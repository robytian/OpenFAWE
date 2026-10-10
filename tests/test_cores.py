"""M11-M13: source-law core growth, material stretch and shared-edge consistency."""

from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4
import unittest
import xml.etree.ElementTree as ET
import numpy as np

from openfawe.aero.cores import evolve_radius, edge_radii, core_identity
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings, CirculationConvergenceError
from openfawe.aero.wake import WakeConfig, start_wake, advance_wake, wake_network, current_influence, pack_rings
from openfawe.aero.vortex import segments_velocity, CoreModel
from openfawe.aero.methods import AeroBackend
from openfawe.aero_archive import AeroReader
from openfawe.aero_demo import run_demo


class CoreEvolutionTests(unittest.TestCase):
    def test_m11_age_strain_inviscid_and_semigroup(self):
        radius=np.array([.01,.1,1.]);old=np.array([1.,2.,3.]);new=old*np.array([.25,1.,5.])
        np.testing.assert_allclose(evolve_radius(radius,old,new,.5,.03)**2,radius**2+.015,rtol=1e-12,atol=1e-14)
        result=evolve_radius(radius,old,new,.5,0.,stretching=True)
        np.testing.assert_allclose(result**2*new,radius**2*old,rtol=1e-12,atol=1e-14)
        half=np.sqrt(old*new)
        two=evolve_radius(evolve_radius(radius,old,half,.25,.03,stretching=True),half,new,.25,.03,stretching=True)
        np.testing.assert_allclose(two,evolve_radius(radius,old,new,.5,.03,stretching=True),rtol=1e-12,atol=1e-14)
        # Independent integrating-factor solution at fixed positive strain rate.
        sigma=2.;dt=.3;rate=.004;q0=.04
        expected=q0*np.exp(-sigma*dt)+rate/sigma*(1-np.exp(-sigma*dt))
        self.assertAlmostEqual(float(evolve_radius(.2,1.,np.exp(sigma*dt),dt,rate,stretching=True)**2),expected,14)

    def test_m11_invalid_inputs_and_method_configurations(self):
        for args in [(0,1,2,.1,1),(.1,0,2,.1,1),(.1,1,-2,.1,1),(.1,1,2,-.1,1),(.1,1,2,.1,-1),(.1,1,np.inf,.1,1)]:
            with self.assertRaises(ValueError):evolve_radius(*args)
        with self.assertRaises(ValueError):WakeConfig(core_evolution='unknown')
        with self.assertRaises(ValueError):WakeConfig(core_evolution='age',core_model=CoreModel.NONE)
        with self.assertRaises(ValueError):AeroBackend('vsm',wake=WakeConfig(core_evolution='age'))

    def test_m12_zero_gamma_core_age_birth_and_immutability(self):
        cfg=WakeConfig(dt_s=.1,core_evolution='age');wing=Wing.ellipse(4)
        polar=SectionPolar([-30,30],[0,0],[0,0],[0,0]);wind=[2,0,.1]
        state=start_wake(wing,polar,cfg,wind);rate=core_identity(cfg)[2]
        for step in range(1,5):
            previous=state
            state=advance_wake(previous,polar,cfg,wind)
            repeat=advance_wake(previous,polar,cfg,wind)
            np.testing.assert_array_equal(repeat.cores.span,state.cores.span)
            age=state.time_s-state.release_times_s
            np.testing.assert_allclose(state.cores.span**2,
                np.broadcast_to((cfg.core_radius_m**2+rate*age)[:,None],(step,4)),rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(state.cores.stream**2,
                np.broadcast_to((cfg.core_radius_m**2+rate*(age-cfg.dt_s/2))[:,None],(step,5)),rtol=1e-12,atol=1e-14)
            self.assertEqual(len(wake_network(state)[1]),0)
        with self.assertRaises(ValueError):state.cores.span[0,0]=1
        with self.assertRaises(ValueError):advance_wake(state,polar,replace(cfg,core_evolution='constant'),wind)
        with self.assertRaises(ValueError):advance_wake(state,polar,replace(cfg,eddy_viscosity_factor=2),wind)

    def test_m12_shared_edge_operator_equals_whole_network(self):
        for policy in ('age','age_stretch'):
            cfg=WakeConfig(dt_s=.1,core_evolution=policy,eddy_viscosity_factor=100)
            wing=Wing.ellipse(6);polar=SectionPolar.linear();wind=[1,0,.1]
            state=start_wake(wing,polar,cfg,wind)
            for _ in range(3):state=advance_wake(state,polar,cfg,wind)
            nodes,edges,gamma=wake_network(state)
            full=segments_velocity(wing.cp,nodes,edges,gamma,model=cfg.core_model,
                core_radius=edge_radii(wing,edges,state.cores,cfg.core_radius_m),singularity='zero')
            operator=current_influence(wing,state.free_rows[0],state.cores,cfg)
            nodes,edges,gamma=pack_rings(state.free_rows,state.older_circulation,wing.panel_edges)
            known=segments_velocity(wing.cp,nodes,edges,gamma,model=cfg.core_model,
                core_radius=edge_radii(wing,edges+2*wing.node_count,state.cores,cfg.core_radius_m))
            np.testing.assert_allclose(full,known+np.einsum('ijk,j->ik',operator,state.solution.circulation),rtol=1e-10,atol=1e-10)
            old=state.cores.span.copy()
            restricted=replace(cfg,solve=SolveSettings(max_iterations=1))
            with self.assertRaises(CirculationConvergenceError):
                # Explicitly construct a counterfactual frozen-state probe;
                # production advances must not silently switch configuration.
                advance_wake(replace(state,config=restricted),polar,restricted,wind)
            np.testing.assert_array_equal(state.cores.span,old)

    def test_m13_timestep_convergence(self):
        for policy in ('age','age_stretch'):
            forces=[];oldest=[]
            for dt in (.1,.05,.025):
                cfg=WakeConfig(dt_s=dt,core_evolution=policy);polar=SectionPolar.linear()
                state=start_wake(Wing.ellipse(8),polar,cfg,[1,0,.1])
                for _ in range(round(1/dt)):state=advance_wake(state,polar,cfg,[1,0,.1])
                forces.append(state.solution.force[2]);oldest.append(state.free_rows[-1])
            self.assertLess(abs(forces[-1]/forces[-2]-1),.01)
            errors=[np.max(np.linalg.norm(b-a,axis=1)) for a,b in zip(oldest,oldest[1:])]
            self.assertLess(errors[-1],errors[0])

    def test_m13_archive_and_vtk_material_cores(self):
        parent=Path(__file__).resolve().parents[1]/'local'/'core-tests'
        parent.mkdir(parents=True,exist_ok=True)
        tmp=parent/uuid4().hex
        tmp.mkdir()
        try:
            path=tmp/'age'
            run_demo(path,panels=4,duration_s=.1,core_evolution='age_stretch',export_vtk=True)
            reader=AeroReader(path)
            try:
                frame=reader.frame(2)
                self.assertEqual(frame['core_evolution'],'age_stretch')
                self.assertEqual(frame['span_core_radius_m'].shape,(2,4))
                root=ET.parse(path/'paraview/wake-000002.vtp')
                radii=np.fromstring(root.find('.//DataArray[@Name="core_radius_m"]').text,sep=' ')
                np.testing.assert_allclose(radii,frame['network_core_radius_m'],rtol=1e-12,atol=1e-12)
            finally:reader.close()
        finally:
            if tmp.resolve().parent != parent.resolve():
                raise ValueError('Refusing cleanup outside test workspace')
            shutil.rmtree(tmp)


if __name__=='__main__':unittest.main()
