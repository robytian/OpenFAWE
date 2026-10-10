"""OpenFAWE VSM-Free Vortex Wake: source limit, history and state checks."""
import json
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from openfawe.aero.aircraft import load_aircraft, WingAssembly, PanelPolars
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.vsm import solve_vsm, geometry, vsm_influence, vsm_self_correction
from openfawe.aero.vsm_wake import solve_finite_vsm, vsm_wake_influence
from openfawe.aero.wake import WakeConfig, wake_network
from openfawe.aero.methods import AeroBackend
from openfawe.aero.cores import edge_radii

ROOT=Path(__file__).resolve().parents[1]
NEWTON=SolveSettings(solver='newton',max_iterations=60)


class VsmWakeTests(unittest.TestCase):
    def test_vf01_finite_wake_converges_to_original_vsm(self):
        definition=load_aircraft(ROOT/'cases/aircraft/v3_cad_attached.json')
        config=WakeConfig(core_model=CoreModel.NONE,solve=NEWTON)
        fixture=json.loads((ROOT/'tests/fixtures/v3_reference.json').read_text())
        for row in fixture['cases']:
            wind=np.array(row['inflow_m_s']);direction=wind/np.linalg.norm(wind)
            downstream=definition.wing.trailing_edge+1e5*direction
            result=solve_finite_vsm(definition.wing,definition.polar,wind,[downstream],config=config)
            assert_allclose(result.circulation,row['circulation'],atol=5e-5,rtol=5e-5)
            assert_allclose(result.cl,row['cl_section'],atol=5e-5,rtol=5e-5)
            self.assertLessEqual(result.residual,result.residual_limit)

    def test_vf01_both_operators_match_semi_infinite_limit(self):
        wing=Wing.ellipse(8);wind=np.array([10.,0.,1.]);direction=wind/np.linalg.norm(wind)
        config=WakeConfig(core_model=CoreModel.NONE)
        for load in (False,True):
            target=vsm_influence(wing,direction,at_load_points=load)
            errors=[]
            for length in (10.,100.,1000.):
                matrix=vsm_wake_influence(wing,[wing.trailing_edge+length*direction],
                    config=config,at_load_points=load)
                errors.append(float(np.max(abs(matrix-target))))
            self.assertGreater(errors[0],errors[1]);self.assertGreater(errors[1],errors[2])
            self.assertLess(errors[-1],5e-8)

    def test_vf02_saved_network_reconstructs_closure_and_load_induction(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=np.array([5.,0.,.5])
        config=WakeConfig(dt_s=.05,solve=NEWTON,core_evolution='age_stretch')
        backend=AeroBackend('vsm_free_wake',wake=config)
        state=backend.start(wing,polar,wind)
        for _ in range(3):state=backend.advance(state,polar,wind)
        nodes,edges,gamma=wake_network(state)
        radii=edge_radii(wing,edges,state.cores,config.core_radius_m)
        cp,chord,normal,tangent,span=geometry(wing)
        induced=segments_velocity(cp,nodes,edges,gamma,model=config.core_model,core_radius=radii)
        induced-=vsm_self_correction(wing,cp)*state.solution.circulation[:,None]
        velocity=wind+induced
        alpha=np.arctan2(np.sum(velocity*normal,axis=1),np.sum(velocity*tangent,axis=1))
        assert_allclose(alpha,state.solution.polar_alpha_rad,atol=1e-12)
        keep=~np.all(edges<wing.node_count,axis=1)
        load=segments_velocity(wing.cp,nodes,edges[keep],gamma[keep],model=config.core_model,core_radius=radii[keep])
        assert_allclose(load,state.solution.induced_velocity,atol=1e-12)
        balance=np.zeros(len(nodes));np.add.at(balance,edges[:,0],gamma);np.add.at(balance,edges[:,1],-gamma)
        assert_allclose(balance,0.,atol=1e-12)

    def test_vf03_exact_zero_circulation_clock_and_method_ownership(self):
        wing=Wing.ellipse(2);polar=SectionPolar([-30.,30.],[0.,0.],[0.,0.],[0.,0.]);wind=np.array([2.,0.,.1])
        backend=AeroBackend('vsm_free_wake',wake=WakeConfig(dt_s=.1,max_rows=120))
        state=backend.start(wing,polar,wind,time_s=10.3)
        for i in range(1,121):state=backend.advance(state,polar,wind,next_time_s=10.3+i*.1)
        self.assertEqual(state.time_s,22.3)
        assert_allclose(state.free_rows[-1],wing.trailing_edge+12*wind,atol=1e-12)
        assert_allclose(state.solution.circulation,0.,atol=0)
        with self.assertRaises(ValueError):AeroBackend('free_wake').advance(state,polar,wind)
        with self.assertRaises(ValueError):AeroBackend('free_wake').induced_velocity(state,[[1.,2.,3.]])
        with self.assertRaises(ValueError):AeroBackend('vsm_free_wake',vsm_core_policy='source_legacy')

    def test_vf04_repeated_and_failed_steps_preserve_state(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=[5.,0.,.5]
        backend=AeroBackend('vsm_free_wake',wake=WakeConfig(dt_s=.05,solve=NEWTON))
        state=backend.start(wing,polar,wind);before=state.solution.circulation.copy()
        a=backend.advance(state,polar,wind);b=backend.advance(state,polar,wind)
        assert_allclose(a.free_rows,b.free_rows,atol=0,rtol=0)
        assert_allclose(a.solution.force,b.solution.force,atol=0,rtol=0)
        with self.assertRaises(ValueError):backend.advance(state,polar,wind,next_time_s=.051)
        assert_allclose(state.solution.circulation,before,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
