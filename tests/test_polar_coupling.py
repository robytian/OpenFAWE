"""Independent limits, derivatives and failure gates of optional polar coupling."""
from dataclasses import replace, asdict
from pathlib import Path
import json
import os
import h5py
import importlib.util
import shutil
from uuid import uuid4
from contextlib import contextmanager, closing
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.methods import AeroBackend, rigid_velocity_field
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    build_lifting_surface_mesh, lifting_surface_influence)
from openfawe.aero.polar_coupling import (SectionReferences, ClosureProblem,
    PolarCouplingError, inverse_reference, source_values)
from openfawe.aero.wake import WakeConfig
from openfawe.aero_archive import AeroWriter, AeroReader, _validate_lattice_frame
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file
from openfawe.run_catalog import inspect_run


@contextmanager
def scratch(parent):
    path=parent/uuid4().hex;path.mkdir()
    try:yield path
    finally:
        if path.resolve().parent != parent.resolve():
            raise RuntimeError('test cleanup escaped parent')
        shutil.rmtree(path)


def wing(span=4., panels=4):
    y=np.linspace(-span/2,span/2,panels+1)
    le=np.column_stack((y*0,y,y*0))
    return Wing(le,le+[1.,0.,0.])


def table(slope=5.,cd=.02,cm=-.03):
    a=np.linspace(-30,30,121)
    return SectionPolar(a,slope*np.sin(np.deg2rad(a)),a*0+cd,a*0+cm)


class PolarCouplingTests(unittest.TestCase):
    def test_exact_section_reference_and_rotation(self):
        w=wing()
        for nc in (1,4,8):
            mesh=build_lifting_surface_mesh(w,LiftingSurfaceConfig(nc,chordwise_spacing='cosine'))
            ref=SectionReferences(w,mesh)
            g,_=ref.response(np.zeros(w.panels))
            assert_allclose(2*g.sum(axis=1),np.tile([0,2*np.pi],(w.panels,1)),atol=2e-14)
            assert_allclose(ref.moment(g,np.full(w.panels,.1)),0,atol=2e-15)
        ref=SectionReferences(w,build_lifting_surface_mesh(w,LiftingSurfaceConfig(1)))
        delta=np.array([.01,.03,.05,.07]);g,dg=ref.response(delta,derivative=True)
        assert_allclose(2*g.sum(axis=1),np.column_stack((-2*np.pi*np.tan(delta),delta*0+2*np.pi)),atol=3e-14)
        assert_allclose(2*dg.sum(axis=1),np.column_stack((-2*np.pi/np.cos(delta)**2,delta*0)),atol=3e-14)

    def problem(self):
        w=wing();cfg=WakeConfig(core_radius_m=.02)
        settings=LiftingSurfaceConfig(3,polar_coupling='grid_section')
        mesh=build_lifting_surface_mesh(w,settings)
        inflow=np.array([10.,0.,1.]);far=w.trailing_edge+settings.wake_length_m*inflow/np.linalg.norm(inflow)
        operator=lifting_surface_influence(w,mesh,cfg,far)
        return ClosureProblem(w,mesh,operator,np.tile(inflow,(mesh.panels,1)),np.tile(inflow,(w.panels,1)),table(),settings)

    def test_source_camber_reference_against_retained_independent_calculation(self):
        data=json.loads((Path(__file__).parent/'fixtures/polar_coupling_geometric_reference.json').read_text(encoding='utf-8'))
        le=np.array([[0.,-1.,0.],[0.,1.,0.]])
        offsets=np.zeros((2,len(data['x']),3));offsets[:,:,2]=data['z']
        w=Wing(le,le+[1.,0.,0.],data['x'],offsets)
        for row in data['records']:
            mesh=build_lifting_surface_mesh(w,LiftingSurfaceConfig(row['panels'],chordwise_spacing='cosine',camber_normal='reconstructed_slope'))
            reference=SectionReferences(w,mesh);g,_=reference.response(np.zeros(1))
            alpha=np.array([np.deg2rad(row['alpha_deg'])])
            cl=2*g.sum(axis=1)@np.array([np.cos(alpha[0]),np.sin(alpha[0])])
            assert_allclose(cl,row['CL_circulation'],atol=3e-12)
            assert_allclose(reference.moment(g,alpha),row['Cm_quarter_chord'],atol=3e-12)

    def test_analytic_jacobian_against_central_differences(self):
        problem=self.problem();d=np.array([.01,.02,.03,.04]);h=1e-6
        analytic=problem.evaluate(d,jacobian=True)['jacobian']
        numeric=np.column_stack([(problem.evaluate(d+v)['residual']-problem.evaluate(d-v)['residual'])/(2*h)
                                 for v in np.eye(4)*h])
        assert_allclose(analytic,numeric,rtol=2e-8,atol=2e-9)

    def test_exact_potential_polar_identity_and_default_unchanged(self):
        w=wing();wind=[10.,0.,1.];cfg=WakeConfig(core_radius_m=.02)
        settings=LiftingSurfaceConfig(4)
        pure=start_lifting_surface(w,cfg,wind,settings=settings)
        unused=start_lifting_surface(w,cfg,wind,settings=settings,polar=table(slope=1))
        assert_array_equal(pure.solution.lattice_force,unused.solution.lattice_force)
        alpha=np.arcsin(2*pure.solution.circulation/np.linalg.norm(wind)/(2*np.pi))
        knots=np.unique(np.r_[-30.,np.rad2deg(alpha),30.])
        polar=SectionPolar(knots,2*np.pi*np.sin(np.deg2rad(knots)),knots*0,knots*0)
        coupled=start_lifting_surface(w,cfg,wind,polar=polar,settings=replace(settings,polar_coupling='grid_section'))
        self.assertEqual(coupled.solution.polar_coupling.iterations,0)
        assert_array_equal(pure.solution.lattice_circulation,coupled.solution.lattice_circulation)
        assert_allclose(pure.solution.lattice_force,coupled.solution.lattice_force,atol=2e-13)
        assert_allclose(pure.solution.lattice_couple,coupled.solution.lattice_couple,atol=2e-13)

    def test_infinite_span_synthetic_lift_drag_and_moment(self):
        w=wing(span=1e5,panels=1);alpha=np.deg2rad(5.)
        wind=10*np.array([np.cos(alpha),0,np.sin(alpha)])
        cfg=WakeConfig(core_radius_m=.01)
        settings=LiftingSurfaceConfig(1,wake_length_m=1e8,polar_coupling='grid_section')
        polar=table(slope=.8*2*np.pi,cd=.03,cm=-.05)
        state=start_lifting_surface(w,cfg,wind,settings=settings,polar=polar)
        c=state.solution.polar_coupling
        assert_allclose(c.delta_rad,np.arctan(.2*np.tan(alpha)),atol=4e-6)
        assert_allclose(c.effective_alpha_rad,alpha,atol=2e-6)
        force=state.solution.panel_force.sum(axis=0);moment=state.solution.panel_couple.sum(axis=0)
        qarea=.5*cfg.density_kg_m3*100*w.area.sum()
        assert_allclose(force@np.array([-np.sin(alpha),0,np.cos(alpha)])/qarea,polar.coefficients(alpha)[0],atol=1e-5)
        assert_allclose(force@(wind/10)/qarea,.03,atol=1e-5)
        assert_allclose(moment[1]/qarea,-.05,atol=1e-5)

    def test_failure_bounds_reverse_flow_and_solver_trace(self):
        w=wing();settings=LiftingSurfaceConfig(2,polar_coupling='grid_section');cfg=WakeConfig()
        for wind,polar in [([-10,0,1],table()),([10,0,20],table()),([10,0,1],None),
                            ([10,0,1],table(-1)),([10,0,1],SectionPolar([-1,1],[-.1,.1],[0,0],[0,0],bounds='clamp'))]:
            with self.subTest(wind=wind,polar=str(type(polar))):
                with self.assertRaises(ValueError):
                    start_lifting_surface(w,cfg,wind,settings=settings,polar=polar)
        with self.assertRaises(PolarCouplingError) as failure:
            start_lifting_surface(w,cfg,[10,0,1],settings=replace(settings,polar_max_virtual_incidence_deg=.0001),polar=table())
        self.assertTrue(failure.exception.trace)
        with self.assertRaises(ValueError):
            replace(settings,steady_mirror_projection=True)
        with self.assertRaises(ValueError):
            inverse_reference(np.array([[0,2*np.pi]]),np.array([.1]),np.array([np.deg2rad(170)]))

    def test_v3_bounded_explicit_seed_recovers_same_positive_slope_root(self):
        from openfawe.aero.aircraft import load_aircraft
        from openfawe.aero.reference import inflow
        root=Path(__file__).resolve().parents[1]
        definition=load_aircraft(root/'cases/aircraft/v3_cad_camber_integral.json')
        wind=inflow(5e5*1.4639e-5/(.395*6.5),5.412846420261046)
        settings=LiftingSurfaceConfig(8,chordwise_spacing='cosine',camber_normal='reconstructed_slope',polar_coupling='grid_section')
        cfg=WakeConfig(core_radius_m=.15)
        with self.assertRaisesRegex(PolarCouplingError,'positive active lift slope'):
            start_lifting_surface(definition.wing,cfg,wind,settings=settings,polar=definition.polar,free_wake=True)
        answers=[]
        for seed in (2.,3.,4.):
            state=start_lifting_surface(definition.wing,cfg,wind,
                settings=replace(settings,polar_initial_virtual_incidence_deg=seed),polar=definition.polar,free_wake=True)
            answers.append(state.solution.circulation)
            self.assertLess(np.max(abs(state.solution.polar_coupling.cl_residual)),settings.polar_cl_tolerance)
        for other in answers[1:]:assert_allclose(other,answers[0],atol=2e-8,rtol=2e-8)

    def test_source_knots_right_slopes_and_last_left(self):
        p=SectionPolar([-10,0,10],[-.1,0,1],[0,0,0],[0,0,0])
        _,s=source_values((p,p),np.array([0,np.deg2rad(10)]),np.zeros(2))
        assert_allclose(s,1/np.deg2rad(10))

    def test_nonuniform_rigid_field_sampling(self):
        w=wing();settings=LiftingSurfaceConfig(3,polar_coupling='grid_section')
        body=rigid_velocity_field([.1,0,0],[0,.02,0])
        inflow=lambda p,t:np.column_stack((10+.05*p[:,1],p[:,0]*0,1+.02*p[:,0]))
        state=start_lifting_surface(w,WakeConfig(),inflow,settings=settings,polar=table(),structural_velocity=body)
        c=state.solution.polar_coupling
        assert_allclose(c.ambient_m_s,inflow(c.reference_points_m,0)-body(c.reference_points_m,0))
        self.assertLess(np.max(abs(c.cl_residual)),settings.polar_cl_tolerance)

    def test_release_jacobian_and_unchanged_physical_unsteady_loads(self):
        from openfawe.aero.lifting_surface import _loads, _wake_induction
        for release,theta in [('current',1.),('trapezoidal',.5)]:
            w=wing();p=table();cfg=WakeConfig(dt_s=.03,core_radius_m=.05)
            settings=LiftingSurfaceConfig(2,polar_coupling='grid_section',wake_release=release,gamma_derivative='bdf2')
            backend=AeroBackend('vlm_free_wake',wake=cfg,lifting_surface=settings)
            state=backend.start(w,p,[10.,0.,.5])
            state=backend.advance(state,p,[10.,0.,.7]);wind=np.array([10.,0.,.9])
            state=backend.advance(state,p,wind)
            c=state.solution.polar_coupling;mesh=state.mesh
            force,couple=_loads(w,mesh,state.solution.lattice_circulation,state.free_rows,
                state.wake_circulation,state.cores,cfg,wind,(0,0,0),state.time_s,state.solution.lattice_gamma_rate)
            assert_array_equal(c.potential_force_N,force);assert_array_equal(c.potential_couple_N_m,couple)
            self.assertGreater(np.max(abs(state.solution.lattice_gamma_rate)),.1)
            operator=lifting_surface_influence(w,mesh,cfg)
            for j in range(w.panels):
                unit=np.zeros((1,w.panels));unit[0,j]=1
                operator[:,-w.panels+j]+=theta*_wake_induction(w,mesh,state.free_rows[:1],unit,mesh.collocation_points,state.cores,cfg)
            history=state.wake_circulation.copy();history[0]=(1-theta)*state.previous_lattice_circulation[-1]
            known=_wake_induction(w,mesh,state.free_rows,history,mesh.collocation_points,state.cores,cfg)
            problem=ClosureProblem(w,mesh,operator,known+wind,np.tile(wind,(w.panels,1)),p,settings)
            d=c.delta_rad;h=1e-6;analytic=problem.evaluate(d,jacobian=True)['jacobian']
            numeric=np.column_stack([(problem.evaluate(d+v)['residual']-problem.evaluate(d-v)['residual'])/(2*h) for v in np.eye(w.panels)*h])
            assert_allclose(analytic,numeric,rtol=3e-8,atol=3e-9)

    def test_coupled_straight_labels_seed_provenance_and_case_binding(self):
        from openfawe.aero.polar_coupling import encode_polars,polar_identity,polar_tables
        from openfawe.aero_archive import POLAR_LATTICE_SCHEMA
        parent=Path(__file__).resolve().parents[1]/'local'/'polar-coupling-tests';parent.mkdir(parents=True,exist_ok=True)
        with scratch(parent) as root:
            w=wing();p=table();backend=AeroBackend('vlm',lifting_surface=LiftingSurfaceConfig(2,polar_coupling='grid_section'))
            identity=polar_identity(encode_polars(polar_tables(p,w.panels)))
            case=dict(aero_method='vlm',aerodynamics=backend.metadata(),coupled_source_polars_sha256=identity)
            run=root/'run';writer=AeroWriter(run,case)
            state=backend.start(w,p,[10,0,.5]);writer.append(state);writer.finish()
            self.assertIn('source-polar coupled',inspect_run(run).core)
            with closing(AeroReader(run)) as reader:
                self.assertEqual(reader.manifest['schema'],POLAR_LATTICE_SCHEMA)
                self.assertEqual(reader.frame(0)['polar_input_binding'],'case_hash_checked')
            output=root/'vtk';export_paraview(run,output)
            text=''.join(f.read_text(encoding='utf-8') for f in output.glob('*.vtp'))
            self.assertIn('virtual_boundary_residual_m_s',text)
            self.assertIn('polar_correction_0_none_1_grid_section',text)
            if importlib.util.find_spec('PySide6'):
                os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
                from PySide6 import QtWidgets
                from openfawe.aero_viewer import AeroReplay
                app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
                window=AeroReplay(run)
                try:
                    self.assertIn('Section polars: source-polar coupled',window.info.text())
                    self.assertNotIn('Section polars: not used',window.info.text())
                    self.assertIn('VIRTUAL BOUNDARY',window.info.text())
                finally:window.close();app.processEvents()
            # A refreshed file checksum cannot waive the v4 seed contract.
            with h5py.File(run/'wake.h5','r+') as f:
                g=f['frames']['000000'];trace=json.loads(bytes(g['polar_trace_json'][:]).decode('utf-8'))
                del trace[0]['initial_virtual_incidence_rad'];del g['polar_trace_json']
                g.create_dataset('polar_trace_json',data=np.frombuffer(json.dumps(trace).encode('utf-8'),dtype=np.uint8))
            manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'));manifest['wake_sha256']=sha256_file(run/'wake.h5')
            (run/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'actual initial seed'):AeroReader(run)
            writer=AeroWriter(root/'wrong-input',dict(case,coupled_source_polars_sha256='0'*64))
            try:
                with self.assertRaisesRegex(ValueError,'declared case input'):writer.append(state)
            finally:writer.fail(ValueError('deliberate wrong input identity test'))

    def test_archive_free_history_export_and_tamper(self):
        w=wing();p=table();wind=[10.,0.,.5]
        parent=Path(__file__).resolve().parents[1]/'local'/'polar-coupling-tests'
        parent.mkdir(parents=True,exist_ok=True)
        for release in ('previous','current','trapezoidal'):
            with self.subTest(release=release),scratch(parent) as tmp:
                backend=AeroBackend('vlm_free_wake',wake=WakeConfig(dt_s=.03,core_radius_m=.05),
                    lifting_surface=LiftingSurfaceConfig(2,polar_coupling='grid_section',wake_release=release,gamma_derivative='bdf2'))
                run=Path(tmp)/'run';writer=AeroWriter(run,dict(aero_method=backend.method.value,aerodynamics=backend.metadata()))
                try:
                    state=backend.start(w,p,wind);writer.append(state)
                    for _ in range(3):
                        state=backend.advance(state,p,wind);writer.append(state)
                    writer.finish()
                except BaseException as error:
                    writer.fail(error);raise
                with closing(AeroReader(run)) as reader:
                    frames=[reader.frame(i) for i in range(reader.count)]
                    frame=frames[-1]
                    self.assertTrue(frame['section_polar_used'])
                    self.assertEqual(frame['gamma_rate_stencil'],'bdf2')
                    for key in ('polar_cl_residual','polar_delta_rad','polar_source_coefficients','polar_correction_couple_N_m'):
                        bad=dict(frame);bad[key]=frame[key].copy();bad[key].flat[0]+=.01
                        with self.assertRaises(ValueError,msg=key):_validate_lattice_frame(bad)
                export_paraview(run,Path(tmp)/'vtk')
                with self.assertRaisesRegex(ValueError,'change coupled source polars'):
                    backend.advance(state,table(slope=4),wind)


if __name__=='__main__':unittest.main()
