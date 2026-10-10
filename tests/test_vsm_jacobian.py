"""Independent derivatives, unchanged defaults and explicit policy transport."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from uuid import uuid4
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.wing import Wing, SectionPolar, SolveSettings, solve_circulation
from openfawe.aero.aircraft import PanelPolars
from openfawe.aero.vsm import (geometry, solve_vsm, solve_vsm_operator,
                              _vsm_closure, _vsm_closure_jacobian)
from openfawe.aero.nonlinear import (solve_newton, solve_newton_picard,
                                    NonlinearSolveError, HybridSolveError, validate_solver_trace)
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero_archive import AeroWriter, AeroReader
from openfawe.provenance import sha256_file

AN = SolveSettings(solver='newton', jacobian='active_segment', max_iterations=60)


class PolarDerivativeTests(unittest.TestCase):
    def table(self,bounds='error'):
        return SectionPolar([-10.,0.,5.,20.],[-.7,0.,.2,1.4],[.01]*4,[0.]*4,'test',bounds)

    def test_segments_knots_endpoints_and_units(self):
        p=self.table()
        for a, expected in [(-10.,.07),(0.,.04),(5.,.08),(20.,.08)]:
            self.assertAlmostEqual(float(p.cl_slope(np.deg2rad(a))),expected*180/np.pi)
        knot=np.deg2rad(5.);h=1e-7
        right=(p.coefficients(knot+h)[0]-p.coefficients(knot)[0])/h
        left=(p.coefficients(knot)[0]-p.coefficients(knot-h)[0])/h
        self.assertAlmostEqual(float(p.cl_slope(knot)),float(right),places=7)
        self.assertGreater(abs(right-left),1.)
        for a in [-10.0001,20.0001]:
            with self.assertRaises(ValueError):p.cl_slope(np.deg2rad(a))

    def test_clamp_and_panel_ownership(self):
        p=self.table('clamp');angles=np.deg2rad([-11.,-9.999,19.999,21.])
        expected=np.array([0.,.07,.08,0.])*180/np.pi
        assert_allclose(p.cl_slope(angles),expected,rtol=0,atol=1e-14)
        assert_allclose(PanelPolars((p,)*4).cl_slope(angles),expected)
        with self.assertRaises(ValueError):PanelPolars((p,)).cl_slope(angles)


class JacobianTests(unittest.TestCase):
    def arguments(self,per_panel=True,bounds='error'):
        # Swept, twisted, dihedral geometry: s is not orthogonal to t and n.
        y=np.linspace(-2.,2.,7);le=np.column_stack([.13*y+.03*y*y,y,.06*y*y])
        chord=np.column_stack([np.ones(7),.09*y,.06*y]);wing=Wing(le,le+chord)
        _,c,n,t,s=geometry(wing);rng=np.random.default_rng(382)
        ambient=np.linspace(6.,9.,6)[:,None]*t+.8*n+.3*s
        known=.02*t+.05*n+.01*rng.normal(size=(6,3))
        matrix=.013*rng.normal(size=(6,6,3));gamma=np.linspace(.1,.3,6)
        tables=tuple(SectionPolar([-40.,-5.,3.,9.,25.,45.],
            np.array([-2.,-.3,.15,.65,1.4,1.8])*(1+.01*i),[.01]*6,[0.]*6,'synthetic',bounds)
            for i in range(6))
        polar=PanelPolars(tables) if per_panel else tables[0]
        args=(ambient,matrix,known,c,t,n,s,np.linalg.norm(np.cross(ambient,s),axis=1),polar)
        return wing,gamma,args

    def test_all_columns_with_known_induction_and_nonorthogonal_axes(self):
        for per_panel in [False,True]:
            wing,gamma,args=self.arguments(per_panel)
            self.assertGreater(np.max(abs(np.sum(args[4]*args[6],axis=1))),.05)
            J=_vsm_closure_jacobian(gamma,*args);h=1e-6
            for j in range(wing.panels):
                delta=np.eye(wing.panels)[j]*h
                numerical=(_vsm_closure(gamma+delta,*args)[0]-(gamma+delta)
                          -_vsm_closure(gamma-delta,*args)[0]+(gamma-delta))/(2*h)
                assert_allclose(J[:,j],numerical,rtol=2e-7,atol=2e-9)

    def test_clamped_lift_still_has_velocity_squared_derivative(self):
        wing,gamma,args=self.arguments(bounds='clamp')
        polar=SectionPolar([-2.,0.,2.],[-.1,0.,.1],[.01]*3,[0.]*3,'clamp','clamp')
        args=(*args[:-1],polar)
        self.assertTrue(np.all(np.rad2deg(_vsm_closure(gamma,*args)[2])>2.))
        J=_vsm_closure_jacobian(gamma,*args);h=1e-6
        self.assertGreater(np.max(abs(J+np.eye(6))),1e-5)
        for j in range(6):
            delta=np.eye(6)[j]*h
            difference=(_vsm_closure(gamma+delta,*args)[0]-_vsm_closure(gamma-delta,*args)[0])/(2*h)-np.eye(6)[:,j]
            assert_allclose(J[:,j],difference,rtol=2e-7,atol=2e-9)

    def test_generic_callback_errors_do_not_retry(self):
        for bad in [np.zeros((2,1)),np.full((2,2),np.nan),np.eye(2)*1j,[['bad']*2]*2]:
            with self.assertRaises(ValueError):
                solve_newton(lambda x:x+1,np.zeros(2),AN,jacobian=lambda x:bad)
        calls=[]
        def invalid(x):calls.append(x.copy());raise ValueError('invalid active segment')
        with self.assertRaisesRegex(ValueError,'invalid active segment'):
            solve_newton_picard(lambda x:x+1,np.zeros(2),replace(AN,solver='newton_picard'),jacobian=invalid)
        self.assertEqual(len(calls),1)

    def test_failure_payload_records_actual_domain_rejections(self):
        def target(x):
            if x[0]>0:raise ValueError('domain')
            return x+1
        diagnostic={}
        with self.assertRaises(NonlinearSolveError) as caught:
            solve_newton(target,[0.],AN,jacobian=lambda x:-np.eye(1),diagnostics=diagnostic)
        self.assertIs(caught.exception.diagnostics,diagnostic)
        assert_array_equal(diagnostic['circulation'],[0.])
        self.assertEqual(diagnostic['updates'],0)
        self.assertEqual(diagnostic['domain_rejected'],32)
        self.assertEqual(diagnostic['no_decrease'],0)

    def test_only_accepted_iterates_call_derivative_and_count_updates(self):
        values=[];diagnostic={}
        def derivative(x):values.append(x.copy());return np.diag(-2*x)
        root,iterations=solve_newton(lambda x:x+1-x*x,[.5],AN,
            jacobian=derivative,diagnostics=diagnostic)
        self.assertAlmostEqual(root[0],1.,places=8)
        self.assertEqual(len(values),diagnostic['updates'])
        self.assertEqual(iterations,diagnostic['updates']+1)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.parent=Path(__file__).resolve().parents[1]/'local/jacobian-tests'
        self.parent.mkdir(parents=True,exist_ok=True)
        self.root=self.parent/uuid4().hex;self.root.mkdir()

    def tearDown(self):
        if self.root.resolve().parent!=self.parent.resolve():raise ValueError('cleanup escaped test workspace')
        shutil.rmtree(self.root)

    def test_rejection_of_unsupported_consumers(self):
        with self.assertRaises(ValueError):SolveSettings(jacobian='active_segment')
        with self.assertRaises(ValueError):SolveSettings(solver='newton',jacobian='unknown')
        for mode in ['free_wake','vsm_free_wake','vlm','vlm_free_wake','ll_straight']:
            with self.assertRaises(ValueError):AeroBackend(mode,vsm_solve=AN)
        with self.assertRaises(ValueError):WakeConfig(solve=AN)
        wing=Wing.ellipse(4)
        with self.assertRaises(ValueError):solve_circulation(wing,SectionPolar.linear(),[5.,0.,.5],np.zeros((4,4,3)),settings=AN)
        class Other:
            coefficients=SectionPolar.linear().coefficients
            cl_slope=SectionPolar.linear().cl_slope
        with self.assertRaises(ValueError):solve_vsm(wing,Other(),[0.,0.,0.],settings=AN)

    def test_zero_and_small_real_root_at_unchanged_gate(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear()
        zero=solve_vsm(wing,polar,[0.,0.,0.],settings=AN)
        assert_array_equal(zero.panel_force,np.zeros((6,3)))
        fd=solve_vsm(wing,polar,[5.,0.,.5],settings=replace(AN,jacobian='finite_difference'))
        an=solve_vsm(wing,polar,[5.,0.,.5],settings=AN)
        self.assertLessEqual(an.residual,an.residual_limit)
        assert_allclose(an.circulation,fd.circulation,rtol=1e-8,atol=1e-9)
        trace=validate_solver_trace(an.solver_trace_json,an.residual,an.residual_limit,6)
        self.assertEqual(trace['jacobian'],'active_segment')
        self.assertEqual(trace['schema'],'newton_v1')
        for solver in ['newton','newton_picard']:
            backend=AeroBackend('vsm',vsm_solve=replace(AN,solver=solver))
            state=backend.start(wing,polar,[5.,0.,.5])
            self.assertEqual(json.loads(state.solution.solver_trace_json)['jacobian'],'active_segment')
            self.assertEqual(backend.metadata()['solve']['jacobian'],'active_segment')
            with self.assertRaises(ValueError):AeroBackend('vsm').advance(state,polar,[5.,0.,.5])

    def test_plain_newton_cannot_fall_into_unreported_picard(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear()
        with patch('openfawe.aero.nonlinear.solve_newton',return_value=(np.full(4,10.),1)):
            with self.assertRaises(NonlinearSolveError):
                solve_vsm_operator(wing,polar,[5.,0.,.5],np.zeros((4,4,3)),np.zeros((4,4,3)),
                                   settings=SolveSettings(solver='newton'))

    def test_archive_roundtrip_and_trace_policy_tamper(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();backend=AeroBackend('vsm',vsm_solve=AN)
        state=backend.start(wing,polar,[5.,0.,.5])
        path=self.root/'run'
        writer=AeroWriter(path,{'aero_method':'vsm','aerodynamics':backend.metadata()})
        writer.append(state);writer.finish()
        reader=AeroReader(path)
        try:self.assertEqual(reader.frame(0)['vsm_jacobian_policy'],'active_segment')
        finally:reader.close()
        with h5py.File(path/'wake.h5','r+') as f:
            trace=json.loads(f['frames/000000'].attrs['solver_trace_json']);trace['jacobian']='finite_difference'
            f['frames/000000'].attrs['solver_trace_json']=json.dumps(trace)
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        manifest['wake_sha256']=sha256_file(path/'wake.h5')
        (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaises(ValueError):AeroReader(path)

    def make_archive(self,method='vsm',zero=False):
        settings=AN if method=='vsm' else SolveSettings()
        backend=AeroBackend(method,vsm_solve=settings)
        state=backend.start(Wing.ellipse(4),SectionPolar.linear(),[0.,0.,0.] if zero else [5.,0.,.5])
        path=self.root/uuid4().hex;writer=AeroWriter(path,dict(aero_method=method,aerodynamics=backend.metadata()))
        writer.append(state);writer.finish();return path

    def rehash(self,path):
        p=path/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'))
        m['wake_sha256']=sha256_file(path/'wake.h5');m['case_sha256']=sha256_file(path/'case.json')
        p.write_text(json.dumps(m),encoding='utf-8')

    def test_case_missing_derivative_means_FD_for_both_writer_and_reader(self):
        backend=AeroBackend('vsm',vsm_solve=AN);state=backend.start(Wing.ellipse(4),SectionPolar.linear(),[5.,0.,.5])
        writer=AeroWriter(self.root/'wrong',dict(aero_method='vsm',aerodynamics=AeroBackend('vsm').metadata()))
        try:
            with self.assertRaisesRegex(ValueError,'case and state VSM Jacobian'):writer.append(state)
        finally:writer.file.close()
        path=self.make_archive();case=json.loads((path/'case.json').read_text(encoding='utf-8'))
        del case['aerodynamics']['solve']['jacobian'];(path/'case.json').write_text(json.dumps(case),encoding='utf-8');self.rehash(path)
        with self.assertRaisesRegex(ValueError,'case and archived VSM Jacobian'):AeroReader(path)

    def test_unknown_orphan_and_mismatched_policy_markers(self):
        for location in ['missing_frame','unknown','root_mismatch','manifest_mismatch']:
            path=self.make_archive()
            with h5py.File(path/'wake.h5','r+') as f:
                if location=='missing_frame':del f['frames/000000'].attrs['vsm_jacobian_policy']
                elif location=='unknown':f['frames/000000'].attrs['vsm_jacobian_policy']='unknown'
                elif location=='root_mismatch':f.attrs['vsm_jacobian_policy']='finite_difference'
            if location=='manifest_mismatch':
                p=path/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'));m['vsm_jacobian_policy']='finite_difference';p.write_text(json.dumps(m),encoding='utf-8')
            self.rehash(path)
            with self.assertRaisesRegex(ValueError,'VSM Jacobian archive'):AeroReader(path)

    def test_non_VSM_derivative_markers_and_traces_rejected(self):
        for location in ['frame','root','manifest','trace']:
            path=self.make_archive('free_wake')
            with h5py.File(path/'wake.h5','r+') as f:
                if location=='frame':f['frames/000000'].attrs['vsm_jacobian_policy']='active_segment'
                elif location=='root':f.attrs['vsm_jacobian_policy']='active_segment'
                elif location=='trace':f['frames/000000'].attrs['solver_trace_json']=json.dumps(dict(jacobian='active_segment'))
            if location=='manifest':
                p=path/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'));m['vsm_jacobian_policy']='active_segment';p.write_text(json.dumps(m),encoding='utf-8')
            self.rehash(path)
            with self.assertRaisesRegex(ValueError,'non-VSM archive'):AeroReader(path)

    def test_zero_active_archive_and_update_count_validation(self):
        path=self.make_archive(zero=True);reader=AeroReader(path)
        try:self.assertEqual(reader.frame(0)['vsm_jacobian_policy'],'active_segment')
        finally:reader.close()
        state=AeroBackend('vsm',vsm_solve=AN).start(Wing.ellipse(4),SectionPolar.linear(),[5.,0.,.5])
        trace=json.loads(state.solution.solver_trace_json);trace['updates']=999
        with self.assertRaisesRegex(ValueError,'updates contradict'):
            validate_solver_trace(json.dumps(trace),state.solution.residual,state.solution.residual_limit,4)

    def test_production_hybrid_failure_payload_is_retained_by_archive(self):
        backend=AeroBackend('vsm',vsm_solve=replace(AN,solver='newton_picard',max_iterations=1,fallback_max_iterations=1))
        with self.assertRaises(HybridSolveError) as caught:
            backend.start(Wing.ellipse(4),SectionPolar.linear(),[5.,0.,.5])
        error=caught.exception;self.assertIsNotNone(error.diagnostics)
        self.assertIn('circulation',error.diagnostics);self.assertIn('linear_solve',error.diagnostics)
        writer=AeroWriter(self.root/'failed',dict(aero_method='vsm',aerodynamics=backend.metadata()))
        writer.fail(error);m=json.loads((writer.directory/'manifest.json').read_text(encoding='utf-8'))
        data=m['solver_diagnostics'];p=writer.directory/data['arrays_file']
        self.assertEqual(data['arrays_sha256'],sha256_file(p))
        with np.load(p) as arrays:assert_array_equal(arrays['circulation'],error.diagnostics['circulation'])

    def test_reused_diagnostics_do_not_keep_stale_failures(self):
        data=dict(history=[dict(update=999)],domain_rejected=32,no_decrease=7)
        solve_newton(lambda x:np.ones_like(x),[0.],AN,jacobian=lambda x:-np.eye(1),diagnostics=data)
        self.assertNotIn('domain_rejected',data);self.assertNotIn('no_decrease',data)
        self.assertEqual([h['update'] for h in data['history']],[1])
        self.assertEqual(data['linear_solve'],'solve')

    def test_singular_linear_system_records_least_squares_direction(self):
        data={}
        gamma,_=solve_newton(lambda x:np.array([1.,x[1]]),[0.,0.],AN,
            jacobian=lambda x:np.diag([-1.,0.]),diagnostics=data)
        assert_array_equal(gamma,[1.,0.])
        self.assertEqual(data['linear_solve'],'lstsq')


if __name__=='__main__':unittest.main()
