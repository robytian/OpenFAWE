"""Native-source component contracts and explicit VSM core policy ownership."""
from dataclasses import replace
import importlib.util,json,shutil,sys,unittest
from pathlib import Path
from uuid import uuid4
import h5py
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from openfawe.aero.vsm import (_pinned_finite,_pinned_ray,_source_ray,solve_vsm,vsm_influence,
    horseshoe_influence,core_kernel_identity)
from openfawe.aero.vortex import FilamentSingularity
from openfawe.aero.wing import Wing,SectionPolar,SolveSettings
from openfawe.aero.methods import AeroBackend
from openfawe.aero_archive import AeroWriter,AeroReader
from openfawe.provenance import sha256_file
from openfawe.run_catalog import inspect_run

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'docs/tasks/OF-0006-vsmcore20260928'
NEWTON=SolveSettings(solver='newton',max_iterations=60)


class CoreProfileTests(unittest.TestCase):
    def test_native_six_point_oracle(self):
        evidence=json.loads((TASK/'evidence/kernel-probe.json').read_text(encoding='utf-8'))
        for row in evidence['records']:
            p=np.asarray(row['points_m']);a=np.zeros(3);b=np.array([1.,0.,0.])
            actual=(_pinned_ray(p,a,b,1.) if row['kind']=='ray' else
                    _pinned_finite(p,a,b,1.,row['kind']=='bound'))
            assert_allclose(actual,row['native_m_s'],rtol=1e-10,atol=1e-10)

    def test_rotation_translation_reversal_and_oriented_reflection(self):
        a=np.zeros(3);b=np.array([1.,0.,0.]);p=np.array([[.4,.01,.02],[1.3,1e-5,2e-5],[-.7,1e-5,2e-5]])
        axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
        K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R=np.eye(3)+np.sin(.73)*K+(1-np.cos(.73))*K@K
        M=np.diag([1.,-1.,1.]);offset=np.array([.2,.3,-.4])
        for bound in [False,True]:
            v=_pinned_finite(p,a,b,2.7,bound)
            assert_allclose(_pinned_finite(p@R.T,R@a,R@b,2.7,bound),v@R.T,rtol=1e-9,atol=1e-9)
            assert_allclose(_pinned_finite(p+offset,a+offset,b+offset,2.7,bound),v,rtol=1e-9,atol=1e-9)
            assert_allclose(_pinned_finite(p,b,a,2.7,bound),-v,rtol=1e-9,atol=1e-9)
            assert_allclose(_pinned_finite(p@M.T,M@b,M@a,2.7,bound),v@M.T,rtol=1e-9,atol=1e-9)
        v=_pinned_ray(p,a,b,2.7)
        assert_allclose(_pinned_ray(p,a,7*b,2.7),v,rtol=0,atol=0)
        assert_allclose(_pinned_ray(p@R.T,R@a,R@b,2.7),v@R.T,rtol=1e-9,atol=1e-9)
        assert_allclose(_pinned_ray(p+offset,a+offset,b,2.7),v,rtol=1e-9,atol=1e-9)
        assert_allclose(-_pinned_ray(p@M.T,M@a,M@b,2.7),v@M.T,rtol=1e-9,atol=1e-9)

    def test_boundary_continuity_and_source_inner_limits(self):
        a=np.zeros(3);b=np.array([1.,0.,0.]);U=2.7
        for kind in ['bound','trailing','ray']:
            cutoff=.05 if kind=='bound' else 4*1.25643*1.48e-5/U
            p=np.array([[.4,cutoff*(1-1e-7),0.],[.4,cutoff*(1+1e-7),0.]])
            v=_pinned_ray(p,a,b,U) if kind=='ray' else _pinned_finite(p,a,b,U,kind=='bound')
            assert_allclose(v[0],v[1],rtol=3e-7,atol=1e-12)
        p=np.array([[.4,1e-8,0.],[.4,4e-8,0.]])
        v=_pinned_finite(p,a,b,U,False)
        assert_allclose(np.linalg.norm(v[0])/np.linalg.norm(v[1]),1.,rtol=1e-9)
        v=_pinned_ray(p,a,b,U)
        assert_allclose(np.linalg.norm(v[0])/np.linalg.norm(v[1]),2.,rtol=1e-9)

    def test_pinned_axis_and_bad_policy_rejection(self):
        a=np.zeros(3);b=np.array([1.,0.,0.]);wing=Wing.ellipse(4)
        for x in [-1.,.4,2.]:
            with self.assertRaises(FilamentSingularity):_pinned_finite(np.array([[x,0.,0.]]),a,b,1.,True)
            with self.assertRaises(FilamentSingularity):_pinned_ray(np.array([[x,0.,0.]]),a,b,1.)
            with self.assertRaises(FilamentSingularity):_source_ray(np.array([[x,0.,0.]]),a,b,1.)
        with self.assertRaises(FilamentSingularity):_pinned_finite(np.array([[.4,1e-15,0.]]),a,b,1.,True)
        with self.assertRaises(FilamentSingularity):_pinned_ray(np.array([[.4,1e-16,0.]]),a,b,1.)
        self.assertTrue(np.isfinite(_pinned_ray(np.array([[.4,1.1e-16,0.]]),a,b,1.)).all())
        for value in ['unknown',1,False,[],np.array(['none'])]:
            with self.assertRaises(ValueError):AeroBackend('vsm',vsm_core_policy=value)
            with self.assertRaises(ValueError):solve_vsm(wing,SectionPolar.linear(),[5.,0.,.5],core_policy=value)
        with self.assertRaises(ValueError):vsm_influence(wing,[1.,0.,0.],source_core_speed=1.)
        with self.assertRaises(ValueError):vsm_influence(wing,[1.,0.,0.],source_core_speed=1.,source_core_policy='none')
        for mode in ['free_wake','vsm_free_wake','vlm','vlm_free_wake','ll_straight']:
            with self.assertRaises(ValueError):AeroBackend(mode,vsm_core_policy='source_pinned_v1')

    def test_legacy_and_none_numeric_and_metadata_bridge(self):
        modules={}
        for name in ['vsm','methods']:
            module_name='openfawe.aero._core_before_'+name
            path=TASK/'retained/review-r1-input/src/openfawe/aero'/(name+'.py')
            spec=importlib.util.spec_from_file_location(module_name,path)
            m=importlib.util.module_from_spec(spec);sys.modules[module_name]=m;spec.loader.exec_module(m);modules[name]=m
        wing=Wing.ellipse(8);polar=SectionPolar.linear();wind=[5.,0.,.5]
        for policy in ['source_legacy','none']:
            a=modules['vsm'].solve_vsm(wing,polar,wind,settings=NEWTON,core_policy=policy)
            b=solve_vsm(wing,polar,wind,settings=NEWTON,core_policy=policy)
            for k,v in vars(a).items():
                if isinstance(v,np.ndarray):self.assertEqual(v.tobytes(),getattr(b,k).tobytes(),k)
            old=modules['methods'].AeroBackend('vsm',vsm_core_policy=policy).metadata()
            new=AeroBackend('vsm',vsm_core_policy=policy).metadata()
            # The isolated old module imports today's settings dataclass;
            # remove that artificial added field, absent from real old metadata.
            self.assertEqual(old['solve'].pop('jacobian'), 'finite_difference')
            self.assertEqual(old['solve'].pop('closure'), 'source')   # stage-4 field, likewise absent from real old metadata
            self.assertEqual(old['solve'].pop('span_flow_fraction'), 0.)
            self.assertEqual(old['solve'].pop('reverse_flow'), 'coupled')
            self.assertEqual(old['solve'].pop('reverse_fade'), .2)
            self.assertEqual(json.dumps(old,sort_keys=True),json.dumps(new,sort_keys=True))

    def test_policy_routes_probe_and_advance_without_mutating_state(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=[5.,0.,.5]
        backend=AeroBackend('vsm',vsm_core_policy='source_pinned_v1',vsm_solve=NEWTON)
        state=backend.start(wing,polar,wind);before=state.solution.circulation.tobytes()
        a,b=wing.bound[wing.panel_edges[1]]
        axis=(b-a)/np.linalg.norm(b-a);r=np.cross(axis,[1.,0.,0.]);r/=np.linalg.norm(r)
        p=np.array([.5*(a+b)+.01*np.linalg.norm(b-a)*r])
        matrix=horseshoe_influence(wing,state.solution.wake_direction,p,source_core_speed=np.linalg.norm(wind),source_core_policy='source_pinned_v1')
        assert_allclose(backend.induced_velocity(state,p),np.einsum('ijk,j->ik',matrix,state.solution.circulation),rtol=1e-12,atol=1e-12)
        for policy in ['none','source_legacy']:
            other=AeroBackend('vsm',vsm_core_policy=policy)
            with self.assertRaises(ValueError):other.induced_velocity(state,p)
            with self.assertRaises(ValueError):other.advance(state,polar,wind)
            other_matrix=horseshoe_influence(wing,state.solution.wake_direction,p,
                source_core_speed=None if policy=='none' else np.linalg.norm(wind),source_core_policy=policy)
            self.assertGreater(np.max(abs(matrix-other_matrix)),1e-3)
        again=backend.advance(state,polar,wind)
        self.assertEqual(again.core_policy,'source_pinned_v1');self.assertEqual(before,state.solution.circulation.tobytes())


class CoreArchiveTests(unittest.TestCase):
    def setUp(self):
        self.parent=ROOT/'local/core-profile-tests';self.parent.mkdir(exist_ok=True,parents=True)
        self.root=self.parent/uuid4().hex;self.root.mkdir()

    def tearDown(self):
        if self.root.resolve().parent!=self.parent.resolve():raise ValueError('cleanup escaped test workspace')
        shutil.rmtree(self.root)

    def make(self,policy):
        output=self.root/policy
        backend=AeroBackend('vsm',vsm_core_policy=policy,vsm_solve=NEWTON)
        state=backend.start(Wing.ellipse(4),SectionPolar.linear(),[5.,0.,.5])
        writer=AeroWriter(output,dict(aero_method='vsm',name='core contract',aerodynamics=backend.metadata()))
        writer.append(state);writer.finish()
        return output

    def rehash(self,p):
        m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
        m['wake_sha256']=sha256_file(p/'wake.h5');m['case_sha256']=sha256_file(p/'case.json')
        (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')

    def test_all_policies_roundtrip_and_catalog(self):
        for policy in ['none','source_legacy','source_pinned_v1']:
            p=self.make(policy);reader=AeroReader(p)
            try:
                frame=reader.frame(0);self.assertEqual(frame['vsm_core_policy'],policy)
                self.assertEqual(frame['vsm_core_kernel_identity'],core_kernel_identity(policy))
            finally:reader.close()
            self.assertEqual(inspect_run(p).core,'VSM / '+policy)

    def test_unknown_missing_and_case_tamper_rejected(self):
        p=self.make('source_pinned_v1')
        with h5py.File(p/'wake.h5','r+') as f:f['frames/000000'].attrs['vsm_core_policy']='unknown'
        self.rehash(p)
        with self.assertRaises(ValueError):AeroReader(p)
        with h5py.File(p/'wake.h5','r+') as f:del f['frames/000000'].attrs['vsm_core_policy']
        self.rehash(p)
        with self.assertRaises(ValueError):AeroReader(p)
        with h5py.File(p/'wake.h5','r+') as f:f['frames/000000'].attrs['vsm_core_policy']='source_pinned_v1'
        c=json.loads((p/'case.json').read_text(encoding='utf-8'));c['aerodynamics']['core_policy']='none'
        (p/'case.json').write_text(json.dumps(c)+'\n',encoding='utf-8');self.rehash(p)
        with self.assertRaises(ValueError):AeroReader(p)

    def test_old_unmarked_archive_reports_unrecorded(self):
        p=self.make('source_legacy')
        with h5py.File(p/'wake.h5','r+') as f:
            for k in ['vsm_core_policy','vsm_core_kernel_identity']:
                del f.attrs[k];del f['frames/000000'].attrs[k]
        m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
        for k in ['vsm_core_policy','vsm_core_kernel_identity']:del m[k]
        (p/'manifest.json').write_text(json.dumps(m)+'\n',encoding='utf-8');self.rehash(p)
        reader=AeroReader(p)
        try:self.assertEqual(reader.frame(0)['vsm_core_policy'],'unrecorded')
        finally:reader.close()
        self.assertEqual(inspect_run(p).core,'VSM / unrecorded')

    def test_identity_tamper_and_orphan_markers_rejected(self):
        for location in ['frame','root','manifest','orphan_frame','orphan_root','orphan_manifest']:
            with self.subTest(location=location):
                p=self.make('none')
                if location.startswith('orphan'):
                    with h5py.File(p/'wake.h5','r+') as f:
                        for k in ['vsm_core_policy','vsm_core_kernel_identity']:
                            del f.attrs[k];del f['frames/000000'].attrs[k]
                        if location=='orphan_frame':f['frames/000000'].attrs['vsm_core_kernel_identity']='orphan'
                        if location=='orphan_root':f.attrs['vsm_core_kernel_identity']='orphan'
                    m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
                    for k in ['vsm_core_policy','vsm_core_kernel_identity']:del m[k]
                    if location=='orphan_manifest':m['vsm_core_kernel_identity']='orphan'
                    (p/'manifest.json').write_text(json.dumps(m)+'\n',encoding='utf-8')
                elif location=='manifest':
                    m=json.loads((p/'manifest.json').read_text(encoding='utf-8'));m['vsm_core_kernel_identity']='wrong'
                    (p/'manifest.json').write_text(json.dumps(m)+'\n',encoding='utf-8')
                else:
                    with h5py.File(p/'wake.h5','r+') as f:
                        group=f if location=='root' else f['frames/000000']
                        group.attrs['vsm_core_kernel_identity']='wrong'
                self.rehash(p)
                with self.assertRaises(ValueError):AeroReader(p)
                shutil.rmtree(p)

    def test_writer_rejects_case_mismatch_and_policy_switch(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=[5.,0.,.5]
        legacy=AeroBackend('vsm',vsm_solve=NEWTON);pinned=AeroBackend('vsm',vsm_core_policy='source_pinned_v1',vsm_solve=NEWTON)
        a=legacy.start(wing,polar,wind);b=pinned.advance(pinned.start(wing,polar,wind),polar,wind)
        writer=AeroWriter(self.root/'mismatch',dict(aero_method='vsm',aerodynamics=dict(core_policy='none')))
        try:
            with self.assertRaises(ValueError):writer.append(a)
        finally:writer.file.close()
        writer=AeroWriter(self.root/'switch',dict(aero_method='vsm'))
        try:
            writer.append(a)
            with self.assertRaises(ValueError):writer.append(b)
        finally:writer.file.close()


if __name__=='__main__':unittest.main()
