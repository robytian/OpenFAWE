"""M01-M07: source/analytical verification and explicit method ownership."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET
import numpy as np
import h5py

from openfawe.aero.wing import Wing,SectionPolar,SolveSettings,CirculationConvergenceError
from openfawe.aero.vortex import segment_velocity,FilamentSingularity
from openfawe.aero.vsm import solve_vsm,semi_infinite_velocity,geometry
from openfawe.aero.methods import AeroBackend,AeroMethod
from openfawe.aero.wake import WakeConfig,start_wake,advance_wake
from openfawe.aero_demo import run_demo
from openfawe.aero_archive import AeroReader,AeroWriter,LEGACY_SCHEMA
from openfawe.aero_vtk import frame_network
from openfawe.provenance import sha256_file


def fixtures():
    return json.loads((Path(__file__).parent/'fixtures/vsm_source.json').read_text())['cases']


def coefficients(solution,wind,area):
    wind=np.asarray(wind);direction=wind/np.linalg.norm(wind)
    lift=np.cross(direction,[0,1,0]);lift/=np.linalg.norm(lift)
    q=.5*1.225*np.dot(wind,wind)*area
    return np.array([solution.force@lift/q,solution.force@direction/q])


class VsmVerification(unittest.TestCase):
    def test_m01_m02_original_source_cases(self):
        for c in fixtures():
            with self.subTest(case=c['name']):
                wing=Wing(c['leading_edge'],c['trailing_edge'])
                compatibility=c['name'] in ('nonlinear_ar12_3deg','nonlinear_ar12_6deg')
                polar=SectionPolar(*np.array(c['polar']).T,bounds='clamp' if compatibility else 'error')
                result=solve_vsm(wing,polar,c['inflow_m_s'],settings=SolveSettings(relaxation=.03,max_iterations=2000))
                np.testing.assert_allclose(result.circulation,c['circulation_m2_s'],rtol=5e-5,atol=5e-5)
                cl,cd=coefficients(result,c['inflow_m_s'],c['area_m2'])
                self.assertLess(abs(cl-c['cl']),1e-4)
                self.assertLess(abs(cd-c['cd']),2e-5)
                np.testing.assert_allclose(result.collocation_points,c['collocation_points'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(wing.cp,c['load_points'],rtol=0,atol=1e-12)
                self.assertLessEqual(result.residual,result.residual_limit)

    def test_m02_source_clamping_is_explicit_and_default_rejects(self):
        for c in fixtures()[2:4]:
            wing=Wing(c['leading_edge'],c['trailing_edge'])
            polar=SectionPolar(*np.array(c['polar']).T)
            with self.assertRaisesRegex(ValueError,'outside'):
                solve_vsm(wing,polar,c['inflow_m_s'])
        with self.assertRaises(ValueError): SectionPolar([-10,10],[-1,1],[0,0],[0,0],bounds='guess')

    def test_m03_original_ellipse_analytical_tolerances(self):
        for c in fixtures()[:2]:
            wing=Wing(c['leading_edge'],c['trailing_edge'])
            result=solve_vsm(wing,SectionPolar(*np.array(c['polar']).T),c['inflow_m_s'])
            cl,cd=coefficients(result,c['inflow_m_s'],c['area_m2'])
            ar=15.709**2/c['area_m2'];alpha=np.arctan2(c['inflow_m_s'][2],c['inflow_m_s'][0])
            theory=2*np.pi*alpha/(1+2/ar)
            self.assertLess(abs(cl-theory),.1)
            self.assertLess(abs(cd-theory**2/(np.pi*ar)),.001)

    def test_m04_semi_infinite_long_segment_limit_and_singularities(self):
        points=np.array([[1,2,3],[-2,1,4],[3,-2,1.]])
        values=semi_infinite_velocity([0,0,0],[1,0,0],points)
        exact=np.array([segment_velocity([0,0,0],[1e6,0,0],p,1.) for p in points])
        np.testing.assert_allclose(values,exact,rtol=1e-6,atol=1e-10)
        np.testing.assert_array_equal(semi_infinite_velocity([0,0,0],[1,0,0],[[-1,0,0]]),0)
        with self.assertRaises(FilamentSingularity):semi_infinite_velocity([0,0,0],[1,0,0],[[1,0,0]])
        with self.assertRaises(ValueError):semi_infinite_velocity([0,0,0],[0,0,0],points)

    def test_m04_covariant_force_moment_and_symmetry(self):
        wing=Wing.ellipse(12);wind=np.array([10.,0.,1.])
        alpha=np.array([-20.,0.,20.])
        polar=SectionPolar(alpha,np.deg2rad(alpha)*2*np.pi,[.02]*3,[.03]*3)
        a=solve_vsm(wing,polar,wind)
        angle=.53;rotation=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        shift=np.array([3.,2.,1.]);moved=wing.transformed(rotation,shift)
        b=solve_vsm(moved,polar,rotation@wind)
        np.testing.assert_allclose(b.force,rotation@a.force,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(b.moment(moved),rotation@a.moment(wing)+np.cross(shift,b.force),rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(a.circulation,a.circulation[::-1],rtol=1e-10,atol=1e-10)
        self.assertAlmostEqual(a.force[1],0,places=10)
        zero=solve_vsm(wing,SectionPolar.linear(),[0,0,0])
        np.testing.assert_array_equal(zero.force,0)
        with self.assertRaises(ValueError):solve_vsm(wing,polar,[0,1,0])

    def test_m05_nonconvergence_input_ownership(self):
        wing=Wing.ellipse(8);initial=np.zeros(8);old=wing.leading_edge.copy()
        with self.assertRaises(CirculationConvergenceError):
            solve_vsm(wing,SectionPolar.linear(),[10,0,1],initial=initial,settings=SolveSettings(max_iterations=1))
        np.testing.assert_array_equal(initial,0);np.testing.assert_array_equal(wing.leading_edge,old)
        result=solve_vsm(wing,SectionPolar.linear(),[10,0,1])
        with self.assertRaises(ValueError):result.circulation.flags.writeable=True


class MethodContractTests(unittest.TestCase):
    def test_m05_selection_and_cross_method_state_rejection(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=[5,0,.5]
        a=AeroBackend('vsm');b=AeroBackend()
        self.assertEqual(b.method,AeroMethod.FREE_WAKE)
        with self.assertRaises(ValueError):AeroBackend('VSM_typo')
        with self.assertRaises(ValueError):AeroBackend('vsm',WakeConfig(free_wake=False))
        sa=a.start(wing,polar,wind);sb=b.start(wing,polar,wind)
        with self.assertRaises(ValueError):a.advance(sb,polar,wind)
        with self.assertRaises(ValueError):b.advance(sa,polar,wind)
        with self.assertRaises(ValueError):a.advance(sa,polar,wind,next_time_s=4)
        self.assertEqual(sa.step,0)

    def test_m05_method_specific_inflow_sampling(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear()
        for mode in AeroMethod:
            backend=AeroBackend(mode);seen=[]
            def wind(points,time):
                seen.append(points.copy());return np.tile([5.,0.,.5],(len(points),1))
            backend.start(wing,polar,wind)
            np.testing.assert_array_equal(seen[0],backend.evaluation_points(wing))
        self.assertGreater(np.max(np.abs(AeroBackend('vsm').evaluation_points(wing)-wing.cp)),.1)

    def test_m06_vsm_has_no_path_history(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear();backend=AeroBackend('vsm');wind=[5,0,.5]
        a=backend.start(wing,polar,wind)
        b=backend.start(wing.transformed(np.eye(3),[0,0,2]),polar,[6,0,.2])
        aa=backend.advance(a,polar,wind,next_wing=wing)
        bb=backend.advance(b,polar,wind,next_wing=wing)
        np.testing.assert_array_equal(aa.solution.force,bb.solution.force)
        self.assertEqual(aa.free_rows.shape,(0,7,3));self.assertEqual(a.step,0)

    def test_m06_free_wake_wrapper_preserves_original_results(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=[5,0,.5];backend=AeroBackend()
        direct=start_wake(wing,polar,backend.wake,wind);wrapped=backend.start(wing,polar,wind)
        for _ in range(3):
            direct=advance_wake(direct,polar,backend.wake,wind)
            wrapped=backend.advance(wrapped,polar,wind)
        np.testing.assert_array_equal(wrapped.free_rows,direct.free_rows)
        np.testing.assert_array_equal(wrapped.solution.force,direct.solution.force)


class MethodArchiveTests(unittest.TestCase):
    def setUp(self):
        self.parent=Path(__file__).resolve().parents[1]/'local'/'method-tests'
        self.parent.mkdir(parents=True,exist_ok=True);self.root=self.parent/uuid4().hex;self.root.mkdir()

    def tearDown(self):
        if self.root.resolve().parent!=self.parent.resolve():raise ValueError('invalid cleanup path')
        shutil.rmtree(self.root.resolve())

    def test_m07_both_modes_archive_and_export(self):
        for mode in AeroMethod:
            output=self.root/mode.value
            result=run_demo(output,panels=4,dt_s=.1,duration_s=.3,aero_method=mode,export_vtk=True)
            reader=AeroReader(output)
            try:
                self.assertEqual(reader.method,mode.value);frame=reader.frame(3)
                self.assertEqual(frame['aero_method'],mode.value)
                self.assertEqual(result['wake_rows'],0 if mode in (AeroMethod.VSM,AeroMethod.VLM,AeroMethod.LL_STRAIGHT) else 3)
                self.assertEqual(len(frame['free_rows']),result['wake_rows'])
                if mode in (AeroMethod.VLM,AeroMethod.VLM_FREE_WAKE):
                    self.assertNotIn('polar_clipped',frame)
                    self.assertIn('lattice_circulation_m2_s',frame)
                else:
                    self.assertFalse(np.any(frame['polar_clipped']))
                nodes,edges,gamma=frame_network(frame)
                self.assertTrue(np.isfinite(nodes).all());self.assertLess(edges.max(),len(nodes))
                self.assertEqual(len(edges),len(gamma))
                piece=ET.parse(output/'paraview/wake-000003.vtp').getroot().find('./PolyData/Piece')
                self.assertEqual(int(piece.attrib['NumberOfLines']),len(edges))
                if mode==AeroMethod.VSM:
                    self.assertIn('polar_alpha_rad',frame)
                    self.assertGreater(np.max(np.abs(frame['polar_alpha_rad']-frame['alpha_rad'])),1e-5)
                    before=frame['panel_force'].copy();frame['vsm_display_wake_length_m']=200
                    self.assertGreater(frame_network(frame)[0][:,0].max(),100)
                    np.testing.assert_array_equal(frame['panel_force'],before)
            finally:reader.close()

    def test_m07_unknown_mode_creates_no_output(self):
        with self.assertRaises(ValueError):run_demo(self.root/'bad',aero_method='unknown')
        self.assertFalse((self.root/'bad').exists())

    def test_m07_legacy_v1_read_compatibility(self):
        output=self.root/'legacy';run_demo(output,panels=4,dt_s=.1,duration_s=.2)
        case=json.loads((output/'case.json').read_text());case.pop('aero_method');case.pop('aerodynamics')
        (output/'case.json').write_text(json.dumps(case),encoding='utf-8')
        with h5py.File(output/'wake.h5','r+') as f:
            f.attrs['schema']=LEGACY_SCHEMA;del f.attrs['aero_method']
            for group in f['frames'].values():del group.attrs['aero_method']
        manifest=json.loads((output/'manifest.json').read_text());manifest['schema']=LEGACY_SCHEMA
        manifest.pop('aero_method');manifest['case_sha256']=sha256_file(output/'case.json');manifest['wake_sha256']=sha256_file(output/'wake.h5')
        (output/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        reader=AeroReader(output)
        try:self.assertEqual(reader.frame(2)['aero_method'],'free_wake')
        finally:reader.close()

    def test_m07_wrong_state_cannot_be_archived(self):
        state=AeroBackend('vsm').start(Wing.ellipse(4),SectionPolar.linear(),[5,0,.5])
        writer=AeroWriter(self.root/'wrong',{'aero_method':'free_wake'})
        try:
            with self.assertRaises(ValueError):writer.append(state)
        finally:writer.fail(ValueError('intentional mismatch'))


if __name__=='__main__':unittest.main()
