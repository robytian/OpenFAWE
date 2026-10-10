"""Source identity, original-code comparisons and real-aircraft input guards."""
import json
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from openfawe.aero.aircraft import load_aircraft
from openfawe.aero.reference import inflow,load_coefficients,refine_aircraft,source_curve
from openfawe.aero.vsm import solve_vsm,geometry

ROOT=Path(__file__).resolve().parents[1]


class ReferenceAircraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3=load_aircraft(ROOT/'cases/aircraft/v3_cad_attached.json')
        cls.m600=load_aircraft(ROOT/'cases/aircraft/m600_neutral.json')

    def test_r01_m600_actual_reference_lines_and_proper_frame(self):
        raw=json.loads(self.m600.raw_json);p=raw['provenance'];R=np.array(p['source_to_target'])
        assert_allclose(R.T@R,np.eye(3),atol=1e-15);self.assertEqual(np.linalg.det(R),1.)
        self.assertEqual(self.m600.wing.panels,95);self.assertEqual(len(self.m600.wing.wings),9)
        for source,wing in zip(p['source_surfaces'],self.m600.wing.wings):
            q=np.array(source['quarter_chord']);q=q[::-1] if source['reverse'] else q
            assert_allclose(wing.bound,q,rtol=0,atol=1e-12)
            reconstructed=wing.bound[::-1] if source['reverse'] else wing.bound
            reconstructed=reconstructed@R-np.array(source['origin'])
            assert_allclose(reconstructed,np.array(source['raw_nodes'])[:,:3],rtol=0,atol=1e-12)

    def test_r01_neutral_control_matches_original_source_knots(self):
        tables=json.loads((ROOT/'cases/reference/m600_airfoil_tables.json').read_text())['tables']
        actual=json.loads(self.m600.raw_json)['polars']
        for ident,rows in tables.items():
            x=np.array([r['control'] for r in rows]);a=np.array([r['data'] for r in rows])
            for key,col in [('cl',1),('cd',2),('cm',3)]:
                assert_allclose(actual[ident][key],[np.interp(0.,x,a[:,i,col]) for i in range(a.shape[1])],atol=1e-12,rtol=0)

    def test_r01_v3_negative_drag_source_is_rejected_not_clipped(self):
        with self.assertRaisesRegex(ValueError,'Cd must be nonnegative'):
            load_aircraft(ROOT/'cases/aircraft/v3_cad.json')
        for polar in self.v3.polar.polars:
            self.assertEqual(polar.alpha_deg[-1],21.5)
            with self.assertRaises(ValueError):polar.coefficients(np.deg2rad(22))

    def test_r02_five_original_v3_circulation_cases(self):
        reference=json.loads((ROOT/'tests/fixtures/v3_reference.json').read_text())
        before=self.v3.wing.bound.copy()
        for case in reference['cases']:
            with self.subTest(alpha=case['alpha_deg'],beta=case['beta_deg']):
                result=solve_vsm(self.v3.wing,self.v3.polar,case['inflow_m_s'])
                assert_allclose(result.circulation,case['circulation'],rtol=5e-5,atol=5e-5)
                assert_allclose(geometry(self.v3.wing)[0],case['collocation_points'],atol=1e-12,rtol=0)
                assert_allclose(result.cl,case['cl_section'],rtol=5e-5,atol=5e-5)
        assert_allclose(self.v3.wing.bound,before,atol=0,rtol=0)

    def test_r07_refinement_preserves_source_stations_domain_and_geometry(self):
        refined=refine_aircraft(self.v3,2)
        self.assertEqual(refined.wing.panels,72)
        assert_allclose(refined.wing.wings[0].leading_edge[::2],self.v3.wing.leading_edge,atol=0,rtol=0)
        assert_allclose(refined.wing.wings[0].trailing_edge[::2],self.v3.wing.trailing_edge,atol=0,rtol=0)
        for p in refined.polar.polars:self.assertEqual(p.alpha_deg[-1],21.5)
        self.assertIs(refine_aircraft(self.v3,1),self.v3)
        with self.assertRaises(ValueError):refine_aircraft(refined,2)
        for n in [True,0,-1,1.5]:
            with self.assertRaises(ValueError):refine_aircraft(self.v3,n)

    def test_r07_wind_axes_and_moment_origin(self):
        wing=self.v3.wing;wind=inflow(20,6,5);result=solve_vsm(wing,self.v3.polar,wind)
        c=load_coefficients(result,wing,wind,20,2,reference=[0,0,0])
        d=load_coefficients(result,wing,wind,20,2,reference=[1,2,3])
        assert_allclose([d[k] for k in ['CL','CD','CS']],[c[k] for k in ['CL','CD','CS']],atol=0,rtol=0)
        shift=-np.cross([1,2,3],result.force)/(.5*1.225*20**2*20*2)
        assert_allclose([d[k]-c[k] for k in ['CMx','CMy','CMz']],shift,atol=1e-12)
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        moved=wing.transformed(rotation,[2,3,4]);s=solve_vsm(moved,self.v3.polar,wind@rotation.T)
        rotated=load_coefficients(s,moved,wind@rotation.T,20,2,reference=[2,3,4],body_y=rotation[:,1])
        assert_allclose([rotated[k] for k in ['CL','CD','CS']],[c[k] for k in ['CL','CD','CS']],atol=1e-10)

    def test_r07_original_de2019_wing_curve_and_bounds(self):
        source=json.loads((ROOT/'cases/reference/de2019_source.json').read_text())
        table=source['parameters']['initAircraft'];x=np.array(table['alpha'])
        result=source_curve(source,x)
        assert_allclose(result['CL'],table['wing_cL_Static'],atol=0,rtol=0)
        assert_allclose(result['CD'],table['wing_cD_Static'],atol=0,rtol=0)
        with self.assertRaises(ValueError):source_curve(source,x[0]-1)

    def test_r07_reference_inflow_and_units_guards(self):
        assert_allclose(np.linalg.norm(inflow(20,6,5)),20)
        for values in [(0,0,0),(20,90,0),(20,0,90),(20,np.nan,0)]:
            with self.assertRaises(ValueError):inflow(*values)


if __name__=='__main__':unittest.main()
