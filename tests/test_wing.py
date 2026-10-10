"""W01-W06 checks against frozen criteria and pinned upstream data."""

import json
from pathlib import Path
import unittest
import numpy as np

from openfawe.aero.wing import (
    Wing, SectionPolar, SolveSettings, CirculationConvergenceError,
    full_cosine_control_points, cp_to_nodes, influence_matrix,
    solve_circulation, solve_steady,
)


def source_fixture():
    data = json.loads((Path(__file__).parent/"fixtures/olaf_ellipse.json").read_text())
    y, chord = np.array(data["span_nodes_m"]), np.array(data["chord_nodes_m"])
    wing = Wing(np.c_[-.25*chord,y,0*y], np.c_[.75*chord,y,0*y])
    return wing, SectionPolar(*np.array(data["polar_alpha_cl_cd_cm"]).T), data


def force_coefficients(solution, wind, area):
    direction = np.array(wind)/np.linalg.norm(wind)
    lift = np.cross(direction, [0,1,0]); lift /= np.linalg.norm(lift)
    pressure_area = .5*1.225*np.dot(wind,wind)*area
    return solution.force@lift/pressure_area, solution.force@direction/pressure_area


class WingTests(unittest.TestCase):
    def test_w01_geometry_and_full_cosine_coordinates(self):
        wing = Wing([[-.25,0,0],[-.25,1,0],[-.25,3,0]], [[.75,0,0],[.75,1,0],[.75,3,0]])
        np.testing.assert_allclose(wing.cp_span, [1/3,5/3], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(wing.area,[1,2],rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(wing.tangent,[[1,0,0]]*2)
        np.testing.assert_allclose(wing.normal,[[0,0,1]]*2)
        np.testing.assert_allclose(wing.cp[:,0],0,atol=1e-13)
        np.testing.assert_allclose(full_cosine_control_points([0,1,2,3]),[.5,1.5,2.5])
        np.testing.assert_allclose(full_cosine_control_points([0,3]),[1.5])
        np.testing.assert_allclose(cp_to_nodes([.5,1.5],[2,4],[0,1,2]),[1,3,5])

    def test_w02_residual_zero_and_pure_span_flow(self):
        wing=Wing.ellipse(8);polar=SectionPolar.linear(); matrix=influence_matrix(wing)
        for wind in ([0,0,0],[0,3,0]):
            solution=solve_circulation(wing,polar,wind,matrix)
            np.testing.assert_array_equal(solution.circulation,0)
            np.testing.assert_array_equal(solution.force,0)
            self.assertLessEqual(solution.residual,solution.residual_limit)
        solution=solve_steady(wing,polar,[1,0,.1])
        self.assertLessEqual(solution.residual,solution.residual_limit)

    def test_w02_nonconvergence_does_not_modify_inputs(self):
        wing=Wing.ellipse(8); seed=np.zeros(8);before=wing.leading_edge.copy()
        with self.assertRaises(CirculationConvergenceError) as caught:
            solve_steady(wing,SectionPolar.linear(),[1,0,.1],initial=seed,settings=SolveSettings(max_iterations=1))
        self.assertGreater(caught.exception.residual,caught.exception.limit)
        np.testing.assert_array_equal(seed,0);np.testing.assert_array_equal(wing.leading_edge,before)

    def test_w03_elliptic_theory_and_mesh_refinement(self):
        wind=[1,0,.1];area=np.pi*5/4;ar=25/area
        theory=2*np.pi*np.arctan(.1)/(1+2/ar)
        results=[]
        for count in (20,40,80):
            result=solve_steady(Wing.ellipse(count),SectionPolar.linear(),wind,wake_length=2000)
            cl,cd=force_coefficients(result,wind,area)
            results.append(cl)
            self.assertLess(abs(cl/theory-1),.02)
            self.assertGreater(cd,0)
            self.assertLess(abs(cd/(cl**2/(np.pi*ar))-1),.05)
        self.assertLess(abs(results[-1]-results[-2]),abs(results[1]-results[0]))

    def test_w04_straight_wake_length_sensitivity(self):
        wing=Wing.ellipse(40);polar=SectionPolar.linear()
        values=[force_coefficients(solve_steady(wing,polar,[1,0,.1],wake_length=spans*5),[1,0,.1],np.pi*5/4)[0] for spans in (100,200,400)]
        self.assertLess(abs(values[2]/values[1]-1),.001)

    def test_w05_exact_original_39_panels_and_historical_nodal_cl(self):
        wing,polar,data=source_fixture();self.assertEqual(wing.panels,39)
        result=solve_steady(wing,polar,data['inflow_m_s'],wake_length=2000)
        nodal=cp_to_nodes(wing.cp_span,result.induced_velocity,wing.span)+data['inflow_m_s']
        cl=polar.coefficients(np.arctan2(nodal[:,2],nodal[:,0]))[0]
        error=cl-np.array(data['last_reference_step']['cl'])
        self.assertLessEqual(np.sqrt(np.mean(error**2)),.003)
        self.assertLessEqual(np.max(np.abs(error)),.01)
        self.assertIn('dirty',data['source_build_description'])

    def test_w06_density_sign_and_moment_reference(self):
        wing=Wing.ellipse(12);polar=SectionPolar.linear();wind=[1,0,.1]
        s=solve_steady(wing,polar,wind);double=solve_steady(wing,polar,wind,density=2.45)
        np.testing.assert_allclose(double.force,2*s.force,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(double.circulation,s.circulation,rtol=1e-12,atol=1e-13)
        reverse=solve_steady(wing,polar,[1,0,-.1])
        np.testing.assert_allclose(reverse.circulation,-s.circulation,rtol=1e-12,atol=1e-13)
        shift=np.array([1,2,3])
        np.testing.assert_allclose(s.moment(wing,shift),s.moment(wing)-np.cross(shift,s.force),atol=1e-12)

    def test_w06_proper_rotation_and_translation(self):
        wing=Wing.ellipse(12);wind=np.array([1,0,.1]);polar=SectionPolar.linear()
        angle=.6;rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        moved=wing.transformed(rot,[2,3,4])
        solution=solve_steady(wing,polar,wind); other=solve_steady(moved,polar,rot@wind)
        np.testing.assert_allclose(other.circulation,solution.circulation,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(other.force,rot@solution.force,rtol=1e-11,atol=1e-12)

    def test_w06_known_2d_force_and_profile_drag(self):
        wing=Wing([[-.25,-1,0],[-.25,0,0],[-.25,1,0]],[[.75,-1,0],[.75,0,0],[.75,1,0]])
        polar=SectionPolar([-10,0,10],[-1,0,1],[.1,.1,.1],[.02,.02,.02])
        angle=np.deg2rad(5);wind=np.array([np.cos(angle),0,np.sin(angle)])*10
        solution=solve_circulation(wing,polar,wind,np.zeros((2,2,3)))
        lift=np.array([-np.sin(angle),0,np.cos(angle)])
        expected=.5*1.225*100*2*(.5*lift+.1*wind/10)
        np.testing.assert_allclose(solution.force,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(solution.panel_couple[:,1],.5*1.225*100*.02,rtol=1e-12)

    def test_w06_readonly_and_invalid_geometry_polar_inputs(self):
        wing=Wing.ellipse(8)
        with self.assertRaises(ValueError): wing.bound.flags.writeable=True
        with self.assertRaises(ValueError): Wing(np.zeros((3,3)),np.zeros((3,3)))
        with self.assertRaises(ValueError): wing.transformed(np.diag([1,-1,1]),[0,0,0])
        with self.assertRaises(ValueError): SectionPolar([0,0],[0,1],[0,0],[0,0])
        with self.assertRaises(ValueError): SectionPolar([0,1],[0,1],[-.1,0],[0,0])
        with self.assertRaises(ValueError): SectionPolar.linear().coefficients(np.deg2rad(21))
        with self.assertRaises(ValueError): SolveSettings(relaxation=0)
        with self.assertRaises(ValueError): solve_steady(wing,SectionPolar.linear(),[0,0,0])


if __name__=='__main__': unittest.main()
