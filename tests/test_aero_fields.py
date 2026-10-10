"""M14 off-body field reconstruction and rigid motion at method-specific points."""
import unittest
import numpy as np
from openfawe.aero.methods import AeroBackend, rigid_velocity
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.wake import WakeConfig, wake_network
from openfawe.aero.vortex import segment_velocity, FilamentSingularity
from openfawe.aero.cores import edge_radii
from openfawe.aero.vsm import semi_infinite_velocity
from openfawe.aero.vsm import solve_vsm


class AeroFieldTests(unittest.TestCase):
    def test_vsm_public_defaults_match_backend_on_finer_mesh(self):
        for panels in (20,40,80):
            wing=Wing.ellipse(panels);polar=SectionPolar.linear()
            direct=solve_vsm(wing,polar,[1,0,.1])
            selected=AeroBackend('vsm').start(wing,polar,[1,0,.1]).solution
            self.assertLessEqual(direct.residual,direct.residual_limit)
            self.assertLess(np.max(np.abs(np.rad2deg(direct.polar_alpha_rad))),20.)
            np.testing.assert_array_equal(direct.circulation,selected.circulation)

    def test_m14_field_matches_explicit_filaments(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear();points=np.array([[2.,.2,1.],[3.,-1.,.5]])
        for method in ('free_wake','vsm'):
            backend=AeroBackend(method,vsm_core_policy='none');state=backend.start(wing,polar,[1,0,.1])
            state=backend.advance(state,polar,[1,0,.1]);expected=np.zeros_like(points)
            if method=='free_wake':
                nodes,edges,gamma=wake_network(state)
                radii=edge_radii(wing,edges,state.cores,backend.wake.core_radius_m)
                for (a,b),g,r in zip(edges,gamma,radii):
                    expected+=np.array([segment_velocity(nodes[a],nodes[b],p,g,
                        model=backend.wake.core_model,core_radius=r) for p in points])
            else:
                for (a,b),g in zip(wing.panel_edges,state.solution.circulation):
                    for start,end in [(wing.bound[a],wing.bound[b]),(wing.bound[b],wing.trailing_edge[b]),(wing.trailing_edge[a],wing.bound[a])]:
                        expected+=np.array([segment_velocity(start,end,p,g) for p in points])
                    expected+=g*(semi_infinite_velocity(wing.trailing_edge[b],state.solution.wake_direction,points)-
                                 semi_infinite_velocity(wing.trailing_edge[a],state.solution.wake_direction,points))
            np.testing.assert_allclose(backend.induced_velocity(state,points),expected,rtol=1e-10,atol=1e-10)
            with self.assertRaises(ValueError):backend.induced_velocity(state,[1,2,3])
            with self.assertRaises(FilamentSingularity):backend.induced_velocity(state,wing.bound[:1])

    def test_m14_rigid_motion_and_field_frame_covariance(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear();rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
        shift=np.array([3.,5.,2.]);points=np.array([[3,.2,1.],[2,-1,.5]])
        for method in ('free_wake','vsm'):
            backend=AeroBackend(method,wake=WakeConfig(core_evolution='age_stretch' if method=='free_wake' else 'constant'))
            velocity=rigid_velocity(backend.evaluation_points(wing),[0,0,.02],[0,.001,0])
            moved=wing.transformed(rotation,shift)
            velocity2=rigid_velocity(backend.evaluation_points(moved),np.array([0,0,.02])@rotation.T,
                                     np.array([0,.001,0])@rotation.T,shift)
            np.testing.assert_allclose(velocity2,velocity@rotation.T,atol=1e-12)
            a=backend.start(wing,polar,[1,0,.1],structural_velocity=velocity)
            b=backend.start(moved,polar,np.array([1,0,.1])@rotation.T,structural_velocity=velocity2)
            np.testing.assert_allclose(b.solution.force,a.solution.force@rotation.T,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(backend.induced_velocity(b,points@rotation.T+shift),
                                       backend.induced_velocity(a,points)@rotation.T,rtol=1e-9,atol=1e-9)
        with self.assertRaises(ValueError):rigid_velocity([1,2,3],[0,0,0],[0,0,0])
        with self.assertRaises(ValueError):AeroBackend('free_wake').induced_velocity(a,points)


if __name__=='__main__':unittest.main()
