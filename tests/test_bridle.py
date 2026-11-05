"""Bridle point of the kite (G4 contract section 3 item 4): geometry, the source's expressions, the case file."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from openfawe.flight.kinematics import Pose
from openfawe.tether.bridle import Bridle, load_bridle

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"cases/aircraft/m600_makani_rigid_body.json"
TO_MAKANI = np.diag([-1., 1., -1.])                       # OpenFAWE body axes <-> Makani body axes (its own inverse)


def turned(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    about_x = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    about_y = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    about_z = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    return about_z@about_y@about_x


def source_bridle_point(Xg, Vb, omega_b, dcm_g2b, pivot_b, radius, Xg_node, Vg_node):
    """Wing::CalcBridlePoint of the source written out line by line, in the source's own quantities.

    ``Vb`` and ``omega_b`` are the body-origin velocity and the angular velocity in body components,
    ``dcm_g2b`` turns ground components into body components.
    """
    y_axis = dcm_g2b[1]
    Xg_pivot = Xg+dcm_g2b.T@pivot_b
    Rg = Xg_node-Xg_pivot
    knot = Rg-np.dot(y_axis, Rg)*y_axis
    norm = max(np.linalg.norm(knot), 1e-9)
    knot = knot/np.linalg.norm(knot)
    Xg_point = Xg_pivot+radius*knot
    Vb_pivot = Vb+np.cross(omega_b, pivot_b)
    Vb_from_pivot = dcm_g2b@Vg_node-Vb_pivot
    plane = Vb_from_pivot.copy()
    plane[1] = 0.
    Rb_knot = dcm_g2b@knot
    Vb_point = plane-np.dot(Rb_knot, plane)*Rb_knot
    Vb_point = Vb_pivot+radius/norm*Vb_point
    return Xg_point, dcm_g2b.T@Vb_point


class BridleTests(unittest.TestCase):
    def setUp(self):
        self.bridle = Bridle([0.1494, -0.5091, -0.13035], 4.786)
        self.pose = Pose(0., np.array([40., -15., 120.]), turned(0.3, -0.5, 0.8), np.array([30., 12., -4.]), np.array([0.2, -0.4, 0.1]))
        self.node, self.node_velocity = np.array([20., -30., 95.]), np.array([25., 10., -6.])

    def test_the_point_is_at_the_bridle_radius_in_the_plane_normal_to_the_span_axis(self):
        point, _ = self.bridle.point(self.pose, self.node, self.node_velocity)
        pivot, pivot_velocity = self.bridle.pivot(self.pose)
        span = self.pose.rotation[:, 1]
        arm = point-pivot
        self.assertAlmostEqual(np.linalg.norm(arm), 4.786, places=12)
        self.assertAlmostEqual(float(arm@span), 0., places=12)
        to_node = self.node-pivot
        in_plane = to_node-(to_node@span)*span
        assert_allclose(np.cross(arm, in_plane), 0., atol=1e-10)                 # same direction ...
        self.assertGreater(float(arm@in_plane), 0.)                              # ... and the same sense
        assert_allclose(pivot, self.pose.position_m+self.pose.rotation@self.bridle.pivot_body_m, atol=1e-13)
        assert_allclose(pivot_velocity, self.pose.point_velocity(pivot[None, :])[0], atol=1e-13)

    def test_position_and_velocity_equal_the_source_written_in_its_own_axes(self):
        point, velocity = self.bridle.point(self.pose, self.node, self.node_velocity)
        # The source works in Makani body axes; turn every body quantity into them (the inertial frame is free).
        dcm_g2b = (self.pose.rotation@TO_MAKANI).T
        expected_point, expected_velocity = source_bridle_point(
            self.pose.position_m, dcm_g2b@self.pose.velocity_m_s, dcm_g2b@self.pose.angular_velocity_rad_s, dcm_g2b,
            TO_MAKANI@self.bridle.pivot_body_m, self.bridle.radius_m, self.node, self.node_velocity)
        assert_allclose(point, expected_point, rtol=0, atol=1e-12)
        assert_allclose(velocity, expected_velocity, rtol=0, atol=1e-12)

    def test_without_rotation_of_the_body_the_velocity_is_the_time_derivative_of_the_position(self):
        steady = Pose(0., self.pose.position_m, self.pose.rotation, self.pose.velocity_m_s, np.zeros(3))
        node_acceleration = np.array([3., -2., 5.])

        def at(time):
            pose = Pose(time, steady.position_m+time*steady.velocity_m_s, steady.rotation, steady.velocity_m_s, np.zeros(3))
            node = self.node+time*self.node_velocity+.5*time*time*node_acceleration
            return self.bridle.point(pose, node, self.node_velocity+time*node_acceleration)
        h = 1e-5
        _, velocity = at(0.)
        assert_allclose(velocity, (at(h)[0]-at(-h)[0])/(2*h), rtol=0, atol=1e-7)
        # With the body turning, the source's expression leaves out the turning of the swing plane with the body:
        # here the body rolls about the inertial x axis and the node stands still.
        rate = 0.4

        def rolling(time):
            c, s = np.cos(rate*time), np.sin(rate*time)
            rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])@steady.rotation
            pose = Pose(time, steady.position_m, rotation, np.zeros(3), np.array([rate, 0., 0.]))
            return self.bridle.point(pose, self.node, np.zeros(3))
        _, stated = rolling(0.)
        true = (rolling(h)[0]-rolling(-h)[0])/(2*h)
        self.assertGreater(np.linalg.norm(true-stated), 0.05*np.linalg.norm(true))

    def test_a_node_on_the_pivot_line_gives_a_finite_point(self):
        pivot, _ = self.bridle.pivot(self.pose)
        point, velocity = self.bridle.point(self.pose, pivot+2.*self.pose.rotation[:, 1], np.zeros(3))
        self.assertTrue(np.all(np.isfinite(point)) and np.all(np.isfinite(velocity)))
        self.assertLessEqual(np.linalg.norm(point-pivot), 4.786*(1.+1e-12))      # no direction is defined; nothing blows up

    def test_the_case_file_holds_the_source_values_in_openfawe_axes(self):
        bridle = load_bridle(CASE)
        port, starboard = np.array([-0.1494, -5.8843, 0.13035]), np.array([-0.1494, 5.8661, 0.13035])     # config/m600/wing.py
        pivot = .5*(port+starboard)+np.array([0., -0.5, 0.])
        assert_allclose(bridle.pivot_body_m, TO_MAKANI@pivot, rtol=0, atol=1e-15)
        self.assertEqual(bridle.radius_m, 4.7860)
        with self.assertRaises(ValueError):
            Bridle([0., 0., 0.], 0.)


if __name__ == "__main__":
    unittest.main()
