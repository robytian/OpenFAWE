"""OF-0011 revision 6 addendum 62 (F2-Ma): GS02's opt-in moving vessel frame. Default unchanged (no option = the fixed
onshore frame, bit for bit the same as the vessel frame given explicitly as NED); with a moving vessel frame the
anchor and the GSG angles follow the frame as rigid-body kinematics say."""
import math
import unittest
from pathlib import Path

import numpy as np

from openfawe.ground import gs02 as G

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "cases" / "ground" / "makani_gs02_parker_ranch.json"


def make(vessel=None):
    gs = G.GroundStationV2(G.load_gs02_params(PARAMS), vessel_frame=vessel)
    gs.x[:] = [0.4, 0.02, -2.0, 0.3, 0.05, 0.0]
    return gs


def moving_frame():
    dcm = G.angle_to_dcm(0.3, 0.05, -0.04, "zyx")          # yaw, pitch, roll of the buoy (g to v)
    return G.Frame(dcm, np.array([3.0, -2.0, 1.5]), np.array([0.4, -0.1, 0.2]), np.array([0.01, -0.02, 0.03]))


class Default(unittest.TestCase):
    def test_explicit_ned_is_bitwise_the_default(self):
        a, b = make(), make(G.Frame.root())
        for x in (a.x, a.x + 0.1):
            pa, va = a.anchor_pos_vel_ned(x)
            pb, vb = b.anchor_pos_vel_ned(x)
            self.assertTrue(np.array_equal(pa, pb) and np.array_equal(va, vb))
            self.assertEqual(a.tether_dir_to_angles([0.3, -0.5, -0.8], x), b.tether_dir_to_angles([0.3, -0.5, -0.8], x))
        a.set_tether_force([1e4, -2e3, -5e4])
        b.set_tether_force([1e4, -2e3, -5e4])
        self.assertTrue(np.array_equal(a.tether_force_p, b.tether_force_p))
        self.assertIsNone(make().vessel_frame)


class Moving(unittest.TestCase):
    def test_anchor_position_is_carried_by_the_vessel(self):
        fixed, frame = make(), moving_frame()
        moving = make(frame)
        p_fixed, _ = fixed.anchor_pos_vel_ned()      # the anchor in the vessel frame (fixed: vessel = NED)
        p_moving, _ = moving.anchor_pos_vel_ned()
        self.assertTrue(np.allclose(p_moving, frame.to_root("position", p_fixed), rtol=0, atol=1e-12))

    def test_anchor_velocity_adds_the_vessel_motion(self):
        fixed, frame = make(), moving_frame()
        moving = make(frame)
        p_fixed, v_fixed = fixed.anchor_pos_vel_ned()
        _, v_moving = moving.anchor_pos_vel_ned()
        r_g = frame.dcm.T @ p_fixed
        expected = frame.dcm.T @ v_fixed + frame.vel + np.cross(frame.omega, r_g)
        self.assertTrue(np.allclose(v_moving, expected, rtol=0, atol=1e-12))

    def test_gsg_angles_rotate_with_the_vessel(self):
        fixed, frame = make(), moving_frame()
        moving = make(frame)
        d_v = np.array([0.3, -0.5, -0.8])
        d_v /= np.linalg.norm(d_v)
        a = fixed.tether_dir_to_angles(d_v)
        b = moving.tether_dir_to_angles(d_v)        # the same direction in vessel coordinates: same angles
        self.assertTrue(all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(a, b)))

    def test_tether_force_in_platform_frame(self):
        fixed, frame = make(), moving_frame()
        moving = make(frame)
        f_g = np.array([1e4, -2e3, -5e4])
        moving.set_tether_force(f_g)
        fixed.set_tether_force(frame.dcm @ f_g)       # the same force seen in vessel axes
        self.assertTrue(np.allclose(moving.tether_force_p, fixed.tether_force_p, rtol=0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
