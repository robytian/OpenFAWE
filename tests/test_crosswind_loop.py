"""Tests of the transcribed Makani nominal crosswind loop (openfawe.flight.crosswind_loop)."""
import math
import unittest
from pathlib import Path

import numpy as np
from scipy import integrate

from openfawe.flight import crosswind_loop as cl
from openfawe.flight.kinematics import Pose, circular_motion, relative_wind_body, uniform_translation

REPO = Path(__file__).resolve().parents[1]
CASE = REPO/"cases/flight/m600_crosswind_playbook_land.json"


def passive(axis, angle):
    """Direction-cosine matrix of a frame turned by ``angle`` about ``axis`` (parent -> rotated components)."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    if axis == "y":
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


class UtilTests(unittest.TestCase):
    def test_wrap_saturate_crossfade(self):
        self.assertAlmostEqual(cl.wrap(7.0, 0.0, 2*np.pi), 7.0-2*np.pi)
        self.assertAlmostEqual(cl.wrap(-1.0, 0.0, 2*np.pi), 2*np.pi-1.0)
        self.assertEqual(cl.wrap(0.0, 0.0, 2*np.pi), 0.0)
        self.assertEqual(cl.saturate(5.0, -1.0, 1.0), 1.0)
        self.assertEqual(cl.saturate(-5.0, -1.0, 1.0), -1.0)
        with self.assertRaises(ValueError):
            cl.saturate(0.0, 1.0, -1.0)
        self.assertAlmostEqual(cl.crossfade(1.0, 3.0, 0.5, 0.0, 1.0), 2.0)
        self.assertEqual(cl.crossfade(1.0, 3.0, -1.0, 0.0, 1.0), 1.0)
        self.assertEqual(cl.crossfade(1.0, 3.0, 2.0, 0.0, 1.0), 3.0)
        self.assertEqual(cl.crossfade(1.0, 3.0, 5.0, 5.0, 5.0), 1.0)     # x_low == x_high: the source gets 0/DBL_MIN = 0 at x_low
        self.assertEqual(cl.crossfade(1.0, 3.0, 5.5, 5.0, 5.0), 3.0)     # and an infinite ratio, saturated to 1, above it
        self.assertEqual(cl.crossfade(1.0, 3.0, 4.5, 5.0, 5.0), 1.0)

    def test_circular_interp1(self):
        x = np.linspace(0.0, 2*np.pi, 5)
        y = [0.0, 1.0, 0.0, -1.0, 0.0]
        self.assertAlmostEqual(cl.circular_interp1(x, y, np.pi/4), 0.5)
        self.assertAlmostEqual(cl.circular_interp1(x, y, np.pi/4+2*np.pi), 0.5)
        self.assertAlmostEqual(cl.circular_interp1(x, y, -np.pi/4), -0.5)
        with self.assertRaises(ValueError):
            cl.circular_interp1(x, [0.0, 1.0, 0.0, -1.0, 0.5], 0.1)

    def test_angle_to_dcm_xzy_is_the_composition_of_passive_rotations(self):
        r1, r2, r3 = 0.3, -0.2, 0.7
        dcm = cl.angle_to_dcm_xzy(r1, r2, r3)
        expected = passive("y", r3)@passive("z", r2)@passive("x", r1)
        np.testing.assert_allclose(dcm, expected, atol=1e-14)
        np.testing.assert_allclose(dcm@dcm.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(dcm), 1.0)


class PlaybookConstructionTests(unittest.TestCase):
    def test_periodic_spline_table_constant_and_control_points(self):
        table = cl.periodic_spline_table([2.0]*6)
        self.assertEqual(len(table), 50)
        np.testing.assert_allclose(table, 2.0, atol=1e-12)
        points = [41.0, 50.023, 60.1149, 57.129, 42.0, 41.0]
        fine = cl.periodic_spline_table(points, table_length=6001)
        self.assertEqual(fine[0], fine[-1])
        # Control point i sits at the FBL fraction (i + 1/2)/6, i.e. at the controller loop angle 2 pi (8 - 2 i)/12.
        for i, value in enumerate(points):
            index = ((8-2*i)*500) % 6000
            self.assertAlmostEqual(fine[index], value, places=9)

    def test_convolve_smooth_edges(self):
        self.assertAlmostEqual(cl.convolve_smooth([155.0, 130.0, 125.0], [0.5, 1.0, 0.5])[1], 135.0)
        self.assertAlmostEqual(cl.convolve_smooth([155.0, 130.0, 125.0], [0.5, 1.0, 0.5])[0], 148.75)
        wrapped = cl.convolve_smooth([1.0, 2.0, 3.0, 1.0], [1.0, 1.0, 1.0], edges=["continuous"])
        self.assertAlmostEqual(wrapped[0], (3.0+1.0+2.0)/3)                # the point before the overlapping end is used
        self.assertAlmostEqual(wrapped[-1], (3.0+1.0+2.0)/3)
        cut = cl.convolve_smooth([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], edges=["cutoff"])
        self.assertAlmostEqual(cut[0], 1.5)
        with self.assertRaises(ValueError):
            cl.convolve_smooth([1.0, 2.0, 3.0], [1.0, 1.0])
        two_d = cl.convolve_smooth([[1.0, 2.0, 1.0], [3.0, 4.0, 3.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                                   edges=["extend", "continuous"])
        self.assertAlmostEqual(two_d[0][1], (2.0+2.0+4.0)/3)

    def test_adjust_wind_speed(self):
        self.assertAlmostEqual(cl.adjust_wind_speed(3.0, -1.0, 9.0, 0.9), 3.6)
        self.assertAlmostEqual(cl.adjust_wind_speed(9.0, -1.0, 9.0, 0.9), 9.0)
        self.assertAlmostEqual(cl.adjust_wind_speed(19.0, -1.0, 9.0, 0.9), 18.0)

    def test_land_case_file_builds_the_controller_entries(self):
        entries, raw, data = cl.load_playbook(CASE)
        self.assertEqual(len(entries), 9)
        np.testing.assert_allclose([e["wind_speed"] for e in entries], [3.6, 5.4, 7.2, 9.0, 10.8, 12.6, 14.4, 16.2, 18.0])
        nine = entries[3]
        self.assertAlmostEqual(nine["path_radius_target"], 135.0)          # smoothed across the rows, not the raw 130 m
        self.assertAlmostEqual(nine["beta_lookup"][0], (0.0188+0.00305-3*0.0384)/5, places=12)
        weights = np.array([0.0, 0.1111, 0.2222, 0.3333, 0.2222, 0.1111, 0.0])
        elevations = np.array([0.69, 0.67, 0.6, 0.55, 0.4821, 0.5188, 0.57])
        self.assertAlmostEqual(nine["elevation"], float(weights@elevations/weights.sum()), places=12)
        self.assertAlmostEqual(nine["azi_offset"], (0.5*0.0982+0.05+0.5*0.0)/2, places=12)
        self.assertEqual(raw[3]["path_radius_target"], 130.0)
        self.assertEqual(nine["airspeed_lookup"][0], nine["airspeed_lookup"][-1])
        self.assertLess(max(nine["alpha_lookup"]), max(raw[3]["alpha_lookup"]))
        with self.assertRaises(ValueError):
            cl.build_playbook(data["playbook_parameterized"][::-1], data["smoothing_params"], data["wind_adjust_params"])

    def test_playbook_entry_blends_bracketing_rows(self):
        entries, _, _ = cl.load_playbook(CASE)
        nine = cl.playbook_entry(entries, 9.0)
        self.assertEqual(nine["rows_blended"], (3, 4))
        np.testing.assert_allclose(nine["alpha_lookup"], entries[3]["alpha_lookup"])
        mid = cl.playbook_entry(entries, 9.9)
        np.testing.assert_allclose(mid["airspeed_lookup"], 0.5*(np.array(entries[3]["airspeed_lookup"])+np.array(entries[4]["airspeed_lookup"])))
        self.assertAlmostEqual(mid["path_radius_target"], 0.5*(entries[3]["path_radius_target"]+entries[4]["path_radius_target"]))
        low = cl.playbook_entry(entries, 1.0)
        np.testing.assert_allclose(low["alpha_lookup"], entries[0]["alpha_lookup"])
        high = cl.playbook_entry(entries, 30.0)
        np.testing.assert_allclose(high["alpha_lookup"], entries[-1]["alpha_lookup"])
        self.assertEqual(high["rows_blended"], (8, 8))


class LawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entries, _, data = cl.load_playbook(CASE)
        cls.entry = cl.playbook_entry(entries, 9.0)
        cls.constants = cl.LoopConstants.from_case(data)
        sphere = cls.constants.tether_length_m*1.001+cls.constants.bridle_radius_m
        cls.distance = cl.NominalLoop.sphere_centre_distance(sphere, cls.entry["path_radius_target"])
        cls.loop = cl.NominalLoop(cls.entry, [9.0, 0.0, 0.0], cls.constants, centre_distance_m=cls.distance)

    def test_constants_from_case(self):
        c = self.constants
        self.assertAlmostEqual(c.tether_roll_nom_rad, math.atan2(0.5, 4.7860+0.13035))
        self.assertAlmostEqual(c.effective_mass_kg, 1730.6+0.917*425.8/3)
        self.assertAlmostEqual(c.simple_aero.CL(0.0), 2.02175)
        self.assertAlmostEqual(c.simple_aero.CY(-0.0384), -1.4205533*-0.0384-0.08687)

    def test_frames(self):
        np.testing.assert_allclose(cl.sph_to_vec_g(0.0, 0.55, 1.0), [math.cos(0.55), 0.0, -math.sin(0.55)], atol=1e-15)
        centre = np.array([300.0, 20.0, -200.0])
        dcm = cl.dcm_g_to_cw(centre)
        np.testing.assert_allclose(dcm@dcm.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(dcm), 1.0)
        np.testing.assert_allclose(dcm[0], -centre/np.linalg.norm(centre))
        self.assertAlmostEqual(dcm[1]@[0.0, 0.0, 1.0], 0.0)                 # y_cw is horizontal
        f2g = cl.dcm_f_to_g([-40.0, 3.0, 5.0], centre)
        np.testing.assert_allclose(f2g.T@f2g, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(f2g[:, 0], np.array([40.0, -3.0, -5.0])/np.linalg.norm([40.0, 3.0, 5.0]))
        self.assertAlmostEqual(f2g[:, 1]@centre, 0.0)                        # y_f is tangent to the sphere

    def test_curvature_to_tether_roll(self):
        c = self.constants
        k = -1.0/135.0
        expected = (c.simple_aero.CY(-0.0187)-k*c.effective_mass_kg/(0.5*c.air_density_kg_m3*c.reference_area_m2))/c.simple_aero.CL(0.06)
        self.assertAlmostEqual(cl.curvature_to_tether_roll(k, c.simple_aero.CL(0.06), c.simple_aero.CY(-0.0187), c), expected)
        with self.assertRaises(ValueError):
            cl.curvature_to_tether_roll(k, 0.0, 0.0, c)

    def test_geometry_of_the_loop(self):
        loop = self.loop
        self.assertAlmostEqual(loop.kite_on_sphere_m, self.constants.tether_length_m*1.001+self.constants.bridle_radius_m)
        makani = cl.NominalLoop(self.entry, [9.0, 0.0, 0.0], self.constants)
        self.assertAlmostEqual(makani.centre_distance_m, 425.8)
        self.assertGreater(makani.kite_on_sphere_m, 425.8+4.786)             # Makani's idealisation overshoots the tether
        for angle in (0.0, 1.0, 2.5, 4.0, 5.5):
            position, vel_hat, acc_hat = loop.loop_geometry(angle)
            self.assertAlmostEqual(np.linalg.norm(position), loop.kite_on_sphere_m)
            self.assertAlmostEqual(np.linalg.norm(position-loop.path_centre_g), loop.radius_m)
            self.assertAlmostEqual(vel_hat@acc_hat, 0.0)
            self.assertAlmostEqual(vel_hat@loop.path_centre_g, 0.0)
        with self.assertRaises(ValueError):
            cl.NominalLoop.sphere_centre_distance(100.0, 135.0)

    def test_kinematics_close_on_the_commanded_airspeed(self):
        loop = self.loop
        for angle in (0.0, 0.7, 2.0, 3.3, 4.6, 6.0):
            velocity, _ = loop.makani_kinematics(angle)
            self.assertAlmostEqual(np.linalg.norm(velocity), loop.kite_speed(angle))
            self.assertAlmostEqual(np.linalg.norm(loop.wind_g-velocity), loop.airspeed(angle))
        self.assertEqual(loop.airspeed(0.3), cl.saturate(loop.airspeed(0.3), 25.0, 80.0))
        with self.assertRaises(ValueError):
            cl.NominalLoop(self.entry, [60.0, 0.0, 0.0], self.constants, centre_distance_m=self.distance).kite_speed(0.0)

    def test_attitude_reproduces_the_commands(self):
        loop = self.loop
        for angle in (0.2, 1.4, 2.9, 4.1, 5.7):
            position, rotation = loop.pose_world(angle)
            velocity = cl.MAKANI_G_TO_WORLD@loop.makani_kinematics(angle)[0]
            pose = Pose(0.0, position, rotation, velocity, np.zeros(3))
            _, speed, alpha_deg, beta_deg = relative_wind_body(pose, loop.wind_world())
            self.assertAlmostEqual(speed, loop.airspeed(angle), places=9)
            self.assertAlmostEqual(alpha_deg, math.degrees(loop.alpha_command(angle)[0]), places=9)
            self.assertAlmostEqual(beta_deg, -math.degrees(loop.beta_command(angle)), places=9)   # sideslip changes sign
            g2b = loop.attitude_g2b(angle)
            np.testing.assert_allclose(g2b@g2b.T, np.eye(3), atol=1e-13)
            # The roll about the fly-frame x axis is the commanded tether roll: dcm_f2b = Ry(alpha) Rz(-beta) Rx(-roll).
            position_g, vel_hat, _ = loop.loop_geometry(angle)
            f2g = cl.dcm_f_to_g(loop.wind_g-loop.kite_speed(angle)*vel_hat, position_g)
            f2b = g2b@f2g
            roll, _ = loop.tether_roll_command(angle)
            self.assertAlmostEqual(math.atan2(f2b[1, 2], f2b[1, 1]), -roll, places=12)
            self.assertAlmostEqual(-math.asin(f2b[1, 0]), -loop.beta_command(angle), places=12)

    def test_roll_command_and_its_saturation(self):
        loop = self.loop
        angles = np.linspace(0.0, 2*np.pi, 73)
        rolls = np.array([loop.tether_roll_command(a) for a in angles])
        self.assertTrue(np.all(rolls[:, 0] == rolls[:, 1]))                   # not saturated at the 20 kN band
        self.assertGreater(rolls[:, 0].min(), 0.1)
        self.assertLess(rolls[:, 0].max(), 0.4)
        needed = loop.tension_for_unsaturated_roll()
        self.assertTrue(5000.0 < needed < 10000.0)
        tight = cl.NominalLoop(self.entry, [9.0, 0.0, 0.0], self.constants, centre_distance_m=self.distance, roll_limit_tension_N=5000.0)
        tight_rolls = np.array([tight.tether_roll_command(a) for a in angles])
        self.assertTrue(np.any(tight_rolls[:, 0] != tight_rolls[:, 1]))
        self.assertAlmostEqual(tight_rolls[:, 0].max(), self.constants.tether_roll_nom_rad+0.2)
        wide = cl.NominalLoop(self.entry, [9.0, 0.0, 0.0], self.constants, centre_distance_m=self.distance, roll_limit_tension_N=None)
        self.assertAlmostEqual(wide.roll_excursion(), 0.48)
        # The aerodynamic curvature: geometric -1/R with the corrections removed.
        for angle in (0.5, 2.0, 4.0):
            speed = loop.kite_speed(angle)
            expected = (-1.0/loop.radius_m-loop.gravity_curvature(angle, speed))/(1.0+loop.apparent_wind_curvature_factor(angle, speed))
            self.assertAlmostEqual(loop.aero_curvature(angle), expected)
        self.assertEqual(loop.alpha_command(1.0)[1], False)                   # no fade towards alpha_min at these airspeeds

    def test_linear_reading_matches_the_controller_interpolation(self):
        linear = cl.NominalLoop(self.entry, [9.0, 0.0, 0.0], self.constants, centre_distance_m=self.distance, interpolation="linear")
        for angle in (0.1, 1.234, 5.9):
            self.assertAlmostEqual(linear.alpha_nom(angle), cl.circular_interp1(self.entry["lookup_loop_angle"], self.entry["alpha_lookup"], angle))
        difference = self.loop.interpolation_difference()
        self.assertLess(difference["airspeed_lookup"], 0.05)
        self.assertLess(difference["alpha_lookup"], 2e-4)
        with self.assertRaises(ValueError):
            cl.NominalLoop(self.entry, [9.0, 0.0, 0.0], self.constants, interpolation="cubic")

    def test_period_and_loop_angle_in_time(self):
        loop = self.loop
        period = loop.period_s()
        quad, _ = integrate.quad(lambda a: loop.radius_m/loop.kite_speed(a), 0.0, 2*np.pi, limit=400,
                                 points=list(loop.loop_angles[1:-1]))
        self.assertAlmostEqual(period, quad, places=8)
        self.assertAlmostEqual(loop.loop_angle_at(period), -2*np.pi, places=10)
        self.assertAlmostEqual(loop.loop_angle_at(0.0), 0.0)
        times = np.linspace(0.0, 2.5*period, 40)
        angles = np.array([loop.loop_angle_at(t) for t in times])
        self.assertTrue(np.all(np.diff(angles) < 0.0))
        for t in (0.37*period, 1.2*period):
            angle = loop.loop_angle_at(t)
            laps = math.floor(t/period)
            back, _ = integrate.quad(lambda a: loop.radius_m/loop.kite_speed(a), angle+2*np.pi*laps, 0.0, limit=400,
                                     points=list(-loop.loop_angles[1:-1]))
            self.assertAlmostEqual(back+laps*period, t, places=9)

    def test_motion_samples(self):
        loop = self.loop
        motion, table = loop.motion(2, 60)
        self.assertEqual(len(motion.time_s), 121)
        self.assertAlmostEqual(table["dt_s"]*60, table["period_s"])
        np.testing.assert_allclose(motion.position_m[60:], motion.position_m[:61], atol=1e-9)      # lap 2 repeats lap 1
        np.testing.assert_allclose(motion.rotation[60:], motion.rotation[:61], atol=1e-12)
        np.testing.assert_allclose(motion.velocity_m_s[60:], motion.velocity_m_s[:61], atol=1e-8)
        self.assertIsNone(motion.angular_acceleration_rad_s2)
        self.assertEqual(len(table["angular_acceleration_rad_s2"]), 121)
        self.assertEqual(table["samples_with_roll_saturated"], 0)
        stored, _ = loop.motion(1, 12, store_angular_acceleration=True)
        self.assertIsNotNone(stored.angular_acceleration_rad_s2)
        # Velocity and angular velocity against the law's own differences and against Makani's feed-forward.
        for k in (3, 17, 41):
            angle = table["loop_angle_rad"][k]
            np.testing.assert_allclose(motion.velocity_m_s[k], cl.MAKANI_G_TO_WORLD@loop.makani_kinematics(angle)[0], rtol=1e-9)
            makani = loop.body_rates_to_world(angle, loop.makani_pqr(angle))
            self.assertLess(np.linalg.norm(makani-motion.angular_velocity_rad_s[k])/np.linalg.norm(makani), 1e-5)
            feed_forward = cl.MAKANI_G_TO_WORLD@loop.makani_kinematics(angle)[1]
            self.assertLess(np.linalg.norm(feed_forward-motion.acceleration_m_s2[k])/np.linalg.norm(feed_forward), 2e-2)
        coarse_interval = motion.consistency("interval")["angular_velocity_vs_attitude"]["relative"]
        coarse_fourth = motion.consistency("fourth_order_central")["angular_velocity_vs_attitude"]["relative"]
        self.assertLess(coarse_fourth, coarse_interval/10)
        self.assertLess(motion.consistency()["velocity_vs_position"]["relative"], 1e-4)
        settings = loop.settings()
        self.assertEqual(settings["interpolation"], "spline")
        self.assertAlmostEqual(settings["kite_distance_from_ground_point_m"], loop.kite_on_sphere_m)
        with self.assertRaises(ValueError):
            loop.motion(0, 60)


class ConsistencySchemeTests(unittest.TestCase):
    def test_fourth_order_attitude_scheme_on_a_constant_rotation(self):
        motion = circular_motion(np.arange(41)*0.02, 130.0, 47.0, 9.0, 3.0)
        for scheme in ("interval", "fourth_order_central"):
            report = motion.consistency(scheme)
            self.assertEqual(report["attitude_scheme"], scheme)
            self.assertLess(report["angular_velocity_vs_attitude"]["relative"], 1e-9)
        with self.assertRaises(ValueError):
            motion.consistency("sixth")
        short = uniform_translation(np.arange(3)*0.1, [1.0, 0.0, 0.0])
        self.assertEqual(short.consistency("fourth_order_central")["attitude_scheme"], "interval")   # fall-back with 3 samples


if __name__ == "__main__":
    unittest.main()


class GoldenValueTests(unittest.TestCase):
    """Values of the law at the 9 m/s land entry on the tether sphere, frozen from the reviewed implementation of 2026-10-05.

    The sign-sensitive parts of the transcription (azimuth-offset sense, loop direction and gravity angle, apparent-wind
    factor, feed-forward phase, the law's density and masses) are verified by reading against makani 1d66923 (review
    A5R-16); these numbers pin them so that a later change is noticed. Receipt: results/a5-law-check.json of the G4 task.
    """

    @classmethod
    def setUpClass(cls):
        entries, _, data = cl.load_playbook(CASE)
        entry = cl.playbook_entry(entries, 9.0)
        constants = cl.LoopConstants.from_case(data)
        centre = cl.NominalLoop.sphere_centre_distance(constants.tether_length_m+constants.bridle_radius_m, entry["path_radius_target"])
        cls.loop = cl.NominalLoop(entry, [9.0, 0.0, 0.0], constants, centre_distance_m=centre)

    def test_path_centre_period_and_masses(self):
        np.testing.assert_allclose(self.loop.path_centre_g, [346.8856758888633, 17.202265893921403, -215.7630953319897], rtol=1e-9)
        self.assertAlmostEqual(self.loop.period_s(), 18.71768577426334, places=7)
        self.assertAlmostEqual(self.loop.constants.effective_mass_kg, 1860.7528666666667, places=9)
        self.assertAlmostEqual(self.loop.constants.tether_roll_nom_rad, 0.10135298647679533, places=12)

    def test_speeds_rolls_and_curvatures_at_five_loop_angles(self):
        expected = {0.3: (41.16862652828226, 0.30711743728485397, -0.006942888673627358, 0.08452606353677226),
                    1.4: (54.01150180033212, 0.3267714701742539, -0.009627269860312097, 0.08559582516741596),
                    2.9: (54.78765646082685, 0.31749892054514084, -0.009386523671388165, 0.06263341723954187),
                    4.1: (43.09878545270389, 0.19887276550012317, -0.004275478439845023, 0.02892493700700082),
                    5.7: (36.81698876877324, 0.20467082103845868, -0.0033060958477896278, 0.07170376685546478)}
        for angle, (speed, roll, curvature, alpha) in expected.items():
            self.assertAlmostEqual(self.loop.kite_speed(angle), speed, places=8)
            self.assertAlmostEqual(self.loop.tether_roll_command(angle)[0], roll, places=10)
            self.assertAlmostEqual(self.loop.aero_curvature(angle), curvature, places=12)
            self.assertAlmostEqual(self.loop.alpha_command(angle)[0], alpha, places=12)

    def test_roll_uses_the_commands_before_their_saturation(self):
        # A tight saturation of the incidence command changes the attitude but not the roll of the curvature balance.
        from dataclasses import replace
        tight = replace(self.loop.constants, alpha_cmd_max_rad=0.05)
        loop = cl.NominalLoop(self.loop.entry, [9.0, 0.0, 0.0], tight, centre_distance_m=self.loop.centre_distance_m)
        self.assertEqual(loop.alpha_command(1.4)[0], 0.05)
        self.assertAlmostEqual(loop.tether_roll_command(1.4)[0], self.loop.tether_roll_command(1.4)[0], places=12)


class TranscriptionCorrectionTests(unittest.TestCase):
    """Two deviations of the hand transcription from the source, corrected 2026-10-07 (decision 0033).

    Found by check L3 of the controller-library trial (the library's own parameters against the case file):
    ``DBL_TOL`` was 1e-12 (``common/c_math/util.h`` line 36: 1e-9) and ``dalpha_dairspeed`` was wrong from the 11th
    significant digit (``config/m600/control/crosswind.py`` line 277: ``-deg2rad(8.0/15.0)``).
    """

    OLD_DALPHA = -0.009308422677494552

    def test_the_two_constants_are_those_of_the_source(self):
        self.assertEqual(cl.DBL_TOL, 1e-9)
        _, _, data = cl.load_playbook(CASE)
        self.assertEqual(cl.LoopConstants.from_case(data).dalpha_dairspeed, -math.radians(8.0/15.0))
        self.assertNotEqual(self.OLD_DALPHA, -math.radians(8.0/15.0))

    def test_the_corrections_leave_the_nominal_loops_bit_for_bit(self):
        # The floor DBL_TOL only acts on vectors shorter than itself, and the incidence fade (the only place of
        # dalpha_dairspeed) starts metres per second above the commanded airspeed of every entry: nothing computed
        # with the earlier values (the A5 inputs of G4, the step-1b cases of G5) changes.
        from dataclasses import replace
        from unittest import mock
        entries, _, data = cl.load_playbook(CASE)
        constants = cl.LoopConstants.from_case(data)
        angles = np.linspace(0.0, 2.0*np.pi, 64, endpoint=False)

        def values(wind, constants):
            entry = cl.playbook_entry(entries, wind)
            sphere = constants.tether_length_m+constants.bridle_radius_m
            loop = cl.NominalLoop(entry, [wind, 0.0, 0.0], constants,
                                  centre_distance_m=cl.NominalLoop.sphere_centre_distance(sphere, entry["path_radius_target"]))
            rows, fading = [], False
            for angle in angles:
                position, rotation = loop.pose_world(angle)
                alpha, fade = loop.alpha_command(angle)
                fading = fading or bool(fade)
                rows.append(np.concatenate([position, rotation.ravel(), [alpha, loop.kite_speed(angle), loop.tether_roll_command(angle)[0]]]))
            return np.array(rows), fading, loop.period_s()
        for wind in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0):
            new, fading, period = values(wind, constants)
            with mock.patch.object(cl, "DBL_TOL", 1e-12):
                old, old_fading, old_period = values(wind, replace(constants, dalpha_dairspeed=self.OLD_DALPHA))
            self.assertFalse(fading or old_fading, wind)
            self.assertTrue(np.array_equal(new, old), wind)
            self.assertEqual(period, old_period, wind)
