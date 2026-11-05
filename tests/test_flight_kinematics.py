"""Prescribed rigid-body motion input (G4 contract: K2 kinematic field, K3 self-consistency)."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from openfawe.flight.kinematics import (RigidMotion, circular_motion, pitch_manoeuvre, relative_wind_body, rotation_from_vector,
                                        rotation_vector, trapezoid_profile, uniform_translation)

RADIUS, SPEED, WIND, ALPHA, DT = 130.0, 47.0, 9.0, 3.0, 0.02


class RotationVectorTests(unittest.TestCase):
    def test_round_trip_small_and_large_angles(self):
        for vector in ([0., 0., 0.], [1e-10, -2e-10, 3e-10], [0.3, -0.2, 0.5], [2.0, 1.0, -1.5]):
            assert_allclose(rotation_vector(rotation_from_vector(vector)), vector, rtol=1e-12, atol=1e-15)

    def test_half_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            rotation_vector(rotation_from_vector([np.pi, 0., 0.]))


class CircularMotionTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(60)*DT
        self.motion = circular_motion(self.times, RADIUS, SPEED, WIND, ALPHA)

    def test_origin_stays_on_the_circle_with_the_declared_speed(self):
        m = self.motion
        assert_allclose(np.linalg.norm(m.position_m, axis=1), RADIUS, rtol=1e-14)
        assert_allclose(m.position_m[:, 0], 0., atol=0.)
        assert_allclose(np.linalg.norm(m.velocity_m_s, axis=1), SPEED, rtol=1e-14)
        assert_allclose(m.angular_velocity_rad_s, np.broadcast_to([SPEED/RADIUS, 0., 0.], (60, 3)), rtol=1e-14, atol=0.)

    def test_span_axis_is_radial_and_relative_wind_has_the_declared_incidence(self):
        m = self.motion
        for k in (0, 17, 59):
            pose = m.pose(self.times[k])
            assert_allclose(pose.rotation[:, 1], pose.position_m/RADIUS, rtol=0, atol=1e-14)
            relative, speed, alpha, beta = relative_wind_body(pose, [WIND, 0., 0.])
            self.assertAlmostEqual(speed, np.hypot(WIND, SPEED), places=12)
            self.assertAlmostEqual(alpha, ALPHA, places=11)
            self.assertAlmostEqual(beta, 0., places=11)
            self.assertGreater(pose.rotation[0, 2], 0.)       # body z has a downwind component

    def test_point_velocity_is_origin_velocity_plus_rotation(self):
        pose = self.motion.pose(self.times[5])
        body = np.array([[0.3, 12.8, 0.1], [0.3, -12.8, 0.1]])          # two wing tips in body axes
        points = pose.position_m+body@pose.rotation.T
        expected = pose.velocity_m_s+np.cross(pose.angular_velocity_rad_s, points-pose.position_m)
        assert_allclose(self.motion.structural_velocity(points, self.times[5]), expected, rtol=0, atol=0)
        # Along the flight direction the outer tip is faster than the inner one by Omega times their distance.
        along = expected@(pose.velocity_m_s/SPEED)
        self.assertAlmostEqual(along[0]-along[1], SPEED/RADIUS*25.6, places=11)

    def test_k3_samples_are_self_consistent_within_the_frozen_limit(self):
        report = self.motion.consistency()
        self.assertEqual(report["scheme"], "fourth_order_central")
        for name in ("velocity_vs_position", "angular_velocity_vs_attitude", "acceleration_vs_velocity"):
            self.assertLessEqual(report[name]["relative"], 1e-6, name)
        self.assertLessEqual(report["angular_acceleration_vs_angular_velocity"]["max_abs"], 1e-12)

    def test_k3_report_counts_what_it_compared(self):
        report = self.motion.consistency()
        self.assertEqual(report["velocity_vs_position"]["compared"], 56)            # 60 samples less two at each end
        self.assertEqual(report["angular_velocity_vs_attitude"]["compared"], 59)    # every interval
        self.assertIsNone(report["angular_acceleration_vs_angular_velocity"]["relative"])   # zero reference: no ratio

    def test_k3_detects_wrong_conventions(self):
        m = self.motion
        def variant(**changes):
            fields = dict(time_s=m.time_s, position_m=m.position_m, rotation=m.rotation, velocity_m_s=m.velocity_m_s,
                          angular_velocity_rad_s=m.angular_velocity_rad_s)
            return RigidMotion(**{**fields, **changes}).consistency()
        body_rate = np.einsum("nji,nj->ni", m.rotation, m.angular_velocity_rad_s)       # body components, not inertial
        self.assertGreater(variant(angular_velocity_rad_s=body_rate)["angular_velocity_vs_attitude"]["relative"], 1e-2)
        self.assertGreater(variant(angular_velocity_rad_s=-m.angular_velocity_rad_s)["angular_velocity_vs_attitude"]["relative"], 1.)
        transposed = np.transpose(m.rotation, (0, 2, 1))                                # inertial-to-body matrices
        self.assertGreater(variant(rotation=transposed)["angular_velocity_vs_attitude"]["relative"], 1e-2)
        self.assertGreater(variant(velocity_m_s=1.001*m.velocity_m_s)["velocity_vs_position"]["relative"], 5e-4)

    def test_interpolation_follows_the_constant_rate_rotation(self):
        coarse = circular_motion(np.arange(4)*0.5, RADIUS, SPEED, WIND, ALPHA)
        exact = circular_motion(np.array([0.5, 0.75, 1.0]), RADIUS, SPEED, WIND, ALPHA, phase_rad=SPEED/RADIUS*0.5)
        pose = coarse.pose(0.75)
        assert_allclose(pose.rotation, exact.rotation[1], rtol=0, atol=1e-13)
        assert_allclose(pose.angular_velocity_rad_s, exact.angular_velocity_rad_s[1], rtol=0, atol=1e-15)
        # Position and velocity are interpolated linearly by declaration, not along the arc.
        assert_allclose(pose.position_m, .5*(coarse.position_m[1]+coarse.position_m[2]), rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            coarse.pose(1.6)


class TrapezoidProfileTests(unittest.TestCase):
    def test_rest_to_rest_move_reaches_the_distance_with_the_declared_peak_rate(self):
        ramp, cruise, distance = 0.5, 2.5, -1.5
        time = np.linspace(-1., 5., 601)
        angle, rate, acceleration = trapezoid_profile(time, distance, ramp, cruise)
        assert_allclose(angle[time <= 0.], 0., atol=0.)
        assert_allclose(angle[time >= 2*ramp+cruise], distance, rtol=0, atol=1e-15)
        self.assertAlmostEqual(rate.min(), distance/(ramp+cruise), places=14)          # peak rate, negative move
        self.assertEqual(rate[0], 0.)
        self.assertEqual(rate[-1], 0.)
        # A negative move: the rate falls first (-1 rad/s2 = peak/ramp), is constant, then rises back to rest.
        for lo, hi, value in ((0., ramp, -1.), (ramp, ramp+cruise, 0.), (ramp+cruise, 2*ramp+cruise, 1.)):
            inside = (time > lo+1e-9) & (time < hi-1e-9)
            assert_allclose(acceleration[inside], value, rtol=0, atol=1e-12)
        assert_allclose(acceleration[(time < 0.) | (time > 2*ramp+cruise+1e-9)], 0., atol=0.)
        # At a corner the acceleration is that of the phase about to begin.
        corner = trapezoid_profile(np.array([0., ramp, ramp+cruise, 2*ramp+cruise]), distance, ramp, cruise)[2]
        assert_allclose(corner, [-1., 0., 1., 0.], rtol=0, atol=1e-12)
        # The angle is the integral of the rate (trapezoid rule is exact on the linear pieces between the corners).
        integral = np.concatenate(([0.], np.cumsum(.5*(rate[1:]+rate[:-1])*np.diff(time))))
        assert_allclose(angle, integral, rtol=0, atol=1e-12)
        # The rate is continuous, the angle is monotone.
        self.assertLessEqual(np.abs(np.diff(rate)).max(), 1.0*0.01+1e-12)
        self.assertTrue(np.all(np.diff(angle) <= 1e-15))

    def test_triangular_move_and_validation(self):
        angle, rate, _ = trapezoid_profile(np.array([0., 0.5, 1.]), 2., 0.5, 0.)
        assert_allclose(angle, [0., 1., 2.], atol=1e-15)
        assert_allclose(rate, [0., 4., 0.], atol=1e-15)
        for ramp, cruise in ((0., 1.), (-1., 1.), (1., -0.1)):
            with self.assertRaises(ValueError):
                trapezoid_profile(np.zeros(2), 1., ramp, cruise)


class PitchManoeuvreTests(unittest.TestCase):
    DT, RAMP, CRUISE = 0.02, 25, 127       # samples: corner times fall on the grid

    def motion(self, speed=25.):
        dt, ramp, cruise = self.DT, self.RAMP*self.DT, self.CRUISE*self.DT
        down = 25*dt
        up = down+2*ramp+cruise+50*dt
        times = np.arange(455)*dt
        return times, pitch_manoeuvre(times, speed, 90., [(down, -87., ramp, cruise), (up, 87., ramp, cruise)])

    def test_incidence_history_is_the_declared_one_at_constant_airspeed(self):
        times, motion = self.motion()
        seen = []
        for k in range(len(times)):
            _, speed, alpha, beta = relative_wind_body(motion.pose(times[k]), np.zeros(3))
            self.assertAlmostEqual(speed, 25., places=12)
            self.assertAlmostEqual(beta, 0., places=12)
            seen.append(alpha)
        seen = np.array(seen)
        assert_allclose(seen[:26], 90., rtol=0, atol=1e-11)                 # dwell before the move
        assert_allclose(seen[202:253], 3., rtol=0, atol=1e-11)              # dwell at the low incidence
        assert_allclose(seen[-26:], 90., rtol=0, atol=1e-11)
        self.assertTrue(np.all(np.diff(seen[25:203]) < 0.) and np.all(np.diff(seen[252:430]) > 0.))
        # The second half mirrors the first: same incidence at mirrored samples.
        assert_allclose(seen[25:203], seen[252:430][::-1], rtol=0, atol=1e-10)

    def test_rates_stay_within_the_makani_transition_limits(self):
        _, motion = self.motion()
        rate = motion.angular_velocity_rad_s
        assert_allclose(rate[:, [0, 2]], 0., atol=0.)
        self.assertLessEqual(np.abs(rate[:, 1]).max(), 0.5)                               # source S6 of the contract
        self.assertGreater(np.abs(rate[:, 1]).max(), 0.499)
        self.assertLessEqual(np.abs(np.diff(rate[:, 1])).max()/self.DT, 1.0+1e-9)
        self.assertIsNone(motion.angular_acceleration_rad_s2)                             # discontinuous: not stored
        # Pitching down in incidence is a negative rotation about +Y with these axes.
        self.assertLess(rate[100, 1], 0.)
        self.assertGreater(rate[340, 1], 0.)

    def test_k3_samples_are_self_consistent(self):
        _, motion = self.motion()
        report = motion.consistency()
        self.assertLessEqual(report["velocity_vs_position"]["relative"], 1e-6)
        self.assertLessEqual(report["angular_velocity_vs_attitude"]["relative"], 1e-6)
        self.assertEqual(report["angular_velocity_vs_attitude"]["compared"], 454)

    def test_body_axes_and_validation(self):
        times = np.arange(3)*0.1
        level = pitch_manoeuvre(times, 10., 0., [])
        assert_allclose(level.rotation[0], np.eye(3), atol=0.)
        upright = pitch_manoeuvre(times, 10., 90., [])
        assert_allclose(upright.rotation[0][:, 0], [0., 0., -1.], atol=1e-15)             # chord (aft) points down: nose up
        assert_allclose(upright.rotation[0][:, 2], [1., 0., 0.], atol=1e-15)              # body z along the relative wind
        assert_allclose(level.position_m[-1], [-2., 0., 0.], atol=1e-15)
        with self.assertRaises(ValueError):
            pitch_manoeuvre(times, 0., 3., [])
        longer = np.arange(11)*0.1
        pitch_manoeuvre(longer, 10., 3., [(0., 5., 0.1, 0.1), (0.3, 5., 0.1, 0.1)])       # back to back is allowed
        with self.assertRaises(ValueError):                                               # second move starts inside the first
            pitch_manoeuvre(longer, 10., 3., [(0., 5., 0.1, 0.1), (0.2, 5., 0.1, 0.1)])
        with self.assertRaises(ValueError):                                               # move runs past the last sample
            pitch_manoeuvre(longer, 10., 3., [(0.8, 5., 0.1, 0.1)])
        with self.assertRaises(ValueError):                                               # move starts before the first sample
            pitch_manoeuvre(longer, 10., 3., [(-0.1, 5., 0.1, 0.1)])


class UniformTranslationTests(unittest.TestCase):
    def test_translation_is_consistent_and_keeps_attitude(self):
        times = np.arange(12)*0.1
        motion = uniform_translation(times, [-20., 0., 1.], origin=[1., 2., 3.])
        assert_allclose(motion.pose(0.5).position_m, [1.-10., 2., 3.5], rtol=0, atol=1e-13)
        report = motion.consistency()
        self.assertLessEqual(report["velocity_vs_position"]["relative"], 1e-12)
        self.assertEqual(report["angular_velocity_vs_attitude"]["max_abs"], 0.)


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_samples(self):
        good = dict(time_s=[0., 1.], position_m=np.zeros((2, 3)), rotation=np.broadcast_to(np.eye(3), (2, 3, 3)),
                    velocity_m_s=np.zeros((2, 3)), angular_velocity_rad_s=np.zeros((2, 3)))
        RigidMotion(**good)
        for change in (dict(time_s=[0., 0.]), dict(position_m=np.zeros((3, 3))),
                       dict(rotation=np.broadcast_to(2*np.eye(3), (2, 3, 3))),
                       dict(rotation=np.broadcast_to(np.diag([1., 1., -1.]), (2, 3, 3))),
                       dict(velocity_m_s=np.full((2, 3), np.nan))):
            with self.assertRaises(ValueError):
                RigidMotion(**{**good, **change})

    def test_second_order_scheme_on_non_uniform_samples_is_declared(self):
        times = np.array([0., 0.1, 0.25, 0.3, 0.5, 0.55])
        motion = uniform_translation(times, [3., 0., 0.])
        report = motion.consistency()
        self.assertEqual(report["scheme"], "second_order_central")
        self.assertLessEqual(report["velocity_vs_position"]["relative"], 1e-12)


if __name__ == "__main__":
    unittest.main()


class FigureEightMotionTests(unittest.TestCase):
    """OF-0009 case C2: the constant-speed Lissajous figure-eight on the tether sphere."""

    @classmethod
    def setUpClass(cls):
        from openfawe.flight.kinematics import FigureEightPath, figure_eight_motion
        cls.path = FigureEightPath(302., 17., np.radians(27.), np.radians(20.), np.radians(5.))
        n = int(round(cls.path.period_s/0.025))
        cls.dt = cls.path.period_s/n
        cls.n = n
        cls.motion, cls.built = figure_eight_motion(np.arange(2*n+1)*cls.dt, 302., 17., 10., 5.41, elevation_deg=27.,
                                                    azimuth_amplitude_deg=20., elevation_amplitude_deg=5.)

    def test_origin_stays_on_the_sphere_at_the_declared_speed_and_closes_after_one_period(self):
        radius = np.linalg.norm(self.motion.position_m, axis=1)
        assert_allclose(radius, 302., rtol=0., atol=1e-9)
        assert_allclose(np.linalg.norm(self.motion.velocity_m_s, axis=1), 17., rtol=0., atol=1e-10)
        assert_allclose(self.motion.position_m[self.n], self.motion.position_m[0], rtol=0., atol=1e-9)
        assert_allclose(self.motion.rotation[self.n], self.motion.rotation[0], rtol=0., atol=1e-12)
        self.assertAlmostEqual(self.built.period_s, self.path.period_s)
        self.assertAlmostEqual(self.built.length_m/self.built.speed, self.built.period_s)

    def test_path_is_the_lying_eight_with_the_declared_amplitudes(self):
        position = self.motion.position_m[:self.n]
        elevation = np.degrees(np.arcsin(position[:, 2]/302.))
        azimuth = np.degrees(np.arctan2(position[:, 1], position[:, 0]))
        assert_allclose([elevation.min(), elevation.max()], [22., 32.], atol=0.02)
        assert_allclose([azimuth.min(), azimuth.max()], [-20., 20.], atol=0.02)
        # the eight: two azimuth extremes per period, four elevation extremes
        self.assertEqual(int(np.sum(np.diff(np.sign(np.diff(azimuth))) != 0)), 2)
        self.assertEqual(int(np.sum(np.diff(np.sign(np.diff(elevation))) != 0)), 4)
        # down-loop: rising through the centre crossing at the start
        self.assertGreater(self.motion.velocity_m_s[0, 2], 0.)

    def test_relative_wind_at_the_origin_has_the_declared_incidence_and_no_sideslip(self):
        for k in range(0, self.n, 37):
            pose = self.motion.pose(self.motion.time_s[k])
            relative, speed, alpha, beta = relative_wind_body(pose, [10., 0., 0.])
            self.assertAlmostEqual(alpha, 5.41, places=9)
            self.assertAlmostEqual(beta, 0., places=9)
            # lift side towards the outward radial direction
            self.assertGreater(float(pose.rotation[:, 2]@pose.position_m), 0.)
            # proper rotation with the span axis normal to the radial direction
            assert_allclose(pose.rotation.T@pose.rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(pose.rotation[:, 1]@pose.position_m)/302., 0., places=9)

    def test_k3_samples_are_self_consistent_within_the_frozen_limit(self):
        report = self.motion.consistency("fourth_order_central")
        for name in ("velocity_vs_position", "angular_velocity_vs_attitude", "acceleration_vs_velocity"):
            self.assertLessEqual(report[name]["relative"], 1e-6, name)
        self.assertNotIn("angular_acceleration_vs_angular_velocity", report)      # not stored, as for the A5 loop

    def test_arc_length_inversion_is_exact_and_validation(self):
        from openfawe.flight.kinematics import FigureEightPath
        theta = self.path.theta(np.array([0., 0.3*self.path.period_s, 0.9*self.path.period_s]))
        s = self.path.s_table[np.searchsorted(self.path.theta_table, theta, side="right")-1]
        self.assertAlmostEqual(float(theta[0]), 0.)
        for value, target in zip(theta, [0., 0.3*self.path.length_m, 0.9*self.path.length_m]):
            j = int(np.searchsorted(self.path.theta_table, value, side="right")-1)
            total = self.path.s_table[j]+self.path._arc(self.path.theta_table[j], value)
            self.assertAlmostEqual(float(total), target, places=8)
        with self.assertRaises(ValueError):
            FigureEightPath(300., 17., np.radians(27.), np.radians(20.), 0.)
        with self.assertRaises(ValueError):
            FigureEightPath(300., 17., np.radians(88.), np.radians(20.), np.radians(5.))


class SphericalCircleMotionTests(unittest.TestCase):
    """OF-0009 cases C2c/C2e: the constant-speed circle on the tether sphere with the figure-eight's attitude rule."""

    @classmethod
    def setUpClass(cls):
        from openfawe.flight.kinematics import spherical_circle_motion
        cls.n = 200
        period = 2.*np.pi*302.*np.sin(np.radians(17.6))/17.
        cls.dt = period/cls.n
        cls.motion, cls.path = spherical_circle_motion(np.arange(2*cls.n+1)*cls.dt, 302., 17., 10., 5.41, elevation_deg=31.9,
                                                       angular_radius_deg=17.6)

    def test_circle_on_the_sphere_at_constant_speed_closing_after_one_period(self):
        position = self.motion.position_m
        assert_allclose(np.linalg.norm(position, axis=1), 302., atol=1e-9)
        assert_allclose(np.linalg.norm(self.motion.velocity_m_s, axis=1), 17., atol=1e-10)
        # angular distance from the centre direction is the angular radius everywhere
        cosines = position@self.path.axis/302.
        assert_allclose(np.degrees(np.arccos(cosines)), 17.6, atol=1e-9)
        assert_allclose(position[self.n], position[0], atol=1e-9)
        self.assertAlmostEqual(self.path.period_s, self.n*self.dt)
        self.assertAlmostEqual(self.path.radius*np.sin(self.path.rho), 302.*np.sin(np.radians(17.6)))

    def test_attitude_rule_and_k3(self):
        for k in range(0, self.n, 23):
            pose = self.motion.pose(self.motion.time_s[k])
            _, _, alpha, beta = relative_wind_body(pose, [10., 0., 0.])
            self.assertAlmostEqual(alpha, 5.41, places=9)
            self.assertAlmostEqual(beta, 0., places=9)
            self.assertGreater(float(pose.rotation[:, 2]@pose.position_m), 0.)
        report = self.motion.consistency("fourth_order_central")
        for name in ("velocity_vs_position", "angular_velocity_vs_attitude", "acceleration_vs_velocity"):
            self.assertLessEqual(report[name]["relative"], 1e-6, name)

    def test_sense_and_validation(self):
        from openfawe.flight.kinematics import SphericalCirclePath, spherical_circle_motion
        reversed_motion, _ = spherical_circle_motion(np.arange(5)*self.dt, 302., 17., 10., 5.41, elevation_deg=31.9, angular_radius_deg=17.6, sense=-1)
        assert_allclose(reversed_motion.velocity_m_s[0], -self.motion.velocity_m_s[0], atol=1e-9)
        with self.assertRaises(ValueError):
            SphericalCirclePath(302., 17., np.radians(31.9), np.radians(95.))
        with self.assertRaises(ValueError):
            SphericalCirclePath(302., 17., np.radians(90.), np.radians(10.))
