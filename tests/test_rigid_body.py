"""Rigid-body propagation under given loads (G5 plant contract, step 1a): analytic solutions, invariants, conventions."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from openfawe.flight.kinematics import Pose, circular_motion, rotation_from_vector, rotation_vector
from openfawe.flight.residual import RigidBody, account, load_rigid_body
from openfawe.flight.rigid_body import (GRAVITY_M_S2, RigidState, angular_momentum, kinetic_energy, no_loads, propagate,
                                        rigid_step, state_from_pose)

ROOT = Path(__file__).resolve().parents[1]
BODY = load_rigid_body(ROOT/"cases/aircraft/m600_makani_rigid_body.json")
R0 = rotation_from_vector(0.7*np.array([1., 2., 3.])/np.sqrt(14.))
G = np.array(GRAVITY_M_S2)


def angle(a, b):
    return float(np.linalg.norm(rotation_vector(a.T@b)))


class StateTests(unittest.TestCase):
    def test_a_pose_and_its_state_describe_the_same_motion_of_every_body_point(self):
        pose = Pose(1.5, np.array([3., -2., 40.]), R0, np.array([-30., 4., 2.]), np.array([0.2, -0.4, 0.1]))
        state = state_from_pose(pose, BODY)
        arm = R0@BODY.centre_of_mass_body_m
        assert_allclose(state.position_cm_m, pose.position_m+arm, rtol=0, atol=1e-14)
        # The centre of mass is a material point of the body: its velocity is the point velocity of the pose there.
        assert_allclose(state.velocity_cm_m_s, pose.point_velocity(state.position_cm_m[None, :])[0], rtol=0, atol=1e-13)
        assert_allclose(R0@state.angular_velocity_body_rad_s, pose.angular_velocity_rad_s, rtol=0, atol=1e-15)
        back = state.pose(BODY)
        for name in ("position_m", "velocity_m_s", "angular_velocity_rad_s", "rotation"):
            assert_allclose(getattr(back, name), getattr(pose, name), rtol=0, atol=1e-13, err_msg=name)
        self.assertEqual(back.time_s, 1.5)

    def test_states_are_checked_and_cannot_be_written_to(self):
        state = RigidState(0., np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            state.position_cm_m[0] = 1.
        with self.assertRaises(ValueError):
            RigidState(0., np.zeros(2), np.eye(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(FloatingPointError):                    # a state outside the finite range is a failure, not a value
            RigidState(0., np.array([np.nan, 0., 0.]), np.eye(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(FloatingPointError):
            rigid_step(BODY, state, lambda t, s: (np.array([np.inf, 0., 0.]), np.zeros(3)), 0.01)
        with self.assertRaises(ValueError):                             # an attitude that is not a rotation is refused
            RigidState(0., np.zeros(3), 1.001*np.eye(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):                             # nor a reflection
            RigidState(0., np.zeros(3), np.diag([1., 1., -1.]), np.zeros(3), np.zeros(3))
        for step in (0., -0.01, np.nan):
            with self.assertRaises(ValueError):
                rigid_step(BODY, state, no_loads, step)
        with self.assertRaises(ValueError):
            rigid_step("body", state, no_loads, 0.01)
        with self.assertRaises(ValueError):
            propagate(BODY, state, no_loads, 0.01, -1)
        self.assertEqual(len(propagate(BODY, state, no_loads, 0.01, 0)), 1)

    def test_the_time_of_a_long_run_is_the_start_plus_whole_steps(self):
        states = propagate(BODY, RigidState(3., np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3)), no_loads, 0.002, 500)
        self.assertEqual([states[k].time_s for k in (0, 1, 500)], [3., 3.+0.002, 3.+500*0.002])


class TranslationTests(unittest.TestCase):
    def test_a_projectile_with_a_constant_force_follows_the_parabola_to_rounding(self):
        force = np.array([3000., 0., 0.])
        start = RigidState(0., np.array([10., -20., 300.]), R0, np.array([-40., 5., 12.]), np.zeros(3))
        states = propagate(BODY, start, lambda t, s: (force, np.zeros(3)), 0.01, 300)
        a = G+force/BODY.mass_kg
        for s in states[::50]:
            t = s.time_s
            assert_allclose(s.position_cm_m, start.position_cm_m+start.velocity_cm_m_s*t+0.5*a*t*t, rtol=1e-13, atol=1e-11)
            assert_allclose(s.velocity_cm_m_s, start.velocity_cm_m_s+a*t, rtol=1e-13, atol=1e-12)
            self.assertTrue(np.array_equal(s.rotation, R0))             # no moment, no rotation: the attitude is untouched

    def test_gravity_is_a_declared_input(self):
        start = RigidState(0., np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
        still = propagate(BODY, start, no_loads, 0.01, 10, gravity_m_s2=(0., 0., 0.))[-1]
        self.assertTrue(np.array_equal(still.position_cm_m, np.zeros(3)))
        moon = propagate(BODY, start, no_loads, 0.01, 10, gravity_m_s2=(0., 0., -1.62))[-1]
        assert_allclose(moon.position_cm_m, [0., 0., -0.5*1.62*0.01], rtol=1e-13, atol=1e-16)


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.diagonal = RigidBody(BODY.mass_kg, BODY.centre_of_mass_body_m, np.diag(np.diag(BODY.inertia_body_kg_m2)))

    def test_steady_rotation_about_a_principal_axis_while_falling_moves_the_body_origin_on_its_analytic_path(self):
        omega = np.array([0., 0.5, 0.])
        start = RigidState(0., np.array([10., -20., 300.]), R0, np.array([-40., 5., 12.]), omega)
        c = self.diagonal.centre_of_mass_body_m
        for s in propagate(self.diagonal, start, no_loads, 0.01, 300)[::60]:
            t = s.time_s
            rotation = R0@rotation_from_vector(omega*t)
            self.assertLess(angle(rotation, s.rotation), 1e-13)
            pose = s.pose(self.diagonal)
            centre = start.position_cm_m+start.velocity_cm_m_s*t+0.5*G*t*t
            assert_allclose(pose.position_m, centre-rotation@c, rtol=1e-13, atol=1e-11)
            assert_allclose(pose.velocity_m_s, start.velocity_cm_m_s+G*t-np.cross(rotation@omega, rotation@c), rtol=1e-13, atol=1e-12)

    def test_a_constant_moment_about_a_principal_body_axis_gives_a_quadratic_angle_of_the_right_sign(self):
        torque = 2000.
        rate = torque/self.diagonal.inertia_body_kg_m2[1, 1]
        start = RigidState(0., np.zeros(3), R0, np.zeros(3), np.zeros(3))
        states = propagate(self.diagonal, start, lambda t, s: (np.zeros(3), s.rotation@np.array([0., torque, 0.])), 0.01, 200)
        last = states[-1]
        assert_allclose(last.angular_velocity_body_rad_s, [0., rate*2., 0.], rtol=1e-13, atol=1e-15)
        self.assertLess(angle(R0@rotation_from_vector(np.array([0., 0.5*rate*4., 0.])), last.rotation), 1e-13)
        # The opposite moment turns the other way: the sign is not lost in an even function.
        other = propagate(self.diagonal, start, lambda t, s: (np.zeros(3), s.rotation@np.array([0., -torque, 0.])), 0.01, 200)[-1]
        self.assertGreater(angle(other.rotation, last.rotation), 0.5)

    def test_a_force_through_the_centre_of_mass_turns_nothing_and_a_moment_moves_nothing(self):
        start = RigidState(0., np.zeros(3), R0, np.zeros(3), np.zeros(3))
        pushed = propagate(BODY, start, lambda t, s: (np.array([0., 500., 0.]), np.zeros(3)), 0.01, 50, gravity_m_s2=(0., 0., 0.))[-1]
        self.assertTrue(np.array_equal(pushed.rotation, R0))
        turned = propagate(BODY, start, lambda t, s: (np.zeros(3), np.array([0., 0., 800.])), 0.01, 50, gravity_m_s2=(0., 0., 0.))[-1]
        self.assertTrue(np.array_equal(turned.position_cm_m, np.zeros(3)))
        self.assertGreater(np.linalg.norm(turned.angular_velocity_body_rad_s), 0.)
        # The moment is taken in inertial components: its impulse is the change of the inertial angular momentum.
        assert_allclose(angular_momentum(BODY, turned), [0., 0., 800.*0.5], rtol=1e-9, atol=1e-9)

    def test_torque_free_rotation_with_the_full_inertia_tensor_keeps_momentum_energy_and_orthogonality(self):
        omega = np.array([0.3, 0.5, -0.2])
        states = propagate(BODY, RigidState(0., np.zeros(3), R0, np.zeros(3), omega), no_loads, 0.002, 2000, gravity_m_s2=(0., 0., 0.))
        h0, e0 = angular_momentum(BODY, states[0]), kinetic_energy(BODY, states[0])[1]
        self.assertGreater(np.linalg.norm(states[-1].angular_velocity_body_rad_s-omega), 0.01)   # the body does tumble
        for s in states[::200]:
            assert_allclose(angular_momentum(BODY, s), h0, rtol=0, atol=1e-11*np.linalg.norm(h0))
            self.assertLess(abs(kinetic_energy(BODY, s)[1]-e0), 1e-11*e0)
            self.assertLess(np.abs(s.rotation.T@s.rotation-np.eye(3)).max(), 1e-13)

    def test_an_axisymmetric_body_follows_the_closed_form_and_the_error_falls_with_the_fourth_power_of_the_step(self):
        a, c = 32729.9725, 9961.76625
        body = RigidBody(1730.6, np.zeros(3), np.diag([a, a, c]))
        omega = np.array([0.3, 0.2, 0.5])
        h = R0@(body.inertia_body_kg_m2@omega)
        spin = (c-a)/a*omega[2]

        def exact(t):
            return (rotation_from_vector(h/np.linalg.norm(h)*(np.linalg.norm(h)/a*t))@R0@rotation_from_vector(np.array([0., 0., -spin*t])),
                    np.array([omega[0]*np.cos(spin*t)-omega[1]*np.sin(spin*t), omega[0]*np.sin(spin*t)+omega[1]*np.cos(spin*t), omega[2]]))
        errors = []
        for step, count in ((0.04, 100), (0.02, 200), (0.01, 400)):
            last = propagate(body, RigidState(0., np.zeros(3), R0, np.zeros(3), omega), no_loads, step, count, gravity_m_s2=(0., 0., 0.))[-1]
            rotation, rate = exact(last.time_s)
            errors.append(angle(rotation, last.rotation))
            assert_allclose(last.angular_velocity_body_rad_s, rate, rtol=0, atol=1e-7)
        self.assertLess(errors[-1], 1e-9)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertTrue(3.7 <= np.log2(coarse/fine) <= 4.3, errors)


class StateDependentLoadTests(unittest.TestCase):
    """Loads that read the stage state: handing them the state of the start of the step would lose the fourth order."""
    STEPS = ((0.04, 125), (0.02, 250), (0.01, 500))                    # 5 s

    def orders(self, start, loads, exact_position, body=BODY):
        errors = []
        for step, count in self.STEPS:
            last = propagate(body, start, loads, step, count)[-1]
            errors.append(float(np.linalg.norm(last.position_cm_m-exact_position(last.time_s))))
        return errors, [np.log2(a/b) for a, b in zip(errors, errors[1:])]

    def test_a_spring_on_the_centre_of_mass_gives_the_harmonic_motion_with_fourth_order_error(self):
        rate, anchor = 2., np.array([10., -20., 300.])
        start = RigidState(0., anchor+np.array([5., -3., 2.]), R0, np.array([1., -2., 0.5]), np.zeros(3))
        rest = anchor+G/rate**2
        amplitude, v0 = start.position_cm_m-rest, start.velocity_cm_m_s
        errors, orders = self.orders(start, lambda t, s: (-BODY.mass_kg*rate**2*(s.position_cm_m-anchor), np.zeros(3)),
                                     lambda t: rest+amplitude*np.cos(rate*t)+v0/rate*np.sin(rate*t))
        self.assertLess(errors[-1], 1e-6)
        self.assertGreater(errors[0], 1e-7)                             # above rounding: the order is measured, not noise
        for order in orders:
            self.assertTrue(3.7 <= order <= 4.3, (errors, orders))

    def test_linear_drag_gives_the_exponential_approach_to_the_terminal_velocity(self):
        rate, v0 = 0.3, np.array([-40., 5., 12.])
        start = RigidState(0., np.zeros(3), R0, v0, np.zeros(3))
        last = propagate(BODY, start, lambda t, s: (-BODY.mass_kg*rate*s.velocity_cm_m_s, np.zeros(3)), 0.01, 500)[-1]
        decay, terminal = np.exp(-rate*5.), G/rate
        assert_allclose(last.velocity_cm_m_s, terminal+(v0-terminal)*decay, rtol=1e-10, atol=1e-10)
        assert_allclose(last.position_cm_m, terminal*5.+(v0-terminal)*(1.-decay)/rate, rtol=1e-10, atol=1e-9)

    def test_a_rotational_damper_about_a_principal_axis_slows_the_spin_exponentially(self):
        body = RigidBody(BODY.mass_kg, BODY.centre_of_mass_body_m, np.diag(np.diag(BODY.inertia_body_kg_m2)))
        rate, spin = 0.3, 0.5
        damping = rate*body.inertia_body_kg_m2[1, 1]
        start = RigidState(0., np.zeros(3), R0, np.zeros(3), np.array([0., spin, 0.]))
        last = propagate(body, start, lambda t, s: (np.zeros(3), s.rotation@(-damping*s.angular_velocity_body_rad_s)), 0.01, 500,
                         gravity_m_s2=(0., 0., 0.))[-1]
        decay = np.exp(-rate*5.)
        assert_allclose(last.angular_velocity_body_rad_s, [0., spin*decay, 0.], rtol=0, atol=1e-11)
        self.assertLess(angle(R0@rotation_from_vector(np.array([0., spin/rate*(1.-decay), 0.])), last.rotation), 1e-11)

    def test_a_thrust_fixed_in_the_body_follows_the_attitude_at_every_stage(self):
        body = RigidBody(BODY.mass_kg, BODY.centre_of_mass_body_m, np.diag(np.diag(BODY.inertia_body_kg_m2)))
        spin, thrust, v0 = 0.5, 4000., np.array([-40., 5., 12.])
        a = thrust/body.mass_kg
        start = RigidState(0., np.zeros(3), R0, v0, np.array([0., spin, 0.]))

        def exact(t):
            return v0*t+0.5*G*t*t+a*(R0@np.array([(1.-np.cos(spin*t))/spin**2, 0., (np.sin(spin*t)-spin*t)/spin**2]))
        errors, orders = self.orders(start, lambda t, s: (s.rotation@np.array([thrust, 0., 0.]), np.zeros(3)), exact, body)
        self.assertLess(errors[-1], 1e-8)
        self.assertGreater(errors[0], 1e-9)
        for order in orders:
            self.assertTrue(3.7 <= order <= 4.3, (errors, orders))


class ResidualInverseTests(unittest.TestCase):
    """The required loads of a motion (definitions of the residual account), given back, reproduce the motion."""
    RADIUS, SPEED = 130., 47.

    def at(self, time):
        motion = circular_motion(np.array([0., time if time > 0. else 1.]), self.RADIUS, self.SPEED, 9., 3.)
        k = 1 if time > 0. else 0
        return motion.pose(motion.time_s[k]), motion.acceleration_m_s2[k]

    def required(self, time, sign=1.):
        pose, acceleration = self.at(time)
        omega = pose.angular_velocity_rad_s
        arm = pose.rotation@BODY.centre_of_mass_body_m
        inertia = pose.rotation@BODY.inertia_body_kg_m2@pose.rotation.T
        return (BODY.mass_kg*(acceleration+np.cross(omega, np.cross(omega, arm)))-BODY.mass_kg*G,
                sign*np.cross(omega, inertia@omega))

    def test_the_load_function_is_the_residual_account_of_the_motion_without_model_loads(self):
        times = np.arange(6)*0.02
        motion = circular_motion(times, self.RADIUS, self.SPEED, 9., 3.)
        rows, _ = account([dict(step=k, time_s=float(t)) for k, t in enumerate(times)], motion, BODY)
        for t, row in zip(times, rows):
            force, moment = self.required(float(t))
            assert_allclose(force, row["required_force_inertial_N"], rtol=1e-12, atol=1e-9)
            assert_allclose(moment, row["required_moment_cm_inertial_Nm"], rtol=1e-12, atol=1e-9)
        self.assertGreater(np.linalg.norm(self.required(0.)[1]), 100.)          # the motion does need a moment

    def test_the_circle_is_recovered_and_a_moment_of_the_wrong_sign_is_not(self):
        start = state_from_pose(self.at(0.)[0], BODY)
        last = propagate(BODY, start, lambda t, s: self.required(t), 0.002, 1000)[-1]
        exact = self.at(last.time_s)[0]
        pose = last.pose(BODY)
        assert_allclose(pose.position_m, exact.position_m, rtol=0, atol=1e-10*self.RADIUS)
        assert_allclose(pose.velocity_m_s, exact.velocity_m_s, rtol=0, atol=1e-10*self.SPEED)
        self.assertLess(angle(exact.rotation, last.rotation), 1e-10)
        wrong = propagate(BODY, start, lambda t, s: self.required(t, sign=-1.), 0.002, 1000)[-1]
        self.assertGreater(angle(exact.rotation, wrong.rotation), 1e-3)          # the check can tell


if __name__ == "__main__":
    unittest.main()
