"""Residual and work account of a prescribed motion (G4 contract section 3 items 5 and 6)."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from openfawe.flight.kinematics import RigidMotion, circular_motion, pitch_manoeuvre, uniform_translation
from openfawe.flight.residual import RigidBody, account, accelerations, load_rigid_body, rotor_inertial_moments
from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set
from openfawe.rotor.table import RotorTable

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"cases/aircraft/m600_makani_rigid_body.json"
INERTIA = np.array([[900., 30., -20.], [30., 400., 10.], [-20., 10., 1100.]])
RADIUS, SPEED, WIND = 130., 47., 9.
G = np.array([0., 0., -9.81])


def records(motion, **loads):
    """Step records holding only what the account reads; ``loads`` name constant inertial vectors or functions of the step."""
    rows = []
    for step, time in enumerate(motion.time_s):
        row = dict(step=step, time_s=float(time))
        for key, value in loads.items():
            row[key] = np.asarray(value(step) if callable(value) else value, dtype=float).tolist()
        rows.append(row)
    return rows


def no_load(motion):
    return records(motion, force_inertial_N=np.zeros(3), moment_origin_inertial_Nm=np.zeros(3))


def torque_free_motion(steps, dt, omega_body):
    """A free rigid body turning about its centre of mass: Euler's equations integrated in body axes (RK4)."""
    inverse = np.linalg.inv(INERTIA)

    def rate(omega):
        return inverse@(-np.cross(omega, INERTIA@omega))

    def skew(v):
        return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

    def derivative(state):
        rotation, omega = state
        return rotation@skew(omega), rate(omega)
    rotation, omega = np.eye(3), np.asarray(omega_body, dtype=float)
    rotations, inertial_rates, inertial_accelerations = [], [], []
    for _ in range(steps+1):
        rotations.append(rotation)
        inertial_rates.append(rotation@omega)
        inertial_accelerations.append(rotation@rate(omega))          # d(R w)/dt = R dw/dt, since (R [w]x) w = 0
        k1 = derivative((rotation, omega))
        k2 = derivative((rotation+.5*dt*k1[0], omega+.5*dt*k1[1]))
        k3 = derivative((rotation+.5*dt*k2[0], omega+.5*dt*k2[1]))
        k4 = derivative((rotation+dt*k3[0], omega+dt*k3[1]))
        rotation = rotation+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        omega = omega+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        u, _, vt = np.linalg.svd(rotation)                             # back onto the rotation group
        rotation = u@vt
    n = steps+1
    return RigidMotion(np.arange(n)*dt, np.zeros((n, 3)), np.array(rotations), np.zeros((n, 3)), np.array(inertial_rates),
                       np.zeros((n, 3)), np.array(inertial_accelerations), description="torque-free rotation")


class ResidualTests(unittest.TestCase):
    def test_a_point_mass_on_a_circle_needs_the_centripetal_force_and_its_weight(self):
        motion = circular_motion(np.arange(6)*0.02, RADIUS, SPEED, WIND, 3.)
        body = RigidBody(1700., np.zeros(3), INERTIA)
        rows, summary = account(no_load(motion), motion, body)
        for row, position in zip(rows, motion.position_m):
            centripetal = -1700.*SPEED**2/RADIUS*position/RADIUS
            assert_allclose(row["required_force_inertial_N"], centripetal-1700.*G, rtol=1e-13, atol=1e-9)
        self.assertEqual(summary["acceleration_source"], dict(linear="stored with the motion", angular="stored with the motion"))
        self.assertEqual((summary["loads_included"], summary["loads_absent"]), (["aero"], ["rotor", "tether"]))

    def test_a_centre_of_mass_off_the_origin_moves_on_its_own_circle(self):
        # The span axis (body y) points radially outwards, so a centre of mass at y = d runs on the radius R + d.
        motion = circular_motion(np.arange(6)*0.02, RADIUS, SPEED, WIND, 3.)
        d, mass, rate = 2.5, 1700., SPEED/RADIUS
        rows, _ = account(no_load(motion), motion, RigidBody(mass, [0., d, 0.], INERTIA), gravity_m_s2=np.zeros(3))
        for row, position in zip(rows, motion.position_m):
            assert_allclose(row["required_force_inertial_N"], -mass*rate**2*(RADIUS+d)*position/RADIUS, rtol=1e-12, atol=1e-9)
            self.assertAlmostEqual(row["kinetic_energy_J"], .5*mass*(rate*(RADIUS+d))**2+.5*rate**2*(motion.pose(row["time_s"]).rotation
                                   @ INERTIA@motion.pose(row["time_s"]).rotation.T)[0, 0], delta=1e-6)

    def test_a_free_body_turning_about_its_centre_of_mass_needs_no_moment(self):
        # The motion comes from Euler's equations in body axes; the account works in inertial components.
        motion = torque_free_motion(200, 0.002, [0.4, 1.1, -0.7])
        body = RigidBody(50., np.zeros(3), INERTIA)
        momentum = np.array([r@INERTIA@r.T@w for r, w in zip(motion.rotation, motion.angular_velocity_rad_s)])
        scale = np.linalg.norm(momentum[0])
        # The check of the trajectory: the angular momentum of the generated motion stays put in the inertial frame.
        assert_allclose(momentum, np.broadcast_to(momentum[0], momentum.shape), rtol=0, atol=1e-9*scale)
        rows, _ = account(no_load(motion), motion, body, gravity_m_s2=np.zeros(3))
        gyroscopic = max(np.linalg.norm(np.cross(w, r@INERTIA@r.T@w)) for r, w in zip(motion.rotation, motion.angular_velocity_rad_s))
        self.assertGreater(gyroscopic, 100.)                                              # the two terms that cancel are large
        # With the stored accelerations this is a check of the account's frame algebra (the stored rate is Euler's
        # equation at the stored state, so nothing about the trajectory is tested here): body-axis dynamics against the
        # account's inertial-component formula.
        for row in rows:
            self.assertLess(np.linalg.norm(row["required_moment_cm_inertial_Nm"]), 1e-10*gyroscopic)
            self.assertLess(np.linalg.norm(row["required_force_inertial_N"]), 1e-12)
        # Without stored accelerations they are differenced: second order in the sample spacing.
        coarse = RigidMotion(motion.time_s, motion.position_m, motion.rotation, motion.velocity_m_s, motion.angular_velocity_rad_s)
        rows, summary = account(no_load(coarse), coarse, body, gravity_m_s2=np.zeros(3))
        self.assertEqual(summary["acceleration_source"]["angular"], "central difference of the stored velocity")
        inner = max(np.linalg.norm(row["required_moment_cm_inertial_Nm"]) for row in rows[1:-1])
        self.assertLess(inner, 1e-4*gyroscopic)
        self.assertGreater(inner, 0.)

    def test_loads_are_taken_about_the_centre_of_mass_and_the_missing_load_is_their_negative(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        motion = uniform_translation(np.arange(4)*0.1, [5., 0., 2.], rotation=rotation)
        body = RigidBody(100., [0.3, -0.2, 0.5], INERTIA)
        force, arm, torque = np.array([40., -10., 25.]), np.array([1.5, 2., -0.5]), np.array([3., 4., 5.])     # inertial
        rows, summary = account(records(motion, force_inertial_N=force, moment_origin_inertial_Nm=np.cross(arm, force),
                                        rotor_force_inertial_N=np.zeros(3), rotor_moment_origin_inertial_Nm=torque,
                                        tether_force_inertial_N=-2*force, tether_moment_origin_inertial_Nm=np.zeros(3)),
                                motion, body, gravity_m_s2=np.zeros(3))
        offset = rotation@body.centre_of_mass_body_m
        expected_moment = np.cross(arm-offset, force)+torque+np.cross(-offset, -2*force)
        for row in rows:
            assert_allclose(row["required_force_inertial_N"], force, atol=1e-12)          # -(F + 0 - 2F)
            assert_allclose(row["required_moment_cm_inertial_Nm"], -expected_moment, atol=1e-11)
            assert_allclose(row["required_force_body_N"], rotation.T@force, atol=1e-12)
            self.assertAlmostEqual(row["power_W"]["aero"], float(force@[5., 0., 2.]), places=10)
            self.assertAlmostEqual(row["power_W"]["required"], float(force@[5., 0., 2.]), places=10)
            self.assertAlmostEqual(row["kinetic_energy_rate_W"], 0., places=9)            # steady translation
        self.assertEqual(summary["loads_included"], ["aero", "rotor", "tether"])
        self.assertIn("rotor gyroscopic and reaction moments", summary["not_modelled"])
        named, other = account(no_load(motion), motion, body, not_included=("pylons", "fuselage aerodynamics"))
        self.assertEqual(other["not_modelled"], ["pylons", "fuselage aerodynamics"])
        mixed = no_load(motion)
        mixed[2]["tether_force_inertial_N"] = [0., 0., 0.]
        with self.assertRaises(ValueError):                                               # the loads must be the same in every record
            account(mixed, motion, body)

    def test_the_power_account_closes_on_the_rate_of_the_kinetic_energy(self):
        dt, steps = 0.01, 60
        times = np.arange(steps+1)*dt
        motion = pitch_manoeuvre(times, 25., 30., [(5*dt, -27., 20*dt, 10*dt)])           # angular acceleration not stored
        body = RigidBody(1700., [0.08, 0., -0.13], INERTIA)
        load = lambda step: np.array([100.+step, 0., 2000.-10.*step])                    # noqa: E731
        rows, summary = account(records(motion, force_inertial_N=load, moment_origin_inertial_Nm=np.array([0., 500., 0.])), motion, body)
        self.assertEqual(summary["acceleration_source"]["angular"], "central difference of the stored velocity")
        energy = np.array([row["kinetic_energy_J"] for row in rows])
        rate = np.array([row["kinetic_energy_rate_W"] for row in rows])
        numerical = np.gradient(energy, times)
        scale = np.abs(numerical).max()
        self.assertGreater(scale, 1.)
        # Away from the corners of the pitch-rate trapezoid (steps 5, 25, 35, 55), where the true acceleration jumps.
        smooth = [k for k in range(2, steps-1) if min(abs(k-c) for c in (5, 25, 35, 55)) > 1]
        assert_allclose(rate[smooth], numerical[smooth], rtol=0, atol=2e-3*scale)
        sampled = np.array([row["kinetic_energy_rate_from_samples_W"] for row in rows])
        assert_allclose(sampled, numerical, rtol=1e-12, atol=1e-9*scale)                  # the account reports this quotient
        self.assertGreater(summary["largest_kinetic_energy_rate_gap_W"], 2e-3*scale)      # ... and the gap at the corners
        for row in rows:                                                                  # bookkeeping: the rate is the sum of the powers
            total = sum(row["power_W"].values())
            self.assertAlmostEqual(total, row["kinetic_energy_rate_W"], delta=1e-9*max(1., abs(total)))

    def test_rotor_gyroscopic_and_reaction_moments(self):
        table = RotorTable.from_makani_json(ROOT/"cases/rotors/makani/rotor_rev4.json")
        discs = load_rotor_set(ROOT/"cases/rotors/m600_rotors_makani.json", table)
        dt = 0.02
        motion = pitch_manoeuvre(np.arange(8)*dt, 25., 30., [(0., -27., 3*dt, dt)])
        inertia = load_rotor_inertia(ROOT/"cases/rotors/m600_rotors_makani.json")
        self.assertEqual(inertia, 1.56)                                                   # 1.23 propeller + 0.33 motor rotor (Makani)
        one = discs[:1]
        rows = records(motion, rotor_omega_rad_s=lambda step: [120.+50.*step*dt])
        gyro, reaction = rotor_inertial_moments(rows, motion, one, inertia)
        for k, row in enumerate(rows):
            pose = motion.pose(row["time_s"])
            momentum = inertia*(120.+50.*k*dt)*one[0].direction*(pose.rotation@one[0].axis)
            assert_allclose(gyro[k], np.cross(momentum, pose.angular_velocity_rad_s), atol=1e-12)     # L x omega (rotor.cc)
            assert_allclose(reaction[k], -inertia*50.*one[0].direction*(pose.rotation@one[0].axis), rtol=1e-10, atol=1e-10)
        self.assertGreater(np.abs(gyro).max(), 10.)
        # The eight M600 rotors at one speed: four of each sense on parallel axes, the gyroscopic moments cancel.
        equal = records(motion, rotor_omega_rad_s=np.full(8, 120.))
        gyro, reaction = rotor_inertial_moments(equal, motion, discs, inertia)
        assert_allclose(gyro, 0., atol=1e-10)
        assert_allclose(reaction, 0., atol=1e-12)
        # Each row holds two rotors of each sense, so the rows may differ in speed and the moments still cancel ...
        by_row = records(motion, rotor_omega_rad_s=np.array([120.]*4+[150.]*4))
        assert_allclose(rotor_inertial_moments(by_row, motion, discs, inertia)[0], 0., atol=1e-10)
        # ... but not when the two senses run at different speeds.
        by_sense = np.array([120. if disc.direction > 0 else 150. for disc in discs])
        uneven = records(motion, rotor_omega_rad_s=by_sense)
        self.assertGreater(np.abs(rotor_inertial_moments(uneven, motion, discs, inertia)[0]).max(), 10.)
        # In the account the gyroscopic moment does no work; it enters the moment balance.
        body = RigidBody(1700., np.zeros(3), INERTIA)
        loaded = records(motion, force_inertial_N=np.zeros(3), moment_origin_inertial_Nm=np.zeros(3),
                         rotor_force_inertial_N=np.zeros(3), rotor_moment_origin_inertial_Nm=np.zeros(3),
                         rotor_omega_rad_s=by_sense)
        with_rotors, summary = account(loaded, motion, body, discs=discs, rotor_inertia_kg_m2=inertia)
        without, _ = account(loaded, motion, body)
        gyro, _ = rotor_inertial_moments(loaded, motion, discs, inertia)
        for a, b, moment in zip(with_rotors, without, gyro):
            assert_allclose(np.subtract(b["required_moment_cm_inertial_Nm"], a["required_moment_cm_inertial_Nm"]), moment, atol=1e-9)
            self.assertAlmostEqual(a["power_W"]["rotor_gyroscopic_and_reaction"], 0., places=9)
        self.assertNotIn("rotor gyroscopic and reaction moments", summary["not_modelled"])
        with self.assertRaises(ValueError):
            account(loaded, motion, body, discs=discs)
        with self.assertRaises(ValueError):                                               # two records: no speed rate to take
            rotor_inertial_moments(loaded[:2], motion, discs, inertia)

    def test_accelerations_and_set_up(self):
        motion = circular_motion(np.arange(5)*0.02, RADIUS, SPEED, WIND, 3.)
        linear, angular, _ = accelerations(motion)
        assert_allclose(linear, motion.acceleration_m_s2, atol=0)
        bare = RigidMotion(motion.time_s, motion.position_m, motion.rotation, motion.velocity_m_s, motion.angular_velocity_rad_s)
        linear, angular, source = accelerations(bare)
        assert_allclose(linear, motion.acceleration_m_s2, rtol=0, atol=2e-3*np.abs(motion.acceleration_m_s2).max())
        self.assertEqual(source["linear"], "central difference of the stored velocity")
        with self.assertRaises(ValueError):                                               # a record off the samples
            account([dict(step=0, time_s=0.011, force_inertial_N=[0., 0., 0.], moment_origin_inertial_Nm=[0., 0., 0.])],
                    motion, RigidBody(1., np.zeros(3), INERTIA))
        with self.assertRaises(ValueError):
            RigidBody(1., np.zeros(3), np.diag([1., -1., 1.]))
        self.assertEqual(account([], motion, RigidBody(1., np.zeros(3), INERTIA)), ([], dict(steps=0)))

    def test_the_case_file_holds_the_source_values_in_openfawe_axes(self):
        body = load_rigid_body(CASE)
        mass = 1667.9+51.5+10.6+0.6                                                        # config/m600/wing.py, serial 06 crosswind
        source = np.array([[30260.0, 21.35, 33.90], [21.35, 9210.0, 16.51], [33.90, 16.51, 36620.0]])*mass/1600.0
        to_makani = np.diag([-1., 1., -1.])
        self.assertAlmostEqual(body.mass_kg, mass, places=10)
        assert_allclose(to_makani@body.centre_of_mass_body_m, [-0.086, 0.011, 0.121], atol=1e-15)
        assert_allclose(to_makani@body.inertia_body_kg_m2@to_makani, source, rtol=1e-14)
        self.assertLess(body.inertia_body_kg_m2[0, 1], 0.)                                 # the xy product changes sign, xz does not
        self.assertGreater(body.inertia_body_kg_m2[0, 2], 0.)


if __name__ == "__main__":
    unittest.main()
