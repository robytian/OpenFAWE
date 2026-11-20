"""Free flight with frozen controls: aerodynamics, rotors and tether in the loop (G5 plant contract, step 1b part two)."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero import polars as P
from openfawe.aero.handover import SectionHandover
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing
from openfawe.flight import free_flight
from openfawe.flight.free_flight import Release, gyroscopic_moment, held_loads, rotor_momentum_body, run_free
from openfawe.flight.kinematics import circular_motion, pitch_manoeuvre, rotation_from_vector, uniform_translation
from openfawe.flight.plant import HeldLoads, PlantTether, plant_step, rates
from openfawe.flight.prescribed import PrescribedRotors, PrescribedTether, run_prescribed
from openfawe.flight.residual import RigidBody, account, rotor_inertial_moments
from openfawe.flight.rigid_body import RigidState, kinetic_energy, state_from_pose
from openfawe.rotor.actuator_disc import ActuatorDisc
from openfawe.rotor.table import RotorTable
from openfawe.tether.bridle import Bridle
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings

ROOT = Path(__file__).resolve().parents[1]
NACA = ROOT/"cases/polars/makani/naca0012_high_aoa.json"
NEWTON = SolveSettings(solver="newton", max_iterations=60)
RADIUS, SPEED, WIND, ALPHA, DT = 130.0, 47.0, 9.0, 3.0, 0.02
BODY = RigidBody(1700., [0.08, 0.01, -0.13], np.array([[32000., -20., 35.], [-20., 9700., -15.], [35., -15., 38700.]]))
G = np.array([0., 0., -9.81])
R0 = rotation_from_vector(0.7*np.array([1., 2., 3.])/np.sqrt(14.))


def backend(steps, dt=DT):
    return AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=0.15, solve=NEWTON))


def rotor_table():
    """A small table in the Makani layout (as in the prescribed-motion tests): thrust falls with the axial freestream."""
    omegas, speeds = np.linspace(40., 260., 12), np.linspace(0.1, 90., 9)
    w, v = np.meshgrid(omegas, speeds, indexing="ij")
    return RotorTable("synthetic", 2.3, omegas, speeds, 0.18-0.002*v-1e-5*w*v/60., -0.08+0.0015*v)


def discs():
    """Two discs ahead of the wing, off the body origin, turning in the same sense so that their momenta add."""
    table = rotor_table()
    axis = [-np.cos(np.radians(3.)), 0., -np.sin(np.radians(3.))]
    return (ActuatorDisc("upper", [-1.9, 1.2, 1.2], axis, 1, table), ActuatorDisc("lower", [-1.6, -3.6, -1.6], axis, 1, table))


def circle(steps):
    return circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA), np.array([WIND, 0., 0.])


class LoadConversionTests(unittest.TestCase):
    def test_held_loads_are_about_the_centre_of_mass_in_body_axes(self):
        force, point = np.array([300., -80., 1500.]), np.array([2.0, -1.5, 0.7])         # a force at a body point (body axes)
        origin_moment = R0@np.cross(point, force)
        held = held_loads(BODY, R0, R0@force, origin_moment)
        assert_allclose(held.force_body_N, force, rtol=0, atol=1e-10)
        assert_allclose(held.moment_cm_body_Nm, np.cross(point-BODY.centre_of_mass_body_m, force), rtol=0, atol=1e-9)

    def test_rates_are_the_equations_of_motion_at_one_state(self):
        state = RigidState(0.3, [1., 2., 3.], R0, [4., -5., 6.], [0.2, -0.1, 0.3])
        held = HeldLoads([100., -200., 5000.], [300., -800., 200.])
        acceleration, spin = rates(BODY, state, held=held)
        assert_allclose(acceleration, R0@held.force_body_N/BODY.mass_kg+G, rtol=1e-14)
        omega, inertia = state.angular_velocity_body_rad_s, BODY.inertia_body_kg_m2
        assert_allclose(inertia@spin, held.moment_cm_body_Nm-np.cross(omega, inertia@omega), rtol=1e-12)
        with self.assertRaises(ValueError):
            rates(BODY, state, nodes=np.zeros((3, 3)))

    def test_the_gyroscopic_moment_has_the_sign_and_size_of_the_residual_account(self):
        steps = 3
        motion, wind = circle(steps)
        rotors = PrescribedRotors(discs(), [150., 120.], nodes_per_ring=8)
        run = run_prescribed(backend(steps), Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.),
                             motion, wind, rotors=rotors)
        self.assertEqual(run.status, "complete", run.error)
        gyro, reaction = rotor_inertial_moments(run.records, motion, rotors.discs, 1.56)
        momentum = rotor_momentum_body(rotors, rotors.omegas(0.), 1.56)
        assert_allclose(momentum, 1.56*(150.+120.)*np.array([-np.cos(np.radians(3.)), 0., -np.sin(np.radians(3.))]), rtol=1e-14)
        for k in range(steps+1):
            state = state_from_pose(motion.pose(motion.time_s[k]), BODY)
            assert_allclose(gyroscopic_moment(state, momentum), gyro[k], rtol=1e-12, atol=1e-9)
            assert_allclose(reaction[k], 0., atol=1e-9)
        self.assertGreater(np.linalg.norm(gyro[0]), 10.)


class PinnedRunTests(unittest.TestCase):
    """With the body pinned to the samples of a reference motion the driver must give the prescribed-motion run."""

    def test_wing_alone_gives_the_loads_of_the_prescribed_run(self):
        steps = 5
        motion, wind = circle(steps)
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        reference = run_prescribed(backend(steps), wing, polar, motion, wind)
        start = state_from_pose(motion.pose(0.), BODY)
        run = run_free(backend(steps), wing, polar, BODY, start, wind, times=motion.time_s, pin_to=motion,
                       external=Release(BODY, motion))
        self.assertEqual((run.status, len(run.records), run.aero_solves), ("complete", steps+1, steps+1))
        for mine, theirs in zip(run.records, reference.records):
            scale = np.linalg.norm(theirs["force_inertial_N"])
            assert_allclose(mine["force_inertial_N"], theirs["force_inertial_N"], rtol=0, atol=1e-9*scale)
            assert_allclose(mine["moment_origin_inertial_Nm"], theirs["moment_origin_inertial_Nm"], rtol=0,
                            atol=1e-9*np.linalg.norm(theirs["moment_origin_inertial_Nm"]))
            self.assertAlmostEqual(mine["alpha_deg"], theirs["alpha_deg"], delta=1e-9)
            self.assertTrue(mine["pinned"])
        self.assertIn("plant_work_J", run.records[0])
        self.assertNotIn("plant_work_J", run.records[-1])           # no step follows the last record

    def test_hand_over_rotors_and_tether_give_the_records_of_the_prescribed_run(self):
        steps = 12
        times = np.arange(steps+1)*DT
        motion = pitch_manoeuvre(times, 25., 12., [(2*DT, 6., 3*DT, 3*DT)])           # 12 deg -> 18 deg: through the blend
        wind = np.zeros(3)
        degrees = np.array([-90., -14., 0., 14., 90.])
        low = SectionPolar(degrees, np.array([-1.3, -1.3, 0., 1.3, 1.3]), np.array([.3, .02, .008, .02, .3]), np.zeros(5), "low")
        handover = SectionHandover(low, P.section_polar(P.load_makani_airfoil(NACA), 0.), seeding="zero")
        rotors = PrescribedRotors(discs(), 150., nodes_per_ring=8)
        wing = Wing.ellipse(8)
        line = LumpedTether(TetherProperties(length_m=60.), TetherSettings(num_nodes=6, air_density_kg_m3=1.225))
        bridle = Bridle([0.1494, -0.5091, -0.13035], 4.786)
        first = motion.pose(0.)
        ground = first.position_m+np.array([30., 0., -55.])
        prescribed_tether = PrescribedTether(line, ground, bridle)
        nodes, speeds = prescribed_tether.start(motion)
        reference = run_prescribed(backend(steps), wing, None, motion, wind, handover=handover, rotors=rotors,
                                   tether=prescribed_tether, keep_tether_states=(0, 3))
        self.assertEqual(reference.status, "complete", reference.error)
        assert_allclose(reference.tether_states[0][0], nodes)
        self.assertEqual(sorted(reference.tether_states), [0, 3])
        run = run_free(backend(steps), wing, None, BODY, state_from_pose(first, BODY), wind, times=times, handover=handover,
                       rotors=rotors, rotor_inertia_kg_m2=1.56, tether=PlantTether(line, ground, bridle), nodes=nodes,
                       node_velocities=speeds, pin_to=motion, external=Release(BODY, motion))
        self.assertEqual(run.status, "complete", run.error)
        weights = [record["handover_weight"] for record in run.records]
        self.assertTrue(0. < max(weights) and min(weights) < max(weights))            # the blend was entered
        for mine, theirs in zip(run.records, reference.records):
            for key in ("force_inertial_N", "moment_origin_inertial_Nm", "rotor_force_inertial_N", "rotor_moment_origin_inertial_Nm"):
                assert_allclose(mine[key], theirs[key], rtol=0, atol=1e-8*max(1., np.linalg.norm(theirs[key])), err_msg=key)
            self.assertEqual((mine["closure"], mine["seeding"]), (theirs["closure"], theirs["seeding"]))
            self.assertAlmostEqual(mine["handover_weight"], theirs["handover_weight"], delta=1e-12)
        # The line is carried by the plant inside a step and by the interpolated pose in the prescribed run: close, not equal.
        mean = np.mean([record["tether_tension_kite_N"] for record in reference.records])
        gaps = [abs(a["tether_tension_kite_N"]-b["tether_tension_kite_N"]) for a, b in zip(run.records, reference.records)]
        self.assertLess(max(gaps), 2e-2*mean)
        self.assertGreater(mean, 100.)

    def test_keeping_tether_states_changes_no_step_of_the_prescribed_run(self):
        steps = 3
        motion = uniform_translation(np.arange(steps+1)*DT, [-1.5, 0., 1.0], origin=[-40., 0., 51.2])
        wind = np.array([12., 0., 0.8])
        line = LumpedTether(TetherProperties(length_m=60.), TetherSettings(num_nodes=6, air_density_kg_m3=1.225))
        tether = PrescribedTether(line, np.zeros(3), Bridle([0.1494, -0.5091, -0.13035], 4.786))
        args = (backend(steps), Wing.ellipse(8), SectionPolar.linear(limit_deg=60.), motion, wind)
        plain = run_prescribed(*args, tether=tether)
        kept = run_prescribed(*args, tether=tether, keep_tether_states=range(steps+1))
        self.assertEqual(plain.tether_states, {})
        self.assertEqual([r["tether_force_inertial_N"] for r in plain.records], [r["tether_force_inertial_N"] for r in kept.records])
        self.assertEqual([r["force_inertial_N"] for r in plain.records], [r["force_inertial_N"] for r in kept.records])
        assert_allclose(kept.tether_states[steps][0], kept.tether_state[0])


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.motion, self.wind = circle(6)
        self.release = Release(BODY, self.motion, release_time_s=3*DT)
        self.model = dict(force_inertial_N=np.array([100., -2000., 30000.]), moment_cm_inertial_Nm=np.array([50., 400., -90.]))

    def test_before_the_release_the_body_has_the_accelerations_of_the_reference(self):
        rows, _ = account([dict(step=k, time_s=float(t)) for k, t in enumerate(self.motion.time_s)], self.motion, BODY)
        for k in (0, 2):
            state = state_from_pose(self.motion.pose(self.motion.time_s[k]), BODY)
            held = HeldLoads(state.rotation.T@self.model["force_inertial_N"], state.rotation.T@self.model["moment_cm_inertial_Nm"])
            self.release.begin_step(self.motion.time_s[k])
            acceleration, spin = rates(BODY, state, held=held, external=self.release)
            assert_allclose(BODY.mass_kg*acceleration, rows[k]["inertia_force_inertial_N"], rtol=1e-12)
            inertia, omega = BODY.inertia_body_kg_m2, state.angular_velocity_body_rad_s
            assert_allclose(state.rotation@(inertia@spin+np.cross(omega, inertia@omega)),
                            rows[k]["angular_momentum_rate_inertial_Nm"], rtol=1e-10, atol=1e-8)

    def test_the_trim_is_taken_once_at_the_release_and_turns_with_the_body(self):
        at = state_from_pose(self.motion.pose(3*DT), BODY)
        expected = self.release._cancel(3*DT, self.model)
        self.release.begin_step(3*DT)
        force, moment = self.release(3*DT, at, self.model)
        assert_allclose(force, expected[0])
        assert_allclose(moment, expected[1])
        turned = RigidState(4*DT, at.position_cm_m+1., at.rotation@rotation_from_vector([0.2, -0.1, 0.3]), at.velocity_cm_m_s,
                            at.angular_velocity_body_rad_s)
        other = dict(force_inertial_N=np.ones(3), moment_cm_inertial_Nm=np.ones(3))       # the model loads no longer matter
        self.release.begin_step(4*DT)
        later = self.release(4*DT, turned, other)
        assert_allclose(turned.rotation.T@later[0], at.rotation.T@expected[0], rtol=1e-13)
        assert_allclose(turned.rotation.T@later[1], at.rotation.T@expected[1], rtol=1e-13)

    def test_without_trim_nothing_is_applied_after_the_release(self):
        release = Release(BODY, self.motion, release_time_s=DT, trim=False)
        state = state_from_pose(self.motion.pose(2*DT), BODY)
        release.begin_step(2*DT)
        assert_allclose(release(2*DT, state, self.model), 0.)
        self.assertIsNone(release.trim_body)
        with self.assertRaises(ValueError):
            release.reference(1e3)


    def test_the_release_is_decided_by_the_step_not_by_the_stage_time(self):
        # The last stage of the step before the release is at the release time; it still belongs to that step.
        at = state_from_pose(self.motion.pose(3*DT), BODY)
        with self.assertRaises(ValueError):
            self.release(3*DT, at, self.model)                # a release time needs the driver's begin_step
        self.release.begin_step(2*DT)
        force, moment = self.release(3*DT, at, self.model)
        self.assertIsNone(self.release.trim_body)
        assert_allclose(force, self.release._cancel(3*DT, self.model)[0])
        self.release.begin_step(3*DT)
        other = dict(force_inertial_N=np.array([7., 8., 9.]), moment_cm_inertial_Nm=np.array([1., 2., 3.]))
        self.release(3*DT, at, other)
        assert_allclose(at.rotation@self.release.trim_body[0], self.release._cancel(3*DT, other)[0])

    def test_through_the_plant_the_trim_is_built_from_the_loads_of_the_release_step(self):
        # Two steps with different held loads, as a driver makes them: the trim must be the contract's expression
        # m a_ref(t0) - m g - F(t0) with the loads of the step that starts at t0, at the state that step starts from.
        before, at_release = HeldLoads([100., -200., 5000.], [300., -800., 200.]), HeldLoads([400., 100., 7000.], [-100., 500., 900.])
        state = state_from_pose(self.motion.pose(2*DT), BODY)
        self.release.begin_step(2*DT)
        plant_step(BODY, RigidState(2*DT, state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s),
                   step_s=DT, substeps=10, held=before, external=self.release)
        self.assertIsNone(self.release.trim_body)             # the stage at t0 that ends the step before did not release
        start = state_from_pose(self.motion.pose(3*DT), BODY)
        start = RigidState(3*DT, start.position_cm_m, start.rotation, start.velocity_cm_m_s, start.angular_velocity_body_rad_s)
        self.release.begin_step(3*DT)
        acceleration, spin = rates(BODY, start, held=at_release, external=self.release)
        inertia_force, momentum_rate = self.release.reference(3*DT)
        assert_allclose(self.release.trim_body[0], start.rotation.T@(inertia_force-BODY.mass_kg*G)-at_release.force_body_N, rtol=1e-12)
        assert_allclose(self.release.trim_body[1], start.rotation.T@momentum_rate-at_release.moment_cm_body_Nm, rtol=1e-10, atol=1e-8)
        assert_allclose(BODY.mass_kg*acceleration, inertia_force, rtol=1e-12, atol=1e-9)   # the release starts on the reference acceleration
        inertia, omega = BODY.inertia_body_kg_m2, start.angular_velocity_body_rad_s
        assert_allclose(start.rotation@(inertia@spin+np.cross(omega, inertia@omega)), momentum_rate, rtol=1e-10, atol=1e-8)


class GyroscopicTests(unittest.TestCase):
    def test_a_torque_free_body_with_a_spinning_rotor_keeps_the_total_angular_momentum(self):
        # Body plus rotor: d/dt [R (I w + L)] = 0 when the moment on the body is L x w (L fixed in body axes).
        momentum = np.array([900., 0., 50.])
        inertia = BODY.inertia_body_kg_m2

        def run(sign):
            state = RigidState(0., [0., 0., 0.], R0, [1., 2., 3.], [0.3, -0.2, 0.5])
            total = state.rotation@(inertia@state.angular_velocity_body_rad_s+momentum)
            for _ in range(50):
                state, _, _, _ = plant_step(BODY, state, step_s=DT, substeps=10, gravity_m_s2=np.zeros(3),
                                            external=lambda time, stage, model: (np.zeros(3), sign*gyroscopic_moment(stage, momentum)))
            return total, state.rotation@(inertia@state.angular_velocity_body_rad_s+momentum), state

        total, after, state = run(1.)
        assert_allclose(after, total, rtol=0, atol=1e-8*np.linalg.norm(total))
        self.assertGreater(np.linalg.norm(state.angular_velocity_body_rad_s-[0.3, -0.2, 0.5]), 1e-3)   # the rotor does act
        total, after, _ = run(-1.)                                # the other sign does not conserve it
        self.assertGreater(np.linalg.norm(after-total), 1e-3*np.linalg.norm(total))


class FreeRunTests(unittest.TestCase):
    def setUp(self):
        self.wing, self.polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)

    def test_a_body_held_on_a_straight_path_by_a_cancelling_load_stays_on_it_without_pinning(self):
        steps = 5
        motion = uniform_translation(np.arange(steps+1)*DT, [-30., 0., 0.], rotation=rotation_from_vector([0., -0.05, 0.]),
                                     origin=[0., 0., 100.])
        wind = np.array([9., 0., 0.])
        reference = run_prescribed(backend(steps), self.wing, self.polar, motion, wind)
        run = run_free(backend(steps), self.wing, self.polar, BODY, state_from_pose(motion.pose(0.), BODY), wind,
                       times=motion.time_s, external=Release(BODY, motion))
        self.assertEqual(run.status, "complete", run.error)
        for mine, theirs in zip(run.records, reference.records):
            assert_allclose(mine["position_m"], theirs["position_m"], rtol=0, atol=1e-10)
            assert_allclose(mine["force_inertial_N"], theirs["force_inertial_N"], rtol=0,
                            atol=1e-9*np.linalg.norm(theirs["force_inertial_N"]))
            self.assertFalse(mine["pinned"])

    def test_released_flight_moves_under_its_loads_and_the_work_account_is_the_kinetic_energy(self):
        steps = 6
        motion, wind = circle(steps)
        release = Release(BODY, motion, release_time_s=2*DT)
        run = run_free(backend(steps), self.wing, self.polar, BODY, state_from_pose(motion.pose(0.), BODY), wind,
                       times=motion.time_s, pin_to=motion, pin_until=2, external=release)
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual([record["pinned"] for record in run.records], [True, True, True, False, False, False, False])
        self.assertIsNotNone(release.trim_body)
        # The trim is the contract's expression with the loads of the release step (record 2), not of the step before.
        at = state_from_pose(motion.pose(2*DT), BODY)
        inertia_force, _ = release.reference(2*DT)
        expected = at.rotation.T@(inertia_force-BODY.mass_kg*G)
        assert_allclose(release.trim_body[0], expected-run.records[2]["held_force_body_N"], rtol=1e-11)
        self.assertGreater(np.linalg.norm(release.trim_body[0]-(expected-run.records[1]["held_force_body_N"])), 1e-3)
        # At the release the trim balances the reference acceleration; afterwards nothing corrects, so the path drifts.
        drift = np.linalg.norm(np.array(run.records[-1]["position_m"])-motion.position_m[-1])
        self.assertTrue(1e-9 < drift < 0.5)
        for k in range(2, steps):
            record, after = run.records[k], run.records[k+1]
            work = sum(record["plant_work_J"].values())
            self.assertAlmostEqual(after["kinetic_energy_J"]-record["kinetic_energy_J"], work,
                                   delta=1e-8*sum(abs(v) for v in record["plant_work_J"].values()))
        final = run.state
        self.assertEqual(final.time_s, motion.time_s[-1])
        assert_allclose(sum(kinetic_energy(BODY, final)), run.records[-1]["kinetic_energy_J"])

    def test_the_predictor_corrector_solves_twice_per_step_and_stays_close_to_the_held_scheme(self):
        steps = 4
        motion, wind = circle(steps)
        runs = {}
        for scheme in ("hold", "heun"):
            runs[scheme] = run_free(backend(steps), self.wing, self.polar, BODY, state_from_pose(motion.pose(0.), BODY), wind,
                                    times=motion.time_s, external=Release(BODY, motion, release_time_s=0.), scheme=scheme)
            self.assertEqual(runs[scheme].status, "complete", runs[scheme].error)
        self.assertEqual((runs["hold"].aero_solves, runs["heun"].aero_solves), (steps+1, 2*steps+1))
        self.assertIn("predictor_force_body_N", runs["heun"].records[0])
        self.assertNotIn("predictor_force_body_N", runs["hold"].records[0])
        gap = np.linalg.norm(np.array(runs["hold"].records[-1]["position_cm_m"])-runs["heun"].records[-1]["position_cm_m"])
        self.assertTrue(0. < gap < 1e-2)
        # The first record is the same solve in both schemes.
        self.assertEqual(runs["hold"].records[0]["force_inertial_N"], runs["heun"].records[0]["force_inertial_N"])

    def test_a_failed_step_stops_the_run_names_the_part_and_keeps_the_earlier_records(self):
        steps = 4
        motion, wind = circle(steps)
        args = (backend(steps), self.wing, self.polar, BODY, state_from_pose(motion.pose(0.), BODY), wind)
        calls = []

        def failing_plant(*a, **k):
            calls.append(1)
            if len(calls) == 3:
                raise FloatingPointError("tether state left the finite range (time step too large?)")
            return free_flight_plant(*a, **k)
        free_flight_plant = free_flight.plant_step
        with mock.patch.object(free_flight, "plant_step", side_effect=failing_plant):
            run = run_free(*args, times=motion.time_s, external=Release(BODY, motion))
        self.assertEqual((run.status, run.failed_step, run.failed_module, len(run.records)), ("failed", 2, "plant", 2))
        self.assertIn("finite range", run.error)
        narrow = SectionPolar.linear(limit_deg=1.0)                 # the incidence leaves this table at the first solve
        run = run_free(args[0], self.wing, narrow, *args[3:], times=motion.time_s)
        self.assertEqual((run.status, run.failed_step, run.failed_module, run.records), ("failed", 0, "aerodynamics", []))

    def test_set_up_is_checked(self):
        steps = 2
        motion, wind = circle(steps)
        start = state_from_pose(motion.pose(0.), BODY)
        args = (backend(steps), self.wing, self.polar, BODY, start, wind)
        line = LumpedTether(TetherProperties(length_m=60.), TetherSettings(num_nodes=6, air_density_kg_m3=1.225))
        tether = PlantTether(line, np.zeros(3), Bridle([0.1494, -0.5091, -0.13035], 4.786))
        nodes = np.zeros((6, 3))
        with self.assertRaises(ValueError):
            run_free(*args, times=np.arange(steps+1)*0.03)                      # not the wake time step
        with self.assertRaises(ValueError):
            run_free(*args, times=motion.time_s, scheme="rk4")
        with self.assertRaises(ValueError):
            run_free(*args, times=motion.time_s, nodes=nodes, node_velocities=nodes)   # a node state without a tether
        with self.assertRaises(ValueError):
            run_free(*args, times=motion.time_s, tether=tether)                 # a tether without its node state
        with self.assertRaises(ValueError):
            run_free(*args, times=motion.time_s, rotor_inertia_kg_m2=1.5)       # an inertia without rotors
        with self.assertRaises(ValueError):
            run_free(*args, times=motion.time_s, tether=tether, nodes=nodes, node_velocities=nodes, substeps=1)   # above the bound
        with self.assertRaises(ValueError):
            run_free(*args, times=motion.time_s, pin_to="motion")
        degrees = np.array([-90., -14., 0., 14., 90.])
        low = SectionPolar(degrees, np.array([-1.3, -1.3, 0., 1.3, 1.3]), np.array([.3, .02, .008, .02, .3]), np.zeros(5), "low")
        panel = SectionHandover(low, P.section_polar(P.load_makani_airfoil(NACA), 0.), weight_basis="panel")
        with self.assertRaises(ValueError):
            run_free(args[0], self.wing, None, *args[3:], times=motion.time_s, handover=panel)


if __name__ == "__main__":
    unittest.main()
