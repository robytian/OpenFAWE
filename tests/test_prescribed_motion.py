"""Vortex-wake aerodynamics along a prescribed motion (G4 contract: K2 panel velocity, K1 in small)."""
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero import polars as P
from openfawe.aero.aircraft import WingAssembly
from openfawe.aero.handover import (TAKEOFF_LANDING_SOLVE, BlendedPolars, PanelBlendedPolars, SectionHandover, makani_weight,
                                    panel_weights)
from openfawe.aero.methods import AeroBackend
from openfawe.aero.nonlinear import NonlinearSolveError
from openfawe.aero.vsm import geometry
from openfawe.aero.vsm_wake import advance_vsm_wake, start_vsm_wake
from openfawe.aero.wake import WakeConfig, ring_induction
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing
from openfawe.flight import prescribed
from openfawe.flight.kinematics import RigidMotion, circular_motion, pitch_manoeuvre, uniform_translation
from openfawe.flight.prescribed import PrescribedRotors, PrescribedTether, run_prescribed, settle_tether
from openfawe.flight.residual import RigidBody, account
from openfawe.tether.bridle import Bridle
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings
from openfawe.rotor.actuator_disc import ActuatorDisc, disc_basis
from openfawe.rotor.coupled import advance_with_rotors, rotor_totals
from openfawe.rotor.table import RotorTable

NACA = Path(__file__).resolve().parents[1] / "cases/polars/makani/naca0012_high_aoa.json"

NEWTON = SolveSettings(solver="newton", max_iterations=60)
RADIUS, SPEED, WIND, ALPHA, DT = 130.0, 47.0, 9.0, 3.0, 0.02


def backend(steps, dt=DT):
    return AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=0.15, solve=NEWTON))


class PanelVelocityTests(unittest.TestCase):
    def test_k2_closure_and_load_velocities_are_wind_minus_body_velocity(self):
        # Wide table: at the impulsive start the tip panels of this coarse wing reach about -19 deg.
        wing, polar = Wing.ellipse(10, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 3
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind = np.array([WIND, 0., 0.])
        run = run_prescribed(backend(steps), wing, polar, motion, wind, keep_states=range(steps+1))
        self.assertEqual(run.status, "complete")
        for index, state in run.states.items():
            pose = motion.pose(motion.time_s[index])
            placed = wing.transformed(pose.rotation, pose.position_m)
            closure_points = geometry(placed)[0]
            expected = wind-(pose.velocity_m_s+np.cross(pose.angular_velocity_rad_s, closure_points-pose.position_m))
            scale = np.linalg.norm(expected, axis=1).max()
            assert_allclose(state.solution.closure_ambient_velocity, expected, rtol=0, atol=1e-12*scale)
            load = wind-(pose.velocity_m_s+np.cross(pose.angular_velocity_rad_s, placed.cp-pose.position_m))
            assert_allclose(state.solution.relative_velocity-state.solution.induced_velocity, load, rtol=0, atol=1e-12*scale)
            # Independent closed form for a circle about the X axis through the origin: body velocity
            # Omega e_X x p, without the pose's origin velocity or position.
            spin = np.array([SPEED/RADIUS, 0., 0.])
            assert_allclose(state.solution.closure_ambient_velocity, wind-np.cross(spin, closure_points), rtol=0, atol=1e-12*scale)
            assert_allclose(state.solution.relative_velocity-state.solution.induced_velocity,
                            wind-np.cross(spin, placed.cp), rtol=0, atol=1e-12*scale)
        # The panel speed grows with the radius: outer tip against inner tip.
        speeds = np.linalg.norm(run.states[0].solution.closure_ambient_velocity, axis=1)
        self.assertGreater(speeds[-1], speeds[0])


    def test_k2_holds_on_an_assembly_of_surfaces(self):
        def rectangle(panels, span, chord, offset):
            y = np.linspace(-.5*span, .5*span, panels+1)
            leading = np.column_stack([np.zeros_like(y), y, np.zeros_like(y)])+np.asarray(offset)
            return Wing(leading, leading+np.array([chord, 0., 0.]))
        assembly = WingAssembly((rectangle(6, 25.0, 1.3, (0., 0., 0.)), rectangle(4, 4.0, 0.8, (6.5, 0., 0.4))), ("wing", "tail"))
        motion = circular_motion(np.arange(3)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind, spin = np.array([WIND, 0., 0.]), np.array([SPEED/RADIUS, 0., 0.])
        run = run_prescribed(backend(2), assembly, SectionPolar.linear(limit_deg=60.), motion, wind, keep_states=(0, 2))
        self.assertEqual(run.status, "complete")
        for index, state in run.states.items():
            pose = motion.pose(motion.time_s[index])
            placed = assembly.transformed(pose.rotation, pose.position_m)
            self.assertEqual(state.solution.closure_ambient_velocity.shape, (10, 3))
            assert_allclose(state.solution.closure_ambient_velocity, wind-np.cross(spin, geometry(placed)[0]), rtol=0, atol=1e-10)


class PrescribedRunTests(unittest.TestCase):
    def test_translation_through_still_air_equals_the_fixed_wing_in_a_wind(self):
        # Wing set off the body origin, so that the moment about the origin is not zero by symmetry.
        wing, polar = Wing.ellipse(8).transformed(np.eye(3), np.array([0.5, 0.3, 0.1])), SectionPolar.linear()
        wind, steps, dt = np.array([12., 0., 0.8]), 6, 0.05
        fixed = backend(steps, dt)
        state = fixed.start(wing, polar, wind)
        reference, moments = [state.solution.force], [state.solution.moment(wing)]
        for index in range(1, steps+1):
            state = fixed.advance(state, polar, wind, next_time_s=index*dt)
            reference.append(state.solution.force)
            moments.append(state.solution.moment(wing))
        motion = uniform_translation(np.arange(steps+1)*dt, -wind)
        run = run_prescribed(backend(steps, dt), wing, polar, motion, np.zeros(3))
        self.assertEqual(run.status, "complete")
        self.assertEqual(len(run.records), steps+1)
        for record, force, moment in zip(run.records, reference, moments):
            assert_allclose(record["force_inertial_N"], force, rtol=1e-9, atol=1e-9*np.linalg.norm(force))
            # The moment about the moving body origin equals the fixed wing's moment about its origin.
            assert_allclose(record["moment_origin_inertial_Nm"], moment, rtol=1e-9, atol=1e-9*np.linalg.norm(moment))
            assert_allclose(record["moment_origin_body_Nm"], moment, rtol=1e-9, atol=1e-9*np.linalg.norm(moment))
            assert_allclose(record["force_body_N"], force, rtol=1e-9, atol=1e-9*np.linalg.norm(force))
            self.assertAlmostEqual(record["airspeed_m_s"], np.linalg.norm(wind), places=12)
            self.assertAlmostEqual(record["alpha_deg"], np.degrees(np.arctan2(0.8, 12.)), places=10)
            self.assertAlmostEqual(record["aero_power_W"], float(np.dot(force, -wind)), delta=1e-8*abs(np.dot(force, wind)))

    def test_circle_run_reports_body_loads_and_flight_condition(self):
        # Wide table: at the impulsive start the tip panels of this coarse wing reach about -19 deg.
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 5
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        run = run_prescribed(backend(steps), wing, polar, motion, [WIND, 0., 0.], keep_states=(steps,))
        self.assertEqual(run.status, "complete")
        last = run.records[-1]
        # Power of the loads on the turning body, summed panel by panel with each panel's own velocity.
        solution, pose = run.states[steps].solution, motion.pose(motion.time_s[steps])
        placed = wing.transformed(pose.rotation, pose.position_m)
        panel_power = (np.sum(solution.panel_force*pose.point_velocity(placed.cp))
                       + np.sum(solution.panel_couple@pose.angular_velocity_rad_s))
        self.assertAlmostEqual(last["aero_power_W"], panel_power, delta=1e-9*abs(panel_power))
        self.assertLessEqual(run.motion_consistency["velocity_vs_position"]["relative"], 1e-6)
        self.assertAlmostEqual(last["airspeed_m_s"], np.hypot(WIND, SPEED), places=10)
        self.assertAlmostEqual(last["alpha_deg"], ALPHA, places=10)
        self.assertGreater(last["force_body_N"][2], 0.)                    # lift along body z
        # The body-frame force is the inertial force seen from the turning body: same magnitude.
        self.assertAlmostEqual(np.linalg.norm(last["force_body_N"]), np.linalg.norm(last["force_inertial_N"]), places=8)
        # The lift pulls downwind (+X), against which a tether would act.
        self.assertGreater(last["force_inertial_N"][0], 0.)

    def test_a_failing_step_stops_the_run_and_is_recorded(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear(limit_deg=2.0)        # polar table ends at 2 deg
        steps = 3
        motion = uniform_translation(np.arange(steps+1)*0.05, [-10., 0., -3.])   # incidence about 17 deg
        run = run_prescribed(backend(steps, 0.05), wing, polar, motion, np.zeros(3))
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.failed_step, 0)
        self.assertEqual(run.records, [])
        self.assertEqual(run.error_type, "ValueError")
        self.assertIn("outside the declared polar table", run.error)

    def test_a_failure_after_good_steps_keeps_their_records_and_the_last_good_state(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear(limit_deg=30.0)
        times = np.arange(4)*0.05
        velocity = np.array([[-10., 0., -0.3], [-10., 0., -0.3], [-10., 0., -9.], [-10., 0., -9.]])   # 1.7 deg, then 42 deg
        position = np.vstack([np.zeros(3), np.cumsum(.05*velocity[:-1], axis=0)])
        motion = RigidMotion(times, position, np.broadcast_to(np.eye(3), (4, 3, 3)), velocity, np.zeros((4, 3)))
        run = run_prescribed(backend(3, 0.05), wing, polar, motion, np.zeros(3))
        self.assertEqual((run.status, run.failed_step, len(run.records)), ("failed", 2, 2))
        self.assertAlmostEqual(run.final_state.time_s, 0.05, places=14)
        self.assertEqual(run.final_state.step, 1)

    def test_a_fixed_point_solver_that_does_not_converge_is_a_recorded_failure(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear()
        slow = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=0.05, max_rows=3, core_radius_m=0.15,
                                                            solve=SolveSettings(max_iterations=1)))
        run = run_prescribed(slow, wing, polar, uniform_translation(np.arange(4)*0.05, [-10., 0., -0.5]), np.zeros(3))
        self.assertEqual((run.status, run.failed_step, run.error_type), ("failed", 0, "CirculationConvergenceError"))
        self.assertEqual(run.failed_iterations, 1)

    def test_set_up_mistakes_raise_instead_of_being_booked_as_failed_steps(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear()
        motion = uniform_translation(np.arange(4)*0.05, [-10., 0., -0.5])
        with self.assertRaises(ValueError):                       # wake row limit below the number of steps
            run_prescribed(backend(2, 0.05), wing, polar, motion, np.zeros(3))
        with self.assertRaises(ValueError):                       # wind is not a (3,) vector
            run_prescribed(backend(3, 0.05), wing, polar, motion, np.zeros(2))
        times = np.array([0., 0.05, 0.1+1e-12, 0.15+1e-12])       # passes a loose check, not the wake's own
        with self.assertRaises(ValueError):
            run_prescribed(backend(3, 0.05), wing, polar, uniform_translation(times, [-10., 0., -0.5]), np.zeros(3))

    def test_a_truncated_wake_runs_past_the_row_limit(self):
        wing, polar = Wing.ellipse(8), SectionPolar.linear()
        steps, dt = 9, 0.05
        motion = uniform_translation(np.arange(steps+1)*dt, [-12., 0., -0.8])
        cut = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=dt, max_rows=4, core_radius_m=0.15, solve=NEWTON, retained_rows=4))
        run = run_prescribed(cut, wing, polar, motion, np.zeros(3))
        self.assertEqual((run.status, len(run.records), run.final_state.step, len(run.final_state.free_rows)), ("complete", 10, 9, 4))
        with self.assertRaises(ValueError):                       # the same row limit without the truncation
            run_prescribed(backend(4, dt), wing, polar, motion, np.zeros(3))

    def test_motion_spacing_must_match_the_wake_time_step(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear()
        motion = uniform_translation(np.arange(4)*0.1, [-10., 0., 0.])
        with self.assertRaises(ValueError):
            run_prescribed(backend(3, 0.05), wing, polar, motion, np.zeros(3))


class HandoverRunTests(unittest.TestCase):
    """The driver with the section hand-over: which model each step uses, the seeding and its bookkeeping."""
    STEPS = 50

    def setUp(self):
        self.wing = Wing.ellipse(8)
        # Crosswind section: linear to 14 deg, constant beyond (a linear table to 60 deg has no root past stall).
        degrees = np.array([-90., -14., 0., 14., 90.])
        self.low = SectionPolar(degrees, np.array([-1.3, -1.3, 0., 1.3, 1.3]), np.array([.3, .02, .008, .02, .3]),
                                np.zeros(5), "test crosswind section")
        self.high = P.section_polar(P.load_makani_airfoil(NACA), 0.)
        dt = DT
        # 30 deg -> 3 deg -> 30 deg: 5 steps dwell, 15-step moves (ramp 5, cruise 5), 10 steps at 3 deg, 5 steps dwell.
        self.times = np.arange(self.STEPS+1)*dt
        self.motion = pitch_manoeuvre(self.times, 25., 30., [(5*dt, -27., 5*dt, 5*dt), (30*dt, 27., 5*dt, 5*dt)])

    def run_with(self, seeding, **options):
        return run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3),
                              handover=SectionHandover(self.low, self.high, seeding=seeding), **options)

    def test_inside_the_crosswind_range_the_run_is_the_production_run_bit_for_bit(self):
        steps, dt = 6, 0.05
        motion = uniform_translation(np.arange(steps+1)*dt, [-12., 0., -0.8])            # incidence 3.8 deg
        low = SectionPolar.linear()
        plain = run_prescribed(backend(steps, dt), self.wing, low, motion, np.zeros(3))
        handed = run_prescribed(backend(steps, dt), self.wing, None, motion, np.zeros(3),
                                handover=SectionHandover(low, Forbidden(), seeding="zero"))
        self.assertEqual((plain.status, handed.status), ("complete", "complete"))
        for a, b in zip(plain.records, handed.records):
            for name in ("force_inertial_N", "moment_origin_inertial_Nm", "aero_power_W", "iterations"):
                self.assertTrue(np.array_equal(a[name], b[name]), name)
            self.assertEqual((b["handover_weight"], b["closure"], b["stall_switches"]), (0., "source", None))
        self.assertEqual([r["seeding"] for r in handed.records], ["start"]+["previous"]*steps)

    def test_pitch_through_the_hand_over_uses_each_model_where_the_rule_says(self):
        run = self.run_with("previous")
        self.assertEqual(run.status, "complete", run.error)
        weight = np.array([r["handover_weight"] for r in run.records])
        alpha = np.array([r["alpha_deg"] for r in run.records])
        assert_allclose(weight[:6], 1., atol=0.)
        assert_allclose(weight[20:31], 0., atol=0.)                                       # dwell at 3 deg
        assert_allclose(weight[-6:], 1., atol=0.)
        inside = (alpha > 5.) & (alpha < 15.)
        self.assertTrue(inside.any() and np.all((weight[inside] > 0.) & (weight[inside] < 1.)))
        for record in run.records:
            # ``closure`` is read from the solution: this is what the solver actually used.
            expected = "source" if record["handover_weight"] == 0. else "full_incidence"
            self.assertEqual(record["closure"], expected)
            self.assertEqual(record["stall_switches"] is None, expected == "source")
            self.assertEqual(record["regime_counts"] is None, expected == "source")
            self.assertIn("total_steps_over_limit", record["incidence_steps"])
            self.assertIsNone(record["zero_seed_failure"])
        self.assertEqual({r["seeding"] for r in run.records[1:]}, {"previous"})
        # One wake for the whole run: a row is released every step, whichever model the step used.
        self.assertEqual(run.final_state.step, self.STEPS)
        self.assertEqual(len(run.final_state.free_rows), self.STEPS)

    def test_out_of_range_option_records_the_panels_only_inside_the_blend_and_changes_nothing_when_none_is_outside(self):
        # The low table of this fixture spans +-60 deg and the manoeuvre stays inside it: the option changes no value and
        # the per-step list is empty inside the blend, None at weight 0 and 1.
        plain = self.run_with("previous")
        option = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3),
                                handover=SectionHandover(self.low, self.high, seeding="previous", out_of_range="hover"))
        self.assertEqual(option.status, "complete", option.error)
        for a, b in zip(plain.records, option.records):
            self.assertTrue(np.array_equal(a["force_body_N"], b["force_body_N"]))
            inside = 0. < b["handover_weight"] < 1.
            self.assertEqual(b["panels_outside_crosswind_table"], [] if inside else None)
            self.assertEqual(b["most_panels_outside_crosswind_table"], 0 if inside else None)
            self.assertIsNone(a["panels_outside_crosswind_table"])
        self.assertIsNone(option.failed_panels_outside_crosswind_table)

    def test_hover_side_steps_are_the_solver_called_with_the_blended_data_and_the_prescribed_setting(self):
        weight = [prescribed.relative_wind_body(self.motion.pose(t), np.zeros(3))[2] for t in self.times]
        handover = SectionHandover(self.low, self.high)
        blend = next(k for k in range(1, self.STEPS) if 0.2 < handover.select(weight[k], 0.).weight < 0.8)
        run = self.run_with("previous", keep_states=(0, blend-1, blend), steps=blend)
        self.assertEqual(run.status, "complete", run.error)
        config = replace(backend(self.STEPS).wake, solve=TAKEOFF_LANDING_SOLVE)
        # Step 0 (weight one): a direct start with the hover data alone.
        pose = self.motion.pose(self.times[0])
        placed = self.wing.transformed(pose.rotation, pose.position_m)
        direct = start_vsm_wake(placed, BlendedPolars(self.low, self.high, 1.0), config, np.zeros(3), time_s=pose.time_s,
                                structural_velocity=self.motion.structural_velocity)
        self.assertTrue(np.array_equal(run.states[0].solution.circulation, direct.solution.circulation))
        self.assertTrue(np.array_equal(run.states[0].solution.panel_force, direct.solution.panel_force))
        # A step inside the blend: a direct advance from the stored previous state with the mixture of that step.
        pose = self.motion.pose(self.times[blend])
        placed = self.wing.transformed(pose.rotation, pose.position_m)
        w = run.records[blend]["handover_weight"]
        self.assertTrue(0.2 < w < 0.8)
        previous = replace(run.states[blend-1], config=config)
        direct = advance_vsm_wake(previous, BlendedPolars(self.low, self.high, w), config, np.zeros(3), next_wing=placed,
                                  structural_velocity=self.motion.structural_velocity, next_time_s=pose.time_s)
        self.assertTrue(np.array_equal(run.states[blend].solution.circulation, direct.solution.circulation))
        self.assertTrue(np.array_equal(run.states[blend].free_rows, direct.free_rows))
        # The mixture matters: the hover data alone give another circulation at this step.
        other = advance_vsm_wake(previous, BlendedPolars(self.low, self.high, 1.0), config, np.zeros(3), next_wing=placed,
                                 structural_velocity=self.motion.structural_velocity, next_time_s=pose.time_s)
        self.assertGreater(np.abs(other.solution.circulation-direct.solution.circulation).max(), 1e-3)
        # Re-binding between the two settings leaves the wake history as it was.
        self.assertTrue(np.array_equal(previous.free_rows, run.states[blend-1].free_rows))
        self.assertTrue(np.array_equal(previous.older_circulation, run.states[blend-1].older_circulation))

    def test_the_hand_over_keeps_the_far_wake_limits_when_it_switches_the_solve_settings(self):
        limited = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=DT, max_rows=12, core_radius_m=0.15, solve=NEWTON,
                                                               retained_rows=12, free_convection_rows=6))
        run = run_prescribed(limited, self.wing, None, self.motion, np.zeros(3), handover=SectionHandover(self.low, self.high))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual((run.final_state.step, len(run.final_state.free_rows)), (self.STEPS, 12))
        self.assertEqual((run.final_state.config.retained_rows, run.final_state.config.free_convection_rows), (12, 6))
        self.assertEqual({r["closure"] for r in run.records}, {"source", "full_incidence"})

    def test_zero_seeding_applies_to_the_hover_side_only_and_is_recorded(self):
        run = self.run_with("zero")
        self.assertEqual(run.status, "complete", run.error)
        for record in run.records[1:]:
            self.assertEqual(record["seeding"], "previous" if record["handover_weight"] == 0. else "zero")
        self.assertEqual(run.records[0]["seeding"], "start")
        self.assertIn("zero", {r["seeding"] for r in run.records})
        # No claim that the two seedings give the same loads: past stall the solution is not unique and the
        # wake history differs afterwards (decision 0025 amendment 2); the case scripts report both.

    def test_a_failed_zero_seeded_solve_is_retried_once_from_the_previous_circulation(self):
        original, calls = prescribed.advance_vsm_wake, []

        def flaky(state, *args, initial_circulation=None, **kwargs):
            calls.append((state.step+1, initial_circulation is not None))
            if initial_circulation is not None and state.step+1 in (2, 3):
                raise NonlinearSolveError("declared test failure of the zero-seeded solve", iterations=7)
            return original(state, *args, initial_circulation=initial_circulation, **kwargs)
        with mock.patch.object(prescribed, "advance_vsm_wake", flaky):
            run = self.run_with("zero", steps=4)
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual([r["seeding"] for r in run.records], ["start", "zero", "retry_previous", "retry_previous", "zero"])
        self.assertEqual(calls, [(1, True), (2, True), (2, False), (3, True), (3, False), (4, True)])
        # The cause of every retry is kept with the step.
        self.assertEqual(run.records[2]["zero_seed_failure"],
                         dict(type="NonlinearSolveError", message="declared test failure of the zero-seeded solve", iterations=7))
        self.assertIsNone(run.records[1]["zero_seed_failure"])
        self.assertIsNone(run.records[4]["zero_seed_failure"])

    def test_when_every_zero_seeded_solve_fails_the_run_is_the_previous_seeded_run_bit_for_bit(self):
        original = prescribed.advance_vsm_wake

        def never_from_zero(state, *args, initial_circulation=None, **kwargs):
            if initial_circulation is not None:
                raise NonlinearSolveError("declared test failure of the zero-seeded solve", iterations=1)
            return original(state, *args, initial_circulation=None, **kwargs)
        with mock.patch.object(prescribed, "advance_vsm_wake", never_from_zero):
            retried = self.run_with("zero")
        plain = self.run_with("previous")
        self.assertEqual((retried.status, plain.status), ("complete", "complete"))
        for a, b in zip(retried.records, plain.records):
            self.assertTrue(np.array_equal(a["force_inertial_N"], b["force_inertial_N"]))
            self.assertEqual(a["iterations"], b["iterations"])
            if a["step"] and a["handover_weight"] > 0.:
                self.assertEqual((a["seeding"], a["zero_seed_failure"]["iterations"]), ("retry_previous", 1))
        self.assertTrue(np.array_equal(retried.final_state.free_rows, plain.final_state.free_rows))

    def test_the_step_fails_only_when_the_retry_fails_too_and_both_errors_are_kept(self):
        original = prescribed.advance_vsm_wake

        def failing(state, *args, initial_circulation=None, **kwargs):
            if state.step+1 == 3:
                raise NonlinearSolveError("from zero" if initial_circulation is not None else "from previous",
                                          iterations=5 if initial_circulation is not None else 9)
            return original(state, *args, initial_circulation=initial_circulation, **kwargs)
        with mock.patch.object(prescribed, "advance_vsm_wake", failing):
            run = self.run_with("zero", steps=4)
        self.assertEqual((run.status, run.failed_step, run.error_type, len(run.records)), ("failed", 3, "NonlinearSolveError", 3))
        self.assertIn("from previous", run.error)
        self.assertEqual(run.failed_iterations, 9)
        self.assertEqual(run.zero_seed_failure, dict(type="NonlinearSolveError", message="from zero", iterations=5))
        self.assertEqual(run.final_state.step, 2)

    def test_a_step_without_relative_wind_is_a_recorded_failure_that_keeps_the_earlier_records(self):
        times = np.arange(4)*DT
        velocity = np.array([[-25., 0., -1.], [-25., 0., -1.], [0., 0., 0.], [0., 0., 0.]])
        position = np.vstack([np.zeros(3), np.cumsum(DT*velocity[:-1], axis=0)])
        motion = RigidMotion(times, position, np.broadcast_to(np.eye(3), (4, 3, 3)), velocity, np.zeros((4, 3)))
        run = run_prescribed(backend(3), self.wing, None, motion, np.zeros(3), handover=SectionHandover(SectionPolar.linear(), self.high))
        self.assertEqual((run.status, run.failed_step, run.error_type, len(run.records)), ("failed", 2, "ValueError", 2))
        self.assertIn("no relative wind", run.error)
        self.assertEqual(run.final_state.step, 1)

    def test_hand_over_set_up_is_checked(self):
        handover = SectionHandover(self.low, self.high)
        with self.assertRaises(ValueError):                       # section data given twice
            run_prescribed(backend(self.STEPS), self.wing, self.low, self.motion, np.zeros(3), handover=handover)
        with self.assertRaises(ValueError):                       # not a SectionHandover
            run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3), handover=dict(seeding="zero"))
        straight = AeroBackend("vsm", vsm_solve=NEWTON)
        with self.assertRaises(ValueError):                       # the hand-over needs the free-wake method
            run_prescribed(straight, self.wing, None, self.motion, np.zeros(3), handover=handover)
        other = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=DT, max_rows=self.STEPS, core_radius_m=0.15,
                                                             solve=SolveSettings(solver="newton", max_iterations=30)))
        with self.assertRaises(ValueError):                       # weight zero would not be the backend's own step
            run_prescribed(other, self.wing, None, self.motion, np.zeros(3), handover=handover)


class PanelBasisRunTests(unittest.TestCase):
    """The driver with one hand-over weight per panel (option (c) of owner item 乙, an opt-in trial)."""
    STEPS = 50

    def setUp(self):
        self.wing = Wing.ellipse(8)
        degrees = np.array([-90., -14., 0., 14., 90.])
        self.low = SectionPolar(degrees, np.array([-1.3, -1.3, 0., 1.3, 1.3]), np.array([.3, .02, .008, .02, .3]),
                                np.zeros(5), "test crosswind section")
        self.high = P.section_polar(P.load_makani_airfoil(NACA), 0.)
        self.times = np.arange(self.STEPS+1)*DT
        self.motion = pitch_manoeuvre(self.times, 25., 30., [(5*DT, -27., 5*DT, 5*DT), (30*DT, 27., 5*DT, 5*DT)])

    def rule(self, basis, seeding="previous", low=None, **options):
        return SectionHandover(self.low if low is None else low, self.high, seeding=seeding, weight_basis=basis, **options)

    def test_in_a_uniform_stream_past_a_body_that_does_not_rotate_the_two_bases_are_one_model_bit_for_bit(self):
        steps, dt = 6, 0.05
        for incidence, low, closure in ((3.8, SectionPolar.linear(), "source"), (12., self.low, "full_incidence"), (30., self.low, "full_incidence")):
            a = np.radians(incidence)
            motion = uniform_translation(np.arange(steps+1)*dt, [-12.*np.cos(a), 0., -12.*np.sin(a)])
            for seeding in ("previous", "zero"):
                origin = run_prescribed(backend(steps, dt), self.wing, None, motion, np.zeros(3), handover=self.rule("origin", seeding, low))
                panel = run_prescribed(backend(steps, dt), self.wing, None, motion, np.zeros(3), handover=self.rule("panel", seeding, low))
                self.assertEqual((origin.status, panel.status), ("complete", "complete"), panel.error)
                for a_row, b_row in zip(origin.records, panel.records):
                    for name in ("force_inertial_N", "moment_origin_inertial_Nm", "aero_power_W", "iterations", "residual", "closure",
                                 "seeding", "handover_weight", "stall_switches", "regime_counts"):
                        self.assertTrue(np.array_equal(a_row[name], b_row[name]), (incidence, seeding, name))
                    self.assertEqual(b_row["closure"], closure)
                    self.assertEqual(b_row["panel_weights"], [b_row["handover_weight"]]*self.wing.panels)    # the very number of the origin
                    self.assertIsNone(a_row["panel_weights"])
                    self.assertIsNone(a_row["panels_by_weight"])
                self.assertTrue(np.array_equal(origin.final_state.free_rows, panel.final_state.free_rows))
                self.assertEqual(origin.final_state.config, panel.final_state.config)

    def test_with_sideslip_and_a_turned_attitude_the_two_bases_are_still_one_model_bit_for_bit(self):
        # Sideslip 18 deg at incidence 3 deg: the weight comes from the sideslip (blend between 15 and 25 deg).
        steps, dt = 5, 0.05
        a, b = np.radians(3.), np.radians(18.)
        relative = 14.*np.array([np.cos(a)*np.cos(b), np.sin(b), np.sin(a)*np.cos(b)])    # relative wind in body axes
        rotation = prescribed.np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])@np.array(
            [[np.cos(0.3), 0., np.sin(0.3)], [0., 1., 0.], [-np.sin(0.3), 0., np.cos(0.3)]])
        motion = uniform_translation(np.arange(steps+1)*dt, -(rotation@relative), rotation=rotation)
        origin = run_prescribed(backend(steps, dt), self.wing, None, motion, np.zeros(3), handover=self.rule("origin", "zero"))
        panel = run_prescribed(backend(steps, dt), self.wing, None, motion, np.zeros(3), handover=self.rule("panel", "zero"))
        self.assertEqual((origin.status, panel.status), ("complete", "complete"), panel.error)
        weight = origin.records[0]["handover_weight"]
        self.assertTrue(0.2 < weight < 0.4)                                              # (18 - 15)/10
        self.assertAlmostEqual(origin.records[0]["beta_deg"], 18., places=9)
        for x, y in zip(origin.records, panel.records):
            for name in ("force_inertial_N", "moment_origin_inertial_Nm", "iterations", "residual", "closure", "seeding", "handover_weight"):
                self.assertTrue(np.array_equal(x[name], y[name]), name)
            self.assertEqual(y["panel_weights"], [weight]*self.wing.panels)
        self.assertTrue(np.array_equal(origin.final_state.free_rows, panel.final_state.free_rows))

    def test_inside_the_crosswind_range_the_panel_basis_is_the_production_run_bit_for_bit(self):
        steps, dt = 6, 0.05
        motion = uniform_translation(np.arange(steps+1)*dt, [-12., 0., -0.8])            # incidence 3.8 deg
        low = SectionPolar.linear()
        plain = run_prescribed(backend(steps, dt), self.wing, low, motion, np.zeros(3))
        handed = run_prescribed(backend(steps, dt), self.wing, None, motion, np.zeros(3),
                                handover=SectionHandover(low, Forbidden(), seeding="zero", weight_basis="panel"))
        self.assertEqual((plain.status, handed.status), ("complete", "complete"), handed.error)
        for a, b in zip(plain.records, handed.records):
            for name in ("force_inertial_N", "moment_origin_inertial_Nm", "aero_power_W", "iterations"):
                self.assertTrue(np.array_equal(a[name], b[name]), name)
            self.assertEqual((b["handover_weight"], b["closure"], b["panel_weight_range"]), (0., "source", [0., 0.]))
            self.assertEqual(b["panels_by_weight"], dict(crosswind_only=8, blend=0, hover_only=0))
        self.assertEqual([r["seeding"] for r in handed.records], ["start"]+["previous"]*steps)

    def test_a_rotating_body_gives_every_panel_the_weight_of_its_own_velocity_and_one_weight_above_zero_puts_the_step_on_the_hover_side(self):
        run = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3), handover=self.rule("panel"))
        origin = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3), handover=self.rule("origin"))
        self.assertEqual((run.status, origin.status), ("complete", "complete"), run.error)
        differing = lifted = 0
        for record, other in zip(run.records, origin.records):
            pose = self.motion.pose(record["time_s"])
            placed = self.wing.transformed(pose.rotation, pose.position_m)
            # Still air, no rings: the oncoming flow at a point of the body is minus the velocity of that point. A panel
            # takes the larger of the weights at its control point (3/4 chord) and at its load point (1/4 chord).
            expected = np.zeros(self.wing.panels)
            for points in (geometry(placed)[0], placed.cp):
                body = -(pose.velocity_m_s+np.cross(pose.angular_velocity_rad_s, points-pose.position_m))@pose.rotation
                expected = np.maximum(expected, [makani_weight(np.degrees(np.arctan2(w, u)), np.degrees(np.arcsin(v/np.sqrt(u*u+v*v+w*w))))
                                                 for u, v, w in body])
            assert_allclose(record["panel_weights"], expected, rtol=0, atol=1e-12)
            weights = np.array(record["panel_weights"])
            self.assertEqual(record["handover_weight"], other["handover_weight"])        # the origin's weight is recorded unchanged
            self.assertEqual(record["closure"], "source" if weights.max() == 0. else "full_incidence")
            self.assertEqual(record["panel_weight_range"], [weights.min(), weights.max()])
            counts = record["panels_by_weight"]
            self.assertEqual(sum(counts.values()), self.wing.panels)
            self.assertEqual(counts["blend"], int(np.sum((weights > 0.) & (weights < 1.))))
            if np.all(pose.angular_velocity_rad_s == 0.):
                self.assertTrue(np.all(weights == record["handover_weight"]))
            differing += bool(np.any(weights != record["handover_weight"]))
            lifted += bool(record["handover_weight"] == 0. and weights.max() > 0.)
        # On this wing the load points lie on the pitch axis through the origin, so they carry the origin's weight; the
        # control points behind them carry a larger one while the nose rises (the pitch-up) and a smaller one while it falls.
        self.assertGreaterEqual(differing, 3)
        self.assertGreater(lifted, 0)                        # a step the origin basis runs with the production model
        self.assertEqual(run.final_state.step, self.STEPS)
        self.assertEqual(len(run.final_state.free_rows), self.STEPS)                    # one wake for the whole run

    def test_a_wind_that_differs_along_the_span_moves_single_panels_while_the_origin_stays_at_weight_zero(self):
        steps, dt = 6, 0.05
        motion = uniform_translation(np.arange(steps+1)*dt, [-12., 0., -0.8])            # incidence 3.8 deg at the origin
        low = SectionPolar.linear(limit_deg=60.)

        def wind(points, time_s):                                                        # an upwash over the outer right wing
            out = np.zeros_like(points)
            out[points[:, 1] > 1.8, 2] = 1.7
            return out
        expected = makani_weight(np.degrees(np.arctan2(2.5, 12.)), 0.)                   # 11.8 deg: inside the blend
        self.assertTrue(0.6 < expected < 0.75)
        origin = run_prescribed(backend(steps, dt), self.wing, None, motion, wind, handover=self.rule("origin", "zero", low))
        panel = run_prescribed(backend(steps, dt), self.wing, None, motion, wind, handover=self.rule("panel", "zero", low))
        self.assertEqual((origin.status, panel.status), ("complete", "complete"), panel.error)
        for a, b in zip(origin.records, panel.records):
            self.assertEqual((a["handover_weight"], a["closure"], b["handover_weight"], b["closure"]), (0., "source", 0., "full_incidence"))
            assert_allclose(b["panel_weights"], [0.]*6+[expected]*2, rtol=0, atol=1e-12)
            self.assertEqual(b["panels_by_weight"], dict(crosswind_only=6, blend=2, hover_only=0))
            self.assertIsNotNone(b["stall_switches"])                                    # the fields of the full-incidence closure
        self.assertEqual([r["seeding"] for r in panel.records], ["start"]+["zero"]*steps)   # the declared seeding of the hover side
        self.assertEqual(panel.final_state.config.solve, TAKEOFF_LANDING_SOLVE)
        # The two runs are different models on these steps (another closure, other data on two panels); no claim about
        # the size of the difference: on this small wing the tip panels sit on different roots under the two closures.
        self.assertFalse(np.array_equal(origin.records[-1]["force_body_N"], panel.records[-1]["force_body_N"]))

    def test_weights_of_zeros_and_ones_only_still_report_a_panel_the_hover_option_moved(self):
        # The upwash puts the two outer panels on the hover data alone (weight 1); the others are at weight 0. The left tip
        # panel (weight 0) leaves the +-15 deg crosswind table: under 'error' the step fails, under 'hover' that panel gets
        # the hover data - and the receipt says so, although no weight lies strictly between 0 and 1.
        steps, dt = 4, 0.05
        motion = uniform_translation(np.arange(steps+1)*dt, [-12., 0., -0.8])
        low = SectionPolar.linear(limit_deg=15.)

        def wind(points, time_s):
            out = np.zeros_like(points)
            out[points[:, 1] > 1.8, 2] = 4.0
            return out
        refused = run_prescribed(backend(steps, dt), self.wing, None, motion, wind, handover=self.rule("panel", "zero", low))
        self.assertEqual((refused.status, refused.failed_module), ("failed", "aerodynamics"))
        self.assertEqual(refused.failed_panel_weights, [0.]*6+[1.]*2)
        self.assertIsNone(refused.failed_panels_outside_crosswind_table)                 # the option is off
        run = run_prescribed(backend(steps, dt), self.wing, None, motion, wind, handover=self.rule("panel", "zero", low, out_of_range="hover"))
        self.assertEqual(run.status, "complete", run.error)
        for record in run.records:
            self.assertEqual(record["panel_weights"], [0.]*6+[1.]*2)
            self.assertEqual(record["panels_by_weight"], dict(crosswind_only=6, blend=0, hover_only=2))
            self.assertEqual((record["handover_weight"], record["closure"]), (0., "full_incidence"))
            self.assertIsNotNone(record["panels_outside_crosswind_table"])
            self.assertGreaterEqual(record["most_panels_outside_crosswind_table"], len(record["panels_outside_crosswind_table"]))
        moved = [r["panels_outside_crosswind_table"] for r in run.records[1:]]
        self.assertTrue(all(panels == [0] for panels in moved), moved)                    # the left tip panel, weight 0

    def test_a_failure_before_the_choice_is_made_is_a_failed_step_without_weights_and_without_a_retry(self):
        original, calls = prescribed.advance_vsm_wake, []

        def broken(state, *args, **kwargs):
            calls.append(state.step+1)
            if state.step+1 == 2:
                raise ValueError("declared test failure before the chooser is called")
            return original(state, *args, **kwargs)
        with mock.patch.object(prescribed, "advance_vsm_wake", broken):
            run = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3), handover=self.rule("panel", "zero"), steps=4)
        self.assertEqual((run.status, run.failed_step, len(run.records)), ("failed", 2, 2))
        self.assertEqual(calls, [1, 2])                                                  # one attempt at the failed step
        self.assertIsNone(run.failed_panel_weights)
        self.assertIsNone(run.zero_seed_failure)
        self.assertEqual(run.final_state.step, 1)

    def test_the_retry_of_a_step_with_unequal_weights_makes_the_same_choice_and_keeps_the_count_of_the_first_attempt(self):
        steps, dt = 4, 0.05
        motion = uniform_translation(np.arange(steps+1)*dt, [-12., 0., -0.8])
        low = SectionPolar.linear(limit_deg=15.)

        def wind(points, time_s):
            out = np.zeros_like(points)
            out[points[:, 1] > 1.8, 2] = 4.0
            return out
        original, choices = prescribed.advance_vsm_wake, []

        def flaky(state, *args, chooser=None, **kwargs):
            def wrapped(*onset):
                polar, solve, seed = chooser(*onset)
                choices.append((state.step+1, seed is not None, type(polar).__name__, polar.weights.tolist(), polar))
                if seed is not None and state.step+1 == 2:
                    polar.most_out_of_range = 5                                          # as if the zero-seeded attempt had seen five panels
                    raise NonlinearSolveError("declared test failure of the zero-seeded solve", iterations=7)
                return polar, solve, seed
            return original(state, *args, chooser=wrapped, **kwargs)
        with mock.patch.object(prescribed, "advance_vsm_wake", flaky):
            run = run_prescribed(backend(steps, dt), self.wing, None, motion, wind,
                                 handover=self.rule("panel", "zero", low, out_of_range="hover"))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual([r["seeding"] for r in run.records], ["start", "zero", "retry_previous", "zero", "zero"])
        first, second = [c for c in choices if c[0] == 2]
        self.assertEqual((first[1], second[1]), (True, False))
        self.assertEqual((first[2], second[2]), ("PanelBlendedPolars", "PanelBlendedPolars"))
        self.assertEqual(first[3], second[3])                                            # the same weights both times
        self.assertIsNot(first[4], second[4])                                            # built anew by the retry ...
        self.assertEqual(run.records[2]["most_panels_outside_crosswind_table"], 5)       # ... which keeps the first attempt's count
        self.assertEqual(run.records[3]["most_panels_outside_crosswind_table"], 1)

    def test_the_rings_of_the_rotors_enter_the_weights_at_their_positions_of_the_step(self):
        steps = 8
        # Just below the 5 deg at which the weight leaves zero, with the body pitched by that angle (pitch_manoeuvre turns
        # the body, uniform_translation with the identity attitude would not tell a rotation from its transpose).
        motion = pitch_manoeuvre(np.arange(steps+1)*DT, 25., 4.9, [])
        self.assertFalse(np.allclose(motion.rotation[0], np.eye(3)))
        rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3)
        low = SectionPolar.linear(limit_deg=60.)
        rule = self.rule("panel", "zero", low)
        run = run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=rule, rotors=rotors,
                             keep_states=range(steps+1))
        bare = run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=rule)
        self.assertEqual((run.status, bare.status), ("complete", "complete"), run.error)
        self.assertEqual(run.records[0]["panel_weights"], bare.records[0]["panel_weights"])          # no ring yet at the start
        self.assertTrue(all(r["panel_weights"] == [0.]*8 for r in bare.records))
        moved = 0
        for k in range(1, steps+1):
            # The same step once more from the kept state, with a chooser that only listens: the ring induction it is
            # given is that of the rings at their new positions, and the recorded weights are those of that flow.
            pose = motion.pose(motion.time_s[k])
            placed = self.wing.transformed(pose.rotation, pose.position_m)
            before = run.states[k-1]
            heard = {}

            def listen(time_s, points, ring_velocity, load_points, ring_load):
                heard.update(points=points.copy(), rings=ring_velocity.copy(), time=time_s, load_points=load_points.copy(),
                             load_rings=ring_load.copy())
                return low, NEWTON, None
            again = advance_vsm_wake(replace(before, config=backend(steps).wake), None, backend(steps).wake, np.zeros(3), next_wing=placed,
                                     structural_velocity=motion.structural_velocity, next_time_s=pose.time_s, chooser=listen)
            self.assertTrue(np.array_equal(heard["points"], geometry(placed)[0]))
            self.assertTrue(np.array_equal(heard["load_points"], placed.cp))
            self.assertTrue(np.array_equal(heard["rings"], ring_induction(again.rings, geometry(placed)[0], again.config)))
            self.assertTrue(np.array_equal(heard["load_rings"], ring_induction(again.rings, placed.cp, again.config)))
            self.assertGreater(np.abs(heard["rings"]).max(), 0.05)
            onset = prescribed.panel_onset_body(pose, heard["points"], np.zeros(3), motion.structural_velocity, heard["rings"], pose.time_s)
            assert_allclose(onset, (-pose.velocity_m_s+heard["rings"])@pose.rotation, rtol=0, atol=1e-13)
            at_load = prescribed.panel_onset_body(pose, heard["load_points"], np.zeros(3), motion.structural_velocity, heard["load_rings"],
                                                  pose.time_s)
            origin_weight = run.records[k]["handover_weight"]
            self.assertEqual(run.records[k]["panel_weights"],
                             np.maximum(panel_weights(onset, fallback=origin_weight), panel_weights(at_load, fallback=origin_weight)).tolist())
            moved += run.records[k]["panel_weights"] != bare.records[k]["panel_weights"]
        self.assertGreater(moved, 0)                         # the rings changed a weight on some step

    def test_the_onset_record_is_what_the_weights_were_taken_from_and_changes_no_step(self):
        steps = 8
        motion = pitch_manoeuvre(np.arange(steps+1)*DT, 25., 4.9, [])
        rotor_discs = discs()
        rotors = PrescribedRotors(rotor_discs, 120., nodes_per_ring=8, surface_core_m=0.3)
        rule = self.rule("panel", "zero", SectionPolar.linear(limit_deg=60.))
        plain = run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=rule, rotors=rotors)
        run = run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=rule, rotors=rotors, panel_onset_steps=(3, 6))
        self.assertEqual((plain.status, run.status), ("complete", "complete"), run.error)
        self.assertEqual([k for k, r in enumerate(run.records) if "panel_onset" in r], [3, 4, 5, 6])
        self.assertFalse(any("panel_onset" in r for r in plain.records))
        for a, b in zip(plain.records, run.records):                                     # the diagnostic changes no step
            self.assertEqual({k: v for k, v in b.items() if k != "panel_onset"}, a)
        self.assertTrue(np.array_equal(plain.final_state.free_rows, run.final_state.free_rows))
        lifted = 0
        for k in (3, 4, 5, 6):
            record, entry = run.records[k], run.records[k]["panel_onset"]
            pose = motion.pose(motion.time_s[k])
            control, load = np.array(entry["control"]), np.array(entry["load"])
            self.assertEqual(control.shape, (self.wing.panels, 3))
            # The recorded weights are those of the recorded flows, to the last bit.
            weight = record["handover_weight"]
            self.assertEqual(record["panel_weights"],
                             np.maximum(panel_weights(control, fallback=weight), panel_weights(load, fallback=weight)).tolist())
            # Without the rings the flow is minus the velocity of the body (still air, no rotation here).
            for name in ("control_without_rings", "load_without_rings"):
                assert_allclose(np.array(entry[name]), np.tile(-pose.velocity_m_s@pose.rotation, (self.wing.panels, 1)), rtol=0, atol=1e-13)
            # The rings' part, disc by disc, adds up to the difference.
            for name, flow in (("control", control), ("load", load)):
                parts = np.array(entry[f"rings_by_disc_{name}"])
                self.assertEqual(parts.shape, (len(rotor_discs), self.wing.panels, 3))
                assert_allclose(parts.sum(axis=0), flow-np.array(entry[f"{name}_without_rings"]), rtol=0, atol=1e-12)
                self.assertTrue(all(np.abs(part).max() > 0. for part in parts))         # every disc has rings by step 3
            lifted += max(record["panel_weights"]) > 0.
        self.assertGreater(lifted, 0)                                                    # the rings did lift a weight above zero
        # Without rotors the record has the flows and no split by disc.
        bare = run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=rule, panel_onset_steps=(2, 2))
        self.assertEqual(sorted(bare.records[2]["panel_onset"]), ["axes", "control", "control_without_rings", "load", "load_without_rings"])
        self.assertEqual(bare.records[2]["panel_onset"]["control"], bare.records[2]["panel_onset"]["control_without_rings"])
        # The option belongs to the per-panel basis.
        with self.assertRaises(ValueError):
            run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=self.rule("origin"), panel_onset_steps=(3, 6))
        with self.assertRaises(ValueError):
            run_prescribed(backend(steps), self.wing, None, motion, np.zeros(3), handover=rule, panel_onset_steps=(6, 3))
        with self.assertRaises(ValueError):
            run_prescribed(backend(steps), self.wing, SectionPolar.linear(), motion, np.zeros(3), panel_onset_steps=(3, 6))

    def test_a_failed_zero_seeded_solve_is_retried_once_and_the_choice_of_the_step_is_the_same_both_times(self):
        original, calls = prescribed.advance_vsm_wake, []

        def flaky(state, *args, chooser=None, **kwargs):
            def wrapped(*onset):
                polar, solve, seed = chooser(*onset)
                calls.append((state.step+1, seed is not None, type(polar).__name__))
                if seed is not None and state.step+1 in (2, 3):
                    raise NonlinearSolveError("declared test failure of the zero-seeded solve", iterations=7)
                return polar, solve, seed
            return original(state, *args, chooser=wrapped, **kwargs)
        with mock.patch.object(prescribed, "advance_vsm_wake", flaky):
            run = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3), handover=self.rule("panel", "zero"), steps=4)
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual([r["seeding"] for r in run.records], ["start", "zero", "retry_previous", "retry_previous", "zero"])
        self.assertEqual([c[:2] for c in calls], [(1, True), (2, True), (2, False), (3, True), (3, False), (4, True)])
        self.assertEqual(run.records[2]["zero_seed_failure"],
                         dict(type="NonlinearSolveError", message="declared test failure of the zero-seeded solve", iterations=7))
        self.assertIsNone(run.records[1]["zero_seed_failure"])

    def test_a_failed_step_keeps_its_panel_weights_both_errors_and_the_earlier_records(self):
        # A crosswind table that ends at 6 deg: the first step on which a panel uses it (weight below 1) is outside it.
        narrow = SectionPolar.linear(limit_deg=6.)
        run = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3), handover=self.rule("panel", "zero", narrow))
        self.assertEqual((run.status, run.error_type, run.failed_module), ("failed", "ValueError", "aerodynamics"))
        weights = np.array(run.failed_panel_weights)
        self.assertEqual(len(weights), self.wing.panels)
        self.assertTrue(np.any(weights < 1.))
        self.assertEqual(len(run.records), run.failed_step)
        self.assertTrue(all(r["panel_weight_range"] == [1., 1.] for r in run.records))  # until then every panel was on the hover data
        self.assertEqual(run.zero_seed_failure["type"], "ValueError")                    # the zero-seeded attempt failed the same way
        self.assertEqual(run.final_state.step, run.failed_step-1)
        self.assertIsNone(run.failed_panels_outside_crosswind_table)                     # the option that would list them is off
        option = run_prescribed(backend(self.STEPS), self.wing, None, self.motion, np.zeros(3),
                                handover=self.rule("panel", "zero", narrow, out_of_range="hover"), steps=run.failed_step)
        self.assertEqual(option.status, "complete", option.error)                        # with it, that panel gets the hover data
        last = option.records[-1]
        self.assertEqual(last["panel_weights"], weights.tolist())
        self.assertGreater(last["most_panels_outside_crosswind_table"], 0)

    def test_the_panel_basis_keeps_the_far_wake_limits_and_a_step_without_wind_at_the_origin_still_fails(self):
        limited = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=DT, max_rows=12, core_radius_m=0.15, solve=NEWTON,
                                                               retained_rows=12, free_convection_rows=6))
        run = run_prescribed(limited, self.wing, None, self.motion, np.zeros(3), handover=self.rule("panel"))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual((run.final_state.step, len(run.final_state.free_rows)), (self.STEPS, 12))
        self.assertEqual((run.final_state.config.retained_rows, run.final_state.config.free_convection_rows), (12, 6))
        self.assertEqual({r["closure"] for r in run.records}, {"source", "full_incidence"})
        times = np.arange(4)*DT
        velocity = np.array([[-25., 0., -1.], [-25., 0., -1.], [0., 0., 0.], [0., 0., 0.]])
        position = np.vstack([np.zeros(3), np.cumsum(DT*velocity[:-1], axis=0)])
        motion = RigidMotion(times, position, np.broadcast_to(np.eye(3), (4, 3, 3)), velocity, np.zeros((4, 3)))
        still = run_prescribed(backend(3), self.wing, None, motion, np.zeros(3),
                               handover=SectionHandover(SectionPolar.linear(), self.high, weight_basis="panel"))
        self.assertEqual((still.status, still.failed_step, len(still.records)), ("failed", 2, 2))
        self.assertIsNone(still.failed_panel_weights)                                    # the weights were not reached


def rotor_table():
    """A small table in the Makani layout: thrust falls with the axial freestream, power negative while thrusting."""
    omegas, speeds = np.linspace(40., 260., 12), np.linspace(0.1, 90., 9)
    w, v = np.meshgrid(omegas, speeds, indexing="ij")
    return RotorTable("synthetic", 2.3, omegas, speeds, 0.18-0.002*v-1e-5*w*v/60., -0.08+0.0015*v)


def discs():
    """Two discs ahead of the wing, off the body origin and off each other's axis, turning in opposite senses."""
    table = rotor_table()
    axis = [-np.cos(np.radians(3.)), 0., -np.sin(np.radians(3.))]
    return (ActuatorDisc("upper", [-1.9, 1.2, 1.2], axis, 1, table), ActuatorDisc("lower", [-1.6, -3.6, -1.6], axis, -1, table))


class RotorRunTests(unittest.TestCase):
    """Rotor discs carried along the prescribed motion (G4 contract section 3 item 3)."""

    def test_a_placed_disc_is_the_body_disc_turned_and_moved(self):
        disc = discs()[0]
        rotation = prescribed.np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])          # body x -> inertial y
        placed = disc.placed(rotation, [10., 20., 30.])
        assert_allclose(placed.centre_m, np.array([10., 20., 30.])+rotation@disc.centre_m, atol=1e-14)
        assert_allclose(placed.axis, rotation@disc.axis, atol=1e-15)
        self.assertEqual((placed.direction, placed.label, placed.table, placed.diameter_m), (disc.direction, disc.label, disc.table, disc.diameter_m))
        assert_allclose(np.linalg.norm(placed.rim(8)-placed.centre_m, axis=1), disc.radius_m, atol=1e-13)
        assert_allclose((placed.rim(8)-placed.centre_m)@placed.axis, 0., atol=1e-13)

    def test_the_rim_polygon_is_fixed_to_the_body_at_every_attitude(self):
        disc = discs()[0]
        for degrees in (0., 20., 29., 31., 60., 90., 135.):                               # pitch about the body y axis
            a = np.radians(degrees)
            rotation = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])
            placed = disc.placed(rotation, [3., -2., 7.])
            for phase in (0., 0.3):
                assert_allclose(placed.rim(7, phase), np.array([3., -2., 7.])+disc.rim(7, phase)@rotation.T, atol=1e-12)
            # Placing twice is placing once with the product of the rotations.
            again = placed.placed(rotation, [0., 0., 0.])
            assert_allclose(again.rim(7), placed.rim(7)@rotation.T, atol=1e-12)
        with self.assertRaises(ValueError):
            disc_basis(disc.axis, reference=disc.axis)

    def test_translation_at_a_turned_attitude_equals_the_turned_fixed_wing_in_the_same_relative_wind(self):
        # The body is pitched 25 deg and yawed 10 deg and keeps that attitude. The fixed reference is built by hand:
        # wing and discs turned by the same rotation, the rim reference turned with them.
        pitch, yaw = np.radians(25.), np.radians(10.)
        about_y = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0., 1., 0.], [-np.sin(pitch), 0., np.cos(pitch)]])
        about_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        rotation = about_z@about_y
        body_wing, polar = Wing.ellipse(8).transformed(np.eye(3), np.array([0.5, 0.3, 0.1])), SectionPolar.linear(limit_deg=60.)
        wind, steps, dt, omega = rotation@np.array([20., 0., 1.]), 5, 0.05, 150.
        body_discs = discs()
        fixed_discs = tuple(ActuatorDisc(d.label, rotation@d.centre_m, rotation@d.axis, d.direction, d.table,
                                         rim_reference=rotation@disc_basis(d.axis)[0]) for d in body_discs)
        fixed, fixed_wing = backend(steps, dt), body_wing.transformed(rotation, np.zeros(3))
        state = fixed.start(fixed_wing, polar, wind)
        wing_force, rotor_rows = [state.solution.force], []
        for index in range(1, steps+1):
            state, rows = advance_with_rotors(state, polar, fixed.wake, wind, fixed_discs, np.full(2, omega), nodes_per_ring=8,
                                              next_time_s=index*dt)
            wing_force.append(state.solution.force)
            rotor_rows.append(rows)
        motion = uniform_translation(np.arange(steps+1)*dt, -wind, rotation=rotation)
        run = run_prescribed(backend(steps, dt), body_wing, polar, motion, np.zeros(3),
                             rotors=PrescribedRotors(body_discs, omega, nodes_per_ring=8))
        self.assertEqual(run.status, "complete", run.error)
        for record, force in zip(run.records, wing_force):
            assert_allclose(record["force_inertial_N"], force, rtol=0, atol=1e-8*np.linalg.norm(force))
            assert_allclose(record["force_body_N"], rotation.T@force, rtol=0, atol=1e-8*np.linalg.norm(force))
        for record, rows in zip(run.records, rotor_rows):
            assert_allclose(record["rotor_thrust_N"], [row["thrust_N"] for row in rows], rtol=1e-8)
            force, moment = rotor_totals(rows, fixed_discs)
            assert_allclose(record["rotor_force_inertial_N"], force, rtol=0, atol=1e-8*np.linalg.norm(force))
            assert_allclose(record["rotor_moment_origin_body_Nm"], rotation.T@moment, rtol=0, atol=1e-8*np.linalg.norm(moment))
            # In body axes the thrust points along the body's own disc axes whatever the attitude.
            expected = sum(row["thrust_N"]*d.axis for row, d in zip(rows, body_discs))
            assert_allclose(record["rotor_force_body_N"], expected, rtol=0, atol=1e-8*np.linalg.norm(expected))

    def test_translation_through_still_air_with_rotors_equals_the_fixed_wing_with_rotors_in_a_wind(self):
        # Wide table: under the slipstream of the discs the sections leave the default range.
        wing, polar = Wing.ellipse(8).transformed(np.eye(3), np.array([0.5, 0.3, 0.1])), SectionPolar.linear(limit_deg=60.)
        wind, steps, dt, omega = np.array([20., 0., 1.]), 6, 0.05, 150.
        fixed, rotors = backend(steps, dt), discs()
        state = fixed.start(wing, polar, wind)
        wing_force, rotor_rows = [state.solution.force], []
        for index in range(1, steps+1):
            state, rows = advance_with_rotors(state, polar, fixed.wake, wind, rotors, np.full(2, omega), nodes_per_ring=8,
                                              next_time_s=index*dt)
            wing_force.append(state.solution.force)
            rotor_rows.append(rows)                           # loads at the time the rings were released: step index-1
        motion = uniform_translation(np.arange(steps+1)*dt, -wind)
        run = run_prescribed(backend(steps, dt), wing, polar, motion, np.zeros(3),
                             rotors=PrescribedRotors(rotors, omega, nodes_per_ring=8))
        self.assertEqual(run.status, "complete", run.error)
        for record, force in zip(run.records, wing_force):
            assert_allclose(record["force_inertial_N"], force, rtol=0, atol=1e-8*np.linalg.norm(force))
        self.assertGreater(np.linalg.norm(np.subtract(run.records[-1]["force_inertial_N"], wing_force[0])), 1.)   # a real comparison
        for record, rows in zip(run.records, rotor_rows):
            assert_allclose(record["rotor_thrust_N"], [row["thrust_N"] for row in rows], rtol=1e-8)
            assert_allclose(record["rotor_axial_freestream_m_s"], [row["axial_freestream_m_s"] for row in rows], rtol=1e-8)
            force, moment = rotor_totals(rows, rotors)        # about the fixed wing's origin = the moving body origin
            assert_allclose(record["rotor_force_inertial_N"], force, rtol=0, atol=1e-8*np.linalg.norm(force))
            assert_allclose(record["rotor_moment_origin_inertial_Nm"], moment, rtol=0, atol=1e-8*np.linalg.norm(moment))
            assert_allclose(record["rotor_moment_origin_body_Nm"], moment, rtol=0, atol=1e-8*np.linalg.norm(moment))
            # Body moving at -wind: the thrust (along -x) does positive work on the body.
            self.assertAlmostEqual(record["rotor_power_on_body_W"], float(np.dot(force, -wind)), delta=1e-8*abs(np.dot(force, wind)))
            self.assertGreater(record["rotor_power_on_body_W"], 0.)
            self.assertLess(record["rotor_shaft_power_W"], 0.)                             # the motors drive the rotors
        # The slipstream of the first steps changes the wing load: the rings are in the wing's solve.
        plain = run_prescribed(backend(steps, dt), wing, polar, motion, np.zeros(3))
        self.assertEqual(run.records[0]["force_inertial_N"], plain.records[0]["force_inertial_N"])    # no ring yet at step 0
        self.assertGreater(np.linalg.norm(np.subtract(run.records[-1]["force_inertial_N"], plain.records[-1]["force_inertial_N"])), 1e-3)
        self.assertEqual(run.final_state.rings.count, 2*8*(steps+1))
        self.assertEqual(run.records[-1]["ring_nodes"], 2*8*(steps+1))

    def test_the_hub_velocity_is_the_velocity_of_the_body_point_at_the_hub(self):
        # A section without lift carries no circulation, so the apparent wind at a hub is the wind minus the
        # hub velocity; on the circle about the X axis that is  wind - Omega e_X x hub.
        degrees = np.array([-90., 0., 90.])
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar(degrees, np.zeros(3), np.full(3, .01), np.zeros(3), "no lift")
        steps = 2
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, 0.)
        wind = np.array([WIND, 0., 0.])
        rotors = discs()
        run = run_prescribed(backend(steps), wing, polar, motion, wind, rotors=PrescribedRotors(rotors, 0.))   # no thrust, no ring strength
        self.assertEqual(run.status, "complete", run.error)
        spin = np.array([SPEED/RADIUS, 0., 0.])
        record, pose = run.records[0], motion.pose(0.)
        self.assertLess(np.abs(run.final_state.solution.circulation).max(), 1e-9)
        outer = []
        for disc, axial in zip(rotors, record["rotor_axial_freestream_m_s"]):
            placed = disc.placed(pose.rotation, pose.position_m)
            expected = -np.dot(wind-np.cross(spin, placed.centre_m), placed.axis)
            self.assertAlmostEqual(axial, expected, delta=1e-9*abs(expected))
            outer.append(np.linalg.norm(placed.centre_m[1:]))
        axial = record["rotor_axial_freestream_m_s"]
        self.assertGreater(abs(axial[0]-axial[1]), 0.5)                                    # the two hubs are at different radii
        self.assertEqual(np.argmax(np.abs(axial)), np.argmax(outer))                       # the hub further out is faster

    def test_rotors_at_rest_change_no_bit_of_the_lifting_surface_solution(self):
        wing, polar = Wing.ellipse(8), SectionPolar.linear(limit_deg=60.)
        steps = 4
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind = np.array([WIND, 0., 0.])
        plain = run_prescribed(backend(steps), wing, polar, motion, wind)
        resting = run_prescribed(backend(steps), wing, polar, motion, wind, rotors=PrescribedRotors(discs(), 0.))
        self.assertEqual((plain.status, resting.status), ("complete", "complete"))
        for a, b in zip(plain.records, resting.records):
            for name in ("force_inertial_N", "moment_origin_inertial_Nm", "aero_power_W", "iterations"):
                self.assertTrue(np.array_equal(a[name], b[name]), name)
            self.assertEqual((b["rotor_thrust_N"], b["rotor_shaft_power_W"]), ([0., 0.], 0.))
            self.assertNotIn("rotor_thrust_N", a)
        self.assertIsNone(plain.final_state.rings)

    def test_loads_are_about_the_body_origin_and_the_work_uses_the_hub_velocities(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps, omega = 2, np.array([150., 120.])
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        rotors = discs()
        run = run_prescribed(backend(steps), wing, polar, motion, np.array([WIND, 0., 0.]),
                             rotors=PrescribedRotors(rotors, lambda time: omega, nodes_per_ring=8))
        self.assertEqual(run.status, "complete", run.error)
        record, pose = run.records[1], motion.pose(motion.time_s[1])
        self.assertEqual(record["rotor_omega_rad_s"], omega.tolist())
        thrust, power = np.array(record["rotor_thrust_N"]), np.array(record["rotor_power_W"])
        self.assertTrue(np.all(thrust > 0.) and np.all(power < 0.) and thrust[0] > thrust[1])
        force = sum(t*disc.axis for t, disc in zip(thrust, rotors))                         # body axes
        torque = [disc.direction*p/w*disc.axis for p, w, disc in zip(power, omega, rotors)]
        moment = sum(np.cross(disc.centre_m, t*disc.axis)+q for t, q, disc in zip(thrust, torque, rotors))
        assert_allclose(record["rotor_force_body_N"], force, rtol=1e-12, atol=1e-9)
        assert_allclose(record["rotor_moment_origin_body_Nm"], moment, rtol=1e-12, atol=1e-8)
        assert_allclose(record["rotor_force_inertial_N"], pose.rotation@force, rtol=1e-12, atol=1e-9)
        hub_velocity = [pose.rotation.T@pose.velocity_m_s+np.cross(pose.rotation.T@pose.angular_velocity_rad_s, disc.centre_m) for disc in rotors]
        work = sum(t*np.dot(disc.axis, v)+np.dot(q, pose.rotation.T@pose.angular_velocity_rad_s)
                   for t, q, v, disc in zip(thrust, torque, hub_velocity, rotors))
        self.assertAlmostEqual(record["rotor_power_on_body_W"], work, delta=1e-10*abs(work))
        self.assertAlmostEqual(record["rotor_shaft_power_W"], power.sum(), delta=1e-9*abs(power.sum()))
        self.assertEqual(record["rotor_table_saturated"], [False, False])
        self.assertEqual((record["ring_nodes"], record["ring_surface_core_m"]), (2*8*2, 0.15))

    def test_a_rotor_step_without_a_solution_fails_the_step_and_names_the_module(self):
        wing, polar = Wing.ellipse(8), SectionPolar.linear(limit_deg=60.)
        steps = 4
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        real, calls = prescribed.shed_rotor_rings, []

        def failing(state, *args, **kwargs):
            calls.append(state.step)
            if len(calls) == 3:
                raise ValueError("momentum theory has no solution")
            return real(state, *args, **kwargs)
        with mock.patch.object(prescribed, "shed_rotor_rings", failing):
            run = run_prescribed(backend(steps), wing, polar, motion, np.array([WIND, 0., 0.]),
                                 rotors=PrescribedRotors(discs(), 150., nodes_per_ring=8))
        self.assertEqual((run.status, run.failed_step, run.failed_module, run.error_type), ("failed", 2, "rotors", "ValueError"))
        self.assertEqual(len(run.records), 2)
        self.assertEqual((run.final_state.step, run.final_state.rings.count), (1, 2*8*2))    # the last good state, rings and all
        self.assertEqual(run.failed_step_aerodynamics["step"], 2)                            # what the surfaces gave at the failed step
        aero = run_prescribed(backend(steps), wing, SectionPolar.linear(limit_deg=1.), motion, np.array([WIND, 0., 0.]),
                              rotors=PrescribedRotors(discs(), 150.))
        self.assertEqual((aero.status, aero.failed_module), ("failed", "aerodynamics"))

    def test_rotors_run_through_the_hand_over_and_their_rings_stay_in_the_wake(self):
        wing = Wing.ellipse(8)
        degrees = np.array([-90., -14., 0., 14., 90.])
        low = SectionPolar(degrees, np.array([-1.3, -1.3, 0., 1.3, 1.3]), np.array([.3, .02, .008, .02, .3]), np.zeros(5), "test crosswind section")
        high = P.section_polar(P.load_makani_airfoil(NACA), 0.)
        steps = 30
        motion = pitch_manoeuvre(np.arange(steps+1)*DT, 25., 30., [(5*DT, -27., 5*DT, 5*DT)])     # 30 deg -> 3 deg
        run = run_prescribed(backend(steps), wing, None, motion, np.zeros(3), handover=SectionHandover(low, high, seeding="zero"),
                             rotors=PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual({r["closure"] for r in run.records}, {"source", "full_incidence"})      # both sides of the hand-over
        self.assertEqual(run.final_state.rings.count, 2*8*(steps+1))
        self.assertEqual(run.final_state.rings.surface_core_m, 0.3)
        self.assertEqual(run.records[-1]["ring_surface_core_m"], 0.3)
        axial = np.array([r["rotor_axial_freestream_m_s"] for r in run.records])
        self.assertTrue(np.all(axial[-1] > axial[0]))                                    # the discs turn into the wind with the body

    def test_a_speed_schedule_is_followed_step_by_step(self):
        wing, polar = Wing.ellipse(8), SectionPolar.linear(limit_deg=60.)
        steps = 3
        motion = uniform_translation(np.arange(steps+1)*DT, [-20., 0., -1.])

        def schedule(time_s):
            return np.array([100.+2000.*time_s, 140.])
        run = run_prescribed(backend(steps), wing, polar, motion, np.zeros(3), rotors=PrescribedRotors(discs(), schedule, nodes_per_ring=8))
        self.assertEqual(run.status, "complete", run.error)
        speeds = np.array([r["rotor_omega_rad_s"] for r in run.records])
        assert_allclose(speeds, [[100.+2000.*k*DT, 140.] for k in range(steps+1)], rtol=1e-14)
        thrust = np.array([r["rotor_thrust_N"] for r in run.records])
        self.assertTrue(np.all(np.diff(thrust[:, 0]) > 50.))                             # the first rotor speeds up
        self.assertLess(np.ptp(thrust[:, 1]), 0.05*thrust[0, 1])                         # the second keeps its speed
        table = discs()[0].table                                                         # first step: no ring yet, wing induction only
        axial = run.records[0]["rotor_axial_freestream_m_s"][0]
        self.assertAlmostEqual(thrust[0, 0], table.thrust_N(100., axial, 1.225), delta=1e-9*thrust[0, 0])

    def test_a_turbine_state_beyond_the_momentum_limit_fails_the_step_and_keeps_the_solved_surfaces(self):
        # A table whose thrust turns strongly negative with the axial freestream: at 20 m/s and 250 rad/s the
        # thrust is about -19 kN and V^2 + 2T/(rho A) is negative, so momentum theory has no tube strength.
        omegas, speeds = np.linspace(40., 260., 12), np.linspace(0.1, 90., 9)
        w, v = np.meshgrid(omegas, speeds, indexing="ij")
        braking = RotorTable("braking", 2.3, omegas, speeds, 0.05-0.02*v, -0.08+0.0015*v)
        disc = ActuatorDisc("one", [-1.9, 1.2, 1.2], [-1., 0., 0.], 1, braking)
        wing, polar = Wing.ellipse(8), SectionPolar.linear(limit_deg=60.)
        motion = uniform_translation(np.arange(4)*DT, [-20., 0., -1.])
        run = run_prescribed(backend(3), wing, polar, motion, np.zeros(3), rotors=PrescribedRotors((disc,), 250.))
        self.assertEqual((run.status, run.failed_step, run.failed_module, run.error_type), ("failed", 0, "rotors", "ValueError"))
        self.assertIn("momentum theory has no solution", run.error)
        self.assertEqual((run.records, run.final_state), ([], None))
        plain = run_prescribed(backend(3), wing, polar, motion, np.zeros(3))
        self.assertEqual(run.failed_step_aerodynamics["force_inertial_N"], plain.records[0]["force_inertial_N"])
        self.assertNotIn("rotor_thrust_N", run.failed_step_aerodynamics)
        # With the declared extension the same rotor state has a ring: step 0 is recorded. (What that ring then does
        # to the wing is another matter - here the wing's solve fails at the next step, and it is booked as such.)
        extended = run_prescribed(backend(3), wing, polar, motion, np.zeros(3),
                                  rotors=PrescribedRotors((disc,), 250., turbine_extension="glauert"))
        self.assertLess(extended.records[0]["rotor_thrust_N"][0], -10000.)
        self.assertIn((extended.status, extended.failed_module), (("complete", None), ("failed", "aerodynamics")))

    def test_the_far_wake_limit_cuts_the_rows_of_the_surfaces_and_leaves_the_rings(self):
        wing, polar = Wing.ellipse(8), SectionPolar.linear(limit_deg=60.)
        steps = 6
        limited = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=DT, max_rows=steps, core_radius_m=0.15, solve=NEWTON, retained_rows=3))
        motion = uniform_translation(np.arange(steps+1)*DT, [-20., 0., -1.])
        run = run_prescribed(limited, wing, polar, motion, np.zeros(3), rotors=PrescribedRotors(discs(), 150., nodes_per_ring=8))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual(len(run.final_state.free_rows), 3)
        self.assertEqual(run.final_state.rings.count, 2*8*(steps+1))                     # the rings are not truncated

    def test_rotor_set_up_is_checked(self):
        with self.assertRaises(ValueError):
            PrescribedRotors((), 100.)
        for bad in (dict(surface_core_m=0.), dict(surface_core_m=-0.3), dict(surface_core_m=float("nan")),
                    dict(turbine_extension="buhl"), dict(ring_phase_rad=float("inf"))):
            with self.assertRaises(ValueError):
                PrescribedRotors(discs(), 100., **bad)
        wing, polar = Wing.ellipse(8), SectionPolar.linear()
        motion = uniform_translation(np.arange(3)*DT, [-12., 0., 0.])
        # A schedule that goes wrong later in the run is a set-up mistake: it raises before the first step.
        late = PrescribedRotors(discs(), lambda time_s: np.full(2, 100.-4000.*time_s))    # negative at the last sample
        with self.assertRaises(ValueError):
            run_prescribed(backend(2), wing, polar, motion, np.zeros(3), rotors=late)
        with self.assertRaises(ValueError):                                               # no wake network for the rings
            run_prescribed(AeroBackend("vsm"), wing, polar, motion, np.zeros(3), rotors=PrescribedRotors(discs(), 100.))
        with self.assertRaises(ValueError):
            PrescribedRotors(discs(), [100., 100., 100.])
        with self.assertRaises(ValueError):
            PrescribedRotors(discs(), -1.)
        with self.assertRaises(ValueError):
            PrescribedRotors(discs(), 100., nodes_per_ring=2)
        wing, polar = Wing.ellipse(8), SectionPolar.linear()
        motion = uniform_translation(np.arange(3)*DT, [-12., 0., 0.])
        with self.assertRaises(ValueError):
            run_prescribed(backend(2), wing, polar, motion, np.zeros(3), rotors=discs())


def short_tether(nodes=6, **options):
    """A 60 m line from the origin to a body near (-40, 0, 51), with the M600 bridle geometry."""
    line = LumpedTether(TetherProperties(length_m=60.0), TetherSettings(num_nodes=nodes))
    return PrescribedTether(line, np.zeros(3), Bridle([0.1494, -0.5091, -0.13035], 4.786), **options)


class TetherRunTests(unittest.TestCase):
    """The tether and the bridle in the prescribed-motion driver (G4 contract section 3 item 4)."""

    def setUp(self):
        self.wing, self.polar = Wing.ellipse(8), SectionPolar.linear(limit_deg=60.)
        self.wind = np.array([12., 0., 0.8])
        self.steps, self.dt = 5, 0.02
        # The body drifts slowly away from the ground point, so the line stays taut and its tension grows.
        self.motion = uniform_translation(np.arange(self.steps+1)*self.dt, [-1.5, 0., 1.0], origin=[-40., 0., 51.2])

    def test_the_line_is_advanced_between_the_steps_with_its_own_time_step_and_the_bridle_follows_it(self):
        tether = short_tether(substeps=10)
        run = run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, tether=tether)
        self.assertEqual(run.status, "complete", run.error)
        # The same line marched by hand: ten RK4 steps of 2 ms per step, the far end at the bridle point of the
        # interpolated pose and of the last node of every stage.
        positions, velocities = tether.start(self.motion)
        ends, small = tether.ends(self.motion), self.dt/10
        for index in range(1, self.steps+1):
            time = self.motion.time_s[index-1]
            for _ in range(10):
                positions, velocities = tether.line.step(time, positions, velocities, small, ends, lambda points, t: np.broadcast_to(self.wind, points.shape))
                time += small
        # (The driver takes its sub-times from the samples; the hand march adds them up. Same line code, so this is a
        # check of the driver's bookkeeping - which steps, which times, which ends - not of the line model.)
        assert_allclose(run.tether_state[0], positions, rtol=0, atol=1e-9)
        assert_allclose(run.tether_state[1], velocities, rtol=0, atol=1e-7)
        self.assertEqual((run.tether_settings["substeps"], run.tether_settings["line_step_s"]), (10, self.dt/10))
        self.assertGreater(run.tether_settings["rk4_step_bound_s"], run.tether_settings["line_step_s"])
        positions, velocities = run.tether_state           # the loads below are checked on the driver's own state
        record, pose = run.records[-1], self.motion.pose(self.motion.time_s[-1])
        point, point_velocity = tether.bridle.point(pose, positions[-1], velocities[-1])
        out = tether.line.forces(positions, velocities, np.zeros(3), np.zeros(3), point, point_velocity, np.broadcast_to(self.wind, positions.shape))
        assert_allclose(record["tether_force_inertial_N"], out["end"], rtol=1e-13)
        assert_allclose(record["tether_moment_origin_inertial_Nm"], np.cross(point-pose.position_m, out["end"]), rtol=1e-12, atol=1e-9)
        assert_allclose(record["tether_force_body_N"], pose.rotation.T@out["end"], rtol=1e-13, atol=1e-9)
        assert_allclose(record["bridle_point_body_m"], pose.rotation.T@(point-pose.position_m), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(np.subtract(record["bridle_point_body_m"], tether.bridle.pivot_body_m)), 4.786, places=10)
        self.assertAlmostEqual(record["tether_tension_kite_N"], float(out["tension"][-1]), delta=1e-9*abs(out["tension"][-1]))
        expected_power = float(np.dot(out["end"], pose.velocity_m_s))                     # no rotation of the body here
        self.assertAlmostEqual(record["tether_power_on_body_W"], expected_power, delta=1e-10*abs(expected_power))
        # The lifting surfaces do not see the line: their loads are those of the run without it.
        plain = run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind)
        for a, b in zip(plain.records, run.records):
            self.assertEqual(a["force_inertial_N"], b["force_inertial_N"])
        self.assertNotIn("tether_force_inertial_N", plain.records[0])
        self.assertIsNone(plain.tether_state)

    def test_a_taut_line_pulls_the_body_towards_the_ground_point(self):
        tether = settle_tether(short_tether(), self.motion, self.wind, duration_s=1.0)
        run = run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, tether=tether)
        self.assertEqual(run.status, "complete", run.error)
        first, last = run.records[0], run.records[-1]
        pose = self.motion.pose(0.)
        point = pose.position_m+pose.rotation@np.array(first["bridle_point_body_m"])
        towards_ground = -point/np.linalg.norm(point)
        force = np.array(first["tether_force_inertial_N"])
        # Stretched just beyond its natural length: with 18 MN of stiffness a centimetre is already kilonewtons.
        self.assertGreater(first["tether_tension_kite_N"], 2e3)
        self.assertEqual(first["tether_slack_segments"], 0)
        self.assertGreater(float(force@towards_ground), 0.98*np.linalg.norm(force))       # along the line, a little sag and drag
        # The body starts moving away at 1.8 m/s: the last segment lengthens and its damper pulls as well.
        self.assertGreater(first["tether_damping_kite_N"], 100.)
        pull = first["tether_tension_kite_N"]+first["tether_damping_kite_N"]
        self.assertAlmostEqual(np.linalg.norm(force), pull, delta=0.01*pull)
        self.assertGreater(last["tether_tension_kite_N"], first["tether_tension_kite_N"])   # the body moves away
        self.assertLess(last["tether_power_on_body_W"], 0.)                               # the line does negative work on it

    def test_a_settled_line_on_a_body_at_rest_is_in_balance(self):
        held = uniform_translation(np.arange(3)*self.dt, np.zeros(3), origin=[-40., 0., 51.2])
        # The transverse swing of the line dies out slowly (only its drag damps it): twenty seconds for this short
        # line of three nodes, marched with 5 ms steps (its stability limit is about 7 ms).
        tether = settle_tether(short_tether(nodes=3, substeps=4), held, self.wind, duration_s=20.0)
        self.assertLess(np.abs(tether.velocities).max(), 0.2)
        run = run_prescribed(backend(2, self.dt), self.wing, self.polar, held, self.wind, tether=tether)
        self.assertEqual(run.status, "complete", run.error)
        record = run.records[-1]
        on_body, on_ground = np.array(record["tether_force_inertial_N"]), np.array(record["tether_ground_force_inertial_N"])
        weight = np.array([0., 0., -60.*0.917*9.81])
        drag = np.array(record["tether_aero_force_inertial_N"])
        # The line pulls both ends towards itself; what the two ends hold together is its weight and its drag.
        assert_allclose(on_body+on_ground, weight+drag, rtol=0, atol=0.02*record["tether_tension_kite_N"])
        self.assertGreater(drag[0], 1.)                                                   # the wind blows along +x
        self.assertLess(abs(record["tether_damping_kite_N"]), 0.01*record["tether_tension_kite_N"])
        self.assertAlmostEqual(record["tether_power_on_body_W"], 0., places=9)            # a body at rest takes no work
        self.assertGreater(record["tether_lowest_node_height_m"], 0.)

    def test_the_line_alone_follows_the_motion_as_the_coupled_run_does(self):
        # The warm-up marches the line through every sample with the driver's sub-times; the state it returns is the one
        # the coupled run reaches with the same motion (the aerodynamics do not act on the line).
        tether = short_tether(substeps=10)
        warmed, report = prescribed.run_tether_alone(tether, self.motion, self.wind, record_every=2)
        run = run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, tether=tether)
        assert_allclose(warmed.positions, run.tether_state[0], rtol=0, atol=1e-12)
        assert_allclose(warmed.velocities, run.tether_state[1], rtol=0, atol=1e-12)
        self.assertEqual([r["sample"] for r in report], [2, 4, 5])                 # every second sample and the last one
        self.assertAlmostEqual(report[-1]["tension_kite_N"], run.records[-1]["tether_tension_kite_N"], places=9)
        self.assertEqual(warmed.settings(self.dt)["started_from"], "given state")
        with self.assertRaises(ValueError):
            prescribed.run_tether_alone(short_tether(substeps=1), self.motion, self.wind)
        with self.assertRaises(ValueError):
            prescribed.run_tether_alone(tether, self.motion, self.wind, record_every=0)

    def test_the_attitude_scheme_of_the_sample_check_is_passed_through(self):
        run = run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, attitude_scheme="fourth_order_central")
        self.assertEqual(run.motion_consistency["attitude_scheme"], "fourth_order_central")
        with self.assertRaises(ValueError):
            run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, attitude_scheme="sixth")

    def test_a_line_step_beyond_the_stability_bound_is_refused_before_the_first_step(self):
        # One line step per 0.02 s is far beyond the stability limit of this line (a few milliseconds).
        with self.assertRaises(ValueError) as caught:
            run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, tether=short_tether(substeps=1))
        self.assertIn("stability bound", str(caught.exception))
        # The bound is the Gershgorin bound of the chain: here the end nodes, held by a half and a whole segment.
        line = short_tether().line
        stiffness, mass = 18e6/10., 0.917*10.
        self.assertAlmostEqual(line.rk4_step_bound_s(), 2*np.sqrt(2)/np.sqrt(2*(2*stiffness+stiffness)/mass), places=12)
        # It is a safe bound: a little below it the undamped chain keeps its energy bounded over many periods.
        from openfawe.tether.lumped_line import fixed_ends
        free = LumpedTether(TetherProperties(length_m=60.0, bending_stiffness_Nm2=0., section_drag_coeff=0.),
                            TetherSettings(num_nodes=6, longitudinal_damping_ratio=0., gravity_m_s2=0.))
        start, end = np.zeros(3), np.array([60.3, 0., 0.])
        positions, velocities = free.straight_state(start, end)
        positions = positions+np.array([0.02, 0., 0.])*np.resize([1., -1.], 6)[:, None]     # the sawtooth: the stiffest mode
        step, ends = 0.98*free.rk4_step_bound_s(), fixed_ends(start, end)
        first = free.energy(positions, velocities, start, end)["total"]
        for index in range(4000):
            positions, velocities = free.step(index*step, positions, velocities, step, ends)
        self.assertLess(free.energy(positions, velocities, start, end)["total"], 1.01*first)
        with self.assertRaises(ValueError):                                               # the air densities must agree
            thin = LumpedTether(TetherProperties(length_m=60.0), TetherSettings(num_nodes=6, air_density_kg_m3=1.0))
            run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind,
                           tether=PrescribedTether(thin, np.zeros(3), Bridle([0.1494, -0.5091, -0.13035], 4.786)))

    def test_a_line_that_leaves_the_finite_range_fails_the_step_and_names_the_tether(self):
        tether = short_tether()
        real, calls = tether.line.step, []

        def diverging(*args, **kwargs):
            calls.append(args[0])
            if len(calls) > 25:                                                           # in the third step of the run
                raise FloatingPointError("tether state left the finite range (time step too large?)")
            return real(*args, **kwargs)
        with mock.patch.object(tether.line, "step", diverging):
            run = run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, tether=tether)
        self.assertEqual((run.status, run.failed_step, run.failed_module, run.error_type), ("failed", 3, "tether", "FloatingPointError"))
        self.assertEqual(len(run.records), 3)
        self.assertEqual(run.failed_step_aerodynamics["step"], 3)
        self.assertEqual(run.final_state.step, 2)
        assert_allclose(calls[:11], self.motion.time_s[0]+np.arange(11)*self.dt/10, rtol=0, atol=1e-15)   # the sub-times of step 1
        # The line state kept is that of the last recorded step, not the half-advanced one.
        again = run_prescribed(backend(2, self.dt), self.wing, self.polar, self.motion, self.wind, tether=short_tether(), steps=2)
        np.testing.assert_array_equal(run.tether_state[0], again.tether_state[0])

    def test_the_last_stage_of_a_step_lands_on_the_sample_also_far_from_time_zero_and_with_many_substeps(self):
        # Sub-times are taken from the samples, so the last stage of the last step asks for the pose of the last
        # sample itself; summing the small steps would drift past it and be refused by the motion.
        times = 1234.5+np.arange(4)*self.dt
        motion = uniform_translation(times, [-1.5, 0., 1.0], origin=[-40., 0., 51.2])
        wake = WakeConfig(dt_s=self.dt, max_rows=3, core_radius_m=0.15, solve=NEWTON)
        run = run_prescribed(AeroBackend("vsm_free_wake", wake=wake), self.wing, self.polar, motion, self.wind, tether=short_tether(substeps=37))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual(len(run.records), 4)

    def test_rotors_tether_and_the_residual_account_run_together(self):
        steps = 4
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind = np.array([WIND, 0., 0.])
        ground = np.array([-200., 0., 0.])                       # upwind of the circle's centre, on its axis
        line = LumpedTether(TetherProperties(length_m=235.0), TetherSettings(num_nodes=8))
        tether = PrescribedTether(line, ground, Bridle([0.1494, -0.5091, -0.13035], 4.786))
        rotors = PrescribedRotors(discs(), 150., nodes_per_ring=8)
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        run = run_prescribed(backend(steps), wing, polar, motion, wind, rotors=rotors, tether=tether)
        self.assertEqual(run.status, "complete", run.error)
        body = RigidBody(1700., [0.08, 0., -0.13], np.diag([32000., 9700., 38700.]))
        rows, summary = account(run.records, motion, body, discs=rotors.discs, rotor_inertia_kg_m2=1.56,
                                not_included=("pylons", "fuselage aerodynamics"))
        self.assertEqual(summary["loads_included"], ["aero", "rotor", "tether", "rotor gyroscopic and reaction moments"])
        self.assertEqual(summary["loads_absent"], [])
        for row, record in zip(rows, run.records):
            pose = motion.pose(record["time_s"])
            offset = pose.rotation@body.centre_of_mass_body_m
            total = sum(np.array(record[key]) for key in ("force_inertial_N", "rotor_force_inertial_N", "tether_force_inertial_N"))
            rate = SPEED/RADIUS
            # The centre of mass sits at the body origin's radius plus the radial part of its offset (the span axis is radial).
            radial = pose.position_m/RADIUS
            centre = pose.position_m+offset
            inertia_force = -1700.*rate**2*np.array([0., centre[1], centre[2]])
            expected = inertia_force-total-1700.*np.array([0., 0., -9.81])
            assert_allclose(row["required_force_inertial_N"], expected, rtol=1e-10, atol=1e-6)
            # The circle is steady: the kinetic energy does not change, so the powers must add up to nothing.
            self.assertAlmostEqual(row["kinetic_energy_rate_W"], 0., delta=1e-6*abs(row["power_W"]["required"]))
            self.assertAlmostEqual(row["kinetic_energy_rate_from_samples_W"], 0., delta=1e-6*abs(row["power_W"]["required"]))
            self.assertGreater(np.linalg.norm(radial), 0.99)
        self.assertGreater(summary["largest_required_force_N"], 1000.)

    def test_tether_set_up_is_checked(self):
        line = LumpedTether(TetherProperties(length_m=60.0), TetherSettings(num_nodes=6))
        bridle = Bridle([0., 0., 0.], 4.786)
        with self.assertRaises(ValueError):
            PrescribedTether(line, [0., 0., float("nan")], bridle)
        with self.assertRaises(ValueError):
            PrescribedTether(line, np.zeros(3), bridle, substeps=0)
        with self.assertRaises(ValueError):
            PrescribedTether(line, np.zeros(3), bridle, positions=np.zeros((6, 3)))            # velocities missing
        with self.assertRaises(ValueError):
            PrescribedTether(line, np.zeros(3), bridle, positions=np.zeros((5, 3)), velocities=np.zeros((5, 3)))
        with self.assertRaises(ValueError):
            PrescribedTether("line", np.zeros(3), bridle)
        with self.assertRaises(ValueError):
            run_prescribed(backend(self.steps, self.dt), self.wing, self.polar, self.motion, self.wind, tether=line)
        with self.assertRaises(ValueError):                       # the ground point on the span line through the pivot
            PrescribedTether(line, [-40.+0.1494, 30., 51.2-0.13035], Bridle([0.1494, -0.5091, -0.13035], 4.786)).start(self.motion)


class Forbidden:
    """Section data that must not be evaluated."""

    def __getattr__(self, name):
        raise AssertionError(f"hover section data evaluated inside the crosswind range: {name}")


if __name__ == "__main__":
    unittest.main()
