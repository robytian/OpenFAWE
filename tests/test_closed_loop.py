"""Closed loop of G5 step 2 (closed-loop contract): the parts that need no controller library."""
import math
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero.makani_flap_increments import ELEVATOR, RUDDER, FlapIncrements, FlapSettings
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing
from openfawe.flight import makani_interface as mi
from openfawe.flight.closed_loop import Externals
from openfawe.flight.free_flight import Release, run_free
from openfawe.flight.kinematics import circular_motion, relative_wind_body, rotation_from_vector
from openfawe.flight.residual import RigidBody
from openfawe.flight.rigid_body import RigidState, state_from_pose

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT/"cases/polars/makani/m600_aswing_baseline.json"
BODY = RigidBody(1700., [0.08, 0.01, -0.13], np.array([[32000., -20., 35.], [-20., 9700., -15.], [35., -15., 38700.]]))
G = np.array([0., 0., -9.81])
DT = 0.02


def synthetic(offset=0.0):
    """Tables that are bilinear in the two angles, so that the interpolation must return them exactly."""
    alphas, betas = np.array([-10., 0., 4., 12.]), np.array([-8., 0., 8.])
    a, b = np.meshgrid(alphas, betas, indexing="ij")
    derivatives = np.array([[(flap+1)*1e-3*(1.0+0.01*a)+(coefficient-2.5)*1e-4*b for coefficient in range(6)] for flap in range(8)])
    return FlapIncrements(alphas, betas, derivatives, np.full(a.shape, offset)+0.1*offset*a)


class FlapIncrementTests(unittest.TestCase):
    def test_the_tables_are_read_bilinearly_and_held_outside_the_grid(self):
        flaps = synthetic()
        for alpha, beta in ((-10., -8.), (4., 8.), (2.5, -3.), (11.9, 7.9)):
            expected = np.array([[(flap+1)*1e-3*(1.0+0.01*alpha)+(coefficient-2.5)*1e-4*beta for coefficient in range(6)] for flap in range(8)])
            assert_allclose(flaps.derivatives_at(math.radians(alpha), math.radians(beta)), expected, rtol=1e-13, atol=1e-18)
        assert_allclose(flaps.derivatives_at(math.radians(50.), math.radians(-50.)), flaps.derivatives[:, :, -1, 0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            FlapIncrements([0., 1.], [0., 1.], np.zeros((8, 6, 2, 3)), np.zeros((2, 2)))

    def test_the_deflections_are_treated_as_in_the_source(self):
        flaps, deg = synthetic(offset=0.5), math.radians
        offset = 0.5+0.1*0.5*2.0                              # de1 at alpha = 2 deg
        zero = np.zeros(8)

        def effective(index, value):
            given = zero.copy()
            given[index] = value
            return flaps.effective_deflections_deg(given, deg(2.), 0.)
        assert_allclose(effective(RUDDER, deg(20.))[RUDDER], 10.+0.4692*10., rtol=1e-13)
        assert_allclose(effective(RUDDER, deg(-25.))[RUDDER], -10.-0.5374*15., rtol=1e-13)
        assert_allclose(effective(RUDDER, deg(5.))[RUDDER], 5., rtol=1e-13)
        assert_allclose(effective(RUDDER, deg(60.))[RUDDER], math.degrees(0.5), rtol=1e-13)      # scaled to 33.5 deg, then saturated
        assert_allclose(effective(0, deg(8.))[0], 5., rtol=1e-13)
        assert_allclose(effective(3, -0.7)[3], math.degrees(-0.5), rtol=1e-13)
        assert_allclose(effective(ELEVATOR, deg(-3.))[ELEVATOR], -3.-offset, rtol=1e-13)
        assert_allclose(effective(ELEVATOR, 0.6)[ELEVATOR], math.degrees(0.5)-offset, rtol=1e-13)
        self.assertEqual(effective(0, deg(2.))[1], 0.)        # the others are untouched
        with self.assertRaises(FloatingPointError):
            flaps.effective_deflections_deg([np.nan]+[0.]*7, 0., 0.)

    def test_the_thrust_coefficient_scales_the_tail_surfaces_only(self):
        flaps, given = synthetic(), np.radians([1., 2., -1., 0.5, 3., -2., -4., 6.])
        plain = np.concatenate(flaps.coefficients(0.03, 0.01, given))
        scaled = np.concatenate(flaps.coefficients(0.03, 0.01, given, thrust_coeff=-0.2))
        table = flaps.derivatives_at(0.03, 0.01)
        degrees = np.degrees(given)
        assert_allclose(plain, degrees@table, rtol=1e-13)
        assert_allclose(plain-scaled, 0.2*(degrees[ELEVATOR]*table[ELEVATOR]+degrees[RUDDER]*table[RUDDER]), rtol=1e-11)

    def test_loads_are_dimensional_at_the_origin_in_the_openfawe_body_axes(self):
        flaps, settings = synthetic(), FlapSettings()
        given = np.radians([2., -3., 0., 0., 1., -2., -4., 6.])
        through_air = np.array([-40., 1.5, -3.])               # OpenFAWE body axes: forward is -x, up is +z
        force, moment, condition = flaps.loads(through_air, 1.1, given)
        makani = np.array([40., 1.5, 3.])
        alpha, beta = math.atan2(3., 40.), math.asin(1.5/np.linalg.norm(makani))
        self.assertAlmostEqual(condition["alpha_rad"], alpha, places=14)
        self.assertAlmostEqual(condition["beta_rad"], beta, places=14)
        pressure = 0.5*1.1*float(makani@makani)
        cf, cm = flaps.coefficients(alpha, beta, given)
        assert_allclose(force, pressure*settings.area_m2*cf*[-1., 1., -1.], rtol=1e-13)
        assert_allclose(moment, pressure*settings.area_m2*np.array([settings.span_m, settings.chord_m, settings.span_m])*cm*[-1., 1., -1.], rtol=1e-13)
        with self.assertRaises(ValueError):
            flaps.loads(np.zeros(3), 1.1, given)

    def test_the_makani_database_has_the_signs_its_own_check_asks_for(self):
        flaps = FlapIncrements.from_makani_json(DATABASE)
        self.assertEqual(flaps.derivatives.shape, (8, 6, 12, 11))
        for alpha in (0., 3., 6.):
            table = flaps.derivatives_at(math.radians(alpha), 0.)
            self.assertTrue(table[0, 3] >= 0 and table[1, 3] >= 0 and table[4, 3] <= 0 and table[5, 3] <= 0)   # ailerons: roll
            self.assertTrue(table[ELEVATOR, 4] <= 0 and table[RUDDER, 5] <= 0)                               # pitch, yaw
        # A positive elevator (trailing edge down) pitches the nose down: in the OpenFAWE axes (y the same) the moment is negative.
        _, moment, _ = flaps.loads([-45., 0., -2.], 1.026, [0.]*6+[math.radians(2.), 0.])
        self.assertLess(moment[1], 0.)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def state(self):
        rng = self.rng
        return RigidState(1.5, rng.uniform(-300, 300, 3), rotation_from_vector(rng.uniform(-2, 2, 3)), rng.uniform(-50, 50, 3),
                          rng.uniform(-1, 1, 3))

    def test_the_frames_map_back_and_forth(self):
        for _ in range(20):
            state = self.state()
            truth = mi.makani_truth(BODY, state, [3., -4., 0.5], acceleration_origin_m_s2=[1., 2., 3.])
            assert_allclose(truth.dcm_g2b@truth.dcm_g2b.T, np.eye(3), atol=1e-13)
            self.assertAlmostEqual(np.linalg.det(truth.dcm_g2b), 1.0, places=12)
            back = mi.plant_from_makani(BODY, state.time_s, truth.Xg, truth.Vg, truth.dcm_g2b, truth.pqr)
            assert_allclose(back.position_cm_m, state.position_cm_m, rtol=1e-13, atol=1e-11)
            assert_allclose(back.velocity_cm_m_s, state.velocity_cm_m_s, rtol=1e-13, atol=1e-12)
            assert_allclose(back.rotation, state.rotation, atol=1e-14)
            assert_allclose(back.angular_velocity_body_rad_s, state.angular_velocity_body_rad_s, atol=1e-15)
            assert_allclose(truth.Ag, [1., -2., -3.])           # z of g points down, y is mirrored
            assert_allclose(truth.wind_g, [3., 4., -0.5])

    def test_the_apparent_wind_has_makanis_incidence_and_the_other_sign_of_sideslip(self):
        for _ in range(20):
            state = self.state()
            wind = self.rng.uniform(-10, 10, 3)
            truth = mi.makani_truth(BODY, state, wind, acceleration_origin_m_s2=np.zeros(3))
            speed, alpha, beta = mi.apparent_wind_sph(truth.apparent_wind_b)
            relative, v, alpha_deg, beta_deg = relative_wind_body(state.pose(BODY), wind)
            self.assertAlmostEqual(speed, v, places=11)
            self.assertAlmostEqual(alpha, math.radians(alpha_deg), places=13)
            self.assertAlmostEqual(beta, -math.radians(beta_deg), places=13)

    def test_the_tether_force_in_spherical_form(self):
        assert_allclose(mi.tether_force_sph([0., 0., 1000.]), [1000., 0., 0.])
        tension, roll, pitch = mi.tether_force_sph([0., -300., 1000.])
        self.assertAlmostEqual(roll, math.atan2(300., 1000.), places=15)       # a force towards -y is a positive roll
        self.assertEqual(pitch, 0.)
        self.assertAlmostEqual(mi.tether_force_sph([200., 0., 1000.])[2], math.atan2(200., 1000.), places=15)
        # A force on the kite along the OpenFAWE body -z (towards the belly) is +z in Makani's body axes.
        state = RigidState(0., [0., 0., 100.], np.eye(3), [0., 0., 0.], [0., 0., 0.])
        truth = mi.makani_truth(BODY, state, np.zeros(3), acceleration_origin_m_s2=np.zeros(3), tether_force_inertial_N=[0., 0., -5.])
        assert_allclose(truth.tether_force_b, [0., 0., 5.])

    def test_the_origin_acceleration_of_a_body_turning_about_its_centre_of_mass(self):
        state = RigidState(0., [1., 2., 3.], rotation_from_vector([0.3, -0.2, 0.5]), [0., 0., 0.], [0.4, -0.3, 0.2])
        arm = state.rotation@BODY.centre_of_mass_body_m
        omega = state.rotation@state.angular_velocity_body_rad_s
        spin = np.array([0.05, 0.02, -0.03])
        expected = np.array([1., 0., 2.])+np.cross(state.rotation@spin, -arm)+np.cross(omega, np.cross(omega, -arm))
        assert_allclose(mi.origin_acceleration(BODY, state, [1., 0., 2.], spin), expected, rtol=1e-13)

    def test_the_wind_direction_is_the_azimuth_the_wind_points_to(self):
        self.assertAlmostEqual(mi.wind_direction([0., -9., 0.]), -math.pi/2.)
        self.assertAlmostEqual(mi.wind_direction([-1., 0., 0.]), -math.pi)       # wrapped to [-pi, pi)
        self.assertAlmostEqual(mi.wind_direction([1., 0., 0.], azi_ref_offset=math.pi), -math.pi)

    def test_the_servo_reference_model_against_a_fine_integration(self):
        gain, rate = 2.0*math.pi*15.91, 0.85
        for step in (-0.3, 0.005, -0.2):                       # inside the limits of A2 (-81 deg .. +10 deg)
            servo = mi.ServoReference(np.zeros(8))
            command = np.zeros(8)
            command[1] = step
            exact = [servo.step(command, 0.01)[1] for _ in range(50)]
            def slope(angle):
                return min(max(gain*(step-angle), -rate), rate)
            angle, fine, h = 0.0, [], 1e-5
            for k in range(50*1000):                           # classical fourth-order scheme (explicit Euler at this step is
                k1 = slope(angle)                              # itself 1.5e-6 off)
                k2 = slope(angle+0.5*h*k1)
                k3 = slope(angle+0.5*h*k2)
                k4 = slope(angle+h*k3)
                angle += h*(k1+2.0*k2+2.0*k3+k4)/6.0
                if (k+1) % 1000 == 0:
                    fine.append(angle)
            assert_allclose(exact, fine, rtol=0, atol=1e-9)
        servo = mi.ServoReference(np.zeros(8))
        self.assertTrue(servo.step([0.3]+[0.]*7, 0.01) is not None and servo.rate_limited[0] and not servo.rate_limited[1])

    def test_the_servo_stops_at_its_limits_and_the_rudder_goes_through_its_linkage(self):
        servo = mi.ServoReference(np.zeros(8))
        command = np.zeros(8)
        command[0], command[7] = math.radians(30.), math.radians(15.)
        for _ in range(300):
            flaps = servo.step(command, 0.01)
        self.assertAlmostEqual(flaps[0], math.radians(10.), places=9)             # A1 stops at +10 deg
        self.assertTrue(servo.at_limit[0] and not servo.at_limit[7])
        self.assertAlmostEqual(flaps[7], math.radians(15.), places=9)             # the rudder reaches its command
        self.assertAlmostEqual(servo.shaft[7], math.asin(math.radians(15.)/0.401), places=9)
        start = mi.ServoReference([0.]*7+[0.1])
        assert_allclose(start.flaps_rad, [0.]*7+[0.1], atol=1e-15)

    def test_rotor_speeds_follow_the_command_within_the_rate_limit(self):
        assert_allclose(mi.rate_limited_speeds([100., 150.], None, 0.01), [100., 150.])
        assert_allclose(mi.rate_limited_speeds([100., 150.], [120., 148.], 0.01), [115., 150.])


class HookTests(unittest.TestCase):
    def setUp(self):
        self.wing, self.polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        self.steps = 4
        self.motion = circular_motion(np.arange(self.steps+1)*DT, 130.0, 47.0, 9.0, 3.0)
        self.wind = np.array([9., 0., 0.])

    def backend(self):
        return AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=DT, max_rows=self.steps, core_radius_m=0.15,
                                                            solve=SolveSettings(solver="newton", max_iterations=60)))

    def run_free(self, trim=True, **options):
        return run_free(self.backend(), self.wing, self.polar, BODY, state_from_pose(self.motion.pose(0.), BODY), self.wind,
                        times=self.motion.time_s, external=Release(BODY, self.motion, release_time_s=0., trim=trim), **options)

    def test_hooks_that_add_nothing_leave_every_record_as_it_is(self):
        plain = self.run_free()
        hooked = self.run_free(before_aero=lambda *args: None, extra_loads=lambda pose, record: (np.zeros(3), np.zeros(3), {}))
        self.assertEqual(plain.status, "complete", plain.error)
        self.assertEqual(plain.records, hooked.records)

    def test_extra_loads_enter_the_held_loads_and_the_record(self):
        force, moment = np.array([300., -200., 1500.]), np.array([50., 800., -120.])
        calls = []

        def before(index, time_s, state, nodes, node_velocities, acceleration):
            calls.append((index, time_s, state, acceleration))
        plain = self.run_free(trim=False)                     # released without a trim load: the external callback adds nothing
        run = self.run_free(trim=False, before_aero=before, extra_loads=lambda pose, record: (force, moment, dict(marker=record["step"])))
        self.assertEqual(run.status, "complete", run.error)
        first, same = run.records[0], plain.records[0]
        rotation, arm = np.array(first["rotation"]), np.array(first["rotation"])@BODY.centre_of_mass_body_m
        assert_allclose(np.array(first["held_force_body_N"])-same["held_force_body_N"], rotation.T@force, rtol=1e-12)
        assert_allclose(np.array(first["held_moment_cm_body_Nm"])-same["held_moment_cm_body_Nm"],
                        rotation.T@(moment-np.cross(arm, force)), rtol=1e-11)
        self.assertEqual([record["marker"] for record in run.records], list(range(self.steps+1)))
        # The hook is called once per step before the solve; from the second step on it is given the plant's
        # acceleration at the new state under the loads held over the step before (and the external callback of that
        # step, which is zero here).
        self.assertEqual([call[0] for call in calls], list(range(self.steps+1)))
        self.assertIsNone(calls[0][3])
        for k in (1, 2):
            state, (acceleration, spin) = calls[k][2], calls[k][3]
            held = np.array(run.records[k-1]["held_force_body_N"])
            assert_allclose(acceleration, state.rotation@held/BODY.mass_kg+G, rtol=1e-12)
        self.assertGreater(np.linalg.norm(np.array(run.records[-1]["position_cm_m"])-plain.records[-1]["position_cm_m"]), 1e-6)

    def test_the_hooks_are_refused_with_the_predictor_corrector(self):
        with self.assertRaises(ValueError):
            self.run_free(scheme="heun", extra_loads=lambda pose, record: (np.zeros(3), np.zeros(3), {}))
        with self.assertRaises(ValueError):
            self.run_free(scheme="heun", before_aero=lambda *args: None)

    def test_externals_add_up_and_pass_the_step_notice_on(self):
        release = Release(BODY, self.motion, release_time_s=DT, trim=False)
        both = Externals(release, lambda time_s, state, model: (np.array([1., 2., 3.]), np.array([4., 5., 6.])), None)
        both.begin_step(2*DT)
        state = state_from_pose(self.motion.pose(2*DT), BODY)
        force, moment = both(2*DT, state, dict(force_inertial_N=np.zeros(3), moment_cm_inertial_Nm=np.zeros(3)))
        assert_allclose(force, [1., 2., 3.])
        assert_allclose(moment, [4., 5., 6.])


# ---------------------------------------------------------------------------------------------------------------------
# CrosswindLoop against a stand-in for the controller library (the library itself is not available to the test set)
# ---------------------------------------------------------------------------------------------------------------------
import ctypes  # noqa: E402

from openfawe.flight.closed_loop import CrosswindLoop  # noqa: E402
from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set  # noqa: E402
from openfawe.rotor.table import RotorTable  # noqa: E402


def _struct(name, fields):
    return type(name, (ctypes.Structure,), dict(_fields_=fields))


class StandInApi:
    """The names ``CrosswindLoop`` uses of the binding, with a "controller" whose outputs are a known function of the call count."""
    kNumFlaps = kNumMotors = 8
    kStackingStateNormal, kApparentWindSolutionTypePitot, kExperimentTypeNoTest = 0, 1, 0
    kFlightModeTransIn, kFlightModeCrosswindNormal, kFlightModeCrosswindPrepTransOut = 6, 7, 8
    D = ctypes.c_double
    Vec3 = _struct("Vec3", [("x", D), ("y", D), ("z", D)])
    Vec2 = _struct("Vec2", [("x", D), ("y", D)])
    Mat3 = _struct("Mat3", [("d", (D*3)*3)])
    ApparentWindSph = _struct("ApparentWindSph", [("v", D), ("alpha", D), ("beta", D)])
    TetherForceSph = _struct("TetherForceSph", [("tension", D), ("roll", D), ("pitch", D)])
    _Wind = _struct("_Wind", [("solution_type", ctypes.c_int), ("vector", Vec3), ("sph", ApparentWindSph), ("sph_f", ApparentWindSph)])
    _JoyData = _struct("_JoyData", [("throttle", D)])
    JoystickEstimate = _struct("JoystickEstimate", [("valid", ctypes.c_bool), ("throttle_f", D), ("data", _JoyData)])
    _Tether = _struct("_Tether", [("valid", ctypes.c_bool), ("vector", Vec3), ("vector_f", Vec3), ("sph", TetherForceSph), ("tension_f", D)])
    _WindG = _struct("_WindG", [("valid", ctypes.c_bool), ("vector", Vec3), ("vector_f", Vec3), ("vector_f_slow", Vec3), ("dir_f", D),
                                ("dir_f_playbook", D), ("speed_f", D), ("speed_f_playbook", D)])
    _Vessel = _struct("_Vessel", [("dcm_g2v", Mat3)])
    _Experiment = _struct("_Experiment", [("active_type", ctypes.c_int), ("staged_type", ctypes.c_int)])
    PerchAziEstimate = _struct("PerchAziEstimate", [("valid", ctypes.c_bool), ("angle", D)])
    StateEstimate = _struct("StateEstimate", [
        ("stacking_state", ctypes.c_int), ("dcm_g2b", Mat3), ("pqr", Vec3), ("pqr_f", Vec3), ("Xg", Vec3), ("Vg", Vec3), ("Vg_f", Vec3),
        ("Ag", Vec3), ("Vb", Vec3), ("Vb_f", Vec3), ("Ab_f", Vec3), ("acc_norm_f", D), ("gps_active", ctypes.c_bool), ("rho", D),
        ("apparent_wind", _Wind), ("joystick", JoystickEstimate), ("tether_force_b", _Tether), ("wind_g", _WindG), ("wind_aloft_g", _WindG),
        ("vessel", _Vessel), ("experiment", _Experiment), ("perch_azi", PerchAziEstimate)])
    ControlOutput = _struct("ControlOutput", [("flaps", D*8), ("motor_speed_upper_limit", D*8), ("motor_speed_lower_limit", D*8),
                                               ("motor_torque", D*8)])
    CrosswindState = _struct("CrosswindState", [("calls", ctypes.c_int)])
    ControlOutputState = _struct("ControlOutputState", [("flaps_z1", D*8)])
    HoverState = _struct("HoverState", [("unused", D)])
    FlightStatus = _struct("FlightStatus", [("flight_mode", ctypes.c_int), ("last_flight_mode", ctypes.c_int),
                                             ("flight_mode_first_entry", ctypes.c_bool), ("flight_mode_time", D)])
    _Output = _struct("_Output", [("flap_offsets", D*8)])
    _Crosswind = _struct("_Crosswind", [("output", _Output)])
    _RotorControl = _struct("_RotorControl", [("idle_speed", D)])
    _ControlOutputParams = _struct("_ControlOutputParams", [("unused", D)])
    ControlParams = _struct("ControlParams", [("crosswind", _Crosswind), ("control_output", _ControlOutputParams), ("rotor_control", _RotorControl)])
    _Telemetry = _struct("_Telemetry", [("loop_angle", D), ("path_center_g", Vec3), ("path_radius_target", D), ("alpha_cmd", D), ("beta_cmd", D),
                                        ("tether_roll_cmd", D), ("airspeed_cmd", D), ("k_aero_cmd", D), ("k_geom_cmd", D), ("k_geom_curr", D),
                                        ("k_aero_curr", D), ("current_pos_cw", Vec2), ("target_pos_cw", Vec2), ("thrust_ff", D), ("thrust_fb", D),
                                        ("int_tether_roll_error", D), ("int_beta_error", D), ("int_aileron", D), ("int_elevator", D),
                                        ("int_rudder", D), ("int_thrust", D), ("int_crosstrack", D), ("pqr_cmd", Vec3)])
    OFFSETS = (-0.2, -0.2, 0.0, 0.0, -0.04, -0.04, 0.004, 0.008)

    class GlobalSystemParams:
        _ts = ctypes.c_double(0.01)

        @classmethod
        def in_dll(cls, library, name):
            return type("G", (), dict(ts=ctypes.pointer(cls._ts)))()

    def __init__(self, flaps_of, speeds_of):
        self._lib = None
        self.flaps_of, self.speeds_of = flaps_of, speeds_of
        self.params = self.ControlParams()
        self.params.crosswind.output.flap_offsets = (ctypes.c_double*8)(*self.OFFSETS)
        self.params.rotor_control.idle_speed = 30.0
        self.telemetry = self._Telemetry()
        self.seen, self.init_flaps = [], None

    def GetSystemParams(self):
        ns = type("N", (), {})
        system = ns()
        system.ground_station, system.phys, system.wing, system.tether = ns(), ns(), ns(), ns()
        system.ground_station.azi_ref_offset, system.phys.rho, system.wing.m, system.tether.length = 0.0, 1.026, 1730.6, 425.8
        return type("P", (), dict(contents=system))()

    def GetControlParams(self):
        return type("P", (), dict(contents=self.params))()

    def GetCrosswindTelemetry(self):
        return type("P", (), dict(contents=self.telemetry))()

    def ApparentWindCartToSph(self, vector, sph):
        v = vector._obj
        sph._obj.v, sph._obj.alpha, sph._obj.beta = mi.apparent_wind_sph([v.x, v.y, v.z])

    def TetherForceCartToSph(self, vector, sph):
        v = vector._obj
        sph._obj.tension, sph._obj.roll, sph._obj.pitch = mi.tether_force_sph([v.x, v.y, v.z])

    def CrosswindInit(self, est, flaps_z1, detwist, revolutions, loop_angle, params, hover, state):
        self.init_flaps = (list(flaps_z1), detwist, revolutions, loop_angle)
        state._obj.calls = 0

    def ControlOutputInit(self, params, state):
        for k in range(8):
            state._obj.flaps_z1[k] = 0.0

    def CrosswindStep(self, status, est, params, state, raw):
        call = state._obj.calls
        self.seen.append((call, status._obj.flight_mode, status._obj.last_flight_mode, bool(status._obj.flight_mode_first_entry),
                          status._obj.flight_mode_time, est._obj.acc_norm_f, est._obj.wind_g.dir_f, est._obj.apparent_wind.sph.alpha))
        for k in range(8):
            raw._obj.flaps[k] = self.flaps_of(call)[k]
            raw._obj.motor_speed_upper_limit[k] = raw._obj.motor_speed_lower_limit[k] = self.speeds_of(call)[k]
        state._obj.calls = call+1

    def ControlOutputStep(self, raw, joystick, perch_azi, params, state, out):
        ctypes.memmove(ctypes.addressof(out._obj), ctypes.addressof(raw._obj), ctypes.sizeof(raw._obj))


class CrosswindLoopTests(unittest.TestCase):
    """Which command acts in which step, and the loads the loop adds (review finding CL-05 of the closed-loop step)."""

    @classmethod
    def setUpClass(cls):
        cls.discs = load_rotor_set(ROOT/"cases/rotors/m600_rotors_makani.json",
                                   RotorTable.from_makani_json(ROOT/"cases/rotors/makani/rotor_rev4.json"))
        cls.inertia = load_rotor_inertia(ROOT/"cases/rotors/m600_rotors_makani.json")
        cls.flaps = FlapIncrements.from_makani_json(DATABASE)

    def setUp(self):
        offsets = np.array(StandInApi.OFFSETS)
        self.flaps_of = lambda call: offsets+0.01*(call+1)*np.array([1., 1., 0., 0., -1., -1., 0.5, -0.5])
        self.speeds_of = lambda call: np.full(8, 150.0)+np.array([0., 2., 9., 30., 0., -2., -9., -30.])*call
        self.api = StandInApi(self.flaps_of, self.speeds_of)
        self.wind = np.array([-3., 8.4, 0.])
        self.loop = CrosswindLoop(self.api, BODY, self.wind, start_step=2, step_s=0.01, density_kg_m3=1.026, flap_increments=self.flaps,
                                  discs=self.discs, rotor_inertia_kg_m2=self.inertia)
        self.state = RigidState(0.0, np.array([120., -300., 250.]), rotation_from_vector([0.3, -0.2, 1.1]), np.array([-30., -10., 28.]),
                                np.array([0.02, -0.05, 0.2]))
        self.acceleration = (np.array([1., -2., 3.]), np.array([0.01, 0.02, -0.03]))

    def step(self, index):
        self.loop.before_aero(index, 0.01*index, self.state, None, None, self.acceleration)
        return self.loop.extra_loads(self.state.pose(BODY), {})

    def test_nothing_happens_before_the_start_and_the_first_call_needs_an_acceleration(self):
        self.loop.before_aero(0, 0.0, self.state, None, None, None)
        force, moment, fields = self.loop.extra_loads(self.state.pose(BODY), {})
        self.assertEqual((force.tolist(), moment.tolist(), fields), ([0.]*3, [0.]*3, dict(controller_active=False)))
        assert_allclose(self.loop.rotor_speeds(0.0), 30.0)                  # idle before the start (schedule of the rings)
        assert_allclose(self.loop.external(0.0, self.state, {})[1], 0.0)
        self.assertEqual(self.api.seen, [])
        with self.assertRaises(ValueError):
            self.loop.before_aero(2, 0.02, self.state, None, None, None)

    def test_the_controller_is_entered_as_the_contract_says(self):
        self.step(2)
        flaps, detwist, revolutions, loop_angle = self.api.init_flaps
        assert_allclose(flaps, StandInApi.OFFSETS, atol=1e-15)                # flaps_z1 of CrosswindInit: the offsets
        self.assertEqual((detwist, revolutions, loop_angle), (0.0, 0, 0.0))
        assert_allclose(list(self.loop.output_state.flaps_z1), StandInApi.OFFSETS, atol=1e-15)   # and of the output stage
        call, mode, last, first, mode_time, acc_norm, direction, alpha = self.api.seen[0]
        self.assertEqual((mode, last, first, mode_time), (7, 8, True, 0.0))  # crosswind normal, entered from prep-trans-out
        self.step(3)
        self.assertEqual(self.api.seen[1][3:5], (False, 0.01))
        # the estimate: specific-force norm of the body origin, the azimuth the wind blows to, Makani's incidence
        origin = mi.origin_acceleration(BODY, self.state, *self.acceleration)
        self.assertAlmostEqual(acc_norm, np.linalg.norm(origin-G), places=12)
        self.assertAlmostEqual(direction, math.atan2(-8.4, -3.), places=12)   # g has y and z opposite to W
        _, _, alpha_deg, _ = relative_wind_body(self.state.pose(BODY), self.wind)
        self.assertAlmostEqual(alpha, math.radians(alpha_deg), places=12)

    def test_the_flaps_of_a_step_are_the_servo_state_at_its_start(self):
        servo, acting = mi.ServoReference(np.array(StandInApi.OFFSETS)), []
        for index in range(2, 7):
            _, _, fields = self.step(index)
            acting.append(fields["flaps_acting_rad"])
            assert_allclose(fields["flaps_acting_rad"], servo.flaps_rad, rtol=0, atol=1e-15)   # before this step's command
            assert_allclose(self.loop.log[-1]["flaps_command_rad"], self.flaps_of(index-2), rtol=0, atol=1e-15)
            servo.step(self.flaps_of(index-2), 0.01)
        assert_allclose(acting[0], StandInApi.OFFSETS, atol=1e-15)            # the first step: the offsets, whatever the command
        self.assertGreater(np.abs(np.array(acting[1])-acting[0]).max(), 1e-4)

    def test_rotor_speeds_are_the_commands_of_the_step_within_the_rate_limit(self):
        speeds = []
        for index in range(2, 6):
            _, _, fields = self.step(index)
            speeds.append(np.array(fields["rotor_omega_rad_s"]))
        assert_allclose(speeds[0], self.speeds_of(0))                           # the first command is taken as it is
        expected = speeds[0]+np.clip(self.speeds_of(1)-speeds[0], -5.0, 5.0)    # 500 rad/s^2 over 0.01 s
        assert_allclose(speeds[1], expected)
        self.assertTrue(self.loop.log[1]["rotor_speed_rate_limited"] and not self.loop.log[0]["rotor_speed_rate_limited"])
        assert_allclose(self.loop.omega_rates, (speeds[3]-speeds[2])/0.01)

    def test_rotor_loads_are_the_table_loads_about_the_origin_and_the_tail_sees_the_thrust_of_the_step_before(self):
        pose = self.state.pose(BODY)
        force, moment, fields = self.step(2)
        expected_force, expected_moment, coefficients = np.zeros(3), np.zeros(3), []
        for disc, omega in zip(self.discs, self.speeds_of(0)):
            placed = disc.placed(pose.rotation, pose.position_m)
            row = placed.loads(omega, self.wind-pose.point_velocity(placed.centre_m[None, :])[0], 1.026)
            expected_force += row["force_N"]
            expected_moment += np.cross(placed.centre_m-pose.position_m, row["force_N"])+row["moment_Nm"]
            coefficients.append(row["thrust_N"]/(0.5*1.026*placed.area_m2*max(abs(row["axial_freestream_m_s"]), 30.0)**2))
        assert_allclose(fields["rotor_force_inertial_N"], expected_force, rtol=1e-13)
        assert_allclose(fields["rotor_moment_origin_inertial_Nm"], expected_moment, rtol=1e-12)
        self.assertEqual(fields["flap_thrust_coeff_used"], 0.0)                 # nothing before the first step
        self.assertAlmostEqual(fields["rotor_thrust_coeff"], float(np.mean(coefficients)), places=14)
        through_air = pose.rotation.T@(pose.velocity_m_s-self.wind)
        flap_force, flap_moment, _ = self.flaps.loads(through_air, 1.026, np.array(StandInApi.OFFSETS), 0.0)
        assert_allclose(force, expected_force+pose.rotation@flap_force, rtol=1e-12)
        assert_allclose(moment, expected_moment+pose.rotation@flap_moment, rtol=1e-12)
        _, _, second = self.step(3)
        self.assertEqual(second["flap_thrust_coeff_used"], fields["rotor_thrust_coeff"])

    def test_gyroscopic_and_reaction_moments(self):
        for index in range(2, 5):
            self.step(index)
        force, moment = self.loop.external(0.04, self.state, {})
        sense = np.array([disc.direction for disc in self.discs], dtype=float)
        axes = np.array([disc.axis for disc in self.discs])
        momentum = (self.inertia*self.loop.omegas*sense)@axes                   # body axes
        rate = (self.inertia*self.loop.omega_rates*sense)@axes
        expected = self.state.rotation@(np.cross(momentum, self.state.angular_velocity_body_rad_s)-rate)
        assert_allclose(force, 0.0)
        assert_allclose(moment, expected, rtol=1e-13)
        self.assertGreater(np.linalg.norm(rate), 0.0)                           # the unequal speed changes leave a net reaction

    def test_a_non_finite_output_stops_the_run_and_is_logged(self):
        self.api.flaps_of = lambda call: np.array([0., 0., 0., 0., 0., float("nan"), 0., 0.])
        with self.assertRaises(FloatingPointError):
            self.step(2)
        self.assertIsNotNone(self.loop.failed)
        self.assertTrue(self.loop.log[-1]["non_finite_outputs"])

    def test_a_non_finite_command_the_output_stage_would_hide_is_still_seen(self):
        # Makani's Saturate turns a NaN into the lower limit; the stand-in output stage does the same here.
        def output_stage(raw, joystick, perch_azi, params, state, out):
            for k in range(8):
                value = raw._obj.flaps[k]
                out._obj.flaps[k] = -0.5 if value != value else value
                out._obj.motor_speed_upper_limit[k] = out._obj.motor_speed_lower_limit[k] = raw._obj.motor_speed_upper_limit[k]
        self.api.ControlOutputStep = output_stage
        self.api.flaps_of = lambda call: np.array([0., 0., 0., float("nan"), 0., 0., 0., 0.])
        with self.assertRaises(FloatingPointError):
            self.step(2)
        self.assertEqual(self.loop.log[-1]["non_finite_outputs"], ["raw.flaps[3]"])

    def test_discs_without_any_rotor_load_are_refused(self):
        loop = CrosswindLoop(self.api, BODY, self.wind, start_step=2, step_s=0.01, density_kg_m3=1.026, flap_increments=self.flaps,
                             discs=self.discs, rotor_inertia_kg_m2=self.inertia, rotor_tables=False)
        loop.before_aero(2, 0.02, self.state, None, None, self.acceleration)
        with self.assertRaises(ValueError):
            loop.extra_loads(self.state.pose(BODY), {})

    def test_without_a_flap_increment_model_only_the_rotor_loads_are_added(self):
        # for a driver whose aerodynamic model holds the control surfaces itself (flight.reference_flight)
        loop = CrosswindLoop(self.api, BODY, self.wind, start_step=2, step_s=0.01, density_kg_m3=1.026, flap_increments=None,
                             discs=self.discs, rotor_inertia_kg_m2=self.inertia)
        loop.before_aero(2, 0.02, self.state, None, None, self.acceleration)
        before = loop.thrust_coeff
        force, moment, fields = loop.extra_loads(self.state.pose(BODY), {})
        assert_allclose(force, fields["rotor_force_inertial_N"], rtol=0, atol=0)
        assert_allclose(moment, fields["rotor_moment_origin_inertial_Nm"], rtol=0, atol=0)
        self.assertNotIn("flap_force_inertial_N", fields)
        self.assertEqual(before, 0.0)
        self.assertEqual(loop.thrust_coeff, fields["rotor_thrust_coeff"])       # updated for the step after

    def test_the_controllers_constants_are_recorded_beside_the_plants(self):
        self.assertEqual(self.loop.consistency["density_controller_kg_m3"], 1.026)
        self.assertEqual(self.loop.consistency["mass_plant_kg"], BODY.mass_kg)
        self.assertEqual(self.loop.consistency["mass_controller_kg"], 1730.6)


if __name__ == "__main__":
    unittest.main()
