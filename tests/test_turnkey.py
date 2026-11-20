"""Plan 3.7 part 3 (full-cycle contract revision 6): the pieces of ``ground.turnkey`` and ``ground.turnkey_scoring``.

Without the library: the flying system's rates against part 2's (no added loads), the added loads and rotor moments,
the blown-wing force against a transcription in Makani's body axes, the period loads, the joystick schedule and the
transcribed Makani scores. With the library (OPENFAWE_MAKANI_FULL_BUILD; a skip fails the judged run): the corrected
``Ab_f`` of ``ground.perch.kite_estimate`` (K4).
"""
import math
import os
from pathlib import Path
import unittest

import numpy as np

from openfawe.ground import gs02 as G
from openfawe.ground import perch as P
from openfawe.ground import turnkey as T
from openfawe.ground import turnkey_scoring as S
from openfawe.flight.makani_interface import MAKANI_BODY_TO_BODY
from openfawe.flight.plant import HeldLoads
from openfawe.flight.rigid_body import RigidState

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT/"cases"/"ground"/"makani_gs02_parker_ranch.json"
DATA = ROOT/"cases"/"polars"/"makani"
BUILD = os.environ.get("OPENFAWE_MAKANI_FULL_BUILD")


def rotors():
    from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set
    from openfawe.rotor.table import RotorTable
    discs = load_rotor_set(ROOT/"cases/rotors/m600_rotors_makani.json",
                           RotorTable.from_makani_json(ROOT/"cases/rotors/makani/rotor_rev4.json"))
    return discs, load_rotor_inertia(ROOT/"cases/rotors/m600_rotors_makani.json")


def database():
    from openfawe.aero.makani_database import MakaniDatabase
    return MakaniDatabase.from_files(DATA/"m600_aswing_baseline.json", spoiler_path=DATA/"spoiler_offset_tables.json",
                                     high_path=DATA/"hover_model_sn02_wind_10_no_wake_model.json")


def initialised(system):
    init, system = P.perched_initialisation(system.params, system)
    return init["wing_state"], init["nodes"], init["node_velocities"], system


def moved(state, rotation_vector, omega, velocity):
    from openfawe.flight.kinematics import rotation_from_vector
    return RigidState(state.time_s, state.position_cm_m + np.array([0.3, -0.2, 0.5]),
                      state.rotation@rotation_from_vector(np.asarray(rotation_vector, dtype=float)),
                      np.asarray(velocity, dtype=float), np.asarray(omega, dtype=float))


class FlyingSystemRates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = G.load_gs02_params(PARAMS)
        cls.discs, cls.inertia = rotors()

    def test_no_added_loads_equals_part_two(self):
        # Zero held loads, rotors at rest and no wind: the same derivatives as PerchedSystem (part 2, judged in G4-G7).
        flying = T.FlyingSystem(self.params, self.discs, self.inertia, np.zeros(3))
        state, nodes, speeds, flying = initialised(flying)
        perched = P.PerchedSystem(self.params)
        _, _, _, perched = initialised(perched)
        s = moved(state, [0.02, -0.01, 0.03], [0.1, -0.2, 0.05], [0.4, 0.1, -0.3])
        a = flying.rates(s, nodes, speeds, flying.gs.x)
        b = perched.rates(s, nodes, speeds, perched.gs.x)
        for x, y in zip(a[:6], b[:6]):
            np.testing.assert_array_equal(x, y)

    def test_added_loads_and_rotor_moments(self):
        flying = T.FlyingSystem(self.params, self.discs, self.inertia, np.zeros(3))
        state, nodes, speeds, flying = initialised(flying)
        s = moved(state, [0.02, -0.01, 0.03], [0.1, -0.2, 0.05], [0.4, 0.1, -0.3])
        base = flying.rates(s, nodes, speeds, flying.gs.x)
        force, moment = np.array([120.0, -40.0, 900.0]), np.array([30.0, -250.0, 75.0])
        omegas = np.linspace(100.0, 170.0, 8)
        rates = np.linspace(-50.0, 40.0, 8)
        flying.set_period(HeldLoads(force, moment), omegas, rates)
        added = flying.rates(s, nodes, speeds, flying.gs.x)
        body = flying.body
        np.testing.assert_allclose(added[1] - base[1], s.rotation@force/body.mass_kg, rtol=0, atol=1e-12)
        sense = np.array([d.direction for d in self.discs], dtype=float)
        axes = np.array([d.axis for d in self.discs], dtype=float)
        momentum = (self.inertia*omegas*sense)@axes
        gyro_body = np.cross(momentum, s.angular_velocity_body_rad_s)
        reaction_body = (-self.inertia*rates*sense)@axes
        expected = np.linalg.solve(body.inertia_body_kg_m2, moment + gyro_body + reaction_body)
        np.testing.assert_allclose(added[2] - base[2], expected, rtol=0, atol=1e-10)

    def test_tether_wind(self):
        # The uniform wind enters every node's drag. Perched, no node is active and the staging node (18) is the only
        # one with drag: its acceleration changes by the change of CalcAeroForceSeg over its mass, nothing else does.
        wind = np.array([7.66, 6.43, 0.0])
        flying = T.FlyingSystem(self.params, self.discs, self.inertia, np.zeros(3))
        windy = T.FlyingSystem(self.params, self.discs, self.inertia, wind)
        state, nodes, speeds, flying = initialised(flying)
        _, _, _, windy = initialised(windy)
        speeds = speeds + np.array([0.2, -0.1, 0.05])
        a = flying.tether_evaluation(state, nodes, speeds, flying.gs.x)
        b = windy.tether_evaluation(state, nodes, speeds, windy.gs.x)
        self.assertEqual(a[2]["indices"], b[2]["indices"])
        self.assertEqual(a[2]["indices"].start, 19)
        np.testing.assert_array_equal(windy.node_wind, np.tile(wind, (windy.line.n, 1)))
        line, out = windy.line, b[2]
        change = (line._aero_one(out["start_point"], out["end_point"], speeds[18], wind)
                  - line._aero_one(out["start_point"], out["end_point"], speeds[18], np.zeros(3)))
        expected = np.zeros_like(nodes)
        expected[18] = change/out["masses"][18]
        np.testing.assert_array_equal(np.delete(b[1], 18, axis=0), np.delete(a[1], 18, axis=0))
        np.testing.assert_allclose(b[1][18] - a[1][18], expected[18], rtol=1e-7, atol=0)   # spring terms are far larger
        self.assertGreater(float(np.linalg.norm(change)), 0.0)


class BlownWing(unittest.TestCase):
    def test_against_makani_body_axes(self):
        # rotor.cc CalcBlownWingForceMomentPos in Makani's body axes: F = T a_m, rotated by AngleToDcm(0, -theta, 0,
        # ZYX) (the DCM of a pitch -theta), minus F; theta = Crossfade(0.3, 0, V, 6, 10). Mapped to OpenFAWE axes.
        from openfawe.rotor.makani_corrections import blown_wing_force
        for axis_o, thrust, v in (([-1.0, 0.0, 0.0], 2500.0, 0.0), ([-0.99, 0.0, 0.14], 3100.0, 7.5),
                                  ([-1.0, 0.0, 0.0], 1800.0, 12.0), ([-0.9, 0.3, 0.3], -400.0, 9.0)):
            axis_o = np.asarray(axis_o)/np.linalg.norm(axis_o)
            theta = 0.3 + (0.0 - 0.3)*min(max((v - 6.0)/4.0, 0.0), 1.0)
            p = -theta
            dcm = np.array([[math.cos(p), 0.0, -math.sin(p)], [0.0, 1.0, 0.0], [math.sin(p), 0.0, math.cos(p)]])
            f_m = thrust*(MAKANI_BODY_TO_BODY@axis_o)
            expected = MAKANI_BODY_TO_BODY@(dcm@f_m - f_m)
            np.testing.assert_allclose(blown_wing_force(thrust*axis_o, v), expected, rtol=0, atol=1e-9)


class PeriodLoads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = G.load_gs02_params(PARAMS)
        cls.discs, cls.inertia = rotors()
        cls.aero = database()

    def test_rotors_at_rest_aero_only(self):
        from openfawe.flight.reference_flight import reference_loads_body
        wind = np.array([7.66, 6.43, 0.0])
        system = T.FlyingSystem(self.params, self.discs, self.inertia, wind)
        state, _, _, system = initialised(system)
        flaps = np.radians([1.0, -2.0, 3.0, 0.5, -1.0, 2.0, -4.0, 5.0])
        held, coefficient, fields = T.period_loads(system, self.aero, state, flaps, 0.2, 1.026)
        aero, _, _ = reference_loads_body(self.aero, system.body, state, wind, 1.026, flaps, 0.2)
        np.testing.assert_array_equal(held.force_body_N, aero.force_body_N + 0.0)
        self.assertEqual(coefficient, 0.0)
        self.assertEqual(fields["rotor_thrust_N"], [0.0]*8)

    def test_rotor_loads_and_blown_wing(self):
        from openfawe.flight.free_flight import held_loads
        from openfawe.rotor.makani_corrections import blown_wing_force
        wind = np.array([7.66, 6.43, 0.0])
        system = T.FlyingSystem(self.params, self.discs, self.inertia, wind)
        state, _, _, system = initialised(system)
        omegas = np.linspace(150.0, 190.0, 8)
        system.set_period(HeldLoads(), omegas, np.zeros(8))
        off, coefficient, f_off = T.period_loads(system, self.aero, state, np.zeros(8), 0.0, 1.026)
        on, _, f_on = T.period_loads(system, self.aero, state, np.zeros(8), 0.0, 1.026, blown_wing=True)
        pose = state.pose(system.body)
        blown = sum(blown_wing_force(t*d.axis, v) for t, d, v in zip(f_off["rotor_thrust_N"], self.discs,
                                                                     f_off["rotor_axial_freestream_m_s"]))
        extra = held_loads(system.body, pose.rotation, pose.rotation@blown, np.zeros(3))
        np.testing.assert_allclose(on.force_body_N - off.force_body_N, extra.force_body_N, rtol=0, atol=1e-9)
        np.testing.assert_allclose(on.moment_cm_body_Nm - off.moment_cm_body_Nm, extra.moment_cm_body_Nm, rtol=0, atol=1e-8)
        self.assertGreater(min(f_off["rotor_thrust_N"]), 0.0)
        area = self.discs[0].area_m2
        expected = np.mean([t/(0.5*1.026*area*max(abs(v), 30.0)**2) for t, v in zip(f_off["rotor_thrust_N"],
                                                                                   f_off["rotor_axial_freestream_m_s"])])
        self.assertAlmostEqual(coefficient, expected, places=14)


class Joystick(unittest.TestCase):
    def test_turnkey_schedule(self):
        class Api:
            kJoystickSwitchPositionUp, kJoystickSwitchPositionMiddle = 2, 1
        api = Api()
        self.assertEqual((T.joystick_at(api, 0).switch_position, T.joystick_at(api, 0).throttle), (2, 0.0))
        self.assertEqual(T.joystick_at(api, 1009).switch_position, 2)
        self.assertEqual(T.joystick_at(api, 1010).switch_position, 1)
        self.assertEqual(T.joystick_at(api, 1099).throttle, 0.0)
        self.assertEqual(T.joystick_at(api, 1100).throttle, 0.81)
        self.assertEqual(T.joystick_at(api, 81999).throttle, 0.81)
        self.assertEqual(T.joystick_at(api, 82000).throttle, 0.39)


class Scores(unittest.TestCase):
    NAMES = {0: "PilotHover", 1: "Perched", 2: "HoverAscend", 3: "HoverPayOut", 12: "HoverDescend", 9: "TransIn"}

    def records(self, modes, **fields):
        out = []
        for k, mode in enumerate(modes):
            r = dict(t=k*0.01, flight_mode=mode, controller_flight_mode=mode, gain_ramp_scale=1.0, rotor_omega_rad_s=[100.0]*8, rotor_thrust_N=[1000.0]*8,
                     tension_bridle_N=5000.0, payout_m=10.0, tension_cmd_N=5000.0, est_tension_N=5000.0,
                     thrust_moment=[0.0]*4, thrust_moment_avail=[0.0]*4, platform_azi=0.0, Xg=[0.0, 0.0, 0.0])
            for key, value in fields.items():
                r[key] = value[k] if isinstance(value, list) and len(value) == len(modes) else value
            out.append(r)
        return out

    def test_lp_filtfilt_against_scipy(self):
        from scipy import signal
        t = np.arange(400)*0.01
        x = np.sin(2*np.pi*0.5*t) + 0.3*np.sin(2*np.pi*20*t)
        b, a = signal.butter(1, 4.0/50.0, "lowpass")
        np.testing.assert_array_equal(S.lp_filtfilt(t, x, 4.0), signal.filtfilt(b, a, x, padtype=None))
        gap = np.concatenate([t[:200], t[250:]])
        y = np.concatenate([x[:200], x[250:]])
        expected = np.concatenate([signal.filtfilt(b, a, y[:200], padtype=None), signal.filtfilt(b, a, y[200:], padtype=None)])
        np.testing.assert_array_equal(S.lp_filtfilt(gap, y, 4.0), expected)
        self.assertTrue(math.isnan(S.lp_filtfilt(t[:9], x[:9], 4.0)[0]))

    def test_intervals_and_saturation(self):
        self.assertEqual(S.get_intervals(np.array([1, 2, 3, 7, 8, 12])), [(1, 3), (7, 8), (12, 12)])
        modes = [2]*20
        tm = [[0.0]*4 for _ in modes]
        for i in range(5, 10):
            tm[i] = [1.0, 0.0, 0.0, 0.0]
        recs = self.records(modes, thrust_moment=tm)
        idx = S.select(recs, self.NAMES, ("HoverAscend",))
        limits = dict(max_thrust=50.0, min_moment_y=-1.0, max_moment_y=1.0, min_moment_z=-1.0, max_moment_z=1.0)
        self.assertAlmostEqual(S.saturation_duration_s(recs, idx, "thrust", limits), 0.04)
        self.assertEqual(S.saturation_duration_s(recs, idx, "moment_y", limits), 0.0)

    def test_selection_descend_gain_and_missing_modes(self):
        modes = [2]*30 + [12]*30
        gain = [1.0]*30 + [1.0]*10 + [0.9]*20
        recs = self.records(modes, gain_ramp_scale=gain)
        self.assertEqual(S.select(recs, self.NAMES, ("HoverDescend",)), list(range(30, 40)))
        self.assertAlmostEqual(S.duration(recs, S.select(recs, self.NAMES, ("HoverAscend",))), 0.29)
        result = S.p4(recs, self.NAMES)
        by = {row["label"]: row for row in result["scores"]}
        self.assertFalse(result["passed"])                       # TransIn and the other modes never reached
        self.assertIsNone(by["max rotor speed, TransIn (1013-1015)"]["value"])
        self.assertFalse(by["max rotor speed, TransIn (1013-1015)"]["passed"])
        self.assertTrue(by["min tension, HoverAscend (924-926)"]["passed"])

    def test_selection_reads_the_controller_label(self):
        # Makani's scores select by ControlTelemetry.flight_mode, the mode the controller ran in (review P3C-01): the
        # first record whose planner mode is the new one still belongs to the old mode.
        planner = [2]*10 + [3]*10
        controller = [2]*11 + [3]*9
        recs = self.records(planner, controller_flight_mode=controller)
        self.assertEqual(S.select(recs, self.NAMES, ("HoverAscend",)), list(range(11)))
        self.assertEqual(S.select(recs, self.NAMES, ("HoverAscend",), key="flight_mode"), list(range(10)))

    def test_limits_are_strict_at_the_bad_value(self):
        self.assertFalse(S.upper_passes(225.0, 225.0))
        self.assertTrue(S.upper_passes(224.999, 225.0))
        self.assertFalse(S.lower_passes(1.0, 1.0))
        self.assertFalse(S.range_passes(1.5, 1.5, 15.0))
        self.assertTrue(S.range_passes(1.51, 1.5, 15.0))
        self.assertFalse(S.upper_passes(None, 225.0))
        self.assertFalse(S.upper_passes(float("nan"), 225.0))

    def test_tension_control_payout_filter(self):
        modes = [3]*10
        payout = [5.0, 6.0, 7.0, 8.0, 6.5, 9.0, 10.0, 6.0, 5.0, 4.0]
        cmd = [5000.0 + 1000.0*i for i in range(10)]
        recs = self.records(modes, payout_m=payout, tension_cmd_N=cmd)
        idx = S.select(recs, self.NAMES, ("HoverPayOut",))
        chosen = range(2, 7)                                     # first to last period with a pay-out of 7 m or more
        expected = math.sqrt(np.mean([(1000.0*i)**2 for i in chosen]))/1e3
        self.assertAlmostEqual(S.tension_control_rms_kn(recs, idx), expected, places=12)

    def test_perched_position(self):
        modes = [12]*5
        gain = [1.0, 0.5, 1e-9, 0.0, 0.0]
        xg = [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [1.0, 7.0, -4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        recs = self.records(modes, gain_ramp_scale=gain, Xg=xg, platform_azi=math.pi/2)
        # Platform azimuth 90 deg: x_p = y_g, y_p = -x_g.
        error = S.perched_position_error_m(recs, self.NAMES, [7.0, -1.0, 0.0])
        self.assertAlmostEqual(error, 0.0, places=12)


@unittest.skipUnless(BUILD, "OPENFAWE_MAKANI_FULL_BUILD not set")
class RunWiring(unittest.TestCase):
    # The first 13 s of the flight (review P3C-05): the records follow K3 (wiring_checks); the rotors stay at rest
    # until the first period after one whose output has run_motors and not stop_motors, then speed up by at most
    # 5 rad/s per period from zero; the controller label lags the planner's by one period at every switch.
    def test_first_seconds(self):
        from openfawe.flight.makani_full import load_full_api
        api = load_full_api(BUILD)
        params = G.load_gs02_params(PARAMS)
        discs, inertia = rotors()
        result = T.run_turnkey(params, api, database(), discs, inertia, max_periods=1300)
        records = result["records"]
        self.assertEqual(result["end"]["reason"], "max_periods")
        self.assertEqual(len(records), 1300)
        self.assertEqual(S.wiring_checks(records, result["settings"]), [])
        first_run = next(i for i, r in enumerate(records) if r["run_motors"] and not r["stop_motors"])
        self.assertGreaterEqual(first_run, T.THROTTLE_CROSSWIND_PERIOD)
        for r in records[:first_run + 1]:
            self.assertEqual(r["rotor_omega_rad_s"], [0.0]*8)
        self.assertGreater(max(records[first_run + 1]["rotor_omega_rad_s"]), 0.0)
        self.assertLessEqual(max(records[first_run + 1]["rotor_omega_rad_s"]), 5.0 + 1e-12)
        for a, b in zip(records, records[1:]):
            self.assertEqual(b["controller_flight_mode"], a["flight_mode"])
        self.assertEqual(records[0]["controller_flight_mode"], int(api.kFlightModePilotHover))
        # A changed record is caught.
        broken = [dict(r) for r in records]
        broken[1200]["rotor_omega_rad_s"] = [w + 1e-9 for w in broken[1200]["rotor_omega_rad_s"]]
        self.assertTrue(S.wiring_checks(broken, result["settings"]))


@unittest.skipUnless(BUILD, "OPENFAWE_MAKANI_FULL_BUILD not set")
class EstimateAcceleration(unittest.TestCase):
    # K4: Ab_f is the kinematic acceleration in body axes (dcm_g2b Ag); acc_norm_f the norm of the specific force.
    def test_ab_f(self):
        from openfawe.flight.makani_full import load_full_api
        from openfawe.flight.makani_interface import G_TO_WORLD, makani_truth, origin_acceleration
        api = load_full_api(BUILD)
        params = G.load_gs02_params(PARAMS)
        system = P.PerchedSystem(params)
        state = moved(system.initial_wing_state(), [0.1, 0.2, -0.3], [0.2, -0.1, 0.3], [1.0, -2.0, 0.5])
        acc, spin = np.array([0.5, -1.5, 2.0]), np.array([0.1, 0.3, -0.2])
        est = P.kite_estimate(api, system, state, acc, spin, np.array([10.0, 20.0, 3000.0]), P.makani_wind_g(params))
        truth = makani_truth(system.body, state, G_TO_WORLD@P.makani_wind_g(params),
                             acceleration_origin_m_s2=origin_acceleration(system.body, state, acc, spin))
        ab = np.array([est.Ab_f.x, est.Ab_f.y, est.Ab_f.z])
        np.testing.assert_allclose(ab, truth.dcm_g2b@truth.Ag, rtol=0, atol=1e-12)
        g_g = np.array([0.0, 0.0, 9.81])
        self.assertAlmostEqual(est.acc_norm_f, float(np.linalg.norm(truth.Ag - g_g)), places=12)
        # At rest: the kinematic acceleration is zero and the specific force is g.
        rest = P.kite_estimate(api, system, system.initial_wing_state(), np.zeros(3), np.zeros(3), np.zeros(3),
                               P.makani_wind_g(params))
        self.assertEqual([rest.Ab_f.x, rest.Ab_f.y, rest.Ab_f.z], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(rest.acc_norm_f, 9.81, places=12)


if __name__ == "__main__":
    unittest.main()
