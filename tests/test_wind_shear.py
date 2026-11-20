"""Plan 3.9b (power-energy contract revision 1 addendum 5, E5b): ``ground.wind_shear`` and its wiring in
``ground.turnkey`` and ``ground.perch.kite_estimate``.

Without the library: the wind at several heights is ``calc_wind_shear`` times the uniform wind's direction; half the
curl is the hand-computed forward difference of 10 m; the body rates are Makani's ``dcm_g2b wind_omega_g`` in the plant's
body axes and reach the database only through the lookup's angular velocity; every tether node takes the wind at its own
height, in ``FlyingSystem``'s tether evaluation too; the ground wind sensor is 21.022 m above ground; the exponent 0 gives
the uniform wind bit for bit; the options are refused where they are not defined. With the library
(OPENFAWE_MAKANI_FULL_BUILD; a skip fails the judged run): the estimate's ground wind is the sensor's and its wind
aloft and apparent wind the kite's; TurnKey with the exponent 0 through the shear path gives the records of TurnKey
without it (power system and blown wing on, the first 1200 periods), and with 1/7 different ones.
"""
import json
import math
import os
from pathlib import Path
import struct
import unittest

import numpy as np

from openfawe.flight.makani_interface import G_TO_WORLD, MAKANI_BODY_TO_BODY
from openfawe.flight.rigid_body import RigidState
from openfawe.ground import gs02 as G
from openfawe.ground import perch as P
from openfawe.ground import turnkey as T
from openfawe.ground.makani_wind import calc_wind_shear
from openfawe.ground.wind_shear import FINITE_DIFFERENCE_STEP_M, ShearWind

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT/"cases"/"ground"/"makani_gs02_parker_ranch.json"
DATA = ROOT/"cases"/"polars"/"makani"
BUILD = os.environ.get("OPENFAWE_MAKANI_FULL_BUILD")
ALPHAS = (0.0, 1.0/7.0, 0.2)


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


def bits(values):
    """Bit patterns (+0 and -0 differ): the contract's "bit for bit"."""
    return [struct.pack("<d", float(v)) for v in np.ravel(np.asarray(values, dtype=float))]


def rotation(rotation_vector):
    from openfawe.flight.kinematics import rotation_from_vector
    return rotation_from_vector(np.asarray(rotation_vector, dtype=float))


class Wind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = G.load_gs02_params(PARAMS)
        phys = cls.params.raw["phys_sim"]
        cls.speed, cls.direction = float(phys["wind_speed"]), float(phys["wind_direction"])
        cls.unit = np.array([-math.cos(cls.direction), -math.sin(cls.direction), 0.0])
        cls.shears = {a: ShearWind.from_params(cls.params, a, ground_z_g_m=T.GROUND_Z_G) for a in ALPHAS}

    def test_wind_at_heights(self):
        # Addendum 5 item 1: speed CalcWindShear(ground_z - z_g, 21, 10, alpha), the uniform wind's direction.
        self.assertEqual((self.speed, self.shears[0.2].ref_height_agl_m), (10.0, 21.0))
        for alpha, shear in self.shears.items():
            for z_g in (-400.0, -150.0, -51.3, -14.878, -14.9, -4.0, 0.0, 4.0, 6.122, 9.0):
                h = T.GROUND_Z_G - z_g
                expected = calc_wind_shear(h, 21.0, 10.0, alpha)*self.unit
                got = shear.wind_g((12.5, -7.25, z_g))
                self.assertEqual(got.tolist(), expected.tolist())
                self.assertEqual(shear.wind_world_at(G_TO_WORLD@np.array([12.5, -7.25, z_g])).tolist(),
                                 (G_TO_WORLD@expected).tolist())
                speed = 10.0*max(h/21.0, 0.1)**alpha                      # independent of the transcription
                self.assertAlmostEqual(float(np.linalg.norm(got)), speed, delta=1e-14*speed)
                np.testing.assert_allclose(got/np.linalg.norm(got), self.unit, rtol=0, atol=1e-15)
        # 21 m above ground: the reference speed; 2.1 m and below: the floor of the ratio.
        self.assertEqual(self.shears[1.0/7.0].speed_at(21.0), 10.0)
        self.assertEqual(self.shears[0.2].speed_at(-3.0), self.shears[0.2].speed_at(2.1))
        self.assertGreater(self.shears[0.2].speed_at(150.0), self.shears[1.0/7.0].speed_at(150.0))
        self.assertGreater(self.shears[1.0/7.0].speed_at(150.0), 10.0)

    def test_exponent_zero_is_the_uniform_wind(self):
        uniform = P.makani_wind_g(self.params)
        shear = self.shears[0.0]
        for z_g in (-300.0, -20.0, 0.0, 30.0):
            self.assertEqual(bits(shear.wind_g((1.0, 2.0, z_g))), bits(uniform))
            self.assertEqual(bits(shear.wind_omega_g((1.0, 2.0, z_g))), bits([0.0, 0.0, 0.0]))
        self.assertEqual(bits(shear.sensor_wind_g()), bits(uniform))
        nodes = np.array([[3.0, -1.0, h] for h in (-6.0, 0.0, 15.0, 80.0, 210.0)])
        self.assertEqual(bits(shear.node_winds_world(nodes)), bits(np.tile(G_TO_WORLD@uniform, (5, 1))))

    def test_half_curl_by_forward_differences(self):
        # environment.cc:122-158 by hand: only the z step changes the height (z + 10 m in g is 10 m lower).
        step = FINITE_DIFFERENCE_STEP_M
        self.assertEqual(step, 10.0)
        for alpha in (1.0/7.0, 0.2):
            shear = self.shears[alpha]
            for pos in ((30.0, -20.0, -120.0), (-4.0, 2.0, -12.0), (0.0, 0.0, 1.0)):
                h = T.GROUND_Z_G - pos[2]
                w = calc_wind_shear(h, 21.0, 10.0, alpha)*self.unit
                below = calc_wind_shear(T.GROUND_Z_G - (pos[2] + step), 21.0, 10.0, alpha)*self.unit
                zero = (w[2] - w[2])/step
                dudz, dvdz = (below[0] - w[0])/step, (below[1] - w[1])/step
                expected = [0.5*(zero - dvdz), 0.5*(dudz - zero), 0.5*((w[1] - w[1])/step - (w[0] - w[0])/step)]
                got = shear.wind_omega_g(pos)
                self.assertEqual(got.tolist(), expected)
                self.assertEqual(got[2], 0.0)
                # Sense (NED, z down): 0.5 curl of U(h) u is 0.5 U'(h) (u_y, -u_x, 0), so omega x u points down
                # (+z) with the size 0.5 U' (here the difference over the 10 m below), positive as the wind grows.
                size = 0.5*(calc_wind_shear(h, 21.0, 10.0, alpha) - calc_wind_shear(h - step, 21.0, 10.0, alpha))/step
                self.assertGreater(size, 0.0)
                self.assertAlmostEqual(float(np.linalg.norm(got)), size, delta=1e-15)
                self.assertAlmostEqual(float(np.cross(got, self.unit)[2]), size, delta=1e-12*size)

    def test_body_rates(self):
        # wing.cc:555: dcm_g2b wind_omega_g in Makani's body axes, with Makani's dcm_g2b of the state (makani_truth, the
        # estimate's attitude); the plant's body axes differ from Makani's by MAKANI_BODY_TO_BODY.
        from openfawe.flight.makani_interface import makani_truth
        shear = self.shears[0.2]
        body = P.PerchedSystem(self.params).body
        state = RigidState(0.0, np.array([40.0, -25.0, 130.0]), rotation([0.4, -1.1, 0.7]), np.array([20.0, 5.0, -3.0]),
                           np.array([0.1, 0.2, -0.3]))
        pose = state.pose(body)
        truth = makani_truth(body, state, np.zeros(3), acceleration_origin_m_s2=np.zeros(3))
        omega_g = shear.wind_omega_g(np.asarray(truth.Xg, dtype=float))
        makani_b = np.asarray(truth.dcm_g2b, dtype=float)@omega_g
        np.testing.assert_allclose(shear.wind_rates_body(pose), MAKANI_BODY_TO_BODY@makani_b, rtol=0, atol=1e-17)
        self.assertGreater(float(np.linalg.norm(omega_g)), 1e-3)

    def test_rates_reach_only_the_lookup(self):
        # reference_loads_body: the rates are added to the body's angular velocity for the database only.
        from openfawe.flight.reference_flight import reference_loads_body
        aero = database()
        body = P.PerchedSystem(self.params).body
        state = RigidState(0.0, np.array([10.0, -5.0, 120.0]), rotation([0.1, -0.2, 0.3]), np.array([30.0, 40.0, -5.0]),
                           np.array([0.2, -0.1, 0.4]))
        wind = G_TO_WORLD@self.shears[0.2].wind_g((10.0, 5.0, -120.0))
        rates = np.array([0.01, -0.02, 0.005])
        flaps = np.zeros(8)
        plain = reference_loads_body(aero, body, state, wind, 1.075, flaps, 0.0)
        added = reference_loads_body(aero, body, state, wind, 1.075, flaps, 0.0, wind_rates_body=rates)
        pose = state.pose(body)
        through_air = pose.rotation.T@(pose.velocity_m_s - wind)
        force, moment, condition = aero.loads(through_air, state.angular_velocity_body_rad_s + rates, 1.075, flaps, 0.0)
        self.assertEqual(added[0].force_body_N.tolist(), force.tolist())
        self.assertEqual(added[1].tolist(), moment.tolist())
        self.assertEqual(added[2]["airspeed_m_s"], plain[2]["airspeed_m_s"])
        self.assertEqual(added[2]["alpha_rad"], plain[2]["alpha_rad"])
        self.assertNotEqual(added[0].moment_cm_body_Nm.tolist(), plain[0].moment_cm_body_Nm.tolist())

    def test_tether_nodes_each_at_their_height(self):
        shear = self.shears[1.0/7.0]
        nodes = np.array([[3.0, -1.0, z] for z in (-8.0, -6.122, 0.0, 14.9, 60.0, 140.0, 300.0)])
        each = shear.node_winds_world(nodes)
        for i, p in enumerate(nodes):
            self.assertEqual(each[i].tolist(), shear.wind_world_at(p).tolist())
        speeds = np.linalg.norm(each, axis=1)
        self.assertEqual(speeds[0], speeds[1])                          # both below 2.1 m: the floor
        self.assertTrue(np.all(np.diff(speeds[1:]) > 0.0))

    def test_flying_system_passes_the_node_winds(self):
        discs, inertia = rotors()
        for alpha in (None, 0.2):
            shear = None if alpha is None else self.shears[alpha]
            flying = T.FlyingSystem(self.params, discs, inertia, G_TO_WORLD@P.makani_wind_g(self.params), wind_shear=shear)
            init, flying = P.perched_initialisation(self.params, flying)
            state, nodes, speeds = init["wing_state"], init["nodes"], init["node_velocities"]
            nodes = nodes + np.array([0.0, 0.0, 1.0])*np.arange(len(nodes))[:, None]       # distinct heights
            seen = []
            original = flying.line.derivatives

            def spy(*args, **kwargs):
                seen.append(np.array(kwargs["wind"], dtype=float))
                return original(*args, **kwargs)
            flying.line.derivatives = spy
            flying.tether_evaluation(state, nodes, speeds, flying.gs.x)
            self.assertEqual(len(seen), 1)
            expected = flying.node_wind if shear is None else shear.node_winds_world(nodes)
            self.assertEqual(seen[0].tolist(), np.asarray(expected).tolist())
            if shear is not None:
                self.assertEqual(len(set(np.linalg.norm(seen[0], axis=1).tolist())), len(nodes))

    def test_ground_wind_sensor(self):
        # config/base_station/wind_sensor.py: (2.32, 0.026, -14.9) m in the platform frame, 21.022 m above ground.
        for alpha, shear in self.shears.items():
            self.assertEqual(shear.sensor_height_agl_m(), T.GROUND_Z_G + 14.9)
            self.assertAlmostEqual(shear.sensor_height_agl_m(), 21.022, places=12)
            self.assertEqual(shear.sensor_wind_g().tolist(), (calc_wind_shear(T.GROUND_Z_G + 14.9, 21.0, 10.0, alpha)*self.unit).tolist())
        self.assertAlmostEqual(float(np.linalg.norm(self.shears[0.2].sensor_wind_g())), 10.0*(21.022/21.0)**0.2, delta=1e-13)

    def test_refused(self):
        with self.assertRaises(ValueError):
            ShearWind(10.0, 0.0, 0.2, ref_height_agl_m=0.0, ground_z_g_m=T.GROUND_Z_G)
        with self.assertRaises(ValueError):
            ShearWind(10.0, 0.0, float("nan"), ground_z_g_m=T.GROUND_Z_G)
        with self.assertRaises(ValueError):
            ShearWind(0.0, 0.0, 0.2, ground_z_g_m=T.GROUND_Z_G)
        discs, inertia = rotors()
        good = self.shears[0.2]
        for kwargs in (dict(wind_shear=good, segment=object()), dict(wind_shear=0.2),
                       dict(wind_shear=ShearWind(10.0, self.direction, 0.2, ground_z_g_m=6.0)),
                       dict(wind_shear=ShearWind(9.0, self.direction, 0.2, ground_z_g_m=T.GROUND_Z_G)),
                       dict(wind_shear=ShearWind(10.0, self.direction + 0.1, 0.2, ground_z_g_m=T.GROUND_Z_G))):
            with self.assertRaises(ValueError):
                T.run_turnkey(self.params, None, None, discs, inertia, **kwargs)

    def test_settings(self):
        s = self.shears[1.0/7.0].settings()
        self.assertEqual((s["exponent"], s["ref_height_agl_m"], s["speed_at_ref_m_s"], s["finite_difference_step_m"]),
                         (1.0/7.0, 21.0, 10.0, 10.0))
        self.assertEqual(s["sensor_height_agl_m"], T.GROUND_Z_G + 14.9)
        json.dumps(s)


@unittest.skipUnless(BUILD, "OPENFAWE_MAKANI_FULL_BUILD not set")
class WithController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openfawe.flight.makani_full import load_full_api
        cls.api = load_full_api(BUILD)
        cls.params = G.load_gs02_params(PARAMS)

    def test_estimate_winds(self):
        # The ground wind is the sensor's; the wind aloft and the apparent wind are the kite's (Pitot solution).
        from openfawe.flight.makani_interface import makani_truth, origin_acceleration, wind_direction
        system = P.PerchedSystem(self.params)
        state = RigidState(0.0, np.array([10.0, -5.0, 120.0]), rotation([0.1, -0.2, 0.3]), np.array([30.0, 40.0, -5.0]),
                           np.array([0.2, -0.1, 0.4]))
        acc, spin, force = np.array([0.5, -1.5, 2.0]), np.array([0.1, 0.3, -0.2]), np.array([10.0, 20.0, 3000.0])
        kite, sensor = np.array([-8.0, -9.0, 0.0]), np.array([-7.0, -7.5, 0.0])
        est = P.kite_estimate(self.api, system, state, acc, spin, force, kite, sensor_wind_g=sensor)
        plain = P.kite_estimate(self.api, system, state, acc, spin, force, kite)
        offset = float(self.api.GetSystemParams().contents.ground_station.azi_ref_offset)
        vec = lambda v: [v.x, v.y, v.z]                                  # noqa: E731
        for name in ("vector", "vector_f", "vector_f_slow"):
            self.assertEqual(vec(getattr(est.wind_g, name)), sensor.tolist())
            self.assertEqual(vec(getattr(est.wind_aloft_g, name)), kite.tolist())
            self.assertEqual(vec(getattr(plain.wind_g, name)), kite.tolist())
        self.assertEqual(est.wind_g.dir_f, wind_direction(sensor, offset))
        self.assertEqual(est.wind_g.speed_f, float(np.linalg.norm(sensor)))
        self.assertEqual(est.wind_aloft_g.speed_f_playbook, float(np.linalg.norm(kite)))
        self.assertEqual((est.wind_aloft_g.dir_f, est.wind_aloft_g.speed_f), (plain.wind_aloft_g.dir_f, plain.wind_aloft_g.speed_f))
        self.assertEqual(vec(est.apparent_wind.vector), vec(plain.apparent_wind.vector))
        truth = makani_truth(system.body, state, G_TO_WORLD@kite, acceleration_origin_m_s2=origin_acceleration(system.body, state, acc, spin))
        self.assertEqual(vec(est.apparent_wind.vector), [float(v) for v in truth.apparent_wind_b])

    def test_turnkey_exponent_zero_and_one_seventh(self):
        # E5a in small: the exponent 0 through the shear path gives the records of TurnKey without the shear (power
        # system and blown wing on, through the first rotor speed-up); 1/7 changes them. One child process per flight,
        # as the judged runs (the controller library keeps its state within a process).
        import sys
        import tempfile
        sys.path.insert(0, str(ROOT/"docs"/"tasks"/"OF-0008-g5control20261006"))
        import turnkey_checks as tk
        lines, finals = {}, {}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for name, alpha, periods in (("none", None, 1200), ("zero", 0.0, 1200), ("seventh", 1.0/7.0, 300)):
                receipt = tk.run_flight(out, name, BUILD, blown=True, max_periods=periods, trial=True, power=True, shear=alpha)
                self.assertEqual(receipt["returncode"], 0, receipt["log_tail"])
                finals[name] = receipt["final"]
                lines[name] = (out/f"{name}-records.jsonl").read_text(encoding="utf-8").splitlines()
        for name, periods in (("none", 1200), ("zero", 1200), ("seventh", 300)):
            self.assertEqual(finals[name]["end"]["reason"], "max_periods")
            self.assertEqual(len(lines[name]), periods)
        first = next((i for i, (a, b) in enumerate(zip(lines["zero"], lines["none"])) if a != b), None)
        self.assertIsNone(first, f"the first differing period is {first}")
        self.assertNotIn("wind_shear", finals["none"]["settings"])
        self.assertEqual(finals["zero"]["settings"]["wind_shear"]["exponent"], 0.0)
        self.assertEqual({k: v for k, v in finals["zero"]["settings"].items() if k != "wind_shear"}, finals["none"]["settings"])
        self.assertEqual(finals["seventh"]["settings"]["wind_shear"]["exponent"], 1.0/7.0)
        self.assertNotEqual(lines["seventh"], lines["none"][:300])
        # The wiring (addendum 8 item 4): in the 1/7 flight the recorded airspeed and aerodynamic loads and the rotors'
        # axial freestream (the power system's too) are those of the wind at the body origin of the period's start, with
        # the wind-induced rates in the lookup.
        from types import SimpleNamespace
        from openfawe.flight.reference_flight import reference_loads_body
        discs, _ = rotors()
        aero, body = database(), P.PerchedSystem(self.params).body
        density = float(self.params.raw["phys_sim"]["air_density"])
        shear = ShearWind.from_params(self.params, 1.0/7.0, ground_z_g_m=T.GROUND_Z_G)
        uniform = G_TO_WORLD@P.makani_wind_g(self.params)
        differs = 0
        for k in (0, 120, 299):
            r = json.loads(lines["seventh"][k])
            state = RigidState(r["t"], np.array(r["position_cm_m"]), np.array(r["rotation"]), np.array(r["velocity_cm_m_s"]),
                               np.array(r["angular_velocity_body_rad_s"]))
            pose = state.pose(body)
            wind = G_TO_WORLD@shear.origin_wind_g(pose)
            held, moment, condition = reference_loads_body(aero, body, state, wind, density, np.array(r["flaps_acting_rad"]),
                                                           r["thrust_coeff_used"], wind_rates_body=shear.wind_rates_body(pose))
            self.assertEqual(bits(held.force_body_N), bits(r["aero_force_body_N"]), k)
            self.assertEqual(bits(moment), bits(r["aero_moment_origin_body_Nm"]), k)
            self.assertEqual(condition["airspeed_m_s"], r["airspeed_m_s"], k)
            without = reference_loads_body(aero, body, state, wind, density, np.array(r["flaps_acting_rad"]), r["thrust_coeff_used"])
            self.assertNotEqual(bits(without[1]), bits(r["aero_moment_origin_body_Nm"]), k)
            freestreams = T.rotor_freestreams(SimpleNamespace(body=body, discs=discs, wind=wind), state)
            self.assertEqual(bits(freestreams), bits(r["rotor_axial_freestream_m_s"]), k)
            self.assertEqual(bits(r["power_freestream_m_s"]), bits(r["rotor_axial_freestream_m_s"]), k)
            differs += int(bits(T.rotor_freestreams(SimpleNamespace(body=body, discs=discs, wind=uniform), state)) != bits(freestreams))
        self.assertGreater(differs, 0)

if __name__ == "__main__":
    unittest.main()
