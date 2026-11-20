"""Reference plant of the G5 task (reference-plant contract): Makani's aerodynamic databases and the driver that flies on them."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero.makani_database import (COEFFICIENTS, DatabaseSettings, MakaniDatabase, blend_coefficient,
                                           dcm_wind_to_body)
from openfawe.aero.makani_flap_increments import FlapIncrements
from openfawe.flight.kinematics import rotation_from_vector
from openfawe.flight.plant import HeldLoads, plant_step
from openfawe.flight.reference_flight import run_reference
from openfawe.flight.residual import RigidBody
from openfawe.flight.rigid_body import RigidState

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"cases/polars/makani"
LOW, SPOILERS, HIGH = DATA/"m600_aswing_baseline.json", DATA/"spoiler_offset_tables.json", DATA/"hover_model_sn02_wind_10_no_wake_model.json"
BODY = RigidBody(1700., [0.08, 0.01, -0.13], np.array([[32000., -20., 35.], [-20., 9700., -15.], [35., -15., 38700.]]))
ZERO = np.zeros(8)


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(LOW.read_text())
        cls.full = MakaniDatabase.from_files(LOW, spoiler_path=SPOILERS, high_path=HIGH)
        cls.bare = MakaniDatabase.from_files(LOW, settings=DatabaseSettings().without_offsets())
        cls.rest = cls.full.omega_hat_0                       # the rates at which the rate terms vanish

    def test_wind_axes_are_those_of_the_source(self):
        alpha, beta = 0.31, -0.17
        dcm = dcm_wind_to_body(alpha, beta)
        r1, r2 = -beta, alpha                                # AngleToDcm(-beta, alpha, 0, ZYX), common/c_math/geometry.c lines 226-234
        expected = [[math.cos(r1)*math.cos(r2), math.sin(r1)*math.cos(r2), -math.sin(r2)], [-math.sin(r1), math.cos(r1), 0.0],
                    [math.cos(r1)*math.sin(r2), math.sin(r1)*math.sin(r2), math.cos(r2)]]
        assert_allclose(dcm, expected, rtol=0, atol=1e-16)
        velocity = dcm@[50.0, 0.0, 0.0]                      # the velocity through the air is along the wind x axis
        self.assertAlmostEqual(math.atan2(velocity[2], velocity[0]), alpha, places=15)
        self.assertAlmostEqual(math.asin(velocity[1]/50.0), beta, places=15)

    def test_the_bare_tables_are_returned_at_their_nodes(self):
        for i, alpha in enumerate(self.bare.alphas_deg):
            for j, beta in enumerate(self.bare.betas_deg):
                got = self.bare.low_coefficients(math.radians(alpha), math.radians(beta), self.rest, ZERO)
                expected = [self.raw[f"{name}tot"][0][i][j] for name in COEFFICIENTS]
                assert_allclose(got, expected, rtol=1e-13, atol=1e-15)

    def test_the_database_passes_the_sources_own_validity_check(self):
        # avl_aero_database.cc lines 281-296: the checks are made at zero incidence and sideslip, in radians.
        def at(alpha, beta):
            return self.full.low_coefficients(alpha, beta, self.rest, ZERO)
        self.assertLess(at(0.1, 0.0)[2], at(0.0, 0.0)[2])       # CZ falls with the incidence
        self.assertLess(at(0.0, 0.1)[1], at(0.0, 0.0)[1])       # CY falls with the sideslip
        table = self.full.flap_increments.derivatives_at(0.0, 0.0)
        self.assertTrue(table[0, 3] >= 0 and table[1, 3] >= 0 and table[4, 3] <= 0 and table[5, 3] <= 0 and table[6, 4] <= 0 and table[7, 5] <= 0)

    def test_the_coefficient_offsets_act_in_wind_axes_at_every_node(self):
        for alpha in self.full.alphas_deg:
            for beta in self.full.betas_deg:
                a, b = math.radians(alpha), math.radians(beta)
                dcm = dcm_wind_to_body(a, b)
                with_offsets, without = (model.low_coefficients(a, b, self.rest, ZERO) for model in (self.full, self.bare))
                assert_allclose(dcm.T@with_offsets[:3]-dcm.T@without[:3], [-0.075, 0.0, 0.125], rtol=0, atol=1e-12)
                assert_allclose(with_offsets[3:]-without[3:], [0.01, 0.0, 0.0], rtol=0, atol=1e-12)

    def test_the_rate_and_flap_terms_add_as_in_the_source(self):
        i, j = 5, 6                                             # a node: alpha 3 deg, beta 3 deg
        alpha, beta = (math.radians(grid[k]) for grid, k in ((self.bare.alphas_deg, i), (self.bare.betas_deg, j)))
        omega_hat = np.array([0.02, -0.004, -0.09])
        flaps = np.radians([2.0, -3.0, 0.0, 0.0, 1.0, -2.0, -4.0, 6.0])     # A4, A5 at zero: no spoilers
        thrust = -0.12
        d = omega_hat-np.array(self.raw["omega_hat"])
        for k, name in enumerate(COEFFICIENTS):
            expected = self.raw[f"{name}tot"][0][i][j]+sum(self.raw[f"{name}{rate}"][0][i][j]*d[m] for m, rate in enumerate("pqr"))
            for flap in range(8):
                degrees = math.degrees(flaps[flap])-(self.raw["de1"][0][i][j] if flap == 6 else 0.0)
                expected += self.raw[f"{name}d{flap+1}"][0][i][j]*degrees*((1.0+thrust) if flap in (6, 7) else 1.0)
            self.assertAlmostEqual(self.bare.low_coefficients(alpha, beta, omega_hat, flaps, thrust)[k], expected, delta=1e-12*max(1.0, abs(expected)))

    def test_the_flap_terms_are_those_of_the_increment_module(self):
        increments = FlapIncrements.from_makani_json(LOW)
        flaps = np.radians([2.0, -3.0, 0.0, 0.0, 1.0, -2.0, -4.0, 16.0])
        for alpha, beta in ((0.03, 0.01), (-0.1, 0.2), (0.08, -0.05)):
            delta = self.full.low_coefficients(alpha, beta, self.rest, flaps, 0.2)-self.full.low_coefficients(alpha, beta, self.rest, ZERO, 0.2)
            force, moment = increments.coefficients(alpha, beta, flaps, 0.2)
            assert_allclose(delta, np.concatenate([force, moment]), rtol=1e-12, atol=1e-15)

    def test_the_blend_weight(self):
        deg = math.radians
        for alpha, expected in ((5.0, 0.0), (6.0, 0.1), (10.0, 0.5), (15.0, 1.0), (-15.0, 0.0), (-16.0, 0.1), (0.0, 0.0), (40.0, 1.0)):
            self.assertAlmostEqual(self.full.blend_weight(deg(alpha), 0.0), expected, places=12)
        for beta, expected in ((15.0, 0.0), (16.0, 0.1), (20.0, 0.5), (-16.0, 0.1)):
            self.assertAlmostEqual(self.full.blend_weight(0.0, deg(beta)), expected, places=12)
        self.assertAlmostEqual(self.full.blend_weight(deg(6.0), deg(16.0)), 1.0-0.9*0.9, places=12)
        self.assertAlmostEqual(blend_coefficient(deg(7.0), deg(-20.0), deg(10.0), deg(10.0)), 0.2, places=12)

    def test_between_the_stall_angles_the_two_databases_are_mixed_by_the_weight(self):
        alpha, beta = math.radians(8.0), math.radians(2.0)
        omega_hat, flaps = np.array([0.01, 0.002, -0.1]), np.radians([1.0, 1.0, -40.0, -40.0, -2.0, -2.0, 3.0, -4.0])
        self.assertGreater(np.abs(self.full.spoiler_coefficients(alpha, flaps)).max(), 1e-6)    # they would show if they were added
        force, moment, info = self.full.coefficients(alpha, beta, omega_hat, flaps, 0.1)
        weight = self.full.blend_weight(alpha, beta)
        self.assertAlmostEqual(weight, 0.3, places=12)
        expected = (weight*self.full.high_coefficients(alpha, beta, omega_hat, flaps)
                    + (1.0-weight)*self.full.low_coefficients(alpha, beta, omega_hat, flaps, 0.1))
        assert_allclose(np.concatenate([force, moment]), expected, rtol=1e-13, atol=0)
        self.assertFalse(info["spoilers"])                      # the spoiler offsets are for the low-incidence database alone
        high_only = self.full.coefficients(math.radians(20.0), 0.0, omega_hat, flaps)
        assert_allclose(np.concatenate(high_only[:2]), self.full.high_coefficients(math.radians(20.0), 0.0, omega_hat, flaps), rtol=0, atol=0)

    def test_spoiler_offsets(self):
        raw = json.loads(SPOILERS.read_text())
        self.assertEqual(raw["flap_list"], [0, 0, 1, 1, 0, 0, 0, 0])               # A4 and A5
        for i, alpha in enumerate(raw["alphads"]):
            for k, delta in enumerate(raw["deltads"]):
                flaps = ZERO.copy()
                flaps[3] = math.radians(delta)                                      # A5 alone
                got = self.full.spoiler_coefficients(math.radians(alpha), flaps)
                expected = [raw[f"{name}_offset"][1][i][k]+raw[f"{name}_offset"][0][i][-1] for name in COEFFICIENTS]
                assert_allclose(got, expected, rtol=1e-12, atol=1e-15)
        assert_allclose(self.full.spoiler_coefficients(0.02, ZERO), 0.0, atol=0)   # nothing at zero deflection
        flaps = ZERO.copy()
        flaps[2] = flaps[3] = math.radians(-40.0)
        with_spoilers = self.full.coefficients(0.03, 0.0, self.rest, flaps)
        self.assertTrue(with_spoilers[2]["spoilers"])
        low = self.full.low_coefficients(0.03, 0.0, self.rest, flaps)
        assert_allclose(np.concatenate(with_spoilers[:2])-low, self.full.spoiler_coefficients(0.03, flaps), rtol=1e-12, atol=1e-15)

    def test_the_high_incidence_tables_are_read_in_the_sources_layout(self):
        raw = json.loads(HIGH.read_text())
        na, nb = raw["num_alphas"], raw["num_betas"]
        i, j = 13, 7                                            # a node of the two angles
        picks = [1, 3, 0, 4, 2, 1, 20, 3]                       # a node of every flap grid
        flaps = np.array([raw[f"delta{flap+1}s"][k] for flap, k in enumerate(picks)])
        omega_hat = np.array([0.03, -0.002, 0.05])              # below the roll rate of the damping limit
        expected = np.zeros(6)
        for c in range(6):
            flat = c*na*nb+i*nb+j
            expected[c] = raw["cfm"][flat]+sum(raw[f"dcfm_d{rate}"][flat]*omega_hat[m] for m, rate in enumerate("pqr"))
            for flap, k in enumerate(picks):
                nd = len(raw[f"delta{flap+1}s"])
                index = c*na*nb*nd+i*nb*nd+j*nd+k
                expected[c] += raw[f"dcfm{flap+1}"][index]+sum(raw[f"dcfm{flap+1}_d{rate}"][index]*omega_hat[m] for m, rate in enumerate("pqr"))
        got = self.full.high_coefficients(raw["alphas"][i], raw["betas"][j], omega_hat, flaps)
        assert_allclose(got, expected, rtol=1e-12, atol=1e-14)

    def test_the_high_incidence_tables_between_their_nodes_and_outside_their_grids(self):
        from scipy.interpolate import RegularGridInterpolator
        h, rng = self.full.high, np.random.default_rng(7)
        on_nodes = np.array([grid[int(np.argmin(np.abs(grid)))] for grid in h["deltas"]])
        base = RegularGridInterpolator((h["alphas"], h["betas"]), np.moveaxis(h["cfm"], 0, -1))
        for _ in range(50):
            alpha, beta = rng.uniform(h["alphas"][0], h["alphas"][-1]), rng.uniform(h["betas"][0], h["betas"][-1])
            delta = rng.uniform(h["deltas"][7][0], h["deltas"][7][-1])
            flaps = on_nodes.copy()
            expected = base([[alpha, beta]])[0]
            for f in range(8):
                table = RegularGridInterpolator((h["alphas"], h["betas"], h["deltas"][f]), np.moveaxis(h["flap_cfm"][f], 0, -1))
                expected = expected+table([[alpha, beta, delta if f == 7 else on_nodes[f]]])[0]
            flaps[7] = delta
            assert_allclose(self.full.high_coefficients(alpha, beta, np.zeros(3), flaps), expected, rtol=1e-11, atol=1e-13)
        corner = self.full.high_coefficients(h["alphas"][-1], h["betas"][0], np.zeros(3), on_nodes)
        beyond = on_nodes.copy()
        beyond[0] = h["deltas"][0][-1]+0.4
        held = on_nodes.copy()
        held[0] = h["deltas"][0][-1]
        assert_allclose(self.full.high_coefficients(h["alphas"][-1]+0.3, h["betas"][0]-0.2, np.zeros(3), on_nodes), corner, rtol=0, atol=0)
        assert_allclose(self.full.high_coefficients(0.4, 0.1, np.zeros(3), beyond), self.full.high_coefficients(0.4, 0.1, np.zeros(3), held), rtol=0, atol=0)

    def test_the_roll_damping_limit_of_the_high_incidence_database(self):
        raw = json.loads(HIGH.read_text())
        alpha, beta = raw["alphas"][13], raw["betas"][5]          # where the tabulated roll damping is +3.8
        flaps = np.array([raw[f"delta{flap+1}s"][-1] for flap in range(8)])
        slow, fast = (self.full.high_coefficients(alpha, beta, [p, 0.0, 0.0], flaps) for p in (0.1, 0.2))
        rest = self.full.high_coefficients(alpha, beta, [0.0, 0.0, 0.0], flaps)
        damping = (slow[3]-rest[3])/0.1                        # the derivative without the flap parts' share of the limit
        base = self.full.high["rates"][0][3, 13, 5]
        self.assertGreater(base, -0.2)                         # otherwise the limit would change nothing here
        flap_part = damping-base
        self.assertAlmostEqual(fast[3]-rest[3], 0.2*(min(base, -0.2)+flap_part), places=12)
        middle = self.full.high_coefficients(alpha, beta, [0.125, 0.0, 0.0], flaps)       # half way through the fade
        self.assertAlmostEqual(middle[3]-rest[3], 0.125*(0.5*base+0.5*min(base, -0.2)+flap_part), places=12)

    def test_loads_are_dimensional_at_the_origin_in_the_openfawe_body_axes(self):
        through_air, omega = np.array([-40.0, 1.5, -3.0]), np.array([0.02, 0.05, 0.25])
        flaps = np.radians([2.0, -3.0, 0.0, 0.0, 1.0, -2.0, -4.0, 6.0])
        force, moment, condition = self.full.loads(through_air, omega, 1.026, flaps, -0.1)
        v = np.array([40.0, 1.5, 3.0])                                              # Makani's body axes
        speed = float(np.linalg.norm(v))
        alpha, beta, q = math.atan2(3.0, 40.0), math.asin(1.5/speed), 0.5*1.026*speed**2
        omega_hat = np.array([-0.02*25.66, 0.05*1.28, -0.25*25.66])/(2.0*speed)
        cf, cm, _ = self.full.coefficients(alpha, beta, omega_hat, flaps, -0.1)
        expected_moment = q*32.9*np.array([25.66, 1.28, 25.66])*cm+[0.0, 3500.0*math.sin(alpha), 0.0]
        assert_allclose(force, np.array([-1.0, 1.0, -1.0])*(q*32.9*cf), rtol=1e-13)
        assert_allclose(moment, np.array([-1.0, 1.0, -1.0])*expected_moment, rtol=1e-13)
        self.assertAlmostEqual(condition["alpha_rad"], alpha, places=15)
        assert_allclose(condition["omega_hat"], omega_hat, rtol=1e-14)
        nose_up = self.full.loads(np.array([-math.cos(math.radians(30.0)), 0.0, -math.sin(math.radians(30.0))])*30.0, np.zeros(3), 1.0, ZERO)
        no_moment = MakaniDatabase.from_files(LOW, spoiler_path=SPOILERS, high_path=HIGH,
                                              settings=DatabaseSettings(empirical_pitching_moment_Nm=0.0))
        bare = no_moment.loads(np.array([-math.cos(math.radians(30.0)), 0.0, -math.sin(math.radians(30.0))])*30.0, np.zeros(3), 1.0, ZERO)
        self.assertAlmostEqual(nose_up[1][1]-bare[1][1], 1750.0, places=9)           # 3500 N m * sin(30 deg), about +y in both axes


class ConstantLoads:
    """A reference model whose loads do not depend on the state."""
    def __init__(self, force, moment):
        self.force, self.moment, self.calls = np.array(force, dtype=float), np.array(moment, dtype=float), 0

    def loads(self, velocity, omega, density, flaps, thrust_coeff):
        self.calls += 1
        return self.force, self.moment, dict(airspeed_m_s=float(np.linalg.norm(velocity)), flaps=list(flaps), thrust_coeff=thrust_coeff)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.state = RigidState(0.0, np.array([120., -300., 250.]), rotation_from_vector([0.3, -0.2, 1.1]), np.array([-30., -10., 28.]),
                                np.array([0.02, -0.05, 0.2]))
        self.times = np.arange(6)*0.01
        self.wind = np.array([-3., 8.4, 0.])

    def test_without_loads_the_driver_is_the_plant(self):
        run = run_reference(ConstantLoads([0., 0., 0.], [0., 0., 0.]), BODY, self.state, self.wind, times=self.times, density_kg_m3=1.0, substeps=4)
        state = self.state
        for k in range(5):
            state = plant_step(BODY, state, step_s=float(self.times[k+1]-self.times[k]), substeps=4)[0]
            state = RigidState(self.times[k+1], state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s)
            self.assertEqual(run.records[k+1]["position_cm_m"], state.position_cm_m.tolist())
            self.assertEqual(run.records[k+1]["rotation"], state.rotation.tolist())
            self.assertEqual(run.records[k+1]["angular_velocity_body_rad_s"], state.angular_velocity_body_rad_s.tolist())
        self.assertEqual(run.status, "complete")

    def test_loads_that_do_not_depend_on_the_state_give_the_same_run_held_or_per_stage(self):
        def run(per_stage):
            return run_reference(ConstantLoads([300., -200., 15000.], [50., 800., -120.]), BODY, self.state, self.wind, times=self.times,
                                 density_kg_m3=1.0, substeps=4, per_stage=per_stage)
        held, staged = run(False), run(True)
        for a, b in zip(held.records, staged.records):
            for key in ("position_cm_m", "velocity_cm_m_s", "rotation", "angular_velocity_body_rad_s"):
                self.assertEqual(a[key], b[key])
        # and the held loads are those of the model about the centre of mass
        assert_allclose(held.records[0]["held_force_body_N"], [300., -200., 15000.])
        assert_allclose(held.records[0]["held_moment_cm_body_Nm"],
                        np.array([50., 800., -120.])-np.cross(BODY.centre_of_mass_body_m, [300., -200., 15000.]), rtol=1e-14)
        state = plant_step(BODY, self.state, step_s=0.01, substeps=4, held=HeldLoads(*(np.array(held.records[0][key]) for key in
                                                                                  ("held_force_body_N", "held_moment_cm_body_Nm"))))[0]
        self.assertEqual(held.records[1]["position_cm_m"], state.position_cm_m.tolist())

    def test_per_stage_loads_are_those_of_the_state_of_each_stage(self):
        class Linear(ConstantLoads):
            """Loads linear in the velocity through the air and in the angular velocity."""
            def loads(self, velocity, omega, density, flaps, thrust_coeff):
                return -40.0*np.asarray(velocity), -900.0*np.asarray(omega), dict(airspeed_m_s=float(np.linalg.norm(velocity)))
        wind = self.wind

        def external(time_s, stage, model):                    # the same loads written directly for the plant
            pose = stage.pose(BODY)
            force = -40.0*(pose.rotation.T@(pose.velocity_m_s-wind))
            moment = -900.0*stage.angular_velocity_body_rad_s-np.cross(BODY.centre_of_mass_body_m, force)
            return stage.rotation@force, stage.rotation@moment
        run = run_reference(Linear([0.]*3, [0.]*3), BODY, self.state, wind, times=self.times, density_kg_m3=1.0, substeps=3, per_stage=True)
        state = self.state
        for k in range(5):
            state = plant_step(BODY, state, step_s=float(self.times[k+1]-self.times[k]), substeps=3, external=external)[0]
            state = RigidState(self.times[k+1], state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s)
            assert_allclose(run.records[k+1]["position_cm_m"], state.position_cm_m, rtol=1e-14)
            assert_allclose(run.records[k+1]["angular_velocity_body_rad_s"], state.angular_velocity_body_rad_s, rtol=1e-12, atol=1e-15)
        held = run_reference(Linear([0.]*3, [0.]*3), BODY, self.state, wind, times=self.times, density_kg_m3=1.0, substeps=3)
        self.assertGreater(np.linalg.norm(np.array(held.records[-1]["velocity_cm_m_s"])-run.records[-1]["velocity_cm_m_s"]), 1e-9)

    def test_a_failure_of_the_model_inside_the_stages_is_the_models(self):
        class Late(ConstantLoads):
            def loads(self, *args):
                self.calls += 1
                if self.calls == 9:
                    raise FloatingPointError("not finite")
                return self.force, self.moment, dict(airspeed_m_s=1.0)
        run = run_reference(Late([0., 0., 1.], [0., 0., 0.]), BODY, self.state, self.wind, times=self.times, density_kg_m3=1.0, substeps=2,
                            per_stage=True)
        self.assertEqual((run.status, run.failed_module), ("failed", "aerodynamics"))

    def test_the_hooks_are_called_in_order_and_feed_the_model(self):
        calls, model = [], ConstantLoads([0., 0., 9000.], [0., 0., 0.])
        start = (np.array([1., 2., 3.]), np.array([0.1, 0.2, 0.3]))

        def before(index, time_s, state, nodes, node_velocities, acceleration):
            calls.append((index, time_s, acceleration))
        run = run_reference(model, BODY, self.state, self.wind, times=self.times, density_kg_m3=1.0, substeps=2, before_step=before,
                            flaps=lambda: np.full(8, 0.01*len(calls)), thrust_coeff=lambda: 0.1*len(calls), start_acceleration=start,
                            extra_loads=lambda pose, record: (np.array([10., 0., 0.]), np.zeros(3), dict(marker=record["step"])))
        self.assertEqual([call[0] for call in calls], list(range(6)))
        self.assertIs(calls[0][2], start)
        acceleration = calls[1][2][0]                           # under the loads held over the step before
        held = np.array(run.records[0]["held_force_body_N"])
        assert_allclose(acceleration, np.array(run.records[1]["rotation"])@held/BODY.mass_kg+[0., 0., -9.81], rtol=1e-12)
        self.assertEqual([record["marker"] for record in run.records], list(range(6)))
        self.assertEqual(run.records[2]["aero_condition"]["flaps"], [0.03]*8)
        self.assertAlmostEqual(run.records[2]["thrust_coeff_used"], 0.3)
        first = run.records[0]
        assert_allclose(np.array(first["held_force_body_N"])-[0., 0., 9000.], np.array(first["rotation"]).T@[10., 0., 0.], rtol=0, atol=1e-9)

    def test_a_failing_model_stops_the_run_with_its_module(self):
        class Failing(ConstantLoads):
            def loads(self, *args):
                if self.calls == 3:
                    raise FloatingPointError("not finite")
                return super().loads(*args)
        run = run_reference(Failing([0., 0., 1.], [0., 0., 0.]), BODY, self.state, self.wind, times=self.times, density_kg_m3=1.0)
        self.assertEqual((run.status, run.failed_step, run.failed_module, len(run.records)), ("failed", 3, "aerodynamics", 3))


if __name__ == "__main__":
    unittest.main()
