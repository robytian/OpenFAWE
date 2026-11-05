"""Hand-over between crosswind and hover section data (decision 0025, amendments 1 and 2; G4 contract section 3 item 2)."""
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero import polars as P
from openfawe.aero.handover import (CROSSWIND_SOLVE, MAKANI_M600_STALL, TAKEOFF_LANDING_SOLVE, BlendedPolars,
                                    PanelBlendedPolars, SectionHandover, incidence_steps, makani_weight, panel_weights,
                                    stall_switches)
from openfawe.aero.methods import AeroBackend
from openfawe.aero.vsm import geometry
from openfawe.aero.vsm_wake import advance_vsm_wake
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing

ROOT = Path(__file__).resolve().parents[1]
NACA = ROOT / "cases/polars/makani/naca0012_high_aoa.json"


class Forbidden:
    """Section data that must not be evaluated."""

    def __getattr__(self, name):
        raise AssertionError(f"section data evaluated although its weight is zero: {name}")


class WeightTests(unittest.TestCase):
    def test_low_incidence_range_has_zero_weight(self):
        for alpha, beta in ((0., 0.), (-15., 0.), (5., 0.), (5., 15.), (-15., -15.), (3., -2.2)):
            self.assertEqual(makani_weight(alpha, beta), 0., (alpha, beta))

    def test_outside_the_blend_the_weight_is_one(self):
        for alpha, beta in ((15., 0.), (-25., 0.), (90., 0.), (0., 25.), (0., -25.), (-170., 3.)):
            self.assertEqual(makani_weight(alpha, beta), 1., (alpha, beta))

    def test_blend_values_at_the_stall_angles_and_their_combination(self):
        for alpha, beta in ((10., 0.), (-20., 0.), (0., 20.), (0., -20.)):
            self.assertAlmostEqual(makani_weight(alpha, beta), 0.5, places=14)
        self.assertAlmostEqual(makani_weight(10., 20.), 0.75, places=14)       # 1 - (1 - 0.5)(1 - 0.5)
        self.assertAlmostEqual(makani_weight(7.5, 0.), 0.25, places=14)
        values = [makani_weight(a, 0.) for a in np.linspace(5., 15., 41)]
        self.assertTrue(np.all(np.diff(values) > 0.))

    def test_other_stall_parameters_are_used(self):
        stall = dict(MAKANI_M600_STALL, high_alpha_deg=30.)
        self.assertEqual(makani_weight(20., 0., stall), 0.)
        self.assertEqual(makani_weight(20., 0.), 1.)


class BlendedPolarsTests(unittest.TestCase):
    def setUp(self):
        self.low = SectionPolar.linear(limit_deg=60.)
        self.high = P.section_polar(P.load_makani_airfoil(NACA), 0.)
        self.alpha = np.deg2rad([-4., 2., 9.])

    def test_weight_zero_and_one_never_touch_the_other_set(self):
        only_low = BlendedPolars(self.low, Forbidden(), 0.)
        only_high = BlendedPolars(Forbidden(), self.high, 1.)
        for a, b in zip(only_low.coefficients(self.alpha), self.low.coefficients(self.alpha)):
            assert_allclose(a, b, rtol=0, atol=0)
        for a, b in zip(only_high.coefficients(self.alpha), self.high.coefficients(self.alpha)):
            assert_allclose(a, b, rtol=0, atol=0)
        assert_allclose(only_low.attached(self.alpha), self.low.attached(self.alpha))
        assert_allclose(only_high.attached(self.alpha), self.high.attached(self.alpha))

    def test_mixture_is_linear_in_the_weight(self):
        blended = BlendedPolars(self.low, self.high, 0.3)
        for mixed, a, b in zip(blended.coefficients(self.alpha), self.low.coefficients(self.alpha), self.high.coefficients(self.alpha)):
            assert_allclose(mixed, 0.7*np.asarray(a)+0.3*np.asarray(b), rtol=0, atol=1e-15)
        assert_allclose(blended.cl_slope(self.alpha), 0.7*self.low.cl_slope(self.alpha)+0.3*self.high.cl_slope(self.alpha),
                        rtol=0, atol=1e-13)
        # Once the hover data take part, the attached-range label is theirs.
        assert_allclose(blended.attached(self.alpha), self.high.attached(self.alpha))

    def test_weight_outside_the_unit_interval_is_rejected(self):
        for weight in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                BlendedPolars(self.low, self.high, weight)


class SectionHandoverTests(unittest.TestCase):
    def setUp(self):
        self.low, self.high = SectionPolar.linear(limit_deg=60.), P.section_polar(P.load_makani_airfoil(NACA), 0.)

    def test_crosswind_side_is_the_production_model_itself(self):
        choice = SectionHandover(self.low, self.high, seeding="zero").select(3., -2.2)
        self.assertEqual(choice.weight, 0.)
        self.assertIs(choice.polar, self.low)
        self.assertIs(choice.solve, CROSSWIND_SOLVE)
        self.assertFalse(choice.zero_seed)                       # zero seeding belongs to the hover side only

    def test_hover_side_uses_the_blend_the_prescribed_setting_and_the_declared_seeding(self):
        for seeding, zero in (("previous", False), ("zero", True)):
            choice = SectionHandover(self.low, self.high, seeding=seeding).select(8., 0.)
            self.assertAlmostEqual(choice.weight, 0.3, places=14)
            self.assertIsInstance(choice.polar, BlendedPolars)
            self.assertAlmostEqual(choice.polar.weight, 0.3, places=14)
            self.assertIs(choice.solve, TAKEOFF_LANDING_SOLVE)
            self.assertEqual(choice.zero_seed, zero)
        self.assertEqual(TAKEOFF_LANDING_SOLVE.closure, "full_incidence")
        self.assertEqual((TAKEOFF_LANDING_SOLVE.span_flow_fraction, TAKEOFF_LANDING_SOLVE.reverse_flow,
                          TAKEOFF_LANDING_SOLVE.reverse_fade), (0.05, "strip", 0.2))

    def test_validation(self):
        with self.assertRaises(ValueError):
            SectionHandover(self.low, self.high, seeding="continuation")
        with self.assertRaises(ValueError):                      # the two sides swapped
            SectionHandover(self.low, self.high, low_solve=TAKEOFF_LANDING_SOLVE)
        with self.assertRaises(ValueError):
            SectionHandover(self.low, self.high, high_solve=CROSSWIND_SOLVE)
        with self.assertRaises(ValueError):
            SectionHandover(self.low, self.high, stall=dict(low_alpha_deg=-20.))
        with self.assertRaises(ValueError):
            SectionHandover(self.low, self.high, stall=dict(MAKANI_M600_STALL, blending_deg=0.))

    def test_metadata_names_the_rule_and_the_open_choice(self):
        data = SectionHandover(self.low, self.high, seeding="zero").metadata()
        self.assertEqual(data["seeding"], "zero")
        self.assertEqual(data["stall"], MAKANI_M600_STALL)
        self.assertEqual((data["low_closure"], data["high_closure"]), ("source", "full_incidence"))
        # Both solve settings are in the receipt, the hover side's iteration limit and fallback included.
        self.assertEqual(data["low_solve"]["solver"], "newton")
        self.assertEqual((data["high_solve"]["solver"], data["high_solve"]["max_iterations"], data["high_solve"]["closure"]),
                         ("newton_picard", 60, "full_incidence"))
        self.assertIn("fallback_max_iterations", data["high_solve"])


class IndicatorTests(unittest.TestCase):
    def test_stall_switches_count_label_changes_on_the_main_surface_only(self):
        regime = np.array([0, 0, 1, 1, 0, 2, 2, 0, 1, 0, 1])
        solution = SimpleNamespace(panel_regime=regime)
        self.assertEqual(stall_switches(solution, 8), (4, 4))    # 0011 0220: changes at 2, 4, 5, 7
        self.assertEqual(stall_switches(solution, 11), (7, 6))
        self.assertEqual(stall_switches(SimpleNamespace(panel_regime=None), 8), (None, None))
        self.assertEqual(stall_switches(SimpleNamespace(), 8), (None, None))

    def test_incidence_steps_are_taken_within_each_surface(self):
        alpha = np.deg2rad([2., 3., 12., 12.5, 170., -175., -174.])
        solution = SimpleNamespace(polar_alpha_rad=alpha)
        assembly = SimpleNamespace(names=("wing", "tail"), wings=(SimpleNamespace(panels=4), SimpleNamespace(panels=3)))
        report = incidence_steps(solution, assembly)
        self.assertAlmostEqual(report["wing"]["max_step_deg"], 9., places=10)
        self.assertEqual(report["wing"]["steps_over_limit"], 1)
        # 170 -> -175 deg is a 15 deg step across the branch cut, not 345 deg; the wing-tail junction is not a step.
        self.assertAlmostEqual(report["tail"]["max_step_deg"], 15., places=10)
        self.assertEqual(report["tail"]["steps_over_limit"], 1)
        self.assertEqual(report["total_steps_over_limit"], 2)
        single = incidence_steps(SimpleNamespace(polar_alpha_rad=alpha[:4]), SimpleNamespace(panels=4), limit_deg=10.)
        self.assertEqual((single["wing"]["steps_over_limit"], single["limit_deg"]), (0, 10.))


class SeedArgumentTests(unittest.TestCase):
    def setUp(self):
        self.wing, self.polar = Wing.ellipse(8), SectionPolar.linear()
        self.config = WakeConfig(dt_s=0.05, max_rows=4, core_radius_m=0.15, solve=SolveSettings(solver="newton", max_iterations=60))
        self.wind = np.array([12., 0., 0.8])
        backend = AeroBackend("vsm_free_wake", wake=self.config)
        self.state = backend.advance(backend.start(self.wing, self.polar, self.wind), self.polar, self.wind, next_time_s=0.05)

    def advance(self, **seed):
        return advance_vsm_wake(self.state, self.polar, self.config, self.wind, next_time_s=0.1, **seed)

    def test_default_seed_is_the_previous_circulation_bit_for_bit(self):
        default = self.advance()
        explicit = self.advance(initial_circulation=self.state.solution.circulation)
        for name in ("circulation", "panel_force", "induced_velocity"):
            self.assertTrue(np.array_equal(getattr(default.solution, name), getattr(explicit.solution, name)), name)
        self.assertEqual(default.solution.iterations, explicit.solution.iterations)

    def test_zero_seed_reaches_the_same_attached_flow_root_and_leaves_the_old_state_alone(self):
        before = self.state.solution.circulation.copy()
        zero = self.advance(initial_circulation=np.zeros(self.wing.panels))
        default = self.advance()
        assert_allclose(zero.solution.circulation, default.solution.circulation, rtol=0,
                        atol=1e-6*np.abs(default.solution.circulation).max())
        self.assertTrue(np.array_equal(self.state.solution.circulation, before))
        self.assertTrue(np.array_equal(zero.free_rows, default.free_rows))        # the wake geometry does not depend on the seed

    def test_a_failed_solve_leaves_the_state_as_it_was_so_that_a_retry_is_legitimate(self):
        before = {name: getattr(self.state, name).copy() for name in ("free_rows", "older_circulation", "release_times_s")}
        gamma = self.state.solution.circulation.copy()
        reference = self.advance()
        narrow = SectionPolar.linear(limit_deg=0.5)                                # the table ends below the incidence
        with self.assertRaises(ValueError):
            advance_vsm_wake(self.state, narrow, self.config, self.wind, next_time_s=0.1,
                             initial_circulation=np.zeros(self.wing.panels))
        for name, value in before.items():
            self.assertTrue(np.array_equal(getattr(self.state, name), value), name)
        self.assertTrue(np.array_equal(self.state.solution.circulation, gamma))
        self.assertEqual(self.state.step, 1)
        again = self.advance()
        self.assertTrue(np.array_equal(again.solution.circulation, reference.solution.circulation))
        self.assertTrue(np.array_equal(again.free_rows, reference.free_rows))

    def test_wrong_seed_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.advance(initial_circulation=np.zeros(self.wing.panels+1))
        with self.assertRaises(ValueError):
            self.advance(initial_circulation=np.full(self.wing.panels, np.nan))



class OutOfRangeOptionTests(unittest.TestCase):
    """Option (a) of owner item 乙 as an opt-in: a panel outside the crosswind table gets the hover data alone."""

    def setUp(self):
        from openfawe.aero.aircraft import PanelPolars
        self.low = PanelPolars([SectionPolar.linear(limit_deg=20.)]*3)
        self.high = P.section_polar(P.load_makani_airfoil(NACA), 0.)
        self.alpha = np.deg2rad([-4., 25., 9.])                       # the middle panel is outside the +-20 deg table

    def test_default_raises_as_before(self):
        blended = BlendedPolars(self.low, self.high, 0.3)
        with self.assertRaises(ValueError):
            blended.coefficients(self.alpha)
        self.assertEqual(blended.out_of_range, "error")

    def test_hover_option_gives_the_outside_panel_the_hover_data_and_leaves_the_others_blended(self):
        blended = BlendedPolars(self.low, self.high, 0.3, out_of_range="hover")
        inside = np.deg2rad([-4., 0., 9.])
        mixed = blended.coefficients(self.alpha)
        low_inside = self.low.coefficients(inside)
        high = self.high.coefficients(self.alpha)
        for m, a, b in zip(mixed, low_inside, high):
            assert_allclose(m[[0, 2]], (0.7*np.asarray(a)+0.3*np.asarray(b))[[0, 2]], rtol=0, atol=1e-15)
            self.assertAlmostEqual(m[1], np.asarray(b)[1], places=15)
        self.assertEqual(blended.last_out_of_range.tolist(), [1])
        self.assertEqual(blended.most_out_of_range, 1)
        blended.cl_slope(inside)                                        # inside: the blend as usual, the counter keeps its maximum
        self.assertEqual(blended.last_out_of_range.tolist(), [])
        self.assertEqual(blended.most_out_of_range, 1)
        assert_allclose(blended.cl_slope(inside), 0.7*self.low.cl_slope(inside)+0.3*self.high.cl_slope(inside), rtol=0, atol=1e-13)

    def test_weight_zero_and_one_are_untouched_by_the_option(self):
        only_low = BlendedPolars(self.low, Forbidden(), 0., out_of_range="hover")
        inside = np.deg2rad([-4., 0., 9.])
        for a, b in zip(only_low.coefficients(inside), self.low.coefficients(inside)):
            assert_allclose(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError):                              # weight 0 is the production model: it still raises outside
            only_low.coefficients(self.alpha)

    def test_handover_passes_the_option_through_and_validates_it(self):
        rule = SectionHandover(self.low, self.high, out_of_range="hover")
        choice = rule.select(12., 0.)                                     # inside the blend (weight 0.7 at 12 deg)
        self.assertEqual(choice.polar.out_of_range, "hover")
        self.assertIs(rule.select(5., 0.).polar, self.low)                # weight 0: the crosswind data themselves
        self.assertEqual(rule.metadata()["out_of_range"], "hover")
        self.assertEqual(SectionHandover(self.low, self.high).metadata()["out_of_range"], "error")
        with self.assertRaises(ValueError):
            SectionHandover(self.low, self.high, out_of_range="clamp")
        with self.assertRaises(ValueError):
            BlendedPolars(self.low, self.high, 0.3, out_of_range="clamp")

class PanelWeightTests(unittest.TestCase):
    """Option (c) of owner item 乙 as an opt-in: one weight per panel from the panel's own oncoming flow."""

    def setUp(self):
        from openfawe.aero.aircraft import PanelPolars
        self.low = PanelPolars([SectionPolar.linear(limit_deg=20.)]*4)
        self.high = P.section_polar(P.load_makani_airfoil(NACA), 0.)

    @staticmethod
    def flow(alpha_deg, beta_deg, speed=10.):
        a, b = np.radians(alpha_deg), np.radians(beta_deg)
        return speed*np.array([np.cos(a)*np.cos(b), np.sin(b), np.sin(a)*np.cos(b)])

    def test_every_panel_gets_the_simulator_weight_of_its_own_flow(self):
        cases = [(3., 0.), (12., 0.), (40., 0.), (-22., 0.), (3., 25.), (170., 0.)]
        weights = panel_weights([self.flow(a, b) for a, b in cases])
        for weight, (a, b) in zip(weights, cases):
            # atan2(w, u) and asin(v/V) of the flow built from the two angles give the angles back to rounding.
            self.assertAlmostEqual(weight, makani_weight(a, b), places=12)
        self.assertEqual(weights[0], 0.)
        self.assertEqual((weights[2], weights[5]), (1., 1.))                 # 170 deg: flow from behind is on the hover data
        self.assertAlmostEqual(weights[1], 0.7, places=12)

    def test_a_panel_without_oncoming_flow_takes_the_weight_of_the_origin(self):
        weights = panel_weights([[0., 0., 0.], self.flow(3., 0.)], fallback=0.4)
        self.assertEqual(weights.tolist(), [0.4, 0.])
        with self.assertRaises(ValueError):
            panel_weights([[np.nan, 0., 0.]])
        other = dict(MAKANI_M600_STALL, high_alpha_deg=0.)
        self.assertAlmostEqual(panel_weights([self.flow(3., 0.)], other)[0], 0.8, places=12)

    def test_the_mixture_is_panel_by_panel_and_exact_at_weight_zero_and_one(self):
        weights = np.array([0., 0.3, 1., 0.6])
        alpha = np.deg2rad([-4., 9., 2., 14.])
        mixed = PanelBlendedPolars(self.low, self.high, weights)
        low, high = self.low.coefficients(alpha), self.high.coefficients(alpha)
        for m, a, b in zip(mixed.coefficients(alpha), low, high):
            a, b = np.asarray(a), np.asarray(b)
            self.assertEqual((m[0], m[2]), (a[0], b[2]))                         # exact: no arithmetic on them
            assert_allclose(m[[1, 3]], ((1.-weights)*a+weights*b)[[1, 3]], rtol=0, atol=1e-15)
        slope = mixed.cl_slope(alpha)
        self.assertEqual((slope[0], slope[2]), (self.low.cl_slope(alpha)[0], self.high.cl_slope(alpha)[2]))
        assert_allclose(slope[[1, 3]], ((1.-weights)*self.low.cl_slope(alpha)+weights*self.high.cl_slope(alpha))[[1, 3]], rtol=0, atol=1e-13)
        attached = mixed.attached(alpha)
        self.assertEqual(attached[0], self.low.attached(alpha)[0])              # crosswind label where only that set is used
        self.assertTrue(np.array_equal(attached[1:], self.high.attached(alpha)[1:]))
        self.assertEqual(mixed.weight, 1.)                                      # the largest panel weight
        self.assertFalse(mixed.weights.flags.writeable)

    def test_a_panel_on_the_hover_data_alone_cannot_make_the_crosswind_table_raise(self):
        alpha = np.deg2rad([-4., 9., 55., 14.])                                 # panel 2 is outside the +-20 deg table
        mixed = PanelBlendedPolars(self.low, self.high, [0., 0.3, 1., 0.6])
        values = mixed.coefficients(alpha)
        self.assertEqual(values[0][2], np.asarray(self.high.coefficients(alpha)[0])[2])
        self.assertEqual((mixed.last_out_of_range.tolist(), mixed.most_out_of_range), ([], 0))

    def test_a_panel_that_uses_the_crosswind_table_outside_its_range_raises_unless_the_option_says_hover(self):
        for weights in ([0., 0.3, 1., 0.6], [0., 0., 1., 0.6]):                 # panel 1 in the blend, or on the crosswind data alone
            alpha = np.deg2rad([-4., 25., 2., 14.])
            with self.assertRaises(ValueError):
                PanelBlendedPolars(self.low, self.high, weights).coefficients(alpha)
            option = PanelBlendedPolars(self.low, self.high, weights, out_of_range="hover")
            values = option.coefficients(alpha)
            for m, b in zip(values, self.high.coefficients(alpha)):
                self.assertEqual(m[1], np.asarray(b)[1])                         # the hover data alone for that panel
            self.assertEqual((option.last_out_of_range.tolist(), option.most_out_of_range), ([1], 1))
            self.assertEqual(option.cl_slope(alpha)[1], self.high.cl_slope(alpha)[1])
            option.coefficients(np.deg2rad([-4., 9., 2., 14.]))                 # inside again: the list empties, the maximum stays
            self.assertEqual((option.last_out_of_range.tolist(), option.most_out_of_range), ([], 1))

    def test_a_panel_whose_flow_is_the_origins_gets_the_origins_weight_to_the_last_bit_also_with_sideslip(self):
        from openfawe.flight.kinematics import relative_wind_body, rotation_from_vector
        rng = np.random.default_rng(20261006)
        inside_beta_blend = 0
        for _ in range(300):
            rotation = rotation_from_vector(rng.normal(size=3))
            pose = SimpleNamespace(rotation=rotation, velocity_m_s=rng.normal(size=3)*20.)
            wind = rng.normal(size=3)*8.
            relative, speed, alpha, beta = relative_wind_body(pose, wind)
            weight = panel_weights([pose.rotation.T@(np.asarray(wind, dtype=float)-pose.velocity_m_s)])[0]
            self.assertEqual(weight, makani_weight(alpha, beta))                 # equal, not close
            inside_beta_blend += 15. < abs(beta) < 25.
        self.assertGreater(inside_beta_blend, 10)                               # the range in which the weight depends on sideslip

    def test_the_label_follows_the_data_and_a_choice_with_panel_weights_can_be_compared(self):
        alpha = np.deg2rad([-4., 25., 2., 14.])                                 # panel 1 (weight 0) is outside the +-20 deg table
        option = PanelBlendedPolars(self.low, self.high, [0., 0., 1., 0.6], out_of_range="hover")
        clamped = np.deg2rad([-4., 20., 2., 14.])
        expected = self.high.attached(alpha).copy()
        expected[0] = self.low.attached(clamped)[0]                             # only panel 0 is on the crosswind data alone
        self.assertTrue(np.array_equal(option.attached(alpha), expected))
        plain = PanelBlendedPolars(self.low, self.high, [0., 0., 1., 0.6])       # 'error': the label of a weight-0 panel is the crosswind set's
        expected[1] = self.low.attached(clamped)[1]
        self.assertTrue(np.array_equal(plain.attached(alpha), expected))
        rule = SectionHandover(self.low, self.high, weight_basis="panel")
        a, b = rule.select_panels([self.flow(3., 0.)]*4, 0.), rule.select_panels([self.flow(3., 0.)]*4, 0.)
        self.assertEqual(a, b)                                                  # the weight array takes no part in equality

    def test_weights_are_validated(self):
        for weights in ([0., 1.2], [np.nan, 0.], [-0.1, 0.]):
            with self.assertRaises(ValueError):
                PanelBlendedPolars(self.low, self.high, weights)
        with self.assertRaises(ValueError):
            PanelBlendedPolars(self.low, self.high, [0., 1.], out_of_range="clamp")
        with self.assertRaises(ValueError):                                     # one weight per panel
            PanelBlendedPolars(self.low, self.high, [0., 0.5, 1.]).coefficients(np.deg2rad([1., 2., 3., 4.]))

    def test_the_rule_picks_the_production_model_the_single_weight_blend_or_the_panel_blend(self):
        rule = SectionHandover(self.low, self.high, seeding="zero", weight_basis="panel", out_of_range="hover")
        quiet = rule.select_panels([self.flow(3., 0.)]*4, 0.)
        self.assertIs(quiet.polar, self.low)                                    # every weight zero: the crosswind data themselves
        self.assertIs(quiet.solve, CROSSWIND_SOLVE)
        self.assertEqual((quiet.weight, quiet.zero_seed, quiet.panel_weights.tolist()), (0., False, [0.]*4))
        uniform = rule.select_panels([self.flow(12., 0.)]*4, makani_weight(12., 0.))
        origin = rule.select(12., 0.)
        self.assertIsInstance(uniform.polar, BlendedPolars)                     # all equal: the origin basis itself
        # The flow rebuilt from the two angles returns them to rounding only; the weight of the origin is passed through as given.
        self.assertAlmostEqual(uniform.polar.weight, origin.polar.weight, places=12)
        self.assertEqual(uniform.polar.weight, uniform.panel_weights[0])
        self.assertEqual((uniform.weight, uniform.zero_seed), (origin.weight, True))
        self.assertEqual(uniform.polar.out_of_range, "hover")
        self.assertIs(uniform.solve, TAKEOFF_LANDING_SOLVE)
        mixed = rule.select_panels([self.flow(3., 0.), self.flow(12., 0.), self.flow(40., 0.), self.flow(3., 0.)], 0.)
        self.assertIsInstance(mixed.polar, PanelBlendedPolars)
        self.assertEqual(mixed.weight, 0.)                                      # the origin's weight is kept as it is
        assert_allclose(mixed.panel_weights, [0., 0.7, 1., 0.], rtol=0, atol=1e-12)
        self.assertTrue(np.array_equal(mixed.polar.weights, mixed.panel_weights))
        self.assertEqual((mixed.polar.out_of_range, mixed.zero_seed), ("hover", True))
        self.assertIs(mixed.solve, TAKEOFF_LANDING_SOLVE)                       # one panel above zero puts the step on the hover side
        self.assertIsNone(origin.panel_weights)
        # With the flow at the load points as well, a panel takes the larger of its two weights.
        both = rule.select_panels([self.flow(3., 0.), self.flow(12., 0.), self.flow(40., 0.), self.flow(3., 0.)], 0.,
                                  [self.flow(12., 0.), self.flow(3., 0.), self.flow(3., 0.), self.flow(3., 0.)])
        assert_allclose(both.panel_weights, [0.7, 0.7, 1., 0.], rtol=0, atol=1e-12)
        self.assertIs(rule.select_panels([self.flow(3., 0.)]*4, 0., [self.flow(4., 0.)]*4).polar, self.low)
        with self.assertRaises(ValueError):
            rule.select_panels([self.flow(3., 0.)]*4, 0., [self.flow(3., 0.)]*3)

    def test_the_basis_is_declared_validated_and_in_the_metadata(self):
        self.assertEqual(SectionHandover(self.low, self.high).weight_basis, "origin")
        self.assertEqual(SectionHandover(self.low, self.high).metadata()["weight_basis"], "origin")
        self.assertEqual(SectionHandover(self.low, self.high, weight_basis="panel").metadata()["weight_basis"], "panel")
        with self.assertRaises(ValueError):
            SectionHandover(self.low, self.high, weight_basis="section")


class ChooserArgumentTests(unittest.TestCase):
    """The optional chooser of the wake advance: called after the convection, default path untouched."""

    def setUp(self):
        self.wing, self.polar = Wing.ellipse(8), SectionPolar.linear()
        self.config = WakeConfig(dt_s=0.05, max_rows=4, core_radius_m=0.15, solve=SolveSettings(solver="newton", max_iterations=60))
        self.wind = np.array([12., 0., 0.8])
        backend = AeroBackend("vsm_free_wake", wake=self.config)
        self.state = backend.advance(backend.start(self.wing, self.polar, self.wind), self.polar, self.wind, next_time_s=0.05)

    def test_a_chooser_that_returns_the_same_data_and_settings_changes_no_bit(self):
        seen = []

        def chooser(time_s, points, ring_velocity, load_points, ring_load):
            seen.append((points.copy(), ring_velocity.copy(), time_s, load_points.copy(), ring_load.copy()))
            return self.polar, self.config.solve, None
        default = advance_vsm_wake(self.state, self.polar, self.config, self.wind, next_time_s=0.1)
        chosen = advance_vsm_wake(self.state, None, self.config, self.wind, next_time_s=0.1, chooser=chooser)
        for name in ("circulation", "panel_force", "induced_velocity"):
            self.assertTrue(np.array_equal(getattr(default.solution, name), getattr(chosen.solution, name)), name)
        self.assertEqual(default.solution.iterations, chosen.solution.iterations)
        self.assertTrue(np.array_equal(default.free_rows, chosen.free_rows))
        self.assertEqual(chosen.config, self.config)
        self.assertEqual(len(seen), 1)                                          # called once
        points, ring_velocity, time_s, load_points, ring_load = seen[0]
        self.assertTrue(np.array_equal(points, geometry(self.wing)[0]))         # the 3/4-chord control points
        self.assertTrue(np.array_equal(load_points, self.wing.cp))              # the quarter-chord load points
        self.assertTrue(np.array_equal(ring_velocity, np.zeros((self.wing.panels, 3))))    # no rings in this state
        self.assertTrue(np.array_equal(ring_load, np.zeros((self.wing.panels, 3))))
        self.assertEqual(time_s, default.time_s)

    def test_the_chosen_settings_and_seed_are_the_ones_used_and_the_state_is_bound_to_them(self):
        other = SolveSettings(solver="newton", max_iterations=37)
        zero = np.zeros(self.wing.panels)
        chosen = advance_vsm_wake(self.state, None, self.config, self.wind, next_time_s=0.1,
                                  chooser=lambda *onset: (self.polar, other, zero))
        self.assertIs(chosen.config.solve, other)
        self.assertEqual(chosen.config, WakeConfig(dt_s=0.05, max_rows=4, core_radius_m=0.15, solve=other))
        # The same step asked for in the older way: the state re-bound to the settings, the seed passed in.
        rebound = replace(self.state, config=replace(self.config, solve=other))
        explicit = advance_vsm_wake(rebound, self.polar, rebound.config, self.wind, next_time_s=0.1, initial_circulation=zero)
        self.assertTrue(np.array_equal(explicit.solution.circulation, chosen.solution.circulation))
        self.assertEqual(explicit.solution.iterations, chosen.solution.iterations)
        self.assertTrue(np.array_equal(explicit.free_rows, chosen.free_rows))

    def test_a_chooser_excludes_the_other_two_ways_of_saying_the_same_and_its_seed_is_checked(self):
        chooser = lambda *onset: (self.polar, self.config.solve, None)        # noqa: E731
        with self.assertRaises(ValueError):
            advance_vsm_wake(self.state, self.polar, self.config, self.wind, next_time_s=0.1, chooser=chooser)
        with self.assertRaises(ValueError):
            advance_vsm_wake(self.state, None, self.config, self.wind, next_time_s=0.1, chooser=chooser,
                             initial_circulation=np.zeros(self.wing.panels))
        with self.assertRaises(ValueError):
            advance_vsm_wake(self.state, None, self.config, self.wind, next_time_s=0.1,
                             chooser=lambda *onset: (self.polar, self.config.solve, np.zeros(3)))
        with self.assertRaises(ValueError):                                     # the settings are validated by the configuration
            advance_vsm_wake(self.state, None, self.config, self.wind, next_time_s=0.1,
                             chooser=lambda *onset: (self.polar, "newton", None))
        self.assertEqual(self.state.step, 1)                                    # the old state is as it was


if __name__ == "__main__":
    unittest.main()
