"""Full-incidence VSM closure (stage 4, decision 0020): default path unchanged, new regimes solve.

Contract docs/tasks/OF-0007-wingfull20261001/contract.zh.md §3 items 2-3. The
'source' closure is compared bitwise against the retained pre-stage-4 module
(docs/tasks/OF-0006-vsmcore20260928/retained/review-r1-input) on the archived
M600 case; the full-incidence closure is exercised on attached, reverse, 90 deg
and hover-like (zero ambient, history induction only) flows with the Makani
NACA 0012 full-circle table.
"""
import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from openfawe.aero import polars as P
from openfawe.aero.aircraft import PanelPolars, load_aircraft
from openfawe.aero.methods import AeroBackend
from openfawe.aero.vsm import (REGIMES, _vsm_closure_full, _vsm_closure_full_jacobian, geometry,
                               onset_weight, solve_vsm, solve_vsm_operator, vsm_influence)
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing, solve_settings_metadata

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "docs/tasks/OF-0006-vsmcore20260928/retained/review-r1-input/src/openfawe/aero/vsm.py"
M600 = ROOT / "cases/qualification/m600_makani_reconstructed.json"
NACA = ROOT / "cases/polars/makani/naca0012_high_aoa.json"
FULL = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence")


def flat_plate_polar():
    return P.section_polar(P.load_makani_airfoil(NACA), 0.)


def inflow(speed, alpha_deg, beta_deg=0.):
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    return speed * np.array([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)])


class DefaultPathUnchanged(unittest.TestCase):
    def test_settings_metadata_and_validation(self):
        self.assertNotIn("closure", solve_settings_metadata(SolveSettings()))
        self.assertEqual(solve_settings_metadata(FULL)["closure"], "full_incidence")
        with self.assertRaises(ValueError):
            SolveSettings(closure="other")
        self.assertIn("full-incidence", AeroBackend("vsm", vsm_solve=FULL).metadata()["closure"])
        self.assertIn("source VSM", AeroBackend("vsm").metadata()["closure"])

    @unittest.skipUnless(RETAINED.exists(), "retained pre-stage-4 module not present")
    def test_m600_source_closure_bitwise_against_retained_module(self):
        spec = importlib.util.spec_from_file_location("openfawe.aero._vsm_before_stage4", RETAINED)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        aircraft = load_aircraft(M600)
        wind = inflow(20., 5.4)
        for settings in (SolveSettings(relaxation=.03, max_iterations=2000),
                         SolveSettings(solver="newton_picard", max_iterations=60)):
            a = old.solve_vsm(aircraft.wing, aircraft.polar, wind, settings=settings)
            b = solve_vsm(aircraft.wing, aircraft.polar, wind, settings=settings)
            for key, value in vars(a).items():
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.tobytes(), getattr(b, key).tobytes(), key)
            self.assertEqual(b.closure, "source")
            self.assertIsNone(b.panel_regime)


class FullIncidence(unittest.TestCase):
    def setUp(self):
        self.wing = Wing.ellipse(12, span=6., root_chord=1.)
        self.polar = flat_plate_polar()

    def test_attached_flow_agrees_with_source_closure_to_second_order(self):
        wind = inflow(10., 4.)
        source = solve_vsm(self.wing, self.polar, wind, settings=SolveSettings(solver="newton_picard", max_iterations=60))
        full = solve_vsm(self.wing, self.polar, wind, settings=FULL)
        self.assertEqual(full.closure, "full_incidence")
        self.assertTrue(np.all(full.panel_regime == 0))
        # Source: Gamma = c CL |v x s|^2 / (2 |a x s|); full: c CL |v x s| / 2. They differ by
        # |v x s| / |a x s| - 1 = O((w/U)^2), a few tenths of a percent for this wing.
        ratio = full.force[2] / source.force[2]
        self.assertAlmostEqual(ratio, 1., delta=5e-3)
        self.assertGreater(full.force[2], 0.)

    def test_ninety_degree_flow_is_pure_drag_without_circulation(self):
        wind = inflow(10., 90.)
        full = solve_vsm(self.wing, self.polar, wind, settings=FULL)
        np.testing.assert_allclose(full.circulation, 0., atol=1e-12)
        cl, cd, _ = self.polar.coefficients(np.pi / 2)
        self.assertAlmostEqual(float(cl), 0., places=12)
        expected = .5 * 1.225 * 100. * float(cd) * float(np.sum(self.wing.area))
        self.assertAlmostEqual(full.force[2] / expected, 1., delta=1e-6)
        # No attached range is declared on this table, so the label stays 0 (undeclared).
        self.assertTrue(np.all(full.panel_regime == 0))

    def test_reverse_flow_solves_and_is_labelled(self):
        wind = inflow(10., 180. - 4.)   # from the trailing edge towards the leading edge
        full = solve_vsm(self.wing, self.polar, wind, settings=FULL)
        self.assertTrue(np.all(full.panel_regime == 2))
        self.assertTrue(np.all(np.abs(full.polar_alpha_rad) > np.pi / 2))
        self.assertGreater(full.force[2], 0.)
        self.assertLess(full.force[0], 0.)   # drag along the reversed stream
        self.assertTrue(np.all(np.isfinite(full.panel_force)))

    def test_hover_like_zero_ambient_with_history_induction(self):
        # No ambient flow; a slipstream (known induction) blows the inner panels only.
        cp, chord, normal, tangent, span = geometry(self.wing)
        blown = np.abs(cp[:, 1]) < 1.5
        known = np.where(blown[:, None], 8. * tangent + 0.8 * normal, 0.)
        operator = vsm_influence(self.wing, np.array([1., 0., 0.]))
        load_operator = vsm_influence(self.wing, np.array([1., 0., 0.]), at_load_points=True)
        with self.assertRaises(ValueError):
            solve_vsm_operator(self.wing, self.polar, np.zeros(3), operator, load_operator, known_induction=known,
                               known_load_induction=known, settings=SolveSettings(solver="newton_picard", max_iterations=60))
        full = solve_vsm_operator(self.wing, self.polar, np.zeros(3), operator, load_operator, known_induction=known,
                                  known_load_induction=known, settings=FULL)
        self.assertEqual(full.closure, "full_incidence")
        self.assertTrue(np.all(full.circulation[blown] != 0.))
        self.assertTrue(np.all(full.panel_regime[~blown] == REGIMES.index("no_inflow")) or
                        np.all(full.circulation[~blown] != 0.))   # unblown panels see only induced flow
        self.assertTrue(np.all(np.isfinite(full.panel_force)))
        self.assertGreater(full.force[2], 0.)

    def test_exact_jacobian_matches_finite_differences(self):
        wind = inflow(10., 30.)
        cp, chord, normal, tangent, span = geometry(self.wing)
        ambient = np.tile(wind, (self.wing.panels, 1))
        matrix = vsm_influence(self.wing, wind / np.linalg.norm(wind))
        known = np.zeros((self.wing.panels, 3))
        rng = np.random.default_rng(3)
        gamma = rng.normal(0., 2., self.wing.panels)
        args = (ambient, matrix, known, chord, tangent, normal, span, self.polar)
        exact = _vsm_closure_full_jacobian(gamma, *args)
        step = 1e-6
        numeric = np.empty_like(exact)
        for j in range(self.wing.panels):
            d = np.zeros_like(gamma)
            d[j] = step
            numeric[:, j] = ((_vsm_closure_full(gamma + d, *args)[0] - gamma - d)
                             - (_vsm_closure_full(gamma - d, *args)[0] - gamma + d)) / (2 * step)
        np.testing.assert_allclose(exact, numeric, rtol=0, atol=2e-6)

    def test_panel_polars_regime_labels(self):
        polars = PanelPolars([self.polar] * self.wing.panels)
        self.assertTrue(np.all(polars.attached(np.zeros(self.wing.panels))))
        labelled = PanelPolars([SectionPolar(self.polar.alpha_deg, self.polar.cl, self.polar.cd, self.polar.cm,
                                             periodic=True, attached_deg=(-10., 10.))] * self.wing.panels)
        full = solve_vsm(self.wing, labelled, inflow(10., 30.), settings=FULL)
        self.assertTrue(np.all(full.panel_regime == 1))


class DeclaredRegimes(unittest.TestCase):
    """span_flow_fraction and reverse_flow='strip' (model parameters, off by default)."""

    def setUp(self):
        self.wing = Wing.ellipse(12, span=6., root_chord=1.)
        self.polar = flat_plate_polar()

    def test_settings_validation_and_metadata(self):
        self.assertNotIn("span_flow_fraction", solve_settings_metadata(FULL))
        self.assertNotIn("reverse_flow", solve_settings_metadata(FULL))
        tuned = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence",
                              span_flow_fraction=.05, reverse_flow="strip")
        meta = solve_settings_metadata(tuned)
        self.assertEqual((meta["span_flow_fraction"], meta["reverse_flow"], meta["reverse_fade"]), (.05, "strip", .2))
        self.assertNotIn("reverse_fade", solve_settings_metadata(FULL))
        self.assertIs(type(SolveSettings(closure="full_incidence", span_flow_fraction=np.float32(.25)).span_flow_fraction), float)
        with self.assertRaises(ValueError):
            SolveSettings(closure="full_incidence", reverse_fade=.1)       # belongs to reverse_flow='strip'
        with self.assertRaises(ValueError):
            SolveSettings(closure="full_incidence", reverse_flow="strip", reverse_fade=0.)
        with self.assertRaises(ValueError):
            SolveSettings(closure="full_incidence", reverse_flow="strip", reverse_fade=1.5)
        with self.assertRaises(ValueError):
            SolveSettings(span_flow_fraction=.05)                      # belongs to the full-incidence closure
        with self.assertRaises(ValueError):
            SolveSettings(closure="full_incidence", reverse_flow="other")
        with self.assertRaises(ValueError):
            SolveSettings(closure="full_incidence", span_flow_fraction=.6)

    def test_reverse_strip_has_no_circulation_and_section_loads(self):
        settings = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence", reverse_flow="strip")
        wind = inflow(10., 180.-8.)
        solution = solve_vsm(self.wing, self.polar, wind, settings=settings)
        np.testing.assert_array_equal(solution.circulation, 0.)
        self.assertTrue(np.all(solution.panel_regime == 2))
        # No induction anywhere: every panel sees the ambient flow, loads are pure strip theory.
        cp, chord, normal, tangent, span = geometry(self.wing)
        alpha = np.arctan2(normal @ wind, tangent @ wind)
        cl, cd, _ = self.polar.coefficients(alpha)
        q_c_dl = .5*1.225*100.*chord*np.linalg.norm(self.wing.dl, axis=1)
        drag = wind/np.linalg.norm(wind)
        lift = np.cross(drag, span)
        expected = np.sum(q_c_dl[:, None]*(cl[:, None]*lift+cd[:, None]*drag), axis=0)
        np.testing.assert_allclose(solution.force, expected, rtol=1e-9, atol=1e-9)
        # Forward flow is untouched by the option.
        forward = inflow(10., 4.)
        np.testing.assert_array_equal(solve_vsm(self.wing, self.polar, forward, settings=settings).circulation,
                                      solve_vsm(self.wing, self.polar, forward, settings=FULL).circulation)

    def test_pure_span_flow_carries_no_circulation(self):
        settings = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence", span_flow_fraction=.05)
        wind = 10.*np.array([.02, 1., .01])/np.linalg.norm([.02, 1., .01])     # 1.3 deg off the span axis
        solution = solve_vsm(self.wing, self.polar, wind, settings=settings)
        inner = np.abs(self.wing.cp[:, 1]) < 2.5                               # panels whose span axis is along y
        np.testing.assert_array_equal(solution.circulation[inner], 0.)
        self.assertTrue(np.all(solution.panel_regime[inner] == REGIMES.index("no_inflow")))
        self.assertTrue(np.all(np.isfinite(solution.panel_force)))
        # Normal flow is untouched by the option.
        forward = inflow(10., 4.)
        np.testing.assert_array_equal(solve_vsm(self.wing, self.polar, forward, settings=settings).circulation,
                                      solve_vsm(self.wing, self.polar, forward, settings=FULL).circulation)

    def test_onset_weights(self):
        cp, chord, normal, tangent, span = geometry(self.wing)
        n = self.wing.panels
        ones, off = onset_weight(np.tile(inflow(10., 4.), (n, 1)), tangent, normal, .05, .2)
        np.testing.assert_array_equal(ones, 1.)
        self.assertFalse(off.any())
        # No onset flow at all (hover outside every slipstream): weight one, not switched off.
        still, off = onset_weight(np.zeros((n, 3)), tangent, normal, .05, .2)
        np.testing.assert_array_equal(still, 1.)
        self.assertFalse(off.any())
        # Span-flow fade: zero at sigma <= f, one at sigma >= 2f, smooth in between.
        inner = np.abs(self.wing.cp[:, 1]) < 2.5
        for sigma, expected in ((.04, 0.), (.05, 0.), (.075, .5), (.10, 1.), (.5, 1.)):
            wind = np.tile(10.*np.array([sigma, np.sqrt(1.-sigma**2), 0.]), (n, 1))
            weight, off = onset_weight(wind, tangent, normal, .05)
            np.testing.assert_allclose(weight[inner], expected, atol=2e-3)
            self.assertEqual(bool(off[inner].all()), sigma <= .05)
        # Reverse fade in the onset chordwise fraction: one at 90 deg, zero from cos = -fade.
        fade = .2
        for alpha, expected in ((89., 1.), (90., 1.), (np.rad2deg(np.arccos(-fade/2)), .5),
                                (np.rad2deg(np.arccos(-fade)), 0.), (150., 0.), (-95.739170477266787, .5)):
            weight, off = onset_weight(np.tile(inflow(10., alpha), (n, 1)), tangent, normal, 0., fade)
            np.testing.assert_allclose(weight[inner], expected, atol=2e-2)
            self.assertFalse(off.any())

    def test_weighted_jacobian_matches_finite_differences(self):
        cp, chord, normal, tangent, span = geometry(self.wing)
        direction = np.array([-.01, .99, .12])
        ambient = np.tile(10.*direction/np.linalg.norm(direction), (self.wing.panels, 1))
        matrix = vsm_influence(self.wing, np.array([1., 0., 0.]))
        known = np.zeros((self.wing.panels, 3))
        gamma = np.random.default_rng(5).normal(0., .3, self.wing.panels)
        onset = onset_weight(ambient, tangent, normal, .08, .2)
        self.assertTrue(np.any((onset[0] > 1e-3) & (onset[0] < 1.-1e-3)))      # some panels really are inside a fade
        args = (ambient, matrix, known, chord, tangent, normal, span, self.polar, onset)
        np.testing.assert_allclose(_vsm_closure_full(gamma, *args)[0], onset[0]*_vsm_closure_full(gamma, *args[:-1])[0], rtol=0, atol=0)
        exact = _vsm_closure_full_jacobian(gamma, *args)
        step = 1e-6
        numeric = np.empty_like(exact)
        for j in range(self.wing.panels):
            d = np.zeros_like(gamma)
            d[j] = step
            numeric[:, j] = ((_vsm_closure_full(gamma+d, *args)[0]-gamma-d)-(_vsm_closure_full(gamma-d, *args)[0]-gamma+d))/(2*step)
        np.testing.assert_allclose(exact, numeric, rtol=0, atol=5e-6)

    def test_reverse_fade_removes_the_jump_at_ninety_degrees(self):
        # A section with lift at 90 deg (cl = 0.5 + sin 2a): a hard reverse switch makes the target jump there.
        grid = np.arange(-180., 180.5, 1.)
        lifting = SectionPolar(grid, .5+np.sin(2*np.deg2rad(grid)), np.full(grid.shape, .1), np.zeros(grid.shape),
                               "lift at ninety", periodic=True)
        cp, chord, normal, tangent, span = geometry(self.wing)
        matrix = vsm_influence(self.wing, np.array([0., 0., 1.]))
        known = np.zeros((self.wing.panels, 3))
        gamma = np.zeros(self.wing.panels)

        def jump(fade):
            targets = []
            for alpha in (89.999, 90.001):
                ambient = np.tile(inflow(10., alpha), (self.wing.panels, 1))
                onset = onset_weight(ambient, tangent, normal, 0., fade)
                targets.append(_vsm_closure_full(gamma, ambient, matrix, known, chord, tangent, normal, span, lifting, onset)[0])
            return np.max(np.abs(targets[1]-targets[0])), np.max(np.abs(targets[0]))

        smooth, scale = jump(.2)
        hard, _ = jump(1e-9)
        self.assertGreater(scale, 1.)                    # the sections do carry circulation at 90 deg
        self.assertLess(smooth, 2e-3*scale)              # faded: continuous
        self.assertGreater(hard, .9*scale)               # switched: the whole target disappears (the test can see a jump)

    def test_default_path_never_evaluates_weights(self):
        with mock.patch("openfawe.aero.vsm.onset_weight", side_effect=AssertionError("weights evaluated")):
            solution = solve_vsm(self.wing, self.polar, inflow(10., 120.), settings=FULL)
        self.assertIsNone(solution.circulation_weight)

    def test_weights_follow_the_history_induction_and_are_reported(self):
        # Zero ambient, uniform reverse history induction (hover-like): every panel is stripped.
        cp, chord, normal, tangent, span = geometry(self.wing)
        settings = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence", reverse_flow="strip")
        operator = vsm_influence(self.wing, np.array([1., 0., 0.]))
        load = np.zeros_like(operator)
        for known_flow, expected in ((inflow(8., 170.), 0.), (inflow(8., 5.), 1.)):
            solution = solve_vsm_operator(self.wing, self.polar, np.zeros(3), operator, load,
                                          known_induction=known_flow, known_load_induction=known_flow, settings=settings)
            np.testing.assert_array_equal(solution.circulation_weight, expected)
            self.assertEqual(bool(np.any(solution.circulation != 0.)), bool(expected))

    def test_exact_cross_plane_flow_on_a_swept_surface_has_a_finite_jacobian(self):
        # M600 vertical tail, wind exactly along z: v.t = v.n = 0 with |v x s| > 0 on the swept fin panels.
        aircraft = load_aircraft(ROOT / "cases/qualification/m600_makani_xfoil_polars.json")
        wind = np.array([0., 0., 10.])
        common = dict(solver="newton_picard", max_iterations=60, closure="full_incidence", span_flow_fraction=.05, reverse_flow="strip")
        analytic = solve_vsm(aircraft.wing, aircraft.polar, wind, settings=SolveSettings(jacobian="active_segment", **common))
        numeric = solve_vsm(aircraft.wing, aircraft.polar, wind, settings=SolveSettings(**common))
        np.testing.assert_allclose(analytic.circulation, numeric.circulation, rtol=0, atol=1e-8)
        fin = np.abs(aircraft.wing.cp[:, 1]) < .05
        np.testing.assert_array_equal(analytic.circulation_weight[fin & (aircraft.wing.cp[:, 2] > 0)], 0.)


if __name__ == "__main__":
    unittest.main()
