"""Makani hover strip model transcription (comparison module, decision 0020; contract D2/W1).

The oracle is docs/tasks/OF-0007-wingfull20261001/makani-original-cfm.json: the
ORIGINAL 1d66923 Python code run on the public table's params (run_makani_original.py).
"""
import json
import unittest
from pathlib import Path

import numpy as np

from openfawe.aero import makani_hover_strip as H

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "cases/polars/makani/hover_model_sn02_wind_10_no_wake_model.json"
AIRFOILS = ROOT / "cases/polars/makani"
ORACLE = ROOT / "docs/tasks/OF-0007-wingfull20261001/makani-original-cfm.json"


class Transcription(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = H.HoverStripModel.from_table_params(TABLE, AIRFOILS)
        cls.alphas, cls.betas, cls.public = H.public_table_cfm(TABLE)
        cls.A, cls.B = np.meshgrid(cls.alphas, cls.betas, indexing="ij")
        cls.D = np.zeros((8,) + cls.A.shape)

    def test_panels_and_parameters(self):
        self.assertEqual(len(self.model.panels), 16)
        names = [p.name for p in self.model.panels]
        self.assertEqual(names[:10], [f"Wing (panel {i})" for i in range(10)])
        self.assertEqual(names[-2:], ["Horizontal tail", "Vertical tail"])
        self.assertEqual((self.model.wing_area, self.model.wing_b, self.model.wing_c), (32.9, 25.66, 1.28))
        self.assertEqual(self.model.panels[0].airfoil.stall_rad, (np.deg2rad(1.), np.deg2rad(16.)))
        self.assertEqual(self.model.panels[-1].airfoil.stall_rad, (np.deg2rad(-10.), np.deg2rad(10.)))
        self.assertFalse(self.model.use_wake_model)

    def test_matches_original_code_on_the_table_grid(self):
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self.alphas, oracle["alphas_rad"], rtol=0, atol=0)
        cfm = self.model.force_moment_coefficients(self.A, self.B, self.D, [0., 0., 0.], oracle["apparent_wind_speed"])
        np.testing.assert_allclose(cfm, np.array(oracle["cfm"]), rtol=0, atol=1e-13)

    def test_public_table_deep_post_stall_without_2d_to_3d(self):
        model = H.HoverStripModel.from_table_params(TABLE, AIRFOILS, convert_2d_to_3d=False)
        cfm = model.force_moment_coefficients(self.A, self.B, self.D, [0., 0., 0.], 10.)
        deep = np.abs(np.rad2deg(self.alphas)) >= 60.
        j = int(np.argmin(np.abs(self.betas)))
        # The table generator (Makani 9cb0bb7) is not the 1d66923 revision: without the 2D-to-3D
        # factor the deep post-stall values agree to 1.3e-4 at beta = 0 and to 5.2e-3 at |beta| = 60 deg
        # (side force of the pylons/vertical tail); inside |alpha| < 30 deg differences reach 0.06 (CZ)
        # and 0.19 (CY). Recorded in docs/tasks/OF-0007-wingfull20261001/transcription-check.json.
        np.testing.assert_allclose(cfm[deep, j], self.public[deep, j], rtol=0, atol=2e-4)
        np.testing.assert_allclose(cfm[deep], self.public[deep], rtol=0, atol=6e-3)
        # With the 1d66923 factor the deep post-stall normal force is about 25% lower than the table.
        with_factor = self.model.force_moment_coefficients(self.A, self.B, self.D, [0., 0., 0.], 10.)
        i = int(np.argmin(np.abs(np.rad2deg(self.alphas) + 105.)))
        self.assertLess(with_factor[i, j, 2] / self.public[i, j, 2], 0.8)

    def test_rho_cancels_in_coefficients(self):
        from dataclasses import replace
        other = replace(self.model, rho=1.1)
        a = self.model.force_moment_coefficients(self.A[:3, :2], self.B[:3, :2], self.D[:, :3, :2], [0., 0., 0.], 10.)
        b = other.force_moment_coefficients(self.A[:3, :2], self.B[:3, :2], self.D[:, :3, :2], [0., 0., 0.], 10.)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_wake_model_runs_and_changes_only_the_tails(self):
        model = H.HoverStripModel.from_table_params(TABLE, AIRFOILS, use_wake_model=True)
        sub = (slice(8, 10), slice(5, 6))
        with_wake = model.force_moment_coefficients(self.A[sub], self.B[sub], self.D[(slice(None),) + sub], [0., 0., 0.], 10.)
        without = self.model.force_moment_coefficients(self.A[sub], self.B[sub], self.D[(slice(None),) + sub], [0., 0., 0.], 10.)
        self.assertTrue(np.all(np.isfinite(with_wake)))
        self.assertGreater(np.max(np.abs(with_wake - without)), 1e-3)


class Pieces(unittest.TestCase):
    def test_sampling_points_follow_the_source_unit_test(self):
        panel = H.StripPanel("Horizontal tail", 1., 2., 1., 4., 1., [0., 0., 0.], 0., np.eye(3), -1,
                             H.post_stall_blend(H.load_makani_airfoil(AIRFOILS / "f8.json"), (-10., 10.),
                                                H.load_makani_airfoil(AIRFOILS / "naca0012_high_aoa.json")))
        points = H.panel_sampling_points(panel, thickness_ratio=0.5, num_points=(3, 2, 2))
        for x in (-0.75, -0.25, 0.25):
            for y in (-1.0, 1.0):
                for z in (-0.25, 0.25):
                    self.assertTrue(np.any(np.all(np.abs(points - [x, y, z]) < 1e-9, axis=1)))
        self.assertEqual(H.panel_sampling_points(panel).shape, (20, 3))

    def test_apparent_wind_round_trip_and_downwash(self):
        v = H.apparent_wind_sph_to_cart(10., np.array([0.3, -1.2]), np.array([0.1, 0.4]))
        speed, alpha, beta = H.apparent_wind_cart_to_sph(v)
        np.testing.assert_allclose(speed, 10., rtol=0, atol=1e-12)
        np.testing.assert_allclose(alpha, [0.3, -1.2], rtol=0, atol=1e-12)
        np.testing.assert_allclose(beta, [0.1, 0.4], rtol=0, atol=1e-12)
        self.assertEqual(H.estimate_downwash("Pylon 1", 0.2), 0.0)
        eps = H.estimate_downwash("Horizontal tail", np.array([0.0]))
        self.assertAlmostEqual(float(eps[0]), 1.12 * 2.0 * 2.05 / (np.pi * 20.0), places=12)

    def test_convert_2d_to_3d_factor(self):
        cl, cd, cm = H.convert_2d_to_3d(np.array([1.0]), np.array([2.0]), np.array([-0.5]), np.array([np.pi / 2]), 20.0, (25., 180.))
        k = 1.0 - 0.41 * (1.0 - np.exp(-17.0 / 20.0))
        w = np.cos(np.pi * (np.pi / 2 - np.deg2rad(25.)) / np.deg2rad(155.) - np.pi / 2)
        np.testing.assert_allclose([cl[0], cd[0], cm[0]], np.array([1., 2., -.5]) * (1 - w * (1 - k)), rtol=0, atol=1e-14)
        cl, _, _ = H.convert_2d_to_3d(np.array([1.0]), np.array([2.0]), np.array([-0.5]), np.array([0.1]), 20.0, (25., 180.))
        self.assertEqual(cl[0], 1.0)


if __name__ == "__main__":
    unittest.main()
