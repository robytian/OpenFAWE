"""Full-circle section polars (stage 4, decision 0020): Makani transcription and periodic tables.

Contract docs/tasks/OF-0007-wingfull20261001/contract.zh.md (D1). The source
blend is reproduced with the scipy path of the original code when scipy is
installed; the remaining tests use the verbatim tables in cases/polars/makani.
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from openfawe.aero import polars as P
from openfawe.aero.wing import SectionPolar

ROOT = Path(__file__).resolve().parents[1]
MAKANI = ROOT / "cases/polars/makani"
STALLS = {"oref": (1., 16.), "ref": (1., 16.), "f8": (-10., 10.), "mp6d": (-10., 10.), "approx_blade_rj8": (-10., 10.)}

try:
    import scipy.interpolate as si
except ImportError:  # pragma: no cover
    si = None


def source_interp(flaps, alphas, coeffs):
    """analysis/aero/airfoil.py _GetInterpolation (scipy path)."""
    if len(flaps) > 1:
        return si.RectBivariateSpline(flaps, alphas, np.array(coeffs), kx=1, ky=1).ev
    return lambda _, x: np.interp(x, alphas, coeffs[0])


def source_blend(data, stall_angles, naca):
    """analysis/aero/airfoil.py _AddPostStallData, line for line."""
    alphas = np.linspace(-np.pi, np.pi, 400)
    full = copy.deepcopy(data)
    full['alphas'] = alphas.tolist()
    past = np.maximum(np.maximum(stall_angles[0]-alphas, alphas-stall_angles[1]), 0.0)
    weights = 1.0-np.minimum(past/0.1, 1.0)
    for coeff in ('lift_coeffs', 'drag_coeffs', 'moment_coeffs'):
        full[coeff] = np.zeros((len(data['flap_deflections']), len(alphas)))
        low = source_interp(data['flap_deflections'], data['alphas'], data[coeff])
        high = source_interp(naca['flap_deflections'], naca['alphas'], naca[coeff])
        for i, flap in enumerate(data['flap_deflections']):
            for j in range(len(alphas)):
                full[coeff][i][j] = weights[j]*low(flap, alphas[j])+(1.0-weights[j])*high(flap, alphas[j])
    return full


class MakaniTranscription(unittest.TestCase):
    def setUp(self):
        self.high = P.load_makani_airfoil(MAKANI/"naca0012_high_aoa.json")

    def test_tables_load_verbatim(self):
        for name in STALLS:
            data = json.loads((MAKANI/f"{name}.json").read_text())
            airfoil = P.load_makani_airfoil(MAKANI/f"{name}.json")
            self.assertEqual(airfoil.cl.shape, (len(data['flap_deflections']), len(data['alphas'])))
            np.testing.assert_array_equal(airfoil.cl, np.array(data['lift_coeffs']))
            self.assertEqual(len(airfoil.source_sha256), 64)

    def test_lookup_clamps_and_interpolates_bilinearly(self):
        airfoil = P.load_makani_airfoil(MAKANI/"oref.json")
        # Below the table (0 deg) the clamped value equals the first column.
        cl_low, _, _ = airfoil.lookup(np.array([-0.5, airfoil.alpha_rad[0]]), 0.)
        self.assertEqual(cl_low[0], cl_low[1])
        index = int(np.where(airfoil.flap_rad == 0.)[0][0])
        self.assertEqual(cl_low[1], airfoil.cl[index, 0])
        # Midway between two flap rows at a knot: arithmetic mean of the rows.
        flap = .5*(airfoil.flap_rad[index]+airfoil.flap_rad[index+1])
        cl_mid, _, _ = airfoil.lookup(airfoil.alpha_rad[7], flap)
        self.assertAlmostEqual(float(cl_mid), .5*(airfoil.cl[index, 7]+airfoil.cl[index+1, 7]), places=14)
        # GetCoeffs wraps the angle: 2 pi + a equals a.
        a = np.array([0.1, 0.3])
        for ref, got in zip(airfoil.coefficients(a), airfoil.coefficients(a+2*np.pi)):
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)

    @unittest.skipUnless(si is not None, "scipy not installed")
    def test_blend_matches_source_scipy_path(self):
        naca = json.loads((MAKANI/"naca0012_high_aoa.json").read_text())
        for name, stall in STALLS.items():
            data = json.loads((MAKANI/f"{name}.json").read_text())
            reference = source_blend(data, [np.deg2rad(s) for s in stall], naca)
            blended = P.post_stall_blend(P.load_makani_airfoil(MAKANI/f"{name}.json"), stall, self.high)
            for key, mine in (('lift_coeffs', blended.cl), ('drag_coeffs', blended.cd), ('moment_coeffs', blended.cm)):
                np.testing.assert_allclose(mine, np.array(reference[key]), rtol=0, atol=4e-15)
            functions = [source_interp(reference['flap_deflections'], reference['alphas'], reference[k])
                         for k in ('lift_coeffs', 'drag_coeffs', 'moment_coeffs')]
            rng = np.random.default_rng(7)
            alpha, delta = rng.uniform(-7, 7, 500), rng.uniform(-.6, .6, 500)
            wrapped = np.mod(alpha+np.pi, 2*np.pi)-np.pi
            saturated = np.clip(delta, min(reference['flap_deflections']), max(reference['flap_deflections']))
            for function, mine in zip(functions, blended.coefficients(alpha, delta)):
                np.testing.assert_allclose(mine, function(saturated, wrapped), rtol=0, atol=4e-15)

    def test_blend_is_source_inside_and_naca_beyond_the_band(self):
        airfoil = P.load_makani_airfoil(MAKANI/"f8.json")
        blended = P.post_stall_blend(airfoil, (-10., 10.), self.high)
        self.assertEqual(len(blended.alpha_rad), P.BLEND_POINTS)
        inside = np.abs(blended.alpha_rad) <= np.deg2rad(10.)
        np.testing.assert_allclose(blended.cl[0, inside], airfoil.lookup(blended.alpha_rad[inside], 0.)[0], rtol=0, atol=1e-15)
        beyond = np.abs(blended.alpha_rad) >= np.deg2rad(10.)+P.BLEND_WIDTH_RAD
        np.testing.assert_allclose(blended.cl[0, beyond], self.high.lookup(blended.alpha_rad[beyond], 0.)[0], rtol=0, atol=1e-15)

    def test_section_polar_is_periodic_with_declared_attached_range(self):
        blended = P.post_stall_blend(P.load_makani_airfoil(MAKANI/"oref.json"), (1., 16.), self.high)
        polar = P.section_polar(blended)
        self.assertTrue(polar.periodic)
        self.assertEqual(polar.attached_deg, (1., 16.))
        self.assertEqual((polar.alpha_deg[0], polar.alpha_deg[-1]), (-180., 180.))
        a = np.array([0.2, 2.0, -3.0, np.pi, -np.pi, 10.])
        for ref, got in zip(polar.coefficients(a), polar.coefficients(a+2*np.pi)):
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(polar.attached(np.deg2rad([0., 5., 20., 90., -90.])), [False, True, False, False, False])
        # Knot values equal the blended table row at zero flap.
        index = int(np.where(blended.flap_rad == 0.)[0][0])
        np.testing.assert_array_equal(polar.cl, blended.cl[index])
        polar.cl_slope(a)  # defined everywhere on a periodic table


class PeriodicSectionPolar(unittest.TestCase):
    def test_periodic_requires_full_circle_and_wraps(self):
        with self.assertRaises(ValueError):
            SectionPolar([-170., 0., 180.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], periodic=True)
        polar = SectionPolar([-180., -90., 0., 90., 180.], [0., -1., 0., 1., 0.], [1., 1., 0., 1., 1.], [0.]*5, periodic=True)
        cl, cd, _ = polar.coefficients(np.array([np.pi/2, np.pi/2+2*np.pi, -3*np.pi/2]))
        np.testing.assert_allclose(cl, [1., 1., 1.], rtol=0, atol=1e-14)
        self.assertTrue(np.all(polar.attached([0., 3.])))
        with self.assertRaises(ValueError):
            SectionPolar([-180., 0., 180.], [0.]*3, [0.]*3, [0.]*3, periodic=True, attached_deg=(10., 5.))
        with self.assertRaises(ValueError):
            SectionPolar([-10., 0., 10.], [0.]*3, [0.]*3, [0.]*3, attached_deg=(-20., 5.))

    def test_default_polar_behaviour_unchanged(self):
        polar = SectionPolar.linear()
        self.assertFalse(polar.periodic)
        self.assertIsNone(polar.attached_deg)
        with self.assertRaises(ValueError):
            polar.coefficients(np.deg2rad(25.))
        np.testing.assert_allclose(polar.coefficients(0.1)[0], 2*np.pi*0.1, rtol=0, atol=1e-14)


class Viterna(unittest.TestCase):
    def test_extension_is_continuous_and_periodic(self):
        base = SectionPolar.linear(limit_deg=30.)
        polar = P.viterna_extension(base, (-12., 12.), 20.)
        self.assertTrue(polar.periodic)
        cl_s, cd_s, _ = base.coefficients(np.deg2rad(12.))
        cl, cd, _ = polar.coefficients(np.deg2rad(np.array([12., 12.+1e-9, 90., 180., -12., -12.-1e-9])))
        self.assertAlmostEqual(float(cl[0]), float(cl_s), places=12)
        self.assertAlmostEqual(float(cl[1]), float(cl_s), places=6)
        self.assertAlmostEqual(float(cd[1]), float(cd_s), places=6)
        self.assertAlmostEqual(float(cd[2]), 1.11+0.018*20., places=12)   # CDmax at 90 deg
        self.assertAlmostEqual(float(cl[2]), 0., places=12)
        self.assertAlmostEqual(float(cl[3]), 0., places=12)
        self.assertAlmostEqual(float(cl[4]), -float(cl_s), places=12)
        self.assertAlmostEqual(float(cl[5]), -float(cl_s), places=6)
        self.assertTrue(np.all(polar.cd >= 0.))


if __name__ == "__main__":
    unittest.main()
