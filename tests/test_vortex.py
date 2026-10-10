# SPDX-License-Identifier: Apache-2.0
# K03/K04 case structure adapted from OpenFAST v5.0.0 Test_BiotSavart_Sgmt.
# Independent analytical/invariance/error cases added for OpenFAWE.
"""K01-K09 verification; original tolerance semantics are preserved."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from openfawe.aero import CoreModel, FilamentSingularity, ring_velocity, segment_velocity, segments_velocity

A = np.array([0., 0., -1.])
B = np.array([0., 0., 1.])


class VortexTests(unittest.TestCase):
    def equal(self, actual, expected, rtol=1e-12, atol=1e-14):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def test_k01_closed_form(self):
        self.equal(segment_velocity(A, B, [1, 0, 0], 1), [0, 1 / (2 * np.pi * np.sqrt(2)), 0])

    def test_k02_symmetric_all_cores(self):
        for h in (.05, 1., 10.):
            ideal = 1 / (2 * np.pi * h * np.sqrt(1 + h * h))
            q = (h / .5) ** 2
            for model in CoreModel:
                with self.subTest(h=h, model=model):
                    factors = [1, min(q, 1), 1 if -1.25643*q < -10 else -np.expm1(-1.25643*q),
                               q / np.sqrt(1 + q*q),
                               (2*h*h*(1+h*h)) / (2*h*h*(1+h*h) + 4*.5**2),
                               h*h*np.sqrt(1+h*h)/((h*h+.5**2)*np.sqrt(1+h*h+.5**2))]
                    self.equal(segment_velocity(A, B, [h, 0, 0], 1, model=model, core_radius=.5),
                               [0, ideal*factors[model], 0])

    def test_k02_lamb_cutoff_is_distinct_from_full_exponential(self):
        for exponent in (9.99, 10.01):
            h = .5 * np.sqrt(exponent / 1.25643)
            ideal = 1 / (2 * np.pi * h * np.sqrt(1+h*h))
            full = ideal * -np.expm1(-exponent)
            velocity = segment_velocity(A, B, [h, 0, 0], 1, model=2, core_radius=.5)[1]
            self.equal(velocity, full if exponent < 10 else ideal)
            if exponent > 10:
                self.assertGreater(abs(velocity-full), 1e-7)

    def test_k02_recorded_upstream_welib_comparison(self):
        path = Path(__file__).resolve().parents[1] / "docs/tasks/OF-0006/upstream-kernel-check.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data["direct_reference_results"]:
            actual = segment_velocity(A, B, case["point"], 1, model=case["core_model_id"], core_radius=.5)
            # All recorded Lamb points are either below the cutoff or have an
            # exponentially negligible difference. Dedicated boundary test above.
            self.equal(actual, case["velocity_m_per_s"])

    def test_k03_upstream_scalar_batch_original_absolute_tolerance(self):
        for point in ([0, 0, 0], A, [.05, 0, -.5], [10, 0, 0]):
            for model in CoreModel:
                scalar = segment_velocity(A, B, point, 1, model=model, core_radius=.5, singularity="zero")
                batch = segments_velocity([point], [A, B], [[0, 1]], 1, model=model,
                                          core_radius=.5, singularity="zero")[0]
                self.equal(scalar, batch, rtol=0, atol=1e-4)

    def test_k04_subdivision_original_and_tighter_criteria(self):
        for point in ([0, 0, 0], A, [.05, 0, -.5], [100, 0, -.5]):
            for model in [*list(CoreModel)[:4], CoreModel.ROSENHEAD_MOORE]:
                whole = segment_velocity(A, B, point, 1, model=model, core_radius=.5, singularity="zero")
                split = segments_velocity([point], [A, [0, 0, 0], B], [[0, 1], [1, 2]], 1,
                                          model=model, core_radius=.5, singularity="zero")[0]
                self.equal(whole, split, rtol=0, atol=1e-4)
                if point[0] != 0:
                    self.equal(whole, split, rtol=1e-11, atol=1e-13)

    def test_k04_offset_is_not_subdivision_invariant(self):
        point = [.05, 0, -.5]
        whole = segment_velocity(A, B, point, 1, model=4, core_radius=.5)
        split = segments_velocity([point], [A, [0, 0, 0], B], [[0, 1], [1, 2]], 1, model=4, core_radius=.5)[0]
        self.assertGreater(np.linalg.norm(whole-split), 1e-3)

    def test_k05_orientation_linearity_ring_and_owned_output(self):
        corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
        points = np.array([[0, 0, 0], [.3, .2, .7]])
        for model in CoreModel:
            u = segment_velocity(A, B, [1, 0, .2], 2, model=model, core_radius=.5)
            self.equal(segment_velocity(B, A, [1, 0, .2], 2, model=model, core_radius=.5), -u)
            self.equal(segment_velocity(A, B, [1, 0, .2], -2, model=model, core_radius=.5), -u)
            self.equal(segment_velocity(A, B, [1, 0, .2], 4, model=model, core_radius=.5), 2*u)
            ring = ring_velocity(corners, points, 1, model=model, core_radius=.5)
            edges = sum(segments_velocity(points, corners, [[j, (j+1)%4]], 1, model=model,
                                          core_radius=.5) for j in range(4))
            self.equal(ring, edges)
            self.equal(ring_velocity(corners[::-1], points, 1, model=model, core_radius=.5), -ring)
        # Independent analytical velocity at the centre of a square of side 2.
        self.equal(ring_velocity(corners, [[0, 0, 0]], 1)[0], [0, 0, np.sqrt(2)/np.pi])
        before = corners.copy(), points.copy()
        out = ring_velocity(corners, points, 1)
        out[:] = 99
        self.equal(corners, before[0]); self.equal(points, before[1])

    def test_k06_proper_rotation_and_translation(self):
        axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(.7)*k + (1-np.cos(.7))*(k@k)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)
        shift = np.array([2., -3., .7])
        for model in CoreModel:
            for h in (.05, 1., 10.):
                point = np.array([h, 0, 0])
                expected = rotation @ segment_velocity(A, B, point, 1, model=model, core_radius=.5)
                self.equal(segment_velocity(rotation@A+shift, rotation@B+shift, rotation@point+shift,
                                            1, model=model, core_radius=.5), expected)

    def test_k07_scale_covariance(self):
        for model in CoreModel:
            for h in (.05, 1., 10.):
                p = np.array([h, 0, -.5])
                expected = segment_velocity(A, B, p, 1, model=model, core_radius=.5)
                for scale in (1e-3, 1., 1e3):
                    self.equal(scale*segment_velocity(scale*A, scale*B, scale*p, 1, model=model,
                                                       core_radius=scale*.5), expected, rtol=1e-11, atol=1e-13)

    def test_k08_singular_exclusions_and_zero_core(self):
        for model in CoreModel:
            with self.assertRaises(FilamentSingularity):
                segment_velocity(A, B, [0, 0, 0], 1, model=model, core_radius=.5)
            self.equal(segment_velocity(A, B, A, 1, model=model, core_radius=.5, singularity="zero"), [0, 0, 0])
            self.equal(segment_velocity(A, B, [1, 0, 0], 1, model=model, core_radius=0),
                       segment_velocity(A, B, [1, 0, 0], 1))
        self.equal(segment_velocity(A, B, [0, 0, 2], 1), [0, 0, 0])
        result = segments_velocity([[0, 0, 0], [1, 0, 0]], [A, B], [[0, 1]], 1,
                                   exclusions=[[0, 0], [0, 0]])
        self.equal(result[0], [0, 0, 0]); self.equal(result[1], segment_velocity(A, B, [1, 0, 0], 1))

    def test_k08_invalid_inputs(self):
        default = dict(points=[[1, 0, 0]], nodes=[A, B], connectivity=[[0, 1]], circulation=1)
        bad = [dict(points=[1, 0, 0]), dict(points=[[np.nan, 0, 0]]), dict(nodes=[[0, 0, 0], [0, 0, 0]]),
               dict(connectivity=[[0., 1.]]), dict(connectivity=[[False, True]]), dict(connectivity=[[0, 2]]),
               dict(connectivity=[[0, -1]]), dict(circulation=np.inf), dict(circulation=[1, 2]),
               dict(core_radius=-1), dict(core_radius=[1, 2]), dict(model=6), dict(model=2.), dict(model=True),
               dict(exclusions=[[1, 0]]), dict(exclusions=[[0, 1]]), dict(exclusions=[[0., 0.]]),
               dict(chunk_size=0), dict(chunk_size=True), dict(chunk_size=1.1), dict(singularity="ignore"),
               dict(points=[[1j, 0, 0]]), dict(points=[["1", "0", "0"]])]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                segments_velocity(**(default | change))
        with self.assertRaises(ValueError):
            ring_velocity(np.zeros((3, 3)), [[1, 0, 0]], 1)

    def test_k08_batch_chunks_per_segment_inputs_and_empty_shapes(self):
        rng = np.random.default_rng(60912)
        points, nodes = rng.normal(size=(17, 3)), rng.normal(size=(6, 3))
        conn = np.array([[0, 1], [2, 3], [4, 5]])
        gamma, radius = np.array([1, -2, .3]), np.array([0, .2, .5])
        saved = [v.copy() for v in (points, nodes, conn, gamma, radius)]
        for value in (points, nodes, conn, gamma, radius):
            value.flags.writeable = False
        expected = sum(segments_velocity(points, nodes, conn[j:j+1], gamma[j], model=3,
                                         core_radius=radius[j]) for j in range(3))
        for chunk in (1, 4, 4096):
            actual = segments_velocity(points, nodes, conn, gamma, model=3, core_radius=radius, chunk_size=chunk)
            np.testing.assert_array_equal(actual, expected)
        for value, original in zip((points, nodes, conn, gamma, radius), saved):
            np.testing.assert_array_equal(value, original)
        self.assertEqual(segments_velocity(np.empty((0, 3)), nodes, conn, gamma).shape, (0, 3))
        self.equal(segments_velocity(points, np.empty((0, 3)), np.empty((0, 2), dtype=int), 1), np.zeros_like(points))

    def test_independent_biot_savart_quadrature(self):
        # Integrate dl x r / |r|^3, not the implementation's closed formula.
        start, end, point = np.array([-.7, .3, -.8]), np.array([1.2, -.4, 1.]), np.array([.5, 1.2, .4])
        abscissa, weight = np.polynomial.legendre.leggauss(96)
        delta = end-start
        r = point - (start + ((abscissa+1)/2)[:, None]*delta)
        integral = np.sum((weight/2)[:, None]*np.cross(delta, r)/np.linalg.norm(r, axis=1)[:, None]**3, axis=0)
        self.equal(segment_velocity(start, end, point, 1.7), 1.7/(4*np.pi)*integral)

    def test_near_interior_stable_denominator(self):
        h = 1e-9
        self.equal(segment_velocity(A, B, [h, 0, 0], 1)[1], 1/(2*np.pi*h*np.sqrt(1+h*h)))

    def test_k09_import_isolation(self):
        code = "import sys; import openfawe.aero; assert not any(k.startswith(('PySide6', 'pyqtgraph', 'matplotlib', 'openfawe.lines')) for k in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
