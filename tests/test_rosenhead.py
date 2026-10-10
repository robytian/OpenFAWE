"""Independent integral and invariance checks for Rosenhead-Moore segments."""
import unittest
from unittest.mock import patch

import numpy as np

from openfawe.aero import vortex as v


def quadrature(start, end, point, circulation, radius, order=128):
    """Integrate the regularized Biot-Savart law, without endpoint formulas."""
    start, end, point = [np.asarray(x, dtype=float) for x in (start, end, point)]
    x, weight = np.polynomial.legendre.leggauss(order)
    delta = end - start
    r = point - (start + ((x + 1) / 2)[:, None] * delta)
    denominator = (np.sum(r*r, axis=1) + radius**2)**1.5
    return circulation / (4*np.pi) * np.sum(
        (weight / 2)[:, None] * np.cross(delta, r) / denominator[:, None], axis=0)


def bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                  np.asarray(expected).view(np.uint64))


class RosenheadTests(unittest.TestCase):
    def velocity(self, start, end, point, gamma=1.7, radius=.25, **kwargs):
        return v.segment_velocity(start, end, point, gamma,
            model=v.CoreModel.ROSENHEAD_MOORE, core_radius=radius, **kwargs)

    def test_ids_and_independent_gauss_integral(self):
        self.assertEqual([int(m) for m in v.CoreModel], list(range(6)))
        start, end = [-.7, .3, -.8], [1.2, -.4, 1.]
        for point in ([.5, 1.2, .4], [.3, .5, .2], [1.3, -.5, 1.1], [-1., .4, -.9]):
            for radius in (.1, .3, 1.4):
                with self.subTest(point=point, radius=radius):
                    q64 = quadrature(start, end, point, 1.7, radius, 64)
                    q128 = quadrature(start, end, point, 1.7, radius)
                    np.testing.assert_allclose(q64, q128, rtol=3e-12, atol=3e-14)
                    np.testing.assert_allclose(self.velocity(start, end, point, radius=radius),
                                               q128, rtol=3e-12, atol=3e-14)

    def test_subdivision_orientation_and_scale(self):
        start, end = np.array([-.7, .3, -.8]), np.array([1.2, -.4, 1.])
        point = np.array([.5, 1.2, .4])
        nodes = start + np.array([0., .13, .42, .77, 1.])[:, None]*(end-start)
        expected = self.velocity(start, end, point)
        split = v.segments_velocity([point], nodes, [[0,1],[1,2],[2,3],[3,4]], 1.7,
            model=v.CoreModel.ROSENHEAD_MOORE, core_radius=.25)[0]
        np.testing.assert_allclose(split, expected, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(self.velocity(end, start, point), -expected, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(self.velocity(start, end, point, gamma=-1.7), -expected, rtol=0, atol=0)
        for scale in (1e-9, 1e-3, 1e3, 1e9):
            actual = scale*self.velocity(scale*start, scale*end, scale*point, radius=scale*.25)
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-14)

    def test_infinite_line_limit_is_scully_not_vatistas_n2(self):
        h, radius, gamma, half_length = .7, .3, 1.7, 1e5
        expected = [0., gamma*h/(2*np.pi*(h*h+radius*radius)), 0.]
        actual = self.velocity([0.,0.,-half_length], [0.,0.,half_length], [h,0.,0.],
                               gamma=gamma, radius=radius)
        np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=0)

    def test_axis_policy_including_zero_strength_and_endpoints(self):
        start, end = [0.,0.,-1.], [0.,0.,1.]
        for gamma in (0., 1.7):
            for point in (start, [0.,0.,0.], end):
                with self.assertRaises(v.FilamentSingularity):
                    self.velocity(start, end, point, gamma=gamma)
                bits(self.velocity(start, end, point, gamma=gamma, singularity="zero"), np.zeros(3))
        bits(self.velocity(start, end, [0.,0.,2.]), np.zeros(3))

    def test_zero_radius_and_mixed_batch_reference_are_bitwise(self):
        rng = np.random.default_rng(20260930)
        points, nodes = rng.normal(size=(11,3)), rng.normal(size=(70,3))
        edges = np.arange(70).reshape(-1,2)
        gamma, radius = rng.normal(size=35), rng.uniform(.05,.7,size=35)
        gamma[::7] = 0.; radius[::5] = 0.
        ideal = v.segments_velocity(points, nodes, edges, gamma)
        bits(v.segments_velocity(points, nodes, edges, gamma,
            model=v.CoreModel.ROSENHEAD_MOORE, core_radius=0.), ideal)
        exclusions = [[0,0], [1,31], [3,32]]
        with patch.object(v, "_sum_batched", v._sum_reference):
            reference = v.segments_velocity(points, nodes, edges, gamma,
                model=v.CoreModel.ROSENHEAD_MOORE, core_radius=radius, exclusions=exclusions)
        for chunk in (1, 4, 4096):
            bits(v.segments_velocity(points, nodes, edges, gamma,
                model=v.CoreModel.ROSENHEAD_MOORE, core_radius=radius,
                exclusions=exclusions, chunk_size=chunk), reference)

    def test_far_exterior_avoids_endpoint_cosine_cancellation(self):
        # Endpoint cosines would both round to one here. Direct integration
        # retains the tiny, nonzero field and is independent of rationalization.
        start, end = [0.,0.,0.], [1.,0.,0.]
        for point in ([1e8, .001, .002], [-1e8, .001, .002], [1e4, 1., -2.]):
            expected = quadrature(start, end, point, 1.7, .25)
            actual = self.velocity(start, end, point)
            self.assertGreater(np.linalg.norm(actual), 0.)
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)

    def test_small_positive_core_near_interior(self):
        h, radius = 1e-9, 2e-9
        expected = [0., 1.7*h/(2*np.pi*(h*h+radius*radius)*np.sqrt(1+h*h+radius*radius)), 0.]
        np.testing.assert_allclose(self.velocity([0,0,-1], [0,0,1], [h,0,0], radius=radius),
                                   expected, rtol=2e-13, atol=0)


if __name__ == "__main__":
    unittest.main()
