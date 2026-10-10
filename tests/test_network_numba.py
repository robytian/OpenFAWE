"""Compiled wake-network bookkeeping must be bitwise identical to the Python loops.

pack_rings(compiled=True) and incident_pairs reproduce wake.pack_rings and the
exclusion loop of wake._network_convection: same edges in the same order, same
summed strengths (same addition order), same (point, segment) pairs in the
same order, including coincident nodes and signed zeros.
"""
import importlib.util
import unittest

import numpy as np

from openfawe.aero.wake import WakeConfig, pack_rings, compiled_network

HAS_NUMBA = importlib.util.find_spec("numba") is not None


def reference_pairs(nodes, edges, first):
    points = nodes[first:]
    locations = {}
    for index, point in enumerate(points):
        locations.setdefault(tuple(point), []).append(index)
    pairs = []
    for segment, (a, b) in enumerate(edges):
        for node in (a, b):
            pairs.extend((index, segment) for index in locations.get(tuple(nodes[node]), ()))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def wavy_rows(rng, depth, width):
    x = np.arange(depth)[:, None] * np.ones((1, width)) * 0.7
    y = np.ones((depth, 1)) * np.linspace(-3., 3., width)[None, :]
    z = 0.2 * np.sin(x) * np.cos(y) + 0.01 * rng.normal(size=x.shape)
    return np.stack((x, y, z), axis=-1)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class CompiledNetwork(unittest.TestCase):
    def check(self, rows, strengths, panel_edges):
        nodes_a, edges_a, gamma_a = pack_rings(rows, strengths, panel_edges)
        nodes_b, edges_b, gamma_b = pack_rings(rows, strengths, panel_edges, compiled=True)
        np.testing.assert_array_equal(nodes_a, nodes_b)
        np.testing.assert_array_equal(edges_a, edges_b)
        self.assertEqual(edges_b.dtype, np.int64)
        np.testing.assert_array_equal(gamma_a.view(np.uint64), gamma_b.view(np.uint64))
        first = rows.shape[1]
        pairs_a = reference_pairs(nodes_a, edges_a, first)
        from openfawe.aero.network_numba import incident_pairs
        pairs_b = incident_pairs(nodes_b, edges_b, first)
        np.testing.assert_array_equal(pairs_a, pairs_b)
        self.assertEqual(pairs_b.dtype, np.int64)
        return edges_b, gamma_b, pairs_b

    def test_random_networks_bitwise(self):
        rng = np.random.default_rng(7)
        for depth, width in ((2, 3), (3, 5), (12, 9), (41, 17)):
            rows = wavy_rows(rng, depth, width)
            strengths = rng.normal(size=(depth - 1, width - 1))
            # Repeated and cancelling strengths exercise the exact-zero filter
            # and the per-edge summation order.
            strengths[1:, :] = strengths[:-1, :] * rng.choice([1., -1., 0.5], size=(depth - 2, width - 1))
            strengths[0, 0] = 0.
            panel_edges = np.column_stack([np.arange(width - 1), np.arange(1, width)])
            self.check(rows, strengths, panel_edges)

    def test_non_sequential_panels_and_shared_root(self):
        # Two surfaces whose root nodes coincide physically (distinct IDs,
        # same coordinates), plus a panel list in a scrambled order and a
        # reversed panel: the pair lists must still match the reference.
        rng = np.random.default_rng(3)
        depth, width = 9, 8
        rows = wavy_rows(rng, depth, width)
        rows[:, 4, :] = rows[:, 3, :]  # node 4 coincides with node 3 in every row
        rows[0, 1, 2] = -0.0            # signed zero must hash like +0.0
        rows[1, 1, 2] = 0.0
        panel_edges = np.array([[0, 1], [2, 1], [2, 3], [4, 5], [6, 5], [6, 7]])
        strengths = rng.normal(size=(depth - 1, len(panel_edges)))
        edges, gamma, pairs = self.check(rows, strengths, panel_edges)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(np.all(np.diff(pairs[:, 1]) >= 0))  # reference order is by segment

    def test_archived_configuration_switch(self):
        self.assertFalse(compiled_network(WakeConfig()))
        self.assertTrue(compiled_network(WakeConfig(velocity_backend="numba")))
        from openfawe.aero.vortex import TreeSettings
        self.assertTrue(compiled_network(WakeConfig(velocity_backend="tree", tree=TreeSettings())))

    def test_long_wake_like_history(self):
        # Depth 400 with 37 nodes: the size of a 400-step V3 run; strengths
        # decay and alternate like shed circulation differences.
        rng = np.random.default_rng(11)
        depth, width = 400, 37
        rows = wavy_rows(rng, depth, width)
        base = rng.normal(size=width - 1)
        strengths = base[None, :] * np.exp(-0.01 * np.arange(depth - 1))[:, None]
        strengths[::7] = 0.
        panel_edges = np.column_stack([np.arange(width - 1), np.arange(1, width)])
        edges, gamma, pairs = self.check(rows, strengths, panel_edges)
        # Every free end point of every edge is one pair (bound-row ends are not targets).
        self.assertEqual(len(pairs), int(np.sum(edges >= width)))


if __name__ == "__main__":
    unittest.main()
