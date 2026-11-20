"""OF-0010 revision 22 addendum 4 (B3, H5): ``WakeConfig.frozen_rows_drift`` - the rows beyond ``free_convection_rows``
take the ambient velocity ('ambient', the default, unchanged) or the ambient velocity plus the rotor rings' induction alone
('ambient_rings'), checked on a small made-up network against direct Biot-Savart sums."""
import unittest

import numpy as np

from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.wake import WakeConfig, _network_convection
from openfawe.aero.wing import wake_config_metadata


def network():
    """Two nodes per row: a bound row, the trailing-edge row and three free rows (rows 0..4), then a square ring."""
    first = 2
    rows = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                     [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]],
                     [[1.0, 0.1, 0.2], [1.0, 1.1, 0.1]],
                     [[1.6, 0.2, 0.3], [1.6, 1.2, 0.2]],
                     [[2.3, 0.3, 0.3], [2.3, 1.3, 0.4]]])
    ring = np.array([[0.2, -0.4, 1.0], [0.2, 1.4, 1.0], [0.9, 1.4, 1.3], [0.9, -0.4, 1.3]])
    nodes = np.vstack([rows.reshape(-1, 3), ring])
    ring_first = rows.size//3
    edges, gamma = [], []
    for j in range(len(rows)):                        # spanwise filaments of every row
        edges.append([2*j, 2*j+1]); gamma.append(1.0+0.3*j)
    for j in range(len(rows)-1):                      # streamwise filaments between rows
        edges += [[2*j, 2*j+2], [2*j+1, 2*j+3]]; gamma += [0.4+0.1*j, -0.4-0.1*j]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):     # the ring
        edges.append([ring_first+a, ring_first+b]); gamma.append(2.5)
    edges, gamma = np.asarray(edges, dtype=np.int64), np.asarray(gamma, dtype=float)
    radii = np.full(len(edges), 0.05)
    return first, nodes, edges, gamma, radii, ring_first


class FrozenRowsDrift(unittest.TestCase):
    def inflow(self, points, time_s):
        return np.broadcast_to([3.0, 0.5, -0.2], (len(points), 3)).copy()

    def test_config(self):
        self.assertEqual(WakeConfig().frozen_rows_drift, 'ambient')
        self.assertNotIn('frozen_rows_drift', wake_config_metadata(WakeConfig()))
        self.assertNotIn('frozen_rows_drift', wake_config_metadata(WakeConfig(free_convection_rows=2)))
        c = WakeConfig(free_convection_rows=2, frozen_rows_drift='ambient_rings')
        self.assertEqual(wake_config_metadata(c)['frozen_rows_drift'], 'ambient_rings')
        with self.assertRaisesRegex(ValueError, 'frozen_rows_drift'):
            WakeConfig(frozen_rows_drift='ambient_rings')
        with self.assertRaisesRegex(ValueError, 'frozen_rows_drift'):
            WakeConfig(free_convection_rows=2, frozen_rows_drift='rings')

    def test_frozen_rows_take_the_rings_alone(self):
        first, nodes, edges, gamma, radii, ring_first = network()
        points = nodes[first:]
        rows_end = ring_first-first
        kw = dict(core_model=CoreModel.LAMB_OSEEN, core_radius_m=0.05, free_convection_rows=1)
        plain = _network_convection(nodes, edges, gamma, first, radii, WakeConfig(**kw), self.inflow, 0.0, ring_first)
        rings = _network_convection(nodes, edges, gamma, first, radii, WakeConfig(**kw, frozen_rows_drift='ambient_rings'),
                                    self.inflow, 0.0, ring_first)
        free = first*1
        ambient = self.inflow(points, 0.0)
        # the default: the frozen rows (points free..rows_end) keep the ambient velocity alone
        np.testing.assert_array_equal(plain[free:rows_end], ambient[free:rows_end])
        # the free row and the ring nodes are the same with and without the option
        np.testing.assert_array_equal(plain[:free], rings[:free])
        np.testing.assert_array_equal(plain[rows_end:], rings[rows_end:])
        # with the option the frozen rows take the ring edges' induction, nothing of the wing's own system
        ring_edge = np.all(edges >= ring_first, axis=1)
        self.assertEqual(int(ring_edge.sum()), 4)
        expected = ambient[free:rows_end]+segments_velocity(points[free:rows_end], nodes, edges[ring_edge], gamma[ring_edge],
                                                           model=CoreModel.LAMB_OSEEN, core_radius=radii[ring_edge])
        np.testing.assert_array_equal(rings[free:rows_end], expected)
        self.assertGreater(float(np.abs(rings[free:rows_end]-plain[free:rows_end]).max()), 1e-3)
        # without rings in the network the option changes nothing
        wing_only = ~np.any(edges >= ring_first, axis=1)
        a = _network_convection(nodes[:ring_first], edges[wing_only], gamma[wing_only], first, radii[wing_only],
                                WakeConfig(**kw), self.inflow, 0.0, None)
        b = _network_convection(nodes[:ring_first], edges[wing_only], gamma[wing_only], first, radii[wing_only],
                                WakeConfig(**kw, frozen_rows_drift='ambient_rings'), self.inflow, 0.0, None)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
