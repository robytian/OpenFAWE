"""Batched VSM attached-ring operator must equal the per-panel reference bitwise.

For the accelerated backends, vsm_wake_influence assembles the unit-current
operator in one kernel pass (vortex_numba.pair_matrix + assemble_columns).
The reference is one pack_rings + segments_velocity call per panel with the
same compiled kernel; the two must agree bit for bit at the closure points
(with the self correction) and at the load points (bound row dropped).
"""
import importlib.util
import unittest

import numpy as np

from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.cores import edge_radii

HAS_NUMBA = importlib.util.find_spec("numba") is not None


def tapered_wing(panels=11, seed=3):
    rng = np.random.default_rng(seed)
    span = np.linspace(-4., 4., panels + 1)
    chord = 1.2 - 0.05 * np.abs(span)
    leading = np.column_stack((-0.25 * chord + 0.1 * np.sin(span), span, 0.15 * np.cos(0.4 * span)))
    trailing = leading + np.column_stack((chord, np.zeros_like(chord), 0.02 * rng.normal(size=chord.shape)))
    return Wing(leading, trailing)


def reference_operator(wing, rows, config, cores, at_load_points, backend):
    from openfawe.aero.wake import pack_rings
    from openfawe.aero.vsm import geometry, vsm_self_correction
    points = wing.cp if at_load_points else geometry(wing)[0]
    matrix = np.zeros((wing.panels, wing.panels, 3))
    for panel in range(wing.panels):
        strengths = np.zeros((len(rows) - 1, wing.panels))
        strengths[:, panel] = 1.
        nodes, edges, gamma = pack_rings(rows, strengths, wing.panel_edges)
        if at_load_points:
            retained = ~np.all(edges < wing.node_count, axis=1)
            edges, gamma = edges[retained], gamma[retained]
        matrix[:, panel] = segments_velocity(points, nodes, edges, gamma, model=config.core_model,
                                             core_radius=edge_radii(wing, edges, cores, config.core_radius_m),
                                             backend=backend)
    if not at_load_points:
        index = np.arange(wing.panels)
        matrix[index, index] -= vsm_self_correction(wing, points)
    return matrix


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class BatchedOperator(unittest.TestCase):
    def test_bitwise_against_per_panel_reference(self):
        from openfawe.aero.vsm_wake import vsm_wake_influence
        wing = tapered_wing()
        rng = np.random.default_rng(9)
        downstream = [wing.trailing_edge + [0.9, 0., 0.05] + 0.03 * rng.normal(size=wing.trailing_edge.shape)]
        for core_model in (CoreModel.LAMB_OSEEN, CoreModel.NONE):
            config = WakeConfig(core_model=core_model, core_radius_m=0.12, velocity_backend="numba")
            for rows_used in ((), tuple(downstream)):
                for at_load in (False, True):
                    rows = [wing.bound, wing.trailing_edge, *rows_used]
                    expected = reference_operator(wing, rows, config, None, at_load, "numba")
                    actual = vsm_wake_influence(wing, rows_used, config=config, at_load_points=at_load)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    # And the NumPy-kernel reference agrees to kernel round-off.
                    numpy_ref = reference_operator(wing, rows, WakeConfig(core_model=core_model, core_radius_m=0.12),
                                                   None, at_load, "numpy")
                    scale = np.max(np.abs(numpy_ref))
                    np.testing.assert_allclose(actual, numpy_ref, rtol=0, atol=1e-12 * scale)

    def test_singular_pair_falls_back_to_reference_exception(self):
        from openfawe.aero.vsm_wake import vsm_wake_influence
        from openfawe.aero.vortex import FilamentSingularity
        wing = tapered_wing(panels=5)
        # A downstream row passing exactly through a closure point makes the
        # reference raise; the batched path must not return a matrix instead.
        from openfawe.aero.vsm import geometry
        cp = geometry(wing)[0]
        row = wing.trailing_edge.copy()
        row[2] = cp[2] + (row[2] - wing.trailing_edge[2])
        row[3] = cp[2] + 2. * (row[3] - row[2])
        config = WakeConfig(velocity_backend="numba", core_radius_m=0.05)
        with self.assertRaises((FilamentSingularity, ValueError)):
            vsm_wake_influence(wing, [row], config=config)


if __name__ == "__main__":
    unittest.main()
