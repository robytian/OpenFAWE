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


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class BatchedCrossClass(unittest.TestCase):
    """Node 3 contract addendum 9: the cross-class blocks of OF-0010 P2 in one pass per class equal the per-panel loop
    (one pack_rings + segments_velocity call per panel, the same compiled kernel) bit for bit, at the closure and the
    load points, on the M600's P2 surfaces."""

    def reference(self, wing, rows, points, config, surfaces):
        from openfawe.aero.surface_cores import CROSS_MODEL
        from openfawe.aero.wake import pack_rings
        matrix = np.full((wing.panels, wing.panels, 3), np.nan)
        width = wing.node_count
        for panel in range(wing.panels):
            targets = np.flatnonzero(surfaces.panel_class != surfaces.panel_class[panel])
            strengths = np.zeros((len(rows) - 1, wing.panels))
            strengths[:, panel] = 1.
            nodes, edges, gamma = pack_rings(rows, strengths, wing.panel_edges)
            radii = np.maximum(edge_radii(wing, edges, None, config.core_radius_m), surfaces.edge_radius(edges, width))
            matrix[targets, panel] = segments_velocity(points[targets], nodes, edges, gamma, model=CROSS_MODEL,
                                                       core_radius=radii, backend="numba")
        return matrix

    def test_bitwise_on_the_m600_p2_surfaces(self):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root/"docs/tasks/OF-0008-g5control20261006"))
        from openfawe.flight import platform_surfaces as PF
        from openfawe.aero.vsm import geometry
        from openfawe.aero.vsm_wake import _batched_cross_class
        p1 = PF.default_p1(root)
        surfaces = PF.p2_surfaces(p1)
        rotation = np.array([[0.96, -0.2, 0.19], [0.21, 0.98, 0.0], [-0.186, 0.04, 0.98]])
        rotation, _ = np.linalg.qr(rotation)
        wing = p1.period(np.radians([3., -2., 1., 0., 4., -5., 2., 1.])).assembly.transformed(rotation, np.array([5., -3., 120.]))
        rng = np.random.default_rng(4)
        config = WakeConfig(core_radius_m=0.15, velocity_backend="numba")
        for downstream in ((), (wing.trailing_edge + [0.6, 0.1, -0.05] + 0.02*rng.normal(size=wing.trailing_edge.shape),)):
            rows = [wing.bound, wing.trailing_edge, *downstream]
            for points in (geometry(wing)[0], wing.cp):
                expected = self.reference(wing, rows, points, config, surfaces)
                actual = np.full_like(expected, np.nan)
                self.assertTrue(_batched_cross_class(actual, wing, rows, points, config, None, surfaces))
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                cross = ~np.isnan(expected)
                np.testing.assert_array_equal(actual[cross].view(np.uint64), expected[cross].view(np.uint64))
                self.assertGreater(int(cross.sum()), 0)

    def test_singular_pair_writes_nothing(self):
        from openfawe.aero.vsm_wake import _batched_cross_class
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        from openfawe.flight import platform_surfaces as PF
        p1 = PF.default_p1(root)
        surfaces = PF.p2_surfaces(p1)
        wing = p1.period(np.zeros(8)).assembly
        config = WakeConfig(core_radius_m=0.15, velocity_backend="numba")
        rows = [wing.bound, wing.trailing_edge]
        cls = surfaces.panel_class
        target = int(np.flatnonzero(cls != cls[0])[0])
        points = wing.cp.copy()
        points[target] = 0.5*(wing.bound[wing.panel_edges[0][0]] + wing.bound[wing.panel_edges[0][1]])   # on panel 0's bound
        matrix = np.zeros((wing.panels, wing.panels, 3))
        self.assertFalse(_batched_cross_class(matrix, wing, rows, points, config, None, surfaces))
        self.assertFalse(np.any(matrix))


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class BatchedHubWinds(unittest.TestCase):
    """Node 3 contract addendum 9 supplement 4: the network's induction at several hubs in one sum equals the per-disc
    ``hub_apparent_wind`` (zero ambient, zero hub velocity, no rings) bit for bit."""

    def test_bitwise_against_the_per_disc_calls(self):
        import types
        from openfawe.aero.methods import AeroBackend
        from openfawe.aero.vortex import TreeSettings
        from openfawe.aero.wing import SolveSettings
        from openfawe.flight.kinematics import circular_motion
        from openfawe.flight.prescribed import run_prescribed
        from openfawe.rotor.coupled import hub_apparent_wind, hub_wing_winds
        wing, polar = tapered_wing(panels=9), SectionPolar.linear(limit_deg=60.)
        steps = 6
        motion = circular_motion(np.arange(steps+1)*0.02, 120.0, 45.0, 9.0, 3.0)
        for options in (dict(velocity_backend="numba"),
                        dict(velocity_backend="tree", tree=TreeSettings(theta=0.2, order=2, leaf_size=32))):
            backend = AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=0.02, max_rows=steps, core_radius_m=0.15,
                                                                   solve=SolveSettings(solver="newton", max_iterations=60),
                                                                   **options))
            run = run_prescribed(backend, wing, polar, motion, [9., 0., 0.], keep_states=(steps,))
            state = run.states[steps]
            pose = motion.pose(steps*0.02)
            centres = pose.position_m + (pose.rotation@np.array([[1.5, y, 0.4] for y in (-3.5, -1.2, 1.2, 3.5)]).T).T
            batched = hub_wing_winds(state, state.config, centres)
            for i, centre in enumerate(centres):
                single = hub_apparent_wind(state, state.config, types.SimpleNamespace(centre_m=centre), i, (0.0, 0.0, 0.0))
                np.testing.assert_array_equal(batched[i].view(np.uint64), single.view(np.uint64))


if __name__ == "__main__":
    unittest.main()
