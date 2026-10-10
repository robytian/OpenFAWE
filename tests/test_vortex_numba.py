"""Compiled direct Biot-Savart backend against the NumPy reference (OF-0006-accel20261001).

Acceptance A1 (frozen before implementation): for every point,
|u_numba - u_numpy| <= 1e-12 * sum_s |u_s|, with identical exception type and
message for rejected inputs. A2: results do not depend on the thread count.
"""
import importlib.util
import unittest

import numpy as np

from openfawe.aero.vortex import CoreModel, FilamentSingularity, TreeSettings, segments_velocity
from openfawe.aero.wake import WakeConfig, kernel_options
from openfawe.aero.wing import Wing, SectionPolar, wake_config_metadata
from openfawe.aero.methods import AeroBackend

HAS_NUMBA = importlib.util.find_spec("numba") is not None
TOLERANCE = 1e-12


def contribution_scale(points, nodes, edges, gamma, model, radius, exclusions=None, singularity="raise"):
    """sum_s |u_s| per point, from the NumPy reference, one segment at a time."""
    gamma = np.broadcast_to(np.asarray(gamma, float), (len(edges),))
    radius = np.broadcast_to(np.asarray(radius, float), (len(edges),))
    scale = np.zeros(len(points))
    for s, edge in enumerate(edges):
        keep = np.ones(len(points), bool)
        if exclusions is not None and len(exclusions):
            keep[exclusions[exclusions[:, 1] == s, 0]] = False
        if not keep.any():
            continue
        u = segments_velocity(points[keep], nodes, edge[None], gamma[s:s+1], model=model,
                              core_radius=radius[s:s+1], singularity=singularity)
        scale[keep] += np.linalg.norm(u, axis=1)
    return scale


def random_network(seed, nodes=60, segments=90, points=70):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(nodes, 3))
    edges = rng.integers(0, nodes, size=(segments, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    gamma = rng.normal(size=len(edges))
    gamma[::7] = 0.
    radius = rng.uniform(.01, .6, size=len(edges))
    targets = rng.normal(size=(points, 3))
    # Points close to segment lines: interior near-axis, exterior collinear, offsets.
    a, b = xyz[edges[:5, 0]], xyz[edges[:5, 1]]
    normal = np.cross(b - a, [0.3, -0.2, 0.9])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    near = [a + .4*(b - a) + 1e-9*normal, a + .4*(b - a) + 1e-13*normal,
            a + 1.7*(b - a), a - .5*(b - a), a + .5*(b - a) + .05*normal]
    targets = np.vstack([targets, *near])
    return xyz, edges, gamma, radius, targets


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class CompiledKernelEquivalence(unittest.TestCase):
    def check(self, points, nodes, edges, gamma, model, radius, **kwargs):
        ref = segments_velocity(points, nodes, edges, gamma, model=model, core_radius=radius, **kwargs)
        out = segments_velocity(points, nodes, edges, gamma, model=model, core_radius=radius,
                                backend="numba", **kwargs)
        scale = contribution_scale(points, nodes, edges, gamma, model, radius,
                                   kwargs.get("exclusions"), kwargs.get("singularity", "raise"))
        error = np.linalg.norm(out - ref, axis=1)
        self.assertTrue(np.all(error <= TOLERANCE * scale), msg=f"model {model}: {np.max(error - TOLERANCE*scale)}")
        return ref, out

    def test_all_core_models_scalar_and_per_segment(self):
        for seed in (3, 4):
            nodes, edges, gamma, radius, points = random_network(seed)
            for model in CoreModel:
                for core in (0., .2, radius):
                    with self.subTest(seed=seed, model=model.name, core=np.ndim(core)):
                        self.check(points, nodes, edges, gamma, model, core, singularity="zero")

    def test_exclusions_match_reference(self):
        nodes, edges, gamma, radius, _ = random_network(5)
        # Evaluate at the network's own nodes; exclude every incident segment.
        incident = [(n, s) for s, (a, b) in enumerate(edges) for n in (a, b)]
        rng = np.random.default_rng(9)
        extra = [(int(p), int(s)) for p, s in zip(rng.integers(0, len(nodes), 30),
                                                    rng.integers(0, len(edges), 30))]
        pairs = np.asarray(incident + extra + incident[:5], dtype=np.int64)  # duplicates allowed
        for model in (CoreModel.NONE, CoreModel.LAMB_OSEEN, CoreModel.ROSENHEAD_MOORE):
            with self.subTest(model=model.name):
                self.check(nodes, nodes, edges, gamma, model, radius, exclusions=pairs)

    def test_singular_point_raises_identically(self):
        nodes = np.array([[0., 0., 0.], [1., 0., 0.]])
        point = np.array([[.3, 0., 0.]])
        for gamma in (1., 0.):
            with self.assertRaises(FilamentSingularity) as ref:
                segments_velocity(point, nodes, [[0, 1]], gamma)
            with self.assertRaises(FilamentSingularity) as out:
                segments_velocity(point, nodes, [[0, 1]], gamma, backend="numba")
            self.assertEqual(str(ref.exception), str(out.exception))
        zero = segments_velocity(point, nodes, [[0, 1]], 1., singularity="zero", backend="numba")
        np.testing.assert_array_equal(zero, 0.)

    def test_float_range_failure_raises_identically(self):
        nodes = np.array([[0., 0., 0.], [1e-160, 0., 0.]])
        point = np.array([[0., 1e160, 0.]])
        with self.assertRaises(ValueError) as ref:
            segments_velocity(point, nodes, [[0, 1]], 1.)
        with self.assertRaises(ValueError) as out:
            segments_velocity(point, nodes, [[0, 1]], 1., backend="numba")
        self.assertEqual(str(ref.exception), str(out.exception))
        # A subnormal core makes the core ratio overflow: the reference rejects
        # it under np.errstate, so the compiled path must not return a value.
        nodes = np.array([[0., 0., 0.], [1., 0., 0.]])
        point = np.array([[.5, 1., 0.]])
        for model in (CoreModel.RANKINE, CoreModel.LAMB_OSEEN, CoreModel.VATISTAS):
            with self.assertRaises(ValueError) as ref:
                segments_velocity(point, nodes, [[0, 1]], 1., model=model, core_radius=1e-310)
            with self.assertRaises(ValueError) as out:
                segments_velocity(point, nodes, [[0, 1]], 1., model=model, core_radius=1e-310,
                                  backend="numba")
            self.assertEqual(str(ref.exception), str(out.exception))

    def test_serial_parallel_and_thread_count_bitwise(self):
        import numba
        from openfawe.aero import vortex_numba
        nodes, edges, gamma, radius, points = random_network(6, nodes=200, segments=400, points=600)
        base = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN,
                                 core_radius=radius, singularity="zero", backend="numba")
        previous = numba.get_num_threads()
        old_limit = vortex_numba.PARALLEL_MIN_PAIRS
        try:
            for limit in (0, 10**12):
                vortex_numba.PARALLEL_MIN_PAIRS = limit
                for threads in sorted({1, numba.config.NUMBA_NUM_THREADS}):
                    numba.set_num_threads(threads)
                    again = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN,
                                              core_radius=radius, singularity="zero", backend="numba")
                    np.testing.assert_array_equal(base.view(np.uint64), again.view(np.uint64))
        finally:
            vortex_numba.PARALLEL_MIN_PAIRS = old_limit
            numba.set_num_threads(previous)

    def test_threaded_driver_is_a_distinct_cached_function(self):
        # Regression (2026-10-01 local benchmark): Numba's disk cache is keyed by the
        # Python function and signature, not by ``parallel``; compiling one source
        # twice made both drivers load the same (serial) cache entry.
        from openfawe.aero import vortex_numba
        threaded, serial = vortex_numba.sum_segments, vortex_numba.sum_segments_serial
        self.assertTrue(threaded.targetoptions.get("parallel"))
        self.assertFalse(serial.targetoptions.get("parallel"))
        self.assertIsNot(threaded.py_func, serial.py_func)
        self.assertNotEqual(threaded.py_func.__qualname__, serial.py_func.__qualname__)
        self.assertNotEqual(threaded.py_func.__code__.co_firstlineno,
                            serial.py_func.__code__.co_firstlineno)


class BackendSelection(unittest.TestCase):
    def test_config_validation_and_legacy_metadata(self):
        legacy = wake_config_metadata(WakeConfig())
        self.assertNotIn("velocity_backend", legacy)
        self.assertNotIn("tree", legacy)
        self.assertEqual(wake_config_metadata(WakeConfig(velocity_backend="numba"))["velocity_backend"], "numba")
        with self.assertRaises(ValueError):
            WakeConfig(velocity_backend="gpu")
        with self.assertRaises(ValueError):
            WakeConfig(velocity_backend="tree")
        with self.assertRaises(ValueError):
            WakeConfig(velocity_backend="numba", tree=TreeSettings())
        with self.assertRaises(ValueError):
            segments_velocity([[0., 1., 0.]], [[0., 0., 0.], [1., 0., 0.]], [[0, 1]], 1., backend="cuda")
        tree = WakeConfig(velocity_backend="tree", tree=TreeSettings())
        self.assertEqual(kernel_options(tree), dict(backend="numba"))
        self.assertEqual(kernel_options(tree, large=True), dict(backend="tree", tree=TreeSettings()))
        self.assertEqual(kernel_options(WakeConfig(), large=True), dict(backend="numpy"))
        for bad in (dict(theta=1.), dict(theta=0.), dict(order=3), dict(leaf_size=0), dict(core_clearance=-1.)):
            with self.assertRaises(ValueError):
                TreeSettings(**bad)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class CompiledRunEquivalence(unittest.TestCase):
    def test_short_vsm_and_ll_free_wake_histories(self):
        y = np.linspace(-3., 3., 7)
        le = np.column_stack((.1*np.abs(y), y, .05*y**2))
        wing = Wing(le, le + [[1., 0., 0.]])
        polar = SectionPolar.linear()
        wind = np.array([15., 0., 1.2])
        for method in ("vsm_free_wake", "free_wake"):
            histories = {}
            for backend in ("numpy", "numba"):
                config = WakeConfig(dt_s=.05, max_rows=6, core_radius_m=.05, velocity_backend=backend)
                aero = AeroBackend(method, wake=config)
                state = aero.start(wing, polar, wind)
                rows = []
                for step in range(1, 7):
                    state = aero.advance(state, polar, wind, next_time_s=step*.05)
                    rows.append((state.solution.circulation.copy(), state.solution.force.copy(),
                                 state.free_rows.copy()))
                histories[backend] = rows
            for (g0, f0, r0), (g1, f1, r1) in zip(histories["numpy"], histories["numba"]):
                np.testing.assert_allclose(g1, g0, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(f1, f0, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(r1, r0, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
