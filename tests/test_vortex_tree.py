"""Tree (Barnes-Hut type) Biot-Savart backend: V0 controlled checks (OF-0006-accel20261001).

The tree is an opt-in approximation. These tests check that it converges to
the direct sum as the opening ratio decreases and the expansion order rises,
that excluded and singular pairs never enter the far field, and that results
do not depend on the thread count.
"""
import importlib.util
import unittest

import numpy as np

from openfawe.aero.vortex import CoreModel, FilamentSingularity, TreeSettings, segments_velocity

HAS_NUMBA = importlib.util.find_spec("numba") is not None


def ring(radius=1.0, count=64, centre=(0., 0., 0.)):
    angle = np.linspace(0., 2*np.pi, count, endpoint=False)
    nodes = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), np.zeros(count))) + centre
    edges = np.column_stack((np.arange(count), (np.arange(count)+1) % count))
    return nodes, edges


def sheet(seed=11, rows=24, span=17):
    """A wavy, wake-like ring lattice with shed circulation differences."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0., 12., rows)[:, None]*np.ones((1, span))
    y = np.ones((rows, 1))*np.linspace(-4., 4., span)[None, :]
    z = .3*np.sin(.7*x)*np.cos(.5*y) + .05*rng.normal(size=x.shape)
    nodes = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    edges, gamma = [], []
    strength = np.cos(np.linspace(-1.3, 1.3, span-1))[None, :]*np.exp(-.05*np.arange(rows-1))[:, None]
    for i in range(rows-1):
        for j in range(span-1):
            a, b, c, d = i*span+j, i*span+j+1, (i+1)*span+j+1, (i+1)*span+j
            for p, q in ((a, b), (b, c), (c, d), (d, a)):
                edges.append((p, q))
                gamma.append(strength[i, j])
    return nodes, np.asarray(edges), np.asarray(gamma)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TreeAccuracy(unittest.TestCase):
    def test_converges_to_direct_with_theta_and_order(self):
        nodes, edges, gamma = sheet()
        rng = np.random.default_rng(2)
        points = nodes + .07*rng.normal(size=nodes.shape)
        direct = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN,
                                   core_radius=.1, backend="numba")
        scale = np.max(np.linalg.norm(direct, axis=1))
        errors = {}
        for order in (0, 1, 2):
            for theta in (.6, .4, .25, .15):
                out = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN,
                                        core_radius=.1, backend="tree",
                                        tree=TreeSettings(theta=theta, order=order, leaf_size=8))
                errors[order, theta] = np.max(np.linalg.norm(out-direct, axis=1))/scale
        for order in (0, 1, 2):
            sequence = [errors[order, theta] for theta in (.6, .4, .25, .15)]
            self.assertTrue(all(b < a for a, b in zip(sequence, sequence[1:])), msg=(order, sequence))
        for theta in (.4, .25, .15):
            self.assertLess(errors[2, theta], errors[1, theta])
            self.assertLess(errors[1, theta], errors[0, theta])
        self.assertLess(errors[2, .15], 2e-4)

    def test_tiny_theta_is_the_direct_sum(self):
        nodes, edges, gamma = sheet(rows=10, span=9)
        points = nodes + [0., 0., .2]
        direct = segments_velocity(points, nodes, edges, gamma, model=CoreModel.RANKINE,
                                   core_radius=.05, backend="numba")
        out = segments_velocity(points, nodes, edges, gamma, model=CoreModel.RANKINE, core_radius=.05,
                                backend="tree", tree=TreeSettings(theta=1e-9, order=2, leaf_size=4))
        scale = np.max(np.linalg.norm(direct, axis=1))
        np.testing.assert_allclose(out, direct, rtol=0, atol=1e-13*scale)

    def test_ring_dipole_far_field(self):
        nodes, edges = ring(count=128)
        far = np.array([[6., 1., 4.], [0., 0., 9.], [-7., 3., -2.]])
        direct = segments_velocity(far, nodes, edges, 1., backend="numba")
        scale = np.linalg.norm(direct, axis=1)
        errors = []
        for order in (0, 1, 2):
            out = segments_velocity(far, nodes, edges, 1., backend="tree",
                                    tree=TreeSettings(theta=.5, order=order, leaf_size=200))
            errors.append(np.linalg.norm(out-direct, axis=1)/scale)
        # A closed ring has no monopole: order 0 misses the whole dipole field.
        np.testing.assert_allclose(errors[0], 1., atol=1e-12)
        # A centred circular ring has no quadrupole either; the first omitted
        # term is the octupole, relative size about (R/d)^2.
        distance = np.linalg.norm(far, axis=1)
        self.assertTrue(np.all(errors[1] < 2./distance**2))
        np.testing.assert_allclose(errors[2], errors[1], rtol=1e-6)
        # An off-centre, tilted, elliptical loop has a quadrupole: order 2 helps.
        angle = np.linspace(0., 2*np.pi, 96, endpoint=False)
        loop = np.column_stack((1.4*np.cos(angle)+.3, .8*np.sin(angle), .4*np.sin(2*angle)+.2*np.cos(angle)))
        edges = np.column_stack((np.arange(96), (np.arange(96)+1) % 96))
        direct = segments_velocity(far, loop, edges, 1., backend="numba")
        first, second = [np.max(np.linalg.norm(segments_velocity(far, loop, edges, 1., backend="tree",
                                tree=TreeSettings(theta=.5, order=order, leaf_size=200))-direct, axis=1)
                                / np.linalg.norm(direct, axis=1)) for order in (1, 2)]
        self.assertLess(second, first/3)

    def test_exclusions_and_incident_targets_stay_near_field(self):
        nodes, edges, gamma = sheet(rows=12, span=9)
        incident = np.asarray([(n, s) for s, (a, b) in enumerate(edges) for n in (a, b)], dtype=np.int64)
        direct = segments_velocity(nodes, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=.1,
                                   exclusions=incident, backend="numba")
        exact = segments_velocity(nodes, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=.1,
                                  exclusions=incident, backend="tree",
                                  tree=TreeSettings(theta=1e-9, order=2, leaf_size=4))
        scale = np.max(np.linalg.norm(direct, axis=1))
        np.testing.assert_allclose(exact, direct, rtol=0, atol=1e-13*scale)
        approx = segments_velocity(nodes, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=.1,
                                   exclusions=incident, backend="tree",
                                   tree=TreeSettings(theta=.3, order=2, leaf_size=4))
        self.assertLess(np.max(np.linalg.norm(approx-direct, axis=1))/scale, 1e-2)
        # A far excluded segment must never leak through a node expansion:
        # give it a huge circulation and compare with the direct exclusion.
        loud = gamma.copy()
        loud[-1] = 1e6
        far_pair = np.array([[0, len(edges)-1]], dtype=np.int64)
        silent = gamma.copy()
        silent[-1] = 0.
        settings = TreeSettings(theta=.6, order=2, leaf_size=4)
        # Opening decisions depend on geometry and exclusions only, so the two
        # traversals are identical and the excluded strength must not matter.
        tree = segments_velocity(nodes[:1], nodes, edges, loud, exclusions=far_pair, singularity="zero",
                                 backend="tree", tree=settings)
        quiet = segments_velocity(nodes[:1], nodes, edges, silent, exclusions=far_pair, singularity="zero",
                                  backend="tree", tree=settings)
        np.testing.assert_array_equal(tree, quiet)
        # Without the exclusion the loud segment is clearly present.
        leaked = segments_velocity(nodes[:1], nodes, edges, loud, singularity="zero", backend="tree", tree=settings)
        self.assertGreater(np.linalg.norm(leaked-quiet), 1e3*np.linalg.norm(quiet))

    def test_singular_and_unsupported_cores(self):
        nodes, edges, gamma = sheet(rows=6, span=5)
        point = .5*(nodes[edges[3, 0]] + nodes[edges[3, 1]])[None]
        settings = TreeSettings(theta=.4, order=2, leaf_size=2)
        with self.assertRaises(FilamentSingularity) as ref:
            segments_velocity(point, nodes, edges, gamma)
        with self.assertRaises(FilamentSingularity) as out:
            segments_velocity(point, nodes, edges, gamma, backend="tree", tree=settings)
        self.assertEqual(str(ref.exception), str(out.exception))
        for model in (CoreModel.VATISTAS, CoreModel.OFFSET, CoreModel.ROSENHEAD_MOORE):
            with self.assertRaises(ValueError):
                segments_velocity(point + 1., nodes, edges, gamma, model=model, core_radius=.1,
                                  backend="tree", tree=settings)

    def test_thread_count_bitwise(self):
        import numba
        nodes, edges, gamma = sheet()
        points = nodes + [0., 0., .3]
        settings = TreeSettings(theta=.35, order=2, leaf_size=16)
        base = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=.1,
                                 backend="tree", tree=settings)
        previous = numba.get_num_threads()
        try:
            for threads in sorted({1, numba.config.NUMBA_NUM_THREADS}):
                numba.set_num_threads(threads)
                again = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN,
                                          core_radius=.1, backend="tree", tree=settings)
                np.testing.assert_array_equal(base.view(np.uint64), again.view(np.uint64))
        finally:
            numba.set_num_threads(previous)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class FarFieldForm(unittest.TestCase):
    def test_contracted_expansion_equals_term_by_term_form(self):
        from openfawe.aero.vortex_tree import (far_velocity, multipole_velocity_explicit,
                                               node_contractions)
        rng = np.random.default_rng(5)
        count = 7
        m0 = rng.normal(size=(count, 3))
        m1 = rng.normal(size=(count, 3, 3))
        m2 = rng.normal(size=(count, 3, 3, 3))
        m2 = .5*(m2 + m2.transpose(0, 1, 3, 2))  # moments are symmetric in the last two indices
        w1, t2, w2 = node_contractions(m1, m2)
        for node in range(count):
            for order in (0, 1, 2):
                r = rng.normal(size=3)*3.
                expected = np.zeros(3)
                multipole_velocity_explicit(r[0], r[1], r[2], node, m0, m1, m2, order, expected)
                actual = np.array(far_velocity(r[0], r[1], r[2], node, m0, m1, m2, w1, t2, w2, order))
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14*np.abs(expected).max())


class TreeScope(unittest.TestCase):
    def test_tree_only_for_wake_on_wake_sums(self):
        # Wing-point induction and attached-ring operators stay direct, so
        # loads use the exact induced velocity of the current wake geometry.
        from openfawe.aero.wake import WakeConfig, kernel_options
        settings = TreeSettings(theta=.3, order=2, leaf_size=32)
        config = WakeConfig(velocity_backend="tree", tree=settings)
        self.assertEqual(kernel_options(config, large=True), dict(backend="tree", tree=settings))
        self.assertEqual(kernel_options(config), dict(backend="numba"))
        self.assertEqual(kernel_options(WakeConfig()), dict(backend="numpy"))
        self.assertEqual(kernel_options(WakeConfig(), large=True), dict(backend="numpy"))
        self.assertEqual(kernel_options(WakeConfig(velocity_backend="numba"), large=True),
                         dict(backend="numba"))
        with self.assertRaises(ValueError):
            WakeConfig(velocity_backend="tree")
        with self.assertRaises(ValueError):
            WakeConfig(velocity_backend="tree", tree=settings, core_model=CoreModel.VATISTAS)


if __name__ == "__main__":
    unittest.main()
