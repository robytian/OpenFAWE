"""Lumped-mass tether (OF-0007-tether20261001): analytic controls D1-D7 of contract.zh.md.

T1 catenary shape and end tension, T2 elastic extension, T3 taut-string
frequencies, T4 pendulum period, T5/T6 energy balance, T7 node convergence,
T8 action/reaction at the end points, T9 honest failure on an unstable step.
"""
import unittest

import numpy as np

from openfawe.tether.lumped_line import (LumpedTether, TetherProperties, TetherSettings,
                                         fixed_ends)


def catenary_reference(span, length, g=9.81):
    """Inextensible catenary between equal-height supports: returns (a, sag, horizontal tension / w)."""
    from math import sinh, cosh
    # Solve 2 a sinh(span / (2a)) = length for a (bisection).
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 2 * mid * sinh(span / (2 * mid)) > length:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    sag = a * (cosh(span / (2 * a)) - 1.0)
    return a, sag


def chain_shape(line, positions, start, end):
    return line.chain(positions, start, end)


class CatenaryAndExtension(unittest.TestCase):
    def test_T1_catenary_shape_and_tension(self):
        """Exact static solution of the lumped chain, then its convergence to the continuous catenary.

        With link forces F_k = F_0 + k m g e_z (node balance) and link vectors
        L0_k (1 + |F_k| / EA) F_k / |F_k|, the two unknowns (H, V0) of F_0 follow
        from the end-point closure. The model's node forces must vanish there
        (verifies the spring/gravity assembly); sag and H/w must converge to the
        inextensible catenary as N grows.
        """
        from scipy.optimize import fsolve
        props = TetherProperties(length_m=100.0, linear_density_kg_m=0.917, tensile_stiffness_N=18e6,
                                 bending_stiffness_Nm2=0.0, section_drag_coeff=0.0)
        start, end = np.array([0., 0., 0.]), np.array([80., 0., 0.])
        a, sag = catenary_reference(80.0, 100.0)
        w_line = props.linear_density_kg_m * 9.81
        results = {}
        for n in (10, 20, 40, 80):
            line = LumpedTether(props, TetherSettings(num_nodes=n))
            w = line.node_mass * 9.81

            def links(hv):
                forces = np.array([[hv[0], 0.0, hv[1] + k * w] for k in range(n + 1)])
                magnitude = np.linalg.norm(forces, axis=1)
                return (line.natural * (1 + magnitude / props.tensile_stiffness_N) / magnitude)[:, None] * forces

            def closure(hv):
                total = np.sum(links(hv), axis=0)
                return [total[0] - 80.0, total[2]]
            hv = fsolve(closure, [a * w_line, -0.5 * n * w], xtol=1e-14)
            self.assertLess(np.max(np.abs(closure(hv))), 1e-9)
            points = start + np.cumsum(links(hv), axis=0)
            x = points[:-1]
            np.testing.assert_allclose(points[-1], end, atol=1e-9)
            out = line.forces(x, np.zeros_like(x), start, np.zeros(3), end, np.zeros(3))
            self.assertLess(np.max(np.abs(out["node"])), 1e-9 * np.max(out["tension"]))   # exact equilibrium
            np.testing.assert_allclose(out["start"], [hv[0], 0.0, hv[1]], rtol=1e-10)
            results[n] = dict(sag=-float(np.min(points[:, 2])), h_over_w=hv[0] / w_line)
        errors = {n: abs(r["sag"] - sag) / sag for n, r in results.items()}
        self.assertLess(errors[20], 5e-3, msg=(results, sag, a))
        self.assertLess(errors[40], errors[20])
        self.assertLess(errors[80], errors[40])
        self.assertLess(abs(results[40]["h_over_w"] - a) / a, 5e-3, msg=(results[40], a))

    def test_T2_elastic_extension_without_gravity(self):
        props = TetherProperties(length_m=100.0, tensile_stiffness_N=1e6, bending_stiffness_Nm2=0.0,
                                 section_drag_coeff=0.0)
        line = LumpedTether(props, TetherSettings(num_nodes=8, gravity_m_s2=0.0))
        strain = 2e-3
        start, end = np.zeros(3), np.array([100.0 * (1 + strain), 0., 0.])
        x, v = line.straight_state(start, end)
        out = line.forces(x, v, start, np.zeros(3), end, np.zeros(3))
        np.testing.assert_allclose(out["tension"], 1e6 * strain, rtol=1e-12)
        np.testing.assert_allclose(out["node"], 0.0, atol=1e-12 * 1e6 * strain)
        np.testing.assert_allclose(out["start"], [1e6 * strain, 0., 0.], rtol=1e-12)
        np.testing.assert_allclose(out["end"], [-1e6 * strain, 0., 0.], rtol=1e-12)


class Dynamics(unittest.TestCase):
    def test_T3_taut_string_frequencies(self):
        # Pre-strained string without gravity, bending or drag; first mode f1 = sqrt(T/mu) / (2L).
        length, strain = 50.0, 1e-3
        props = TetherProperties(length_m=length, linear_density_kg_m=0.917, tensile_stiffness_N=1e6,
                                 bending_stiffness_Nm2=0.0, section_drag_coeff=0.0)
        n = 40
        line = LumpedTether(props, TetherSettings(num_nodes=n, gravity_m_s2=0.0, longitudinal_damping_ratio=0.0))
        start, end = np.zeros(3), np.array([length * (1 + strain), 0., 0.])
        x, v = line.straight_state(start, end)
        tension = 1e6 * strain
        span_x = x[:, 0] / end[0]
        for mode in (1, 2, 3):
            xi = x.copy()
            xi[:, 2] = 0.05 * np.sin(mode * np.pi * span_x)
            vi = np.zeros_like(v)
            expected = mode / (2 * length) * np.sqrt(tension / props.linear_density_kg_m)
            dt = 0.25 * line.longitudinal_period()
            steps = int(round(3.0 / expected / dt))
            history = []
            time = 0.0
            for _ in range(steps):
                xi, vi = line.step(time, xi, vi, dt, fixed_ends(start, end))
                time += dt
                history.append(float(xi[n // 2 if mode % 2 else n // 4, 2]))
            signal = np.asarray(history) - np.mean(history)
            spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
            freqs = np.fft.rfftfreq(len(signal), dt)
            measured = freqs[np.argmax(spectrum[1:]) + 1]
            # Lumped masses on a string: discrete dispersion ~ (pi mode / N)^2 / 24.
            self.assertLess(abs(measured - expected) / expected, 0.02 + (np.pi * mode / n)**2 / 12, msg=(mode, measured, expected))

    def test_T4_pendulum_period(self):
        # One heavy node on a short stiff line swinging as a pendulum (small amplitude).
        props = TetherProperties(length_m=2.0, linear_density_kg_m=5.0, tensile_stiffness_N=1e7,
                                 bending_stiffness_Nm2=0.0, section_drag_coeff=0.0)
        line = LumpedTether(props, TetherSettings(num_nodes=1, longitudinal_damping_ratio=0.0))
        # Only the start is a pivot: the end point sits 0.5 m below the node's rest position, so the
        # second half segment (natural length 1 m) stays slack and carries no force.
        start, end = np.zeros(3), np.array([0., 0., -1.5])
        x = np.array([[np.sin(0.05), 0., -np.cos(0.05)]])     # node at 1.0 m (half segment) from the pivot
        v = np.zeros_like(x)
        ends = fixed_ends(start, end)
        dt = 0.1 * line.longitudinal_period()
        g = 9.81
        expected_period = 2 * np.pi * np.sqrt(1.0 / g) * (1 + 0.05**2 / 16)
        time, crossings, previous = 0.0, [], x[0, 0]
        while time < 4 * expected_period:
            x, v = line.step(time, x, v, dt, ends)
            time += dt
            if previous < 0 <= x[0, 0]:
                crossings.append(time)
            previous = x[0, 0]
        self.assertGreaterEqual(len(crossings), 3)
        period = np.mean(np.diff(crossings))
        self.assertLess(abs(period - expected_period) / expected_period, 0.01, msg=(period, expected_period))

    def test_T5_energy_conservation_without_damping(self):
        # Pre-strained taut string with a transverse initial shape (smooth motion, no slack
        # snapping): RK4 at dt = 0.1 of the shortest longitudinal period keeps the energy.
        props = TetherProperties(length_m=60.0, linear_density_kg_m=0.917, tensile_stiffness_N=2e6,
                                 bending_stiffness_Nm2=0.0, section_drag_coeff=0.0)
        line = LumpedTether(props, TetherSettings(num_nodes=12, longitudinal_damping_ratio=0.0))
        start, end = np.zeros(3), np.array([60.0 * 1.002, 0., 0.])
        x, v = line.straight_state(start, end)
        x[:, 2] += 0.5 * np.sin(np.pi * (np.arange(12) + 0.5) / 12)
        ends = fixed_ends(start, end)
        dt = 0.1 * line.longitudinal_period()
        e0 = line.energy(x, v, start, end)
        scale = e0["elastic"] + abs(e0["gravity"])
        drift, time = 0.0, 0.0
        for _ in range(20000):
            x, v = line.step(time, x, v, dt, ends)
            time += dt
            drift = max(drift, abs(line.energy(x, v, start, end)["total"] - e0["total"]))
        self.assertGreater(line.energy(x, v, start, end)["kinetic"], 0.0)
        self.assertLess(drift / scale, 1e-4, msg=drift / scale)

    def test_T6_power_balance_with_damping(self):
        props = TetherProperties(length_m=60.0, linear_density_kg_m=0.917, tensile_stiffness_N=2e6,
                                 bending_stiffness_Nm2=0.0, section_drag_coeff=0.0)
        line = LumpedTether(props, TetherSettings(num_nodes=12, longitudinal_damping_ratio=0.1))
        start, end = np.zeros(3), np.array([50., 0., 0.])
        x, v = line.straight_state(start, end)
        x[:, 2] -= 10.0 * np.sin(np.pi * (np.arange(12) + 0.5) / 12)
        ends = fixed_ends(start, end)
        dt = 0.05 * line.longitudinal_period()
        e0 = line.energy(x, v, start, end)["total"]
        dissipated, time = 0.0, 0.0
        for _ in range(8000):
            # Dissipation integrated with the midpoint rule: damping force along each segment
            # times the segment length rate, evaluated at the half step.
            def dissipation(xx, vv):
                out = line.forces(xx, vv, start, np.zeros(3), end, np.zeros(3))
                points = line.chain(xx, start, end)
                speeds = line.chain(vv, np.zeros(3), end * 0.0)
                seg = points[1:] - points[:-1]
                length = np.linalg.norm(seg, axis=1)
                rate = np.einsum("ij,ij->i", seg, speeds[1:] - speeds[:-1]) / length
                return float(np.sum(out["damping"] * rate))
            k1x, k1v = line.rhs(time, x, v, ends, None)
            xm, vm = x + dt / 2 * k1x, v + dt / 2 * k1v
            dissipated += dissipation(xm, vm) * dt
            x, v = line.step(time, x, v, dt, ends)
            time += dt
        e1 = line.energy(x, v, start, end)["total"]
        residual = abs((e0 - e1) - dissipated) / max(abs(e0 - e1), 1e-9)
        self.assertGreater(e0 - e1, 0.0)
        self.assertLess(residual, 2e-2, msg=(e0, e1, dissipated, residual))


class Conventions(unittest.TestCase):
    def test_T8_action_reaction_and_sum_of_forces(self):
        line = LumpedTether(TetherProperties(bending_stiffness_Nm2=35.0), TetherSettings(num_nodes=19))
        start, end = np.array([0., 0., 1.0]), np.array([200., 30., 350.])
        x, v = line.straight_state(start, end)
        rng = np.random.default_rng(4)
        x = x + 2.0 * rng.normal(size=x.shape)
        v = 3.0 * rng.normal(size=v.shape)
        wind = np.full_like(x, 8.0) * np.array([1., 0.2, 0.])
        out = line.forces(x, v, start, np.array([0., 0., 0.]), end, np.array([5., 0., 1.]), wind)
        gravity = np.array([0., 0., -line.node_mass * 9.81 * line.s.num_nodes])
        internal_and_aero = np.sum(out["node"], axis=0) - gravity - np.sum(out["aero"], axis=0)
        # Internal forces on nodes + forces on the two end bodies sum to zero.
        np.testing.assert_allclose(internal_and_aero + out["start"] + out["end"], 0.0,
                                   atol=1e-9 * np.max(np.abs(out["node"])))
        self.assertTrue(np.all(out["tension"] >= 0.0))

    def test_T9_unstable_step_fails_honestly(self):
        line = LumpedTether(TetherProperties(bending_stiffness_Nm2=0.0, section_drag_coeff=0.0),
                            TetherSettings(num_nodes=19, longitudinal_damping_ratio=0.0))
        start, end = np.zeros(3), np.array([420., 0., 0.])
        x, v = line.straight_state(start, end)
        x[:, 2] -= 5.0
        ends = fixed_ends(start, end)
        dt = 5.0 * line.longitudinal_period()
        with self.assertRaises((FloatingPointError, ValueError)):
            time = 0.0
            for _ in range(2000):
                x, v = line.step(time, x, v, dt, ends)
                time += dt
                if np.max(np.abs(x)) > 1e6:
                    raise FloatingPointError("diverged")

    def test_an_end_that_depends_on_the_line_is_evaluated_with_the_state_of_every_stage(self):
        line = LumpedTether(TetherProperties(length_m=60.0), TetherSettings(num_nodes=6))
        start, end = np.zeros(3), np.array([40.0, 5.0, 30.0])
        positions, velocities = line.straight_state(start, end)
        fixed = fixed_ends(start, end)
        seen = []

        def same(time, nodes, speeds):                    # the same fixed ends through the state-dependent interface
            seen.append((time, nodes.copy()))
            return start, np.zeros(3), end, np.zeros(3)
        same.uses_state = True
        a, b = (positions, velocities), (positions, velocities)
        for step in range(5):
            a = line.step(step * 1e-3, *a, 1e-3, fixed)
            b = line.step(step * 1e-3, *b, 1e-3, same)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(len(seen), 5 * 4)                # four stages per step
        np.testing.assert_allclose([time for time, _ in seen[:4]], [0.0, 0.5e-3, 0.5e-3, 1e-3], rtol=0, atol=1e-18)
        # The stage states, not the step's first state (the line starts at rest, so the second stage has not moved yet).
        self.assertFalse(np.array_equal(seen[0][1], seen[3][1]))

        def following(time, nodes, speeds):               # the far end rides 2 m beyond the last node, along x
            return start, np.zeros(3), nodes[-1] + np.array([2.0, 0.0, 0.0]), speeds[-1]
        following.uses_state = True
        c = line.step(0.0, positions, velocities, 1e-3, following)
        d = line.step(0.0, positions, velocities, 1e-3, fixed)
        # With the end riding on the last node the last segment is 2 m long instead of its natural 5 m: slack, so
        # the last node is no longer held from that side and falls back along the line under the pull of the others.
        self.assertFalse(np.array_equal(c[0], d[0]))
        self.assertLess(c[1][-1] @ (end - start), d[1][-1] @ (end - start))

    def test_curvature_matches_source_linear_system(self):
        rng = np.random.default_rng(1)
        line = LumpedTether()
        v1, v2 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        kappa = line._curvature(v1, v2)
        for a, b, k in zip(v1, v2, kappa):
            cross = np.cross(a, b)
            A = np.array([cross, a, b])
            r = np.linalg.solve(A, [0., a @ a / 2, -(b @ b) / 2])
            expected = cross / np.linalg.norm(cross) / np.linalg.norm(r)
            np.testing.assert_allclose(k, expected, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
