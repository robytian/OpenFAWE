"""Nonzero-circulation numerical combinations and independent geometry checks.

These are verification of bounded component transitions. They do not qualify
the time order of the coupled aerodynamic model or an aircraft operating range.
"""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.aircraft import PanelPolars, WingAssembly
from openfawe.aero.cores import edge_radii
from openfawe.aero.methods import AeroBackend
from openfawe.aero.vortex import segment_velocity
from openfawe.aero.wake import WakeConfig, convect_rows, wake_network
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing


METHODS = ('free_wake', 'vsm_free_wake')
INTEGRATORS = ('euler', 'heun', 'rk4')
POLICIES = ('constant', 'age', 'age_stretch')
NEWTON = SolveSettings(solver='newton', max_iterations=60)


def _frozen_array_values(state):
    values = [state.free_rows, state.older_circulation, state.release_times_s,
              state.solution.circulation, state.solution.panel_force]
    if state.cores is not None:
        values.extend((state.cores.span, state.cores.stream))
    return tuple(np.array(value, copy=True) for value in values)


def _independent_midpoint_geometry(state, config, wind, subdivisions):
    """Explicit midpoint using independent point/segment RHS assembly.

    The verified scalar filament kernel is shared; graph summation, incident
    exclusions and midpoint stepping do not call production convection helpers.
    Bound geometry, strengths and material cores remain frozen by definition.
    """
    nodes, edges, strengths = wake_network(state)
    first = state.wing.node_count
    fixed = nodes[:first].copy()
    radii = edge_radii(state.wing, edges, state.cores, config.core_radius_m)
    moving = nodes[first:].copy()
    dt = config.dt_s / subdivisions

    def derivative(points):
        all_nodes = np.vstack((fixed, points))
        velocity = np.broadcast_to(wind, points.shape).copy()
        for i, point in enumerate(points):
            for (a, b), strength, radius in zip(edges, strengths, radii):
                if np.array_equal(point, all_nodes[a]) or np.array_equal(point, all_nodes[b]):
                    continue
                velocity[i] += segment_velocity(all_nodes[a], all_nodes[b], point,
                    strength, model=config.core_model, core_radius=radius)
        return velocity

    for _ in range(subdivisions):
        midpoint = moving + .5 * dt * derivative(moving)
        moving += dt * derivative(midpoint)
    return moving.reshape(state.step + 1, first, 3)


class NumericalMatrixTests(unittest.TestCase):
    def test_nonzero_history_matrix_preserves_strengths_and_pure_retries(self):
        wing = Wing.ellipse(4)
        polar = SectionPolar.linear()
        wind = np.array([2., 0., .15])
        for method in METHODS:
            for integrator in INTEGRATORS:
                for policy in POLICIES:
                    with self.subTest(method=method, integrator=integrator, policy=policy):
                        config = WakeConfig(dt_s=.03, max_rows=3, solve=NEWTON,
                            convection_integrator=integrator, core_evolution=policy,
                            geometry_substeps=2, eddy_viscosity_factor=10.)
                        backend = AeroBackend(method, wake=config)
                        state = backend.start(wing, polar, wind)
                        self.assertGreater(np.max(np.abs(state.solution.circulation)), .01)
                        for step in range(1, 4):
                            saved = _frozen_array_values(state)
                            advanced = backend.advance(state, polar, wind)
                            repeated = backend.advance(state, polar, wind)
                            for old, current in zip(saved, _frozen_array_values(state)):
                                assert_array_equal(current, old)
                            for actual, repeat in zip(_frozen_array_values(advanced),
                                                      _frozen_array_values(repeated)):
                                assert_array_equal(actual, repeat)
                            self.assertEqual(advanced.step, step)
                            self.assertEqual(len(advanced.free_rows), step)
                            assert_array_equal(advanced.release_times_s,
                                               np.r_[state.time_s, state.release_times_s])
                            if state.step:
                                assert_array_equal(advanced.older_circulation[0],
                                                   state.solution.circulation)
                                assert_array_equal(advanced.older_circulation[1:],
                                                   state.older_circulation)
                            nodes, edges, gamma = wake_network(advanced)
                            balance = np.zeros(len(nodes))
                            np.add.at(balance, edges[:, 0], gamma)
                            np.add.at(balance, edges[:, 1], -gamma)
                            assert_allclose(balance, 0., atol=1e-12, rtol=0.)
                            radii = edge_radii(wing, edges, advanced.cores,
                                               config.core_radius_m)
                            self.assertTrue(np.isfinite(radii).all())
                            self.assertTrue(np.all(radii > 0.))
                            self.assertLessEqual(advanced.solution.residual,
                                                 advanced.solution.residual_limit)
                            self.assertTrue(np.isfinite(advanced.solution.force).all())
                            if policy == 'constant':
                                self.assertIsNone(advanced.cores)
                                assert_array_equal(radii, config.core_radius_m)
                            else:
                                self.assertEqual(advanced.cores.span.shape, (step, wing.panels))
                                self.assertEqual(advanced.cores.stream.shape, (step, wing.node_count))
                            state = advanced

    def test_touching_surface_equivalence_for_both_backends_and_integrators(self):
        leading = np.array([[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]])
        trailing = leading + [1., 0., 0.]
        whole = Wing(leading, trailing)
        split = WingAssembly((Wing(leading[:2], trailing[:2]),
                              Wing(leading[1:], trailing[1:])), ('left', 'right'))
        polar = SectionPolar.linear()
        tables = PanelPolars((polar, polar))
        wind = np.array([2., 0., .2])
        for method in METHODS:
            for integrator in INTEGRATORS:
                for policy in ('constant', 'age_stretch'):
                    with self.subTest(method=method, integrator=integrator, policy=policy):
                        config = WakeConfig(dt_s=.03, max_rows=3, solve=NEWTON,
                            convection_integrator=integrator, core_evolution=policy,
                            geometry_substeps=2)
                        backend = AeroBackend(method, wake=config)
                        a = backend.start(whole, polar, wind)
                        b = backend.start(split, tables, wind)
                        for _ in range(3):
                            a = backend.advance(a, polar, wind)
                            b = backend.advance(b, tables, wind)
                            assert_allclose(b.free_rows[:, [0, 1, 3]], a.free_rows,
                                            atol=1e-11, rtol=1e-11)
                            assert_array_equal(b.free_rows[:, 1], b.free_rows[:, 2])
                            assert_allclose(b.solution.circulation, a.solution.circulation,
                                            atol=1e-10, rtol=1e-10)
                            assert_allclose(b.solution.force, a.solution.force,
                                            atol=1e-10, rtol=1e-10)
                            if a.cores is not None:
                                assert_allclose(b.cores.span, a.cores.span,
                                                atol=1e-12, rtol=1e-12)
                                assert_allclose(b.cores.stream[:, [0, 1, 3]], a.cores.stream,
                                                atol=1e-12, rtol=1e-12)

    def test_stage_dependent_induction_against_independent_midpoint_reference(self):
        leading = np.array([[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]])
        wing = Wing(leading, leading + [1., 0., 0.])
        polar = SectionPolar.linear()
        wind = np.array([2., 0., .2])
        config = WakeConfig(dt_s=.2, solve=NEWTON)
        # A nonzero frozen circulation graph has genuinely position-dependent
        # self induction. A uniform ambient field alone would be exact here.
        state = AeroBackend('free_wake', wake=config).start(wing, polar, wind)
        saved = _frozen_array_values(state)
        references = [_independent_midpoint_geometry(state, config, wind, n)
                      for n in (64, 128, 256)]
        # Extrapolate independent second-order midpoint references, and retain
        # the change as an uncertainty screen for the production comparison.
        extrapolated = [(4 * b - a) / 3 for a, b in zip(references, references[1:])]
        reference = extrapolated[-1]
        uncertainty = np.max(np.abs(extrapolated[-1] - extrapolated[0]))
        errors = {}
        for integrator in INTEGRATORS:
            errors[integrator] = []
            for subdivisions in (1, 2, 4):
                result = convect_rows(state, replace(config,
                    convection_integrator=integrator, geometry_substeps=subdivisions), wind)
                errors[integrator].append(float(np.max(np.abs(result - reference))))
            self.assertLess(errors[integrator][1], errors[integrator][0])
            self.assertLess(errors[integrator][2], errors[integrator][1])
            self.assertLess(uncertainty, .1 * errors[integrator][2])
        self.assertLess(errors['rk4'][0], errors['heun'][0])
        self.assertLess(errors['heun'][0], errors['euler'][0])
        for old, current in zip(saved, _frozen_array_values(state)):
            assert_array_equal(current, old)


if __name__ == '__main__':
    unittest.main()
