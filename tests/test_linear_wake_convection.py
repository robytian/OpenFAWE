"""Independent stage rings, manufactured ODE orders and actual free-wake history."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, build_lifting_surface_mesh, lifting_surface_network,
    lifting_surface_induced_velocity, _linear_convection_network)
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.wake import WakeConfig, pack_rings
from openfawe.aero.wing import Wing
from openfawe.aero_archive import AeroWriter, AeroReader
from openfawe.aero_vtk import export_paraview


def rectangle():
    le = np.array([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]])
    return Wing(le, le + [1., 0., 0.])


def settings():
    return LiftingSurfaceConfig(3, wake_basis='linear_potential', wake_release='current',
        gamma_derivative='bdf2', chordwise_spacing='cosine', wake_quadrature='te_graded')


class LinearWakeConvectionTests(unittest.TestCase):
    def test_constant_stage_potential_against_ordinary_physical_rings(self):
        wing = rectangle()
        for model in (CoreModel.NONE, CoreModel.RANKINE, CoreModel.LAMB_OSEEN, CoreModel.VATISTAS):
            cfg = WakeConfig(dt_s=.04, core_radius_m=.07, core_model=model, free_wake=True)
            state = start_lifting_surface(wing, cfg, [4., 0., .1], settings=settings(), free_wake=True)
            next_wing = Wing(wing.leading_edge + [.03, -.02, .01], wing.trailing_edge + [.03, -.02, .01])
            next_mesh = build_lifting_surface_mesh(next_wing, settings())
            bound = state.mesh.vortex_rows + .5 * (next_mesh.vortex_rows - state.mesh.vortex_rows)
            moving = (wing.trailing_edge + [.16, .01, .004])[None]
            network, radii, canonical = _linear_convection_network(state, cfg, moving, .5, next_mesh)
            ordinary = pack_rings(np.concatenate((bound, moving)),
                np.vstack((state.solution.lattice_circulation, state.solution.circulation)), wing.panel_edges)
            direct_radii = np.zeros(len(ordinary[1]))
            for k, (first, last) in enumerate(ordinary[1]):
                a, b = first // wing.node_count, last // wing.node_count
                if model != CoreModel.NONE and (a == b and a > 3 or a != b and min(a, b) >= 3):
                    direct_radii[k] = .07
            points = np.array([[.35, .15, .35], [1.2, .8, .7], [3., -.7, .2]])
            assert_allclose(segments_velocity(points, *network, model=model, core_radius=radii),
                segments_velocity(points, *ordinary, model=model, core_radius=direct_radii), rtol=2e-12, atol=2e-13)
            assert_array_equal(network[0][:3 * wing.node_count], bound[:-1].reshape(-1, 3))
            # Topological endpoint identity must hold even if interpolation rounds by one ulp.
            assert_array_equal(canonical[-wing.node_count:], moving[0])
            initial, _, _ = _linear_convection_network(state, cfg, wing.trailing_edge[None], 0.)
            for actual, expected in zip(initial, lifting_surface_network(state)):
                assert_array_equal(actual, expected)

    def test_zero_gamma_manufactured_geometry_orders(self):
        wing, dt, rate = rectangle(), .3, .3
        wind = lambda p, t: np.column_stack((np.full(len(p), 4.), rate * p[:, 1], np.zeros(len(p))))
        expected = wing.trailing_edge + [4 * dt, 0., 0.]
        expected[:, 1] = wing.trailing_edge[:, 1] * np.exp(rate * dt)
        for integrator, minimum in (('euler', 1.7), ('heun', 3.3), ('rk4', 12.)):
            errors = []
            for substeps in (1, 2, 4):
                cfg = WakeConfig(dt_s=dt, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
                    free_wake=True, convection_integrator=integrator, geometry_substeps=substeps)
                state = start_lifting_surface(wing, cfg, wind, settings=settings(), free_wake=True)
                state = advance_lifting_surface(state, cfg, wind)
                assert_array_equal(state.solution.lattice_circulation, 0.)
                errors.append(float(np.max(np.abs(state.free_rows[0] - expected))))
            self.assertGreater(errors[0] / errors[1], minimum)
            self.assertGreater(errors[1] / errors[2], minimum)

    def test_nonzero_gamma_galilean_covariance_for_all_integrators(self):
        wing, wind, boost = rectangle(), np.array([4., 0., .1]), np.array([.4, -.2, .15])
        for integrator in ('euler', 'heun', 'rk4'):
            cfg = WakeConfig(dt_s=.03, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
                free_wake=True, core_evolution='age_stretch', convection_integrator=integrator, geometry_substeps=2)
            stationary = start_lifting_surface(wing, cfg, wind, settings=settings(), free_wake=True)
            translated = start_lifting_surface(wing, cfg, wind + boost, structural_velocity=boost,
                settings=settings(), free_wake=True)
            for k in range(1, 4):
                stationary = advance_lifting_surface(stationary, cfg, wind)
                shift = k * cfg.dt_s * boost
                next_wing = Wing(wing.leading_edge + shift, wing.trailing_edge + shift)
                translated = advance_lifting_surface(translated, cfg, wind + boost,
                    next_wing=next_wing, structural_velocity=boost)
                assert_allclose(translated.free_rows, stationary.free_rows + shift, rtol=1e-12, atol=2e-12)
                assert_allclose(translated.solution.lattice_circulation,
                    stationary.solution.lattice_circulation, rtol=2e-11, atol=2e-12)
                assert_allclose(translated.solution.force, stationary.solution.force, rtol=2e-11, atol=2e-11)
                assert_allclose(translated.cores.span, stationary.cores.span, rtol=2e-12, atol=2e-13)

    def test_free_material_stretch_trace_flux_and_repeatability(self):
        cfg = WakeConfig(dt_s=.02, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
            free_wake=True, core_evolution='age_stretch', eddy_viscosity_factor=0., convection_integrator='rk4')
        wing, wind = rectangle(), [4., 0., .1]
        state = start_lifting_surface(wing, cfg, wind, settings=settings(), free_wake=True)
        stretch_seen = False
        for k in range(4):
            old = state
            old_rows = np.concatenate((old.wing.trailing_edge[None], old.free_rows))
            old_q = np.vstack((np.full((1, wing.panels), .07**2), old.cores.span**2))
            state = advance_lifting_surface(old, cfg, wind)
            repeated = advance_lifting_surface(old, cfg, wind)
            assert_array_equal(state.free_rows, repeated.free_rows)
            assert_array_equal(state.solution.force, repeated.solution.force)
            assert_array_equal(state.wake_node_potential_m2_s[1:], old.wake_node_potential_m2_s)
            a, b = wing.panel_edges.T
            old_length = np.linalg.norm(old_rows[:, b] - old_rows[:, a], axis=2)
            new_length = np.linalg.norm(state.free_rows[:, b] - state.free_rows[:, a], axis=2)
            assert_allclose(state.cores.span**2 * new_length, old_q * old_length, rtol=1e-12, atol=1e-14)
            stretch_seen |= bool(np.max(np.abs(new_length - old_length)) > 1e-10)
            relative = np.array(wind) + lifting_surface_induced_velocity(state, state.mesh.collocation_points)
            assert_allclose(np.sum(relative * state.mesh.normals, axis=1), 0., atol=2e-12)
            nodes, edges, gamma = lifting_surface_network(state)
            balance = np.zeros(len(nodes))
            np.add.at(balance, edges[:, 0], gamma)
            np.add.at(balance, edges[:, 1], -gamma)
            assert_allclose(balance, 0., atol=2e-15)
        self.assertTrue(stretch_seen)

    def test_free_archive_contract_vtk_and_prescribed_compatibility(self):
        parent = Path(__file__).resolve().parents[1] / 'local/linear-convection-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / uuid4().hex
        root.mkdir()
        try:
            for free in (False, True):
                name = 'free' if free else 'prescribed'
                cfg = WakeConfig(dt_s=.02, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
                    free_wake=free, core_evolution='age_stretch', convection_integrator='rk4')
                state = start_lifting_surface(rectangle(), cfg, [4., 0., .1], settings=settings(), free_wake=True)
                writer = AeroWriter(root / name, {'aero_method': 'vlm_free_wake'})
                try:
                    writer.append(state)
                    for k in range(4):
                        state = advance_lifting_surface(state, cfg, [4., 0., .1])
                        writer.append(state)
                    writer.finish()
                except BaseException as error:
                    writer.fail(error)
                    raise
                reader = AeroReader(root / name)
                try:
                    for k in range(5):
                        frame = reader.frame(k)
                    assert_array_equal(frame['free_rows'], state.free_rows)
                    self.assertEqual(reader.manifest.get('linear_wake_convection_contract'),
                        'attached_frozen_nodes_v1' if free else None)
                finally:
                    reader.close()
            export_paraview(root / 'free', root / 'vtk')
            tree = ET.parse(root / 'vtk/wake-000004.vtp')
            self.assertIsNotNone(tree.find('.//Points/DataArray'))
            bad = root / 'bad-contract'
            shutil.copytree(root / 'free', bad)
            manifest = json.loads((bad / 'manifest.json').read_text(encoding='utf-8'))
            manifest.pop('linear_wake_convection_contract')
            (bad / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'convection contract'):
                AeroReader(bad)
        finally:
            if root.resolve().parent != parent.resolve() or not parent.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
                raise ValueError('Refusing cleanup outside the test workspace')
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
