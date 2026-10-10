"""Independent steady references and bounded UVLM numerical contracts."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.aircraft import WingAssembly
from openfawe.aero.lifting_surface import (
    LiftingSurfaceConfig, VlmState, VlmWakeState, build_lifting_surface_mesh,
    start_lifting_surface, advance_lifting_surface, lifting_surface_network,
    lifting_surface_edge_radii, lifting_surface_induced_velocity,
    lifting_surface_influence, _convect, _steady_mirror_permutation)
from openfawe.aero.vortex import CoreModel, FilamentSingularity
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing


def rectangle(panels=4, span=4., chord=1.):
    y = np.linspace(-span/2, span/2, panels+1)
    le = np.column_stack((np.zeros_like(y), y, np.zeros_like(y)))
    return Wing(le, le+[chord, 0., 0.])


class LiftingSurfaceTests(unittest.TestCase):
    def test_true_chordwise_mesh_and_immutable_ownership(self):
        wing = rectangle()
        mesh = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(4))
        self.assertEqual(mesh.panels, 16)
        assert_allclose(mesh.collocation_points[:, 0].reshape(4, 4)[:, 0],
                        [.1875, .4375, .6875, .9375])
        assert_allclose(mesh.vortex_rows[:, 0, 0], [.0625, .3125, .5625, .8125, 1.])
        assert_allclose(mesh.normals, np.tile([0., 0., 1.], (16, 1)))
        assert_allclose(mesh.area.sum(), 4.)
        for array in (mesh.vertices, mesh.cells, mesh.normals, mesh.strip_index):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_unchanged_dlr_panelaero_ring_circulation_reference(self):
        reference = json.loads((Path(__file__).parent/'fixtures/lifting_surface_reference.json').read_text())
        for case in reference['cases']:
            with self.subTest(case=case['name']):
                wings = tuple(Wing(s['leading_edge'], s['trailing_edge']) for s in case['surfaces'])
                wing = wings[0] if len(wings) == 1 else WingAssembly(wings, tuple(str(i) for i in range(len(wings))))
                alpha = np.deg2rad(case['alpha_deg'])
                wind = case['speed_m_s']*np.array([np.cos(alpha), 0., np.sin(alpha)])
                settings = LiftingSurfaceConfig(4, wake_length_m=1e7, wake_direction=(1., 0., 0.))
                config = WakeConfig(core_model=CoreModel.NONE)
                mesh = build_lifting_surface_mesh(wing, settings)
                # The coupled source fixture has wake/load-midpoint crossings.
                # Compare its boundary equation without relabeling a singular
                # Joukowski force field as an accepted complete state.
                operator = lifting_surface_influence(wing, mesh, config,
                    wing.trailing_edge+[settings.wake_length_m, 0., 0.])
                matrix = np.einsum('ijk,ik->ij', operator, mesh.normals)
                gamma = np.linalg.solve(matrix, -mesh.normals@wind).reshape(4, wing.panels)
                expected = np.asarray(case['gamma_rings_m2_s']).reshape(-1, 4).T
                assert_allclose(gamma, expected,
                                rtol=reference['rtol'], atol=reference['atol'])
                self.assertEqual(np.linalg.matrix_rank(gamma), 4)
                if len(wings) == 1:
                    state = start_lifting_surface(wing, config, wind, settings=settings)
                    assert_allclose(state.solution.lattice_circulation, expected,
                                    rtol=reference['rtol'], atol=reference['atol'])
                    self.assertLess(state.solution.boundary_residual_m_s, 1e-11)

    def test_explicit_steady_mirror_projection_on_dlr_wing_tail(self):
        reference = json.loads((Path(__file__).parent/'fixtures/lifting_surface_reference.json').read_text())
        case = next(item for item in reference['cases'] if item['name'].startswith('wing_tail'))
        surfaces = tuple(Wing(item['leading_edge'], item['trailing_edge'])
                         for item in case['surfaces'])
        wing = WingAssembly(surfaces, ('wing', 'tail'))
        alpha = np.deg2rad(case['alpha_deg'])
        wind = case['speed_m_s']*np.array([np.cos(alpha), 0., np.sin(alpha)])
        config = WakeConfig(core_model=CoreModel.NONE)
        settings = LiftingSurfaceConfig(4, wake_length_m=1e7,
                                        wake_direction=(1., 0., 0.))
        with self.assertRaises(FilamentSingularity):
            start_lifting_surface(wing, config, wind, settings=settings)
        projected = replace(settings, steady_mirror_projection=True)
        state = start_lifting_surface(wing, config, wind, settings=projected)
        rounded_far = state.prescribed_row.copy()
        rounded_far[0, 0] = np.nextafter(rounded_far[0, 0], np.inf)
        _steady_mirror_permutation(wing, state.mesh, rounded_far, wind, (0., 0., 0.))
        expected = np.asarray(case['gamma_rings_m2_s']).reshape(-1, 4).T
        assert_allclose(state.solution.lattice_circulation, expected,
                        rtol=reference['rtol'], atol=reference['atol'])
        self.assertLess(state.solution.boundary_residual_m_s,
                        projected.linear_residual_tolerance_m_s)
        self.assertGreater(state.solution.mirror_preprojection_max_abs_m2_s, 0.)
        self.assertLess(state.solution.mirror_preprojection_max_abs_m2_s,
                        projected.mirror_projection_atol_m2_s +
                        projected.mirror_projection_rtol*np.max(np.abs(state.solution.lattice_circulation)))
        self.assertLess(state.solution.unprojected_boundary_residual_m_s, 1e-12)
        self.assertTrue(np.isfinite(state.solution.force).all())
        self.assertTrue(np.isfinite(state.solution.lattice_couple).all())
        nodes, edges, _ = lifting_surface_network(state)
        on_axis = np.all(nodes[edges][:, :, 1:] == 0., axis=(1, 2))
        self.assertFalse(np.any(on_axis))
        with self.assertRaisesRegex(ValueError, 'constant y-symmetric'):
            start_lifting_surface(wing, config, wind+[0., 1e-6, 0.], settings=projected)
        with self.assertRaisesRegex(ValueError, 'unavailable for free wakes'):
            start_lifting_surface(wing, config, wind, settings=projected,
                                  free_wake=True)
        with self.assertRaisesRegex(ValueError, 'circulation asymmetry'):
            start_lifting_surface(wing, config, wind,
                                  settings=replace(projected, mirror_projection_rtol=1e-14,
                                                   mirror_projection_atol_m2_s=1e-14))
        moved_tail = surfaces[1].transformed(np.eye(3), [0., 1e-6, 0.])
        asymmetric = WingAssembly((surfaces[0], moved_tail), ('wing', 'tail'))
        with self.assertRaisesRegex(ValueError, 'mirrored geometry'):
            start_lifting_surface(asymmetric, config, wind, settings=projected)

    def test_twisted_bilinear_boundary_normal_and_integrated_area_are_distinct(self):
        wing = Wing([[0., 0., 0.], [0., 2., 0.]], [[1., 0., 0.], [1., 2., .4]])
        mesh = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(2))
        for i in range(2):
            tangent = np.array([1., 0., .2])
            span_at_cp = np.array([0., 2., .4*(i+.75)/2])
            expected = np.cross(tangent, span_at_cp)
            assert_allclose(mesh.normals[i], expected/np.linalg.norm(expected))
        self.assertGreater(np.max(np.abs(mesh.normals-mesh.area_normals)), .01)

    def test_zero_angle_symmetry_sign_and_high_aspect_ratio_limit(self):
        config = WakeConfig(core_model=CoreModel.NONE)
        wing = rectangle(1, span=4000.)
        settings = LiftingSurfaceConfig(8, wake_length_m=1e8, wake_direction=(1., 0., 0.))
        zero = start_lifting_surface(wing, config, [20., 0., 0.], settings=settings)
        assert_array_equal(zero.solution.lattice_circulation, np.zeros((8, 1)))
        values = []
        for alpha in (-.005, .005):
            wind = 20*np.array([np.cos(alpha), 0., np.sin(alpha)])
            state = start_lifting_surface(wing, config, wind, settings=settings)
            values.append(state.solution.force)
            cl = state.solution.force[2]/(.5*config.density_kg_m3*20**2*4000.)
            self.assertLess(abs(cl/(2*np.pi*alpha)-1), .002)
        assert_allclose(values[0][2], -values[1][2], rtol=1e-12)
        assert_allclose(values[0][0], values[1][0], rtol=1e-12)

    def test_rotation_translation_and_galilean_velocity(self):
        wing = rectangle()
        config = WakeConfig(dt_s=.02, max_rows=2)
        angle = .32
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        translation = np.array([2., -1., 3.])
        frame_speed = np.array([1., 2., -.5])
        wind = np.array([10., .15, .6])
        for free in (False, True):
            state = start_lifting_surface(wing, config, wind, free_wake=free)
            transformed = start_lifting_surface(wing.transformed(rotation, translation), config,
                rotation@wind+frame_speed, free_wake=free, structural_velocity=frame_speed)
            assert_allclose(transformed.solution.lattice_circulation, state.solution.lattice_circulation, rtol=1e-12, atol=1e-12)
            assert_allclose(transformed.solution.force, rotation@state.solution.force, rtol=1e-11, atol=1e-11)
            if free:
                state = advance_lifting_surface(state, config, wind)
                transformed = advance_lifting_surface(transformed, config, rotation@wind+frame_speed,
                    next_wing=wing.transformed(rotation, translation+frame_speed*config.dt_s),
                    structural_velocity=frame_speed)
                assert_allclose(transformed.free_rows, state.free_rows@rotation.T+translation+frame_speed*config.dt_s, rtol=1e-12, atol=1e-12)
                assert_allclose(transformed.solution.force, rotation@state.solution.force, rtol=1e-10, atol=1e-10)

    def test_material_kelvin_history_and_pure_retries_across_integrators_cores(self):
        wing = rectangle(2)
        for integrator in ('euler', 'heun', 'rk4'):
            for policy in ('constant', 'age', 'age_stretch'):
                with self.subTest(integrator=integrator, core=policy):
                    config = WakeConfig(dt_s=.03, max_rows=3, convection_integrator=integrator,
                                        core_evolution=policy, geometry_substeps=2)
                    state = start_lifting_surface(wing, config, [4., 0., .2],
                                                  settings=LiftingSurfaceConfig(2), free_wake=True)
                    for step in range(3):
                        old_gamma = state.solution.circulation.copy()
                        old_rows = state.free_rows.copy()
                        advanced = advance_lifting_surface(state, config, [4., 0., .2])
                        repeated = advance_lifting_surface(state, config, [4., 0., .2])
                        assert_array_equal(state.free_rows, old_rows)
                        assert_array_equal(advanced.free_rows, repeated.free_rows)
                        assert_array_equal(advanced.wake_circulation[0], old_gamma)
                        assert_array_equal(advanced.wake_circulation[1:], state.wake_circulation)
                        nodes, edges, gamma = lifting_surface_network(advanced)
                        divergence = np.zeros(len(nodes))
                        np.add.at(divergence, edges[:, 0], gamma)
                        np.add.at(divergence, edges[:, 1], -gamma)
                        assert_allclose(divergence, 0., atol=1e-15)
                        assert_allclose(advanced.solution.lattice_gamma_rate,
                            (advanced.solution.lattice_circulation-state.solution.lattice_circulation)/config.dt_s)
                        state = advanced
                    with self.assertRaisesRegex(ValueError, 'row limit'):
                        advance_lifting_surface(state, config, [4., 0., .2])

    def test_induced_free_convection_galilean_invariance_all_integrators(self):
        wing = rectangle(2)
        wind, boost = np.array([5., 0., .2]), np.array([3., -.8, .6])
        for integrator in ('euler', 'heun', 'rk4'):
            config = WakeConfig(dt_s=.01, convection_integrator=integrator)
            settings = LiftingSurfaceConfig(2)
            fixed = start_lifting_surface(wing, config, wind, free_wake=True, settings=settings)
            moving = start_lifting_surface(wing, config, wind+boost, free_wake=True,
                settings=settings, structural_velocity=boost)
            for step in range(1, 3):
                fixed = advance_lifting_surface(fixed, config, wind)
                moving = advance_lifting_surface(moving, config, wind+boost,
                    next_wing=wing.transformed(np.eye(3), boost*step*config.dt_s),
                    structural_velocity=boost)
                assert_allclose(moving.free_rows, fixed.free_rows+boost*step*config.dt_s, atol=2e-14, rtol=1e-13)
                assert_allclose(moving.solution.lattice_circulation, fixed.solution.lattice_circulation,
                                atol=2e-13, rtol=2e-13)
                assert_allclose(moving.solution.force, fixed.solution.force, atol=2e-11, rtol=2e-12)

    def test_bound_sheet_is_inviscid_and_wake_has_material_cores(self):
        wing = rectangle(2)
        config = WakeConfig(core_evolution='age_stretch', dt_s=.03)
        state = start_lifting_surface(wing, config, [4., 0., .2], free_wake=True)
        assert_array_equal(lifting_surface_edge_radii(state), 0.)
        state = advance_lifting_surface(state, config, [4., 0., .2])
        edges = lifting_surface_network(state)[1]
        radii = lifting_surface_edge_radii(state)
        row = edges//wing.node_count
        attached = (row[:, 0] < 4) | (row[:, 1] < 4) | np.all(row == 4, axis=1)
        assert_array_equal(radii[attached], 0.)
        self.assertTrue(np.all(radii[~attached] >= config.core_radius_m))

    def test_spatial_velocity_callbacks_sample_boundary_and_edge_loads(self):
        wing = rectangle(2)
        seen = []
        def body(points, time):
            seen.append(points.copy())
            return np.column_stack((np.zeros(len(points)), np.zeros(len(points)), .3*points[:, 0]))
        state = start_lifting_surface(wing, WakeConfig(), [10., 0., .5], structural_velocity=body)
        assert_allclose(state.lattice_structural_velocity[:, 2], .3*state.mesh.collocation_points[:, 0])
        assert_allclose(state.structural_velocity[:, 2], .3*wing.cp[:, 0])
        self.assertTrue(any(np.any(np.isclose(points[:, 0], .0625)) for points in seen))
        with self.assertRaisesRegex(ValueError, 'spatial callable'):
            start_lifting_surface(wing, WakeConfig(), [10., 0., .5], structural_velocity=np.zeros((8, 3)))

    def test_exact_force_moment_reduction_and_start_rate_convention(self):
        wing = rectangle(3)
        config = WakeConfig(dt_s=.04)
        state = start_lifting_surface(wing, config, [10., .3, .6], free_wake=True)
        assert_array_equal(state.solution.lattice_gamma_rate, 0.)
        state = advance_lifting_surface(state, config, [10., .3, .6])
        solution = state.solution
        forces = solution.lattice_force.reshape(-1, 3)
        couples = solution.lattice_couple.reshape(-1, 3)
        reference = np.array([.2, -.1, .4])
        direct = np.sum(np.cross(state.mesh.collocation_points-reference, forces)+couples, axis=0)
        assert_allclose(solution.force, forces.sum(axis=0), atol=1e-12)
        assert_allclose(solution.moment(wing, reference), direct, atol=1e-12)

    def test_flat_plate_noncirculatory_added_mass_limit(self):
        # The two-dimensional plate's normal added mass per span is pi*rho*c²/4.
        # Initially Gamma=0, so the first released ring has zero strength. A
        # prescribed normal acceleration isolates the discrete Gamma-rate term.
        wing = rectangle(1, span=4000.)
        config = WakeConfig(dt_s=.01, core_model=CoreModel.NONE)
        errors = []
        for nc in (4, 8, 16):
            state = start_lifting_surface(wing, config, [20., 0., 0.],
                settings=LiftingSurfaceConfig(nc), free_wake=True)
            state = advance_lifting_surface(state, config, [20., 0., .01])
            computed = config.density_kg_m3*np.sum(
                state.solution.lattice_gamma_rate.ravel()*state.mesh.area)
            exact = np.pi/4*config.density_kg_m3*4000.
            errors.append(abs(computed/exact-1))
        self.assertTrue(errors[0] > errors[1] > errors[2])
        self.assertLess(errors[-1], .01)

    def test_frozen_geometry_integrator_order_with_affine_ambient_field(self):
        wing = rectangle(2)
        # Zero Gamma eliminates self induction. The ODE dx/dt=a*x+U has an
        # independent exponential solution for every TE node, including shear.
        a, speed = .7, 2.
        def wind(points, time):
            return np.column_stack((speed+a*points[:, 0], np.zeros(len(points)), np.zeros(len(points))))
        ratios = {}
        for integrator in ('euler', 'heun', 'rk4'):
            errors = []
            for dt in (.2, .1):
                config = WakeConfig(dt_s=dt, convection_integrator=integrator)
                state = start_lifting_surface(wing, config, wind, free_wake=True)
                moved = _convect(state, config, wind)[0]
                exact = (wing.trailing_edge[:, 0]+speed/a)*np.exp(a*dt)-speed/a
                errors.append(np.max(np.abs(moved[:, 0]-exact)))
            ratios[integrator] = errors[0]/errors[1]
        self.assertGreater(ratios['euler'], 3.8)
        self.assertGreater(ratios['heun'], 7.5)
        self.assertGreater(ratios['rk4'], 30.)

    def test_state_settings_clock_topology_and_probe_contracts(self):
        wing = rectangle(2)
        config = WakeConfig(dt_s=.05)
        state = start_lifting_surface(wing, config, [4., 0., .2], free_wake=True)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            advance_lifting_surface(state, replace(config, core_radius_m=.2), [4., 0., .2])
        with self.assertRaisesRegex(ValueError, 'configuration'):
            lifting_surface_induced_velocity(state, [[0., 0., 2.]], replace(config, core_model=CoreModel.RANKINE))
        with self.assertRaisesRegex(ValueError, 'connectivity'):
            advance_lifting_surface(state, config, [4., 0., .2], next_wing=rectangle(3))
        with self.assertRaisesRegex(ValueError, 'next time'):
            advance_lifting_surface(state, config, [4., 0., .2], next_time_s=.06)
        for array in (state.solution.lattice_circulation, state.solution.lattice_force,
                      state.structural_velocity, state.free_rows):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        velocity = lifting_surface_induced_velocity(state, [[.5, 0., 2.]])
        self.assertTrue(np.isfinite(velocity).all())


if __name__ == '__main__':
    unittest.main()
