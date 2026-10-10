"""Independent manufactured loads, full operators, and dual archive contracts."""

from dataclasses import replace
from pathlib import Path
from contextlib import closing
import importlib.util
import json
import os
import shutil
import unittest
from uuid import uuid4
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, lifting_surface_induced_velocity, _loads)
from openfawe.aero.wake import WakeConfig
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.wake_potential import potential_network, potential_edge_radii
from openfawe.aero.polar_coupling import ClosureProblem, encode_polars, polar_tables, polar_identity
from openfawe.aero_archive import (AeroWriter, AeroReader, POTENTIAL_POLAR_LATTICE_SCHEMA,
    POTENTIAL_LATTICE_SCHEMA, _validate_lattice_frame)
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file
from openfawe.run_catalog import inspect_run


def wing():
    le = np.array([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]])
    return Wing(le, le + [1., 0., 0.])


def settings(**changes):
    return replace(LiftingSurfaceConfig(3, chordwise_spacing='cosine',
        wake_basis='linear_potential', wake_release='current', gamma_derivative='bdf2',
        wake_quadrature='te_graded', polar_coupling='grid_section'), **changes)


def config(**changes):
    return replace(WakeConfig(dt_s=.025, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
        core_evolution='age_stretch', convection_integrator='rk4'), **changes)


def polar():
    angles = np.linspace(-25, 25, 101)
    return SectionPolar(angles, 5*np.sin(np.deg2rad(angles)), angles*0+.02, angles*0-.03)


class LinearWakePolarTests(unittest.TestCase):
    def test_zero_lift_manufactured_drag_moment_all_integrators(self):
        w, p = wing(), polar()
        for integrator in ('euler', 'heun', 'rk4'):
            for free in (False, True):
                with self.subTest(integrator=integrator, free=free):
                    cfg = config(convection_integrator=integrator, free_wake=free)
                    state = start_lifting_surface(w, cfg, [4., 0., 0.], settings=settings(), polar=p, free_wake=True)
                    for k in range(4):
                        if k:
                            state = advance_lifting_surface(state, cfg, [4., 0., 0.], polar=p)
                        qarea = .5*cfg.density_kg_m3*16*w.area.sum()
                        assert_allclose(state.solution.lattice_circulation, 0., atol=2e-13)
                        assert_allclose(state.solution.force, [qarea*.02, 0., 0.], atol=2e-12)
                        assert_allclose(state.solution.moment(w), [0., -qarea*.03, 0.], atol=2e-12)
                        assert_allclose(state.solution.polar_coupling.potential_force_N, 0., atol=2e-12)

    def test_exact_thin_source_identity_prescribed_and_free(self):
        w = wing()
        for free in (False, True):
            cfg = config(free_wake=free)
            pure_settings = settings(polar_coupling='none')
            states = [start_lifting_surface(w, cfg, [4., 0., .1], settings=pure_settings, free_wake=True)]
            for k in range(1, 5):
                states.append(advance_lifting_surface(states[-1], cfg, [4., 0., .1+.005*k]))
            alphas = []
            for k, state in enumerate(states):
                speed = np.linalg.norm([4., 0., .1+.005*k])
                alphas.extend(np.rad2deg(np.arcsin(2*state.solution.circulation/(speed*2*np.pi))))
            # Symmetric strips can differ by one ulp; do not create artificial
            # zero-slope intervals between such manufactured duplicate knots.
            knots = np.unique(np.round(np.r_[-30., alphas, 30.], 12))
            p = SectionPolar(knots, 2*np.pi*np.sin(np.deg2rad(knots)), knots*0, knots*0)
            state = start_lifting_surface(w, cfg, [4., 0., .1], settings=settings(), polar=p, free_wake=True)
            for k, pure in enumerate(states):
                if k:
                    state = advance_lifting_surface(state, cfg, [4., 0., .1+.005*k], polar=p)
                assert_allclose(state.solution.lattice_circulation, pure.solution.lattice_circulation, atol=2e-12, rtol=2e-12)
                assert_allclose(state.solution.force, pure.solution.force, atol=2e-10, rtol=2e-12)
                assert_allclose(state.solution.moment(w), pure.solution.moment(w), atol=2e-10, rtol=2e-12)
                assert_allclose(state.free_rows, pure.free_rows, atol=2e-12)

    def test_full_linear_operator_jacobian_bdf_history_and_load_routing(self):
        w, p, cfg = wing(), polar(), config()
        state = start_lifting_surface(w, cfg, [4., 0., .1], settings=settings(), polar=p, free_wake=True)
        two_back = None
        for k in range(1, 4):
            previous = state
            wind = np.array([4., 0., .1+.02*k])
            state = advance_lifting_surface(previous, cfg, wind, polar=p)
            repeated = advance_lifting_surface(previous, cfg, wind, polar=p)
            assert_array_equal(state.solution.force, repeated.solution.force)
            assert_array_equal(state.wake_node_potential_m2_s[1:], previous.wake_node_potential_m2_s)
            assert_array_equal(state.solution.polar_coupling.trace_json, repeated.solution.polar_coupling.trace_json)
            expected_rate = ((state.solution.lattice_circulation-previous.solution.lattice_circulation)/cfg.dt_s
                if two_back is None else (3*state.solution.lattice_circulation-4*previous.solution.lattice_circulation
                    +two_back.solution.lattice_circulation)/(2*cfg.dt_s))
            assert_allclose(state.solution.lattice_gamma_rate, expected_rate, atol=2e-13)
            two_back = previous
        mesh, c = state.mesh, state.solution.polar_coupling
        points = mesh.collocation_points
        velocity = wind + lifting_surface_induced_velocity(state, points)
        assert_allclose(np.sum(velocity*c.virtual_normals, axis=1), 0., atol=2e-12)
        assert_allclose(np.sum(velocity*mesh.normals, axis=1), c.geometric_residual_m_s, atol=2e-12)
        self.assertGreater(np.max(np.abs(c.geometric_residual_m_s)), 1e-4)
        force, couple = _loads(w, mesh, state.solution.lattice_circulation, state.free_rows,
            state.wake_circulation, state.cores, cfg, wind, [0., 0., 0.], state.time_s,
            state.solution.lattice_gamma_rate, state.wake_node_potential_m2_s, state.settings)
        assert_array_equal(c.potential_force_N, force)
        assert_array_equal(c.potential_couple_N_m, couple)
        assert_allclose(state.solution.lattice_force, force+c.correction_force_N, atol=2e-13)
        assert_allclose(state.solution.lattice_couple, couple+c.correction_couple_N_m, atol=2e-13)
        # Reconstruct each entire bound+current-TE column via a separate full graph,
        # rather than calling the production split influence/history assembly.
        def field(gamma, phi):
            network = potential_network(w, mesh, gamma, state.free_rows, phi, 4, 'te_graded')
            radii = potential_edge_radii(w, mesh, state.free_rows, network[1], state.cores, .07, 4, 'te_graded')
            return segments_velocity(points, *network, model=cfg.core_model, core_radius=radii)
        operator = np.empty((mesh.panels, mesh.panels, 3))
        for j in range(mesh.panels):
            unit = np.zeros_like(state.solution.lattice_circulation)
            unit.flat[j] = 1.
            phi = np.vstack((unit[-1], np.zeros_like(state.wake_node_potential_m2_s[1:])))
            operator[:, j] = field(unit, phi)
        known = field(np.zeros_like(state.solution.lattice_circulation),
            np.vstack((np.zeros((1, w.panels)), state.wake_node_potential_m2_s[1:])))
        problem = ClosureProblem(w, mesh, operator, known+wind, np.tile(wind, (w.panels, 1)), p, state.settings)
        d, h = c.delta_rad, 1e-6
        assert_allclose(problem.evaluate(d)['gamma'], state.solution.lattice_circulation.reshape(-1), atol=2e-12)
        numeric = np.column_stack([(problem.evaluate(d+v)['residual']-problem.evaluate(d-v)['residual'])/(2*h)
            for v in np.eye(w.panels)*h])
        assert_allclose(problem.evaluate(d, jacobian=True)['jacobian'], numeric, atol=3e-9, rtol=3e-8)

    def test_dual_v6_archive_real_consumers_and_corruption(self):
        parent = Path(__file__).resolve().parents[1] / 'local/linear-polar-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / uuid4().hex
        root.mkdir()
        try:
            w, p, cfg = wing(), polar(), config()
            identity = polar_identity(encode_polars(polar_tables(p, w.panels)))
            run = root / 'run'
            case = dict(aero_method='vlm_free_wake', coupled_source_polars_sha256=identity)
            state = start_lifting_surface(w, cfg, [4., 0., .1], settings=settings(), polar=p, free_wake=True)
            writer = AeroWriter(run, case)
            try:
                writer.append(state)
                for k in range(3):
                    state = advance_lifting_surface(state, cfg, [4., 0., .1+.02*k], polar=p)
                    writer.append(state)
                writer.finish()
            except BaseException as error:
                writer.fail(error)
                raise
            before = sha256_file(run / 'wake.h5')
            with closing(AeroReader(run)) as reader:
                self.assertEqual(reader.manifest['schema'], POTENTIAL_POLAR_LATTICE_SCHEMA)
                frame = reader.frame(3)
                self.assertEqual(frame['polar_input_binding'], 'case_hash_checked')
                self.assertIn('virtual', frame['boundary_condition'].lower())
                assert_allclose(frame['panel_force'].sum(axis=0), state.solution.force)
                for key in ('wake_node_potential_m2_s', 'polar_correction_couple_N_m', 'polar_geometric_residual_m_s'):
                    bad = dict(frame)
                    bad[key] = frame[key].copy()
                    bad[key].flat[0] += .01
                    with self.assertRaises(ValueError, msg=key):
                        _validate_lattice_frame(bad)
            self.assertIn('source-polar coupled', inspect_run(run).core)
            export_paraview(run, root / 'vtk')
            vtk = (root / 'vtk/wing-000003.vtp').read_text(encoding='utf-8')
            self.assertIn('wake_basis_0_constant_1_linear_potential', vtk)
            self.assertIn('virtual_boundary_residual_m_s', vtk)
            if importlib.util.find_spec('PySide6'):
                os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
                from PySide6 import QtWidgets
                from openfawe.aero_viewer import AeroReplay
                app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
                window = AeroReplay(run)
                try:
                    window.timer.stop()
                    window.show()
                    window.seek(1)
                    app.processEvents()
                    self.assertIn('VIRTUAL BOUNDARY', window.info.text())
                    self.assertIn('source-polar coupled', window.info.text().lower())
                    self.assertFalse(window.grab().isNull())
                finally:
                    window.close()
            self.assertEqual(before, sha256_file(run / 'wake.h5'))
            for name in ('downgrade', 'missing-closure', 'missing-material', 'missing-free'):
                bad = root / name
                shutil.copytree(run, bad)
                manifest = json.loads((bad / 'manifest.json').read_text(encoding='utf-8'))
                if name == 'downgrade':
                    manifest['schema'] = POTENTIAL_LATTICE_SCHEMA
                    with h5py.File(bad / 'wake.h5', 'r+') as file:
                        file.attrs['schema'] = POTENTIAL_LATTICE_SCHEMA
                elif name == 'missing-closure':
                    with h5py.File(bad / 'wake.h5', 'r+') as file:
                        del file['frames']['000000'].attrs['polar_coupling_contract']
                elif name == 'missing-material':
                    manifest.pop('wake_potential_contract')
                else:
                    manifest.pop('linear_wake_convection_contract')
                manifest['wake_sha256'] = sha256_file(bad / 'wake.h5')
                (bad / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError, msg=name):
                    AeroReader(bad)
        finally:
            if root.resolve().parent != parent.resolve() or not parent.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
                raise ValueError('Refusing cleanup outside test workspace')
            shutil.rmtree(root)

    def test_source_coupled_rotation_and_galilean_covariance(self):
        w, p = wing(), polar()
        angle = .31
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        offset, translation_velocity = np.array([2., -.3, 1.]), np.array([.7, .2, -.1])
        def transformed(time):
            position = offset + time*translation_velocity
            return Wing(w.leading_edge@rotation.T+position, w.trailing_edge@rotation.T+position)
        for integrator in ('euler', 'heun', 'rk4'):
            cfg = config(convection_integrator=integrator, geometry_substeps=2)
            wind = np.array([4., 0., .1])
            state = start_lifting_surface(w, cfg, wind, settings=settings(), polar=p, free_wake=True)
            changed = start_lifting_surface(transformed(0), cfg, rotation@wind+translation_velocity,
                settings=settings(), polar=p, free_wake=True, structural_velocity=translation_velocity)
            for k in range(4):
                if k:
                    wind = np.array([4., 0., .1+.01*k])
                    state = advance_lifting_surface(state, cfg, wind, polar=p)
                    changed = advance_lifting_surface(changed, cfg, rotation@wind+translation_velocity,
                        next_wing=transformed(state.time_s), structural_velocity=translation_velocity, polar=p)
                position = offset+state.time_s*translation_velocity
                assert_allclose(changed.solution.lattice_circulation, state.solution.lattice_circulation, atol=3e-11, rtol=3e-11)
                assert_allclose(changed.solution.force, rotation@state.solution.force, atol=3e-9, rtol=3e-10)
                assert_allclose(changed.solution.moment(changed.wing, reference=position),
                    rotation@state.solution.moment(w), atol=3e-9, rtol=3e-10)
                assert_allclose(changed.free_rows, state.free_rows@rotation.T+position, atol=3e-11, rtol=3e-11)


if __name__ == '__main__':
    unittest.main()
