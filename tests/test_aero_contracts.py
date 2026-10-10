"""Six-backend motion, configuration ownership and archive trust boundaries."""
from dataclasses import replace, fields
from pathlib import Path
import json
import shutil
from uuid import uuid4
import unittest
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from openfawe.aero.methods import AeroBackend, AeroMethod, rigid_velocity_field, rigid_velocity
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings
from openfawe.aero.lifting_surface import LiftingSurfaceConfig
from openfawe.aero.vsm import solve_vsm, solve_vsm_operator, horseshoe_influence
from openfawe.aero_archive import AeroReader, AeroWriter, replay_config
from openfawe.aero_vtk import export_paraview, frame_network
from openfawe.provenance import sha256_file


class AeroContractTests(unittest.TestCase):
    def setUp(self):
        self.wing = Wing([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]],
                         [[1., -2., 0.], [1., 0., 0.], [1., 2., 0.]])
        self.polar = SectionPolar.linear()
        self.wind = [5., 0., .2]
        self.config = WakeConfig(dt_s=.02, solve=SolveSettings(solver='newton', max_iterations=60))
        parent = Path(__file__).resolve().parents[1]/'local'/'contract-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.root = parent/uuid4().hex
        self.root.mkdir()
        self.addCleanup(self.cleanup_output, parent)

    def cleanup_output(self, parent):
        if self.root.resolve().parent != parent.resolve():
            raise ValueError('refusing cleanup outside test directory')
        shutil.rmtree(self.root)

    def backend(self, method):
        return AeroBackend(method, wake=self.config, lifting_surface=LiftingSurfaceConfig(2))

    def test_spatial_rigid_velocity_contract_all_six_methods(self):
        omega, translation = np.array([0., .02, 0.]), np.array([.1, 0., .03])
        theta = omega[1]*self.config.dt_s
        rotation = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.],
                             [-np.sin(theta), 0., np.cos(theta)]])
        moved = self.wing.transformed(rotation, translation*self.config.dt_s)
        for method in AeroMethod:
            with self.subTest(method=method):
                backend = self.backend(method)
                body = rigid_velocity_field(translation, omega)
                state = backend.start(self.wing, self.polar, self.wind, structural_velocity=body)
                updated_body = rigid_velocity_field(translation, omega, translation*self.config.dt_s)
                result = backend.advance(state, self.polar, self.wind, next_wing=moved,
                                         structural_velocity=updated_body)
                points = backend.evaluation_points(moved)
                saved = (result.lattice_structural_velocity if method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE)
                         else result.structural_velocity)
                np.testing.assert_allclose(saved, rigid_velocity(points, translation, omega, translation*self.config.dt_s))
                self.assertTrue(np.isfinite(result.solution.force).all())
                self.assertLessEqual(result.solution.residual, result.solution.residual_limit)
        for method in ('vlm', 'vlm_free_wake'):
            backend = self.backend(method)
            with self.assertRaisesRegex(ValueError, 'spatial callable'):
                backend.start(self.wing, self.polar, self.wind,
                    structural_velocity=rigid_velocity(backend.evaluation_points(self.wing), translation, omega))

    def test_config_ownership_applies_to_advances_and_field_probes(self):
        for method in AeroMethod:
            with self.subTest(method=method):
                backend = self.backend(method)
                state = backend.start(self.wing, self.polar, self.wind)
                changed = replace(backend, wake=replace(self.config, density_kg_m3=2.))
                for operation in (lambda: changed.advance(state, self.polar, self.wind),
                                  lambda: changed.induced_velocity(state, [[2., 0., 2.]])):
                    with self.assertRaisesRegex(ValueError, 'config'):
                        operation()
                self.assertEqual(state.step, 0)

    def test_vsm_spatial_load_point_sampling_and_unchanged_closure(self):
        def sheared(points, time):
            flow = np.broadcast_to(self.wind, points.shape).copy()
            flow[:, 2] += points[:, 0]-.75
            return flow

        # This rigid pitch field is zero at closure and nonzero at load points.
        body = rigid_velocity_field([0., 0., 0.], [0., 1., 0.], [.75, 0., 0.])
        for method in ('vsm', 'vsm_free_wake'):
            with self.subTest(method=method):
                backend = self.backend(method)
                baseline = backend.start(self.wing, self.polar, self.wind)
                shear = backend.start(self.wing, self.polar, sheared)
                pitch = backend.start(self.wing, self.polar, self.wind, structural_velocity=body)
                for state in (shear, pitch):
                    np.testing.assert_array_equal(state.solution.circulation, baseline.solution.circulation)
                    np.testing.assert_array_equal(state.solution.polar_alpha_rad, baseline.solution.polar_alpha_rad)
                    self.assertGreater(np.max(abs(state.solution.panel_force-baseline.solution.panel_force)), .1)
                    np.testing.assert_allclose(state.solution.relative_velocity-state.solution.induced_velocity,
                                               sheared(self.wing.cp, 0.), atol=1e-14, rtol=0)
                    np.testing.assert_array_equal(state.solution.closure_ambient_velocity,
                                                  np.broadcast_to(self.wind, (self.wing.panels, 3)))
                next_baseline = backend.advance(baseline, self.polar, self.wind)
                next_pitch = backend.advance(pitch, self.polar, self.wind, structural_velocity=body)
                np.testing.assert_array_equal(next_baseline.solution.circulation, next_pitch.solution.circulation)
                np.testing.assert_allclose(next_pitch.solution.relative_velocity-next_pitch.solution.induced_velocity,
                                           np.array(self.wind)-body(self.wing.cp, next_pitch.time_s), atol=1e-14)
                self.assertGreater(np.max(abs(next_pitch.solution.panel_force-next_baseline.solution.panel_force)), .1)
                self.assertIn('quarter chord', backend.metadata()['load_point_ambient_sampling'])

    def test_vsm_uniform_vector_array_callable_and_low_level_compatibility(self):
        wind = np.asarray(self.wind)
        for method in ('vsm', 'vsm_free_wake'):
            backend = self.backend(method)
            vector = backend.start(self.wing, self.polar, wind)
            array = backend.start(self.wing, self.polar, np.broadcast_to(wind, (self.wing.panels, 3)))
            function = backend.start(self.wing, self.polar, lambda points, time: np.broadcast_to(wind, points.shape))
            for candidate in (array, function):
                for field in fields(vector.solution):
                    np.testing.assert_array_equal(getattr(candidate.solution, field.name),
                                                  getattr(vector.solution, field.name))
            if method == 'vsm':
                legacy = solve_vsm(self.wing, self.polar, wind, settings=backend.vsm_solve)
                for field in fields(legacy):
                    if field.name == 'load_ambient_sampling':
                        continue
                    np.testing.assert_array_equal(getattr(legacy, field.name), getattr(vector.solution, field.name))
                self.assertEqual(legacy.load_ambient_sampling, 'legacy_closure_reuse')
                self.assertEqual(vector.solution.load_ambient_sampling, 'separate_load_input_v1')
                probe = np.array([[2., 1., 2.]])
                expected = np.einsum('ijk,j->ik', horseshoe_influence(self.wing, vector.solution.wake_direction,
                    probe, source_core_speed=float(np.linalg.norm(wind)),
                    source_core_policy='source_legacy'), vector.solution.circulation)
                np.testing.assert_array_equal(backend.induced_velocity(vector, probe), expected)
                changed = replace(vector, solution=replace(vector.solution,
                    relative_velocity=vector.solution.relative_velocity+[0., 0., 3.]))
                np.testing.assert_array_equal(backend.induced_velocity(changed, probe), expected)

    def test_vsm_zero_closure_and_explicit_load_ambient_are_finite(self):
        result = solve_vsm(self.wing, self.polar, [0., 0., 0.], load_relative_ambient=[1., 0., .1])
        np.testing.assert_array_equal(result.circulation, [0., 0.])
        np.testing.assert_array_equal(result.panel_force, np.zeros((2, 3)))
        np.testing.assert_array_equal(result.relative_velocity, [[1., 0., .1]]*2)
        np.testing.assert_array_equal(result.closure_ambient_velocity, np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            solve_vsm(self.wing, self.polar, self.wind, load_relative_ambient=[np.nan, 0., 0.])

    def test_vsm_load_flow_domain_is_checked_before_forces(self):
        for load in ([-5., 0., .2], [0., 5., 0.], [[5., 0., .2], [0., 0., 0.]]):
            with self.subTest(load=load), self.assertRaisesRegex(ValueError, 'VSM load'):
                solve_vsm(self.wing, self.polar, self.wind, load_relative_ambient=load)
        operator = np.zeros((self.wing.panels, self.wing.panels, 3))
        for known in ([-10., 0., 0.], [-5., 2., -.2], [-5., 0., -.2]):
            with self.subTest(known=known), self.assertRaisesRegex(ValueError, 'VSM load total'):
                solve_vsm_operator(self.wing, self.polar, self.wind, operator, operator,
                    known_load_induction=known)
        for method in ('vsm', 'vsm_free_wake'):
            def reversing(points, time):
                return np.column_stack((20.*(points[:, 0]-.5), np.zeros(len(points)), np.full(len(points), .2)))
            with self.assertRaisesRegex(ValueError, 'VSM load ambient requires forward'):
                self.backend(method).start(self.wing, self.polar, reversing)

    def test_vsm_reader_distinguishes_legacy_sampling_without_rewriting(self):
        for method in ('vsm', 'vsm_free_wake'):
            path = self.root/method
            writer = AeroWriter(path, {'aero_method': method})
            writer.append(self.backend(method).start(self.wing, self.polar, self.wind))
            writer.finish()
            reader = AeroReader(path)
            try:
                self.assertEqual(reader.frame(0)['vsm_load_ambient'], 'separate_load_input_v1')
            finally:
                reader.close()
            # Construct a pre-fix fixture, not a modification of a retained run.
            with h5py.File(path/'wake.h5', 'r+') as file:
                group = file['frames/000000']
                del group['closure_ambient_velocity']
                for attrs in (file.attrs, group.attrs):
                    for name in ('vsm_load_ambient', 'load_point_ambient_sampling'):
                        if name in attrs:
                            del attrs[name]
            manifest = json.loads((path/'manifest.json').read_text())
            manifest['wake_sha256'] = sha256_file(path/'wake.h5')
            (path/'manifest.json').write_text(json.dumps(manifest))
            before = {p.name:sha256_file(p) for p in path.iterdir() if p.is_file()}
            reader = AeroReader(path)
            try:
                frame = reader.frame(0)
                self.assertEqual(frame['vsm_load_ambient'], 'legacy_closure_reuse')
                self.assertIn('closure-point ambient', frame['alpha_rad_definition'])
                self.assertIn('Legacy', frame['load_point_ambient_sampling'])
            finally:
                reader.close()
            self.assertEqual(before, {p.name:sha256_file(p) for p in path.iterdir() if p.is_file()})

    def test_vsm_array_without_marker_is_unrecorded_and_root_does_not_guess(self):
        path = self.root/'unrecorded'
        writer = AeroWriter(path, {'aero_method':'vsm'})
        self.addCleanup(lambda: writer.file.close() if writer.file else None)
        self.assertNotIn('vsm_load_ambient', writer.file.attrs)
        self.assertNotIn('load_point_ambient_sampling', writer.file.attrs)
        writer.append(self.backend('vsm').start(self.wing, self.polar, self.wind))
        writer.finish()
        with h5py.File(path/'wake.h5', 'r+') as file:
            group = file['frames/000000']
            self.assertIn('closure_ambient_velocity', group)
            del group.attrs['vsm_load_ambient']
            # A root-level claim must not substitute for the missing frame marker.
            file.attrs['vsm_load_ambient'] = 'separate_load_input_v1'
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        manifest['wake_sha256'] = sha256_file(path/'wake.h5')
        (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        before = sha256_file(path/'wake.h5')
        reader = AeroReader(path)
        try:
            frame = reader.frame(0)
            self.assertEqual(frame['vsm_load_ambient'], 'unrecorded')
            self.assertIn('unrecorded', frame['alpha_rad_definition'])
        finally:
            reader.close()
        self.assertEqual(before, sha256_file(path/'wake.h5'))
        export_paraview(path, self.root/'unrecorded-vtk')
        for name in ('wing-000000.vtp', 'wake-000000.vtp'):
            data = {i.attrib['Name']:i.text for i in ET.parse(self.root/'unrecorded-vtk'/name).findall('.//FieldData/DataArray')}
            self.assertEqual(float(data['vsm_load_ambient_0_legacy_closure_reuse_1_separate_input_v1_2_unrecorded']), 2.)
            self.assertNotIn('vsm_load_ambient_0_legacy_closure_reuse_1_separate_input_v1', data)

    def test_invalid_residual_is_atomic_then_six_minimal_archives_export(self):
        for method in AeroMethod:
            with self.subTest(method=method):
                backend = self.backend(method)
                state = backend.start(self.wing, self.polar, self.wind)
                path = self.root/method.value
                writer = AeroWriter(path, {'aero_method': method.value})
                self.addCleanup(lambda w=writer: w.file.close() if w.file else None)
                bad = replace(state, solution=replace(state.solution, residual=float('nan')))
                with self.assertRaises(ValueError):
                    writer.append(bad)
                self.assertEqual(writer.count, 0)
                self.assertEqual(len(writer.file['frames']), 0)
                writer.append(state)
                moved = self.wing.transformed(np.eye(3), [self.config.dt_s, 0., 0.])
                state = backend.advance(state, self.polar, self.wind, next_wing=moved)
                writer.append(state)
                writer.finish()
                reader = AeroReader(path)
                try:
                    frame = reader.frame(1)
                    self.assertAlmostEqual(frame['kinematic_mismatch_m_s'], 1., places=12)
                    self.assertEqual(replay_config({}, frame)['dt_s'], self.config.dt_s)
                    self.assertEqual(frame['unsteady_term'], 'rho_dgamma_dt_area_normal' if method==AeroMethod.VLM_FREE_WAKE else 'none')
                    if method in (AeroMethod.VSM, AeroMethod.VSM_FREE_WAKE):
                        np.testing.assert_array_equal(frame['closure_ambient_velocity'], state.solution.closure_ambient_velocity)
                        self.assertIn('quarter chord', frame['load_point_ambient_sampling'])
                    if method != AeroMethod.VSM:
                        for actual, saved in zip(frame_network(frame), ('network_nodes_m','network_edges','network_circulation_m2_s')):
                            np.testing.assert_array_equal(actual, frame[saved])
                finally:
                    reader.close()
                output = self.root/(method.value+'-vtk')
                export_paraview(path, output)
                for name in ('wing-000001.vtp', 'wake-000001.vtp'):
                    data = {item.attrib['Name']:item.text for item in ET.parse(output/name).findall('.//FieldData/DataArray')}
                    self.assertEqual(float(data['force_model_0_quasi_steady_polar_1_bound_edge_joukowski']),
                                     int(method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE)))
                    self.assertEqual(float(data['unsteady_term_0_none_1_rho_dgamma_dt_area_normal']),
                                     int(method == AeroMethod.VLM_FREE_WAKE))

    def test_reader_rejects_nonfinite_residual_even_with_updated_content_hash(self):
        for method in AeroMethod:
            backend = self.backend(method)
            path = self.root/method.value
            writer = AeroWriter(path, {'aero_method': method.value})
            writer.append(backend.start(self.wing, self.polar, self.wind))
            writer.finish()
            with h5py.File(path/'wake.h5', 'r+') as file:
                key = 'boundary_residual_m_s' if method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE) else 'residual_m2_s'
                file['frames/000000'].attrs[key] = np.nan
            manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            manifest['wake_sha256'] = sha256_file(path/'wake.h5')
            (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'residual'):
                AeroReader(path)

    def test_lattice_condition_limit_is_explicit_and_default_is_diagnostic(self):
        backend = self.backend('vlm')
        state = backend.start(self.wing, self.polar, self.wind)
        condition = state.solution.matrix_condition_number
        self.assertGreater(condition, 1.)
        restricted = replace(backend, lifting_surface=replace(backend.lifting_surface, max_condition_number=1.))
        with self.assertRaisesRegex(ValueError, 'condition'):
            restricted.start(self.wing, self.polar, self.wind)


if __name__ == '__main__':
    unittest.main()
