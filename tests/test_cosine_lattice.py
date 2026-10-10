"""Independent geometry, line-vortex and replay checks for chord clustering."""
from dataclasses import fields, replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, build_lifting_surface_mesh,
    lifting_surface_influence, start_lifting_surface, advance_lifting_surface)
from openfawe.aero.vortex import CoreModel
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.methods import AeroBackend
from openfawe.aero_archive import AeroReader, AeroWriter
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file


class CosineLatticeTests(unittest.TestCase):
    def test_option_validation_and_one_panel_exact_equivalence(self):
        for value in ('', 'COSINE', 'linear', None, 0):
            with self.assertRaisesRegex(ValueError, 'spacing'):
                LiftingSurfaceConfig(chordwise_spacing=value)
        wing = Wing([[0., -2., 0.], [.1, 2., .2]], [[1., -2., 0.], [.8, 2., .5]])
        meshes = [build_lifting_surface_mesh(wing, LiftingSurfaceConfig(1, chordwise_spacing=s))
                  for s in ('uniform', 'cosine')]
        for field in fields(meshes[0]):
            assert_array_equal(getattr(meshes[0], field.name), getattr(meshes[1], field.name))

    def test_cosine_twisted_tapered_geometry_and_vector_area(self):
        le = np.array([[0., -2., 0.], [.3, 2., .1]])
        te = np.array([[1.2, -2., .1], [1., 2., .6]])
        wing = Wing(le, te)
        mesh = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(5, chordwise_spacing='cosine'))
        # Independently use sin squared rather than the source cosine expression.
        fractions = np.sin(np.arange(6)*np.pi/10)**2
        for i, (a, b) in enumerate(zip(fractions[:-1], fractions[1:])):
            cp_fraction = a+.75*(b-a)
            chord = te-le
            assert_allclose(mesh.collocation_points[i], np.mean(le+cp_fraction*chord, axis=0), atol=1e-15)
            assert_allclose(mesh.vortex_rows[i], le+(a+.25*(b-a))*chord, atol=1e-15)
            t = np.mean(chord, axis=0)
            s = le[1]-le[0]+cp_fraction*(chord[1]-chord[0])
            normal = np.cross(t, s)
            assert_allclose(mesh.normals[i], normal/np.linalg.norm(normal), atol=1e-15)
            p = mesh.vertices[mesh.cells[i]]
            polygon_area = -.5*np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0)
            assert_allclose(mesh.area[i]*mesh.area_normals[i], polygon_area, atol=1e-15)
        total = np.sum(mesh.area[:, None]*mesh.area_normals, axis=0)
        full = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(1))
        assert_allclose(total, full.area[0]*full.area_normals[0], atol=2e-15)

    def test_cosine_source_matrix_matches_infinite_span_line_vortices(self):
        wing = Wing([[0., -2000., 0.], [0., 2000., 0.]], [[1., -2000., 0.], [1., 2000., 0.]])
        config = WakeConfig(core_model=CoreModel.NONE)
        for nc in (4, 8, 16):
            mesh = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(nc, chordwise_spacing='cosine'))
            x = mesh.collocation_points[:, 0, None]
            front, back = mesh.vortex_rows[:-1, 0, 0], mesh.vortex_rows[1:, 0, 0]
            expected = (-1/(x-front)+1/(x-back))/(2*np.pi)
            actual = lifting_surface_influence(wing, mesh, config)[:, :, 2]
            assert_allclose(actual, expected, rtol=0, atol=1e-8)

    def test_cosine_force_moment_reduction_and_archive_provenance(self):
        parent = Path(__file__).resolve().parents[1]/'local'/'cosine-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex
        root.mkdir()
        self.addCleanup(self.clean, root, parent)
        wing = Wing([[0., -2., 0.], [0., 0., .1], [.2, 2., .3]],
                    [[1., -2., 0.], [1., 0., .2], [.9, 2., .5]])
        settings = LiftingSurfaceConfig(4, chordwise_spacing='cosine', max_condition_number=1e8)
        for method in ('vlm', 'vlm_free_wake'):
            backend = AeroBackend(method, wake=WakeConfig(dt_s=.03), lifting_surface=settings)
            self.assertEqual(backend.metadata()['lifting_surface']['chordwise_spacing'], 'cosine')
            state = backend.start(wing, SectionPolar.linear(), [5., .1, .2])
            path = root/method
            writer = AeroWriter(path, {'aero_method':method})
            writer.append(state)
            state = backend.advance(state, SectionPolar.linear(), [5., .1, .25])
            writer.append(state)
            writer.finish()
            force = state.solution.lattice_force.reshape(-1, 3)
            couple = state.solution.lattice_couple.reshape(-1, 3)
            assert_allclose(force.sum(axis=0), state.solution.force, rtol=1e-13, atol=1e-13)
            lattice_moment = np.sum(couple+np.cross(state.mesh.collocation_points, force), axis=0)
            strip_moment = np.sum(state.solution.panel_couple+np.cross(wing.cp, state.solution.panel_force), axis=0)
            assert_allclose(lattice_moment, strip_moment, rtol=1e-13, atol=1e-13)
            self.assertLessEqual(state.solution.residual, state.solution.residual_limit)
            reader = AeroReader(path)
            try:
                frame = reader.frame(1)
                self.assertEqual(frame['chordwise_spacing'], 'cosine')
                self.assertEqual(json.loads(frame['lattice_settings_json'])['max_condition_number'], 1e8)
                self.assertEqual(frame['condition_limit_status'], 'passed')
                assert_array_equal(frame['lattice_nodes_m'], state.mesh.vertices)
            finally:
                reader.close()
            export_paraview(path, root/(method+'-vtk'))
            for name in ('wing-000001.vtp', 'wake-000001.vtp'):
                data = {i.attrib['Name']:i.text for i in ET.parse(root/(method+'-vtk')/name).findall('.//FieldData/DataArray')}
                self.assertEqual(float(data['chordwise_spacing_0_uniform_1_cosine']), 1.)
                self.assertGreater(float(data['condition_roundoff_indicator']), 0.)

    def test_legacy_uniform_archive_defaults_and_spacing_mismatch(self):
        parent = Path(__file__).resolve().parents[1]/'local'/'cosine-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex
        root.mkdir()
        self.addCleanup(self.clean, root, parent)
        wing = Wing([[0., -2., 0.], [0., 2., 0.]], [[1., -2., 0.], [1., 2., 0.]])
        backend = AeroBackend('vlm')
        path = root/'legacy'
        writer = AeroWriter(path, {'aero_method':'vlm'})
        writer.append(backend.start(wing, SectionPolar.linear(), [5., 0., .2]))
        writer.finish()
        with h5py.File(path/'wake.h5', 'r+') as file:
            attrs = file['frames/000000'].attrs
            del attrs['chordwise_spacing']
            settings = json.loads(attrs['lattice_settings_json'])
            del settings['chordwise_spacing']
            attrs['lattice_settings_json'] = json.dumps(settings)
        def update_fixture_hash():
            manifest = json.loads((path/'manifest.json').read_text())
            manifest['wake_sha256'] = sha256_file(path/'wake.h5')
            (path/'manifest.json').write_text(json.dumps(manifest))
        update_fixture_hash()
        before = sha256_file(path/'wake.h5')
        reader = AeroReader(path)
        try:
            self.assertEqual(reader.frame(0)['chordwise_spacing'], 'uniform')
        finally:
            reader.close()
        self.assertEqual(before, sha256_file(path/'wake.h5'))
        with h5py.File(path/'wake.h5', 'r+') as file:
            file['frames/000000'].attrs['chordwise_spacing'] = 'cosine'
        update_fixture_hash()
        with self.assertRaisesRegex(ValueError, 'spacing'):
            AeroReader(path)

    def test_archive_checks_saved_mesh_and_local_time_resolution(self):
        parent = Path(__file__).resolve().parents[1]/'local'/'cosine-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex
        root.mkdir()
        self.addCleanup(self.clean, root, parent)
        wing = Wing([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]],
                    [[1., -2., 0.], [1.5, 0., 0.], [2., 2., 0.]])
        cfg = WakeConfig(dt_s=.03)
        settings = LiftingSurfaceConfig(4, chordwise_spacing='cosine')
        backend = AeroBackend('vlm_free_wake', wake=cfg, lifting_surface=settings)
        path = root/'original'
        writer = AeroWriter(path, {'aero_method':'vlm_free_wake'})
        writer.append(backend.start(wing, SectionPolar.linear(), [5., 0., .2]))
        writer.finish()
        reader = AeroReader(path)
        try:
            frame = reader.frame(0)
            smallest = 1.25*np.sin(np.pi/8)**2
            self.assertAlmostEqual(frame['minimum_chordwise_cell_length_m'], smallest, places=14)
            self.assertAlmostEqual(frame['prescribed_step_to_chord_ratio'],
                                   cfg.dt_s*np.hypot(5., .2)/smallest, places=13)
        finally:
            reader.close()
        for error in ('false_uniform_label', 'changed_vortex_row'):
            altered = root/error
            shutil.copytree(path, altered)
            with h5py.File(altered/'wake.h5', 'r+') as file:
                group = file['frames/000000']
                if error == 'false_uniform_label':
                    values = json.loads(group.attrs['lattice_settings_json'])
                    values['chordwise_spacing'] = 'uniform'
                    group.attrs['lattice_settings_json'] = json.dumps(values)
                    group.attrs['chordwise_spacing'] = 'uniform'
                else:
                    group['lattice_vortex_rows_m'][1, 0, 0] += .001
            manifest = json.loads((altered/'manifest.json').read_text(encoding='utf-8'))
            manifest['wake_sha256'] = sha256_file(altered/'wake.h5')
            (altered/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, 'mesh'):
                AeroReader(altered)

    @staticmethod
    def clean(root, parent):
        if root.resolve().parent != parent.resolve():
            raise ValueError('refusing cleanup outside test directory')
        shutil.rmtree(root)

    def test_saved_network_must_match_straight_and_free_lattice_states(self):
        parent = Path(__file__).resolve().parents[1]/'local'/'cosine-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex
        root.mkdir()
        self.addCleanup(self.clean, root, parent)
        wing = Wing([[0., -2., 0.], [0., 2., 0.]], [[1., -2., 0.], [1., 2., 0.]])
        for method in ('vlm', 'vlm_free_wake'):
            backend = AeroBackend(method, wake=WakeConfig(dt_s=.02),
                lifting_surface=LiftingSurfaceConfig(4, chordwise_spacing='cosine'))
            path = root/method
            writer = AeroWriter(path, {'aero_method':method})
            state = backend.start(wing, SectionPolar.linear(), [5., 0., .2])
            writer.append(state)
            writer.append(backend.advance(state, SectionPolar.linear(), [5., 0., .3]))
            writer.finish()
            for field in ('network_circulation_m2_s', 'network_nodes_m', 'network_edges'):
                altered = root/(method+'-'+field)
                shutil.copytree(path, altered)
                with h5py.File(altered/'wake.h5', 'r+') as file:
                    dataset = file['frames/000001'][field]
                    if field == 'network_circulation_m2_s':
                        dataset[-1] += .01
                    elif field == 'network_nodes_m':
                        dataset[-1, 0] += .01
                    else:
                        dataset[0, :] = dataset[0, :][::-1]
                manifest = json.loads((altered/'manifest.json').read_text(encoding='utf-8'))
                manifest['wake_sha256'] = sha256_file(altered/'wake.h5')
                (altered/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.subTest(method=method, field=field), self.assertRaisesRegex(ValueError, 'network'):
                    AeroReader(altered)


if __name__ == '__main__':
    unittest.main()
