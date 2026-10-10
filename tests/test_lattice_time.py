"""Optional UVLM time schemes: physical residuals, history and corrupt replay."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
import xml.etree.ElementTree as ET
from uuid import uuid4

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, lifting_surface_induced_velocity, lifting_surface_network,
    build_lifting_surface_mesh, _solve)
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing
from openfawe.aero_archive import AeroReader, AeroWriter
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file


def wing():
    return Wing([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]],
                [[1., -2., 0.], [1., 0., 0.], [1., 2., 0.]])


class LatticeTimeTests(unittest.TestCase):
    def test_public_guards_and_missing_multistep_history(self):
        for kwargs in ({'wake_release':'bad'}, {'gamma_derivative':'rk4'}):
            with self.assertRaises(ValueError):
                LiftingSurfaceConfig(**kwargs)
        for kwargs in ({'wake_release':'current'}, {'gamma_derivative':'bdf2'}):
            settings = LiftingSurfaceConfig(**kwargs)
            with self.assertRaisesRegex(ValueError, 'transient'):
                AeroBackend('vlm', lifting_surface=settings)
            with self.assertRaisesRegex(ValueError, 'transient'):
                start_lifting_surface(wing(), WakeConfig(), [4., 0., .1], settings=settings)
        config = WakeConfig(dt_s=.02)
        state = start_lifting_surface(wing(), config, [4.,0.,.1], free_wake=True,
            settings=LiftingSurfaceConfig(2, gamma_derivative='bdf2'))
        state = advance_lifting_surface(state, config, [4.,0.,.1])
        with self.assertRaisesRegex(ValueError, 'history'):
            replace(state, previous_lattice_circulation=None)
        with self.assertRaises(ValueError):
            state.previous_lattice_circulation.setflags(write=True)

    def test_all_release_derivative_pairs_physical_boundary_and_material_history(self):
        for release, theta in [('previous',0.), ('current',1.), ('trapezoidal',.5)]:
            for derivative in ('backward_euler', 'bdf2'):
                with self.subTest(release=release, derivative=derivative):
                    config = WakeConfig(dt_s=.02, core_evolution='age_stretch')
                    settings = LiftingSurfaceConfig(3, chordwise_spacing='cosine',
                        wake_release=release, gamma_derivative=derivative)
                    state = start_lifting_surface(wing(), config, [4.,0.,.2],
                        settings=settings, free_wake=True)
                    for k in range(1, 4):
                        old = state
                        wind = np.array([4., .1*k, .2+.05*k])
                        state = advance_lifting_surface(old, config, wind)
                        repeat = advance_lifting_surface(old, config, wind)
                        assert_array_equal(state.solution.force, repeat.solution.force)
                        assert_array_equal(state.wake_circulation[1:], old.wake_circulation)
                        assert_allclose(state.wake_circulation[0],
                            (1-theta)*old.solution.circulation+theta*state.solution.circulation,
                            rtol=0, atol=0)
                        actual_velocity = wind+lifting_surface_induced_velocity(
                            state, state.mesh.collocation_points)
                        assert_allclose(np.sum(actual_velocity*state.mesh.normals, axis=1),
                                        0., atol=1e-12)
                        nodes, edges, gamma = lifting_surface_network(state)
                        balance = np.zeros(len(nodes))
                        np.add.at(balance, edges[:,0], gamma)
                        np.add.at(balance, edges[:,1], -gamma)
                        assert_allclose(balance, 0., atol=2e-15)
                        if k == 1:
                            assert_array_equal(state.solution.lattice_gamma_rate,
                                (state.solution.lattice_circulation-old.solution.lattice_circulation)/config.dt_s)

    def test_bdf2_quadratic_rate_manufactured_solution(self):
        # Gamma(t)=a*t^2 has exactly zero derivative at t=0. The current
        # no-wake, zero-normal-inflow equation has Gamma=0 independently.
        w = wing(); config = WakeConfig(dt_s=.03)
        settings = LiftingSurfaceConfig(3, gamma_derivative='bdf2')
        mesh = build_lifting_surface_mesh(w, settings)
        a = np.arange(1, 7).reshape(3, 2)
        solution, _ = _solve(w, mesh, settings, config, [4.,0.,0.], [0.,0.,0.],
            0., np.empty((0,3,3)), np.empty((0,2)), None,
            previous=a*config.dt_s**2, previous_previous=a*(2*config.dt_s)**2)
        assert_allclose(solution.lattice_gamma_rate, 0., atol=1e-16)
        assert_allclose(solution.force, 0., atol=1e-15)

    def test_archive_new_history_and_corruptions_with_refreshed_outer_hash(self):
        parent = Path(__file__).resolve().parents[1]/'local/lattice-time-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex; root.mkdir()
        try:
            config = WakeConfig(dt_s=.03)
            for release in ('previous', 'current', 'trapezoidal'):
                for derivative in ('backward_euler', 'bdf2'):
                    path = root/f'{release}-{derivative}'
                    settings = LiftingSurfaceConfig(2, wake_release=release, gamma_derivative=derivative)
                    state = start_lifting_surface(wing(), config, [4.,0.,.1],
                        settings=settings, free_wake=True)
                    writer = AeroWriter(path, {'aero_method':'vlm_free_wake'})
                    try:
                        writer.append(state)
                        for k in range(3):
                            state = advance_lifting_surface(state, config, [4.,0.,.2+k*.03])
                            writer.append(state)
                        writer.finish()
                    except BaseException as error:
                        writer.fail(error); raise
                    reader = AeroReader(path)
                    try:
                        frame = reader.frame(3)
                        self.assertEqual(frame['gamma_rate_stencil'], derivative)
                        self.assertEqual(frame['wake_release'], release)
                        self.assertEqual(frame['lattice_time_status'], 'explicit_history_checked_v1')
                    finally:
                        reader.close()
            source = root/'trapezoidal-bdf2'
            for field in ('lattice_gamma_rate_m2_s2', 'previous_lattice_circulation_m2_s',
                          'gamma_rate_stencil', 'condition_operator', 'lattice_time_contract',
                          'wake_circulation_m2_s', 'trailing_edge_jump_m2_s'):
                bad = root/('corrupt-'+field); shutil.copytree(source, bad)
                with h5py.File(bad/'wake.h5', 'r+') as file:
                    frame = file['frames/000003']
                    if field in frame:
                        frame[field][0] += .1
                    elif field == 'lattice_time_contract':
                        del frame.attrs[field]
                    else:
                        frame.attrs[field] = 'incorrect'
                manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
                manifest['wake_sha256'] = sha256_file(bad/'wake.h5')
                (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'lattice|Gamma'):
                    AeroReader(bad)
            # A genuine legacy lattice has neither new settings nor time fields.
            legacy = root/'legacy'; shutil.copytree(root/'previous-backward_euler', legacy)
            with h5py.File(legacy/'wake.h5', 'r+') as file:
                for frame in file['frames'].values():
                    settings = json.loads(frame.attrs['lattice_settings_json'])
                    for name in ('wake_release', 'gamma_derivative'):
                        del settings[name]
                    frame.attrs['lattice_settings_json'] = json.dumps(settings)
                    for name in ('lattice_time_contract', 'wake_release', 'gamma_derivative',
                                 'gamma_rate_stencil', 'condition_operator', 'trailing_edge_jump_definition'):
                        del frame.attrs[name]
                    del frame['previous_lattice_circulation_m2_s']
            manifest = json.loads((legacy/'manifest.json').read_text(encoding='utf-8'))
            manifest['wake_sha256'] = sha256_file(legacy/'wake.h5')
            (legacy/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            before = sha256_file(legacy/'wake.h5')
            reader = AeroReader(legacy)
            try:
                self.assertEqual(reader.frame(3)['lattice_time_status'],
                                 'legacy_previous_BE; history not independently verified')
            finally:
                reader.close()
            export_paraview(legacy, root/'legacy-vtk')
            tree = ET.parse(root/'legacy-vtk/wing-000003.vtp')
            flag = tree.find('.//FieldData/DataArray[@Name="time_history_checked_0_legacy_1_explicit"]')
            self.assertEqual(float(flag.text), 0.)
            self.assertEqual(sha256_file(legacy/'wake.h5'), before)
        finally:
            if root.resolve().parent != parent.resolve():
                raise ValueError('Refusing test cleanup outside its workspace')
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
