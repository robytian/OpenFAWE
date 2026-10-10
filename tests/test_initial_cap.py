"""Independent initial kernel, right limits and non-downgradable archive checks."""

from contextlib import closing
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, lifting_surface_network, lifting_surface_edge_radii,
    lifting_surface_induced_velocity, _linear_convection_network)
from openfawe.aero.polar_coupling import encode_polars, polar_tables, polar_identity
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.wake import WakeConfig, pack_rings
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero_archive import (AeroWriter, AeroReader, INITIAL_CAP_LATTICE_SCHEMA,
    LATTICE_SCHEMA, POLAR_LATTICE_SCHEMA, POTENTIAL_LATTICE_SCHEMA,
    POTENTIAL_POLAR_LATTICE_SCHEMA)
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file
from openfawe.run_catalog import inspect_run


ROOT = Path(__file__).resolve().parents[1]


def wing():
    le = np.array([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]])
    return Wing(le, le+[1., 0., 0.])


def settings(basis, **changes):
    return replace(LiftingSurfaceConfig(8, chordwise_spacing='cosine', initial_cap='wake_core',
        wake_basis=basis, wake_release='current' if basis == 'linear_potential' else 'previous',
        gamma_derivative='bdf2' if basis == 'linear_potential' else 'backward_euler',
        wake_quadrature='te_graded' if basis == 'linear_potential' else 'gauss'), **changes)


class InitialCapTests(unittest.TestCase):
    def test_initial_operator_and_probe_match_independent_unit_graphs(self):
        w, wind = wing(), np.array([4., 0., .1])
        for basis in ('constant', 'linear_potential'):
            for model in (CoreModel.NONE, CoreModel.RANKINE, CoreModel.LAMB_OSEEN, CoreModel.VATISTAS):
                with self.subTest(basis=basis, model=model):
                    cfg = WakeConfig(dt_s=.01, core_model=model, core_radius_m=.05)
                    state = start_lifting_surface(w, cfg, wind, settings=settings(basis), free_wake=True)
                    mesh, nc = state.mesh, state.settings.chordwise_panels
                    operator = np.empty((mesh.panels, mesh.panels, 3))
                    for k in range(mesh.panels):
                        unit = np.zeros_like(state.solution.lattice_circulation)
                        unit.flat[k] = 1.
                        network = pack_rings(mesh.vortex_rows, unit, w.panel_edges)
                        cap = np.all(network[1]//w.node_count == nc, axis=1)
                        radii = np.where(cap, cfg.core_radius_m, 0.) if model != CoreModel.NONE else np.zeros(len(cap))
                        operator[:, k] = segments_velocity(mesh.collocation_points, *network,
                            model=model, core_radius=radii)
                    matrix = np.einsum('ijk,ik->ij', operator, mesh.normals)
                    expected = np.linalg.solve(matrix, -mesh.normals@wind)
                    assert_allclose(state.solution.lattice_circulation.ravel(), expected, atol=4e-12, rtol=4e-12)
                    nodes, edges, gamma = lifting_surface_network(state)
                    cap = np.all(edges//w.node_count == nc, axis=1)
                    radii = np.where(cap, cfg.core_radius_m, 0.) if model != CoreModel.NONE else np.zeros(len(cap))
                    assert_array_equal(lifting_surface_edge_radii(state), radii)
                    field = segments_velocity(mesh.collocation_points, nodes, edges, gamma,
                        model=model, core_radius=radii)
                    assert_array_equal(lifting_surface_induced_velocity(state, mesh.collocation_points), field)
                    assert_allclose(np.sum((wind+field)*mesh.normals, axis=1), 0., atol=3e-12)
                    if basis == 'linear_potential':
                        stage, stage_radii, _ = _linear_convection_network(state, cfg, w.trailing_edge, 0.)
                        for actual, original in zip(stage, (nodes, edges, gamma)):
                            assert_array_equal(actual, original)
                        assert_array_equal(stage_radii, radii)

    def test_retained_core_limits_have_no_fixed_jump(self):
        w, wind = wing(), np.array([4., 0., .1])
        for basis in ('constant', 'linear_potential'):
            for model, radius in ((CoreModel.NONE, .05), (CoreModel.LAMB_OSEEN, .15),
                                  (CoreModel.LAMB_OSEEN, .05), (CoreModel.LAMB_OSEEN, .015)):
                errors, rates = [], []
                for dt in (1e-5, 1e-7, 1e-9):
                    cfg = WakeConfig(dt_s=dt, free_wake=False, core_model=model, core_radius_m=radius)
                    state = start_lifting_surface(w, cfg, wind, settings=settings(basis), free_wake=True)
                    after = advance_lifting_surface(state, cfg, wind)
                    errors.append(np.max(np.abs(after.solution.lattice_circulation-state.solution.lattice_circulation)))
                    rates.append(np.max(np.abs(after.solution.lattice_gamma_rate)))
                with self.subTest(basis=basis, model=model, radius=radius):
                    self.assertLess(errors[-1], .02*errors[-2])
                    self.assertLess(errors[-1], 1e-7)
                    assert_allclose(rates[-1], rates[-2], rtol=.02, atol=1e-5)

    def test_first_euler_convection_uses_initial_cap_field(self):
        w, wind = wing(), np.array([4., 0., .1])
        for basis in ('constant', 'linear_potential'):
            cfg = WakeConfig(dt_s=.001, free_wake=True, core_model=CoreModel.LAMB_OSEEN,
                core_radius_m=.05, convection_integrator='euler')
            state = start_lifting_surface(w, cfg, wind, settings=settings(basis), free_wake=True)
            nodes, edges, gamma = lifting_surface_network(state)
            cap = np.all(edges//w.node_count == state.settings.chordwise_panels, axis=1)
            radius = np.where(cap, cfg.core_radius_m, 0.)
            exclusions = [(j, k) for j, point in enumerate(w.trailing_edge)
                for k, edge in enumerate(edges) if any(np.array_equal(point, nodes[a]) for a in edge)]
            velocity = wind+segments_velocity(w.trailing_edge, nodes, edges, gamma,
                model=cfg.core_model, core_radius=radius,
                exclusions=np.asarray(exclusions, dtype=int).reshape(-1, 2))
            after = advance_lifting_surface(state, cfg, wind)
            assert_allclose(after.free_rows[0], w.trailing_edge+cfg.dt_s*velocity, atol=2e-14, rtol=2e-14)

    def test_legacy_initial_arrays_match_sealed_startup_records(self):
        w, wind = wing(), np.array([4., 0., .1])
        task = ROOT/'docs/tasks/OF-0006-teremedy20260927'
        for basis, filename in (('constant', 'constant-start-cap-probe.json'),
                                ('linear_potential', 'start-cap-probe.json')):
            records = json.loads((task/filename).read_text(encoding='utf-8'))
            for case in records['cases']:
                cfg = WakeConfig(dt_s=.1, free_wake=False, core_model=CoreModel[case['core_model']],
                    core_radius_m=case['radius_m'])
                state = start_lifting_surface(w, cfg, wind,
                    settings=settings(basis, initial_cap='legacy_zero'), free_wake=True)
                assert_array_equal(state.solution.lattice_circulation, case['initial_gamma_m2_s'])

    def test_unsupported_initial_cap_choices_are_rejected(self):
        with self.assertRaises(ValueError):
            LiftingSurfaceConfig(initial_cap='unrecorded')
        with self.assertRaises(ValueError):
            start_lifting_surface(wing(), WakeConfig(), [4., 0., .1], settings=settings('constant'))
        for basis in ('constant', 'linear_potential'):
            with self.assertRaises(ValueError):
                start_lifting_surface(wing(), WakeConfig(core_model=CoreModel.OFFSET), [4., 0., .1],
                    settings=settings(basis), free_wake=True)

    def test_zero_circulation_and_none_still_require_explicit_archive_contract(self):
        parent = ROOT/'local/initial-cap-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root, success = parent/uuid4().hex, False
        root.mkdir()
        try:
            for basis in ('constant', 'linear_potential'):
                for core in (CoreModel.NONE, CoreModel.LAMB_OSEEN):
                    path = root/f'{basis}-{core.name}'
                    state = start_lifting_surface(wing(), WakeConfig(core_model=core), [4., 0., 0.],
                        settings=settings(basis), free_wake=True)
                    writer = AeroWriter(path, dict(aero_method='vlm_free_wake'))
                    writer.append(state)
                    writer.finish()
                    with closing(AeroReader(path)) as reader:
                        self.assertEqual(reader.manifest['schema'], INITIAL_CAP_LATTICE_SCHEMA)
                        self.assertEqual(len(reader.frame(0)['network_edges']), 0)
                    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
                    manifest.pop('initial_cap_contract')
                    (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        AeroReader(path)
            success = True
        finally:
            if success:
                if root.resolve().parent != parent.resolve() or not parent.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Refusing cleanup outside test workspace')
                shutil.rmtree(root)

    def test_all_v7_combinations_consumers_and_hashed_corruption(self):
        parent = ROOT/'local/initial-cap-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root, success = parent/uuid4().hex, False
        root.mkdir()
        try:
            w = wing()
            angles = np.linspace(-25., 25., 101)
            polar = SectionPolar(angles, 5*np.sin(np.deg2rad(angles)), angles*0+.02, angles*0-.03)
            for basis in ('constant', 'linear_potential'):
                for coupled in (False, True):
                    run = root/f'{basis}-{coupled}'
                    cfg = WakeConfig(dt_s=.025, core_radius_m=.05, core_evolution='age_stretch')
                    selected = settings(basis, chordwise_panels=3,
                        polar_coupling='grid_section' if coupled else 'none')
                    state = start_lifting_surface(w, cfg, [4., 0., .1], settings=selected, polar=polar, free_wake=True)
                    case = dict(aero_method='vlm_free_wake')
                    if coupled:
                        case['coupled_source_polars_sha256'] = polar_identity(encode_polars(polar_tables(polar, w.panels)))
                    writer = AeroWriter(run, case)
                    try:
                        writer.append(state)
                        for _ in range(2):
                            state = advance_lifting_surface(state, cfg, [4., 0., .1], polar=polar)
                            writer.append(state)
                        writer.finish()
                    except BaseException as error:
                        writer.fail(error)
                        raise
                    before = sha256_file(run/'wake.h5')
                    with closing(AeroReader(run)) as reader:
                        self.assertEqual(reader.manifest['schema'], INITIAL_CAP_LATTICE_SCHEMA)
                        frame = reader.frame(2)
                        assert_array_equal(frame['network_core_radius_m'], lifting_surface_edge_radii(state))
                        assert_allclose(frame['panel_force'].sum(axis=0), state.solution.force, atol=2e-12)
                    self.assertIn('birth-core initial cap', inspect_run(run).core)
                    export_paraview(run, root/f'vtk-{basis}-{coupled}')
                    vtk = (root/f'vtk-{basis}-{coupled}'/'wing-000000.vtp').read_text(encoding='utf-8')
                    self.assertIn('initial_cap_0_legacy_1_birth_core', vtk)
                    if coupled and basis == 'linear_potential' and importlib.util.find_spec('PySide6'):
                        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
                        from PySide6 import QtWidgets
                        from openfawe.aero_viewer import AeroReplay
                        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
                        window = AeroReplay(run)
                        try:
                            window.timer.stop()
                            window.show()
                            window.seek(0)
                            app.processEvents()
                            self.assertIn('birth-core continuity', window.info.text())
                            self.assertFalse(window.grab().isNull())
                        finally:
                            window.close()
                    self.assertEqual(before, sha256_file(run/'wake.h5'))
                    for mode in ('downgrade', 'missing-manifest-cap', 'missing-frame-cap', 'wrong-radius'):
                        bad = root/f'bad-{basis}-{coupled}-{mode}'
                        shutil.copytree(run, bad)
                        manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
                        with h5py.File(bad/'wake.h5', 'r+') as file:
                            if mode == 'downgrade':
                                schema = (POTENTIAL_POLAR_LATTICE_SCHEMA if coupled else POTENTIAL_LATTICE_SCHEMA) if basis == 'linear_potential' else (POLAR_LATTICE_SCHEMA if coupled else LATTICE_SCHEMA)
                                manifest['schema'] = schema
                                file.attrs['schema'] = schema
                            elif mode == 'missing-manifest-cap':
                                manifest.pop('initial_cap_contract')
                            elif mode == 'missing-frame-cap':
                                del file['frames']['000000'].attrs['initial_cap_contract']
                            else:
                                file['frames']['000000']['network_core_radius_m'][:] = 0.
                        manifest['wake_sha256'] = sha256_file(bad/'wake.h5')
                        (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                        with self.assertRaises(ValueError, msg=(basis, coupled, mode)):
                            AeroReader(bad)
            success = True
        finally:
            if success:
                if root.resolve().parent != parent.resolve() or not parent.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Refusing cleanup outside test workspace')
                shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
