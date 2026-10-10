"""Independent regularized partial-ring and physical core history checks."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, lifting_surface_induced_velocity, lifting_surface_edge_radii,
    lifting_surface_network)
from openfawe.aero.vortex import CoreModel, ring_velocity, segments_velocity
from openfawe.aero.wake import WakeConfig, pack_rings
from openfawe.aero.wake_potential import refine_potential, potential_edge_radii
from openfawe.aero.wing import Wing
from openfawe.aero_archive import AeroWriter, AeroReader
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file


def rectangle():
    le = np.array([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]])
    return Wing(le, le+[1., 0., 0.])


def settings():
    return LiftingSurfaceConfig(3, wake_basis='linear_potential', wake_release='current',
        gamma_derivative='bdf2', wake_quadrature='te_graded')


class LinearWakeCoreTests(unittest.TestCase):
    def test_regularized_field_against_independent_partial_ring_integral(self):
        wing = Wing([[0., -1., 0.], [0., 1., 0.]], [[1., -1., 0.], [1., 1., 0.]])
        rows = np.array([wing.trailing_edge, wing.trailing_edge+[1., 0., 0.]])
        phi = np.array([[.7], [-.2]])
        points = np.array([[.1, .2, .5], [.6, -.5, -.3], [3., .5, .7]])
        cores = SimpleNamespace(span=np.array([[.12]]), stream=np.array([[.9, 1.1]]))
        mesh = SimpleNamespace(vortex_rows=rows[:1])
        a, b = rows[0]; c, d = rows[1]
        q, weights = leggauss(96); q, weights = (q+1)/2, weights/2
        for model in (CoreModel.RANKINE, CoreModel.LAMB_OSEEN, CoreModel.VATISTAS):
            def partial(s):
                # Four explicit closed edges, independently prescribed cores.
                radius = np.array([0., 1.1, np.sqrt(.08**2+s*(.12**2-.08**2)), .9])
                return segments_velocity(points, [a, b, b+s*(d-b), a+s*(c-a)],
                    np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), np.ones(4),
                    model=model, core_radius=radius)
            reference = phi[1, 0]*partial(1.)-(phi[1, 0]-phi[0, 0])*sum(
                weight*partial(s) for s, weight in zip(q, weights))
            errors = []
            for order in (4, 8):
                vertices, strengths = refine_potential(rows, phi, order)
                network = pack_rings(vertices, strengths)
                radii = potential_edge_radii(wing, mesh, rows[1:], network[1], cores,
                    .08, order, 'gauss', include_bound=False)
                errors.append(np.max(np.abs(segments_velocity(points, *network,
                    model=model, core_radius=radii)-reference)))
            self.assertLess(errors[1], 2e-7)
            self.assertLess(errors[1], errors[0]/20)
            constant_phi = np.ones((2, 1))*.7
            vertices, strengths = refine_potential(rows, constant_phi, 8)
            network = pack_rings(vertices, strengths)
            radii = potential_edge_radii(wing, mesh, rows[1:], network[1], cores,
                .08, 8, 'gauss', include_bound=False)
            assert_allclose(segments_velocity(points, *network, model=model, core_radius=radii),
                            .7*partial(1.), rtol=1e-12, atol=1e-10)

    def test_positive_cores_keep_material_age_and_boundary_residual(self):
        initial, dt, nu = .07, .02, 1.4639e-5
        rate = 4*1.25643*nu
        for policy in ('constant', 'age', 'age_stretch'):
            cfg = WakeConfig(dt_s=dt, core_radius_m=initial, core_model=CoreModel.LAMB_OSEEN,
                free_wake=False, core_evolution=policy, kinematic_viscosity_m2_s=nu)
            state = start_lifting_surface(rectangle(), cfg, [4., 0., .1],
                                         settings=settings(), free_wake=True)
            for k in range(1, 5):
                old = state
                state = advance_lifting_surface(old, cfg, [4., 0., .1])
                repeated = advance_lifting_surface(old, cfg, [4., 0., .1])
                assert_array_equal(state.solution.force, repeated.solution.force)
                assert_array_equal(state.wake_node_potential_m2_s[1:], old.wake_node_potential_m2_s)
                velocity = np.array([4., 0., .1])+lifting_surface_induced_velocity(
                    state, state.mesh.collocation_points)
                assert_allclose(np.sum(velocity*state.mesh.normals, axis=1), 0., atol=2e-12)
                if policy != 'constant':
                    ages = dt*np.arange(1, k+1)
                    assert_allclose(state.cores.span**2,
                        np.broadcast_to((initial**2+rate*ages)[:, None], state.cores.span.shape),
                        rtol=1e-12, atol=1e-14)
                    assert_allclose(state.cores.stream**2,
                        np.broadcast_to((initial**2+rate*(ages-dt/2))[:, None], state.cores.stream.shape),
                        rtol=1e-12, atol=1e-14)
                radii = lifting_surface_edge_radii(state)
                nodes, edges, gamma = lifting_surface_network(state)
                self.assertGreater(radii.max(), 0.)
                balance = np.zeros(len(nodes))
                np.add.at(balance, edges[:, 0], gamma); np.add.at(balance, edges[:, 1], -gamma)
                assert_allclose(balance, 0., atol=2e-15)
        with self.assertRaisesRegex(ValueError, 'OFFSET|offset'):
            start_lifting_surface(rectangle(), replace(cfg, core_model=CoreModel.OFFSET,
                core_evolution='constant'), [4., 0., .1], settings=settings(), free_wake=True)

    def test_physical_stretch_conserves_inviscid_q_times_length(self):
        cfg = WakeConfig(dt_s=.02, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
            free_wake=False, core_evolution='age_stretch', eddy_viscosity_factor=0.)
        wind = lambda p, t: np.column_stack((np.full(len(p), 4.), .1*p[:, 1], np.full(len(p), .1)))
        state = start_lifting_surface(rectangle(), cfg, wind, settings=settings(), free_wake=True)
        for k in range(4):
            old = state
            state = advance_lifting_surface(old, cfg, wind)
            a, b = old.wing.panel_edges.T
            old_rows = np.concatenate((old.wing.trailing_edge[None], old.free_rows))
            old_length = np.linalg.norm(old_rows[:, b]-old_rows[:, a], axis=2)
            new_length = np.linalg.norm(state.free_rows[:, b]-state.free_rows[:, a], axis=2)
            old_q = np.vstack((np.full((1, old.wing.panels), cfg.core_radius_m**2), old.cores.span**2))
            assert_allclose(state.cores.span**2*new_length, old_q*old_length, rtol=1e-12, atol=1e-14)
            self.assertLess(state.cores.span[0].max(), cfg.core_radius_m)

    def test_archive_vtk_and_self_consistent_forged_core_history(self):
        parent = Path(__file__).resolve().parents[1]/'local/linear-core-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex; root.mkdir()
        try:
            cfg = WakeConfig(dt_s=.02, core_radius_m=.07, core_model=CoreModel.LAMB_OSEEN,
                free_wake=False, core_evolution='age_stretch')
            state = start_lifting_surface(rectangle(), cfg, [4., 0., .1],
                                         settings=settings(), free_wake=True)
            writer = AeroWriter(root/'run', {'aero_method':'vlm_free_wake'})
            try:
                writer.append(state)
                for k in range(3):
                    state = advance_lifting_surface(state, cfg, [4., 0., .1])
                    writer.append(state)
                writer.finish()
            except BaseException as error:
                writer.fail(error); raise
            reader = AeroReader(root/'run')
            try:
                frame = reader.frame(3)
                self.assertEqual(reader.manifest['wake_core_mapping'], 'material_edge_q_v1')
                assert_array_equal(frame['network_core_radius_m'], lifting_surface_edge_radii(state))
                assert_array_equal(frame['span_core_radius_m'], state.cores.span)
            finally:
                reader.close()
            export_paraview(root/'run', root/'vtk')
            tree = ET.parse(root/'vtk/wake-000003.vtp')
            value = tree.find('.//CellData/DataArray[@Name="core_radius_m"]')
            assert_array_equal(np.fromstring(value.text, sep=' '), frame['network_core_radius_m'])
            bad = root/'bad-history'; shutil.copytree(root/'run', bad)
            with h5py.File(bad/'wake.h5', 'r+') as file:
                group = file['frames/000003']; span = group['span_core_radius_m'][:]
                span[-1] *= 1.1; group['span_core_radius_m'][:] = span
                forged = replace(state.cores, span=span)
                network = lifting_surface_network(state)
                group['network_core_radius_m'][:] = potential_edge_radii(state.wing, state.mesh,
                    state.free_rows, network[1], forged, cfg.core_radius_m,
                    state.settings.wake_quadrature_order, state.settings.wake_quadrature)
            manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
            manifest['wake_sha256'] = sha256_file(bad/'wake.h5')
            (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'adjacent evolution'):
                AeroReader(bad)
        finally:
            if root.resolve().parent != parent.resolve():
                raise ValueError('Refusing cleanup outside test workspace')
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
