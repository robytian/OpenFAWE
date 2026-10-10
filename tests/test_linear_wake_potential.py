"""Independent quadrature, production history and corrupt-archive contracts."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.aircraft import WingAssembly
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, lifting_surface_network, lifting_surface_induced_velocity,
    lifting_surface_edge_radii, build_lifting_surface_mesh, _convect)
from openfawe.aero.methods import AeroBackend
from openfawe.aero.vortex import CoreModel, ring_velocity, segments_velocity, FilamentSingularity
from openfawe.aero.wake import WakeConfig, pack_rings
from openfawe.aero.wake_potential import (refine_potential, potential_network,
    near_junction_distance, quadrature_rule)
from openfawe.aero.wing import Wing
from openfawe.aero_archive import AeroWriter, AeroReader, POTENTIAL_LATTICE_SCHEMA
from openfawe.aero_vtk import export_paraview, frame_network
from openfawe.provenance import sha256_file
from openfawe.run_catalog import inspect_run


def rectangle():
    le = np.array([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]])
    return Wing(le, le+[1., 0., 0.])


def config(**kwargs):
    return WakeConfig(dt_s=.02, free_wake=False, core_model=CoreModel.NONE, **kwargs)


def settings(**kwargs):
    return LiftingSurfaceConfig(3, wake_basis='linear_potential', wake_release='current',
                               gamma_derivative='bdf2', **kwargs)


class LinearWakePotentialTests(unittest.TestCase):
    def test_quadrature_against_independent_partial_ring_integral(self):
        # Independent integration by parts: phi_back*R(1) minus
        # (phi_back-phi_front)*integral R(s) ds. No production refine/pack
        # operation constructs the reference, and every partial ring is closed.
        rows = np.array([[[1., -1., 0.], [1., 1., .1]],
                         [[2., -1.1, .3], [2.2, 1.1, .2]]])
        phi = np.array([[.7], [-.2]])
        points = np.array([[.1, .2, .5], [.6, -.5, -.3], [3., .5, .7]])
        a, b = rows[0]; c, d = rows[1]
        full = ring_velocity([a, b, d, c], points, 1.)
        q, w = leggauss(96); q, w = (q+1)/2, w/2
        integral = sum(weight*ring_velocity([a, b, b+s*(d-b), a+s*(c-a)], points, 1.)
                       for s, weight in zip(q, w))
        reference = phi[1, 0]*full-(phi[1, 0]-phi[0, 0])*integral
        errors = []
        for order in (4, 8):
            vertices, values = refine_potential(rows, phi, order)
            network = pack_rings(vertices, values)
            actual = segments_velocity(points, *network)
            errors.append(np.max(np.abs(actual-reference)))
        self.assertLess(errors[1], 2e-7)
        self.assertLess(errors[1], errors[0]/20)

    def test_constant_potential_exact_field_and_disconnected_flux(self):
        w = WingAssembly((rectangle(), Wing([[2., -1., 1.], [2., 1., 1.]],
                                            [[2.5, -1., 1.], [2.5, 1., 1.]])), ('main', 'tail'))
        s = settings(wake_quadrature='te_graded')
        mesh = build_lifting_surface_mesh(w, s)
        gamma = np.arange(1., mesh.panels+1).reshape(3, w.panels)/10
        free = np.array([w.trailing_edge+[.3, 0., 0.], w.trailing_edge+[.6, 0., 0.]])
        phi = np.tile(gamma[-1], (3, 1))
        points = [[-.2, .6, .7], [3., 2., 2.]]
        network = potential_network(w, mesh, gamma, free, phi, 8, 'te_graded')
        original = pack_rings(np.concatenate((mesh.vortex_rows, free)),
                              np.vstack((gamma, phi[:-1])), w.panel_edges)
        assert_allclose(segments_velocity(points, *network),
                        segments_velocity(points, *original), rtol=2e-14, atol=2e-15)
        nodes, edges, strengths = network
        balance = np.zeros(len(nodes))
        np.add.at(balance, edges[:, 0], strengths); np.add.at(balance, edges[:, 1], -strengths)
        assert_allclose(balance, 0., atol=2e-15)
        self.assertGreater(near_junction_distance(w, mesh), 0.)

    def test_public_guards_and_immutable_state(self):
        for kwargs in ({'wake_basis':'bad'}, {'wake_quadrature_order':True},
                       {'wake_quadrature_order':3}, {'wake_quadrature':'bad'},
                       {'wake_basis':'linear_potential'}, {'wake_quadrature_order':8}):
            with self.assertRaises(ValueError):
                LiftingSurfaceConfig(**kwargs)
        s = settings()
        for cfg in (replace(config(), core_model=CoreModel.OFFSET),):
            with self.assertRaisesRegex(ValueError, 'linear_potential'):
                start_lifting_surface(rectangle(), cfg, [4., 0., .1], settings=s, free_wake=True)
        with self.assertRaisesRegex(ValueError, 'linear_potential'):
            AeroBackend('vlm', wake=config(), lifting_surface=s)
        with self.assertRaises(ValueError):
            start_lifting_surface(rectangle(), config(), [4., 0., .1],
                settings=settings(polar_coupling='grid_section'), free_wake=True)
        state = start_lifting_surface(rectangle(), config(), [4., 0., .1], settings=s, free_wake=True)
        with self.assertRaises(ValueError):
            replace(state, wake_node_potential_m2_s=None)
        with self.assertRaises(ValueError):
            state.wake_node_potential_m2_s.setflags(write=True)
        with self.assertRaises(ValueError):
            refine_potential(np.zeros((2, 3, 3)), np.zeros((2, 2)), 4)
        with self.assertRaises(ValueError):
            quadrature_rule(4, 0.)

    def test_actual_boundary_bdf2_and_material_history(self):
        cfg = config()
        for quadrature in ('gauss', 'te_graded'):
            state = start_lifting_surface(rectangle(), cfg, [4., 0., .1],
                settings=settings(chordwise_spacing='cosine', wake_quadrature=quadrature), free_wake=True)
            two_back = None
            for k in range(1, 5):
                old = state
                wind = np.array([4., .04*k, .1+.03*k])
                state = advance_lifting_surface(old, cfg, wind)
                repeat = advance_lifting_surface(old, cfg, wind)
                assert_array_equal(state.solution.force, repeat.solution.force)
                assert_array_equal(state.wake_node_potential_m2_s[1:], old.wake_node_potential_m2_s)
                assert_array_equal(state.wake_node_potential_m2_s[0], state.solution.circulation)
                assert_array_equal(state.wake_circulation,
                    .5*(state.wake_node_potential_m2_s[:-1]+state.wake_node_potential_m2_s[1:]))
                velocity = wind+lifting_surface_induced_velocity(state, state.mesh.collocation_points)
                assert_allclose(np.sum(velocity*state.mesh.normals, axis=1), 0., atol=2e-12)
                expected = ((state.solution.lattice_circulation-old.solution.lattice_circulation)/cfg.dt_s
                    if k == 1 else (3*state.solution.lattice_circulation-4*old.solution.lattice_circulation+
                                     two_back)/(2*cfg.dt_s))
                assert_array_equal(state.solution.lattice_gamma_rate, expected)
                assert_array_equal(state.solution.trailing_edge_jump_m2_s, np.zeros(2))
                assert_array_equal(lifting_surface_edge_radii(state),
                                   np.zeros(len(lifting_surface_network(state)[1])))
                two_back = old.solution.lattice_circulation
            with self.assertRaises(ValueError):
                replace(state, wake_circulation=state.wake_circulation+.01)
            nodes, edges, _ = lifting_surface_network(state)
            with self.assertRaises(FilamentSingularity):
                lifting_surface_induced_velocity(state, nodes[edges[:1]].mean(axis=1))

    def test_physical_node_convection_has_correct_stage_times(self):
        for integrator in ('euler', 'heun', 'rk4'):
            cfg = config(convection_integrator=integrator, geometry_substeps=2)
            state = start_lifting_surface(rectangle(), cfg, [4., 0., .1],
                settings=settings(), free_wake=True, time_s=.3)
            field = lambda p, t: np.broadcast_to([1+t*t, 0., 0.], p.shape)
            actual = _convect(state, cfg, field)
            h = cfg.dt_s/2; expected = 0.
            for substep in range(2):
                t = state.time_s+substep*h
                expected += (h*(1+t*t) if integrator == 'euler' else
                    h*(1+(t*t+(t+h)**2)/2) if integrator == 'heun' else
                    h+((t+h)**3-t**3)/3)
            assert_allclose(actual, state.wing.trailing_edge[None]+[expected, 0., 0.], atol=2e-15, rtol=0)

    def test_archive_history_network_catalog_and_vtk(self):
        parent = Path(__file__).resolve().parents[1]/'local/linear-wake-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex; root.mkdir()
        try:
            cfg, s = config(), settings(wake_quadrature='te_graded')
            backend = AeroBackend('vlm_free_wake', wake=cfg, lifting_surface=s)
            state = backend.start(rectangle(), None, [4., 0., .1])
            writer = AeroWriter(root/'run', {'aero_method':'vlm_free_wake', 'aerodynamics':backend.metadata()})
            try:
                writer.append(state)
                for k in range(3):
                    state = backend.advance(state, None, [4., 0., .15+.03*k])
                    writer.append(state)
                writer.finish()
            except BaseException as error:
                writer.fail(error); raise
            reader = AeroReader(root/'run')
            try:
                self.assertEqual(reader.manifest['schema'], POTENTIAL_LATTICE_SCHEMA)
                frame = reader.frame(3)
                self.assertEqual(frame['lattice_time_status'], 'explicit_history_checked_v2')
                assert_array_equal(frame['wake_node_potential_m2_s'], state.wake_node_potential_m2_s)
                for actual, expected in zip(frame_network(frame), lifting_surface_network(state)):
                    assert_array_equal(actual, expected)
            finally:
                reader.close()
            self.assertIn('nodal linear potential', inspect_run(root/'run').core)
            export_paraview(root/'run', root/'vtk')
            tree = ET.parse(root/'vtk/wake-000003.vtp')
            marker = tree.find('.//FieldData/DataArray[@Name="wake_basis_0_constant_1_linear_potential"]')
            self.assertEqual(float(marker.text), 1.)
            values = tree.find('.//CellData/DataArray[@Name="circulation_m2_s"]')
            assert_array_equal(np.fromstring(values.text, sep=' '), frame['network_circulation_m2_s'])
            for field in ('wake_node_potential_m2_s', 'network_core_radius_m', 'lattice_time_contract'):
                bad = root/('bad-'+field); shutil.copytree(root/'run', bad)
                with h5py.File(bad/'wake.h5', 'r+') as file:
                    group = file['frames/000003']
                    if field in group:
                        group[field][0] += .1
                    else:
                        group.attrs[field] = 'incorrect'
                manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
                manifest['wake_sha256'] = sha256_file(bad/'wake.h5')
                (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    AeroReader(bad)
            # A self-consistent forged graph must still fail adjacent material
            # history checks; an outer hash is not evidence of causal validity.
            bad = root/'bad-causal'; shutil.copytree(root/'run', bad)
            with h5py.File(bad/'wake.h5', 'r+') as file:
                group = file['frames/000003']; phi = group['wake_node_potential_m2_s'][:]
                phi[-1] += .1
                group['wake_node_potential_m2_s'][:] = phi
                group['wake_circulation_m2_s'][:] = .5*(phi[:-1]+phi[1:])
                network = potential_network(state.wing, state.mesh, state.solution.lattice_circulation,
                    state.free_rows, phi, s.wake_quadrature_order, s.wake_quadrature)
                for name, value in zip(('network_nodes_m', 'network_edges', 'network_circulation_m2_s'), network):
                    del group[name]; group.create_dataset(name, data=value)
                del group['network_core_radius_m']; group.create_dataset('network_core_radius_m', data=np.zeros(len(network[1])))
            manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
            manifest['wake_sha256'] = sha256_file(bad/'wake.h5')
            (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nodal potential history'):
                AeroReader(bad)
        finally:
            if root.resolve().parent != parent.resolve():
                raise ValueError('Refusing cleanup outside test workspace')
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
