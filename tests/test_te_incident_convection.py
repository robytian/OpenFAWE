"""Independent endpoint/filament checks for the optional convection-only core."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.aircraft import WingAssembly
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, lifting_surface_network, lifting_surface_edge_radii,
    _linear_convection_network, _endpoint_exclusions)
from openfawe.aero.vortex import CoreModel, segment_velocity, segments_velocity
from openfawe.aero.wake import WakeConfig, pack_rings
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.initial_rate import InitialGammaRate
from openfawe.aero_archive import AeroWriter, AeroReader
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file
from openfawe.run_catalog import inspect_run


def rectangle():
    le = np.array([[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]])
    return Wing(le, le+[1., 0., 0.])


def settings(**kw):
    return LiftingSurfaceConfig(3, wake_basis='linear_potential', wake_release='current',
        initial_cap='wake_core', bound_convection_core='te_incident_birth', **kw)


def stage_velocity(state, points, fraction, wind):
    network, radii, canonical = _linear_convection_network(state, state.config, points, fraction)
    exclusions = _endpoint_exclusions(points, network[1], canonical)
    return np.asarray(wind)+segments_velocity(points, *network, model=state.config.core_model,
        core_radius=radii, exclusions=exclusions)


class IncidentConvectionTests(unittest.TestCase):
    def test_single_segment_analytic_endpoint_and_regularized_limit(self):
        gamma, theta = 1.2, .6
        ray = np.array([np.cos(theta), 0., np.sin(theta)])
        expected = gamma*np.tan(theta/2)/(4*np.pi)
        ideal, regular = [], []
        for distance in (1e-5, 1e-6, 1e-7):
            point = np.array([1., 0., 0.])+distance*ray
            ideal.append(np.linalg.norm(segment_velocity([0., 0., 0.], [1., 0., 0.],
                point, gamma))*distance)
            regular.append(np.linalg.norm(segment_velocity([0., 0., 0.], [1., 0., 0.],
                point, gamma, model=CoreModel.LAMB_OSEEN, core_radius=.1)))
        assert_allclose(ideal, expected, rtol=2e-9)
        assert_allclose(np.array(regular[:-1])/regular[1:], 10., rtol=2e-8)

    def test_policy_domain_is_explicit(self):
        for kw in ({'bound_convection_core':'unknown'}, {'bound_convection_core':True},
                   {'bound_convection_core':'te_incident_birth'}):
            with self.assertRaises(ValueError): LiftingSurfaceConfig(**kw)
        for cfg, transient in ((WakeConfig(core_model=CoreModel.NONE), True),
                               (WakeConfig(core_model=CoreModel.OFFSET), True),
                               (WakeConfig(free_wake=False), True), (WakeConfig(), False)):
            with self.assertRaises(ValueError):
                start_lifting_surface(rectangle(), cfg, [1., 0., .1],
                    settings=settings(), free_wake=transient)
        for model in (CoreModel.RANKINE, CoreModel.LAMB_OSEEN, CoreModel.VATISTAS):
            start_lifting_surface(rectangle(), WakeConfig(core_model=model), [1., 0., .1],
                settings=settings(), free_wake=True)

    def test_stage_field_against_independent_constant_ring_graph(self):
        wing, cfg = rectangle(), WakeConfig(dt_s=.02, core_radius_m=.1)
        state = start_lifting_surface(wing, cfg, [1., 0., .1], settings=settings(), free_wake=True)
        moving = (wing.trailing_edge+[.03, .005, .002])[None]
        network, radii, canonical = _linear_convection_network(state, cfg, moving, .5)
        ordinary = pack_rings(np.concatenate((state.mesh.vortex_rows, moving)),
            np.vstack((state.solution.lattice_circulation, state.solution.circulation)))
        expected = np.zeros(len(ordinary[1]))
        for k, (first, last) in enumerate(ordinary[1]):
            a, b = first//wing.node_count, last//wing.node_count
            if min(a, b) >= 3 or (a != b and max(a, b) == 3): expected[k] = .1
        points = np.array([[.5, .1, .25], [1.01, -1.95, .01], [2., .8, .3]])
        assert_allclose(segments_velocity(points, *network, model=cfg.core_model, core_radius=radii),
            segments_velocity(points, *ordinary, model=cfg.core_model, core_radius=expected),
            rtol=2e-12, atol=2e-13)
        width, nc = wing.node_count, state.settings.chordwise_panels
        # TE span edges cancel exactly. Every chordwise filament continues into
        # the newborn stream edge with the same oriented strength, without fitting.
        edges, gamma = ordinary[1:]
        self.assertFalse(np.any(np.all(edges//width == nc, axis=1)))
        lookup = dict(zip(map(tuple, edges), gamma))
        for node in range(width):
            assert_array_equal(lookup[((nc-1)*width+node, nc*width+node)],
                               lookup[(nc*width+node, (nc+1)*width+node)])
        balance = np.zeros(len(ordinary[0]))
        np.add.at(balance, edges[:, 0], gamma); np.add.at(balance, edges[:, 1], -gamma)
        assert_allclose(balance, 0., atol=2e-15)

    def test_every_te_right_limit_duplicate_roots_and_age_policies(self):
        base = rectangle()
        duplicate = WingAssembly((Wing(base.leading_edge[:2], base.trailing_edge[:2]),
            Wing(base.leading_edge[1:], base.trailing_edge[1:])), ('left', 'right'))
        for wing in (base, duplicate):
            for policy in ('constant', 'age', 'age_stretch'):
                cfg = WakeConfig(dt_s=.01, core_radius_m=.1, core_evolution=policy,
                    convection_integrator='rk4')
                state = start_lifting_surface(wing, cfg, [1., 0., .1],
                    settings=settings(), free_wake=True)
                for _ in range(2): state = advance_lifting_surface(state, cfg, [1., 0., .1])
                legacy = replace(state, settings=replace(state.settings, bound_convection_core='legacy_zero'))
                moving = np.concatenate((state.mesh.vortex_rows[-1:], state.free_rows)).reshape(-1, 3)
                first = stage_velocity(state, moving, 0., [1., 0., .1])
                errors = []
                for epsilon in (1e-5, 1e-6, 1e-7):
                    displaced = moving+epsilon*first
                    actual = stage_velocity(state, displaced, epsilon/cfg.dt_s, [1., 0., .1])
                    errors.append(np.linalg.norm(actual[:wing.node_count]-first[:wing.node_count], axis=1))
                    new = _linear_convection_network(state, cfg, displaced, epsilon/cfg.dt_s)
                    old = _linear_convection_network(legacy, cfg, displaced, epsilon/cfg.dt_s)
                    for a, b in zip(new[0], old[0]): assert_array_equal(a, b)
                    assert_array_equal(new[2], old[2])
                    assert_array_equal(_endpoint_exclusions(displaced, new[0][1], new[2]),
                                       _endpoint_exclusions(displaced, old[0][1], old[2]))
                    changed = new[1] != old[1]
                    self.assertTrue(changed.any())
                    expected = []
                    for k, (a, b) in enumerate(old[0][1]):
                        if old[1][k] != 0 or max(a, b) >= 4*wing.node_count: continue
                        if any(np.array_equal(old[0][0][index], te)
                               for index in (a, b) for te in state.mesh.vortex_rows[-1]):
                            expected.append(k)
                    assert_array_equal(np.flatnonzero(changed), expected)
                    assert_array_equal(new[1][changed], .1)
                    self.assertTrue(np.all(new[0][1][changed] < 4*wing.node_count))
                    assert_array_equal(new[1][~changed], old[1][~changed])
                self.assertLess(max(errors[-1]), 1e-5)
                assert_allclose(errors[0]/errors[1], 10., rtol=.01)
                assert_allclose(errors[1]/errors[2], 10., rtol=.01)
                for a, b in zip(lifting_surface_network(state), lifting_surface_network(legacy)):
                    assert_array_equal(a, b)
                assert_array_equal(lifting_surface_edge_radii(state), lifting_surface_edge_radii(legacy))
                if state.cores is not None:
                    self.assertGreater(state.cores.stream[0].min(), .1)

    def test_option_preserves_initial_boundary_and_loads_and_covariance(self):
        wing, boost = rectangle(), np.array([.4, -.2, .15])
        for integrator in ('euler', 'heun', 'rk4'):
            cfg = WakeConfig(dt_s=.02, core_radius_m=.1, convection_integrator=integrator,
                core_evolution='age_stretch', geometry_substeps=2)
            options = settings(chordwise_spacing='cosine', wake_quadrature='te_graded')
            stationary = start_lifting_surface(wing, cfg, [1., 0., .1], settings=options, free_wake=True)
            legacy = start_lifting_surface(wing, cfg, [1., 0., .1],
                settings=replace(options, bound_convection_core='legacy_zero'), free_wake=True)
            assert_array_equal(stationary.solution.lattice_circulation, legacy.solution.lattice_circulation)
            assert_array_equal(stationary.solution.force, legacy.solution.force)
            translated = start_lifting_surface(wing, cfg, np.array([1., 0., .1])+boost,
                structural_velocity=boost, settings=options, free_wake=True)
            for k in range(1, 4):
                stationary = advance_lifting_surface(stationary, cfg, [1., 0., .1])
                shift = k*cfg.dt_s*boost
                translated = advance_lifting_surface(translated, cfg, np.array([1., 0., .1])+boost,
                    structural_velocity=boost, next_wing=Wing(wing.leading_edge+shift, wing.trailing_edge+shift))
                assert_allclose(translated.free_rows, stationary.free_rows+shift, rtol=2e-12, atol=2e-12)
                assert_allclose(translated.solution.force, stationary.solution.force, rtol=2e-11, atol=2e-11)

    def test_microscopic_tip_is_stiff_with_a_finite_right_limit(self):
        wing = Wing.ellipse(8)
        cfg = WakeConfig(dt_s=.05, core_radius_m=.15)
        options = replace(settings(chordwise_spacing='cosine', wake_quadrature='te_graded'),
                          chordwise_panels=8)
        state = start_lifting_surface(wing, cfg, [1., 0., .1], settings=options, free_wake=True)
        moving = state.mesh.vortex_rows[-1].copy()
        first = stage_velocity(state, moving, 0., [1., 0., .1])
        errors = []
        for epsilon in (1e-12, 1e-13, 1e-14):
            value = stage_velocity(state, moving+epsilon*first, epsilon/cfg.dt_s, [1., 0., .1])
            errors.append(np.linalg.norm((value-first)[[0, -1]], axis=1).max())
        assert_allclose(np.array(errors[:-1])/errors[1:], 10., rtol=.01)
        # This limit test deliberately does not qualify practical ODE time steps.

    def test_curved_te_legacy_cap_field_against_direct_closed_rings(self):
        le = np.array([[0., -.1, .02], [0., 0., 0.], [0., .1, .02]])
        wing, cfg = Wing(le, le+[1., 0., 0.]), WakeConfig(core_radius_m=.1)
        chosen = replace(settings(), initial_cap='legacy_zero')
        state = start_lifting_surface(wing, cfg, [1., 0., .1], settings=chosen, free_wake=True)
        points = state.mesh.vortex_rows[-1].copy()
        ordinary = pack_rings(state.mesh.vortex_rows, state.solution.lattice_circulation)
        expected = np.broadcast_to([1., 0., .1], points.shape).copy()
        for k, (a, b) in enumerate(ordinary[1]):
            ra, rb = a//wing.node_count, b//wing.node_count
            radius = .1 if max(ra, rb) == 3 else 0.
            for j, point in enumerate(points):
                if any(np.array_equal(point, ordinary[0][i]) for i in (a, b)): continue
                expected[j] += segment_velocity(ordinary[0][a], ordinary[0][b], point,
                    ordinary[2][k], model=cfg.core_model, core_radius=radius)
        assert_allclose(stage_velocity(state, points, 0., [1., 0., .1]), expected,
                        rtol=2e-12, atol=2e-13)
        legacy = replace(state, settings=replace(chosen, bound_convection_core='legacy_zero'))
        self.assertGreater(np.linalg.norm(expected-stage_velocity(legacy, points, 0., [1., 0., .1])), 1e-9)

    def test_archive_binding_replay_and_consumer_labels(self):
        parent = Path(__file__).resolve().parents[1]/'local/incident-core-tests'
        parent.mkdir(parents=True, exist_ok=True); root = parent/uuid4().hex; root.mkdir()
        success = False
        try:
            cfg = WakeConfig(dt_s=.02, core_radius_m=.1, core_evolution='age_stretch',
                convection_integrator='rk4')
            state = start_lifting_surface(rectangle(), cfg, [1., 0., .1], settings=settings(), free_wake=True)
            writer = AeroWriter(root/'run', {'aero_method':'vlm_free_wake'})
            try:
                writer.append(state)
                for _ in range(3):
                    state = advance_lifting_surface(state, cfg, [1., 0., .1]); writer.append(state)
                writer.finish()
            except BaseException as error:
                writer.fail(error); raise
            reader = AeroReader(root/'run')
            try:
                frame = reader.frame(3)
                self.assertEqual(frame['linear_wake_convection_contract'],
                    'attached_frozen_nodes_te_incident_birth_v2')
                assert_array_equal(frame['free_rows'], state.free_rows)
                assert_array_equal(frame['network_core_radius_m'], lifting_surface_edge_radii(state))
            finally:
                reader.close()
            self.assertIn('TE-incident birth core in convection', inspect_run(root/'run').core)
            export_paraview(root/'run', root/'vtk')
            tree = ET.parse(root/'vtk/wake-000003.vtp')
            self.assertEqual(float(tree.find(".//FieldData/DataArray[@Name='convection_bound_core_radius_m']").text), .1)
            for name, value in (('bound_convection_core','legacy_zero'),
                                ('convection_bound_core_radius_m',.2),
                                ('linear_wake_convection_contract','attached_frozen_nodes_v1'),
                                ('bound_core_model','ideal')):
                bad = root/('bad-manifest-'+name); shutil.copytree(root/'run', bad)
                manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
                manifest[name] = value
                (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError): AeroReader(bad)
                bad = root/('bad-frame-'+name); shutil.copytree(root/'run', bad)
                with h5py.File(bad/'wake.h5','r+') as f: f['frames']['000003'].attrs[name] = value
                manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
                manifest['wake_sha256'] = sha256_file(bad/'wake.h5')
                (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'convection core|convection requires'):
                    AeroReader(bad)
            bad = root/'quarantined'; shutil.copytree(root/'run', bad)
            case = json.loads((bad/'case.json').read_text(encoding='utf-8'))
            case['counterfactual'] = {'scope':'Unregistered frozen-field diagnostic'}
            (bad/'case.json').write_text(json.dumps(case), encoding='utf-8')
            manifest = json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
            manifest['case_sha256'] = sha256_file(bad/'case.json')
            (bad/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'quarantined counterfactual'): AeroReader(bad)
            record = inspect_run(bad)
            self.assertEqual(record.status, 'quarantined / diagnostic')
            self.assertFalse(record.ready)
            success = True
        finally:
            if success:
                if root.resolve().parent != parent.resolve(): raise ValueError('Unsafe test cleanup')
                shutil.rmtree(root)

    def test_archive_schema_combinations_and_source_polar_coupling(self):
        parent = Path(__file__).resolve().parents[1]/'local/incident-core-tests'
        parent.mkdir(parents=True, exist_ok=True); root = parent/uuid4().hex; root.mkdir()
        success = False
        try:
            cfg = WakeConfig(dt_s=.01, core_radius_m=.1, convection_integrator='heun')
            for cap, rate in (('legacy_zero', False), ('wake_core', False), ('wake_core', True)):
                for coupled in (False, True):
                    chosen = replace(settings(), initial_cap=cap,
                        gamma_derivative='bdf2_consistent' if rate else 'bdf2',
                        polar_coupling='grid_section' if coupled else 'none')
                    polar = SectionPolar.linear(5.) if coupled else None
                    supplied = (InitialGammaRate(np.zeros((3, 2)), 0., 'declared_kinematic',
                        json.dumps({'source':'fixed wing with zero initial normal forcing'})) if rate else None)
                    state = start_lifting_surface(rectangle(), cfg, [1., 0., 0.],
                        settings=chosen, free_wake=True, polar=polar, initial_gamma_rate=supplied)
                    path = root/f'{cap}-{rate}-{coupled}'
                    writer = AeroWriter(path, {'aero_method':'vlm_free_wake'})
                    try:
                        writer.append(state)
                        state = advance_lifting_surface(state, cfg, [1., 0., .01], polar=polar)
                        writer.append(state); writer.finish()
                    except BaseException as error:
                        writer.fail(error); raise
                    reader = AeroReader(path)
                    try:
                        frame = reader.frame(1)
                        assert_array_equal(frame['panel_force'].sum(axis=0), state.solution.force)
                        self.assertEqual(frame['bound_convection_core'], 'te_incident_birth')
                        self.assertEqual(frame['section_polar_used'], coupled)
                        expected = 'v8' if rate else 'v7' if cap == 'wake_core' else 'v6' if coupled else 'v5'
                        self.assertTrue(reader.manifest['schema'].endswith(expected))
                    finally: reader.close()
            success = True
        finally:
            if success:
                if root.resolve().parent != parent.resolve(): raise ValueError('Unsafe test cleanup')
                shutil.rmtree(root)


if __name__ == '__main__': unittest.main()
