"""Production routing and transactional failure of the static surface guard."""
from dataclasses import replace
import csv
import importlib.util
import json
from pathlib import Path
import shutil
from uuid import uuid4
import unittest
from unittest.mock import patch

import numpy as np
import h5py

from openfawe.aero.geometry_guard import StaticSurfaceGuard, WakeGeometryViolation
from openfawe.aero.methods import AeroBackend
from openfawe.aero.lifting_surface import LiftingSurfaceConfig
from openfawe.aero.wake import WakeConfig, make_surface_guard
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero_archive import AeroWriter


def wing():
    y = np.linspace(-1., 1., 4)
    le = np.column_stack((np.zeros(4), y, np.zeros(4)))
    return Wing(le, le+[1., 0., 0.])


class WakeGuardIntegrationTests(unittest.TestCase):
    def test_all_six_methods_clear_a_stationary_wing(self):
        geometry, polar = wing(), SectionPolar.linear()
        for method in ('ll_straight', 'vsm', 'free_wake', 'vsm_free_wake', 'vlm', 'vlm_free_wake'):
            with self.subTest(method=method):
                config = WakeConfig(dt_s=.02, max_rows=2, surface_guard='reject_static')
                backend = AeroBackend(method, wake=config,
                    lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
                state = backend.start(geometry, polar, [2., 0., 0.])
                advanced = backend.advance(state, polar, [2., 0., 0.])
                self.assertEqual(state.step, 0)
                self.assertEqual(advanced.step, 1)

    def test_prediction_contact_is_rejected_before_commit_in_all_free_backends(self):
        geometry, polar = wing(), SectionPolar.linear()
        # Explicit diagnostic obstacle: not an aerodynamic body or extra lift surface.
        guard = StaticSurfaceGuard.from_triangles(geometry,
            [[[1.1, -3., -3.], [1.1, 3., -3.], [1.1, 0., 3.]]], surface_ids=['diagnostic_obstacle'])
        for method in ('free_wake', 'vsm_free_wake', 'vlm_free_wake'):
            for integrator in ('euler', 'heun', 'rk4'):
                with self.subTest(method=method, integrator=integrator):
                    backend = AeroBackend(method, wake=WakeConfig(dt_s=.1, max_rows=2,
                        convection_integrator=integrator, surface_guard='reject_static'),
                        lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
                    state = backend.start(geometry, polar, [2., 0., 0.])
                    original = state.free_rows.copy(), state.solution.circulation.copy()
                    with patch('openfawe.aero.wake.make_surface_guard', return_value=guard):
                        with self.assertRaises(WakeGeometryViolation) as caught:
                            backend.advance(state, polar, [2., 0., 0.])
                    report = caught.exception.geometry_report
                    self.assertFalse(report['acceptable'])
                    self.assertTrue(any(x['target_surface'] == 'diagnostic_obstacle' for x in report['contacts']))
                    self.assertEqual(report['stage'], {'euler':'accepted_geometry',
                        'heun':'heun_predictor', 'rk4':'rk4_stage2'}[integrator])
                    np.testing.assert_array_equal(state.free_rows, original[0])
                    np.testing.assert_array_equal(state.solution.circulation, original[1])
                    self.assertEqual(state.step, 0)

    def test_moving_geometry_and_live_policy_changes_are_not_silently_accepted(self):
        geometry = wing()
        config = WakeConfig(surface_guard='reject_static')
        with self.assertRaises(NotImplementedError):
            make_surface_guard(geometry, config, geometry.transformed(np.eye(3), [.1, 0., 0.]))
        for change in ({'surface_guard':'push'}, {'guard_chordwise_panels':0}):
            with self.assertRaises(ValueError):
                replace(config, **change)
        backend = AeroBackend('free_wake', wake=config)
        state = backend.start(geometry, SectionPolar.linear(), [2., 0., 0.])
        with self.assertRaises(ValueError):
            replace(backend, wake=replace(config, surface_guard='off')).advance(
                state, SectionPolar.linear(), [2., 0., 0.])

    def test_linear_potential_current_uvlm_rejects_each_integrator_without_commit(self):
        geometry, polar = wing(), SectionPolar.linear()
        guard = StaticSurfaceGuard.from_triangles(geometry,
            [[[1.1, -3., -3.], [1.1, 3., -3.], [1.1, 0., 3.]]],
            surface_ids=['diagnostic_obstacle'])
        for integrator, stage, time_s in (
                ('euler', 'accepted_geometry', .1),
                ('heun', 'heun_predictor', .1),
                ('rk4', 'rk4_stage2', .05)):
            with self.subTest(integrator=integrator):
                backend = AeroBackend('vlm_free_wake',
                    wake=WakeConfig(dt_s=.1, max_rows=2,
                        convection_integrator=integrator, surface_guard='reject_static'),
                    lifting_surface=LiftingSurfaceConfig(chordwise_panels=2,
                        wake_basis='linear_potential', wake_release='current'))
                state = backend.start(geometry, polar, [2., 0., .05])
                names = ('free_rows', 'wake_circulation', 'wake_node_potential_m2_s',
                         'release_times_s')
                original = {name: getattr(state, name).copy() for name in names}
                circulation = state.solution.lattice_circulation.copy()
                with patch('openfawe.aero.wake.make_surface_guard', return_value=guard):
                    with self.assertRaises(WakeGeometryViolation) as caught:
                        backend.advance(state, polar, [2., 0., .05])
                report = caught.exception.geometry_report
                self.assertEqual(report['stage'], stage)
                self.assertEqual(report['time_s'], time_s)
                self.assertFalse(report['acceptable'])
                self.assertTrue(any(c['target_surface'] == 'diagnostic_obstacle'
                                    for c in report['contacts']))
                for name in names:
                    np.testing.assert_array_equal(getattr(state, name), original[name])
                np.testing.assert_array_equal(state.solution.lattice_circulation, circulation)
                self.assertEqual((state.step, state.time_s), (0, 0.))

    def test_all_straight_methods_reject_real_obstacle_contacts_on_start_and_advance(self):
        geometry, polar = wing(), SectionPolar.linear()
        guard = StaticSurfaceGuard.from_triangles(geometry,
            [[[1.1, -3., -3.], [1.1, 3., -3.], [1.1, 0., 3.]]],
            surface_ids=['diagnostic_obstacle'])
        for method in ('ll_straight', 'vsm', 'vlm'):
            with self.subTest(method=method):
                backend = AeroBackend(method, wake=WakeConfig(surface_guard='reject_static'),
                    lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
                state = backend.start(geometry, polar, [2., 0., .05])
                original = state.solution.circulation.copy()
                with patch('openfawe.aero.wake.make_surface_guard', return_value=guard):
                    for action in ('start', 'advance'):
                        with self.subTest(action=action):
                            with self.assertRaises(WakeGeometryViolation) as caught:
                                if action == 'start':
                                    backend.start(geometry, polar, [2., 0., .05])
                                else:
                                    backend.advance(state, polar, [2., 0., .05])
                            report = caught.exception.geometry_report
                            self.assertEqual(report['stage'], 'prescribed_straight_rays'
                                if method == 'vsm' else 'prescribed_straight')
                            self.assertTrue(any(c['target_surface'] == 'diagnostic_obstacle'
                                                for c in report['contacts']))
                np.testing.assert_array_equal(state.solution.circulation, original)
                self.assertEqual((state.step, state.time_s), (0, 0.))

    def test_retained_campaign_runner_archives_the_actual_rejected_stage(self):
        # The 38-case campaign uses this unchanged run_one function. Its existing
        # rectangle option keeps this interface test tiny; it is not kite validation.
        repository = Path(__file__).resolve().parents[1]
        path = repository/'docs/tasks/OF-0006-polarcoupled20260924/evidence/run_comparison.py'
        specification = importlib.util.spec_from_file_location('guard_campaign_fixture', path)
        driver = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(driver)
        y = np.linspace(-4., 4., 9)
        le = np.column_stack((np.zeros_like(y), y, np.zeros_like(y)))
        geometry = Wing(le, le+[1., 0., 0.])
        guard = StaticSurfaceGuard.from_triangles(geometry,
            [[[1.1, -10., -10.], [1.1, 10., -10.], [1.1, 0., 10.]]],
            surface_ids=['diagnostic_obstacle'])
        job = dict(name='guard-runner-integration', model='rectangle', variant='rectangle',
            method='vlm_free_wake', alpha=0., beta=0., duration=.02, dt=.02,
            chordwise_panels=2, chordwise_spacing='uniform', normal_scheme='cell_secant',
            core='constant', integrator='heun', wake_length_m=10., release='current')
        parent = repository/'local'
        directory = parent/('guard-runner-test-'+uuid4().hex)
        directory.mkdir(parents=True)
        try:
            output = directory/'failed'
            constructor = driver.WakeConfig
            with patch.object(driver, 'WakeConfig', side_effect=lambda **kwargs:
                    replace(constructor(**kwargs), surface_guard='reject_static')):
                with patch('openfawe.aero.wake.make_surface_guard', return_value=guard):
                    result = driver.run_one(dict(job=job, output=str(output)))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error_type'], 'WakeGeometryViolation')
            self.assertEqual(result['attempted_failure_time_s'], .02)
            self.assertEqual(result['frames'], 1)
            self.assertEqual(result['last']['time_s'], 0.)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            report = manifest['geometry_guard_failure']
            self.assertEqual(report['stage'], 'heun_predictor')
            self.assertEqual(report['time_s'], .02)
            self.assertFalse(report['acceptable'])
            self.assertTrue(any(c['target_surface'] == 'diagnostic_obstacle'
                                for c in report['contacts']))
            with h5py.File(output/'wake.h5', 'r') as handle:
                self.assertEqual(len(handle['frames']), 1)
                self.assertEqual(handle['frames']['000000'].attrs['time_s'], 0.)
            with (output/'history.csv').open(encoding='utf-8', newline='') as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 1)
            self.assertEqual(float(records[0]['time_s']), 0.)
            self.assertEqual(json.loads((output/'result.json').read_text(encoding='utf-8')), result)
        finally:
            self.assertTrue(directory.resolve().is_relative_to(parent.resolve()))
            shutil.rmtree(directory)

    def test_geometry_failure_is_serialized_in_a_failed_archive(self):
        geometry = wing()
        guard = StaticSurfaceGuard.from_triangles(geometry,
            [[[1.1, -3., -3.], [1.1, 3., -3.], [1.1, 0., 3.]]])
        with self.assertRaises(WakeGeometryViolation) as caught:
            guard.check_rows(geometry.trailing_edge[None],
                             (geometry.trailing_edge+[.2, 0., 0.])[None], geometry)
        parent = Path(__file__).resolve().parents[1]/'local'
        directory = parent/('guard-test-'+uuid4().hex)
        directory.mkdir(parents=True)
        try:
            output = directory/'failed'
            writer = AeroWriter(output, {'name':'guard failure', 'aero_method':'free_wake'})
            writer.fail(caught.exception)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['geometry_guard_failure'], caught.exception.geometry_report)
        finally:
            self.assertTrue(directory.resolve().is_relative_to(parent.resolve()))
            shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
