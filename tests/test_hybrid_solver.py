"""Residual-preserving retries, successful-root compatibility and trace transport."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from uuid import uuid4
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from openfawe.aero.nonlinear import (solve_newton, solve_newton_picard,
    NonlinearSolveError, HybridSolveError, HybridLoadReevaluationError,
    validate_solver_trace, section_solver_trace)
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings, solve_steady
from openfawe.aero.vsm import solve_vsm
from openfawe.aero.wake import WakeConfig
from openfawe.aero.methods import AeroBackend
from openfawe.aero_archive import AeroWriter, AeroReader
from openfawe.aero_vtk import export_paraview

ROOT = Path(__file__).resolve().parents[1]


class HybridSolverTests(unittest.TestCase):
    def test_settings_and_json_roundtrip(self):
        settings = SolveSettings(solver='newton_picard')
        self.assertEqual(SolveSettings(**json.loads(json.dumps(asdict(settings)))), settings)
        for kwargs in ({'fallback_relaxations': ()}, {'fallback_relaxations': (.3,)},
                       {'fallback_relaxations': (float('nan'),)}, {'fallback_max_iterations': 0},
                       {'fallback_max_iterations': True}):
            with self.assertRaises((ValueError, TypeError)):
                SolveSettings(**kwargs)

    def test_success_preserves_exact_newton_root(self):
        settings = SolveSettings(solver='newton_picard')
        seed = np.array([.3, .7])
        target = lambda x: x+1-x*x
        expected, iterations = solve_newton(target, seed, settings)
        actual, count, text = solve_newton_picard(target, seed, settings)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(count, iterations)
        trace = json.loads(text)
        self.assertFalse(trace['fallback_used'])
        self.assertEqual(len(trace['attempts']), 1)
        np.testing.assert_array_equal(seed, [.3, .7])

    def test_real_newton_exhaustion_falls_back_at_unchanged_tolerance(self):
        settings = SolveSettings(solver='newton_picard', max_iterations=1)
        target = lambda x: x+1-x*x
        seed = np.array([.5])
        with self.assertRaises(NonlinearSolveError):
            solve_newton(target, seed, settings)
        root, _, text = solve_newton_picard(target, seed, settings)
        trace = json.loads(text)
        self.assertTrue(trace['fallback_used'])
        self.assertEqual(trace['attempts'][0]['iterations'], 1)
        self.assertAlmostEqual(root[0], 1., places=7)
        self.assertLessEqual(max(abs(target(root)-root)),
                            settings.absolute_tolerance+settings.relative_tolerance*np.mean(abs(root)))

    def test_retry_restarts_seed_and_domain_failure_moves_to_next_relaxation(self):
        seed = np.zeros(1)
        def target(x):
            if x[0] > 1.5:
                raise ValueError('outside test domain')
            return 20.-19.*x
        with patch('openfawe.aero.nonlinear.solve_newton', side_effect=NonlinearSolveError('forced', iterations=1)):
            root, _, text = solve_newton_picard(target, seed, SolveSettings(solver='newton_picard'))
        trace = json.loads(text)
        self.assertEqual(trace['attempts'][1]['status'], 'domain_error')
        self.assertEqual(trace['relaxation'], .03)
        self.assertAlmostEqual(root[0], 1., places=7)
        np.testing.assert_array_equal(seed, [0.])

    def test_invalid_seed_and_complete_exhaustion(self):
        with self.assertRaises(ValueError):
            solve_newton_picard(lambda x: np.full_like(x, np.nan), [0.], SolveSettings())
        seed = np.zeros(1)
        with self.assertRaises(HybridSolveError) as caught:
            solve_newton_picard(lambda x: x+1., seed,
                SolveSettings(solver='newton_picard', max_iterations=1, fallback_max_iterations=2))
        attempts = json.loads(caught.exception.attempts_json)
        self.assertEqual(len(attempts), 3)
        self.assertEqual([x['status'] for x in attempts], ['failed', 'exhausted', 'exhausted'])
        np.testing.assert_array_equal(seed, [0.])

    def test_last_picard_update_is_checked(self):
        with patch('openfawe.aero.nonlinear.solve_newton', side_effect=NonlinearSolveError('forced', iterations=1)):
            root, _, trace = solve_newton_picard(lambda x: 10.-9.*x, [0.],
                SolveSettings(solver='newton_picard', fallback_max_iterations=1))
        np.testing.assert_array_equal(root, [1.])
        self.assertEqual(json.loads(trace)['attempts'][-1]['iterations'], 2)

    def test_ll_vsm_success_loads_are_identical(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear()
        original = SolveSettings(solver='newton', max_iterations=60)
        for solve in (solve_steady, solve_vsm):
            reference = solve(wing, polar, [5., 0., .5], settings=original)
            candidate = solve(wing, polar, [5., 0., .5], settings=replace(original, solver='newton_picard'))
            for name in ('circulation','panel_force','panel_couple','alpha_rad'):
                np.testing.assert_array_equal(getattr(candidate,name), getattr(reference,name))
            trace = validate_solver_trace(candidate.solver_trace_json, candidate.residual,
                                          candidate.residual_limit, wing.panels)
            self.assertFalse(trace['fallback_used'])
            self.assertEqual(len(trace['polar_segment_indices']), wing.panels)
            trace['attempts'][-1]['residual_m2_s'] = 100.
            with self.assertRaises(ValueError):
                validate_solver_trace(json.dumps(trace), candidate.residual, candidate.residual_limit, wing.panels)

    def test_state_archive_and_paraview_trace_roundtrip(self):
        parent = ROOT/'local'/'hybrid-tests'
        parent.mkdir(parents=True, exist_ok=True)
        path = parent/uuid4().hex
        path.mkdir()
        try:
            wing, polar = Wing.ellipse(4), SectionPolar.linear()
            config = WakeConfig(dt_s=.1, solve=SolveSettings(solver='newton_picard', max_iterations=1))
            backend = AeroBackend('free_wake', wake=config)
            case = dict(name='hybrid trace', aero_method='free_wake', aerodynamics=backend.metadata(),
                        wake=asdict(config), inflow_m_s=[5.,0.,.5], duration_s=.1,
                        polar={k:getattr(polar,k).tolist() for k in ('alpha_deg','cl','cd','cm')})
            writer = AeroWriter(path/'run', case)
            state = backend.start(wing, polar, [5.,0.,.5])
            self.assertTrue(json.loads(state.solution.solver_trace_json)['fallback_used'])
            writer.append(state)
            first = state.solution.circulation.copy()
            state = backend.advance(state, polar, [5.,0.,.5], next_time_s=.1)
            writer.append(state)
            writer.finish()
            reader = AeroReader(path/'run')
            try:
                self.assertEqual(reader.frame(1)['solver_trace_json'], state.solution.solver_trace_json)
                np.testing.assert_array_equal(reader.frame(0)['circulation'], first)
            finally:
                reader.close()
            export_paraview(path/'run', path/'vtk')
            fields = ET.parse(path/'vtk'/'wing-000001.vtp').findall('.//FieldData/DataArray')
            self.assertTrue(any(x.attrib['Name']=='solver_path_0_newton_1_picard' for x in fields))
        finally:
            if path.resolve().parent != parent.resolve():
                raise ValueError('refusing cleanup outside test workspace')
            shutil.rmtree(path)

    def test_failed_advance_keeps_previous_wake_state(self):
        wing, polar = Wing.ellipse(4), SectionPolar.linear()
        backend = AeroBackend('free_wake', wake=WakeConfig(
            dt_s=.1, solve=SolveSettings(solver='newton_picard')))
        state = backend.start(wing, polar, [5.,0.,.5])
        gamma, rows = state.solution.circulation.copy(), state.free_rows.copy()
        with patch('openfawe.aero.nonlinear.solve_newton_picard',
                   side_effect=HybridSolveError([dict(solver='newton',status='failed')])):
            with self.assertRaises(HybridSolveError):
                backend.advance(state, polar, [5.,0.,.5], next_time_s=.1)
        self.assertEqual(state.step, 0)
        self.assertEqual(state.time_s, 0.)
        np.testing.assert_array_equal(state.solution.circulation, gamma)
        np.testing.assert_array_equal(state.free_rows, rows)

    def test_real_ll_vsm_fallback_passes_load_reevaluation(self):
        wing, polar = Wing.ellipse(6), SectionPolar.linear()
        for solve in (solve_steady, solve_vsm):
            with self.subTest(solver=solve.__name__):
                result=solve(wing,polar,[5.,0.,.5],settings=SolveSettings(
                    solver='newton_picard',max_iterations=1))
                trace=validate_solver_trace(result.solver_trace_json,result.residual,result.residual_limit,wing.panels)
                self.assertTrue(trace['fallback_used'])
                self.assertEqual(trace['attempts'][0]['status'],'failed')
                self.assertLessEqual(result.residual,result.residual_limit)

    def test_load_reevaluation_failure_retains_trace_and_state(self):
        wing, polar = Wing.ellipse(4), SectionPolar.linear()
        config=WakeConfig(dt_s=.1,solve=SolveSettings(solver='newton_picard'))
        for method in ('free_wake','vsm_free_wake'):
            backend=AeroBackend(method,wake=config)
            state=backend.start(wing,polar,[5.,0.,.5])
            saved_gamma=state.solution.circulation.copy()
            trace=state.solution.solver_trace_json
            with patch('openfawe.aero.nonlinear.solve_newton_picard',
                       return_value=(np.zeros(wing.panels),1,trace)):
                with self.assertRaises(HybridLoadReevaluationError) as caught:
                    backend.advance(state,polar,[5.,0.,.5],next_time_s=.1)
            self.assertEqual(caught.exception.solver_trace_json,trace)
            self.assertGreater(caught.exception.residual,caught.exception.limit)
            self.assertEqual(state.step,0)
            np.testing.assert_array_equal(state.solution.circulation,saved_gamma)

    def test_linear_algebra_failure_can_trigger_picard(self):
        with patch('numpy.linalg.solve',side_effect=np.linalg.LinAlgError('test failure')), \
             patch('numpy.linalg.lstsq',side_effect=np.linalg.LinAlgError('test failure')):
            root,_,trace=solve_newton_picard(lambda x: np.ones_like(x),[0.],SolveSettings(solver='newton_picard'))
        self.assertTrue(json.loads(trace)['fallback_used'])
        np.testing.assert_allclose(root,[1.],atol=1e-8,rtol=0)

    def test_malformed_trace_fields_raise_value_error(self):
        wing,polar=Wing.ellipse(4),SectionPolar.linear()
        result=solve_steady(wing,polar,[5.,0.,.5],settings=SolveSettings(solver='newton_picard'))
        original=json.loads(result.solver_trace_json)
        mutations=[lambda t:t.pop('delta_seed_linf_m2_s'),
                   lambda t:t.update(delta_seed_linf_m2_s='not a number'),
                   lambda t:t.update(relaxation=.03),
                   lambda t:t['attempts'][0].update(solver='picard'),
                   lambda t:t.update(polar_segment_indices=3),
                   lambda t:t.update(polar_outside_table=[0]*wing.panels),
                   lambda t:t.update(attempts='not attempts')]
        for mutate in mutations:
            trace=json.loads(json.dumps(original));mutate(trace)
            with self.assertRaises(ValueError):
                validate_solver_trace(json.dumps(trace),result.residual,result.residual_limit,wing.panels)
        with self.assertRaises(ValueError):
            validate_solver_trace('[]',result.residual,result.residual_limit,wing.panels)

    def test_clamped_polar_metadata_marks_constant_extension(self):
        polar=SectionPolar([-10.,0.,10.],[-1.,0.,1.],[0.,0.,0.],[0.,0.,0.],bounds='clamp')
        angles=np.deg2rad([-20.,0.,20.])
        trace=json.loads(section_solver_trace('{}',polar,angles,0.,1e-8))
        self.assertEqual(trace['polar_outside_table'],[True,False,True])
        self.assertIn('zero slope',trace['polar_segment_convention'])
        np.testing.assert_array_equal(polar.coefficients(angles)[0],[-1.,0.,1.])


if __name__ == '__main__':
    unittest.main()
