"""OpenFAWE root-finder verification against exact and upstream results."""
import json
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from openfawe.aero.nonlinear import solve_newton, NonlinearSolveError
from openfawe.aero.wing import SolveSettings, solve_steady
from openfawe.aero.aircraft import load_aircraft
from openfawe.aero.vortex import CoreModel
from openfawe.aero.reference import inflow, refine_aircraft, load_coefficients
from openfawe.aero.vsm import solve_vsm
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig

ROOT = Path(__file__).resolve().parents[1]
NEWTON = SolveSettings(solver='newton', max_iterations=60)


class NonlinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3 = load_aircraft(ROOT/'cases/aircraft/v3_cad_attached.json')

    def test_unstable_fixed_point_has_exact_admissible_root(self):
        seed = np.zeros(2)
        matrix = np.array([[2., .3], [.2, 1.5]])
        root = np.array([1., 2.])
        def target(x):
            if np.any(x < -.1) or np.any(x > 2.1):
                raise ValueError('outside domain')
            return matrix @ (x-root)+root
        result, _ = solve_newton(target, seed, NEWTON)
        assert_allclose(result, root, atol=1e-9)
        assert_allclose(seed, [0., 0.], atol=0)

    def test_invalid_input_and_absent_root_remain_failures(self):
        with self.assertRaises(ValueError):
            SolveSettings(solver='automatic-silent-fallback')
        def target(x):
            if x[0] < 0 or x[0] > 1:
                raise ValueError('outside domain')
            return x+1
        with self.assertRaises(NonlinearSolveError):
            solve_newton(target, np.array([.5]), NEWTON)
        with self.assertRaises(ValueError):
            solve_newton(target, np.array([-1.]), NEWTON)

    def test_final_allowed_update_is_checked(self):
        result, iterations = solve_newton(lambda x: 2*x-1., np.zeros(1),
            SolveSettings(solver='newton', max_iterations=1))
        assert_allclose(result, [1.], atol=1e-9)
        self.assertEqual(iterations, 1)

    def test_invalid_trials_and_one_sided_derivatives_are_rejected(self):
        rejected = []
        def logarithmic(x):
            if x[0] <= 0:
                rejected.append(float(x[0]))
                raise ValueError('logarithm domain')
            return x+np.log(x)
        result, _ = solve_newton(logarithmic, np.array([10.]), NEWTON)
        assert_allclose(result, [1.], atol=1e-8)
        self.assertTrue(rejected)
        above = []
        def bounded(x):
            if not 0 <= x[0] <= 1:
                above.append(float(x[0]))
                raise ValueError('bounded domain')
            return 2*x-.25
        result, _ = solve_newton(bounded, np.ones(1), NEWTON)
        assert_allclose(result, [.25], atol=1e-8)
        self.assertTrue(any(x>1 for x in above))

    def test_same_five_original_vsm_results_with_newton(self):
        source = json.loads((ROOT/'tests/fixtures/v3_reference.json').read_text())
        for case in source['cases']:
            with self.subTest(alpha=case['alpha_deg'], beta=case['beta_deg']):
                result = solve_vsm(self.v3.wing, self.v3.polar, case['inflow_m_s'], settings=NEWTON)
                assert_allclose(result.circulation, case['circulation'], rtol=5e-5, atol=5e-5)
                assert_allclose(result.cl, case['cl_section'], rtol=5e-5, atol=5e-5)
                self.assertLessEqual(result.residual, result.residual_limit)

    def test_two_failed_ll_initializations_recover_same_picard_root(self):
        for beta in (-8.061, 8.074):
            wind = inflow(5e5*1.4639e-5/2.5675, 7.4, beta)
            args = dict(wake_length=2000., model=CoreModel.LAMB_OSEEN, core_radius=.15)
            with self.assertRaises(ValueError):
                solve_steady(self.v3.wing, self.v3.polar, wind,
                             settings=SolveSettings(relaxation=.03, max_iterations=4000), **args)
            fixed = solve_steady(self.v3.wing, self.v3.polar, wind, initial=np.zeros(self.v3.wing.panels),
                                 settings=SolveSettings(relaxation=.03, max_iterations=4000), **args)
            newton = solve_steady(self.v3.wing, self.v3.polar, wind, settings=NEWTON, **args)
            assert_allclose(newton.circulation, fixed.circulation, atol=1e-7, rtol=1e-7)
            self.assertLessEqual(newton.residual, newton.residual_limit)
            self.assertLess(np.rad2deg(newton.alpha_rad).max(), 21.5)

    def test_refined_vsm_recovers_same_unmodified_closure(self):
        definition = refine_aircraft(self.v3, 2)
        slow = solve_vsm(definition.wing, definition.polar, inflow(3., 6.),
                         settings=SolveSettings(relaxation=.005, max_iterations=20000))
        newton = solve_vsm(definition.wing, definition.polar, inflow(3., 6.), settings=NEWTON)
        assert_allclose(newton.force, slow.force, rtol=1e-7, atol=1e-7)

    def test_actual_backend_records_solver_and_preserves_step(self):
        for method in ('vsm', 'free_wake'):
            backend = AeroBackend(method, wake=WakeConfig(dt_s=.1, max_rows=2, solve=NEWTON), vsm_solve=NEWTON)
            first = backend.start(self.v3.wing, self.v3.polar, inflow(3., 6.))
            before = first.solution.circulation.copy()
            second = backend.advance(first, self.v3.polar, inflow(3., 6.))
            assert_allclose(first.solution.circulation, before, atol=0, rtol=0)
            self.assertLessEqual(second.solution.residual, second.solution.residual_limit)
            settings = backend.metadata()['solve'] if method=='vsm' else backend.metadata()['wake']['solve']
            self.assertEqual(settings['solver'], 'newton')

    def test_moment_axes_against_known_yaw_rotation(self):
        class KnownMoment:
            force = np.zeros(3)
            def moment(self, wing, reference):
                return np.array([1., 2., 3.])
        arguments = dict(solution=KnownMoment(), wing=None, velocity=inflow(1.,0.,30.),
                         area=1., chord=1., density=2.)
        body = load_coefficients(**arguments)
        wind = load_coefficients(**arguments, moment_frame='wind')
        assert_allclose([body[k] for k in ('CMx','CMy','CMz')], [1.,2.,3.], atol=1e-14)
        assert_allclose([wind[k] for k in ('CMx','CMy','CMz')],
                        [np.sqrt(3)/2+1., -.5+np.sqrt(3), 3.], atol=1e-14)
        with self.assertRaises(ValueError):
            load_coefficients(**arguments, moment_frame='unknown')

    def test_raw_supported_exact_endpoints_without_bound_extension(self):
        source = load_aircraft(ROOT/'cases/aircraft/v3_cad_raw_supported.json')
        raw = json.loads((ROOT/'cases/aircraft/v3_cad.json').read_text())
        for panel, polar in enumerate(source.polar.polars):
            name = raw['surfaces'][0]['polar_ids'][panel]
            original = raw['polars'][name]
            for endpoint in (0, -1):
                angle = np.deg2rad(polar.alpha_deg[endpoint])
                value = polar.coefficients(angle)
                index = original['alpha_deg'].index(polar.alpha_deg[endpoint])
                assert_allclose(value, [original[k][index] for k in ('cl','cd','cm')], atol=0, rtol=0)
                outside = np.nextafter(angle, -np.inf if endpoint==0 else np.inf)
                with self.assertRaises(ValueError):
                    polar.coefficients(outside)


if __name__ == '__main__':
    unittest.main()
