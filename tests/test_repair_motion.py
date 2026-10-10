"""Regression contract for moving bound geometry in LL/VSM wake stages."""
from dataclasses import replace
import unittest
import numpy as np
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig, convect_rows
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings


class MovingBoundTests(unittest.TestCase):
    def test_galilean_covariance_all_integrators_and_substeps(self):
        wing, polar = Wing.ellipse(4), SectionPolar.linear()
        wind = np.array([5., 0., .3]); boost = np.array([1.3, -.2, .1])
        for name in ('free_wake', 'vsm_free_wake'):
            for integrator in ('euler', 'heun', 'rk4'):
                for substeps in (1, 2, 4):
                    with self.subTest(method=name, integrator=integrator, substeps=substeps):
                        config = WakeConfig(dt_s=.02, convection_integrator=integrator,
                            geometry_substeps=substeps, solve=SolveSettings(solver='newton', max_iterations=60))
                        b = AeroBackend(name, wake=config)
                        fixed = b.start(wing, polar, wind)
                        moving = b.start(wing, polar, wind+boost, structural_velocity=boost)
                        for step in (1, 2):
                            time = step*config.dt_s
                            fixed = b.advance(fixed, polar, wind, next_time_s=time)
                            moving = b.advance(moving, polar, wind+boost,
                                next_wing=wing.transformed(np.eye(3), time*boost),
                                structural_velocity=boost, next_time_s=time)
                            self.assertLessEqual(np.max(abs(moving.free_rows-time*boost-fixed.free_rows))/5., 1e-9)
                            for field in ('force', 'circulation'):
                                reference = getattr(fixed.solution, field)
                                difference = getattr(moving.solution, field)-reference
                                self.assertLessEqual(np.linalg.norm(difference)/np.linalg.norm(reference), 1e-8)

    def test_fixed_endpoint_identity_old_te_convection_and_topology_guard(self):
        wing, polar = Wing.ellipse(4), SectionPolar.linear()
        wind = np.array([5., 0., .3]); boost = np.array([1.3, -.2, .1])
        for integrator in ('euler', 'heun', 'rk4'):
            config = WakeConfig(dt_s=.02, free_wake=False, convection_integrator=integrator,
                                geometry_substeps=4)
            state = AeroBackend('free_wake', wake=config).start(wing, polar, wind)
            rows = convect_rows(state, config, wind)
            np.testing.assert_array_equal(rows, convect_rows(state, config, wind, next_wing=replace(wing)))
            endpoint = wing.transformed(np.eye(3), config.dt_s*boost)
            moving = convect_rows(state, config, wind, next_wing=endpoint)
            # In a prescribed ambient-only wake the old TE follows the fluid,
            # independently of the aircraft endpoint displacement.
            np.testing.assert_allclose(moving[0], wing.trailing_edge+config.dt_s*wind, rtol=0, atol=1e-14)
            with self.assertRaisesRegex(ValueError, 'topology'):
                convect_rows(state, config, wind, next_wing=Wing.ellipse(6))


if __name__ == '__main__':
    unittest.main()
