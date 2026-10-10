"""Independent geometry-ODE checks, separate from full aerodynamic accuracy."""

from dataclasses import replace
import unittest
import numpy as np

from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig, start_wake, convect_rows, node_convection_velocity


class ConvectionTests(unittest.TestCase):
    def setUp(self):
        self.wing = Wing.ellipse(4)
        self.zero = SectionPolar([-30,30], [0,0], [0,0], [0,0])

    def test_euler_retains_previous_geometry_exactly(self):
        config = WakeConfig(dt_s=.1)
        state = start_wake(self.wing, SectionPolar.linear(), config, [2,0,.1])
        expected = state.wing.trailing_edge[None]+config.dt_s*node_convection_velocity(state,config,[2,0,.1])
        np.testing.assert_array_equal(convect_rows(state,config,[2,0,.1]),expected)

    def test_rk4_uses_position_and_time_stages(self):
        config = WakeConfig(dt_s=.4, free_wake=False, convection_integrator='rk4')
        state = start_wake(self.wing,self.zero,config,[2,0,0])
        # x'=x+2t, y'=0, z'=0, x(t)=(x0+2)*exp(t)-2*(t+1).
        def field(points,time):
            values = np.zeros_like(points)
            values[:,0] = points[:,0]+2*time
            return values
        exact = self.wing.trailing_edge.copy()
        exact[:,0] = (exact[:,0]+2)*np.exp(config.dt_s)-2*(config.dt_s+1)
        errors = []
        for substeps in (1,2,4):
            rows = convect_rows(state,replace(config,geometry_substeps=substeps),field)
            errors.append(float(np.max(np.abs(rows[0]-exact))))
        self.assertGreater(errors[0]/errors[1],13)
        self.assertGreater(errors[1]/errors[2],14)
        np.testing.assert_array_equal(state.free_rows,np.empty((0,self.wing.node_count,3)))

    def test_substeps_do_not_create_extra_circulation_or_release_rows(self):
        for method in ('free_wake','vsm_free_wake'):
            for policy in ('constant','age','age_stretch'):
                config = WakeConfig(dt_s=.1, convection_integrator='rk4',
                                    geometry_substeps=3, core_evolution=policy)
                backend = AeroBackend(method,wake=config)
                state = backend.start(self.wing,self.zero,[2,0,0])
                old = state
                for step in (1,2,3):
                    state = backend.advance(state,self.zero,[2,0,0])
                    self.assertEqual(state.step,step)
                    self.assertEqual(len(state.free_rows),step)
                    np.testing.assert_allclose(state.release_times_s,np.arange(step-1,-1,-1)*.1,atol=1e-15)
                    for row,time in zip(state.free_rows,state.release_times_s):
                        np.testing.assert_allclose(row,self.wing.trailing_edge+[2*(state.time_s-time),0,0],atol=1e-14)
                    np.testing.assert_array_equal(state.older_circulation,0)
                self.assertEqual(old.step,0)

    def test_heun_second_order_for_time_dependent_growth(self):
        config=WakeConfig(dt_s=.4,free_wake=False,convection_integrator='heun')
        state=start_wake(self.wing,self.zero,config,[2,0,0])
        def field(points,time):
            result=np.zeros_like(points)
            result[:,0]=points[:,0]+2*time
            return result
        exact=(self.wing.trailing_edge[:,0]+2)*np.exp(.4)-2*1.4
        errors=[]
        for n in (2,4,8):
            rows=convect_rows(state,replace(config,geometry_substeps=n),field)
            errors.append(np.max(np.abs(rows[0,:,0]-exact)))
        self.assertGreater(errors[0]/errors[1],3.6)
        self.assertGreater(errors[1]/errors[2],3.8)

    def test_heun_retains_single_physical_release(self):
        for method in ('free_wake','vsm_free_wake'):
            cfg=WakeConfig(dt_s=.1,convection_integrator='heun',geometry_substeps=4)
            backend=AeroBackend(method,wake=cfg)
            state=backend.start(self.wing,self.zero,[2,0,0])
            advanced=backend.advance(state,self.zero,[2,0,0])
            self.assertEqual(advanced.step,1)
            np.testing.assert_array_equal(advanced.release_times_s,[0])
            np.testing.assert_allclose(advanced.free_rows[0],self.wing.trailing_edge+[.2,0,0],atol=1e-14)

    def test_invalid_integrator_and_substeps(self):
        for kwargs in ({'convection_integrator':'implicit'}, {'geometry_substeps':0},
                       {'geometry_substeps':1.5}, {'geometry_substeps':True}):
            with self.assertRaises(ValueError):
                WakeConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
