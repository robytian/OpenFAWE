"""T01-T05: physical time, circulation ownership and explicit Euler checks."""

from dataclasses import replace
import unittest
import numpy as np

from openfawe.aero.vortex import segment_velocity
from openfawe.aero.wing import Wing,SectionPolar,SolveSettings,CirculationConvergenceError
from openfawe.aero.wake import WakeConfig,start_wake,advance_wake,wake_network,node_convection_velocity


class WakeTests(unittest.TestCase):
    def test_t01_uniform_zero_gamma_convection_and_release_age(self):
        wing=Wing.ellipse(4); polar=SectionPolar([-30,30],[0,0],[0,0],[0,0])
        cfg=WakeConfig(dt_s=.1); wind=np.array([2.,0.,.1])
        state=start_wake(wing,polar,cfg,wind)
        for step in range(1,5):
            state=advance_wake(state,polar,cfg,wind)
            self.assertEqual(state.step,step)
            for row,time in zip(state.free_rows,state.release_times_s):
                np.testing.assert_allclose(row,wing.trailing_edge+(state.time_s-time)*wind,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(state.solution.circulation,0)

    def test_t02_old_strengths_and_pure_repeat(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear();cfg=WakeConfig();wind=[1,0,.1]
        a=start_wake(wing,polar,cfg,wind);b=advance_wake(a,polar,cfg,wind)
        repeat=advance_wake(a,polar,cfg,wind)
        np.testing.assert_array_equal(b.free_rows,repeat.free_rows)
        np.testing.assert_array_equal(b.solution.circulation,repeat.solution.circulation)
        self.assertEqual(a.step,0);self.assertEqual(a.free_rows.shape[0],0)
        c=advance_wake(b,polar,cfg,wind)
        np.testing.assert_array_equal(c.older_circulation[0],b.solution.circulation)
        np.testing.assert_array_equal(c.release_times_s,[b.time_s,a.time_s])
        d=advance_wake(c,polar,cfg,wind)
        np.testing.assert_array_equal(d.older_circulation[1],b.solution.circulation)

    def test_t02_failed_step_and_row_limit_leave_state_unchanged(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear();cfg=WakeConfig();state=start_wake(wing,polar,cfg,[1,0,.1])
        before=state.solution.circulation.copy()
        restricted=replace(cfg,solve=SolveSettings(max_iterations=1))
        with self.assertRaises(CirculationConvergenceError):
            advance_wake(replace(state,config=restricted),polar,restricted,[1,0,.1])
        self.assertEqual(state.step,0);np.testing.assert_array_equal(state.solution.circulation,before)
        limited=replace(cfg,max_rows=1)
        once=advance_wake(start_wake(wing,polar,limited,[1,0,.1]),polar,limited,[1,0,.1])
        with self.assertRaisesRegex(ValueError,'row limit'): advance_wake(once,polar,limited,[1,0,.1])
        with self.assertRaises(ValueError): advance_wake(state,polar,cfg,[1,0,.1],next_time_s=3)
        with self.assertRaises(ValueError): WakeConfig(dt_s=0)
        with self.assertRaises(ValueError): WakeConfig(core_radius_m=0)

    def test_t03_closed_strength_graph_and_te_cancellation(self):
        wing=Wing.ellipse(6);polar=SectionPolar.linear();cfg=WakeConfig();state=start_wake(wing,polar,cfg,[1,0,.1])
        state=advance_wake(state,polar,cfg,[1,0,.1])
        nodes,edges,strength=wake_network(state)
        balance=np.zeros(len(nodes))
        np.add.at(balance,edges[:,0],strength);np.add.at(balance,edges[:,1],-strength)
        np.testing.assert_allclose(balance,0,rtol=0,atol=1e-12)
        width=wing.panels+1
        for a,b in edges:
            self.assertFalse(width<=a<2*width and width<=b<2*width, 'Equal bound/first-wake strength cancels the TE span edge')
        self.assertGreater(np.count_nonzero(strength),0)

    def test_t04_explicit_segment_sum_and_free_frozen_difference(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();cfg=WakeConfig();wind=np.array([1.,0.,.1])
        state=start_wake(wing,polar,cfg,wind);nodes,edges,gamma=wake_network(state);first=wing.panels+1
        expected=[]
        for point_id in range(first,len(nodes)):
            velocity=wind.copy()
            for edge,strength in zip(edges,gamma):
                if point_id in edge: continue
                velocity+=segment_velocity(nodes[edge[0]],nodes[edge[1]],nodes[point_id],strength,model=cfg.core_model,core_radius=cfg.core_radius_m)
            expected.append(velocity)
        expected=np.array(expected).reshape(1,first,3)
        np.testing.assert_allclose(node_convection_velocity(state,cfg,wind),expected,rtol=1e-11,atol=1e-12)
        ambient_config=replace(cfg,free_wake=False)
        ambient_state=start_wake(wing,polar,ambient_config,wind)
        moved=advance_wake(state,polar,cfg,wind);frozen=advance_wake(ambient_state,polar,ambient_config,wind)
        np.testing.assert_allclose(moved.free_rows[0],wing.trailing_edge+cfg.dt_s*expected[0],rtol=1e-11,atol=1e-12)
        self.assertGreater(np.linalg.norm(moved.free_rows-frozen.free_rows),1e-5)

    def test_t05_fixed_duration_timestep_sensitivity(self):
        forces=[]
        for dt in (.1,.05,.025):
            cfg=WakeConfig(dt_s=dt);polar=SectionPolar.linear();state=start_wake(Wing.ellipse(8),polar,cfg,[1,0,.1])
            for step in range(round(1/dt)): state=advance_wake(state,polar,cfg,[1,0,.1])
            self.assertAlmostEqual(state.time_s,1)
            forces.append(state.solution.force[2])
        self.assertLess(abs(forces[-1]/forces[-2]-1),.05)
        self.assertLess(abs(forces[-1]-forces[-2]),abs(forces[1]-forces[0]))

    def test_prescribed_motion_moves_only_new_wing_and_sets_relative_velocity(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();cfg=WakeConfig(free_wake=False)
        state=start_wake(wing,polar,cfg,[1,0,.1])
        shifted=wing.transformed(np.eye(3),[0,0,.002])
        advanced=advance_wake(state,polar,cfg,[1,0,.1],next_wing=shifted,structural_velocity=[0,0,.04])
        np.testing.assert_array_equal(advanced.wing.bound,shifted.bound)
        np.testing.assert_allclose(advanced.free_rows[0],wing.trailing_edge+cfg.dt_s*np.array([1,0,.1]))
        np.testing.assert_allclose(advanced.solution.relative_velocity-advanced.solution.induced_velocity,[[1,0,.06]]*4,atol=1e-12)


if __name__=='__main__': unittest.main()
