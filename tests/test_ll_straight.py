"""Straight LL reference, covariance, ownership and exact saved-wake checks."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from uuid import uuid4
import numpy as np

from openfawe.aero.ll_straight import LLStraightConfig, straight_network, network_radii
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings, solve_steady
from openfawe.aero.vortex import CoreModel, FilamentSingularity, segments_velocity
from openfawe.aero_archive import AeroReader, AeroWriter, _validate_straight_ll_frame
from openfawe.aero_demo import run_demo
from openfawe.aero_vtk import frame_network


def backend(**kwargs):
    return AeroBackend('ll_straight', **kwargs)


def cl(solution, wind, area):
    direction = np.asarray(wind)/np.linalg.norm(wind)
    lift = np.cross(direction, [0., 1., 0.])
    return float(solution.force@lift/(.5*1.225*np.dot(wind, wind)*area))


class StraightLLTests(unittest.TestCase):
    def test_invalid_candidate_rejected_before_frame_creation_and_retry(self):
        parent=Path(__file__).resolve().parents[1]/'local/ll-straight-tests'
        parent.mkdir(parents=True,exist_ok=True)
        wing, polar = Wing.ellipse(6), SectionPolar.linear()
        b=backend();state=b.start(wing,polar,[5.,0.,.3])
        writer=AeroWriter(parent/uuid4().hex,{'aero_method':'ll_straight'})
        try:
            bad_length=replace(state,prescribed_wake_row=state.prescribed_wake_row+[1.,0.,0.])
            bad_residual=replace(state,solution=replace(state.solution,
                residual=state.solution.residual_limit*2))
            for invalid in (bad_length,bad_residual):
                with self.assertRaises(ValueError):writer.append(invalid)
                self.assertEqual(writer.count,0)
                self.assertIsNone(writer.last_time)
                self.assertEqual(list(writer.file['frames']),[])
            writer.append(state)
            frame=writer.file['frames']['000000']
            np.testing.assert_array_equal(frame['circulation'][:],state.solution.circulation)
            np.testing.assert_array_equal(frame['panel_force'][:],state.solution.panel_force)
            writer.finish()
        finally:
            if writer.file.id.valid:writer.fail(ValueError('test interrupted'))
        reader=AeroReader(writer.directory)
        try:
            np.testing.assert_array_equal(reader.frame(0)['prescribed_wake_row_m'],state.prescribed_wake_row)
        finally:reader.close()

    def test_legacy_steady_equivalence_all_finite_kernels(self):
        wing, polar, wind = Wing.ellipse(12), SectionPolar.linear(), [5., 0., .3]
        for core in CoreModel:
            cfg = WakeConfig(core_model=core)
            b = backend(wake=cfg)
            state = b.start(wing, polar, wind)
            direct = solve_steady(wing, polar, wind, wake_length=1000., model=core,
                core_radius=cfg.core_radius_m, settings=cfg.solve)
            np.testing.assert_allclose(state.solution.circulation, direct.circulation, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(state.solution.panel_force, direct.panel_force, rtol=1e-12, atol=1e-12)

    def test_unchanged_upstream_llt_small_angle_scope(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/ll_straight_reference.json').read_text(encoding='utf-8'))
        for row in fixture['cases']:
            with self.subTest(case=row['name']):
                wing = Wing(row['leading_edge'], row['trailing_edge'])
                polar = SectionPolar(*np.array(row['polar']).T)
                b = backend(wake=WakeConfig(core_model=CoreModel.NONE,
                    solve=SolveSettings(solver='newton', max_iterations=60)),
                    ll_straight=LLStraightConfig(wake_length_m=1e6))
                s = b.start(wing, polar, row['inflow_m_s']).solution
                reference = np.array(row['circulation_m2_s'])
                self.assertLessEqual(np.max(np.abs(s.circulation-reference))/np.max(np.abs(reference)), .01)
                self.assertLessEqual(abs(cl(s,row['inflow_m_s'],row['area_m2'])/row['cl']-1), .01)
                np.testing.assert_allclose(wing.cp, row['control_points'], rtol=0, atol=1e-12)

    def test_relative_wind_galilean_and_rotation(self):
        wing, polar, wind = Wing.ellipse(10), SectionPolar.linear(), np.array([5., .1, .3])
        b = backend(); a = b.start(wing, polar, wind)
        translation = np.array([30., -2., 3.])
        c = b.start(wing, polar, wind+translation, structural_velocity=translation)
        np.testing.assert_allclose(c.prescribed_wake_row, a.prescribed_wake_row, rtol=0, atol=3e-13)
        np.testing.assert_allclose(c.solution.force, a.solution.force, rtol=1e-12, atol=1e-12)
        angle=.27; rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        posed=wing.transformed(rotation,[3.,4.,5.])
        d=b.start(posed,polar,rotation@wind)
        np.testing.assert_allclose(d.solution.force, rotation@a.solution.force, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(d.prescribed_wake_row,a.prescribed_wake_row@rotation.T+[3.,4.,5.],atol=1e-12)

    def test_no_released_history_or_path_dependence(self):
        wing, polar = Wing.ellipse(8), SectionPolar.linear()
        b=backend();a=b.start(wing,polar,[5.,0.,.3]);c=b.start(wing,polar,[5.,0.,.4])
        previous=a.solution.force.copy()
        for i in range(1,4):
            a=b.advance(a,polar,[5.,0.,.3],next_time_s=i*b.wake.dt_s)
        c=b.advance(c,polar,[5.,0.,.3])
        np.testing.assert_array_equal(a.solution.force,previous)
        np.testing.assert_array_equal(a.solution.force,c.solution.force)
        self.assertEqual(a.free_rows.shape,(0,wing.node_count,3));self.assertEqual(a.step,3)
        with self.assertRaises(ValueError):a.prescribed_wake_row[0,0]=2.

    def test_configuration_and_time_guards(self):
        for changes in ({'core_evolution':'age'},{'free_wake':False}):
            with self.assertRaises(ValueError):backend(wake=WakeConfig(**changes))
        b=backend();w=Wing.ellipse(6);p=SectionPolar.linear();s=b.start(w,p,[5,0,.2])
        with self.assertRaises(ValueError):backend(ll_straight=LLStraightConfig(200.)).advance(s,p,[5,0,.2])
        with self.assertRaises(ValueError):backend(wake=replace(b.wake,dt_s=.1)).advance(s,p,[5,0,.2])
        with self.assertRaises(ValueError):b.advance(s,p,[5,0,.2],next_time_s=.04)
        with self.assertRaises(ValueError):AeroBackend('free_wake').advance(s,p,[5,0,.2])
        with self.assertRaises(ValueError):b.start(w,p,[0,0,0])
        zero=backend(ll_straight=LLStraightConfig(wake_direction=(1,0,0))).start(w,p,[0,0,0])
        np.testing.assert_array_equal(zero.solution.force,0)

    def test_finite_network_conservation_probe_and_load_routing(self):
        w=Wing.ellipse(8);p=SectionPolar.linear();b=backend();s=b.start(w,p,[5,0,.3])
        nodes,edges,gamma=straight_network(s);divergence=np.zeros(len(nodes))
        np.add.at(divergence,edges[:,0],gamma);np.add.at(divergence,edges[:,1],-gamma)
        np.testing.assert_allclose(divergence,0,atol=1e-13)
        points=np.array([[2.,.3,.7],[-1.,.2,1.5]])
        direct=segments_velocity(points,nodes,edges,gamma,model=b.wake.core_model,core_radius=network_radii(s))
        np.testing.assert_array_equal(b.induced_velocity(s,points),direct)
        offset=np.array([1.,2.,3.]);moment=s.solution.moment(w,offset)
        np.testing.assert_allclose(moment,s.solution.moment(w)-np.cross(offset,s.solution.force),atol=1e-12)
        double=backend(wake=replace(b.wake,density_kg_m3=2.45)).start(w,p,[5,0,.3])
        np.testing.assert_allclose(double.solution.force,2*s.solution.force,atol=1e-12)

    def test_saved_exact_finite_wake_and_no_aging(self):
        parent=Path(__file__).resolve().parents[1]/'local/ll-straight-tests';parent.mkdir(parents=True,exist_ok=True)
        # Plain workspace mkdir inherits the working ACL on Windows; preserve
        # the small verification output instead of changing temp-directory ACLs.
        folder=parent/uuid4().hex;folder.mkdir()
        run=folder/'run'
        run_demo(run,panels=6,aero_method='ll_straight',duration_s=.1,heave_m=0,
            straight_wake_length_m=120.,export_vtk=True)
        reader=AeroReader(run)
        try:
            frame=reader.frame(2);self.assertEqual(frame['section_polar_used'],True)
            self.assertEqual(frame['prescribed_wake_length_m'],120.)
            np.testing.assert_allclose(frame['prescribed_wake_row_m']-frame['trailing_edge'],
                np.tile(120*np.array([5,0,.5])/np.linalg.norm([5,0,.5]),(7,1)))
            np.testing.assert_array_equal(frame_network(frame)[0],frame['network_nodes_m'])
            altered={**frame,'network_circulation_m2_s':frame['network_circulation_m2_s']+1}
            with self.assertRaises(ValueError):_validate_straight_ll_frame(altered)
        finally:reader.close()


if __name__=='__main__':
    unittest.main()
