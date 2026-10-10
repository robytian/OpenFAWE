"""T06: real computed-state archive/replay/export, not synthetic power data."""

import json
from dataclasses import asdict
from pathlib import Path
import shutil
from uuid import uuid4
import unittest
import xml.etree.ElementTree as ET
import numpy as np

from openfawe.aero_demo import run_demo
from openfawe.aero_archive import AeroReader,AeroWriter,method_label,field_semantics
from openfawe.aero_vtk import export_paraview,frame_network
from openfawe.aero.wake import WakeConfig,start_wake,wake_network
from openfawe.aero.wing import Wing,SectionPolar
from openfawe.aero.methods import AeroBackend


def write_vsm_free_case(directory,dt=.1,duration=.3):
    """Compute an actual small third-method archive for archive/UI verification."""
    cfg=WakeConfig(dt_s=dt,core_evolution='age_stretch')
    backend=AeroBackend('vsm_free_wake',wake=cfg)
    wing=Wing.ellipse(4).transformed(np.eye(3),[0,0,4.])
    polar=SectionPolar.linear();wind=[5.,0.,.5]
    case=dict(name='ellipse VSM free wake',geometry='analytic elliptic wing, 5 m span / 1 m root chord; not DE2019',
        panels=4,aero_method='vsm_free_wake',aerodynamics=backend.metadata(),wake=asdict(cfg),
        inflow_m_s=wind,base_position_m=[0,0,4.],heave_amplitude_m=0.,duration_s=duration,
        polar={k:getattr(polar,k).tolist() for k in ('alpha_deg','cl','cd','cm')})
    writer=AeroWriter(directory,case)
    try:
        state=backend.start(wing,polar,wind)
        writer.append(state)
        for i in range(1,round(duration/dt)+1):
            state=backend.advance(state,polar,wind,next_time_s=i*dt)
            writer.append(state)
        writer.finish()
    except BaseException as error:
        writer.fail(error);raise
    return state


class AeroArchiveTests(unittest.TestCase):
    def test_field_semantics_separate_closure_load_and_validity(self):
        for method in ('ll_straight','free_wake','vsm','vsm_free_wake','vlm','vlm_free_wake'):
            fields=field_semantics(method)
            self.assertEqual(fields['section_polar_used'],method not in ('vlm','vlm_free_wake'))
            if method in ('vsm','vsm_free_wake'):
                self.assertIn('three-quarter',fields['polar_alpha_rad_definition'])
                self.assertIn('normal-to-span',fields['reynolds_definition'])
            if fields['section_polar_used']:
                self.assertEqual(fields['polar_validity_status'],'unknown')
                self.assertIn('no Reynolds interpolation',fields['polar_reynolds_policy'])

    def setUp(self):
        self.parent=Path(__file__).resolve().parents[1]/'local'/'aero-tests'
        self.parent.mkdir(parents=True,exist_ok=True)
        self.root=self.parent/uuid4().hex
        self.root.mkdir()

    def tearDown(self):
        resolved=self.root.resolve()
        if resolved.parent != self.parent.resolve():
            raise ValueError('Refusing cleanup outside test workspace')
        shutil.rmtree(resolved)

    def test_roundtrip_true_times_and_source_subpackage_hashes(self):
        output=self.root/'run';result=run_demo(output,panels=4,dt_s=.1,duration_s=.3)
        self.assertEqual(result['frames'],4)
        reader=AeroReader(output)
        try:
            np.testing.assert_allclose(reader.times,[0,.1,.2,.3],rtol=0,atol=1e-14)
            last=reader.frame(3)
            self.assertEqual(last['free_rows'].shape,(3,5,3))
            np.testing.assert_allclose(last['relative_velocity']-last['induced_velocity']+last['structural_velocity'],[[5,0,.5]]*4,atol=1e-12)
            np.testing.assert_allclose(last['panel_force'].sum(axis=0),result['final_force_N'])
            self.assertLessEqual(last['residual_m2_s'],last['residual_limit_m2_s'])
            self.assertIn('openfawe/aero/wing.py',reader.manifest['source']['files_sha256'])
            self.assertIn('openfawe/aero/wake.py',reader.manifest['source']['files_sha256'])
            self.assertFalse(any('power' in key for key in last))
            last['free_rows'][:]=999
            self.assertFalse(np.all(reader.frame(3)['free_rows']==999))
            with self.assertRaises(ValueError): reader.frame(-1)
        finally: reader.close()

    def test_existing_run_and_export_are_not_overwritten(self):
        output=self.root/'run';run_demo(output,panels=4,dt_s=.1,duration_s=.2)
        original=(output/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError): run_demo(output,panels=4,dt_s=.1,duration_s=.2)
        self.assertEqual((output/'manifest.json').read_bytes(),original)
        export_paraview(output,self.root/'vtk')
        with self.assertRaises(FileExistsError): export_paraview(output,self.root/'vtk')

    def test_incomplete_failed_and_tampered_runs_are_rejected(self):
        writer=AeroWriter(self.root/'incomplete',{'name':'failed fixture'})
        state=start_wake(Wing.ellipse(4),SectionPolar.linear(),WakeConfig(),[1,0,.1])
        try:
            writer.append(state)
            with self.assertRaises(ValueError): writer.append(state)
            with self.assertRaises(ValueError): AeroReader(self.root/'incomplete')
        finally: writer.fail(ValueError('deliberate failure'))
        with self.assertRaises(ValueError): AeroReader(self.root/'incomplete')
        output=self.root/'tamper';run_demo(output,panels=4,dt_s=.1,duration_s=.2)
        with (output/'wake.h5').open('ab') as stream: stream.write(b'changed')
        with self.assertRaises(ValueError): AeroReader(output)

    def test_paraview_collection_matches_physical_frames_and_connectivity(self):
        output=self.root/'run';run_demo(output,panels=4,dt_s=.1,duration_s=.2,export_vtk=True)
        datasets=ET.parse(output/'paraview/wake.pvd').getroot().find('Collection').findall('DataSet')
        self.assertEqual(len(datasets),6)
        np.testing.assert_allclose([float(d.attrib['timestep']) for d in datasets],[0,0,.1,.1,.2,.2])
        reader=AeroReader(output)
        try:
            nodes,edges,gamma=frame_network(reader.frame(2))
            piece=ET.parse(output/'paraview/wake-000002.vtp').getroot().find('./PolyData/Piece')
            self.assertEqual(int(piece.attrib['NumberOfPoints']),len(nodes))
            self.assertEqual(int(piece.attrib['NumberOfLines']),len(edges))
            raw=piece.find('./Lines/DataArray[@Name="connectivity"]').text
            np.testing.assert_array_equal(np.fromstring(raw,sep=' ',dtype=int).reshape(-1,2),edges)
            raw=piece.find('./CellData/DataArray[@Name="circulation_m2_s"]').text
            np.testing.assert_allclose(np.fromstring(raw,sep=' '),gamma,rtol=0,atol=0)
        finally: reader.close()

    def test_invalid_duration_creates_no_run(self):
        for kwargs in [{'duration_s':.25,'dt_s':.1},{'duration_s':100},{'dt_s':0}]:
            path=self.root/'invalid'
            with self.assertRaises(ValueError): run_demo(path,**kwargs)
            self.assertFalse(path.exists())

    def test_vsm_free_wake_roundtrip_actual_network_polars_and_vtk_identity(self):
        output=self.root/'vsm-free'
        state=write_vsm_free_case(output,duration=.2)
        reader=AeroReader(output)
        try:
            self.assertEqual(reader.method,'vsm_free_wake')
            self.assertEqual(method_label(reader.method),'OpenFAWE VSM\u2013Free Vortex Wake')
            frame=reader.frame(2)
            self.assertEqual(frame['free_rows'].shape,(2,5,3))
            self.assertIn('polar_alpha_rad',frame)
            self.assertIn('collocation_points',frame)
            self.assertIn('network_core_radius_m',frame)
            self.assertNotIn('prescribed_wake_direction',frame)
            np.testing.assert_array_equal(frame['polar_alpha_rad'],state.solution.polar_alpha_rad)
            np.testing.assert_array_equal(frame['collocation_points'],state.solution.collocation_points)
            np.testing.assert_array_equal(frame['control_points'],state.wing.cp)
            for actual,expected in zip(frame_network(frame),wake_network(state)):
                np.testing.assert_array_equal(actual,expected)
            self.assertIn('openfawe/aero/vsm_wake.py',reader.manifest['source']['files_sha256'])
        finally:reader.close()
        export_paraview(output,self.root/'third-vtk')
        piece=ET.parse(self.root/'third-vtk/wake-000002.vtp').getroot().find('./PolyData/Piece')
        field=piece.find('./FieldData/DataArray[@Name="aero_method_id_0_ll_free_wake_1_vsm_straight_2_vsm_free_wake"]')
        self.assertEqual(int(field.text),2)
        self.assertIsNone(piece.find('./FieldData/DataArray[@Name="aero_method_id_0_free_wake_1_vsm"]'))
        self.assertIsNotNone(piece.find('./CellData/DataArray[@Name="core_radius_m"]'))

    def test_third_method_archive_rejects_cross_method_states(self):
        wing=Wing.ellipse(4);polar=SectionPolar.linear();wind=[5,0,.5]
        states={m:AeroBackend(m).start(wing,polar,wind) for m in ('free_wake','vsm_free_wake')}
        for target,source in [('free_wake','vsm_free_wake'),('vsm_free_wake','free_wake')]:
            writer=AeroWriter(self.root/target,{'aero_method':target})
            try:
                with self.assertRaisesRegex(ValueError,'state and archived'):
                    writer.append(states[source])
                self.assertEqual(writer.count,0)
            finally:writer.fail(ValueError('intentional ownership check'))


if __name__=='__main__': unittest.main()
