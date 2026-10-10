"""Exact lattice archive, finite-network export and five-method replay checks."""

from dataclasses import asdict,replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from openfawe.aero.methods import AeroBackend
from openfawe.aero.lifting_surface import LiftingSurfaceConfig, lifting_surface_network, lifting_surface_edge_radii, start_lifting_surface
from openfawe.aero.aircraft import WingAssembly
from openfawe.aero.vortex import CoreModel
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero_archive import AeroReader, AeroWriter, LATTICE_SCHEMA, LEGACY_SCHEMA, method_label
from openfawe.aero_vtk import export_paraview, frame_network
from openfawe.provenance import sha256_file
from openfawe.run_catalog import discover_runs, comparison_notes


class LiftingSurfaceArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=Path(__file__).resolve().parents[1]/'local'/'lattice-archive-tests'
        cls.parent.mkdir(parents=True,exist_ok=True)
        cls.root=cls.parent/uuid4().hex
        cls.root.mkdir()
        y=np.linspace(-2.,2.,5)
        le=np.column_stack([np.zeros(5),y,np.full(5,3.)])
        wing=Wing(le,le+[1.,0.,0.])
        polar=SectionPolar.linear();wind=[5.,0.,.2]
        cfg=WakeConfig(dt_s=.05,core_radius_m=.02)
        cls.states={}
        for method in ('free_wake','vsm','vsm_free_wake','vlm','vlm_free_wake'):
            method_cfg=replace(cfg,core_evolution='age_stretch') if method=='vlm_free_wake' else cfg
            backend=AeroBackend(method,wake=method_cfg,
                                lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
            case={'name':'Rectangular wing archive verification','model':'Rectangular wing',
                  'geometry':'Flat rectangular wing; 4 m span / 1 m chord',
                  'panels':4,'aero_method':method,'aerodynamics':backend.metadata(),
                  'wake':asdict(method_cfg),'inflow_m_s':wind,
                  # Deliberately supplied to all adapters: VLM must explicitly
                  # record that a present input polar is not its closure law.
                  'polar':{key:getattr(polar,key).tolist() for key in ('alpha_deg','cl','cd','cm')}}
            writer=AeroWriter(cls.root/method,case)
            try:
                state=backend.start(wing,polar,wind)
                writer.append(state)
                for index in range(1,3):
                    state=backend.advance(state,polar,wind,next_time_s=index*cfg.dt_s)
                    writer.append(state)
                writer.finish()
                cls.states[method]=state
            except BaseException as error:
                writer.fail(error)
                raise

    @classmethod
    def tearDownClass(cls):
        resolved=cls.root.resolve()
        if resolved.parent!=cls.parent.resolve():
            raise ValueError('Refusing cleanup outside test workspace')
        shutil.rmtree(resolved)

    def test_roundtrip_lattice_network_units_and_force_routing(self):
        for method in ('vlm','vlm_free_wake'):
            with self.subTest(method=method):
                reader=AeroReader(self.root/method)
                try:
                    state=self.states[method]
                    frame=reader.frame(2)
                    self.assertEqual(reader.manifest['schema'],LATTICE_SCHEMA)
                    np.testing.assert_array_equal(frame['lattice_nodes_m'],state.mesh.vertices)
                    np.testing.assert_array_equal(frame['lattice_cells'],state.mesh.cells)
                    self.assertEqual(frame['lattice_cells'].shape,(8,4))
                    np.testing.assert_array_equal(frame['lattice_circulation_m2_s'],state.solution.lattice_circulation.ravel())
                    np.testing.assert_array_equal(frame['lattice_panel_force_N'],state.solution.lattice_force.reshape(-1,3))
                    for actual,expected in zip(frame_network(frame),lifting_surface_network(state)):
                        np.testing.assert_array_equal(actual,expected)
                    np.testing.assert_array_equal(frame['network_core_radius_m'],lifting_surface_edge_radii(state))
                    np.testing.assert_allclose(frame['lattice_panel_force_N'].sum(axis=0),frame['panel_force'].sum(axis=0),rtol=1e-14,atol=1e-14)
                    self.assertFalse(frame['section_polar_used'])
                    self.assertNotIn('polar_clipped',frame)
                    self.assertNotIn('residual_m2_s',frame)
                    self.assertLessEqual(frame['boundary_residual_m_s'],frame['boundary_residual_limit_m_s'])
                    self.assertEqual(len(frame['free_rows']),0 if method=='vlm' else 2)
                    self.assertEqual('prescribed_wake_row_m' in frame,method=='vlm')
                    self.assertEqual('wake_circulation_m2_s' in frame,method=='vlm_free_wake')
                    if method=='vlm_free_wake':
                        np.testing.assert_array_equal(frame['span_core_radius_m'],state.cores.span)
                        np.testing.assert_array_equal(frame['stream_core_radius_m'],state.cores.stream)
                        self.assertEqual(frame['core_evolution'],'age_stretch')
                    self.assertIn('openfawe/aero/lifting_surface.py',reader.manifest['source']['files_sha256'])
                    frame['lattice_circulation_m2_s'][:]=999.
                    self.assertFalse(np.all(reader.frame(2)['lattice_circulation_m2_s']==999.))
                finally:
                    reader.close()

    def test_steady_mirror_projection_diagnostics_survive_archive(self):
        reference = json.loads((Path(__file__).parent/'fixtures/lifting_surface_reference.json').read_text())
        item = next(case for case in reference['cases'] if case['name'].startswith('wing_tail'))
        wing = WingAssembly(tuple(Wing(s['leading_edge'], s['trailing_edge'])
                                  for s in item['surfaces']), ('wing', 'tail'))
        alpha = np.deg2rad(item['alpha_deg'])
        wind = item['speed_m_s']*np.array([np.cos(alpha), 0., np.sin(alpha)])
        settings = LiftingSurfaceConfig(4, wake_length_m=1e7,
                                        wake_direction=(1., 0., 0.),
                                        steady_mirror_projection=True)
        state = start_lifting_surface(wing, WakeConfig(core_model=CoreModel.NONE),
                                      wind, settings=settings)
        path = self.root/'dlr-steady-mirror'
        writer = AeroWriter(path, {'name': 'DLR steady mirror diagnostic', 'aero_method': 'vlm'})
        try:
            writer.append(state)
            writer.finish()
        except BaseException as error:
            writer.fail(error)
            raise
        reader = AeroReader(path)
        try:
            frame = reader.frame(0)
            self.assertEqual(frame['mirror_preprojection_max_abs_m2_s'],
                             state.solution.mirror_preprojection_max_abs_m2_s)
            self.assertEqual(frame['unprojected_boundary_residual_m_s'],
                             state.solution.unprojected_boundary_residual_m_s)
            self.assertLess(frame['boundary_residual_m_s'],
                            frame['boundary_residual_limit_m_s'])
        finally:
            reader.close()
        tampered = self.root/'dlr-steady-mirror-tampered'
        shutil.copytree(path, tampered)
        with h5py.File(tampered/'wake.h5', 'r+') as archive:
            gamma = archive['frames']['000000']['lattice_circulation_m2_s']
            gamma[0] += 1e-3
        manifest_path = tampered/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['wake_sha256'] = sha256_file(tampered/'wake.h5')
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mirror circulation is not exactly paired'):
            AeroReader(tampered)

    def test_new_method_writer_rejects_state_substitution_without_creating_frame(self):
        for target,source in [('vlm','vlm_free_wake'),('vlm_free_wake','vlm'),('vlm','vsm'),('vlm_free_wake','free_wake')]:
            writer=AeroWriter(self.root/f'reject-{target}-{source}',{'aero_method':target})
            try:
                with self.assertRaisesRegex(ValueError,'state and archived'):
                    writer.append(self.states[source])
                self.assertEqual(writer.count,0)
                self.assertEqual(len(writer.file['frames']),0)
            finally:
                writer.fail(ValueError('Intentional test of method ownership'))

    def test_vtk_exports_full_mesh_exact_finite_wake_and_truthful_method_fields(self):
        five_id='aero_method_id_0_ll_free_wake_1_vsm_straight_2_vsm_free_wake_3_vlm_straight_4_uvlm_free_wake'
        old_id='aero_method_id_0_ll_free_wake_1_vsm_straight_2_vsm_free_wake'
        for method,code in [('vlm',3),('vlm_free_wake',4)]:
            directory=self.root/(method+'-vtk')
            export_paraview(self.root/method,directory)
            collection=ET.parse(directory/'wake.pvd').getroot().find('Collection')
            np.testing.assert_allclose([float(item.attrib['timestep']) for item in collection],[0,0,.05,.05,.1,.1])
            reader=AeroReader(self.root/method)
            try:
                frame=reader.frame(2)
                wing=ET.parse(directory/'wing-000002.vtp').getroot().find('./PolyData/Piece')
                wake=ET.parse(directory/'wake-000002.vtp').getroot().find('./PolyData/Piece')
                self.assertEqual(int(wing.attrib['NumberOfPolys']),8)
                for piece in (wing,wake):
                    self.assertEqual(int(piece.find(f'./FieldData/DataArray[@Name="{five_id}"]').text),code)
                    self.assertIsNone(piece.find(f'./FieldData/DataArray[@Name="{old_id}"]'))
                    self.assertIsNone(piece.find('./FieldData/DataArray[@Name="aero_method_id_0_free_wake_1_vsm"]'))
                    self.assertIsNotNone(piece.find('./FieldData/DataArray[@Name="boundary_residual_max_m_s"]'))
                    self.assertEqual(float(piece.find('./FieldData/DataArray[@Name="section_polar_used_0_false_1_true"]').text),0.)
                def values(piece,section,name,dtype=float):
                    return np.fromstring(piece.find(f'./{section}/DataArray[@Name="{name}"]').text or '',sep=' ',dtype=dtype)
                np.testing.assert_array_equal(values(wing,'Polys','connectivity',int).reshape(-1,4),frame['lattice_cells'])
                np.testing.assert_array_equal(values(wing,'CellData','circulation_m2_s'),frame['lattice_circulation_m2_s'])
                np.testing.assert_array_equal(values(wing,'CellData','panel_force_N').reshape(-1,3),frame['lattice_panel_force_N'])
                np.testing.assert_array_equal(values(wake,'Lines','connectivity',int).reshape(-1,2),frame['network_edges'])
                np.testing.assert_array_equal(values(wake,'CellData','core_radius_m'),frame['network_core_radius_m'])
                self.assertIsNone(wing.find('./CellData/DataArray[@Name="Cl"]'))
            finally:
                reader.close()

    def test_old_v2_and_legacy_v1_reader_compatibility(self):
        reader=AeroReader(self.root/'free_wake')
        try:
            self.assertEqual(reader.manifest['schema'],'openfawe.aero.v2')
            expected=reader.frame(2)['panel_force']
        finally:
            reader.close()
        legacy=self.root/'legacy-schema-fixture'
        shutil.copytree(self.root/'free_wake',legacy)
        with h5py.File(legacy/'wake.h5','r+') as archive:
            archive.attrs['schema']=LEGACY_SCHEMA
        manifest=json.loads((legacy/'manifest.json').read_text(encoding='utf-8'))
        manifest.update(schema=LEGACY_SCHEMA,wake_sha256=sha256_file(legacy/'wake.h5'))
        (legacy/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        reader=AeroReader(legacy)
        try:
            np.testing.assert_array_equal(reader.frame(2)['panel_force'],expected)
        finally:
            reader.close()

    @unittest.skipUnless(importlib.util.find_spec('PySide6'),'Optional Qt environment absent')
    def test_platform_five_method_selection_actual_mesh_and_comparison(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6 import QtCore,QtWidgets
        from openfawe.platform import OpenFAWEPlatform
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        paths=[self.root/method for method in self.states]
        window=OpenFAWEPlatform(paths)
        try:
            window.show();app.processEvents()
            self.assertEqual(len(window.records),5)
            self.assertEqual(window.method_filter.count(),6)
            for index in range(window.run_list.count()):
                item=window.run_list.item(index)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                method=item.data(QtCore.Qt.ItemDataRole.UserRole).method
                if method in ('vlm','vlm_free_wake'):
                    window.run_list.setCurrentItem(item)
                    replay=window.replay
                    self.assertIsNotNone(replay)
                    self.assertIn(method_label(method),replay.windowTitle())
                    replay.timer.stop();replay.seek(2)
                    replay.loads.setChecked(True);replay.points.setChecked(True)
                    self.assertEqual(len(replay.canvas.frame['lattice_cells']),8)
                    self.assertIn('Residual =',replay.info.text())
                    self.assertIn(' m/s',replay.info.text())
                    self.assertNotIn('m²/s',replay.info.text())
                    self.assertIn('Section polars: not used',replay.info.text())
                    app.processEvents()
                    self.assertFalse(window.grab().isNull())
            self.assertEqual(len(window.comparison_curves),5)
            self.assertIn('inviscid thin lifting surfaces',window.comparison_note.text())
        finally:
            window.close();app.processEvents()


if __name__=='__main__':
    unittest.main()
