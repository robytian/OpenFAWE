"""M08-M10: joint disconnected surfaces, polar mapping and source comparison."""
from pathlib import Path
import json
import shutil
from uuid import uuid4
import unittest
import numpy as np

from openfawe.aero.aircraft import WingAssembly,PanelPolars,load_aircraft
from openfawe.aero.wing import Wing,SectionPolar
from openfawe.aero.vsm import solve_vsm
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import wake_network
from openfawe.aero_demo import run_demo
from openfawe.aero_archive import AeroReader
from openfawe.aero_vtk import frame_network


def source_case():
    c=json.loads((Path(__file__).parent/'fixtures/vsm_assembly.json').read_text())
    wings=tuple(Wing(v['leading_edge'],v['trailing_edge']) for v in c['components'])
    geometry=WingAssembly(wings,tuple(v['name'] for v in c['components']))
    tables=tuple(SectionPolar(*np.array(v['polar']).T) for v in c['components'])
    polar=PanelPolars(tuple(p for w,p in zip(wings,tables) for _ in range(w.panels)))
    return geometry,polar,c


class AircraftTests(unittest.TestCase):
    def test_m08_joint_vsm_original_source(self):
        wing,polar,c=source_case();result=solve_vsm(wing,polar,c['inflow_m_s'])
        np.testing.assert_allclose(result.circulation,c['circulation_m2_s'],rtol=5e-5,atol=5e-5)
        np.testing.assert_allclose(result.panel_force,c['panel_force_N'],rtol=5e-5,atol=5e-5)
        np.testing.assert_allclose(result.moment(wing),c['moment_Nm'],rtol=5e-5,atol=5e-5)
        self.assertEqual(wing.panels,14);self.assertEqual(wing.node_count,16)
        self.assertNotIn([8,9],wing.panel_edges.tolist())

    def test_m09_single_surface_assembly_preserves_each_backend(self):
        w=Wing.ellipse(6);p=SectionPolar.linear();assembly=WingAssembly((w,),('wing',))
        tables=PanelPolars((p,)*6)
        for method in ('vsm','free_wake'):
            b=AeroBackend(method);a=b.start(w,p,[5,0,.5]);c=b.start(assembly,tables,[5,0,.5])
            a=b.advance(a,p,[5,0,.5]);c=b.advance(c,tables,[5,0,.5])
            np.testing.assert_allclose(a.solution.force,c.solution.force,rtol=0,atol=1e-10)
            np.testing.assert_allclose(a.free_rows,c.free_rows,rtol=0,atol=1e-10)

    def test_m09_mutual_induction_and_far_separation(self):
        wing,polar,c=source_case();wind=c['inflow_m_s'];w=wing.wings[0]
        p=SectionPolar(*np.array(c['components'][0]['polar']).T)
        far=WingAssembly((w,w.transformed(np.eye(3),[0,1000,0])),('a','b'))
        tables=PanelPolars((p,)*(2*w.panels))
        for method in ('vsm','free_wake'):
            b=AeroBackend(method);solo=b.start(w,p,wind);separate=b.start(far,tables,wind)
            np.testing.assert_allclose(separate.solution.force,2*solo.solution.force,rtol=1e-5,atol=1e-8)
            joint=b.start(wing,polar,wind)
            self.assertGreater(np.linalg.norm(joint.solution.circulation[:w.panels]-solo.solution.circulation),1e-5)

    def test_m09_assembly_wake_graph_has_no_connector(self):
        wing,polar,c=source_case();b=AeroBackend();state=b.start(wing,polar,c['inflow_m_s'])
        for _ in range(3):state=b.advance(state,polar,c['inflow_m_s'])
        nodes,edges,gamma=wake_network(state);balance=np.zeros(len(nodes))
        np.add.at(balance,edges[:,0],gamma);np.add.at(balance,edges[:,1],-gamma)
        np.testing.assert_allclose(balance,0,atol=1e-12)
        groups=np.tile(np.r_[np.zeros(9),np.ones(7)],5)
        self.assertTrue(np.all(groups[edges[:,0]]==groups[edges[:,1]]))
        self.assertEqual(state.free_rows.shape,(3,16,3))

    def test_m10_input_mapping_roundtrip_and_replay(self):
        case=Path(__file__).resolve().parents[1]/'cases/aircraft/wing_tail_reference.json'
        definition=load_aircraft(case);self.assertEqual(definition.wing.panels,14)
        alpha=np.zeros(14)
        cl=definition.polar.coefficients(alpha)[0]
        np.testing.assert_allclose(cl[:8],0);np.testing.assert_allclose(cl[8:],.1)
        parent=Path(__file__).resolve().parents[1]/'local/aircraft-tests';parent.mkdir(parents=True,exist_ok=True)
        root=parent/uuid4().hex;root.mkdir()
        try:
            for method in ('vsm','free_wake'):
                run_demo(root/method,aircraft_path=case,aero_method=method,dt_s=.1,duration_s=.2,export_vtk=True)
                reader=AeroReader(root/method)
                try:
                    self.assertEqual(reader.case['aircraft_input_sha256'],definition.source_sha256)
                    frame=reader.frame(2);self.assertEqual(len(frame['panel_edges']),14)
                    self.assertEqual(len(frame['leading_edge']),16)
                    nodes,edges,gamma=frame_network(frame);self.assertTrue(np.isfinite(nodes).all())
                finally:reader.close()
            raw=json.loads(definition.raw_json)
            for key,value in [('length_unit','ft'),('unexpected',1),('surfaces',[])]:
                broken=dict(raw);broken[key]=value
                (root/'bad.json').write_text(json.dumps(broken),encoding='utf-8')
                with self.assertRaises(ValueError):load_aircraft(root/'bad.json')
            (root/'bad.json').write_text('{"name":"a","name":"b"}',encoding='utf-8')
            with self.assertRaises(ValueError):load_aircraft(root/'bad.json')
        finally:
            if root.resolve().parent!=parent.resolve():raise ValueError('invalid cleanup')
            shutil.rmtree(root.resolve())

    def test_m10_invalid_topology_names_and_polars(self):
        w=Wing.ellipse(4);p=SectionPolar.linear()
        with self.assertRaises(ValueError):WingAssembly((w,w),('same','same'))
        with self.assertRaises(ValueError):PanelPolars((p,)).coefficients([0,0])


if __name__=='__main__':unittest.main()
