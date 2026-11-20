"""OF-0010 revision 13, S-U5: the free flight's 3D frame - in a vortex period the vortex state's wake and rotor rings, in a
table period the hook's own rings, without a hook the aircraft only (as before); and the spec's new choices."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openfawe_studio import engine, turnkey_worker  # noqa: E402
from openfawe_studio.spec import Spec  # noqa: E402


class Frame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        built = engine.build(Spec(steps=3, compute="cpu_direct", rotors=True))
        cls.state = list(engine.march(built))[-1].state            # a VSM free-wake state with rotor rings
        cls.wing = built.wing
        cls.pose = (np.eye(3), np.array([1.0, 2.0, 3.0]))

    def test_vortex_period(self):
        from openfawe.aero.wake import extended_network
        hook = SimpleNamespace(kind="vsm", main=SimpleNamespace(state=self.state), rings=None)
        arrays, wake = turnkey_worker.flight_frame(self.wing, hook, *self.pose)
        nodes, edges, gamma, ring_first = extended_network(self.state)
        self.assertTrue(wake)
        self.assertEqual(len(arrays["nodes"]), len(nodes))
        self.assertEqual(arrays["ring_first"], ring_first)
        self.assertEqual(arrays["surface_nodes"], 2*self.state.wing.node_count)          # bound and trailing-edge rows
        self.assertEqual(len(arrays["nodes"])-arrays["ring_first"], len(self.state.rings.nodes))
        self.assertEqual(len(arrays["ring_source"]), len(self.state.rings.source))       # per ring node
        self.assertEqual(len(arrays["bound_gamma"]), len(arrays["panel_edges"]))
        from openfawe_studio.runstore import wake_lines
        kept, _, ring = wake_lines(dict(arrays))
        self.assertTrue(len(kept) and ring.any())

    def test_table_period_and_no_hook(self):
        rings = self.state.rings
        hook = SimpleNamespace(kind="strip", main=None, rings=rings)
        arrays, wake = turnkey_worker.flight_frame(self.wing, hook, *self.pose)
        self.assertTrue(wake)
        self.assertEqual((len(arrays["nodes"]), arrays["ring_first"]), (len(rings.nodes), 0))
        arrays, wake = turnkey_worker.flight_frame(self.wing, None, *self.pose)
        self.assertFalse(wake)
        self.assertEqual(len(arrays["nodes"]), 0)
        np.testing.assert_array_equal(arrays["leading_edge"], self.wing.leading_edge+self.pose[1])


class Choices(unittest.TestCase):
    def test_defaults_and_refusals(self):
        s = Spec()
        self.assertEqual((s.flight_aero, s.power_system, s.blown_wing, s.keep_rows), ("tables", True, True, None))  # ST-03: E4
        Spec(motion="free_flight", flight_aero="vortex", compute="cpu_tree").validate()
        for bad in (dict(flight_aero="other"), dict(motion="free_flight", flight_aero="vortex", compute="cpu_direct"),
                    dict(keep_rows=100), dict(flight_frame_every=100, wake_frame_every=150)):    # review STR-08
            with self.assertRaises(ValueError):
                Spec(**bad).validate()
        self.assertEqual(Spec.from_dict(dict(Spec().to_dict(), power_system=True)).power_system, True)


class Thinning(unittest.TestCase):
    def test_long_records_read_thinned_and_exported_whole(self):
        import json
        import tempfile
        from openfawe_studio import runstore
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path/"frames").mkdir()
            (path/"spec.json").write_text(json.dumps(dict(spec=Spec().to_dict(), reference_area_m2=1.0, reference_chord_m=1.0)),
                                          encoding="utf-8")
            with (path/"records.jsonl").open("w", encoding="utf-8") as stream:
                for k in range(1001):
                    stream.write(json.dumps(dict(k=k, t=0.01*k))+"\n")
            old = runstore.THIN_ABOVE_BYTES, runstore.THIN_ROWS
            runstore.THIN_ABOVE_BYTES, runstore.THIN_ROWS = 100, 100
            try:
                run = runstore.RunFolder(path)
            finally:
                runstore.THIN_ABOVE_BYTES, runstore.THIN_ROWS = old
            ks = [r["k"] for r in run.records]
            self.assertEqual((run.stride, ks[0], ks[-1]), (11, 0, 1000))
            self.assertLessEqual(len(ks), 102)
            self.assertEqual(runstore.RunFolder(path, records=False).records, [])
            runstore.export_csv(run, path/"out.csv")
            self.assertEqual(len((path/"out.csv").read_text(encoding="utf-8").splitlines()), 1002)


if __name__ == "__main__":
    unittest.main()
