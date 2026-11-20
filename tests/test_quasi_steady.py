"""OF-0011 revision 7 addendum 12, R6-0: the quasi-steady group of lifting surfaces (``flight.quasi_steady``)."""
import json
import unittest
from pathlib import Path

import numpy as np

from openfawe.aero.aircraft import PanelPolars, WingAssembly
from openfawe.aero.handover import TAKEOFF_LANDING_SOLVE
from openfawe.aero.vsm import geometry, solve_vsm
from openfawe.aero.wing import SectionPolar, Wing
from openfawe.flight.kinematics import Pose
from openfawe.flight.quasi_steady import QuasiSteadyGroup


def case():
    c = json.loads((Path(__file__).parent/"fixtures/vsm_assembly.json").read_text())
    wings = tuple(Wing(v["leading_edge"], v["trailing_edge"]) for v in c["components"])
    geometry_ = WingAssembly(wings, tuple(v["name"] for v in c["components"]))
    tables = tuple(SectionPolar(*np.array(v["polar"]).T) for v in c["components"])
    polar = PanelPolars(tuple(p for w, p in zip(wings, tables) for _ in range(w.panels)))
    return geometry_, polar, np.asarray(c["inflow_m_s"], dtype=float)


def pose(t=0.0):
    return Pose(t, np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))


class QuasiSteady(unittest.TestCase):
    def test_equals_the_steady_vsm(self):
        wing, polar, inflow = case()
        group = QuasiSteadyGroup("qs", inflow, density_kg_m3=1.026)
        force, moment, fields = group.extra_loads(pose(), {}, wing, polar)
        cp = geometry(wing)[0]
        direct = solve_vsm(wing, polar, np.tile(inflow, (len(cp), 1)), load_relative_ambient=np.tile(inflow, (wing.panels, 1)),
                           density=1.026, settings=TAKEOFF_LANDING_SOLVE)
        self.assertTrue(np.array_equal(force, direct.force))
        self.assertTrue(np.array_equal(moment, direct.moment(wing, reference=np.zeros(3))))
        self.assertEqual(fields["qs_wake_model"], "quasi_steady")
        self.assertIsNone(group.state)

    def test_induction_enters_the_inflow(self):
        wing, polar, inflow = case()
        extra = np.array([0.3, -0.2, 0.5])
        a = QuasiSteadyGroup("qs", inflow, density_kg_m3=1.026, induction=lambda p: np.tile(extra, (len(p), 1)))
        b = QuasiSteadyGroup("qs", inflow + extra, density_kg_m3=1.026)
        fa, ma, _ = a.extra_loads(pose(), {}, wing, polar)
        fb, mb, _ = b.extra_loads(pose(), {}, wing, polar)
        self.assertTrue(np.array_equal(fa, fb))
        self.assertTrue(np.array_equal(ma, mb))

    def test_far_field_and_order(self):
        wing, polar, inflow = case()
        group = QuasiSteadyGroup("qs", inflow, density_kg_m3=1.026)
        self.assertTrue(np.array_equal(group.induced_at(np.ones((2, 3))), np.zeros((2, 3))))
        group.extra_loads(pose(), {}, wing, polar)
        chord = float(np.max(np.linalg.norm(wing.trailing_edge - wing.leading_edge, axis=1)))
        far = group.induced_at(np.array([[0.0, 0.0, 1000.0*chord], [0.0, 1000.0*chord, 0.0]]))
        self.assertLess(float(np.abs(far).max()), 1e-6*float(np.linalg.norm(inflow)))
        near = group.induced_at(np.array([[0.0, 0.0, 0.5*chord]]))
        self.assertGreater(float(np.abs(near).max()), 0.0)
        with self.assertRaises(RuntimeError):
            group.extra_loads(pose(), {}, wing, polar)              # the same time again


if __name__ == "__main__":
    unittest.main()
