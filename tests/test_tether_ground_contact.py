"""Tether ground contact (full-cycle contract revision 7, option 甲): Makani's contactor template on the active and the
staging nodes of the variable-length tether, spring only (the contactor sits in the ground frame, so its velocity is zero
and CalcContactForce's damping and friction vanish). Off by default; with the option off nothing changes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openfawe.ground import gs02 as G
from openfawe.ground import perch as P
from openfawe.tether.lumped_line import TetherProperties, TetherSettings
from openfawe.tether.variable_line import GroundContact, StagingSettings, VariableLengthTether, makani_ground_contact

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT/"cases"/"ground"/"makani_gs02_parker_ranch.json"


def m600_line(ground=None):
    params = G.load_gs02_params(PARAMS)
    raw = params.raw
    t, ts = raw["tether"], raw["tether_sim"]
    props = TetherProperties(t["length"], t["linear_density"], t["outer_diameter"], t["tensile_stiffness"],
                             t["bending_stiffness"], t["section_drag_coeff"])
    settings = TetherSettings(num_nodes=ts["num_nodes"], longitudinal_damping_ratio=ts["longitudinal_damping_ratio_active"],
                              bending_damping_ratio=ts["bending_damping_ratio_active"],
                              air_density_kg_m3=raw["phys_sim"]["air_density"], gravity_m_s2=raw["phys"]["g"])
    staging = StagingSettings(ts["stiff_len_lim"], ts["longitudinal_damping_ratio_staged"])
    if ground == "makani":
        ground = makani_ground_contact(props, settings, -6.122)
    return VariableLengthTether(props, settings, staging, ground=ground), props, settings, raw


def chain(line, free_length, depth):
    """Nodes on a straight line from the anchor at height h0 to an end point, pushed down so that the middle ones are
    ``depth`` metres below z = -6.122 (the ground of the g frame in the z-up frame)."""
    n = line.n
    start, end = np.array([0.0, 0.0, -5.0]), np.array([free_length*0.9, 0.0, -5.0])
    x = np.linspace(0.0, 1.0, n+2)[1:-1]
    nodes = start + (end-start)*x[:, None]
    nodes[:, 2] = -6.122 - depth*np.sin(np.pi*x)
    return nodes, np.zeros_like(nodes), start, np.zeros(3), end, np.zeros(3)


class GroundContactTests(unittest.TestCase):
    def test_template_spring(self):
        line, props, settings, raw = m600_line("makani")
        m0 = props.linear_density_kg_m*props.length_m/settings.num_nodes
        self.assertEqual(line.ground.spring_const_N_m, m0*raw["phys"]["g"]/0.01)
        self.assertEqual(line.ground.ground_height_m, -6.122)
        # a template node at rest is held at 1 cm (tether_sim.py: ground_penetration = 0.01)
        f = line.ground_forces(np.array([[0.0, 0.0, -6.132]]))
        self.assertAlmostEqual(f[0, 2], m0*raw["phys"]["g"], places=9)
        self.assertEqual(f[0, 0], 0.0)
        self.assertEqual(f[0, 1], 0.0)
        self.assertTrue(np.array_equal(line.ground_forces(np.array([[3.0, 2.0, -6.0]])), np.zeros((1, 3))))
        with self.assertRaises(ValueError):
            GroundContact(-6.122, 0.0)
        with self.assertRaises(ValueError):
            VariableLengthTether(props, settings, ground="ground")

    def test_off_and_above_ground_unchanged(self):
        """No option: the default; and with the option but every node above the plane: the same node forces."""
        plain, *_ = m600_line()
        contact, *_ = m600_line("makani")
        for free_length in (60.0, 200.0, 425.8):
            nodes, speeds, s, sv, e, ev = chain(plain, free_length, -3.0)          # 3 m above the ground
            a = plain.forces(free_length, nodes, speeds, s, sv, e, ev)
            b = contact.forces(free_length, nodes, speeds, s, sv, e, ev)
            self.assertTrue(np.array_equal(a["node"], b["node"]), free_length)
            self.assertNotIn("ground_force", a)
            self.assertTrue(np.array_equal(b["ground_force"], np.zeros_like(nodes)))

    def test_active_and_staging_nodes_only(self):
        """Below the plane: the node force grows by k * depth on z exactly for the active and the staging node."""
        plain, *_ = m600_line()
        contact, *_ = m600_line("makani")
        free_length = 200.0
        nodes, speeds, s, sv, e, ev = chain(plain, free_length, 0.5)
        a = plain.forces(free_length, nodes, speeds, s, sv, e, ev)
        b = contact.forces(free_length, nodes, speeds, s, sv, e, ev)
        idx = a["indices"]
        self.assertGreater(idx.start, 1)
        expected = contact.ground_forces(nodes)
        first = idx.start - 1
        diff = b["node"]-a["node"]
        self.assertTrue(np.allclose(diff[first:], expected[first:], rtol=1e-12, atol=1e-9))
        self.assertTrue(np.array_equal(diff[:first], np.zeros_like(diff[:first])))
        self.assertGreater(float(np.sum(expected[first:, 2])), 0.0)
        self.assertTrue(np.array_equal(b["ground_force"][:first], np.zeros_like(b["ground_force"][:first])))

    def test_spring_equals_makani_value_and_contact_force(self):
        """The spring constant equals the value Makani's config evaluates (cases file), and the force equals the
        transcribed Contactor::CalcContactForce (ground.perch.contact_force) with the contactor's zero velocity."""
        line, props, settings, raw = m600_line("makani")
        self.assertEqual(line.ground.spring_const_N_m, raw["tether_sim"]["ground_contactor_template"]["spring_const"])
        from openfawe.flight.makani_interface import G_TO_WORLD
        template = raw["tether_sim"]["ground_contactor_template"]
        for depth in (0.004, 0.01, 0.37):
            pos_g = np.array([12.0, -3.0, 6.122+depth])
            f_g = P.contact_force(template, pos_g, np.zeros(3), np.array([12.0, -3.0, 6.122]), np.zeros(3))
            mine = line.ground_forces((G_TO_WORLD@pos_g)[None, :])[0]
            self.assertTrue(np.allclose(G_TO_WORLD@f_g, mine, rtol=1e-13, atol=0.0), depth)

    def test_independent_of_node_velocities_and_no_active_node(self):
        contact, *_ = m600_line("makani")
        free_length = 200.0
        nodes, speeds, s, sv, e, ev = chain(contact, free_length, 0.5)
        rng = np.random.default_rng(3)
        moving = rng.normal(size=speeds.shape)
        a = contact.forces(free_length, nodes, speeds, s, sv, e, ev)
        b = contact.forces(free_length, nodes, moving, s, sv, e, ev)
        self.assertTrue(np.array_equal(a["ground_force"], b["ground_force"]))
        short = 5.0                                       # below L0/2 + stiff_len_lim: no active node, staging node n-1
        nodes, speeds, s, sv, e, ev = chain(contact, short, 0.2)
        c = contact.forces(short, nodes, speeds, s, sv, e, ev)
        self.assertFalse(c["indices"].active)
        expected = np.zeros_like(nodes)
        expected[-1] = contact.ground_forces(nodes[-1:])[0]
        self.assertTrue(np.array_equal(c["ground_force"], expected))
        self.assertGreater(expected[-1, 2], 0.0)

    def test_p6_prime_through_the_records_loader(self):
        """P6' of turnkey_checks.py reads the contact fields through flight_records (review GC-01)."""
        path = ROOT/"docs"/"tasks"/"OF-0008-g5control20261006"/"turnkey_checks.py"
        spec = importlib.util.spec_from_file_location("turnkey_checks_for_test", path)
        tc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tc)
        names = {0: "Perched", 1: "HoverTransOut"}
        rows = []
        for k, (mode, low, depth) in enumerate([(0, 1.0, 0.0), (1, -0.02, 0.02), (1, -0.04, 0.04), (1, 0.5, 0.0)]):
            rows.append(dict(k=k, t=0.01*k, flight_mode=mode, controller_flight_mode=mode, origin_height_m=50.0, min_node_height_m=low,
                             slack_segments=0, Xg=[0.0, 0.0, -50.0], wing_pos_g_cmd=[1.0, 0.0, -50.0],
                             ground_contact_nodes=int(depth > 0), ground_contact_nodes_active=int(depth > 0), max_penetration_m=depth,
                             max_penetration_active_m=depth, max_penetration_staging_m=0.0, deepest_node_index=7,
                             deepest_node_xy_m=[100.0, 2.0], ground_force_N=1000.0*depth))
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "x-records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            records = tc.flight_records(Path(folder), "x")
        result = tc.p6_prime(records, names)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["contact_by_mode"]["HoverTransOut"]["periods_in_contact"], 2)
        self.assertEqual(result["contact_by_mode"]["HoverTransOut"]["deepest"]["node"], 7)
        self.assertFalse(result["p6_as_written"]["passed"])
        records[2]["min_node_height_m"] = -0.06
        self.assertFalse(tc.p6_prime(records, names)["passed"])
        for r in records:
            r.pop("ground_contact_nodes")
        self.assertFalse(tc.p6_prime(records, names)["passed"])

    def test_perched_system_option(self):
        params = G.load_gs02_params(PARAMS)
        with self.assertRaises(ValueError):
            P.PerchedSystem(params, tether_ground_contact=True)
        system = P.PerchedSystem(params, tether_ground_contact=True, ground_z_g_m=6.122)
        self.assertEqual(system.line.ground.ground_height_m, -6.122)
        self.assertIsNone(P.PerchedSystem(params).line.ground)
        with self.assertRaises(ValueError):
            P.PerchedSystem(params, ground_z_g_m=6.122)


if __name__ == "__main__":
    unittest.main()
