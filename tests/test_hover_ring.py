"""Unit checks V1-V4 and H2(e) of the hover-ring contract (``hover-ring-contract.zh.md``, revision 2; decision 0045)."""

import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openfawe.aero import hover_ring as HR
from openfawe.aero import makani_hover_strip as H
from openfawe.aero.aircraft import load_aircraft
from openfawe.aero.vortex import CoreModel
from openfawe.aero.vsm import geometry
from openfawe.rotor import makani_corrections

REPO = Path(__file__).resolve().parents[1]
CASE = REPO/"cases/qualification/m600_makani_xfoil_polars_pylons.json"
ROTORS = REPO/"cases/rotors/m600_rotors_makani.json"
TABLE = REPO/"cases/polars/makani/hover_model_sn02_wind_10_no_wake_model.json"
AIRFOILS = REPO/"cases/polars/makani"
THRUST = {"bottom": 2133.443132820832, "top": 2430.3405421791686}


def aircraft():
    if not hasattr(aircraft, "cached"):
        aircraft.cached = load_aircraft(CASE)
    return aircraft.cached


class V1StripLoads(unittest.TestCase):
    def test_section_loads_equal_an_independent_evaluation(self):
        ac = aircraft()
        sections = HR.Sections.from_aircraft(ac)
        _, chord, normal, tangent, span = geometry(ac.wing)
        surfaces = ac.wing.wings
        bound_mid = np.vstack([0.5*(np.asarray(s.bound)[:-1]+np.asarray(s.bound)[1:]) for s in surfaces])
        length = np.concatenate([np.linalg.norm(np.diff(np.asarray(s.bound), axis=0), axis=1) for s in surfaces])
        np.testing.assert_allclose(sections.points, bound_mid, rtol=0, atol=1e-15)
        for alpha_deg, beta_deg in ((90.0, 0.0), (-90.0, 15.0), (67.5, -30.0), (105.0, 7.5), (10.0, 0.0)):
            wind = HR.wind_from_makani(10.0, math.radians(alpha_deg), math.radians(beta_deg))
            rng = np.random.default_rng(abs(int(alpha_deg*10+beta_deg*100))+1)
            u = wind+rng.normal(0.0, 2.0, size=sections.points.shape)          # non-uniform inflow as with rings
            force, couple, alpha = sections.loads(u, 1.2)
            for k in range(len(chord)):
                s, t, n = span[k], tangent[k], normal[k]
                a = math.atan2(float(u[k]@n), float(u[k]@t))
                cl, cd, cm = ac.polar.polars[k].coefficients(a)
                across = u[k]-(u[k]@s)*s
                q = 0.5*1.2*float(across@across)
                d = math.cos(a)*t+math.sin(a)*n
                d = d/np.linalg.norm(d)
                lift = np.cross(d, s)
                lift = lift/np.linalg.norm(lift)
                d = np.cross(s, lift)                                          # contract revision 3: as the VSM
                f = q*chord[k]*length[k]*(cl*lift+cd*d)
                c = q*chord[k]**2*length[k]*cm*s
                scale = 1e-12*np.linalg.norm(f)+1e-9
                self.assertLessEqual(float(np.max(np.abs(force[k]-f))), scale)
                self.assertLessEqual(float(np.max(np.abs(couple[k]-c))), 1e-12*np.linalg.norm(c)+1e-9)
                self.assertAlmostEqual(float(alpha[k]), a, delta=1e-14)

    def test_same_as_the_vsm_full_incidence_loads_with_zero_circulation(self):
        from openfawe.aero.vsm import _full_incidence_loads
        ac = aircraft()
        sections = HR.Sections.from_aircraft(ac)
        _, chord, normal, tangent, span = geometry(ac.wing)
        rng = np.random.default_rng(7)
        u = HR.wind_from_makani(10.0, math.radians(80.0), math.radians(-20.0))+rng.normal(0.0, 3.0, size=sections.points.shape)
        force, couple, alpha = sections.loads(u, 1.2)
        cl, cd, cm = sections.coefficients(alpha)
        across = u-np.sum(u*span, axis=1)[:, None]*span
        n = len(chord)
        _, _, vsm_alpha, vsm_force, vsm_couple = _full_incidence_loads(np.zeros(n), u, np.zeros((n, n, 3)), np.zeros((n, 3)), ac.wing,
                                                                        tangent, normal, span, chord, cl, cd, cm,
                                                                        np.sum(across*across, axis=1), 1.2)
        np.testing.assert_allclose(alpha, vsm_alpha, rtol=0, atol=1e-14)
        self.assertLessEqual(float(np.max(np.abs(force-vsm_force))), 1e-12*float(np.max(np.abs(vsm_force)))+1e-9)
        self.assertLessEqual(float(np.max(np.abs(couple-vsm_couple))), 1e-12*float(np.max(np.abs(vsm_couple)))+1e-9)

    def test_sub_strips_carry_the_bound_length(self):
        ac = aircraft()
        one, four = HR.Sections.from_aircraft(ac), HR.Sections.from_aircraft(ac, sub_strips=4)
        self.assertEqual(len(four.points), 4*len(one.points))
        np.testing.assert_allclose(np.add.reduceat(four.length, np.arange(0, len(four.length), 4)), one.length, rtol=1e-14)
        wind = HR.wind_from_makani(10.0, math.radians(90.0), 0.0)
        f1, c1, _ = one.loads(np.broadcast_to(wind, one.points.shape), 1.2)
        f4, c4, _ = four.loads(np.broadcast_to(wind, four.points.shape), 1.2)
        np.testing.assert_allclose(f4.sum(axis=0), f1.sum(axis=0), rtol=1e-12)         # uniform inflow: same total force


class V2RingInduction(unittest.TestCase):
    def test_load_point_induction_equals_ring_wake_induced_velocity(self):
        ac = aircraft()
        sections = HR.Sections.from_aircraft(ac)
        discs = HR.discs_from_case(json.loads(ROTORS.read_text(encoding="utf-8")), THRUST)
        wind = HR.wind_from_makani(10.0, math.radians(90.0), math.radians(15.0))
        run = HR.HoverRun(sections, discs, wind)
        for _ in range(6):
            run.advance()
        ours = HR.ring_induction(run.wake, sections.points, run.settings)
        reference = run.wake.induced_velocity(sections.points, CoreModel.LAMB_OSEEN, 0.3, singularity="raise")
        self.assertLessEqual(float(np.max(np.abs(ours-reference))), 1e-12)
        self.assertEqual(run.wake.surface_core_m, 0.3)

    def test_ring_circulation_and_retention(self):
        disc = HR.discs_from_case(json.loads(ROTORS.read_text(encoding="utf-8")), THRUST)[0]
        ring = disc.ring(0.01, 1.2, 16, 0.0)
        self.assertAlmostEqual(ring.circulation, 2133.443132820832*0.01/(1.2*math.pi*1.15**2), places=12)
        self.assertAlmostEqual(ring.circulation, 4.279118089400057, places=9)
        sections = HR.Sections.from_aircraft(aircraft())
        run = HR.HoverRun(sections, (disc,), np.zeros(3), HR.RingSettings(retention_s=0.05))
        for _ in range(9):
            record = run.advance()
        self.assertEqual(record["rings"], 5)          # 0.05 s at dt 0.01: the five newest rings


class V3SingleDisc(unittest.TestCase):
    def test_disc_mean_inflow_in_still_air(self):
        """Contract revision 3 item 5: D3's procedure (six equal-area annuli x 16 azimuths offset half a step, physical
        core, singularity 'zero'), evaluated after each step's convection, mean over t_(k+1) in [0.8, 1.0] s."""
        disc = HR.PrescribedDisc("one", np.zeros(3), np.array([-1.0, 0.0, 0.0]), 1.15, 2133.443132820832)
        sections = HR.Sections.from_aircraft(aircraft())
        far = np.array([1.0e3, 0.0, 0.0])                                     # the wing is not in the slipstream's way
        run = HR.HoverRun(HR.Sections(sections.points[:1]+far, sections.panel[:1], sections.chord[:1], sections.length[:1],
                                      sections.tangent[:1], sections.normal[:1], sections.span[:1], sections.surface[:1],
                                      sections.names, sections.polar), (disc,), np.zeros(3))
        means = []
        probes = run.disc_probes()
        while run.time_s < 1.0-1e-9:
            record = run.advance(post_probes=probes)
            if run.time_s >= 0.8-1e-9:                                       # t_(k+1), after this step's convection
                means.append(HR.axial_mean(disc, record["post_probes"]["disc0"]))
        v_i = math.sqrt(2133.443132820832/(2*1.2*math.pi*1.15**2))
        ratio = float(np.mean(means))/v_i
        self.assertEqual(len(means), 21)
        self.assertGreaterEqual(ratio, 0.80, ratio)
        self.assertLessEqual(ratio, 1.00, ratio)


class StepReplay(unittest.TestCase):
    def test_one_step_from_the_primitives(self):
        """The order of one step (contract M3): shed, keep the newest K, loads with these rings, convect (RK4, physical core)."""
        from openfawe.rotor.ring_wake import RingWake, convect_rings
        from openfawe.aero.vortex import segments_velocity
        sections = HR.Sections.from_aircraft(aircraft())
        discs = HR.discs_from_case(json.loads(ROTORS.read_text(encoding="utf-8")), THRUST)
        wind = HR.wind_from_makani(10.0, math.radians(90.0), math.radians(-15.0))
        run = HR.HoverRun(sections, discs, wind, HR.RingSettings(retention_s=0.03))
        for _ in range(5):
            run.advance()
        before, t = run.wake, run.time_s
        record = run.advance()
        wake = before
        for index, disc in enumerate(discs):
            wake = wake.shed(disc.ring(0.01, 1.2, 16, t, float(-wind@disc.axis)), index)
        wake = wake.released_since(t-0.03+0.005)
        self.assertEqual(wake.count, 3*8*16)
        induced = segments_velocity(sections.points, wake.nodes, wake.edges, wake.circulation, model=CoreModel.LAMB_OSEEN,
                                    core_radius=0.3, singularity="raise", backend="numpy")
        force, couple, _ = sections.loads(wind+induced, 1.2)
        self.assertLessEqual(float(np.max(np.abs(record["force"]-force))), 1e-12*float(np.max(np.abs(force))))
        moved = convect_rings(wake, 0.01, CoreModel.LAMB_OSEEN, 0.15, ambient=lambda p: np.broadcast_to(wind, p.shape).copy(),
                              integrator="rk4", backend="numpy")
        self.assertLessEqual(float(np.max(np.abs(run.wake.nodes-moved.nodes))), 1e-12)


class V4StripReference(unittest.TestCase):
    def test_r2_equals_the_public_table_on_its_nodes(self):
        model = H.HoverStripModel.from_table_params(TABLE, AIRFOILS, use_wake_model=False, convert_2d_to_3d=False)
        alphas, betas, cfm = H.public_table_cfm(TABLE)
        a_deg, b_deg = np.rad2deg(alphas), np.rad2deg(betas)
        for alpha in (67.5, -67.5, 105.0, -105.0):
            for beta in (0.0, 7.5, -7.5, 15.0, -15.0, 30.0, -30.0):
                i, j = int(np.argmin(np.abs(a_deg-alpha))), int(np.argmin(np.abs(b_deg-beta)))
                ours = model.force_moment_coefficients(np.array([math.radians(alpha)]), np.array([math.radians(beta)]),
                                                       np.zeros((8, 1)), np.zeros(3), 10.0)[0]
                self.assertLessEqual(abs(ours[2]-cfm[i, j, 2]), 6e-3, (alpha, beta))
                self.assertLessEqual(abs(ours[4]-cfm[i, j, 4]), 6e-3, (alpha, beta))


class H2eNoDoubleCounting(unittest.TestCase):
    def test_the_module_names_no_empirical_correction(self):
        source = (REPO/"src/openfawe/aero/hover_ring.py").read_text(encoding="utf-8")
        code = "\n".join(line.split("#")[0] for line in source.splitlines())
        import ast
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | \
                {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | \
                {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names} | \
                {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        for forbidden in ("makani_corrections", "blown_wing_force", "cl_weight", "FUDGE_SCALE", "convert_2d_to_3d",
                          "sum_wake_velocity_increments", "_panel_force_moment", "HoverStripModel"):
            self.assertNotIn(forbidden, names)
        self.assertNotIn("makani_corrections", code)


    def test_the_load_path_calls_no_empirical_correction(self):
        def refuse(*args, **kwargs):
            raise AssertionError("an empirical correction was called on the H-ring-S load path")
        sections = HR.Sections.from_aircraft(aircraft())
        discs = HR.discs_from_case(json.loads(ROTORS.read_text(encoding="utf-8")), THRUST)
        with mock.patch.object(makani_corrections, "blown_wing_force", refuse), \
                mock.patch.object(makani_corrections, "blown_wing_corrections", refuse), \
                mock.patch.object(H, "sum_wake_velocity_increments", refuse), \
                mock.patch.object(H, "convert_2d_to_3d", refuse), \
                mock.patch.object(H.HoverStripModel, "force_moment_coefficients", refuse):
            run = HR.HoverRun(sections, discs, HR.wind_from_makani(10.0, math.radians(90.0), 0.0))
            for _ in range(3):
                record = run.advance()
        self.assertTrue(np.all(np.isfinite(record["force"])))


if __name__ == "__main__":
    unittest.main()
