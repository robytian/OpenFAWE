"""Plan 3.7 part 2, criteria G5 and G7 (full-cycle contract revision 4): the perch contact and the perched wing with
the full controller.

G5(a), (b): the contact force and the panel geometry against hand computations from contactor.cc and
ground_station_util.cc. G5(c): the wing at Makani's initial pose (6 cm above the perched pose) with gravity and the
contactors only. G7: ground station, variable tether, contactors, wing (gravity only) and part 1's full controller
for TurnKey's first 11 s; needs the library (OPENFAWE_MAKANI_FULL_BUILD; a skip fails the judged run).

Both runs measure how far the wing moves on the perch: the centre of mass in the platform frame and the attitude
relative to the platform, from the state at 2.0 s (after the drop onto the perch) to the end.
"""
import math
import os
from pathlib import Path
import unittest

import numpy as np

from openfawe.ground import gs02 as G
from openfawe.ground import perch as P

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT/"cases"/"ground"/"makani_gs02_parker_ranch.json"
BUILD = os.environ.get("OPENFAWE_MAKANI_FULL_BUILD")
SETTLE_S = 2.0


def on_perch(records, gs_params, reference_time_s=SETTLE_S):
    """Largest displacement of the centre of mass in the platform frame and largest attitude change relative to the
    platform (deg), after ``reference_time_s``; records carry position_cm (world), rotation (body to world) and
    platform_azi."""
    from openfawe.flight.makani_interface import G_TO_WORLD

    def rel(r):
        dcm = G.angle_to_dcm(r["platform_azi"], 0.0, 0.0, "zyx")         # NED -> platform
        return dcm@(G_TO_WORLD@r["position_cm"]), dcm@G_TO_WORLD@r["rotation"]
    ref = next(r for r in records if r["t"] >= reference_time_s - 1e-9)
    p0, R0 = rel(ref)
    disp, ang = 0.0, 0.0
    for r in records:
        if r["t"] < ref["t"]:
            continue
        p, R = rel(r)
        disp = max(disp, float(np.linalg.norm(p - p0)))
        ang = max(ang, math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R0.T@R) - 1.0)/2.0)))))
    return disp, ang


class G5Contact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = G.load_gs02_params(PARAMS)
        cls.panel = cls.params.sim["panel"]
        cls.contactors = cls.params.raw["perch_contactors"]

    # G5(a): spring along the penetration, damping only while approaching, friction viscous below 1 m/s and Coulomb
    # above (contactor.cc:60-97), written out.
    def test_a_contact_force(self):
        c = self.contactors[0]
        k, d, mu = c["spring_const"], c["damping_coeff"], c["friction_coeff"]
        collision = np.array([0.3, -0.2, 1.0])
        normal = np.array([0.0, 0.6, 0.8])
        tangent = np.array([1.0, 0.0, 0.0])
        for depth, vn, vt in ((0.01, 0.2, 0.3), (0.01, -0.2, 0.3), (0.003, 0.0, 2.5), (0.02, 0.5, -1.7), (0.0, 0.1, 0.1)):
            pos = collision + depth*normal              # offset points from the surface into the object
            vel = vn*normal + vt*tangent + np.array([0.1, 0.2, 0.3])
            cvel = np.array([0.1, 0.2, 0.3])            # the surface point's own velocity
            spring = -k*depth*normal
            damping = -d*vn*normal if (vn*depth > 0) else np.zeros(3)
            slide = vt*tangent
            if abs(vt) > 1.0:
                slide = slide/abs(vt)
            friction = -mu*k*depth*slide
            got = P.contact_force(c, pos, vel, collision, cvel)
            self.assertLessEqual(np.abs(got - (spring + damping + friction)).max(), 1e-12*max(1.0, k*depth*mu), (depth, vn, vt))

    # G5(b): collision points of the panel surface against points worked out by hand.
    def test_b_panel_geometry(self):
        gs = G.GroundStationV2(self.params)
        gs.init(0.7, -200.0)
        frames = gs.frames()
        port, star = self.panel["port"], self.panel["starboard"]
        cp, cs = np.array(port["center_panel"]), np.array(star["center_panel"])
        top = port["z_extents_p"][0]

        def panel_point(xy, z_p):
            """A panel-frame point with the given panel x, y and platform-frame z (solving for the panel z)."""
            a = frames.panel.transform_to(frames.platform, "position", [xy[0], xy[1], 0.0])
            b = frames.panel.transform_to(frames.platform, "position", [xy[0], xy[1], 1.0])
            zeta = (z_p - a[2])/(b[2] - a[2])
            return np.array([xy[0], xy[1], zeta])
        # Inside the port disk only, well below the top: the radial projection onto the port circle.
        u = np.array([math.cos(1.2), math.sin(1.2)])
        xy = cp + 3.5*u
        self.assertGreaterEqual(np.linalg.norm(xy - cs), star["radius"])
        point = panel_point(xy, top + 1.5)
        hit, where = P.contact_perch_panel(point, frames.platform, frames.panel, self.panel)
        self.assertTrue(hit)
        self.assertLessEqual(np.abs(where - np.array([*(cp + 4.0*u), point[2]])).max(), 1e-12)
        # Inside the starboard disk only.
        u = np.array([math.cos(-1.3), math.sin(-1.3)])
        xy = cs + 3.7*u
        self.assertGreaterEqual(np.linalg.norm(xy - cp), port["radius"])
        point = panel_point(xy, top + 1.0)
        hit, where = P.contact_perch_panel(point, frames.platform, frames.panel, self.panel)
        self.assertTrue(hit)
        self.assertLessEqual(np.abs(where - np.array([*(cs + 4.0*u), point[2]])).max(), 1e-12)
        # Inside both disks: the intersection of the two circles on the contactor's side of the centre line.
        d = cs - cp
        dist = np.linalg.norm(d)
        e = d/dist
        perp = np.array([-e[1], e[0]])
        a = dist/2.0                                           # equal radii
        h = math.sqrt(port["radius"]**2 - a*a)
        for side in (1.0, -1.0):
            xy = cp + a*e + side*0.8*h*perp
            point = panel_point(xy, top + 1.2)
            hit, where = P.contact_perch_panel(point, frames.platform, frames.panel, self.panel)
            self.assertTrue(hit)
            self.assertLessEqual(np.abs(where[:2] - (cp + a*e + side*h*perp)).max(), 1e-12)
        # Deep inside a disk, 1 cm below the top plane: the top plane is nearer.
        u = np.array([math.cos(0.3), math.sin(0.3)])
        xy = cp + 2.0*u
        point = panel_point(xy, top + 0.01)
        hit, where = P.contact_perch_panel(point, frames.platform, frames.panel, self.panel)
        self.assertTrue(hit)
        p_platform = frames.panel.transform_to(frames.platform, "position", point)
        expected = frames.platform.transform_to(frames.panel, "position", [p_platform[0], p_platform[1], top])
        self.assertLessEqual(np.abs(where - expected).max(), 1e-12)
        # Above the top plane, and outside both disks: no contact.
        self.assertFalse(P.contact_perch_panel(panel_point(xy, top - 0.01), frames.platform, frames.panel, self.panel)[0])
        far = cp + 5.0*np.array([math.cos(2.5), math.sin(2.5)])
        self.assertFalse(P.contact_perch_panel(panel_point(far, top + 1.0), frames.platform, frames.panel, self.panel)[0])
        # Too close to a cylinder axis: the source stops (LOG(FATAL)).
        with self.assertRaises(RuntimeError):
            P.contact_perch_panel(panel_point(cp + 0.3*u, top + 1.0), frames.platform, frames.panel, self.panel)

    # G5(c): gravity and the contactors only (no tether), 10 s after the drop: displacement on the perch <= 5 mm,
    # attitude <= 0.1 deg, contact force sum within 0.5% of the weight.
    def test_c_static_on_the_perch(self):
        system = P.PerchedSystem(self.params)
        system.tether_on = False
        system.gs.init(self.params.sim["initial_platform_azi"], -228.5)
        state = system.initial_wing_state()
        nodes = speeds = np.zeros((system.line.n, 3))
        records, h = [], 1e-3
        for k in range(int(round((SETTLE_S + 10.0)/h)) + 1):
            if k % 10 == 0:
                records.append(dict(t=k*h, position_cm=state.position_cm_m.copy(), rotation=state.rotation.copy(),
                                    platform_azi=system.gs.platform_azi))
            if k < int(round((SETTLE_S + 10.0)/h)):
                state, nodes, speeds = system.step(state, nodes, speeds, h)
        disp, ang = on_perch(records, self.params)
        force, _, _ = system.contact_loads(state, system.gs.x)
        weight = system.body.mass_kg*self.params.raw["phys"]["g"]
        self.__class__.static = dict(displacement_m=disp, attitude_deg=ang, force_N=force.tolist(), weight_N=weight)
        self.assertLessEqual(abs(np.linalg.norm(force) - weight), 0.005*weight)
        self.assertLessEqual(disp, 5e-3)
        self.assertLessEqual(ang, 0.1)


@unittest.skipUnless(BUILD, "OPENFAWE_MAKANI_FULL_BUILD not set")
class G7PerchedWithController(unittest.TestCase):
    # G7: no assert for 11 s; Perched after the switch moves to the middle at 10.1 s; the ground station stays in
    # Reel with a zero drum command and the drum at rest; the wing stays on the perch (<= 5 mm from the 2 s state).
    def test_perched_turnkey(self):
        from openfawe.flight.makani_full import load_full_api
        api = load_full_api(BUILD)
        params = G.load_gs02_params(PARAMS)
        result = P.run_perched_turnkey(params, api, duration_s=11.0)
        records = result["records"]
        modes = [r["flight_mode"] for r in records]
        first_perched = next((r["t"] for r in records if r["flight_mode"] == api.kFlightModePerched), None)
        disp, ang = on_perch(records, params)
        self.__class__.summary = dict(first_perched_s=first_perched, displacement_m=disp, attitude_deg=ang,
                                      platform_azimuth_change_rad=records[-1]["platform_azi"] - records[0]["platform_azi"],
                                      wall_s=result["wall_s"], init=result["init"])
        self.assertEqual(modes[-1], api.kFlightModePerched)
        self.assertIsNotNone(first_perched)
        self.assertGreaterEqual(first_perched, 10.1)
        self.assertTrue(all(r["gs_mode"] == G.MODE_REEL for r in records))
        self.assertTrue(all(r["drum_vel_cmd"] == 0.0 for r in records))
        self.assertLessEqual(max(abs(r["drum_omega"]) for r in records), 1e-12)
        self.assertLessEqual(disp, 5e-3)


class ContactBranch(unittest.TestCase):
    """OF-0011 revision 7 addendum 4 supplement 4 (H7-R6): ``contact_branch`` against ``contact_perch_panel``, and the
    event signature with ``"branches"`` against ``"all"``."""

    @classmethod
    def setUpClass(cls):
        cls.params = G.load_gs02_params(PARAMS)
        cls.panel = cls.params.sim["panel"]
        gs = G.GroundStationV2(cls.params)
        gs.init(0.7, -200.0)
        cls.frames = gs.frames()
        port, star = cls.panel["port"], cls.panel["starboard"]
        cls.cp, cls.cs = np.array(port["center_panel"]), np.array(star["center_panel"])

    def panel_point(self, xy, z_p):
        frames = self.frames
        a = frames.panel.transform_to(frames.platform, "position", [xy[0], xy[1], 0.0])
        b = frames.panel.transform_to(frames.platform, "position", [xy[0], xy[1], 1.0])
        return np.array([xy[0], xy[1], (z_p - a[2])/(b[2] - a[2])])

    def both(self, point):
        f = self.frames
        try:
            hit, where = P.contact_perch_panel(point, f.platform, f.panel, self.panel)
        except RuntimeError:
            hit = where = "raised"
        try:
            branch = P.contact_branch(point, f.platform, f.panel, self.panel)
        except RuntimeError:
            branch = "raised"
        return hit, where, branch

    def test_agrees_with_contact_perch_panel(self):
        port, star = self.panel["port"], self.panel["starboard"]
        tops = sorted([port["z_extents_p"][0], star["z_extents_p"][0]])
        heights = [tops[0] - 0.05, tops[0] + 0.004, 0.5*(tops[0] + tops[1]) if tops[1] > tops[0] else tops[0] + 0.3,
                   tops[1] + 0.02, tops[1] + 1.2]
        middle = 0.5*(self.cp + self.cs)
        span = 2.0*max(port["radius"], star["radius"]) + np.linalg.norm(self.cs - self.cp)
        counted = dict(hit=0, miss=0, raised=0, top=0, cylinder=0)
        for z_p in heights:
            for gx in np.linspace(-0.5*span, 0.5*span, 23):
                for gy in np.linspace(-0.5*span, 0.5*span, 23):
                    point = self.panel_point(middle + np.array([gx, gy]), z_p)
                    hit, where, branch = self.both(point)
                    if hit == "raised":
                        self.assertEqual(branch, "raised")
                        counted["raised"] += 1
                        continue
                    self.assertEqual(bool(hit), len(branch) == 5, (point, branch))
                    if not hit:
                        counted["miss"] += 1
                        continue
                    counted["hit"] += 1
                    surface = "cylinder" if where[2] == point[2] else "top"
                    self.assertEqual(branch[3], surface, (point, branch))
                    counted[surface] += 1
        for key in ("hit", "miss", "top", "cylinder"):
            self.assertGreater(counted[key], 0, counted)

    def test_branch_changes_into_both_circles(self):
        # Along the port circle, 1 cm inside it and deep in the panel (the cylinder nearest), through the corner where
        # the starboard circle crosses it: inside the port disk only the collision point is the projection on the port
        # circle, inside both it is the circles' intersection. The point jumps there (H7-0b, k = 1 622) while the
        # contact does not change; the branch changes with it.
        d = self.cs - self.cp
        dist = np.linalg.norm(d)
        e, radius = d/dist, self.panel["port"]["radius"]
        perp = np.array([-e[1], e[0]])
        a = 0.5*dist                                            # equal radii
        corner = self.cp + a*e + np.sqrt(radius*radius - a*a)*perp
        theta0 = math.atan2(corner[1] - self.cp[1], corner[0] - self.cp[0])
        z_p = self.panel["port"]["z_extents_p"][0] + 2.5
        seen = []
        for theta in np.linspace(theta0 - 0.05, theta0 + 0.05, 2001):
            xy = self.cp + (radius - 0.01)*np.array([math.cos(theta), math.sin(theta)])
            hit, where, branch = self.both(self.panel_point(xy, z_p))
            self.assertEqual(len(branch), 5)
            self.assertEqual(branch[3], "cylinder")
            seen.append((branch[:3], where))
        kinds = [b for b, _ in seen]
        self.assertIn(("two", (True, False), 0.0), kinds)
        self.assertIn(("two", (True, True), 1.0), kinds)
        changes = [j for j in range(1, len(seen)) if seen[j][0] != seen[j - 1][0]]
        self.assertEqual(len(changes), 1)
        moves = [np.linalg.norm(seen[j][1] - seen[j - 1][1]) for j in range(1, len(seen)) if j not in changes]
        jump = np.linalg.norm(seen[changes[0]][1] - seen[changes[0] - 1][1])
        self.assertGreater(jump, 10.0*max(moves))

    def test_held_point_is_the_own_point(self):
        # Supplement 5 (H7-R8): with the contactor's own branch the held collision point is contact_perch_panel's, bit
        # for bit, on the same grid as above.
        port, star = self.panel["port"], self.panel["starboard"]
        tops = sorted([port["z_extents_p"][0], star["z_extents_p"][0]])
        middle = 0.5*(self.cp + self.cs)
        span = 2.0*max(port["radius"], star["radius"]) + np.linalg.norm(self.cs - self.cp)
        f, compared = self.frames, 0
        for z_p in (tops[0] + 0.004, tops[1] + 0.02, tops[1] + 1.2, tops[1] + 2.5):
            for gx in np.linspace(-0.5*span, 0.5*span, 23):
                for gy in np.linspace(-0.5*span, 0.5*span, 23):
                    point = self.panel_point(middle + np.array([gx, gy]), z_p)
                    hit, where, branch = self.both(point)
                    if hit == "raised" or not hit:
                        continue
                    held = P.contact_point_held(point, f.platform, f.panel, self.panel, branch)
                    np.testing.assert_array_equal(held, where)
                    compared += 1
        self.assertGreater(compared, 100)
        self.assertIsNone(P.contact_point_held(self.panel_point(middle, tops[0] - 0.5), f.platform, f.panel, self.panel,
                                               ("above",)))

    def test_signature_branches_against_all(self):
        from openfawe.ground.turnkey import FlyingSystem
        init, system = P.perched_initialisation(self.params)

        class Probe:
            event_signature = FlyingSystem.event_signature
            first_node = FlyingSystem.first_node

            def __init__(self, base):
                self.base, self.held_start, self.floating = base, None, None

            def __getattr__(self, name):
                return getattr(self.base, name)

        probe = Probe(system)
        state, nodes, speeds = init["wing_state"], init["nodes"], init["node_velocities"]
        import dataclasses
        lowered = dataclasses.replace(state, position_cm_m=state.position_cm_m - np.array([0.0, 0.0, 0.25]))
        hits = []
        for s in (state, lowered):
            probe.event_kinds = "all"
            sig_all = probe.event_signature(s, nodes, speeds)
            probe.event_kinds = "branches"
            sig_branches = probe.event_signature(s, nodes, speeds)
            probe.event_kinds = "first"
            self.assertEqual(probe.event_signature(s, nodes, speeds), sig_all[:1])
            self.assertEqual(sig_branches[:2], sig_all[:2])
            self.assertEqual(tuple(len(b) == 5 for b in sig_branches[2]), sig_all[2])
            self.assertEqual(len(sig_branches[2]), len(system.contactors))
            hits.append(any(sig_all[2]))
        self.assertEqual(hits, [False, True])           # the initial pose is above the perch; lowered, it touches


    def test_depth_flag(self):
        # Supplement 6 (H7-R10): with depth_m the signature also says whether a contactor is at least that far into the
        # panel; lowering the wing past the first touch by 3e-9 m it is not yet, by 3e-8 m it is.
        import dataclasses
        init, system = P.perched_initialisation(self.params)
        state = init["wing_state"]

        def lowered(dz):
            return dataclasses.replace(state, position_cm_m=state.position_cm_m - np.array([0.0, 0.0, dz]))

        def touching(dz):
            return any(len(b) == 5 for b in system.contact_branches(lowered(dz), system.gs.x))
        lo, hi = 0.0, 0.5
        self.assertFalse(touching(lo))
        self.assertTrue(touching(hi))
        for _ in range(80):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if not touching(mid) else (lo, mid)
        for extra, expected in ((3e-9, False), (3e-8, True)):
            pairs = system.contact_branches(lowered(hi + extra), system.gs.x, depth_m=1e-8)
            plain = system.contact_branches(lowered(hi + extra), system.gs.x)
            self.assertEqual(tuple(b for b, _ in pairs), plain)
            self.assertEqual(any(deep for _, deep in pairs), expected, extra)
            self.assertTrue(all(len(b) == 5 for b, deep in pairs if deep))


if __name__ == "__main__":
    unittest.main()
