"""Plan 3.7 part 2, criteria G1-G3 and G6 (full-cycle contract revision 4): the GS02 ground station transcription.

Every expected value is written here from Makani's definitions (cited per test), not taken from the module: the free
length and anchor by the piecewise formulas, the filters by an independent derivation, the mode machine and the
HighTension switching logic by a small replica written from the source's transition rules. G3(b) and G6 need
Makani's own estimator and the tether/bridle models: G3(b) runs only with the full controller library
(OPENFAWE_MAKANI_FULL_BUILD; a skip fails the judged run).
"""
import math
import os
from pathlib import Path
import unittest

import numpy as np

from openfawe.ground import gs02 as G

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT/"cases"/"ground"/"makani_gs02_parker_ranch.json"
BUILD = os.environ.get("OPENFAWE_MAKANI_FULL_BUILD")
PI = math.pi


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def gauss_legendre(f, a, b, n=200):
    x, w = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5*(a + b), 0.5*(b - a)
    return half*float(np.sum(w*np.array([f(mid + half*xi) for xi in x])))


def rk4(gs, dt, steps, on_period=None, period=0.01):
    """Integrate the ground station's continuous states with fixed-step RK4; ``on_period(t)`` before each period."""
    t, per = 0.0, int(round(period/dt))
    for k in range(steps):
        if on_period is not None and k % per == 0:
            on_period(k*dt)
        x = gs.x.copy()
        k1 = gs.derivatives(x)
        k2 = gs.derivatives(x + 0.5*dt*k1)
        k3 = gs.derivatives(x + 0.5*dt*k2)
        k4 = gs.derivatives(x + dt*k3)
        gs.x = x + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
    return t


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = G.load_gs02_params(PARAMS)
        cls.g = cls.params.gs02
        cls.sim = cls.params.sim
        cls.angles = cls.g["drum_angles"]
        cls.L = cls.params.tether_length

    def new(self):
        G.HighTensionController.reset_function_static()
        return G.GroundStationV2(self.params)


class G1FreeLengthAndAnchor(Base):
    def racetrack_coefficients(self):
        """ground_station_util.cc:119-137 written out: the tether on the racetrack is a curve whose radius goes from
        gsg_pos_drum.z to the drum radius and whose x goes from gsg_pos_drum.x to wrap_start_posx_drum over pi."""
        r0, r1 = self.g["gsg_pos_drum"][2], self.g["drum_radius"]
        x0, x1 = self.g["gsg_pos_drum"][0], self.sim["wrap_start_posx_drum"]
        span = self.angles["racetrack_high"] - self.angles["racetrack_low"]
        dr, dx = (r1 - r0)/span, (x1 - x0)/span
        return lambda tau: math.sqrt((r0 + dr*tau)**2 + dx**2 + dr**2), span

    # G1(a): closed-form racetrack length against Gauss-Legendre (200 points) of the integrand; full racetrack 5.1 m.
    def test_a_racetrack_integral(self):
        gs = self.new()
        integrand, span = self.racetrack_coefficients()
        for tau in (0.0, 1e-6, 0.3, 1.0, 2.0, 3.0, span):
            theta = self.angles["racetrack_high"] - tau
            closed = gs.racetrack.wrapped_length(theta)
            numeric = gauss_legendre(integrand, 0.0, tau) if tau > 0 else 0.0
            self.assertLessEqual(abs(closed - numeric), 1e-9, tau)
        full = gs.racetrack.wrapped_length(self.angles["racetrack_low"])
        self.assertLessEqual(abs(full - self.g["racetrack_tether_length"]), 0.01)   # ground_station_util_test.cc:104-107
        self.assertEqual(gs.racetrack.wrapped_length(self.angles["racetrack_low"] - 0.1), full)  # saturated

    def k(self, which):
        return math.sqrt(self.sim[f"dx_dtheta_{which}_wrap"]**2 + self.g["drum_radius"]**2)

    def by_hand(self, theta):
        """Free length by sections (ground_station_v2.cc:770-799)."""
        integrand, span = self.racetrack_coefficients()
        rh, rl, wl = self.angles["racetrack_high"], self.angles["racetrack_low"], self.angles["wide_wrap_low"]
        if theta >= rh:
            return self.L
        if theta >= rl:
            return self.L - gauss_legendre(integrand, 0.0, rh - theta)
        full = gauss_legendre(integrand, 0.0, span)
        if theta >= wl:
            return self.L - full - (rl - theta)*self.k("wide")
        return self.L - full - (rl - wl)*self.k("wide") - (wl - theta)*self.k("main")

    # G1(b): slopes on the wraps (central differences), full length above -pi, continuity at the joints.
    def test_b_slopes_and_continuity(self):
        gs = self.new()
        h = 0.01
        rl, wl = self.angles["racetrack_low"], self.angles["wide_wrap_low"]
        for theta in np.linspace(gs.min_drum_angle() + 2*h, wl - 2*h, 50):
            slope = (gs.free_length(theta + h) - gs.free_length(theta - h))/(2*h)
            self.assertLessEqual(abs(slope - self.k("main")), 1e-8)
        for theta in np.linspace(wl + 2*h, rl - 2*h, 50):
            slope = (gs.free_length(theta + h) - gs.free_length(theta - h))/(2*h)
            self.assertLessEqual(abs(slope - self.k("wide")), 1e-8)
        for theta in (-PI, -PI + 1e-9, -2.0, 0.0, 3.0):
            self.assertEqual(gs.free_length(theta), self.L)
        for theta in np.concatenate([np.linspace(-229.0, -3.2, 40), [wl, rl, -PI]]):
            self.assertLessEqual(abs(gs.free_length(theta) - self.by_hand(theta)), 1e-9, theta)
        # The sections meet: each joint from both sides, with a step far below the tolerance times the slope.
        eps = 1e-14
        for joint in (-PI, rl, wl):
            self.assertLessEqual(abs(gs.free_length(joint) - gs.free_length(joint - eps)), 1e-12, joint)
            self.assertLessEqual(abs(gs.free_length(joint) - gs.free_length(joint + eps)), 1e-12, joint)

    def anchor_by_hand(self, psi, psi_dot, theta, omega):
        """ground_station_v2.cc:812-877 written out, with the frames of UpdateDerivedStates (684-720)."""
        g, sim = self.g, self.sim
        origin, gsg = np.array(g["drum_origin_p"]), np.array(g["gsg_pos_drum"])
        rh, rl, wl = self.angles["racetrack_high"], self.angles["racetrack_low"], self.angles["wide_wrap_low"]
        Rp = rot_z(psi).T
        w_platform = np.array([0.0, 0.0, psi_dot])
        if theta > rh:
            p = Rp@(origin + rot_x(theta).T@gsg)
            v = np.cross(w_platform, p) + Rp@np.cross(np.array([omega, 0.0, 0.0]), rot_x(theta).T@gsg)
            return p, v
        rin = math.hypot(gsg[1], gsg[2])
        if theta >= rl:      # racetrack (theta < rh) - the branch also holds at theta == rh with dtheta 0
            dtheta = theta - rh if theta < rh else 0.0
            span = rh - rl
            q = np.array([gsg[0] + (gsg[0] - sim["wrap_start_posx_drum"])/span*dtheta, 0.0,
                          -rin + (g["drum_radius"] - rin)/span*dtheta])
            u = (np.array([(gsg[0] - sim["wrap_start_posx_drum"])/span*omega, 0.0, (g["drum_radius"] - rin)/span*omega])
                 if theta < rh else np.zeros(3))
        else:
            base = np.array([sim["wrap_start_posx_drum"], 0.0, -g["drum_radius"]])
            if theta >= wl:
                q = base + np.array([sim["dx_dtheta_wide_wrap"]*(theta - rl), 0.0, 0.0])
                u = np.array([sim["dx_dtheta_wide_wrap"]*omega, 0.0, 0.0])
            else:
                q = base + np.array([sim["dx_dtheta_wide_wrap"]*(wl - rl) + sim["dx_dtheta_main_wrap"]*(theta - wl), 0.0, 0.0])
                u = np.array([sim["dx_dtheta_main_wrap"], 0.0, 0.0])      # B2: no factor omega, as coded (867)
        p = Rp@(origin + q)
        return p, np.cross(w_platform, p) + Rp@u

    # G1(c): the anchor, section by section; on the main wrap the velocity as coded (B2).
    def test_c_anchor(self):
        gs = self.new()
        rl, wl = self.angles["racetrack_low"], self.angles["wide_wrap_low"]
        for psi, psi_dot in ((0.0, 0.0), (2.2689, 0.07), (-1.3, -0.2)):
            for theta in (1.0, 0.0, -1.0, -PI + 1e-6, -PI, -4.0, -6.0, rl, -8.0, -10.0, wl, -50.0, -228.5):
                for omega in (0.0, 0.37, -1.1):
                    gs.x[:] = 0.0
                    gs.x[gs.PSI], gs.x[gs.PSI_DOT], gs.x[gs.THETA], gs.x[gs.OMEGA] = psi, psi_dot, theta, omega
                    pos, vel = gs.anchor_pos_vel_ned()
                    p, v = self.anchor_by_hand(psi, psi_dot, theta, omega)
                    self.assertLessEqual(np.abs(pos - p).max(), 1e-12, (psi, theta, omega))
                    self.assertLessEqual(np.abs(vel - v).max(), 1e-12, (psi, theta, omega))
        # B2 explicitly: on the main wrap the anchor's drum-relative speed does not scale with the drum rate.
        gs.x[:] = 0.0
        gs.x[gs.THETA] = -50.0
        speeds = []
        for omega in (0.0, 1.0, 5.0):
            gs.x[gs.OMEGA] = omega
            speeds.append(np.linalg.norm(gs.anchor_pos_vel_ned()[1]))
        self.assertEqual(speeds[0], speeds[1])
        self.assertEqual(speeds[1], speeds[2])
        self.assertLessEqual(abs(speeds[0] - abs(self.sim["dx_dtheta_main_wrap"])), 1e-15)

    # G1(d): MinDrumAngle (801-810) by hand.
    def test_d_min_drum_angle(self):
        gs = self.new()
        integrand, span = self.racetrack_coefficients()
        rl, wl = self.angles["racetrack_low"], self.angles["wide_wrap_low"]
        main_length = self.L - gauss_legendre(integrand, 0.0, span) - (rl - wl)*self.k("wide")
        self.assertLessEqual(abs(gs.min_drum_angle() - (wl - main_length/self.k("main"))), 1e-9)
        self.assertLessEqual(abs(gs.free_length(gs.min_drum_angle())), 1e-9)


class G2DynamicsAndControllers(Base):
    # G2(a): drum second-order and platform first-order responses to a step command, against the analytic solutions,
    # with RK4 at the step of the perched runs (1e-3 s).
    def test_a_servo_responses(self):
        gs = self.new()
        gs.init(0.3, -3.0)
        gs.mode.val = G.MODE_REEL
        u_drum, u_azi = 0.4, -0.05
        gs.reel_controller.drum_vel_cmd.val = u_drum
        gs.reel_controller.azi_vel_cmd.val = u_azi
        wn, zeta, kp = self.sim["winch_drive_natural_freq"], self.sim["winch_drive_damping_ratio"], self.sim["azi_accel_kp"]
        wd = wn*math.sqrt(1 - zeta**2)
        dt, worst_drum, worst_azi, worst_psi = 1e-3, 0.0, 0.0, 0.0
        for k in range(1, 3001):
            rk4(gs, dt, 1)
            t = k*dt
            omega = u_drum*(1 - math.exp(-zeta*wn*t)*(math.cos(wd*t) + zeta/math.sqrt(1 - zeta**2)*math.sin(wd*t)))
            psi_dot = u_azi*(1 - math.exp(-kp*t))
            psi = 0.3 + u_azi*(t - (1 - math.exp(-kp*t))/kp)
            worst_drum = max(worst_drum, abs(gs.drum_omega - omega))
            worst_azi = max(worst_azi, abs(gs.platform_azi_vel - psi_dot))
            worst_psi = max(worst_psi, abs(gs.platform_azi - psi))
        self.assertLessEqual(worst_drum, 1e-6)
        self.assertLessEqual(worst_azi, 1e-6)
        self.assertLessEqual(worst_psi, 1e-6)

    # G2(b): the helpers against independent derivations.
    def test_b_helpers(self):
        # Lpf2: Tustin's substitution s = 2/T (z-1)/(z+1) in w^2/(s^2 + 2 zeta w s + w^2), written out.
        for w, zeta, T in ((5.0, 1.0, 0.01), (2.0, 0.7, 0.01), (30.0, 0.3, 0.02)):
            a, b = G.second_order_lowpass_coefficients(w/2.0/PI, zeta, T)
            d = np.array([4/T**2 + 4*zeta*w/T + w**2, -8/T**2 + 2*w**2, 4/T**2 - 4*zeta*w/T + w**2])
            n = w**2*np.array([1.0, 2.0, 1.0])
            self.assertLessEqual(np.abs(np.array(a) - d/d[0]).max(), 1e-12)
            self.assertLessEqual(np.abs(np.array(b) - n/d[0]).max(), 1e-12)
            # Direct form II (the module) against direct form I (written here) on a random input.
            u = np.random.default_rng(3).normal(size=200)
            z, y_ii, y_i = (0.0, 0.0), [], []
            for x in u:
                y, z = G.lpf2(float(x), w/2.0/PI, zeta, T, z)
                y_ii.append(y)
            nb, da = n/d[0], d/d[0]
            for i in range(len(u)):
                acc = nb[0]*u[i] + (nb[1]*u[i-1] if i >= 1 else 0) + (nb[2]*u[i-2] if i >= 2 else 0)
                acc -= (da[1]*y_i[i-1] if i >= 1 else 0) + (da[2]*y_i[i-2] if i >= 2 else 0)
                y_i.append(acc)
            self.assertLessEqual(np.abs(np.array(y_ii) - np.array(y_i)).max(), 1e-12)
        # AttenuateAndApplyDeadZone (34-58): v_max = sqrt(2 a |e|), then a dead zone that shifts by sqrt(2 a |h|).
        a, h = 0.2, 0.01
        u = math.sqrt(2*a*h)
        for v, e in ((1.0, 4.0), (1.0, 0.5), (-1.0, 0.5), (0.01, 4.0), (-0.01, 4.0), (0.3, -0.1), (0.0, 1.0)):
            vmax = math.sqrt(2*a*abs(e))
            m = min(max(v, -vmax), vmax)
            expected = m + u if m < -u else (0.0 if m < u else m - u)
            self.assertLessEqual(abs(G.attenuate_and_apply_dead_zone(v, e, a, h) - expected), 1e-12)
        # RateLimit (filter.c:103-110).
        y = 0.0
        for target, expected in ((1.0, 0.0011), (1.0, 0.0022), (-1.0, 0.0011), (0.0011, 0.0011)):
            out, y = G.rate_limit(target, -0.11, 0.11, 0.01, y)
            self.assertLessEqual(abs(out - expected), 1e-12)
            self.assertEqual(out, y)
        # Wrap with C fmod semantics, Sign as an integer.
        self.assertLessEqual(abs(G.wrap(7.0, -PI, PI) - (7.0 - 2*PI)), 1e-15)
        self.assertLessEqual(abs(G.wrap(-7.0, -PI, PI) - (-7.0 + 2*PI)), 1e-15)
        self.assertEqual((G.sign(-2.0), G.sign(0.0), G.sign(1e-300)), (-1, 0, 1))

    def transform_replica(self, ht2reel, s, psi, theta, wing_azi, unpause):
        """TransformController's stage logic (241-328), written from the source: (next stage, complete)."""
        p = self.params.mclaren["transform"]
        key = "ht2reel" if ht2reel else "reel2ht"
        target = wing_azi + p["azi_offset_from_wing"] + p[f"azi_targets_{key}"][s]
        e = G.wrap(target - psi, -PI, PI)
        azi_ok = -p[f"azi_tols_{key}"][s] < e < p[f"azi_tols_{key}"][s]
        wt, wtol = p[f"winch_targets_{key}"][s], p[f"winch_tols_{key}"][s]
        winch_ok = wt - wtol < theta < wt + wtol
        ok = azi_ok and winch_ok
        if ht2reel:
            last, nxt = 4, min(s + 1, 4)
            if not unpause and s in (1, 2):
                ok = False
        else:
            last, nxt = 1, (4 if s == 0 else max(s - 1, 1))
            if not unpause and s == 3:
                ok = False
        return (nxt if ok else s), (s == last and ok)

    def run_transform(self, ht2reel, theta0, psi0, wing_azi, pauses, max_time_s=300.0):
        """Drive a transform with scripted commands; check each period's stage against the replica."""
        gs = self.new()
        gs.init(psi0, theta0)
        if ht2reel:
            gs.mode.val, gs.mode_z1.val = G.MODE_HIGH_TENSION, G.MODE_HIGH_TENSION
        gs.levelwind_engaged = True
        request = G.MODE_REEL if ht2reel else G.MODE_HIGH_TENSION
        opposite = G.MODE_HIGH_TENSION if ht2reel else G.MODE_REEL
        final = G.MODE_REEL if ht2reel else G.MODE_HIGH_TENSION
        tc = gs.transform_controller
        record = dict(stages=[], waited=[], ignored=0, checked=0)
        pause_since = None
        t, k = 0.0, 0
        while t < max_time_s:
            t = k*0.01
            mode_before, stage_before = gs.mode.val, tc.transform_stage.val
            # Scripted unpause: one second after the targets of a pausing stage are met.
            unpause = False
            if mode_before == G.MODE_TRANSFORM and stage_before in pauses:
                met = self.transform_replica(ht2reel, stage_before, gs.platform_azi, gs.drum_angle, wing_azi, True)[0] != stage_before
                if met:
                    if pause_since is None:
                        pause_since = t
                        record["waited"].append(stage_before)
                    unpause = t - pause_since >= 1.0
                else:
                    pause_since = None
            else:
                pause_since = None
            # A mode command for the way back while the transform is in its second stage: ignored by the source.
            mode_cmd = opposite if (mode_before == G.MODE_TRANSFORM and len(record["stages"]) == 2) else request
            record["ignored"] += int(mode_before == G.MODE_TRANSFORM and mode_cmd == opposite)
            gs.set_from_command(gs_azi_target=wing_azi, gs_azi_dead_zone=0.0, winch_velocity=0.0, detwist_position=0.0,
                                gs_mode_request=mode_cmd, gs_unpause_transform=unpause)
            predicted = (self.transform_replica(ht2reel, stage_before, gs.platform_azi, gs.drum_angle, wing_azi, unpause)
                         if mode_before == G.MODE_TRANSFORM else None)
            gs.discrete_update(t)
            if gs.mode.val == G.MODE_TRANSFORM:
                if mode_before != G.MODE_TRANSFORM:     # StartTransform wins this period (first write): stage 0
                    self.assertEqual((tc.transform_stage.val, tc.transform_complete.val, tc.ht2reel.val), (0, False, ht2reel))
                else:
                    self.assertEqual((tc.transform_stage.val, tc.transform_complete.val), predicted, t)
                    record["checked"] += 1
                if not record["stages"] or record["stages"][-1] != tc.transform_stage.val:
                    record["stages"].append(tc.transform_stage.val)
            elif mode_before == G.MODE_TRANSFORM:
                self.assertTrue(predicted[1] or tc.transform_complete.val is False)
                self.assertEqual(gs.mode.val, final)
                break
            rk4(gs, 1e-3, 10)
            k += 1
        return gs, record

    # G2(c): the mode machine on scripted commands: Reel -> HT through stages 0, 4, 3, 2, 1 (waiting at the end of 3),
    # HT -> Reel through 0, 1, 2, 3, 4 (waiting at the ends of 1 and 2); a mode command during the transform is
    # ignored; each stage advances on the period the replica says.
    def test_c_mode_machine(self):
        wing = 1.0
        gs, rec = self.run_transform(False, -2*PI, wing - PI/2, wing, pauses=(3,))
        self.assertEqual(gs.mode.val, G.MODE_HIGH_TENSION)
        self.assertGreater(rec["checked"], 100)
        self.assertEqual(rec["stages"], [0, 4, 3, 2, 1])
        self.assertIn(3, rec["waited"])
        self.assertGreater(rec["ignored"], 0)
        gs, rec = self.run_transform(True, 0.0, wing - PI, wing, pauses=(1, 2))
        self.assertEqual(gs.mode.val, G.MODE_REEL)
        self.assertGreater(rec["checked"], 100)
        self.assertEqual(rec["stages"], [0, 1, 2, 3, 4])
        self.assertEqual(sorted(set(rec["waited"])), [1, 2])
        self.assertGreater(rec["ignored"], 0)

    # G2(d): Reel drum command - zero while the levelwind is disengaged; at the upper drum limit the command before the
    # rate limiter is zero, so the output only decays at the rate limit (max_drum_accel x ts per period) to zero and the
    # drum comes to rest just past the limit (the decay and the servo lag; the stop point is reported).
    def test_d_reel_drum_command(self):
        gs = self.new()
        reel = self.params.mclaren["reel"]
        upper, decrement = reel["drum_angle_upper_limit"], reel["max_drum_accel"]*self.params.mclaren["ts"]
        gs.init(1.0, upper - 0.3)

        def command(engaged, v=1.0):
            gs.levelwind_engaged = engaged
            gs.set_from_command(gs_azi_target=1.0, gs_azi_dead_zone=0.0, winch_velocity=v, detwist_position=0.0,
                                gs_mode_request=G.MODE_REEL, gs_unpause_transform=False)
        t = 0.0
        for _ in range(50):
            command(False)
            gs.discrete_update(t)
            self.assertEqual(gs.reel_controller.drum_vel_cmd.val, 0.0)
            rk4(gs, 1e-3, 10)
            t += 0.01
        self.assertEqual(gs.drum_angle, upper - 0.3)
        rows = []
        for _ in range(4000):
            command(True)
            theta = gs.drum_angle
            gs.discrete_update(t)
            rows.append((theta, gs.reel_controller.drum_vel_cmd.val))
            rk4(gs, 1e-3, 10)
            t += 0.01
        self.assertGreater(max(c for _, c in rows), 0.0)                   # it does pay out below the limit
        first = next(i for i, (theta, _) in enumerate(rows) if theta >= upper)
        previous = rows[first - 1][1]
        for theta, c in rows[first:]:
            self.assertLessEqual(abs(c - max(previous - decrement, 0.0)), 1e-12)
            previous = c
        self.assertEqual(rows[-1][1], 0.0)
        self.assertLessEqual(abs(gs.drum_omega), 1e-6)
        self.assertGreater(gs.drum_angle, upper)
        self.assertLessEqual(gs.drum_angle - upper, 0.05)
        self.__class__.reel_stop_past_limit_rad = gs.drum_angle - upper

    def ht_replica(self, steps):
        """RunControlSwitchingLogic and the HPU model (384-546), written from the source, on the scripted inputs."""
        p = self.params.mclaren["high_tension"]
        state, static, hpu, hpu_last, demand, t0, first, brake_t0 = 0, 0, 1, 1, 1, 0.0, True, 0.0
        out = []
        for t, torque, a_err, azi_vel in steps:
            if state == 0:
                demand = 1
                if torque > p["m_max_azi_ht"] and a_err > p["a_control_threshold_azi_ht"]:
                    static = 1
                if torque < -p["m_max_azi_ht"] and a_err < -p["a_control_threshold_azi_ht"]:
                    static = 6
            elif state == 1:
                demand = 1
                if torque > p["m_control_threshold_azi_ht"]:
                    static = 2
            elif state == 2:
                demand = 2
                if hpu == 2:
                    static = 3
                if torque < p["m_control_threshold_azi_ht"]:
                    static = 1
            elif state == 3:
                if abs(a_err) < p["a_control_tolerance_azi_ht"] or a_err < 0:
                    static, first = 5, True
            elif state == 5:
                if first:
                    first, brake_t0 = False, t
                if abs(azi_vel) < p["n_control_tolerance_azi_ht"] or t > brake_t0 + p["t_threshold_wait_azi_ht"]:
                    static = 0
            elif state == 6:
                demand = 1
                if torque < -p["m_control_threshold_azi_ht"]:
                    static = 7
            elif state == 7:
                demand = 2
                if torque > -p["m_control_threshold_azi_ht"]:
                    static = 6
                if hpu == 2:
                    static = 8
            elif state == 8:
                if abs(a_err) < p["a_control_tolerance_azi_ht"] or a_err > 0:
                    static, first = 5, True
            state = static
            if hpu_last != demand:
                t0 = t
            if t > t0 + 0.4:
                hpu = demand
            hpu_last = demand
            out.append(state)
        return out

    # G2(e): HighTension brake state machine on scripted torques and azimuth errors, period by period.
    def test_e_high_tension_state_machine(self):
        for sign in (1.0, -1.0):
            gs = self.new()
            ht = gs.high_tension_controller
            script = []
            t = 0.0
            for k in range(1200):
                if k < 20:
                    torque, a_err, azi_vel = 0.0, 0.0, 0.0
                elif k < 600:
                    torque, a_err, azi_vel = 6000.0, 0.2, 0.0
                elif k < 800:
                    torque, a_err, azi_vel = 6000.0, 0.01, 0.1
                else:
                    torque, a_err, azi_vel = 0.0, 0.01, 0.01
                script.append((t, sign*torque, sign*a_err, azi_vel))
                t += 0.01
            states = []
            for t, torque, a_err, azi_vel in script:
                wing_azi = 1.0
                azi_pos = wing_azi - PI - a_err      # a_error = Wrap(wing_azi - azi_pos - pi) = a_err
                ht.set_common_inputs(t, azi_pos, azi_vel, 0.0, 0.0, wing_azi)
                ht.tether_torque.update(t, torque)
                ht.discrete_update(t)
                states.append(ht.n_state_machine.val)
            self.assertEqual(states, self.ht_replica(script))
            path = [s for i, s in enumerate(states) if i == 0 or s != states[i-1]]
            self.assertEqual(path, [0, 1, 2, 3, 5, 0] if sign > 0 else [0, 6, 7, 8, 5, 0])


class G3SensorsAndEstimate(Base):
    # G3(a): tether_engaged (ground_station_v2_sensors.cc:67-84) as a hand table; B7 raises.
    def test_a_tether_engaged_table(self):
        R, H, T, M = G.MODE_REEL, G.MODE_HIGH_TENSION, G.MODE_TRANSFORM, G.MODE_MANUAL
        for stage in range(5):
            for cmd in (R, H, T, M):
                self.assertTrue(G.tether_engaged(R, cmd, stage))
                self.assertTrue(G.tether_engaged(M, cmd, stage))
                self.assertFalse(G.tether_engaged(H, cmd, stage))
            self.assertEqual(G.tether_engaged(T, R, stage), stage == 4)
            self.assertEqual(G.tether_engaged(T, H, stage), stage in (0, 4))
            for cmd in (T, M):
                with self.assertRaises(RuntimeError):
                    G.tether_engaged(T, cmd, stage)

    def gs_in(self, mode, mode_cmd, stage, psi, theta, phi):
        gs = self.new()
        gs.init(psi, theta)
        gs.x[gs.PHI] = phi
        gs.mode.val, gs.transform_stage.val, gs.mode_cmd = mode, stage, mode_cmd
        return gs

    # G3(b): round trip - a tether direction through the GS02 sensor model, part 1's sensor chain and Makani's own
    # ground estimate gives back its elevation in the ground frame, in each mode and stage where the estimate uses the
    # sensor the model reports.
    @unittest.skipUnless(BUILD, "OPENFAWE_MAKANI_FULL_BUILD not set")
    def test_b_round_trip_through_the_estimator(self):
        from openfawe.flight.makani_full import FullController, Joystick, gs02_sensors, load_full_api
        api = load_full_api(BUILD)
        cells = [(G.MODE_REEL, G.MODE_REEL, 0, "levelwind"), (G.MODE_HIGH_TENSION, G.MODE_HIGH_TENSION, 0, "gsg")]
        cells += [(G.MODE_TRANSFORM, cmd, s, "gsg") for cmd in (G.MODE_REEL, G.MODE_HIGH_TENSION) for s in (1, 2, 3)]
        directions = [np.array(v, dtype=float) for v in ((0.6, -0.5, -0.62), (-0.2, 0.9, -0.1), (0.3, 0.3, -0.9),
                                                          (0.9, 0.1, 0.05))]
        worst = 0.0
        for mode, cmd, stage, source in cells:
            for psi, theta, phi in ((0.4, -2.0, 0.9), (2.2689, -0.5, 3.54), (-2.5, 0.3, -1.2)):
                gs = self.gs_in(mode, cmd, stage, psi, theta, phi)
                for d in directions:
                    d = d/np.linalg.norm(d)
                    ele, _, yoke, term = gs.tether_dir_to_angles(d)
                    readings = G.sensor_readings(gs, ele, yoke, term)
                    self.assertEqual(readings["tether_engaged"], source == "levelwind")
                    fc = FullController(api, follow_init_state=False)
                    sensors = gs02_sensors(api, fc.system, drum_angle_rad=readings["drum_angle_rad"],
                                           platform_azimuth_rad=readings["platform_azimuth_rad"], gs_mode=readings["gs_mode"],
                                           transform_stage=readings["transform_stage"], proximity=readings["proximity"],
                                           detwist_angle_rad=readings["detwist_angle_rad"],
                                           levelwind_elevation_rad=readings["levelwind_elevation_rad"],
                                           gsg_yoke_rad=readings["gsg_yoke_rad"],
                                           gsg_termination_rad=readings["gsg_termination_rad"])
                    est = api.StateEstimate()
                    for _ in range(int(fc.control.estimator.ground_station.num_debounce) + 1):
                        est, _, _ = fc.ground_estimate(est, sensors, Joystick(switch_position=api.kJoystickSwitchPositionUp))
                    self.assertEqual((est.gs_mode, est.gs_transform_stage), (mode, stage))
                    self.assertTrue(est.tether_ground_angles.elevation_valid)
                    expected = math.atan2(-d[2], math.hypot(d[0], d[1]))
                    error = abs(est.tether_ground_angles.elevation_g - expected)
                    worst = max(worst, error)
                    self.assertLessEqual(error, 1e-9, (mode, cmd, stage, psi, theta, phi, d))
        self.assertLessEqual(worst, 1e-9)


class G6Initialisation(Base):
    # G6: Makani's bisection for the perched drum angle (full_system.cc:866-914) on the perched geometry.
    def test_bisection(self):
        from openfawe.ground.perch import perched_initialisation
        init, _ = perched_initialisation(self.params)
        L, rho, g = self.L, self.params.raw["tether"]["linear_density"], self.params.raw["phys"]["g"]
        self.assertGreaterEqual(init["tension_N"], L*rho*g/100.0)
        self.assertLessEqual(init["tension_N"], L*rho*g/2.0)
        r = self.g["drum_radius"]
        self.assertGreaterEqual(init["drum_angle_rad"]*r, -L - 1.1*PI*r)        # EstimatorWinchSetPayout's bound
        self.assertGreater(init["free_length_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
