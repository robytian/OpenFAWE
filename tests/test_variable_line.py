"""Plan 3.7 part 2, criterion G4 (full-cycle contract revision 4): the variable-length tether.

Expected values are written here by hand from Makani's tether.cc: the node table from the activation thresholds,
the staging-node equations from tether.cc:439-485 and 602-753, the static shape from the elastic catenary (Irvine's
closed form, solved here with SciPy). G4(e) integrates 423 s twice (pay-out and reel-in) and runs only with
OPENFAWE_SLOW_TESTS=1 (a skip fails the judged run).
"""
from fractions import Fraction
import math
import os
import unittest

import numpy as np
from scipy.optimize import fsolve

from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings
from openfawe.tether.variable_line import Indices, VariableLengthTether

SLOW = os.environ.get("OPENFAWE_SLOW_TESTS") == "1"
RELAX_S = 120.0                 # relaxation time of G4(d)
AVERAGE_S = 60.0                # averaging window at its end
CHORDS = (0.98, 1.0005)         # chord / free length: slack and taut


def rk4(line, free_length, x, v, ends, t, dt):
    """One RK4 step of the tether alone; ``free_length(t)`` and ``ends(t) -> (start, start_vel, end, end_vel)``."""
    def rates(time, xs, vs):
        s, sv, e, ev = ends(time)
        dx, dv, _ = line.derivatives(free_length(time), xs, vs, s, sv, e, ev)
        return dx, dv
    k1 = rates(t, x, v)
    k2 = rates(t + dt/2, x + dt/2*k1[0], v + dt/2*k1[1])
    k3 = rates(t + dt/2, x + dt/2*k2[0], v + dt/2*k2[1])
    k4 = rates(t + dt, x + dt*k3[0], v + dt*k3[1])
    return x + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), v + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])


class G4VariableLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.line = VariableLengthTether()
        cls.n, cls.L = cls.line.n, cls.line.p.length_m

    def hand_indices(self, length):
        """Node k (0..N-1) is in the air iff L >= L0/2 + (N-1-k) L0 + 1 m (k <= N-2), node N-1 iff L >= L0/2 + 1 m;
        exact rational arithmetic."""
        L, L0 = Fraction(length), Fraction(self.L)/self.n
        lim = Fraction(1)
        active = [k for k in range(self.n) if L >= L0/2 + (self.n - 1 - k)*L0 + lim]
        start = min(active) if active else self.n
        L_end = min(L0/2, L)
        L_start = L - (L0/2 + (self.n - 1 - start)*L0) if active else (L - L_end if L > L_end else Fraction(0))
        return start, float(L_start), float(L_end)

    # G4(a): first active node, last node and the first and last natural lengths against the hand table.
    def test_a_index_table(self):
        L0 = self.L/self.n
        lengths = [0.5, 3.0, 11.0, 11.205, 12.0, 12.2, 30.0, 34.0, 100.0, 200.0, 300.0, 400.0, 425.0, self.L]
        for k in range(self.n):
            threshold = L0/2 + (self.n - 1 - k)*L0 + 1.0
            lengths += [threshold - 1e-6, threshold + 1e-6]
        for length in lengths:
            if length > self.L:
                continue
            got = self.line.indices(length)
            start, L_start, L_end = self.hand_indices(length)
            self.assertEqual(got.start, start, length)
            self.assertEqual(got.end, self.n - 1)
            self.assertLessEqual(abs(got.L_end - L_end), 1e-12, length)
            self.assertLessEqual(abs(got.L_start - L_start), 1e-9, length)
            self.assertGreaterEqual(got.L_start if got.active else 1.0, 1.0)       # a first segment is never < 1 m

    # G4(b): airborne mass rho L_free whenever a node is active.
    def test_b_airborne_mass(self):
        rho = self.line.p.linear_density_kg_m
        for length in np.linspace(12.3, self.L, 97):
            idx = self.line.indices(length)
            self.assertTrue(idx.active)
            self.assertLessEqual(abs(self.line.airborne_mass(idx) - rho*length), 1e-12*rho*length, length)

    def states(self):
        rng = np.random.default_rng(11)
        out = []
        for end, noise in ((np.array([300.0, 0.0, 280.0]), 0.3), (0.8*425.95*np.array([1.0, 0.0, 0.0]) + 0.6*425.95*np.array([0.0, 0.3, 0.954]), 0.01),
                           (np.array([400.0, 50.0, 150.0]), 0.05)):
            x, v = LumpedTether().straight_state(np.zeros(3), end)
            out.append((x + rng.normal(0, noise, x.shape), rng.normal(0, 0.5, x.shape), end, rng.normal(0, 3, 3),
                        rng.normal(0, 2, x.shape)))
        return out

    # G4(c): at full free length the forces equal LumpedTether's for the same state (node forces, both end forces).
    def test_c_full_length_equals_lumped_tether(self):
        lumped = LumpedTether()
        worst = dict(node=0.0, start=0.0, end=0.0)
        rows = []
        for x, v, end, end_vel, wind in self.states():
            a = lumped.forces(x, v, np.zeros(3), np.zeros(3), end, end_vel, wind)
            b = self.line.forces(self.L, x, v, np.zeros(3), np.zeros(3), end, end_vel, wind)
            rel = dict(node=float((np.linalg.norm(a["node"] - b["node"], axis=1)/np.linalg.norm(a["node"], axis=1)).max()))
            for key in ("start", "end"):
                rel[key] = float(np.linalg.norm(a[key] - b[key])/np.linalg.norm(a[key]))
            rows.append(rel)
            worst = {k: max(worst[k], rel[k]) for k in worst}
        # Reported with the result (not judged): the first natural length of the two models at full free length, and
        # the same comparison with the variable tether evaluated at LumpedTether's first natural length (L0/2).
        idx = self.line.indices(self.L)
        same = Indices(idx.start, lumped.natural[0], idx.end, idx.L_end)
        same_worst = 0.0
        for x, v, end, end_vel, wind in self.states():
            a = lumped.forces(x, v, np.zeros(3), np.zeros(3), end, end_vel, wind)
            b = self.line.forces(self.L, x, v, np.zeros(3), np.zeros(3), end, end_vel, wind, indices=same)
            same_worst = max(same_worst, float((np.linalg.norm(a["node"] - b["node"], axis=1)/np.linalg.norm(a["node"], axis=1)).max()),
                             *(float(np.linalg.norm(a[k] - b[k])/np.linalg.norm(a[k])) for k in ("start", "end")))
        self.__class__.full_length = dict(worst=worst, cases=rows, L_start_m=idx.L_start, half_segment_m=lumped.natural[0],
                                          difference_m=idx.L_start - lumped.natural[0],
                                          worst_at_lumped_first_length=same_worst)
        for key, value in worst.items():
            self.assertLessEqual(value, 1e-12, key)

    def catenary(self, natural, l, h):
        """Elastic catenary through (0,0) and (l,h) (z up): unstretched length ``natural``, weight w per unstretched
        metre; returns (H, Tz0, x(s), z(s)) with the tension T(s) = (H, Tz0 + w s) (Irvine 1981, written out)."""
        p = self.line.p
        w, EA = p.linear_density_kg_m*self.line.s.gravity_m_s2, p.tensile_stiffness_N

        def x(s, H, Tz0):
            return H*s/EA + H/w*(math.asinh((Tz0 + w*s)/H) - math.asinh(Tz0/H))

        def z(s, H, Tz0):
            return (Tz0*s + w*s*s/2)/EA + (math.sqrt(H*H + (Tz0 + w*s)**2) - math.sqrt(H*H + Tz0*Tz0))/w

        def residual(u):
            H, Tz0 = math.exp(u[0]), u[1]
            return [x(natural, H, Tz0) - l, z(natural, H, Tz0) - h]
        stretch = math.hypot(l, h)/natural - 1.0
        H0 = EA*stretch if stretch > 1e-6 else w*natural/(8.0*max(math.sqrt(3.0*max(-stretch, 1e-6)/8.0), 1e-3))
        sol = fsolve(residual, [math.log(H0), -w*natural/2 + w*natural*h/max(math.hypot(l, h), 1e-9)/2], xtol=1e-13)
        self.assertLessEqual(max(abs(r) for r in residual(sol)), 1e-9)
        H, Tz0 = math.exp(sol[0]), sol[1]
        return H, Tz0, (lambda s: x(s, H, Tz0)), (lambda s: z(s, H, Tz0))

    def static_case(self, length, angle, chord_ratio):
        """One G4(d) case: start on the catenary at the nodes' unstretched arc lengths, relax with the model's own
        damping for RELAX_S, average the last AVERAGE_S (the transverse modes are weakly damped); return the relative
        differences of the end tensions and of the sag, and the residual node speed."""
        line = self.line
        chord = chord_ratio*length
        l, h = chord*math.cos(angle), chord*math.sin(angle)
        H, Tz0, cx, cz = self.catenary(length, l, h)
        idx = line.indices(length)
        s_nodes = np.concatenate([[idx.L_start], idx.L_start + line.L0*np.arange(1, line.n - idx.start)])
        x = np.zeros((line.n, 3))
        x[idx.start:] = np.array([[cx(s), 0.0, cz(s)] for s in s_nodes])
        if idx.start >= 1:
            x[:idx.start] = x[idx.start]*0.0                          # staged nodes start at the anchor's side
        v = np.zeros_like(x)
        start, end = np.zeros(3), np.array([l, 0.0, h])
        ends = lambda t: (start, np.zeros(3), end, np.zeros(3))          # noqa: E731
        dt, t = 1e-3, 0.0
        sums, count = [0.0, 0.0, np.zeros((line.n - idx.start, 3))], 0
        steps = int(round(RELAX_S/dt))
        for k in range(steps):
            x, v = rk4(line, lambda t: length, x, v, ends, t, dt)
            t += dt
            if k >= steps - int(round(AVERAGE_S/dt)):
                out = line.forces(length, x, v, start, np.zeros(3), end, np.zeros(3))
                sums[0] += np.linalg.norm(out["start"])
                sums[1] += np.linalg.norm(out["end"])
                sums[2] += x[idx.start:]
                count += 1
        t_start, t_end, nodes = sums[0]/count, sums[1]/count, sums[2]/count
        c_start = math.hypot(H, Tz0)
        c_end = math.hypot(H, Tz0 + line.p.linear_density_kg_m*line.s.gravity_m_s2*length)
        chord_z = lambda xx: h/l*xx                                        # noqa: E731
        sag_nodes = max(chord_z(p[0]) - p[2] for p in nodes)
        sag_cat = max(chord_z(cx(s)) - cz(s) for s in s_nodes)
        return dict(length=length, angle_deg=math.degrees(angle), chord_ratio=chord_ratio,
                    start_tension=t_start/c_start - 1, end_tension=t_end/c_end - 1, sag=sag_nodes/sag_cat - 1,
                    sag_m=sag_cat, horizontal_tension_N=H, residual_speed=float(np.abs(v[idx.start:]).max()))

    # G4(d): constant free length, relaxed with the model's own damping, against the elastic catenary: end tensions
    # and sag (the largest vertical distance below the chord of the nodes, against the catenary's at the nodes'
    # unstretched arc lengths), within 0.5%; 80, 200 and 425.8 m, chord level and at 30 degrees; a slack chord (0.98 of
    # the length) and a taut one (1.0005).
    @unittest.skipUnless(SLOW, "OPENFAWE_SLOW_TESTS not set")
    def test_d_static_catenary(self):
        results = []
        for length in (80.0, 200.0, self.L):
            for angle in (0.0, math.radians(30.0)):
                for ratio in CHORDS:
                    results.append(self.static_case(length, angle, ratio))
        self.__class__.static_results = results
        for r in results:
            for key in ("start_tension", "end_tension", "sag"):
                self.assertLessEqual(abs(r[key]), 0.005, r)

    # G4(e), single steps: the staging node's and the next staging node's equations (tether.cc:439-485, 602-753),
    # written out for one state.
    def test_e_staging_node_equations(self):
        line = self.line
        rng = np.random.default_rng(5)
        p = line.p
        for length in (3.0, 12.4, 34.0, 50.0, 140.0):
            idx = line.indices(length)
            x = rng.normal(0, 3, (line.n, 3)) + np.linspace([0, 0, 0], [0, 0, length], line.n)
            v = rng.normal(0, 1, (line.n, 3))
            start, start_vel = np.array([0.1, -0.2, 0.3]), np.array([0.01, 0.02, -0.03])
            end, end_vel = np.array([0.0, 0.0, length]) + rng.normal(0, 0.5, 3), rng.normal(0, 1, 3)
            dx, dv, out = line.derivatives(length, x, v, start, start_vel, end, end_vel)
            s = idx.start
            i = s - 1
            L_first = idx.L_start + idx.L_end if s > idx.end else idx.L_start
            L_first0 = line.L0 if s <= idx.end else line.L0/2
            L_pre = max(0.5, L_first - L_first0)
            L_post = L_first - L_pre
            post = x[s] if s <= idx.end else end

            def spring(a, b, L0):
                d = b - a
                n = np.linalg.norm(d)
                if n <= L0:
                    return np.zeros(3)
                ea = p.tensile_stiffness_N*(min(L0, 1.0)/1.0)**3
                return ea*(n - L0)/L0*d/n
            c_staged = 0.14*math.sqrt(2*p.tensile_stiffness_N*p.linear_density_kg_m)
            mid = 0.5*(x[i] + x[s]) if s <= idx.end else end
            axis = mid - start
            u = axis/np.linalg.norm(axis)
            vn = v[i] - np.dot(v[i], u)*u
            aero = -0.5*line.s.air_density_kg_m3*p.section_drag_coeff*p.outer_diameter_m*np.linalg.norm(axis)*np.linalg.norm(vn)*vn
            m0 = p.linear_density_kg_m*p.length_m/line.n
            force = spring(x[i], post, L_post) - spring(start, x[i], L_pre) - c_staged*v[i] + aero + np.array([0, 0, -m0*9.81])
            self.assertLessEqual(np.abs(dv[i] - force/m0).max(), 1e-9*max(1.0, np.abs(force/m0).max()), length)
            self.assertEqual(dx[i].tolist(), v[i].tolist())
            j = s - 2
            if j >= 0:
                rate = 20.0/(line.L0/3.0)
                self.assertLessEqual(np.abs(dx[j] - rate*(0.5*(x[i] + start) - x[j])).max(), 1e-12*rate*100)
                self.assertLessEqual(np.abs(dv[j] - rate*(0.5*(v[i] + start_vel) - v[j])).max(), 1e-12*rate*100)
                self.assertEqual(out["node"][j].tolist(), [0.0, 0.0, 0.0])
            for k in range(max(s - 2, 0)):
                self.assertEqual(dx[k].tolist(), [0.0, 0.0, 0.0])
                self.assertEqual(dv[k].tolist(), [0.0, 0.0, 0.0])

    # G4(e): pay out at 2 m/s from 3 m to the full length and reel back in: finite throughout, nodes switched on and
    # off at the thresholds of (a); the start-force jump at each switch is recorded (reported, not judged).
    @unittest.skipUnless(SLOW, "OPENFAWE_SLOW_TESTS not set")
    def test_e_pay_out_and_reel_in(self):
        line = self.line
        speed, low = 2.0, 3.0
        duration = (self.L - low)/speed
        direction = np.array([math.cos(math.radians(30.0)), 0.0, math.sin(math.radians(30.0))])
        start = np.zeros(3)
        for sign in (1.0, -1.0):
            free = (lambda t: low + speed*t) if sign > 0 else (lambda t: self.L - speed*t)
            ends = lambda t: (start, np.zeros(3), 0.995*free(t)*direction, 0.995*sign*speed*direction)   # noqa: E731
            if sign > 0:
                x, v = line.init_nodes(start, 0.995*free(0.0)*direction, free(0.0))
            dt, t, last = 1e-3, 0.0, line.indices(free(0.0)).start
            switches = []
            for _ in range(int(round(duration/dt))):
                x_prev, v_prev, t_prev = x, v, t
                x, v = rk4(line, free, x, v, ends, t, dt)
                t += dt
                self.assertTrue(np.isfinite(x).all() and np.isfinite(v).all())
                now = line.indices(free(t)).start
                if now != last:
                    before = np.linalg.norm(line.forces(free(t_prev), x_prev, v_prev, *ends(t_prev))["start"])
                    after = np.linalg.norm(line.forces(free(t), x, v, *ends(t))["start"])
                    switches.append((t, last, now, float(before), float(after)))
                    last = now
            self.__class__.switches = getattr(self.__class__, "switches", {})
            self.__class__.switches["pay_out" if sign > 0 else "reel_in"] = switches
            L0 = line.L0
            for t_switch, before, after, _, _ in switches:
                node = min(before, after) if sign > 0 else max(before, after) - 1
                threshold = L0/2 + (self.n - 1 - node)*L0 + 1.0
                self.assertLessEqual(abs(free(t_switch) - threshold), speed*dt + 1e-9, (sign, t_switch, before, after))
            self.assertEqual(len(switches), self.n)


if __name__ == "__main__":
    unittest.main()
