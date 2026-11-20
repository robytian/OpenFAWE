"""Spanwise artificial viscosity for the VSM closure past stall (OF-0010 revision 7 with its addendum 1, P5c, option 乙1).

The closure's target circulation F(Gamma) = 1/2 c CL(alpha) |v x s| is replaced by the solution of
(I - diag(mu) L) Gamma_target = F(Gamma), so that the nonlinear solve finds (I - diag(mu) L) Gamma = F(Gamma).
mu_i = max(0, -k S_s CL'_i / dz_i^2): k = 0.035, S_s the planform area sum(c dz) of the lifting surface that holds panel
i, dz_i its spanwise length |dl_i|, CL'_i the central difference of its table over +-0.5 deg at the current closure angle
(per radian), re-evaluated at every iterate. L is the three-point second difference within each lifting surface (no
coupling between surfaces); a free tip uses (L Gamma)_0 = -4 Gamma_0 + 4/3 Gamma_1 (the quadratic that vanishes at the
tip), an end that joins another surface uses zero
gradient (L Gamma)_0 = Gamma_1 - Gamma_0 (this contract's adaptation; an end joins when its bound node lies within
``junction_m`` = 0.5 m of another surface's bound line - on the M600 only the fin's lower end, 0.39 m from the horizontal
tail, the next nearest end being 2.1 m away). The term is on only while some panel's wrapped
closure angle lies beyond its table's angles of the largest or smallest lift coefficient within -30...+30 deg (the
P5a rule); otherwise the target is returned as it is (the same object).

Records (review P5cR-05, P5cR-10; nothing numerical): ``last`` holds the latest call's state (the acceptance check of
``solve_vsm_operator`` makes the last call at the accepted iterate) with each surface's deviation measure
P = 100/b_s sum_i (mu_i (L Gamma)_i / F_i) dz_i in per cent (Simonet, Roncin, Faure & Daridon 2024, eq. 27 as read from a
text extraction of the preprint; here mu (L Gamma) = Gamma - F at the call's own solution, b_s = sum dz of the surface,
panels with F_i = 0 left out); ``tally`` counts the calls, those with the term on and the switches of the gate between
consecutive calls since ``reset_tally``.

After Li, Gaunaa, Pirrung & Lonbaek (2026), J. Phys.: Conf. Ser. 3224 042051 (TORQUE 2026), as implemented in the TU Delft
Vortex-Step-Method (awegroup/Vortex-Step-Method, commit 716eb8d, MIT licence, ``Solver._regularize_gamma_target``); the
artificial-viscosity term goes back to Chattot (2004), AIAA 2004-0220. k was derived for uniformly discretised
rectangular lifting-line wings, not for the three-quarter-chord VSM, non-uniform panels or a free wake.

Nothing in the package imports this module; with it unused every result is unchanged.
"""
import numpy as np
from scipy.linalg import solve_banded

STALL_WINDOW_DEG = (-30.0, 30.0)
K_DEFAULT = 0.035
DELTA_DEG = 0.5
JUNCTION_M = 0.5
CACHE_LIMIT = 4096                                     # review P5cR-04: the stall-angle cache is bounded


def table_stall_angles_deg(polar):
    """(angle of the smallest, of the largest lift coefficient) of a section table within -30...+30 deg, knots only, ties
    to the smaller angle, both interior (the rule of ``flight.platform_surfaces.stall_angles_deg``)."""
    alpha = np.asarray(polar.alpha_deg, dtype=float)
    cl_all = np.asarray(polar.cl, dtype=float)
    inside = (alpha >= STALL_WINDOW_DEG[0]) & (alpha <= STALL_WINDOW_DEG[1])
    if not np.any(inside):
        raise ValueError("a section table without knots in the stall window")
    a, cl = alpha[inside], cl_all[inside]
    edges = np.interp(STALL_WINDOW_DEG, alpha, cl_all)
    low, high = int(np.argmin(cl)), int(np.argmax(cl))
    if not (cl[high] > edges.max() and cl[low] < edges.min()):
        raise ValueError("the largest or the smallest lift coefficient lies at the edge of the stall window")
    return float(a[low]), float(a[high])


def polyline_distance(point, nodes):
    """Shortest distance from ``point`` to the polyline through ``nodes``."""
    best = np.inf
    for a, b in zip(nodes[:-1], nodes[1:]):
        d = b-a
        t = float(np.clip(np.dot(point-a, d)/np.dot(d, d), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point-(a+t*d))))
    return best


def second_difference_bands(count, free_start, free_end):
    """(diagonal, superdiagonal, subdiagonal) of L for one surface of ``count`` panels."""
    if count < 2:
        raise ValueError("a lifting surface needs at least two panels for the spanwise viscosity")
    diag = np.full(count, -2.0)
    sup = np.ones(count-1)
    sub = np.ones(count-1)
    if free_start:
        diag[0], sup[0] = -4.0, 4.0/3.0
    else:
        diag[0], sup[0] = -1.0, 1.0
    if free_end:
        diag[-1], sub[-1] = -4.0, 4.0/3.0
    else:
        diag[-1], sub[-1] = -1.0, 1.0
    return diag, sup, sub


class SpanwiseViscosity:
    """The regularisation of the module text for one assembly (body axes; lengths are invariant under the rigid motion and
    the tail's turn). Call it as ``regularized = viscosity(target, alpha_rad, panel_polars)``."""

    def __init__(self, assembly, *, k=K_DEFAULT, delta_deg=DELTA_DEG, junction_m=JUNCTION_M):
        wings = list(getattr(assembly, "wings", [assembly]))
        self.k, self.delta = float(k), np.radians(float(delta_deg))
        if not (self.k > 0 and self.delta > 0):
            raise ValueError("k and the difference step must be positive")
        names = list(getattr(assembly, "names", [None]*len(wings)))
        nodes = [np.asarray(w.bound, dtype=float) for w in wings]
        self.blocks, dz, chord, area = [], [], [], []
        start = 0
        for index, w in enumerate(wings):
            n = int(w.panels)
            lengths = np.linalg.norm(np.asarray(w.dl, dtype=float), axis=1)
            le, te = np.asarray(w.leading_edge, dtype=float), np.asarray(w.trailing_edge, dtype=float)
            chords = np.linalg.norm(0.5*((te-le)[:-1]+(te-le)[1:]), axis=1)
            others = [p for j, p in enumerate(nodes) if j != index]

            def joined(point):
                return bool(others) and min(polyline_distance(point, p) for p in others) <= junction_m
            first, last = nodes[index][0], nodes[index][-1]
            free_start, free_end = not joined(first), not joined(last)
            self.blocks.append(dict(name=names[index], start=start, stop=start+n, free=(free_start, free_end),
                                    bands=second_difference_bands(n, free_start, free_end)))
            dz.append(lengths)
            chord.append(chords)
            area.append(np.full(n, float(np.sum(chords*lengths))))
            start += n
        self.panels = start
        self.dz, self.chord, self.area = np.concatenate(dz), np.concatenate(chord), np.concatenate(area)
        self._angles = {}
        self.last = dict(active=0, mu_max=0.0, on=False, deviation_percent=None)
        self.reset_tally()

    def reset_tally(self):
        """Start counting the calls afresh (the platform group calls it at the start of each step)."""
        self.tally = dict(calls=0, on=0, switches=0)
        self._previous_on = None

    def settings(self):
        return dict(model="spanwise artificial viscosity after Li, Gaunaa, Pirrung & Lonbaek (2026), as in the TU Delft "
                          "Vortex-Step-Method (716eb8d); OF-0010 revision 7 addendum 1",
                    k=self.k, difference_deg=float(np.degrees(self.delta)),
                    surfaces=[dict(name=b["name"], panels=b["stop"]-b["start"], free_ends=list(b["free"])) for b in self.blocks])

    def stall_angles(self, panel_polars):
        out = np.empty((self.panels, 2))
        for j, table in enumerate(panel_polars.polars):
            key = id(table)
            entry = self._angles.get(key)
            if entry is None or entry[0] is not table:  # review P5cR-04: an id is trusted only with its own table
                if len(self._angles) >= CACHE_LIMIT:
                    self._angles.clear()
                entry = self._angles[key] = (table, table_stall_angles_deg(table))
            out[j] = entry[1]
        return out

    def coefficients(self, target, alpha_rad, panel_polars):
        """mu per panel at these angles (zeros while no panel is beyond its stall angles) and whether the term is on."""
        alpha = np.asarray(alpha_rad, dtype=float)
        if alpha.shape != (self.panels,) or len(panel_polars.polars) != self.panels:
            raise ValueError("one angle and one table per panel")
        wrapped = np.degrees(np.mod(alpha+np.pi, 2*np.pi)-np.pi)
        angles = self.stall_angles(panel_polars)
        if not np.any((wrapped > angles[:, 1]) | (wrapped < angles[:, 0])):
            return np.zeros(self.panels), False
        slope = (panel_polars.coefficients(alpha+self.delta)[0]-panel_polars.coefficients(alpha-self.delta)[0])/(2*self.delta)
        return np.maximum(0.0, -self.k*self.area*slope/self.dz**2), True

    def __call__(self, target, alpha_rad, panel_polars):
        mu, on = self.coefficients(target, alpha_rad, panel_polars)
        self.last = dict(active=int(np.sum(mu > 0)), mu_max=float(np.max(mu)) if on else 0.0, on=bool(on), deviation_percent=None)
        self.tally["calls"] += 1
        self.tally["on"] += int(bool(on))
        if self._previous_on is not None and bool(on) != self._previous_on:
            self.tally["switches"] += 1
        self._previous_on = bool(on)
        if not on or not np.any(mu > 0):
            return target
        target = np.asarray(target, dtype=float)
        out = np.empty_like(target)
        for block in self.blocks:
            a, b = block["start"], block["stop"]
            diag, sup, sub = block["bands"]
            m = mu[a:b]
            if not np.any(m > 0):
                out[a:b] = target[a:b]
                continue
            ab = np.zeros((3, b-a))
            ab[1] = 1.0-m*diag
            ab[0, 1:] = -m[:-1]*sup
            ab[2, :-1] = -m[1:]*sub
            out[a:b] = solve_banded((1, 1), ab, target[a:b])
        self.last["deviation_percent"] = self.deviation_percent(out, target)
        return out

    def deviation_percent(self, gamma, target):
        """Simonet et al.'s P per surface (module text) for the solution ``gamma`` of (I - diag(mu) L) gamma = ``target``."""
        out = []
        for block in self.blocks:
            a, b = block["start"], block["stop"]
            f, g, dz = np.asarray(target[a:b], dtype=float), np.asarray(gamma[a:b], dtype=float), self.dz[a:b]
            keep = f != 0
            out.append(float(100.0*np.sum((g[keep]-f[keep])/f[keep]*dz[keep])/np.sum(dz)))
        return out


def sawtooth(circulation, panels):
    """mean |delta^2 Gamma| / max |Gamma| over the interior of the first ``panels`` panels (a report metric, after the TU
    Delft VSM tests)."""
    g = np.asarray(circulation, dtype=float)[:panels]
    if panels < 3 or not np.any(g):
        return 0.0
    return float(np.mean(np.abs(g[2:]-2*g[1:-1]+g[:-2]))/np.max(np.abs(g)))
