"""Wake-length law: the starting vortex as a far-field dipole (paper session, 2026-10-08).

Derivation (own analysis, Section 4 of an earlier manuscript). A wake of length L closed by the starting
vortex differs from the steady semi-infinite wake by a horseshoe of strength -Gamma_0 whose bound segment
is the starting vortex at distance x behind the closure points. At the wing centre this horseshoe induces

  dw(x) = Gamma_0 [ b'/(4 pi x sqrt(x^2 + b'^2/4)) - (1 - x/sqrt(x^2 + b'^2/4)) / (pi b') ]
        -> Gamma_0 b' / (8 pi x^2)          for x >> b',

and Gamma_0 b' = 2 int_0^{b/2} Gamma dy = L/(rho U) = U S C_L / 2 for any spanwise loading (b' = Betz
spacing). A uniform downwash dw lowers the lift like a change of incidence -dw/U, so

  eps_L(x) = |dC_L|/C_L = C_La dw/(U C_L) -> C_La S / (16 pi x^2),     L_eps = sqrt(C_La S / (16 pi eps)).

For the free wake the shed vorticity of the lift build-up lies at all rows, not only at L: the same
superposition of horseshoes gives eps_shed(N) = C_La S/(16 pi) sum_m (dG_m / G_inf) / x_m^2, with
G = sum_i Gamma_i dl_{y,i} (the impulse of each released ring) and x_m the distance of row m.

Inputs: unchanged repository code at e58677c (solve_finite_vsm for the straight finite wakes and the lift
slope), core-growth/{M600,V3}-d0/result.json (free-wake histories, constant core 0.15 m, tree backend),
Table 5 (free-wake values at 80/160 chords), stored states state_0513 (M600) and state_0360 (V3).
Usage: wake_length_law.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper1/scripts/convergence"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import accel_bench as ab  # noqa: E402
import wakelib as wl  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.aero.wing import SolveSettings  # noqa: E402
from openfawe.aero.vsm_wake import solve_finite_vsm  # noqa: E402

CORE = 0.15
CHORDS = (2.5, 3.5, 5, 7, 10, 14, 20, 28, 40, 56, 80, 113, 160, 320, 1000, 3000)
# Table 5 free-wake values at the two longest wakes (lift), for the extrapolated reference (p = 2)
TABLE5 = {"V3": (40, 0.572674, 80, 0.572748), "M600": (80, 1.941550, 160, 1.942216)}
STATES = {"V3": ("v3", 360), "M600": ("m600", 513)}


def config():
    return WakeConfig(dt_s=0.02, max_rows=10, core_radius_m=CORE, convection_integrator="euler",
                      core_evolution="constant", solve=SolveSettings(solver="newton", max_iterations=60))


def finite(model, length_m, alpha=None):
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model])
    area, chord, speed = ab.reference_values(model)
    wind = np.asarray(ab.inflow(speed, ab.ALPHA[model] if alpha is None else alpha, 0.0), float)
    wing = definition.wing
    rows = [wing.trailing_edge + length_m * wind / np.linalg.norm(wind)]
    sol = solve_finite_vsm(wing, definition.polar, wind, rows, config=config())
    return sol, wing, ab.load_coefficients(sol, wing, wind, area, chord)


def betz(model):
    """Gamma_0, Betz spacing b' of the main wing and the impulse check sum(Gamma dl_y) vs U S C_L / 2."""
    area, chord, speed = ab.reference_values(model)
    sol, wing, coeff = finite(model, 1000 * chord)
    gamma = np.asarray(sol.circulation)
    parts = wl.surfaces(wing)
    main_panels = len(parts[0].area) if hasattr(parts[0], "area") else wing.panels
    dly = np.asarray(wing.dl)[:, 1]
    cp_y = np.asarray(wing.cp)[:, 1]
    g, y, d = gamma[:main_panels], cp_y[:main_panels], dly[:main_panels]
    gamma0 = float(np.abs(g).max())
    half = float(np.abs((g * np.abs(d))[y > 0]).sum())
    impulse_all = float(abs((gamma * dly).sum()))
    return dict(gamma0=gamma0, bprime=2.0 * half / gamma0, span=float(2 * np.abs(np.asarray(wing.trailing_edge)[:parts[0].node_count, 1]).max()),
                impulse_wing=float(abs((g * d).sum())), impulse_all=impulse_all, US_CL_half=speed * area * coeff["CL"] / 2.0,
                CL=coeff["CL"])


def lift_slope(model, delta=0.25):
    base = ab.ALPHA[model]
    up = finite(model, 1000 * ab.reference_values(model)[1], base + delta)[2]["CL"]
    down = finite(model, 1000 * ab.reference_values(model)[1], base - delta)[2]["CL"]
    return (up - down) / np.deg2rad(2 * delta)


def law(x, cla, area, bprime=None):
    far = cla * area / (16 * np.pi * x ** 2)
    if bprime is None:
        return far
    root = np.sqrt(x ** 2 + bprime ** 2 / 4)
    ratio = 2 * x / root - 8 * x ** 2 / bprime ** 2 * (1 - x / root)
    return far * ratio


def shed_prediction(model, cla, area, chord, speed, dt):
    """eps from the actual shed-vorticity history of the stored free-wake state (straight row positions)."""
    key, step = STATES[model]
    state = wl.read_state(key, step)
    wing, strengths = state["wing"], state["strengths"]
    dly = np.asarray(wing.dl)[:, 1]
    # strengths[0] = bound ring (Gamma^N); strengths[r], r >= 1, = ring released at step N - r + 1
    history = np.array([(strengths[step - n + 1] * dly).sum() for n in range(1, step + 1)])   # G^n, n = 1..N
    g_inf = None
    out = []
    for n_steps in range(1, step + 1):
        g = history[:n_steps]                                       # G^1..G^N
        shed = np.diff(np.r_[0.0, g])                              # dG released at steps 1..N
        # row released at step n lies (N - n + 1) U dt behind the trailing edge; closure points 0.25 c ahead of it
        x = (n_steps - np.arange(1, n_steps + 1) + 1) * speed * dt + 0.25 * chord
        out.append((n_steps, shed, x))
    return history, out


def main(path):
    result = dict(created="2026-10-08", code_commit="e58677c", core_m=CORE, note=__doc__.splitlines()[0], models={})
    for model in ("V3", "M600"):
        area, chord, speed = ab.reference_values(model)
        dt = ab.ARCHIVED_DT[model]
        entry = dict(area=area, chord=chord, speed=speed, dt=dt)
        entry["CLa"] = float(lift_slope(model))
        entry["betz"] = betz(model)
        bprime = entry["betz"]["bprime"]
        # straight finite wake
        rows = []
        for n in CHORDS:
            rows.append(dict(chords=n, **finite(model, n * chord)[2]))
        ref = rows[-1]["CL"]
        for r in rows:
            x = r["chords"] * chord + 0.25 * chord
            r["eps"] = (ref - r["CL"]) / ref
            r["law_far"] = float(law(x, entry["CLa"], area))
            r["law_exact"] = float(law(x, entry["CLa"], area, bprime))
        entry["straight"] = rows
        # free wake history
        record = json.loads((HERE / f"core-growth/{model}-d0/result.json").read_text())["records"]
        l1, f1, l2, f2 = TABLE5[model]
        f_inf = f2 + (f2 - f1) / ((l2 / l1) ** 2 - 1)
        entry["free_reference"] = dict(f_inf=f_inf, from_table5=TABLE5[model])
        history, shed = shed_prediction(model, entry["CLa"], area, chord, speed, dt)
        g_inf_ext = history[-1]
        free = []
        for rec in record[1:]:
            n = rec["step"]
            length = n * speed * dt
            x = length + 0.25 * chord
            _, dg, xs = shed[n - 1]
            eps_shed = entry["CLa"] * area / (16 * np.pi) * float(np.sum(dg / g_inf_ext / xs ** 2))
            free.append(dict(step=n, chords=length / chord, CL=rec["CL"], CD=rec["CD"], CMy=rec["CMy"],
                             eps=(f_inf - rec["CL"]) / f_inf, law_far=float(law(x, entry["CLa"], area)),
                             law_exact=float(law(x, entry["CLa"], area, bprime)), law_shed=eps_shed))
        entry["free"] = free
        entry["shed_history_G_over_Ginf"] = (history / history[-1]).tolist()
        result["models"][model] = entry
        print(model, f"CLa={entry['CLa']:.4f}/rad", "betz", {k: round(v, 5) for k, v in entry["betz"].items()}, flush=True)
        for r in rows:
            print(f"  S {r['chords']:6g}c eps={r['eps']:.3e} far={r['law_far']:.3e} exact={r['law_exact']:.3e} ratio={r['eps'] / r['law_far'] if r['law_far'] else 0:.3f}")
        for f in free:
            if f["step"] in (8, 16, 32, 45, 64, 90, 128, 180, 257) or abs(f["chords"] - round(f["chords"])) < 0.02 and round(f["chords"]) in (5, 10, 20, 40, 80):
                print(f"  F {f['chords']:6.1f}c eps={f['eps']:.3e} far={f['law_far']:.3e} shed={f['law_shed']:.3e} ratioF={f['eps'] / f['law_far']:.3f} ratioShed={f['eps'] / f['law_shed']:.3f}")
    Path(path).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
