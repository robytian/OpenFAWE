"""Velocity induced at the closure points by the far end of the wake (paper session, 2026-10-08).
Unchanged code e58677c: vsm_wake_influence(wing, rows) is the closure operator of rings that all carry the
current circulation. The difference between a wake ending at x and a long wake of the same kind is the field of
the starting-vortex end. It is split into the downwash dw (along -n) and the streamwise component du, and the
lift deficit is estimated as eps = C_w <dw>/(U C_L) - 2 <du>/U (incidence change and dynamic pressure),
with <.> weighted by the panel impulse Gamma_i dl_y,i."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wake_length_law as wll
import accel_bench as ab
import wakelib as wl
from openfawe.aero.vsm_wake import vsm_wake_influence, solve_finite_vsm

resp = json.load(open("response-test.json"))
swap = json.load(open("geometry-swap.json"))
out = {}
for model, key, long_step, steps in (("M600", "m600", 513, (128, 257)), ("V3", "v3", 360, (90, 180))):
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model]); wing, polar = definition.wing, definition.polar
    area, chord, speed = ab.reference_values(model); dt = ab.ARCHIVED_DT[model]
    wind = np.asarray(ab.inflow(speed, ab.ALPHA[model], 0.0), float); d = wind / speed; n = np.cross(d, [0.0, 1.0, 0.0])
    cfg = wll.config()
    long_free = list(wl.read_state(key, long_step)["rows"][2:])
    cw = resp[model]["straight"]["CLa_eff"]
    for N in steps:
        free = list(wl.read_state(key, N)["rows"][2:])
        straight = [wing.trailing_edge + m * speed * dt * d for m in range(1, N + 1)]
        long_straight = [wing.trailing_edge + 3000 * chord * d]
        res = {}
        for name, short, long in (("free", free, long_free), ("straight", straight, long_straight)):
            sol = solve_finite_vsm(wing, polar, wind, short, config=cfg)
            gamma = np.asarray(sol.circulation)
            cl = ab.load_coefficients(sol, wing, wind, area, chord)["CL"]
            dv = np.einsum("ijk,j->ik", vsm_wake_influence(wing, short, config=cfg) - vsm_wake_influence(wing, long, config=cfg), gamma)
            weight = gamma * np.asarray(wing.dl)[:, 1]
            weight = weight / weight.sum()
            dw = float(weight @ (-(dv @ n))); du = float(weight @ (dv @ d))
            res[name] = dict(dw_over_U=dw / speed, du_over_U=du / speed, eps_w=cw * dw / (speed * cl), eps_u=-2 * du / speed,
                             eps_sum=cw * dw / (speed * cl) - 2 * du / speed, eps_actual=swap[f"{model}-{N}"][f"{name}-current"]["eps"])
        out[f"{model}-{N}"] = res
        for name, r in res.items():
            print(model, N, name, {k: f"{v:.3e}" for k, v in r.items()}, flush=True)
json.dump(out, open("far-end-velocity.json", "w"), indent=1)
