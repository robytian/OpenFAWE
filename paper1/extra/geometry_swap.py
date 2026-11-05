"""Free-wake lift deficit: geometry versus circulation history (paper session, 2026-10-08).
Unchanged code e58677c: history_vsm_induction + solve_finite_vsm reproduce one free-wake step from a stored
state; the wake rows are then replaced by straight rows (m U dt along the stream) and/or the ring strengths by
the current circulation, to separate the effect of wake deformation from that of the shed-vorticity history."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wake_length_law as wll
import accel_bench as ab
import wakelib as wl
from openfawe.aero.vsm_wake import solve_finite_vsm, history_vsm_induction

REF = {"free": {"M600": 1.9424380, "V3": 0.5727727},        # extrapolated free-wake reference (Table 5, p = 2)
       "straight": {"M600": 1.9479760, "V3": 0.5738161}}    # 3000-chord straight wake, law-corrected (duhamel.json)
out = {}
for model, key, steps in (("M600", "m600", (128, 257)), ("V3", "v3", (90, 180))):
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model]); wing, polar = definition.wing, definition.polar
    area, chord, speed = ab.reference_values(model)
    dt = ab.ARCHIVED_DT[model]
    wind = np.asarray(ab.inflow(speed, ab.ALPHA[model], 0.0), float)
    d = wind / speed
    cfg = wll.config()
    for N in steps:
        st = wl.read_state(key, N)
        free = st["rows"][2:]                      # free rows 1..N
        strengths = st["strengths"]
        straight = np.array([wing.trailing_edge + m * speed * dt * d for m in range(1, N + 1)])
        current = np.repeat(strengths[1][None], N - 1, axis=0)
        res = {}
        ratio = strengths[2:N + 1] / strengths[1][None]          # relative history of every ring
        for gname, rows in (("free", free), ("straight", straight)):
            sol = solve_finite_vsm(wing, polar, wind, list(rows), config=cfg, initial=strengths[1])
            c = ab.load_coefficients(sol, wing, wind, area, chord)
            res[f"{gname}-current"] = dict(CL=c["CL"], CD=c["CD"], CMy=c["CMy"])
            guess = strengths[1].copy()
            for it in range(8):                                    # history scaled to the solved circulation
                known, known_load = history_vsm_induction(wing, list(rows), ratio * guess[None], None, cfg)
                sol = solve_finite_vsm(wing, polar, wind, list(rows[:1]), config=cfg, known_induction=known,
                                       known_load_induction=known_load, initial=guess)
                change = float(np.abs(sol.circulation - guess).max()); guess = np.asarray(sol.circulation).copy()
                if change < 1e-12:
                    break
            c = ab.load_coefficients(sol, wing, wind, area, chord)
            res[f"{gname}-history"] = dict(CL=c["CL"], CD=c["CD"], CMy=c["CMy"], fixed_point_change=change, iterations=it + 1)
        for k, v in res.items():
            v["eps"] = (REF[k.split("-")[0]][model] - v["CL"]) / REF[k.split("-")[0]][model]
        out[f"{model}-{N}"] = res
        print(model, N, {k: (round(v["CL"], 6), f"{v['eps']:.3e}") for k, v in res.items()}, flush=True)
json.dump(out, open("geometry-swap.json", "w"), indent=1)
