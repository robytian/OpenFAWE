"""Response of the closure to a uniform downwash in the straight and in the free-wake geometry (paper session).
Unchanged code e58677c: solve_finite_vsm with known_induction = -dw * n (n = lift direction) at closure and load points;
free-wake rows taken from the stored state (all rings at the current circulation)."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wake_length_law as wll
import accel_bench as ab
import wakelib as wl
from openfawe.aero.vsm_wake import solve_finite_vsm

out = {}
for model, key, n_state, rows_n in (("M600", "m600", 513, 257), ("V3", "v3", 360, 180)):
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model]); wing, polar = definition.wing, definition.polar
    area, chord, speed = ab.reference_values(model)
    wind = np.asarray(ab.inflow(speed, ab.ALPHA[model], 0.0), float)
    d = wind / speed; n = np.cross(d, [0.0, 1.0, 0.0])
    state = wl.read_state(key, n_state)
    free_rows = list(state["rows"][2:2 + rows_n])
    straight_rows = [wing.trailing_edge + k * speed * ab.ARCHIVED_DT[model] * d for k in range(1, rows_n + 1)]
    res = {}
    for name, rows in (("straight", straight_rows), ("free", free_rows)):
        vals = []
        for dw in (0.0, 1e-3 * speed):
            known = -dw * n
            sol = solve_finite_vsm(wing, polar, wind, rows, config=wll.config(), known_induction=known, known_load_induction=known)
            vals.append(ab.load_coefficients(sol, wing, wind, area, chord))
        cl0, cl1 = vals[0]["CL"], vals[1]["CL"]
        res[name] = dict(CL0=cl0, CL1=cl1, CLa_eff=(cl0 - cl1) / 1e-3, CD0=vals[0]["CD"], CD1=vals[1]["CD"],
                         CMy0=vals[0]["CMy"], CMy1=vals[1]["CMy"])
        print(model, name, {k: round(v, 6) for k, v in res[name].items()}, flush=True)
    res["ratio_free_over_straight"] = res["free"]["CLa_eff"] / res["straight"]["CLa_eff"]
    print(model, "ratio", res["ratio_free_over_straight"])
    out[model] = res
json.dump(out, open("response-test.json", "w"), indent=1)
