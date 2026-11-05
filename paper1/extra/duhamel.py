"""Duhamel superposition of the straight finite-wake response over the shed history (paper session, 2026-10-08).

The wake of the impulsively started wing is a superposition of horseshoes: one of the current strength at
the wing and, for every wake row m, one of strength -dG_m starting at that row (dG_m = shed impulse of the
row). To first order the lift deficit is therefore

  eps_F(N) = sum_m (dG_m / G_inf) * eps_S(x_m),

where eps_S(x) is the lift deficit of the steady solution with a straight wake that ends at x (all rings at
the current circulation), computed here with the unchanged code (solve_finite_vsm, e58677c) on the row grid
x_m = m U dt, and dG_m is read from the stored free-wake state. Compared with the free-wake history of
core-growth/*-d0 (constant core 0.15 m). eps_S is referenced to the 3000-chord value corrected by the far-field
law (eps(3000 c) ~ 3e-8 and 3e-7).
Usage: duhamel.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wake_length_law as wll  # noqa: E402
import accel_bench as ab  # noqa: E402  (path set by wake_length_law)
import wakelib as wl  # noqa: E402

STEPS = {"V3": 180, "M600": 257}


def main(path):
    law = json.loads((HERE / "wake-length-law.json").read_text())["models"]
    result = dict(created="2026-10-08", code_commit="e58677c", note=__doc__.splitlines()[0], models={})
    for model in ("V3", "M600"):
        area, chord, speed = ab.reference_values(model)
        dt = ab.ARCHIVED_DT[model]
        cla = law[model]["CLa"]
        x_ref = 3000 * chord
        ref = wll.finite(model, x_ref)[2]
        f_inf = {k: ref[k] for k in ("CL", "CD", "CMy")}
        f_inf["CL"] = ref["CL"] * (1 + wll.law(x_ref + 0.25 * chord, cla, area))
        steps = STEPS[model]
        response = []
        for m in range(1, steps + 1):
            c = wll.finite(model, m * speed * dt)[2]
            response.append(dict(m=m, chords=m * speed * dt / chord, CL=c["CL"], CD=c["CD"], CMy=c["CMy"]))
        eps_s = np.array([(f_inf["CL"] - r["CL"]) / f_inf["CL"] for r in response])
        dcd_s = np.array([(r["CD"] - f_inf["CD"]) / f_inf["CD"] for r in response])
        dcm_s = np.array([(r["CMy"] - f_inf["CMy"]) for r in response])
        # shed history from the stored state
        key, n_state = wll.STATES[model]
        state = wl.read_state(key, n_state)
        dly = np.asarray(state["wing"].dl)[:, 1]
        strengths = state["strengths"]
        history = np.array([(strengths[n_state - n + 1] * dly).sum() for n in range(1, n_state + 1)])
        g_inf = history[-1]
        record = json.loads((HERE / f"core-growth/{model}-d0/result.json").read_text())["records"]
        f_free_inf = law[model]["free_reference"]["f_inf"]
        rows = []
        for rec in record[1:]:
            n = rec["step"]
            shed = np.diff(np.r_[0.0, history[:n]]) / g_inf          # released at steps 1..n
            rows_m = n - np.arange(1, n + 1) + 1                      # row index m of each release
            pred = float(np.sum(shed * eps_s[rows_m - 1]))
            pred_cd = float(np.sum(shed * dcd_s[rows_m - 1]))
            pred_cm = float(np.sum(shed * dcm_s[rows_m - 1]))
            rows.append(dict(step=n, chords=n * speed * dt / chord, eps_free=(f_free_inf - rec["CL"]) / f_free_inf,
                             eps_duhamel=pred, eps_straight_same_L=float(eps_s[n - 1]),
                             eps_far=float(wll.law(n * speed * dt + 0.25 * chord, cla, area)),
                             dCD_rel_straight_same_L=float(dcd_s[n - 1]), dCD_rel_duhamel=pred_cd, dCMy_duhamel=pred_cm,
                             CL=rec["CL"], CD=rec["CD"], CMy=rec["CMy"]))
        result["models"][model] = dict(f_inf_straight=f_inf, f_inf_free_CL=f_free_inf, response=response, rows=rows)
        print(model, "f_inf", f_inf, flush=True)
        for r in rows:
            if r["step"] % (16 if model == "V3" else 32) == 0 or r["step"] in (45, 90, 180, 257):
                print(f"  {r['chords']:6.1f}c free={r['eps_free']:.3e} duhamel={r['eps_duhamel']:.3e} "
                      f"ratio={r['eps_free'] / r['eps_duhamel']:.3f} straight={r['eps_straight_same_L']:.3e} far={r['eps_far']:.3e}")
    Path(path).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
