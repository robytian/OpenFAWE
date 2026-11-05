"""Roll-up and descent of the free wake (paper session, 2026-10-08).

Reads the stored free-wake states (unchanged code e58677c, runs p1-*-planes-wakelength) and evaluates,
for the trailing vorticity of the main wing on each half-span, the classical inviscid invariants of
Betz [Betz 1932; Spalart 1998]:

  Gamma_h(s)  circulation of the half wake (sum of the trailing-filament strengths),
  y_c(s)      lateral centroid  sum(g*y)/sum(g)        -> invariant, equal to int Gamma dy / Gamma_0,
  n_c(s)      vertical centroid sum(g*n)/sum(g)        -> descends with w = Gamma_h / (2 pi b'), b' = 2 y_c,
  R2(s)       second moment about the centroid          -> approximately invariant (Betz),
  f_r(s)      fraction of Gamma_h carried by filaments within r of the tip filament (roll-up progress).

The trailing filaments are classified by their node column (Lagrangian identity): a column belongs to the
right half if its trailing-edge node has y > 0. Positions are midpoints of the filament between rows k and
k+1; s is the distance downstream of the trailing-edge centroid of all surfaces along the free stream,
n the coordinate normal to the stream (lift direction), y the spanwise coordinate.

Usage: rollup.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wakelib as wl  # noqa: E402

STEPS = {"m600": (128, 257, 513), "v3": (90, 180, 360)}


def frame(key, state):
    stream = wl.freestream(key, 1.0)
    direction = stream / np.linalg.norm(stream)
    axis_n = np.cross(direction, [0.0, 1.0, 0.0])
    te_centre = state["rows"][1].mean(axis=0)
    return direction, axis_n, te_centre


def analyse(key, step):
    state = wl.read_state(key, step)
    wing, rows, strengths, chord, speed = state["wing"], state["rows"], state["strengths"], state["chord"], state["speed"]
    parts = wl.surfaces(wing)
    main = parts[0].node_count                                  # columns of the main wing come first
    width = rows.shape[1]
    trailing = wl.trailing_strength(strengths, wing.panel_edges, width)   # (depth, width), interval k: rows k..k+1
    direction, axis_n, te_centre = frame(key, state)
    te_wing = rows[1, :main]
    right = np.where(te_wing[:, 1] > 1e-9)[0]
    tip = right[np.argmax(te_wing[right, 1])]
    span = 2.0 * te_wing[:, 1].max()
    # bound circulation of the main wing (ring 0) and Betz centroid from it
    panel_edges = np.asarray(wing.panel_edges)
    main_panels = np.where(panel_edges[:, 1] < main)[0]
    bound = strengths[0, main_panels]
    gamma0 = float(np.abs(bound).max())
    out = dict(model=key, step=step, chord=chord, speed=speed, span=span, gamma0=gamma0, core=float(state["radii"].max()),
               state=str(state["path"]), rows=[])
    sign = None
    for k in range(1, len(strengths)):                          # wake intervals: TE row 1 .. last row
        g = trailing[k, right]
        mid = 0.5 * (rows[k, right] + rows[k + 1, right])
        rel = mid - te_centre
        s = float(np.mean(rel @ direction))
        y = mid[:, 1]
        n = rel @ axis_n
        total = g.sum()
        if sign is None:
            sign = np.sign(total)
        yc, nc = (g * y).sum() / total, (g * n).sum() / total
        r2 = (g * ((y - yc) ** 2 + (n - nc) ** 2)).sum() / total
        # roll-up progress: share of Gamma_h within radius r of the tip filament (r in chords)
        t = list(right).index(tip)
        dist = np.hypot(y - y[t], n - n[t]) / chord
        frac = {f"{r:g}": float(g[dist <= r].sum() / total) for r in (0.5, 1.0, 2.0)}
        # filament positions of the right half in the cross plane (for drawing the spiral)
        out["rows"].append(dict(k=k, s=s, gamma_h=float(total), yc=float(yc), nc=float(nc), R2=float(r2), frac=frac,
                                ytip=float(y[t]), ntip=float(n[t]), nmin=float(n.min())))
    out["betz_yc"] = out["rows"][0]["yc"]
    return out, dict(rows=rows, right=right, trailing=trailing, direction=direction, axis_n=axis_n, te_centre=te_centre)


def descent(entry, s_lo, s_hi):
    rows = [r for r in entry["rows"] if s_lo <= r["s"] <= s_hi]
    s = np.array([r["s"] for r in rows])
    nc = np.array([r["nc"] for r in rows])
    slope, intercept = np.polyfit(s, nc, 1)
    gh = np.mean([r["gamma_h"] for r in rows])
    bprime = 2.0 * np.mean([r["yc"] for r in rows])
    w_pair = abs(gh) / (2.0 * np.pi * bprime)
    return dict(s_lo=s_lo, s_hi=s_hi, slope=float(slope), w_over_U=float(-slope), pair_w_over_U=float(w_pair / entry["speed"]),
                ratio=float(-slope * entry["speed"] / w_pair), gamma_h=float(gh), bprime=float(bprime))


if __name__ == "__main__":
    result = dict(created="2026-10-08", code_commit="e58677c (states written by the accel runs)", note=__doc__.splitlines()[0],
                  cases={})
    for key, steps in STEPS.items():
        for step in steps:
            entry, _ = analyse(key, step)
            b, c = entry["span"], entry["chord"]
            length = entry["rows"][-1]["s"]
            # descent fitted over the part of the wake between 1 span and 2 spans before the starting vortex
            windows = {}
            for lo, hi in ((1.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 6.0)):
                if hi * b < length - 2.0 * b:
                    windows[f"{lo:g}-{hi:g} spans"] = descent(entry, lo * b, hi * b)
            entry["descent"] = windows
            result["cases"][f"{key}-{step}"] = entry
            yc = np.array([r["yc"] for r in entry["rows"]])
            s = np.array([r["s"] for r in entry["rows"]])
            inside = s < max(length - 2.0 * b, 0.5 * length)
            print(key, step, f"L={length / c:.1f}c b={b / c:.2f}c Gamma0={entry['gamma0']:.3f}",
                  f"Gh(TE)={entry['rows'][0]['gamma_h']:.3f} betz yc/(b/2)={entry['betz_yc'] / (b / 2):.4f}",
                  f"yc range (excl. last 2 spans or half) {yc[inside].min() / (b / 2):.4f}..{yc[inside].max() / (b / 2):.4f}", flush=True)
            for name, d in windows.items():
                print("   ", name, f"w/U={d['w_over_U']:.5f} pair={d['pair_w_over_U']:.5f} ratio={d['ratio']:.3f}")
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1), encoding="utf-8")
