"""Impulse-weighted centroid of the shed vorticity of the main wing in the stored free-wake states (paper session,
2026-10-08). Shed filament of row k: strength of ring k-1 minus ring k (ring N closed by zero), weighted by its
spanwise projection dl_y; positions relative to the trailing-edge centroid of the main wing, x along the free stream,
n normal to it (lift direction). Usage: shed_centroid.py OUT.json"""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wakelib as wl
out = {}
for key, steps in (("m600", (128, 257, 513)), ("v3", (90, 180, 360))):
    for N in steps:
        s = wl.read_state(key, N); rows, w, st = s["rows"], s["wing"], s["strengths"]
        d = wl.freestream(key, 1.0); d /= np.linalg.norm(d); n = np.cross(d, [0, 1, 0])
        main = wl.surfaces(w)[0].node_count; npan = len(wl.surfaces(w)[0].area)
        pe = np.asarray(w.panel_edges)[:npan]; a, b = pe[:, 0], pe[:, 1]
        te_c = rows[1, :main].mean(axis=0)
        tot = xs = ns = 0.0
        for k in range(2, N + 2):
            newer = st[k - 1][:npan]; older = st[k][:npan] if k < len(st) else np.zeros(npan)
            imp = (newer - older) * (rows[k, b, 1] - rows[k, a, 1])
            rel = 0.5 * (rows[k, a] + rows[k, b]) - te_c
            tot += imp.sum(); xs += (imp * (rel @ d)).sum(); ns += (imp * (rel @ n)).sum()
        out[f"{key}-{N}"] = dict(x_over_c=xs / tot / s["chord"], n_over_c=ns / tot / s["chord"], n_over_x=ns / xs)
        print(key, N, {k: round(v, 4) for k, v in out[f"{key}-{N}"].items()})
json.dump(out, open(sys.argv[1], "w"), indent=1)
