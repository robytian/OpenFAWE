"""Compare the meridional planes of two runs (Sec. III E, Fig. 17): the c3s run without rotors and the r1 run with the eight
rotors, both with rows and rings frozen beyond one lap, after four laps. Input are the ``*_meridional.npz`` files written by
``wake_figures.py`` (display core 3 m, grid 2 m, coordinates in loop diameters D from the path centre); the crossing positions
are those of the complete free wake C1a (results/c1a_meridional_deficit.json), read with the same window.

Written: OUT (json) with, per run, the deficit at each crossing window (largest value, area above 0.1 and 0.3), and in bands
of 0.5 D between 0 and 2 D downstream the mean, root mean square, largest and smallest deficit and the area where the flow is
faster than the wind by more than 0.1 U_inf, plus the location of the largest and the smallest value over 0 < x < 2 D.

Usage: meridional_compare.py LABEL=NPZ LABEL=NPZ OUT.json [--crossings results/c1a_meridional_deficit.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent


def stats(npz, crossings, half_window=0.08):
    d = np.load(npz)
    x, z, df = d["x"], d["z"], d["deficit"]
    D = float(d["diameter"])
    cell = float((x[0, 1]-x[0, 0])*(z[1, 0]-z[0, 0]))*D*D
    out = dict(source=str(npz), display_core_m=float(d["vis_core"]), cell_m2=cell, crossings=[], bands=[])
    for c in crossings:
        half = (z < 0) if c["half"] == "bottom" else (z > 0)
        sel = half & (np.abs(x-c["x_D"]) < half_window)
        v = df[sel]
        out["crossings"].append(dict(half=c["half"], x_D=c["x_D"], deficit_max=float(v.max()),
                                     area_deficit_above_0p1_m2=float((v > 0.1).sum()*cell),
                                     area_deficit_above_0p3_m2=float((v > 0.3).sum()*cell)))
    for lo in (0.0, 0.5, 1.0, 1.5):
        sel = (x > lo) & (x < lo+0.5)
        v = df[sel]
        out["bands"].append(dict(x_D=[lo, lo+0.5], mean=float(v.mean()), rms=float(np.sqrt((v**2).mean())),
                                 deficit_max=float(v.max()), deficit_min=float(v.min()),
                                 area_faster_than_wind_by_0p1_m2=float((v < -0.1).sum()*cell)))
    sel = (x > 0) & (x < 2)
    for name, pick in (("largest", np.nanargmax), ("smallest", np.nanargmin)):
        k = pick(np.where(sel, df, np.nan))
        i, j = np.unravel_index(k, df.shape)
        out[f"{name}_deficit"] = dict(value=float(df[i, j]), x_D=float(x[i, j]), z_D=float(z[i, j]))
    v = df[sel]
    out["whole_0_2D"] = dict(mean=float(v.mean()), rms=float(np.sqrt((v**2).mean())),
                             mean_positive_part=float(np.clip(v, 0, None).mean()),
                             mean_negative_part=float(np.clip(v, None, 0).mean()))
    return out


def main(argv):
    parser = argparse.ArgumentParser(prog="meridional_compare.py")
    parser.add_argument("runs", nargs=2, help="LABEL=NPZ")
    parser.add_argument("out")
    parser.add_argument("--crossings", default=str(TASK/"results"/"c1a_meridional_deficit.json"))
    args = parser.parse_args(argv)
    crossings = json.loads(Path(args.crossings).read_text(encoding="utf-8"))["crossings"]
    result = dict(window="|x - x_crossing| < 0.08 D, half plane of the crossing (z < 0 bottom, z > 0 top)",
                  crossings_from=str(args.crossings), runs={})
    for item in args.runs:
        label, path = item.split("=", 1)
        result["runs"][label] = stats(path, crossings)
    Path(args.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    for label, r in result["runs"].items():
        print(label, "crossings", [(c["half"][0], round(c["x_D"], 2), round(c["deficit_max"], 3), round(c["area_deficit_above_0p1_m2"]))
                                   for c in r["crossings"]])
        print(label, "bands", [(b["x_D"][0], round(b["mean"], 4), round(b["rms"], 4), round(b["deficit_max"], 2), round(b["deficit_min"], 2))
                               for b in r["bands"]])
        print(label, "extremes", r["largest_deficit"], r["smallest_deficit"], r["whole_0_2D"])


if __name__ == "__main__":
    main(sys.argv[1:])
