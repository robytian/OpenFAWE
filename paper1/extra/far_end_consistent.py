"""Eq. (25) against the free-geometry deficit with one consistent reference (paper session, 2026-10-08).
far-end-velocity.json takes the field of the far end relative to the longest stored free wake (M600 513 steps, V3 360 steps);
here the deficit of the free geometry with all vortex rings at the current circulation is taken relative to the same long
free geometry (solve_finite_vsm with all its rows, unchanged code e58677c), so that both sides share the reference."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wake_length_law as wll
import accel_bench as ab
import wakelib as wl
from openfawe.aero.vsm_wake import solve_finite_vsm
swap = json.load(open("geometry-swap.json")); far = json.load(open("far-end-velocity.json"))
out = {}
for model, key, long_step, steps in (("M600", "m600", 513, (128, 257)), ("V3", "v3", 360, (90, 180))):
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model]); wing, polar = definition.wing, definition.polar
    area, chord, speed = ab.reference_values(model)
    wind = np.asarray(ab.inflow(speed, ab.ALPHA[model], 0.0), float)
    rows = list(wl.read_state(key, long_step)["rows"][2:])
    cl_long = ab.load_coefficients(solve_finite_vsm(wing, polar, wind, rows, config=wll.config()), wing, wind, area, chord)["CL"]
    for N in steps:
        cl = swap[f"{model}-{N}"]["free-current"]["CL"]
        eps = (cl_long - cl) / cl_long
        pred = far[f"{model}-{N}"]["free"]["eps_sum"]
        out[f"{model}-{N}"] = dict(CL_long_current=cl_long, CL_current=cl, eps_consistent=eps, eps_eq25=pred, ratio=pred / eps)
        print(model, N, out[f"{model}-{N}"], flush=True)
json.dump(out, open("far-end-consistent.json", "w"), indent=1)
