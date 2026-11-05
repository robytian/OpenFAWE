"""Evidence of Sec. III E (OF-0009): the effect of the rotors on the wing over the first second of flight for every ring
treatment, time step and parameter that was run, from the receipts (cloud runs of 2026-10-22 and 2026-10-24 and node-2
runs), in one table for the figures and the paper.

The effect is the run with rotors minus the run without rotors at the same time step and backend, averaged over 0.5 to
1.0 s of flight, for C_Z, C_m and C_X in body axes (the reference values of c3_pair.per_lap: q = rho V^2/2 from the
recorded airspeed, area 32.9285 m^2, chord 1.2831 m, moment about the body origin). Each row carries the settings read
from the receipt (time step, treatment, surface core, refinement, ring core, nodes, kept steps, rotor speed) and the
mean total thrust and shaft power over the same window.

Treatments (from the receipt settings): free rings (free convection for all kept steps, no bound core), tube A (free
during the first step, then drifting with the wind), tube A' (drifting from release), tube A'' (drifting with the wind
plus gamma/2 along the disc axis; with ambient spacing the released circulation uses the ambient inflow), model B (free
rings, bound vortex regularised over r_b for their convection).

Usage: rotor_evidence.py OUT.json RECEIPT_DIR [RECEIPT_DIR ...]     (each RECEIPT_DIR holds <run>/checks.json)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

RHO, AREA, CHORD = 1.225, 32.9285, 1.2831


def window(data, t0=0.5, t1=1.0):
    d = data["per_step"]["A"]
    t = np.asarray(d["time_s"])
    keep = (t >= t0-1e-9) & (t <= t1+1e-9)
    q = 0.5*RHO*np.asarray(d["airspeed_m_s"])[keep]**2
    f = np.asarray(d["force_body_N"], float)[keep]
    m = np.asarray(d["moment_origin_body_Nm"], float)[keep]
    out = dict(CZ=float(np.mean(f[:, 2]/(q*AREA))), CX=float(np.mean(f[:, 0]/(q*AREA))), Cm=float(np.mean(m[:, 1]/(q*AREA*CHORD))),
               samples=int(keep.sum()))
    if d.get("rotor_thrust_total_N") and len(d["rotor_thrust_total_N"]) == len(t):
        out["thrust_N"] = float(np.mean(np.asarray(d["rotor_thrust_total_N"], float)[keep]))
        out["shaft_W"] = float(np.mean(np.asarray(d["rotor_shaft_power_W"], float)[keep]))
    return out


def treatment(r):
    if r.get("ring_bound_core_m"):
        return "model B"
    free, keep = r.get("free_convection_ring_steps"), r.get("retained_ring_steps")
    if free == 0:
        if r.get("ring_sheet_drift"):
            return "tube A'' ambient" if r.get("ring_ambient_spacing") else "tube A''"
        return "tube A'"
    if free == 1:
        return "tube A"
    return "free rings"


def main(out, dirs):
    runs = {}
    for d in dirs:
        for p in sorted(glob.glob(str(Path(d)/"*"/"checks.json"))):
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            if data.get("status") != "complete" or not data["settings"].get("single"):
                continue
            s = data["settings"]
            if s["steps"] > s["steps_per_lap"]/2:          # first-second runs only
                continue
            runs[Path(p).parent.name+"@"+Path(d).name] = data
    refs = {(round(v["settings"]["dt_s"], 7), v["settings"]["backend"]): (k, v) for k, v in runs.items() if not v["settings"].get("rotors")}
    rows = []
    for name, data in sorted(runs.items()):
        s = data["settings"]
        r = s.get("rotors")
        if not r:
            continue
        key = (round(s["dt_s"], 7), s["backend"])
        if key not in refs:
            rows.append(dict(run=name, missing_reference=list(key)))
            continue
        ref_name, ref = refs[key]
        a, b = window(data), window(ref)
        if a["samples"] != b["samples"]:
            raise SystemExit(f"{name}: samples differ from {ref_name}")
        rows.append(dict(run=name, reference=ref_name, dt_s=s["dt_s"], backend=s["backend"], treatment=treatment(r),
                         surface_core_m=r.get("surface_core_m"), overlap=r.get("ring_surface_overlap"), ring_core_m=r.get("ring_core_m"),
                         nodes=r.get("nodes_per_ring"), kept_steps=r.get("retained_ring_steps"), omega_rad_s=r.get("omega_rad_s"),
                         bound_core_m=r.get("ring_bound_core_m"), dCZ=a["CZ"]-b["CZ"], dCm=a["Cm"]-b["Cm"], dCX=a["CX"]-b["CX"],
                         thrust_N=a.get("thrust_N"), shaft_W=a.get("shaft_W"), samples=a["samples"]))
    Path(out).write_text(json.dumps(dict(script="rotor_evidence.py", window_s=[0.5, 1.0], rows=rows), indent=1)+"\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    for row in main(sys.argv[1], sys.argv[2:]):
        if "missing_reference" in row:
            print(row["run"], "no reference", row["missing_reference"])
            continue
        print(f"{row['run']:<52} {row['treatment']:<16} dt {row['dt_s']:.5f} sc {row['surface_core_m']} ov {row['overlap']} "
              f"nodes {row['nodes']} keep {row['kept_steps']} rc {row['ring_core_m']} om {row['omega_rad_s']} "
              f"dCZ {row['dCZ']:+.4f} dCm {row['dCm']:+.4f} dCX {row['dCX']:+.4f} T {row['thrust_N'] or 0:.0f}")
