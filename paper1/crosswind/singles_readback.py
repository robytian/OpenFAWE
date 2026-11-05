"""Read-back of the single-unit production checks and of the f and g groups of node 2 (OF-0009, 2026-10-24), against
the criteria written before the runs (contract section 8, 2026-10-22 17:34, 18:17, 18:22, 21:30 and 21:55 BST).

  retention  p1k (rings kept 4 s) against p1a (2 s), laps 1 and 2: the rotor effect (with rotors minus p1n, the same
             dt 0.01 s) must agree within 0.003 in C_Z and 10 % in C_m, lap by lap.
  dt         p1b (dt 0.02 s) against p1a (0.01 s): the rotor effect per lap (p1b minus c3s, p1a minus p1n); reported
             only, as the dt error of the pairs.
  f group    the surface-core ladder at dt 0.0025 and 0.00125 s (model A, rings free during their first step, no sheet
             refinement): the rotor effect averaged over 0.5-1 s against the no-rotor runs at the same dt
             (rdt-m600-dt00250/00125-norotors-1s); reported against the first-order dt extrapolation of the cloud
             (C_Z +0.0381, C_m +0.0270 at 0.1 m; +0.0392, +0.0267 at 0.075 m).
  g group    criterion V1 of model B: the rotor effect over 0.5-1 s at dt 0.02, 0.01 (cloud, recorded values),
             0.005 and 0.0025 s (node 2, against ta-dt0050-norotors-1s and rdt-m600-dt00250-norotors-1s). Passed if
             the differences of neighbouring levels decrease and |E(0.005) - E(0.0025)| <= 0.003 (C_Z) and
             <= 10 % of |E(0.0025)| (C_m).

Coefficients over a time window are formed step by step with q = rho V^2 / 2 from the recorded airspeed (rho 1.225,
area 32.9285 m^2, chord 1.2831 m, the moment about the body origin), the same reference values as c3_pair.per_lap.

Usage: singles_readback.py OUT.json RECEIPTS_DIR      (RECEIPTS_DIR/<run>/checks.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

RHO, AREA, CHORD = 1.225, 32.9285, 1.2831
CLOUD_V1 = {0.02: dict(CZ=-0.117, Cm=-0.029), 0.01: dict(CZ=-0.087, Cm=-0.014)}   # contract section 8, 2026-10-22
EXTRAPOLATION = {0.1: dict(CZ=0.0381, Cm=0.0270), 0.075: dict(CZ=0.0392, Cm=0.0267)}


def load(root, name):
    return json.loads((Path(root)/name/"checks.json").read_text(encoding="utf-8"))


def lap_effect(with_rotors, without, lap):
    a, b = with_rotors["per_lap"]["A"][lap-1], without["per_lap"]["A"][lap-1]
    return dict(CZ=a["force_body_coefficient_mean"][2]-b["force_body_coefficient_mean"][2],
                Cm=a["moment_body_coefficient_mean"][1]-b["moment_body_coefficient_mean"][1])


def window(data, t0=0.5, t1=1.0):
    d = data["per_step"]["A"]
    t = np.array(d["time_s"])
    keep = (t >= t0-1e-9) & (t <= t1+1e-9)
    q = 0.5*RHO*np.array(d["airspeed_m_s"])[keep]**2
    force = np.array(d["force_body_N"], dtype=float)[keep]
    moment = np.array(d["moment_origin_body_Nm"], dtype=float)[keep]
    return dict(CZ=float(np.mean(force[:, 2]/(q*AREA))), Cm=float(np.mean(moment[:, 1]/(q*AREA*CHORD))), samples=int(keep.sum()),
                t=[float(t[keep].min()), float(t[keep].max())])


def window_effect(root, rotors, reference):
    a, b = window(load(root, rotors)), window(load(root, reference))
    if a["samples"] != b["samples"]:
        raise SystemExit(f"{rotors} and {reference} differ in their samples")
    return dict(run=rotors, reference=reference, CZ=a["CZ"]-b["CZ"], Cm=a["Cm"]-b["Cm"], samples=a["samples"], window_s=a["t"])


def main(out, root):
    report = dict(script="singles_readback.py")
    p1a, p1n, p1b, p1k, c3s = (load(root, n) for n in ("p1a-m600-single-rotors-adp-dt010-4laps", "p1n-m600-single-dt010-4laps",
                                                        "p1b-m600-single-rotors-adp-dt020-4laps",
                                                        "p1k-m600-single-rotors-adp-dt010-keep400-2laps", "c3s-m600-single-frozen1-5laps"))
    for run, ref in ((p1a, p1n), (p1k, p1n), (p1b, c3s)):
        if abs(run["settings"]["dt_s"]-ref["settings"]["dt_s"]) > 1e-12 or run["settings"]["steps_per_lap"] != ref["settings"]["steps_per_lap"]:
            raise SystemExit("reference with another step")
    rows = []
    for lap in (1, 2):
        e2, e4 = lap_effect(p1a, p1n, lap), lap_effect(p1k, p1n, lap)
        dz, dm = e4["CZ"]-e2["CZ"], e4["Cm"]-e2["Cm"]
        rows.append(dict(lap=lap, keep2s=e2, keep4s=e4, CZ_difference=dz, Cm_relative_difference=dm/abs(e2["Cm"]),
                         passed=bool(abs(dz) <= 0.003 and abs(dm) <= 0.10*abs(e2["Cm"]))))
    report["retention"] = dict(criterion="|dCZ| <= 0.003 and |dCm| <= 10 % lap by lap (laps 1-2)", laps=rows,
                               verdict="passed" if all(r["passed"] for r in rows) else "failed",
                               K7=dict(p1k=p1k["K7"]["A"]["verdict"]), elapsed_h=p1k["elapsed_s"]/3600.)
    rows = []
    for lap in range(1, 5):
        e1, e2 = lap_effect(p1a, p1n, lap), lap_effect(p1b, c3s, lap)
        rows.append(dict(lap=lap, dt001=e1, dt002=e2, CZ_difference=e2["CZ"]-e1["CZ"], Cm_difference=e2["Cm"]-e1["Cm"],
                         Cm_relative_difference=(e2["Cm"]-e1["Cm"])/abs(e1["Cm"])))
    report["dt"] = dict(note="reported only; the dt error of the pairs at 0.02 s (first order)", laps=rows,
                        K7=dict(p1b=p1b["K7"]["A"]["verdict"]), last_pair=p1b["K7_pairs"]["A"][-1], elapsed_h=p1b["elapsed_s"]/3600.)
    f = []
    for name in ("sc-dt0025-n32-sc030-1s", "sc-dt0025-n32-sc015-1s", "sc-dt0025-n64-sc015-1s", "sc-dt0025-n64-sc030-1s",
                 "sc-dt00125-n64-sc015-1s", "sc-dt00125-n64-sc0075-1s"):
        reference = "rdt-m600-dt00250-norotors-1s" if "dt0025-" in name else "rdt-m600-dt00125-norotors-1s"
        row = window_effect(root, name, reference)
        row["surface_core_m"] = load(root, name)["settings"]["rotors"]["surface_core_m"]
        row["nodes"] = load(root, name)["settings"]["rotors"]["nodes_per_ring"]
        f.append(row)
    report["f_group"] = dict(note="model A at small dt, against the cloud extrapolation of model A' with refinement", runs=f,
                             extrapolation=EXTRAPOLATION)
    levels = {0.005: window_effect(root, "mb-dt0050-rb064-1s", "ta-dt0050-norotors-1s"),
              0.0025: window_effect(root, "mb-dt0025-rb064-1s", "rdt-m600-dt00250-norotors-1s")}
    e = {dt: dict(CZ=v["CZ"], Cm=v["Cm"]) for dt, v in CLOUD_V1.items()}
    e.update({dt: dict(CZ=v["CZ"], Cm=v["Cm"]) for dt, v in levels.items()})
    order = [0.02, 0.01, 0.005, 0.0025]
    steps = [abs(e[b]["CZ"]-e[a]["CZ"]) for a, b in zip(order, order[1:])]
    steps_m = [abs(e[b]["Cm"]-e[a]["Cm"]) for a, b in zip(order, order[1:])]
    decreasing = all(y < x for x, y in zip(steps, steps[1:])) and all(y < x for x, y in zip(steps_m, steps_m[1:]))
    last_z, last_m = abs(e[0.005]["CZ"]-e[0.0025]["CZ"]), abs(e[0.005]["Cm"]-e[0.0025]["Cm"])
    report["g_group_V1"] = dict(levels={str(k): v for k, v in e.items()}, node2=levels, CZ_steps=steps, Cm_steps=steps_m,
                                decreasing=decreasing, last_CZ=last_z, last_Cm_relative=last_m/abs(e[0.0025]["Cm"]),
                                verdict="passed" if decreasing and last_z <= 0.003 and last_m <= 0.10*abs(e[0.0025]["Cm"]) else "failed")
    Path(out).write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
