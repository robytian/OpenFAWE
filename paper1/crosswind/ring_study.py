"""Read-back of the rotor-ring treatment runs rs1-rs8 (Sec. III E, contract section 8, 2026-10-20): one lap of the single
M600 with the eight rotors, the settings of r1 except one parameter each, against the first lap of r1 (rings all kept, ring
core = wake core 0.15 m, surface core 0.3 m, 16 nodes), and against the first lap of c3s (no rotors) for the size of the
rotor effect itself.

Per run, over lap 1: lap means of the wing and tail body-axis coefficients C_X, C_Z, C_m (pitch), the aerodynamic power,
the rotor shaft power and thrust, the ring nodes within the surface core of a load point; the change against r1, also as a
fraction of the rotor effect (r1 minus c3s) for C_X, C_Z and C_m; per step, the largest and the root-mean-square change of
C_Z against r1 at the same step, and the step-to-step scatter of C_Z (root mean square of its second difference, which the
smooth motion keeps near zero and the ring nodes inside the surface core raise; steps after the first 50, every step
needed).

Inputs: the r1 and c3s ``history.json`` (every step in ``records``) and the receipt folder of each rs run (``checks.json``
written by c3_pair.py; one-lap runs keep every step in ``per_step``).

Written: OUT.json and a markdown table on standard output.

Usage: ring_study.py OUT.json --r1 R1_HISTORY --c3s C3S_HISTORY RS_DIR [RS_DIR ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AREA_M2, CHORD_M, RHO = 32.9285, 1.2831, 1.225


def coefficients(force, moment, airspeed):
    q = 0.5*RHO*np.asarray(airspeed, dtype=float)**2
    return np.asarray(force, dtype=float)/(q[:, None]*AREA_M2), np.asarray(moment, dtype=float)/(q[:, None]*AREA_M2*CHORD_M)


def history_lap1(path):
    h = json.loads(Path(path).read_text(encoding="utf-8"))
    n = int(h["settings"]["steps_per_lap"])
    recs = [r for r in h["records"] if 1 <= int(r["step"]) <= n]
    body = [r["bodies"]["A"] for r in recs]
    cf, cm = coefficients([b["force_body_N"] for b in body], [b["moment_origin_body_Nm"] for b in body], [b["airspeed_m_s"] for b in body])
    return dict(step=np.array([int(r["step"]) for r in recs]), CZ=cf[:, 2], per_lap=h["per_lap"]["A"][0], settings=h["settings"],
                elapsed_s=float(np.sum(h["step_times_s"][1:n+1])) if h.get("step_times_s") else None)


def receipt_lap1(directory):
    d = json.loads((Path(directory)/"checks.json").read_text(encoding="utf-8"))
    n = int(d["settings"]["steps_per_lap"])
    a = d["per_step"]["A"]
    step = np.asarray(a["step"])
    keep = (step >= 1) & (step <= n)
    cf, cm = coefficients(np.asarray(a["force_body_N"])[keep], np.asarray(a["moment_origin_body_Nm"])[keep], np.asarray(a["airspeed_m_s"])[keep])
    return dict(step=step[keep], CZ=cf[:, 2], per_lap=d["per_lap"]["A"][0], settings=d["settings"], elapsed_s=float(d["elapsed_s"]),
                status=d["status"], name=Path(directory).name)


def lap_means(per_lap):
    c, m = per_lap["force_body_coefficient_mean"], per_lap["moment_body_coefficient_mean"]
    out = dict(CX=c[0], CZ=c[2], Cm=m[1], lift_N=per_lap["lift_body_z_mean_N"], aero_power_W=per_lap["aero_power_mean_W"])
    if "rotor_shaft_power_W" in per_lap:
        out.update(shaft_power_W=per_lap["rotor_shaft_power_W"]["mean"], thrust_N=per_lap["rotor_thrust_total_N"]["mean"],
                   ring_nodes_in_core_mean=per_lap["ring_nodes_within_surface_core"]["mean"],
                   ring_nodes_in_core_max=per_lap["ring_nodes_within_surface_core"]["max"])
    return out


START_STEPS = 50   # the impulsive start dominates the second difference over the first steps (c3s: 0.1 there, 9e-6 after)


def scatter(cz, step):
    keep = np.asarray(step) > START_STEPS
    cz = np.asarray(cz)[keep]
    if len(cz) < 3 or np.any(np.diff(np.asarray(step)[keep]) != 1):
        return None
    return float(np.sqrt(np.mean(np.diff(cz, 2)**2)))


def setting_label(s):
    r = s.get("rotors") or {}
    keys = ("retained_ring_steps", "ring_core_m", "surface_core_m", "nodes_per_ring", "ring_bound_core_m", "free_convection_ring_steps",
            "ring_surface_overlap")
    return {k: r.get(k) for k in keys}


def main(argv):
    parser = argparse.ArgumentParser(prog="ring_study.py")
    parser.add_argument("out")
    parser.add_argument("--r1", required=True)
    parser.add_argument("--c3s", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    r1, c3s = history_lap1(args.r1), history_lap1(args.c3s)
    ref, base = lap_means(r1["per_lap"]), lap_means(c3s["per_lap"])
    effect = {k: ref[k]-base[k] for k in ("CX", "CZ", "Cm")}
    result = dict(reference=dict(run="r1 lap 1", **ref, settings=setting_label(r1["settings"]), cz_step_scatter=scatter(r1["CZ"], r1["step"])),
                  no_rotors=dict(run="c3s lap 1", **base, cz_step_scatter=scatter(c3s["CZ"], c3s["step"])), rotor_effect=effect, runs=[])
    for path in args.runs:
        rs = receipt_lap1(path)
        means = lap_means(rs["per_lap"])
        common, i_r1, i_rs = np.intersect1d(r1["step"], rs["step"], return_indices=True)
        dcz = rs["CZ"][i_rs]-r1["CZ"][i_r1]
        row = dict(run=rs["name"], status=rs["status"], settings=setting_label(rs["settings"]), elapsed_s=rs["elapsed_s"], **means,
                   change={k: means[k]-ref[k] for k in ref if k in means},
                   change_relative={k: (means[k]-ref[k])/abs(ref[k]) for k in ("CZ", "lift_N", "aero_power_W", "shaft_power_W", "thrust_N")
                                    if k in means and ref.get(k)},
                   fraction_of_rotor_effect={k: (means[k]-ref[k])/effect[k] for k in effect},
                   cz_step_change_max=float(np.max(np.abs(dcz))) if len(dcz) else None,
                   cz_step_change_rms=float(np.sqrt(np.mean(dcz**2))) if len(dcz) else None,
                   cz_step_scatter=scatter(rs["CZ"], rs["step"]), steps_compared=int(len(common)))
        result["runs"].append(row)
    Path(args.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(f"rotor effect (r1 - c3s, lap 1): dCX {effect['CX']:+.4f}  dCZ {effect['CZ']:+.4f}  dCm {effect['Cm']:+.4f}; "
          f"C_Z step scatter r1 {result['reference']['cz_step_scatter']:.2e}, c3s {result['no_rotors']['cz_step_scatter']:.2e}")
    print("| run | setting | C_Z | dC_Z/rotor effect | dC_m/rotor effect | shaft power (kW) | thrust (kN) | max abs dC_Z per step | C_Z step scatter | nodes in core (mean/max) | time (h) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in result["runs"]:
        f = r["fraction_of_rotor_effect"]
        print(f"| {r['run']} | {r['settings']} | {r['CZ']:.4f} | {f['CZ']:+.3f} | {f['Cm']:+.3f} | {r.get('shaft_power_W', float('nan'))/1e3:.1f} | "
              f"{r.get('thrust_N', float('nan'))/1e3:.2f} | {r['cz_step_change_max']:.4f} | {r['cz_step_scatter'] if r['cz_step_scatter'] is None else format(r['cz_step_scatter'], '.2e')} | "
              f"{r.get('ring_nodes_in_core_mean', float('nan')):.1f}/{r.get('ring_nodes_in_core_max', '-')} | {r['elapsed_s']/3600:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
