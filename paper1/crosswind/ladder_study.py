"""Read-back of the discretisation ladder of the rotor rings (contract section 8, 2026-10-20 21:57 BST, written before the
runs): one lap of the single M600 with the eight rotors at three levels refined together by a ratio of 2, the ring core
equal to the node spacing of a ring, rings kept 2 s, surface core 0.3 m and wing core 0.15 m unchanged,

    L3  dt 0.04 s,  8 nodes, ring core 0.9 m,   keep  50    rl3   (no rotors: rl3n)
    L1  dt 0.02 s, 16 nodes, ring core 0.45 m,  keep 100    rl1   (no rotors: lap 1 of c3s)
    L2  dt 0.01 s, 32 nodes, ring core 0.225 m, keep 200    rl2   (no rotors: rl2n)
    P   dt 0.01 s, 16 nodes, ring core 0.45 m,  keep 200    rl2p  (paper 2's control step; no rotors: rl2n)

For every level the rotor effect E_q = q(with rotors) - q(no rotors, same dt), lap-1 means of the body-axis coefficients
C_X, C_Z, C_m (pitch) and the aerodynamic power of wing and tails; the shaft power and thrust of the rotors.

Judged (contract): C1  |E(L1) - E(L2)| <= 2 % |E(L2)| for C_Z and <= 5 % for C_m;
                   C3  |E(P)  - E(L2)| <= 2 % |E(L2)| for C_Z and <= 5 % for C_m.
Reported only: with three levels, the ratio of the differences, and when it is monotone the observed order, the
extrapolated value and the grid convergence index of the fine level (Celik et al. 2008, constant ratio r = 2, safety factor
1.25); the no-rotor differences between the levels (the wing's own sensitivity to dt in the loop); the step-to-step scatter
of C_Z as the root mean square of C_Z minus its centred moving mean over the odd number of steps nearest to 0.2 s, after
the first 1 s and inside the lap (independent of dt, unlike the second difference used by ring_study.py).

Usage: ladder_study.py OUT.json --c3s C3S_HISTORY --rl1 DIR --rl2 DIR --rl2n DIR [--rl3 DIR --rl3n DIR] [--rl2p DIR]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_study import history_lap1, lap_means, receipt_lap1   # noqa: E402

KEYS = ("CX", "CZ", "Cm", "aero_power_W")
LIMITS = dict(CZ=0.02, Cm=0.05)
START_S, WINDOW_S, R = 1.0, 0.2, 2.0


def dt_of(settings):
    return float(settings["dt_s"])


def scatter(cz, step, dt):
    """RMS of C_Z about its centred moving mean over the odd step count nearest to 0.2 s, steps after the first 1 s."""
    cz, step = np.asarray(cz, dtype=float), np.asarray(step)
    if len(cz) < 3 or np.any(np.diff(step) != 1):
        return None
    w = max(3, int(round(WINDOW_S/dt)) | 1)
    if abs((w-2)*dt-WINDOW_S) < abs(w*dt-WINDOW_S):
        w -= 2
    h = w//2
    mean = np.convolve(cz, np.ones(w)/w, mode="valid")          # centred on cz[h:-h]
    centre = cz[h:len(cz)-h]
    t = step[h:len(cz)-h]*dt
    keep = t > START_S
    return dict(window_steps=int(w), window_s=w*dt, rms=float(np.sqrt(np.mean((centre[keep]-mean[keep])**2))))


def level(name, run, base):
    m, n = lap_means(run["per_lap"]), lap_means(base["per_lap"])
    rotor = run["settings"].get("rotors") or {}
    return dict(run=name, dt_s=dt_of(run["settings"]), steps_per_lap=int(run["settings"]["steps_per_lap"]),
                nodes_per_ring=rotor.get("nodes_per_ring"), ring_core_m=rotor.get("ring_core_m"),
                surface_core_m=rotor.get("surface_core_m"), retained_ring_steps=rotor.get("retained_ring_steps"),
                ring_bound_core_m=rotor.get("ring_bound_core_m"), free_convection_ring_steps=rotor.get("free_convection_ring_steps"),
                ring_surface_overlap=rotor.get("ring_surface_overlap"),
                with_rotors={k: m[k] for k in m}, no_rotors={k: n[k] for k in KEYS},
                effect={k: m[k]-n[k] for k in KEYS},
                cz_scatter=scatter(run["CZ"], run["step"], dt_of(run["settings"])),
                cz_scatter_no_rotors=scatter(base["CZ"], base["step"], dt_of(base["settings"])),
                elapsed_h=None if run.get("elapsed_s") is None else run["elapsed_s"]/3600.0)


def compare(coarse, fine, label):
    rel = {k: abs(coarse["effect"][k]-fine["effect"][k])/abs(fine["effect"][k]) for k in KEYS}
    ok = {k: rel[k] <= LIMITS[k] for k in LIMITS}
    return dict(criterion=label, relative_difference_of_effect=rel, limits=LIMITS, passed_each=ok, passed=all(ok.values()))


def order(l3, l1, l2):
    """Celik et al. 2008 with phi1 the fine (L2), phi2 the base (L1), phi3 the coarse (L3) level, r = 2."""
    out = {}
    for k in KEYS:
        p1, p2, p3 = l2["effect"][k], l1["effect"][k], l3["effect"][k]
        e21, e32 = p2-p1, p3-p2
        row = dict(fine=p1, base=p2, coarse=p3, e21=e21, e32=e32)
        if e21 == 0.0 or e32 == 0.0:
            row.update(kind="a difference is zero")
        elif e32/e21 < 0.0:
            row.update(kind="oscillatory", error_band=max(abs(e21), abs(e32)))
        elif abs(e21) >= abs(e32):
            row.update(kind="not convergent", error_band=max(abs(e21), abs(e32)))
        else:
            p = math.log(abs(e32/e21))/math.log(R)
            ext = (R**p*p1-p2)/(R**p-1.0)
            row.update(kind="monotone", observed_order=p, extrapolated=ext, ea21=abs(e21/p1),
                       gci_fine=1.25*abs(e21/p1)/(R**p-1.0))
        out[k] = row
    return out


def main(argv):
    ap = argparse.ArgumentParser(prog="ladder_study.py")
    ap.add_argument("out")
    ap.add_argument("--c3s", required=True)
    for name in ("rl1", "rl2", "rl2n"):
        ap.add_argument("--"+name, required=True)
    for name in ("rl3", "rl3n", "rl2p"):
        ap.add_argument("--"+name)
    a = ap.parse_args(argv)
    c3s = history_lap1(a.c3s)
    c3s["settings"].setdefault("dt_s", 0.02)
    rl2n = receipt_lap1(a.rl2n)
    levels = dict(L1=level("rl1", receipt_lap1(a.rl1), c3s), L2=level("rl2", receipt_lap1(a.rl2), rl2n))
    if a.rl3 and a.rl3n:
        levels["L3"] = level("rl3", receipt_lap1(a.rl3), receipt_lap1(a.rl3n))
    if a.rl2p:
        levels["P"] = level("rl2p", receipt_lap1(a.rl2p), rl2n)
    result = dict(levels=levels, C1=compare(levels["L1"], levels["L2"], "C1: L1 against L2"))
    if "P" in levels:
        result["C3"] = compare(levels["P"], levels["L2"], "C3: P against L2")
    if "L3" in levels:
        result["three_levels"] = order(levels["L3"], levels["L1"], levels["L2"])
    base = levels["L1"]["no_rotors"]
    result["no_rotor_dt_sensitivity"] = {lv: {k: levels[lv]["no_rotors"][k]-base[k] for k in KEYS} for lv in levels if lv != "L1"}
    Path(a.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print("| level | dt (s) | nodes | ring core (m) | keep | E C_Z | E C_m | E C_X | E P_aero (kW) | shaft (kW) | thrust (kN) | scatter C_Z | time (h) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lv, r in levels.items():
        e, w, s = r["effect"], r["with_rotors"], r["cz_scatter"]
        print(f"| {lv} | {r['dt_s']:.3f} | {r['nodes_per_ring']} | {r['ring_core_m']} | {r['retained_ring_steps']} | {e['CZ']:+.5f} | {e['Cm']:+.5f} | "
              f"{e['CX']:+.5f} | {e['aero_power_W']/1e3:+.2f} | {w.get('shaft_power_W', float('nan'))/1e3:.1f} | {w.get('thrust_N', float('nan'))/1e3:.3f} | "
              f"{'-' if s is None else format(s['rms'], '.2e')} | {'-' if r['elapsed_h'] is None else format(r['elapsed_h'], '.2f')} |")
    for key in ("C1", "C3"):
        if key in result:
            c = result[key]
            print(f"{c['criterion']}: C_Z {c['relative_difference_of_effect']['CZ']:.4f} (limit 0.02), C_m {c['relative_difference_of_effect']['Cm']:.4f} "
                  f"(limit 0.05) -> {'passed' if c['passed'] else 'not passed'}")
    if "three_levels" in result:
        for k, row in result["three_levels"].items():
            print(f"three levels {k}: {row['kind']}" + (f", order {row['observed_order']:.2f}, GCI {row['gci_fine']:.4f}" if row["kind"] == "monotone" else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
