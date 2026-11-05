"""Read-back of the sheet refinement S of the rotor rings (contract section 8, 2026-10-22 18:22 BST, criteria written
before the runs). All runs are the first second of the single M600 on its loop, the tube of 16:20 (rings free for one
step, then drifting with the wind), 16 nodes, ring core 0.45 m, rings kept 2 s.

The rotor effect E_q of a run is the mean over 0.5 s < t <= 1 s of q(with rotors) - q(no rotors, same dt), for the
body-axis coefficients C_Z and C_m (reference area 32.9285 m^2, chord 1.2831 m, the airspeed of every step).

  S1  sheet refinement (lambda 2, surface core 0.3 m) at dt 0.02 and 0.01 s against dt 0.00125 s without it:
      |E - E_ref| <= 0.003 (C_Z) and <= 10 % |E_ref| (C_m), for both.
  S2  report: refinement at dt 0.02 s with surface cores 0.3, 0.15, 0.1, 0.075 m; read: if the change 0.1 -> 0.075 m is
      within 0.003 (C_Z) and 10 % (C_m), the surface core is a numerical parameter in that window.
  S3  mean step time with the refinement <= 1.10 x without (dt 0.02 s, surface core 0.3 m, same machine, one thread,
      runs one after the other).

Model A' (contract section 8, 18:55 BST: rings drift with the wind from their release) is read with --set aprime
(criteria S1', S3' with the same limits; S2 rerun on it).

Usage: sheet_study.py OUT.json --runs DIR [--set a|aprime] [--ref-rotors NAME --ref-plain NAME]
"""
import argparse
import json
from pathlib import Path

import numpy as np

AREA_M2, CHORD_M, RHO = 32.9285, 1.2831, 1.225
WINDOW = (0.5, 1.0)
LIMIT_CZ, LIMIT_CM = 0.003, 0.10
SETS = dict(a=dict(plain="t936b", s02="s936", s01="s1872", c015="s936c015", c010="s936c010", c0075="s936c0075"),
            aprime=dict(plain="z936", s02="zs936", s01="zs1872", c015="zs936c015", c010="zs936c010", c0075="zs936c0075",
                        plain01="z1872"))


def load(path):
    data = json.loads((Path(path)/"checks.json").read_text())
    a = data["per_step"]["A"]
    dt = float(data["settings"]["dt_s"])
    t = np.asarray(a["step"])*dt
    q = 0.5*RHO*np.asarray(a["airspeed_m_s"], dtype=float)**2
    f, m = np.asarray(a["force_body_N"], dtype=float), np.asarray(a["moment_origin_body_Nm"], dtype=float)
    times = np.asarray(data.get("step_times_s") or [], dtype=float)
    rotors = data["settings"].get("rotors") or {}
    return dict(t=t, dt=dt, CZ=f[:, 2]/(q*AREA_M2), Cm=m[:, 1]/(q*AREA_M2*CHORD_M), step_times=times,
                elapsed=data.get("elapsed_s"), overlap=rotors.get("ring_surface_overlap"),
                surface_core=rotors.get("surface_core_m"), free=rotors.get("free_convection_ring_steps"))


def effect(run, plain):
    n = min(len(run["t"]), len(plain["t"]))
    if np.any(np.abs(run["t"][:n]-plain["t"][:n]) > 1e-9):
        raise ValueError("the runs do not share their time samples")
    t = run["t"][:n]
    sel = (t > WINDOW[0]) & (t <= WINDOW[1]+1e-9)
    return {k: float(np.mean(run[k][:n][sel]-plain[k][:n][sel])) for k in ("CZ", "Cm")}


def compare(e, ref):
    dz, dm = abs(e["CZ"]-ref["CZ"]), abs(e["Cm"]-ref["Cm"])/abs(ref["Cm"])
    return dict(abs_diff_CZ=dz, rel_diff_Cm=dm, passed=bool(dz <= LIMIT_CZ and dm <= LIMIT_CM))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sheet_study.py")
    ap.add_argument("out")
    ap.add_argument("--runs", required=True, help="folder with the run directories")
    ap.add_argument("--set", default="a", choices=sorted(SETS))
    ap.add_argument("--ref-rotors", default="f14976g", help="tube without refinement at dt 0.00125 s")
    ap.add_argument("--ref-plain", default="n14976g", help="no rotors at dt 0.00125 s")
    a = ap.parse_args(argv)
    d = Path(a.runs)
    names = SETS[a.set]
    runs = {name: load(d/name) for name in ("n936", "n1872", *names.values(), a.ref_rotors, a.ref_plain)
            if (d/name/"checks.json").exists()}
    plain = {0.02: runs["n936"], 0.01: runs.get("n1872")}
    ref = effect(runs[a.ref_rotors], runs[a.ref_plain])
    out = dict(set=a.set, names=names, reference=dict(run=a.ref_rotors, dt_s=runs[a.ref_rotors]["dt"], effect=ref), runs={})
    for name in names.values():
        if name not in runs:
            continue
        r = runs[name]
        e = effect(r, plain[round(r["dt"], 2)])
        out["runs"][name] = dict(dt_s=r["dt"], overlap=r["overlap"], surface_core_m=r["surface_core"], effect=e,
                                 mean_step_time_s=float(np.mean(r["step_times"][1:])) if len(r["step_times"]) > 1 else None,
                                 elapsed_s=r["elapsed"])
    s1 = {name: compare(out["runs"][name]["effect"], ref) for name in (names["s02"], names["s01"]) if name in out["runs"]}
    out["S1"] = dict(limits=dict(CZ=LIMIT_CZ, Cm=LIMIT_CM), each=s1, passed=bool(len(s1) == 2 and all(v["passed"] for v in s1.values())))
    ladder = [n for n in (names["s02"], names["c015"], names["c010"], names["c0075"]) if n in out["runs"]]
    out["S2"] = dict(ladder={n: out["runs"][n]["effect"] for n in ladder})
    if names["c010"] in out["runs"] and names["c0075"] in out["runs"]:
        out["S2"]["reading_0.1_to_0.075"] = compare(out["runs"][names["c010"]]["effect"], out["runs"][names["c0075"]]["effect"])
    if names["plain"] in out["runs"] and names["s02"] in out["runs"]:
        a0, a1 = out["runs"][names["plain"]]["mean_step_time_s"], out["runs"][names["s02"]]["mean_step_time_s"]
        out["S3"] = dict(mean_step_time_plain_s=a0, mean_step_time_sheet_s=a1, ratio=a1/a0, limit=1.10, passed=bool(a1/a0 <= 1.10))
    Path(a.out).write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    print(f"reference ({a.ref_rotors}, dt {out['reference']['dt_s']}): E C_Z {ref['CZ']:+.5f}, E C_m {ref['Cm']:+.5f}")
    print("| run | dt (s) | overlap | surface core (m) | E C_Z | E C_m | mean step (s) |")
    print("|---|---|---|---|---|---|---|")
    for name, r in out["runs"].items():
        st = r["mean_step_time_s"]
        print(f"| {name} | {r['dt_s']:.4f} | {r['overlap']} | {r['surface_core_m']} | {r['effect']['CZ']:+.5f} | {r['effect']['Cm']:+.5f} | "
              f"{'-' if st is None else format(st, '.3f')} |")
    for name, c in out["S1"]["each"].items():
        print(f"S1 {name}: |dC_Z| {c['abs_diff_CZ']:.4f} (<= 0.003), |dC_m|/|C_m| {c['rel_diff_Cm']:.3f} (<= 0.10) -> {'passed' if c['passed'] else 'not passed'}")
    if "reading_0.1_to_0.075" in out["S2"]:
        c = out["S2"]["reading_0.1_to_0.075"]
        print(f"S2 0.1 -> 0.075 m: |dC_Z| {c['abs_diff_CZ']:.4f}, |dC_m|/|C_m| {c['rel_diff_Cm']:.3f} -> {'within' if c['passed'] else 'outside'} the limits")
    if "S3" in out:
        print(f"S3: {out['S3']['mean_step_time_sheet_s']:.3f} / {out['S3']['mean_step_time_plain_s']:.3f} s = {out['S3']['ratio']:.3f} (<= 1.10) "
              f"-> {'passed' if out['S3']['passed'] else 'not passed'}")


if __name__ == "__main__":
    main()
