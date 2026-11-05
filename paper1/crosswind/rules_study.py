"""Read-back of the rule checks R' at the proposed production settings of the rotor rings (contract section 8,
2026-10-22 19:43 BST, written before the runs). First second of the single M600, dt 0.01 s; base: model A' (rings drift
with the wind from release), sheet refinement lambda 2, surface core 0.1 m, ring core 0.45 m, 16 nodes, rings kept 2 s.
The rotor effect is the mean over 0.5 s < t <= 1 s of q(rotors) - q(no rotors, n1872) for C_Z and C_m (sheet_study).

Numerical parameters (rings kept 0.5, 1, 4 s; 32 nodes; ring core 0.15, 0.9 m) are insensitive when the effect differs
from the base by at most 0.003 (C_Z) and 10 % (C_m). Model and physical items are reported only: model A'' (sheet drift),
only the newest ring kept (no slipstream), rotor speeds 60, 160, 200 rad/s. R-base reruns the base on the current code
and is compared with zs1872c010 (the same settings run before the rebase onto the OF-0010 merge c76fb4d7).

Usage: rules_study.py OUT.json --runs DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sheet_study import effect, load

LIMIT_CZ, LIMIT_CM = 0.003, 0.10
NUMERICAL = {"R-keep050": "rings kept 0.5 s", "R-keep100": "rings kept 1 s", "R-keep400": "rings kept 4 s",
             "R-n32": "32 nodes per ring", "R-core015": "ring core 0.15 m", "R-core090": "ring core 0.9 m"}
REPORTED = {"R-adp": "model A'' (sheet drift)", "R-keep001": "newest ring only (no slipstream)",
            "R-om060": "60 rad/s", "R-om160": "160 rad/s", "R-om200": "200 rad/s"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rules_study.py")
    ap.add_argument("out")
    ap.add_argument("--runs", required=True)
    a = ap.parse_args(argv)
    d = Path(a.runs)
    plain = load(d/"n1872")
    runs, effects, status = {}, {}, {}
    for name in ["R-base", "zs1872c010", *NUMERICAL, *REPORTED]:
        path = d/name/"checks.json"
        if not path.exists():
            status[name] = "missing"
            continue
        data = json.loads(path.read_text())
        status[name] = data.get("status")
        if data.get("status") != "complete":
            continue
        runs[name] = load(d/name)
        effects[name] = effect(runs[name], plain)
    base = effects.get("R-base")
    out = dict(base=base, status=status, consistency=None, numerical={}, reported={})
    if base is not None and "zs1872c010" in effects:
        old = effects["zs1872c010"]
        out["consistency"] = dict(previous=old, abs_diff_CZ=abs(base["CZ"]-old["CZ"]), abs_diff_Cm=abs(base["Cm"]-old["Cm"]))
    for group, labels in (("numerical", NUMERICAL), ("reported", REPORTED)):
        for name, label in labels.items():
            if name not in effects or base is None:
                continue
            e = effects[name]
            dz, dm = abs(e["CZ"]-base["CZ"]), abs(e["Cm"]-base["Cm"])/abs(base["Cm"])
            row = dict(label=label, effect=e, abs_diff_CZ=dz, rel_diff_Cm=dm,
                       mean_step_time_s=float(np.mean(runs[name]["step_times"][1:])))
            if group == "numerical":
                row["insensitive"] = bool(dz <= LIMIT_CZ and dm <= LIMIT_CM)
            out[group][name] = row
    Path(a.out).write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    if base is not None:
        print(f"base: E C_Z {base['CZ']:+.5f}, E C_m {base['Cm']:+.5f}")
    if out["consistency"]:
        c = out["consistency"]
        print(f"rerun of the base against zs1872c010: |dC_Z| {c['abs_diff_CZ']:.2e}, |dC_m| {c['abs_diff_Cm']:.2e}")
    print("| run | item | E C_Z | E C_m | |dC_Z| | |dC_m|/|C_m| | reading |")
    print("|---|---|---|---|---|---|---|")
    for group in ("numerical", "reported"):
        for name, r in out[group].items():
            reading = ("insensitive" if r["insensitive"] else "sensitive") if group == "numerical" else "report"
            print(f"| {name} | {r['label']} | {r['effect']['CZ']:+.5f} | {r['effect']['Cm']:+.5f} | {r['abs_diff_CZ']:.4f} | "
                  f"{r['rel_diff_Cm']:.3f} | {reading} |")
    missing = [k for k, v in status.items() if v != "complete"]
    if missing:
        print("not complete:", ", ".join(f"{k} ({status[k]})" for k in missing))


if __name__ == "__main__":
    main()
