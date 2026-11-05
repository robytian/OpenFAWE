"""Read-back of the K7 periodicity check and of the rotor effect for the OF-0009 runs (2026-10-23).

Two questions, answered from the receipts only (no new physics, no change to K7):

1. Where does a K7 failure of unit B come from? K7 (``c3_pair.periodicity``, written before the runs) compares the
   total load (aerodynamic + tether, + rotor when present) of each step with the same step one lap earlier and divides
   by the norm of the total load of that step. This script recomputes that number from ``history.json`` and splits it:
   the aerodynamic part alone, the tether part (identical between laps under prescribed motion), the same difference
   divided by the largest total-load norm of the lap instead of the local one, and the loop angle of the largest value.
   It also reports how many steps exceed 0.5 % and 1 %. The written verdict is not changed here; the split only shows
   the cause.
2. The rotor effect of the single unit: per lap, the run with rotors minus the run without (same path, same dt).
3. K7a (contract section 8, 2026-10-24 12:16 BST, written before the p5, p4, p2a and p2b results): the same lap-to-lap
   comparison on the wing aerodynamic loads only, normalised step by step, limit 1 %, reported beside K7.

Usage: k7_readback.py OUT.json [--history NAME=PATH/history.json ...] [--pair NAME=ROTORS/checks.json,NONE/checks.json ...]
       [--k7a RUN/checks.json ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def split(history_path):
    data = json.loads(Path(history_path).read_text(encoding="utf-8"))
    records, s = data["records"], data["settings"]
    n, laps = s["steps_per_lap"], s["laps"]
    names = list(records[-1]["bodies"])

    def block(lap, name, key):
        return np.array([records[k]["bodies"][name][key] for k in range((lap-1)*n+1, lap*n+1)], dtype=float)

    out = dict(steps_per_lap=n, laps=laps, backend=s["backend"], free_convection_rows=s["free_convection_rows"],
               written_K7=data["K7"], units={})
    for name in names:
        rows = []
        for lap in range(2, laps+1):
            row = dict(lap=lap, against=lap-1)
            rotors = "rotor_force_body_N" in records[-1]["bodies"][name]
            for kind, aero_key, tether_key, rotor_key in (("force", "force_body_N", "tether_force_body_N", "rotor_force_body_N"),
                                                          ("moment", "moment_origin_body_Nm", "tether_moment_origin_body_Nm",
                                                           "rotor_moment_origin_body_Nm")):
                aero0, aero1 = block(lap-1, name, aero_key), block(lap, name, aero_key)
                teth0, teth1 = block(lap-1, name, tether_key), block(lap, name, tether_key)
                total0, total1 = aero0+teth0, aero1+teth1
                if rotors:   # the same total as c3_pair.total_loads
                    rot0, rot1 = block(lap-1, name, rotor_key), block(lap, name, rotor_key)
                    total0, total1 = total0+rot0, total1+rot1
                local = np.linalg.norm(total1-total0, axis=1)/np.linalg.norm(total0, axis=1)
                aero = np.linalg.norm(aero1-aero0, axis=1)/np.linalg.norm(aero0, axis=1)
                lapmax = np.linalg.norm(total1-total0, axis=1)/np.linalg.norm(total0, axis=1).max()
                k = int(local.argmax())
                row[kind] = dict(K7_local_max=float(local.max()), aero_only_max=float(aero.max()),
                                 lap_max_normalised_max=float(lapmax.max()),
                                 tether_difference_max=float(np.abs(teth1-teth0).max()),
                                 loop_angle_of_max_deg=round(360.*k/n, 1),
                                 total_norm_at_max=float(np.linalg.norm(total0, axis=1)[k]),
                                 total_norm_lap_max=float(np.linalg.norm(total0, axis=1).max()),
                                 aero_norm_at_max=float(np.linalg.norm(aero0, axis=1)[k]),
                                 aero_steps_above_0p5_percent=int((aero > 0.005).sum()),
                                 aero_steps_above_1_percent=int((aero > 0.01).sum()))
                if rotors:
                    row[kind]["rotor_difference_max"] = float(np.abs(rot1-rot0).max())
                    row[kind]["rotor_norm_at_max"] = float(np.linalg.norm(rot0, axis=1)[k])
            rows.append(row)
        out["units"][name] = rows
    return out


def k7a(checks_path, limit=0.01):
    """K7a of contract section 8 (2026-10-24 12:16 BST, written before the p5, p4, p2a and p2b results): per unit, the
    last lap against the one before, step by step, the wing aerodynamic force and moment about the body origin (the
    receipt's ``per_step``, without tether and rotor), the norm of the difference over the norm at that step, the largest
    value of the lap, limit 1 %. With rotors also the wing plus rotor force (no limit; the rotor moment is not in the
    receipt). Reported beside K7, which is not changed."""
    data = json.loads(Path(checks_path).read_text(encoding="utf-8"))
    s = data["settings"]
    n, laps = s["steps_per_lap"], s["laps"]
    out = dict(run=Path(checks_path).parent.name, steps_per_lap=n, laps=laps, backend=s["backend"], limit=limit,
               written_K7={k: v.get("verdict") for k, v in data["K7"].items()}, units={})
    if data.get("status") != "complete" or laps < 2:
        out["verdict"] = "not evaluated"
        return out
    for name, d in data["per_step"].items():
        index = {k: i for i, k in enumerate(d["step"])}
        last = [(laps-1)*n+k for k in range(1, n+1)]
        before = [(laps-2)*n+k for k in range(1, n+1)]
        if not all(k in index for k in last+before):
            out["units"][name] = dict(verdict="not evaluated", reason="the receipt lacks dense steps of the last two laps")
            continue
        rows = dict(lap=laps, against=laps-1)
        pick = lambda key, steps: np.array([d[key][index[k]] for k in steps], dtype=float)
        parts = [("force", "force_body_N"), ("moment", "moment_origin_body_Nm")]
        for kind, key in parts:
            a, b = pick(key, before), pick(key, last)
            rel = np.linalg.norm(b-a, axis=1)/np.linalg.norm(a, axis=1)
            k = int(rel.argmax())
            rows[kind] = dict(max=float(rel.max()), mean=float(rel.mean()), loop_angle_of_max_deg=round(360.*k/n, 1),
                              steps_above_0p5_percent=int((rel > 0.005).sum()), steps_above_1_percent=int((rel > 0.01).sum()))
        if d.get("rotor_force_body_N") and len(d["rotor_force_body_N"]) == len(d["step"]):
            a = pick("force_body_N", before)+pick("rotor_force_body_N", before)
            b = pick("force_body_N", last)+pick("rotor_force_body_N", last)
            rel = np.linalg.norm(b-a, axis=1)/np.linalg.norm(a, axis=1)
            rows["wing_plus_rotor_force"] = dict(max=float(rel.max()), mean=float(rel.mean()),
                                                 loop_angle_of_max_deg=round(360.*int(rel.argmax())/n, 1))
        rows["verdict"] = "passed" if max(rows["force"]["max"], rows["moment"]["max"]) <= limit else "failed"
        out["units"][name] = rows
    return out


def rotor_effect(with_rotors, without):
    a = json.loads(Path(with_rotors).read_text(encoding="utf-8"))
    b = json.loads(Path(without).read_text(encoding="utf-8"))
    for key in ("steps_per_lap", "dt_s", "wind_m_s", "case", "playbook", "single"):   # laps may differ: laps compared up to the shorter run
        if a["settings"][key] != b["settings"][key]:
            raise SystemExit(f"the two runs differ in {key}")
    rows = []
    for la, lb in zip(a["per_lap"]["A"], b["per_lap"]["A"]):
        rows.append(dict(lap=la["lap"],
                         force_coefficient_with=la["force_body_coefficient_mean"], force_coefficient_without=lb["force_body_coefficient_mean"],
                         force_coefficient_change=list(np.subtract(la["force_body_coefficient_mean"], lb["force_body_coefficient_mean"])),
                         moment_coefficient_with=la["moment_body_coefficient_mean"], moment_coefficient_without=lb["moment_body_coefficient_mean"],
                         moment_coefficient_change=list(np.subtract(la["moment_body_coefficient_mean"], lb["moment_body_coefficient_mean"])),
                         lift_relative_change=la["lift_body_z_mean_N"]/lb["lift_body_z_mean_N"]-1.,
                         aero_power_with_W=la["aero_power_mean_W"], aero_power_without_W=lb["aero_power_mean_W"],
                         rotor_shaft_power_W=la.get("rotor_shaft_power_W"), rotor_thrust_total_N=la.get("rotor_thrust_total_N"),
                         rotor_power_on_body_mean_W=la.get("rotor_power_on_body_mean_W"),
                         ring_nodes_within_surface_core=la.get("ring_nodes_within_surface_core")))
    return dict(with_rotors=dict(K3=a["K3"]["A"]["worst_relative"], K7=a["K7"]["A"], last_pair=a["K7_pairs"]["A"][-1],
                                 elapsed_s=a["elapsed_s"], backend=a["settings"]["backend"], dt_s=a["settings"]["dt_s"],
                                 rotors=a["settings"]["rotors"]),
                without=dict(K3=b["K3"]["A"]["worst_relative"], K7=b["K7"]["A"], last_pair=b["K7_pairs"]["A"][-1],
                             elapsed_s=b["elapsed_s"], backend=b["settings"]["backend"]),
                per_lap=rows)


def pair_effect(with_rotors, without):
    """Pairs: per unit and lap, the run with rotors minus the run without on the same paths and dt (lap means of C_Z,
    C_m, lift, wing aerodynamic power), and the B/A ratios of both runs."""
    a = json.loads(Path(with_rotors).read_text(encoding="utf-8"))
    b = json.loads(Path(without).read_text(encoding="utf-8"))
    for key in ("steps_per_lap", "dt_s", "wind_m_s", "case", "playbook", "spacing_diameters", "offset_diameters", "phase_deg"):
        if a["settings"][key] != b["settings"][key]:
            raise SystemExit(f"the two runs differ in {key}")
    laps = min(len(a["per_lap"]["A"]), len(b["per_lap"]["A"]))
    out = dict(with_rotors=dict(run=Path(with_rotors).parent.name, K3=a["K3"], K7={k: v["verdict"] for k, v in a["K7"].items()},
                                last_pairs={k: v[-1] for k, v in a["K7_pairs"].items()}, elapsed_s=a["elapsed_s"],
                                backend=a["settings"]["backend"], laps=a["settings"]["laps"]),
               without=dict(run=Path(without).parent.name, K7={k: v["verdict"] for k, v in b["K7"].items()},
                            backend=b["settings"]["backend"], laps=b["settings"]["laps"]),
               units={}, ratios=[])
    for name in a["per_lap"]:
        rows = []
        for lap in range(1, laps+1):
            x, y = a["per_lap"][name][lap-1], b["per_lap"][name][lap-1]
            rows.append(dict(lap=lap, CZ_with=x["force_body_coefficient_mean"][2], CZ_without=y["force_body_coefficient_mean"][2],
                             CZ_change=x["force_body_coefficient_mean"][2]-y["force_body_coefficient_mean"][2],
                             Cm_with=x["moment_body_coefficient_mean"][1], Cm_without=y["moment_body_coefficient_mean"][1],
                             Cm_change=x["moment_body_coefficient_mean"][1]-y["moment_body_coefficient_mean"][1],
                             lift_relative_change=x["lift_body_z_mean_N"]/y["lift_body_z_mean_N"]-1.,
                             aero_power_with_W=x["aero_power_mean_W"], aero_power_without_W=y["aero_power_mean_W"],
                             rotor_shaft_power_W=x.get("rotor_shaft_power_W"), rotor_thrust_total_N=x.get("rotor_thrust_total_N"),
                             ring_nodes_within_surface_core=x.get("ring_nodes_within_surface_core")))
        out["units"][name] = rows
    if "B" in a["per_lap"]:
        for lap in range(1, laps+1):
            ra, rb = a["per_lap"], b["per_lap"]
            out["ratios"].append(dict(lap=lap,
                                      lift_B_over_A_with=ra["B"][lap-1]["lift_body_z_mean_N"]/ra["A"][lap-1]["lift_body_z_mean_N"],
                                      lift_B_over_A_without=rb["B"][lap-1]["lift_body_z_mean_N"]/rb["A"][lap-1]["lift_body_z_mean_N"],
                                      power_B_over_A_with=ra["B"][lap-1]["aero_power_mean_W"]/ra["A"][lap-1]["aero_power_mean_W"],
                                      power_B_over_A_without=rb["B"][lap-1]["aero_power_mean_W"]/rb["A"][lap-1]["aero_power_mean_W"],
                                      shaft_B_over_A=ra["B"][lap-1]["rotor_shaft_power_W"]["mean"]/ra["A"][lap-1]["rotor_shaft_power_W"]["mean"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("--history", action="append", default=[], help="NAME=path/to/history.json")
    parser.add_argument("--pair", action="append", default=[], help="NAME=with_rotors_checks.json,without_checks.json")
    parser.add_argument("--pairs", action="append", default=[], help="NAME=pair_with_rotors/checks.json,pair_without/checks.json")
    parser.add_argument("--k7a", action="append", default=[], help="path/to/checks.json (K7a of 2026-10-24 12:16 BST)")
    args = parser.parse_args()
    report = dict(script="k7_readback.py", k7_split={}, rotor_effect={}, k7a={})
    for path in args.k7a:
        row = k7a(path)
        report["k7a"][row["run"]] = row
    for item in args.history:
        name, path = item.split("=", 1)
        report["k7_split"][name] = split(path)
    for item in args.pair:
        name, paths = item.split("=", 1)
        with_rotors, without = paths.split(",")
        report["rotor_effect"][name] = rotor_effect(with_rotors, without)
    for item in args.pairs:
        name, paths = item.split("=", 1)
        with_rotors, without = paths.split(",")
        report.setdefault("pair_effect", {})[name] = pair_effect(with_rotors, without)
    Path(args.out).write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
