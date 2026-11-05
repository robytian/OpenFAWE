"""V2: tree histories against the exact Numba direct history (OF-0006-accel20261001).

Frozen budget (tree-design.md): for every step |dCL|/|CL| <= 5e-4,
|dCD|/|CD| <= 5e-4, |dCMy| <= 2e-4; plus <= 1/10 of the smallest
discretisation effect measured in Phase C (checked in the Phase C report).
Usage: check_v2_histories.py OUT.json REFERENCE_DIR TREE_DIR [TREE_DIR ...]
"""
import json
import sys
from pathlib import Path

BUDGET = dict(CL=5e-4, CD=5e-4, CMy=2e-4)


def load(path):
    return json.loads((Path(path) / "history.json").read_text(encoding="utf-8"))


def compare(reference, candidate):
    a, b = reference["records"], candidate["records"]
    steps = min(len(a), len(b))
    worst = dict(CL=0.0, CD=0.0, CMy=0.0)
    where = dict(CL=0, CD=0, CMy=0)
    for k in range(steps):
        diffs = dict(CL=abs(a[k]["CL"] - b[k]["CL"]) / abs(a[k]["CL"]),
                     CD=abs(a[k]["CD"] - b[k]["CD"]) / abs(a[k]["CD"]),
                     CMy=abs(a[k]["CMy"] - b[k]["CMy"]))
        for key, value in diffs.items():
            if value > worst[key]:
                worst[key], where[key] = value, k
    ref_time = sum(reference["step_times_s"][:steps])
    new_time = sum(candidate["step_times_s"][:steps])
    passed = (reference["status"] == candidate["status"] and len(a) == len(b)
              and all(worst[key] <= BUDGET[key] for key in BUDGET))
    return dict(candidate=candidate["settings"]["out"], settings={k: candidate["settings"][k] for k in
                ("backend", "theta", "order", "leaf")}, status=[reference["status"], candidate["status"]],
                frames=[len(a), len(b)], worst=worst, worst_step=where, budget=BUDGET, within_budget=passed,
                final=dict(reference={k: a[steps - 1][k] for k in ("CL", "CD", "CMy")},
                           candidate={k: b[steps - 1][k] for k in ("CL", "CD", "CMy")}),
                wall_s=dict(reference=ref_time, candidate=new_time, ratio=ref_time / new_time if new_time else None),
                last_step_s=dict(reference=reference["step_times_s"][steps - 1],
                                 candidate=candidate["step_times_s"][steps - 1]))


def main(output, reference_dir, candidates):
    reference = load(reference_dir)
    rows = [compare(reference, load(c)) for c in candidates]
    report = dict(reference=str(reference_dir), reference_settings=reference["settings"],
                  environment=reference["environment"], comparisons=rows,
                  note="Same 2-core cloud machine, one Numba thread per run; timings indicative.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(dict(candidate=Path(row["candidate"]).name, worst=row["worst"],
                              within_budget=row["within_budget"], speed_ratio=row["wall_s"]["ratio"],
                              last_step_s=row["last_step_s"])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
