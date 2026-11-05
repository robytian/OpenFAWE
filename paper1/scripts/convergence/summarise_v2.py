"""Summarise V2: tree histories against the exact Numba direct history (OF-0006-accel20261001).

For each model, compares every tree candidate run with the direct reference
run step by step, applies the frozen budget (tree-design.md):
|dCL|/|CL| <= 5e-4, |dCD|/|CD| <= 5e-4, |dCMy| <= 2e-4 at every step,
and writes v2-summary.json plus figure V2 (per-step differences, log scale,
budget as a horizontal rule). Usage:
  summarise_v2.py OUT_DIR RUNS_DIR
Run directory names: {v3,m600}-vsm-direct-200 and v2-{v3,m600}-tree-*.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BUDGET = dict(CL=5e-4, CD=5e-4, CMy=2e-4)
# Categorical slots 1-4 of the reference palette (adjacent pairs validated), plus markers.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def load(path):
    return json.loads((Path(path) / "history.json").read_text(encoding="utf-8"))


def differences(reference, candidate):
    a, b = reference["records"], candidate["records"]
    steps = min(len(a), len(b))
    series = dict(CL=[], CD=[], CMy=[])
    for k in range(steps):
        series["CL"].append(abs(a[k]["CL"] - b[k]["CL"]) / abs(a[k]["CL"]))
        series["CD"].append(abs(a[k]["CD"] - b[k]["CD"]) / abs(a[k]["CD"]))
        series["CMy"].append(abs(a[k]["CMy"] - b[k]["CMy"]))
    return series


def label(settings):
    return f"θ={settings['theta']}, p={settings['order']}, leaf {settings['leaf']}"


def main(out_dir, runs_dir):
    runs = Path(runs_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(budget=BUDGET, models={})
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True)
    for row, model in enumerate(("v3", "m600")):
        reference = load(runs / f"{model}-vsm-direct-200")
        candidates = sorted((p for p in runs.glob(f"v2-{model}-tree-*") if p.is_dir()),
                            key=lambda p: (-load(p)["settings"]["theta"], load(p)["settings"]["order"]))
        entries = []
        for index, path in enumerate(candidates):
            candidate = load(path)
            series = differences(reference, candidate)
            steps = min(len(reference["records"]), len(candidate["records"]))
            worst = {k: max(v) for k, v in series.items()}
            within = (reference["status"] == candidate["status"] == "complete"
                      and len(reference["records"]) == len(candidate["records"])
                      and all(worst[k] <= BUDGET[k] for k in BUDGET))
            ref_wall = sum(reference["step_times_s"][:steps])
            new_wall = sum(candidate["step_times_s"][:steps])
            entries.append(dict(run=path.name, settings={k: candidate["settings"][k] for k in
                                                         ("backend", "theta", "order", "leaf")},
                                status=candidate["status"], frames=len(candidate["records"]),
                                worst=worst, worst_step={k: v.index(max(v)) for k, v in series.items()},
                                within_budget=within, wall_s=dict(reference=ref_wall, tree=new_wall,
                                                                  speedup=ref_wall / new_wall),
                                last_step_s=dict(reference=reference["step_times_s"][steps - 1],
                                                 tree=candidate["step_times_s"][steps - 1]),
                                final=dict(reference={k: reference["records"][steps - 1][k] for k in BUDGET},
                                           tree={k: candidate["records"][steps - 1][k] for k in BUDGET})))
            for column, key in enumerate(("CL", "CD", "CMy")):
                ax = axes[row][column]
                values = [max(v, 1e-16) for v in series[key]]
                ax.plot(range(len(values)), values, color=COLOURS[index], linewidth=1.3,
                        marker=MARKERS[index], markevery=25, markersize=4, label=label(candidate["settings"]))
        report["models"][model] = dict(reference=reference["settings"]["out"],
                                       reference_status=reference["status"],
                                       reference_wall_s=reference["elapsed_s"], candidates=entries)
        for column, key in enumerate(("CL", "CD", "CMy")):
            ax = axes[row][column]
            ax.axhline(BUDGET[key], color=RULE, linewidth=1.0, linestyle=(0, (4, 3)))
            ax.text(1, BUDGET[key] * 1.25, "budget", fontsize=7, color=MUTED)
            ax.set_yscale("log")
            ax.set_ylim(1e-9, 3e-3)
            ax.grid(True, which="major", color=GRID, linewidth=0.6)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            title = {"CL": "|ΔCL| / |CL|", "CD": "|ΔCD| / |CD|", "CMy": "|ΔCMy|"}[key]
            if row == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if column == 0:
                ax.set_ylabel(("V3" if model == "v3" else "M600") + " VSM free wake", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("Time step", fontsize=8.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle("Tree code against the exact direct sum: coefficient differences along 200-step histories",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"V2_tree_history_differences.{suffix}", dpi=300)
    report["note"] = ("Reference: exact Numba direct sum. Wing-point induction is direct in both runs; "
                      "the tree enters only through wake-node convection. One Numba thread per run on the "
                      "2-core cloud machine; two runs in parallel; timings indicative.")
    (out / "v2-summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for model, content in report["models"].items():
        for entry in content["candidates"]:
            print(model, entry["run"], {k: f"{v:.2e}" for k, v in entry["worst"].items()},
                  "within" if entry["within_budget"] else "OUTSIDE", f"x{entry['wall_s']['speedup']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
