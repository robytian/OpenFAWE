"""Figure C: VSM free-wake convergence ladders (OF-0006-accel20261001).

Reads the JSON written by analyse_convergence.py and draws, per model, the
relative deviation of CL, CD and CMy from the finest (or reference) level of
each ladder: time step, wake length, spanwise refinement and core radius
(sensitivity). One axis per panel; missing or failed levels are left out and
listed in the console. Usage: plot_convergence.py ANALYSIS.json OUT_PREFIX
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs).
SERIES = {"CL": ("#2a78d6", "o"), "CD": ("#eb6834", "s"), "CMy": ("#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CHORD = {"V3": 0.395 * 6.5, "M600": 1.2831}
SPEED = {"V3": 5e5 * 1.4639e-5 / (0.395 * 6.5), "M600": 20.0}


def ladder_axis(ladder_id, rows, model):
    """x values and label for a ladder."""
    def value(r, f):
        try:
            return f(r)
        except (TypeError, KeyError):
            return None
    if ladder_id.startswith("C1"):
        return [value(r, lambda r: r["dt_s"] * SPEED[model] / CHORD[model]) for r in rows], "U∞ Δt / c", True
    if ladder_id.startswith("C2"):
        return [r["wake_chords"] for r in rows], "wake length, chords", True
    if ladder_id.startswith("C3"):
        return [value(r, lambda r: r["refine"] or 1) for r in rows], "spanwise subdivision", False
    return [value(r, lambda r: r["core_m"] / CHORD[model]) for r in rows], "core radius / c", True


def main(analysis_path, prefix):
    report = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    ladders = report["ladders"]
    kinds = ["C1", "C2", "C3b", "C3s", "C4"]
    titles = {"C1": "Time step (fixed T)", "C2": "Wake length (archived Δt)",
              "C3b": "Spanwise panels, core 0.075 m", "C3s": "Spanwise panels, core 0.15 m / n",
              "C4": "Core radius (sensitivity)"}
    fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.0))
    missing = []
    for row, model in enumerate(("V3", "M600")):
        for column, kind in enumerate(kinds):
            ax = axes[row][column]
            key = next((k for k in ladders if k.startswith(f"{kind}-{model}-")), None)
            if key is None:
                ax.set_axis_off()
                continue
            rows = ladders[key]["rows"]
            x_all, label, log = ladder_axis(key, rows, model)
            valid = [i for i, r in enumerate(rows) if r.get("CL") is not None and x_all[i] is not None]
            if len(valid) < 2:
                missing.append(key)
                ax.text(0.5, 0.5, "fewer than two levels", transform=ax.transAxes, ha="center", color=MUTED,
                        fontsize=8)
                ax.set_axis_off()
                continue
            # Reference: finest level (C1/C3), longest wake (C2), archived core 0.15 m (C4).
            if kind == "C4":
                reference = next((i for i in valid if abs((rows[i].get("core_m") or 0.15) - 0.15) < 1e-12), valid[-1])
            else:
                reference = valid[-1]
            for name, (colour, marker) in SERIES.items():
                ref = rows[reference][name]
                deviation = [100.0 * (rows[i][name] - ref) / abs(ref) if name != "CMy" else 100.0 * (rows[i][name] - ref)
                             for i in valid]
                x = [x_all[i] for i in valid]
                ax.plot(x, deviation, color=colour, marker=marker, markersize=5, linewidth=1.6,
                        label=name if name != "CMy" else "CMy (×100, absolute)")
            ax.axhline(0.0, color=MUTED, linewidth=0.8)
            if log:
                ax.set_xscale("log")
            if kind in ("C1",):
                ax.invert_xaxis()
            ax.grid(True, which="major", color=GRID, linewidth=0.6)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if row == 0:
                ax.set_title(titles[kind], fontsize=9, color=INK)
            ax.set_xlabel(label, fontsize=8, color=INK)
            if column == 0:
                ax.set_ylabel(f"{model} VSM free wake\ndeviation from reference level, %", fontsize=8.5, color=INK)
            failed = [r["run"] for r in rows if r.get("status") not in (None, "complete", "missing")]
            pending = [r["run"] for r in rows if r.get("status") == "missing"]
            if failed:
                ax.text(0.02, 0.04, "failed: " + ", ".join(failed), transform=ax.transAxes, fontsize=6.5, color="#b3261e")
            if pending:
                ax.text(0.02, 0.10, "pending: " + ", ".join(pending), transform=ax.transAxes, fontsize=6.5, color=MUTED)
            if log:
                from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
                ticks = sorted({x_all[i] for i in valid})
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("VSM + free vortex wake: numerical sensitivity ladders (tree θ = 0.2, order 2; "
                 "reference = finest level, longest wake, or 0.15 m core)", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{prefix}.{suffix}", dpi=300)
    print("missing/incomplete ladders:", missing)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
