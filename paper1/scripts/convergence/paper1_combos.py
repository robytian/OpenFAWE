"""Paper 1, figures 4/5: free-wake values of VSM and LL at converged wake length, in plotted-series format.

Reads the run_history outputs of jobs-paper1-b-20261002.txt (owner's PC, local/accel-20261001/runs)
and writes, next to this script in paper1-combos/:
  plotted-series-freewake-converged.csv   rows (figure, series, wake, alpha_deg, quantity, value,
                                          ci99_halfwidth) for Fig1 (V3) and Fig2 (M600), series
                                          "OpenFAWE VSM" / "OpenFAWE LL", wake "free"; same columns
                                          and labels as OF-0006-paperfig20260930/figures-v1/plotted-series.csv
  combos-summary.json                     every value with its run, step, wake length, the value of the
                                          earlier figure (40 steps) and the difference; SHA-256 of inputs
CMy is transferred from the origin to the figure's moment reference exactly as
OF-0006-kitererun20260929/analyse_campaign.py does (CM_ref = CM_0 - r x F / (q S c)); the transfer is
checked by reproducing the earlier figure's values at step 40 of the same runs.
Usage: paper1_combos.py RUNS_DIR
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
OLD_CASES = REPO/"docs/tasks/OF-0006-kitererun20260929/evidence-v1/cases.csv"
MOMENT_REFERENCE = REPO/"docs/tasks/OF-0006-latticeloads20260924/evidence/moment-reference.json"
EXTERNAL = REPO/"docs/tasks/OF-0006-external20260924/evidence/external-comparison.json"
DENSITY = 1.225
# model: (figure label, reference area m2, reference chord m, converged step, earlier-figure step)
MODELS = {"V3": ("Fig1", 0.46*6.5**2, 0.395*6.5, 180, 40), "M600": ("Fig2", 32.9285, 1.2831, 330, 40)}
RUNS = [("V3", "vsm_free_wake", 3.0810784665258444, "p1b-v3-a308-vsm"), ("V3", "vsm_free_wake", 5.412846420261046, "p1b-v3-a541-vsm"),
        ("V3", "vsm_free_wake", 7.350324457982788, "c2-v3-a735-long360"),
        ("V3", "free_wake", 3.0810784665258444, "p1b-v3-a308-ll"), ("V3", "free_wake", 5.412846420261046, "p1b-v3-a541-ll"),
        ("V3", "free_wake", 7.350324457982788, "p1b-v3-a735-ll"),
        ("M600", "vsm_free_wake", -4.0, "p1b-m600-am4-vsm"), ("M600", "vsm_free_wake", 0.0, "p1b-m600-a0-vsm"),
        ("M600", "vsm_free_wake", 4.0, "p1b-m600-a4-vsm"),
        ("M600", "free_wake", -4.0, "p1b-m600-am4-ll"), ("M600", "free_wake", 0.0, "p1b-m600-a0-ll"),
        ("M600", "free_wake", 4.0, "p1b-m600-a4-ll")]
SERIES = {"vsm_free_wake": "OpenFAWE VSM", "free_wake": "OpenFAWE LL"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references():
    v3 = json.loads(MOMENT_REFERENCE.read_text(encoding="utf-8"))["reference_points_m"]["paper_CG"]
    m600 = next(r["reference_m"] for r in json.loads(EXTERNAL.read_text(encoding="utf-8"))["records"] if r["model"] == "M600")
    return {"V3": np.array(v3, dtype=float), "M600": np.array(m600, dtype=float)}


def reference_cmy(row, speed, area, chord, reference):
    force = np.array([row["Fx_N"], row["Fy_N"], row["Fz_N"]])
    scale = 0.5*DENSITY*speed**2*area*chord
    moment = np.array([row["CMx"], row["CMy"], row["CMz"]])-np.cross(reference, force)/scale
    return float(moment[1])


def main(runs_dir):
    runs_dir = Path(runs_dir)
    out = TASK/"paper1-combos"
    out.mkdir(exist_ok=True)
    refs = references()
    old = {(r["model"], r["method"], round(float(r["alpha_deg"]), 6)): r for r in csv.DictReader(OLD_CASES.open(encoding="utf-8"))
           if r["coupling"] == "none"}
    rows, summary, checks = [], [], []
    for model, method, alpha, name in RUNS:
        figure, area, chord, step, old_step = MODELS[model]
        path = runs_dir/name/"history.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        if history["status"] != "complete":
            raise RuntimeError(f"{name}: {history['status']} {history['error']}")
        speed, records = history["speed_m_s"], history["records"]
        final = records[step]
        earlier = records[old_step]
        values = dict(CL=final["CL"], CD=final["CD"], CMy=reference_cmy(final, speed, area, chord, refs[model]))
        early = dict(CL=earlier["CL"], CD=earlier["CD"], CMy=reference_cmy(earlier, speed, area, chord, refs[model]))
        previous = old[(model, method, round(alpha, 6))]
        archived = dict(CL=float(previous["CL"]), CD=float(previous["CD"]), CMy=float(previous["CMy"]))
        # The earlier figure used 40 steps of the same configuration: step 40 here must reproduce it.
        checks.append(dict(run=name, max_abs_difference_step40_vs_archived=max(abs(early[q]-archived[q]) for q in archived)))
        half = records[step//2]
        for quantity in ("CL", "CD", "CMy"):
            rows.append(dict(figure=figure, series=SERIES[method], wake="free", alpha_deg=repr(alpha), quantity=quantity,
                             value=repr(values[quantity]), ci99_halfwidth=""))
        summary.append(dict(model=model, method=method, alpha_deg=alpha, run=name, step=step, time_s=final["time_s"],
                            wake_chords=final["time_s"]*speed/history["chord_m"], values=values,
                            earlier_figure=dict(step=old_step, wake_chords=earlier["time_s"]*speed/history["chord_m"], values=archived),
                            change_pct={q: 100*(values[q]/archived[q]-1) for q in archived},
                            change_last_half_of_run={q: (values[q]-(half[q] if q != "CMy" else reference_cmy(half, speed, area, chord, refs[model])))
                                                     for q in values},
                            moment_reference_m=refs[model].tolist(), git_commit=history["settings"].get("git_commit"),
                            history_sha256=sha(path)))
    with (out/"plotted-series-freewake-converged.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "series", "wake", "alpha_deg", "quantity", "value", "ci99_halfwidth"])
        writer.writeheader()
        writer.writerows(rows)
    worst = max(c["max_abs_difference_step40_vs_archived"] for c in checks)
    (out/"combos-summary.json").write_text(json.dumps(dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), job_file="docs/tasks/OF-0006-viz20261001/jobs-paper1-b-20261002.txt",
        note="LSi/LSp (lifting-surface) free-wake points are not recomputed here; they remain at the earlier 40-step values",
        density_kg_m3=DENSITY, step40_reproduces_archived_max_abs=worst, step40_checks=checks, cases=summary,
        inputs={str(OLD_CASES.relative_to(REPO)): sha(OLD_CASES), str(MOMENT_REFERENCE.relative_to(REPO)): sha(MOMENT_REFERENCE),
                str(EXTERNAL.relative_to(REPO)): sha(EXTERNAL)}), indent=1)+"\n", encoding="utf-8")
    print("step-40 reproduction of the earlier figure values: max |difference| = %.2e" % worst)
    for s in summary:
        v, a = s["values"], s["earlier_figure"]["values"]
        print(f"{s['model']:4s} {s['method']:13s} alpha {s['alpha_deg']:+.4f} @ {s['wake_chords']:.0f} c: CL {v['CL']:.5f} ({s['change_pct']['CL']:+.2f}%)"
              f" CD {v['CD']:.5f} ({s['change_pct']['CD']:+.2f}%) CMy {v['CMy']:+.5f} (was {a['CMy']:+.5f})")


if __name__ == "__main__":
    main(sys.argv[1])
