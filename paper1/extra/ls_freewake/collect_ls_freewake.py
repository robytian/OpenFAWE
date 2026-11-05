"""Collect the lifting-surface free-wake runs (runs/lsf-*) into the plotted-series file of Fig. 4.

Writes ../../fig/plotted-series-freewake-converged.csv = the v1 rows (VSM, LL; kept verbatim from
plotted-series-freewake-converged-v1.csv) + rows for "OpenFAWE LSi"/"OpenFAWE LSp" at the final step,
and ls-freewake-summary.json with every value, its run, wake length, the 40-step value of the earlier
figure (archived cases.csv) and the change, the step-40 reproduction check, the tree-against-direct
check on V3, and the SHA-256 of each history.json.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "fig"
V1 = FIG / "plotted-series-freewake-converged-v1.csv"
OUT = FIG / "plotted-series-freewake-converged.csv"
RUNS = [("V3", "Fig1", "LSi", "lsf-v3-a308-lsi"), ("V3", "Fig1", "LSi", "lsf-v3-a541-lsi"), ("V3", "Fig1", "LSi", "lsf-v3-a735-lsi"),
        ("V3", "Fig1", "LSp", "lsf-v3-a308-lsp"), ("V3", "Fig1", "LSp", "lsf-v3-a541-lsp"), ("V3", "Fig1", "LSp", "lsf-v3-a735-lsp"),
        ("M600", "Fig2", "LSi", "lsf-m600-am4-lsi"), ("M600", "Fig2", "LSi", "lsf-m600-a0-lsi"), ("M600", "Fig2", "LSi", "lsf-m600-a4-lsi"),
        ("M600", "Fig2", "LSp", "lsf-m600-am4-lsp"), ("M600", "Fig2", "LSp", "lsf-m600-a0-lsp"), ("M600", "Fig2", "LSp", "lsf-m600-a4-lsp")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not V1.exists():
        OUT.rename(V1)
    rows = list(csv.DictReader(V1.open(encoding="utf-8")))
    summary, missing = [], []
    for model, figure, variant, name in RUNS:
        path = HERE / "runs" / name / "history.json"
        if not path.exists():
            missing.append(name)
            continue
        h = json.loads(path.read_text(encoding="utf-8"))
        if h["status"] != "complete":
            missing.append(f"{name} ({h['status']}: {h['error']})")
            continue
        final, r40 = h["records"][-1], h["records"][40]
        alpha = h["settings"]["alpha"]
        values = dict(CL=final["CL"], CD=final["CD"], CMy=final["CMy_ref"])
        for q, v in values.items():
            rows.append(dict(figure=figure, series=f"OpenFAWE {variant}", wake="free", alpha_deg=repr(alpha), quantity=q,
                             value=repr(v), ci99_halfwidth=""))
        archived = h["archived_step40_check"]["archived"]
        summary.append(dict(model=model, variant=variant, alpha_deg=alpha, run=name, engine=h["settings"]["engine"],
                            steps=h["settings"]["steps"], wake_chords=final["wake_chords"], values=values,
                            change_from_40_steps_pct={q: 100 * (values[q] / archived[q] - 1) for q in ("CL", "CD", "CMy")},
                            change_last_tenth_of_run={q: values[q] - h["records"][int(0.9 * h["settings"]["steps"])][k]
                                                      for q, k in (("CL", "CL"), ("CD", "CD"), ("CMy", "CMy_ref"))},
                            step40_reproduces_archived_max_abs=h["archived_step40_check"]["max_abs_difference_step40"],
                            elapsed_s=h["elapsed_s"], git_commit=h["git_commit"], history_sha256=sha(path)))
    check = None
    direct, tree = HERE / "runs/lsf-v3-a541-lsp/history.json", HERE / "runs/lsf-v3-a541-lsp-tree/history.json"
    if direct.exists() and tree.exists():
        a, b = (json.loads(p.read_text(encoding="utf-8"))["records"] for p in (direct, tree))
        n = min(len(a), len(b))
        check = dict(runs=[direct.parent.name, tree.parent.name], steps_compared=n,
                     max_rel_dCL=max(abs(b[i]["CL"] - a[i]["CL"]) / abs(a[i]["CL"]) for i in range(1, n)),
                     max_rel_dCD=max(abs(b[i]["CD"] - a[i]["CD"]) / abs(a[i]["CD"]) for i in range(1, n)),
                     max_abs_dCMy=max(abs(b[i]["CMy_ref"] - a[i]["CMy_ref"]) for i in range(1, n)),
                     final=dict(direct=dict(CL=a[n - 1]["CL"], CD=a[n - 1]["CD"], CMy=a[n - 1]["CMy_ref"]),
                                tree=dict(CL=b[n - 1]["CL"], CD=b[n - 1]["CD"], CMy=b[n - 1]["CMy_ref"])))
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "series", "wake", "alpha_deg", "quantity", "value", "ci99_halfwidth"])
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "ls-freewake-summary.json").write_text(json.dumps(dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), producer_sha256=sha(__file__), runner_sha256=sha(HERE / "run_ls_freewake.py"),
        v1_rows_sha256=sha(V1), output=str(OUT.relative_to(HERE.parents[1])), output_sha256=sha(OUT),
        missing=missing, tree_against_direct_v3=check, cases=summary), indent=1) + "\n", encoding="utf-8")
    for s in summary:
        v, c = s["values"], s["change_from_40_steps_pct"]
        print(f"{s['model']:4s} {s['variant']} a={s['alpha_deg']:+.3f} @ {s['wake_chords']:.0f} c ({s['engine']}): "
              f"CL {v['CL']:.5f} ({c['CL']:+.2f}% vs 40 steps) CD {v['CD']:.5f} ({c['CD']:+.2f}%) CMy {v['CMy']:+.5f}")
    if missing:
        print("missing:", missing)
    if check:
        print("tree vs direct (V3 LSp 5.41):", {k: v for k, v in check.items() if k.startswith("max")})


if __name__ == "__main__":
    main()
