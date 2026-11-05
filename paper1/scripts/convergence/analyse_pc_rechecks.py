"""Phase C completion on the owner's PC: direct-sum rechecks of the finest dt levels
and the 800-step (160-chord) M600 wake-length extension.

Reads the run directories (copied from openfawe-accel/local/accel-20261001/runs)
and writes evidence/convergence-c-pc-rechecks.json with: tree-vs-direct
differences at the fixed time, the dt ladders rebuilt with direct values at
the finest levels (observed order, GCI, Celik et al. 2008 with Fs = 1.25), the
wake-length sequence at 10/20/40/80/160 chords and its drift.
Usage: analyse_pc_rechecks.py RUNS_DIR OUT.json
"""
import json
import math
import sys
from pathlib import Path

KEYS = ("CL", "CD", "CMy")


def history(runs, name):
    return json.loads((runs / name / "history.json").read_text(encoding="utf-8"))


def at(h, t):
    best = min(h["records"], key=lambda r: abs(r["time_s"] - t))
    if abs(best["time_s"] - t) > 1e-9 * max(1.0, t):
        raise ValueError(f"no record at t={t}")
    return best


def triple(f1, f2, f3, r=2.0):
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or (e21 > 0) != (e32 > 0):
        return dict(monotone=False, e21=e21, e32=e32)
    p = math.log(abs(e21 / e32)) / math.log(r)
    return dict(monotone=True, e21=e21, e32=e32, observed_order=p, richardson=f3 + e32 / (r**p - 1),
                gci_fine=1.25 * abs(e32 / f3) / (r**p - 1))


def main(runs, out):
    runs = Path(runs)
    report = dict(source="owner's PC, openfawe-accel at 589269e, run_jobs.bat 12 threads, 2026-10-01 20:26-20:49",
                  tree_vs_direct=[], dt_ladders={}, wake_length={})
    for model, T, pairs in (("M600", 1.6, [("c1-m600-dt0010", "c1d-m600-dt0010"), ("c1-m600-dt0005", "c1d-m600-dt0005")]),
                            ("V3", 16.0, [("c1-v3-dt0050", "c1d-v3-dt0050")])):
        for tree, direct in pairs:
            ht, hd = history(runs, tree), history(runs, direct)
            rt, rd = at(ht, T), at(hd, T)
            report["tree_vs_direct"].append(dict(
                model=model, tree_run=tree, direct_run=direct, time_s=T,
                direct_settings={k: hd["settings"].get(k) for k in ("backend", "steps", "git_commit", "git_dirty")},
                direct_elapsed_s=hd.get("elapsed_s"), tree_elapsed_s_cloud=ht.get("elapsed_s"),
                values={k: dict(tree=rt[k], direct=rd[k], diff=rd[k] - rt[k], relative=(rd[k] - rt[k]) / rd[k]) for k in KEYS}))
    ladders = {
        "M600": (1.6, ["c1-m600-dt0040", "c1-m600-dt0020", "c1d-m600-dt0010", "c1d-m600-dt0005"], [0.04, 0.02, 0.01, 0.005]),
        "V3": (16.0, ["c1-v3-dt0400", "c1-v3-dt0200", "c1-v3-dt0100", "c1d-v3-dt0050"], [0.4, 0.2, 0.1, 0.05]),
    }
    for model, (T, names, dts) in ladders.items():
        values = [at(history(runs, n), T) for n in names]
        entry = dict(time_s=T, runs=names, dt_s=dts, levels={}, triples={})
        for k in KEYS:
            seq = [v[k] for v in values]
            entry["levels"][k] = seq
            entry["triples"][k] = [triple(*seq[i:i + 3]) for i in range(len(seq) - 2)]
        report["dt_ladders"][model] = entry
    h = history(runs, "c2-m600-long800")
    chord, speed = h.get("chord_m") or 1.2831, h.get("speed_m_s") or 20.0
    seq = {}
    for n in (10, 20, 40, 80, 160):
        t = n * chord / speed
        r = min(h["records"], key=lambda r: abs(r["time_s"] - t))
        seq[n] = dict(time_s=r["time_s"], **{k: r[k] for k in KEYS})
    tail = [r for r in h["records"] if r["time_s"] >= 0.9 * h["records"][-1]["time_s"]]
    h4 = history(runs, "c2-m600-long400")
    r4 = min(h4["records"], key=lambda r: abs(r["time_s"] - 80 * chord / speed))
    report["wake_length"]["M600"] = dict(
        run="c2-m600-long800", settings={k: h["settings"].get(k) for k in ("backend", "steps", "theta", "order", "leaf", "git_commit")},
        elapsed_s=h.get("elapsed_s"), chord_m=chord, speed_m_s=speed, at_chords=seq,
        adjacent_relative={k: [(seq[b][k] - seq[a][k]) / seq[b][k] for a, b in ((10, 20), (20, 40), (40, 80), (80, 160))] for k in KEYS},
        drift_last_10_percent={k: max(r[k] for r in tail) - min(r[k] for r in tail) for k in KEYS},
        cloud_400_vs_pc_800_at_80_chords={k: dict(cloud=r4[k], pc=seq[80][k]) for k in KEYS})
    Path(out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in report["tree_vs_direct"]:
        print(row["model"], row["direct_run"], {k: f"{v['relative']:+.1e}" for k, v in row["values"].items()})
    for model, entry in report["dt_ladders"].items():
        for k in KEYS:
            print(model, k, [f"{v:.6f}" for v in entry["levels"][k]],
                  [("p=%.2f GCI=%.4f%%" % (t["observed_order"], 100 * t["gci_fine"])) if t["monotone"] else "non-monotone" for t in entry["triples"][k]])
    print("wake length adjacent rel %:", {k: [f"{100*v:+.3f}" for v in vals] for k, vals in report["wake_length"]["M600"]["adjacent_relative"].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
