"""Phase C analysis: VSM free-wake convergence ladders (OF-0006-accel20261001).

Reads run directories written by run_history.py and reports, per ladder,
the observables at the fixed physical time, signed level differences,
observed order, Richardson extrapolation and GCI (Fs = 1.25) where three
successive levels are monotone (Celik et al. 2008; verify before citing).
Usage: analyse_convergence.py OUT_PREFIX LADDER_SPEC.json
LADDER_SPEC: {"ladders": [{"id", "model", "kind": "levels"|"time",
               "runs": [dirs, coarse -> fine], "ratio": r, "time_s": T,
               "wake_chords": [10, 20, 40, 80] (kind=time)}]}
"""
import csv
import json
import math
import sys
from pathlib import Path

KEYS = ("CL", "CD", "CMy")


def load(path):
    """History of a run directory; a missing or unfinished run is reported, not fatal."""
    history = Path(path) / "history.json"
    if not history.exists():
        return dict(status="missing", error="no history.json (not run or still running)", records=[],
                    settings=dict(steps=None, backend=None, out=str(path)), dt_s=None, chord_m=None,
                    speed_m_s=None, elapsed_s=None, step_times_s=[])
    return json.loads(history.read_text(encoding="utf-8"))


def at_time(history, time_s):
    best = min(history["records"], key=lambda r: abs(r["time_s"] - time_s))
    if abs(best["time_s"] - time_s) > 1e-9 * max(1.0, time_s):
        return None
    return best


def drift(history, time_s, fraction=0.1):
    rows = [r for r in history["records"] if time_s * (1 - fraction) - 1e-12 <= r["time_s"] <= time_s + 1e-12]
    return {k: (max(r[k] for r in rows) - min(r[k] for r in rows)) if rows else None for k in KEYS}


def order_and_gci(f1, f2, f3, ratio):
    """f1 coarse, f3 fine. Returns dict or a non-monotone flag."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or (e21 > 0) != (e32 > 0):
        return dict(monotone=False, e21=e21, e32=e32)
    p = math.log(abs(e21 / e32)) / math.log(ratio)
    extrapolated = f3 + (f3 - f2) / (ratio**p - 1)
    gci_fine = 1.25 * abs((f3 - f2) / f3) / (ratio**p - 1) if f3 else None
    return dict(monotone=True, e21=e21, e32=e32, observed_order=p, richardson=extrapolated,
                gci_fine=gci_fine)


def level_ladder(spec):
    rows, values = [], []
    for run in spec["runs"]:
        history = load(run)
        record = at_time(history, spec["time_s"]) if history["records"] else None
        values.append(record)
        rows.append(dict(ladder=spec["id"], run=Path(run).name, status=history["status"], error=history["error"],
                         dt_s=history["dt_s"], steps=history["settings"]["steps"],
                         refine=history["settings"].get("refine"), core_m=history["settings"].get("core"),
                         backend=history["settings"]["backend"], theta=history["settings"].get("theta"),
                         wall_s=history["elapsed_s"],
                         **({k: record[k] for k in KEYS} if record else {k: None for k in KEYS}),
                         **{f"drift_{k}": v for k, v in drift(history, spec["time_s"]).items()}))
    analysis = {}
    for key in KEYS:
        series = [v[key] if v else None for v in values]
        steps = []
        for i in range(len(series) - 1):
            if series[i] is not None and series[i + 1] is not None:
                steps.append(dict(levels=[i, i + 1], difference=series[i + 1] - series[i],
                                  relative=(series[i + 1] - series[i]) / abs(series[i + 1])))
        triples = []
        if spec.get("ratio"):
            for i in range(len(series) - 2):
                if None not in series[i:i + 3]:
                    triples.append(dict(levels=[i, i + 1, i + 2],
                                        **order_and_gci(*series[i:i + 3], spec["ratio"])))
        analysis[key] = dict(values=series, differences=steps, triples=triples)
    return rows, analysis


def time_ladder(spec):
    history = load(spec["runs"][0])
    if not history["records"]:
        rows = [dict(ladder=spec["id"], run=Path(spec["runs"][0]).name, status=history["status"], wake_chords=c,
                     time_s=None, CL=None, CD=None, CMy=None) for c in spec["wake_chords"]]
        return rows, {k: dict(values=[None] * len(spec["wake_chords"]), differences=[]) for k in KEYS}
    chord, speed = history["chord_m"], history["speed_m_s"]
    rows, values = [], []
    for chords in spec["wake_chords"]:
        time_s = chords * chord / speed
        candidates = [r for r in history["records"] if r["time_s"] <= time_s + 1e-12]
        record = candidates[-1] if candidates and candidates[-1]["time_s"] >= time_s - history["dt_s"] else None
        values.append(record)
        rows.append(dict(ladder=spec["id"], run=Path(spec["runs"][0]).name, status=history["status"],
                         wake_chords=chords, time_s=record["time_s"] if record else None,
                         **({k: record[k] for k in KEYS} if record else {k: None for k in KEYS})))
    analysis = {}
    for key in KEYS:
        series = [v[key] if v else None for v in values]
        analysis[key] = dict(values=series, differences=[
            dict(levels=[i, i + 1], difference=series[i + 1] - series[i],
                 relative=(series[i + 1] - series[i]) / abs(series[i + 1]))
            for i in range(len(series) - 1) if series[i] is not None and series[i + 1] is not None])
    return rows, analysis


def main(prefix, spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    all_rows, report = [], dict(spec=spec, ladders={})
    for ladder in spec["ladders"]:
        rows, analysis = (time_ladder if ladder["kind"] == "time" else level_ladder)(ladder)
        all_rows.extend(rows)
        report["ladders"][ladder["id"]] = dict(rows=rows, analysis=analysis)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in all_rows for k in r}, key=lambda k: (k not in ("ladder", "run"), k))
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for ladder, content in report["ladders"].items():
        print(ladder, json.dumps({k: [None if v is None else round(v, 6) for v in content["analysis"][k]["values"]]
                                  for k in KEYS}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
