"""Acceptance A3: full archived 40-step histories, NumPy reference vs Numba backend.

Criterion (frozen in codex-brief.md before running): every step's circulation,
CL/CD/CS/CMx/CMy/CMz and force components agree to
|diff| <= 1e-10 * max(|ref|, 1e-3*|CL_ref|) (coefficients/forces) and wake node
positions to 1e-10 * reference chord. Also records wall times (same machine).
"""
import json
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import numpy as np  # noqa: E402
import accel_bench as ab  # noqa: E402

CASES = [("V3", "vsm_free_wake"), ("M600", "vsm_free_wake"), ("V3", "free_wake"), ("M600", "free_wake")]


def history(model, method, backend):
    captured = []
    backend_options = dict(velocity_backend=backend)
    result = ab.run(model, method, steps=40, wake_options=backend_options,
                    keep_states=tuple(range(41)))
    for step in range(41):
        state = result["states"].get(step)
        if state is not None:
            captured.append(dict(gamma=state.solution.circulation.copy(),
                                 rows=state.free_rows.copy()))
    return result, captured


def main(output):
    report = dict(environment=ab.environment(), criterion="A3 frozen in codex-brief.md", cases=[])
    # Compile once outside the timed runs; compile time recorded separately.
    tick = time.perf_counter()
    ab.run("V3", "vsm_free_wake", steps=1, wake_options=dict(velocity_backend="numba"))
    report["numba_warmup_s"] = time.perf_counter() - tick
    for model, method in CASES:
        ref, ref_states = history(model, method, "numpy")
        new, new_states = history(model, method, "numba")
        chord = ref["chord"]
        worst = dict(coefficient=0.0, gamma=0.0, rows_over_chord=0.0)
        channels = {}
        for a, b in zip(ref["records"], new["records"]):
            cl = abs(a["CL"])
            for key in ("CL", "CD", "CS", "CMx", "CMy", "CMz", "Fx_N", "Fy_N", "Fz_N"):
                scale = max(abs(a[key]), 1e-3 * cl) if not key.startswith("F") else max(abs(a[key]), 1e-3 * abs(a["Fz_N"]))
                relative = abs(a[key] - b[key]) / scale
                worst["coefficient"] = max(worst["coefficient"], relative)
                entry = channels.setdefault(key, dict(max_relative=0.0, max_abs_diff=0.0, max_abs_value=0.0,
                                                      max_abs_diff_over_CL_or_Fz=0.0))
                entry["max_relative"] = max(entry["max_relative"], relative)
                entry["max_abs_diff"] = max(entry["max_abs_diff"], abs(a[key] - b[key]))
                entry["max_abs_value"] = max(entry["max_abs_value"], abs(a[key]))
                lift = abs(a["Fz_N"]) if key.startswith("F") else cl
                entry["max_abs_diff_over_CL_or_Fz"] = max(entry["max_abs_diff_over_CL_or_Fz"],
                                                          abs(a[key] - b[key]) / max(lift, 1e-300))
        for a, b in zip(ref_states, new_states):
            scale = max(np.max(np.abs(a["gamma"])), 1e-300)
            worst["gamma"] = max(worst["gamma"], float(np.max(np.abs(a["gamma"] - b["gamma"])) / scale))
            if a["rows"].size:
                worst["rows_over_chord"] = max(worst["rows_over_chord"],
                                               float(np.max(np.abs(a["rows"] - b["rows"]))) / chord)
        passed = (ref["status"] == new["status"] and len(ref["records"]) == len(new["records"])
                  and worst["coefficient"] <= 1e-10 and worst["gamma"] <= 1e-10 and worst["rows_over_chord"] <= 1e-10)
        case = dict(model=model, method=method, status=[ref["status"], new["status"]],
                    errors=[ref["error"], new["error"]], frames=[len(ref["records"]), len(new["records"])],
                    worst_relative=worst, channels=channels, passed=passed,
                    elapsed_s=dict(numpy=ref["elapsed_s"], numba=new["elapsed_s"]),
                    speedup=ref["elapsed_s"] / new["elapsed_s"],
                    final=dict(numpy=ref["records"][-1], numba=new["records"][-1]))
        report["cases"].append(case)
        print(json.dumps(dict(model=model, method=method, passed=passed, worst=worst,
                              numpy_s=round(ref["elapsed_s"], 1), numba_s=round(new["elapsed_s"], 1))), flush=True)
    report["all_passed"] = all(c["passed"] for c in report["cases"])
    ab.dump(output, report)
    return report


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else TASK / "evidence" / "a3-run-equivalence.json")
