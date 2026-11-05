"""Thread-scaling and backend benchmark on the owner's machine (OF-0006-accel20261001).

Measures, for the VSM free-wake configuration of the archived campaign:
  K  kernel timing on frozen 400-step V3/M600 wake networks: NumPy reference
     (once), Numba direct and tree (theta 0.3 / 0.2, order 2, leaf 32) at each
     thread count; thread-count bitwise invariance; tree error against direct.
  H  time-marching: 80-step histories (numpy once, numba and tree at each
     thread count) and one 400-step tree history at the largest thread count,
     with per-step wall times.
Everything is written to OUT_DIR/local-benchmark.json plus a Chinese summary.
Usage: local_benchmark.py OUT_DIR STATES_DIR [--threads 1,2,4,8,16] [--quick]
Numerical results are not interpreted here; the exact/approximate status of
each backend is established by the A3/V1/V2 evidence, not by this timing run.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import numpy as np  # noqa: E402
import accel_bench as ab  # noqa: E402
from openfawe.aero.vortex import TreeSettings, segments_velocity  # noqa: E402

TREES = {"tree-t03": TreeSettings(theta=0.3, order=2, leaf_size=32),
         "tree-t02": TreeSettings(theta=0.2, order=2, leaf_size=32)}


def cpu_name():
    try:
        import subprocess
        if os.name == "nt":
            out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20).stdout
            lines = [line.strip() for line in out.splitlines() if line.strip() and "Name" not in line]
            return lines[0] if lines else platform.processor()
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            return next(line.split(":", 1)[1].strip() for line in stream if line.startswith("model name"))
    except Exception:  # noqa: BLE001 - diagnostics only
        return platform.processor()


def timed(function, repeats):
    """Median wall time over repeats and the first result."""
    times, result = [], None
    for _ in range(repeats):
        tick = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - tick)
    return float(np.median(times)), result


def kernel_block(state_path, threads, repeats, do_numpy):
    import numba
    data = np.load(state_path)
    nodes, edges, gamma, radii = data["nodes"], data["edges"], data["gamma"], data["radii"]
    first, speed, model = int(data["first"]), float(data["speed"]), int(data["core_model"])
    points, exclusions = nodes[first:], data["exclusions"]
    common = dict(model=model, core_radius=radii, exclusions=exclusions)
    block = dict(state=Path(state_path).name, targets=len(points), segments=len(edges), rows={})
    numba.set_num_threads(1)
    direct_1 = segments_velocity(points, nodes, edges, gamma, backend="numba", **common)
    if do_numpy:
        seconds, reference = timed(lambda: segments_velocity(points, nodes, edges, gamma, **common), 1)
        block["rows"]["numpy-1"] = dict(backend="numpy", threads=1, seconds=seconds,
                                        max_diff_vs_numba_over_U=float(np.max(np.abs(reference - direct_1)) / speed))
    for count in threads:
        numba.set_num_threads(count)
        seconds, direct = timed(lambda: segments_velocity(points, nodes, edges, gamma, backend="numba", **common), repeats)
        block["rows"][f"numba-{count}"] = dict(
            backend="numba", threads=count, seconds=seconds,
            bitwise_equal_to_one_thread=bool(np.array_equal(direct.view(np.uint64), direct_1.view(np.uint64))))
        for name, settings in TREES.items():
            seconds, tree = timed(lambda: segments_velocity(points, nodes, edges, gamma, backend="tree",
                                                            tree=settings, **common), repeats)
            error = np.linalg.norm(tree - direct_1, axis=1) / speed
            block["rows"][f"{name}-{count}"] = dict(
                backend="tree", theta=settings.theta, order=settings.order, leaf=settings.leaf_size,
                threads=count, seconds=seconds, max_err_over_U=float(error.max()),
                rms_err_over_U=float(np.sqrt(np.mean(error**2))))
    return block


def history_block(model, threads, steps, long_steps, do_numpy):
    import numba
    block = dict(model=model, steps=steps, rows={})

    def run(backend, count, n, settings=None):
        numba.set_num_threads(count)
        options = dict(velocity_backend=backend)
        if settings is not None:
            options["tree"] = settings
        result = ab.run(model, "vsm_free_wake", steps=n, wake_options=options)
        final = result["records"][-1] if result["records"] else {}
        return dict(backend=backend, threads=count, steps=n, status=result["status"], error=result["error"],
                    elapsed_s=result["elapsed_s"], step_times_s=result["step_times"],
                    final={k: final.get(k) for k in ("CL", "CD", "CMy")},
                    settings=None if settings is None else dict(theta=settings.theta, order=settings.order,
                                                                 leaf=settings.leaf_size))
    if do_numpy:
        block["rows"]["numpy-1"] = run("numpy", 1, steps)
    for count in threads:
        block["rows"][f"numba-{count}"] = run("numba", count, steps)
        block["rows"][f"tree-t03-{count}"] = run("tree", count, steps, TREES["tree-t03"])
    largest = max(threads)
    block["rows"][f"tree-t03-{largest}-long"] = run("tree", largest, long_steps, TREES["tree-t03"])
    block["rows"][f"numba-{largest}-long"] = run("numba", largest, long_steps)
    return block


def summary_zh(report):
    lines = ["OpenFAWE 气动加速：本机线程扩展测试结果", f"机器：{report['environment'].get('cpu')}，"
             f"逻辑核 {report['environment'].get('logical_cpus')}，Numba {report['environment'].get('numba')}，"
             f"线程层 {report['environment'].get('numba_threading_layer')}", ""]
    for block in report["kernel"]:
        lines.append(f"[冻结尾迹核函数] {block['state']}：目标点 {block['targets']}，涡段 {block['segments']}")
        rows = block["rows"]
        base = rows.get("numpy-1", {}).get("seconds")
        for key, row in rows.items():
            extra = ""
            if row["backend"] == "tree":
                extra = f"，误差 max {row['max_err_over_U']:.1e} U / rms {row['rms_err_over_U']:.1e} U"
            if row["backend"] == "numba":
                extra = "，与单线程逐位相同" if row["bitwise_equal_to_one_thread"] else "，与单线程不同！"
            speed = f"，相对 NumPy 加速 {base / row['seconds']:.1f}x" if base else ""
            lines.append(f"  {key:>14}: {row['seconds']:8.3f} s{speed}{extra}")
        lines.append("")
    for block in report["history"]:
        lines.append(f"[时间推进] {block['model']} VSM 自由尾迹")
        for key, row in block["rows"].items():
            last = row["step_times_s"][-1] if row["step_times_s"] else float("nan")
            lines.append(f"  {key:>18}: {row['steps']} 步 {row['elapsed_s']:8.1f} s，末步 {last:.3f} s/步，"
                         f"状态 {row['status']}，CL={row['final'].get('CL')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--quick", action="store_true", help="skip the NumPy reference timings")
    parser.add_argument("--kernel-only", action="store_true", help="frozen-wake kernel timings only (no histories)")
    parser.add_argument("--name", default="local-benchmark", help="output file stem (default local-benchmark)")
    args = parser.parse_args()
    import numba
    available = numba.config.NUMBA_NUM_THREADS
    threads = sorted({min(int(v), available) for v in args.threads.split(",") if v})
    args.out.mkdir(parents=True, exist_ok=True)
    json_path, text_path = args.out / f"{args.name}.json", args.out / f"{args.name}.zh.txt"
    report = dict(environment=dict(ab.environment(), cpu=cpu_name(), numba_max_threads=available,
                                   threads_tested=threads, git=ab.git_state()),
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), kernel=[], history=[])
    # Warm the JIT caches once (excluded from timings).
    ab.run("V3", "vsm_free_wake", steps=2, wake_options=dict(velocity_backend="numba"))
    ab.run("V3", "vsm_free_wake", steps=2, wake_options=dict(velocity_backend="tree", tree=TREES["tree-t03"]))
    for state in sorted(args.states.glob("*_step0400.npz")):
        print("kernel", state.name, flush=True)
        report["kernel"].append(kernel_block(state, threads, repeats=3, do_numpy=not args.quick))
        ab.dump(json_path, report)
    if not args.kernel_only:
        for model in ("V3", "M600"):
            print("history", model, flush=True)
            report["history"].append(history_block(model, threads, steps=80, long_steps=400, do_numpy=not args.quick))
            ab.dump(json_path, report)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ab.dump(json_path, report)
    text_path.write_text(summary_zh(report), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
