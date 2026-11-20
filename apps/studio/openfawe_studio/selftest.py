"""OF-0013 revision 1 RS-2 and addendum 4: the packaged studio's self-test (``OpenFAWE Studio.exe --self-test [FOLDER]``).

Parts (each a dict with ``passed`` True / False, or None when skipped, and what was seen):

1. ``cases``: every bundled example (``examples/<name>/run`` at the data root) opens as a ``RunFolder`` and its last stored
   frame is drawn off screen (``video.render_frames``, 640 x 360) to a PNG that is not of a single colour.
2. ``turnkey``: ``turnkey_worker.run`` in this process: the table TurnKey (the studio's free flight), 20 periods, a frame
   every period, the bundled controller library. Each record line is parsed, keys naming a wall time (``wall`` or
   ``seconds`` in the name) are dropped and the line is written again with sorted keys; the SHA-256 of the lines is
   compared with ``selftest/turnkey-20.json`` (written by the packaging job from source with the same library). Lines that
   differ are compared as numbers as well (the largest relative difference is reported; only the identity is the release
   condition, and only on the packaging machine).
3. ``exports``: from the run of part 2: ParaView (last frame), CSV, and a GIF of the 20 frames at 10 frames/s (2 s).
4. ``gpu``: when ``openfawe.aero.vortex_cuda.available()`` in this process and in a spawned child process: the GPU tree
   code against the CPU tree code on a 468-row synthetic wake sheet, with the bounds of the GPU tree contract's GTA
   ('fp64' and 'mixed'). Addendum 20 supplement 2: skipped only without an NVIDIA driver; with a driver and no usable
   GPU (here or in the child) the part fails, with the import traceback in the report.

The report goes to ``self-test.json`` in the folder (default: ``~/OpenFAWE Studio runs/self-test-<UTC time>``) with the
paths written relative to that folder or to the data root; ``main`` returns 0 when no part failed.
"""
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

REFERENCE = "selftest/turnkey-20.json"
PERIODS = 20
GIF_FPS = 10
TIME_KEYS = ("wall", "seconds")
SHEET_ROWS, TREE, BLOCK = 468, dict(theta=0.2, order=2, leaf_size=32), 128


# ----- the records of part 2 -------------------------------------------------------------------------------------------
def canonical_lines(records_path):
    """(lines, dropped keys): every record with the wall-time keys removed, as JSON with sorted keys."""
    lines, dropped = [], set()
    for line in Path(records_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for key in [k for k in record if any(word in k.lower() for word in TIME_KEYS)]:
            dropped.add(key)
            record.pop(key)
        lines.append(json.dumps(record, sort_keys=True))
    return lines, sorted(dropped)


def digest(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _numbers(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _numbers(value[key], f"{prefix}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _numbers(item, f"{prefix}[{i}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        yield prefix, value
    else:
        yield prefix, float(value)


def largest_difference(lines, reference_lines):
    """dict(lines compared, lines differing, structure_differs, max relative difference and where; addendum 22 (REL-09):
    also the largest scaled difference |a - b| / max(1, |a|, |b|) with its values - a relative difference of a quantity
    that is nearly zero (a sideslip angle on the perch) says little)."""
    out = dict(compared=min(len(lines), len(reference_lines)), differing=0, structure_differs=len(lines) != len(reference_lines),
               max_relative=0.0, where=None, max_scaled=0.0, where_scaled=None, values_scaled=None)
    for k, (a, b) in enumerate(zip(lines, reference_lines)):
        if a == b:
            continue
        out["differing"] += 1
        x, y = dict(_numbers(json.loads(a))), dict(_numbers(json.loads(b)))
        if set(x) != set(y):
            out["structure_differs"] = True
            continue
        for key, u in x.items():
            v = y[key]
            if isinstance(u, float) and isinstance(v, float):
                if math.isnan(u) and math.isnan(v):
                    continue
                d = abs(u - v)/max(abs(v), 1e-300) if u != v else 0.0
                if d > out["max_relative"]:
                    out.update(max_relative=d, where=f"line {k+1}{key}")
                s = abs(u - v)/max(1.0, abs(u), abs(v)) if u != v else 0.0
                if s > out["max_scaled"]:
                    out.update(max_scaled=s, where_scaled=f"line {k+1}{key}", values_scaled=[u, v])
            elif u != v:
                out["structure_differs"] = True
    return out


def turnkey_run(run_dir, build=None):
    """Part 2's flight: returns (run folder, the worker's last message)."""
    import queue
    import threading
    from . import turnkey_worker
    from .spec import Spec
    spec = Spec(name="self-test", motion="free_flight", flight_aero="tables", max_periods=PERIODS, flight_frame_every=1,
                makani_build="" if build is None else str(build))
    messages, frames = queue.Queue(), queue.Queue(maxsize=1)
    turnkey_worker.run(spec.to_dict(), str(run_dir), messages, frames, threading.Event(), threading.Event())
    last = None
    while not messages.empty():
        item = messages.get_nowait()
        if item[0] != "step":
            last = item
    return Path(run_dir), last


def machine():
    """The processor as Python reports it and the logical CPU count (no computer or user name): which machine computed
    a reference. The identity of the 20 periods is the release condition only on the packaging machine (addendum 4)."""
    import os
    name = platform.processor() or platform.machine()
    try:
        from openfawe.compute.hardware import cpu_model
        name = cpu_model() or name
    except Exception:  # noqa: BLE001
        pass
    return dict(processor=name, logical_cpus=os.cpu_count(), system=platform.system())


def write_reference(out_path, run_dir, build):
    """The packaging job (from source): part 2's flight and its digest, for the bundle."""
    run, last = turnkey_run(run_dir, build)
    lines, dropped = canonical_lines(run/"records.jsonl")
    from .turnkey_worker import library_sha256
    value = dict(what="OF-0013 RS-2: the table TurnKey, 20 periods, from source in the packaging environment",
                 periods=len(lines), sha256=digest(lines), dropped_keys=dropped, last_message=list(map(str, last or ())),
                 library=library_sha256(build), python=platform.python_version(), machine=machine(), lines=lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    return value


# ----- the parts ----------------------------------------------------------------------------------------------------------
def _image_ok(image):
    import numpy as np
    image = np.asarray(image)
    return bool(image.size) and bool((image != image.reshape(-1, image.shape[-1])[0]).any())


def part_cases(root, out):
    from .runstore import RunFolder
    from .video import render_frames
    examples = sorted(p for p in (root/"examples").glob("*") if (p/"run").is_dir()) if (root/"examples").is_dir() else []
    if not examples:
        return dict(passed=None, note="no examples bundled")
    rows = {}
    for case in examples:
        row = rows.setdefault(case.name, {})
        try:
            run = RunFolder(case/"run")
            row.update(records=len(run.records), frames=len(run.frame_steps))
            if not run.frame_steps:
                row.update(passed=False, error="no stored frame")
                continue
            from PIL import Image
            image = next(render_frames(run, steps=[run.frame_steps[-1]], size=(640, 360)))
            Image.fromarray(image).save(out/f"case-{case.name}.png")
            row.update(passed=_image_ok(image), image=f"case-{case.name}.png")
        except Exception as error:  # noqa: BLE001
            row.update(passed=False, error=f"{type(error).__name__}: {error}", trace=traceback.format_exc(limit=4))
    return dict(passed=all(r.get("passed") for r in rows.values()), cases=rows)


def part_turnkey(root, out, reference):
    tick = time.time()
    run, last = turnkey_run(out/"turnkey-20")
    lines, dropped = canonical_lines(run/"records.jsonl")
    row = dict(periods=len(lines), sha256=digest(lines), dropped_keys=dropped, last_message=list(map(str, last or ())),
               wall_s=time.time() - tick)
    if reference is None:
        row.update(passed=len(lines) == PERIODS and (last or ("",))[0] == "finished", reference=None,
                   note="no reference in the bundle: only that 20 periods ran")
        return row
    same_machine = reference.get("machine") == machine()
    row.update(reference_sha256=reference["sha256"], identical=row["sha256"] == reference["sha256"],
               packaging_machine=same_machine)
    if not row["identical"]:
        row["difference"] = largest_difference(lines, reference["lines"])
    complete = len(lines) == PERIODS and (last or ("",))[0] == "finished"
    if same_machine:
        row["passed"] = row["identical"] and complete
    else:                                     # addendum 4: elsewhere the identity is reported, not required
        row["passed"] = complete
        row["note"] = ("another machine than the packaging one: the comparison with the reference is for information "
                       "(processors may round the last digits differently)")
    return row


def part_exports(out):
    run_dir = out/"turnkey-20"
    from .freeflight_io import export_flight_csv, export_flight_paraview
    from .runstore import RunFolder
    from .video import export_video
    run = RunFolder(run_dir)
    row = {}
    try:
        pvd = export_flight_paraview(run, out/"paraview", steps=[run.frame_steps[-1]])
        import vtk
        files = sorted((out/"paraview").glob("*.vtp"))
        points = 0
        for file in files:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(file))
            reader.Update()
            points += reader.GetOutput().GetNumberOfPoints()
        row["paraview"] = dict(pvd=pvd.name, vtp_files=len(files), points=points, passed=bool(files) and points > 0)
    except Exception as error:  # noqa: BLE001
        row["paraview"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
    try:
        csv_path = export_flight_csv(run, out/"records.csv")
        rows = Path(csv_path).read_text(encoding="utf-8").count("\n") - 1
        row["csv"] = dict(rows=rows, passed=rows == PERIODS)
    except Exception as error:  # noqa: BLE001
        row["csv"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
    try:
        gif = export_video(run, out/"run.gif", fps=GIF_FPS, size=(640, 360))
        from PIL import Image
        with Image.open(gif) as image:
            count = getattr(image, "n_frames", 1)
        row["video"] = dict(file=Path(gif).name, frames=count, seconds=count/GIF_FPS, passed=count == PERIODS)
    except Exception as error:  # noqa: BLE001
        row["video"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
    row["passed"] = all(part["passed"] for part in row.values())
    return row


def sheet(rows, nodes_per_row=110, span=26.0, dx=0.45, seed=0):
    """A wake-like sheet of spanwise and trailing segments: a copy of ``tree_parallel_checks.sheet``
    (docs/tasks/OF-0006-accel20261001), the generator of the GPU tree contract's synthetic sheets."""
    import numpy as np
    rng = np.random.default_rng(seed)
    y = np.linspace(-span/2, span/2, nodes_per_row)
    grid = np.zeros((rows, nodes_per_row, 3))
    for r in range(rows):
        s = r*dx
        grid[r, :, 0] = s+0.02*rng.standard_normal(nodes_per_row)
        grid[r, :, 1] = y*(1+0.002*r)+0.5*np.sin(0.01*r)
        grid[r, :, 2] = -0.03*s+0.3*np.sin(y/3.0+0.02*r)
    starts, ends = [], []
    for r in range(rows):
        starts.append(grid[r, :-1])
        ends.append(grid[r, 1:])
        if r+1 < rows:
            starts.append(grid[r])
            ends.append(grid[r+1])
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    vector = ends-starts
    length = np.linalg.norm(vector, axis=1)
    targets = grid[:min(rows, 187)].reshape(-1, 3)+np.array([0.0, 0.0, 0.011])
    gamma = np.linspace(0.5, 1.5, len(starts))
    return dict(points=targets, starts=starts, directions=vector/length[:, None], lengths=length, gamma=gamma,
                radius=np.full(len(starts), 0.075))


TRACEBACK_HEAD, TRACEBACK_TAIL = 40, 80


def short_traceback(error):
    """Addendum 20 supplement 2 (report): an exception's formatted traceback, the first 40 and last 80 lines when longer."""
    lines = "".join(traceback.format_exception(type(error), error, error.__traceback__)).splitlines()
    if len(lines) <= TRACEBACK_HEAD + TRACEBACK_TAIL:
        return lines
    return lines[:TRACEBACK_HEAD] + [f"... {len(lines) - TRACEBACK_HEAD - TRACEBACK_TAIL} lines left out ..."] \
        + lines[-TRACEBACK_TAIL:]


def _numba_cuda_origin():
    """OF-0013 addendum 20 (report): the file numba.cuda was loaded from; supplement 2: the traceback when it failed."""
    try:
        from numba import cuda
        return getattr(cuda, "__file__", None), None
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}", short_traceback(error)


STATE_MODULES = ("cuda", "cuda.core", "cuda.bindings", "cuda.pathfinder", "numba", "numba.cuda")


def _path_view(path):
    """What a module's ``__path__`` holds without iterating it (a ``_NamespacePath`` recalculates when iterated)."""
    if path is None:
        return None
    view = dict(type=type(path).__name__)
    for attr in ("_name", "_path", "_last_parent_path"):
        if hasattr(path, attr):
            value = getattr(path, attr)
            view[attr.lstrip("_")] = value if isinstance(value, str) else [str(v) for v in list(value)[:8]] \
                if isinstance(value, (list, tuple)) else repr(value)[:300]
    if isinstance(path, (list, tuple)):
        view["entries"] = [str(v) for v in path[:8]]
    return view


def import_state():
    """Addendum 20 supplement 4 (report): the modules and namespace paths the GPU import goes through."""
    rows = {}
    for name in STATE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            rows[name] = None
            continue
        spec = getattr(module, "__spec__", None)
        rows[name] = dict(file=getattr(module, "__file__", None), path=_path_view(getattr(module, "__path__", None)),
                          loader=type(getattr(spec, "loader", None)).__name__ if spec is not None else None)
    return dict(modules=rows, sys_path=[str(p) for p in sys.path[:12]],
                path_hooks=[getattr(h, "__qualname__", repr(h))[:120] for h in sys.path_hooks])


def nvidia_driver():
    """Addendum 20 supplement 2: an NVIDIA display driver is installed (its CUDA driver library loads)."""
    import ctypes
    try:
        if sys.platform == "win32":
            ctypes.WinDLL("nvcuda.dll")
        else:
            ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def _vortex_cuda():
    """``openfawe.aero.vortex_cuda`` through ``sys.modules`` (``importlib``), not the package attribute: the same module,
    and a test can stand in for it (addendum 20 supplement 2, result 3)."""
    import importlib
    return importlib.import_module("openfawe.aero.vortex_cuda")


def child_probe():
    """Addendum 20 supplement 2 (REL-02): what a spawned child process sees - a free flight runs in one."""
    import os
    row = dict(cuda_home=os.environ.get("CUDA_HOME"))
    try:
        vc = _vortex_cuda()
        row.update(available=bool(vc.available()), probe=vc.probe_error())
    except Exception as error:  # noqa: BLE001
        row.update(available=False, probe=f"{type(error).__name__}: {error}", traceback=short_traceback(error))
    return row


def _in_child(function, timeout_s=600):
    import multiprocessing as mp
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            return pool.apply_async(function).get(timeout_s)
    except Exception as error:  # noqa: BLE001
        return dict(available=False, probe=f"child process: {type(error).__name__}: {error}")


def _not_usable(row, driver):
    """No driver: skipped. A driver but no usable GPU: failed (supplement 2, REL-03)."""
    if driver:
        return dict(row, passed=False, note="an NVIDIA driver is installed but the GPU path does not work")
    return dict(row, passed=None, note="no NVIDIA driver: skipped")


def part_gpu():
    origin, origin_traceback = _numba_cuda_origin()
    driver = nvidia_driver()
    seen = dict(numba_cuda=origin, numba_cuda_traceback=origin_traceback, driver=driver)
    if origin_traceback:
        seen["import_state"] = import_state()
    try:
        vc = _vortex_cuda()
        if not vc.available():
            return _not_usable(dict(seen, probe=str(vc.probe_error())), driver)
    except Exception as error:  # noqa: BLE001
        return _not_usable(dict(seen, probe=f"{type(error).__name__}: {error}"), driver)
    child = _in_child(child_probe)
    if not child.get("available"):
        return _not_usable(dict(seen, probe="the main process can use the GPU, a child process cannot", child=child), driver)
    import numpy as np
    from openfawe.aero import vortex_cuda_tree as ct
    from openfawe.aero import vortex_numba as vn
    from openfawe.aero import vortex_tree as vt
    from openfawe.aero.vortex import CudaSettings, TreeSettings
    case = sheet(SHEET_ROWS, seed=SHEET_ROWS)
    args = (case["points"], case["starts"], case["directions"], case["lengths"], case["gamma"], case["radius"], 2,
            (np.zeros(len(case["points"])+1, dtype=np.int64), np.zeros(0, dtype=np.int64)), "raise")
    tree = TreeSettings(**TREE)
    row = dict(rows=SHEET_ROWS, targets=len(case["points"]), segments=len(case["starts"]), tree=TREE, block=BLOCK)
    tick = time.perf_counter()
    direct, direct_flag = vn.direct_sum(*args)
    row["direct_s"] = time.perf_counter() - tick
    cpu, cpu_flag = vt.tree_sum(*args, tree)
    norm = lambda v: np.linalg.norm(v, axis=1)          # noqa: E731
    tree_error = norm(cpu - direct)
    scale = float(norm(cpu).max())
    row.update(cpu_tree_flag=bool(cpu_flag), direct_flag=bool(direct_flag), max_speed=scale,
               cpu_tree_error_max=float(tree_error.max()), cpu_tree_error_rms=float(np.sqrt(np.mean(tree_error**2))))
    for precision in ("fp64", "mixed"):
        settings = CudaSettings(precision, BLOCK)
        ct.cuda_tree_sum(*args, tree, settings)                       # compilation
        gpu, flag = ct.cuda_tree_sum(*args, tree, settings)
        error = norm(gpu - cpu)
        part = dict(max=float(error.max()), rms=float(np.sqrt(np.mean(error**2))), flag_same=bool(flag) == bool(cpu_flag))
        if precision == "fp64":
            part.update(bound=1e-10*scale, passed=part["max"] <= 1e-10*scale and part["flag_same"])
        else:
            part.update(bound_max=0.1*row["cpu_tree_error_max"], bound_rms=0.1*row["cpu_tree_error_rms"],
                        passed=part["max"] <= 0.1*row["cpu_tree_error_max"] and part["rms"] <= 0.1*row["cpu_tree_error_rms"]
                        and part["flag_same"])
        row[precision] = part
    row["device"] = str(vc.cuda.get_current_device().name)
    row["passed"] = row["fp64"]["passed"] and row["mixed"]["passed"]
    row.update(seen, child=child)
    return row


def _guard(name, function, *args):
    try:
        return function(*args)
    except Exception as error:  # noqa: BLE001
        return dict(passed=False, error=f"{type(error).__name__}: {error}", trace=traceback.format_exc(limit=6))


def run_selftest(out=None):
    from . import APP_NAME, __version__, resources
    resources.ensure_library()
    root = resources.data_root()
    if out is None:
        out = Path.home()/f"{APP_NAME} runs"/time.strftime("self-test-%Y%m%d-%H%M%S", time.gmtime())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reference_file = root/REFERENCE
    reference = json.loads(reference_file.read_text(encoding="utf-8")) if reference_file.is_file() else None
    from .turnkey_worker import default_build, library_sha256
    controller = Path(default_build()).resolve()
    report = dict(what="OpenFAWE Studio self-test (OF-0013 RS-2)", version=__version__, frozen=bool(getattr(sys, "frozen", False)),
                  python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                  controller=dict(folder=str(controller.relative_to(root)) if controller.is_relative_to(root) else controller.name,
                                  library=library_sha256(controller)),
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = out/"self-test.json"

    def save():
        path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    parts = report.setdefault("parts", {})
    for name, function, args in (("cases", part_cases, (root, out)), ("turnkey", part_turnkey, (root, out, reference)),
                                 ("exports", part_exports, (out,)), ("gpu", part_gpu, ())):
        tick = time.time()
        parts[name] = _guard(name, function, *args)
        parts[name]["part_wall_s"] = time.time() - tick
        save()
    report["passed"] = all(part.get("passed") is not False for part in parts.values())
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    return report, path


def main(out=None):
    report, path = run_selftest(out)
    summary = {name: part.get("passed") for name, part in report["parts"].items()}
    if sys.stdout is not None:                  # the windowed executable on Windows has no console
        print(json.dumps(dict(passed=report["passed"], parts=summary, report=str(path)), indent=1))
    return 0 if report["passed"] else 1
