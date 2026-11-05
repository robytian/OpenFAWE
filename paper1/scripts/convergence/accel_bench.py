"""Benchmark and comparison harness for OF-0006-accel20261001.

Builds the archived V3/M600 free-wake configurations exactly as the retained
driver docs/tasks/OF-0006-latticegap20260923/evidence/run_lattice_comparison.py
(run_one) does: same aircraft files, reference speeds, inflow, WakeConfig
(core 0.15 m Lamb-Oseen, Euler, constant cores, Newton 60) and AeroBackend.
Only the velocity backend and the run length/resolution are varied here.
This is orchestration; no aerodynamic algorithm lives in this file.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.reference import inflow, load_coefficients, refine_aircraft  # noqa: E402
from openfawe.aero.wake import WakeConfig, wake_network  # noqa: E402
from openfawe.aero.wing import SolveSettings  # noqa: E402

CASES = {"V3": "cases/aircraft/v3_cad_camber_integral.json",
         "M600": "cases/qualification/m600_makani_reconstructed.json"}
# Archived campaign values (run_campaign.py jobs()).
ARCHIVED_DT = {"V3": 0.2, "M600": 0.02}
ALPHA = {"V3": 5.412846420261046, "M600": 0.0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_values(model):
    if model == "M600":
        return 32.9285, 1.2831, 20.0
    area, chord = 0.46 * 6.5**2, 0.395 * 6.5
    return area, chord, 5e5 * 1.4639e-5 / chord


def build(model, method="vsm_free_wake", *, alpha=None, dt=None, steps=40, refine=1,
          core_radius=0.15, wake_options=None, beta=0.0):
    """Return (backend, wing, polar, wind, area, chord, config) for one run."""
    definition = load_aircraft(REPO / CASES[model])
    if refine != 1:
        definition = refine_aircraft(definition, refine)
    wing, polar = definition.wing, definition.polar
    area, chord, speed = reference_values(model)
    alpha = ALPHA[model] if alpha is None else alpha
    dt = ARCHIVED_DT[model] if dt is None else dt
    options = dict(wake_options or {})
    config = WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=core_radius,
                        convection_integrator="euler", core_evolution="constant",
                        solve=SolveSettings(solver="newton", max_iterations=60), **options)
    backend = AeroBackend(method, wake=config)
    return backend, wing, polar, inflow(speed, alpha, beta), area, chord, config


def run(model, method="vsm_free_wake", *, steps=40, keep_states=(), progress=False, **kwargs):
    """Run an impulsive-start history; returns records, timings and kept states."""
    backend, wing, polar, wind, area, chord, config = build(model, method, steps=steps, **kwargs)
    records, kept, step_times = [], {}, []
    started = time.perf_counter()
    status, error = "complete", None
    try:
        state = backend.start(wing, polar, wind)
        for index in range(steps + 1):
            tick = time.perf_counter()
            if index:
                state = backend.advance(state, polar, wind, next_time_s=index * config.dt_s)
            step_times.append(time.perf_counter() - tick)
            solution = state.solution
            row = dict(step=index, time_s=state.time_s,
                       **load_coefficients(solution, wing, wind, area, chord),
                       Fx_N=float(solution.force[0]), Fy_N=float(solution.force[1]),
                       Fz_N=float(solution.force[2]))
            records.append(row)
            if index in keep_states:
                kept[index] = state
            if progress and index % 20 == 0:
                print(f"  {model} {method} step {index} CL={row['CL']:.6f} t={time.perf_counter()-started:.1f}s",
                      flush=True)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        state = None
    return dict(model=model, method=method, steps=steps, status=status, error=error,
                records=records, step_times=step_times, elapsed_s=time.perf_counter() - started,
                states=kept, final_state=state, config=config, wing=wing, chord=chord,
                speed=float(np.linalg.norm(wind)))


def network_of(state):
    """Wake network (nodes, edges, gamma) and edge radii of a free-wake state."""
    from openfawe.aero.cores import edge_radii
    nodes, edges, gamma = wake_network(state)
    radii = edge_radii(state.wing, edges, state.cores, state.config.core_radius_m)
    return nodes, edges, gamma, radii


def environment():
    import numpy
    info = dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), numpy=numpy.__version__)
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in stream if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    import os
    info["logical_cpus"] = os.cpu_count()
    try:
        import numba
        info["numba"] = numba.__version__
        info["numba_threads"] = numba.get_num_threads()
        try:
            info["numba_threading_layer"] = numba.threading_layer()
        except ValueError:
            info["numba_threading_layer"] = "not initialised"
    except ImportError:
        info["numba"] = None
    return info


def git_state():
    """Commit and dirtiness of the source tree the benchmark imports (None without git)."""
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True, timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "tests"], cwd=REPO,
                                    capture_output=True, text=True, check=True, timeout=30).stdout.strip())
        return dict(commit=commit, dirty=dirty)
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return str(obj)
    path.write_text(json.dumps(value, indent=1, default=default) + "\n", encoding="utf-8")
