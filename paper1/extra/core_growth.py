"""Age-dependent vortex-core growth against the constant core (paper session, 2026-10-08).

Runs the unchanged repository code (commit e58677c) with the archived harness settings of the
convergence ladders (VSM + free wake, Euler convection, Newton closure, tree backend theta 0.2 /
order 2 / leaf 32) and WakeConfig.core_evolution = 'age' (OLAF growth law
rc^2 = rc0^2 + 4 alpha delta nu t, alpha = 1.25643, nu = 1.4639e-5 m2/s) for one eddy-viscosity
factor delta, or 'constant'. Writes OUT/result.json (history, final loads, per-surface split for
the M600) and OUT/state.npz (final wake network with edge radii).
Usage: core_growth.py OUT MODEL STEPS DELTA   (DELTA = 0 -> constant core)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper1/scripts/convergence"))
import accel_bench as ab  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.aero.wing import SolveSettings  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_loads import split  # noqa: E402


def main(out, model, steps, delta):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model])
    wing, polar = definition.wing, definition.polar
    area, chord, speed = ab.reference_values(model)
    wind = ab.inflow(speed, ab.ALPHA[model], 0.0)
    evolution = "constant" if delta == 0 else "age"
    config = WakeConfig(dt_s=ab.ARCHIVED_DT[model], max_rows=steps, core_radius_m=0.15, convection_integrator="euler",
                        core_evolution=evolution, eddy_viscosity_factor=float(delta) if delta else 1.0,
                        solve=SolveSettings(solver="newton", max_iterations=60), velocity_backend="tree",
                        tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
    backend = AeroBackend("vsm_free_wake", wake=config)
    records, status, error = [], "complete", None
    tick = time.perf_counter()
    try:
        state = backend.start(wing, polar, wind)
        for index in range(steps + 1):
            if index:
                state = backend.advance(state, polar, wind, next_time_s=index * config.dt_s)
            records.append(dict(step=index, time_s=state.time_s, **ab.load_coefficients(state.solution, wing, wind, area, chord)))
            if index % 20 == 0:
                print(model, delta, index, round(records[-1]["CL"], 6), round(time.perf_counter() - tick, 1), flush=True)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        status, error, state = "failed", f"{type(exc).__name__}: {exc}", None
    result = dict(model=model, steps=steps, dt_s=config.dt_s, eddy_viscosity_factor=delta, core_evolution=evolution,
                  core_radius_m=0.15, kinematic_viscosity_m2_s=config.kinematic_viscosity_m2_s, status=status, error=error,
                  elapsed_s=time.perf_counter() - tick, chord_m=chord, speed_m_s=float(np.linalg.norm(wind)),
                  wake_chords=steps * config.dt_s * float(np.linalg.norm(wind)) / chord, records=records,
                  git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    if state is not None:
        result["final"] = records[-1]
        if model == "M600":
            result["surfaces"] = split(state, state.wing, wind, area, chord)
        nodes, edges, gamma, radii = ab.network_of(state)
        np.savez_compressed(out / "state.npz", nodes=nodes, edges=edges, gamma=gamma, radii=radii, first=state.wing.node_count,
                            chord=chord, speed=float(np.linalg.norm(wind)))
        result["core_radius_range_m"] = [float(radii.min()), float(radii.max())]
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(dict(model=model, delta=delta, status=status, error=error, final=result.get("final"),
                          elapsed=round(result["elapsed_s"], 1))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
