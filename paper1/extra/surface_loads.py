"""Loads of the M600 split by lifting surface for three vortex-core radii (paper session, 2026-10-02).

Runs the unchanged repository code (commit e58677c) through the archived harness
docs/tasks/OF-0006-accel20261001/accel_bench.py: VSM + free wake, alpha 0 deg, dt 0.02 s,
tree backend theta 0.2 / order 2 / leaf 32 (the settings of the convergence ladders).
For each core radius the total coefficients (which must reproduce the ladder values)
and the contribution of the wing, the horizontal tail and the vertical tail are written.
Usage: surface_loads.py OUT.json STEPS
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


def split(state, wing, wind, area, chord, density=1.225):
    solution = state.solution
    speed = float(np.linalg.norm(wind))
    drag = wind / speed
    lift = np.cross(drag, [0.0, 1.0, 0.0])
    lift /= np.linalg.norm(lift)
    q_s = 0.5 * density * speed**2 * area
    moment = np.cross(wing.cp, solution.panel_force) + solution.panel_couple
    out, start = {}, 0
    for name, part in zip(wing.names, wing.wings):
        index = slice(start, start + part.panels)
        force = solution.panel_force[index].sum(axis=0)
        out[name] = dict(CL=float(force @ lift / q_s), CD=float(force @ drag / q_s),
                         CMy=float(moment[index].sum(axis=0)[1] / (q_s * chord)),
                         mean_alpha_deg=float(np.degrees(solution.alpha_rad[index]).mean()),
                         mean_gamma_over_Uc=float(solution.circulation[index].mean() / (speed * chord)))
        start += part.panels
    return out


def main(out, steps):
    options = dict(velocity_backend="tree", tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    result = dict(model="M600", method="vsm_free_wake", alpha_deg=0.0, dt_s=0.02, steps=steps, backend="tree 0.2/2/32",
                  git_commit=commit, cores={})
    for core in (0.075, 0.15, 0.30):
        tick = time.perf_counter()
        run = ab.run("M600", "vsm_free_wake", steps=steps, core_radius=core, wake_options=options,
                     keep_states=(steps,))
        entry = dict(status=run["status"], error=run["error"], elapsed_s=time.perf_counter() - tick)
        if run["status"] == "complete":
            state = run["states"][steps]
            backend, wing, polar, wind, area, chord, config = ab.build("M600", "vsm_free_wake", steps=steps,
                                                                       core_radius=core, wake_options=options)
            record = run["records"][-1]
            parts = split(state, state.wing, wind, area, chord)
            entry.update(total={k: record[k] for k in ("CL", "CD", "CMy")}, surfaces=parts,
                         check_sum={k: sum(p[k] for p in parts.values()) - record[k] for k in ("CL", "CD", "CMy")})
        result["cores"][f"{core:g}"] = entry
        print(core, entry.get("total"), round(entry["elapsed_s"], 1), flush=True)
    Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
