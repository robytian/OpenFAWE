"""Run one archived-configuration free-wake history and store it (OF-0006-accel20261001).

Usage:
  run_history.py OUT MODEL STEPS [--method vsm_free_wake] [--backend numba|numpy|tree]
                 [--theta T --order P --leaf L] [--dt DT] [--refine N] [--core R]
                 [--save 40,200]
                 [--rotors-omega W [--rotor-nodes N] [--rotor-surface-core R]
                  [--rotor-set PATH] [--rotor-table PATH] [--turbine-extension glauert]]
Writes OUT/history.json (records, per-step wall time, settings, environment,
status) and OUT/state_XXXX.npz wake networks at the requested steps. With
--rotors-omega the eight M600 discs (OF-0007-rotor20261001) shed rings into the
wake each step; per-step rotor loads go to OUT/rotors.json and the saved states
include the ring network.
"""
import argparse
import json
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import numpy as np  # noqa: E402
import accel_bench as ab  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.vsm import geometry  # noqa: E402
from save_states import exclusions_for  # noqa: E402


def run_with_rotors(args, options, save):
    """ab.run's loop with the rotor discs shedding rings each step (OF-0007-rotor20261001)."""
    from openfawe.rotor.actuator_disc import load_rotor_set
    from openfawe.rotor.coupled import advance_with_rotors, rotor_totals
    from openfawe.rotor.table import RotorTable
    backend, wing, polar, wind, area, chord, config = ab.build(args.model, args.method, steps=args.steps, dt=args.dt,
                                                               refine=args.refine, core_radius=args.core,
                                                               wake_options=options, alpha=args.alpha, beta=args.beta)
    discs = load_rotor_set(args.rotor_set, RotorTable.from_makani_json(args.rotor_table))
    omegas = np.full(len(discs), args.rotors_omega)
    records, kept, step_times, rotor_rows = [], {}, [], []
    started = time.perf_counter()
    status, error = "complete", None
    state = None
    try:
        state = backend.start(wing, polar, wind)
        for index in range(args.steps + 1):
            tick = time.perf_counter()
            if index:
                state, loads = advance_with_rotors(state, polar, config, wind, discs, omegas,
                                                   nodes_per_ring=args.rotor_nodes,
                                                   surface_core_m=args.rotor_surface_core,
                                                   turbine_extension=args.turbine_extension,
                                                   next_time_s=index * config.dt_s)
                force, moment = rotor_totals(loads, discs)
                rotor_rows.append(dict(step=index, time_s=state.time_s, total_force_N=force.tolist(),
                                       total_moment_Nm=moment.tolist(), loads=loads))
            step_times.append(time.perf_counter() - tick)
            solution = state.solution
            row = dict(step=index, time_s=state.time_s, **ab.load_coefficients(solution, wing, wind, area, chord),
                       Fx_N=float(solution.force[0]), Fy_N=float(solution.force[1]), Fz_N=float(solution.force[2]))
            records.append(row)
            if index in save:
                kept[index] = state
            if index % 20 == 0:
                print(f"  {args.model} {args.method}+rotors step {index} CL={row['CL']:.6f} "
                      f"t={time.perf_counter() - started:.1f}s", flush=True)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    return dict(model=args.model, method=args.method, steps=args.steps, status=status, error=error, records=records,
                step_times=step_times, elapsed_s=time.perf_counter() - started, states=kept, final_state=state,
                config=config, wing=wing, chord=chord, speed=float(np.linalg.norm(wind))), rotor_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("model")
    parser.add_argument("steps", type=int)
    parser.add_argument("--method", default="vsm_free_wake")
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--leaf", type=int, default=32)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--core", type=float, default=0.15)
    parser.add_argument("--save", default="")
    parser.add_argument("--alpha", type=float, default=None, help="angle of attack (deg); default: archived campaign value")
    parser.add_argument("--beta", type=float, default=0.0, help="sideslip (deg)")
    parser.add_argument("--rotors-omega", type=float, default=None, help="all discs at this angular speed (rad/s)")
    parser.add_argument("--rotor-nodes", type=int, default=16)
    parser.add_argument("--rotor-surface-core", type=float, default=None)
    parser.add_argument("--rotor-set", default=str(ab.REPO / "cases/rotors/m600_rotors_makani.json"))
    parser.add_argument("--rotor-table", default=str(ab.REPO / "cases/rotors/makani/rotor_rev4.json"))
    parser.add_argument("--turbine-extension", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    options = dict(velocity_backend=args.backend)
    if args.backend == "tree":
        options["tree"] = TreeSettings(theta=args.theta, order=args.order, leaf_size=args.leaf)
    save = {int(v) for v in args.save.split(",") if v}
    settings = dict(vars(args), out=str(args.out))
    try:
        import subprocess
        settings["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ab.REPO, capture_output=True,
                                                text=True, timeout=20).stdout.strip()
        settings["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "src", "tests"], cwd=ab.REPO,
                                                    capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:  # noqa: BLE001 - provenance only
        settings["git_commit"] = None
    started = time.time()
    if args.rotors_omega is None:
        result = ab.run(args.model, args.method, steps=args.steps, dt=args.dt, refine=args.refine,
                        core_radius=args.core, wake_options=options, keep_states=tuple(save), progress=True,
                        alpha=args.alpha, beta=args.beta)
        rotor_rows = None
    else:
        result, rotor_rows = run_with_rotors(args, options, save)
        ab.dump(args.out / "rotors.json", rotor_rows)
    for step, state in result["states"].items():
        nodes, edges, gamma, radii = ab.network_of(state)
        if getattr(state, "rings", None) is not None and len(state.rings.edges):
            from openfawe.aero.wake import extended_network
            nodes, edges, gamma, ring_first = extended_network(state)
            radii = np.concatenate((np.broadcast_to(radii, (len(edges) - len(state.rings.edges),)),
                                    np.full(len(state.rings.edges), state.config.core_radius_m)))
        first = state.wing.node_count
        np.savez_compressed(args.out / f"state_{step:04d}.npz", nodes=nodes, edges=edges, gamma=gamma,
                            radii=radii, first=first,
                            wing_points=np.vstack((geometry(state.wing)[0], state.wing.cp)),
                            exclusions=exclusions_for(nodes, edges, first), chord=result["chord"],
                            speed=result["speed"], core_model=int(state.config.core_model))
    report = dict(settings=settings, status=result["status"], error=result["error"],
                  frames=len(result["records"]), elapsed_s=result["elapsed_s"],
                  started_unix=started, dt_s=result["config"].dt_s, chord_m=result["chord"],
                  speed_m_s=result["speed"], records=result["records"], step_times_s=result["step_times"],
                  environment=ab.environment())
    ab.dump(args.out / "history.json", report)
    print(json.dumps(dict(out=str(args.out), status=result["status"], error=result["error"],
                          frames=len(result["records"]), elapsed_s=round(result["elapsed_s"], 1))), flush=True)


if __name__ == "__main__":
    main()
