"""Paper 1, Fig. 4: lifting-surface (LSi, LSp) free-wake runs to the converged wake length.

Orchestration only (paper session, 2026-10-09). The backend is built exactly as the
retained drivers of OF-0006-kitererun20260929 build it (run_lattice_comparison.py for
coupling 'none' = LSi, run_comparison.py for 'grid_section' = LSp): WakeConfig(core 0.15 m,
Euler, constant cores, Newton 60), LiftingSurfaceConfig(8 cosine chordwise panels,
reconstructed_slope normals, wake_length 1000 m, release 'previous', backward_euler),
V3 LSp free wake with 2 deg initial virtual incidence. Only the run length differs
(V3 180 steps = 40 c, M600 as given), and optionally the summation engine:
--engine numba routes lifting_surface.segments_velocity to the compiled direct sum of
openfawe.aero.vortex (same kernel, same per-pair arithmetic and per-point order; the
archived lifting-surface code calls the numpy reference). The 40-step values are
checked against docs/tasks/OF-0006-kitererun20260929/evidence-v1/cases.csv.

Usage: run_ls_freewake.py OUT MODEL {LSi|LSp} ALPHA STEPS [--engine numpy|numba] [--save 40,180]
Writes OUT/history.json (records with CL, CD, CMy about the origin and about the
figure's moment reference, per-step wall time, settings, git commit, check against the
archived 40-step value).
"""
import argparse
import csv
import functools
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402

from openfawe.aero import lifting_surface as ls_module  # noqa: E402
from openfawe.aero import vortex  # noqa: E402
from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.lifting_surface import LiftingSurfaceConfig  # noqa: E402
from openfawe.aero.ll_straight import LLStraightConfig  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.reference import inflow, load_coefficients  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.aero.wing import SolveSettings  # noqa: E402

CASES = {"V3": "cases/aircraft/v3_cad_camber_integral.json",
         "M600": "cases/qualification/m600_makani_reconstructed.json"}
DT = {"V3": 0.2, "M600": 0.02}
ARCHIVED = REPO / "docs/tasks/OF-0006-kitererun20260929/evidence-v1/cases.csv"
MOMENT_REFERENCE = REPO / "docs/tasks/OF-0006-latticeloads20260924/evidence/moment-reference.json"
EXTERNAL = REPO / "docs/tasks/OF-0006-external20260924/evidence/external-comparison.json"
DENSITY = 1.225
COUPLING = {"LSi": "none", "LSp": "grid_section"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_values(model):
    if model == "M600":
        return 32.9285, 1.2831, 20.0
    area, chord = 0.46 * 6.5**2, 0.395 * 6.5
    return area, chord, 5e5 * 1.4639e-5 / chord


def moment_reference(model):
    if model == "V3":
        return np.array(json.loads(MOMENT_REFERENCE.read_text(encoding="utf-8"))["reference_points_m"]["paper_CG"], float)
    return np.array(next(r["reference_m"] for r in json.loads(EXTERNAL.read_text(encoding="utf-8"))["records"]
                         if r["model"] == "M600"), float)


def reference_cmy(row, speed, area, chord, reference):
    force = np.array([row["Fx_N"], row["Fy_N"], row["Fz_N"]])
    scale = 0.5 * DENSITY * speed**2 * area * chord
    moment = np.array([row["CMx"], row["CMy"], row["CMz"]]) - np.cross(reference, force) / scale
    return float(moment[1])


def archived_value(model, variant, alpha):
    key = ("vlm_free_wake", COUPLING[variant])
    for row in csv.DictReader(ARCHIVED.open(encoding="utf-8")):
        if row["model"] == model and (row["method"], row["coupling"]) == key and abs(float(row["alpha_deg"]) - alpha) < 1e-9:
            return dict(CL=float(row["CL"]), CD=float(row["CD"]), CMy_zero_origin=float(row["CMy_zero_origin"]),
                        CMy=float(row["CMy"]), name=row["name"])
    raise KeyError((model, variant, alpha))


def build(model, variant, alpha, steps):
    definition = load_aircraft(REPO / CASES[model])
    wing, polar = definition.wing, definition.polar
    area, chord, speed = reference_values(model)
    config = WakeConfig(dt_s=DT[model], max_rows=steps, core_radius_m=0.15, convection_integrator="euler",
                        core_evolution="constant", solve=SolveSettings(solver="newton", max_iterations=60))
    surface = dict(chordwise_panels=8, chordwise_spacing="cosine", camber_normal="reconstructed_slope",
                   wake_length_m=1000.0, wake_release="previous", gamma_derivative="backward_euler")
    if variant == "LSp":
        surface.update(polar_coupling="grid_section",
                       polar_initial_virtual_incidence_deg=2.0 if model == "V3" else 0.0)
    backend = AeroBackend("vlm_free_wake", wake=config, lifting_surface=LiftingSurfaceConfig(**surface),
                          ll_straight=LLStraightConfig(wake_length_m=1000.0))
    return backend, wing, polar, inflow(speed, alpha, 0.0), area, chord, speed, config, definition.source_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("model", choices=("V3", "M600"))
    parser.add_argument("variant", choices=("LSi", "LSp"))
    parser.add_argument("alpha", type=float)
    parser.add_argument("steps", type=int)
    parser.add_argument("--engine", default="numpy", choices=("numpy", "numba", "tree"))
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--leaf", type=int, default=32)
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.engine == "numba":
        # Same kernel, compiled direct sum; the archived lifting-surface module calls the reference.
        ls_module.segments_velocity = functools.partial(vortex.segments_velocity, backend="numba")
    elif args.engine == "tree":
        # Same policy as openfawe.aero.wake.kernel_options: only the wake-on-wake convection sum
        # (the one call of lifting_surface._convect that passes endpoint exclusions) uses the tree
        # (theta, order, leaf as in the paper); the operator and wing-point inductions stay direct.
        from openfawe.aero.vortex import TreeSettings
        tree = TreeSettings(theta=args.theta, order=args.order, leaf_size=args.leaf)

        def dispatch(points, nodes, connectivity, circulation, **kwargs):
            if "exclusions" in kwargs and len(points) > 64:
                return vortex.segments_velocity(points, nodes, connectivity, circulation, backend="tree", tree=tree, **kwargs)
            return vortex.segments_velocity(points, nodes, connectivity, circulation, backend="numba", **kwargs)
        ls_module.segments_velocity = dispatch
    backend, wing, polar, wind, area, chord, speed, config, case_sha = build(args.model, args.variant, args.alpha, args.steps)
    reference = moment_reference(args.model)
    save = {int(v) for v in args.save.split(",") if v}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    records, step_times, kept = [], [], {}
    status, error = "complete", None
    started = time.perf_counter()
    try:
        state = backend.start(wing, polar, wind)
        for index in range(args.steps + 1):
            tick = time.perf_counter()
            if index:
                state = backend.advance(state, polar, wind, next_time_s=index * config.dt_s)
            step_times.append(time.perf_counter() - tick)
            solution = state.solution
            row = dict(step=index, time_s=state.time_s, **load_coefficients(solution, wing, wind, area, chord),
                       Fx_N=float(solution.force[0]), Fy_N=float(solution.force[1]), Fz_N=float(solution.force[2]))
            row["CMy_ref"] = reference_cmy(row, speed, area, chord, reference)
            row["wake_chords"] = state.time_s * speed / chord
            records.append(row)
            if index in save:
                kept[index] = state
            if index % 20 == 0 or index == args.steps:
                print(f"  {args.model} {args.variant} a={args.alpha:+.3f} step {index} ({row['wake_chords']:.1f} c) "
                      f"CL={row['CL']:.6f} CD={row['CD']:.6f} CMy_ref={row['CMy_ref']:+.6f} "
                      f"t={time.perf_counter() - started:.1f}s", flush=True)
            if index % 10 == 0:
                (args.out / "progress.json").write_text(json.dumps(dict(step=index, elapsed_s=time.perf_counter() - started,
                                                                        records=records, step_times_s=step_times)) + "\n",
                                                        encoding="utf-8")
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    check = None
    if len(records) > 40:
        archived = archived_value(args.model, args.variant, args.alpha)
        r40 = records[40]
        check = dict(archived_run=archived["name"],
                     max_abs_difference_step40=max(abs(r40["CL"] - archived["CL"]), abs(r40["CD"] - archived["CD"]),
                                                   abs(r40["CMy"] - archived["CMy_zero_origin"]),
                                                   abs(r40["CMy_ref"] - archived["CMy"])),
                     step40=dict(CL=r40["CL"], CD=r40["CD"], CMy_zero_origin=r40["CMy"], CMy_ref=r40["CMy_ref"]),
                     archived=archived)
    report = dict(settings=dict(vars(args), out=str(args.out)), status=status, error=error, frames=len(records),
                  elapsed_s=elapsed, dt_s=config.dt_s, chord_m=chord, speed_m_s=speed, area_m2=area,
                  moment_reference_m=reference.tolist(), git_commit=git, git_dirty_src=dirty,
                  case_sha256=case_sha, archived_cases_sha256=sha(ARCHIVED), producer_sha256=sha(__file__),
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
                                   machine=platform.machine()),
                  archived_step40_check=check, records=records, step_times_s=step_times)
    (args.out / "history.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(out=str(args.out), status=status, error=error, frames=len(records), elapsed_s=round(elapsed, 1),
                          step40_check=None if check is None else check["max_abs_difference_step40"])), flush=True)


if __name__ == "__main__":
    main()
