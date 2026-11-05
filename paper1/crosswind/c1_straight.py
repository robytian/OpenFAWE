"""Case C1d of the OF-0009 contract: the M600 on the nominal loop solved pose by pose with the straight wake (quasi-steady).

The same motion samples as ``c1_loop.py`` (nominal loop of the Makani land playbook at ``--wind``, sphere at the
declared strain, spline reading, 936 steps per lap at 9 m/s) are solved with the straight-wake vortex step method:
at every pose the semi-infinite straight wake is trailed along the mean relative flow of the moving panels and the
closure is solved once, so that the solution holds no history of the loop. The difference to the free-wake run of
``c1_loop.py`` at equal phase is the self-induced effect of the wake of the earlier loops (Sec. III D of the paper).
No tether (the tether does not change the aerodynamic solution of the prescribed motion). One lap by default.

Usage (direct): c1_straight.py OUT_DIR [--wind 9] [--laps 1] [--steps-per-lap N] [--strain 0.0054375]
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.handover import CROSSWIND_SOLVE  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.flight import crosswind_loop as cl  # noqa: E402
from openfawe.flight.prescribed import run_prescribed  # noqa: E402

CASE = "cases/qualification/m600_makani_reconstructed.json"
PLAYBOOK = "cases/flight/m600_crosswind_playbook_land.json"
RIGID_BODY = "cases/aircraft/m600_makani_rigid_body.json"
AREA_M2, CHORD_M = 32.9285, 1.2831
BASE_DT_S = 0.02


def main(out, argv):
    parser = argparse.ArgumentParser(prog="c1_straight.py")
    parser.add_argument("--wind", type=float, default=9.0)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--steps-per-lap", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--strain", type=float, default=0.0054375)
    parser.add_argument("--surface-guard", default="off", help="WakeConfig.surface_guard for the straight rays (default off: the loop is far from any surface)")
    args = parser.parse_args(argv)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    entries, raw_entries, playbook = cl.load_playbook(REPO/PLAYBOOK)
    constants = cl.LoopConstants.from_case(playbook)
    entry = cl.playbook_entry(entries, args.wind)
    sphere = constants.tether_length_m*(1.0+args.strain)+constants.bridle_radius_m
    centre = cl.NominalLoop.sphere_centre_distance(sphere, entry["path_radius_target"])
    loop = cl.NominalLoop(entry, [args.wind, 0.0, 0.0], constants, centre_distance_m=centre, interpolation="spline", roll_limit_tension_N=20000.0)
    period = loop.period_s()
    steps_per_lap = int(round(period/BASE_DT_S)) if args.steps_per_lap is None else args.steps_per_lap
    motion, table = loop.motion(args.laps, steps_per_lap)
    dt = table["dt_s"]
    steps = args.laps*steps_per_lap if args.steps is None else min(args.steps, args.laps*steps_per_lap)
    wind = loop.wind_world()
    definition = load_aircraft(REPO/CASE)
    config = WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=0.15, solve=CROSSWIND_SOLVE, surface_guard=args.surface_guard)
    backend = AeroBackend("vsm", wake=config)
    started, tick = time.time(), time.perf_counter()

    def progress(record):
        if record["step"] % 100 == 0 or record["step"] == steps:
            print(f"step {record['step']}/{steps} lap_angle={np.degrees(table['loop_angle_rad'][record['step']]):.0f} alpha={record['alpha_deg']:.2f} "
                  f"it={record['iterations']} |F|={np.linalg.norm(record['force_body_N']):.0f}N elapsed {time.perf_counter()-tick:.0f}s", flush=True)

    run = run_prescribed(backend, definition.wing, definition.polar, motion, wind, steps=steps, progress=progress, attitude_scheme="fourth_order_central")

    def git(*a):
        try:
            done = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=60)
            return done.stdout.strip() if done.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None
    settings = dict(case=CASE, playbook=PLAYBOOK, wind_m_s=args.wind, laps=args.laps, steps_per_lap=steps_per_lap, dt_s=dt, steps=steps, period_s=period,
                    strain=args.strain, sphere_radius_m=sphere, method="vsm straight wake, quasi-steady at every pose", core_radius_m=0.15,
                    reference_area_m2=AREA_M2, reference_chord_m=CHORD_M, loop=loop.settings(), aerodynamics=backend.metadata(),
                    scope="wing and tail aerodynamic loads; no tether, rotors, fuselage or hand-over; the straight wake holds no history",
                    script="paper1/crosswind/c1_straight.py", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    git_commit=git("rev-parse", "HEAD"), src_clean=(git("status", "--porcelain", "--", "src") == ""), platform=platform.platform())
    report = dict(settings=settings, status=run.status, error=run.error, failed_step=run.failed_step, frames=len(run.records),
                  elapsed_s=time.perf_counter()-tick, started_unix=started, K3=run.motion_consistency,
                  commands=dict(time_s=table["time_s"], loop_angle_rad=table["loop_angle_rad"], airspeed_m_s=table["airspeed_m_s"], kite_speed_m_s=table["kite_speed_m_s"]),
                  motion=dict(position_m=motion.position_m.tolist(), rotation=motion.rotation.tolist(), velocity_m_s=motion.velocity_m_s.tolist(),
                              angular_velocity_rad_s=motion.angular_velocity_rad_s.tolist()),
                  records=run.records, step_times_s=run.step_seconds)
    (out/"history.json").write_text(json.dumps(report, indent=1, default=str)+"\n", encoding="utf-8")
    print(json.dumps(dict(out=str(out), status=run.status, error=run.error, frames=len(run.records), elapsed_s=round(report["elapsed_s"], 1))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
