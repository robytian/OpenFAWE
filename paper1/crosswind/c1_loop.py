"""Case C1 of the paper task (OF-0009): the M600 on the nominal crosswind loop of the Makani land playbook, with the tether,
and the wake network stored at the end of every lap. A copy of ``OF-0007-g4motion20261002/a5_loop.py`` (its text follows)
with two additions: ``--save-laps`` stores the wake network at the end of every lap as ``state_<step>.npz``
(nodes, edges, gamma, first = node count of the lifting surfaces; the format of ``a2_circle.py``), and the receipt names
this script. Nothing else is changed: the motion, the hand-over, the tether warm-up and the far-wake flags are those of
A5, so ``--free-rows 0 --keep-rows 0`` runs the complete free wake (no far-wake treatment).

Original text of a5_loop.py:

Motion (contract section 4, settings of section 8, entry of 2026-10-05): the body origin runs on the controller's
nominal circle at the playbook wind speed (``openfawe.flight.crosswind_loop``): radius, elevation and azimuth offset of
the smoothed land playbook entry, the kite speed that gives the commanded airspeed in the uniform wind, the attitude of
the controller's command (incidence and sideslip schedules, tether roll from the curvature balance with the
feed-forward term and the saturation band). The circle is put on the sphere the tether and the bridle reach
(``--strain`` is the declared tether strain that sets that sphere; ``--centre makani`` puts the centre at the tether
length as the controller does, which the real tether cannot reach and which is therefore not run with the tether).
The tables are read through a periodic cubic spline so that the motion is C2 (``--interpolation linear`` is the
controller's reading; its samples are not self-consistent to 1e-6 and K3 then says so). The time step is one lap
divided by ``--steps-per-lap`` (default: the lap time over 0.02 s rounded, 936 at 9 m/s), so that equal phases of
consecutive laps fall on steps (K7 subtracts step by step).

Aerodynamics as in A3/A4 (``a3_pitch.py``): VSM + free wake with the section hand-over (Makani weight from the body
origin incidence and sideslip; zero seeding with one recorded retry), tree backend, constant core 0.15 m, explicit
Euler. Far wake by decision 0028 item 1(c): the newest ``--free-rows`` rows convect freely (default a tenth of a lap),
only the newest ``--keep-rows`` rows are kept (default one lap). The tether (``openfawe.flight.prescribed.PrescribedTether``:
lumped line, bridle point, 10 sub-steps) is first run alone along ``--warmup-laps`` laps of the same motion and the
coupled run starts from the state it reached (contract section 8, 2026-10-04 02:00: the start transient is the
caller's to declare); its lap-to-lap tensions are in the receipt. ``--assembly pylons`` runs the four pylons alone on
the same motion (decision 0028: their loads are added afterwards; the tether is then left out so that it is counted
once). ``--rotors OMEGA`` adds the eight rotors at a constant speed as in A4 (no speed is declared for A5 yet).

Reported (thresholds are those frozen in the contract; none is set here):
  K3  self-consistency of the motion samples with the fourth-order attitude check (contract section 8, 2026-10-05;
      the angular acceleration is not stored with the motion, see ``NominalLoop.motion``);
  K7  periodicity of the total body loads (aerodynamic plus tether, body axes, about the body origin) of lap n against
      lap n-1 at equal phase, by the contract rule (lap 3 against lap 2 within 1 %, else lap 5 against lap 4). Under
      the far-wake truncation this is only half of K7: the second half, the comparison with a run that keeps two laps,
      is made by ``a5_truncation_compare.py`` (contract section 8, 2026-10-05), and this script's verdict says so;
  K9  retries, failed step, stall switches over the limit, steps with weight > 0 (the share the contract asks for);
  the residual force and moment of the rigid body (``openfawe.flight.residual``; the loads of this run only, the
  pylons' loads are added by ``a5_pair.py``), the tether tension range per lap, the power account and the K5-type
  impulses per lap.
A failed step stops the run and is recorded; nothing is smoothed or replaced.

Usage (direct):      a5_loop.py OUT_DIR [--wind 9] [--laps 3] [--steps-per-lap N] [--steps N] [--backend tree|numba|numpy]
                                        [--assembly main|pylons] [--seeding zero|previous] [--strain 0.0] [--centre sphere|makani]
                                        [--interpolation spline|linear] [--roll-limit-tension 20000] [--no-tether]
                                        [--warmup-laps 2] [--free-rows N] [--keep-rows K] [--rotors OMEGA [--ring-surface-core 0.3]]
Usage (PC job line): NAME --script paper1/crosswind/c1_loop.py --strain 0.0054375 --free-rows 0 --keep-rows 0 --save-laps
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
G4 = TASK/"g4"
REPO = TASK.parents[1]
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero.aircraft import PanelPolars, WingAssembly, load_aircraft  # noqa: E402
from openfawe.aero.handover import CROSSWIND_SOLVE, SectionHandover  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig, wake_network  # noqa: E402
from openfawe.flight import crosswind_loop as cl  # noqa: E402
from openfawe.flight.prescribed import PrescribedRotors, PrescribedTether, run_prescribed, run_tether_alone  # noqa: E402
from openfawe.flight.residual import account, load_rigid_body  # noqa: E402
from openfawe.rotor.actuator_disc import load_rotor_set  # noqa: E402
from openfawe.rotor.table import RotorTable  # noqa: E402
from openfawe.tether.bridle import load_bridle  # noqa: E402
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings  # noqa: E402

LOW_CASE = "cases/qualification/m600_makani_reconstructed.json"
HIGH_CASE = "cases/qualification/m600_makani_xfoil_polars.json"
LOW_PYLON_CASE = "cases/qualification/m600_makani_reconstructed_pylons.json"
HIGH_PYLON_CASE = "cases/qualification/m600_makani_xfoil_polars_pylons.json"
PLAYBOOK = "cases/flight/m600_crosswind_playbook_land.json"
RIGID_BODY = "cases/aircraft/m600_makani_rigid_body.json"
ROTOR_SET = "cases/rotors/m600_rotors_makani.json"
ROTOR_TABLE = "cases/rotors/makani/rotor_rev4.json"
AREA_M2, CHORD_M = 32.9285, 1.2831          # reference area and chord of the archived M600 runs
BASE_DT_S = 0.02
K3_LIMIT, K7_LIMIT, SWITCH_LIMIT = 1e-6, 0.01, 4
CONTRACT = dict(wind_m_s=9.0, strain=0.0, centre="sphere", interpolation="spline", roll_limit_tension_N=20000.0, warmup_laps=2,
                seeding="zero", backend="tree", substeps=10, far_wake_free_lap_fraction=0.1, far_wake_keep_laps=1)
TRUNCATION = ("K7 under the far-wake truncation has two parts (contract section 8, 2026-10-05): the lap-to-lap periodicity "
              "judged here and the comparison with the run that keeps two laps, judged by a5_truncation_compare.py; "
              "this run alone gives at most 'periodicity passed'")


def _script(name):
    spec = importlib.util.spec_from_file_location(f"g4_{name}_for_c1", G4/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A3 = _script("a3_pitch")


def consistency_verdict(report):
    """K3 from a ``RigidMotion.consistency`` report: worst relative entry against the frozen limit."""
    entries = {k: v for k, v in report.items() if isinstance(v, dict)}
    relative = {k: v["relative"] for k, v in entries.items() if v["relative"] is not None and v["compared"] > 0}
    worst = max(relative.values()) if relative else None
    return dict(limit=K3_LIMIT, worst_relative=worst, passed=None if worst is None else bool(worst <= K3_LIMIT),
                entries_in_verdict=sorted(relative), attitude_scheme=report.get("attitude_scheme"),
                zero_scale_entries={k: v["max_abs"] for k, v in entries.items() if v["relative"] is None}, report=report)


def first_samples(motion, count):
    """The motion restricted to its first ``count`` samples (the laps are identical, so the warm-up can reuse them)."""
    from openfawe.flight.kinematics import RigidMotion
    return RigidMotion(motion.time_s[:count], motion.position_m[:count], motion.rotation[:count], motion.velocity_m_s[:count],
                       motion.angular_velocity_rad_s[:count],
                       None if motion.acceleration_m_s2 is None else motion.acceleration_m_s2[:count],
                       None if motion.angular_acceleration_rad_s2 is None else motion.angular_acceleration_rad_s2[:count],
                       description=motion.description)


def contract_case(args, with_tether):
    """True when the run carries the settings of the contract entry (2026-10-05 17:59) in either assembly of decision 0028.

    The strain is part of them (the +-0.0005 runs are trials), and a shortened run (``--steps``) is never the contract case.
    """
    return bool(args.wind == CONTRACT["wind_m_s"] and args.backend == CONTRACT["backend"] and args.seeding == CONTRACT["seeding"]
                and args.interpolation == CONTRACT["interpolation"] and args.centre == CONTRACT["centre"]
                and args.strain == CONTRACT["strain"] and args.steps is None
                and args.roll_limit_tension == CONTRACT["roll_limit_tension_N"] and args.steps_per_lap is None
                and args.free_rows is None and args.keep_rows is None and args.rotors is None
                and (with_tether or args.assembly == "pylons") and args.warmup_laps == CONTRACT["warmup_laps"] and args.laps >= 3)


def total_loads(record):
    """Body-axis force and moment about the body origin of everything this run modelled on the body."""
    force = np.asarray(record["force_body_N"], dtype=float)
    moment = np.asarray(record["moment_origin_body_Nm"], dtype=float)
    for name in ("tether", "rotor"):
        if f"{name}_force_body_N" in record:
            force = force+np.asarray(record[f"{name}_force_body_N"], dtype=float)
            moment = moment+np.asarray(record[f"{name}_moment_origin_body_Nm"], dtype=float)
    return force, moment


def periodicity(records, steps_per_lap, loads=total_loads):
    """Per lap pair, the largest relative change of the loads at equal phase (lap n step k against lap n-1 step k)."""
    force = np.array([loads(r)[0] for r in records])
    moment = np.array([loads(r)[1] for r in records])
    pairs, lap = [], 2
    while lap*steps_per_lap < len(records):
        now = slice((lap-1)*steps_per_lap+1, lap*steps_per_lap+1)
        before = slice((lap-2)*steps_per_lap+1, (lap-1)*steps_per_lap+1)
        entry = dict(lap=lap, against=lap-1)
        for name, values in (("force", force), ("moment", moment)):
            scale = np.linalg.norm(values[before], axis=1)
            difference = np.linalg.norm(values[now]-values[before], axis=1)/np.where(scale > 0., scale, np.inf)
            entry[f"{name}_max_relative"] = float(difference.max())
            entry[f"{name}_mean_relative"] = float(difference.mean())
            entry[f"{name}_max_absolute"] = float(np.linalg.norm(values[now]-values[before], axis=1).max())
        pairs.append(entry)
        lap += 1
    return pairs


def periodicity_verdict(pairs, status, contract_case):
    """The lap-to-lap half of K7 by the contract rule: "periodicity passed" / "failed" / "not evaluated" with the reason."""
    if status != "complete":
        return dict(verdict="not evaluated", reason="the run failed", scope=TRUNCATION)
    if not contract_case:
        return dict(verdict="not evaluated", reason="settings differ from contract case A5", scope=TRUNCATION)
    by_lap = {p["lap"]: p for p in pairs}

    def within(lap):
        return max(by_lap[lap]["force_max_relative"], by_lap[lap]["moment_max_relative"]) <= K7_LIMIT
    if 3 not in by_lap:
        return dict(verdict="not evaluated", reason="fewer than three laps", scope=TRUNCATION)
    if within(3):
        return dict(verdict="periodicity passed", reason="lap 3 against lap 2 within the limit; the truncation comparison is still needed", scope=TRUNCATION)
    if 5 not in by_lap:
        return dict(verdict="not evaluated", reason="lap 3 against lap 2 exceeds the limit and lap 5 was not computed", scope=TRUNCATION)
    return dict(verdict="periodicity passed" if within(5) else "failed", reason="decided by lap 5 against lap 4", scope=TRUNCATION)


def per_lap(records, steps_per_lap, dt):
    """Per completed lap: impulses of the total loads, tether tension range, powers, hand-over counts."""
    rows, lap = [], 1
    while lap*steps_per_lap < len(records):
        part = records[(lap-1)*steps_per_lap+1:lap*steps_per_lap+1]
        force = np.array([total_loads(r)[0] for r in part])
        moment = np.array([total_loads(r)[1] for r in part])
        aero = np.array([r["force_body_N"] for r in part])
        row = dict(lap=lap, steps=len(part),
                   total_force_impulse_body_Ns=(dt*force.sum(axis=0)).tolist(), total_moment_impulse_origin_body_Nms=(dt*moment.sum(axis=0)).tolist(),
                   aero_force_impulse_body_Ns=(dt*aero.sum(axis=0)).tolist(),
                   aero_force_body_N=dict(largest=float(np.linalg.norm(aero, axis=1).max()), smallest=float(np.linalg.norm(aero, axis=1).min())),
                   aero_power_W=dict(mean=float(np.mean([r["aero_power_W"] for r in part])), smallest=float(min(r["aero_power_W"] for r in part)),
                                     largest=float(max(r["aero_power_W"] for r in part))),
                   steps_weight_positive=int(sum(r["handover_weight"] > 0. for r in part)),
                   retries=sum(r["seeding"] == "retry_previous" for r in part),
                   steps_with_stall_switches_over_limit=int(sum((r["stall_switches"] or 0) > SWITCH_LIMIT for r in part)),
                   largest_iterations=int(max(r["iterations"] for r in part)))
        if "tether_tension_kite_N" in part[0]:
            tension = np.array([r["tether_tension_kite_N"] for r in part])
            row.update(tether_tension_kite_N=dict(smallest=float(tension.min()), largest=float(tension.max()), mean=float(tension.mean())),
                       tether_slack_steps=int(sum(r["tether_slack_segments"] > 0 for r in part)),
                       tether_lowest_node_height_m=float(min(r["tether_lowest_node_height_m"] for r in part)),
                       tether_power_on_body_W=dict(mean=float(np.mean([r["tether_power_on_body_W"] for r in part]))))
        if "rotor_shaft_power_W" in part[0]:
            shaft = np.array([r["rotor_shaft_power_W"] for r in part])
            row.update(rotor_shaft_power_W=dict(mean=float(shaft.sum(axis=1).mean())))
        rows.append(row)
        lap += 1
    return rows


def residual_summary(rows, summary, steps_per_lap):
    """The residual account condensed: largest and lap-3 statistics of the required force and moment."""
    if not rows:
        return summary
    required = np.array([r["required_force_inertial_N"] for r in rows])
    moment = np.array([r["required_moment_cm_inertial_Nm"] for r in rows])
    modelled = np.array([r["modelled_force_inertial_N"] for r in rows])
    out = dict(summary)
    out["required_force_N_over_modelled"] = float(np.linalg.norm(required, axis=1).max()/max(np.linalg.norm(modelled, axis=1).max(), 1e-300))
    lap3 = slice(2*steps_per_lap+1, 3*steps_per_lap+1)
    if len(rows) > 3*steps_per_lap:
        out["lap_3"] = dict(required_force_N=dict(largest=float(np.linalg.norm(required[lap3], axis=1).max()),
                                                  mean=float(np.linalg.norm(required[lap3], axis=1).mean())),
                            required_moment_Nm=dict(largest=float(np.linalg.norm(moment[lap3], axis=1).max()),
                                                    mean=float(np.linalg.norm(moment[lap3], axis=1).mean())),
                            modelled_force_N=dict(largest=float(np.linalg.norm(modelled[lap3], axis=1).max())),
                            power_W={k: float(np.mean([r["power_W"][k] for r in rows[lap3]])) for k in rows[lap3.start]["power_W"]})
    gaps = [abs(r["kinetic_energy_rate_W"]-r["kinetic_energy_rate_from_samples_W"]) for r in rows if "kinetic_energy_rate_from_samples_W" in r]
    out["kinetic_energy_rate_gap_W"] = dict(largest=float(max(gaps)), mean=float(np.mean(gaps))) if gaps else None
    return out


def main(out, argv):
    parser = argparse.ArgumentParser(prog="a5_loop.py")
    parser.add_argument("--wind", type=float, default=CONTRACT["wind_m_s"], help="uniform wind speed along +x [m/s], also the playbook wind")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--steps-per-lap", type=int, default=None, help="default: lap time / 0.02 s, rounded")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps (smoke runs)")
    parser.add_argument("--backend", default="tree", choices=("tree", "numba", "numpy"))
    parser.add_argument("--assembly", default="main", choices=("main", "pylons"))
    parser.add_argument("--seeding", default=CONTRACT["seeding"], choices=("zero", "previous"))
    parser.add_argument("--strain", type=float, default=CONTRACT["strain"], help="declared tether strain that sets the sphere the circle lies on")
    parser.add_argument("--centre", default=CONTRACT["centre"], choices=("sphere", "makani"),
                        help="sphere: centre so that the kite stays on the tether-plus-bridle sphere; makani: centre at the tether length (no tether)")
    parser.add_argument("--interpolation", default=CONTRACT["interpolation"], choices=("spline", "linear"))
    parser.add_argument("--roll-limit-tension", type=float, default=CONTRACT["roll_limit_tension_N"],
                        help="tension the saturation band of the roll command is evaluated at [N]; negative: the wide band of a load-cell fault")
    parser.add_argument("--no-tether", action="store_true", help="aerodynamics only (the residual then lacks the tether)")
    parser.add_argument("--warmup-laps", type=int, default=CONTRACT["warmup_laps"], help="laps the line runs alone before the coupled run")
    parser.add_argument("--free-rows", type=int, default=None, help="far wake: newest rows convected freely (default a tenth of a lap)")
    parser.add_argument("--keep-rows", type=int, default=None, help="far wake: newest rows kept (default one lap; 0: keep everything)")
    parser.add_argument("--rotors", type=float, default=None, metavar="OMEGA", help="the eight rotors at this constant angular speed [rad/s]")
    parser.add_argument("--ring-surface-core", type=float, default=A3.A4_SURFACE_CORE_M)
    parser.add_argument("--ring-nodes", type=int, default=A3.A4_RING_NODES)
    parser.add_argument("--save-laps", action="store_true", help="store the wake network at the end of every lap (state_<step>.npz)")
    args = parser.parse_args(argv)
    if args.laps < 1 or args.warmup_laps < 0:
        raise SystemExit("--laps must be positive and --warmup-laps non-negative")
    if args.centre == "makani" and not args.no_tether:
        raise SystemExit("the centre at the tether length puts the kite beyond the tether's reach: use it with --no-tether")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    # ----- the motion -----------------------------------------------------------------------------------------------
    entries, raw_entries, playbook = cl.load_playbook(REPO/PLAYBOOK)
    constants = cl.LoopConstants.from_case(playbook)
    entry = cl.playbook_entry(entries, args.wind)
    bridle = load_bridle(REPO/RIGID_BODY)
    if abs(constants.tether_length_m-TetherProperties().length_m) > 1e-9 or abs(constants.bridle_radius_m-bridle.radius_m) > 1e-9:
        raise SystemExit("the playbook case's tether length or bridle radius differs from the line and bridle the run uses")
    sphere = constants.tether_length_m*(1.0+args.strain)+constants.bridle_radius_m
    centre = cl.NominalLoop.sphere_centre_distance(sphere, entry["path_radius_target"]) if args.centre == "sphere" else None
    loop = cl.NominalLoop(entry, [args.wind, 0.0, 0.0], constants, centre_distance_m=centre, interpolation=args.interpolation,
                          roll_limit_tension_N=None if args.roll_limit_tension < 0 else args.roll_limit_tension)
    period = loop.period_s()
    steps_per_lap = int(round(period/BASE_DT_S)) if args.steps_per_lap is None else args.steps_per_lap
    tick = time.perf_counter()
    motion, table = loop.motion(args.laps, steps_per_lap)
    motion_seconds = time.perf_counter()-tick
    dt = table["dt_s"]
    full_steps = args.laps*steps_per_lap
    steps = full_steps if args.steps is None else min(args.steps, full_steps)
    wind = loop.wind_world()

    # ----- the aircraft ---------------------------------------------------------------------------------------------
    low_case, high_case = (LOW_CASE, HIGH_CASE) if args.assembly == "main" else (LOW_PYLON_CASE, HIGH_PYLON_CASE)
    low, high = load_aircraft(REPO/low_case), load_aircraft(REPO/high_case)
    if not (np.array_equal(high.wing.cp, low.wing.cp) and np.array_equal(high.wing.dl, low.wing.dl)):
        raise SystemExit("the two cases do not share one geometry")
    if args.assembly == "pylons":
        (body, low_polar), (_, high_polar) = A3.pylons_only(low), A3.pylons_only(high)
    else:
        body, low_polar, high_polar = low.wing, low.polar, high.polar
    handover = SectionHandover(low_polar, high_polar, seeding=args.seeding)
    free_rows = int(round(CONTRACT["far_wake_free_lap_fraction"]*steps_per_lap)) if args.free_rows is None else args.free_rows
    keep_rows = CONTRACT["far_wake_keep_laps"]*steps_per_lap if args.keep_rows is None else args.keep_rows
    options = dict(velocity_backend=args.backend)
    if args.backend == "tree":
        options["tree"] = TreeSettings(theta=0.2, order=2, leaf_size=32)
    config = WakeConfig(dt_s=dt, max_rows=max(steps, keep_rows, 1), core_radius_m=0.15, convection_integrator="euler", core_evolution="constant",
                        solve=CROSSWIND_SOLVE, free_convection_rows=free_rows if free_rows > 0 else None,
                        retained_rows=keep_rows if keep_rows > 0 else None, **options)
    backend = AeroBackend("vsm_free_wake", wake=config)

    # ----- rotors (optional) and tether -----------------------------------------------------------------------------
    rotors = rotor_settings = None
    if args.rotors is not None:
        rotors = PrescribedRotors(load_rotor_set(REPO/ROTOR_SET, RotorTable.from_makani_json(REPO/ROTOR_TABLE)), args.rotors,
                                  nodes_per_ring=args.ring_nodes, surface_core_m=args.ring_surface_core)
        rotor_settings = dict(omega_rad_s=args.rotors, discs=len(rotors.discs), nodes_per_ring=rotors.nodes_per_ring,
                              surface_core_m=rotors.surface_core_m, rotor_set=ROTOR_SET, table=ROTOR_TABLE,
                              note="no rotor speed is declared for A5; a run with rotors is a trial")
    tether = warmup = None
    with_tether = not args.no_tether and args.assembly == "main"
    if with_tether:
        line = LumpedTether(TetherProperties(), TetherSettings(air_density_kg_m3=config.density_kg_m3))
        tether = PrescribedTether(line, loop.ground_point_world(), bridle, substeps=CONTRACT["substeps"])
        if args.warmup_laps:
            tick = time.perf_counter()
            warm_motion = first_samples(motion, args.warmup_laps*steps_per_lap+1) if args.warmup_laps <= args.laps else loop.motion(args.warmup_laps, steps_per_lap)[0]
            every = next(d for d in range(max(steps_per_lap//8, 1), 0, -1) if steps_per_lap % d == 0)   # divides the lap: lap ends are recorded
            tether, report = run_tether_alone(tether, warm_motion, wind, record_every=every)
            lap_ends = [r for r in report if r["sample"] % steps_per_lap == 0]
            warmup = dict(laps=args.warmup_laps, elapsed_s=time.perf_counter()-tick, samples=report,
                          tension_at_lap_ends_N=[r["tension_kite_N"] for r in lap_ends],
                          lap_end_tension_change=[abs(b["tension_kite_N"]-a["tension_kite_N"])/abs(a["tension_kite_N"])
                                                  for a, b in zip(lap_ends[:-1], lap_ends[1:])],
                          largest_node_speed_at_end_m_s=report[-1]["largest_node_speed_m_s"],
                          starts_from="the line's state after the warm-up; the body pose at the first sample equals the one the warm-up ended at")

    # ----- settings and provenance ----------------------------------------------------------------------------------
    contract = contract_case(args, with_tether)
    q = 0.5*config.density_kg_m3*np.mean(table["airspeed_m_s"])**2
    names = list(body.names)
    settings = dict(case="A5", low_case=low_case, high_case=high_case, playbook=PLAYBOOK, rigid_body=RIGID_BODY, assembly=args.assembly,
                    surfaces=names, switch_count_surface=names[0], wind_m_s=args.wind, laps=args.laps, steps_per_lap=steps_per_lap,
                    period_s=period, dt_s=dt, steps=steps, full_steps=full_steps, full_length=bool(steps == full_steps),
                    strain=args.strain, sphere_radius_m=sphere if args.centre == "sphere" else None, centre=args.centre,
                    loop=loop.settings(), playbook_entry=dict(wind_speed=entry["wind_speed"], rows_blended=entry["rows_blended"],
                                                               path_radius_target=entry["path_radius_target"], elevation=entry["elevation"],
                                                               azi_offset=entry["azi_offset"],
                                                               raw_row=dict(radius=raw_entries[entry["rows_blended"][0]]["path_radius_target"],
                                                                            elevation=raw_entries[entry["rows_blended"][0]]["elevation"],
                                                                            azi_offset=raw_entries[entry["rows_blended"][0]]["azi_offset"])),
                    commands=dict(airspeed_m_s=dict(smallest=min(table["airspeed_m_s"]), largest=max(table["airspeed_m_s"])),
                                  kite_speed_m_s=dict(smallest=min(table["kite_speed_m_s"]), largest=max(table["kite_speed_m_s"])),
                                  alpha_cmd_deg=dict(smallest=float(np.degrees(min(table["alpha_cmd_rad"]))), largest=float(np.degrees(max(table["alpha_cmd_rad"])))),
                                  beta_cmd_deg=dict(smallest=float(np.degrees(min(table["beta_cmd_rad"]))), largest=float(np.degrees(max(table["beta_cmd_rad"])))),
                                  tether_roll_cmd_deg=dict(smallest=float(np.degrees(min(table["tether_roll_cmd_rad"]))),
                                                           largest=float(np.degrees(max(table["tether_roll_cmd_rad"])))),
                                  samples_with_roll_saturated=table["samples_with_roll_saturated"], samples_with_alpha_fading=table["samples_with_alpha_fading"],
                                  samples_with_airspeed_saturated=table["samples_with_airspeed_saturated"],
                                  tension_for_unsaturated_roll_N=table["tension_for_unsaturated_roll_N"],
                                  largest_alpha_cmd_rate_deg_s=float(np.degrees(np.abs(np.diff(table["alpha_cmd_rad"])).max()/dt)),
                                  alpha_cmd_rate_limit_deg_s=4.0, largest_beta_cmd_rate_deg_s=float(np.degrees(np.abs(np.diff(table["beta_cmd_rad"])).max()/dt)),
                                  beta_cmd_rate_limit_deg_s=1.0, derivatives=table["derivatives"]),
                    seeding=args.seeding, backend=args.backend, tree=dict(theta=0.2, order=2, leaf_size=32) if args.backend == "tree" else None,
                    core_radius_m=0.15, density_kg_m3=config.density_kg_m3, dynamic_pressure_mean_airspeed_Pa=float(q),
                    reference_area_m2=AREA_M2, reference_chord_m=CHORD_M,
                    far_wake=dict(free_convection_rows=config.free_convection_rows, retained_rows=config.retained_rows,
                                  free_laps=None if config.free_convection_rows is None else config.free_convection_rows/steps_per_lap,
                                  kept_laps=None if config.retained_rows is None else config.retained_rows/steps_per_lap),
                    tether=None if tether is None else dict(tether.settings(dt), warmup=warmup, properties=dict(TetherProperties().__dict__),
                                                            node_count=TetherSettings().num_nodes),
                    rotors=rotor_settings, contract_settings_a5=contract,
                    scope=(("wing and tail aerodynamic loads" if args.assembly == "main" else "the four pylons alone, as an assembly of their own")
                           + (" with the tether at the bridle point" if tether is not None else "; no tether")
                           + (" and the eight rotor discs" if rotors is not None else "; no rotors")+"; no fuselage"),
                    not_modelled=(["pylons" if args.assembly == "main" else "wing and tails"]+(["tether"] if tether is None else [])
                                  + (["rotors"] if rotors is None else [])+["fuselage aerodynamics", "wind shear", "ground contact of the line"]),
                    k7_rule=TRUNCATION, motion_build_s=motion_seconds, handover=handover.metadata(), aerodynamics=backend.metadata(),
                    playbook_sha256=hashlib.sha256((REPO/PLAYBOOK).read_bytes()).hexdigest(),
                    rigid_body_sha256=hashlib.sha256((REPO/RIGID_BODY).read_bytes()).hexdigest(), **A3.provenance(low_case, high_case))
    settings["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings["script"] = "paper1/crosswind/c1_loop.py (copy of OF-0007-g4motion20261002/a5_loop.py with --save-laps)"
    settings["saved_lap_states"] = [lap*steps_per_lap for lap in range(1, args.laps+1) if lap*steps_per_lap <= steps] if args.save_laps else []
    started, tick = time.time(), time.perf_counter()

    def progress(record):
        if record["step"] % 25 == 0 or record["step"] == steps:
            tension = f"T={record['tether_tension_kite_N']:.0f}N " if "tether_tension_kite_N" in record else ""
            print(f"step {record['step']}/{steps} lap_angle={np.degrees(table['loop_angle_rad'][record['step']]):.0f} alpha={record['alpha_deg']:.2f} "
                  f"w={record['handover_weight']:.2f} seed={record['seeding']} switches={record['stall_switches']} it={record['iterations']} "
                  f"|F|={np.linalg.norm(record['force_body_N']):.0f}N {tension}elapsed {time.perf_counter()-tick:.0f}s", flush=True)

    keep = [lap*steps_per_lap for lap in range(1, args.laps+1) if lap*steps_per_lap <= steps] if args.save_laps else []
    run = run_prescribed(backend, body, None, motion, wind, handover=handover, rotors=rotors, tether=tether, steps=steps,
                         progress=progress, attitude_scheme="fourth_order_central", keep_states=keep)
    for step, state in run.states.items():
        nodes, edges, gamma = wake_network(state)
        np.savez_compressed(out/f"state_{step:05d}.npz", nodes=nodes, edges=edges, gamma=gamma, first=state.wing.node_count)
    k3 = consistency_verdict(run.motion_consistency)
    pairs = periodicity(run.records, steps_per_lap)
    aero_pairs = periodicity(run.records, steps_per_lap, loads=lambda r: (np.asarray(r["force_body_N"]), np.asarray(r["moment_origin_body_Nm"])))
    k7 = periodicity_verdict(pairs, run.status, contract)
    weight = np.array([r["handover_weight"] for r in run.records]) if run.records else np.zeros(0)
    seeding = [r["seeding"] for r in run.records]
    retries = [dict(step=r["step"], alpha_deg=r["alpha_deg"], **(r["zero_seed_failure"] or {})) for r in run.records if r["seeding"] == "retry_previous"]
    counts = dict(steps_recorded=len(run.records), seeding_counts={name: seeding.count(name) for name in sorted(set(seeding))},
                  retries=len(retries), retry_steps=retries,
                  steps_weight_positive=int(np.sum(weight > 0.)), share_weight_positive=float(np.mean(weight > 0.)) if len(weight) else None,
                  steps_with_stall_switches_over_limit=int(sum((r["stall_switches"] or 0) > SWITCH_LIMIT for r in run.records)),
                  stall_switch_limit=SWITCH_LIMIT,
                  steps_with_incidence_steps_over_6deg=int(sum(r["incidence_steps"]["total_steps_over_limit"] > 0 for r in run.records)),
                  largest_iterations=int(max(r["iterations"] for r in run.records)) if run.records else None,
                  alpha_deg=dict(smallest=float(min(r["alpha_deg"] for r in run.records)), largest=float(max(r["alpha_deg"] for r in run.records))) if run.records else None,
                  beta_deg=dict(smallest=float(min(r["beta_deg"] for r in run.records)), largest=float(max(r["beta_deg"] for r in run.records))) if run.records else None)
    residual_rows, residual = [], None
    if run.records and args.assembly == "main":
        rigid = load_rigid_body(REPO/RIGID_BODY)
        residual_rows, summary = account(run.records, motion, rigid, not_included=tuple(settings["not_modelled"]))
        residual = residual_summary(residual_rows, summary, steps_per_lap)
    report = dict(settings=settings, status=run.status, error=run.error, error_type=run.error_type, failed_step=run.failed_step,
                  failed_module=run.failed_module, failed_step_aerodynamics=run.failed_step_aerodynamics,
                  failed_iterations=run.failed_iterations, zero_seed_failure_at_failed_step=run.zero_seed_failure,
                  frames=len(run.records), elapsed_s=time.perf_counter()-tick, started_unix=started,
                  K3=k3, K7=k7, K7_pairs_total_loads=pairs, K7_pairs_aero_only=aero_pairs, bookkeeping=counts, per_lap=per_lap(run.records, steps_per_lap, dt),
                  residual=residual, residual_rows=residual_rows, rotor_bookkeeping=None if rotors is None else A3.rotor_bookkeeping(run.records, dt),
                  commands=dict(time_s=table["time_s"], loop_angle_rad=table["loop_angle_rad"], airspeed_m_s=table["airspeed_m_s"],
                                kite_speed_m_s=table["kite_speed_m_s"], alpha_cmd_rad=table["alpha_cmd_rad"], beta_cmd_rad=table["beta_cmd_rad"],
                                tether_roll_cmd_rad=table["tether_roll_cmd_rad"], angular_acceleration_rad_s2=table["angular_acceleration_rad_s2"]),
                  motion=dict(position_m=motion.position_m.tolist(), rotation=motion.rotation.tolist(), velocity_m_s=motion.velocity_m_s.tolist(),
                              angular_velocity_rad_s=motion.angular_velocity_rad_s.tolist(), acceleration_m_s2=motion.acceleration_m_s2.tolist()),
                  records=run.records, step_times_s=run.step_seconds)
    (out/"history.json").write_text(json.dumps(report, indent=1, default=str)+"\n", encoding="utf-8")
    tension = [row.get("tether_tension_kite_N") for row in report["per_lap"]]
    print(json.dumps(dict(out=str(out), status=run.status, error=run.error, failed_step=run.failed_step, failed_module=run.failed_module,
                          frames=len(run.records), elapsed_s=round(report["elapsed_s"], 1), K3=k3["worst_relative"], K3_passed=k3["passed"],
                          K7=k7["verdict"], pairs=[(p["lap"], round(p["force_max_relative"], 5), round(p["moment_max_relative"], 5)) for p in pairs],
                          retries=counts["retries"], switches_over_limit=counts["steps_with_stall_switches_over_limit"],
                          weight_positive=counts["steps_weight_positive"], tension_per_lap=tension,
                          residual_over_modelled=None if residual is None else round(residual["required_force_N_over_modelled"], 4))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
