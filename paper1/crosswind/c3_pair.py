"""Case C3 of the OF-0009 contract: two M600 units on the nominal crosswind loop in one wake, with their tethers.

Both units fly the 9 m/s nominal loop of A5 (``openfawe.flight.crosswind_loop``, settings of ``c1_loop.py``: smoothed
playbook entry, sphere at the declared strain 0.0054375, spline reading, 936 steps per lap, zero seeding through the
production closure with the crosswind section data only, tree backend, constant core 0.15 m, explicit Euler). Unit A's
ground point is the origin; unit B's ground point is shifted downwind by ``--spacing`` loop diameters (2 R of the
smoothed entry, 270 m at 9 m/s) and sideways by ``--offset`` diameters (+Y), and B flies ``--phase`` degrees of loop
angle ahead of A (its samples are A's rolled forward). Each unit has its own lumped-mass tether (Makani M600 line, A5
settings), run alone for ``--warmup-laps`` laps before the coupled run. The two units are solved in one circulation
system and one free wake (``openfawe.flight.multi_prescribed``).

Far wake (contract section 4, C3): frozen beyond ``--free-rows`` (default one lap, 936 rows: older rows drift with the
ambient wind, nothing is deleted); ``--keep-rows 0`` keeps everything (the default). ``--free-rows 0`` runs the complete
free wake. The run length ``--laps`` should cover the arrival of A's wake at B (spacing / wind speed) plus three laps.

Rotors (``--rotors OMEGA``, contract section 4 addition of 2026-10-19): every unit carries the eight rotor discs of
``cases/rotors/m600_rotors_makani.json`` with the Rev4 table at the constant angular speed OMEGA [rad/s] (the C1 rotor
runs use 97.4 rad/s, the maximum-shaft-power speed of the table at the loop-mean commanded airspeed of 46.5 m/s), each
releasing one vortex ring per step into the one ring set of the wake (``openfawe.flight.multi_prescribed``; rings act
on the lifting surfaces with the surface core ``--ring-surface-core``, 0.3 m as in A4). The rings follow the same far-wake
rule as the rows: ``--free-ring-steps`` (default: the value of ``--free-rows``, one lap; 0: every ring convects freely)
is the number of newest steps whose rings convect with the induced velocity, older rings drift with the ambient wind and
nothing is deleted. The rotor force enters the total loads of the K7 periodicity and the per-lap table gives the shaft
power and thrust of each unit.

Reported: K3 of both motions, K7 of each unit (lap n against lap n-1 at equal phase, total loads; limit 1 %, last
pair), per-lap means of both units (loads, power, tension), the loads of unit B relative to unit A lap by lap, and a
compact ``checks.json`` (settings, verdicts, per-lap table and the per-step loads of both units for every step of the
last three laps and every fourth step before; about 2-3 MB for the pairs, which node 2 commits whatever its size, its
2 MB limit applying to ``history.json`` only). The wake summary is computed after both receipts are written and added to
``checks.json`` afterwards (an error of it is recorded there instead).

Usage (direct):      c3_pair.py OUT_DIR [--spacing 1.0] [--offset 0.0] [--phase 0] [--laps 5] [--wind 9] [--steps-per-lap N]
                                        [--steps N] [--backend tree|numba|numpy] [--strain 0.0054375] [--no-tether]
                                        [--warmup-laps 2] [--free-rows 936] [--keep-rows 0] [--save-laps] [--single]
                                        [--rotors OMEGA [--ring-nodes 16] [--ring-surface-core 0.3] [--free-ring-steps N]
                                                        [--ring-core R] [--keep-ring-steps K]]
Usage (PC job line): NAME --script paper1/crosswind/c3_pair.py --spacing 1.0 --laps 5
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
from openfawe.aero.vortex import CudaSettings, TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig, extended_network  # noqa: E402
from openfawe.flight import crosswind_loop as cl  # noqa: E402
from openfawe.flight.kinematics import RigidMotion  # noqa: E402
from openfawe.flight.multi_prescribed import PrescribedBody, run_prescribed_multi  # noqa: E402
from openfawe.flight.prescribed import PrescribedRotors, PrescribedTether, run_tether_alone  # noqa: E402
from openfawe.rotor.actuator_disc import load_rotor_set  # noqa: E402
from openfawe.rotor.table import RotorTable  # noqa: E402
from openfawe.tether.bridle import load_bridle  # noqa: E402
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings  # noqa: E402

CASE = "cases/qualification/m600_makani_reconstructed.json"
PLAYBOOK = "cases/flight/m600_crosswind_playbook_land.json"
RIGID_BODY = "cases/aircraft/m600_makani_rigid_body.json"
ROTOR_SET = "cases/rotors/m600_rotors_makani.json"
ROTOR_TABLE = "cases/rotors/makani/rotor_rev4.json"
AREA_M2, CHORD_M = 32.9285, 1.2831
BASE_DT_S = 0.02
K3_LIMIT, K7_LIMIT = 1e-6, 0.01
CONTRACT = dict(wind_m_s=9.0, strain=0.0054375, warmup_laps=2, substeps=10, free_laps=1, keep_laps=0, separation_m=60.0)


def consistency_verdict(report):
    entries = {k: v for k, v in report.items() if isinstance(v, dict)}
    relative = {k: v["relative"] for k, v in entries.items() if v["relative"] is not None and v["compared"] > 0}
    worst = max(relative.values()) if relative else None
    return dict(limit=K3_LIMIT, worst_relative=worst, passed=None if worst is None else bool(worst <= K3_LIMIT), report=report)


def total_loads(entry):
    force = np.asarray(entry["force_body_N"], dtype=float)
    moment = np.asarray(entry["moment_origin_body_Nm"], dtype=float)
    if "tether_force_body_N" in entry:
        force = force+np.asarray(entry["tether_force_body_N"], dtype=float)
        moment = moment+np.asarray(entry["tether_moment_origin_body_Nm"], dtype=float)
    if "rotor_force_body_N" in entry:
        force = force+np.asarray(entry["rotor_force_body_N"], dtype=float)
        moment = moment+np.asarray(entry["rotor_moment_origin_body_Nm"], dtype=float)
    return force, moment


def periodicity(records, name, steps_per_lap, loads=total_loads):
    pairs, lap = [], 2
    while lap*steps_per_lap < len(records):
        now = records[(lap-1)*steps_per_lap+1:lap*steps_per_lap+1]
        before = records[(lap-2)*steps_per_lap+1:(lap-1)*steps_per_lap+1]
        entry = dict(lap=lap, against=lap-1)
        for index, kind in enumerate(("force", "moment")):
            a = np.array([loads(r["bodies"][name])[index] for r in before])
            b = np.array([loads(r["bodies"][name])[index] for r in now])
            difference = np.linalg.norm(b-a, axis=1)/np.linalg.norm(a, axis=1)
            entry[f"{kind}_max_relative"], entry[f"{kind}_mean_relative"] = float(difference.max()), float(difference.mean())
        pairs.append(entry)
        lap += 1
    return pairs


def verdict(pairs, status):
    if status != "complete":
        return dict(verdict="not evaluated", reason="the run failed")
    if not pairs:
        return dict(verdict="not evaluated", reason="fewer than two complete laps")
    last = pairs[-1]
    within = max(last["force_max_relative"], last["moment_max_relative"]) <= K7_LIMIT
    return dict(verdict="passed" if within else "failed", reason=f"lap {last['lap']} against lap {last['against']}", limit=K7_LIMIT)


def per_lap(records, name, steps_per_lap):
    rows, lap = [], 1
    while lap*steps_per_lap < len(records):
        part = [r["bodies"][name] for r in records[(lap-1)*steps_per_lap+1:lap*steps_per_lap+1]]
        q = np.array([r["dynamic_pressure_Pa"] for r in part])[:, None]
        force = np.array([r["force_body_N"] for r in part])
        moment = np.array([r["moment_origin_body_Nm"] for r in part])
        row = dict(lap=lap, force_body_mean_N=force.mean(axis=0).tolist(), force_body_coefficient_mean=(force/(q*AREA_M2)).mean(axis=0).tolist(),
                   moment_body_coefficient_mean=(moment/(q*AREA_M2*CHORD_M)).mean(axis=0).tolist(),
                   lift_body_z_mean_N=float(force[:, 2].mean()), aero_power_mean_W=float(np.mean([r["aero_power_W"] for r in part])),
                   airspeed_mean_m_s=float(np.mean([r["airspeed_m_s"] for r in part])),
                   alpha_deg_min_max=[float(min(r["alpha_deg"] for r in part)), float(max(r["alpha_deg"] for r in part))])
        if "tether_tension_kite_N" in part[0]:
            tension = np.array([r["tether_tension_kite_N"] for r in part])
            row["tether_tension_kite_N"] = dict(mean=float(tension.mean()), min=float(tension.min()), max=float(tension.max()))
        if "rotor_shaft_power_W" in part[0]:
            shaft = np.array([r["rotor_shaft_power_W"] for r in part])
            thrust = np.array([sum(r["rotor_thrust_N"]) for r in part])
            row["rotor_shaft_power_W"] = dict(mean=float(shaft.mean()), min=float(shaft.min()), max=float(shaft.max()))
            row["rotor_thrust_total_N"] = dict(mean=float(thrust.mean()), min=float(thrust.min()), max=float(thrust.max()))
            row["rotor_force_body_mean_N"] = np.array([r["rotor_force_body_N"] for r in part]).mean(axis=0).tolist()
            row["rotor_power_on_body_mean_W"] = float(np.mean([r["rotor_power_on_body_W"] for r in part]))
            near = [r["ring_nodes_within_surface_core"] for r in part]
            row["ring_nodes_within_surface_core"] = dict(mean=float(np.mean(near)), max=int(max(near)), steps_nonzero=int(np.count_nonzero(near)))
        rows.append(row)
        lap += 1
    return rows


def relative_to_a(laps_a, laps_b):
    out = []
    for a, b in zip(laps_a, laps_b):
        row = dict(lap=a["lap"], lift_ratio_b_over_a=b["lift_body_z_mean_N"]/a["lift_body_z_mean_N"],
                   power_ratio_b_over_a=b["aero_power_mean_W"]/a["aero_power_mean_W"],
                   force_body_ratio_b_over_a=(np.array(b["force_body_mean_N"])/np.array(a["force_body_mean_N"])).tolist())
        if "tether_tension_kite_N" in a and "tether_tension_kite_N" in b:
            row["tension_ratio_b_over_a"] = b["tether_tension_kite_N"]["mean"]/a["tether_tension_kite_N"]["mean"]
        if "rotor_shaft_power_W" in a and "rotor_shaft_power_W" in b:
            row["shaft_power_ratio_b_over_a"] = b["rotor_shaft_power_W"]["mean"]/a["rotor_shaft_power_W"]["mean"]
            row["rotor_thrust_ratio_b_over_a"] = b["rotor_thrust_total_N"]["mean"]/a["rotor_thrust_total_N"]["mean"]
        out.append(row)
    return out


def compact_records(records, names, steps_per_lap, laps):
    """Per-step loads of every body: every step of the last three laps, every fourth step before (6 significant digits)."""
    first_dense = max(0, (laps-3)*steps_per_lap)
    out = {name: dict(step=[], time_s=[], force_body_N=[], moment_origin_body_Nm=[], tension_N=[], airspeed_m_s=[], alpha_deg=[],
                      aero_power_W=[], iterations=[], rotor_shaft_power_W=[], rotor_thrust_total_N=[], rotor_force_body_N=[],
                      ring_nodes_within_surface_core=[]) for name in names}
    for r in records:
        if r["step"] < first_dense and r["step"] % 4:
            continue
        for name in names:
            e, o = r["bodies"][name], out[name]
            o["step"].append(r["step"]); o["time_s"].append(round(r["time_s"], 6)); o["iterations"].append(r["iterations"])
            o["force_body_N"].append([float(f"{v:.6g}") for v in e["force_body_N"]])
            o["moment_origin_body_Nm"].append([float(f"{v:.6g}") for v in e["moment_origin_body_Nm"]])
            o["tension_N"].append(None if "tether_tension_kite_N" not in e else float(f"{e['tether_tension_kite_N']:.6g}"))
            o["airspeed_m_s"].append(float(f"{e['airspeed_m_s']:.6g}")); o["alpha_deg"].append(float(f"{e['alpha_deg']:.6g}"))
            o["aero_power_W"].append(float(f"{e['aero_power_W']:.6g}"))
            if "rotor_shaft_power_W" in e:
                o["rotor_shaft_power_W"].append(float(f"{e['rotor_shaft_power_W']:.6g}")); o["rotor_thrust_total_N"].append(float(f"{sum(e['rotor_thrust_N']):.6g}"))
                o["rotor_force_body_N"].append([float(f"{v:.6g}") for v in e["rotor_force_body_N"]]); o["ring_nodes_within_surface_core"].append(int(e["ring_nodes_within_surface_core"]))
                if "ring_nodes_within_surface_core_all_bodies" in e:
                    o.setdefault("ring_nodes_within_surface_core_all_bodies", []).append(int(e["ring_nodes_within_surface_core_all_bodies"]))
    for o in out.values():
        if not o["rotor_shaft_power_W"]:
            for key in ("rotor_shaft_power_W", "rotor_thrust_total_N", "rotor_force_body_N", "ring_nodes_within_surface_core"):
                del o[key]
    return out


def provenance():
    def git(*args):
        try:
            done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)
        except Exception:  # noqa: BLE001
            return None
        return done.stdout.strip() if done.returncode == 0 else None
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain", "--", "src")
    inputs = git("status", "--porcelain", "--", CASE, PLAYBOOK, RIGID_BODY, ROTOR_SET, ROTOR_TABLE,
                 "paper1/crosswind/c3_pair.py", "paper1/crosswind/wake_analysis.py")
    try:
        import numba
        threads, numba_version = int(numba.get_num_threads()), numba.__version__
    except Exception:  # noqa: BLE001
        threads, numba_version = None, None
    return dict(git_commit=commit, src_clean=None if commit is None or status is None else status == "",
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                case_sha256=hashlib.sha256((REPO/CASE).read_bytes()).hexdigest(),
                playbook_sha256=hashlib.sha256((REPO/PLAYBOOK).read_bytes()).hexdigest(),
                rigid_body_sha256=hashlib.sha256((REPO/RIGID_BODY).read_bytes()).hexdigest(),
                rotor_set_sha256=hashlib.sha256((REPO/ROTOR_SET).read_bytes()).hexdigest(),
                rotor_table_sha256=hashlib.sha256((REPO/ROTOR_TABLE).read_bytes()).hexdigest(),
                wake_analysis_sha256=hashlib.sha256((TASK/"wake_analysis.py").read_bytes()).hexdigest(),
                inputs_clean=None if commit is None or inputs is None else inputs == "",
                threads=threads, versions=dict(python=platform.python_version(), numpy=np.__version__, numba=numba_version),
                platform=platform.platform())


def shifted(motion, offset, roll, n):
    """``n`` samples of the motion, positions shifted by ``offset``, samples rolled forward by ``roll`` steps (periodic laps)."""
    sl = slice(roll, roll+n)
    return RigidMotion(motion.time_s[:n], motion.position_m[sl]+offset, motion.rotation[sl], motion.velocity_m_s[sl],
                       motion.angular_velocity_rad_s[sl], motion.acceleration_m_s2[sl], None,
                       description=f"{motion.description}; shifted {np.asarray(offset).tolist()}, rolled {roll} steps")


def wake_summary(state, bodies, motion_a, steps_per_lap, dt, wind_m_s, diameter, core_radius, stations, count, ring_core=None):
    """Compact description of the final wake for the receipt (node 2 commits no wake states).

    centre: per body, the centre line of its own trailing vorticity (``wake_analysis.centre_line`` on the node columns of
    that body, every fourth row); planes: axial deficit 1 - u/U and in-plane velocity on cross-flow planes at ``stations``
    loop diameters downstream of unit A's path centre (mean position over the last lap), half-width one diameter, from the
    complete network including the rotor rings (Lamb-Oseen, the run's core, CPU tree code), with the azimuthal profile
    about the deficit centre of each plane. Plane values rounded to 4 significant digits.
    """
    sys.path.insert(0, str(TASK))
    import wake_analysis as wa
    nodes, edges, gamma, ring_first = extended_network(state)
    first = state.wing.node_count
    wing_edges = np.all(edges < ring_first, axis=1)
    # the run's cores: the wake core on the wing filaments, the rings' own core (--ring-core) on the ring segments
    radii = np.where(wing_edges, float(core_radius), float(core_radius if ring_core is None else ring_core))
    rows, circulation = wa.wake_rows(nodes[:ring_first], edges[wing_edges], gamma[wing_edges], first)
    offset, centre = 0, {}
    for body in bodies:
        count_nodes = body.wing.node_count
        part = slice(offset, offset+count_nodes)
        line = wa.centre_line(rows[:, part], circulation[:, part], dt, wind_m_s)
        centre[body.name] = [{k: (np.round(v, 3).tolist() if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v))
                              for k, v in row.items()} for row in line[::4]]
        offset += count_nodes
    path_centre = motion_a.position_m[-steps_per_lap:].mean(axis=0)
    planes = []
    for station in stations:
        p = wa.plane_at(path_centre[0]+station*diameter, path_centre[1:], diameter, count, nodes, edges, gamma, wind_m_s, radii, "tree")
        r4 = lambda a: np.vectorize(lambda x: float(f"{x:.4g}"))(np.asarray(a, dtype=float)).tolist()   # noqa: E731
        planes.append(dict(station_diameters=station, x_m=p["x_m"], u=r4(p["u"]), v=r4(p["v"]), deficit=r4(p["deficit"]), in_plane=r4(p["in_plane"]),
                           centre_yz_m=list(p["centre"]), path_centre_yz_m=list(p["path_centre_yz"]), radius_bins_m=p["radius_bins_m"],
                           profile=p["profile"], deficit_max=p["deficit_max"], deficit_mean_positive=p["deficit_mean_positive"],
                           momentum_deficit_m2_s=p["momentum_deficit_m2_s"]))
    return dict(path_centre_m=path_centre.tolist(), diameter_m=diameter, rows=int(rows.shape[0]), ring_nodes=int(len(nodes)-ring_first),
                centre=centre, planes=planes)


def main(out, argv):
    parser = argparse.ArgumentParser(prog="c3_pair.py")
    parser.add_argument("--spacing", type=float, default=1.0, help="downwind spacing of the ground points in loop diameters")
    parser.add_argument("--offset", type=float, default=0.0, help="sideways offset of unit B in loop diameters (+Y)")
    parser.add_argument("--phase", type=float, default=0.0, help="loop angle of unit B ahead of unit A [deg]")
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--wind", type=float, default=CONTRACT["wind_m_s"])
    parser.add_argument("--steps-per-lap", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--backend", default="tree", choices=("tree", "numba", "numpy", "cuda_tree"),
                        help="velocity sums of the wake convection: CPU tree code (default) or the CPU-built tree evaluated on the GPU "
                             "(cuda_tree, mixed precision, suspect pairs in float64 on the GPU: whole-step GPU contract revision 3)")
    parser.add_argument("--strain", type=float, default=CONTRACT["strain"])
    parser.add_argument("--no-tether", action="store_true")
    parser.add_argument("--warmup-laps", type=int, default=CONTRACT["warmup_laps"])
    parser.add_argument("--free-rows", type=int, default=None, help="far wake: newest rows convected freely (default one lap; 0: all)")
    parser.add_argument("--keep-rows", type=int, default=0, help="far wake: newest rows kept (default 0: keep everything)")
    parser.add_argument("--save-laps", action="store_true", help="store the wake network at the end of every lap")
    parser.add_argument("--frames", type=int, default=0, help="store this many wake networks equally spaced in time over the last lap, written as they are reached (figures and animations; author 2026-10-22 22:47)")
    parser.add_argument("--save-final", action="store_true", help="store the wake network of the last step as well (partial-lap runs for figures)")
    parser.add_argument("--single", action="store_true", help="unit A alone (the single-unit reference with the same code path)")
    parser.add_argument("--rotors", type=float, default=None, metavar="OMEGA", help="the eight rotors of every unit at this constant angular speed [rad/s]")
    parser.add_argument("--ring-nodes", type=int, default=16)
    parser.add_argument("--ring-surface-core", type=float, default=0.3)
    parser.add_argument("--ring-core", type=float, default=None, help="core of the ring filaments away from the lifting surfaces [m] (PrescribedRotors.ring_core_m; default: the wake core)")
    parser.add_argument("--ring-bound-core", type=float, default=None, help="core of the bound system of the lifting surfaces where it induces at the ring nodes [m] (PrescribedRotors.ring_bound_core_m, model B of contract section 8, 2026-10-22; default: unchanged)")
    parser.add_argument("--ring-surface-overlap", type=float, default=None, help="core overlap of the ring sheet where the rings induce at lifting-surface points (PrescribedRotors.ring_surface_overlap, S of contract section 8, 2026-10-22 18:22 BST; default: stored rings)")
    parser.add_argument("--keep-ring-steps", type=int, default=None, help="rings of the newest K steps are kept, older rings are removed (PrescribedRotors.retained_ring_steps; default: all kept)")
    parser.add_argument("--ring-drift", action="store_true", help="every ring drifts with the wind from its release, no free step (PrescribedRotors.free_convection_ring_steps = 0, model A' of contract section 8, 2026-10-22 18:55 BST)")
    parser.add_argument("--ring-sheet-drift", action="store_true", help="the rings that do not convect freely drift with the wind plus the sheet's own speed gamma/2 (PrescribedRotors.ring_sheet_drift, model A'' of contract section 8, 2026-10-22 19:16 BST)")
    parser.add_argument("--ring-ambient-spacing", action="store_true", help="with --ring-sheet-drift: release Gamma over the drift spacing (V_ambient + gamma/2) dt so that the tube carries gamma (PrescribedRotors.ring_ambient_spacing, review RA-01 of 2026-10-22)")
    parser.add_argument("--free-ring-steps", type=int, default=None, help="far wake of the rings: newest steps convected freely (default: --free-rows; 0: all)")
    parser.add_argument("--planes", default=None, help="stations of the cross-flow planes of the final wake, loop diameters downstream of A's "
                        "path centre ('' for none; default 0.5,1,1.5,2,2.5,3 alone and 0.5, S+0.5, S+1.5, S+2.5 for a pair at spacing S, off B's loop)")
    parser.add_argument("--plane-count", type=int, default=49)
    args = parser.parse_args(argv)
    if args.frames < 0:
        raise SystemExit("--frames must not be negative")
    if args.laps < 1 or args.warmup_laps < 0:
        raise SystemExit("--laps must be positive and --warmup-laps non-negative")
    if args.keep_ring_steps is not None and args.keep_ring_steps < 1:
        raise SystemExit("--keep-ring-steps must be a positive number of steps")
    if args.ring_core is not None and not args.ring_core > 0:
        raise SystemExit("--ring-core must be a positive radius")
    if args.ring_bound_core is not None and not args.ring_bound_core > 0:
        raise SystemExit("--ring-bound-core must be a positive radius")
    if args.ring_surface_overlap is not None and not args.ring_surface_overlap > 0:
        raise SystemExit("--ring-surface-overlap must be positive")
    # Review RB-03 of c383b09a: a ring option without --rotors would be ignored and not recorded; refuse it.
    ring_options = ("ring_nodes", "ring_surface_core", "ring_core", "ring_bound_core", "ring_surface_overlap", "keep_ring_steps",
                    "free_ring_steps", "ring_drift", "ring_sheet_drift", "ring_ambient_spacing")
    given = ["--"+name.replace("_", "-") for name in ring_options if getattr(args, name) != parser.get_default(name)]
    if args.rotors is None and given:
        parser.error(f"{', '.join(given)} need --rotors")
    if args.ring_drift and args.free_ring_steps is not None:
        parser.error("--ring-drift and --free-ring-steps exclude each other")
    # Review RA-02/RA-03 of 2026-10-22: the sheet drift is defined from the release (needs --ring-drift), and an explicit
    # --free-ring-steps 0 is refused (here it meant "every ring free", in the library 0 means "no ring free").
    if args.ring_sheet_drift and not args.ring_drift:
        parser.error("--ring-sheet-drift needs --ring-drift (the rings drift from their release)")
    if args.ring_ambient_spacing and not args.ring_sheet_drift:
        parser.error("--ring-ambient-spacing needs --ring-sheet-drift")
    if args.free_ring_steps == 0:
        parser.error("--free-ring-steps 0 is ambiguous: omit it with --free-rows 0 for every ring free, or use --ring-drift for none")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    # ----- the loop of unit A and the motions ---------------------------------------------------------------------
    entries, raw_entries, playbook = cl.load_playbook(REPO/PLAYBOOK)
    constants = cl.LoopConstants.from_case(playbook)
    entry = cl.playbook_entry(entries, args.wind)
    bridle = load_bridle(REPO/RIGID_BODY)
    sphere = constants.tether_length_m*(1.0+args.strain)+constants.bridle_radius_m
    centre = cl.NominalLoop.sphere_centre_distance(sphere, entry["path_radius_target"])
    loop = cl.NominalLoop(entry, [args.wind, 0.0, 0.0], constants, centre_distance_m=centre, interpolation="spline", roll_limit_tension_N=20000.0)
    period = loop.period_s()
    steps_per_lap = int(round(period/BASE_DT_S)) if args.steps_per_lap is None else args.steps_per_lap
    roll = int(round(args.phase/360.0*steps_per_lap)) % steps_per_lap
    tick = time.perf_counter()
    long_motion, table = loop.motion(args.laps+1, steps_per_lap)
    dt = table["dt_s"]
    n = args.laps*steps_per_lap+1
    motion_a = shifted(long_motion, np.zeros(3), 0, n)
    diameter = 2.0*float(entry["path_radius_target"])
    offset_b = np.array([args.spacing*diameter, args.offset*diameter, 0.0])
    motion_b = shifted(long_motion, offset_b, roll, n)
    motion_seconds = time.perf_counter()-tick
    full_steps = args.laps*steps_per_lap
    steps = full_steps if args.steps is None else min(args.steps, full_steps)
    wind = loop.wind_world()

    # ----- the aircraft and the backend -------------------------------------------------------------------------------
    definition = load_aircraft(REPO/CASE)
    free_rows = CONTRACT["free_laps"]*steps_per_lap if args.free_rows is None else args.free_rows
    keep_rows = args.keep_rows
    options = dict(velocity_backend=args.backend)
    if args.backend in ("tree", "cuda_tree"):
        options["tree"] = TreeSettings(theta=0.2, order=2, leaf_size=32)
    if args.backend == "cuda_tree":
        options["cuda"] = CudaSettings(precision="mixed", block=128, suspect="gpu64")
    config = WakeConfig(dt_s=dt, max_rows=max(steps, keep_rows, 1), core_radius_m=0.15, convection_integrator="euler", core_evolution="constant",
                        solve=CROSSWIND_SOLVE, free_convection_rows=free_rows if free_rows > 0 else None,
                        retained_rows=keep_rows if keep_rows > 0 else None, **options)
    backend = AeroBackend("vsm_free_wake", wake=config)

    # ----- the rotors (optional) ----------------------------------------------------------------------------------------
    rotors = rotor_settings = None
    if args.rotors is not None:
        free_rings = free_rows if args.free_ring_steps is None else args.free_ring_steps
        if not args.ring_drift and args.keep_ring_steps is not None and free_rings > args.keep_ring_steps:
            print(f"note: free ring steps {free_rings} reduced to the kept ring steps {args.keep_ring_steps} (every kept ring is free)", flush=True)
            free_rings = args.keep_ring_steps                           # the free part cannot exceed what is kept
        rotors = PrescribedRotors(load_rotor_set(REPO/ROTOR_SET, RotorTable.from_makani_json(REPO/ROTOR_TABLE)), args.rotors,
                                  nodes_per_ring=args.ring_nodes, surface_core_m=args.ring_surface_core,
                                  free_convection_ring_steps=0 if args.ring_drift else (free_rings if free_rings > 0 else None),
                                  retained_ring_steps=args.keep_ring_steps, ring_core_m=args.ring_core,
                                  ring_bound_core_m=args.ring_bound_core, ring_surface_overlap=args.ring_surface_overlap,
                                  ring_sheet_drift=args.ring_sheet_drift, ring_ambient_spacing=args.ring_ambient_spacing)
        rotor_settings = dict(omega_rad_s=args.rotors, discs=len(rotors.discs), nodes_per_ring=rotors.nodes_per_ring, surface_core_m=rotors.surface_core_m,
                              free_convection_ring_steps=rotors.free_convection_ring_steps, retained_ring_steps=rotors.retained_ring_steps,
                              ring_core_m=rotors.ring_core_m, ring_bound_core_m=rotors.ring_bound_core_m,
                              ring_surface_overlap=rotors.ring_surface_overlap, ring_sheet_drift=rotors.ring_sheet_drift,
                              ring_ambient_spacing=rotors.ring_ambient_spacing, rotor_set=ROTOR_SET, table=ROTOR_TABLE,
                              rule="constant speed; 97.4 rad/s is the maximum-shaft-power speed of the table at the loop-mean commanded airspeed 46.5 m/s")

    # ----- the tethers and their warm-up -------------------------------------------------------------------------------
    units = [("A", motion_a, np.zeros(3))]+([] if args.single else [("B", motion_b, offset_b)])
    bodies, warmups = [], {}
    for name, motion, ground in units:
        tether = None
        if not args.no_tether:
            line = LumpedTether(TetherProperties(), TetherSettings(air_density_kg_m3=config.density_kg_m3))
            tether = PrescribedTether(line, ground, bridle, substeps=CONTRACT["substeps"])
            if args.warmup_laps:
                tick = time.perf_counter()
                count = args.warmup_laps*steps_per_lap+1
                warm = RigidMotion(motion.time_s[:count], motion.position_m[:count], motion.rotation[:count], motion.velocity_m_s[:count],
                                   motion.angular_velocity_rad_s[:count], motion.acceleration_m_s2[:count], None, description="warm-up")
                every = next(d for d in range(max(steps_per_lap//8, 1), 0, -1) if steps_per_lap % d == 0)
                tether, report = run_tether_alone(tether, warm, wind, record_every=every)
                ends = [r for r in report if r["sample"] % steps_per_lap == 0]
                warmups[name] = dict(laps=args.warmup_laps, elapsed_s=time.perf_counter()-tick, tension_at_lap_ends_N=[r["tension_kite_N"] for r in ends],
                                     lap_end_tension_change=[abs(b["tension_kite_N"]-a["tension_kite_N"])/abs(a["tension_kite_N"]) for a, b in zip(ends[:-1], ends[1:])])
        bodies.append(PrescribedBody(name, definition.wing, definition.polar, motion, tether, rotors=rotors))

    settings = dict(case=CASE, playbook=PLAYBOOK, wind_m_s=args.wind, spacing_diameters=args.spacing, offset_diameters=args.offset,
                    phase_deg=args.phase, phase_steps=roll, loop_diameter_m=diameter, ground_point_b_m=offset_b.tolist(), laps=args.laps,
                    steps_per_lap=steps_per_lap, dt_s=dt, steps=steps, period_s=period, strain=args.strain, sphere_radius_m=sphere,
                    single=args.single, backend=args.backend, core_radius_m=0.15, free_convection_rows=config.free_convection_rows,
                    retained_rows=config.retained_rows, free_laps=None if config.free_convection_rows is None else config.free_convection_rows/steps_per_lap,
                    density_kg_m3=config.density_kg_m3, reference_area_m2=AREA_M2, reference_chord_m=CHORD_M, separation_m=CONTRACT["separation_m"],
                    tether=None if args.no_tether else bodies[0].tether.settings(dt), warmup=warmups, loop=loop.settings(),
                    playbook_entry=dict(wind_speed=entry["wind_speed"], rows_blended=entry["rows_blended"]), motion_build_s=motion_seconds,
                    arrival_estimate_laps=args.spacing*diameter/args.wind/period,
                    rotors=rotor_settings,
                    scope="wing and tail aerodynamic loads of each unit and its tether force"+(" and the eight rotor discs of each unit" if rotors is not None else "")
                          +"; no pylons, fuselage, hand-over or wind shear"+("" if rotors is not None else "; no rotors"),
                    aerodynamics=backend.metadata(), script="paper1/crosswind/c3_pair.py", **provenance())
    started, tick = time.time(), time.perf_counter()
    names = [b.name for b in bodies]

    def progress(record):
        if record["step"] % 25 == 0 or record["step"] == steps:
            parts = []
            for name in names:
                e = record["bodies"][name]
                tension = f" T={e['tether_tension_kite_N']:.0f}N" if "tether_tension_kite_N" in e else ""
                shaft = f" P={e['rotor_shaft_power_W']/1e3:.0f}kW rings={e['ring_nodes']}" if "rotor_shaft_power_W" in e else ""
                parts.append(f"{name}: Fz={e['force_body_N'][2]:.0f}N{tension}{shaft}")
            print(f"step {record['step']}/{steps} lap_angle={np.degrees(table['loop_angle_rad'][record['step']]):.0f} it={record['iterations']} "
                  f"{' '.join(parts)} elapsed {time.perf_counter()-tick:.0f}s", flush=True)

    keep = [lap*steps_per_lap for lap in range(1, args.laps+1) if lap*steps_per_lap <= steps] if args.save_laps else []
    if args.save_final and steps not in keep:
        keep.append(steps)
    if args.frames > 0:
        # Equally spaced over the last lap of the run (its start included, its end is --save-final's step).
        first = max(0, steps-steps_per_lap)
        keep += [s for s in (first+int(round(k*(steps-first)/args.frames)) for k in range(args.frames)) if s not in keep]
    keep = sorted(set(keep))

    def save_state(step, state):
        nodes, edges, gamma, ring_first = extended_network(state)
        np.savez_compressed(out/f"state_{step:05d}.npz", nodes=nodes, edges=edges, gamma=gamma, first=state.wing.node_count, ring_first=ring_first)

    run = run_prescribed_multi(backend, bodies, wind, steps=steps, keep_states=keep, progress=progress, attitude_scheme="fourth_order_central",
                               separation_m=CONTRACT["separation_m"], state_sink=save_state)
    k3 = {name: consistency_verdict(run.motion_consistency[name]) for name in names}
    pairs = {name: periodicity(run.records, name, steps_per_lap) for name in names}
    k7 = {name: verdict(pairs[name], run.status) for name in names}
    laps = {name: per_lap(run.records, name, steps_per_lap) for name in names}
    relative = relative_to_a(laps["A"], laps["B"]) if "B" in laps else None
    checks = dict(settings=settings, status=run.status, error=run.error, error_type=run.error_type, failed_step=run.failed_step,
                  failed_module=run.failed_module, frames=len(run.records), elapsed_s=time.perf_counter()-tick, started_unix=started,
                  K3=k3, K7=k7, K7_pairs=pairs, per_lap=laps, b_relative_to_a=relative, saved_states=keep,
                  step_times_s=[round(s, 4) for s in run.step_seconds],
                  per_step=compact_records(run.records, names, steps_per_lap, args.laps), wake=None)
    (out/"checks.json").write_text(json.dumps(checks, default=str)+"\n", encoding="utf-8")
    report = dict(checks, per_step=None, motions={name: dict(position_m=b.motion.position_m.tolist(), rotation=b.motion.rotation.tolist(),
                                                              velocity_m_s=b.motion.velocity_m_s.tolist(),
                                                              angular_velocity_rad_s=b.motion.angular_velocity_rad_s.tolist()) for name, b in zip(names, bodies)},
                  records=run.records)
    (out/"history.json").write_text(json.dumps(report, indent=1, default=str)+"\n", encoding="utf-8")
    # The wake summary comes after both receipts exist, so that a failure of it cannot cost the run (review R-02, 2026-10-19).
    wake = None
    try:
        if args.planes is None:
            stations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if args.single else [0.5, args.spacing+0.5, args.spacing+1.5, args.spacing+2.5]
        else:
            stations = [float(v) for v in args.planes.split(",") if v.strip()]
        if run.final_state is not None and stations:
            tick_wake = time.perf_counter()
            wake = wake_summary(run.final_state, bodies, motion_a, steps_per_lap, dt, args.wind, diameter, config.core_radius_m,
                                stations, args.plane_count, ring_core=args.ring_core)
            wake["elapsed_s"] = time.perf_counter()-tick_wake
    except Exception as error:  # noqa: BLE001
        wake = dict(error=f"{type(error).__name__}: {str(error)[:300]}")
    checks["wake"] = wake
    (out/"checks.json").write_text(json.dumps(checks, default=str)+"\n", encoding="utf-8")
    print(json.dumps(dict(out=str(out), status=run.status, error=run.error, failed_step=run.failed_step, failed_module=run.failed_module,
                          frames=len(run.records), elapsed_s=round(checks["elapsed_s"], 1), K3={n: k3[n]["worst_relative"] for n in names},
                          K7={n: k7[n]["verdict"] for n in names}, b_relative_to_a=relative[-1] if relative else None,
                          checks_bytes=(out/"checks.json").stat().st_size)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
