"""Checks D1-D5 of the drag contract (G5 step 3, plan 3.2 first part): profile drag and parasitic drag on the vortex plant.

usage: drag_checks.py OUT_DIR BUILD [--parts d1,d2,d3,d4,d5] [--steps N]

D1, D2  the unit tests of ``tests/test_drag_parts.py`` (run here and recorded). D2(a), the default path being
        unchanged bit for bit, is judged on the closed loop itself (contract revision 1): the repeatability run B4 of
        ``closed_loop_checks.py`` is made once more at this commit, untouched, and its history must have the SHA-256
        recorded for B4 before this step (``results/closed-loop-checks-b124-run2.json``).
D3      straight steady flight, sideslip zero, body incidence 0...6 deg, no rotation, no deflection: the lifting
        surfaces with the profile-drag case (production crosswind closure, straight semi-infinite wake) against the
        low-incidence database of Makani's simulator with its offsets (``aero.makani_database``, zero rates, zero
        flaps). Criterion: the lift-to-drag ratio ``CL/(CD + 0.075)`` against the database's within 10 per cent at
        2, 3, 4 and 5 deg. Report: lift, drag and pitching moment on both sides, the profile part of the drag (same
        solve with the archived case), panels whose section incidence is outside the XFOIL table, a free-wake run at
        4 deg; added after the first D3 rows were seen (report only, contract revision 1): the database computed at
        zero rates (Makani's own file) beside the extrapolation of the baseline file, and the two sides at equal lift.
D4      the first lap reflown: case CL1 of the closed-loop checks with the profile-drag case and the parasitic drag
        coefficient 0.075 (case CL1-D). Conditions (i)-(iv) of B3, with slack and height judged over the whole run
        (contract revision 1), and (v) the airspeed within 5 m/s of its command from 5 s after the hand-over.
        ``--steps`` changes the length of the closed loop: such a run judges nothing.
D5      the lead-in and the first 50 closed-loop steps of CL1-D in two processes: bit-identical. Reported with it: the
        drag of the lifting surfaces over the pinned lead-in against the same steps of the untouched B4 run (the
        motion is the same, so the difference is the profile drag: evidence that the child process used the new case).

The closed-loop runs reuse ``closed_loop_checks.py`` unchanged: this script replaces, in its own process, the case
builder by one that swaps the section data, and the driver class by the same class with the parasitic coefficient
set. Every record of such a run carries ``parasitic_drag_coefficient`` and the case settings name the case file.
"""

import argparse
import functools
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(REPO/"docs/tasks/OF-0007-wingfull20261001"))

from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.handover import CROSSWIND_SOLVE, SectionHandover  # noqa: E402
from openfawe.aero.makani_database import DatabaseSettings, MakaniDatabase, dcm_wind_to_body  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.vsm import solve_vsm  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.flight.closed_loop import CrosswindLoop  # noqa: E402

import alpha_sweep as S  # noqa: E402  (straight-flight helpers of the full-incidence task)
import closed_loop_checks as checks  # noqa: E402

ARCHIVED_CASE = "cases/qualification/m600_makani_reconstructed.json"
DRAG_CASE = "cases/qualification/m600_makani_reconstructed_profile_drag.json"
DATABASE = REPO/"cases/polars/makani/m600_aswing_baseline.json"
ZERO_RATE_DATABASE = REPO/"cases/polars/makani/m600_aswing_baseline_zero_angular_rate.json"
B4_RECEIPT = TASK/"results/closed-loop-checks-b124-run2.json"          # the B4 history hashes before this step
PARASITIC = 0.075
ALPHAS_DEG = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
JUDGED_ALPHAS_DEG = (2.0, 3.0, 4.0, 5.0)
RATIO_LIMIT = 0.10
AIRSPEED_LIMIT_M_S = 5.0
SETTLE_S = 5.0
FREE_WAKE_ALPHA_DEG, FREE_WAKE_CHORDS = 4.0, 25.0


# ---------------------------------------------------------------------------------------------------------------------
# D1, D2
# ---------------------------------------------------------------------------------------------------------------------
def unit_tests():
    sys.path.insert(0, str(REPO/"tests"))
    import test_drag_parts
    out = {}
    for name, case in (("D1", test_drag_parts.ProfileDragCaseTests), ("D2", test_drag_parts.ParasiticDragTests)):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        out[name] = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                         names=[line.split(" ")[0] for line in stream.getvalue().splitlines() if "..." in line],
                         passed=bool(result.wasSuccessful() and result.testsRun > 0 and not result.skipped))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------------------------------------------------
def wind_axes(force_makani_body, alpha_rad):
    """Lift and drag coefficients from Makani body-axis force coefficients at zero sideslip."""
    wind = dcm_wind_to_body(alpha_rad, 0.0).T@np.asarray(force_makani_body, dtype=float)
    return float(-wind[2]), float(-wind[0])


def straight(aircraft, alphas_deg):
    rows, seed = [], None
    for alpha in alphas_deg:
        wind = S.inflow_makani(math.radians(alpha), 0.0)
        solution = solve_vsm(aircraft.wing, aircraft.polar, wind, settings=CROSSWIND_SOLVE, initial=seed, density=S.RHO)
        seed = solution.circulation
        cfm = S.makani_coefficients(solution.force, solution.moment(aircraft.wing))
        lift, drag = wind_axes(cfm[:3], math.radians(alpha))
        rows.append(dict(alpha_deg=alpha, CL=lift, CD=drag, Cm=float(cfm[4]), cfm=cfm.tolist(), iterations=int(solution.iterations),
                         section_alpha_deg=np.degrees(solution.polar_alpha_rad).tolist()))
    return rows


def free_wake(aircraft, alpha_deg, chords, dt_s=0.04):
    steps = int(math.ceil(chords*S.MAKANI_REF["c"]/(S.SPEED*dt_s)))
    config = WakeConfig(dt_s=dt_s, max_rows=steps, core_radius_m=0.15, convection_integrator="euler", core_evolution="constant",
                        solve=CROSSWIND_SOLVE, velocity_backend="tree", tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
    backend = AeroBackend("vsm_free_wake", wake=config)
    wind = S.inflow_makani(math.radians(alpha_deg), 0.0)
    state = backend.start(aircraft.wing, aircraft.polar, wind)
    for index in range(1, steps+1):
        state = backend.advance(state, aircraft.polar, wind, next_time_s=index*dt_s)
    cfm = S.makani_coefficients(state.solution.force, state.solution.moment(aircraft.wing))
    lift, drag = wind_axes(cfm[:3], math.radians(alpha_deg))
    return dict(alpha_deg=alpha_deg, chords=chords, steps=steps, dt_s=dt_s, CL=lift, CD=drag, Cm=float(cfm[4]))


def d3():
    drag_case, archived = load_aircraft(REPO/DRAG_CASE), load_aircraft(REPO/ARCHIVED_CASE)
    ours, bare = straight(drag_case, ALPHAS_DEG), straight(archived, ALPHAS_DEG)
    with_offsets = MakaniDatabase.from_files(DATABASE)
    without = MakaniDatabase.from_files(DATABASE, settings=DatabaseSettings().without_offsets())
    table_ends = [(polar.alpha_deg[1], polar.alpha_deg[-2]) for polar in drag_case.polar.polars]      # the XFOIL range of each panel
    rows = []
    for mine, plain in zip(ours, bare):
        alpha = math.radians(mine["alpha_deg"])
        base = with_offsets.low_coefficients(alpha, 0.0, np.zeros(3), np.zeros(8), 0.0)
        raw = without.low_coefficients(alpha, 0.0, np.zeros(3), np.zeros(8), 0.0)
        lift_db, drag_db = wind_axes(base[:3], alpha)
        lift_raw, drag_raw = wind_axes(raw[:3], alpha)
        ratio, ratio_db = mine["CL"]/(mine["CD"]+PARASITIC), lift_db/drag_db
        outside = int(sum(1 for value, (low, high) in zip(mine["section_alpha_deg"], table_ends) if value < low or value > high))
        rows.append(dict(alpha_deg=mine["alpha_deg"],
                         ours=dict(CL=mine["CL"], CD_lifting_surfaces=mine["CD"], CD_of_which_profile=mine["CD"]-plain["CD"],
                                   CD_induced_archived_case=plain["CD"], CD_parasitic=PARASITIC, CD_total=mine["CD"]+PARASITIC,
                                   Cm_origin=mine["Cm"], lift_to_drag=ratio, CL_archived_case=plain["CL"], iterations=mine["iterations"],
                                   panels_with_section_incidence_outside_the_xfoil_table=outside,
                                   section_incidence_range_deg=[float(min(mine["section_alpha_deg"])), float(max(mine["section_alpha_deg"]))]),
                         database=dict(CL_with_offsets=lift_db, CD_with_offsets=drag_db, Cm_origin=float(base[4]), lift_to_drag=ratio_db,
                                       CL_without_offsets=lift_raw, CD_without_offsets=drag_raw),
                         difference=dict(CL=mine["CL"]-lift_db, CD_total=mine["CD"]+PARASITIC-drag_db, Cm_origin=mine["Cm"]-float(base[4]),
                                         CD_lifting_surfaces_minus_database_without_offsets=mine["CD"]-drag_raw,
                                         lift_to_drag_relative=ratio/ratio_db-1.0)))
    # Report only (added after the first rows were seen): Makani's zero-rate file, and the comparison at equal lift.
    zero_rate = MakaniDatabase.from_files(ZERO_RATE_DATABASE, settings=DatabaseSettings().without_offsets())
    lift_db = np.array([row["database"]["CL_with_offsets"] for row in rows])
    drag_db = np.array([row["database"]["CD_with_offsets"] for row in rows])
    lift_raw = np.array([row["database"]["CL_without_offsets"] for row in rows])
    drag_raw = np.array([row["database"]["CD_without_offsets"] for row in rows])
    for row in rows:
        alpha = math.radians(row["alpha_deg"])
        lift_zero, drag_zero = wind_axes(zero_rate.low_coefficients(alpha, 0.0, zero_rate.omega_hat_0, np.zeros(8), 0.0)[:3], alpha)
        row["database"].update(CL_zero_rate_file_without_offsets=lift_zero, CD_zero_rate_file_without_offsets=drag_zero)
        lift = row["ours"]["CL"]
        equal = dict(note="database rows interpolated in their own lift over the incidences of this sweep; None outside them")
        equal["CD_database_with_offsets_at_our_CL"] = float(np.interp(lift, lift_db, drag_db)) if lift_db[0] <= lift <= lift_db[-1] else None
        equal["CD_database_without_offsets_at_our_CL"] = float(np.interp(lift, lift_raw, drag_raw)) if lift_raw[0] <= lift <= lift_raw[-1] else None
        if equal["CD_database_with_offsets_at_our_CL"] is not None:
            equal["lift_to_drag_relative"] = (lift/row["ours"]["CD_total"])/(lift/equal["CD_database_with_offsets_at_our_CL"])-1.0
        if equal["CD_database_without_offsets_at_our_CL"] is not None:
            equal["CD_lifting_surfaces_minus_database_without_offsets"] = row["ours"]["CD_lifting_surfaces"]-equal["CD_database_without_offsets_at_our_CL"]
        row["at_equal_lift"] = equal
    judged = [row for row in rows if row["alpha_deg"] in JUDGED_ALPHAS_DEG]
    worst = max(abs(row["difference"]["lift_to_drag_relative"]) for row in judged)
    wake = free_wake(drag_case, FREE_WAKE_ALPHA_DEG, FREE_WAKE_CHORDS)
    same = next(row for row in rows if row["alpha_deg"] == FREE_WAKE_ALPHA_DEG)
    wake["minus_the_straight_wake"] = dict(CL=wake["CL"]-same["ours"]["CL"], CD=wake["CD"]-same["ours"]["CD_lifting_surfaces"],
                                           Cm=wake["Cm"]-same["ours"]["Cm_origin"])
    return dict(rows=rows, judged_alphas_deg=list(JUDGED_ALPHAS_DEG), limit_relative=RATIO_LIMIT, largest_relative_gap_of_the_ratio=worst,
                free_wake=wake, settings=dict(speed_m_s=S.SPEED, density_kg_m3=S.RHO, reference=S.MAKANI_REF, parasitic=PARASITIC,
                                              ours="profile-drag case, production crosswind closure, straight semi-infinite wake",
                                              database="low-incidence database, zero dimensionless rates, zero deflections, zero thrust coefficient"),
                passed=bool(worst <= RATIO_LIMIT))


# ---------------------------------------------------------------------------------------------------------------------
# D4, D5: the closed loop of closed_loop_checks.py on the case CL1-D
# ---------------------------------------------------------------------------------------------------------------------
def install_case():
    """Make ``closed_loop_checks`` build CL1-D in this process: section data with profile drag, parasitic drag in the driver."""
    original = checks.cl_base

    def cl_base(closed_steps, backend, lead_in):
        case = original(closed_steps, backend, lead_in)
        low, high = load_aircraft(REPO/DRAG_CASE), load_aircraft(REPO/checks.A5.HIGH_CASE)
        if low.wing.panels != case.wing.panels or not np.array_equal(low.wing.bound, case.wing.bound):
            raise ValueError("the profile-drag case does not hold the geometry of the case it replaces")
        case.wing = low.wing
        case.handover = SectionHandover(low.polar, high.polar, seeding=checks.A5.CONTRACT["seeding"])
        case.settings.update(low_case=DRAG_CASE, parasitic_drag_coefficient=PARASITIC, case_name="CL1-D")
        return case
    checks.cl_base = cl_base
    checks.CrosswindLoop = functools.partial(CrosswindLoop, parasitic_drag_coefficient=PARASITIC)


def summarise(api, case, final, records, log, lead_in, returncode, stderr):
    if final is None:
        final = dict(status="aborted", error=f"the process ended with return code {returncode}: {stderr.strip()[-600:]}",
                     failed_step=None, failed_module="process", step_seconds=[], settings={}, controller_failure=None)
    run = SimpleNamespace(records=records, status=final["status"], error=final["error"], failed_step=final["failed_step"],
                          failed_module=final["failed_module"], step_seconds=final["step_seconds"], settings=final["settings"])
    controller = SimpleNamespace(log=log, failed=final["controller_failure"])
    if final["status"] == "aborted" and log:
        run.failed_step = log[-1]["step"]+(0 if len(log)+lead_in > len(records) else 1)
    summary = checks.summarise(api, case, run, controller, lead_in)
    time_s = np.array([(entry["step"]-lead_in)*checks.STEP_S for entry in log])
    gap = np.array([entry["estimate"]["airspeed_m_s"]-entry["airspeed_cmd_m_s"] for entry in log])
    radius = np.array([math.hypot(*entry["current_pos_cw"])-entry["path_radius_target_m"] for entry in log])
    settled = time_s >= SETTLE_S
    drag_fields = [record.get("parasitic_drag_coefficient") for record in records[lead_in:]]
    tracking = dict(after_s=SETTLE_S, steps=int(settled.sum()), limit_m_s=AIRSPEED_LIMIT_M_S,
                    largest_airspeed_error_m_s=float(np.abs(gap[settled]).max()) if settled.any() else None,
                    airspeed_error_m_s_whole_run=[float(gap.min()), float(gap.max())] if len(gap) else None,
                    radius_error_m_whole_run=[float(radius.min()), float(radius.max())] if len(radius) else None,
                    radius_error_m_after_5_s=[float(radius[settled].min()), float(radius[settled].max())] if settled.any() else None,
                    radius_note="report only: Makani's batch scoring calls -25...+50 m full marks; the reference plant's judged run had "
                                "-13.7...+31.8 m; neither is a criterion here")
    tracking["passed"] = bool(settled.any() and tracking["largest_airspeed_error_m_s"] <= AIRSPEED_LIMIT_M_S)
    whole_run = bool(summary.get("largest_slack_segments") == 0 and summary.get("lowest_height_m") is not None
                     and summary["lowest_height_m"] > 0.0)                # (iv) over the whole run (contract revision 1)
    b3_conditions = bool(summary["passed"] and whole_run)
    power = np.array([record.get("rotor_shaft_power_W", np.nan) for record in records[lead_in:lead_in+len(log)]], dtype=float)
    angle = np.unwrap(np.array([entry["loop_angle_rad"] for entry in log])) if log else np.array([])
    lap_end = next((k for k in range(len(angle)) if angle[0]-angle[k] >= 2.0*math.pi), None)
    summary.update(slack_and_height_hold_over_the_whole_run=whole_run,
                   rotor_shaft_power_mean_over_the_first_lap_W=None if lap_end is None else float(np.nanmean(power[:lap_end])),
                   rotor_shaft_power_note="table shaft power, positive while generating; not a power prediction")
    summary.update(conditions_i_to_iv=b3_conditions, tracking=tracking,
                   every_closed_loop_record_carries_the_parasitic_coefficient=bool(drag_fields and all(v == PARASITIC for v in drag_fields)),
                   controller_settings=final.get("controller_settings"), consistency=final.get("consistency"))
    summary["passed"] = bool(b3_conditions and tracking["passed"] and summary["every_closed_loop_record_carries_the_parasitic_coefficient"])
    return summary


def lead_in_drag(records, lead_in):
    """Drag coefficient of the lifting surfaces over the pinned lead-in: force along the relative wind over q S."""
    out = []
    for record in records[:lead_in]:
        wind = np.array(record["relative_wind_body_m_s"], dtype=float)
        out.append(float(np.array(record["force_body_N"], dtype=float)@wind/np.linalg.norm(wind)/(record["dynamic_pressure_Pa"]*S.MAKANI_REF["S"])))
    return np.array(out)


def lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def provenance():
    def git(*args):
        done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()[:200]}")
        return done.stdout.strip()
    import numba
    import scipy
    return dict(git_commit=git("rev-parse", "HEAD"), src_clean=git("status", "--porcelain", "--", "src", "cases") == "",
                src_tree=git("rev-parse", "HEAD:src"), cases_tree=git("rev-parse", "HEAD:cases"), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, platform=platform.platform(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                closed_loop_checks_sha256=hashlib.sha256((TASK/"closed_loop_checks.py").read_bytes()).hexdigest(),
                drag_case_sha256=hashlib.sha256((REPO/DRAG_CASE).read_bytes()).hexdigest(),
                generator_sha256=hashlib.sha256((TASK/"make_profile_drag_case.py").read_bytes()).hexdigest(),
                tests_sha256=hashlib.sha256((REPO/"tests/test_drag_parts.py").read_bytes()).hexdigest())


def main(out, argv):
    parser = argparse.ArgumentParser(prog="drag_checks.py")
    parser.add_argument("build")
    parser.add_argument("--parts", default="d1,d2,d3,d4,d5")
    parser.add_argument("--steps", type=int, default=None, help="closed-loop steps of d4 (default: 1.5 nominal periods); shorter judges nothing")
    parser.add_argument("--d4-child", action="store_true")
    parser.add_argument("--d5-child", default=None)
    args = parser.parse_args(argv)
    lead_in, backend = checks.LEAD_IN, "tree"
    if args.d4_child:
        install_case()
        checks.b3_child(args.build, out, backend, lead_in, args.steps, False, "prep-trans-out")
        return
    if args.d5_child:
        install_case()
        checks.b4_child(args.build, args.d5_child, backend, lead_in)
        return
    out = Path(out)
    library = Path(args.build)/"lib"/"libmakani_controller.so"
    library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()        # a wrong BUILD path fails here, before anything runs
    out.mkdir(parents=True, exist_ok=False)
    started, tick = time.time(), time.perf_counter()
    at_start = provenance()
    default_run = None
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    criteria, history = {}, {}

    def note(text):
        print(f"[{time.perf_counter()-tick:7.0f}s] {text}", flush=True)
    if "d1" in parts or "d2" in parts:
        units = unit_tests()
        for name in ("D1", "D2"):
            if name.lower() in parts:
                criteria[name] = units[name]
                note(f"{name}: {units[name]['passed']} ({units[name]['tests']} tests)")
    if "d2" in parts:
        # D2(a) on the closed loop itself: the untouched B4 run of closed_loop_checks.py, once, against its recorded hash.
        target = out/"d2-default-path-b4.json"
        done = subprocess.run([sys.executable, str(TASK/"closed_loop_checks.py"), str(out), args.build, "--b4-child", str(target),
                               "--backend", backend, "--lead-in", str(lead_in)], capture_output=True, text=True)
        text = target.read_text(encoding="utf-8") if done.returncode == 0 and target.exists() else None
        recorded = json.loads(B4_RECEIPT.read_text(encoding="utf-8"))["criteria"]["B4"]["sha256"]
        digest = None if text is None else hashlib.sha256(text.encode()).hexdigest()
        same = bool(digest is not None and recorded[0] == recorded[1] == digest)
        default_run = None if text is None else json.loads(text)
        criteria["D2"].update(default_path=dict(what="B4 of closed_loop_checks.py repeated untouched at this commit", sha256=digest,
                                                recorded_sha256=recorded, recorded_in=str(B4_RECEIPT.relative_to(REPO)),
                                                bit_identical=same, error=None if text is not None else done.stderr.strip()[-400:]),
                              unit_tests_passed=criteria["D2"]["passed"])
        criteria["D2"]["passed"] = bool(criteria["D2"]["passed"] and same)
        note(f"D2 with the default path: {criteria['D2']['passed']} (history hash equal to the recorded one: {same})")
    if "d3" in parts:
        criteria["D3"] = d3()
        note(f"D3: {criteria['D3']['passed']}; largest gap of the lift-to-drag ratio {criteria['D3']['largest_relative_gap_of_the_ratio']:.4f}")
        for row in criteria["D3"]["rows"]:
            note(f"   alpha {row['alpha_deg']:.0f}: CL {row['ours']['CL']:.3f} / {row['database']['CL_with_offsets']:.3f}  CD "
                 f"{row['ours']['CD_total']:.4f} / {row['database']['CD_with_offsets']:.4f}  L/D {row['ours']['lift_to_drag']:.2f} / "
                 f"{row['database']['lift_to_drag']:.2f} ({100*row['difference']['lift_to_drag_relative']:+.1f}%)")
    if "d4" in parts:
        install_case()
        api = checks.load_api(args.build)
        case = checks.cl_base(args.steps, backend, lead_in)
        trial = args.steps is not None and args.steps != case.full_steps
        note(f"D4: CL1-D, lead-in {lead_in}, closed loop {case.closed_steps} steps{' (not the contract length: judges nothing)' if trial else ''}")
        command = [sys.executable, str(Path(__file__).resolve()), str(out), args.build, "--d4-child"]+(["--steps", str(args.steps)] if args.steps is not None else [])
        child_tick = time.perf_counter()
        done = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        final = json.loads((out/"b3-final.json").read_text(encoding="utf-8")) if (out/"b3-final.json").exists() else None
        records, log = lines(out/"b3-records.jsonl"), lines(out/"b3-log.jsonl")
        summary = summarise(api, case, final, records, log, lead_in, done.returncode, done.stderr)
        summary.update(case="CL1-D", seconds=time.perf_counter()-child_tick, shortened_trial=bool(trial), case_settings=case.settings)
        if trial:
            summary.update(judged=False, would_have_passed=summary["passed"], passed=None)
        criteria["D4"] = summary
        history["CL1-D"] = dict(records=records, controller_log=log)
        note(f"D4: status {summary['status']} {summary['error'] or ''}; lap {summary.get('lap_completed')}; tracking {summary['tracking']}; "
             f"passed {summary['passed']}")
    if "d5" in parts:
        texts = []
        for index in range(2):
            target = out/f"d5-run{index}.json"
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(out), args.build, "--d5-child", str(target)],
                                  capture_output=True, text=True)
            texts.append(target.read_text(encoding="utf-8") if done.returncode == 0 and target.exists() else None)
            if done.returncode:
                note(f"D5 child {index} failed: {done.stderr.strip()[-400:]}")
        same = texts[0] is not None and texts[0] == texts[1]
        first = None if texts[0] is None else json.loads(texts[0])
        carried = bool(first and all(record.get("parasitic_drag_coefficient") == PARASITIC for record in first["records"][lead_in:]))
        evidence = None
        if first is not None and default_run is not None:           # same pinned motion: the difference is the profile drag
            mine, theirs = lead_in_drag(first["records"], lead_in), lead_in_drag(default_run["records"], lead_in)
            evidence = dict(what="drag coefficient of the lifting surfaces over the pinned lead-in, CL1-D minus the untouched B4 run",
                            mean=float(np.mean(mine-theirs)), range=[float((mine-theirs).min()), float((mine-theirs).max())],
                            mean_CL1_D=float(mine.mean()), mean_untouched=float(theirs.mean()), steps=int(len(mine)))
        criteria["D5"] = dict(steps=checks.REPEAT_STEPS, both_finished=bool(all(t is not None for t in texts)),
                              lead_in_drag_against_the_untouched_run=evidence,
                              status=None if first is None else first["status"], bit_identical=bool(same),
                              records_carry_the_parasitic_coefficient=carried,
                              sha256=[None if t is None else hashlib.sha256(t.encode()).hexdigest() for t in texts],
                              passed=bool(same and first["status"] == "complete" and carried))
        note(f"D5: {criteria['D5']['passed']}")
    end = provenance()
    result = dict(what="G5 step 3: profile drag and parasitic drag on the vortex-method plant (drag contract, D1-D5)", parts=parts,
                  criteria=criteria, library_sha256=library_sha256,
                  binding_sha256=hashlib.sha256((Path(args.build)/"lib"/"makani_controller_api.py").read_bytes()).hexdigest(),
                  elapsed_s=time.perf_counter()-tick,
                  started_unix=started, at_start=at_start,
                  code_unchanged_during_the_run=bool(all(at_start[k] == end[k] for k in ("src_tree", "cases_tree", "script_sha256",
                                                                                         "closed_loop_checks_sha256", "drag_case_sha256", "generator_sha256", "tests_sha256"))
                                                     and at_start["src_clean"] and end["src_clean"]), **end)
    text = json.dumps(dict(result, history=history), default=str)+"\n"
    (out/"history.json").write_text(text, encoding="utf-8")
    result["history_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out/"checks.json").write_text(json.dumps(result, indent=1, default=str)+"\n", encoding="utf-8")
    print(json.dumps({name: value.get("passed") for name, value in criteria.items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
