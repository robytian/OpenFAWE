"""Checks E1-E5 of the pylon contract (G5 step 3, plan 3.2 second part): the pylons as a vortex system of their own in the loop.

usage: pylon_loop_checks.py OUT_DIR BUILD [--parts e1,e2,e3,e4,e5] [--steps N]

E1  the wiring of the companion surfaces: the unit tests of ``tests/test_companion.py``; on the case CL1, pinned to
    its nominal loop for 40 steps, the main lifting surfaces hung on the hook once more must give the loads of the
    main solve bit for bit, and the main records must be those of the same run without any companion, bit for bit
    (added after the review, a tightening); the existing tests of the free-flight and closed-loop drivers pass; and the
    repeatability run B4 of ``closed_loop_checks.py``, made untouched at this commit, has its recorded history hash
    (the default path is unchanged bit for bit).
E2  on the same 40 pinned steps the pylon group on the hook gives, bit for bit, the force and the moment of
    ``run_prescribed`` for the same four pylons, motion and settings.
E3  straight steady flight with the pylons (a new numbered check; it does not rejudge D3): main lifting surfaces and
    pylon group solved separately with straight wakes and added, plus the parasitic 0.075, against Makani's database
    with offsets. Criterion: lift-to-drag ratio within 10 per cent at 2, 3, 4 and 5 deg. Report: side force and its
    slope against the controller's simple model and the database, yaw and roll moment slopes, the pylon group alone.
E4  the first lap reflown with the pylons (case CL1-DP = CL1-D of ``drag_checks.py`` plus the pylon group):
    conditions (i)-(iv) over the whole run, (v) the airspeed within 5 m/s of its command from 5 s after the hand-over,
    and no failed solve of the pylon group. ``--steps`` changes the length: such a run judges nothing.
E5  the lead-in and the first 50 closed-loop steps of CL1-DP in two processes: bit-identical.

The closed-loop runs reuse ``closed_loop_checks.py`` and ``drag_checks.py`` unchanged: in its own process this script
installs the case of ``drag_checks`` and replaces the ``run_free`` that ``closed_loop_checks`` calls by one that chains
the pylon group behind the controller's load hook.
"""

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(TASK))

import drag_checks as drag  # noqa: E402  (it puts the other task folders on the path)
from drag_checks import S, checks  # noqa: E402

from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.handover import CROSSWIND_SOLVE, SectionHandover  # noqa: E402
from openfawe.aero.makani_database import DatabaseSettings, MakaniDatabase  # noqa: E402
from openfawe.aero.vsm import solve_vsm  # noqa: E402
from openfawe.flight.companion import CompanionSurfaces, chain_extra_loads  # noqa: E402
from openfawe.flight.free_flight import Release, run_free  # noqa: E402
from openfawe.flight.prescribed import run_prescribed  # noqa: E402

A3, A5 = checks.A3, checks.A5
PYLON_LOW_CASE = "cases/qualification/m600_makani_reconstructed_pylons_profile_drag.json"
PYLON_HIGH_CASE = A5.HIGH_PYLON_CASE
SIMPLE_MODEL = dict(CY_0=-0.08687, dCY_dbeta=-1.4205533)        # the controller's simple aero model (Makani's literals)
PINNED_STEPS = 40
SIDESLIPS_DEG = (-2.0, 0.0, 2.0)
SLOPE_ALPHAS_DEG = (2.0, 4.0)


def pylon_parts():
    low, high = load_aircraft(REPO/PYLON_LOW_CASE), load_aircraft(REPO/PYLON_HIGH_CASE)
    if not (np.array_equal(high.wing.cp, low.wing.cp) and np.array_equal(high.wing.dl, low.wing.dl)):
        raise ValueError("the two pylon cases do not share one geometry")
    (body, low_polar), (_, high_polar) = A3.pylons_only(low), A3.pylons_only(high)
    return body, low_polar, SectionHandover(low_polar, high_polar, seeding=A5.CONTRACT["seeding"])


def pylon_group(backend, wind):
    """The pylon group with the wake settings of ``backend`` (an AeroBackend holds settings only, no state) and that wind."""
    body, _, handover = pylon_parts()
    return CompanionSurfaces("pylons", backend, body, wind, handover=handover)


def install_case():
    """CL1-DP in this process: the case of ``drag_checks`` and the pylon group chained behind the controller's hook."""
    drag.install_case()
    built = checks.cl_base

    def cl_base(closed_steps, backend, lead_in):
        case = built(closed_steps, backend, lead_in)
        case.settings.update(case_name="CL1-DP", pylon_low_case=PYLON_LOW_CASE, pylon_high_case=PYLON_HIGH_CASE,
                             pylons="a vortex system of their own, no mutual induction with wing and tails (decision 0028, model B)")
        return case

    def run_free_with_pylons(backend, wing, polar, body, state, wind, *args, extra_loads=None, **kw):
        group = pylon_group(backend, wind)                        # the same wake settings and wind as the main surfaces, by construction
        return run_free(backend, wing, polar, body, state, wind, *args, extra_loads=chain_extra_loads(extra_loads, group.extra_loads), **kw)
    checks.cl_base = cl_base
    checks.run_free = run_free_with_pylons


# ---------------------------------------------------------------------------------------------------------------------
# E1, E2
# ---------------------------------------------------------------------------------------------------------------------
def unit_tests():
    sys.path.insert(0, str(REPO/"tests"))
    import test_closed_loop
    import test_companion
    import test_free_flight
    out = {}
    for name, module in (("companion", test_companion), ("free_flight", test_free_flight), ("closed_loop", test_closed_loop)):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        out[name] = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                         passed=bool(result.wasSuccessful() and result.testsRun > 0))
    return out


def pinned(backend):
    """CL1 pinned to its nominal loop for 40 steps, with the main surfaces as their own companion and the pylon group."""
    case = checks.cl_base(1, backend, PINNED_STEPS)                   # the untouched case CL1 of closed_loop_checks.py
    state, line = checks.start_of(case)
    times = np.arange(PINNED_STEPS+1)*checks.STEP_S
    release = Release(case.rigid, case.motion, release_time_s=times[-1]+1.0, trim=False)     # never released
    twin = CompanionSurfaces("twin", case.aero(), case.wing, case.wind, handover=case.handover)
    pylons = pylon_group(case.aero(), case.wind)
    run = run_free(case.aero(), case.wing, None, case.rigid, state, case.wind, times=times, handover=case.handover, pin_to=case.motion,
                   external=release, extra_loads=chain_extra_loads(twin.extra_loads, pylons.extra_loads), **line)
    state, line = checks.start_of(case)                                # the same run without any companion
    plain = run_free(case.aero(), case.wing, None, case.rigid, state, case.wind, times=times, handover=case.handover, pin_to=case.motion,
                     external=Release(case.rigid, case.motion, release_time_s=times[-1]+1.0, trim=False), **line)
    main_fields = ("force_inertial_N", "moment_origin_inertial_Nm", "iterations", "handover_weight", "closure", "seeding",
                   "tether_tension_kite_N", "position_m")
    untouched = plain.status == run.status == "complete" and len(plain.records) == len(run.records) and all(
        a[key] == b[key] for a, b in zip(run.records, plain.records) for key in main_fields)
    body, _, handover = pylon_parts()
    reference = run_prescribed(case.aero(), body, None, case.motion, case.wind, handover=handover, steps=PINNED_STEPS)
    pairs = (("twin_force_inertial_N", "force_inertial_N"), ("twin_moment_origin_inertial_Nm", "moment_origin_inertial_Nm"),
             ("twin_force_body_N", "force_body_N"), ("twin_iterations", "iterations"), ("twin_handover_weight", "handover_weight"),
             ("twin_closure", "closure"), ("twin_seeding", "seeding"))
    twin_same = run.status == "complete" and len(run.records) == PINNED_STEPS+1 and all(
        record[a] == record[b] for record in run.records for a, b in pairs)
    force_gap = moment_gap = None
    pylon_same = False
    p2b = None
    if run.status == "complete" and reference.status == "complete" and len(reference.records) == len(run.records):
        mine = np.array([record["pylons_force_inertial_N"] for record in run.records])
        theirs = np.array([record["force_inertial_N"] for record in reference.records])
        mine_m = np.array([record["pylons_moment_origin_inertial_Nm"] for record in run.records])
        theirs_m = np.array([record["moment_origin_inertial_Nm"] for record in reference.records])
        force_gap = float((np.linalg.norm(mine-theirs, axis=1)/np.linalg.norm(theirs, axis=1)).max())
        moment_gap = float((np.linalg.norm(mine_m-theirs_m, axis=1)/np.linalg.norm(theirs_m, axis=1)).max())
        pylon_same = bool(np.array_equal(mine, theirs) and np.array_equal(mine_m, theirs_m))
        # Report only: the same comparison as criterion P2b of the plant contract words it (largest difference over the
        # largest norm of the steps, threshold 1e-7; closure identical step by step; weight within 1e-9).
        force_p2b = float(np.linalg.norm(mine-theirs, axis=1).max()/np.linalg.norm(theirs, axis=1).max())
        moment_p2b = float(np.linalg.norm(mine_m-theirs_m, axis=1).max()/np.linalg.norm(theirs_m, axis=1).max())
        closures = all(a["pylons_closure"] == b["closure"] and a["pylons_seeding"] == b["seeding"] for a, b in zip(run.records, reference.records))
        weight = max(abs(a["pylons_handover_weight"]-b["handover_weight"]) for a, b in zip(run.records, reference.records))
        iterations = all(a["pylons_iterations"] == b["iterations"] for a, b in zip(run.records, reference.records))
        p2b = dict(note="report only: the comparison as P2b of the plant contract words it", threshold=1e-7, force_gap=force_p2b,
                   moment_gap=moment_p2b, closure_and_seeding_identical=bool(closures), largest_weight_gap=float(weight),
                   iterations_identical=bool(iterations),
                   within_the_P2b_threshold=bool(force_p2b <= 1e-7 and moment_p2b <= 1e-7 and closures and weight <= 1e-9))
    pressure = np.array([record["dynamic_pressure_Pa"] for record in run.records]) if run.status == "complete" else None
    pylon_force = None if pressure is None else (np.array([record["pylons_force_body_N"] for record in run.records])
                                                 * np.array([-1.0, 1.0, -1.0])/(pressure[:, None]*S.MAKANI_REF["S"]))
    weights = [record["handover_weight"] for record in run.records] if run.status == "complete" else []
    e1 = dict(steps=PINNED_STEPS, status=[run.status, plain.status], error=[run.error, plain.error], failed_module=run.failed_module,
              main_surfaces_as_their_own_companion_bit_identical=bool(twin_same), fields_compared=[a for a, _ in pairs],
              main_records_untouched_by_the_companions=bool(untouched), main_fields_compared=list(main_fields),
              handover_weight_range_of_the_pinned_steps=[min(weights), max(weights)] if weights else None,
              cannot_show="an error common to the main solve and its twin (in the shared solve call or in the pose itself); with "
                          "weight zero over these steps the blended closure of a group is exercised by the unit test only")
    e2 = dict(steps=PINNED_STEPS, status=[run.status, reference.status], errors=[run.error, reference.error],
              largest_relative_force_gap=force_gap, largest_relative_moment_gap=moment_gap, bit_identical=bool(pylon_same),
              as_criterion_P2b_of_the_plant_contract=p2b,
              pylon_force_coefficients_makani_body_range=None if pylon_force is None else
              dict(CX=[float(pylon_force[:, 0].min()), float(pylon_force[:, 0].max())], CY=[float(pylon_force[:, 1].min()), float(pylon_force[:, 1].max())],
                   CZ=[float(pylon_force[:, 2].min()), float(pylon_force[:, 2].max())]),
              passed=bool(pylon_same))
    return e1, e2


# ---------------------------------------------------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------------------------------------------------
def solve_group(wing, polar, alpha_deg, beta_deg, seed=None):
    wind = S.inflow_makani(math.radians(alpha_deg), math.radians(beta_deg))
    solution = solve_vsm(wing, polar, wind, settings=CROSSWIND_SOLVE, initial=seed, density=S.RHO)
    return S.makani_coefficients(solution.force, solution.moment(wing)), solution


def e3():
    main = load_aircraft(REPO/drag.DRAG_CASE)
    pylons, pylon_polar, _ = pylon_parts()
    with_offsets = MakaniDatabase.from_files(drag.DATABASE)
    without = MakaniDatabase.from_files(drag.DATABASE, settings=DatabaseSettings().without_offsets())
    rows, seeds = [], [None, None]
    for alpha_deg in drag.ALPHAS_DEG:
        alpha = math.radians(alpha_deg)
        cfm_main, solved_main = solve_group(main.wing, main.polar, alpha_deg, 0.0, seeds[0])
        cfm_pylons, solved_pylons = solve_group(pylons, pylon_polar, alpha_deg, 0.0, seeds[1])
        seeds = [solved_main.circulation, solved_pylons.circulation]
        total = cfm_main+cfm_pylons
        lift, surfaces_drag = drag.wind_axes(total[:3], alpha)
        lift_main, drag_main = drag.wind_axes(cfm_main[:3], alpha)
        lift_pylons, drag_pylons = drag.wind_axes(cfm_pylons[:3], alpha)
        base = with_offsets.low_coefficients(alpha, 0.0, np.zeros(3), np.zeros(8), 0.0)
        raw = without.low_coefficients(alpha, 0.0, np.zeros(3), np.zeros(8), 0.0)
        lift_db, drag_db = drag.wind_axes(base[:3], alpha)
        lift_raw, drag_raw = drag.wind_axes(raw[:3], alpha)
        ratio, ratio_db = lift/(surfaces_drag+drag.PARASITIC), lift_db/drag_db
        rows.append(dict(alpha_deg=alpha_deg,
                         ours=dict(CL=lift, CD_lifting_surfaces=surfaces_drag, CD_total=surfaces_drag+drag.PARASITIC, Cm_origin=float(total[4]),
                                   CY_body=float(total[1]), lift_to_drag=ratio, main=dict(CL=lift_main, CD=drag_main, cfm=cfm_main.tolist()),
                                   pylons=dict(CL=lift_pylons, CD=drag_pylons, cfm=cfm_pylons.tolist())),
                         database=dict(CL_with_offsets=lift_db, CD_with_offsets=drag_db, Cm_origin=float(base[4]), CY_body=float(base[1]),
                                       lift_to_drag=ratio_db, CL_without_offsets=lift_raw, CD_without_offsets=drag_raw),
                         difference=dict(CL=lift-lift_db, CD_total=surfaces_drag+drag.PARASITIC-drag_db, Cm_origin=float(total[4]-base[4]),
                                         CY_body=float(total[1]-base[1]),
                                         CD_lifting_surfaces_minus_database_without_offsets=surfaces_drag-drag_raw,
                                         lift_to_drag_relative=ratio/ratio_db-1.0)))
    lift_db = np.array([row["database"]["CL_with_offsets"] for row in rows])
    drag_db = np.array([row["database"]["CD_with_offsets"] for row in rows])
    for row in rows:                                          # report only: the two sides at equal lift
        lift = row["ours"]["CL"]
        at = float(np.interp(lift, lift_db, drag_db)) if lift_db[0] <= lift <= lift_db[-1] else None
        row["at_equal_lift"] = dict(CD_database_with_offsets_at_our_CL=at,
                                    lift_to_drag_relative=None if at is None else at/row["ours"]["CD_total"]-1.0)
    slopes = []
    for alpha_deg in SLOPE_ALPHAS_DEG:                        # report only: side force and lateral moments against sideslip
        alpha, ours, base = math.radians(alpha_deg), {}, {}
        for beta_deg in SIDESLIPS_DEG:
            cfm_main, _ = solve_group(main.wing, main.polar, alpha_deg, beta_deg)
            cfm_pylons, _ = solve_group(pylons, pylon_polar, alpha_deg, beta_deg)
            ours[beta_deg] = dict(total=(cfm_main+cfm_pylons).tolist(), main=cfm_main.tolist(), pylons=cfm_pylons.tolist())
            base[beta_deg] = with_offsets.low_coefficients(alpha, math.radians(beta_deg), np.zeros(3), np.zeros(8), 0.0).tolist()
        span = math.radians(SIDESLIPS_DEG[-1]-SIDESLIPS_DEG[0])

        def slope(table, index, key=None):
            high, low = table[SIDESLIPS_DEG[-1]], table[SIDESLIPS_DEG[0]]
            if key is not None:
                high, low = high[key], low[key]
            return float((high[index]-low[index])/span)
        slopes.append(dict(alpha_deg=alpha_deg, sideslips_deg=list(SIDESLIPS_DEG),
                           ours=dict(CY_at_zero_sideslip=ours[0.0]["total"][1], CY_pylons_at_zero_sideslip=ours[0.0]["pylons"][1],
                                     CY_main_at_zero_sideslip=ours[0.0]["main"][1],
                                     dCY_dbeta=slope(ours, 1, "total"), dCY_dbeta_pylons=slope(ours, 1, "pylons"), dCY_dbeta_main=slope(ours, 1, "main"),
                                     dCl_dbeta=slope(ours, 3, "total"), dCn_dbeta=slope(ours, 5, "total"),
                                     dCn_dbeta_pylons=slope(ours, 5, "pylons"), dCn_dbeta_main=slope(ours, 5, "main")),
                           database=dict(CY_at_zero_sideslip=base[0.0][1], dCY_dbeta=slope(base, 1), dCl_dbeta=slope(base, 3), dCn_dbeta=slope(base, 5)),
                           controller_simple_model=SIMPLE_MODEL))
    judged = [row for row in rows if row["alpha_deg"] in drag.JUDGED_ALPHAS_DEG]
    worst = max(abs(row["difference"]["lift_to_drag_relative"]) for row in judged)
    return dict(rows=rows, sideslip=slopes, judged_alphas_deg=list(drag.JUDGED_ALPHAS_DEG), limit_relative=drag.RATIO_LIMIT,
                largest_relative_gap_of_the_ratio=worst,
                settings=dict(speed_m_s=S.SPEED, density_kg_m3=S.RHO, reference=S.MAKANI_REF, parasitic=drag.PARASITIC,
                              ours="main surfaces (profile-drag case) and pylon group (profile-drag case), separate solves with straight wakes, added",
                              axes="Makani body axes; moments about the body origin; lift and drag in wind axes; side force in body axes"),
                passed=bool(worst <= drag.RATIO_LIMIT))


# ---------------------------------------------------------------------------------------------------------------------
def provenance():
    out = drag.provenance()
    out.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               drag_checks_sha256=hashlib.sha256((TASK/"drag_checks.py").read_bytes()).hexdigest(),
               pylon_case_sha256=hashlib.sha256((REPO/PYLON_LOW_CASE).read_bytes()).hexdigest(),
               companion_tests_sha256=hashlib.sha256((REPO/"tests/test_companion.py").read_bytes()).hexdigest())
    import numba
    others = {"a3_pitch": REPO/"docs/tasks/OF-0007-g4motion20261002/a3_pitch.py", "a5_loop": REPO/"docs/tasks/OF-0007-g4motion20261002/a5_loop.py",
              "alpha_sweep": REPO/"docs/tasks/OF-0007-wingfull20261001/alpha_sweep.py", "test_free_flight": REPO/"tests/test_free_flight.py",
              "test_closed_loop": REPO/"tests/test_closed_loop.py", "b4_receipt": drag.B4_RECEIPT}
    out.update({f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in others.items()})
    out.update(numba_threads=int(numba.get_num_threads()))
    return out


def pylon_report(records, lead_in):
    """Force coefficients of the pylon group over the closed loop (Makani body axes), and whether every record has them."""
    closed = records[lead_in:]
    if not closed or any("pylons_force_body_N" not in record for record in closed):
        return dict(every_record_carries_the_pylon_loads=False)
    force = np.array([record["pylons_force_body_N"] for record in closed])*np.array([-1.0, 1.0, -1.0])
    force = force/(np.array([record["dynamic_pressure_Pa"] for record in closed])[:, None]*S.MAKANI_REF["S"])
    def counts(key):
        out = {}
        for record in closed:
            out[str(record[key])] = out.get(str(record[key]), 0)+1
        return out
    retries = [int(record["step"]) for record in closed if record.get("pylons_zero_seed_failure")]
    return dict(every_record_carries_the_pylon_loads=True,
                force_coefficients_makani_body_range={name: [float(force[:, k].min()), float(force[:, k].max())] for k, name in enumerate(("CX", "CY", "CZ"))},
                largest_iterations=int(max(record["pylons_iterations"] for record in closed)),
                steps_by_seeding=counts("pylons_seeding"), steps_by_closure=counts("pylons_closure"),
                steps_with_a_zero_seed_retry=len(retries), first_retry_steps=retries[:20],
                reading="a retry from the previous circulation after a failed zero-seeded solve is the hand-over rule's own, recorded "
                        "here; a failed solve of the group is one that stops the run (contract revision 2)",
                handover_weight_range=[float(min(record["pylons_handover_weight"] for record in closed)),
                                       float(max(record["pylons_handover_weight"] for record in closed))])


def main(out, argv):
    parser = argparse.ArgumentParser(prog="pylon_loop_checks.py")
    parser.add_argument("build")
    parser.add_argument("--parts", default="e1,e2,e3,e4,e5")
    parser.add_argument("--steps", type=int, default=None, help="closed-loop steps of e4 (default: 1.5 nominal periods); another length judges nothing")
    parser.add_argument("--e4-child", action="store_true")
    parser.add_argument("--e5-child", default=None)
    args = parser.parse_args(argv)
    lead_in, backend = checks.LEAD_IN, "tree"
    if args.e4_child:
        install_case()
        checks.b3_child(args.build, out, backend, lead_in, args.steps, False, "prep-trans-out")
        return
    if args.e5_child:
        install_case()
        checks.b4_child(args.build, args.e5_child, backend, lead_in)
        return
    out = Path(out)
    library = Path(args.build)/"lib"/"libmakani_controller.so"
    library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    started, tick = time.time(), time.perf_counter()
    at_start = provenance()
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    criteria, history = {}, {}

    def note(text):
        print(f"[{time.perf_counter()-tick:7.0f}s] {text}", flush=True)
    if "e1" in parts or "e2" in parts:
        e1, e2 = pinned(backend)
        if "e1" in parts:
            units = unit_tests()
            target = out/"e1-default-path-b4.json"
            done = subprocess.run([sys.executable, str(TASK/"closed_loop_checks.py"), str(out), args.build, "--b4-child", str(target),
                                   "--backend", backend, "--lead-in", str(lead_in)], capture_output=True, text=True)
            text = target.read_text(encoding="utf-8") if done.returncode == 0 and target.exists() else None
            recorded = json.loads(drag.B4_RECEIPT.read_text(encoding="utf-8"))["criteria"]["B4"]["sha256"]
            digest = None if text is None else hashlib.sha256(text.encode()).hexdigest()
            default_same = bool(digest is not None and recorded[0] == recorded[1] == digest)
            e1.update(unit_tests=units, default_path=dict(what="B4 of closed_loop_checks.py repeated untouched at this commit", sha256=digest,
                                                          recorded_sha256=recorded, bit_identical=default_same,
                                                          error=None if text is not None else done.stderr.strip()[-400:]))
            e1["passed"] = bool(e1["main_surfaces_as_their_own_companion_bit_identical"] and e1["main_records_untouched_by_the_companions"]
                                and all(u["passed"] for u in units.values()) and default_same)
            criteria["E1"] = e1
            note(f"E1: {e1['passed']} (twin {e1['main_surfaces_as_their_own_companion_bit_identical']}, main untouched "
                 f"{e1['main_records_untouched_by_the_companions']}, unit tests "
                 f"{ {k: v['passed'] for k, v in units.items()} }, default path {default_same})")
        if "e2" in parts:
            criteria["E2"] = e2
            note(f"E2: {e2['passed']} (force gap {e2['largest_relative_force_gap']}, moment gap {e2['largest_relative_moment_gap']}; "
                 f"as P2b: {e2['as_criterion_P2b_of_the_plant_contract']})")
    if "e3" in parts:
        criteria["E3"] = e3()
        note(f"E3: {criteria['E3']['passed']}; largest gap of the lift-to-drag ratio {criteria['E3']['largest_relative_gap_of_the_ratio']:.4f}")
        for row in criteria["E3"]["rows"]:
            note(f"   alpha {row['alpha_deg']:.0f}: CL {row['ours']['CL']:.3f} / {row['database']['CL_with_offsets']:.3f}  CD "
                 f"{row['ours']['CD_total']:.4f} / {row['database']['CD_with_offsets']:.4f}  L/D {row['ours']['lift_to_drag']:.2f} / "
                 f"{row['database']['lift_to_drag']:.2f} ({100*row['difference']['lift_to_drag_relative']:+.1f}%)  CY {row['ours']['CY_body']:.3f} / "
                 f"{row['database']['CY_body']:.3f}")
        for item in criteria["E3"]["sideslip"]:
            note(f"   alpha {item['alpha_deg']:.0f}: dCY/dbeta {item['ours']['dCY_dbeta']:.3f} (database {item['database']['dCY_dbeta']:.3f}), "
                 f"dCn/dbeta {item['ours']['dCn_dbeta']:.4f} ({item['database']['dCn_dbeta']:.4f}), dCl/dbeta {item['ours']['dCl_dbeta']:.4f} "
                 f"({item['database']['dCl_dbeta']:.4f})")
    if "e4" in parts:
        install_case()
        api = checks.load_api(args.build)
        case = checks.cl_base(args.steps, backend, lead_in)
        trial = args.steps is not None and args.steps != case.full_steps
        note(f"E4: CL1-DP, lead-in {lead_in}, closed loop {case.closed_steps} steps{' (not the contract length: judges nothing)' if trial else ''}")
        command = [sys.executable, str(Path(__file__).resolve()), str(out), args.build, "--e4-child"]+(["--steps", str(args.steps)] if args.steps is not None else [])
        child_tick = time.perf_counter()
        done = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        final = json.loads((out/"b3-final.json").read_text(encoding="utf-8")) if (out/"b3-final.json").exists() else None
        records, log = drag.lines(out/"b3-records.jsonl"), drag.lines(out/"b3-log.jsonl")
        summary = drag.summarise(api, case, final, records, log, lead_in, done.returncode, done.stderr)
        report = pylon_report(records, lead_in)
        summary.update(case="CL1-DP", seconds=time.perf_counter()-child_tick, shortened_trial=bool(trial), case_settings=case.settings,
                       pylons=report)
        summary["passed"] = bool(summary["passed"] and report["every_record_carries_the_pylon_loads"])
        if trial:
            summary.update(judged=False, would_have_passed=summary["passed"], passed=None)
        criteria["E4"] = summary
        history["CL1-DP"] = dict(records=records, controller_log=log)
        note(f"E4: status {summary['status']} {summary['error'] or ''} (failed module {summary.get('failed_module')}); lap "
             f"{summary.get('lap_completed')}; tracking {summary['tracking']}; pylons {report}; passed {summary['passed']}")
    if "e5" in parts:
        texts = []
        for index in range(2):
            target = out/f"e5-run{index}.json"
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(out), args.build, "--e5-child", str(target)],
                                  capture_output=True, text=True)
            texts.append(target.read_text(encoding="utf-8") if done.returncode == 0 and target.exists() else None)
            if done.returncode:
                note(f"E5 child {index} failed: {done.stderr.strip()[-400:]}")
        same = texts[0] is not None and texts[0] == texts[1]
        first = None if texts[0] is None else json.loads(texts[0])
        carried = bool(first and all("pylons_force_inertial_N" in record for record in first["records"]))
        criteria["E5"] = dict(steps=checks.REPEAT_STEPS, both_finished=bool(all(t is not None for t in texts)),
                              status=None if first is None else first["status"], bit_identical=bool(same),
                              every_record_carries_the_pylon_loads=carried,
                              sha256=[None if t is None else hashlib.sha256(t.encode()).hexdigest() for t in texts],
                              passed=bool(same and first["status"] == "complete" and carried))
        note(f"E5: {criteria['E5']['passed']}")
    end = provenance()
    keys = ("src_tree", "cases_tree", "script_sha256", "closed_loop_checks_sha256", "drag_checks_sha256", "drag_case_sha256",
            "pylon_case_sha256", "generator_sha256", "tests_sha256", "companion_tests_sha256", "a3_pitch_sha256", "a5_loop_sha256",
            "alpha_sweep_sha256", "test_free_flight_sha256", "test_closed_loop_sha256", "b4_receipt_sha256")
    result = dict(what="G5 step 3: the pylons as a vortex system of their own in the closed loop (pylon contract, E1-E5)", parts=parts,
                  criteria=criteria, library_sha256=library_sha256,
                  binding_sha256=hashlib.sha256((Path(args.build)/"lib"/"makani_controller_api.py").read_bytes()).hexdigest(),
                  elapsed_s=time.perf_counter()-tick, started_unix=started, at_start=at_start,
                  code_unchanged_during_the_run=bool(all(at_start[k] == end[k] for k in keys) and at_start["src_clean"] and end["src_clean"]),
                  **end)
    text = json.dumps(dict(result, history=history), default=str)+"\n"
    (out/"history.json").write_text(text, encoding="utf-8")
    result["history_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out/"checks.json").write_text(json.dumps(result, indent=1, default=str)+"\n", encoding="utf-8")
    print(json.dumps({name: value.get("passed") for name, value in criteria.items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
