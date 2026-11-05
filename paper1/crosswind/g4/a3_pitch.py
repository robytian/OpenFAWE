"""Cases A3 and A4 of the G4 contract: pitch manoeuvre at constant airspeed, 90 deg -> 3 deg -> 90 deg.

A3 is the manoeuvre without rotors (the default); A4 is the same manoeuvre with the eight rotors at a constant
angular speed (``--rotors OMEGA``; see "Case A4" below).

Motion (contract section 4 and its revision of 2026-10-03, second entry): the M600 wing and tails fly
through still air at a constant speed (25 or 10 m/s) while the incidence at the body origin goes from
90 deg to 3 deg and back; each move is a rest-to-rest trapezoid in pitch rate whose corner times fall on
time steps (``openfawe.flight.kinematics.pitch_manoeuvre``). With the base step 0.02 s: hold 0.5 s,
pitch down 3.54 s (ramps 0.5 s, constant rate 2.54 s), hold 1.0 s, pitch up 3.54 s, hold 0.5 s, i.e.
454 steps; peak rate 0.4995 rad/s and acceleration 0.9990 rad/s^2, inside the limits of source S6
(0.5 rad/s, 1.0 rad/s^2). ``--refine n`` divides the time step by n and keeps the same motion (K5).

Aerodynamics: VSM + free wake with the section hand-over (``openfawe.aero.handover``): the Makani
simulator weight from the incidence and sideslip at the body origin; weight 0 -> crosswind section data
with the production closure, weight > 0 -> blended section data with the take-off/landing setting of
decision 0025. Two seedings of the solve on the hover side exist: ``--seeding zero`` (zero, then one recorded
retry from the previous circulation), which the author chose on 2026-10-03 (decision 0028), and
``--seeding previous``, kept for the earlier comparison runs. Wake: explicit Euler, constant core 0.15 m, tree backend
theta 0.2 / order 2 / leaf 32 unless another backend is asked for. No tether or fuselage; no rotors unless
``--rotors`` is given (case A4); no pylons in the default assembly (two trial assemblies with pylons are described below).

The default assembly is wing and tails without pylons (the "first pass" of the contract's section 8, entries of
2026-10-03). On its own it does not close K4: K4 is judged on the pair of this run and the run of the pylon
assembly, with zero seeding, by ``a3_k4_pair.py`` (contract section 8, entry of 2026-10-03 21:53).

Reported (thresholds are those frozen in the contract; none is set here):
  K3  self-consistency of the motion samples (<= 1e-6);
  K4  every step solved: whether the run stopped at a failed step (it stops at the first) and how many
      steps were not computed, the retries with the error of each zero-seeded attempt, the number of
      steps with more than four stall switches on the main wing (reported, no threshold). A verdict is
      given only for the contract settings (25 or 10 m/s, tree backend, base time step, full length);
      a failed run with those settings is "failed" also when it was shortened;
  K5  (report only) impulses of the body-axis force and of the pitching moment over the whole run, for
      comparison between ``--refine 1`` and ``--refine 2``;
  the share of steps with weight > 0 and inside the blend, the steps with incidence steps above 6 deg,
  the load change across the steps at which the closure switches (weight 0 <-> weight > 0) against the
  neighbouring steps, and the loads of the pitch-up against the pitch-down at the same incidence
  (mirrored samples).
A failed step stops the run and is recorded; nothing is smoothed or replaced.

Two further assemblies use the cases with pylons (contract section 8, entries of 2026-10-03). Neither gets a
K4 verdict of its own.
  ``--assembly pylons``   the four pylons alone, as an assembly of their own (no mutual induction with the wing
                          and the tails): the pylon model chosen by the author (decision 0028). K4 is judged on the
                          PAIR of this run and the default run at one commit, by ``a3_k4_pair.py``, which also adds
                          the loads.
  ``--assembly coupled``  wing, tails and the four pylons solved together as one assembly: a trial of the model
                          that was not chosen.
The motion has no sideslip, so these runs show whether a pylon model runs and what it adds to the loads; the side
force is in the step records (``force_body_N``), not in the tables of the closure switches and of the
pitch-down against pitch-up comparison, which hold CX, CZ and Cm only.
The default, ``--assembly main``, is unchanged: wing and tails, the cases without pylons.

Case A4 (``--rotors OMEGA``; contract section 8, entry of 2026-10-04): the eight rotor discs of
``cases/rotors/m600_rotors_makani.json`` with the Rev4 table are carried by the body at the constant angular speed
OMEGA [rad/s]; each releases one vortex ring per step (``openfawe.flight.prescribed.PrescribedRotors``). The rings
induce on the lifting surfaces with the core ``--ring-surface-core`` (contract value 0.3 m) and convect with the
wake core. ``--ring-phase-deg`` turns the ring polygons about the disc axes (0 in the contract; a trial option).
What "A4 with pylons" is under the pylon model of decision 0028 is NOT decided (contract section 8, 2026-10-04,
item 4: with the author), so no K6 rule for a set of runs exists yet and this script gives no K6 "passed": a run
with the A4 settings of the contract in either assembly (``main`` or ``pylons``, each releasing its own rings) is
"failed" when a step failed and "not evaluated" otherwise. ``contract_settings_a4`` in the receipt is true for
these settings in either assembly: 25 or 10 m/s, tree backend, base time step, zero seeding, 120 rad/s, 16 nodes,
surface core 0.3 m, phase 0. A run with rotors never gets a K4 verdict. The loads of lifting surfaces crossed by
ring filaments are not quantitative (rotor contract, section 7 item 5): K6 judges only whether every step has a
solution. The rotor loads are those of the axial table, also where the stream meets the discs edge-on (at 90 deg
incidence the axial part of 25 m/s is 1.3 m/s) or from behind: a look-up below the table's lowest freestream uses
that value and is counted, and momentum theory is applied to the axial inflow as it is. The rotor moment holds no
gyroscopic or reaction moment. The records keep the rotor loads, the axial freestream of every hub, the clamped
table look-ups and the number of ring nodes inside the surface core of a load point (a lower bound on filament
crossings: nodes only, load points only).

``--out-of-range hover`` (trial of option (a) of author item 乙, G4 README, A4 trial read-back): inside the blend a panel whose
incidence leaves the crosswind table gets the hover data alone instead of raising; the receipts list those panels per step.
Not a contract setting: such a run gets no K4 and no K6 verdict. Together with ``--weight-basis panel`` "inside the blend"
reads "on a step with some panel weight above zero, for a panel of weight below one": a panel of weight zero outside the
table is then moved to the hover data as well, a jump of the whole difference between the two tables for that panel.

``--weight-basis panel`` (trial of option (c) of the same author item; decision 0029, contract section 8, 2026-10-06
21:23): every panel takes its own hand-over weight from the flow that reaches it from outside the lifting surfaces' own
circulation system - ambient wind minus the panel's velocity plus the induction of the rotor rings, in body axes - instead
of the one weight of the body origin (``openfawe.aero.handover``, ``openfawe.flight.prescribed.panel_handover_step``). The
records keep the weight of the origin (``handover_weight``) and the panel weights; the bookkeeping counts the steps on
which the two bases differ. Not a contract setting either: no K4 and no K6 verdict.

Usage (direct):      a3_pitch.py OUT_DIR [--speed 25] [--seeding previous|zero] [--out-of-range error|hover] [--weight-basis origin|panel]
                                         [--refine 1] [--steps N]
                                         [--backend tree|numba|numpy] [--assembly main|coupled|pylons]
                                         [--rotors OMEGA [--ring-surface-core 0.3] [--ring-nodes 16] [--ring-phase-deg 0]]
Usage (PC job line): NAME --script docs/tasks/OF-0007-g4motion20261002/a3_pitch.py --speed 25 --seeding zero
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
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero.aircraft import PanelPolars, WingAssembly, load_aircraft  # noqa: E402
from openfawe.aero.handover import CROSSWIND_SOLVE, SectionHandover  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.flight.kinematics import pitch_manoeuvre  # noqa: E402
from openfawe.flight.prescribed import PrescribedRotors, run_prescribed  # noqa: E402
from openfawe.rotor.actuator_disc import load_rotor_set  # noqa: E402
from openfawe.rotor.table import RotorTable  # noqa: E402

LOW_CASE = "cases/qualification/m600_makani_reconstructed.json"
HIGH_CASE = "cases/qualification/m600_makani_xfoil_polars.json"
LOW_PYLON_CASE = "cases/qualification/m600_makani_reconstructed_pylons.json"
HIGH_PYLON_CASE = "cases/qualification/m600_makani_xfoil_polars_pylons.json"
ROTOR_SET = "cases/rotors/m600_rotors_makani.json"
ROTOR_TABLE = "cases/rotors/makani/rotor_rev4.json"
A4_OMEGA_RAD_S, A4_RING_NODES, A4_SURFACE_CORE_M = 120.0, 16, 0.3     # contract section 8, 2026-10-04
AREA_M2, CHORD_M = 32.9285, 1.2831          # reference area and chord of the archived M600 runs
BASE_DT_S = 0.02
HOLD_START, RAMP, CRUISE, HOLD_LOW, HOLD_END = 25, 25, 127, 50, 25     # in base time steps
HIGH_DEG, LOW_DEG = 90.0, 3.0
K3_LIMIT, SWITCH_LIMIT = 1e-6, 4
PASS = ("wing and tails without pylons: the first pass, or the wing-and-tail half of the K4 pair; this run's own verdict is "
        "not the K4 of G4, which is given only by a3_k4_pair.py on the pair (contract section 8, 2026-10-03)")
TRIALS = dict(
    coupled=("trial of the wing, the tails and the four pylons solved together (not the model chosen by the author: "
             "decision 0028); no K4 verdict"),
    pylons=("the four pylons as an assembly of their own (the author's choice, decision 0028); K4 is judged on the "
            "pair with the wing-and-tail run by a3_k4_pair.py (contract section 8, 2026-10-03 21:53), not on this run alone"))
SCOPES = dict(
    main="wing and tail aerodynamic loads only: no rotors, pylons, tether, bridle or fuselage",
    coupled="aerodynamic loads of the wing, the tails and the four pylons solved together: no rotors, tether, bridle or fuselage",
    pylons="the four pylons alone, as an assembly of their own without mutual induction with the wing and the tails")
TABLE_DEG = (90.0, 75.0, 60.0, 45.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.0, 5.0, 3.0)


def schedule(refine):
    """Time step, number of steps and the two moves; sample indices of the phases."""
    dt = BASE_DT_S/refine
    ramp, cruise = RAMP*BASE_DT_S, CRUISE*BASE_DT_S
    down = HOLD_START*BASE_DT_S
    up = down+2*ramp+cruise+HOLD_LOW*BASE_DT_S
    move = (2*RAMP+CRUISE)*refine
    first = HOLD_START*refine
    second = first+move+HOLD_LOW*refine
    steps = second+move+HOLD_END*refine
    moves = [(down, LOW_DEG-HIGH_DEG, ramp, cruise), (up, HIGH_DEG-LOW_DEG, ramp, cruise)]
    return dt, steps, moves, dict(down=(first, first+move), low=(first+move, second), up=(second, second+move))


def consistency_verdict(report):
    """K3 from a ``RigidMotion.consistency`` report: worst relative entry against the frozen limit."""
    entries = {k: v for k, v in report.items() if isinstance(v, dict)}
    relative = {k: v["relative"] for k, v in entries.items() if v["relative"] is not None and v["compared"] > 0}
    worst = max(relative.values()) if relative else None
    return dict(limit=K3_LIMIT, worst_relative=worst, passed=None if worst is None else bool(worst <= K3_LIMIT),
                entries_in_verdict=sorted(relative),
                zero_scale_entries={k: v["max_abs"] for k, v in entries.items() if v["relative"] is None},
                report=report)


def coefficients(record, q):
    force = np.asarray(record["force_body_N"])/(q*AREA_M2)
    moment = np.asarray(record["moment_origin_body_Nm"])/(q*AREA_M2*CHORD_M)
    return force, moment


def phase_ranges(phases, last_step):
    """Inclusive sample ranges that put every sample in exactly one phase.

    A move's first sample is still at rest (it belongs to the hold before it); its last sample belongs to the move.
    """
    (d0, d1), (u0, u1) = phases["down"], phases["up"]
    return dict(hold_high_start=(0, d0), down=(d0+1, d1), hold_low=(d1+1, u0), up=(u0+1, u1), hold_high_end=(u1+1, last_step))


def bookkeeping(records, phases, last_step):
    """Counts the contract asks for; all taken from the step records."""
    weight = np.array([r["handover_weight"] for r in records])
    solved = np.array([r["stall_switches"] is not None for r in records], dtype=bool)
    switches = np.array([0 if r["stall_switches"] is None else r["stall_switches"] for r in records])
    rough = np.array([r["incidence_steps"]["total_steps_over_limit"] for r in records])
    seeding = [r["seeding"] for r in records]
    retries = [dict(step=r["step"], alpha_deg=r["alpha_deg"], handover_weight=r["handover_weight"], **(r["zero_seed_failure"] or {}))
               for r in records if r["seeding"] == "retry_previous"]
    out = dict(steps_recorded=len(records),
               seeding_counts={name: seeding.count(name) for name in sorted(set(seeding))},
               retries=len(retries), retry_steps=retries,
               retry_error_types={name: [r.get("type") for r in retries].count(name) for name in sorted({str(r.get("type")) for r in retries})},
               steps_weight_positive=int(np.sum(weight > 0.)), steps_inside_blend=int(np.sum((weight > 0.) & (weight < 1.))),
               share_weight_positive=float(np.mean(weight > 0.)) if len(records) else None,
               steps_with_stall_switches_over_limit=int(np.sum(switches > SWITCH_LIMIT)), stall_switch_limit=SWITCH_LIMIT,
               largest_stall_switches=int(switches[solved].max()) if solved.any() else None,
               steps_with_incidence_steps_over_6deg=int(np.sum(rough > 0)),
               largest_iterations=int(max(r["iterations"] for r in records)) if records else None)
    outside = [r.get("most_panels_outside_crosswind_table") for r in records]
    if any(v is not None for v in outside):                   # the out_of_range='hover' trial option was active in the blend
        counts = np.array([v or 0 for v in outside])
        out["panels_outside_crosswind_table"] = dict(
            steps_with_any=int(np.sum(counts > 0)), largest_count=int(counts.max()),
            first_step=int(np.argmax(counts > 0)) if np.any(counts > 0) else None,
            steps_in_the_blend=int(sum(v is not None for v in outside)),
            note="counts are the largest over the evaluations of a step (line-search trials and a zero-seed attempt included); "
                 "the per-step list in the records is the set at the converged circulation")
    per_panel = [r.get("panel_weights") for r in records]
    if any(v is not None for v in per_panel):                 # the weight_basis='panel' trial option
        largest = np.array([max(v) for v in per_panel])
        smallest = np.array([min(v) for v in per_panel])
        blend = np.array([r["panels_by_weight"]["blend"] for r in records])
        differs = np.array([any(x != w for x in v) for v, w in zip(per_panel, weight)])
        lifted = (weight == 0.) & (largest > 0.)
        full = largest > 0.                                   # steps on the full-incidence closure
        toggles = np.flatnonzero(full[1:] != full[:-1])+1
        out["panel_weights"] = dict(
            closure_switches=int(len(toggles)), closure_switch_steps=[int(k) for k in toggles],
            first_step_on_the_full_incidence_closure=int(np.argmax(full)) if full.any() else None,
            last_step_on_the_full_incidence_closure=int(len(full)-1-np.argmax(full[::-1])) if full.any() else None,
            origin_weight_counts_are="steps_weight_positive, steps_inside_blend and share_weight_positive above count the weight of "
                                     "the body origin; with this basis they do not say which model a step used - the counts here do",
            steps_where_some_panel_differs_from_origin=int(differs.sum()),
            first_such_step=int(np.argmax(differs)) if differs.any() else None,
            steps_origin_zero_but_some_panel_positive=int(lifted.sum()),
            first_step_origin_zero_but_some_panel_positive=int(np.argmax(lifted)) if lifted.any() else None,
            steps_with_panels_in_the_blend=int(np.sum(blend > 0)), most_panels_in_the_blend=int(blend.max()),
            steps_on_the_full_incidence_closure=int(np.sum(largest > 0.)),
            steps_all_panels_on_crosswind_data=int(np.sum(largest == 0.)),
            steps_all_panels_on_hover_data=int(np.sum(smallest == 1.)),
            note="handover_weight is the weight of the body origin; a step uses the production crosswind model only when "
                 "every panel weight is zero")
    by_phase = {}
    for name, (a, b) in phase_ranges(phases, last_step).items():
        if a >= len(records) or b < a:
            continue
        b = min(b, len(records)-1)
        part = slice(a, b+1)
        by_phase[name] = dict(first_step=a, last_step=b, steps=b-a+1,
                              steps_with_stall_switches_over_limit=int(np.sum(switches[part] > SWITCH_LIMIT)),
                              steps_with_incidence_steps_over_6deg=int(np.sum(rough[part] > 0)),
                              retries=seeding[part].count("retry_previous"))
    out["by_phase"] = by_phase
    return out


def on_crosswind_side(record):
    """Whether a step used the production crosswind model.

    Origin basis: the weight of the origin is zero (the expression the receipts have always used). Panel basis
    (the record carries ``panel_weights``): every panel weight is zero - there the weight of the origin does not
    say which model a step used.
    """
    per_panel = record.get("panel_weights")
    if per_panel is None:
        return record["handover_weight"] == 0.
    return max(per_panel) == 0.


def closure_switches(records, q):
    """Load change across each step at which the closure switches, against the changes of the steps around it.

    The two models differ at weight zero (source closure with the crosswind data against the full-incidence
    closure with the same data), so a step in the loads is expected there; its size is reported, not judged.
    Which side a step is on: ``on_crosswind_side``. ``weight_before`` and ``weight_after`` are the weights of the
    body origin (with the panel basis they can both be zero at a switch).
    """
    rows = []
    force = np.array([coefficients(r, q)[0] for r in records])
    moment = np.array([coefficients(r, q)[1] for r in records])
    low = [on_crosswind_side(r) for r in records]
    for k in range(1, len(records)):
        before, after = records[k-1]["handover_weight"], records[k]["handover_weight"]
        if low[k-1] == low[k]:
            continue
        near = [j for j in (k-2, k-1, k+1, k+2) if 1 <= j < len(records) and low[j-1] == low[j]]
        rows.append(dict(step=k, alpha_before_deg=records[k-1]["alpha_deg"], alpha_after_deg=records[k]["alpha_deg"],
                         weight_before=before, weight_after=after,
                         force_coefficient_change=float(np.linalg.norm(force[k]-force[k-1])),
                         moment_coefficient_change=float(np.linalg.norm(moment[k]-moment[k-1])),
                         neighbouring_force_changes=[float(np.linalg.norm(force[j]-force[j-1])) for j in near],
                         neighbouring_moment_changes=[float(np.linalg.norm(moment[j]-moment[j-1])) for j in near],
                         CZ_before=float(force[k-1, 2]), CZ_after=float(force[k, 2])))
    return rows


def k4_verdict(status, failed_step, steps, full_steps, frames, counts, contract_settings, assembly="main"):
    """K4 for one run; a verdict only for the contract settings, and never more than "first pass".

    The other assemblies never get a verdict of their own: the one-solve model is a trial of the model the author
    did not choose, and the pylons alone are half of the pair K4 is judged on (``a3_k4_pair.py``). The counts are
    recorded all the same.
    """
    out = dict(limit_failed_steps=0, failed=status != "complete", run_stopped_at_failed_step=failed_step,
               steps_not_computed=steps+1-frames, retries=counts["retries"],
               steps_with_stall_switches_over_limit=counts["steps_with_stall_switches_over_limit"],
               scope=PASS if assembly == "main" else TRIALS[assembly])
    if assembly == "pylons":
        out.update(verdict="not evaluated", reason="the pylons alone are not the model K4 is defined for; K4 is judged on the pair")
    elif assembly != "main":
        out.update(verdict="not evaluated", reason="trial of the pylon model that was not chosen")
    elif not contract_settings:
        out.update(verdict="not evaluated", reason="settings differ from contract case A3 (speed, backend, time step or a hand-over "
                                                   "trial option)")
    elif status != "complete":
        out.update(verdict="failed", reason=f"a step failed (step {failed_step}); {PASS}")
    elif steps != full_steps:
        out.update(verdict="not evaluated", reason="shortened run without a failure")
    else:
        out.update(verdict="passed", reason=f"every step solved; {PASS}")
    return out


K6_PENDING = ("which runs K6 of G4 is judged on when the pylons are an assembly of their own is not decided "
              "(contract section 8, 2026-10-04, item 4); this script gives no K6 'passed'")


def k6_verdict(status, failed_step, failed_module, steps, frames, contract_settings):
    """K6 fields of one run of case A4: "failed" for a failed step under the A4 settings, never "passed"."""
    out = dict(limit_failed_steps=0, failed=status != "complete", run_stopped_at_failed_step=failed_step,
               failed_module=failed_module, steps_not_computed=steps+1-frames,
               scope="one assembly with the eight rotors; "+K6_PENDING)
    if not contract_settings:
        out.update(verdict="not evaluated", reason="settings differ from contract case A4"
                   + (f"; a step failed (step {failed_step}, {failed_module})" if status != "complete" else ""))
    elif status != "complete":
        out.update(verdict="failed", reason=f"a step failed (step {failed_step}, {failed_module})")
    else:
        out.update(verdict="not evaluated", reason=K6_PENDING)
    return out


def rotor_bookkeeping(records, dt):
    """What the rotors did over the run; all taken from the step records (report only)."""
    if not records:
        return None
    thrust = np.array([r["rotor_thrust_N"] for r in records])
    axial = np.array([r["rotor_axial_freestream_m_s"] for r in records])
    shaft = np.array([r["rotor_shaft_power_W"] for r in records])
    near = np.array([r["ring_nodes_within_surface_core"] for r in records])
    clamped = np.array([r["rotor_table_saturated"] for r in records], dtype=bool)
    force = np.array([r["rotor_force_body_N"] for r in records])
    return dict(thrust_N=dict(smallest=float(thrust.min()), largest=float(thrust.max()), first_step=thrust[0].tolist(), last_step=thrust[-1].tolist()),
                axial_freestream_m_s=dict(smallest=float(axial.min()), largest=float(axial.max())),
                steps_with_negative_thrust=int(np.sum(np.any(thrust < 0., axis=1))),
                steps_with_clamped_table_look_up=int(np.sum(np.any(clamped, axis=1))),
                shaft_power_W=dict(smallest=float(shaft.min()), largest=float(shaft.max())),
                shaft_energy_J=float(dt*(shaft[1:]+shaft[:-1]).sum()/2) if len(records) > 1 else None,
                rotor_force_impulse_body_Ns=(dt*(force[1:]+force[:-1]).sum(axis=0)/2).tolist() if len(records) > 1 else None,
                ring_nodes_at_the_end=int(records[-1]["ring_nodes"]),
                steps_with_ring_nodes_within_surface_core=int(np.sum(near > 0)), largest_ring_nodes_within_surface_core=int(near.max()),
                first_step_with_ring_nodes_within_surface_core=int(np.argmax(near > 0)) if np.any(near > 0) else None)


def impulses(records, dt):
    """Trapezoidal time integrals of the body-axis force and moment about the body origin (K5, report only)."""
    force = np.array([r["force_body_N"] for r in records])
    moment = np.array([r["moment_origin_body_Nm"] for r in records])
    if len(records) < 2:
        return None
    return dict(force_body_Ns=(dt*(force[1:]+force[:-1]).sum(axis=0)/2).tolist(),
                moment_origin_body_Nms=(dt*(moment[1:]+moment[:-1]).sum(axis=0)/2).tolist(),
                duration_s=dt*(len(records)-1))


def in_against_out(records, phases, q):
    """Pitch-up against pitch-down at the same incidence: sample ``down0+k`` against ``up1-k``."""
    (d0, d1), (u0, u1) = phases["down"], phases["up"]
    if len(records) <= u1:
        return None
    rows, worst = [], dict(force=0., moment=0.)
    scale_f = max(np.linalg.norm(coefficients(records[k], q)[0]) for k in range(d0, d1+1))
    scale_m = max(np.linalg.norm(coefficients(records[k], q)[1]) for k in range(d0, d1+1))
    differences = []
    for k in range(d1-d0+1):
        down, up = records[d0+k], records[u1-k]
        (fd, md), (fu, mu) = coefficients(down, q), coefficients(up, q)
        differences.append((down["alpha_deg"], np.linalg.norm(fu-fd)/scale_f, np.linalg.norm(mu-md)/scale_m))
    differences = np.array(differences)
    for target in TABLE_DEG:
        k = int(np.argmin(np.abs(differences[:, 0]-target)))
        down, up = records[d0+k], records[u1-k]
        (fd, md), (fu, mu) = coefficients(down, q), coefficients(up, q)
        rows.append(dict(alpha_deg=down["alpha_deg"], alpha_up_deg=up["alpha_deg"], weight=down["handover_weight"],   # of the body origin
                         down=dict(CX=fd[0], CZ=fd[2], Cm=md[1], stall_switches=down["stall_switches"], seeding=down["seeding"]),
                         up=dict(CX=fu[0], CZ=fu[2], Cm=mu[1], stall_switches=up["stall_switches"], seeding=up["seeding"])))
    worst = dict(force_max=float(differences[:, 1].max()), force_mean=float(differences[:, 1].mean()),
                 moment_max=float(differences[:, 2].max()), moment_mean=float(differences[:, 2].mean()),
                 alpha_of_force_max_deg=float(differences[int(np.argmax(differences[:, 1])), 0]),
                 normalised_by="largest coefficient magnitude during the pitch-down",
                 note="the pitch rate has opposite sign in the two moves and the wake differs; a difference is expected, "
                      "its size is the result")
    return dict(table=rows, difference=worst)


def pylons_only(aircraft):
    """The pylon surfaces of a case with pylons as an assembly of their own, with their panel polars."""
    keep = [k for k, name in enumerate(aircraft.wing.names) if name.startswith("Pylon")]
    if not keep:
        raise SystemExit("the case has no pylon surfaces")
    start = sum(w.panels for w in aircraft.wing.wings[:keep[0]])
    assembly = WingAssembly(tuple(aircraft.wing.wings[k] for k in keep), tuple(aircraft.wing.names[k] for k in keep))
    return assembly, PanelPolars(aircraft.polar.polars[start:start+assembly.panels])


def provenance(low_case=LOW_CASE, high_case=HIGH_CASE):
    def git(*args):
        try:
            done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)
        except Exception:  # noqa: BLE001 - provenance only
            return None
        return done.stdout.strip() if done.returncode == 0 else None
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain", "--", "src")
    import numba
    return dict(git_commit=commit, src_clean=None if commit is None or status is None else status == "",
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                low_case_sha256=hashlib.sha256((REPO/low_case).read_bytes()).hexdigest(),
                high_case_sha256=hashlib.sha256((REPO/high_case).read_bytes()).hexdigest(),
                threads=int(numba.get_num_threads()), threading_layer_requested=str(numba.config.THREADING_LAYER),
                versions=dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__),
                platform=platform.platform())


def main(out, argv):
    parser = argparse.ArgumentParser(prog="a3_pitch.py")
    parser.add_argument("--speed", type=float, default=25.0)
    parser.add_argument("--seeding", default="previous", choices=("previous", "zero"))
    parser.add_argument("--out-of-range", default="error", choices=("error", "hover"),
                        help="inside the blend (with --weight-basis panel: on a step with some panel weight above zero), a panel that "
                             "uses the crosswind table and whose incidence leaves it: error (contract) or the hover data alone "
                             "(option (a) of author item 乙, a trial; such a run gets no K4/K6 verdict)")
    parser.add_argument("--weight-basis", default="origin", choices=("origin", "panel"),
                        help="hand-over weight from the body origin (contract) or per panel from its own oncoming flow "
                             "(option (c) of author item 乙, a trial; such a run gets no K4/K6 verdict)")
    parser.add_argument("--refine", type=int, default=1, help="divide the base time step 0.02 s by this integer")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps (smoke runs)")
    parser.add_argument("--backend", default="tree", choices=("tree", "numba", "numpy"))
    parser.add_argument("--assembly", default="main", choices=("main", "coupled", "pylons"),
                        help="main: wing and tails (cases without pylons); coupled: with the pylons, one assembly (trial); "
                             "pylons: the four pylons alone (half of the K4 pair)")
    parser.add_argument("--rotors", type=float, default=None, metavar="OMEGA",
                        help="case A4: the eight rotors at this constant angular speed [rad/s]")
    parser.add_argument("--ring-surface-core", type=float, default=A4_SURFACE_CORE_M,
                        help="core of the rotor rings when they induce at lifting-surface points [m]")
    parser.add_argument("--ring-nodes", type=int, default=A4_RING_NODES, help="nodes of each rotor ring")
    parser.add_argument("--ring-phase-deg", type=float, default=0.0, help="turn the ring polygons about the disc axes [deg]")
    parser.add_argument("--record-panel-onset", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                        help="diagnostic with --weight-basis panel: on these steps record the flow each panel's weight was taken "
                             "from, with and without the rings, and the rings' induction by disc (changes no step)")
    args = parser.parse_args(argv)
    if args.refine < 1:
        raise SystemExit("--refine must be a positive integer")
    if args.rotors is not None and args.assembly == "coupled":
        raise SystemExit("case A4 is defined for the assemblies of decision 0028 (main, pylons), not for the one-solve trial")
    if args.rotors is None and (args.ring_surface_core != A4_SURFACE_CORE_M or args.ring_nodes != A4_RING_NODES or args.ring_phase_deg != 0.0):
        raise SystemExit("the ring options need --rotors")
    if args.record_panel_onset is not None and args.weight_basis != "panel":
        raise SystemExit("--record-panel-onset needs --weight-basis panel")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dt, full_steps, moves, phases = schedule(args.refine)
    steps = full_steps if args.steps is None else min(args.steps, full_steps)
    low_case, high_case = (LOW_CASE, HIGH_CASE) if args.assembly == "main" else (LOW_PYLON_CASE, HIGH_PYLON_CASE)
    low, high = load_aircraft(REPO/low_case), load_aircraft(REPO/high_case)
    if not (np.array_equal(high.wing.cp, low.wing.cp) and np.array_equal(high.wing.dl, low.wing.dl)):
        raise SystemExit("the two cases do not share one geometry")
    if args.assembly == "pylons":
        (body, low_polar), (_, high_polar) = pylons_only(low), pylons_only(high)
    else:
        body, low_polar, high_polar = low.wing, low.polar, high.polar
    scope = SCOPES[args.assembly]
    rotors = rotor_settings = None
    if args.rotors is not None:
        rotors = PrescribedRotors(load_rotor_set(REPO/ROTOR_SET, RotorTable.from_makani_json(REPO/ROTOR_TABLE)), args.rotors,
                                  nodes_per_ring=args.ring_nodes, surface_core_m=args.ring_surface_core,
                                  ring_phase_rad=float(np.radians(args.ring_phase_deg)))
        rotor_settings = dict(omega_rad_s=args.rotors, discs=len(rotors.discs), nodes_per_ring=rotors.nodes_per_ring,
                              surface_core_m=rotors.surface_core_m, turbine_extension=rotors.turbine_extension,
                              ring_phase_deg=args.ring_phase_deg,
                              not_in_the_rotor_loads="gyroscopic and reaction moments; edgewise and reverse inflow use the axial table",
                              rotor_set=ROTOR_SET, rotor_set_sha256=hashlib.sha256((REPO/ROTOR_SET).read_bytes()).hexdigest(),
                              table=ROTOR_TABLE, table_sha256=hashlib.sha256((REPO/ROTOR_TABLE).read_bytes()).hexdigest())
        scope = (f"{'wing and tail' if args.assembly == 'main' else 'pylon'} aerodynamic loads with the eight rotor discs and their "
                 "ring wakes (this assembly releases its own rings); no tether, bridle or fuselage; loads of surfaces crossed by "
                 "ring filaments are not quantitative")
    handover = SectionHandover(low_polar, high_polar, seeding=args.seeding, out_of_range=args.out_of_range,
                               weight_basis=args.weight_basis)
    options = dict(velocity_backend=args.backend)
    if args.backend == "tree":
        options["tree"] = TreeSettings(theta=0.2, order=2, leaf_size=32)
    config = WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=0.15, convection_integrator="euler", core_evolution="constant",
                        solve=CROSSWIND_SOLVE, **options)
    backend = AeroBackend("vsm_free_wake", wake=config)
    motion = pitch_manoeuvre(np.arange(full_steps+1)*dt, args.speed, HIGH_DEG, moves)
    q = 0.5*config.density_kg_m3*args.speed**2
    base_settings = bool(args.speed in (25.0, 10.0) and args.backend == "tree" and args.refine == 1 and args.out_of_range == "error"
                         and args.weight_basis == "origin")
    contract_settings = bool(base_settings and args.assembly == "main" and rotors is None)   # case A3: set for the main assembly only
    # Case A4: the settings of the contract entry of 2026-10-04, in either assembly of decision 0028 (the one-solve
    # trial is refused above). Zero seeding is part of them.
    contract_a4 = bool(base_settings and rotors is not None and args.seeding == "zero" and args.rotors == A4_OMEGA_RAD_S
                       and args.ring_nodes == A4_RING_NODES and args.ring_surface_core == A4_SURFACE_CORE_M
                       and args.ring_phase_deg == 0.0)
    names = list(body.names)
    settings = dict(low_case=low_case, high_case=high_case, assembly=args.assembly, surfaces=names,
                    # "stall_switches" and "main_panels_not_attached" of the step records count panels of this surface
                    # (the driver takes the first surface of the assembly): the main wing, or Pylon 1 in a pylon trial.
                    switch_count_surface=names[0],
                    pylon_panels=None if args.assembly == "main" else int(low.wing.wings[-1].panels),
                    speed_m_s=args.speed, seeding=args.seeding, out_of_range=args.out_of_range, weight_basis=args.weight_basis,
                    refine=args.refine, dt_s=dt,
                    steps=steps, full_steps=full_steps, phases={k: list(v) for k, v in phases.items()},
                    moves=[dict(start_s=m[0], change_deg=m[1], ramp_s=m[2], cruise_s=m[3]) for m in moves],
                    peak_rate_rad_s=float(np.radians(HIGH_DEG-LOW_DEG)/((RAMP+CRUISE)*BASE_DT_S)),
                    peak_acceleration_rad_s2=float(np.radians(HIGH_DEG-LOW_DEG)/((RAMP+CRUISE)*BASE_DT_S)/(RAMP*BASE_DT_S)),
                    backend=args.backend, tree=dict(theta=0.2, order=2, leaf_size=32) if args.backend == "tree" else None,
                    core_radius_m=0.15, density_kg_m3=config.density_kg_m3, dynamic_pressure_Pa=q,
                    reference_area_m2=AREA_M2, reference_chord_m=CHORD_M,
                    scope=scope,
                    contract_settings_a3=contract_settings, contract_settings_a4=contract_a4, rotors=rotor_settings,
                    full_length=bool(steps == full_steps),
                    **{"pass": ("case A4, one assembly with the eight rotors; "+K6_PENDING if rotors is not None
                                else PASS if args.assembly == "main" else TRIALS[args.assembly])},
                    handover=handover.metadata(), aerodynamics=backend.metadata(), **provenance(low_case, high_case))
    started, tick = time.time(), time.perf_counter()

    def progress(record):
        if record["step"] % 25 == 0 or record["step"] == steps:
            rotor = (f"thrust {min(record['rotor_thrust_N']):.0f}..{max(record['rotor_thrust_N']):.0f}N "
                     f"near={record['ring_nodes_within_surface_core']} " if "rotor_thrust_N" in record else "")
            print(f"step {record['step']}/{steps} alpha={record['alpha_deg']:.1f} w={record['handover_weight']:.2f} "
                  f"seed={record['seeding']} switches={record['stall_switches']} it={record['iterations']} "
                  f"Fz_body={record['force_body_N'][2]:.1f}N {rotor}elapsed {time.perf_counter()-tick:.0f}s", flush=True)

    if args.record_panel_onset is not None:               # the key exists only in receipts of diagnostic runs
        settings["panel_onset_steps"] = [int(k) for k in args.record_panel_onset]
    run = run_prescribed(backend, body, None, motion, np.zeros(3), handover=handover, rotors=rotors, steps=steps, progress=progress,
                         panel_onset_steps=args.record_panel_onset)
    k3 = consistency_verdict(run.motion_consistency)
    counts = bookkeeping(run.records, phases, steps)
    k4 = k4_verdict(run.status, run.failed_step, steps, full_steps, len(run.records), counts, contract_settings, args.assembly)
    if rotors is not None:
        k4.update(verdict="not evaluated", reason="a run with rotors is case A4, not A3", scope="case A4 (with rotors): no K4")
    report = dict(settings=settings, status=run.status, error=run.error, error_type=run.error_type, failed_step=run.failed_step,
                  failed_module=run.failed_module, failed_step_aerodynamics=run.failed_step_aerodynamics,
                  failed_panels_outside_crosswind_table=run.failed_panels_outside_crosswind_table,
                  failed_panel_weights=run.failed_panel_weights,
                  failed_iterations=run.failed_iterations, zero_seed_failure_at_failed_step=run.zero_seed_failure,
                  frames=len(run.records), elapsed_s=time.perf_counter()-tick,
                  started_unix=started, K3=k3, K4=k4,
                  K6=None if rotors is None else k6_verdict(run.status, run.failed_step, run.failed_module, steps, len(run.records), contract_a4),
                  rotor_bookkeeping=None if rotors is None else rotor_bookkeeping(run.records, dt),
                  bookkeeping=counts, K5_impulses=impulses(run.records, dt),
                  closure_switches=closure_switches(run.records, q),
                  pitch_up_against_pitch_down=in_against_out(run.records, phases, q),
                  records=run.records, step_times_s=run.step_seconds)
    (out/"history.json").write_text(json.dumps(report, indent=1, default=str)+"\n", encoding="utf-8")
    print(json.dumps(dict(out=str(out), status=run.status, error=run.error, failed_step=run.failed_step, frames=len(run.records),
                          elapsed_s=round(report["elapsed_s"], 1), K3=k3["worst_relative"], K4=k4["verdict"],
                          K6=None if rotors is None else report["K6"]["verdict"], failed_module=run.failed_module,
                          retries=counts["retries"], switches_over_limit=counts["steps_with_stall_switches_over_limit"],
                          rough_steps=counts["steps_with_incidence_steps_over_6deg"])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
