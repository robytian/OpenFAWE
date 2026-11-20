# SPDX-License-Identifier: Apache-2.0
# Transcribed from google/makani commit 1d66923 (Apache-2.0): lib/python/batch_sim/scoring_functions/__init__.py
# (FlightModeFilter lines 306-370, _FilteredOutput, PayoutFilter lines 446-513, DurationScoringFunction lines
# 733-760, the limit semantics of SingleSidedLimitScoringFunction and DoubleSidedLimitScoringFunction),
# scoring_functions_util.py (GetTimeSamp lines 610-622, LpFiltFiltTimeSeries lines 625-653, GetIntervals lines
# 540-562), tether.py (TensionMinScoringFunction, TetherHeightAglMinScoringFunction), rotors.py
# (MaxRotorSpeedsScoringFunction, MaxRotorThrustScoringFunction, ThrustMomentSaturationDurationScoringFunction),
# hover.py (HoverTensionControlScoringFunction, PerchedPositionScoringFunction); the limits of
# analysis/crosswind_batch_sims/batch_sim_params.py. Modifications (2026): the functions read the per-period records
# of ``ground.turnkey`` (one record per controller period) instead of Makani's simulator and controller telemetry.
"""Makani's batch-simulation scores, applied to the records of a TurnKey run (full-cycle contract, revision 6, P4, P7).

The selection follows ``FlightModeFilter``: the periods whose flight mode is one of the listed modes, sorted in time;
for HoverDescend only the periods with the hover gain ramp above 0.95. Values are those of the scores' ``GetOutput``;
a selection with no period gives None (Makani gives no score; P4 counts it as not passed). Pass/fail uses the scores'
own semantics: a one-sided score fails at or beyond its bad limit (``>= bad`` for an upper limit, ``<= bad`` for a
lower one), the duration score fails at or outside its bad limits.

Differences from Makani's inputs, declared: the rotor speeds are the speeds the plant holds over each period (no motor
model); the tension at the bridle point is the plant's truth (Makani's ``wing.tether_force_b.tension``); everything
else (hover telemetry, thrust-moment command) is the controller's own telemetry as Makani records it.
"""

import math

import numpy as np

DESCEND = "HoverDescend"


def get_time_samp(time):
    """GetTimeSamp: the most common sampling interval, rounded to 1 ms; NaN below ten samples."""
    time = np.asarray(time, dtype=float)
    if len(time) < 10:
        return float("nan")
    values, counts = np.unique(np.around(np.diff(time), decimals=3), return_counts=True)
    return float(values[np.argmax(counts)])            # scipy.stats.mode: the smallest of the most common values


def lp_filtfilt(time, series, f_cutoff):
    """LpFiltFiltTimeSeries: a first-order Butterworth low pass run forwards and backwards (filtfilt without padding)
    on every continuous piece of the series (a gap is a step above 1.5 sampling intervals)."""
    from scipy import signal
    time, series = np.asarray(time, dtype=float), np.asarray(series, dtype=float)
    t_samp = get_time_samp(time)
    if math.isnan(t_samp):
        return np.array([float("nan")])
    pieces = np.split(series, np.where(np.diff(time) > 1.5*t_samp)[0] + 1)
    b, a = signal.butter(1, f_cutoff/(0.5/t_samp), "lowpass")
    return np.concatenate([signal.filtfilt(b, a, piece, padtype=None) for piece in pieces])


def get_intervals(indices):
    """GetIntervals: (first, last) of every run of consecutive indices."""
    indices = np.asarray(indices).reshape(-1)
    intervals = []
    if indices.size:
        begin = indices[0]
        for i in range(len(indices) - 1):
            if indices[i + 1] > indices[i] + 1:
                intervals.append((int(begin), int(indices[i])))
                begin = indices[i + 1]
        intervals.append((int(begin), int(indices[-1])))
    return intervals


MODE_KEY = "controller_flight_mode"


def select(records, names, modes, *, descend_gain=0.95, key=MODE_KEY):
    """Indices of the records in ``modes`` (names without ``kFlightMode``), as ``FlightModeFilter`` with a zero time
    threshold; ``names`` maps the mode numbers to names. The mode is the controller telemetry's ``flight_mode``, the
    mode the controller ran in that period (written by PlannerStep before it switches), as Makani's scores read it."""
    wanted = set(modes)
    out = []
    for i, r in enumerate(records):
        name = names.get(r[key])
        if name not in wanted:
            continue
        if name == DESCEND and not r["gain_ramp_scale"] > descend_gain:
            continue
        out.append(i)
    return out


def max_rotor_speed(records, idx):
    return None if not idx else float(max(max(abs(w) for w in records[i]["rotor_omega_rad_s"]) for i in idx))


def max_rotor_thrust(records, idx):
    return None if not idx else float(max(max(abs(f) for f in records[i]["rotor_thrust_N"]) for i in idx))


def min_tension_kn(records, idx, f_cutoff=4.0):
    """TensionMinScoringFunction: minimum of the 4 Hz zero-phase low-passed bridle-point tension [kN]."""
    if not idx:
        return None
    time = [records[i]["t"] for i in idx]
    tension = [records[i]["tension_bridle_N"]*1e-3 for i in idx]
    filtered = lp_filtfilt(time, tension, f_cutoff)
    return float(np.min(filtered))


def duration(records, idx):
    """DurationScoringFunction: last minus first time of the selection (NaN with two samples or fewer)."""
    if not idx:
        return None
    return float(records[idx[-1]]["t"] - records[idx[0]]["t"]) if len(idx) > 2 else float("nan")


def tension_control_rms_kn(records, idx, min_payout_m=7.0):
    """HoverTensionControlScoringFunction behind PayoutFilter(min 7 m): RMS of (tension command - tension) [kN] over
    the selection from its first to its last period with a pay-out of at least ``min_payout_m``."""
    if not idx:
        return None
    payout = np.array([records[i]["payout_m"] for i in idx])
    keep = np.flatnonzero(payout >= min_payout_m)
    if not keep.size:
        return None
    chosen = idx[keep[0]:keep[-1] + 1]
    error = np.array([records[i]["tension_cmd_N"] - records[i]["est_tension_N"] for i in chosen])
    return float(np.sqrt(np.mean(error**2))/1e3)


def saturation_duration_s(records, idx, axis, hover_limits):
    """ThrustMomentSaturationDurationScoringFunction (``axis`` 'thrust', 'moment_y' or 'moment_z'); ``hover_limits``:
    dict(max_thrust, min_moment_y, max_moment_y, min_moment_z, max_moment_z) of the hover parameters."""
    if not idx:
        return None
    column = {"thrust": 0, "moment_y": 2, "moment_z": 3}[axis]
    cmd = np.array([records[i]["thrust_moment"][column] for i in idx])
    avail = np.array([records[i]["thrust_moment_avail"][column] for i in idx])
    if axis == "thrust":
        low, high = -math.inf, hover_limits["max_thrust"]
    else:
        low, high = hover_limits[f"min_{axis}"], hover_limits[f"max_{axis}"]
    t_samp = get_time_samp([records[i]["t"] for i in idx])
    if math.isnan(t_samp):
        return float("nan")
    mask = np.flatnonzero((np.abs(cmd - avail) > 1e-2) | (np.abs(cmd - low) < 1e-2) | (np.abs(cmd - high) < 1e-2))
    if not mask.size:
        return 0.0
    return float(max(b - a for a, b in get_intervals(mask))*t_samp)


def perched_position_error_m(records, names, perched_wing_pos_p):
    """PerchedPositionScoringFunction (onshore: the vessel frame is the ground frame and its origin the buoy's): the
    horizontal distance in the platform frame between the wing's position at the first HoverDescend period with the
    gain ramp below 1e-8 and ``perched_wing_pos_p``."""
    for r in records:
        if names.get(r[MODE_KEY]) == DESCEND and r["gain_ramp_scale"] < 1e-8:
            c, s = math.cos(r["platform_azi"]), math.sin(r["platform_azi"])
            dcm_v2p = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])      # AngleToDcm(azi, 0, 0, 'ZYX')
            pos_p = dcm_v2p@np.asarray(r["Xg"], dtype=float)
            error = pos_p - np.asarray(perched_wing_pos_p, dtype=float)
            return float(math.hypot(error[0], error[1]))
    return None


def upper_passes(value, bad):
    """One-sided upper score: passes below the bad limit; no value does not pass."""
    return value is not None and not math.isnan(value) and value < bad


def lower_passes(value, bad):
    return value is not None and not math.isnan(value) and value > bad


def range_passes(value, bad_low, bad_high):
    return value is not None and not math.isnan(value) and bad_low < value < bad_high


HOVER_OUT = ("HoverAscend", "HoverPayOut", "HoverPrepTransformGsUp", "HoverTransformGsUp", "HoverFullLength", "HoverAccel")
HOVER_BACK = ("HoverTransOut", "HoverPrepTransformGsDown", "HoverTransformGsDown", "HoverReelIn", "HoverDescend")
# P4: (label, kind, modes, bad limit(s), good limit(s)) with batch_sim_params.py line numbers.
P4_SCORES = (
    ("max rotor speed, perch to crosswind (756-761)", "speed", HOVER_OUT, 225.0, 215.0),
    ("max rotor speed, crosswind to perch (762-768)", "speed", HOVER_BACK, 225.0, 215.0),
    ("max rotor speed, TransIn (1013-1015)", "speed", ("TransIn",), 220.0, 210.0),
    ("min tension, perch to transform (830-833)", "tension", ("HoverPayOut", "HoverPrepTransformGsUp", "HoverTransformGsUp"),
     1.0, 2.0),
    ("min tension, transform to perch (834-837)", "tension", ("HoverTransformGsDown", "HoverReelIn"), 1.0, 2.0),
    ("min tension, HoverDescend (920-922)", "tension", ("HoverDescend",), 1.0, 2.0),
    ("min tension, HoverAscend (924-926)", "tension", ("HoverAscend",), 1.0, 2.0),
    ("max rotor thrust, HoverAccel (1021-1023)", "thrust", ("HoverAccel",), 4400.0, 3800.0),
    ("max rotor thrust, TransIn (1024-1026)", "thrust", ("TransIn",), 4400.0, 3800.0),
    ("duration, HoverAccel (936-939)", "duration", ("HoverAccel",), (1.5, 15.0), (3.0, 10.0)),
    ("duration, TransIn (941-945)", "duration", ("TransIn",), (5.0, 25.0), (6.0, 20.0)),
)


def p4(records, names):
    """Every P4 score: value, bad and good limits, whether it passes (P4 passes when all do)."""
    rows = []
    for label, kind, modes, bad, good in P4_SCORES:
        idx = select(records, names, modes)
        known = None
        if kind in ("speed", "thrust"):
            value = max_rotor_speed(records, idx) if kind == "speed" else max_rotor_thrust(records, idx)
            ok = upper_passes(value, bad)
            known = value is not None and not math.isnan(value)
            good_ok = known and value <= good                  # Makani: a score of zero or less at or below the good limit
        elif kind == "tension":
            value = min_tension_kn(records, idx)
            ok = lower_passes(value, bad)
            known = value is not None and not math.isnan(value)
            good_ok = known and value >= good
        else:
            value = duration(records, idx)
            ok = range_passes(value, *bad)
            known = value is not None and not math.isnan(value)
            good_ok = known and good[0] <= value <= good[1]
        rows.append(dict(label=label, modes=list(modes), periods=len(idx), value=value, bad=bad, good=good, passed=bool(ok),
                         within_good=bool(good_ok)))
    return dict(passed=all(row["passed"] for row in rows), scores=rows)


def wiring_checks(records, settings):
    """K3 read back from the records (review P3C-05): periods contiguous from 0; the rotor speeds of period k are the
    gated command of period k-1 reached from the speeds of period k-1 at 500 rad/s^2; the thrust coefficient used in
    period k is the mean coefficient of the rotor loads of period k-1 (30 m/s floor); the flap angles of period k are
    the servo reference model's state after the command of period k-2 moved it over period k-1. Returns a list of
    disagreements (empty: the records follow K3)."""
    from openfawe.flight.makani_interface import ServoReference, rate_limited_speeds
    problems = []
    density, areas = float(settings["density_kg_m3"]), settings["rotor_area_m2"]
    servo, flap_command = ServoReference(np.zeros(8)), np.zeros(8)
    omegas, command, coefficient = np.zeros(len(areas)), np.zeros(len(areas)), 0.0
    for i, r in enumerate(records):
        if r["k"] != i:
            problems.append(f"record {i} is period {r['k']}")
            break
        expected = rate_limited_speeds(command, omegas, 0.01)
        if not np.array_equal(np.asarray(r["rotor_omega_rad_s"]), expected):
            problems.append(f"period {i}: rotor speeds {r['rotor_omega_rad_s']} are not {expected.tolist()}")
        if not math.isclose(r["thrust_coeff_used"], coefficient, rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f"period {i}: thrust coefficient {r['thrust_coeff_used']!r} is not {coefficient!r}")
        if not np.array_equal(np.asarray(r["flaps_acting_rad"]), servo.flaps_rad):
            problems.append(f"period {i}: flap angles are not the servo model's state")
        omegas = np.asarray(r["rotor_omega_rad_s"], dtype=float)
        command = (np.asarray(r["rotor_command_rad_s"], dtype=float) if r["run_motors"] and not r["stop_motors"]
                   else np.zeros(len(areas)))
        coefficient = float(np.mean([t/(0.5*density*a*max(abs(v), 30.0)**2) for t, v, a in
                                     zip(r["rotor_thrust_N"], r["rotor_axial_freestream_m_s"], areas)]))
        servo.step(flap_command, 0.01)
        flap_command = np.asarray(r["flaps_command_rad"], dtype=float)
        if len(problems) > 20:
            problems.append("more disagreements not listed")
            break
    return problems
