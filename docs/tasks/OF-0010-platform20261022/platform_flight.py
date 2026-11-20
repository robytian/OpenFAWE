"""OF-0010 revision 8: the turnkey flight on the platform path - P1, P2, the strip seed (甲'') and P5c's viscosity, the
hook's segment 'whole+orth+p2sv' (CPU tree code) or 'whole+orth+gpu+p2sv' (node 2's 'cuda_tree') - as a runner job.
A trial: it judges nothing (revision 8 says what it reports).

Windows (``--window``):
  hover      the real flight with the segment applied, to ``--periods`` periods (``run_turnkey(max_periods=...)``);
  crosswind  the same flight with tables until the first CrosswindNormal period P0, from P0 the segment computed as a
             shadow (loads not applied; plan 3.8a's shadow, ``whole_vortex_timing.make_shadow_class``) to P0 + ``--periods``;
  whole      the real flight with the segment applied to its end (no period limit).
The flight is part 3's (blown wing off, no power system), as P5a-U6 and P5c-U8. A period's wall time is the perf_counter
difference between consecutive record callbacks; periods over 600 s (a quiet-hours suspension on node 2) are reported
and left out of the means. Every record goes to ``records.jsonl.gz`` in the job folder (it stays on the machine); the
job's history.json (under 2 MB, so node 2's runner returns it) holds the end, the periods by flight mode and by source,
the fallbacks, the viscosity's and the stall counts' summaries, the wall times (every period's for a window, every
tenth's for the whole flight) and, for the windows, the means by mode class and the projection of revision 8.

Usage (a runner job line):
  NAME --script docs/tasks/OF-0010-platform20261022/platform_flight.py --build local/build-makani-win-full
       --segment whole+orth+gpu+p2sv --window hover|crosswind|whole [--periods N]
       [--fallback-ladder 0.1,0.03,0.01,0.003:5000]   (diagnosis only: the platform groups' Picard fallback relaxations and
                                                      iterations in place of TAKEOFF_LANDING_SOLVE's (0.1, 0.03) and 2000)
       [--keep-rows K]   (OF-0010 revision 10: the hook's ``keep_rows``, the kept rows of the lifting surfaces' wake)
       [--free-rows N]   (OF-0010 revision 22 addendum 3, B3 H4: the hook's ``free_rows``, only the N newest wake rows
                          convected with the induced velocity, older rows with the ambient velocity; without it nothing changes)
       [--frozen-drift rings] (revision 22 addendum 4, B3 H5, with --free-rows: the older rows also take the rotor rings' induction)
       [--stall-lag K]   (revision 22 addendum 8, B3 H6: the hook's ``stall_lag`` K, the Oye-type lag of the stalled sections)
       [--power on|jit]  (OF-0010 revision 10: the power system of plan 3.9, ``turnkey_checks.power_system_config``;
                         'jit' (OF-0014 revision 2): the same system integrated on the compiled path, bit for bit)
       [--hover-tables on] [--blown-wing tables]   (OF-0010 revision 11, B: the hook's ``hover_tables``; Makani's
                         blown-wing term only where the database acts, ``run_turnkey(blown_wing='tables')``)
  window hover-shadow (revision 10, WK-H): the flight with tables until the first HoverAscend period H0, from H0 the
  segment computed as a shadow to H0 + ``--periods`` (as 'crosswind' from P0).
  [--checkpoint-every N]  (OF-0010 revision 12: save the run's state every N periods outside the job folder and, when the
                          job is run again after a break, resume from it - ``flight_resume.py``; without it nothing changes)
  window whole with --periods N (revision 12 addendum 1, R2a): the whole flight's settings to N periods.
  [--gpu-precision fp64|mixed] [--gpu-control cpu|gpu]  (GPU whole-step contract revision 5, PM-C; a GPU segment): the
                          cuda_tree precision (default 'mixed', suspect pairs 'gpu64') and where the VSM free wake's
                          induction at the control points is summed (default 'cpu'); without them nothing changes.
  [--gpu-moments cpu|gpu]  (GPU whole-step contract revision 6, S3'; a GPU segment): where the tree's node moments are
                          computed (default 'cpu'); without it nothing changes.
  [--gpu-build cpu|gpu]   (revision 7, S3''; a GPU segment): where the tree is built (the same tree either way; default
                          'cpu'); without it nothing changes.
  [--floating norway|norway-waves]  (OF-0011 revision 6 addendum 77, F3c): the offshore flight on Makani's buoy - the
                          Norway parameters, the wing serial 05 (plant and hook), Makani's buoy under GS02 (still water, or
                          Makani's sea with the C++ export's phases); give the offshore library with --build. Not with a
                          wind shear, turbulence or --wind-speed. Without it nothing changes.
  [--floating oc3-spar|oc3-spar-regular|oc3-spar-jonswap|oc3-spar-fixed]  (OF-0011 revision 7 addendum 1, F3d): F3b's OC3
                          spar (HydroDyn + MoorDyn-C) under GS02 - still water, regular or JONSWAP waves (Makani's sea:
                          1.5 m, 6.5 s, WaveDir 160 deg) - or the fixed foundation (the deck held at the spar's static
                          height); Norway parameters and wing serial 05 as 'norway'; the spar values are not resumable
                          (no --checkpoint-every). 'land-fixed' (the check V2 only): land parameters, a frozen base.
  [--iec CASE [--iec-start T]] (OF-0011 revision 7 addendum 6 supplement 1, P2): an IEC 61400-1 deterministic wind
                          (``ground.iec_wind.IecEventWind``; event from T s, default 10) in place of the uniform wind; not
                          with a shear or turbulence. With an F3d spar value of --floating the shear, turbulence,
                          --wind-speed and --iec are taken, the field's ground at the spar's deck.
  [--record-ring-inflow on] (OF-0011 revision 7 addendum 11, C1-R7): the rotor rings' induced velocity on the wing and
                          tails, mean and largest by surface, into each record (read only).
  [--return-period P]   (OF-0011 revision 7 addendum 20): the joystick's ReturnToPerch throttle from controller period P
                          (``run_turnkey(stop_rules=dict(throttle_return_period=P))``, as ``f3d.py tables --return-period``);
                          with the whole or the hover window.
  [--tether-events depth] (OF-0011 revision 7 addendum 4 supplement 7): the tether and perch events located within each
                          step, the contacts' modes held, as ``f3d.py tables --contact-depth`` (``run_turnkey(tether_events=
                          "depth")``); without it the step is as before.
  [--balance on]          (OF-0011 revision 7 addendum 14, C1-V3; an F3d spar value of --floating only): the spar's
                          ``balance`` - each record gets ``base_balance`` (f3d.make_base(balance=True), addendum 10).
  [--crosswind-params land]  (OF-0011 revision 7 addendum 2, with a Norway --floating value): the land library's whole
                          crosswind parameter block copied into the offshore library before the controller starts
                          (``f3d.swap_crosswind``); passed to the child as the floating value with '+landcw'.
  [--wake-frames-every N] (OF-0010 revision 27, a B' segment): a wake frame (the fields of the wake states, float32 and
                          int32) every N periods solved by the vortex method, in the job folder's wake-frames/; without it
                          nothing changes.
  [--jacobian active]     (GPU whole-step contract revision 9, J): the P2 groups' Newton solve with the analytic
                          ('active_segment') Jacobian; without it nothing changes.
  [--gpu-rings cpu|gpu]   (revision 10, R; a GPU segment): where the rotor rings' induction at the lifting-surface points
                          is summed (default 'cpu'); without it nothing changes.
  [--device auto|cpu|gpu] (revision 10, A): pick the engine from the machine. --segment names the CPU segment (no 'gpu' in
                          it) and no --gpu-* option is given; 'auto' takes the GPU variant of the segment with
                          AUTO_GPU_OPTIONS when a CUDA device is usable, else the segment as named; 'gpu' the GPU variant
                          (and fails without a device), 'cpu' the segment as named. The chosen options are what the run
                          uses (fingerprint, history); history adds ``device``. Judged runs name the engine explicitly.
"""
import gzip
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
G5 = REPO/"docs/tasks/OF-0008-g5control20261006"
FLOATING = dict(norway=REPO/"cases/ground/makani_gs02_norway.json", buoy=REPO/"cases/floating/makani_norway_params.json",
                serial05=REPO/"cases/aircraft/m600_makani_rigid_body_serial05.json",          # OF-0011 rev 6 add 77 (F3c)
                components=REPO/"docs/tasks/OF-0011-floating20261105/results/f1a-reference-46b05132/components-norway.json")
F3D_SPAR = ("oc3-spar", "oc3-spar-regular", "oc3-spar-jonswap")                       # OF-0011 rev 7 add 1 (F3d)
F3D_FLOATING = F3D_SPAR + ("oc3-spar-fixed", "land-fixed")
LAND_CROSSWIND = "+landcw"                                                                 # OF-0011 rev 7 add 2
SEGMENTS = {"whole+orth+p2sv": dict(gpu=False), "whole+orth+gpu+p2sv": dict(gpu=True),
            # diagnosis only (revision 8 addendum 1): the same path without the viscosity, and with the zero seed
            "whole+orth+p2s": dict(gpu=False, seeding="strip", viscosity=False), "whole+orth+gpu+p2s": dict(gpu=True, seeding="strip", viscosity=False),
            "whole+orth+p2": dict(gpu=False, seeding="zero", viscosity=False), "whole+orth+gpu+p2": dict(gpu=True, seeding="zero", viscosity=False),
            # OF-0010 revision 9 (P5d): the strip seed with the local strip fallback, no viscosity
            "whole+orth+p2sl": dict(gpu=False, seeding="strip", viscosity=False, local_strip=True),
            "whole+orth+gpu+p2sl": dict(gpu=True, seeding="strip", viscosity=False, local_strip=True)}
# OF-0010 revision 15 with addendum 1 (B'): B's segment with the slipstream T3 (t3), model D (md) or no rings (nr), and the
# induced (i) or kinematic (k) hub inflow; B' is 'whole+orth+gpu+p2sl+t3i'. Kept apart from SEGMENTS (whose entries the
# earlier steps' tests list); every lookup of a job's segment goes through ALL_SEGMENTS.
B_PRIME_SEGMENTS = {f"{_base}+{_code}{_hub}": dict(SEGMENTS[_base], slipstream=_slip, hub_inflow=_inflow)
                    for _base in ("whole+orth+p2sl", "whole+orth+gpu+p2sl")
                    for _code, _slip in (("t3", "t3"), ("t34", "t3_4s"), ("md", "model_d"), ("nr", "none"))
                    for _hub, _inflow in (("i", "induced"), ("k", "kinematic"))}
# OF-0011 revision 7 addendum 12 (C1-R6): B' on the CPU with the quasi-steady lifting surfaces (straight wake, no history)
QUASI_STEADY_SEGMENTS = {"whole+orth+p2sl+t3i+qs": dict(B_PRIME_SEGMENTS["whole+orth+p2sl+t3i"], quasi_steady=True)}
ALL_SEGMENTS = {**SEGMENTS, **B_PRIME_SEGMENTS, **QUASI_STEADY_SEGMENTS}
B_PRIME_NODES_EVERY = 100                               # revision 15 item 4: the airborne tether nodes every 100 periods
LONG_PERIOD_S = 600.0
TABLE_MODES = ("PilotHover", "Perched")
CROSSWIND_MODES = ("TransIn", "CrosswindNormal", "CrosswindPrepTransOut")
WEIGHTS = dict(table=1293, hover=93677, crosswind=52273)     # part 3's periods per mode class (whole_vortex_timing.py)
KEEP = ("k", "flight_mode", "whole_mode", "whole_model", "whole_shadow_source", "load_source", "whole_main_iterations",
        "whole_main_viscosity_on", "whole_main_viscosity_active", "whole_main_viscosity_mu_max",
        "whole_main_wing_panels_stalled_branch", "whole_main_wing_stall_switches", "whole_main_seeding",
        "whole_main_local_strip_rounds", "whole_main_local_strip_panels", "whole_main_local_strip_panel_regime",
        "whole_main_local_strip_share", "whole_main_local_strip_weighted_panels",
        "platform_local_strip", "whole_main_force_body_N", "whole_main_moment_origin_body_Nm",   # revision 10 (WK)
        "tension_bridle_N", "free_length", "origin_height_m")                                       # revision 11 addendum 2
ENERGY_TERMS = (("aero_applied_J", "power_aero_applied_J"), ("motor_electrical_J", "power_electrical_J"),
                ("motor_mechanical_J", "power_mechanical_J"), ("ground_J", "power_ground_energy_J"),
                ("tether_loss_J", "power_tether_loss_J"), ("resistive_J", "power_resistive_J"), ("speed_loss_J", "power_speed_loss_J"),
                ("hysteresis_J", "power_hysteresis_J"), ("controller_J", "power_controller_J"),
                ("conduction_J", "power_conduction_J"))   # revision 11 addendum 2: plan 3.9's account (power_energy_checks)
WHOLE_SERIES = ("whole_model", "cum_ground_J", "cum_motor_electrical_J", "tension_bridle_N", "free_length", "origin_height_m")


def energy_account(account, cumulative, r, names):
    """Revision 11 addendum 2: add one period's power-system energies to the per-mode account (the terms and signs of plan
    3.9's ``power_energy_checks.check_power_records``: positive ground energy is delivered to the ground; the shaft power
    of the rotor table over the period as ``table_shaft_J``) and return the running ground and motor-electrical sums.
    Records without the power system leave the account untouched (None, None)."""
    if "power_ground_energy_J" not in r:
        return None, None
    mode = names.get(r.get("controller_flight_mode"), str(r.get("controller_flight_mode")))
    a = account.setdefault(mode, dict(periods=0, table_shaft_J=0.0, wing_J=0.0, **{key: 0.0 for key, _ in ENERGY_TERMS},
                                      sources={}))
    a["periods"] += 1
    a["table_shaft_J"] += sum(r.get("rotor_power_W") or [])*0.01
    for key, field in ENERGY_TERMS:
        value = r.get(field)
        a[key] += float(sum(value) if isinstance(value, list) else (value or 0.0))
    a["wing_J"] += float(r["power_ground_energy_J"]) + float(r.get("power_tether_loss_J") or 0.0)   # V_wing I = V_g I + I^2 R
    source = str(r.get("whole_model") or r.get("load_source"))
    a["sources"][source] = a["sources"].get(source, 0)+1
    cumulative["ground"] += float(r["power_ground_energy_J"])
    electrical = r.get("power_electrical_J")
    cumulative["motor"] += float(sum(electrical) if isinstance(electrical, list) else (electrical or 0.0))
    return cumulative["ground"], cumulative["motor"]


def energy_report(account):
    """The whole flight's sums of the per-mode account, its duration and the net ground energy and mean power."""
    if not account:
        return None
    keys = [k for k in next(iter(account.values())) if k not in ("periods", "sources")]
    whole = {key: sum(a[key] for a in account.values()) for key in keys}
    duration = sum(a["periods"] for a in account.values())*0.01
    return dict(by_mode=account, whole=whole, duration_s=duration, net_ground_energy_J=whole["ground_J"],
                net_mean_ground_power_W=whole["ground_J"]/duration if duration else None,
                signs="as plan 3.9 (power_energy_checks): positive ground energy V_g I is delivered to the ground",
                note="from the perch (period 0) to the flight's last period; the drum's mechanical work and Makani's factor-1.5 "
                     "term of plan 3.9 are not summed here")


SHADOW_SERIES = ("whole_shadow_source", "platform_local_strip", "whole_main_local_strip_rounds", "whole_main_local_strip_panels",
                 "whole_main_local_strip_panel_regime", "whole_main_local_strip_share", "whole_main_force_body_N",
                 "whole_main_moment_origin_body_Nm")     # revision 10: node 2 returns history.json, not the records


SHADOW_WINDOWS = dict(crosswind="CrosswindNormal", hover_shadow="HoverAscend")     # the mode whose first period starts the shadow


GPU_OPTIONS = ("--gpu-precision", "--gpu-control", "--gpu-moments", "--gpu-build", "--gpu-rings")   # revisions 5-7, 10
AUTO_GPU_OPTIONS = ("--gpu-control", "gpu", "--gpu-moments", "gpu", "--gpu-build", "gpu")   # revision 10 (A): paper 3's
#                                                  whole-step GPU; '--gpu-rings gpu' is added only after R3 passes


def gpu_variant(segment):
    """The GPU segment of a CPU segment ('gpu' after 'orth'), or None when there is none."""
    parts = segment.split("+")
    if "gpu" in parts or "orth" not in parts:
        return None
    name = "+".join(parts[:parts.index("orth")+1] + ["gpu"] + parts[parts.index("orth")+1:])
    return name if name in ALL_SEGMENTS and ALL_SEGMENTS[name]["gpu"] else None


def resolve_device(args, available=None):
    """Whole-step revision 10 (A): the explicit options ``--device`` stands for. Returns (args, device): without
    ``--device`` the args unchanged and None; else the args without it, with the segment and the GPU options chosen, and
    the record of the choice. ``available`` (tests) replaces the probe of a usable CUDA device."""
    if "--device" not in args:
        return args, None
    asked = args["--device"]
    out = {k: v for k, v in args.items() if k != "--device"}
    if available is None:
        from openfawe.aero.vortex import cuda_backend_available, cuda_unavailable_reason
        available = bool(cuda_backend_available())
        reason = "a CUDA device is usable" if available else f"no usable CUDA device ({cuda_unavailable_reason()})"
    else:
        reason = "given (test)"
    if asked == "gpu" and not available:
        raise SystemExit(f"--device gpu: {reason}")
    chosen = "gpu" if asked == "gpu" or (asked == "auto" and available) else "cpu"
    if chosen == "gpu":
        out["--segment"] = gpu_variant(args["--segment"])
        out.update(zip(AUTO_GPU_OPTIONS[::2], AUTO_GPU_OPTIONS[1::2]))
    return out, dict(asked=asked, chosen=chosen, cuda_available=available, reason=reason, segment=out["--segment"],
                     options=[x for k in GPU_OPTIONS if k in out for x in (k, out[k])])


def parse_args(argv):
    """The job line's options (``main``): a dict, or SystemExit with the usage."""
    args = dict(zip(argv[::2], argv[1::2]))
    known = {"--build", "--segment", "--window", "--periods", "--fallback-ladder", "--keep-rows", "--power", "--hover-tables",
             "--blown-wing", "--checkpoint-every", "--crosswind-periods", "--wind-shear-exponent", "--turbulence",
             "--turbulence-t0", "--free-rows", "--frozen-drift", "--stall-lag", "--wind-speed", "--wake-frames-every",
             "--gpu-precision", "--gpu-control", "--gpu-moments", "--gpu-build", "--floating", "--jacobian",
             "--gpu-rings", "--device", "--crosswind-params", "--iec", "--iec-start", "--record-ring-inflow",
             "--balance", "--tether-events", "--return-period"}
    if len(argv) % 2 or not set(args) <= known or not {"--build", "--segment", "--window"} <= set(args):
        raise SystemExit("usage: --build PATH --segment SEGMENT --window hover|crosswind|hover-shadow|whole [--periods N] "
                         "[--fallback-ladder R,...:N] [--keep-rows K] [--power on]")
    segment, window = args["--segment"], args["--window"]
    if segment not in ALL_SEGMENTS or window not in ("hover", "crosswind", "hover-shadow", "whole") or \
            (window != "whole" and "--periods" not in args):
        raise SystemExit(f"segment one of {sorted(ALL_SEGMENTS)}; window hover, crosswind or hover-shadow with --periods, whole "
                         "with or without")
    if "--periods" in args and (not args["--periods"].isdigit() or int(args["--periods"]) < 1):
        raise SystemExit("--periods takes a positive integer")
    if "--checkpoint-every" in args and (not args["--checkpoint-every"].isdigit() or int(args["--checkpoint-every"]) < 1):
        raise SystemExit("--checkpoint-every takes a positive integer (revision 12)")
    if "--crosswind-periods" in args and (window != "whole" or not args["--crosswind-periods"].isdigit()
                                          or int(args["--crosswind-periods"]) < 1):
        raise SystemExit("--crosswind-periods takes a positive integer, with the whole window (revision 15 addendum 1)")
    if "--wind-shear-exponent" in args:                # revision 19 addendum 6 (B1)
        try:
            exponent = float(args["--wind-shear-exponent"])
        except ValueError:
            exponent = float("nan")
        if not (math.isfinite(exponent) and exponent >= 0.0):                      # B1R-05: not inf or nan
            raise SystemExit("--wind-shear-exponent takes a finite exponent >= 0 (revision 19 addendum 6)")
    if "--turbulence" in args:                         # revision 23 (B2): a TurbSim full-field file as the wind
        if "--wind-shear-exponent" in args:
            raise SystemExit("--turbulence and --wind-shear-exponent are two winds; give one (revision 23)")
        if not (REPO/args["--turbulence"]).resolve().is_file():
            raise SystemExit(f"--turbulence takes the path of a TurbSim .bts file (not found: {args['--turbulence']})")
    if "--wind-speed" in args:                         # revision 18 addendum 8 (18.4): the vortex anchors at 8 and 12 m/s
        try:
            speed = float(args["--wind-speed"])
        except ValueError:
            speed = float("nan")
        if not (math.isfinite(speed) and speed > 0.0):
            raise SystemExit("--wind-speed takes a finite speed > 0 in m/s (revision 18 addendum 8)")
        if "--turbulence" in args:
            raise SystemExit("--wind-speed and --turbulence: the field brings its own mean wind; give one (revision 18 addendum 8)")
    if "--device" in args:                                   # whole-step revision 10 (A): a CPU segment, no GPU option
        if args["--device"] not in ("auto", "cpu", "gpu") or ALL_SEGMENTS[segment]["gpu"] or \
                any(o in args for o in GPU_OPTIONS):
            raise SystemExit("--device takes auto, cpu or gpu, with the CPU segment's name and without --gpu-* options "
                             "(whole-step revision 10, A)")
        if gpu_variant(segment) is None and args["--device"] != "cpu":
            raise SystemExit(f"--device {args['--device']}: the segment {segment} has no GPU variant")
    if any(o in args for o in GPU_OPTIONS) and (
            not ALL_SEGMENTS[segment]["gpu"] or args.get("--gpu-precision", "mixed") not in ("fp64", "mixed")
            or any(args.get(o, "cpu") not in ("cpu", "gpu") for o in GPU_OPTIONS[1:])):
        raise SystemExit("--gpu-precision fp64|mixed, --gpu-control, --gpu-moments, --gpu-build and --gpu-rings cpu|gpu go "
                         "with a GPU segment (whole-step revisions 5-7, 10)")
    if args.get("--record-ring-inflow", "on") != "on":  # OF-0011 revision 7 addendum 11 (C1-R7)
        raise SystemExit("--record-ring-inflow takes 'on'")
    if "--iec" in args or "--iec-start" in args:      # OF-0011 revision 7 addendum 6 supplement 1 (P2)
        from openfawe.ground.iec_wind import CASES as IEC_CASES
        if args.get("--iec") not in IEC_CASES or "--wind-shear-exponent" in args or "--turbulence" in args:
            raise SystemExit("--iec takes one of " + ", ".join(IEC_CASES) + ", not with --wind-shear-exponent or --turbulence")
        try:
            iec_start = float(args.get("--iec-start", "10"))
        except ValueError:
            iec_start = float("nan")
        if not (math.isfinite(iec_start) and iec_start >= 0.0):
            raise SystemExit("--iec-start takes a finite time >= 0 (s), with --iec")
    fields_given = any(o in args for o in ("--wind-shear-exponent", "--turbulence", "--wind-speed", "--iec"))
    if "--floating" in args and (args["--floating"] not in ("norway", "norway-waves") + F3D_FLOATING or  # OF-0011 add 77
                                 (fields_given and args["--floating"] not in F3D_SPAR + ("oc3-spar-fixed",))):
        raise SystemExit("--floating takes norway, norway-waves or (F3d) " + ", ".join(F3D_FLOATING) + "; a wind shear, "
                         "turbulence, --wind-speed or --iec only with the spar values (OF-0011 revision 6 addendum 77, "
                         "revision 7 addenda 1 and 6)")
    if "--crosswind-params" in args and (args["--crosswind-params"] != "land" or args.get("--floating") not in   # add 2
                                         ("norway", "norway-waves", "oc3-spar-fixed") + F3D_SPAR):
        raise SystemExit("--crosswind-params takes 'land', with a Norway --floating value (OF-0011 revision 7 addendum 2)")
    if "--balance" in args and (args["--balance"] != "on" or args.get("--floating") not in F3D_SPAR):  # rev 7 add 14
        raise SystemExit("--balance takes 'on', with an F3d spar value of --floating (" + ", ".join(F3D_SPAR) + ")")
    if "--tether-events" in args and args["--tether-events"] != "depth":        # rev 7 add 4 supplement 7
        raise SystemExit("--tether-events takes 'depth' (OF-0011 revision 7 addendum 4 supplement 7)")
    if "--return-period" in args and (window not in ("whole", "hover") or not args["--return-period"].isdigit()   # add 20
                                      or int(args["--return-period"]) < 1):
        raise SystemExit("--return-period takes a positive integer, with the whole or hover window (OF-0011 revision 7 "
                         "addendum 20)")
    if args.get("--floating") in F3D_SPAR and "--checkpoint-every" in args:      # OF-0011 rev 7 add 1: not resumable
        raise SystemExit(f"--floating {args['--floating']} cannot be resumed (HydroDyn's state): no --checkpoint-every")
    if "--wake-frames-every" in args and (not args["--wake-frames-every"].isdigit() or int(args["--wake-frames-every"]) < 1
                                          or "slipstream" not in ALL_SEGMENTS[segment]):
        raise SystemExit("--wake-frames-every takes a positive integer, with a B' segment (revision 27)")
    if "--turbulence-t0" in args:
        try:
            t0 = float(args["--turbulence-t0"])
        except ValueError:
            t0 = float("nan")
        if "--turbulence" not in args or not (math.isfinite(t0) and t0 >= 0.0):
            raise SystemExit("--turbulence-t0 takes a finite time >= 0 (seconds into the field), with --turbulence (revision 23)")
    if "--keep-rows" in args and (not args["--keep-rows"].isdigit() or int(args["--keep-rows"]) <= 187):
        raise SystemExit("--keep-rows takes an integer above 187")
    if "--free-rows" in args and (not args["--free-rows"].isdigit() or not 1 <= int(args["--free-rows"]) < 187):
        raise SystemExit("--free-rows takes an integer from 1 to 186 (revision 22 addendum 3, B3 H4)")
    if "--frozen-drift" in args and (args["--frozen-drift"] != "rings" or "--free-rows" not in args):
        raise SystemExit("--frozen-drift takes 'rings', with --free-rows (revision 22 addendum 4, B3 H5)")
    if "--stall-lag" in args:
        try:
            lag = float(args["--stall-lag"])
        except ValueError:
            lag = float("nan")
        if not (math.isfinite(lag) and lag > 0.0):
            raise SystemExit("--stall-lag takes a finite number > 0 (revision 22 addendum 8, B3 H6)")
    if args.get("--jacobian", "active") != "active":         # GPU whole-step contract revision 9 (J): Newton's analytic Jacobian
        raise SystemExit("--jacobian takes 'active' (the P2 groups' Newton solve with jacobian='active_segment')")
    if args.get("--power", "on") not in ("on", "jit") or args.get("--hover-tables", "on") != "on":
        raise SystemExit("--power takes 'on' (or 'jit': the same power system on the compiled path, OF-0014 revision 2) and "
                         "--hover-tables 'on'")
    if args.get("--blown-wing", "tables") != "tables" or ("--blown-wing" in args) != ("--hover-tables" in args):
        raise SystemExit("--blown-wing takes 'tables', and goes with --hover-tables on (revision 11)")
    return args


def wind_speed_change(params, wind_speed):
    """Revision 18 addendum 8: ``phys_sim.wind_speed`` = the option's speed, as ``table_variants.py --wind-speed``; with
    '-' nothing is touched and None is returned, else ``[before, after]``."""
    if wind_speed == "-":
        return None
    phys = params.raw["phys_sim"]
    before = float(phys["wind_speed"])
    phys["wind_speed"] = float(wind_speed)
    return [before, float(phys["wind_speed"])]


IEC_SLOT = "iec:"                                       # OF-0011 revision 7 addendum 6 supplement 1: --iec in the shear slot


def shear_slot(args):
    """The child's shear argument: the exponent, ``iec:CASE:START`` for --iec, or '-'."""
    if "--iec" in args:
        return f"{IEC_SLOT}{args['--iec']}:{float(args.get('--iec-start', '10'))!r}"
    return args.get("--wind-shear-exponent", "-")


def iec_of_slot(shear):
    """``(case, start)`` of an ``iec:`` shear slot, else None."""
    if not str(shear).startswith(IEC_SLOT):
        return None
    case, start = str(shear)[len(IEC_SLOT):].rsplit(":", 1)
    return case, float(start)


def fingerprint_items(segment, window, periods, ladder, keep, power, hover_tables, blown, every, crosswind="-", shear="-",
                      turbulence=None, free_rows="-", frozen="-", stall_lag="-", wind_speed="-", gpu_precision="-",
                      gpu_control="-", gpu_moments="-", gpu_build="-", floating="-", jacobian="-", gpu_rings="-",
                      ring_inflow="-", tether_events="-", return_period="-"):
    """The settings of a run that its saved states must match (revision 12; ``flight_resume.fingerprint``): the first nine
    always, every later option only when given (so a run without it keeps the fingerprint it had before the option).
    ``turbulence`` is ``(field sha256, t0_s)`` or None."""
    return ([segment, window, periods, ladder, keep, power, hover_tables, blown, every] +
            ([] if crosswind == "-" else [crosswind]) +
            ([] if shear == "-" else ["shear", shear]) +                       # B1R-02
            ([] if turbulence is None else ["turbulence", *turbulence]) +      # revision 23 (B2)
            ([] if free_rows == "-" else ["free_rows", free_rows]) +
            ([] if frozen == "-" else ["frozen", frozen]) +
            ([] if stall_lag == "-" else ["stall_lag", stall_lag]) +
            ([] if wind_speed == "-" else ["wind_speed", wind_speed]) +        # revision 18 addendum 8
            ([] if gpu_precision == "-" else ["gpu_precision", gpu_precision]) +   # whole-step contract revision 5
            ([] if gpu_control == "-" else ["gpu_control", gpu_control]) +
            ([] if gpu_moments == "-" else ["gpu_moments", gpu_moments]) +         # whole-step contract revision 6
            ([] if gpu_build == "-" else ["gpu_build", gpu_build]) +               # whole-step contract revision 7
            ([] if floating == "-" else ["floating", floating]) +                  # OF-0011 revision 6 addendum 77
            ([] if jacobian == "-" else ["jacobian", jacobian]) +                  # whole-step contract revision 9
            ([] if gpu_rings == "-" else ["gpu_rings", gpu_rings]) +               # whole-step contract revision 10 (R)
            ([] if ring_inflow == "-" else ["ring_inflow", ring_inflow]) +         # OF-0011 revision 7 addendum 11
            ([] if tether_events == "-" else ["tether_events", tether_events]) +   # rev 7 add 4 supplement 7
            ([] if return_period == "-" else ["return_period", return_period]))    # rev 7 add 20


def child(out_dir, build, segment, window, periods, ladder="-", keep="-", power="-", hover_tables="-", blown="-", every="-",
          crosswind="-", shear="-", turbulence="-", turbulence_t0="-", free_rows="-", frozen="-", stall_lag="-", wind_speed="-",
          wake_frames_every="-", gpu_precision="-", gpu_control="-", gpu_moments="-", gpu_build="-", floating="-", jacobian="-",
          gpu_rings="-", ring_inflow="-", balance="-", tether_events="-", return_period="-"):
    sys.path.insert(0, str(G5))
    if ladder != "-" or jacobian != "-":               # diagnosis only (revision 8 addendum 1): the Picard fallback ladder;
        from dataclasses import replace                # whole-step contract revision 9 (J): Newton's analytic Jacobian
        from openfawe.aero.handover import TAKEOFF_LANDING_SOLVE
        from openfawe.flight import platform_surfaces as PFS
        solve = TAKEOFF_LANDING_SOLVE
        if ladder != "-":
            relaxations, iterations = ladder.split(":")
            solve = replace(solve, fallback_relaxations=tuple(float(v) for v in relaxations.split(",")),
                            fallback_max_iterations=int(iterations))
        if jacobian != "-":
            solve = replace(solve, jacobian="active_segment")
        original_init = PFS.PlatformGroup.__init__

        def init(self, *a, **k):
            k.setdefault("solve", solve)
            original_init(self, *a, **k)
        PFS.PlatformGroup.__init__ = init
    import turnkey_checks as TC
    import turnkey_whole_vortex as W
    import whole_vortex_timing as WT
    from openfawe.aero.makani_database import MakaniDatabase
    from openfawe.flight.makani_full import load_full_api
    from openfawe.flight.makani_interface import G_TO_WORLD
    from openfawe.ground import gs02 as G
    from openfawe.ground.perch import PerchedSystem, makani_wind_g
    from openfawe.ground.turnkey import mode_names, run_turnkey
    from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set
    from openfawe.rotor.table import RotorTable
    out_dir = Path(out_dir)
    api = load_full_api(build)
    base = spar = None
    place, land_crosswind = (floating[:-len(LAND_CROSSWIND)], True) if floating.endswith(LAND_CROSSWIND) else (floating, False)
    if place == "-":
        params = G.load_gs02_params(TC.PARAMS)
    elif place in F3D_FLOATING:                        # OF-0011 revision 7 addendum 1 (F3d): F3b's spar, or a frozen base
        sys.path.insert(0, str(REPO/"docs/tasks/OF-0011-floating20261105"))
        import f3b
        import f3d
        from openfawe.ground.turnkey import END_PERIODS, PERIOD_S
        span = periods if (window == "hover" or (window == "whole" and periods)) else END_PERIODS
        spar = f3d.make_base(place, out_dir, span*PERIOD_S + 20.0,         # f3b.fly's t_max
                             **({} if balance == "-" else dict(balance=True)))     # OF-0011 rev 7 add 14 (C1-V3)
        params = G.load_gs02_params(TC.PARAMS if spar["kind"] == "land-fixed" else spar["params"])
        base = spar["base"]
        if spar["kind"] != "land-fixed":
            spar["ground_z_set"] = f3b.set_ground_z(api, spar["ground_z"])   # addendum 83 result 1
        (out_dir/"floating-inputs.json").write_text(json.dumps({k: v for k, v in spar.items() if k not in ("base", "close")},
                                                               default=str, indent=1) + "\n", encoding="utf-8")
    else:                                              # OF-0011 revision 6 addendum 77 (F3c): as f3a.py's resumable child
        import importlib                               # by name: the studio's bundle (this file is one of its task files)
        MB = importlib.import_module("openfawe.floating.makani_buoy")      # carries no offshore code (OF-0013 RS-1's
        TB = importlib.import_module("openfawe.floating.turnkey_base")     # closure test); here the same modules
        from openfawe.flight.residual import load_rigid_body
        params = G.load_gs02_params(FLOATING["norway"])
        waves = place == "norway-waves"
        base = TB.load_makani_buoy_base(FLOATING["buoy"], body=load_rigid_body(FLOATING["serial05"]),
                                        solver=MB.MakaniSolver(Path(build)/"lib"/"libmakani_full_controller.so"),
                                        still_water=not waves, components=FLOATING["components"] if waves else None)
    if land_crosswind:                                 # OF-0011 revision 7 addendum 2: before the controller starts
        sys.path.insert(0, str(REPO/"docs/tasks/OF-0011-floating20261105"))
        import f3d as F3D
        (out_dir/"crosswind-swap.json").write_text(json.dumps(F3D.swap_crosswind(api), indent=1) + "\n", encoding="utf-8")
    wind_speed_changed = wind_speed_change(params, wind_speed)     # revision 18 addendum 8; before the hook and the winds
    aero = MakaniDatabase.from_files(TC.LOW, spoiler_path=TC.SPOILERS, high_path=TC.HIGH)
    discs = load_rotor_set(TC.ROTOR_SET, RotorTable.from_makani_json(TC.ROTOR_TABLE))
    inertia = load_rotor_inertia(TC.ROTOR_SET)
    numbers = {v: k for k, v in mode_names(api).items()}
    args = (PerchedSystem(params).body if base is None or base.body is None else base.body, G_TO_WORLD@makani_wind_g(params),
            discs, numbers, aero)
    kwargs = dict(mode="vortex", density_kg_m3=float(params.raw["phys_sim"]["air_density"]), orthonormalise=True,
                  gpu=ALL_SEGMENTS[segment]["gpu"], platform="p2", seeding=ALL_SEGMENTS[segment].get("seeding", "strip"),
                  **({} if not ALL_SEGMENTS[segment].get("viscosity", True) else dict(viscosity=True)),
                  **({} if not ALL_SEGMENTS[segment].get("local_strip") else dict(local_strip=True)),
                  **({} if keep == "-" else dict(keep_rows=int(keep))),              # OF-0010 revision 10 (WK)
                  **({} if free_rows == "-" else dict(free_rows=int(free_rows))),   # OF-0010 revision 22 addendum 3 (B3, H4)
                  **({} if frozen == "-" else dict(frozen_drift=frozen)),            # revision 22 addendum 4 (B3, H5)
                  **({} if stall_lag == "-" else dict(stall_lag=float(stall_lag))),  # revision 22 addendum 8 (B3, H6)
                  **({} if hover_tables == "-" else dict(hover_tables=True)),       # OF-0010 revision 11 (B)
                  **({} if "slipstream" not in ALL_SEGMENTS[segment] else              # OF-0010 revision 15 (B')
                     dict(slipstream=ALL_SEGMENTS[segment]["slipstream"], hub_inflow=ALL_SEGMENTS[segment]["hub_inflow"],
                          state_folder=str(out_dir/"wake-states"))),
                  **({} if ring_inflow == "-" else dict(ring_inflow=True)),          # OF-0011 rev 7 add 11 (C1-R7)
                  **({} if not ALL_SEGMENTS[segment].get("quasi_steady") else dict(quasi_steady=True)),   # add 12 (C1-R6)
                  **({} if wake_frames_every == "-" else dict(wake_frames=dict(folder=str(out_dir/"wake-frames"),  # revision 27
                                                                               every=int(wake_frames_every)))),
                  **({} if gpu_precision == gpu_control == gpu_moments == gpu_build == gpu_rings == "-" else dict(
                      gpu_settings=dict(
                      **({} if gpu_precision == "-" else dict(precision=gpu_precision)),        # whole-step rev 5
                      **({} if gpu_control == "-" else dict(control=gpu_control)),
                      **({} if gpu_moments == "-" else dict(moments=gpu_moments)),              # whole-step rev 6
                      **({} if gpu_build == "-" else dict(build=gpu_build)),                    # whole-step rev 7
                      **({} if gpu_rings == "-" else dict(rings=gpu_rings))))))                 # whole-step rev 10 (R)
    shadow = window.replace("-", "_") in SHADOW_WINDOWS
    if shadow:
        hook = WT.make_shadow_class(W.WholeFlightVortex)(*args, activate_number=int(numbers[SHADOW_WINDOWS[window.replace("-", "_")]]),
                                                         **kwargs)
    else:
        hook = W.WholeFlightVortex(*args, **kwargs)
    extra = {} if power == "-" else dict(power_system=TC.power_system_config(discs, inertia,  # revision 10: plan 3.9
                                                                            compiled=power == "jit"))   # OF-0014 rev 2
    blown_wing = False if blown == "-" else "tables"                                            # revision 11 (B)
    rows, last = [], [time.perf_counter()]
    crosswind_first = [None]
    names = mode_names(api)
    account, cumulative = {}, dict(ground=0.0, motor=0.0)
    turbulent = None
    if turbulence != "-":                              # revision 23 (B2): the field is read before the fingerprint (its hash)
        from openfawe.ground.turbulent_wind import TurbulentWind
        from openfawe.ground.turnkey import GROUND_Z_G
        field_ground = GROUND_Z_G if base is None else float(base.ground_z_g_m)     # OF-0011 rev 7 add 6 supplement 1
        turbulent = TurbulentWind.from_params(params, (REPO/turbulence).resolve(), ground_z_g_m=field_ground, z_below="saturate",
                                              t0_s=30.0 if turbulence_t0 == "-" else float(turbulence_t0))
    store = run_blob = None
    if every == "-":
        stream = gzip.open(out_dir/"records.jsonl.gz", "wt", encoding="utf-8")
    else:                                              # revision 12: the saved states and the record parts
        import flight_resume as FR
        store = FR.ResumeStore(out_dir.parent.parent/"checkpoints"/out_dir.name, print_=lambda s: print(s, flush=True))
        _, run_blob, caller, _ = store.start(FR.fingerprint(build, fingerprint_items(
            segment, window, periods, ladder, keep, power, hover_tables, blown, every, crosswind, shear,
            None if turbulence == "-" else (turbulent.field.sha256, turbulent.t0_s, turbulent.z_below), free_rows, frozen, stall_lag,
            wind_speed, gpu_precision, gpu_control, gpu_moments, gpu_build, floating, jacobian, gpu_rings, ring_inflow,
            tether_events, return_period)))
        if run_blob is not None:
            account.update(caller["account"])
            cumulative.update(caller["cumulative"])
            rows.extend(store.kept_rows())
        store.caller_state = lambda: dict(account=account, cumulative=cumulative)
        store.every = int(every)

    def on_record(r):
        now = time.perf_counter()
        active = not shadow or hook.active_from is not None
        ground, motor = energy_account(account, cumulative, r, names)       # revision 11 addendum 2
        if active:
            row = {key: r.get(key) for key in KEEP}
            row.update(cum_ground_J=ground, cum_motor_electrical_J=motor)
            row["wall_s"] = now-last[0]
            stalled = row.pop("whole_main_wing_panels_stalled_branch")
            row["stalled"] = None if stalled is None else len(stalled)
            rows.append(row)
            if store is None:
                stream.write(json.dumps(r)+"\n")
            else:
                store.write(r, row)
            if len(rows) % 100 == 0:
                print(f"period {r['k']} {r.get('flight_mode')} {row['wall_s']:.2f} s", flush=True)
        last[0] = now
        if shadow and hook.active_from is not None and int(r["k"]) >= hook.active_from+periods-1:
            raise WT.TimingWindowDone(f"period {r['k']}")
        if crosswind != "-":                           # revision 15 addendum 1 (BP-11): after the run's own first
            if r.get("whole_mode") == "CrosswindNormal" and crosswind_first[0] is None:     # CrosswindNormal period
                crosswind_first[0] = int(r["k"])
            if crosswind_first[0] is not None and int(r["k"]) >= crosswind_first[0]+int(crosswind)-1:
                raise WT.TimingWindowDone(f"period {r['k']}: {crosswind} periods from the first CrosswindNormal period")
    started = time.time()
    if store is not None:
        extra.update(checkpoint=store, resume=run_blob)
    if turbulence != "-":                              # revision 23 (B2): the IEC turbulence field (see turbulent above)
        extra.update(wind_shear=turbulent)
    if shear != "-":                                   # revision 19 addendum 6 (B1): Makani's power-law shear, also on
        from openfawe.ground.wind_shear import ShearWind   # the vortex path (the hook takes the field every period)
        from openfawe.ground.turnkey import GROUND_Z_G
        field_ground = GROUND_Z_G if base is None else float(base.ground_z_g_m)     # OF-0011 rev 7 add 6 supplement 1
        iec = iec_of_slot(shear)
        if iec is not None:                            # OF-0011 rev 7 add 6 supplement 1: an IEC deterministic wind
            from openfawe.ground.iec_wind import IecEventWind
            extra.update(wind_shear=IecEventWind(iec[0], params.raw, ground_z_g_m=field_ground, event_t_start=iec[1]))
        else:
            extra.update(wind_shear=ShearWind.from_params(params, float(shear), ground_z_g_m=field_ground))
    if base is not None:                                   # OF-0011 revision 6 addendum 77 (F3c)
        extra.update(floating=base)
    if spar is not None and spar["kind"] != "land-fixed":   # OF-0011 revision 7 addendum 3 (C1-V3)
        extra.update(record_terminal=True)
    if "slipstream" in ALL_SEGMENTS[segment]:              # revision 15 item 4 (B'): the airborne tether nodes
        extra.update(record_nodes_every=B_PRIME_NODES_EVERY)
    if tether_events != "-":                               # OF-0011 revision 7 addendum 4 supplement 7
        extra.update(tether_events=tether_events)
    if return_period != "-":                               # OF-0011 revision 7 addendum 20 (as f3d.py tables, addendum 7)
        extra.update(stop_rules=dict(throttle_return_period=int(return_period)))
    try:
        result = run_turnkey(params, api, aero, discs, inertia, blown_wing=blown_wing, on_record=on_record, keep_records=False,
                             max_periods=periods if window == "hover" or (window == "whole" and periods) else None,
                             segment=hook, **extra)
    except Exception as error:                         # noqa: BLE001  (outside the flight's own capture: start or restore)
        if spar is not None:
            spar["close"]()
        if run_blob is None:
            raise
        store.mark_unresumable(f"{type(error).__name__}: {error}")    # revision 12 addendum 2 (RSM-01): main runs the
        print(f"the resume failed ({type(error).__name__}: {error}); a fresh start follows", flush=True)  # child again
        sys.exit(FR.RESTART_CODE)
    if spar is not None:                               # OF-0011 revision 7 addendum 1: HydroDyn and MoorDyn ended
        spar["close"]()
    resumed = None if store is None else store.finish(out_dir)
    if store is None:
        stream.close()
    end = dict(end=result["end"], active_from=getattr(hook, "active_from", None), hook_periods=dict(hook.periods),
               fallbacks=hook.fallbacks, settings=hook.settings(), elapsed_s=time.time()-started, rows=rows,
               energy=energy_report(account), **({} if store is None else dict(resume=resumed)),
               **({} if wind_speed_changed is None else dict(wind_speed_m_s=wind_speed_changed)))
    (out_dir/"child.json").write_text(json.dumps(end, default=str)+"\n", encoding="utf-8")


def frames_summary(folder):
    """Revision 27: the wake frames in the job folder (count, bytes, first and last period) - also after a resume."""
    files = sorted(Path(folder).glob("frame-k*.npz"))
    periods = [int(f.stem.split("-k")[1]) for f in files]
    return dict(folder=Path(folder).name, count=len(files), bytes=sum(f.stat().st_size for f in files),
                first_k=periods[0] if periods else None, last_k=periods[-1] if periods else None)


def compact(value):
    """A number to six significant digits (the whole flight's series stays under node 2's 2 MB receipt limit)."""
    return float(f"{value:.6g}") if isinstance(value, float) else value


def mean(values):
    return None if not values else float(sum(values)/len(values))


def summarise(window, data):
    rows = data["rows"]
    for r in rows:
        r["timed"] = r["wall_s"] is not None and r["wall_s"] <= LONG_PERIOD_S
    timed = [r for r in rows if r["timed"]]
    long = [dict(k=r["k"], wall_s=r["wall_s"]) for r in rows if not r["timed"]]
    by_mode, by_source = {}, {}
    for r in rows:
        mode = str(r.get("whole_mode") or r.get("flight_mode"))   # the hook's mode name; the number where it has none
        entry = by_mode.setdefault(mode, dict(periods=0, wall_s=0.0, timed=0))
        entry["periods"] += 1
        if r["timed"]:
            entry["wall_s"] += r["wall_s"]
            entry["timed"] += 1
        source = str(r.get("whole_shadow_source") or r.get("whole_model") or r.get("load_source"))
        by_source[source] = by_source.get(source, 0)+1
    for entry in by_mode.values():
        entry["mean_wall_s"] = entry["wall_s"]/entry["timed"] if entry["timed"] else None
    vsm = [r for r in rows if (r.get("whole_shadow_source") or r.get("whole_model")) == "vsm"]
    out = dict(periods=len(rows), by_flight_mode=by_mode, by_source=by_source, long_periods=long,
               viscosity_on_periods=sum(1 for r in vsm if r.get("whole_main_viscosity_on")),
               mu_max=max((r.get("whole_main_viscosity_mu_max") or 0.0 for r in vsm), default=None),
               stalled_max=max((r.get("stalled") or 0 for r in vsm), default=None),
               iterations_max=max((r.get("whole_main_iterations") or 0 for r in vsm), default=None))
    marked = [r for r in vsm if r.get("whole_main_local_strip_rounds")]      # OF-0010 P5d: the steps solved after marking
    out["local_strip"] = dict(periods=len(marked),
                              rounds={str(n): sum(1 for r in marked if r["whole_main_local_strip_rounds"] == n) for n in (1, 2, 3)},
                              steps=[dict(k=r["k"], mode=r.get("whole_mode"), rounds=r["whole_main_local_strip_rounds"],
                                          panels=r["whole_main_local_strip_panels"],
                                          regime=r.get("whole_main_local_strip_panel_regime"),
                                          share=r.get("whole_main_local_strip_share"),
                                          weighted=r.get("whole_main_local_strip_weighted_panels")) for r in marked[:2000]],
                              panel_counts={str(j): n for j, n in sorted(
                                  Counter(j for r in marked for j in r["whole_main_local_strip_panels"]).items())})
    step = 1 if window != "whole" else 10
    out["series"] = {key: [r.get(key) for r in rows[::step]] for key in ("k", "flight_mode", "wall_s", "stalled", "whole_main_iterations")}
    if window == "whole":                                 # revision 11 addendum 2: the time history, every tenth period
        out["series"].update({key: [compact(r.get(key)) for r in rows[::step]] for key in WHOLE_SERIES})
    if window in ("crosswind", "hover-shadow"):          # revision 10 (WK): what wake_keep_compare.py reads, every period
        out["series"].update({key: [r.get(key) for r in rows] for key in SHADOW_SERIES})
    out["series_every"] = step
    if window != "whole":
        classes = dict(table=[], hover=[], crosswind=[])
        for r in timed:
            mode = str(r.get("whole_mode") or r.get("flight_mode"))
            source = r.get("whole_shadow_source") or r.get("whole_model")
            if mode in TABLE_MODES or source not in ("vsm", "fallback"):
                classes["table"].append(r["wall_s"])
            elif mode in CROSSWIND_MODES:
                classes["crosswind"].append(r["wall_s"])
            else:
                classes["hover"].append(r["wall_s"])
        tail = dict((name, mean(values[len(values)//2:])) for name, values in classes.items())
        out["class_means_s"] = {name: mean(values) for name, values in classes.items()}
        out["class_means_second_half_s"] = tail
        out["class_periods"] = {name: len(values) for name, values in classes.items()}
    return out


def tree_state(status, env=None):
    """``tree_clean`` of the receipt from ``git status --porcelain``'s output: nothing listed - or, in a code snapshot
    (``detached_jobs.py --snapshot`` names the entries it links from the main copy in OPENFAWE_SNAPSHOT_LINKS; node-3
    contract addendum 28), nothing but those links as untracked entries. Any other line (a change, a staged file, an
    untracked entry that is not such a link) is not clean, as before; ``snapshot_links`` (their number) only when the
    variable is set."""
    env = os.environ if env is None else env
    try:
        links = json.loads(env["OPENFAWE_SNAPSHOT_LINKS"]) if "OPENFAWE_SNAPSHOT_LINKS" in env else None
        links = None if links is None else {str(name).rstrip("/") for name in links}
    except (ValueError, TypeError):
        links = None
    lines = [line for line in status.splitlines() if line.strip()]
    if links is None:
        return dict(tree_clean=not lines)
    excused = all(line.startswith("?? ") and line[3:].strip().strip('"').rstrip("/") in links for line in lines)
    return dict(tree_clean=bool(excused), snapshot_links=len(links))


def main(out_dir, argv):
    out_dir = Path(out_dir).resolve()                    # the runner passes it relative to its runs folder; the child runs in REPO
    out_dir.mkdir(parents=True, exist_ok=True)
    args = parse_args(argv)
    args, device = resolve_device(args)                  # whole-step revision 10 (A); without --device: unchanged
    segment, window = args["--segment"], args["--window"]
    periods = int(args["--periods"]) if "--periods" in args else 0
    build = (REPO/args["--build"]).resolve()
    sys.path.insert(0, str(G5))
    import turnkey_checks as TC
    tick = time.time()
    restarted = False
    for attempt in range(2):                         # revision 12 addendum 2: once more from scratch when a resume fails
        with open(out_dir/"child.log", "w" if attempt == 0 else "a", encoding="utf-8") as log:
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(out_dir), str(build), segment,
                                   window, str(periods), args.get("--fallback-ladder", "-"), args.get("--keep-rows", "-"),
                                   args.get("--power", "-"), args.get("--hover-tables", "-"), args.get("--blown-wing", "-"),
                                   args.get("--checkpoint-every", "-"), args.get("--crosswind-periods", "-"),
                                   shear_slot(args), args.get("--turbulence", "-"),
                                   args.get("--turbulence-t0", "-"), args.get("--free-rows", "-"),
                                   args.get("--frozen-drift", "-"), args.get("--stall-lag", "-"), args.get("--wind-speed", "-"),
                                   args.get("--wake-frames-every", "-"), args.get("--gpu-precision", "-"),
                                   args.get("--gpu-control", "-"), args.get("--gpu-moments", "-"), args.get("--gpu-build", "-"),
                                   args.get("--floating", "-") + (LAND_CROSSWIND if "--crosswind-params" in args else ""),
                                   args.get("--jacobian", "-"), args.get("--gpu-rings", "-"),
                                   "on" if "--record-ring-inflow" in args else "-",
                                   "on" if "--balance" in args else "-",           # OF-0011 rev 7 add 14
                                   args.get("--tether-events", "-"),                # rev 7 add 4 supplement 7
                                   args.get("--return-period", "-")],               # rev 7 add 20
                                  cwd=REPO, env=TC.child_env(build), stdout=log, stderr=subprocess.STDOUT, text=True)
        if "--checkpoint-every" not in args or done.returncode != 75 or attempt:      # flight_resume.RESTART_CODE
            break
        restarted = True
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    history = dict(what="OF-0010 revision 8: the turnkey flight on the platform path (a trial; judges nothing)", segment=segment,
                   window=window, periods_asked=periods or None, fallback_ladder=args.get("--fallback-ladder"),
                   keep_rows=None if "--keep-rows" not in args else int(args["--keep-rows"]), power=args.get("--power"),
                   hover_tables=args.get("--hover-tables"), blown_wing=args.get("--blown-wing"),
                   **({} if "--checkpoint-every" not in args else dict(checkpoint_every=int(args["--checkpoint-every"]),
                                                                      restarted_after_failed_resume=restarted)),
                   **({} if "--crosswind-periods" not in args else dict(crosswind_periods=int(args["--crosswind-periods"]))),
                   **({} if "--wind-shear-exponent" not in args else dict(wind_shear_exponent=float(args["--wind-shear-exponent"]))),
                   **({} if "--iec" not in args else dict(iec=args["--iec"],                 # OF-0011 rev 7 add 6 supplement 1
                                                          iec_start_s=float(args.get("--iec-start", "10")))),
                   **({} if "--turbulence" not in args else dict(turbulence=args["--turbulence"],      # revision 23 (B2)
                       turbulence_sha256=__import__("hashlib").sha256((REPO/args["--turbulence"]).resolve().read_bytes()).hexdigest(),
                       turbulence_t0_s=float(args.get("--turbulence-t0", "30")))),
                   **({} if "--free-rows" not in args else dict(free_rows=int(args["--free-rows"]))),
                   **({} if "--frozen-drift" not in args else dict(frozen_drift=args["--frozen-drift"])),
                   **({} if "--stall-lag" not in args else dict(stall_lag=float(args["--stall-lag"]))),
                   **({} if "--wind-speed" not in args else dict(wind_speed_asked_m_s=float(args["--wind-speed"]))),
                   **({} if "--wake-frames-every" not in args else dict(                                   # revision 27
                       wake_frames_every=int(args["--wake-frames-every"]), wake_frames=frames_summary(out_dir/"wake-frames"))),
                   **({} if "--gpu-precision" not in args else dict(gpu_precision=args["--gpu-precision"])),
                   **({} if "--gpu-control" not in args else dict(gpu_control=args["--gpu-control"])),
                   **({} if "--gpu-moments" not in args else dict(gpu_moments=args["--gpu-moments"])),
                   **({} if "--gpu-build" not in args else dict(gpu_build=args["--gpu-build"])),
                   **({} if "--floating" not in args or args["--floating"] in F3D_FLOATING else dict(       # add 77
                       floating=args["--floating"], floating_inputs_sha256={
                       str(p.relative_to(REPO)): __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                       for name, p in FLOATING.items() if name != "components" or args["--floating"] == "norway-waves"})),
                   **({} if args.get("--floating") not in F3D_FLOATING else dict(                      # rev 7 add 1 (F3d)
                       floating=args["--floating"], floating_inputs=json.loads((out_dir/"floating-inputs.json").read_text(
                           encoding="utf-8")) if (out_dir/"floating-inputs.json").exists() else None)),
                   **({} if "--crosswind-params" not in args else dict(crosswind_params=args["--crosswind-params"],  # add 2
                       crosswind_swap=json.loads((out_dir/"crosswind-swap.json").read_text(encoding="utf-8"))
                       if (out_dir/"crosswind-swap.json").exists() else None)),
                   **({} if "--jacobian" not in args else dict(jacobian=args["--jacobian"])),    # whole-step revision 9
                   **({} if "--gpu-rings" not in args else dict(gpu_rings=args["--gpu-rings"])),  # whole-step revision 10 (R)
                   **({} if "--record-ring-inflow" not in args else dict(record_ring_inflow=True)),  # OF-0011 rev 7 add 11
                   **({} if "--balance" not in args else dict(balance=True)),                         # OF-0011 rev 7 add 14
                   **({} if "--tether-events" not in args else dict(tether_events=args["--tether-events"])),  # add 4 supp. 7
                   **({} if "--return-period" not in args else dict(return_period=int(args["--return-period"]))),  # add 20
                   **({} if device is None else dict(device=device)),                           # whole-step revision 10 (A)
                   returncode=done.returncode, seconds=time.time()-tick,
                   provenance=dict(git_commit=git("rev-parse", "HEAD"), **tree_state(git("status", "--porcelain"))),
                   machine=dict(host=platform.node(), processor=platform.processor(), platform=platform.platform(),
                                python=sys.version.split()[0], threads=os.environ.get("NUMBA_NUM_THREADS")),
                   status="failed")
    child_file = out_dir/"child.json"
    if child_file.exists():
        data = json.loads(child_file.read_text(encoding="utf-8"))
        history.update(end=data["end"], active_from=data["active_from"], hook_periods=data["hook_periods"],
                       fallbacks=data["fallbacks"][:50], fallback_count=len(data["fallbacks"]), settings=data["settings"],
                       summary=summarise(window, data), energy=data.get("energy"), status="complete",
                       **({} if "resume" not in data else dict(resume=data["resume"])),
                       **({} if "wind_speed_m_s" not in data else dict(wind_speed_m_s=data["wind_speed_m_s"])))
    else:
        history["child_log_tail"] = (out_dir/"child.log").read_text(encoding="utf-8", errors="replace")[-4000:]
    text = json.dumps(history, default=str)
    if len(text.encode()) > 1_900_000:                       # node 2 returns history.json only under 2 MB
        history["summary"]["series"] = None
        history["summary"]["series_dropped"] = "over the 2 MB receipt limit"
        text = json.dumps(history, default=str)
    (out_dir/"history.json").write_text(text+"\n", encoding="utf-8")
    print(json.dumps(dict(status=history["status"], end=history.get("end"), seconds=history["seconds"])), flush=True)


if __name__ == "__main__":
    if 7 <= len(sys.argv) <= 33 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), *sys.argv[7:33])
    else:
        raise SystemExit("run as a runner job (main) or as '--child ...'")
