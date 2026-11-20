"""Checks P1-P7 of the full-cycle contract, part 3 (``full-cycle-contract.zh.md``, revision 6): TurnKey on the reference
plant, from the perch to HoverDescend and back onto the perch (``openfawe.ground.turnkey``).

usage: turnkey_checks.py OUT --build FULL_BUILD --parts unit,p5,off,on,judge [--trial-end-s S] [--cw10 FILE]

  unit   the unit tests of tests/test_turnkey.py in a child process with the library (receipt unit.json; a skip or a
         failure makes the run not judgeable)
  p5     P5: the first 4000 periods in two child processes; their records must be identical (receipt p5.json)
  off    the judged flight, blown-wing correction off (K1): child process, records off-records.jsonl, tail file
         off-tail.json rewritten every 1 s of simulated time, full records appended every 10 s, off-final.json at the
         end (receipt run-off.json)
  on     the same flight with the blown-wing correction on: reported only (receipt run-on.json)
  judge  P1-P7 from the receipts and the records of 'off', the same quantities for 'on' (judgement.json)

Revision 7, option 甲 (tether ground contact, opt-in; parts added after revision 6's judged runs, the parts above unchanged):
  unitgc  the unit tests of tests/test_tether_ground_contact.py (receipt unitgc.json)
  p5c     P5 with the ground contact on (tags p5ca, p5cb; receipt p5c.json)
  offc    the flight with the ground contact on, blown wing off: judged with P6' (receipt run-offc.json)
  onc     the same with the blown wing on: reported only (receipt run-onc.json)
  judgec  P1-P5 and P7 as revision 6, P6' instead of P6, from unit, unitgc, p5, p5c and offc; the receipt of 'p5' (ground
          contact off, run at this commit) must carry revision 6's P5 records hash 168e4fbf... in both processes (the
          option off changes nothing) (judgement-contact.json)

``--trial-end-s`` shortens 'off' and 'on' (development only; the receipts say so and the judgement refuses them).
``--cw10``: a JSON file of CW-10's flight-mode durations made on the owner's PC (the flight logs stay there,
decision on flight data, section 7); without it that report item says it is missing.

The judgement refuses to judge unless every receipt carries the same commit, a clean tracked tree at start and end, the
same file hashes and library hash, the unit tests passed with nothing skipped, and no input is a trial. Nothing is
retried and no criterion is computed from a run that is not the judged one. The criteria are those of revision 6,
committed before this script.
"""

import argparse
from collections import deque
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO/"src"))

PARAMS = REPO/"cases/ground/makani_gs02_parker_ranch.json"
DATA = REPO/"cases/polars/makani"
LOW, SPOILERS, HIGH = DATA/"m600_aswing_baseline.json", DATA/"spoiler_offset_tables.json", DATA/"hover_model_sn02_wind_10_no_wake_model.json"
ROTOR_SET, ROTOR_TABLE = REPO/"cases/rotors/m600_rotors_makani.json", REPO/"cases/rotors/makani/rotor_rev4.json"
P5_PERIODS = 4000
TAIL_PERIODS, FLUSH_PERIODS = 100, 1000             # K7: last 100 periods every 1 s; the full record every 10 s
UNIT_TESTS = 16
UNIT_TESTS_GC = 7                                    # tests/test_tether_ground_contact.py (revision 7, code-commit addendum)
P5_RECORDS_SHA256 = "168e4fbf37343b9f4147d62fdf5ce36b935a425214c294d27da6e2ad08699dc1"   # revision 6 addendum 2
PENETRATION_LIMIT_M = 0.05                           # P6' (revision 7, option 甲)
FREE_LENGTH_M, FREE_LENGTH_TOL_M = 425.8, 0.05
HASHED = ("src/openfawe/ground/turnkey.py", "src/openfawe/ground/turnkey_scoring.py", "src/openfawe/ground/perch.py",
          "src/openfawe/ground/gs02.py", "src/openfawe/tether/variable_line.py", "src/openfawe/tether/lumped_line.py",
          "src/openfawe/tether/bridle.py", "src/openfawe/flight/makani_full.py", "src/openfawe/flight/makani_interface.py",
          "src/openfawe/flight/reference_flight.py", "src/openfawe/flight/free_flight.py", "src/openfawe/flight/plant.py",
          "src/openfawe/flight/closed_loop.py", "src/openfawe/flight/rigid_body.py", "src/openfawe/flight/kinematics.py",
          "src/openfawe/aero/makani_database.py", "src/openfawe/aero/makani_flap_increments.py",
          "src/openfawe/rotor/actuator_disc.py", "src/openfawe/rotor/table.py", "src/openfawe/rotor/makani_corrections.py",
          "tests/test_turnkey.py", "tests/test_tether_ground_contact.py", "docs/tasks/OF-0008-g5control20261006/turnkey_checks.py",
          "cases/ground/makani_gs02_parker_ranch.json", "cases/aircraft/m600_makani_rigid_body.json",
          "cases/polars/makani/m600_aswing_baseline.json", "cases/polars/makani/spoiler_offset_tables.json",
          "cases/polars/makani/hover_model_sn02_wind_10_no_wake_model.json", "cases/rotors/m600_rotors_makani.json",
          "cases/rotors/makani/rotor_rev4.json", "src/openfawe/ground/wind_shear.py", "src/openfawe/ground/makani_wind.py")


def git(*args):
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()[:300]}")
    return done.stdout.strip()


def sha(path):
    """SHA-256 of a file, read in 16 MiB chunks (a whole flight's records with the power system are about 1.2 GB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(build):
    try:
        commit = git("rev-parse", "HEAD")
        clean = git("status", "--porcelain", "--untracked-files=no", "--", "src", "tests", "cases",
                    "docs/tasks/OF-0008-g5control20261006") == ""
    except (RuntimeError, OSError) as error:
        commit, clean = None, False
        print(f"provenance: {error}", file=sys.stderr)
    lib = Path(build)/"lib"
    library = next((p for p in (lib/"libmakani_full_controller.so", lib/"makani_full_controller.dll") if p.exists()), None)
    return dict(git_commit=commit, tree_clean=clean, files={name: sha(REPO/name) for name in HASHED},
                library=None if library is None else library.name, library_sha256=None if library is None else sha(library),
                binding_sha256=sha(lib/"makani_full_controller_api.py") if (lib/"makani_full_controller_api.py").exists() else None,
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), machine=platform.node())


def child_env(build):
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO/"src"), str(REPO)]), OPENFAWE_MAKANI_FULL_BUILD=str(build))


# --------------------------------------------------------------------------------------------------- child processes
def unit_child(out_file, module="tests.test_turnkey"):
    import unittest
    sys.path.insert(0, str(REPO))
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    names = []

    def walk(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            else:
                names.append(t.id())
    walk(suite)                                   # before the run: the runner lets go of the tests it has run
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    outcome = {name: "ok" for name in names}
    for kind, items in (("FAIL", result.failures), ("ERROR", result.errors)):
        for test, _ in items:
            outcome[test.id()] = kind
    for test, reason in result.skipped:
        outcome[test.id()] = "skipped: " + reason
    Path(out_file).write_text(json.dumps(dict(outcome=outcome, ran=result.testsRun), indent=1) + "\n", encoding="utf-8")


def whole_segment(segment):
    """A 'whole...' segment name: (base, platform step, seeding, viscosity, local strip) - OF-0010's suffixes '+p1', '+p2',
    '+p1m', '+p2m', '+p1s', '+p2s', each optionally followed by 'v' (P5c) and then by 'l' (P5d, revision 9)."""
    text = str(segment)
    suffixes = {"+p1": ("p1", "zero"), "+p2": ("p2", "zero"), "+p1m": ("p1", "mixed"), "+p2m": ("p2", "mixed"),
                "+p1s": ("p1", "strip"), "+p2s": ("p2", "strip")}   # OF-0010
    local = text.endswith("l") and any(text[:-1].endswith((key, key+"v")) for key in suffixes)   # OF-0010 P5d: '+p2sl'
    text = text[:-1] if local else text
    viscous = any(text.endswith(key+"v") for key in suffixes)       # OF-0010 P5c: '+p2sv' etc.
    text_core = text[:-1] if viscous else text
    suffix = next((key for key in suffixes if text_core.endswith(key)), None)
    platform_step, platform_seeding = suffixes[suffix] if suffix else (None, "zero")
    base = text_core[:-len(suffix)] if suffix else text
    return base, platform_step, platform_seeding, viscous, local


def flight_child(tag, folder, build, blown, max_periods, contact=False, segment=None, power=False, shear=None,
                 turbulence=None, turbulence_t0=30.0):
    """One flight in this process: records appended every FLUSH_PERIODS periods, the tail rewritten every TAIL_PERIODS.
    ``segment`` (plan 3.8a): None, or 'record' / 'vortex' for ``turnkey_vortex.VortexSegment`` in that mode; 'vortex+orth'
    also orthonormalises the attitude of the vortex solve's pose (opt-in, ``VortexSegment(orthonormalise=True)``);
    plan 3.8b: 'whole' / 'whole-record' for ``turnkey_whole_vortex.WholeFlightVortex`` (blown wing off); '+orth' also
    orthonormalises the attitude of the hook's pose (contract revision 2 B2, ``orthonormalise=True``), '+gpu' evaluates its
    large sums with the GPU tree code (revision 3, ``gpu=True``): 'whole+orth', 'whole+gpu', 'whole+orth+gpu'; a final
    '+p1' (OF-0010 contract revision 1) adds the platform model's step P1 (``platform='p1'``), e.g. 'whole+orth+p1'.
    ``power`` (plan 3.9a): Makani's stacked power system and the rotors' equation of motion (``power_system_config``).
    ``shear`` (plan 3.9b): None, or the wind-shear exponent (``ground.wind_shear.ShearWind`` of the parameters' wind).
    ``turbulence`` (OF-0010 revision 23, B2): None, or the path of a TurbSim .bts file (``ground.turbulent_wind.TurbulentWind``
    with ``turbulence_t0`` seconds into the field); not together with ``shear``."""
    from openfawe.aero.makani_database import MakaniDatabase
    from openfawe.flight.makani_full import load_full_api
    from openfawe.ground import gs02 as G
    from openfawe.ground.turnkey import mode_names, run_turnkey
    from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set
    from openfawe.rotor.table import RotorTable
    folder = Path(folder)
    api = load_full_api(build)
    params = G.load_gs02_params(PARAMS)
    aero = MakaniDatabase.from_files(LOW, spoiler_path=SPOILERS, high_path=HIGH)
    discs = load_rotor_set(ROTOR_SET, RotorTable.from_makani_json(ROTOR_TABLE))
    inertia = load_rotor_inertia(ROTOR_SET)
    names = mode_names(api)
    tail, buffer, count = deque(maxlen=TAIL_PERIODS), [], [0]
    tick, last = time.time(), [None]
    records = open(folder/f"{tag}-records.jsonl", "w", encoding="utf-8")

    def write_tail():
        tmp = folder/f"{tag}-tail.json.tmp"
        tmp.write_text(json.dumps(list(tail)) + "\n", encoding="utf-8")
        os.replace(tmp, folder/f"{tag}-tail.json")

    def on_record(r):
        line = json.dumps(r)
        tail.append(r)
        buffer.append(line)
        count[0] += 1
        if r["flight_mode"] != last[0]:
            print(f"  t={r['t']:8.2f} s  {names.get(r['flight_mode'], r['flight_mode'])}  gate {r['gates'][r['flight_mode']]}"
                  f"  gs {r['gs_mode']}  L {r['free_length']:.2f} m  T {r['tension_bridle_N']/1e3:.2f} kN"
                  f"  elapsed {time.time()-tick:.0f} s", flush=True)
            last[0] = r["flight_mode"]
        if count[0] % TAIL_PERIODS == 0:
            write_tail()
        if count[0] % FLUSH_PERIODS == 0:
            records.write("\n".join(buffer) + "\n")
            records.flush()
            buffer.clear()
        if count[0] % 10000 == 0:
            print(f"  period {r['k']}  {names.get(r['flight_mode'])}  L {r['free_length']:.1f} m  T {r['tension_bridle_N']/1e3:.1f} kN"
                  f"  omega max {max(r['rotor_omega_rad_s']):.0f}  height {r['origin_height_m']:.1f} m  elapsed {time.time()-tick:.0f} s",
                  flush=True)

    hook = {}
    if shear is not None:                         # plan 3.9b: the opt-in wind shear
        from openfawe.ground.turnkey import GROUND_Z_G
        from openfawe.ground.wind_shear import ShearWind
        hook["wind_shear"] = ShearWind.from_params(params, float(shear), ground_z_g_m=GROUND_Z_G)
    if turbulence is not None:                    # OF-0010 revision 23 (B2): the opt-in IEC turbulence field
        if shear is not None:
            raise ValueError("turbulence and shear are two winds; give one")
        from openfawe.ground.turnkey import GROUND_Z_G
        from openfawe.ground.turbulent_wind import TurbulentWind
        hook["wind_shear"] = TurbulentWind.from_params(params, Path(turbulence), ground_z_g_m=GROUND_Z_G, t0_s=float(turbulence_t0))
    if power:                                     # plan 3.9a: the opt-in power system
        hook["power_system"] = power_system_config(discs, inertia)
    if segment is not None and str(segment).startswith("whole"):     # plan 3.8b: the whole flight on vortex methods
        from openfawe.flight.makani_interface import G_TO_WORLD
        from openfawe.ground.perch import PerchedSystem, makani_wind_g
        sys.path.insert(0, str(TASK))
        import turnkey_whole_vortex
        base, platform_step, platform_seeding, viscous, local = whole_segment(segment)
        whole_mode = {"whole": ("vortex", False, False), "whole+orth": ("vortex", True, False), "whole+gpu": ("vortex", False, True),
                      "whole+orth+gpu": ("vortex", True, True), "whole-record": ("record", False, False)}.get(base)
        if whole_mode is None or (platform_step is not None and whole_mode[0] != "vortex"):
            raise ValueError(f"unknown segment {segment!r} (whole, whole+orth, whole+gpu, whole+orth+gpu or whole-record; "
                             "'+p1', '+p2', '+p1m', '+p2m', '+p1s' or '+p2s' (each optionally followed by 'v', then by 'l') after "
                             "one of the first four)")
        hook["segment"] = turnkey_whole_vortex.WholeFlightVortex(
            PerchedSystem(params).body, G_TO_WORLD@makani_wind_g(params), discs, {v: k for k, v in names.items()}, aero, mode=whole_mode[0],
            density_kg_m3=float(params.raw["phys_sim"]["air_density"]), orthonormalise=whole_mode[1], gpu=whole_mode[2],
            **({} if platform_step is None else dict(platform=platform_step, **({} if platform_seeding == 'zero' else dict(seeding=platform_seeding)),
                                                     **({} if not viscous else dict(viscosity=True)),
                                                     **({} if not local else dict(local_strip=True)))))
    elif segment is not None:                     # plan 3.8a: the segment hook in the given mode
        from openfawe.flight.makani_interface import G_TO_WORLD
        from openfawe.ground.perch import PerchedSystem, makani_wind_g
        sys.path.insert(0, str(TASK))
        import turnkey_vortex
        mode, orthonormalise = turnkey_vortex.parse_segment(segment)   # 'vortex+orth': the opt-in attitude orthonormalisation
        hook["segment"] = (turnkey_vortex.VortexSegment(PerchedSystem(params).body, G_TO_WORLD@makani_wind_g(params),
                                                         int(api.kFlightModeCrosswindNormal), mode=mode,
                                                         density_kg_m3=float(params.raw["phys_sim"]["air_density"]),
                                                         **(dict(orthonormalise=True) if orthonormalise else {})))
    if contact:
        result = run_turnkey(params, api, aero, discs, inertia, blown_wing=blown, max_periods=max_periods, on_record=on_record,
                             keep_records=False, tether_ground_contact=True, **hook)
    else:
        result = run_turnkey(params, api, aero, discs, inertia, blown_wing=blown, max_periods=max_periods, on_record=on_record,
                             keep_records=False, **hook)
    if buffer:
        records.write("\n".join(buffer) + "\n")
    records.close()
    write_tail()
    (folder/f"{tag}-final.json").write_text(json.dumps(dict(end=result["end"], settings=result["settings"], init=result["init"],
                                                            wall_s=result["wall_s"], periods=count[0]), default=str) + "\n",
                                           encoding="utf-8")
    print("  end", json.dumps(result["end"]), f"wall {result['wall_s']:.0f} s", flush=True)


def power_system_config(discs, inertia, compiled=False):
    """Plan 3.9a's power system for the M600 rotor set: the case files, the discs' directions, the plant's inertia.
    ``compiled`` (OF-0014 revision 2, opt-in): integrate it on the compiled path (``makani_power_jit``), bit for bit."""
    from dataclasses import replace
    from openfawe.power.makani_power_sys import PowerSystemConfig
    config = PowerSystemConfig.from_files(REPO/"cases/powertrain/makani_m600_power_sys.json", REPO/"cases/powertrain/makani_m600_motors.json",
                                          [int(d.direction) for d in discs], [float(inertia)]*len(discs))
    return replace(config, compiled=True) if compiled else config


def run_flight(out, tag, build, *, blown, max_periods, trial, contact=False, segment=None, power=False, shear=None,
               turbulence=None, turbulence_t0=30.0):
    start = provenance(build)
    log = out/f"{tag}.log"
    for name in (f"run-{tag}.json", f"{tag}-records.jsonl", f"{tag}-tail.json", f"{tag}-final.json"):
        if (out/name).exists():
            (out/name).unlink()
    tick = time.time()
    with open(log, "w", encoding="utf-8") as stream:
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", tag, str(out), str(build),
                               "on" if blown else "off", str(max_periods or 0), "contact" if contact else "nocontact",
                               f"segment={segment}", *(["power=on"] if power else []),
                               *([f"shear={float(shear)!r}"] if shear is not None else []),
                               *([f"turbulence={Path(turbulence).resolve()}", f"turbulence_t0={float(turbulence_t0)!r}"]
                                 if turbulence is not None else [])], cwd=REPO,
                              env=child_env(build),
                              stdout=stream, stderr=subprocess.STDOUT, text=True)
    final = out/f"{tag}-final.json"
    receipt = dict(tag=tag, blown_wing=bool(blown), max_periods=max_periods, trial=bool(trial), returncode=done.returncode,
                   **(dict(tether_ground_contact=True) if contact else {}), **(dict(segment=segment) if segment else {}),
                   **(dict(power_system=True) if power else {}),
                   **(dict(wind_shear_exponent=float(shear)) if shear is not None else {}),
                   **(dict(turbulence=str(turbulence), turbulence_sha256=sha(Path(turbulence)), turbulence_t0_s=float(turbulence_t0))
                      if turbulence is not None else {}),
                   seconds=time.time() - tick, final=json.loads(final.read_text(encoding="utf-8")) if final.exists() else None,
                   log_tail=log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:],
                   records_sha256=sha(out/f"{tag}-records.jsonl") if (out/f"{tag}-records.jsonl").exists() else None,
                   tail_sha256=sha(out/f"{tag}-tail.json") if (out/f"{tag}-tail.json").exists() else None,
                   provenance_start=start, provenance=provenance(build))
    (out/f"run-{tag}.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
    return receipt


def unit_part(out, build, module="tests.test_turnkey", name="unit", expected=UNIT_TESTS):
    start = provenance(build)
    target = out/f"{name}-outcome.json"
    for old in (target, out/f"{name}.json"):
        if old.exists():
            old.unlink()
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--unit-child", str(target), module], cwd=REPO,
                          env=child_env(build), capture_output=True, text=True, errors="replace")
    (out/f"{name}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else dict(outcome={}, ran=0)
    passed = bool(done.returncode == 0 and len(data["outcome"]) == expected and all(v == "ok" for v in data["outcome"].values()))
    receipt = dict(passed=passed, expected_tests=expected, returncode=done.returncode, **data, provenance_start=start,
                   provenance=provenance(build))
    (out/f"{name}.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    return receipt


def p5_part(out, build, prefix="p5", contact=False):
    if (out/f"{prefix}.json").exists():
        (out/f"{prefix}.json").unlink()
    a = run_flight(out, f"{prefix}a", build, blown=False, max_periods=P5_PERIODS, trial=False, contact=contact)
    b = run_flight(out, f"{prefix}b", build, blown=False, max_periods=P5_PERIODS, trial=False, contact=contact)
    lines = [(out/f"{t}-records.jsonl").read_text(encoding="utf-8").splitlines() if (out/f"{t}-records.jsonl").exists() else []
             for t in (f"{prefix}a", f"{prefix}b")]
    first_difference = next((i for i, (x, y) in enumerate(zip(*lines)) if x != y), None)
    ended = all(r["final"] is not None and r["final"]["end"]["reason"] == "max_periods" and r["returncode"] == 0 for r in (a, b))
    # "past the first node activation" (P5): the tether's first active node index falls below 19 within the periods.
    first_node = next((json.loads(x)["k"] for x in lines[0] if json.loads(x)["start_index_end"] < 19), None)
    passed = bool(ended and len(lines[0]) == len(lines[1]) == P5_PERIODS and first_difference is None and first_node is not None)
    receipt = dict(passed=passed, periods=[len(x) for x in lines], first_difference=first_difference, both_ended_normally=ended,
                   first_node_activation_period=first_node,
                   records_sha256=[a["records_sha256"], b["records_sha256"]], provenance_start=a["provenance_start"],
                   provenance=b["provenance"])
    if contact:
        receipt["tether_ground_contact"] = True
    (out/f"{prefix}.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    return receipt


# --------------------------------------------------------------------------------------------------------- judgement
KEEP = ("k", "t", "flight_mode", "gates", "gs_mode", "start_index_end", "free_length", "free_length_end", "tension_bridle_N",
        "rotor_omega_rad_s", "rotor_thrust_N", "rotor_power_W", "gain_ramp_scale", "tension_cmd_N", "est_tension_N", "payout_m",
        "wing_pos_g_cmd", "Xg", "origin_height_m", "min_node_height_m", "slack_segments", "tether_release", "operator_flags",
        "non_finite", "est_airspeed_m_s", "est_alpha_rad", "thrust_moment", "thrust_moment_avail", "platform_azi",
        "airspeed_m_s", "alpha_rad", "beta_rad", "blend_weight", "contacts", "incomplete", "drum_angle", "rotor_table_saturated",
        "controller_flight_mode", "rotor_command_rad_s", "run_motors", "stop_motors", "thrust_coeff_used",
        "rotor_axial_freestream_m_s", "flaps_acting_rad", "flaps_command_rad", "position_cm_m", "rotation",
        # revision 7, option 甲 (present only in records of runs with the ground contact; review GC-01)
        "ground_contact_nodes", "ground_contact_nodes_active", "max_penetration_m", "max_penetration_active_m",
        "max_penetration_staging_m", "deepest_node_index", "deepest_node_xy_m", "ground_force_N")


def load_records(path, keep=KEEP):
    records = []
    if not Path(path).exists():
        return records
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                r = json.loads(line)
                records.append({key: r[key] for key in keep if key in r})
    return records


def flight_records(out, tag):
    """The records of a flight: the appended file, then the periods of the tail file that come after it."""
    records = load_records(out/f"{tag}-records.jsonl")
    tail = out/f"{tag}-tail.json"
    if tail.exists():
        last = records[-1]["k"] if records else -1
        records += [{key: r[key] for key in KEEP if key in r} for r in json.loads(tail.read_text(encoding="utf-8")) if r["k"] > last]
    return records


def transitions(records):
    out = []
    for previous, r in zip(records, records[1:]):
        if r["flight_mode"] != previous["flight_mode"]:
            out.append(r)
    return out


def p1(receipt, records):
    final = receipt["final"]
    end = None if final is None else final["end"]
    reasons = []
    if receipt["returncode"] != 0:
        reasons.append(f"child exit code {receipt['returncode']}")
    if end is None:
        last = records[-1]["k"] if records else None
        end = dict(reason="failure", type="the child process ended without its final file (an assert of the library, or a crash)",
                   last_recorded_period=last, failing_period=None if last is None else last + 1)
        reasons.append(f"no final file: {end}")
    elif end["reason"] not in ("gain_ramp_done", "descend_timeout"):
        reasons.append(f"end reason {end['reason']}: {end}")
    nonfinite = [r["k"] for r in records if r.get("non_finite")]
    if nonfinite:
        reasons.append(f"non-finite values from period {nonfinite[0]}")
    if any(r.get("incomplete") for r in records):
        reasons.append("an incomplete record (a failure inside a period)")
    released = [r["k"] for r in records if r["tether_release"]]
    if released:
        reasons.append(f"tether_release at period {released[0]}")
    forced = [r["k"] for r in records if any(r["operator_flags"])]
    if forced:
        reasons.append(f"an operator flag set at period {forced[0]}")
    return dict(passed=not reasons, end=end, reasons=reasons)


def p2(records, names, settings):
    from openfawe.ground.turnkey import MODE_ORDER as order
    visited = [names.get(records[0]["flight_mode"])] if records else []
    rows, reasons = [], []
    for r in transitions(records):
        to = names.get(r["flight_mode"], str(r["flight_mode"]))
        frm = visited[-1]
        visited.append(to)
        row = dict(period=r["k"], t=r["t"], from_mode=frm, to_mode=to, gate_word=r["gates"][r["flight_mode"]])
        if frm == "PilotHover" and to == "Perched":
            row["gate_checked"] = False                          # the planner's initial switch (F4-09)
        else:
            row["gate_checked"] = True
            if row["gate_word"] != 0:
                reasons.append(f"{frm} -> {to} at period {r['k']} with gate word {row['gate_word']}")
        if to == "TransIn" and row["gate_word"] != 0:
            row["forced"] = "TransIn pitched-forward force (trans_in_mode.c:69-84)"
        if to == "HoverTransOut" and (r["est_airspeed_m_s"] > settings["transout_airspeed_m_s"]
                                      or r["est_alpha_rad"] < settings["transout_alpha_rad"]):
            row["forced"] = "crosswind forced hand-off on time in flare (crosswind_mode.c:115-127)"
        if "forced" in row:
            reasons.append(f"entered {to} by Makani's forced path: {row['forced']}")
        rows.append(row)
    if visited != list(order):
        reasons.append(f"mode sequence {visited} is not {list(order)}")
    return dict(passed=not reasons, sequence=visited, transitions=rows, reasons=reasons)


def p3(records, names):
    from openfawe.ground.gs02 import MODE_HIGH_TENSION, MODE_REEL, MODE_TRANSFORM
    reasons, out = [], {}
    modes = [names.get(r["flight_mode"]) for r in records]

    def first(name):
        return next((i for i, m in enumerate(modes) if m == name), None)

    def last(name):
        return next((len(modes) - 1 - i for i, m in enumerate(reversed(modes)) if m == name), None)

    def changes(values):
        return sum(1 for a, b in zip(values, values[1:]) if a != b)
    a0, a1, b0 = first("HoverAscend"), last("HoverFullLength"), first("HoverPrepTransformGsDown")
    if a0 is None or a1 is None:
        reasons.append("HoverAscend or HoverFullLength not reached")
    else:
        values = [records[i].get("start_index_end") for i in range(a0, a1 + 1)]
        if None in values:
            reasons.append("a record without the index at the end of its period (a failed period)")
            values = [v for v in values if v is not None] or [None]
        out["out"] = dict(first=values[0], last=values[-1], changes=changes(values),
                          monotone=all(b <= a for a, b in zip(values, values[1:])))
        if not (out["out"]["monotone"] and out["out"]["changes"] == 19 and values[0] == 19 and values[-1] == 0):
            reasons.append(f"Indices.start from HoverAscend to HoverFullLength: {out['out']}")
    if b0 is None:
        reasons.append("HoverPrepTransformGsDown not reached")
    else:
        values = [records[i].get("start_index_end") for i in range(b0, len(records))]
        if None in values:
            reasons.append("a record without the index at the end of its period (a failed period)")
            values = [v for v in values if v is not None] or [None]
        out["back"] = dict(first=values[0], last=values[-1], changes=changes(values),
                           monotone=all(b >= a for a, b in zip(values, values[1:])))
        if not (out["back"]["monotone"] and out["back"]["changes"] == 19):
            reasons.append(f"Indices.start from HoverPrepTransformGsDown to the end: {out['back']}")
    # CrosswindNormal by either label (the planner's after the step, the controller's of the step): the stricter set.
    cw = [r for r, m in zip(records, modes) if m == "CrosswindNormal" or names.get(r.get("controller_flight_mode")) == "CrosswindNormal"]
    if not cw:
        reasons.append("CrosswindNormal not reached")
    else:
        worst = max(max(abs(r["free_length"] - FREE_LENGTH_M), abs(r.get("free_length_end", math.inf) - FREE_LENGTH_M)) for r in cw)
        out["crosswind_free_length_max_difference_m"] = worst
        if not worst <= FREE_LENGTH_TOL_M:
            reasons.append(f"free length in CrosswindNormal differs from 425.8 m by {worst:.4f} m")
    gs_changes, previous = [], None
    for r in records:
        if r["gs_mode"] != previous:
            gs_changes.append(dict(period=r["k"], t=r["t"], gs_mode=r["gs_mode"], flight_mode=names.get(r["flight_mode"])))
            previous = r["gs_mode"]
    out["gs_mode_changes"] = gs_changes
    expected = [MODE_REEL, MODE_TRANSFORM, MODE_HIGH_TENSION, MODE_TRANSFORM, MODE_REEL]
    if [c["gs_mode"] for c in gs_changes] != expected:
        reasons.append(f"ground-station modes {[c['gs_mode'] for c in gs_changes]} are not {expected}")
    return dict(passed=not reasons, reasons=reasons, **out)


def per_mode(records, names, function, key="controller_flight_mode"):
    """``function`` of the records of every mode, by the mode the controller ran in (Makani's scores read it so)."""
    out = {}
    for name in dict.fromkeys(names.get(r[key]) for r in records):
        selected = [r for r in records if names.get(r[key]) == name]
        out[name] = function(selected)
    return out


def p6(records, names):
    low_origin = min((r["origin_height_m"], r["k"]) for r in records) if records else (None, None)
    nodes = [(r["min_node_height_m"], r["k"]) for r in records if r["min_node_height_m"] is not None]
    low_node = min(nodes) if nodes else (None, None)
    reasons = []
    if not records or not low_origin[0] > 0.0:
        reasons.append(f"body origin at or below the ground: {low_origin}")
    if nodes and not low_node[0] > 0.0:
        reasons.append(f"an active tether node at or below the ground: {low_node}")
    agl = {}
    for label, modes in (("HoverFullLength (479-483)", ("HoverFullLength",)),
                         ("HoverTransOut, HoverPrepTransformGsDown (484-490)", ("HoverTransOut", "HoverPrepTransformGsDown"))):
        values = [r["min_node_height_m"] for r in records if names.get(r["controller_flight_mode"]) in modes
                  and r["min_node_height_m"] is not None]
        agl[label] = dict(min_m=min(values) if values else None, bad_m=1.0, good_m=4.0)
    slack = per_mode(records, names, lambda rs: dict(max=max(r["slack_segments"] for r in rs),
                                                      mean=float(np.mean([r["slack_segments"] for r in rs]))))
    return dict(passed=not reasons, reasons=reasons, lowest_origin=dict(height_m=low_origin[0], period=low_origin[1]),
                lowest_active_node=dict(height_m=low_node[0], period=low_node[1]),
                tether_height_agl_min_report=agl, slack_segments_report=slack)


def p7(records, names):
    from openfawe.ground import turnkey_scoring as S
    group = ("HoverPayOut", "HoverPrepTransformGsUp", "HoverTransformGsUp", "HoverFullLength", "HoverPrepTransformGsDown",
             "HoverTransformGsDown", "HoverReelIn")
    out = dict(what="report for the simulation-process document, section 6 item 3 (the 'Makani table' alternative: this run)",
               tension_control_rms_kn_makani_group=S.tension_control_rms_kn(records, S.select(records, names, group)),
               tension_control_rms_kn_by_mode={m: S.tension_control_rms_kn(records, S.select(records, names, (m,))) for m in group},
               limits_kn=dict(good=2.0, bad=5.0))
    hover = [m for m in dict.fromkeys(names.get(r["controller_flight_mode"]) for r in records) if m and m.startswith("Hover")]
    altitude = {}
    for m in hover:
        error = np.array([r["wing_pos_g_cmd"][2] - r["Xg"][2] for r in records if names.get(r["controller_flight_mode"]) == m])
        altitude[m] = dict(max_abs_m=float(np.max(np.abs(error))), rms_m=float(np.sqrt(np.mean(error**2))), periods=int(len(error)))
    out["altitude_error_by_mode"] = altitude
    return out


def on_perch(records, reference_time_s=2.0):
    """Largest displacement of the centre of mass in the platform frame and largest attitude change relative to the
    platform (deg), from the state at ``reference_time_s`` (after the drop onto the perch) to the last record given;
    the computation of part 2's G5/G7 (tests/test_perch.on_perch)."""
    from openfawe.flight.makani_interface import G_TO_WORLD
    from openfawe.ground.gs02 import angle_to_dcm

    def rel(r):
        dcm = angle_to_dcm(r["platform_azi"], 0.0, 0.0, "zyx")
        return dcm@(G_TO_WORLD@np.asarray(r["position_cm_m"])), dcm@G_TO_WORLD@np.asarray(r["rotation"])
    ref = next((r for r in records if r["t"] >= reference_time_s - 1e-9), None)
    if ref is None:
        return None
    p0, R0 = rel(ref)
    disp, ang = 0.0, 0.0
    for r in records:
        if r["t"] < ref["t"]:
            continue
        pos, rot = rel(r)
        disp = max(disp, float(np.linalg.norm(pos - p0)))
        ang = max(ang, math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R0.T@rot) - 1.0)/2.0)))))
    return dict(from_t=ref["t"], to_t=records[-1]["t"], displacement_m=disp, attitude_deg=ang,
                platform_azimuth_change_rad=records[-1]["platform_azi"] - ref["platform_azi"])


def reports(records, names, settings, receipt):
    from openfawe.ground import turnkey_scoring as S
    rep = {}

    def summary(rs):
        power = np.array([sum(r["rotor_power_W"]) for r in rs])
        return dict(first_t=rs[0]["t"], last_t=rs[-1]["t"], duration_s=rs[-1]["t"] - rs[0]["t"],
                    free_length_m=[min(r["free_length"] for r in rs), max(r["free_length"] for r in rs)],
                    tension_bridle_kN=dict(min=min(r["tension_bridle_N"] for r in rs)/1e3,
                                           mean=float(np.mean([r["tension_bridle_N"] for r in rs]))/1e3,
                                           max=max(r["tension_bridle_N"] for r in rs)/1e3),
                    shaft_power_table_W=dict(mean=float(np.mean(power)), min=float(np.min(power)), max=float(np.max(power))),
                    shaft_energy_table_J=float(np.sum(power)*0.01),
                    rotor_speed_max_rad_s=max(max(r["rotor_omega_rad_s"]) for r in rs),
                    blend_weight_max=max((r.get("blend_weight", 0.0) for r in rs), default=None),
                    table_saturated_periods=sum(1 for r in rs if any(r.get("rotor_table_saturated", []))))
    rep["by_mode"] = per_mode(records, names, summary)
    rep["note_power"] = "table shaft power, negative while the rotors drive (consume); energy = sum over periods of power x 0.01 s"
    f4 = json.loads((TASK/"results/full-cycle-part1.json").read_text(encoding="utf-8"))["F4"]["b_base_sequence"]
    entries = {row["to"]: row["t"] for row in f4}
    rep["entry_times_vs_f4_stand_in"] = [dict(mode=row["to_mode"], t=row["t"], f4_t=entries.get(row["to_mode"]))
                                         for row in p2(records, names, settings)["transitions"]]
    ascend = next((i for i, r in enumerate(records) if names.get(r["flight_mode"]) == "HoverAscend"), None)
    if ascend is not None:
        r = records[ascend]
        wind_from = math.atan2(-settings["wind_g_m_s"][1], -settings["wind_g_m_s"][0])
        rep["hover_ascend_entry"] = dict(t=r["t"], tension_bridle_kN=r["tension_bridle_N"]/1e3, platform_azimuth_rad=r["platform_azi"],
                                         wind_from_rad=wind_from,
                                         platform_azimuth_minus_wind_from_rad=(r["platform_azi"] - wind_from + math.pi) % (2*math.pi) - math.pi)
        rep["on_perch_before_hover_ascend"] = on_perch(records[:ascend + 1])
        rep["perched_position_error_m"] = dict(value=S.perched_position_error_m(records, names, settings["perched_wing_pos_p"]),
                                           good=0.38, bad=0.80)
    groups = {"Hover - Perch to CW": ("HoverAscend", "HoverPayOut", "HoverPrepTransformGsUp", "HoverTransformGsUp", "HoverFullLength"),
              "Hover - CW to Perch": ("HoverTransOut", "HoverPrepTransformGsDown", "HoverTransformGsDown", "HoverReelIn", "HoverDescend")}
    rep["saturation_duration_s"] = {f"{g} {axis}": S.saturation_duration_s(records, S.select(records, names, modes), axis,
                                                                           settings["hover_limits"])
                                    for g, modes in groups.items() for axis in ("thrust", "moment_y", "moment_z")}
    rep["saturation_limits_s"] = dict(good=0.5, bad=1.0)
    rep["wall_s"] = None if receipt["final"] is None else receipt["final"]["wall_s"]
    return rep


def judgeable(receipts, own):
    """Same commit, clean tracked tree at start and end, same file, library and binding hashes in every receipt and in
    the judging process itself (``own``: its provenance); no trial among the inputs."""
    reasons, commits, hashes, libraries, bindings = [], set(), set(), set(), set()
    for name, r in list(receipts.items()) + [("judge", dict(provenance_start=own, provenance=own))]:
        for key in ("provenance_start", "provenance"):
            p = r.get(key) or {}
            if not p.get("git_commit"):
                reasons.append(f"{name}: no commit in {key}")
            if not p.get("tree_clean"):
                reasons.append(f"{name}: tree not clean in {key}")
            commits.add(p.get("git_commit"))
            hashes.add(json.dumps(p.get("files"), sort_keys=True))
            libraries.add(p.get("library_sha256"))
            bindings.add(p.get("binding_sha256"))
        if r.get("trial"):
            reasons.append(f"{name} is a trial")
    if len(commits) != 1:
        reasons.append(f"receipts from different commits: {sorted(str(c) for c in commits)}")
    if len(hashes) != 1:
        reasons.append("receipts with different file hashes")
    if len(libraries) != 1 or None in libraries:
        reasons.append("receipts with different or missing library hashes")
    if len(bindings) != 1 or None in bindings:
        reasons.append("receipts with different or missing binding hashes")
    return reasons


def flight_files_match(out, tag, receipt):
    """The records and tail files are the ones the receipt was written with."""
    problems = []
    for kind in ("records", "tail"):
        path = out/(f"{tag}-records.jsonl" if kind == "records" else f"{tag}-tail.json")
        stored = receipt.get(f"{kind}_sha256")
        if stored is None or not path.exists() or sha(path) != stored:
            problems.append(f"{tag}: the {kind} file is not the one of its receipt")
    return problems


def criteria(receipt, records, settings):
    """P1-P4, P6, P7, the reports and the K3 read-back of one flight."""
    from openfawe.ground import turnkey_scoring as S
    j = dict(P1=p1(receipt, records))
    if settings is None:
        j["P2_to_P7"] = "not computed: the flight ended without its final file (P1 does not pass)"
        return j
    names = {int(k): v for k, v in settings["mode_names"].items()}
    j["K3_read_back"] = S.wiring_checks(records, settings)
    j.update(P2=p2(records, names, settings), P3=p3(records, names), P4=S.p4(records, names), P6=p6(records, names),
             P7=p7(records, names), reports=reports(records, names, settings, receipt))
    return j


def judge(out, build, cw10=None):
    own = provenance(build)
    files = dict(unit=out/"unit.json", p5=out/"p5.json", off=out/"run-off.json")
    receipts = {k: json.loads(f.read_text(encoding="utf-8")) for k, f in files.items() if f.exists()}
    reasons = [f"missing receipt: {k}" for k in files if k not in receipts]
    reasons += judgeable(receipts, own)
    if "unit" in receipts and not receipts["unit"]["passed"]:
        reasons.append("the unit tests did not all pass (or one was skipped)")
    off = receipts.get("off")
    if off is not None:
        reasons += flight_files_match(out, "off", off)
    j = dict(what="full-cycle contract part 3, P1-P7 (revision 6)", provenance=own, inputs={k: sha(f) for k, f in files.items() if f.exists()})
    if off is not None:
        records = flight_records(out, "off")
        j.update(criteria(off, records, None if off["final"] is None else off["final"]["settings"]))
        if j.get("K3_read_back"):
            reasons.append(f"the records do not follow K3: {j['K3_read_back'][:3]}")
        j["P5"] = receipts.get("p5", {}).get("passed")
        del records
        j["passed"] = bool(all((j.get(c) or {}).get("passed") for c in ("P1", "P2", "P3", "P4", "P6")) and j["P5"] is True)
    if (out/"run-on.json").exists():
        try:                                        # report only: a failure here must not stop the judgement of 'off'
            on = json.loads((out/"run-on.json").read_text(encoding="utf-8"))
            rep = criteria(on, flight_records(out, "on"), None if on["final"] is None else on["final"]["settings"])
            rep.update(trial=bool(on.get("trial")), files_match=not flight_files_match(out, "on", on),
                       same_commit_as_off=bool(off is not None and on["provenance"]["git_commit"] == off["provenance"]["git_commit"]))
            j["blown_wing_on_report"] = rep
        except Exception as error:                  # noqa: BLE001
            j["blown_wing_on_report"] = dict(error=f"{type(error).__name__}: {str(error)[:300]}")
    j["cw10_mode_durations"] = json.loads(Path(cw10).read_text(encoding="utf-8")) if cw10 else \
        "not available: CW-10's flight logs stay on the owner's PC; the durations are to be made there"
    j["judgeable"] = not reasons
    j["not_judgeable_because"] = reasons
    if reasons:
        j["passed"] = None
    (out/"judgement.json").write_text(json.dumps(j, indent=1, default=str) + "\n", encoding="utf-8")
    return j



def p6_prime(records, names):
    """P6' (revision 7, option 甲; only for runs with the tether ground contact): the body origin above the ground as P6;
    every active tether node no deeper than PENETRATION_LIMIT_M below it; contact time, nodes and depth by mode reported."""
    low_origin = min((r["origin_height_m"], r["k"]) for r in records) if records else (None, None)
    nodes = [(r["min_node_height_m"], r["k"]) for r in records if r["min_node_height_m"] is not None]
    low_node = min(nodes) if nodes else (None, None)
    reasons = []
    if not records or not low_origin[0] > 0.0:
        reasons.append(f"body origin at or below the ground: {low_origin}")
    if nodes and not low_node[0] >= -PENETRATION_LIMIT_M:
        reasons.append(f"an active tether node deeper than {PENETRATION_LIMIT_M} m in the ground: {low_node}")
    if any("ground_contact_nodes" not in r for r in records):
        reasons.append("records without the ground-contact fields: not a run with the ground contact")
    def by_mode(rs):
        deepest = max(rs, key=lambda r: r.get("max_penetration_m", 0.0))
        return dict(periods_in_contact=int(sum(1 for r in rs if r.get("ground_contact_nodes", 0) > 0)),
                    periods_active_in_contact=int(sum(1 for r in rs if r.get("ground_contact_nodes_active", 0) > 0)),
                    max_nodes=int(max(r.get("ground_contact_nodes", 0) for r in rs)),
                    max_penetration_active_m=float(max(r.get("max_penetration_active_m", 0.0) for r in rs)),
                    max_penetration_staging_m=float(max(r.get("max_penetration_staging_m", 0.0) for r in rs)),
                    deepest=dict(period=deepest.get("k"), node=deepest.get("deepest_node_index"), xy_m=deepest.get("deepest_node_xy_m"),
                                 penetration_m=deepest.get("max_penetration_m")),
                    max_ground_force_N=float(max(r.get("ground_force_N", 0.0) for r in rs)))
    contact = per_mode(records, names, by_mode)
    report = p6(records, names)
    return dict(passed=not reasons, reasons=reasons, limit_m=PENETRATION_LIMIT_M,
                lowest_origin=dict(height_m=low_origin[0], period=low_origin[1]),
                lowest_active_node=dict(height_m=low_node[0], period=low_node[1]), contact_by_mode=contact,
                tether_height_agl_min_report=report["tether_height_agl_min_report"], slack_segments_report=report["slack_segments_report"],
                p6_as_written=dict(passed=report["passed"], reasons=report["reasons"]))


def hover_position_error(records, names):
    """Report (revision 7, cause investigation): position command minus position in the hover modes, horizontal and
    vertical, max and RMS (g frame)."""
    out = {}
    for m in [m for m in dict.fromkeys(names.get(r["controller_flight_mode"]) for r in records) if m and m.startswith("Hover")]:
        rows = [r for r in records if names.get(r["controller_flight_mode"]) == m]
        error = np.array([np.asarray(r["wing_pos_g_cmd"]) - np.asarray(r["Xg"]) for r in rows])
        horizontal = np.linalg.norm(error[:, :2], axis=1)
        out[m] = dict(horizontal_max_m=float(horizontal.max()), horizontal_rms_m=float(np.sqrt(np.mean(horizontal**2))),
                      vertical_max_abs_m=float(np.abs(error[:, 2]).max()), periods=len(rows))
    return out


def judge_contact(out, build):
    """Revision 7, option 甲: P1-P5 and P7 as revision 6, P6' for P6, on the blown-wing-off flight with the ground contact;
    the contact-off P5 at this commit must reproduce revision 6's P5 records hash."""
    own = provenance(build)
    files = dict(unit=out/"unit.json", unitgc=out/"unitgc.json", p5=out/"p5.json", p5c=out/"p5c.json", offc=out/"run-offc.json")
    receipts = {k: json.loads(f.read_text(encoding="utf-8")) for k, f in files.items() if f.exists()}
    reasons = [f"missing receipt: {k}" for k in files if k not in receipts]
    reasons += judgeable(receipts, own)
    for name in ("unit", "unitgc"):
        if name in receipts and not receipts[name]["passed"]:
            reasons.append(f"the unit tests of {name} did not all pass (or one was skipped)")
    offc = receipts.get("offc")
    if offc is not None:
        final_settings = (offc.get("final") or {}).get("settings") or {}
        if (offc.get("blown_wing") or not offc.get("tether_ground_contact") or final_settings.get("blown_wing") is not False
                or "tether_ground_contact" not in final_settings):
            reasons.append("run-offc.json is not the blown-wing-off flight with the ground contact (receipt or child settings)")
        reasons += flight_files_match(out, "offc", offc)
    p5c = receipts.get("p5c")
    if p5c is not None:
        if not p5c.get("tether_ground_contact"):
            reasons.append("p5c.json is not a P5 with the ground contact")
        if P5_RECORDS_SHA256 in (p5c.get("records_sha256") or []):   # review GC-04: the option reached the children
            reasons.append("p5c.json has the records of the contact-off P5: the ground contact did not reach the child")
    j = dict(what="full-cycle contract part 3 with the tether ground contact (revision 7, option 甲): P1-P5, P7, P6'", provenance=own,
             inputs={k: sha(f) for k, f in files.items() if f.exists()})
    p5 = receipts.get("p5") or {}
    j["default_unchanged"] = dict(passed=bool(p5.get("passed") and p5.get("records_sha256") == [P5_RECORDS_SHA256]*2
                                              and not p5.get("tether_ground_contact")),
                                  records_sha256=p5.get("records_sha256"), expected=P5_RECORDS_SHA256)
    if offc is not None:
        records = flight_records(out, "offc")
        settings = None if offc["final"] is None else offc["final"]["settings"]
        crit = criteria(offc, records, settings)
        j.update(crit)
        if settings is not None:
            names = {int(k): v for k, v in settings["mode_names"].items()}
            j["P6_prime"] = p6_prime(records, names)
            j["hover_position_error_report"] = hover_position_error(records, names)
        if j.get("K3_read_back"):
            reasons.append(f"the records do not follow K3: {j['K3_read_back'][:3]}")
        j["P5"] = (receipts.get("p5c") or {}).get("passed")
        del records
        j["passed"] = bool(all((j.get(c) or {}).get("passed") for c in ("P1", "P2", "P3", "P4", "P6_prime"))
                           and j["P5"] is True and j["default_unchanged"]["passed"])
    if (out/"run-onc.json").exists():
        try:                                        # report only
            onc = json.loads((out/"run-onc.json").read_text(encoding="utf-8"))
            settings = None if onc["final"] is None else onc["final"]["settings"]
            records = flight_records(out, "onc")
            rep = criteria(onc, records, settings)
            if settings is not None:
                names_on = {int(k): v for k, v in settings["mode_names"].items()}
                rep["P6_prime"] = p6_prime(records, names_on)
                rep["hover_position_error_report"] = hover_position_error(records, names_on)
            rep.update(trial=bool(onc.get("trial")), files_match=not flight_files_match(out, "onc", onc),
                       same_commit_as_offc=bool(offc is not None and onc["provenance"]["git_commit"] == offc["provenance"]["git_commit"]))
            j["blown_wing_on_contact_report"] = rep
        except Exception as error:                  # noqa: BLE001
            j["blown_wing_on_contact_report"] = dict(error=f"{type(error).__name__}: {str(error)[:300]}")
    j["judgeable"] = not reasons
    j["not_judgeable_because"] = reasons
    if reasons:
        j["passed"] = None
    (out/"judgement-contact.json").write_text(json.dumps(j, indent=1, default=str) + "\n", encoding="utf-8")
    return j


def main(out, argv):
    """Entry for the command line and for the PC runner (``run_history.py OUT --script <this> ARGS``); writes
    OUT/history.json last, with the parts that ran and their outcome."""
    parser = argparse.ArgumentParser(prog="turnkey_checks.py")
    parser.add_argument("--build", required=True, help="part 1's full-controller build folder (lib/)")
    parser.add_argument("--parts", default="unit,p5,off,on,judge")
    parser.add_argument("--trial-end-s", type=float, default=None, help="development only: shortened flights, judged as nothing")
    parser.add_argument("--cw10", default=None)
    args = parser.parse_args(argv)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = Path(args.build).resolve()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    trial_periods = None if args.trial_end_s is None else int(round(args.trial_end_s/0.01))
    summary = dict(what="turnkey_checks.py run", parts=parts, trial_end_s=args.trial_end_s, provenance_start=provenance(build), outcome={})
    if "unit" in parts:
        summary["outcome"]["unit"] = unit_part(out, build)["passed"]
        print("unit", summary["outcome"]["unit"], flush=True)
    if "p5" in parts:
        summary["outcome"]["p5"] = p5_part(out, build)["passed"]
        print("P5", summary["outcome"]["p5"], flush=True)
    if "unitgc" in parts:
        summary["outcome"]["unitgc"] = unit_part(out, build, "tests.test_tether_ground_contact", "unitgc", UNIT_TESTS_GC)["passed"]
        print("unitgc", summary["outcome"]["unitgc"], flush=True)
    if "p5c" in parts:
        summary["outcome"]["p5c"] = p5_part(out, build, "p5c", contact=True)["passed"]
        print("P5 (contact)", summary["outcome"]["p5c"], flush=True)
    for tag, blown, contact in (("off", False, False), ("on", True, False), ("offc", False, True), ("onc", True, True)):
        if tag in parts:
            r = run_flight(out, tag, build, blown=blown, max_periods=trial_periods, trial=trial_periods is not None, contact=contact)
            summary["outcome"][tag] = None if r["final"] is None else r["final"]["end"]
            print(tag, json.dumps(summary["outcome"][tag]), flush=True)
    if "judgec" in parts:
        j = judge_contact(out, build)
        summary["outcome"]["judgec"] = dict(judgeable=j["judgeable"], passed=j.get("passed"), reasons=j["not_judgeable_because"],
                                            **{c: (j.get(c) or {}).get("passed") if isinstance(j.get(c), dict) else j.get(c)
                                               for c in ("P1", "P2", "P3", "P4", "P5", "P6_prime", "default_unchanged")})
        print(json.dumps(summary["outcome"]["judgec"]), flush=True)
    if "judge" in parts:
        j = judge(out, build, args.cw10)
        summary["outcome"]["judge"] = dict(judgeable=j["judgeable"], passed=j.get("passed"), reasons=j["not_judgeable_because"],
                                           **{c: (j.get(c) or {}).get("passed") if isinstance(j.get(c), dict) else j.get(c)
                                              for c in ("P1", "P2", "P3", "P4", "P5", "P6")})
        print(json.dumps(summary["outcome"]["judge"]), flush=True)
    summary["provenance"] = provenance(build)
    (out/"history.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    if os.name == "nt" and len(sys.argv) >= 2 and sys.argv[1] in ("--child", "--unit-child"):
        # Windows: msvcrt's assert and abort messages go to stderr instead of a dialog box that would hold the child until
        # someone closes it (Windows full-library contract, WF0 and review WF-R01; plan 3.8a revision 1 item 7)
        import ctypes
        ctypes.cdll.msvcrt._set_error_mode(2)
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        segment = next((a.split("=", 1)[1] for a in sys.argv[8:9] if a.startswith("segment=")), "None")
        flight_child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "on", int(sys.argv[6]) or None,
                     contact=len(sys.argv) > 7 and sys.argv[7] == "contact", segment=None if segment == "None" else segment,
                     power="power=on" in sys.argv[9:10],
                     shear=next((float(a.split("=", 1)[1]) for a in sys.argv[9:11] if a.startswith("shear=")), None),
                     turbulence=next((a.split("=", 1)[1] for a in sys.argv[9:12] if a.startswith("turbulence=")), None),
                     turbulence_t0=next((float(a.split("=", 1)[1]) for a in sys.argv[9:13] if a.startswith("turbulence_t0=")), 30.0))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--unit-child":
        unit_child(sys.argv[2], *sys.argv[3:4])
    else:
        main(sys.argv[1], sys.argv[2:])
