"""Free flight in the studio: the whole-flight TurnKey (openfawe.ground.turnkey.run_turnkey) with Makani's controller -
perch, hover take-off, transition, crosswind loops, landing - with the ground station, winch, lumped tether, rotors and
motors computed, and the Makani aerodynamic database as the aerodynamics (as plan 3.8's flights).

OF-0010 revision 13: ``spec.flight_aero`` 'vortex' or 'vortex_hover' adds the segment hook of the platform path
(``turnkey_whole_vortex.WholeFlightVortex``) built exactly as ``platform_flight.child`` builds it for the segment
'whole+orth+p2sl' (or 'whole+orth+gpu+p2sl' with the CPU + GPU choice): 'vortex' is paper 2's flight B (hover and perch
on the tables, the blown-wing term only there), 'vortex_hover' every flying mode on the vortex method (P5d-U3's set-up).
``spec.power_system`` adds plan 3.9's power system (``turnkey_checks.power_system_config``); the records then carry
``studio_ground_power_kW``, ``studio_motor_power_kW`` and ``studio_ground_energy_MJ`` (derived, for the plots) and the
status the mode-by-mode energy account of ``platform_flight.energy_account``. In the vortex periods the 3D frame holds
the vortex state's wake and rotor rings, in the table periods the hook's own rings; frames with a wake or rings are
stored at most every ``spec.wake_frame_every`` periods.

OF-0010 revision 19 addendum 7 (Studio batch 1): 'vortex_bprime' is paper 2's flight B' (the T3 slipstream and the induced
hub inflow, ``platform_flight``'s segment 'whole+orth+p2sl+t3i' or 'whole+orth+gpu+p2sl+t3i'); a frame without the wake
every ``spec.flight_frame_every`` periods; ``run_turnkey(record_nodes_every=flight_frame_every)`` so that the drawn tether
is the period's start (ST-01: the plant's first evaluation of the period, the record's own ``tether_nodes_Xg_m``);
``run_turnkey``'s settings written to ``settings.json`` and the wind of the parameters to ``spec.json`` (ST-02); the
records flushed every ``FLUSH_EVERY`` periods (ST-07).

The flight runs in this process, a fresh one per flight (the controller library keeps state between calls). The records
are run_turnkey's own (``on_record``). For the 3D view the worker also reads the tether's node positions where the plant
evaluates them: ``FlyingSystem.rates`` is wrapped in this process only, to keep the newest ``nodes`` and the tether's
end points; the wrapper returns what the plant returns and changes nothing.

Frames: TurnKey's world frame (x, y, z up; the ground at z = -6.122 m at Parker Ranch). TurnKey's body frame is the
frame of cases/aircraft/m600_makani_rigid_body.json, x_downstream_y_span_z_up - the frame of the aerodynamic cases and of
the fuselage mesh - so the airframe is drawn with the record's rotation as it is (checked: the record's centre of mass
in body axes equals the rigid-body file's).
"""
import json
import os
import time as _time
from pathlib import Path

import numpy as np

DATA = "cases/polars/makani"
WAKE_FRAME_EVERY = 200                                # the default of spec.wake_frame_every (a frame with the wake: ~10 MB)
FLUSH_EVERY = 100                                     # records written to disk every that many periods (ST-07)
STATUS_EVERY_PERIODS, STATUS_EVERY_S = 1000, 60.0     # status.json with the energy so far (review STU-02)


def vortex_hook(spec, params, api, aero, discs):
    """The segment hook of ``platform_flight.child`` for spec.flight_aero 'vortex' / 'vortex_hover' and its blown-wing
    setting (``run_turnkey(blown_wing=...)``)."""
    from . import resources
    resources.task_modules()                          # ARC-01 (OF-0010 revision 19 addendum 26)
    import turnkey_whole_vortex as W
    from openfawe.flight.makani_interface import G_TO_WORLD
    from openfawe.ground.perch import PerchedSystem, makani_wind_g
    from openfawe.ground.turnkey import mode_names
    numbers = {v: k for k, v in mode_names(api).items()}
    args = (PerchedSystem(params).body, G_TO_WORLD@makani_wind_g(params), discs, numbers, aero)
    kwargs = dict(mode="vortex", density_kg_m3=float(params.raw["phys_sim"]["air_density"]), orthonormalise=True,
                  gpu=spec.compute == "cpu_gpu", platform="p2", seeding="strip", local_strip=True,
                  **({} if spec.keep_rows is None else dict(keep_rows=int(spec.keep_rows))),
                  **(dict(hover_tables=True) if spec.flight_aero in ("vortex", "vortex_bprime") else {}),
                  **(dict(slipstream="t3", hub_inflow="induced") if spec.flight_aero == "vortex_bprime" else {}))
    return W.WholeFlightVortex(*args, **kwargs), ("tables" if spec.flight_aero in ("vortex", "vortex_bprime") else False)


def flight_frame(wing, hook, rotation, position):
    """The wake, rings and lifting surfaces of one free-flight period for the 3D view: (arrays, has_wake)."""
    n0 = len(wing.panel_edges)
    out = dict(nodes=np.zeros((0, 3)), edges=np.zeros((0, 2), dtype=np.int64), gamma=np.zeros(0), ring_first=0,
               surface_nodes=0, ring_source=np.zeros(0, dtype=np.int64), leading_edge=wing.leading_edge@rotation.T+position,
               trailing_edge=wing.trailing_edge@rotation.T+position, panel_edges=np.asarray(wing.panel_edges, dtype=np.int64),
               bound_gamma=np.zeros(n0))
    if hook is None:
        return out, False
    state = None if hook.main is None else getattr(hook.main, "state", None)
    if hook.kind == "vsm" and state is not None:          # the vortex state: its own wake and the rotor rings
        from openfawe.aero.wake import extended_network
        nodes, edges, gamma, ring_first = extended_network(state)
        rings = state.rings
        out.update(nodes=np.asarray(nodes, dtype=float), edges=np.asarray(edges, dtype=np.int64),
                   gamma=np.asarray(gamma, dtype=float), ring_first=int(ring_first), surface_nodes=2*int(state.wing.node_count),
                   ring_source=np.zeros(0, dtype=np.int64) if rings is None or not len(rings.edges) else np.asarray(rings.source, dtype=np.int64),
                   leading_edge=np.asarray(state.wing.leading_edge, dtype=float),
                   trailing_edge=np.asarray(state.wing.trailing_edge, dtype=float),
                   panel_edges=np.asarray(state.wing.panel_edges, dtype=np.int64),
                   bound_gamma=np.asarray(state.solution.circulation, dtype=float))
        return out, True
    rings = hook.rings                                    # a table period: the rings the hook carries on
    if rings is not None and rings.count:
        out.update(nodes=np.asarray(rings.nodes, dtype=float), edges=np.asarray(rings.edges, dtype=np.int64),
                   gamma=np.asarray(rings.circulation, dtype=float), ring_first=0, surface_nodes=0,
                   ring_source=np.asarray(rings.source, dtype=np.int64))
        return out, True
    return out, False


def library_sha256(build):
    """ST-07: the SHA-256 of the controller library file in ``build/lib`` (None when there is none to read)."""
    import hashlib
    for pattern in ("*makani_full_controller*.dll", "*makani_full_controller*.so", "*makani_full_controller*.dylib"):
        for file in sorted((Path(build)/"lib").glob(pattern)):
            if file.is_file():
                return dict(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    return None


def default_build():
    """The Makani controller library: the Windows build in third_party on Windows, OPENFAWE_MAKANI_BUILD elsewhere."""
    from . import resources
    if os.environ.get("OPENFAWE_MAKANI_BUILD"):
        return Path(os.environ["OPENFAWE_MAKANI_BUILD"])
    return resources.controller_build()               # third_party/makani-controller-win-full unless packaged (OF-0013)


def run(spec_dict, run_dir, messages, frames, pause, stop, threads=0):
    from .worker import _offer
    try:
        from . import resources
        resources.ensure_library()
        if os.name == "nt":                             # msvcrt asserts to stderr, not a dialog box (turnkey_checks)
            import ctypes
            ctypes.cdll.msvcrt._set_error_mode(2)
        from openfawe.aero.aircraft import load_aircraft
        from openfawe.aero.makani_database import MakaniDatabase
        from openfawe.flight.makani_full import load_full_api
        from openfawe.flight.makani_interface import G_TO_WORLD
        from openfawe.ground import gs02 as G
        from openfawe.ground import turnkey as T
        from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set
        from openfawe.rotor.table import RotorTable
        from .spec import AIRCRAFT, Spec
        spec = Spec.from_dict(spec_dict)
        build = Path(spec.makani_build) if spec.makani_build else default_build()
        api = load_full_api(build)
        params = G.load_gs02_params(resources.path("cases/ground/makani_gs02_parker_ranch.json"))
        aero = MakaniDatabase.from_files(resources.path(f"{DATA}/m600_aswing_baseline.json"),
                                         spoiler_path=resources.path(f"{DATA}/spoiler_offset_tables.json"),
                                         high_path=resources.path(f"{DATA}/hover_model_sn02_wind_10_no_wake_model.json"))
        discs = load_rotor_set(resources.path(resources.ROTOR_SET), RotorTable.from_makani_json(resources.path(resources.ROTOR_TABLE)))
        inertia = load_rotor_inertia(resources.path(resources.ROTOR_SET))
        names = T.mode_names(api)
        from openfawe.ground.perch import makani_wind_g
        wind_g = makani_wind_g(params)
        wing = load_aircraft(resources.path(AIRCRAFT[spec.aircraft]["file"])).wing
        if threads:
            import numba
            numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
        hook, blown = (None, bool(spec.blown_wing)) if spec.flight_aero == "tables" else vortex_hook(spec, params, api, aero, discs)
        extra = {}
        if spec.power_system:                             # plan 3.9, as platform_flight's --power on
            resources.task_modules()                  # ARC-01 (OF-0010 revision 19 addendum 26)
            import turnkey_checks as TC
            import platform_flight as PFL
            extra = dict(power_system=TC.power_system_config(discs, inertia))
            account, cumulative = {}, dict(ground=0.0, motor=0.0)
    except Exception as error:  # noqa: BLE001
        import traceback
        messages.put(("error", f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"))
        return

    seen = dict(armed=True)
    original = T.FlyingSystem.rates

    def watched(self, state, nodes, speeds, gsx):         # observation only: the plant's own result is returned
        result = original(self, state, nodes, speeds, gsx)
        if seen["armed"]:                             # ST-01: the period's first evaluation, at its start (the
            seen["nodes"], seen["tether"] = nodes, result[-1]["tether"]   # record's pose); on_record arms the next
            seen["armed"] = False
        return result
    T.FlyingSystem.rates = watched

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=False)
    (path/"frames").mkdir()
    from . import __version__
    from .runstore import library_identity
    phys = params.raw["phys_sim"]
    meta = dict(app="OpenFAWE Studio", version=__version__, spec=spec.to_dict(), mode="free flight (TurnKey)", makani_build=str(build),
                makani_library_sha256=library_sha256(build),
                library=library_identity(),
                wind=dict(note="the free flight takes the wind of the GS02 parameter file, not the set-up's inflow fields (ST-02)",
                          speed_m_s=float(phys["wind_speed"]), direction_rad=float(phys["wind_direction"]),
                          air_density_kg_m3=float(phys["air_density"])),
                aerodynamics="Makani aerodynamic database (m600_aswing_baseline, spoiler offsets, hover model)"
                if hook is None else f"the platform path's vortex hook ({spec.flight_aero}; OF-0010 revisions 11 and 13)",
                hook=None if hook is None else hook.settings(), power_system=bool(spec.power_system),
                mode_names={str(k): v for k, v in names.items()},
                blown_wing=blown, case_file=AIRCRAFT[spec.aircraft]["file"],
                reference_area_m2=AIRCRAFT[spec.aircraft]["area_m2"], reference_chord_m=AIRCRAFT[spec.aircraft]["chord_m"],
                frames="TurnKey world (z up, ground at -6.122 m); body frame x_downstream_y_span_z_up (the case frame)")
    (path/"spec.json").write_text(json.dumps(meta, indent=1, default=str)+"\n", encoding="utf-8")
    records = (path/"records.jsonl").open("w", encoding="utf-8", newline="\n")
    status = dict(state="running", periods=0)
    clock = dict(last=0.0, tick=_time.time(), status=_time.time())
    shown = dict(frame=None, stored=None, errors=0)

    def write_status():
        if spec.power_system:
            status["energy"] = PFL.energy_report(account)
        if hook is not None:
            status.update(hook_periods=dict(hook.periods), fallbacks=len(hook.fallbacks), fallback_first=hook.fallbacks[:5])
        tmp = path/"status.json.tmp"
        tmp.write_text(json.dumps(status, indent=1, default=str)+"\n", encoding="utf-8")
        os.replace(tmp, path/"status.json")

    def store(frame):
        saver = np.savez_compressed if len(frame["nodes"]) else np.savez
        saver(path/"frames"/f"frame_{frame['step']:06d}.npz", **{key: value for key, value in frame.items() if key != "label"})
        shown["stored"] = frame["step"]
    messages.put(("started", str(path), dict(count=int(spec.max_periods or T.END_PERIODS), method="turnkey", compute="cpu")))
    every = max(1, int(spec.flight_frame_every))
    wake_every = max(every, int(spec.wake_frame_every))

    class Stopped(Exception):
        pass

    def on_record(record):
        if stop.is_set():
            raise Stopped()
        while pause.is_set() and not stop.is_set():
            stop.wait(0.1)
        k = int(record["k"])
        if spec.power_system and "power_ground_energy_J" in record:     # revision 13: derived for the plots
            PFL.energy_account(account, cumulative, record, names)
            electrical = record.get("power_electrical_J")
            electrical = float(sum(electrical) if isinstance(electrical, list) else (electrical or 0.0))
            record.update(studio_ground_power_kW=float(record["power_ground_energy_J"])/T.PERIOD_S/1e3,
                          studio_motor_power_kW=electrical/T.PERIOD_S/1e3, studio_ground_energy_MJ=cumulative["ground"]/1e6)
        records.write(json.dumps(record, default=float)+"\n")
        if (k+1) % FLUSH_EVERY == 0:                  # ST-07: a crash keeps the records up to the last hundred periods
            records.flush()
        status["periods"] = k+1
        row = dict(record, time_s=float(record["t"]), mode_name=names.get(record["flight_mode"], str(record["flight_mode"])))
        tick = _time.time()
        seconds, clock["previous"] = tick-clock.get("previous", tick), tick          # the wall time of this period
        messages.put(("step", k, row, seconds))
        now = _time.time()
        if (k+1) % STATUS_EVERY_PERIODS == 0 or now-clock["status"] > STATUS_EVERY_S:
            clock["status"] = now
            try:
                write_status()
            except OSError as error:                  # the flight goes on (review STU-04)
                status["status_error"] = f"{type(error).__name__}: {error}"[:200]
        wants_store = k % every == 0
        try:
            if now-clock["last"] < 0.15 and not wants_store:
                return
            clock["last"] = now
            try:                                      # display and frames only: an error here does not end the flight
                show(record, row, k, wants_store)
            except (OSError, ValueError, MemoryError) as error:
                shown["errors"] += 1
                status["frame_error"] = f"period {k}: {type(error).__name__}: {error}"[:200]
        finally:
            seen["armed"] = True                      # ST-01: the next period's first plant evaluation

    def show(record, row, k, wants_store):
        rotation_body = np.asarray(record["rotation"], dtype=float)
        rotation = rotation_body
        position = G_TO_WORLD@np.asarray(record["Xg"], dtype=float)
        line = np.zeros((0, 3))
        ground = np.full(3, np.nan)
        if "nodes" in seen:
            out = seen["tether"]
            nodes = np.asarray(seen["nodes"], dtype=float)
            active = nodes[out["indices"].start:] if out["indices"].active else np.zeros((0, 3))
            line = np.vstack([np.asarray(out["start_point"], float)[None], active, np.asarray(out["end_point"], float)[None]])
            ground = np.array([out["start_point"][0], out["start_point"][1], -T.GROUND_Z_G])   # the station on the ground
        force = rotation_body@np.asarray(record.get("held_aero_force_body_N", record.get("aero_force_body_N", record["held_force_body_N"])),
                                         dtype=float)
        arrays, has_wake = flight_frame(wing, hook, rotation, position)
        if has_wake and wants_store and k % wake_every != 0:     # a frame between the wake frames: pose and tether only
            arrays, has_wake = flight_frame(wing, None, rotation, position)
        frame = dict(step=k, time_s=float(record["t"]), rotation=rotation, position=position, **arrays,
                     force=force, line=line, ground=ground, wind=G_TO_WORLD@np.asarray(wind_g, dtype=float),
                     mode=row["mode_name"], label=row["mode_name"]+(f" [{record.get('whole_model')}]" if hook else ""))
        shown["frame"] = frame
        if wants_store:
            store(frame)
        _offer(frames, frame)

    state = "complete"
    try:
        result = T.run_turnkey(params, api, aero, discs, inertia, blown_wing=blown,
                               max_periods=int(spec.max_periods) or None, on_record=on_record, keep_records=False,
                               record_nodes_every=every, **({} if hook is None else dict(segment=hook)),
                               **({} if int(spec.max_periods or 0) <= T.END_PERIODS else      # ST-05: the UI allows more than
                                  dict(stop_rules=dict(end_periods=int(spec.max_periods)))), **extra)   # the module's 160 000
        status["end"] = result["end"]
        try:                                          # ST-02: run_turnkey's own settings (the wind, the limits, the names)
            (path/"settings.json").write_text(json.dumps(result.get("settings"), indent=1, default=str)+"\n", encoding="utf-8")
        except (OSError, TypeError, ValueError) as error:
            status["settings_error"] = f"{type(error).__name__}: {error}"[:200]
        if result["end"].get("reason") == "failure":
            state = "stopped" if result["end"].get("type") == "Stopped" else "failed"   # the stop request ends the flight
    except Stopped:
        state = "stopped"
    except Exception as error:  # noqa: BLE001
        state = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
    finally:
        T.FlyingSystem.rates = original
        records.close()
        if shown["frame"] is not None and shown["stored"] != shown["frame"]["step"]:
            try:                                      # the last frame shown (review STU-03)
                store(shown["frame"])
            except (OSError, ValueError, MemoryError):
                pass
        status.update(state=state, wall_s=_time.time()-clock["tick"], frame_errors=shown["errors"])
        write_status()
    if state == "failed":
        messages.put(("failed", status["periods"], "turnkey", str(status.get("end") or status.get("error"))))
    else:
        messages.put(("finished", state))
