"""OF-0010 revision 19 addendum 7 (Studio batch 1): the whole flight's (TurnKey's) exports and the import of a flight
computed by the job runner.

- ``flat_all``: every number of a record as a column (scalars; lists and nested lists flattened as key_i, key_i_j; three
  numbers as key_x, key_y, key_z as the plots name them; dicts as key_name, review STR-02) and every text as a text column
  - the CSV of a whole flight (ST-06).
- ``export_flight_csv``: the CSV of a stored free flight, two passes over the records file.
- ``export_flight_paraview``: one set of .vtp files per stored frame - wake and rings, lifting surfaces, fuselage placed
  at the frame's pose, tether, ground station - with the trajectory of every record, collected in one multi-part .pvd
  (A2, an optional export: the studio replays its runs itself). Written with ``openfawe.aero_vtk._vtp`` (no pyvista).
- ``import_job_run``: a folder of ``platform_flight.py`` or ``table_variants.py`` (``records.jsonl.gz``, ``child.json``,
  ``history.json``, the saved wake states) as a studio run: the records unchanged, the energy account and the end as the
  status, frames from the records - the pose every ``frame_every`` periods, the tether where the record has its nodes
  (``tether_nodes_Xg_m``), the wake and rings where a wake state was saved (A4). The lifting surfaces are drawn at the
  recorded pose from the aircraft case (the saved states carry no surface geometry); the ground station is drawn at the
  ground frame's origin.

Review b1-studio1-review-20261105 (STR-01, STR-04, STR-05, STR-07): a saved wake state is the P2 group's (wing, tails and
pylons; ``wake_network``: bound row, trailing-edge row, free rows), so in its frame the surfaces are P2's at the record's
acting flaps placed at the state's own pose with the state's bound circulation, and the surface rows are P2's (2 x its node count) - as the live
frames draw ``state.wing``; the force arrow is the acting load (``held_aero_force_body_N``), the wind the job's (its
settings) or the parameter file's, the mode label the record's ``flight_mode`` (as the live frames), the tether line runs
from the ground anchor to the bridle point where the record has them (``tether_ends_Xg_m``); every PVD time step has
every part (an empty dataset where the frame has none); a truncated or damaged source ends the import as 'truncated' with
the reason, and a run without records cannot be exported.
"""
import csv
import gzip
import json
import re
from pathlib import Path

import numpy as np

FLIGHT_MODE = "free flight (TurnKey)"
IMPORTED_MODE = "free flight (TurnKey), imported from a job run"
PARTS = ("wake", "surfaces", "fuselage", "tether", "station", "trajectory")


def is_flight(run):
    return str(run.meta.get("mode", "")).startswith(FLIGHT_MODE)


def flat_all(record):
    out = {}

    def put(key, value):
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        elif value is None:
            out[key] = np.nan
        elif isinstance(value, str):                     # review STR-02: text as text (whole_model, modes, failures)
            out[key] = value
        elif isinstance(value, dict):                    # review STR-02: the power messages, whole_vsm_failure, ...
            for sub, v in value.items():
                put(f"{key}_{sub}", v)
        elif isinstance(value, (list, tuple)):
            if len(value) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                for axis, v in zip("xyz", value):
                    out[f"{key}_{axis}"] = float(v)
            else:
                for i, v in enumerate(value):
                    put(f"{key}_{i}", v)
    for key, value in record.items():
        put(key, value)
    return out


def _records(path, strict=False):
    """The records of a records file. A line that does not parse ends the reading (a line being written); with ``strict``
    it raises ValueError (review STR-07: an import must not pass a damaged file as complete)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:        # a line being written
                    if strict:
                        raise ValueError(f"{path.name}, line {number}: {error}") from error
                    return


def records_file(run, required=False):
    for name in ("records.jsonl", "records.jsonl.gz"):
        if (run.path/name).exists():
            return run.path/name
    if required:                                         # review STR-07
        raise ValueError(f"{run.path} has no records file (records.jsonl)")
    return None


def _cell(value):
    return repr(value) if isinstance(value, float) else str(value)


def export_flight_csv(run, out_path):
    file = records_file(run, required=True)
    columns = []
    seen = set()
    for record in _records(file):
        for key in flat_all(record):
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with Path(out_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for record in _records(file):
            row = flat_all(record)
            writer.writerow([_cell(row[c]) if c in row else "" for c in columns])
    return out_path


def _fuselage(rotation, position):
    from . import resources
    with np.load(resources.path(resources.FUSELAGE)) as data:
        nodes, quads = np.asarray(data["nodes"], float), np.asarray(data["quads"], np.int64)
    return nodes@np.asarray(rotation, float).T + np.asarray(position, float), quads


EMPTY = dict(wake=("Lines", ("Gamma_m2_s", "ring")), surfaces=("Polys", ("bound_Gamma_m2_s",)), fuselage=("Polys", ("panel",)),
             tether=("Lines", ("segment",)), station=("Lines", ("mast",)))


def empty_part(part):
    """Review STR-05: a part a frame does not have, as an empty dataset with the part's cell arrays."""
    kind, names = EMPTY[part]
    width = 2 if kind == "Lines" else 4
    return (np.zeros((0, 3)), np.zeros((0, width), np.int64), kind,
            {name: np.zeros(0, float if "Gamma" in name else np.int64) for name in names})


def frame_parts(frame):
    """The geometry of one stored frame as (points, cells, kind, cell_data) per part (None where the frame has none)."""
    from .runstore import surface_quads, wake_lines
    parts = {}
    edges, gamma, ring = wake_lines(frame)
    parts["wake"] = None if not len(edges) else (np.asarray(frame["nodes"], float), np.asarray(edges, np.int64), "Lines",
                                                 {"Gamma_m2_s": np.asarray(gamma, float), "ring": ring.astype(np.int64)})
    points, quads = surface_quads(frame)
    parts["surfaces"] = (points, quads, "Polys", {"bound_Gamma_m2_s": np.asarray(frame["bound_gamma"], float)})
    try:
        nodes, quads = _fuselage(frame["rotation"], frame["position"])
        parts["fuselage"] = (nodes, quads, "Polys", {"panel": np.arange(len(quads))})
    except FileNotFoundError:
        parts["fuselage"] = None
    line = np.asarray(frame.get("line", np.zeros((0, 3))), float).reshape(-1, 3)
    parts["tether"] = None if len(line) < 2 else (line, np.column_stack([np.arange(len(line)-1), np.arange(1, len(line))]),
                                                  "Lines", {"segment": np.arange(len(line)-1)})
    ground = np.asarray(frame.get("ground", np.full(3, np.nan)), float).reshape(3)
    parts["station"] = None if not np.all(np.isfinite(ground)) else (
        np.vstack([ground, ground + np.array([0.0, 0.0, 6.0])]), np.array([[0, 1]]), "Lines", {"mast": np.array([0])})
    return parts


def export_flight_paraview(run, out_dir, *, steps=None):
    """A2: one .vtp per part and stored frame, the trajectory once, and ``run.pvd`` with the parts by time."""
    from openfawe.aero_vtk import _vtp
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions = []
    from openfawe.flight.makani_interface import G_TO_WORLD
    for record in _records(records_file(run, required=True)):
        positions.append(G_TO_WORLD@np.asarray(record["Xg"], float))
    trajectory = None
    if len(positions) > 1:
        points = np.asarray(positions)
        _vtp(out/"trajectory.vtp", points, np.column_stack([np.arange(len(points)-1), np.arange(1, len(points))]), "Lines",
             {"period": np.arange(len(points)-1)}, "vsm_free_wake")
        trajectory = "trajectory.vtp"
    entries = []
    for step in (run.frame_steps if steps is None else steps):
        frame = run.frame(step)
        fields = {"period": [int(frame["step"])], "time_s": [float(frame["time_s"])]}
        for part, data in frame_parts(frame).items():
            points, cells, kind, cell_data = empty_part(part) if data is None else data
            name = f"{part}_{int(step):07d}.vtp"
            _vtp(out/name, points, cells, kind, cell_data, "vsm_free_wake", fields)
            entries.append((float(frame["time_s"]), PARTS.index(part), name))
        if trajectory:
            entries.append((float(frame["time_s"]), PARTS.index("trajectory"), trajectory))
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
    lines += [f'    <DataSet timestep="{t!r}" part="{p}" file="{f}"/>' for t, p, f in entries]
    lines += ["  </Collection>", "</VTKFile>", ""]
    (out/"run.pvd").write_text("\n".join(lines), encoding="utf-8")
    return out/"run.pvd"


# ----- import (A4) -----------------------------------------------------------------------------------------------------
def mode_names_from_binding(path=None):
    """Flight-mode number -> name from the controller binding's ``kFlightMode<Name> = N`` lines."""
    from openfawe.ground.turnkey import MODE_ORDER
    from . import resources
    path = Path(path) if path else resources.controller_build()/"lib"/"makani_full_controller_api.py"
    text = path.read_text(encoding="utf-8")
    names = {}
    for name in MODE_ORDER:
        found = re.search(rf"^kFlightMode{name} = (-?\d+)\s*$", text, re.M)
        if found:
            names[int(found.group(1))] = name
    return names


def _flight_aero(history):
    segment = str(history.get("segment") or "")
    if not segment:
        return "tables"
    if segment.endswith("t3i"):
        return "vortex_bprime"
    return "vortex_hover" if history.get("window") == "hover" else "vortex"


def _wake_states(folder):
    """Period -> path of every saved wake state; a damaged file raises ValueError before anything is written (STR-07)."""
    import zipfile
    states = {}
    for path in sorted(Path(folder).glob("**/state-*.npz")):
        try:
            with np.load(path) as data:
                states[int(data["k"])] = path
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"wake state {path.name}: {type(error).__name__}: {error}") from error
    return states


_P2 = {}


def p2_model():
    """The P2 group's surfaces (wing, tails, pylons) whose wake the job runs save (``turnkey_whole_vortex._save_state``,
    ``platform="p2"``), from the repository's files (``platform_surfaces.default_p1``)."""
    if "model" not in _P2:
        from openfawe.flight import platform_surfaces as PF
        from . import resources
        _P2["model"] = PF.default_p1(resources.data_root())
    return _P2["model"]


def p2_assembly(flaps_rad=None):
    """P2's surfaces at the period's flap deflections (the all-moving horizontal tail turns with the elevator; the other
    flaps act through the tables); the node and panel counts do not depend on them."""
    return p2_model().period(np.zeros(8) if flaps_rad is None else np.asarray(flaps_rad, float).reshape(8)).assembly


def _state_arrays(path, flaps_rad=None):
    """A saved wake state as frame arrays: wake and rings, and the P2 surfaces at the state's own pose and the record's
    acting flaps (``flaps_acting_rad``) with the state's bound circulation (review STR-01: the surface rows are P2's,
    2 x its node count)."""
    with np.load(path) as d:
        wake_nodes, wake_edges, wake_gamma = (np.asarray(d["wake_nodes"], float), np.asarray(d["wake_edges"], np.int64),
                                              np.asarray(d["wake_gamma"], float))
        ring_nodes, ring_edges, ring_gamma = (np.asarray(d["ring_nodes"], float).reshape(-1, 3),
                                              np.asarray(d["ring_edges"], np.int64).reshape(-1, 2),
                                              np.asarray(d["ring_circulation"], float))
        source = np.asarray(d["ring_source"], np.int64)
        bound = np.asarray(d["bound_circulation"], float) if "bound_circulation" in d else None
        rotation = np.asarray(d["rotation"], float) if "rotation" in d else None
        position = np.asarray(d["position_m"], float) if "position_m" in d else None
    assembly = p2_assembly(flaps_rad)
    count = int(assembly.node_count)
    if len(wake_nodes) < 2*count or len(wake_nodes) % count:
        raise ValueError(f"wake state {Path(path).name}: {len(wake_nodes)} wake nodes are not rows of the P2 surfaces' {count}")
    panels = len(assembly.panel_edges)
    if bound is not None and len(bound) != panels:
        raise ValueError(f"wake state {Path(path).name}: {len(bound)} bound circulations, P2 has {panels} panels")
    first = len(wake_nodes)
    out = dict(nodes=np.vstack([wake_nodes, ring_nodes]), edges=np.vstack([wake_edges, ring_edges + first]),
               gamma=np.concatenate([wake_gamma, ring_gamma]), ring_first=first, surface_nodes=2*count, ring_source=source)
    if rotation is not None and position is not None:
        placed = assembly.transformed(rotation, position)
        out.update(leading_edge=np.asarray(placed.leading_edge, float), trailing_edge=np.asarray(placed.trailing_edge, float),
                   panel_edges=np.asarray(placed.panel_edges, np.int64),
                   bound_gamma=np.zeros(panels) if bound is None else bound)
    return out


def _job_wind_world(child, history):
    """The job's wind (world axes) from its settings: the hook's ``wind_world_m_s`` (platform_flight) or run_turnkey's
    ``wind_g_m_s`` (table_variants); None when neither is there."""
    from openfawe.flight.makani_interface import G_TO_WORLD
    for source in (child.get("settings") or {}, history.get("settings") or {}):
        if isinstance(source, dict) and source.get("wind_world_m_s") is not None:
            return np.asarray(source["wind_world_m_s"], float).reshape(3), "the job's settings (wind_world_m_s)"
        if isinstance(source, dict) and source.get("wind_g_m_s") is not None:
            return G_TO_WORLD@np.asarray(source["wind_g_m_s"], float).reshape(3), "the job's settings (wind_g_m_s)"
    return None, None


def _parameter_wind_world():
    from openfawe.flight.makani_interface import G_TO_WORLD
    from openfawe.ground import gs02 as G
    from openfawe.ground.perch import makani_wind_g
    from . import resources
    params = G.load_gs02_params(resources.path("cases/ground/makani_gs02_parker_ranch.json"))
    return G_TO_WORLD@np.asarray(makani_wind_g(params), float)


def import_job_run(job_folder, out_root, *, frame_every=100, name=None):
    """A4: the job runner's folder as a new studio run under ``out_root``; returns its path."""
    from openfawe.aero.aircraft import load_aircraft
    from openfawe.flight.makani_interface import G_TO_WORLD
    from openfawe.ground.turnkey import GROUND_Z_G
    from . import __version__, resources
    from .runstore import new_run_dir
    from .spec import AIRCRAFT, Spec
    from .turnkey_worker import flight_frame
    job = Path(job_folder)
    source = job/"records.jsonl.gz" if (job/"records.jsonl.gz").exists() else job/"records.jsonl"
    if not source.exists():
        raise ValueError(f"{job} has no records.jsonl(.gz)")
    history = json.loads((job/"history.json").read_text(encoding="utf-8")) if (job/"history.json").exists() else {}
    child = json.loads((job/"child.json").read_text(encoding="utf-8")) if (job/"child.json").exists() else {}
    states = _wake_states(job)                           # before anything is written (STR-07)
    wind, wind_source = _job_wind_world(child, history)
    if wind is None:
        wind, wind_source = _parameter_wind_world(), "the parameter file (makani_gs02_parker_ranch.json); the job states none"
    aero = _flight_aero(history)
    spec = Spec(name=name or f"imported {job.name}", motion="free_flight", flight_aero=aero,
                compute="cpu_tree", power_system=bool(history.get("power") or child.get("energy")),
                flight_frame_every=int(frame_every), wake_frame_every=int(frame_every))
    path = new_run_dir(out_root, spec.name)
    path.mkdir(parents=True)
    (path/"frames").mkdir()
    names = mode_names_from_binding()
    meta = dict(app="OpenFAWE Studio", version=__version__, spec=spec.to_dict(), mode=IMPORTED_MODE,
                source=dict(folder=str(job), history={k: history.get(k) for k in ("what", "segment", "window", "provenance",
                                                                                    "change", "returncode", "seconds")}),
                mode_names={str(k): v for k, v in names.items()}, case_file=AIRCRAFT[spec.aircraft]["file"],
                reference_area_m2=AIRCRAFT[spec.aircraft]["area_m2"], reference_chord_m=AIRCRAFT[spec.aircraft]["chord_m"],
                wind_world_m_s=wind.tolist(), wind_source=wind_source,
                frames="TurnKey world (z up, ground at -6.122 m); surfaces drawn from the case at the recorded pose, in wake "
                       "frames P2's at the saved state's pose; the ground station drawn at the ground frame's origin")
    (path/"spec.json").write_text(json.dumps(meta, indent=1, default=str)+"\n", encoding="utf-8")
    wing = load_aircraft(resources.path(AIRCRAFT[spec.aircraft]["file"])).wing
    ground = np.array([0.0, 0.0, -GROUND_Z_G])
    count, last, stored, problem = 0, None, 0, None
    wake_errors = []
    with (path/"records.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        try:
            for record in _records(source, strict=True):
                out.write(json.dumps(record)+"\n")
                k = int(record["k"])
                count, last = count + 1, record
                if k % int(frame_every) and k not in states:
                    continue
                rotation = np.asarray(record["rotation"], float)
                position = G_TO_WORLD@np.asarray(record["Xg"], float)
                arrays, _ = flight_frame(wing, None, rotation, position)
                if k in states:
                    try:
                        arrays.update(_state_arrays(states[k], record.get("flaps_acting_rad")))
                    except (OSError, ValueError, KeyError, EOFError) as error:
                        wake_errors.append(f"period {k}: {type(error).__name__}: {error}"[:200])
                nodes = record.get("tether_nodes_Xg_m")
                line = np.zeros((0, 3)) if not nodes else (G_TO_WORLD@np.asarray(nodes, float).reshape(-1, 3).T).T
                ends = record.get("tether_ends_Xg_m")
                if ends is not None:                     # STR-04: from the ground anchor to the bridle point
                    start, end = (G_TO_WORLD@np.asarray(ends, float).reshape(2, 3).T).T
                    line = np.vstack([start[None], line, end[None]])
                number = record.get("flight_mode", record.get("controller_flight_mode"))
                mode = names.get(number, str(number))
                force = record.get("held_aero_force_body_N", record.get("aero_force_body_N", record.get("held_force_body_N")))
                frame = dict(step=k, time_s=float(record["t"]), rotation=rotation, position=position, **arrays,
                             force=rotation@np.asarray([0.0, 0.0, 0.0] if force is None else force, float), line=line,
                             ground=ground, wind=wind, mode=mode,
                             label=mode + (f" [{record.get('whole_model')}]" if "whole_model" in record else ""))
                np.savez_compressed(path/"frames"/f"frame_{k:06d}.npz",
                                    **{key: value for key, value in frame.items() if key != "label"})
                stored += 1
        except (ValueError, KeyError, EOFError, OSError) as error:     # a truncated gzip, a damaged line (STR-07)
            problem = f"{type(error).__name__}: {error}"[:300]
    complete = history.get("status", "complete") == "complete"
    state = "truncated" if problem else ("complete" if complete else str(history.get("status")))
    status = dict(state=state, periods=count, end=child.get("end") or history.get("end"),
                  energy=child.get("energy") or history.get("energy"), imported_frames=stored, wake_states=sorted(states),
                  last_period=None if last is None else int(last["k"]), **({"import_error": problem} if problem else {}),
                  **({"wake_state_errors": wake_errors} if wake_errors else {}))
    (path/"status.json").write_text(json.dumps(status, indent=1, default=str)+"\n", encoding="utf-8")
    return path
