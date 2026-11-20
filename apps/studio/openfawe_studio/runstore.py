"""A studio run on disk, and the exports of a run.

Layout of a run folder (written while the run goes, so a stopped or failed run keeps what was solved):

    spec.json         the Spec, the studio version, the library's source identity and the environment
    records.jsonl     one step record per line (openfawe.flight.prescribed.step_record, plus the wall time 'seconds')
    frames/           frame_<step>.npz: the arrays of engine.frame for every 'frame_every'-th step and the last
    status.json       running / complete / stopped / failed, steps done, error, wall time

Exports: CSV of the records with the coefficients; ParaView (one wake .vtp and one surface .vtp per stored frame and a
.pvd collection). The whole flight's exports and the import of a job run are in freeflight_io.py.
"""
import csv
import datetime as _dt
import json
import re
import time as _time
from pathlib import Path

import numpy as np

from . import APP_NAME, __version__
from .spec import Spec

FRAME_KEYS = ("rotation", "position", "nodes", "edges", "gamma", "ring_source", "leading_edge", "trailing_edge",
              "panel_edges", "bound_gamma", "force", "line", "ground")


def _write_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False, default=str)+"\n", encoding="utf-8")
    temporary.replace(path)


def new_run_dir(root, name):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:40] or "run"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root)/f"{stamp}-{slug}"
    k = 1
    while path.exists():
        k += 1
        path = Path(root)/f"{stamp}-{slug}-{k}"
    return path


def library_identity():
    try:
        from openfawe import provenance
        return dict(source=provenance.source_identity(), environment=provenance.environment())
    except Exception as error:  # noqa: BLE001  (the identity is a record, never a reason to stop a run)
        return dict(error=f"{type(error).__name__}: {error}")


class RunWriter:
    def __init__(self, path, spec, built):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/"frames").mkdir()
        self.spec = spec
        self.started = _time.time()
        self.steps = 0
        meta = dict(app=APP_NAME, version=__version__, spec=spec.to_dict(), case_file=built.case_file,
                    reference_area_m2=built.area_m2, reference_chord_m=built.chord_m,
                    aerodynamics=built.backend.metadata(), motion=built.motion.description,
                    wind=built.wind.describe() if hasattr(built.wind, "describe") else f"{float(built.wind[0]):g} m/s along +x",
                    rotors=None if built.rotors is None else dict(discs=len(built.rotors.discs),
                                                                  omega_rad_s=float(built.spec.rotor_omega_rad_s)),
                    tether=None if built.tether is None else built.tether.settings(built.backend.wake.dt_s),
                    not_in_the_aerodynamics=["fuselage (drawn only)", "the tether's load does not move the aircraft (motion is prescribed)"],
                    library=library_identity(), started_utc=_dt.datetime.now(_dt.timezone.utc).isoformat())
        _write_json(self.path/"spec.json", meta)
        self.records = (self.path/"records.jsonl").open("w", encoding="utf-8", newline="\n")
        self.status("running")

    def status(self, state, error=None):
        _write_json(self.path/"status.json", dict(state=state, steps_done=self.steps, error=error,
                                                  wall_s=_time.time()-self.started))

    def append(self, record, seconds):
        self.records.write(json.dumps(dict(record, seconds=seconds), allow_nan=False)+"\n")
        self.records.flush()
        self.steps += 1
        self.status("running")                       # a run that is killed keeps an honest count

    def save_frame(self, frame):
        arrays = {k: frame[k] for k in FRAME_KEYS}
        np.savez(self.path/"frames"/f"frame_{frame['step']:06d}.npz", step=frame["step"], time_s=frame["time_s"],
                 ring_first=frame["ring_first"], surface_nodes=frame["surface_nodes"], **arrays)

    def close(self, state, error=None):
        self.records.close()
        self.status(state, error)


THIN_ABOVE_BYTES, THIN_ROWS = 50_000_000, 20_000    # a long free flight is read thinned (OF-0010 revision 13, STU-01)


class RunFolder:
    """A stored run, complete or not. ``records=False`` reads no records (frames and status only); a records file larger
    than ``THIN_ABOVE_BYTES`` is read every ``stride``-th line (and the last), at most about ``THIN_ROWS`` records."""

    def __init__(self, path, records=True):
        self.read_records = bool(records)
        self.stride = 1
        self.path = Path(path)
        if not (self.path/"spec.json").exists():
            raise ValueError(f"{self.path} is not a studio run (no spec.json)")
        self.meta = json.loads((self.path/"spec.json").read_text(encoding="utf-8"))
        self.spec = Spec.from_dict(self.meta["spec"])
        self.refresh()

    def refresh(self):
        self.records = []
        file = self.path/"records.jsonl"
        if self.read_records and file.exists():
            if file.stat().st_size > THIN_ABOVE_BYTES:
                with file.open("r", encoding="utf-8") as stream:
                    count = sum(1 for _ in stream)
                self.stride = max(1, -(-count//THIN_ROWS))
            with file.open("r", encoding="utf-8") as stream:
                last = None
                for i, line in enumerate(stream):
                    if not line.strip():
                        continue
                    if i % self.stride and self.stride > 1:
                        last = line
                        continue
                    try:
                        self.records.append(json.loads(line))
                        last = None
                    except json.JSONDecodeError:            # a line being written
                        last = None
                        break
                if last is not None:
                    try:
                        self.records.append(json.loads(last))
                    except json.JSONDecodeError:
                        pass
        self.frame_steps = sorted(int(p.stem.split("_")[1]) for p in (self.path/"frames").glob("frame_*.npz"))
        status = self.path/"status.json"
        self.status = json.loads(status.read_text(encoding="utf-8")) if status.exists() else dict(state="unknown")
        if self.status.get("state") == "running":
            self.status["state"] = "running or interrupted"

    @property
    def area_m2(self):
        return float(self.meta["reference_area_m2"])

    @property
    def chord_m(self):
        return float(self.meta["reference_chord_m"])

    def frame(self, step):
        with np.load(self.path/"frames"/f"frame_{int(step):06d}.npz") as data:
            frame = {k: data[k] for k in data.files}
        for key in ("step", "ring_first", "surface_nodes"):
            frame[key] = int(frame[key])
        frame["time_s"] = float(frame["time_s"])
        return frame


def flat_rows(records, area_m2, chord_m):
    from .engine import flat_record
    rows = [dict(flat_record(r, area_m2, chord_m), seconds=float(r.get("seconds", np.nan))) for r in records]
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns, rows


def _all_records(run):
    """Every record of the run's file, one at a time (a thinned RunFolder still exports them all)."""
    file = run.path/"records.jsonl"
    if not file.exists():
        return
    with file.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return


def export_csv(run, out_path):
    from .engine import flat_record
    columns = []
    for record in _all_records(run):                    # two passes, so that a long flight is never held in memory
        for key in dict(flat_record(record, run.area_m2, run.chord_m), seconds=0.0):
            if key not in columns:
                columns.append(key)
    with Path(out_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for record in _all_records(run):
            row = dict(flat_record(record, run.area_m2, run.chord_m), seconds=float(record.get("seconds", np.nan)))
            writer.writerow([repr(row.get(c, "")) if isinstance(row.get(c), float) else row.get(c, "") for c in columns])
    return out_path


def wake_lines(frame, *, include_surface=False):
    """Edges to draw as wake filaments: the lifting-surface rows and the rings left out (they are drawn as such)."""
    edges = frame["edges"]
    if len(edges) == 0:
        return edges, frame["gamma"], np.zeros(0, dtype=bool)
    ring = edges[:, 0] >= frame["ring_first"]
    keep = np.ones(len(edges), dtype=bool)
    if not include_surface:
        keep &= ~((edges[:, 0] < frame["surface_nodes"]) & (edges[:, 1] < frame["surface_nodes"]))
    return edges[keep], frame["gamma"][keep], ring[keep]


def surface_quads(frame):
    le, te, panels = frame["leading_edge"], frame["trailing_edge"], frame["panel_edges"]
    n = len(le)
    points = np.vstack([le, te])
    quads = np.column_stack([panels[:, 0], panels[:, 1], n+panels[:, 1], n+panels[:, 0]])
    return points, quads


def export_paraview(run, out_dir, *, steps=None):
    import pyvista as pv
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for step in (run.frame_steps if steps is None else steps):
        frame = run.frame(step)
        edges, gamma, ring = wake_lines(frame)
        lines = np.column_stack([np.full(len(edges), 2), edges]).ravel() if len(edges) else np.zeros(0, dtype=np.int64)
        wake = pv.PolyData(frame["nodes"], lines=lines)
        wake.cell_data["Gamma_m2_s"] = gamma
        wake.cell_data["ring"] = ring.astype(np.uint8)
        wake.save(out/f"wake_{step:06d}.vtp", binary=True)
        points, quads = surface_quads(frame)
        surface = pv.PolyData(points, faces=np.column_stack([np.full(len(quads), 4), quads]).ravel())
        surface.cell_data["bound_Gamma_m2_s"] = frame["bound_gamma"]
        surface.save(out/f"surfaces_{step:06d}.vtp", binary=True)
        entries.append((frame["time_s"], step))
    collection = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
    for time_s, step in entries:
        collection.append(f'    <DataSet timestep="{time_s!r}" part="0" file="wake_{step:06d}.vtp"/>')
        collection.append(f'    <DataSet timestep="{time_s!r}" part="1" file="surfaces_{step:06d}.vtp"/>')
    collection += ["  </Collection>", "</VTKFile>", ""]
    (out/"run.pvd").write_text("\n".join(collection), encoding="utf-8")
    return out/"run.pvd"
