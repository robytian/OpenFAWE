"""Bounded, read-only discovery and comparison of saved OpenFAWE aero runs.

Discovery reads small JSON metadata only. Full archive hashes are checked by
AeroReader when a run is actually opened; a declared completion is not validation.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .aero_archive import AeroReader, SUPPORTED_SCHEMAS, METHOD_LABELS
from .provenance import sha256_file


SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", "refs",
                       "raw-sources", "raw", "data", "code", "paraview",
                       "figures", "figures-final", ".codex", ".openai"})
MAX_FRAMES = 10000
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def _json(path, limit):
    if path.stat().st_size > limit:
        raise ValueError(f"metadata exceeds {limit} bytes: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"object metadata required: {path.name}")
    return data


def _fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    path: Path
    name: str
    model: str = "Unknown"
    method: str = "Unknown"
    core: str = "Unknown"
    status: str = "invalid"
    detail: str = ""
    frames: int = 0
    case: dict = field(default_factory=dict, compare=False, repr=False)
    manifest: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def ready(self):
        return self.status == "complete / unchecked"


@dataclass(frozen=True)
class Catalog:
    records: tuple
    notices: tuple


def inspect_run(directory):
    """Read metadata without opening HDF5; keep invalid/incomplete candidates."""
    path = Path(directory).resolve()
    values = dict(path=path, name=path.name)
    try:
        manifest = _json(path / "manifest.json", 2 * 1024 * 1024)
        values["manifest"] = manifest
        if manifest.get("schema") not in SUPPORTED_SCHEMAS:
            raise ValueError("not an OpenFAWE aerodynamic archive schema")
        case = _json(path / "case.json", 16 * 1024 * 1024)
        values.update(case=case, name=str(case.get("name", path.name)))
        method = case.get("aero_method", "free_wake")
        if method not in METHOD_LABELS:
            raise ValueError("unsupported aerodynamic method")
        aircraft = case.get("aircraft_input_json")
        if isinstance(aircraft, str):
            aircraft = json.loads(aircraft)
        model = aircraft.get("name") if isinstance(aircraft, dict) else None
        if not model:
            geometry = str(case.get("geometry", ""))
            model = "Analytical ellipse" if "ellip" in geometry.lower() else case.get("model", case.get("name", "Unknown"))
        selected = case.get("selected_surfaces")
        excluded = case.get("excluded_source_surfaces", [])
        if excluded:
            model = f"{model} ({len(selected or [])} surfaces; subset)"
        aero = case.get("aerodynamics") or {}
        if method == "vsm":
            core = "VSM / " + str(manifest.get("vsm_core_policy", "unrecorded"))
        else:
            wake = case.get("wake") or aero.get("wake") or {}
            radius = wake.get("core_radius_m")
            radius_text = f"{radius:g} m" if isinstance(radius, (int, float)) else "unrecorded radius"
            names = {0: "None", 1: "Rankine", 2: "Lamb-Oseen", 3: "Vatistas", 4: "Offset"}
            kernel = names.get(wake.get("core_model"), "Unrecorded kernel")
            core = f"{kernel} / {wake.get('core_evolution', 'constant')} / {radius_text}"
            if wake.get("free_wake") is False:
                core += " / ambient-only convection"
            if method in ('vlm','ll_straight'):
                core += ' / prescribed finite straight wake'
        if method in ('vlm', 'vlm_free_wake'):
            coupled = manifest.get('polar_coupling_model') == 'grid_section'
            core += ' / source-polar coupled' if coupled else ' / inviscid lattice'
            if manifest.get('wake_basis') == 'linear_potential':
                core += ' / nodal linear potential'
            if manifest.get('bound_convection_core') == 'te_incident_birth':
                core += ' / TE-incident birth core in convection; ideal bound probe field'
            if manifest.get('initial_cap_contract') == 'closed_te_birth_core_v1':
                core += ' / explicit birth-core initial cap'
            if manifest.get('initial_gamma_rate_contract') == 'provided_d0_bdf2_start_v1':
                core += ' / explicit initial Gamma rate: ' + manifest.get('initial_gamma_rate_input', {}).get('provenance', 'unrecorded')
        frames = manifest.get("frames", 0)
        values.update(model=str(model), method=method, core=core,
                      frames=frames if type(frames) is int else 0)
        counterfactual = case.get('counterfactual')
        if counterfactual is not None:
            scope = counterfactual.get('scope') if isinstance(counterfactual, dict) else None
            return RunRecord(**values, status='quarantined / diagnostic',
                detail=str(scope or 'Explicit unregistered counterfactual; native replay disabled.'))
        if manifest.get("status") != "complete":
            return RunRecord(**values, status="incomplete / " + str(manifest.get("status", "unknown")),
                             detail=str(manifest.get("error", "No complete saved trajectory; replay disabled.")))
        if type(frames) is not int or not 1 <= frames <= MAX_FRAMES:
            raise ValueError(f"frame count outside platform limit 1..{MAX_FRAMES}")
        if sha256_file(path / "case.json") != manifest.get("case_sha256"):
            raise ValueError("archive hash mismatch: case.json")
        wake_path = path / "wake.h5"
        if not wake_path.is_file():
            raise ValueError("missing wake.h5")
        if wake_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("wake.h5 exceeds platform 256 MiB loading limit")
        return RunRecord(**values, status="complete / unchecked",
                         detail="Complete in manifest. Full content hashes checked on opening.")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return RunRecord(**values, status="invalid", detail=str(error))


def discover_runs(roots, *, max_depth=8, max_directories=10000, max_runs=500):
    """Scan selected roots, excluding reference/cache trees and directory links.

    Limits stop the scan with an explicit notice; no HDF5 arrays are read here.
    An explicitly selected run is accepted even when its parent name is excluded.
    """
    for value in (max_depth, max_directories, max_runs):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer catalog limits required")
    stack = [(Path(p).resolve(), 0) for p in reversed(tuple(roots))]
    seen, records, notices = set(), [], []
    while stack:
        path, depth = stack.pop()
        if path in seen:
            continue
        if len(seen) >= max_directories or len(records) >= max_runs:
            notices.append("Scan limit reached; select a narrower results directory for more runs.")
            break
        seen.add(path)
        if not path.is_dir():
            notices.append(f"Directory unavailable: {path}")
            continue
        try:
            manifest = path / "manifest.json"
            candidate = (path / "wake.h5").exists()
            if manifest.is_file() and not candidate:
                try:
                    candidate = str(_json(manifest, 2 * 1024 * 1024).get("schema", "")).startswith("openfawe.aero")
                except (OSError, ValueError):
                    candidate = (path / "case.json").is_file()
            if candidate:
                records.append(inspect_run(path))
                continue
            if depth >= max_depth:
                notices.append(f"Depth limit reached: {path}")
                continue
            with os.scandir(path) as entries:
                children = sorted((Path(e.path) for e in entries if e.is_dir(follow_symlinks=False)
                                   and e.name not in SKIP_DIRS and not e.name.startswith(".")
                                   and not getattr(os.path, "isjunction", lambda _: False)(e.path)), key=lambda p: p.name.casefold())
            stack.extend((child, depth + 1) for child in reversed(children))
        except OSError as error:
            notices.append(f"Cannot inspect {path}: {error}")
    return Catalog(tuple(sorted(records, key=lambda r: (r.model, str(r.path)))), tuple(notices))


def filter_runs(records, *, model=None, method=None, core=None):
    return tuple(r for r in records if (model is None or r.model == model)
                 and (method is None or r.method == method) and (core is None or r.core == core))


def comparison_notes(records):
    """Compare recorded identities only; do not imply a matched physical test."""
    if len(records) < 2:
        return "Check at least two completed runs to compare saved loads."
    geometry_keys = ("aircraft_input_sha256", "geometry", "panels", "selected_surfaces", "surface_names", "polar")
    forcing_keys = ("inflow_m_s", "wind_m_s", "base_position_m", "heave_amplitude_m", "heave_frequency_hz",
                    "prescribed_motion", "motion", "alpha_deg", "beta_deg", "structural_velocity")
    geometry = [{k: r.case[k] for k in geometry_keys if k in r.case} for r in records]
    forcing = [{k: r.case[k] for k in forcing_keys if k in r.case} for r in records]
    notes = []
    for name, values in (("geometry/polar", geometry), ("forcing", forcing)):
        if any(not v for v in values):
            notes.append(f"Missing {name} metadata; matching is unconfirmed.")
        elif len({_fingerprint(v) for v in values}) > 1:
            notes.append(f"Different recorded {name}; exploratory comparison only.")
        else:
            notes.append(f"Recorded {name} metadata match.")
    lattice = [r for r in records if r.method in ('vlm','vlm_free_wake')]
    modes = {r.manifest.get('polar_coupling_model','none') for r in lattice}
    if len(modes)>1:
        notes.append('Different lattice closures: inviscid and source-polar coupled runs.')
    if lattice:
        notes.append('VLM/UVLM default to inviscid thin lifting surfaces. Optional grid_section coupling uses source polars and virtual boundary normals; inspect each run settings. Identical input polar metadata does not mean identical physics.')
    if any(r.method == 'vlm_free_wake' for r in records) and any(r.method != 'vlm_free_wake' for r in records):
        notes.append('UVLM includes a circulation-rate normal load; LL/VSM loads are quasi-steady sectional loads without this term. Unsteady load curves are not identical force models.')
    return " ".join(notes) + " Each curve uses its own physical times; no resampling or validation claim."


@dataclass(frozen=True)
class LoadHistory:
    times: np.ndarray
    forces: np.ndarray


class HistoryCache:
    """At most eight copied load histories; never retains open HDF5 readers."""
    def __init__(self, capacity=8):
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError("history cache capacity must be 1..8")
        self.capacity, self._items = capacity, OrderedDict()

    def load(self, record):
        current = inspect_run(record.path)
        if not current.ready:
            raise ValueError(current.detail)
        key = (record.path, current.manifest.get("wake_sha256"), current.manifest.get("case_sha256"),
               (record.path / "wake.h5").stat().st_mtime_ns)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        reader = AeroReader(record.path)
        try:
            times, forces = reader.times.copy(), reader.forces.copy()
            if not np.isfinite(forces).all() or forces.shape != (len(times), 3):
                raise ValueError("nonfinite or invalid saved load history")
            times.flags.writeable = False
            forces.flags.writeable = False
            history = LoadHistory(times, forces)
        finally:
            reader.close()
        self._items[key] = history
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return history

    def clear(self):
        self._items.clear()
