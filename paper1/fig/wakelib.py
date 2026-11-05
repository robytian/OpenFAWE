"""Shared helpers for the paper-1 model and wake figures (paper session, 2026-10-02).

Reads the aircraft cases and the stored free-wake states of OpenFAWE and turns them
into surfaces for rendering. No aerodynamic algorithm lives here: the wake lattice and
its ring circulations are read back from the stored filament network (nodes, edges,
gamma), which wake.pack_rings wrote as rows of width wing.node_count:
row 0 = bound vortex (quarter chord), row 1 = trailing edge, rows 2.. = free rows.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper1/scripts/convergence"))
from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.lifting_surface import LiftingSurfaceConfig, build_lifting_surface_mesh  # noqa: E402
from openfawe.aero.vsm import geometry as vsm_geometry  # noqa: E402

RUNS = Path(__import__("os").environ.get("OPENFAWE_RUNS", REPO / "local/accel-20261001/runs"))
CASES = {"v3": REPO / "cases/aircraft/v3_cad_camber_integral.json",
         "m600": REPO / "cases/qualification/m600_makani_reconstructed.json"}
ALPHA = {"v3": 5.412846420261046, "m600": 0.0}
STATE = {"v3": RUNS / "p1-v3-planes-wakelength", "m600": RUNS / "p1-m600-planes-wakelength"}

INK = "#1a1a1a"
SURFACE = "#c9ced6"
BOUND = "#c0392b"
TRAIL = "#1f5fa8"
POINT = "#111111"
LOAD = "#e69f00"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aircraft(key):
    return load_aircraft(CASES[key])


def freestream(key, speed):
    from openfawe.aero.reference import inflow
    return np.asarray(inflow(speed, ALPHA[key], 0.0), dtype=float)


def surfaces(wing):
    """List of single surfaces of a wing or wing assembly."""
    return list(getattr(wing, "wings", [wing]))


def quads(rows):
    """Faces array (pyvista) for a (m, n, 3) structured sheet of points."""
    m, n = rows.shape[:2]
    index = np.arange(m * n).reshape(m, n)
    cells = np.stack([index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], axis=-1).reshape(-1, 4)
    return np.hstack([np.full((len(cells), 1), 4), cells]).ravel()


def spline_span(rows, factor=8):
    """Display only: cubic spline through the spanwise nodes of every row, (m, n, 3) -> (m, factor*(n-1)+1, 3).

    The computational geometry is piecewise linear between the nodes; the spline is used to draw the
    curved kite smoothly and passes through every node.
    """
    from scipy.interpolate import CubicSpline
    rows = np.asarray(rows, dtype=float)
    middle = rows[len(rows) // 2]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(middle, axis=0), axis=1))]
    fine = np.concatenate([np.linspace(s[i], s[i + 1], factor, endpoint=False) for i in range(len(s) - 1)] + [[s[-1]]])
    return np.stack([CubicSpline(s, row, axis=0)(fine) for row in rows])


def surface_mesh(wing, chordwise=8, smooth=False):
    """pyvista PolyData of the mean surface of every lifting surface (cambered where the case has camber)."""
    import pyvista as pv
    parts = []
    for part in surfaces(wing):
        mesh = build_lifting_surface_mesh(part, LiftingSurfaceConfig(chordwise_panels=chordwise))
        rows = mesh.vertices.reshape(chordwise + 1, part.node_count, 3)
        if smooth:
            rows = spline_span(rows)
        parts.append(pv.PolyData(rows.reshape(-1, 3), quads(rows)))
    return parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]


def strip_outline(wing, chordwise=8, smooth=False):
    """Line segments of the spanwise panel boundaries and the leading/trailing edges (on the mean surface)."""
    lines = []
    for part in surfaces(wing):
        mesh = build_lifting_surface_mesh(part, LiftingSurfaceConfig(chordwise_panels=chordwise))
        rows = mesh.vertices.reshape(chordwise + 1, part.node_count, 3)
        for j in range(part.node_count):
            lines.append(rows[:, j])
        edges = spline_span(rows[[0, -1]]) if smooth else rows[[0, -1]]
        lines.append(edges[0])
        lines.append(edges[1])
    return lines


def polyline(points_list):
    """pyvista PolyData made of several open polylines."""
    import pyvista as pv
    points, cells, start = [], [], 0
    for line in points_list:
        line = np.asarray(line, dtype=float)
        points.append(line)
        cells.append(np.r_[len(line), np.arange(start, start + len(line))])
        start += len(line)
    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.hstack(cells)
    return poly


def read_state(key, step):
    """Stored free-wake state -> rows (depth+1, width, 3), ring strengths (depth, panels), chord, speed."""
    data = np.load(STATE[key] / f"state_{step:04d}.npz")
    wing = aircraft(key).wing
    width = int(data["first"])
    assert width == wing.node_count
    rows = data["nodes"].reshape(-1, width, 3)
    depth = len(rows) - 1
    lookup = {(int(a), int(b)): g for (a, b), g in zip(data["edges"], data["gamma"])}
    panel_edges = np.asarray(wing.panel_edges)
    assert np.all(panel_edges[:, 0] < panel_edges[:, 1])
    jump = np.zeros((depth + 1, len(panel_edges)))
    for r in range(depth + 1):
        for j, (left, right) in enumerate(panel_edges):
            jump[r, j] = lookup.get((r * width + left, r * width + right), 0.0)
    strengths = np.cumsum(jump, axis=0)
    closure = np.abs(strengths[-1]).max()
    assert closure < 1e-9 * max(1.0, np.abs(strengths).max()), closure      # the last row closes every column
    return dict(rows=rows, strengths=strengths[:-1], wing=wing, chord=float(data["chord"]), speed=float(data["speed"]),
                nodes=data["nodes"], edges=data["edges"], gamma=data["gamma"], radii=data["radii"],
                core_model=int(data["core_model"]), path=STATE[key] / f"state_{step:04d}.npz")


def wake_sheet(rows, strengths, panel_edges, first_row=1, last_row=None):
    """pyvista PolyData of the wake lattice (rows first_row..last_row) with the ring circulation as cell data."""
    import pyvista as pv
    last_row = len(rows) - 1 if last_row is None else min(last_row, len(rows) - 1)
    width = rows.shape[1]
    points = rows[first_row:last_row + 1].reshape(-1, 3)
    cells, values = [], []
    for i in range(last_row - first_row):
        for j, (left, right) in enumerate(panel_edges):
            cells.append([4, i * width + left, i * width + right, (i + 1) * width + right, (i + 1) * width + left])
            values.append(strengths[first_row + i, j])
    sheet = pv.PolyData(points, np.asarray(cells).ravel())
    sheet.cell_data["gamma"] = np.asarray(values)
    return sheet


def straight_rows(wing, direction, length, count):
    """Rows of a prescribed straight wake: the trailing edge carried along the stream direction."""
    stations = np.linspace(0.0, length, count + 1)
    return np.concatenate(([wing.bound], wing.trailing_edge[None] + stations[:, None, None] * direction[None, None]), axis=0)


def flat_mesh(wing, smooth=False):
    """pyvista PolyData of the flat chord strips (leading edge to trailing edge), as seen by the line models."""
    import pyvista as pv
    parts = []
    for part in surfaces(wing):
        rows = np.stack([part.leading_edge, part.trailing_edge])
        if smooth:
            rows = spline_span(rows)
        parts.append(pv.PolyData(rows.reshape(-1, 3), quads(rows)))
    return parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]


def flat_outline(wing, smooth=False):
    lines = []
    for part in surfaces(wing):
        lines += [np.stack([part.leading_edge[j], part.trailing_edge[j]]) for j in range(part.node_count)]
        edges = np.stack([part.leading_edge, part.trailing_edge])
        edges = spline_span(edges) if smooth else edges
        lines += [edges[0], edges[1]]
    return lines


def smooth_line(points, factor=8):
    """Display only: cubic spline through the points of a polyline."""
    return spline_span(np.asarray(points, dtype=float)[None], factor)[0]


def trailing_strength(strengths, panel_edges, width):
    """Circulation of the trailing filament on every node column, per row interval: (depth, width)."""
    out = np.zeros((len(strengths), width))
    for p, (left, right) in enumerate(np.asarray(panel_edges)):
        out[:, right] += strengths[:, p]
        out[:, left] -= strengths[:, p]
    return out


def trailing_filaments(rows, strengths, panel_edges, first_row=1, last_row=None):
    """pyvista PolyData of the trailing filaments (one polyline per node column) with |circulation| as point data."""
    import pyvista as pv
    last_row = len(rows) - 1 if last_row is None else min(last_row, len(rows) - 1)
    width = rows.shape[1]
    value = np.abs(trailing_strength(strengths, panel_edges, width))[first_row:last_row]      # per interval
    points, cells, scalars, start = [], [], [], 0
    for j in range(width):
        line = rows[first_row:last_row + 1, j]
        v = value[:, j]
        node = np.r_[v[0], 0.5 * (v[:-1] + v[1:]), v[-1]]
        points.append(line)
        scalars.append(node)
        cells.append(np.r_[len(line), np.arange(start, start + len(line))])
        start += len(line)
    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.hstack(cells)
    poly.point_data["gamma"] = np.hstack(scalars)
    return poly


def shed_filaments(rows, strengths, panel_edges, first_row=1, last_row=None, every=4):
    """pyvista PolyData of shed (spanwise) filaments on every `every`-th wake row, with |circulation| as point data.

    The filament of panel p on row r is shared by the rings r-1 and r and carries their difference; the last row of
    the wake closes the oldest ring and carries its full strength (the starting vortex).
    """
    import pyvista as pv
    final = len(rows) - 1
    last_row = final if last_row is None else min(last_row, final)
    picked = list(range(first_row + every, last_row + 1, every))
    if last_row == final and final not in picked:
        picked.append(final)
    points, cells, scalars, start = [], [], [], 0
    for r in picked:
        value = np.abs(strengths[r - 1]) if r == final else np.abs(strengths[r] - strengths[r - 1])
        for p, (left, right) in enumerate(np.asarray(panel_edges)):
            points.append(rows[r, [left, right]])
            scalars.append([value[p], value[p]])
            cells.append([2, start, start + 1])
            start += 2
    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.hstack(cells)
    poly.point_data["gamma"] = np.hstack(scalars)
    return poly
