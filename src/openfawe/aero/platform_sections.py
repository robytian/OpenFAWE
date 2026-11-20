"""Platform model P1 (OF-0010 contract, revision 1 with its addenda, and revision 2): one section table per section at every incidence,
built for every flap row of Makani's XFOIL tables; the control surfaces in the vortex method; the elevator as the whole
horizontal tail turned.

* **Tables** (``FlapTables``): for each airfoil of Makani's hover model (google/makani 1d66923,
  ``analysis/aero/hover_model/hover_model.py`` 105-152: main wing outer ``oref`` and inner ``ref``, horizontal tail
  ``f8``, vertical tail ``approx_blade_rj8``, pylon ``mp6d``), the XFOIL table blended into the NACA 0012 full-circle
  table past the stall angles on the 400-point grid (``polars.post_stall_blend``, the source's ``_AddPostStallData``),
  with all its flap rows. The stall angles (revision 2, the owner's option 乙 of 2026-10-22): the main wing's two
  tables at their own XFOIL data range, [0, 25] deg (blended only where there is no data); the tails and the pylons at
  the hover model's [-10, 10] deg. The hover model's main-wing values [1, 16] deg are kept as
  ``HOVER_MODEL_STALL_DEG`` for reports only. A panel's table at a deflection is ``polars.section_polar(blend, delta)``:
  the rows interpolated in the flap direction at the alpha knots. At zero deflection the tails' and the pylons' tables
  are those of the existing cases (``m600_makani_xfoil_polars*.json``) bit for bit; the main wing's are not (the cases
  use [1, 16] deg).
* **Out-of-table deflections** are clamped at the table edge, as Makani's ``GetCoeffs``; the excess
  max(0, delta - delta_max, delta_min - delta) is returned for the record.
* **Control map** (``control_map``): each panel's flap number, in Makani's order (A1, A2, A4, A5, A7, A8, elevator,
  rudder), from the CONTROL entries of Makani's AVL file: da1->0, da2->1, da4->2, da5->3, da7->4, da8->5, de1->6,
  dr1->7. A panel carries a control when both AVL sections bounding the segment that contains it declare it; a panel
  that straddles an AVL section is refused. Main-wing panels without a control use the zero row; pylons carry none.
* **Elevator** (``turn_horizontal_tail``): the elevator of the M600 is the whole horizontal tail (de1 hinged at 0 chord;
  Makani's hover model adds delta to the tail's angle). The tail's nodes and camber offsets are turned by delta_e about
  the line through its middle-section leading edge parallel to body +y; positive delta_e is trailing edge down, a
  right-hand turn about +y in the case axes (x downstream, y span, z up). The real hinge line is the swept, kinked
  leading edge; the angle between the two, per half, is returned by ``hinge_angles_deg`` (a declared limitation).
* Not used (contract revision 1, part E): the hover model's flap fudge factor 3, ``cl_weight``, its 2D-to-3D and
  downwash estimates, the simulator's flap scalings and saturations.

Nothing in the package imports this module; with it unused every result is unchanged.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aircraft import PanelPolars, WingAssembly
from .polars import load_makani_airfoil, post_stall_blend, section_polar
from .wing import Wing

FLAPS = ("A1", "A2", "A4", "A5", "A7", "A8", "Ele", "Rud")
AVL_CONTROLS = {"da1": 0, "da2": 1, "da4": 2, "da5": 3, "da7": 4, "da8": 5, "de1": 6, "dr1": 7}
ELEVATOR, RUDDER = 6, 7
STALL_DEG = {"oref": (0., 25.), "ref": (0., 25.), "f8": (-10., 10.), "approx_blade_rj8": (-10., 10.), "mp6d": (-10., 10.)}
HOVER_MODEL_STALL_DEG = {"oref": (1., 16.), "ref": (1., 16.), "f8": (-10., 10.), "approx_blade_rj8": (-10., 10.),
                         "mp6d": (-10., 10.)}      # hover_model.py 105-152 (1d66923); reports only (revision 2)
HIGH_AOA = "naca0012_high_aoa.json"
TAIL, FIN = "Horizontal tail", "Vertical tail"
SPAN_AXIS = {"Wing": 1, TAIL: 1, FIN: 2}            # the coordinate along which each surface's sections are placed


class FlapTables:
    """The blended full-circle tables of the five airfoils with all their flap rows."""

    def __init__(self, directory, names=tuple(STALL_DEG), stall=STALL_DEG):
        directory = Path(directory)
        high = load_makani_airfoil(directory/HIGH_AOA)
        self.high_aoa_sha256 = high.source_sha256
        self.blended = {}
        for name in names:
            self.blended[name] = post_stall_blend(load_makani_airfoil(directory/f"{name}.json"), stall[name], high)
        self._cache = {}

    def edges(self, name):
        rows = self.blended[name].flap_rad
        return float(rows[0]), float(rows[-1])

    def polar(self, name, delta_rad):
        """(SectionPolar at the clamped deflection, excess in rad)."""
        low, high = self.edges(name)
        delta = float(delta_rad)
        if not np.isfinite(delta):
            raise ValueError("non-finite flap deflection")
        excess = max(0.0, delta-high, low-delta)
        clamped = min(max(delta, low), high)
        key = (name, clamped)
        polar = self._cache.get(key)
        if polar is None:
            polar = section_polar(self.blended[name], clamped)
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = polar
        return polar, excess


def read_avl(path):
    """{surface name: [(leading edge xyz, chord, incidence deg, controls), ...]} in file order (Makani's AVL file)."""
    lines = [line.split("!")[0].rstrip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    surfaces, current, k = {}, None, 0
    while k < len(lines):
        word = lines[k].split()[0]
        if word == "SURFACE":
            current = lines[k+1].strip()
            surfaces[current] = []
            k += 2
            continue
        if word == "SECTION" and current is not None:
            values = [float(v) for v in lines[k+1].split()[:5]]
            surfaces[current].append(dict(le=np.array(values[:3]), chord=values[3], incidence_deg=values[4], controls=[]))
            k += 2
            continue
        if word == "CONTROL" and current is not None and surfaces[current]:
            surfaces[current][-1]["controls"].append(lines[k+1].split()[0])
            k += 2
            continue
        k += 1
    return surfaces


def control_map(assembly, avl_path, tolerance_m=1e-6):
    """Per panel of ``assembly`` (a WingAssembly): the flap number in Makani's order, or -1."""
    avl = read_avl(avl_path)
    out = []
    for wing, name in zip(assembly.wings, assembly.names):
        if name.startswith("Pylon"):
            out.extend([-1]*wing.panels)
            continue
        if name not in SPAN_AXIS or name not in avl:
            raise ValueError(f"no AVL surface for '{name}'")
        axis = SPAN_AXIS[name]
        sections = avl[name]
        stations = np.array([s["le"][axis] for s in sections])
        nodes = np.asarray(wing.leading_edge, dtype=float)[:, axis]
        for i in range(wing.panels):
            a, b = sorted((nodes[i], nodes[i+1]))
            found = None
            for j in range(len(sections)-1):
                lo, hi = sorted((stations[j], stations[j+1]))
                if a >= lo-tolerance_m and b <= hi+tolerance_m:
                    found = j
                    break
            if found is None:
                raise ValueError(f"panel {i} of '{name}' straddles an AVL section")
            common = set(sections[found]["controls"]) & set(sections[found+1]["controls"])
            if len(common) > 1:
                raise ValueError(f"panel {i} of '{name}' would carry two controls {sorted(common)}")
            if common:
                control = common.pop()
                if control not in AVL_CONTROLS:
                    raise ValueError(f"unknown AVL control '{control}'")
                out.append(AVL_CONTROLS[control])
            else:
                out.append(-1)
    return np.array(out, dtype=np.int64)


def tail_axis_point(assembly):
    """The leading-edge node of the horizontal tail nearest the body's symmetry plane (its middle AVL section)."""
    tail = assembly.wings[assembly.names.index(TAIL)]
    le = np.asarray(tail.leading_edge, dtype=float)
    return le[int(np.argmin(np.abs(le[:, 1])))].copy()


def rotation_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def turn_horizontal_tail(assembly, delta_e_rad, axis_point):
    """The assembly with the horizontal tail turned by ``delta_e_rad`` about body +y through ``axis_point``.

    Positive is trailing edge down: R_y(delta) maps a point behind the axis (x > 0) to z < 0. The other surfaces are the
    same objects."""
    delta = float(delta_e_rad)
    if delta == 0.0:
        return assembly
    rot = rotation_y(delta)
    pivot = np.asarray(axis_point, dtype=float)
    wings = []
    for wing, name in zip(assembly.wings, assembly.names):
        if name != TAIL:
            wings.append(wing)
            continue
        le = (np.asarray(wing.leading_edge)-pivot)@rot.T+pivot
        te = (np.asarray(wing.trailing_edge)-pivot)@rot.T+pivot
        offsets = None if wing.camber_offset_m is None else np.asarray(wing.camber_offset_m)@rot.T
        wings.append(Wing(le, te, wing.camber_x, offsets))
    return WingAssembly(tuple(wings), assembly.names)


def hinge_angles_deg(assembly, axis_point):
    """Largest angle (deg) between body y and the tail's leading-edge segments (the hinge line of de1 in AVL), for the
    port (y below the axis point) and the starboard half."""
    tail = assembly.wings[assembly.names.index(TAIL)]
    le = np.asarray(tail.leading_edge, dtype=float)
    d = np.diff(le, axis=0)
    angles = np.degrees(np.arccos(np.clip(np.abs(d[:, 1])/np.linalg.norm(d, axis=1), 0., 1.)))
    middle = 0.5*(le[:-1, 1]+le[1:, 1])
    y = float(np.asarray(axis_point, dtype=float)[1])
    return dict(port=float(np.max(angles[middle < y])), starboard=float(np.max(angles[middle > y])))


def tail_fin_overlap(assembly_turned, assembly, fin_name=FIN):
    """True when the turned tail's middle chord line rises to the fin's root within the fin root's chord range."""
    fin = assembly.wings[assembly.names.index(fin_name)]
    fin_le, fin_te = np.asarray(fin.leading_edge, dtype=float), np.asarray(fin.trailing_edge, dtype=float)
    root = int(np.argmin(fin_le[:, 2]))
    x0, x1 = sorted((fin_le[root, 0], fin_te[root, 0]))
    z_root = min(fin_le[root, 2], fin_te[root, 2])
    tail = assembly_turned.wings[assembly_turned.names.index(TAIL)]
    le, te = np.asarray(tail.leading_edge, dtype=float), np.asarray(tail.trailing_edge, dtype=float)
    middle = int(np.argmin(np.abs(le[:, 1])))
    s = np.linspace(0.0, 1.0, 201)
    chord = le[middle]+s[:, None]*(te[middle]-le[middle])
    inside = (chord[:, 0] >= x0) & (chord[:, 0] <= x1)
    return bool(np.any(chord[inside, 2] >= z_root))


def overlap_onset_deg(assembly, axis_point, lo_deg=-30.0):
    """The trailing-edge-up deflection (deg, negative) at which ``tail_fin_overlap`` first holds, by bisection; None if not
    within [lo_deg, 0]."""
    def hit(deg):
        return tail_fin_overlap(turn_horizontal_tail(assembly, np.radians(deg), axis_point), assembly)
    if not hit(lo_deg):
        return None
    a, b = lo_deg, 0.0                       # hit(a) True, hit(b) False (untouched at zero)
    if hit(b):
        return 0.0
    for _ in range(60):
        m = 0.5*(a+b)
        a, b = (m, b) if hit(m) else (a, m)
    return b


@dataclass(frozen=True)
class PanelAirfoils:
    """Per panel: the airfoil name of its table (from the case's polar ids) and its flap number."""
    names: tuple
    flaps: np.ndarray


def panel_airfoils(case_path, assembly, avl_path):
    """Airfoil names per panel read from a case file's ``polar_ids`` (the ids are the airfoil names in the XFOIL cases)."""
    import json
    data = json.loads(Path(case_path).read_text(encoding="utf-8"))
    ids = []
    for surface in data["surfaces"]:
        ids.extend(surface["polar_ids"])
    if len(ids) != assembly.panels:
        raise ValueError("the case's polar ids and the assembly's panels differ")
    unknown = sorted(set(ids)-set(STALL_DEG))
    if unknown:
        raise ValueError(f"polar ids that are not Makani airfoils: {unknown}")
    return PanelAirfoils(tuple(ids), control_map(assembly, avl_path))


def panel_polars(tables, airfoils, flaps_rad, *, panels=None):
    """(PanelPolars of the panels at these flap deflections, excess per flap number (8,)).

    The elevator (flap 6) is geometry (``turn_horizontal_tail``): its panels use the zero row. ``panels`` restricts to a
    slice of the panels (the order of ``airfoils``)."""
    flaps = np.asarray(flaps_rad, dtype=float).reshape(8)
    excess = np.zeros(8)
    out = []
    index = range(len(airfoils.names)) if panels is None else panels
    for i in index:
        name, flap = airfoils.names[i], int(airfoils.flaps[i])
        delta = 0.0 if flap < 0 or flap == ELEVATOR else flaps[flap]
        polar, over = tables.polar(name, delta)
        if flap >= 0 and flap != ELEVATOR:
            excess[flap] = max(excess[flap], over)
        out.append(polar)
    return PanelPolars(tuple(out)), excess
