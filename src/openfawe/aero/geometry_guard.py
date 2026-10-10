"""Pure contact checks against stationary, faceted mean-surface geometry.

The guard checks candidate material segments and straight node-displacement
paths. It is not continuous swept-segment collision detection, a solid-body
boundary condition, or a trajectory repair. Facets are physical camber-cell
triangles, not displaced vortex rings or virtual-incidence surfaces.
"""

from dataclasses import dataclass, field
import hashlib
import json
import operator

import numpy as np

from .vortex import _real, _norm
from .wing import owned


def _surface_layout(wing):
    wings = getattr(wing, 'wings', (wing,))
    names = tuple(getattr(wing, 'names', ('wing',)))
    if len(wings) != len(names) or len(set(names)) != len(names):
        raise ValueError('guard requires uniquely named physical surfaces')
    nodes = tuple(name for part, name in zip(wings, names)
                  for _ in range(part.node_count))
    panels = tuple(name for part, name in zip(wings, names)
                   for _ in range(part.panels))
    if len(nodes) != wing.node_count or len(panels) != wing.panels:
        raise ValueError('guard surface layout differs from wing topology')
    return names, nodes, panels


def _geometry_signature(wing):
    digest = hashlib.sha256()
    digest.update(json.dumps(_surface_layout(wing)[0]).encode('utf-8'))
    for name in ('leading_edge', 'trailing_edge', 'panel_edges',
                 'camber_x', 'camber_offset_m'):
        value = getattr(wing, name, None)
        digest.update(name.encode('ascii'))
        if value is None:
            digest.update(b'none')
        else:
            array = np.asarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive integer')
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be a positive integer') from error
    if integer < 1:
        raise ValueError(f'{name} must be a positive integer')
    return integer


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _inside(point, triangle, drop, tolerance):
    keep = [axis for axis in range(3) if axis != drop]
    vertices, point = triangle[:, keep], point[keep]
    sign = np.sign(_cross2(vertices[1]-vertices[0], vertices[2]-vertices[0]))
    for first, last in ((0, 1), (1, 2), (2, 0)):
        edge = vertices[last]-vertices[first]
        if sign*_cross2(edge, point-vertices[first]) < -tolerance*np.linalg.norm(edge):
            return False
    return True


def _contact_interval(start, end, triangle, normal, tolerance):
    """Return the contact parameter interval and kind, or None.

    Coplanar segments are clipped against the triangle's three half-planes.
    Only roundoff-scale signed distances are snapped to zero; this is not a
    core-sized exclusion band. A stationary point can also produce contact.
    """
    distances = np.array([np.dot(start-triangle[0], normal),
                          np.dot(end-triangle[0], normal)])
    distances[np.abs(distances) <= tolerance] = 0.
    first, last = distances
    if (first > 0 and last > 0) or (first < 0 and last < 0):
        return None
    drop = int(np.argmax(np.abs(normal)))
    if first == 0 and last == 0:
        keep = [axis for axis in range(3) if axis != drop]
        vertices, a, b = triangle[:, keep], start[keep], end[keep]
        sign = np.sign(_cross2(vertices[1]-vertices[0], vertices[2]-vertices[0]))
        lower, upper = 0., 1.
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = vertices[j]-vertices[i]
            values = np.array([sign*_cross2(edge, a-vertices[i]),
                               sign*_cross2(edge, b-vertices[i])])
            values[np.abs(values) <= tolerance*np.linalg.norm(edge)] = 0.
            fa, fb = values
            if fa < 0 and fb < 0:
                return None
            if fa < 0:
                lower = max(lower, float(fa/(fa-fb)))
            elif fb < 0:
                upper = min(upper, float(fa/(fa-fb)))
            if lower > upper:
                return None
        return lower, upper, 'coplanar_contact'
    parameter = 0. if first == 0 else 1. if last == 0 else float(first/(first-last))
    if not 0. <= parameter <= 1.:
        return None
    point = start+parameter*(end-start)
    if not _inside(point, triangle, drop, tolerance):
        return None
    kind = 'endpoint_contact' if parameter in (0., 1.) else 'surface_crossing'
    return parameter, parameter, kind


@dataclass(frozen=True)
class SurfaceContact:
    query_kind: str
    query_id: tuple
    source_surface: str
    target_surface: str
    triangle_id: int
    kind: str
    parameter_start: float
    parameter_end: float
    point_m: tuple
    end_point_m: tuple
    tolerance_m: float

    def to_dict(self):
        return dict(query_kind=self.query_kind, query_id=list(self.query_id),
                    source_surface=self.source_surface, target_surface=self.target_surface,
                    triangle_id=self.triangle_id, kind=self.kind,
                    parameter_start=self.parameter_start, parameter_end=self.parameter_end,
                    point_m=list(self.point_m), end_point_m=list(self.end_point_m),
                    tolerance_m=self.tolerance_m)


@dataclass(frozen=True)
class GeometryGuardReport:
    stage: str
    time_s: float | None
    geometry_sha256: str
    checked_segments: int
    checked_node_paths: int
    broad_phase_pairs: int
    allowed_te_contacts: int
    violation_count: int
    contacts: tuple
    initial_te_allowed: bool

    @property
    def acceptable(self):
        return self.violation_count == 0

    def to_dict(self):
        return dict(schema='openfawe.geometry_guard.v1', stage=self.stage,
                    time_s=self.time_s, geometry_sha256=self.geometry_sha256,
                    acceptable=self.acceptable, checked_segments=self.checked_segments,
                    checked_node_paths=self.checked_node_paths,
                    broad_phase_pairs=self.broad_phase_pairs,
                    allowed_te_contacts=self.allowed_te_contacts,
                    violation_count=self.violation_count,
                    stored_contacts=len(self.contacts),
                    contacts=[contact.to_dict() for contact in self.contacts],
                    initial_te_allowed=self.initial_te_allowed,
                    geometry='static_triangulated_mean_surface',
                    coverage='candidate_segments_and_straight_node_paths',
                    complete_continuous_collision_detection=False,
                    source_camber_interpolation_error_certified=False,
                    solid_body_nonpenetration_enforced=False)


class WakeGeometryViolation(ValueError):
    """Rejected geometry with a JSON-serializable diagnostic payload."""

    def __init__(self, report):
        self.report = report
        self.geometry_report = report.to_dict()
        super().__init__(f'{report.violation_count} surface contact(s) at {report.stage}; '
                         'candidate rejected, accepted state unchanged')


@dataclass(frozen=True, eq=False)
class StaticSurfaceGuard:
    """Cached static triangles with explicit source-surface and TE ownership.

    ``from_mesh`` accepts Wing or WingAssembly and a physical cell mesh.
    ``from_triangles`` accepts explicitly provided facets; a face has no TE
    exemption unless its supplied topology names that same surface and node.
    No external triangle set is inferred to be a watertight solid.
    """

    wing: object = field(repr=False)
    triangles: np.ndarray = field(repr=False)
    surface_ids: tuple
    te_node_ids: tuple
    roundoff_factor: int = 64
    max_events: int = 32
    _normals: np.ndarray = field(init=False, repr=False)
    _lower: np.ndarray = field(init=False, repr=False)
    _upper: np.ndarray = field(init=False, repr=False)
    _scale: float = field(init=False, repr=False)
    _node_surfaces: tuple = field(init=False, repr=False)
    _wing_signature: str = field(init=False, repr=False)
    geometry_sha256: str = field(init=False)

    @property
    def vertices(self):
        """Read-only facet vertices, including shared-vertex duplicates."""
        return self.triangles.reshape(-1, 3)

    def __post_init__(self):
        factor = _positive_integer(self.roundoff_factor, 'roundoff factor')
        _positive_integer(self.max_events, 'event budget')
        triangles = owned(self.triangles)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
            raise ValueError('guard requires finite triangles with shape (n,3,3)')
        surfaces = tuple(self.surface_ids)
        if len(surfaces) != len(triangles) or any(not isinstance(s, str) or not s for s in surfaces):
            raise ValueError('every guard triangle needs a named surface')
        raw_topology = tuple(tuple(nodes) for nodes in self.te_node_ids)
        if any(isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer))
               for nodes in raw_topology for node in nodes):
            raise ValueError('invalid guard TE-node index')
        topology = tuple(tuple(map(int, nodes)) for nodes in raw_topology)
        if len(topology) != len(triangles):
            raise ValueError('every guard triangle needs explicit TE-node ownership')
        _, node_surfaces, _ = _surface_layout(self.wing)
        scale = float(max(np.max(np.abs(triangles)),
                          np.max(_norm(triangles[:, 1:]-triangles[:, :1]))))
        tolerance = factor*np.finfo(float).eps*scale
        normals = np.cross(triangles[:, 1]-triangles[:, 0],
                           triangles[:, 2]-triangles[:, 0])
        areas = _norm(normals)
        lengths = np.maximum(_norm(triangles[:, 1]-triangles[:, 0]),
                             _norm(triangles[:, 2]-triangles[:, 0]))
        if np.any(areas <= tolerance*lengths):
            raise ValueError('guard triangle is degenerate at the declared roundoff scale')
        for index, nodes in enumerate(topology):
            if len(set(nodes)) != len(nodes):
                raise ValueError('duplicate TE node in guard triangle')
            for node in nodes:
                if (isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer))
                        or node < 0 or node >= self.wing.node_count):
                    raise ValueError('invalid guard TE-node index')
                if surfaces[index] != node_surfaces[node]:
                    raise ValueError('TE exemption cannot belong to a different surface')
                if not np.any(_norm(triangles[index]-self.wing.trailing_edge[node]) <= tolerance):
                    raise ValueError('TE topology must name a physical triangle vertex')
        signature = _geometry_signature(self.wing)
        digest = hashlib.sha256(triangles.tobytes())
        digest.update(json.dumps(dict(surfaces=surfaces, te_nodes=topology,
                                      source=signature, roundoff_factor=factor)).encode('utf-8'))
        for name, value in (('triangles', triangles), ('surface_ids', surfaces),
                            ('te_node_ids', topology), ('_node_surfaces', node_surfaces),
                            ('_normals', owned(normals/areas[:, None])),
                            ('_lower', owned(triangles.min(axis=1))),
                            ('_upper', owned(triangles.max(axis=1))), ('_scale', scale),
                            ('_wing_signature', signature), ('geometry_sha256', digest.hexdigest())):
            object.__setattr__(self, name, value)

    @classmethod
    def from_mesh(cls, wing, mesh, *, roundoff_factor=64, max_events=32):
        """Cache triangles from physical mesh vertices/cells, retaining surface IDs."""
        vertices, cells = _real(mesh.vertices, 'physical vertices'), np.asarray(mesh.cells)
        strips = np.asarray(mesh.strip_index)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or cells.ndim != 2
                or cells.shape[1] != 4 or cells.dtype.kind not in 'iu'
                or strips.shape != (len(cells),) or strips.dtype.kind not in 'iu'
                or np.any(cells < 0) or np.any(cells >= len(vertices))):
            raise ValueError('invalid physical quadrilateral mesh')
        width = wing.node_count
        if len(vertices) % width or len(vertices) < 2*width:
            raise ValueError('physical mesh must use source span-node rows')
        scale = max(float(np.max(np.abs(vertices))), float(np.max(_norm(wing.trailing_edge-wing.leading_edge))))
        tolerance = _positive_integer(roundoff_factor, 'roundoff factor')*np.finfo(float).eps*scale
        # LE + (TE - LE) can differ from TE by roundoff in an unchanged physical mesh.
        if (np.any(_norm(vertices[:width]-wing.leading_edge) > tolerance)
                or np.any(_norm(vertices[-width:]-wing.trailing_edge) > tolerance)):
            raise ValueError('guard mesh must include physical LE/TE, not displaced vortex rows')
        _, _, panel_surfaces = _surface_layout(wing)
        if np.any(strips < 0) or np.any(strips >= wing.panels):
            raise ValueError('physical mesh strip ownership is invalid')
        triangles, surfaces, topology = [], [], []
        te_start = len(vertices)-width
        for cell, strip in zip(cells, strips):
            for corners in (cell[[0, 1, 2]], cell[[0, 2, 3]]):
                triangles.append(vertices[corners])
                surfaces.append(panel_surfaces[int(strip)])
                topology.append(tuple(int(node-te_start) for node in corners if node >= te_start))
        return cls(wing, triangles, tuple(surfaces), tuple(topology), roundoff_factor, max_events)

    @classmethod
    def from_triangles(cls, wing, triangles, *, surface_ids=None, te_node_ids=None,
                       roundoff_factor=64, max_events=32):
        """Use explicit external facets; absent TE metadata permits no birth contact."""
        count = len(triangles)
        surfaces = tuple(surface_ids) if surface_ids is not None else ('external',)*count
        topology = tuple(te_node_ids) if te_node_ids is not None else ((),)*count
        return cls(wing, triangles, surfaces, topology, roundoff_factor, max_events)

    def assert_static(self, wing):
        if wing is not self.wing and _geometry_signature(wing) != self._wing_signature:
            raise NotImplementedError('surface guard supports stationary source geometry only')

    def check_rows(self, old_rows, new_rows, wing, *, stage='candidate', time_s=None,
                   allow_initial_te=False):
        """Reject contacts without changing inputs; rows include the convected old TE.

        The arrays have the same (depth, source_node_count, 3) shape. Row zero
        is the old TE at the start of a physical step, followed by material
        history. A predictor call may supply an earlier substep's moving row
        positions instead; only starts exactly at an owned TE can be exempt.
        ``allow_initial_te`` is solely for zero-time field evaluation, never a
        positive-time predictor or accepted new state.
        """
        self.assert_static(wing)
        if not isinstance(stage, str) or not stage:
            raise ValueError('guard stage must be a nonempty string')
        if not isinstance(allow_initial_te, bool):
            raise ValueError('initial TE policy must be boolean')
        if time_s is not None:
            time = _real(time_s, 'guard time')
            if time.shape:
                raise ValueError('guard time must be a scalar')
            time_s = float(time)
        old = _real(old_rows, 'old material rows')
        new = _real(new_rows, 'candidate material rows')
        if (old.ndim != 3 or old.shape[1:] != (wing.node_count, 3)
                or old.shape != new.shape or not len(old)):
            raise ValueError('old and candidate material rows must have the same nonempty shape')
        anchor_tolerance = self.roundoff_factor*np.finfo(float).eps*self._scale
        starts_at_te = _norm(old[0]-wing.trailing_edge) <= anchor_tolerance
        if allow_initial_te and (not np.all(starts_at_te)
                                 or np.any(_norm(new[0]-wing.trailing_edge) > anchor_tolerance)):
            raise ValueError('initial TE exemption requires the unmoved source TE row within roundoff')
        contacts, total, allowed, pairs, segment_count, path_count = [], 0, 0, 0, 0, 0

        def inspect(a, b, kind, query_id, source_node, birth_anchor=None, initial_anchors=()):
            nonlocal total, allowed, pairs, segment_count, path_count
            if kind == 'node_path':
                path_count += 1
            else:
                segment_count += 1
            scale = max(self._scale, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            tolerance = self.roundoff_factor*np.finfo(float).eps*scale
            lower, upper = np.minimum(a, b)-tolerance, np.maximum(a, b)+tolerance
            possible = np.flatnonzero(np.all(self._upper >= lower, axis=1)
                                      & np.all(self._lower <= upper, axis=1))
            length = float(_norm(b-a))
            parameter_tolerance = min(.25, tolerance/length) if length else 0.
            source = self._node_surfaces[source_node]
            for index in possible:
                pairs += 1
                result = _contact_interval(a, b, self.triangles[index], self._normals[index], tolerance)
                if result is None:
                    continue
                lo, hi, event_kind = result
                first_point, last_point = a+lo*(b-a), a+hi*(b-a)
                target = self.surface_ids[index]
                anchors = self.te_node_ids[index]
                permitted = False
                if source == target:
                    if birth_anchor is not None and birth_anchor in anchors:
                        permitted = (lo <= parameter_tolerance and hi <= parameter_tolerance
                                     and _norm(first_point-wing.trailing_edge[birth_anchor]) <= tolerance)
                    if initial_anchors:
                        matching = tuple(node for node in initial_anchors if node in anchors)
                        if len(matching) == 2:
                            permitted = True
                        elif len(matching) == 1:
                            anchor = wing.trailing_edge[matching[0]]
                            permitted = (_norm(first_point-anchor) <= tolerance
                                         and _norm(last_point-anchor) <= tolerance)
                if permitted:
                    allowed += 1
                    continue
                total += 1
                if len(contacts) < self.max_events:
                    contacts.append(SurfaceContact(kind, tuple(query_id), source, target,
                        int(index), event_kind, lo, hi, tuple(map(float, first_point)),
                        tuple(map(float, last_point)), float(tolerance)))

        # Each span edge belongs to its source surface; never join global neighbors.
        for row, positions in enumerate(new):
            for panel, (a, b) in enumerate(wing.panel_edges):
                initial = (int(a), int(b)) if allow_initial_te and row == 0 else ()
                inspect(positions[a], positions[b], 'span_segment', (row, panel), int(a),
                        initial_anchors=initial)
        # Include the newly attached connector, followed by old material stream edges.
        upstream = wing.trailing_edge
        for row, positions in enumerate(new):
            for node in range(wing.node_count):
                initial = (node,) if allow_initial_te and row == 0 else ()
                inspect(upstream[node], positions[node], 'stream_segment', (row, node), node,
                        birth_anchor=node if row == 0 else None, initial_anchors=initial)
            upstream = positions
        for row, (before, after) in enumerate(zip(old, new)):
            for node in range(wing.node_count):
                anchor = (node if row == 0 and starts_at_te[node]
                          else None)
                initial = (node,) if allow_initial_te and row == 0 else ()
                inspect(before[node], after[node], 'node_path', (row, node), node,
                        birth_anchor=anchor, initial_anchors=initial)
        report = GeometryGuardReport(stage, time_s, self.geometry_sha256, segment_count,
            path_count, pairs, allowed, total, tuple(contacts), allow_initial_te)
        if not report.acceptable:
            raise WakeGeometryViolation(report)
        return report
