"""Free vortex-ring wake of actuator discs (standalone and as the extra closed
loops that join the wing's free-wake network).

The rings are closed directed loops carrying one circulation per edge. They
convect with the velocity induced by every segment in the network except the
segments incident on the moved node (the same exclusion rule as
``wake._network_convection``; exact positions only), plus the ambient field.
Nothing is merged or smoothed; rings are deleted only by the opt-in horizon
(``released_since``, used by ``PrescribedRotors.retained_ring_steps``), whole.
"""

from dataclasses import dataclass, field

import numpy as np

from ..aero.vortex import segments_velocity
from ..aero.network_numba import incident_pairs
from .actuator_disc import ShedRing


def _owned(array, dtype=float):
    array = np.array(array, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RingWake:
    """Closed vortex loops: node positions, directed edges, circulation per edge, release times per node."""
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    circulation: np.ndarray = field(default_factory=lambda: np.empty(0))
    release_times_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    source: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))  # disc index per node
    # Core radius used when the rings induce velocity at lifting-surface points
    # (closure and load points). None = the wake configuration's core. A larger
    # value smears the discrete sheet to the panel scale where filaments cross
    # the surface (contract §7.5, decision 0020); it is a model parameter to be
    # reported with its sensitivity. Convection always uses the physical core.
    surface_core_m: float | None = None
    # Opt-in far-wake treatment of the rings (OF-0009 contract, case C3 with rotors): None, the default,
    # convects every ring with the full induction (every existing result unchanged). A whole number K
    # lets only the rings released within the newest K wake steps convect with the induced velocity;
    # older rings drift with the ambient flow alone and nothing is deleted. It is the ring counterpart
    # of ``WakeConfig.free_convection_rows`` (``aero.wake.convect_geometry``) and all rings keep inducing
    # on everything. K = 0 (contract section 8, 2026-10-22 18:55 BST, model A') lets every ring drift with the ambient
    # flow from its release: no ring convects with the induced velocity.
    free_convection_steps: int | None = None
    # Opt-in core radius of the ring filaments as sources in the flow (OF-0009, owner 2026-10-20: how the rotor rings
    # are to be treated): None, the default, uses the wake configuration's core (every existing result unchanged). A
    # value sets the core of the ring segments wherever they induce away from the lifting surfaces in the coupled wake:
    # the convection of the wake rows and of the rings themselves (``aero.wake._edge_radii_extended``) and the inflow at
    # the rotor hubs (``rotor.coupled.hub_apparent_wind``). ``surface_core_m`` still applies at lifting-surface points;
    # the cores of the wing filaments are unchanged. The methods of this class that take a core argument
    # (``induced_velocity``, ``self_velocity``, ``convect_rings``: the hover path) use the core they are given.
    core_m: float | None = None
    # Opt-in core of the bound system of the lifting surfaces where it induces at the ring nodes (OF-0009 contract
    # section 8, 2026-10-22, model B; owner 17:34 BST): None, the default, keeps every existing result. A value sets the
    # core of the wing segments whose two nodes are on the bound row or the trailing-edge row (bound vortex, chordwise
    # and trailing-edge segments of the bound rings) when the convection velocity of the ring nodes is summed
    # (``aero.wake.convect_geometry``). A lifting line stands for a wing of finite chord, whose velocity field within
    # about a chord is not that of a line vortex; a chord-scale core keeps the rings from being drawn into the bound
    # vortex line. The wake rows, the loads, the closure and the hub inflow are unchanged. It is a model parameter.
    bound_core_m: float | None = None
    # Opt-in refinement of the ring sheet where the rings induce at lifting-surface points (OF-0009 contract section 8,
    # 2026-10-22 18:22 BST, "S"; owner 18:13 BST): None, the default, keeps every existing result. A value is the core
    # overlap lambda: the rings of each disc, newest to oldest, are read as one sheet in which ring k stands for the
    # stretch [k-1/2, k+1/2] of the ring index (the newest [0, 1/2], the oldest [K-1/2, K]); where a stretch is longer
    # than surface core / lambda and near the evaluation points, its ring is replaced by sub-rings at the linear
    # interpolation of the neighbouring rings (``surface_sheet``). Only the induction at lifting-surface points uses it
    # (``aero.wake.ring_induction``, ``flight.prescribed.ring_induction_by_disc``); convection and the hub inflow keep
    # the stored rings. One ring per disc per step is still released and stored.
    surface_overlap: float | None = None
    # Opt-in drift of the frozen rings with the sheet's own speed (OF-0009 contract section 8, 2026-10-22 19:16 BST,
    # model A''): False, the default, keeps every existing result. True makes ``shed`` give each ring the velocity
    # gamma/2 along the flow through its disc at release (-axis), gamma its momentum sheet strength, and the rings that
    # do not convect freely (beyond ``free_convection_steps``) move with the ambient flow plus that velocity
    # (``aero.wake.convect_geometry``): the straight vortex cylinder whose wall moves at the mean of the inner and
    # outer axial speeds, as the released circulation gamma (V + gamma/2) dt assumes. ``drift_m_s`` holds it per node.
    sheet_drift: bool = False
    drift_m_s: np.ndarray | None = None
    # Opt-in (review RA-01 of 2026-10-22, with ``sheet_drift`` only): the released circulation uses the spacing at which
    # the ring recedes from its disc under the drift, (V_ambient + gamma/2) dt, instead of the hub-inflow spacing
    # (V_hub + gamma/2) dt, so that the tube carries the momentum strength gamma (``rotor.coupled.shed_rotor_rings``).
    ambient_spacing: bool = False

    def __post_init__(self):
        if not isinstance(self.sheet_drift, (bool, np.bool_)):
            raise ValueError("sheet_drift must be True or False")
        object.__setattr__(self, "sheet_drift", bool(self.sheet_drift))
        if not isinstance(self.ambient_spacing, (bool, np.bool_)):
            raise ValueError("ambient_spacing must be True or False")
        object.__setattr__(self, "ambient_spacing", bool(self.ambient_spacing))
        if self.sheet_drift and self.free_convection_steps is None:
            raise ValueError("sheet_drift needs free_convection_steps (the drift acts on the rings that do not convect freely)")
        if self.ambient_spacing and not self.sheet_drift:
            raise ValueError("ambient_spacing needs sheet_drift")
        if self.surface_overlap is not None and not (np.isfinite(self.surface_overlap) and self.surface_overlap > 0):
            raise ValueError("surface_overlap must be a positive finite number or None")
        if self.bound_core_m is not None and not (np.isfinite(self.bound_core_m) and self.bound_core_m > 0):
            raise ValueError("bound_core_m must be a positive finite radius or None")
        if self.surface_core_m is not None and not (np.isfinite(self.surface_core_m) and self.surface_core_m > 0):
            raise ValueError("surface_core_m must be a positive finite radius or None")
        if self.core_m is not None and not (np.isfinite(self.core_m) and self.core_m > 0):
            raise ValueError("core_m must be a positive finite radius or None")
        steps = self.free_convection_steps
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0):
            raise ValueError("free_convection_steps must be a non-negative whole number of steps or None")
        if steps is not None:
            object.__setattr__(self, "free_convection_steps", int(steps))
        nodes = _owned(self.nodes).reshape(-1, 3)
        edges = _owned(self.edges, np.int64).reshape(-1, 2)
        circulation = _owned(self.circulation).reshape(-1)
        times = _owned(self.release_times_s).reshape(-1)
        source = _owned(self.source, np.int64).reshape(-1)
        if len(circulation) != len(edges) or len(times) != len(nodes) or len(source) != len(nodes):
            raise ValueError("ring wake arrays disagree in length")
        drift = None if self.drift_m_s is None else _owned(self.drift_m_s).reshape(-1, 3)
        if drift is not None and (len(drift) != len(nodes) or not np.all(np.isfinite(drift))):
            raise ValueError("drift_m_s must be finite with one row per ring node")
        object.__setattr__(self, "drift_m_s", drift)
        if len(edges) and (edges.min() < 0 or edges.max() >= len(nodes)):
            raise ValueError("ring wake edge refers to a missing node")
        if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(circulation))):
            raise ValueError("ring wake geometry/circulation must be finite")
        for name, value in (("nodes", nodes), ("edges", edges), ("circulation", circulation),
                            ("release_times_s", times), ("source", source)):
            object.__setattr__(self, name, value)

    @property
    def count(self):
        return len(self.nodes)

    def shed(self, ring: ShedRing, disc_index=0, axis=None):
        """Append one ring (new nodes first, so the newest ring has the lowest ids). With ``sheet_drift`` the disc
        ``axis`` at release is required: the ring's drift is ``-(sheet_strength/2) axis``."""
        shift = ring.count
        drift = None
        if self.sheet_drift:
            if axis is None:
                raise ValueError("sheet_drift needs the disc axis at release")
            axis = np.asarray(axis, dtype=float).reshape(3)
            new = np.tile(-0.5 * float(ring.sheet_strength) * axis, (ring.count, 1))
            old = self.drift_m_s if self.drift_m_s is not None else np.zeros((len(self.nodes), 3))
            drift = np.vstack([new, old])
        return RingWake(np.vstack([ring.nodes, self.nodes]),
                        np.vstack([ring.edges, self.edges + shift]) if len(self.edges) else ring.edges.copy(),
                        np.concatenate([np.full(ring.count, ring.circulation), self.circulation]),
                        np.concatenate([np.full(ring.count, ring.release_time_s), self.release_times_s]),
                        np.concatenate([np.full(ring.count, disc_index, dtype=np.int64), self.source]),
                        self.surface_core_m, self.free_convection_steps, self.core_m, self.bound_core_m,
                        self.surface_overlap, self.sheet_drift, drift, self.ambient_spacing)

    def free_node_count(self, time_s, dt_s):
        """Number of leading ring nodes that convect with the induced velocity at ``time_s`` (None: all).

        With ``free_convection_steps`` K the rings released at t, t-dt, ..., t-(K-1) dt are free (half a step
        of margin, as ``released_since`` counts). The rings are stored newest first (``shed`` prepends), so
        the free nodes are a leading block; a ring set that is not ordered that way is refused.
        """
        if self.free_convection_steps is None:
            return None
        keep = self.release_times_s >= float(time_s)-(self.free_convection_steps-0.5)*float(dt_s)
        count = int(np.count_nonzero(keep))
        if not (np.all(keep[:count]) and not np.any(keep[count:])):
            raise ValueError("the rings are not stored newest first; the frozen far wake of the rings needs that order")
        return count

    def induced_velocity(self, points, model, core_radius, *, exclude_source=None, singularity="raise", **kernel):
        """Velocity of the rings at ``points``; ``exclude_source`` drops one disc's own rings (contract §3)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(self.edges):
            return np.zeros_like(points)
        if exclude_source is None:
            return segments_velocity(points, self.nodes, self.edges, self.circulation, model=model,
                                     core_radius=core_radius, singularity=singularity, **kernel)
        keep = self.source[self.edges[:, 0]] != exclude_source
        if not np.any(keep):
            return np.zeros_like(points)
        return segments_velocity(points, self.nodes, self.edges[keep], self.circulation[keep], model=model,
                                 core_radius=core_radius, singularity=singularity, **kernel)

    def self_velocity(self, model, core_radius, *, singularity="zero", **kernel):
        """Convection velocity of every ring node from all rings (incident segments excluded).

        Identical rings released from the same rim in still air can stack onto
        one another (starting-ring cluster; seen in hover at core 0.30 m,
        2026-10-01): coincident nodes of different rings are then on each
        other's filament end points. With a positive core the regularised
        velocity on the filament is zero, so ``singularity='zero'`` is the
        physical limit here, not a guard against bad geometry as in the wing
        wake; pass ``singularity='raise'`` to detect such events instead.
        """
        if not len(self.edges):
            return np.zeros_like(self.nodes)
        exclusions = incident_pairs(self.nodes, self.edges, 0)
        return segments_velocity(self.nodes, self.nodes, self.edges, self.circulation, model=model,
                                 core_radius=core_radius, exclusions=exclusions, singularity=singularity,
                                 **kernel)

    def released_since(self, time_s):
        """The rings released at or after ``time_s``; older rings are removed whole.

        Opt-in far-wake truncation of the ring set (``PrescribedRotors.retained_ring_steps``). Every node of a ring
        carries the ring's release time, so what stays is a set of closed loops with their circulation; a ring whose
        nodes disagree in release time is refused when the cut falls between them (otherwise it is not looked at).
        Returns ``self`` when no ring is older.
        """
        keep = self.release_times_s >= float(time_s)
        if np.all(keep):
            return self
        first, second = keep[self.edges[:, 0]], keep[self.edges[:, 1]]
        if np.any(first != second):
            raise ValueError("a ring has nodes of different release times and cannot be removed whole")
        renumbered = np.cumsum(keep)-1
        return RingWake(self.nodes[keep], renumbered[self.edges[first]], self.circulation[first], self.release_times_s[keep],
                        self.source[keep], self.surface_core_m, self.free_convection_steps, self.core_m, self.bound_core_m,
                        self.surface_overlap, self.sheet_drift, None if self.drift_m_s is None else self.drift_m_s[keep],
                        self.ambient_spacing)

    def moved(self, nodes):
        return RingWake(nodes, self.edges, self.circulation, self.release_times_s, self.source, self.surface_core_m,
                        self.free_convection_steps, self.core_m, self.bound_core_m, self.surface_overlap, self.sheet_drift,
                        self.drift_m_s, self.ambient_spacing)

    def surface_sheet(self, points, core_m):
        """``(nodes, edges, circulation, source per edge)`` of the rings for induction at lifting-surface ``points``.

        Without ``surface_overlap`` (or without rings) these are the stored arrays. With it, the rings of each disc,
        ordered newest first as 0..K, are read as one sheet in which ring k stands for the stretch of ring index u in
        [k-1/2, k+1/2] clipped to [0, K]. The stretch length s_k is the largest distance between corresponding nodes of
        ring k and its neighbours; m_k = ceil(overlap * s_k / core_m); a ring to be refined with more than
        ``SURFACE_SUBRINGS_MAX`` sub-rings is refused.
        Ring k is replaced by m_k sub-rings when m_k > 1 and a node of ring k or of a neighbour lies within
        2 s_k + 4 core_m + l_k of a point (l_k the longest edge of ring k): sub-ring j sits at the linear interpolation of
        the neighbouring rings' corresponding nodes at the middle u of its sub-interval and carries the ring's
        circulation times the sub-interval's share of the stretch. The returned nodes are the stored nodes followed by
        the sub-ring nodes; the edges are the stored edges of the kept rings, in the stored order, followed by those of
        the sub-rings (refined rings in stored order), so that a sheet with no refined ring is the stored set bitwise.
        The rings of a disc must have equal node counts (corresponding nodes); each ring must be one closed loop of its
        own nodes.
        """
        source_per_edge = self.source[self.edges[:, 0]] if len(self.edges) else np.empty(0, dtype=np.int64)
        if self.surface_overlap is None or not len(self.edges):
            return self.nodes, self.edges, self.circulation, source_per_edge
        if not (np.isfinite(core_m) and core_m > 0):
            raise ValueError("the surface core must be a positive finite radius")
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        starts, ends, edge_block = _ring_blocks(self)
        plan, summary = _sheet_plan(self, starts, ends, edge_block, points, float(core_m), float(self.surface_overlap))
        # The summary of the last sheet is kept on this (immutable) ring set, so that the record of the step
        # (``surface_sheet_summary`` at the same points) does not plan the sheet again; values are unchanged.
        object.__setattr__(self, "_last_sheet", (points.tobytes(), float(core_m), summary))
        if not plan:
            return self.nodes, self.edges, self.circulation, source_per_edge
        refined = np.zeros(len(starts), dtype=bool)
        refined[list(plan)] = True
        kept = ~refined[edge_block]
        out_nodes, out_edges = [self.nodes], [self.edges[kept]]
        out_circ, out_source = [self.circulation[kept]], [source_per_edge[kept]]
        order = np.argsort(edge_block, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(edge_block, minlength=len(starts)))]
        count = len(self.nodes)
        for block in sorted(plan):
            index = order[bounds[block]:bounds[block + 1]]
            local = self.edges[index] - starts[block]
            circ = self.circulation[index]
            disc = int(self.source[starts[block]])
            for position, share in plan[block]:
                out_nodes.append(position)
                out_edges.append(local + count)
                out_circ.append(circ * share)
                out_source.append(np.full(len(local), disc, dtype=np.int64))
                count += len(position)
        return (np.vstack(out_nodes), np.vstack(out_edges).astype(np.int64), np.concatenate(out_circ),
                np.concatenate(out_source))

    def surface_sheet_summary(self, points, core_m):
        """What ``surface_sheet`` refines at ``points``: number of refined rings, their largest sub-ring count and largest
        stretch length (zeros without ``surface_overlap`` or without rings). Diagnostic (review RS-02)."""
        empty = dict(refined_rings=0, max_subrings=0, max_stretch_m=0.0)
        if self.surface_overlap is None or not len(self.edges):
            return empty
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        last = getattr(self, "_last_sheet", None)
        if last is not None and last[0] == points.tobytes() and last[1] == float(core_m):
            return dict(last[2])
        starts, ends, edge_block = _ring_blocks(self)
        return _sheet_plan(self, starts, ends, edge_block, points, float(core_m), float(self.surface_overlap))[1]

    def near_surface_count(self, points, radius):
        """Number of ring nodes within ``radius`` of any of ``points`` (diagnostic for filament crossings)."""
        if not len(self.nodes):
            return 0
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(points):
            return 0
        # Only nodes inside the box of a block of points widened by the radius can be within the radius of a point of
        # that block; the count is the same as over all nodes, without the (nodes x block x 3) temporaries of a long
        # ring wake (several bodies: the blocks follow the point order, so a block rarely spans two bodies).
        # The box is widened by a relative margin far above round-off, so that a node exactly at the radius along one
        # axis is not lost to the rounding of min - radius (the exact test below decides).
        close = np.zeros(len(self.nodes), dtype=bool)
        for start in range(0, len(points), 64):
            block = points[start:start + 64]
            margin = radius + 1e-9 * (radius + float(np.max(np.abs(block))))
            low, high = block.min(axis=0) - margin, block.max(axis=0) + margin
            index = np.flatnonzero(np.all((self.nodes >= low) & (self.nodes <= high), axis=1))
            if not len(index):
                continue
            d2 = np.sum((self.nodes[index][:, None, :] - block[None, :, :]) ** 2, axis=2)
            close[index[np.any(d2 <= radius * radius, axis=1)]] = True
        return int(np.count_nonzero(close))


SURFACE_SUBRINGS_MAX = 64


def _ring_blocks(wake):
    """``(starts, ends, edge_block)``: the rings as contiguous node blocks of one release time and one disc, and the
    block of every edge. Each ring must be one closed loop of its own nodes (as many edges as nodes, all inside)."""
    times, source = wake.release_times_s, wake.source
    change = np.flatnonzero((times[1:] != times[:-1]) | (source[1:] != source[:-1])) + 1
    starts = np.r_[0, change].astype(np.int64)
    ends = np.r_[change, len(wake.nodes)].astype(np.int64)
    first = np.searchsorted(starts, wake.edges[:, 0], side="right") - 1
    second = np.searchsorted(starts, wake.edges[:, 1], side="right") - 1
    if np.any(first != second):
        raise ValueError("a ring edge joins two rings; the sheet refinement needs closed rings")
    if np.any(np.bincount(first, minlength=len(starts)) != ends - starts):
        raise ValueError("every ring must have as many edges as nodes for the sheet refinement")
    return starts, ends, first


def _min_distance(nodes, points, chunk=4096):
    """Smallest distance from each of ``nodes`` to ``points`` (inf without points)."""
    if not len(points):
        return np.full(len(nodes), np.inf)
    out = np.empty(len(nodes))
    for lo in range(0, len(nodes), chunk):
        block = nodes[lo:lo + chunk]
        d2 = np.sum((block[:, None, :] - points[None, :, :]) ** 2, axis=2)
        out[lo:lo + chunk] = np.sqrt(d2.min(axis=1))
    return out


def sheet_subrings(rings, k, m):
    """Sub-rings of ring k of one disc's sheet ``rings`` (K+1, n, 3), newest first: ``[(nodes, share)]`` for m equal
    sub-intervals of [k-1/2, k+1/2] clipped to [0, K] (see ``RingWake.surface_sheet``)."""
    K = len(rings) - 1
    lo, hi = max(0.0, k - 0.5), min(float(K), k + 0.5)
    out = []
    for j in range(m):
        a, b = lo + (hi - lo) * j / m, lo + (hi - lo) * (j + 1) / m
        u = 0.5 * (a + b)
        i0 = min(int(np.floor(u)), K - 1)
        t = u - i0
        out.append((rings[i0] * (1 - t) + rings[i0 + 1] * t, (b - a) / (hi - lo)))
    return out


def _sheet_plan(wake, starts, ends, edge_block, points, core, overlap):
    """``({block: [(sub-ring nodes, share)]}, summary)`` for the rings to be replaced (``RingWake.surface_sheet``); the
    summary holds the number of refined rings, their largest sub-ring count and their largest stretch length."""
    plan = {}
    summary = dict(refined_rings=0, max_subrings=0, max_stretch_m=0.0)
    block_source = wake.source[starts]
    block_time = wake.release_times_s[starts]
    edge_vectors = wake.nodes[wake.edges[:, 1]] - wake.nodes[wake.edges[:, 0]]
    longest = np.zeros(len(starts))
    np.maximum.at(longest, edge_block, np.linalg.norm(edge_vectors, axis=1))
    for disc in np.unique(block_source):
        blocks = np.flatnonzero(block_source == disc)
        blocks = blocks[np.argsort(-block_time[blocks], kind="stable")]           # newest first
        sizes = ends[blocks] - starts[blocks]
        if len(blocks) < 2:
            continue
        if np.any(sizes != sizes[0]):
            raise ValueError("the rings of a disc must have equal node counts for the sheet refinement")
        rings = np.stack([wake.nodes[starts[b]:ends[b]] for b in blocks])
        gaps = np.max(np.linalg.norm(rings[1:] - rings[:-1], axis=2), axis=1)
        stretch = np.zeros(len(blocks))
        stretch[:-1] = gaps
        stretch[1:] = np.maximum(stretch[1:], gaps)
        counts = np.maximum(1, np.ceil(overlap * stretch / core)).astype(np.int64)
        candidates = np.flatnonzero(counts > 1)
        if not len(candidates):
            continue
        threshold = 2.0 * stretch + 4.0 * core + longest[blocks]
        # A bounding sphere per ring gives a lower bound on its node distances; only rings that may be within reach are
        # measured node by node (the decision is the node-distance rule of ``surface_sheet``).
        centres = rings.mean(axis=1)
        lower = _min_distance(centres, points) - np.max(np.linalg.norm(rings - centres[:, None, :], axis=2), axis=1)
        padded = np.r_[np.inf, lower, np.inf]
        lower_near = np.minimum(np.minimum(padded[:-2], padded[1:-1]), padded[2:])        # over k-1, k, k+1
        candidates = candidates[lower_near[candidates] < threshold[candidates]]
        if not len(candidates):
            continue
        needed = np.unique(np.clip(np.concatenate([candidates - 1, candidates, candidates + 1]), 0, len(blocks) - 1))
        nearest = np.full(len(blocks), np.inf)
        nearest[needed] = _min_distance(rings[needed].reshape(-1, 3), points).reshape(len(needed), -1).min(axis=1)
        for k in candidates:
            near = nearest[max(0, k - 1):k + 2].min()
            if near < threshold[k]:
                if counts[k] > SURFACE_SUBRINGS_MAX:
                    raise ValueError(f"the sheet refinement needs {int(counts[k])} sub-rings for one ring "
                                     f"(more than {SURFACE_SUBRINGS_MAX}); reduce the time step or the overlap")
                plan[int(blocks[k])] = sheet_subrings(rings, int(k), int(counts[k]))
                summary["refined_rings"] += 1
                summary["max_subrings"] = max(summary["max_subrings"], int(counts[k]))
                summary["max_stretch_m"] = max(summary["max_stretch_m"], float(stretch[k]))
    return plan, summary


def convect_rings(wake, dt_s, model, core_radius, *, ambient=None, integrator="rk4", **kernel):
    """Advance the ring nodes with their own induction (+ ambient(points) if given), frozen circulation."""
    def rhs(nodes):
        current = wake.moved(nodes)
        velocity = current.self_velocity(model, core_radius, **kernel)
        if ambient is not None:
            velocity = velocity + ambient(nodes)
        return velocity
    x = wake.nodes
    if integrator == "euler":
        new = x + dt_s * rhs(x)
    elif integrator == "rk4":
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt_s * k1)
        k3 = rhs(x + 0.5 * dt_s * k2)
        k4 = rhs(x + dt_s * k3)
        new = x + dt_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        raise ValueError("integrator must be 'euler' or 'rk4'")
    if not np.all(np.isfinite(new)):
        raise FloatingPointError("ring wake convection produced non-finite positions")
    return wake.moved(new)
