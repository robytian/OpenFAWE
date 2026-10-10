# SPDX-License-Identifier: Apache-2.0
"""Closed partial-ring quadrature of a material-node linear wake potential.

The potential jump is in m2/s. Each physical material interval linearly
interpolates its two accepted nodal values. Integration by parts represents
that sheet as partial closed rings, integrated by Gauss quadrature. Auxiliary
vertices are integration points, not newly released material or core history.
Generic panel connectivity permits disconnected lifting surfaces.

See docs/design/linear-wake-potential.md for the restricted production domain
and the retained OF-0006-teremedy20260927 prototype/bridge evidence.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .vortex import _real, _norm
from .wake import count, pack_rings


def quadrature_rule(order, ratio=None):
    """Unit-interval Gauss rule, optionally graded at the physical TE junction."""
    if count(order, 'wake quadrature order', 1) not in (4, 8):
        raise ValueError('wake quadrature order must be 4 or 8')
    ends = [0., 1.]
    if ratio is not None:
        ratio = _real(ratio, 'near-junction ratio')
        if ratio.shape or ratio <= 0:
            raise ValueError('positive geometric near-junction ratio required')
        ends = [0., min(ratio.item(), 1.)]
        while ends[-1] < 1.:
            ends.append(min(2*ends[-1], 1.))
            if len(ends) > 200:
                raise ValueError('near-junction geometry cannot be resolved')
    q, w = leggauss(order)
    nodes, weights = [], []
    for left, right in zip(ends[:-1], ends[1:]):
        nodes.extend(left+(q+1)*(right-left)/2)
        weights.extend(w*(right-left)/2)
    return np.asarray(nodes), np.asarray(weights)


def near_junction_distance(wing, mesh):
    """Minimum CP distance to its own physical TE segment, in metres."""
    first, last = wing.panel_edges[mesh.strip_index].T
    a, b = wing.trailing_edge[first], wing.trailing_edge[last]
    line = b-a
    lengths2 = np.sum(line*line, axis=1)
    if np.any(lengths2 <= 0):
        raise ValueError('nonzero physical TE segments required')
    projection = np.clip(np.sum((mesh.collocation_points-a)*line, axis=1)/lengths2, 0., 1.)
    distance = float(np.min(_norm(mesh.collocation_points-a-projection[:, None]*line)))
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('positive CP-to-TE distance required')
    return distance


def refine_potential(rows, nodal_phi, order, near_distance_m=None):
    """Return auxiliary rows and constant subcell potentials for integration."""
    rows, phi = _real(rows, 'material rows'), _real(nodal_phi, 'nodal wake potential')
    if rows.ndim != 3 or rows.shape[-1] != 3 or len(rows) < 2 or rows.shape[1] < 2:
        raise ValueError('material rows must have shape (intervals+1,nodes,3)')
    if phi.ndim != 2 or len(phi) != len(rows) or phi.shape[1] < 1:
        raise ValueError('one nodal potential per material row and strip required')
    lengths = _norm(np.diff(rows, axis=0))
    if np.any(lengths <= 0):
        raise ValueError('nonzero material intervals required at every node')
    if near_distance_m is not None:
        distance = _real(near_distance_m, 'near-junction distance')
        if distance.shape or distance <= 0:
            raise ValueError('positive near-junction distance required')
        ratio = distance.item()/float(np.max(lengths[0]))
    else:
        ratio = None
    q0, w0 = quadrature_rule(order, ratio)
    first = rows[:1]+np.r_[q0, 1.][:, None, None]*(rows[1:2]-rows[:1])
    cumulative = np.r_[0., np.cumsum(w0)]; cumulative[-1] = 1.
    values = phi[:1]+cumulative[:, None]*(phi[1:2]-phi[:1])
    vertices = np.concatenate((rows[:1], first))
    if len(rows) > 2:
        q, w = quadrature_rule(order)
        positions = np.r_[q, 1.]
        v = rows[1:-1, None]+positions[None, :, None, None]*(rows[2:]-rows[1:-1])[:, None]
        cumulative = np.r_[0., np.cumsum(w)]; cumulative[-1] = 1.
        p = phi[1:-1, None]+cumulative[None, :, None]*(phi[2:]-phi[1:-1])[:, None]
        vertices = np.concatenate((vertices, v.reshape(-1, rows.shape[1], 3)))
        values = np.concatenate((values, p.reshape(-1, phi.shape[1])))
    return vertices, values


def potential_network(wing, mesh, gamma, free_rows, nodal_phi, order, quadrature):
    """Bound rings plus the same integrated wake used by loads and probes."""
    if quadrature not in ('gauss', 'te_graded'):
        raise ValueError('unknown linear wake quadrature')
    gamma = _real(gamma, 'bound circulation')
    free_rows = _real(free_rows, 'free material rows')
    nodal_phi = _real(nodal_phi, 'nodal wake potential')
    if (gamma.shape != (len(mesh.vortex_rows)-1, wing.panels) or
            free_rows.shape != (len(nodal_phi)-1, wing.node_count, 3) or
            nodal_phi.shape[1:] != (wing.panels,) or
            not np.array_equal(nodal_phi[0], gamma[-1])):
        raise ValueError('bound/material potential network shape or TE trace mismatch')
    if not len(free_rows):
        return pack_rings(mesh.vortex_rows, gamma, wing.panel_edges)
    rows = np.concatenate((mesh.vortex_rows[-1:], free_rows))
    distance = near_junction_distance(wing, mesh) if quadrature == 'te_graded' else None
    vertices, values = refine_potential(rows, nodal_phi, order, distance)
    return pack_rings(np.concatenate((mesh.vortex_rows, vertices[1:])),
                      np.vstack((gamma, values)), wing.panel_edges)


def wake_potential_network(wing, mesh, free_rows, nodal_phi, order, quadrature):
    """Wake-only graph, also for the attached unknown and accepted known history."""
    if quadrature not in ('gauss', 'te_graded'):
        raise ValueError('unknown linear wake quadrature')
    rows = np.concatenate((mesh.vortex_rows[-1:], free_rows))
    distance = near_junction_distance(wing, mesh) if quadrature == 'te_graded' else None
    vertices, values = refine_potential(rows, nodal_phi, order, distance)
    return pack_rings(vertices, values, wing.panel_edges)


def material_row_coordinates(rows, order, near_distance_m=None):
    """Physical-interval abscissa of every auxiliary row, including endpoints."""
    rows = _real(rows, 'material rows')
    if rows.ndim != 3 or rows.shape[-1] != 3 or len(rows) < 1:
        raise ValueError('material rows must have shape (intervals+1,nodes,3)')
    if len(rows) == 1:
        return np.array([0.])
    lengths = _norm(np.diff(rows, axis=0))
    if np.any(lengths <= 0):
        raise ValueError('nonzero material intervals required at every node')
    ratio = None if near_distance_m is None else near_distance_m/float(np.max(lengths[0]))
    q0, _ = quadrature_rule(order, ratio)
    first = np.r_[0., q0, 1.]
    if len(rows) == 2:
        return first
    q, _ = quadrature_rule(order)
    rest = np.arange(1, len(rows)-1)[:, None]+np.r_[q, 1.][None]
    return np.r_[first, rest.ravel()]


def potential_edge_radii(wing, mesh, free_rows, edges, cores, initial_radius,
                         order, quadrature, *, include_bound=True, initial_cap=False):
    """Map auxiliary edges to physical material cores; interpolate span rc^2.

    Physical TE/bound edges have zero core except the explicit zero-history
    closing-cap option, which uses the birth radius. The downstream TE
    right-limit is the configured initial radius. Stream subsegments inherit
    their parent interval core; no auxiliary age or strain is introduced.
    This map requires a core kernel additive on collinear stream segments.
    """
    initial = _real(initial_radius, 'initial radius')
    if initial.shape or initial < 0:
        raise ValueError('nonnegative scalar initial radius required')
    if quadrature not in ('gauss', 'te_graded'):
        raise ValueError('unknown linear wake quadrature')
    if not isinstance(initial_cap, bool) or (initial_cap and (not include_bound or len(free_rows))):
        raise ValueError('initial cap radius applies only to a zero-history bound network')
    rows = np.concatenate((mesh.vortex_rows[-1:], free_rows))
    distance = near_junction_distance(wing, mesh) if quadrature == 'te_graded' else None
    coordinate = material_row_coordinates(rows, order, distance)
    if include_bound:
        coordinate = np.r_[np.arange(1-len(mesh.vortex_rows), 0), coordinate]
    width = wing.node_count
    edges = np.asarray(edges)
    if (edges.ndim != 2 or edges.shape[1:] != (2,) or edges.dtype.kind not in 'iu'
            or np.any(edges < 0) or np.any(edges >= len(coordinate)*width)):
        raise ValueError('invalid potential network edge indices')
    if cores is not None and (cores.span.ndim != 2 or cores.stream.ndim != 2 or
            cores.span.shape[1:] != (wing.panels,) or cores.stream.shape[1:] != (width,) or
            len(cores.span) < len(free_rows) or len(cores.stream) < len(free_rows)):
        raise ValueError('physical core history is shorter than material wake')
    lookup = {tuple(sorted(pair)): j for j, pair in enumerate(wing.panel_edges)}
    result = np.zeros(len(edges))
    for k, (first, last) in enumerate(edges):
        ra, a = divmod(int(first), width); rb, b = divmod(int(last), width)
        if ra == rb:
            s = coordinate[ra]
            if initial_cap and s == 0:
                result[k] = initial
            elif s > 0:
                if cores is None:
                    result[k] = initial
                else:
                    strip = lookup[tuple(sorted((a, b)))]
                    left = int(np.floor(s)); fraction = s-left
                    qleft = initial**2 if left == 0 else cores.span[left-1, strip]**2
                    qright = qleft if fraction == 0 else cores.span[left, strip]**2
                    result[k] = np.sqrt(qleft+fraction*(qright-qleft))
        elif a == b and abs(ra-rb) == 1:
            s, t = sorted((coordinate[ra], coordinate[rb]))
            if s >= 0:
                parent = int(np.floor((s+t)/2))
                result[k] = initial if cores is None else cores.stream[parent, a]
        else:
            raise ValueError('edge is not a spanwise or streamwise potential segment')
    return result
