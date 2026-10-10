# SPDX-License-Identifier: Apache-2.0
"""Compiled wake-network bookkeeping for the accelerated velocity backends.

Free-wake steps rebuild the ring network and the incident-endpoint exclusions
of the whole wake every step. The reference implementations (wake.pack_rings,
wake._network_convection) are Python loops whose cost grows with the wake and
is single-threaded, so with a fast tree/threaded kernel they dominate long
runs. The functions here reproduce those loops exactly:

* pack_ring_edges: same directed-edge aggregation as wake.pack_rings, same
  first-appearance edge order, same per-edge summation order starting from
  0.0, same exact-zero filtering. Results are bitwise identical.
* incident_pairs: same (point, segment) pairs, in the same order, as the loop
  in wake._network_convection. Coordinates are matched by exact equality
  through a dictionary keyed by the coordinate triple, exactly like the
  Python tuple keys (so -0.0 and 0.0 coincide, and NaN never matches).

Only the accelerated backends call these; the NumPy default path keeps the
original Python loops.
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict

_KEY = types.UniTuple(types.float64, 3)


@njit(cache=True)
def _ring_contributions(depth, width, panel_edges, strengths):
    """Directed ring-edge contributions in the reference loop order."""
    panels = panel_edges.shape[0]
    total = 4 * (depth - 1) * panels
    lo = np.empty(total, np.int64)
    hi = np.empty(total, np.int64)
    value = np.empty(total)
    c = 0
    vertices = np.empty(4, np.int64)
    for i in range(depth - 1):
        for j in range(panels):
            left = panel_edges[j, 0]
            right = panel_edges[j, 1]
            vertices[0] = i * width + left
            vertices[1] = i * width + right
            vertices[2] = (i + 1) * width + right
            vertices[3] = (i + 1) * width + left
            g = strengths[i, j]
            for k in range(4):
                a = vertices[k]
                b = vertices[(k + 1) % 4]
                if a < b:
                    lo[c] = a
                    hi[c] = b
                    value[c] = g
                else:
                    lo[c] = b
                    hi[c] = a
                    value[c] = -g
                c += 1
    return lo, hi, value


@njit(cache=True)
def _aggregate(lo, hi, value, node_count):
    """Sum contributions per undirected key in sequence order; keep first-appearance order."""
    count = lo.shape[0]
    key = lo * node_count + hi
    order = np.argsort(key, kind="mergesort")  # stable: sequence order within a key
    first = np.empty(count, np.int64)
    sums = np.empty(count)
    keys_lo = np.empty(count, np.int64)
    keys_hi = np.empty(count, np.int64)
    groups = 0
    k = 0
    while k < count:
        start = order[k]
        current = key[start]
        total = 0.0
        while k < count and key[order[k]] == current:
            total = total + value[order[k]]
            k += 1
        first[groups] = start
        sums[groups] = total
        keys_lo[groups] = lo[start]
        keys_hi[groups] = hi[start]
        groups += 1
    appearance = np.argsort(first[:groups], kind="mergesort")
    kept = 0
    for g in range(groups):
        if sums[appearance[g]] != 0.0:
            kept += 1
    edges = np.empty((kept, 2), np.int64)
    gamma = np.empty(kept)
    m = 0
    for g in range(groups):
        index = appearance[g]
        if sums[index] != 0.0:
            edges[m, 0] = keys_lo[index]
            edges[m, 1] = keys_hi[index]
            gamma[m] = sums[index]
            m += 1
    return edges, gamma


def pack_ring_edges(depth, width, panel_edges, strengths):
    """Return (edges, gamma) exactly as the reference loop of wake.pack_rings."""
    panel_edges = np.ascontiguousarray(panel_edges, dtype=np.int64)
    strengths = np.ascontiguousarray(strengths, dtype=np.float64)
    lo, hi, value = _ring_contributions(int(depth), int(width), panel_edges, strengths)
    return _aggregate(lo, hi, value, int(depth) * int(width))


@njit(cache=True)
def _incident_pairs(nodes, edges, first):
    """(point, segment) pairs for every free point coinciding with a segment end."""
    count = nodes.shape[0] - first
    group = Dict.empty(key_type=_KEY, value_type=types.int64)
    point_group = np.empty(count, np.int64)
    groups = 0
    for index in range(count):
        key = (nodes[first + index, 0], nodes[first + index, 1], nodes[first + index, 2])
        if key in group:
            point_group[index] = group[key]
        else:
            group[key] = groups
            point_group[index] = groups
            groups += 1
    # Point indices of each group in increasing order (CSR).
    sizes = np.zeros(groups + 1, np.int64)
    for index in range(count):
        sizes[point_group[index] + 1] += 1
    for g in range(groups):
        sizes[g + 1] += sizes[g]
    cursor = sizes[:-1].copy()
    members = np.empty(count, np.int64)
    for index in range(count):
        g = point_group[index]
        members[cursor[g]] = index
        cursor[g] += 1
    total = 0
    for segment in range(edges.shape[0]):
        for end in range(2):
            node = edges[segment, end]
            key = (nodes[node, 0], nodes[node, 1], nodes[node, 2])
            if key in group:
                g = group[key]
                total += sizes[g + 1] - sizes[g]
    pairs = np.empty((total, 2), np.int64)
    m = 0
    for segment in range(edges.shape[0]):
        for end in range(2):
            node = edges[segment, end]
            key = (nodes[node, 0], nodes[node, 1], nodes[node, 2])
            if key in group:
                g = group[key]
                for k in range(sizes[g], sizes[g + 1]):
                    pairs[m, 0] = members[k]
                    pairs[m, 1] = segment
                    m += 1
    return pairs


def incident_pairs(nodes, edges, first):
    """Exclusion pairs of wake._network_convection, as an (e, 2) int64 array."""
    return _incident_pairs(np.ascontiguousarray(nodes, dtype=np.float64),
                           np.ascontiguousarray(edges, dtype=np.int64), int(first))
