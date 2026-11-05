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
  in wake._network_convection. Coordinates are matched by exact equality.
  GPU whole-step contract revision 4, S3 (b): the matching is now a sort of the
  coordinates' bit patterns instead of a dictionary keyed by the coordinate
  triple; the pairs are those of the dictionary version (kept as
  ``incident_pairs_reference``) element for element. Both match by bit
  pattern: -0.0 and 0.0 are different keys there (the compiled dictionary does
  not unite them, unlike the Python tuple keys of the NumPy path - a difference
  that predates S3), and NaN never matches.

The accelerated backends call these; the NumPy default path keeps the original
Python loops for the wing wake, but the rotor rings' self-convection
(``rotor.ring_wake.RingWake.self_velocity``) calls ``incident_pairs`` with every
kernel, the NumPy one included (the pairs are the reference's either way).
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


def incident_pairs_reference(nodes, edges, first):
    """The dictionary version of ``incident_pairs`` (before S3), kept as its reference (Z1)."""
    return _incident_pairs(np.ascontiguousarray(nodes, dtype=np.float64),
                           np.ascontiguousarray(edges, dtype=np.int64), int(first))


@njit(cache=True)
def _incident_pairs_sorted(nodes, bits, edges, first, order):
    """``_incident_pairs`` with the groups found from ``order`` (the nodes sorted by their coordinates' bit patterns):
    the same group numbers (first appearance among the free points), members, loop and output."""
    n = nodes.shape[0]
    count = n - first
    # no_nan: a node with a NaN coordinate never matches (as in the dictionary); +-inf coordinates match by bits
    no_nan = np.empty(n, np.bool_)
    for node in range(n):
        no_nan[node] = not (np.isnan(nodes[node, 0]) or np.isnan(nodes[node, 1]) or np.isnan(nodes[node, 2]))
    # runs of equal bit patterns along the sorted order; a NaN node is a run of its own
    run_of = np.empty(n, np.int64)
    runs = 0
    previous = -1
    for position in range(n):
        node = order[position]
        same = (previous >= 0 and no_nan[node] and no_nan[previous] and bits[node, 0] == bits[previous, 0]
                and bits[node, 1] == bits[previous, 1] and bits[node, 2] == bits[previous, 2])
        if not same:
            runs += 1
        run_of[node] = runs - 1
        previous = node
    # group numbers in order of first appearance, as the dictionary assigned them
    group_of_run = np.full(runs, -1, np.int64)
    point_group = np.empty(count, np.int64)
    groups = 0
    for index in range(count):
        run = run_of[first + index]
        if group_of_run[run] < 0:
            group_of_run[run] = groups
            groups += 1
        point_group[index] = group_of_run[run]
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
            if no_nan[node]:
                g = group_of_run[run_of[node]]
                if g >= 0:
                    total += sizes[g + 1] - sizes[g]
    pairs = np.empty((total, 2), np.int64)
    m = 0
    for segment in range(edges.shape[0]):
        for end in range(2):
            node = edges[segment, end]
            if no_nan[node]:
                g = group_of_run[run_of[node]]
                if g >= 0:
                    for k in range(sizes[g], sizes[g + 1]):
                        pairs[m, 0] = members[k]
                        pairs[m, 1] = segment
                        m += 1
    return pairs


def incident_pairs(nodes, edges, first):
    """Exclusion pairs of wake._network_convection, as an (e, 2) int64 array (S3 (b): sorted bit patterns)."""
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    # Anything but (n, 3) nodes, (e, 2) edges with indices in [0, n) and 0 <= first <= n goes to the reference itself
    # (its result or its error; out-of-range indices are unchecked there, as before S3, and never occur in the wake).
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or edges.ndim != 2 or edges.shape[1] != 2
            or not 0 <= int(first) <= len(nodes) or (edges.size and (edges.min() < 0 or edges.max() >= len(nodes)))):
        return incident_pairs_reference(nodes, edges, first)
    bits = nodes.view(np.int64)
    order = np.lexsort((bits[:, 2], bits[:, 1], bits[:, 0])).astype(np.int64)
    return _incident_pairs_sorted(nodes, bits, edges, int(first), order)
