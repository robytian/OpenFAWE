# SPDX-License-Identifier: Apache-2.0
# Near-field pairs use the OpenFAST-derived kernel through vortex_numba.
"""Opt-in hierarchical (Barnes-Hut type) Biot-Savart summation for segments.

This is a controlled approximation of ``segments_velocity``; the direct NumPy
and Numba sums remain the reference. Method:

* A binary tree splits the segment midpoints at the median of the longest
  bounding-box axis (stable sort, deterministic) until a leaf holds at most
  ``leaf_size`` segments.
* Each node stores its centre c (bounding-box centre of all segment end
  points), the radius R of the sphere about c containing every end point (and
  hence every segment), the largest core radius, and the exact Cartesian
  moments of its segment line integrals about c, up to second order:
  M0 = sum alpha, M1_j = sum alpha d_j, M2_jk = sum alpha (d_j d_k + L_j L_k/12),
  with alpha = Gamma L, L = B - A and d = midpoint - c.
* The velocity of a node at x, R_x = x - c, is the Taylor series of
  (1/4 pi) sum Gamma int L x K(x - y) dtau, K(r) = r/|r|^3, truncated at
  ``order``: u = (1/4 pi)[M0 x K - M1_j x d_j K + 1/2 M2_jk x d_j d_k K].
  Complete vortex rings have M0 ~ 0 (dipole-like far field), so order >= 1
  is the meaningful setting; order 0 is a diagnostic.
* A node is evaluated by its expansion only if R < theta*|R_x| and
  |R_x| - R >= core_clearance * (largest core radius in the node), and none
  of the target's excluded segments lies in the node. Otherwise its children
  are visited; leaves use the exact pair kernel (cores, singularity flags).
  An excluded or on-segment pair therefore always lies in the near field.
* The far field omits the core factor. With the Lamb-Oseen cutoff used here
  (factor exactly 1 when the distance to the segment line exceeds 2.83 core
  radii) and Rankine, the omission only affects targets very close to the
  extension of a far segment's line; the measured total error includes it.
  Vatistas, offset and Rosenhead-Moore cores are rejected in tree mode.

Each target traverses the tree in a fixed order, so results are deterministic
and independent of the thread count. References for the method family:
J. Barnes and P. Hut, Nature 324 (1986) 446-449; K. Lindsay and R. Krasny,
J. Comput. Phys. 172 (2001) 879-907 (to be verified before citation).
"""

import math

import numpy as np
from numba import njit, prange

from .vortex_numba import FLAG_NONFINITE, FLAG_OK, pair_velocity

SUPPORTED_MODELS = (0, 1, 2)  # NONE, RANKINE, LAMB_OSEEN
_INV_FOUR_PI = 1.0 / (4 * np.pi)
_STACK = 512


@njit(cache=True)
def build_tree(mid, leaf_size):
    """Return (order, lo, hi, left, right, node_count) of a median-split binary tree."""
    n = mid.shape[0]
    order = np.arange(n)
    capacity = 4 * (n // max(leaf_size, 1) + 1) + 1
    lo = np.empty(capacity, np.int64)
    hi = np.empty(capacity, np.int64)
    left = np.full(capacity, -1, np.int64)
    right = np.full(capacity, -1, np.int64)
    lo[0] = 0
    hi[0] = n
    count = 1
    stack = np.empty(capacity, np.int64)
    stack[0] = 0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        a = lo[node]
        b = hi[node]
        if b - a <= leaf_size:
            continue
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for k in range(a, b):
            s = order[k]
            for axis in range(3):
                v = mid[s, axis]
                if v < low[axis]:
                    low[axis] = v
                if v > high[axis]:
                    high[axis] = v
        axis = 0
        extent = high[0] - low[0]
        for candidate in range(1, 3):
            if high[candidate] - low[candidate] > extent:
                extent = high[candidate] - low[candidate]
                axis = candidate
        if not extent > 0.0:
            continue  # coincident midpoints stay one leaf
        indices = order[a:b].copy()
        keys = np.empty(b - a)
        for k in range(b - a):
            keys[k] = mid[indices[k], axis]
        permutation = np.argsort(keys, kind="mergesort")
        for k in range(b - a):
            order[a + k] = indices[permutation[k]]
        middle = (a + b) // 2
        first = count
        second = count + 1
        count += 2
        lo[first] = a
        hi[first] = middle
        lo[second] = middle
        hi[second] = b
        left[node] = first
        right[node] = second
        stack[top] = second
        stack[top + 1] = first
        top += 2
    return order, lo[:count], hi[:count], left[:count], right[:count], count


@njit(cache=True)
def node_moments(order, lo, hi, starts, ends, alpha, length_vectors, radius, gamma):
    """Centre, bounding radius, largest core and moments for every node."""
    count = lo.shape[0]
    centre = np.zeros((count, 3))
    bound = np.zeros(count)
    core = np.zeros(count)
    m0 = np.zeros((count, 3))
    m1 = np.zeros((count, 3, 3))
    m2 = np.zeros((count, 3, 3, 3))
    for node in range(count):
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for k in range(lo[node], hi[node]):
            s = order[k]
            for axis in range(3):
                for value in (starts[s, axis], ends[s, axis]):
                    if value < low[axis]:
                        low[axis] = value
                    if value > high[axis]:
                        high[axis] = value
        for axis in range(3):
            centre[node, axis] = 0.5 * (low[axis] + high[axis])
        r2 = 0.0
        largest = 0.0
        for k in range(lo[node], hi[node]):
            s = order[k]
            for point in (starts[s], ends[s]):
                q = ((point[0] - centre[node, 0]) ** 2 + (point[1] - centre[node, 1]) ** 2
                     + (point[2] - centre[node, 2]) ** 2)
                if q > r2:
                    r2 = q
            if radius[s] > largest:
                largest = radius[s]
            if gamma[s] == 0.0:
                continue
            d = np.empty(3)
            for axis in range(3):
                d[axis] = 0.5 * (starts[s, axis] + ends[s, axis]) - centre[node, axis]
            for i in range(3):
                m0[node, i] += alpha[s, i]
                for j in range(3):
                    m1[node, i, j] += alpha[s, i] * d[j]
                    for l in range(3):
                        m2[node, i, j, l] += alpha[s, i] * (
                            d[j] * d[l] + length_vectors[s, j] * length_vectors[s, l] / 12.0)
        # Small relative pad keeps the containing sphere conservative under roundoff.
        bound[node] = math.sqrt(r2) * (1.0 + 1e-12)
        core[node] = largest
    return centre, bound, core, m0, m1, m2


@njit(cache=True, inline="always")
def _cross_add(scale, a0, a1, a2, b0, b1, b2, out):
    out[0] += scale * (a1 * b2 - a2 * b1)
    out[1] += scale * (a2 * b0 - a0 * b2)
    out[2] += scale * (a0 * b1 - a1 * b0)


@njit(cache=True)
def multipole_velocity_explicit(rx, ry, rz, node, m0, m1, m2, order, out):
    """Term-by-term form of the expansion (kept as the test reference for far_velocity)."""
    r = np.empty(3)
    r[0] = rx
    r[1] = ry
    r[2] = rz
    r2 = rx * rx + ry * ry + rz * rz
    distance = math.sqrt(r2)
    inv3 = 1.0 / (r2 * distance)
    acc = np.zeros(3)
    _cross_add(inv3, m0[node, 0], m0[node, 1], m0[node, 2], rx, ry, rz, acc)
    if order >= 1:
        inv5 = inv3 / r2
        for j in range(3):
            g = np.empty(3)
            for c in range(3):
                g[c] = -3.0 * r[j] * r[c] * inv5
            g[j] += inv3
            _cross_add(-1.0, m1[node, 0, j], m1[node, 1, j], m1[node, 2, j], g[0], g[1], g[2], acc)
    if order >= 2:
        inv5 = inv3 / r2
        inv7 = inv5 / r2
        for j in range(3):
            for k in range(3):
                g = np.empty(3)
                for c in range(3):
                    g[c] = 15.0 * r[j] * r[k] * r[c] * inv7
                    if j == k:
                        g[c] -= 3.0 * r[c] * inv5
                g[j] -= 3.0 * r[k] * inv5
                g[k] -= 3.0 * r[j] * inv5
                _cross_add(0.5, m2[node, 0, j, k], m2[node, 1, j, k], m2[node, 2, j, k],
                           g[0], g[1], g[2], acc)
    out[0] += _INV_FOUR_PI * acc[0]
    out[1] += _INV_FOUR_PI * acc[1]
    out[2] += _INV_FOUR_PI * acc[2]


@njit(cache=True)
def node_contractions(m1, m2):
    """Per-node contractions of the moments used by far_velocity.

    w1_i = eps_iaj M1_aj, t2_i = M2_ijj and W2_ik = eps_iaj M2_ajk. M2 is exactly
    symmetric in its last two indices (products commute in IEEE arithmetic),
    which far_velocity uses to merge two of the three Kronecker terms.
    """
    count = m1.shape[0]
    w1 = np.zeros((count, 3))
    t2 = np.zeros((count, 3))
    w2 = np.zeros((count, 3, 3))
    for node in range(count):
        w1[node, 0] = m1[node, 1, 2] - m1[node, 2, 1]
        w1[node, 1] = m1[node, 2, 0] - m1[node, 0, 2]
        w1[node, 2] = m1[node, 0, 1] - m1[node, 1, 0]
        for i in range(3):
            t2[node, i] = m2[node, i, 0, 0] + m2[node, i, 1, 1] + m2[node, i, 2, 2]
        for k in range(3):
            w2[node, 0, k] = m2[node, 1, 2, k] - m2[node, 2, 1, k]
            w2[node, 1, k] = m2[node, 2, 0, k] - m2[node, 0, 2, k]
            w2[node, 2, k] = m2[node, 0, 1, k] - m2[node, 1, 0, k]
    return w1, t2, w2


@njit(cache=True, inline="always")
def far_velocity(rx, ry, rz, node, m0, m1, m2, w1, t2, w2, order):
    """Truncated node expansion at R = x - c in contracted, allocation-free form.

    Algebraically identical to multipole_velocity_explicit:
      order 0: M0 x R / r^3
      order 1: - w1 / r^3 + 3 (M1 R) x R / r^5
      order 2: 7.5 (M2 : R R) x R / r^7 - 1.5 t2 x R / r^5 - 3 (W2 R) / r^5
    all times 1/(4 pi).
    """
    r2 = rx * rx + ry * ry + rz * rz
    inv3 = 1.0 / (r2 * math.sqrt(r2))
    inv5 = inv3 / r2
    a0 = m0[node, 0]
    a1 = m0[node, 1]
    a2 = m0[node, 2]
    u0 = inv3 * (a1 * rz - a2 * ry)
    u1 = inv3 * (a2 * rx - a0 * rz)
    u2 = inv3 * (a0 * ry - a1 * rx)
    if order >= 1:
        v0 = m1[node, 0, 0] * rx + m1[node, 0, 1] * ry + m1[node, 0, 2] * rz
        v1 = m1[node, 1, 0] * rx + m1[node, 1, 1] * ry + m1[node, 1, 2] * rz
        v2 = m1[node, 2, 0] * rx + m1[node, 2, 1] * ry + m1[node, 2, 2] * rz
        u0 += 3.0 * inv5 * (v1 * rz - v2 * ry) - inv3 * w1[node, 0]
        u1 += 3.0 * inv5 * (v2 * rx - v0 * rz) - inv3 * w1[node, 1]
        u2 += 3.0 * inv5 * (v0 * ry - v1 * rx) - inv3 * w1[node, 2]
    if order >= 2:
        inv7 = inv5 / r2
        q0 = 0.0
        q1 = 0.0
        q2 = 0.0
        r = (rx, ry, rz)
        for j in range(3):
            for k in range(3):
                rr = r[j] * r[k]
                q0 += m2[node, 0, j, k] * rr
                q1 += m2[node, 1, j, k] * rr
                q2 += m2[node, 2, j, k] * rr
        b0 = 7.5 * inv7 * q0 - 1.5 * inv5 * t2[node, 0]
        b1 = 7.5 * inv7 * q1 - 1.5 * inv5 * t2[node, 1]
        b2 = 7.5 * inv7 * q2 - 1.5 * inv5 * t2[node, 2]
        s0 = w2[node, 0, 0] * rx + w2[node, 0, 1] * ry + w2[node, 0, 2] * rz
        s1 = w2[node, 1, 0] * rx + w2[node, 1, 1] * ry + w2[node, 1, 2] * rz
        s2 = w2[node, 2, 0] * rx + w2[node, 2, 1] * ry + w2[node, 2, 2] * rz
        u0 += (b1 * rz - b2 * ry) - 3.0 * inv5 * s0
        u1 += (b2 * rx - b0 * rz) - 3.0 * inv5 * s1
        u2 += (b0 * ry - b1 * rx) - 3.0 * inv5 * s2
    return _INV_FOUR_PI * u0, _INV_FOUR_PI * u1, _INV_FOUR_PI * u2


@njit(cache=True, inline="always")
def _excluded_in(excluded, a, b, lo, hi):
    for k in range(a, b):
        if excluded[k] >= lo and excluded[k] < hi:
            return True
    return False


@njit(cache=True, parallel=True)
def tree_velocity(points, starts, directions, lengths, gamma, radius, model, raise_singular,
                  order, lo, hi, left, right, centre, bound, core, m0, m1, m2, w1, t2, w2,
                  theta, expansion_order, clearance, ex_ptr, ex_pos, out, flags, far_counts):
    """Evaluate all targets; ex_pos holds sorted tree positions of excluded segments."""
    for p in prange(points.shape[0]):
        x0, x1, x2 = points[p, 0], points[p, 1], points[p, 2]
        acc = np.zeros(3)
        flag = FLAG_OK
        far = 0
        e0 = ex_ptr[p]
        e1 = ex_ptr[p + 1]
        stack = np.empty(_STACK, np.int64)
        stack[0] = 0
        top = 1
        while top > 0:
            top -= 1
            node = stack[top]
            rx = x0 - centre[node, 0]
            ry = x1 - centre[node, 1]
            rz = x2 - centre[node, 2]
            distance = math.sqrt(rx * rx + ry * ry + rz * rz)
            if (bound[node] < theta * distance
                    and distance - bound[node] >= clearance * core[node]
                    and not _excluded_in(ex_pos, e0, e1, lo[node], hi[node])):
                fx, fy, fz = far_velocity(rx, ry, rz, node, m0, m1, m2, w1, t2, w2, expansion_order)
                acc[0] += fx
                acc[1] += fy
                acc[2] += fz
                far += 1
            elif left[node] < 0:
                for k in range(lo[node], hi[node]):
                    skip = False
                    for e in range(e0, e1):
                        if ex_pos[e] == k:
                            skip = True
                            break
                    if skip:
                        continue
                    s = order[k]
                    vx, vy, vz, f = pair_velocity(x0, x1, x2, starts[s, 0], starts[s, 1], starts[s, 2],
                                                  directions[s, 0], directions[s, 1], directions[s, 2],
                                                  lengths[s], gamma[s], model, radius[s], raise_singular)
                    if f != FLAG_OK:
                        if flag == FLAG_OK:
                            flag = f
                        continue
                    acc[0] += vx
                    acc[1] += vy
                    acc[2] += vz
            else:
                stack[top] = right[node]
                stack[top + 1] = left[node]
                top += 2
        if not (math.isfinite(acc[0]) and math.isfinite(acc[1]) and math.isfinite(acc[2])) and flag == FLAG_OK:
            flag = FLAG_NONFINITE
        out[p, 0] = acc[0]
        out[p, 1] = acc[1]
        out[p, 2] = acc[2]
        flags[p] = flag
        far_counts[p] = far


@njit(cache=True)
def _positions(ex_ptr, ex_segments, inverse):
    positions = np.empty_like(ex_segments)
    for p in range(ex_ptr.shape[0] - 1):
        a = ex_ptr[p]
        b = ex_ptr[p + 1]
        for k in range(a, b):
            positions[k] = inverse[ex_segments[k]]
        positions[a:b] = np.sort(positions[a:b])
    return positions


class TreeStatistics:
    """Last evaluation counters for diagnostics (not part of the result)."""
    last = {}


def tree_sum(points, starts, directions, lengths, gamma, radius, model, csr, singularity, settings):
    """Return (velocity, any_flag) for validated inputs from segments_velocity.

    ``csr`` is the (pointer, segments) exclusion structure from exclusion_csr.
    """
    if int(model) not in SUPPORTED_MODELS and np.any(radius > 0):
        raise ValueError("tree backend supports core models NONE, RANKINE and LAMB_OSEEN only")
    if settings.theta >= 1.0:
        raise ValueError("tree theta must be below one")
    points = np.ascontiguousarray(points)
    starts = np.ascontiguousarray(starts)
    directions = np.ascontiguousarray(directions)
    lengths = np.ascontiguousarray(lengths)
    gamma = np.ascontiguousarray(gamma)
    radius = np.ascontiguousarray(radius)
    out = np.zeros_like(points)
    flags = np.zeros(len(points), dtype=np.int8)
    far = np.zeros(len(points), dtype=np.int64)
    if len(starts) == 0 or len(points) == 0:
        return out, False
    length_vectors = directions * lengths[:, None]
    ends = starts + length_vectors
    alpha = gamma[:, None] * length_vectors
    mid = 0.5 * (starts + ends)
    order, lo, hi, left, right, _ = build_tree(mid, int(settings.leaf_size))
    centre, bound, core, m0, m1, m2 = node_moments(order, lo, hi, starts, ends, alpha,
                                                   length_vectors, radius, gamma)
    w1, t2, w2 = node_contractions(m1, m2)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    pointer, segments = csr
    positions = _positions(pointer, segments, inverse) if len(segments) else segments
    tree_velocity(points, starts, directions, lengths, gamma, radius, int(model),
                  singularity == "raise", order, lo, hi, left, right, centre, bound, core,
                  m0, m1, m2, w1, t2, w2, float(settings.theta), int(settings.order),
                  float(settings.core_clearance), pointer, positions, out, flags, far)
    TreeStatistics.last = dict(points=len(points), segments=len(starts), nodes=len(lo),
                               mean_far_nodes=float(far.mean()))
    return out, bool(np.any(flags))
