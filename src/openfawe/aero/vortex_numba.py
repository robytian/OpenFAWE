# SPDX-License-Identifier: Apache-2.0
# Compiled counterpart of vortex._evaluate_pairs/_sum_batched, which are
# adapted from OpenFAST v5.0.0 FVW_BiotSavart.f90 (ui_seg_11/ui_seg/ui_quad_n1).
# Modifications (2026): Numba translation with identical per-pair arithmetic,
# per-target serial summation in segment order, parallel over targets.
# See third_party/OPENFAST-NOTICE.md and third_party/LICENSE-OpenFAST.txt.
"""Exact compiled direct Biot-Savart summation for directed segments.

The per-pair arithmetic repeats ``vortex._evaluate_pairs`` operation by
operation (length scaling, rationalized denominator, roundoff axis guard and
all six core branches). Each target point accumulates its contributions
serially in segment input order, exactly like ``vortex._sum_batched``. Threads
only split target points, so the result does not depend on the thread count.
No fast-math or contraction flags are used.

Error policy: pairs that the NumPy path would reject (an on-segment point with
``singularity='raise'``) or arithmetic that leaves the finite range only set a
per-target flag here. The caller then re-runs the NumPy reference, which raises
the historical exception type and message. Results are never repaired.
"""

import math

import numpy as np
from numba import njit, prange

_AXIS_EPS = 8 * np.finfo(np.float64).eps
_LAMB_OSEEN_CUT = float(np.sqrt(10 / 1.25643))
_FOUR_PI = 4 * np.pi

FLAG_OK = 0
FLAG_SINGULAR = 1
FLAG_NONFINITE = 2


@njit(cache=True, inline="always")
def _norm3(x, y, z):
    return math.hypot(math.hypot(x, y), z)


@njit(cache=True, inline="always")
def pair_velocity(px, py, pz, sx, sy, sz, dx, dy, dz, length, gamma, model, radius,
                  raise_singular):
    """Velocity of one directed segment at one point; returns (ux, uy, uz, flag)."""
    ax = (px - sx) / length
    ay = (py - sy) / length
    az = (pz - sz) / length
    t = (ax * dx + ay * dy) + az * dz
    # np.cross(direction, a)
    cx = dy * az - dz * ay
    cy = dz * ax - dx * az
    cz = dx * ay - dy * ax
    h = _norm3(cx, cy, cz)
    na = _norm3(ax, ay, az)
    nb = _norm3(ax - dx, ay - dy, az - dz)
    if not (math.isfinite(h) and math.isfinite(na) and math.isfinite(nb) and math.isfinite(t)):
        return 0.0, 0.0, 0.0, FLAG_NONFINITE
    if h <= _AXIS_EPS * max(1.0, max(na, nb)):
        if raise_singular and t >= 0.0 and t <= 1.0:
            return 0.0, 0.0, 0.0, FLAG_SINGULAR
        return 0.0, 0.0, 0.0, FLAG_OK
    if gamma == 0.0:
        return 0.0, 0.0, 0.0, FLAG_OK
    product = na * nb
    dot = h * h + t * (t - 1.0)
    plus = product + dot
    if dot < 0.0:
        plus = (h * h) / (product - dot)
    denominator = product * plus
    rc = radius / length
    factor = 1.0
    if radius > 0.0 and model != 0:
        if model == 5:  # Rosenhead-Moore, exact regularized line integral
            na = math.hypot(na, rc)
            nb = math.hypot(nb, rc)
            product = na * nb
            transverse2 = h * h + rc * rc
            dot = transverse2 + t * (t - 1.0)
            plus = product + dot
            if dot < 0.0:
                plus = transverse2 / (product - dot)
            denominator = product * plus
        elif model == 4:  # offset
            denominator = denominator + rc * rc
        else:
            ratio = h / rc
            if not math.isfinite(ratio):
                # The reference divides under np.errstate(over='raise'); an
                # overflowing ratio (subnormal core) must fail the same way.
                return 0.0, 0.0, 0.0, FLAG_NONFINITE
            if model == 1:  # Rankine
                limited = min(ratio, 1.0)
                factor = limited * limited
            elif model == 2:  # Lamb-Oseen with OLAF's exponent cutoff
                if ratio <= _LAMB_OSEEN_CUT:
                    factor = -math.expm1(-1.25643 * (ratio * ratio))
            elif model == 3:  # Vatistas n=2
                if ratio < 1.0:
                    q = ratio * ratio
                    factor = q / math.hypot(1.0, q)
                else:
                    inverse = 1.0 / ratio
                    factor = 1.0 / math.hypot(1.0, inverse * inverse)
    coefficient = (gamma / length) / _FOUR_PI * factor * (na + nb) / denominator
    if not (math.isfinite(denominator) and math.isfinite(coefficient) and math.isfinite(product)
            and math.isfinite(dot)):
        return 0.0, 0.0, 0.0, FLAG_NONFINITE
    return coefficient * cx, coefficient * cy, coefficient * cz, FLAG_OK


@njit(cache=True, inline="always")
def _sum_point(p, points, starts, directions, lengths, gamma, radius, model, raise_singular,
               exclusion_ptr, exclusion_segments, out, flags):
    """Fill out[p] with the ordered sum over all non-excluded segments.

    One definition serves both drivers below, so the per-target arithmetic
    and summation order (hence every result bit) do not depend on threading.
    """
    count = starts.shape[0]
    px, py, pz = points[p, 0], points[p, 1], points[p, 2]
    ux = 0.0
    uy = 0.0
    uz = 0.0
    flag = FLAG_OK
    k = exclusion_ptr[p]
    stop = exclusion_ptr[p + 1]
    for s in range(count):
        if k < stop and exclusion_segments[k] == s:
            k += 1
            continue
        vx, vy, vz, f = pair_velocity(px, py, pz, starts[s, 0], starts[s, 1], starts[s, 2],
                                      directions[s, 0], directions[s, 1], directions[s, 2],
                                      lengths[s], gamma[s], model, radius[s], raise_singular)
        if f != FLAG_OK:
            if flag == FLAG_OK:
                flag = f
            continue
        ux += vx
        uy += vy
        uz += vz
    if not (math.isfinite(ux) and math.isfinite(uy) and math.isfinite(uz)) and flag == FLAG_OK:
        flag = FLAG_NONFINITE
    out[p, 0] = ux
    out[p, 1] = uy
    out[p, 2] = uz
    flags[p] = flag


# Two distinct top-level drivers, not one source compiled twice: Numba's
# on-disk cache is keyed by the Python function (qualified name, bytecode,
# signature) and not by the ``parallel`` flag, so ``njit(parallel=True)(f)``
# and ``njit(parallel=False)(f)`` of the same ``f`` share one cache entry and
# whichever compiled last is loaded for both in every later process. That is
# how the threaded kernel silently ran serially (local benchmark 2026-10-01:
# identical times at 1-16 threads). Threads only pay off for large pair
# counts, hence the serial driver for small calls.

@njit(cache=True, parallel=True)
def sum_segments(points, starts, directions, lengths, gamma, radius, model, raise_singular,
                 exclusion_ptr, exclusion_segments, out, flags):
    """Threaded driver: targets are independent, one thread per chunk of targets."""
    for p in prange(points.shape[0]):
        _sum_point(p, points, starts, directions, lengths, gamma, radius, model, raise_singular,
                   exclusion_ptr, exclusion_segments, out, flags)


@njit(cache=True)
def sum_segments_serial(points, starts, directions, lengths, gamma, radius, model,
                        raise_singular, exclusion_ptr, exclusion_segments, out, flags):
    """Single-thread driver for small pair counts (same per-target code)."""
    for p in range(points.shape[0]):
        _sum_point(p, points, starts, directions, lengths, gamma, radius, model, raise_singular,
                   exclusion_ptr, exclusion_segments, out, flags)


PARALLEL_MIN_PAIRS = 200_000


def exclusion_csr(point_count, pairs):
    """Per-point sorted unique excluded segment IDs from an (e, 2) (point, segment) array.

    Equivalent to the reference's {segment: unique points} dictionary: the
    same set of pairs, here grouped by point in increasing segment order.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros(point_count + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs = np.unique(pairs, axis=0)  # sorted by point, then segment
    counts = np.bincount(pairs[:, 0], minlength=point_count)
    pointer = np.zeros(point_count + 1, dtype=np.int64)
    np.cumsum(counts, out=pointer[1:])
    return pointer, np.ascontiguousarray(pairs[:, 1])


def direct_sum(points, starts, directions, lengths, gamma, radius, model, csr,
               singularity):
    """Return (velocity, any_flag) for validated inputs from segments_velocity.

    ``csr`` is the (pointer, segments) exclusion structure from exclusion_csr.
    """
    pointer, segments = csr
    out = np.empty_like(points)
    flags = np.empty(len(points), dtype=np.int8)
    kernel = sum_segments if len(points) * len(starts) >= PARALLEL_MIN_PAIRS else sum_segments_serial
    kernel(np.ascontiguousarray(points), np.ascontiguousarray(starts),
                 np.ascontiguousarray(directions), np.ascontiguousarray(lengths),
                 np.ascontiguousarray(gamma), np.ascontiguousarray(radius), int(model),
                 singularity == "raise", pointer, segments, out, flags)
    return out, bool(np.any(flags))


# ---------------------------------------------------------------------------
# Per-pair kernel matrix and ordered ring-operator assembly (accelerated
# backends only). These reproduce "one segments_velocity call per panel" of
# the attached-ring influence operators bitwise: each (point, segment) pair is
# the same arithmetic, and each panel column accumulates its signed segment
# contributions sequentially from 0.0 in the reference edge order.
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def pair_matrix(points, starts, directions, lengths, radius, model, raise_singular, out, flags):
    """out[p, s, :] = unit-circulation velocity of segment s at point p; flags per point."""
    count = starts.shape[0]
    for p in prange(points.shape[0]):
        px, py, pz = points[p, 0], points[p, 1], points[p, 2]
        flag = FLAG_OK
        for s in range(count):
            vx, vy, vz, f = pair_velocity(px, py, pz, starts[s, 0], starts[s, 1], starts[s, 2],
                                          directions[s, 0], directions[s, 1], directions[s, 2],
                                          lengths[s], 1.0, model, radius[s], raise_singular)
            if f != FLAG_OK and flag == FLAG_OK:
                flag = f
            out[p, s, 0] = vx
            out[p, s, 1] = vy
            out[p, s, 2] = vz
        flags[p] = flag


@njit(cache=True, parallel=True)
def assemble_columns(pairs, column_ptr, column_segment, column_sign, out):
    """out[p, j, :] = sum_k sign_k * pairs[p, segment_k, :], sequential in k from 0.0."""
    for p in prange(pairs.shape[0]):
        for j in range(column_ptr.shape[0] - 1):
            ux = 0.0
            uy = 0.0
            uz = 0.0
            for k in range(column_ptr[j], column_ptr[j + 1]):
                s = column_segment[k]
                sign = column_sign[k]
                # sign is exactly +1.0 or -1.0: the product equals the signed
                # per-pair velocity of the reference call bit for bit.
                ux += sign * pairs[p, s, 0]
                uy += sign * pairs[p, s, 1]
                uz += sign * pairs[p, s, 2]
            out[p, j, 0] = ux
            out[p, j, 1] = uy
            out[p, j, 2] = uz
