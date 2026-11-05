# SPDX-License-Identifier: Apache-2.0
# Tree-code evaluation on a CUDA GPU: the traversal, acceptance test and far-field expansion of vortex_tree.tree_velocity;
# near-field pairs with the kernels of vortex_cuda (themselves vortex_numba.pair_velocity, a translation of OpenFAST v5.0.0
# FVW_BiotSavart.f90 ui_seg_11/ui_seg/ui_quad_n1; see third_party/OPENFAST-NOTICE.md and LICENSE-OpenFAST.txt).
"""GPU evaluation of the hierarchical (Barnes-Hut type) segment sum (GPU tree-code contract, docs/tasks/OF-0006-accel20261001).

The tree - order, node ranges, centres, bounding radii, largest cores, moments and their contractions - is built on the
CPU by the production functions of ``vortex_tree`` exactly as ``tree_sum`` builds it. Each target point is one GPU
thread that walks the tree like ``vortex_tree.tree_velocity``: the same stack order (left child first), the same
acceptance test (bounding radius below theta times the distance to the centre, the core clearance, no excluded segment
of the target inside the node), the same leaf loop that skips the target's excluded segments, the same flag rules and
the same second-order expansion. The acceptance test is evaluated in float64 in every precision, so the nodes chosen are
those of the CPU tree code; only the arithmetic differs:

- ``fp64``: near-field pairs with ``vortex_cuda._pair64`` (the code object of ``pair_velocity``), the expansion with a
  float64 device form of ``far_velocity`` (same arithmetic and order; the tuple subscript is replaced by an equivalent
  conditional choice of the component).
- ``mixed``: target-minus-segment-start and target-minus-node-centre vectors are formed in float64 and rounded to
  float32; pairs with ``vortex_cuda._pair32``, the expansion in float32 (moments and contractions rounded to float32 on
  the host, float32 constants); every term is added to a float64 accumulator.
- ``fp32``: as ``mixed`` with a float32 accumulator.

Single-precision targets with a pair near a segment's axis are "suspect" (the rule of ``vortex_cuda._pair32``). With
``CudaSettings.suspect='cpu'`` (the default) they are recomputed whole on the CPU by ``vortex_tree.tree_velocity`` in
float64, so their values are those of the CPU tree code. With ``suspect='gpu64'`` (whole-step GPU contract, revision 3)
only the suspect pairs are evaluated again, on the GPU, by ``vortex_cuda._pair64`` from float64 inputs, and no target goes
back to the CPU; the other pairs and the far field of such a target stay in single precision (``_kernel_gpu64``).
Each thread has a 512-deep stack like the CPU code; an overflow (which the CPU code does not check) raises. Launches cover
at most ``TARGETS_PER_LAUNCH`` targets (the Windows display watchdog stops a launch after about 2 s).

Wired into ``segments_velocity`` as ``backend='cuda_tree'`` (whole-step GPU contract, stage S1).
"""
import math
import time

import numpy as np
from numba import cuda

from . import vortex_cuda as VC
from . import vortex_tree as VT

STACK = 512
TARGETS_PER_LAUNCH = 65536
PRECISIONS = ("fp64", "mixed", "fp32")
INV_FOUR_PI = VT._INV_FOUR_PI
F0, F1 = VC.F0, VC.F1
F3, F1_5, F7_5 = np.float32(3.0), np.float32(1.5), np.float32(7.5)
F_INV_FOUR_PI = np.float32(VT._INV_FOUR_PI)
FLAG_NONFINITE = VC.FLAG_NONFINITE
_pair64 = VC._pair64
_pair32 = VC._pair32


@cuda.jit(device=True)
def _far64(rx, ry, rz, node, m0, m1, m2, w1, t2, w2, order):
    """``vortex_tree.far_velocity`` on the device, float64; same arithmetic in the same order."""
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
        for j in range(3):
            rj = rx if j == 0 else (ry if j == 1 else rz)
            for k in range(3):
                rk = rx if k == 0 else (ry if k == 1 else rz)
                rr = rj * rk
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
    return INV_FOUR_PI * u0, INV_FOUR_PI * u1, INV_FOUR_PI * u2


@cuda.jit(device=True)
def _far32(rx, ry, rz, node, m0, m1, m2, w1, t2, w2, order):
    """The same expansion in float32 (float32 inputs, moments and constants)."""
    r2 = rx * rx + ry * ry + rz * rz
    inv3 = F1 / (r2 * math.sqrt(r2))
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
        u0 += F3 * inv5 * (v1 * rz - v2 * ry) - inv3 * w1[node, 0]
        u1 += F3 * inv5 * (v2 * rx - v0 * rz) - inv3 * w1[node, 1]
        u2 += F3 * inv5 * (v0 * ry - v1 * rx) - inv3 * w1[node, 2]
    if order >= 2:
        inv7 = inv5 / r2
        q0 = F0
        q1 = F0
        q2 = F0
        for j in range(3):
            rj = rx if j == 0 else (ry if j == 1 else rz)
            for k in range(3):
                rk = rx if k == 0 else (ry if k == 1 else rz)
                rr = rj * rk
                q0 += m2[node, 0, j, k] * rr
                q1 += m2[node, 1, j, k] * rr
                q2 += m2[node, 2, j, k] * rr
        b0 = F7_5 * inv7 * q0 - F1_5 * inv5 * t2[node, 0]
        b1 = F7_5 * inv7 * q1 - F1_5 * inv5 * t2[node, 1]
        b2 = F7_5 * inv7 * q2 - F1_5 * inv5 * t2[node, 2]
        s0 = w2[node, 0, 0] * rx + w2[node, 0, 1] * ry + w2[node, 0, 2] * rz
        s1 = w2[node, 1, 0] * rx + w2[node, 1, 1] * ry + w2[node, 1, 2] * rz
        s2 = w2[node, 2, 0] * rx + w2[node, 2, 1] * ry + w2[node, 2, 2] * rz
        u0 += (b1 * rz - b2 * ry) - F3 * inv5 * s0
        u1 += (b2 * rx - b0 * rz) - F3 * inv5 * s1
        u2 += (b0 * ry - b1 * rx) - F3 * inv5 * s2
    return F_INV_FOUR_PI * u0, F_INV_FOUR_PI * u1, F_INV_FOUR_PI * u2


_KERNELS = {}


def _kernel(precision, stack_size=STACK):
    """The kernel of one precision (the stack depth is a compile-time constant)."""
    key = (precision, int(stack_size))
    if key in _KERNELS:
        return _KERNELS[key]
    depth = int(stack_size)
    if precision == "fp64":
        @cuda.jit
        def kernel(points, starts, directions, lengths, gamma, radius, model, raise_singular, order, lo, hi, left,
                   right, centre, bound, core, m0, m1, m2, w1, t2, w2, theta, expansion_order, clearance, ex_ptr,
                   ex_pos, first, count, out, flags, far_counts, overflow):
            i = cuda.grid(1)
            if i >= count:
                return
            p = first + i
            x0 = points[p, 0]
            x1 = points[p, 1]
            x2 = points[p, 2]
            u0 = 0.0
            u1 = 0.0
            u2 = 0.0
            flag = 0
            far = 0
            e0 = ex_ptr[p]
            e1 = ex_ptr[p + 1]
            stack = cuda.local.array(depth, np.int64)
            stack[0] = 0
            top = 1
            while top > 0:
                top -= 1
                node = stack[top]
                rx = x0 - centre[node, 0]
                ry = x1 - centre[node, 1]
                rz = x2 - centre[node, 2]
                distance = math.sqrt(rx * rx + ry * ry + rz * rz)
                accept = bound[node] < theta * distance and distance - bound[node] >= clearance * core[node]
                if accept:
                    for e in range(e0, e1):
                        if ex_pos[e] >= lo[node] and ex_pos[e] < hi[node]:
                            accept = False
                            break
                if accept:
                    fx, fy, fz = _far64(rx, ry, rz, node, m0, m1, m2, w1, t2, w2, expansion_order)
                    u0 += fx
                    u1 += fy
                    u2 += fz
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
                        vx, vy, vz, f = _pair64(x0, x1, x2, starts[s, 0], starts[s, 1], starts[s, 2],
                                                   directions[s, 0], directions[s, 1], directions[s, 2], lengths[s],
                                                   gamma[s], model, radius[s], raise_singular)
                        if f != 0:
                            if flag == 0:
                                flag = f
                            continue
                        u0 += vx
                        u1 += vy
                        u2 += vz
                else:
                    if top + 2 > depth:
                        overflow[0] = 1
                        break
                    stack[top] = right[node]
                    stack[top + 1] = left[node]
                    top += 2
            if not (math.isfinite(u0) and math.isfinite(u1) and math.isfinite(u2)) and flag == 0:
                flag = FLAG_NONFINITE
            out[p, 0] = u0
            out[p, 1] = u1
            out[p, 2] = u2
            flags[p] = flag
            far_counts[p] = far
    else:
        wide = precision == "mixed"
        accumulator = np.float64 if wide else np.float32

        @cuda.jit
        def kernel(points, starts, directions, lengths, gamma, radius, model, raise_singular, order, lo, hi, left,
                   right, centre, bound, core, m0, m1, m2, w1, t2, w2, theta, expansion_order, clearance, ex_ptr,
                   ex_pos, first, count, out, flags, far_counts, overflow, suspects):
            i = cuda.grid(1)
            if i >= count:
                return
            p = first + i
            x0 = points[p, 0]
            x1 = points[p, 1]
            x2 = points[p, 2]
            u0 = accumulator(0.0)
            u1 = accumulator(0.0)
            u2 = accumulator(0.0)
            flag = 0
            far = 0
            suspect = 0
            e0 = ex_ptr[p]
            e1 = ex_ptr[p + 1]
            stack = cuda.local.array(depth, np.int64)
            stack[0] = 0
            top = 1
            while top > 0:
                top -= 1
                node = stack[top]
                rx = x0 - centre[node, 0]                # float64: the acceptance test is that of the CPU code
                ry = x1 - centre[node, 1]
                rz = x2 - centre[node, 2]
                distance = math.sqrt(rx * rx + ry * ry + rz * rz)
                accept = bound[node] < theta * distance and distance - bound[node] >= clearance * core[node]
                if accept:
                    for e in range(e0, e1):
                        if ex_pos[e] >= lo[node] and ex_pos[e] < hi[node]:
                            accept = False
                            break
                if accept:
                    fx, fy, fz = _far32(np.float32(rx), np.float32(ry), np.float32(rz), node, m0, m1, m2, w1, t2, w2,
                                        expansion_order)
                    u0 += accumulator(fx)
                    u1 += accumulator(fy)
                    u2 += accumulator(fz)
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
                        vx, vy, vz, f, w = _pair32(np.float32(x0 - starts[s, 0]), np.float32(x1 - starts[s, 1]),
                                                      np.float32(x2 - starts[s, 2]), directions[s, 0], directions[s, 1],
                                                      directions[s, 2], lengths[s], gamma[s], model, radius[s],
                                                      raise_singular)
                        if w != 0:
                            suspect = 1
                        if f != 0:
                            if flag == 0:
                                flag = f
                            continue
                        u0 += accumulator(vx)
                        u1 += accumulator(vy)
                        u2 += accumulator(vz)
                else:
                    if top + 2 > depth:
                        overflow[0] = 1
                        break
                    stack[top] = right[node]
                    stack[top + 1] = left[node]
                    top += 2
            if not (math.isfinite(u0) and math.isfinite(u1) and math.isfinite(u2)) and flag == 0:
                flag = FLAG_NONFINITE
            out[p, 0] = u0
            out[p, 1] = u1
            out[p, 2] = u2
            flags[p] = flag
            far_counts[p] = far
            suspects[p] = suspect
    _KERNELS[key] = kernel
    return kernel


def _kernel_gpu64(precision, stack_size=STACK):
    """The single-precision kernel with ``suspect='gpu64'`` (whole-step GPU contract, revision 3): as ``_kernel`` for
    ``mixed``/``fp32`` except for one branch and four more arguments - a pair that ``_pair32`` marks suspect is
    evaluated again by ``_pair64`` from the float64 inputs (``d64``, ``l64``, ``g64``, ``r64``; points and starts are
    float64 already) and its float64 velocity and flag take the place of the float32 pair's. ``suspects`` holds the
    number of such pairs per target.
    A separate function, so that the kernel of ``suspect='cpu'`` (stage S1) is compiled from unchanged source."""
    key = (precision, int(stack_size), "gpu64")
    if key in _KERNELS:
        return _KERNELS[key]
    if precision not in ("mixed", "fp32"):
        raise ValueError("suspect='gpu64' is for the single precisions only")
    depth = int(stack_size)
    wide = precision == "mixed"
    accumulator = np.float64 if wide else np.float32

    @cuda.jit
    def kernel(points, starts, directions, lengths, gamma, radius, model, raise_singular, order, lo, hi, left,
               right, centre, bound, core, m0, m1, m2, w1, t2, w2, theta, expansion_order, clearance, ex_ptr,
               ex_pos, first, count, out, flags, far_counts, overflow, suspects, d64, l64, g64, r64):
        i = cuda.grid(1)
        if i >= count:
            return
        p = first + i
        x0 = points[p, 0]
        x1 = points[p, 1]
        x2 = points[p, 2]
        u0 = accumulator(0.0)
        u1 = accumulator(0.0)
        u2 = accumulator(0.0)
        flag = 0
        far = 0
        suspect = 0
        e0 = ex_ptr[p]
        e1 = ex_ptr[p + 1]
        stack = cuda.local.array(depth, np.int64)
        stack[0] = 0
        top = 1
        while top > 0:
            top -= 1
            node = stack[top]
            rx = x0 - centre[node, 0]                # float64: the acceptance test is that of the CPU code
            ry = x1 - centre[node, 1]
            rz = x2 - centre[node, 2]
            distance = math.sqrt(rx * rx + ry * ry + rz * rz)
            accept = bound[node] < theta * distance and distance - bound[node] >= clearance * core[node]
            if accept:
                for e in range(e0, e1):
                    if ex_pos[e] >= lo[node] and ex_pos[e] < hi[node]:
                        accept = False
                        break
            if accept:
                fx, fy, fz = _far32(np.float32(rx), np.float32(ry), np.float32(rz), node, m0, m1, m2, w1, t2, w2,
                                    expansion_order)
                u0 += accumulator(fx)
                u1 += accumulator(fy)
                u2 += accumulator(fz)
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
                    vx, vy, vz, f, w = _pair32(np.float32(x0 - starts[s, 0]), np.float32(x1 - starts[s, 1]),
                                                  np.float32(x2 - starts[s, 2]), directions[s, 0], directions[s, 1],
                                                  directions[s, 2], lengths[s], gamma[s], model, radius[s],
                                                  raise_singular)
                    if w != 0:                       # revision 3: this pair again in float64, on the GPU
                        suspect += 1
                        ax, ay, az, f64 = _pair64(x0, x1, x2, starts[s, 0], starts[s, 1], starts[s, 2], d64[s, 0],
                                                  d64[s, 1], d64[s, 2], l64[s], g64[s], model, r64[s], raise_singular)
                        if f64 != 0:
                            if flag == 0:
                                flag = f64
                            continue
                        u0 += accumulator(ax)
                        u1 += accumulator(ay)
                        u2 += accumulator(az)
                        continue
                    if f != 0:
                        if flag == 0:
                            flag = f
                        continue
                    u0 += accumulator(vx)
                    u1 += accumulator(vy)
                    u2 += accumulator(vz)
            else:
                if top + 2 > depth:
                    overflow[0] = 1
                    break
                stack[top] = right[node]
                stack[top + 1] = left[node]
                top += 2
        if not (math.isfinite(u0) and math.isfinite(u1) and math.isfinite(u2)) and flag == 0:
            flag = FLAG_NONFINITE
        out[p, 0] = u0
        out[p, 1] = u1
        out[p, 2] = u2
        flags[p] = flag
        far_counts[p] = far
        suspects[p] = suspect
    _KERNELS[key] = kernel
    return kernel


class Statistics:
    last = {}
    last_far_counts = None                 # per-target accepted-node counts of the last call (contract D1 report)
    totals = dict(calls=0, suspect=0, launches=0)

    @classmethod
    def reset_totals(cls):
        cls.totals = dict(calls=0, suspect=0, launches=0)


def cuda_tree_sum(points, starts, directions, lengths, gamma, radius, model, csr, singularity, tree, settings, *,
                  targets_per_launch=None, stack_size=STACK, block=None):
    """Return (velocity, any_flag) like ``vortex_tree.tree_sum`` for the same inputs, evaluated on the GPU.

    ``tree`` is a ``TreeSettings``, ``settings`` a ``CudaSettings`` (its precision and block size). Timing of the CPU
    build, the transfers and the evaluation goes to ``Statistics.last``.
    """
    if not VC.available():
        raise ValueError(f"the cuda tree evaluation needs a CUDA device ({VC.probe_error()})")
    if int(model) not in VT.SUPPORTED_MODELS and np.any(np.asarray(radius) > 0):
        raise ValueError("tree backend supports core models NONE, RANKINE and LAMB_OSEEN only")
    if tree.theta >= 1.0:
        raise ValueError("tree theta must be below one")
    precision = settings.precision
    gpu64 = getattr(settings, "suspect", "cpu") == "gpu64"
    if gpu64 and precision == "fp64":
        raise ValueError("suspect='gpu64' is for the single precisions only")
    block = int(settings.block if block is None else block)
    per = int(TARGETS_PER_LAUNCH if targets_per_launch is None else targets_per_launch)
    points = np.ascontiguousarray(points, dtype=np.float64)
    starts = np.ascontiguousarray(starts, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    lengths = np.ascontiguousarray(lengths, dtype=np.float64)
    gamma = np.ascontiguousarray(gamma, dtype=np.float64)
    radius = np.ascontiguousarray(radius, dtype=np.float64)
    n, m = len(points), len(starts)
    out = np.zeros((n, 3))
    if n == 0 or m == 0:
        Statistics.last = dict(points=n, segments=m, launches=0, suspect=0)
        if gpu64:
            Statistics.last.update(suspect_mode="gpu64", suspect_targets=0, suspect_pairs=0)
        return out, False
    tick = time.perf_counter()
    length_vectors = directions * lengths[:, None]
    ends = starts + length_vectors
    alpha = gamma[:, None] * length_vectors
    mid = 0.5 * (starts + ends)
    order, lo, hi, left, right, _ = VT.build_tree(mid, int(tree.leaf_size))
    build_s = time.perf_counter() - tick
    tick = time.perf_counter()
    centre, bound, core, m0, m1, m2 = VT.node_moments(order, lo, hi, starts, ends, alpha, length_vectors, radius, gamma)
    w1, t2, w2 = VT.node_contractions(m1, m2)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    pointer, segments = csr
    pointer = np.ascontiguousarray(pointer, dtype=np.int64)
    segments = np.ascontiguousarray(segments, dtype=np.int64)
    positions = VT._positions(pointer, segments, inverse) if len(segments) else segments
    moments_s = time.perf_counter() - tick
    raise_singular = singularity == "raise"
    tick = time.perf_counter()
    single = precision != "fp64"
    real = np.float32 if single else np.float64
    to = cuda.to_device
    d = dict(points=to(points), starts=to(starts), order=to(np.ascontiguousarray(order, dtype=np.int64)),
             lo=to(lo), hi=to(hi), left=to(left), right=to(right), centre=to(centre), bound=to(bound), core=to(core),
             ex_ptr=to(pointer), ex_pos=to(positions if len(positions) else np.zeros(1, dtype=np.int64)))
    for name, value in (("directions", directions), ("lengths", lengths), ("gamma", gamma), ("radius", radius),
                        ("m0", m0), ("m1", m1), ("m2", m2), ("w1", w1), ("t2", t2), ("w2", w2)):
        d[name] = to(np.ascontiguousarray(value, dtype=real))
    d_out = cuda.to_device(np.zeros((n, 3), dtype=np.float64 if precision != "fp32" else np.float32))
    d_flags = cuda.to_device(np.zeros(n, dtype=np.int8))
    d_far = cuda.to_device(np.zeros(n, dtype=np.int64))
    d_overflow = cuda.to_device(np.zeros(1, dtype=np.int8))
    d_suspects = cuda.to_device(np.zeros(n, dtype=np.int32 if gpu64 else np.int8)) if single else None
    wide = (to(directions), to(lengths), to(gamma), to(radius)) if gpu64 else ()   # float64 copies (revision 3)
    upload_s = time.perf_counter() - tick
    kernel = _kernel_gpu64(precision, stack_size) if gpu64 else _kernel(precision, stack_size)
    args = (d["points"], d["starts"], d["directions"], d["lengths"], d["gamma"], d["radius"], int(model), raise_singular,
            d["order"], d["lo"], d["hi"], d["left"], d["right"], d["centre"], d["bound"], d["core"], d["m0"], d["m1"],
            d["m2"], d["w1"], d["t2"], d["w2"], float(tree.theta), int(tree.order), float(tree.core_clearance),
            d["ex_ptr"], d["ex_pos"])
    launches, longest = 0, 0.0
    tick = time.perf_counter()
    for first in range(0, n, per):
        count = min(per, n - first)
        grid = (count + block - 1) // block
        started = time.perf_counter()
        tail = (first, count, d_out, d_flags, d_far, d_overflow) + ((d_suspects,) if single else ()) + wide
        kernel[grid, block](*args, *tail)
        cuda.synchronize()
        longest = max(longest, time.perf_counter() - started)
        launches += 1
    evaluate_s = time.perf_counter() - tick
    tick = time.perf_counter()
    if d_overflow.copy_to_host()[0]:
        raise RuntimeError(f"tree traversal stack overflow on the GPU (depth {stack_size})")
    out[:] = d_out.copy_to_host()
    flags = d_flags.copy_to_host()
    far = d_far.copy_to_host()
    if gpu64:                              # revision 3: suspect pairs were done on the GPU; no target goes back to the CPU
        pair_counts = d_suspects.copy_to_host()
        suspect = np.zeros(0, dtype=np.int64)
    else:
        suspect = np.flatnonzero(d_suspects.copy_to_host()) if single else np.zeros(0, dtype=np.int64)
    download_s = time.perf_counter() - tick
    tick = time.perf_counter()
    if len(suspect):                       # contract D3: suspect targets recomputed whole on the CPU tree code (float64)
        sub_ptr = np.zeros(len(suspect) + 1, dtype=np.int64)
        np.cumsum(pointer[suspect + 1] - pointer[suspect], out=sub_ptr[1:])
        sub_pos = (np.concatenate([positions[pointer[i]:pointer[i + 1]] for i in suspect]) if len(positions)
                   else np.zeros(0, dtype=np.int64))
        sub_pos = np.ascontiguousarray(sub_pos, dtype=np.int64)
        again = np.zeros((len(suspect), 3))
        again_flags = np.zeros(len(suspect), dtype=np.int8)
        again_far = np.zeros(len(suspect), dtype=np.int64)
        VT.tree_velocity(np.ascontiguousarray(points[suspect]), starts, directions, lengths, gamma, radius, int(model),
                         raise_singular, order, lo, hi, left, right, centre, bound, core, m0, m1, m2, w1, t2, w2,
                         float(tree.theta), int(tree.order), float(tree.core_clearance), sub_ptr, sub_pos, again,
                         again_flags, again_far)
        out[suspect] = again
        flags[suspect] = again_flags
        far[suspect] = again_far
    recompute_s = time.perf_counter() - tick
    Statistics.last = dict(points=n, segments=m, nodes=len(lo), precision=precision, launches=launches,
                           targets_per_launch=per, longest_launch_s=longest, suspect=int(len(suspect)),
                           mean_far_nodes=float(far.mean()), build_s=build_s, moments_s=moments_s, upload_s=upload_s,
                           evaluate_s=evaluate_s, download_s=download_s, recompute_s=recompute_s)
    Statistics.last_far_counts = far
    Statistics.totals["calls"] += 1
    Statistics.totals["launches"] += launches
    Statistics.totals["suspect"] += int(len(suspect))
    if gpu64:
        targets, pairs = int(np.count_nonzero(pair_counts)), int(pair_counts.sum(dtype=np.int64))
        Statistics.last.update(suspect_mode="gpu64", suspect_targets=targets, suspect_pairs=pairs)
        Statistics.totals["suspect_targets"] = Statistics.totals.get("suspect_targets", 0) + targets
        Statistics.totals["suspect_pairs"] = Statistics.totals.get("suspect_pairs", 0) + pairs
    return out, bool(np.any(flags))
