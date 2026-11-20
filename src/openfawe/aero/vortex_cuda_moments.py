# SPDX-License-Identifier: Apache-2.0
"""The tree's node quantities on a CUDA GPU (GPU whole-step contract, revision 6, stage S3'; opt-in).

``vortex_tree.node_moments`` and ``vortex_tree.node_contractions`` compute, for every node of the tree built by
``vortex_tree.build_tree``, the centre of the box around its segments' end points, the radius of the sphere about that
centre that holds them (padded by 1e-12), the largest core radius, the moments m0, m1, m2 of the segments' vortex
vectors about the centre and their contractions w1, t2, w2. Here each node is one thread block of ``THREADS`` threads:
the threads share out the node's segments, and the block combines their parts in a fixed order, so the same input gives
the same result bit for bit on every call.

- The box (minimum and maximum) and the largest core are exact operations in any order, and the centre is
  0.5 * (low + high) as on the CPU, so centre and core equal ``node_moments`` bit for bit.
- The bounding radius and the moments are sums in another order (and the device may fuse a multiply and an add), so
  they agree with the CPU's to rounding. The contractions are the same differences of moments as ``node_contractions``.

The tree itself (order, ranges, children) is still built on the CPU, so it is the CPU's tree. The results stay on the
device for the evaluation kernels of ``vortex_cuda_tree``; for the single precisions the moments and contractions are
also rounded to float32 on the device.
"""
import math

import numpy as np
from numba import cuda, float64

THREADS = 64                     # threads of a node's block; fixed, because the order of the block's sum depends on it
PARTS = 39                       # m0 (3) + m1 (9) + m2 (27)


class Statistics:
    """Counts for the tests (contract revision 6, M1 (e)): calls of ``device_node_quantities`` and kernel launches."""
    totals = dict(calls=0, launches=0)

    @classmethod
    def reset_totals(cls):
        cls.totals = dict(calls=0, launches=0)


@cuda.jit
def _node_kernel(order, lo, hi, starts, ends, alpha, length_vectors, radius, gamma,
                 centre, bound, core, m0, m1, m2, w1, t2, w2):
    node = cuda.blockIdx.x
    t = cuda.threadIdx.x
    box = cuda.shared.array((6, THREADS), float64)
    big = cuda.shared.array(THREADS, float64)
    acc = cuda.shared.array((PARTS, THREADS), float64)
    far = cuda.shared.array(THREADS, float64)
    mid = cuda.shared.array(3, float64)
    a = lo[node]
    b = hi[node]
    l0 = math.inf
    l1 = math.inf
    l2 = math.inf
    h0 = -math.inf
    h1 = -math.inf
    h2 = -math.inf
    largest = 0.0
    k = a + t
    while k < b:
        s = order[k]
        v = starts[s, 0]
        if v < l0:
            l0 = v
        if v > h0:
            h0 = v
        v = ends[s, 0]
        if v < l0:
            l0 = v
        if v > h0:
            h0 = v
        v = starts[s, 1]
        if v < l1:
            l1 = v
        if v > h1:
            h1 = v
        v = ends[s, 1]
        if v < l1:
            l1 = v
        if v > h1:
            h1 = v
        v = starts[s, 2]
        if v < l2:
            l2 = v
        if v > h2:
            h2 = v
        v = ends[s, 2]
        if v < l2:
            l2 = v
        if v > h2:
            h2 = v
        if radius[s] > largest:
            largest = radius[s]
        k += THREADS
    box[0, t] = l0
    box[1, t] = l1
    box[2, t] = l2
    box[3, t] = h0
    box[4, t] = h1
    box[5, t] = h2
    big[t] = largest
    cuda.syncthreads()
    stride = THREADS // 2
    while stride > 0:
        if t < stride:
            for axis in range(3):
                if box[axis, t + stride] < box[axis, t]:
                    box[axis, t] = box[axis, t + stride]
                if box[3 + axis, t + stride] > box[3 + axis, t]:
                    box[3 + axis, t] = box[3 + axis, t + stride]
            if big[t + stride] > big[t]:
                big[t] = big[t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        for axis in range(3):
            mid[axis] = 0.5 * (box[axis, 0] + box[3 + axis, 0])
    cuda.syncthreads()
    c0 = mid[0]
    c1 = mid[1]
    c2 = mid[2]
    part = cuda.local.array(PARTS, float64)
    for q in range(PARTS):
        part[q] = 0.0
    r2 = 0.0
    k = a + t
    while k < b:
        s = order[k]
        x = starts[s, 0] - c0
        y = starts[s, 1] - c1
        z = starts[s, 2] - c2
        q2 = x * x + y * y + z * z
        if q2 > r2:
            r2 = q2
        x = ends[s, 0] - c0
        y = ends[s, 1] - c1
        z = ends[s, 2] - c2
        q2 = x * x + y * y + z * z
        if q2 > r2:
            r2 = q2
        if gamma[s] != 0.0:
            d0 = 0.5 * (starts[s, 0] + ends[s, 0]) - c0
            d1 = 0.5 * (starts[s, 1] + ends[s, 1]) - c1
            d2 = 0.5 * (starts[s, 2] + ends[s, 2]) - c2
            for i in range(3):
                ai = alpha[s, i]
                part[i] += ai
                for j in range(3):
                    dj = d0 if j == 0 else (d1 if j == 1 else d2)
                    part[3 + 3 * i + j] += ai * dj
                    lj = length_vectors[s, j]
                    for l in range(3):
                        dl = d0 if l == 0 else (d1 if l == 1 else d2)
                        part[12 + 9 * i + 3 * j + l] += ai * (dj * dl + lj * length_vectors[s, l] / 12.0)
        k += THREADS
    for q in range(PARTS):
        acc[q, t] = part[q]
    far[t] = r2
    cuda.syncthreads()
    stride = THREADS // 2
    while stride > 0:
        if t < stride:
            for q in range(PARTS):
                acc[q, t] += acc[q, t + stride]
            if far[t + stride] > far[t]:
                far[t] = far[t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        centre[node, 0] = c0
        centre[node, 1] = c1
        centre[node, 2] = c2
        bound[node] = math.sqrt(far[0]) * (1.0 + 1e-12)
        core[node] = big[0]
        for i in range(3):
            m0[node, i] = acc[i, 0]
            for j in range(3):
                m1[node, i, j] = acc[3 + 3 * i + j, 0]
                for l in range(3):
                    m2[node, i, j, l] = acc[12 + 9 * i + 3 * j + l, 0]
        w1[node, 0] = m1[node, 1, 2] - m1[node, 2, 1]
        w1[node, 1] = m1[node, 2, 0] - m1[node, 0, 2]
        w1[node, 2] = m1[node, 0, 1] - m1[node, 1, 0]
        for i in range(3):
            t2[node, i] = m2[node, i, 0, 0] + m2[node, i, 1, 1] + m2[node, i, 2, 2]
        for k2 in range(3):
            w2[node, 0, k2] = m2[node, 1, 2, k2] - m2[node, 2, 1, k2]
            w2[node, 1, k2] = m2[node, 2, 0, k2] - m2[node, 0, 2, k2]
            w2[node, 2, k2] = m2[node, 0, 1, k2] - m2[node, 1, 0, k2]


@cuda.jit
def _to32(source, target):
    i = cuda.grid(1)
    if i < source.shape[0]:
        target[i] = source[i]


def _rounded32(array):
    """A float32 copy, made on the device, of a contiguous float64 device array."""
    flat = array.reshape(array.size)
    out = cuda.device_array(array.size, dtype=np.float32)
    block = 256
    _to32[(array.size + block - 1) // block, block](flat, out)
    return out.reshape(array.shape)


def device_node_quantities(order, lo, hi, starts, ends, alpha, length_vectors, radius, gamma, *, single=False,
                           device_starts=None, device_order=None):
    """The node quantities of ``vortex_tree.node_moments`` and ``node_contractions`` as device arrays: a dict with
    ``centre``, ``bound``, ``core`` (float64) and ``m0``, ``m1``, ``m2``, ``w1``, ``t2``, ``w2`` (float64, or float32
    rounded on the device when ``single``, with the float64 ones as ``m0_64`` ...). ``device_starts``: the segments'
    starts already on the device (float64); ``device_order``: ``order`` already on the device (int64; the GPU build's)."""
    count = len(lo)
    to = cuda.to_device
    f64 = lambda a: to(np.ascontiguousarray(a, dtype=np.float64))   # noqa: E731
    i64 = lambda a: to(np.ascontiguousarray(a, dtype=np.int64))     # noqa: E731
    d_starts = f64(starts) if device_starts is None else device_starts
    out = dict(centre=cuda.device_array((count, 3)), bound=cuda.device_array(count), core=cuda.device_array(count),
               m0=cuda.device_array((count, 3)), m1=cuda.device_array((count, 3, 3)), m2=cuda.device_array((count, 3, 3, 3)),
               w1=cuda.device_array((count, 3)), t2=cuda.device_array((count, 3)), w2=cuda.device_array((count, 3, 3)))
    Statistics.totals["calls"] += 1
    if count:
        _node_kernel[count, THREADS](i64(order) if device_order is None else device_order, i64(lo), i64(hi), d_starts, f64(ends), f64(alpha), f64(length_vectors),
                                     f64(radius), f64(gamma), out["centre"], out["bound"], out["core"], out["m0"],
                                     out["m1"], out["m2"], out["w1"], out["t2"], out["w2"])
        Statistics.totals["launches"] += 1
    if single:                     # the float64 ones stay under NAME_64 (a CPU recomputation of suspect targets reads them)
        for name in ("m0", "m1", "m2", "w1", "t2", "w2"):
            out[name + "_64"] = out[name]
            out[name] = _rounded32(out[name]) if count else cuda.device_array(out[name].shape, dtype=np.float32)
    return out


def node_quantities(order, lo, hi, starts, ends, alpha, length_vectors, radius, gamma):
    """``device_node_quantities`` copied back to the host in float64: (centre, bound, core, m0, m1, m2, w1, t2, w2)."""
    d = device_node_quantities(order, lo, hi, starts, ends, alpha, length_vectors, radius, gamma)
    return tuple(d[name].copy_to_host() for name in ("centre", "bound", "core", "m0", "m1", "m2", "w1", "t2", "w2"))
