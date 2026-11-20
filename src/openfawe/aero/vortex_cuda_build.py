# SPDX-License-Identifier: Apache-2.0
"""The tree of the hierarchical segment sum built on a CUDA GPU (GPU whole-step contract, revision 7, stage S3''; opt-in).

``vortex_tree.build_tree`` splits the segments level by level: for every node it takes the box around the node's
segment midpoints, the axis of largest extent (on a tie the lower axis), and - when the node holds more than ``leaf_size``
segments and the extent is above zero - sorts the node's range of ``order`` by the midpoint coordinate along that axis
with a stable merge sort and cuts it in the middle. Here the same rule runs on the GPU, level by level:

- one thread block per node finds the box (minimum and maximum, exact operations), the axis and whether it splits;
- the nodes that split are sorted on the GPU by the pair (coordinate, position in the node's range): no two elements
  have the same pair, so every correct sort gives one order, and it is the stable sort's order. The sort is a bitonic
  network whose comparators all put the smaller element first; a node's length is padded to a power of two and the
  padded places are never touched (they would hold infinity, and such a comparator never moves an element);
- the split flags go back to the host, which forms the next level and finally numbers the nodes with ``vortex_tree``'s
  own ``_next_level`` and ``_renumber`` (the CPU build's code).

So ``order``, ``lo``, ``hi``, ``left``, ``right`` and the node count equal ``build_tree``'s element for element. ``order``
stays on the device for the moments and the evaluation; a host copy is returned as well.
"""
import numpy as np
from numba import cuda, float64, int64

from . import vortex_tree as VT

TILE = 1024                 # elements sorted in one block's shared memory (TILE // 2 threads, one comparator each)
BOX_THREADS = 128
STEP_THREADS = 256


class Statistics:
    """Counts for the tests: calls of ``device_build_tree``, levels and kernel launches."""
    totals = dict(calls=0, levels=0, launches=0)

    @classmethod
    def reset_totals(cls):
        cls.totals = dict(calls=0, levels=0, launches=0)


@cuda.jit
def _level_box(mid, order, lo_level, hi_level, leaf_size, split, axis_of):
    node = cuda.blockIdx.x
    t = cuda.threadIdx.x
    box = cuda.shared.array((6, BOX_THREADS), float64)
    a = lo_level[node]
    b = hi_level[node]
    l0 = np.inf
    l1 = np.inf
    l2 = np.inf
    h0 = -np.inf
    h1 = -np.inf
    h2 = -np.inf
    k = a + t
    while k < b:
        s = order[k]
        v = mid[s, 0]
        if v < l0:
            l0 = v
        if v > h0:
            h0 = v
        v = mid[s, 1]
        if v < l1:
            l1 = v
        if v > h1:
            h1 = v
        v = mid[s, 2]
        if v < l2:
            l2 = v
        if v > h2:
            h2 = v
        k += BOX_THREADS
    box[0, t] = l0
    box[1, t] = l1
    box[2, t] = l2
    box[3, t] = h0
    box[4, t] = h1
    box[5, t] = h2
    cuda.syncthreads()
    stride = BOX_THREADS // 2
    while stride > 0:
        if t < stride:
            for axis in range(3):
                if box[axis, t + stride] < box[axis, t]:
                    box[axis, t] = box[axis, t + stride]
                if box[3 + axis, t + stride] > box[3 + axis, t]:
                    box[3 + axis, t] = box[3 + axis, t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        split[node] = 0
        axis_of[node] = 0
        if b - a > leaf_size:
            axis = 0
            extent = box[3, 0] - box[0, 0]
            for candidate in range(1, 3):
                if box[3 + candidate, 0] - box[candidate, 0] > extent:
                    extent = box[3 + candidate, 0] - box[candidate, 0]
                    axis = candidate
            if extent > 0.0:
                split[node] = 1
                axis_of[node] = axis


@cuda.jit
def _fill(mid, order, seg_lo, seg_len, seg_axis, key, tie):
    s = cuda.blockIdx.x
    a = seg_lo[s]
    m = seg_len[s]
    axis = seg_axis[s]
    k = cuda.threadIdx.x
    while k < m:
        key[a + k] = mid[order[a + k], axis]
        tie[a + k] = k
        k += cuda.blockDim.x


@cuda.jit(device=True)
def _greater(k1, t1, k2, t2):
    return k1 > k2 or (k1 == k2 and t1 > t2)


@cuda.jit
def _tile_kernel(key, tie, order, seg_lo, seg_len, tile_size, tiles, full):
    """One tile of ``tile_size`` places of one node in shared memory: the whole network up to ``tile_size`` (``full``),
    or only the strides below ``tile_size`` of a larger merge (the flip and the larger strides were done in global
    memory)."""
    sk = cuda.shared.array(TILE, float64)
    st = cuda.shared.array(TILE, int64)
    so = cuda.shared.array(TILE, int64)
    s = cuda.blockIdx.x // tiles
    base = (cuda.blockIdx.x % tiles) * tile_size
    a = seg_lo[s]
    m = seg_len[s]
    t = cuda.threadIdx.x
    half = tile_size // 2
    for r in range(2):                          # e = t, t + half (one integer type; GPU contract rev 13)
        e = t + r * half
        if t < half and base + e < m:
            sk[e] = key[a + base + e]
            st[e] = tie[a + base + e]
            so[e] = order[a + base + e]
    cuda.syncthreads()
    size = 2 if full else tile_size
    while size <= tile_size:
        if full:
            stride = size // 2                  # the flip of this size
            if t < half:
                block = t // stride
                offset = t % stride
                i = block * size + offset
                j = block * size + size - 1 - offset
                if base + j < m and _greater(sk[i], st[i], sk[j], st[j]):
                    sk[i], sk[j] = sk[j], sk[i]
                    st[i], st[j] = st[j], st[i]
                    so[i], so[j] = so[j], so[i]
            cuda.syncthreads()
            stride = size // 4
        else:
            stride = tile_size // 2
        while stride > 0:
            if t < half:
                i = (t // stride) * 2 * stride + t % stride
                j = i + stride
                if base + j < m and _greater(sk[i], st[i], sk[j], st[j]):
                    sk[i], sk[j] = sk[j], sk[i]
                    st[i], st[j] = st[j], st[i]
                    so[i], so[j] = so[j], so[i]
            cuda.syncthreads()
            stride //= 2
        size *= 2
    for r in range(2):                          # e = t, t + half (one integer type; GPU contract rev 13)
        e = t + r * half
        if t < half and base + e < m:
            key[a + base + e] = sk[e]
            tie[a + base + e] = st[e]
            order[a + base + e] = so[e]


@cuda.jit
def _global_step(key, tie, order, seg_lo, seg_len, segments, padded, size, stride, flip):
    """One stage of the network in global memory over every node of the level (pairs further apart than a tile)."""
    tid = cuda.grid(1)
    pairs = padded // 2
    s = tid // pairs
    if s >= segments:
        return
    t = tid % pairs
    if flip:
        half = size // 2
        i = (t // half) * size + t % half
        j = (t // half) * size + size - 1 - t % half
    else:
        i = (t // stride) * 2 * stride + t % stride
        j = i + stride
    if j >= seg_len[s]:
        return
    a = seg_lo[s]
    if _greater(key[a + i], tie[a + i], key[a + j], tie[a + j]):
        key[a + i], key[a + j] = key[a + j], key[a + i]
        tie[a + i], tie[a + j] = tie[a + j], tie[a + i]
        order[a + i], order[a + j] = order[a + j], order[a + i]


def _sort_segments(d_mid, d_order, d_key, d_tie, seg_lo, seg_len, seg_axis):
    """Sort every given node's range of ``order`` by (midpoint coordinate along its axis, position in the range)."""
    to = cuda.to_device
    segments = len(seg_lo)
    longest = int(seg_len.max())
    padded = 1 << max(int(longest - 1).bit_length(), 1)
    d_lo, d_len = to(np.ascontiguousarray(seg_lo, dtype=np.int64)), to(np.ascontiguousarray(seg_len, dtype=np.int64))
    _fill[segments, STEP_THREADS](d_mid, d_order, d_lo, d_len, to(np.ascontiguousarray(seg_axis, dtype=np.int64)), d_key, d_tie)
    launches = 1
    if padded <= TILE:
        _tile_kernel[segments, TILE // 2](d_key, d_tie, d_order, d_lo, d_len, padded, 1, True)
        return launches + 1
    tiles = padded // TILE
    _tile_kernel[segments * tiles, TILE // 2](d_key, d_tie, d_order, d_lo, d_len, TILE, tiles, True)
    launches += 1
    grid = (segments * (padded // 2) + STEP_THREADS - 1) // STEP_THREADS
    size = 2 * TILE
    while size <= padded:
        _global_step[grid, STEP_THREADS](d_key, d_tie, d_order, d_lo, d_len, segments, padded, size, size // 2, True)
        launches += 1
        stride = size // 4
        while stride >= TILE:
            _global_step[grid, STEP_THREADS](d_key, d_tie, d_order, d_lo, d_len, segments, padded, size, stride, False)
            launches += 1
            stride //= 2
        _tile_kernel[segments * tiles, TILE // 2](d_key, d_tie, d_order, d_lo, d_len, TILE, tiles, False)
        launches += 1
        size *= 2
    return launches


def device_build_tree(mid, leaf_size, *, device_mid=None):
    """``vortex_tree.build_tree(mid, leaf_size)`` with the splitting and sorting on the GPU. Returns (device order,
    order, lo, hi, left, right, node_count); the host arrays equal ``build_tree``'s element for element."""
    mid = np.ascontiguousarray(mid, dtype=np.float64)
    n = mid.shape[0]
    leaf_size = int(leaf_size)
    Statistics.totals["calls"] += 1
    d_mid = cuda.to_device(mid) if device_mid is None else device_mid
    d_order = cuda.to_device(np.arange(n, dtype=np.int64))
    d_key = cuda.device_array(max(n, 1), dtype=np.float64)
    d_tie = cuda.device_array(max(n, 1), dtype=np.int64)
    capacity = 4 * (n // max(leaf_size, 1) + 1) + 1
    lo = np.empty(capacity, np.int64)
    hi = np.empty(capacity, np.int64)
    left = np.full(capacity, -1, np.int64)
    right = np.full(capacity, -1, np.int64)
    lo[0] = 0
    hi[0] = n
    count = 1
    level = np.zeros(1, np.int64)
    while level.shape[0] > 0:
        lo_level = np.ascontiguousarray(lo[level])
        hi_level = np.ascontiguousarray(hi[level])
        d_split = cuda.device_array(level.shape[0], dtype=np.int8)
        d_axis = cuda.device_array(level.shape[0], dtype=np.int64)
        if n:
            _level_box[level.shape[0], BOX_THREADS](d_mid, d_order, cuda.to_device(lo_level), cuda.to_device(hi_level),
                                                    leaf_size, d_split, d_axis)
            split = d_split.copy_to_host().astype(np.bool_)
            axis_of = d_axis.copy_to_host()
            Statistics.totals["launches"] += 1
        else:
            split = np.zeros(level.shape[0], np.bool_)
            axis_of = np.zeros(level.shape[0], np.int64)
        chosen = np.flatnonzero(split)
        if len(chosen):
            Statistics.totals["launches"] += _sort_segments(d_mid, d_order, d_key, d_tie, lo_level[chosen],
                                                            hi_level[chosen] - lo_level[chosen], axis_of[chosen])
        Statistics.totals["levels"] += 1
        level, count = VT._next_level(level, split, lo, hi, left, right, count)
    out_lo, out_hi, out_left, out_right = VT._renumber(lo, hi, left, right, count)
    return d_order, d_order.copy_to_host(), out_lo, out_hi, out_left, out_right, count
