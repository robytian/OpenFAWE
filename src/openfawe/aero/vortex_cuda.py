# SPDX-License-Identifier: Apache-2.0
# Direct Biot-Savart summation on a CUDA GPU for directed segments (opt-in backend 'cuda').
# The per-pair arithmetic is that of vortex_numba.pair_velocity (itself a translation of OpenFAST v5.0.0
# FVW_BiotSavart.f90 ui_seg_11/ui_seg/ui_quad_n1); see third_party/OPENFAST-NOTICE.md and LICENSE-OpenFAST.txt.
"""Direct summation on the GPU (GPU direct-sum contract, docs/tasks/OF-0006-accel20261001).

One thread per target point; the segments are walked in their input order, staged block by block through shared
memory, skipping the excluded pairs exactly like ``vortex_numba._sum_point``. Launches cover consecutive ranges of
segments (the Windows display watchdog stops a launch after about 2 s); each point's running sum, exclusion pointer
and flag stay on the device between launches, so the summation order equals that of one pass.

Precisions (``CudaSettings.precision``):

- ``fp64``: double throughout, no coordinate shift. The device function is built from the very code object of
  ``vortex_numba.pair_velocity`` (only its ``_norm3`` is rebound to a device version).
- ``mixed``: one local origin per thread block (the float64 mean of that block's targets); targets and, when a
  segment tile is loaded, segment starts are shifted in float64 and rounded to float32; the pair arithmetic is float32
  (``_pair32``, branch by branch the same as ``pair_velocity``, float32 constants, axis threshold 8 eps32); each tile's
  partial sum is float32 and is added to a float64 accumulator.
- ``fp32``: as ``mixed`` with a float32 accumulator.

Flags as in vortex_numba (the caller re-runs the NumPy reference to raise its exception). The single-precision kernels
also mark a target "suspect" when any of its pairs lies within 1e-4 segment lengths of the segment's axis (scaled like
the axis guard); suspect targets are recomputed whole on the CPU by ``vortex_numba.direct_sum`` in float64.

``numba.cuda`` is imported only here, and this module only when the 'cuda' backend is selected.
"""
import math
import time
import types

import numpy as np
from numba import cuda

from . import vortex_numba as VN

FLAG_OK, FLAG_SINGULAR, FLAG_NONFINITE = VN.FLAG_OK, VN.FLAG_SINGULAR, VN.FLAG_NONFINITE
BLOCKS = (32, 64, 128, 256)
PAIRS_PER_LAUNCH = {"fp64": 2 ** 27, "mixed": 2 ** 31, "fp32": 2 ** 31}       # contract D2: at most 2^28 / 2^31
SEGMENTS_PER_LAUNCH = {"fp64": 2 ** 14, "mixed": 2 ** 20, "fp32": 2 ** 20}    # a few targets: one block walks them all
SUSPECT_AXIS = 1e-4

# float32 constants of _pair32 (contract D2): every literal of pair_velocity in its float32 form.
F0 = np.float32(0.0)
F1 = np.float32(1.0)
F_AXIS = np.float32(8.0 * np.finfo(np.float32).eps)
F_SUSPECT = np.float32(SUSPECT_AXIS)
F_LAMB = np.float32(1.25643)
F_LAMB_CUT = np.float32(VN._LAMB_OSEEN_CUT)
F_FOUR_PI = np.float32(4.0 * np.pi)


_AVAILABLE = []
_PROBE_ERROR = []
SHIFT_RATIO_LIMIT = 0.5e-4 / float(np.finfo(np.float32).eps)   # max |p - o| / min L that the suspect width covers


def available():
    """True when a CUDA device (or the simulator, NUMBA_ENABLE_CUDASIM=1) can be used: the device is there and a trivial
    kernel compiles and runs (so a missing NVVM or libdevice also counts as "not available"; contract D4). Cached."""
    if not _AVAILABLE:
        try:
            ok = bool(cuda.is_available())
            if ok:
                @cuda.jit
                def probe(out):
                    out[0] = 1.0
                value = cuda.to_device(np.zeros(1))
                probe[1, 1](value)
                ok = bool(value.copy_to_host()[0] == 1.0)
        except Exception as error:  # noqa: BLE001 - any driver, compiler or import problem means "not available"
            ok = False
            _PROBE_ERROR.append(f"{type(error).__name__}: {str(error)[:400]}")
        if not ok and not _PROBE_ERROR:
            _PROBE_ERROR.append("cuda.is_available() is False or the probe kernel gave a wrong value")
        _AVAILABLE.append(ok)
    return _AVAILABLE[0]


def probe_error():
    """Why ``available()`` said no (None when it said yes or was not asked)."""
    return _PROBE_ERROR[0] if _PROBE_ERROR else None


@cuda.jit(device=True)
def _norm3_device(x, y, z):
    return math.hypot(math.hypot(x, y), z)


def _rebuilt(function, **replacements):
    """``function`` (a plain Python function) with some of its globals replaced; same code object."""
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    return types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__,
                              function.__closure__)


PAIR64_SOURCE = _rebuilt(VN.pair_velocity.py_func, _norm3=_norm3_device)
_pair64 = cuda.jit(device=True)(PAIR64_SOURCE)


@cuda.jit(device=True)
def _norm3_32(x, y, z):
    return math.hypot(math.hypot(x, y), z)


@cuda.jit(device=True)
def _pair32(ax, ay, az, dx, dy, dz, length, gamma, model, radius, raise_singular):
    """float32 counterpart of ``vortex_numba.pair_velocity`` given the float32 relative position ``a = p - s`` (not yet
    divided by the length). Returns (ux, uy, uz, flag, suspect). Branches and their order as in pair_velocity."""
    ax = ax / length
    ay = ay / length
    az = az / length
    t = (ax * dx + ay * dy) + az * dz
    cx = dy * az - dz * ay
    cy = dz * ax - dx * az
    cz = dx * ay - dy * ax
    h = _norm3_32(cx, cy, cz)
    na = _norm3_32(ax, ay, az)
    nb = _norm3_32(ax - dx, ay - dy, az - dz)
    if not (math.isfinite(h) and math.isfinite(na) and math.isfinite(nb) and math.isfinite(t)):
        return F0, F0, F0, FLAG_NONFINITE, 0
    scale = max(F1, max(na, nb))
    suspect = 1 if h <= F_SUSPECT * scale else 0
    if h <= F_AXIS * scale:
        if raise_singular and t >= F0 and t <= F1:
            return F0, F0, F0, FLAG_SINGULAR, suspect
        return F0, F0, F0, FLAG_OK, suspect
    if gamma == F0:
        return F0, F0, F0, FLAG_OK, suspect
    product = na * nb
    dot = h * h + t * (t - F1)
    plus = product + dot
    if dot < F0:
        plus = (h * h) / (product - dot)
    denominator = product * plus
    rc = radius / length
    factor = F1
    if radius > F0 and model != 0:
        if model == 5:
            na = math.hypot(na, rc)
            nb = math.hypot(nb, rc)
            product = na * nb
            transverse2 = h * h + rc * rc
            dot = transverse2 + t * (t - F1)
            plus = product + dot
            if dot < F0:
                plus = transverse2 / (product - dot)
            denominator = product * plus
        elif model == 4:
            denominator = denominator + rc * rc
        else:
            ratio = h / rc
            if not math.isfinite(ratio):
                return F0, F0, F0, FLAG_NONFINITE, suspect
            if model == 1:
                limited = min(ratio, F1)
                factor = limited * limited
            elif model == 2:
                if ratio <= F_LAMB_CUT:
                    factor = -math.expm1(-F_LAMB * (ratio * ratio))
            elif model == 3:
                if ratio < F1:
                    q = ratio * ratio
                    factor = q / math.hypot(F1, q)
                else:
                    inverse = F1 / ratio
                    factor = F1 / math.hypot(F1, inverse * inverse)
    coefficient = (gamma / length) / F_FOUR_PI * factor * (na + nb) / denominator
    if not (math.isfinite(denominator) and math.isfinite(coefficient) and math.isfinite(product)
            and math.isfinite(dot)):
        return F0, F0, F0, FLAG_NONFINITE, suspect
    return coefficient * cx, coefficient * cy, coefficient * cz, FLAG_OK, suspect


_KERNELS = {}


def _kernel(precision, block):
    """The kernel for one precision and block size (shared-memory shapes are compile-time constants)."""
    key = (precision, block)
    if key in _KERNELS:
        return _KERNELS[key]
    if precision == "fp64":
        @cuda.jit
        def kernel(points, starts, directions, lengths, gamma, radius, model, raise_singular, ex_ptr, ex_seg,
                   seg_lo, seg_hi, acc, kpos, flags):
            sx = cuda.shared.array(block, np.float64)
            sy = cuda.shared.array(block, np.float64)
            sz = cuda.shared.array(block, np.float64)
            qx = cuda.shared.array(block, np.float64)
            qy = cuda.shared.array(block, np.float64)
            qz = cuda.shared.array(block, np.float64)
            ql = cuda.shared.array(block, np.float64)
            qg = cuda.shared.array(block, np.float64)
            qr = cuda.shared.array(block, np.float64)
            t = cuda.threadIdx.x
            p = cuda.blockIdx.x * block + t
            n = points.shape[0]
            active = p < n
            ux = 0.0
            uy = 0.0
            uz = 0.0
            flag = 0
            k = 0
            stop = 0
            px = 0.0
            py = 0.0
            pz = 0.0
            if active:
                ux = acc[p, 0]
                uy = acc[p, 1]
                uz = acc[p, 2]
                flag = flags[p]
                k = kpos[p]
                stop = ex_ptr[p + 1]
                px = points[p, 0]
                py = points[p, 1]
                pz = points[p, 2]
            tile = seg_lo
            while tile < seg_hi:
                s = tile + t
                if s < seg_hi:
                    sx[t] = starts[s, 0]
                    sy[t] = starts[s, 1]
                    sz[t] = starts[s, 2]
                    qx[t] = directions[s, 0]
                    qy[t] = directions[s, 1]
                    qz[t] = directions[s, 2]
                    ql[t] = lengths[s]
                    qg[t] = gamma[s]
                    qr[t] = radius[s]
                cuda.syncthreads()
                count = min(block, seg_hi - tile)
                if active:
                    for j in range(count):
                        s = tile + j
                        if k < stop and ex_seg[k] == s:
                            k += 1
                            continue
                        vx, vy, vz, f = _pair64(px, py, pz, sx[j], sy[j], sz[j], qx[j], qy[j], qz[j], ql[j], qg[j],
                                                model, qr[j], raise_singular)
                        if f != 0:
                            if flag == 0:
                                flag = f
                            continue
                        ux += vx
                        uy += vy
                        uz += vz
                cuda.syncthreads()
                tile += block
            if active:
                acc[p, 0] = ux
                acc[p, 1] = uy
                acc[p, 2] = uz
                flags[p] = flag
                kpos[p] = k
    else:
        wide = precision == "mixed"
        accumulator = np.float64 if wide else np.float32

        @cuda.jit
        def kernel(points, origins, starts, directions, lengths, gamma, radius, model, raise_singular, ex_ptr,
                   ex_seg, seg_lo, seg_hi, acc, kpos, flags, suspects):
            sx = cuda.shared.array(block, np.float32)
            sy = cuda.shared.array(block, np.float32)
            sz = cuda.shared.array(block, np.float32)
            qx = cuda.shared.array(block, np.float32)
            qy = cuda.shared.array(block, np.float32)
            qz = cuda.shared.array(block, np.float32)
            ql = cuda.shared.array(block, np.float32)
            qg = cuda.shared.array(block, np.float32)
            qr = cuda.shared.array(block, np.float32)
            t = cuda.threadIdx.x
            b = cuda.blockIdx.x
            p = b * block + t
            n = points.shape[0]
            active = p < n
            ox = origins[b, 0]
            oy = origins[b, 1]
            oz = origins[b, 2]
            ux = accumulator(0.0)
            uy = accumulator(0.0)
            uz = accumulator(0.0)
            flag = 0
            suspect = 0
            k = 0
            stop = 0
            px = F0
            py = F0
            pz = F0
            if active:
                ux = acc[p, 0]
                uy = acc[p, 1]
                uz = acc[p, 2]
                flag = flags[p]
                suspect = suspects[p]
                k = kpos[p]
                stop = ex_ptr[p + 1]
                px = points[p, 0]
                py = points[p, 1]
                pz = points[p, 2]
            tile = seg_lo
            while tile < seg_hi:
                s = tile + t
                if s < seg_hi:
                    sx[t] = np.float32(starts[s, 0] - ox)      # the shift in float64, then rounded (contract D2)
                    sy[t] = np.float32(starts[s, 1] - oy)
                    sz[t] = np.float32(starts[s, 2] - oz)
                    qx[t] = directions[s, 0]
                    qy[t] = directions[s, 1]
                    qz[t] = directions[s, 2]
                    ql[t] = lengths[s]
                    qg[t] = gamma[s]
                    qr[t] = radius[s]
                cuda.syncthreads()
                count = min(block, seg_hi - tile)
                if active:
                    tx = F0
                    ty = F0
                    tz = F0
                    for j in range(count):
                        s = tile + j
                        if k < stop and ex_seg[k] == s:
                            k += 1
                            continue
                        vx, vy, vz, f, w = _pair32(px - sx[j], py - sy[j], pz - sz[j], qx[j], qy[j], qz[j], ql[j],
                                                   qg[j], model, qr[j], raise_singular)
                        if w != 0:
                            suspect = 1
                        if f != 0:
                            if flag == 0:
                                flag = f
                            continue
                        tx += vx
                        ty += vy
                        tz += vz
                    ux += accumulator(tx)
                    uy += accumulator(ty)
                    uz += accumulator(tz)
                cuda.syncthreads()
                tile += block
            if active:
                acc[p, 0] = ux
                acc[p, 1] = uy
                acc[p, 2] = uz
                flags[p] = flag
                suspects[p] = suspect
                kpos[p] = k
    _KERNELS[key] = kernel
    return kernel


def segments_per_launch(points, segments, precision, block, cap=None):
    """Segments per launch: a multiple of ``block`` with points * segments <= the precision's pair cap and at most the
    precision's segment cap (a launch over few targets runs on one multiprocessor)."""
    cap = PAIRS_PER_LAUNCH[precision] if cap is None else int(cap)
    per = min(max(1, cap // max(1, points)), SEGMENTS_PER_LAUNCH[precision]) // block * block
    return max(block, per)


def block_origins(points, block):
    """float64 mean of each block's targets (indices b*block .. min(n, (b+1)*block) - 1)."""
    n = len(points)
    blocks = (n + block - 1) // block
    origins = np.empty((blocks, 3))
    for b in range(blocks):
        origins[b] = points[b * block:min(n, (b + 1) * block)].mean(axis=0)
    return origins


class Statistics:
    last = {}
    totals = dict(calls=0, suspect=0, suspect_max_per_call=0, cpu_fallback_calls=0, launches=0)

    @classmethod
    def reset_totals(cls):
        cls.totals = dict(calls=0, suspect=0, suspect_max_per_call=0, cpu_fallback_calls=0, launches=0)


def cuda_sum(points, starts, directions, lengths, gamma, radius, model, csr, singularity, settings, *,
             pairs_per_launch=None):
    """Return (velocity, any_flag) for validated inputs from segments_velocity (like direct_sum and tree_sum)."""
    if not available():
        raise ValueError(f"the cuda velocity backend needs a CUDA device ({probe_error()})")
    precision, block = settings.precision, int(settings.block)
    points = np.ascontiguousarray(points, dtype=np.float64)
    n, m = len(points), len(starts)
    out = np.zeros((n, 3))
    if n == 0 or m == 0:
        Statistics.last = dict(points=n, segments=m, launches=0, suspect=0)
        return out, False
    pointer, segments = csr
    pointer = np.ascontiguousarray(pointer, dtype=np.int64)
    segments = np.ascontiguousarray(segments, dtype=np.int64)
    if len(segments) == 0:
        segments = np.zeros(1, dtype=np.int64)                 # a device array may not be empty; never read
    raise_singular = singularity == "raise"
    per = segments_per_launch(n, m, precision, block, pairs_per_launch)
    grid = (n + block - 1) // block
    kernel = _kernel(precision, block)
    d_ptr, d_seg = cuda.to_device(pointer), cuda.to_device(segments)
    d_kpos = cuda.to_device(pointer[:-1].copy())
    d_flags = cuda.to_device(np.zeros(n, dtype=np.int8))
    launches, longest = 0, 0.0
    shift = dict(max_target_offset_m=0.0, max_offset_over_min_length=0.0)
    if precision == "fp64":
        d_acc = cuda.to_device(np.zeros((n, 3)))
        arrays = [cuda.to_device(np.ascontiguousarray(a, dtype=np.float64)) for a in
                  (points, starts, directions, lengths, gamma, radius)]
        for lo in range(0, m, per):
            tick = time.perf_counter()
            kernel[grid, block](*arrays, int(model), raise_singular, d_ptr, d_seg, lo, min(m, lo + per), d_acc,
                                d_kpos, d_flags)
            cuda.synchronize()
            longest = max(longest, time.perf_counter() - tick)
            launches += 1
        suspects = np.zeros(n, dtype=np.int8)
    else:
        origins = block_origins(points, block)
        offsets = points - np.repeat(origins, block, axis=0)[:n]
        relative = offsets.astype(np.float32)
        largest = float(np.linalg.norm(offsets, axis=1).max())
        shift = dict(max_target_offset_m=largest, max_offset_over_min_length=largest / float(np.min(lengths)))
        if shift["max_offset_over_min_length"] > SHIFT_RATIO_LIMIT:
            # The suspect width (1e-4 segment lengths) covers the float32 rounding of the shift only up to this ratio
            # (review GPUC2-R07): the whole call is summed on the CPU in float64 instead.
            out, flagged = VN.direct_sum(points, starts, directions, lengths, gamma, radius, model, csr, singularity)
            Statistics.last = dict(points=n, segments=m, launches=0, cpu_fallback=True, suspect=0, **shift)
            Statistics.totals["calls"] += 1
            Statistics.totals["cpu_fallback_calls"] += 1
            return out, flagged
        d_acc = cuda.to_device(np.zeros((n, 3), dtype=np.float64 if precision == "mixed" else np.float32))
        d_suspect = cuda.to_device(np.zeros(n, dtype=np.int8))
        arrays = [cuda.to_device(np.ascontiguousarray(relative)), cuda.to_device(np.ascontiguousarray(origins)),
                  cuda.to_device(np.ascontiguousarray(starts, dtype=np.float64))]
        arrays += [cuda.to_device(np.ascontiguousarray(a, dtype=np.float32)) for a in (directions, lengths, gamma, radius)]
        for lo in range(0, m, per):
            tick = time.perf_counter()
            kernel[grid, block](*arrays, int(model), raise_singular, d_ptr, d_seg, lo, min(m, lo + per), d_acc,
                                d_kpos, d_flags, d_suspect)
            cuda.synchronize()
            longest = max(longest, time.perf_counter() - tick)
            launches += 1
        suspects = d_suspect.copy_to_host()
    out[:] = d_acc.copy_to_host()
    flags = d_flags.copy_to_host()
    suspect = np.flatnonzero(suspects)
    gpu_flagged_suspects = int(np.count_nonzero(flags[suspect])) if len(suspect) else 0
    if len(suspect):                       # contract D3: suspect targets recomputed whole on the CPU in float64
        sub_ptr = np.zeros(len(suspect) + 1, dtype=np.int64)
        np.cumsum(pointer[suspect + 1] - pointer[suspect], out=sub_ptr[1:])
        sub_seg = (np.concatenate([csr[1][pointer[i]:pointer[i + 1]] for i in suspect]) if len(csr[1])
                   else np.zeros(0, dtype=np.int64))
        again, flagged = VN.direct_sum(points[suspect], starts, directions, lengths, gamma, radius, model,
                                       (sub_ptr, np.ascontiguousarray(sub_seg, dtype=np.int64)), singularity)
        out[suspect] = again
        flags[suspect] = FLAG_SINGULAR if flagged else FLAG_OK   # direct_sum reports one flag for the call
    bad = ~np.isfinite(out).all(axis=1)
    flags[bad & (flags == 0)] = FLAG_NONFINITE
    Statistics.last = dict(points=n, segments=m, launches=launches, segments_per_launch=per, suspect=int(len(suspect)),
                           gpu_flagged_suspects=gpu_flagged_suspects, longest_launch_s=longest, cpu_fallback=False, **shift)
    Statistics.totals["calls"] += 1
    Statistics.totals["launches"] += launches
    Statistics.totals["suspect"] += int(len(suspect))
    Statistics.totals["suspect_max_per_call"] = max(Statistics.totals["suspect_max_per_call"], int(len(suspect)))
    return out, bool(np.any(flags))


_PAIR32_KERNEL = []


def pair32_values(relative, directions, lengths, gamma, radius, model, raise_singular=True):
    """Evaluate ``_pair32`` once per row (GU(b) of the contract): float32 inputs, returns (u (k,3) float32, flags,
    suspects). ``relative`` is p - s (float32)."""
    if not _PAIR32_KERNEL:
        @cuda.jit
        def kernel(rel, dirs, lens, gam, rad, model, raise_singular, out, flags, suspects):
            i = cuda.grid(1)
            if i < rel.shape[0]:
                vx, vy, vz, f, w = _pair32(rel[i, 0], rel[i, 1], rel[i, 2], dirs[i, 0], dirs[i, 1], dirs[i, 2], lens[i],
                                           gam[i], model, rad[i], raise_singular)
                out[i, 0] = vx
                out[i, 1] = vy
                out[i, 2] = vz
                flags[i] = f
                suspects[i] = w
        _PAIR32_KERNEL.append(kernel)
    k = len(relative)
    out = np.zeros((k, 3), dtype=np.float32)
    flags = np.zeros(k, dtype=np.int8)
    suspects = np.zeros(k, dtype=np.int8)
    if k:
        arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in (relative, directions, lengths, gamma, radius)]
        d_out, d_flags, d_suspects = cuda.to_device(out), cuda.to_device(flags), cuda.to_device(suspects)
        _PAIR32_KERNEL[0][(k + 63) // 64, 64](*[cuda.to_device(a) for a in arrays], int(model), bool(raise_singular),
                                               d_out, d_flags, d_suspects)
        out, flags, suspects = d_out.copy_to_host(), d_flags.copy_to_host(), d_suspects.copy_to_host()
    return out, flags, suspects
