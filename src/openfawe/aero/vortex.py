# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenFAST v5.0.0 FVW_BiotSavart.f90, ui_seg_11/ui_seg/ui_quad_n1.
# Modifications (2026): NumPy API, local scaling, stable denominator, validation,
# explicit singularity/exclusion policy, owned output and bounded-memory summation.
# See third_party/OPENFAST-NOTICE.md and third_party/LICENSE-OpenFAST.txt.
"""Directed finite vortex filaments in Cartesian metres and seconds.

Positive circulation (m^2/s) follows A -> B and the right-hand rule. Velocities
are m/s in the input frame. This is Biot-Savart verification, not a flow solver.
OLAF's Lamb-Oseen exponent cutoff (< -10) is intentionally retained.
Rosenhead-Moore independently integrates dl x r / (r^2 + rc^2)^(3/2).
"""

from dataclasses import dataclass
from enum import IntEnum
import operator

import numpy as np


class CoreModel(IntEnum):
    NONE = 0
    RANKINE = 1
    LAMB_OSEEN = 2
    VATISTAS = 3
    OFFSET = 4
    ROSENHEAD_MOORE = 5


@dataclass(frozen=True)
class TreeSettings:
    """Opt-in hierarchical (Barnes-Hut type) far-field approximation settings.

    theta: opening ratio; a node is expanded only if its bounding-sphere radius
    is below theta times the distance from the target to its centre (0 < theta < 1).
    order: multipole order of the segment-moment expansion (0, 1 or 2).
    leaf_size: maximum number of segments in a leaf node.
    core_clearance: a node is approximated only if every target lies at least
    this many maximum core radii outside the node's bounding sphere.
    The direct NumPy/Numba sums remain the reference; see vortex_tree.
    """
    theta: float = 0.5
    order: int = 2
    leaf_size: int = 32
    core_clearance: float = 3.0

    def __post_init__(self):
        if isinstance(self.theta, (bool, np.bool_)) or not (0 < float(self.theta) < 1):
            raise ValueError("tree theta must lie strictly between 0 and 1")
        if isinstance(self.order, (bool, np.bool_)) or self.order not in (0, 1, 2):
            raise ValueError("tree order must be 0, 1 or 2")
        if isinstance(self.leaf_size, (bool, np.bool_)) or operator.index(self.leaf_size) < 1:
            raise ValueError("tree leaf size must be a positive integer")
        if not np.isfinite(self.core_clearance) or self.core_clearance < 0:
            raise ValueError("tree core clearance must be finite and nonnegative")


@dataclass(frozen=True)
class CudaSettings:
    """Opt-in direct summation on a CUDA GPU (GPU direct-sum contract, docs/tasks/OF-0006-accel20261001).

    precision: 'fp64' (double throughout), 'mixed' (float32 pair arithmetic about per-block local origins, float64
    accumulation) or 'fp32' (float32 accumulation as well). block: threads per block and segments per shared-memory
    tile (32, 64, 128 or 256). See vortex_cuda.

    suspect (whole-step GPU contract, revision 3; only the 'cuda_tree' backend in single precision): 'cpu' (default)
    recomputes every target with a near-axis ("suspect") pair whole on the CPU tree code in float64, as stage S1;
    'gpu64' evaluates just the suspect pairs again on the GPU in float64 and recomputes no target on the CPU. 'cpu' is
    left out of the wake metadata, so the metadata of every earlier configuration is unchanged.

    control (whole-step GPU contract revision 4 addendum 1, S4 (a); only with a wake's 'cuda_tree' velocity backend, which
    ``WakeConfig`` checks when it is built): 'cpu' (default, left out of the metadata) keeps the VSM free wake's induction
    at the control and load points on the CPU's exact direct sum; 'gpu' evaluates it with the GPU direct sum (backend
    'cuda', this precision and block). Only the VSM free wake (``vsm_wake.history_vsm_induction``) reads it; the
    lifting-line and lattice wakes are unaffected.
    A single sum (``segments_velocity``) does not read it: the wake's choice is made in ``vsm_wake.control_options``.
    """
    precision: str = "mixed"
    block: int = 128
    suspect: str = "cpu"
    control: str = "cpu"

    def __post_init__(self):
        if self.precision not in ("fp64", "mixed", "fp32"):
            raise ValueError("cuda precision must be 'fp64', 'mixed' or 'fp32'")
        if isinstance(self.block, (bool, np.bool_)) or self.block not in (32, 64, 128, 256):
            raise ValueError("cuda block must be 32, 64, 128 or 256")
        if self.suspect not in ("cpu", "gpu64"):
            raise ValueError("cuda suspect must be 'cpu' or 'gpu64'")
        if self.suspect == "gpu64" and self.precision == "fp64":
            raise ValueError("cuda suspect='gpu64' is for the single precisions 'mixed' and 'fp32' only")
        if self.control not in ("cpu", "gpu"):
            raise ValueError("cuda control must be 'cpu' or 'gpu'")


def cuda_backend_available():
    """True when the 'cuda' backend can run here (imports numba.cuda; only called when 'cuda' is selected)."""
    try:
        from .vortex_cuda import available
    except Exception:  # noqa: BLE001 - numba.cuda missing or broken means "not available"
        return False
    return available()


def cuda_unavailable_reason():
    """Why the 'cuda' backend is not available here (the probe's exception, or the import error)."""
    try:
        from .vortex_cuda import probe_error
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {str(error)[:400]}"
    return probe_error()


class FilamentSingularity(ValueError):
    """Observation is on a segment/end point and needs an explicit policy."""


def _real(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(value, width, name):
    result = _real(value, name)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width})")
    return result


def _indices(value, width, name):
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != width or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of shape (n, {width})")
    return result


def _per_segment(value, count, name):
    result = _real(value, name)
    if result.ndim == 0:
        return np.full(count, result.item(), dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    return result


def _model(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("model must be an integer CoreModel ID")
    try:
        return CoreModel(operator.index(value))
    except (TypeError, ValueError) as error:
        raise ValueError("model must be an integer CoreModel ID from 0 through 5") from error


def _norm(value):
    return np.hypot(np.hypot(value[..., 0], value[..., 1]), value[..., 2])


def _rosenhead_geometry(na, nb, h, t, rc):
    """Augmented distances for the exact regularized line integral, scaled by L."""
    na, nb = np.hypot(na, rc), np.hypot(nb, rc)
    product = na * nb
    transverse2 = h * h + rc * rc
    dot = transverse2 + t * (t - 1)
    plus = product + dot
    # The augmented Gram determinant is h^2 + rc^2. Rationalization avoids
    # cancellation near the interior; exterior points require no endpoint
    # cosine subtraction even when both endpoint directions nearly coincide.
    opposite = dot < 0
    plus[opposite] = transverse2[opposite] / (product[opposite] - dot[opposite])
    return na, nb, product * plus


def _evaluate(points, start, direction, length, gamma, model, radius, singularity):
    # Normalizing by segment length preserves units without OLAF's unit-scale
    # dimensional epsilon. Very large coordinate offsets still lose precision.
    a = (points - start) / length
    # Explicit component order avoids BLAS choosing different reductions for
    # one-row and multi-row chunks.
    t = (a[:, 0] * direction[0] + a[:, 1] * direction[1]) + a[:, 2] * direction[2]
    cross = np.cross(direction, a)
    h = _norm(cross)
    na, nb = _norm(a), _norm(a - direction)
    # Only roundoff-close to the axis is classified as collinear. This is a
    # dimensionless numerical guard, not an imposed physical core radius.
    axis = h <= 8 * np.finfo(np.float64).eps * np.maximum(1.0, np.maximum(na, nb))
    on_segment = axis & (t >= 0) & (t <= 1)
    if singularity == "raise" and np.any(on_segment):
        raise FilamentSingularity("Point on filament/end point; declare exclusions or singularity='zero'")
    out = np.zeros_like(points)
    active = ~axis
    if not np.any(active) or gamma == 0:
        return out
    ha, taa = h[active], t[active]
    naa, nbb = na[active], nb[active]
    product = naa * nbb
    dot = ha * ha + taa * (taa - 1)
    plus = product + dot
    # (|a||b| + a.b)(|a||b| - a.b) = |a x b|^2; use
    # this identity near an interior filament to avoid subtractive cancellation.
    opposite = dot < 0
    plus[opposite] = ha[opposite] ** 2 / (product[opposite] - dot[opposite])
    denominator = product * plus
    rc = radius / length
    factor = np.ones_like(ha)
    if radius > 0 and model != CoreModel.NONE:
        if model == CoreModel.ROSENHEAD_MOORE:
            naa, nbb, denominator = _rosenhead_geometry(naa, nbb, ha, taa, rc)
        elif model == CoreModel.OFFSET:
            denominator += rc * rc
        else:
            # Ratios outside a model's transition can be classified before
            # squaring, avoiding overflow for a very small positive core.
            ratio = ha / rc
            if model == CoreModel.RANKINE:
                factor = np.minimum(ratio, 1.0) ** 2
            elif model == CoreModel.LAMB_OSEEN:
                inside = ratio <= np.sqrt(10 / 1.25643)
                factor[inside] = -np.expm1(-1.25643 * ratio[inside] ** 2)
            elif model == CoreModel.VATISTAS:
                small = ratio < 1
                q = ratio[small] ** 2
                factor[small] = q / np.hypot(1, q)
                factor[~small] = 1 / np.hypot(1, (1 / ratio[~small]) ** 2)
    coefficient = (gamma / length) / (4 * np.pi) * factor * (naa + nbb) / denominator
    out[active] = coefficient[:, None] * cross[active]
    return out


def _evaluate_pairs(points, starts, directions, lengths, gamma, model, radius, singularity):
    """Evaluate independent, already non-excluded point/segment pairs."""
    a = (points-starts)/lengths[:, None]
    t = (a[:, 0]*directions[:, 0]+a[:, 1]*directions[:, 1])+a[:, 2]*directions[:, 2]
    cross = np.cross(directions, a)
    h = _norm(cross)
    na, nb = _norm(a), _norm(a-directions)
    axis = h <= 8*np.finfo(np.float64).eps*np.maximum(1.0, np.maximum(na, nb))
    if singularity == 'raise' and np.any(axis & (t >= 0) & (t <= 1)):
        raise FilamentSingularity("Point on filament/end point; declare exclusions or singularity='zero'")
    out = np.zeros_like(points)
    active = (~axis) & (gamma != 0)
    if not np.any(active):
        return out
    ha, taa = h[active], t[active]
    naa, nbb = na[active], nb[active]
    product = naa*nbb
    dot = ha*ha+taa*(taa-1)
    plus = product+dot
    opposite = dot < 0
    plus[opposite] = ha[opposite]**2/(product[opposite]-dot[opposite])
    denominator = product*plus
    length = lengths[active]
    radii = radius[active]
    rc = radii/length
    factor = np.ones_like(ha)
    cored = (radii > 0) & (model != CoreModel.NONE)
    if np.any(cored):
        if model == CoreModel.ROSENHEAD_MOORE:
            naa[cored], nbb[cored], denominator[cored] = _rosenhead_geometry(
                naa[cored], nbb[cored], ha[cored], taa[cored], rc[cored])
        elif model == CoreModel.OFFSET:
            denominator[cored] += rc[cored]*rc[cored]
        else:
            ratio = ha[cored]/rc[cored]
            correction = np.ones_like(ratio)
            if model == CoreModel.RANKINE:
                correction = np.minimum(ratio, 1.0)**2
            elif model == CoreModel.LAMB_OSEEN:
                inside = ratio <= np.sqrt(10/1.25643)
                correction[inside] = -np.expm1(-1.25643*ratio[inside]**2)
            elif model == CoreModel.VATISTAS:
                small = ratio < 1
                q = ratio[small]**2
                correction[small] = q/np.hypot(1, q)
                correction[~small] = 1/np.hypot(1, (1/ratio[~small])**2)
            factor[cored] = correction
    coefficient = (gamma[active]/length)/(4*np.pi)*factor*(naa+nbb)/denominator
    out[active] = coefficient[:, None]*cross[active]
    return out


def _sum_reference(points, starts, directions, lengths, edges, gamma, model, radius,
                   excluded, singularity, chunk_size):
    """Frozen original ordered loop for regression and exception precedence."""
    out = np.zeros_like(points)
    for segment in range(len(edges)):
        skipped = excluded.get(segment, np.empty(0, dtype=int))
        for first in range(0, len(points), chunk_size):
            last = min(first + chunk_size, len(points))
            selected = np.arange(first, last)
            selected = selected[~np.isin(selected, skipped)]
            out[selected] += _evaluate(
                points[selected], starts[segment], directions[segment], lengths[segment],
                gamma[segment], model, radius[segment], singularity)
    return out


def _sum_batched(points, starts, directions, lengths, edges, gamma, model, radius,
                 excluded, singularity, chunk_size):
    """Bounded 32-segment arithmetic followed by unchanged ordered additions."""
    out = np.zeros_like(points)
    try:
        for segment_first in range(0, len(edges), 32):
            segments = np.arange(segment_first, min(segment_first+32, len(edges)))
            for point_first in range(0, len(points), chunk_size):
                point_ids = np.arange(point_first, min(point_first+chunk_size, len(points)))
                shape = (len(segments), len(point_ids))
                keep = np.ones(shape, dtype=bool)
                for j, segment in enumerate(segments):
                    if segment in excluded:
                        keep[j] = ~np.isin(point_ids, excluded[segment])
                # Excluded pairs must never reach even the geometry arithmetic.
                pair_points = np.broadcast_to(point_ids[None, :], shape)[keep]
                pair_segments = np.broadcast_to(segments[:, None], shape)[keep]
                pair_values = _evaluate_pairs(points[pair_points], starts[pair_segments],
                    directions[pair_segments], lengths[pair_segments], gamma[pair_segments],
                    model, radius[pair_segments], singularity)
                if np.all(keep):
                    # No masked scatter or fancy-index output copies are needed.
                    # Each point still receives one addition per original segment.
                    values = pair_values.reshape(*shape, 3)
                    target = out[point_first:point_first+len(point_ids)]
                    for j in range(len(segments)):
                        target += values[j]
                    continue
                values = np.zeros((*shape, 3))
                values[keep] = pair_values
                for j in range(len(segments)):
                    selected = point_ids[keep[j]]
                    out[selected] += values[j, keep[j]]
        return out
    except (FilamentSingularity, FloatingPointError) as error:
        # A block can encounter a later segment's error first. Re-execute the
        # frozen sequence to preserve the historical exception type and message.
        _sum_reference(points, starts, directions, lengths, edges, gamma, model, radius,
                       excluded, singularity, chunk_size)
        raise RuntimeError('Batched arithmetic failed while the reference succeeded') from error


def segments_velocity(points, nodes, connectivity, circulation, *, model=CoreModel.NONE,
                      core_radius=0.0, exclusions=None, singularity="raise", chunk_size=4096,
                      backend="numpy", tree=None, cuda=None):
    """Sum directed segments into a new float64 (n,3) velocity array.

    points/nodes: (n,3)/(m,3) metres; connectivity: integer (k,2), zero-based.
    Circulation and core radius may be scalars or per-segment (k,) arrays.
    ``exclusions`` is an optional integer (e,2) array of (point, segment) pairs;
    omitted pairs are never evaluated. Retain it with scientific case inputs.
    ``singularity='zero'`` explicitly selects the upstream on-axis convention,
    also for endpoints and positive cores. Default 'raise' rejects on-segment
    points for every core, including zero circulation. Collinear points outside
    the finite segment return zero. Radius zero uses the ideal-filament limit.

    Segment input order fixes summation order, independently of point chunks.
    Working memory scales with points + nodes + segments + exclusions + chunk
    times at most 32 segments, never their full product. No input is modified.
    Degenerate segments are invalid even if excluded. Geometry that exceeds
    float64's supported arithmetic range raises ValueError; it is not clipped.

    ``backend`` selects the summation engine for the same kernel:
    ``'numpy'`` (default, reference), ``'numba'`` (compiled direct sum with
    identical per-pair arithmetic and per-point order; see vortex_numba) or
    ``'tree'`` (opt-in hierarchical approximation with ``tree`` settings; see
    vortex_tree). A compiled pair that the reference would reject re-runs the
    reference, so exceptions keep their historical type and message.
    ``'cuda'`` (opt-in direct sum on a CUDA GPU with ``cuda`` settings; see vortex_cuda) is accepted only with
    CudaSettings and only where a CUDA device is available. ``'cuda_tree'`` (opt-in: the tree built on the CPU,
    evaluated on a CUDA GPU; see vortex_cuda_tree; whole-step GPU contract S1) needs both ``tree`` and ``cuda``.
    """
    if backend not in ("numpy", "numba", "tree", "cuda", "cuda_tree"):
        raise ValueError("backend must be 'numpy', 'numba', 'tree', 'cuda' or 'cuda_tree'")
    if backend == "cuda_tree":
        if tree is None or cuda is None:
            raise ValueError("backend='cuda_tree' needs both tree and cuda settings")
        if not isinstance(tree, TreeSettings):
            raise ValueError("tree must be TreeSettings")
    else:
        if (tree is not None) != (backend == "tree"):
            raise ValueError("tree settings are required exactly for backend='tree'")
        if (cuda is not None) != (backend == "cuda"):
            raise ValueError("cuda settings are required exactly for backend='cuda'")
    if cuda is not None:
        if not isinstance(cuda, CudaSettings):
            raise ValueError("cuda must be CudaSettings")
        if cuda.suspect != "cpu" and backend != "cuda_tree":
            raise ValueError("cuda suspect='gpu64' is for the cuda_tree backend only")
        if not cuda_backend_available():
            raise ValueError(f"the cuda velocity backend needs a CUDA device ({cuda_unavailable_reason()})")
    points, nodes = _matrix(points, 3, "points"), _matrix(nodes, 3, "nodes")
    edges = _indices(connectivity, 2, "connectivity")
    if np.any(edges < 0) or np.any(edges >= len(nodes)):
        raise ValueError("connectivity index out of range")
    gamma = _per_segment(circulation, len(edges), "circulation")
    radius = _per_segment(core_radius, len(edges), "core_radius")
    if np.any(radius < 0):
        raise ValueError("core_radius must be nonnegative")
    model = _model(model)
    if singularity not in ("raise", "zero"):
        raise ValueError("singularity must be 'raise' or 'zero'")
    if isinstance(chunk_size, (bool, np.bool_)):
        raise ValueError("chunk_size must be a positive integer")
    try:
        chunk_size = operator.index(chunk_size)
    except TypeError as error:
        raise ValueError("chunk_size must be a positive integer") from error
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    excluded = {}
    pairs = np.zeros((0, 2), dtype=np.int64)
    if exclusions is not None:
        pairs = _indices(exclusions, 2, "exclusions")
        if np.any(pairs < 0) or np.any(pairs[:, 0] >= len(points)) or np.any(pairs[:, 1] >= len(edges)):
            raise ValueError("exclusion index out of range")
        if backend == "numpy":
            for point, segment in pairs:
                excluded.setdefault(int(segment), []).append(int(point))
            excluded = {key: np.unique(value) for key, value in excluded.items()}

    def by_segment():
        # The reference (segment -> unique points) view, built only when the
        # compiled kernels must re-run the NumPy path to raise its exception.
        table = {}
        for point, segment in pairs:
            table.setdefault(int(segment), []).append(int(point))
        return {key: np.unique(value) for key, value in table.items()}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            starts = nodes[edges[:, 0]]
            delta = nodes[edges[:, 1]] - starts
            lengths = _norm(delta)
            if np.any(lengths == 0):
                raise ValueError("zero-length segment")
            directions = delta / lengths[:, None]
            if backend == "numpy":
                out = _sum_batched(points, starts, directions, lengths, edges, gamma, model,
                                   radius, excluded, singularity, chunk_size)
            else:
                from .vortex_numba import exclusion_csr
                csr = exclusion_csr(len(points), pairs)
                if backend == "numba":
                    from .vortex_numba import direct_sum
                    out, flagged = direct_sum(points, starts, directions, lengths, gamma,
                                              radius, model, csr, singularity)
                elif backend == "cuda":
                    from .vortex_cuda import cuda_sum
                    out, flagged = cuda_sum(points, starts, directions, lengths, gamma,
                                            radius, model, csr, singularity, cuda)
                elif backend == "cuda_tree":
                    from .vortex_cuda_tree import cuda_tree_sum
                    out, flagged = cuda_tree_sum(points, starts, directions, lengths, gamma,
                                                 radius, model, csr, singularity, tree, cuda)
                else:
                    from .vortex_tree import tree_sum
                    out, flagged = tree_sum(points, starts, directions, lengths, gamma,
                                            radius, model, csr, singularity, tree)
                if flagged:
                    # Reproduce the reference failure (type, message, precedence).
                    _sum_batched(points, starts, directions, lengths, edges, gamma, model,
                                 radius, by_segment(), singularity, chunk_size)
                    raise RuntimeError("compiled kernel flagged a pair that the NumPy reference accepted")
            if not np.isfinite(out).all():
                raise ValueError("velocity exceeds finite float64 range")
            return out
    except FloatingPointError as error:
        raise ValueError("geometry/core/velocity exceeds supported float64 arithmetic range") from error


def segment_velocity(start, end, point, circulation, *, model=CoreModel.NONE,
                     core_radius=0.0, singularity="raise"):
    """Single-segment counterpart of segments_velocity; returns an owned (3,) array."""
    vectors = [_real(value, name) for value, name in ((start, "start"), (end, "end"), (point, "point"))]
    if any(value.shape != (3,) for value in vectors):
        raise ValueError("start, end and point must each have shape (3,)")
    return segments_velocity(vectors[2][None, :], np.stack(vectors[:2]), [[0, 1]], circulation,
                             model=model, core_radius=core_radius, singularity=singularity)[0].copy()


def ring_velocity(corners, points, circulation, *, model=CoreModel.NONE,
                  core_radius=0.0, singularity="raise", exclusions=None, chunk_size=4096):
    """Closed A->B->C->D->A ring (possibly nonplanar); scalar Gamma/core required.

    Exclusion segment indices 0..3 refer to those four directed edges.
    This computes their superposition, not a surface load or circulation solve.
    """
    corners = _real(corners, "corners")
    if corners.shape != (4, 3):
        raise ValueError("corners must have shape (4, 3)")
    if _real(circulation, "circulation").ndim or _real(core_radius, "core_radius").ndim:
        raise ValueError("ring circulation and core_radius must be scalars")
    return segments_velocity(points, corners, [[0, 1], [1, 2], [2, 3], [3, 0]], circulation,
                             model=model, core_radius=core_radius, singularity=singularity,
                             exclusions=exclusions, chunk_size=chunk_size)
