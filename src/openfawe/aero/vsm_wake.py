# SPDX-License-Identifier: Apache-2.0
# Wake update order adapted from the OpenFAST routines named in wake.py.
# The MIT-derived VSM sectional closure is imported from vsm.py.
"""OpenFAWE three-quarter-chord VSM closure with a material free vortex wake.

This combines the existing VSM circulation equation with selectable geometry
convection, circulation-shedding and material-core machinery. It is an OpenFAWE
method combination, not a claim that upstream KiteFAST implements this model.

``surfaces`` (OF-0010 P2, opt-in; ``aero.surface_cores.SurfaceCores``): the induction of one class's vortex elements
(attached rings and free rows) at the control and load points of another class uses the Rosenhead-Moore kernel with
the radius max(AVL strip core, material core), and at the load points keeps that class's bound segments; everything
within a class, the convection and the rings are unchanged. With ``surfaces=None`` (the default) every result is the
same bit for bit.
"""

from dataclasses import dataclass, replace
import numpy as np

from .vortex import _real, segments_velocity
from .vsm import VsmSolution, geometry, vsm_self_correction, solve_vsm_operator
from .wake import (WakeConfig, WakeState, ambient_velocity, pack_rings, retain_newest, row_limit_reached,
                   convect_geometry, validate_state_config, kernel_options, compiled_network,
                   ring_induction, ring_induction_pair)
from .cores import initial_cores, check_cores, advance_cores, edge_radii


@dataclass(frozen=True)
class VsmWakeState(WakeState):
    """Separate VSM/free-wake ownership, retaining the common wake-network data."""

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.solution, VsmSolution):
            raise ValueError('VSM free-wake state requires a VsmSolution')
        if (self.solution.polar_alpha_rad.shape != (self.wing.panels,) or
                self.solution.collocation_points.shape != (self.wing.panels, 3)):
            raise ValueError('VSM free-wake solution and geometry disagree')


def vsm_wake_influence(wing, downstream_rows=(), *, config=WakeConfig(),
                       cores=None, at_load_points=False, surfaces=None):
    """Unit-current-Gamma operator for a finite attached wake network.

    Rows are bound, trailing edge, and zero or more specified downstream rows.
    All rings in this operator carry the current panel circulation. Actual
    transient calls include only the newest downstream row; older strengths
    are provided separately. Multiple rows support an explicit steady finite-
    wake comparison without pretending they are a computed transient history.

    At 3/4 chord, subtract the original infinite-bound diagonal correction.
    At 1/4 chord, omit every row-zero bound span edge and apply no correction.
    All retained finite filaments use the caller's declared material cores.
    """
    if not isinstance(config, WakeConfig):
        raise ValueError('config must be WakeConfig')
    check_cores(cores, config)
    rows = [wing.bound, wing.trailing_edge]
    for row in downstream_rows:
        row = _real(row, 'VSM downstream row')
        if row.shape != wing.bound.shape:
            raise ValueError('VSM downstream row shape mismatch')
        rows.append(row)
    if cores is not None and (cores.span.shape[0] < len(rows)-2 or
            cores.stream.shape[0] < len(rows)-2):
        raise ValueError('insufficient material core rows for VSM operator')
    points = wing.cp if at_load_points else geometry(wing)[0]
    if compiled_network(config):
        matrix = _batched_ring_operator(wing, rows, points, config, cores, at_load_points)
        if matrix is not None:
            if not at_load_points:
                index = np.arange(wing.panels)
                matrix[index, index] -= vsm_self_correction(wing, points)
            if surfaces is not None:
                _cross_class_operator(matrix, wing, rows, points, config, cores, at_load_points, surfaces)
            return matrix
    matrix = np.zeros((wing.panels, wing.panels, 3))
    for panel in range(wing.panels):
        strengths = np.zeros((len(rows)-1, wing.panels))
        strengths[:, panel] = 1.
        nodes, edges, gamma = pack_rings(rows, strengths, wing.panel_edges)
        if at_load_points:
            retained = ~np.all(edges < wing.node_count, axis=1)
            edges, gamma = edges[retained], gamma[retained]
        matrix[:, panel] = segments_velocity(points, nodes, edges, gamma,
            model=config.core_model,
            core_radius=edge_radii(wing, edges, cores, config.core_radius_m),
            **kernel_options(config))
    if not at_load_points:
        index = np.arange(wing.panels)
        matrix[index, index] -= vsm_self_correction(wing, points)
    if surfaces is not None:
        _cross_class_operator(matrix, wing, rows, points, config, cores, at_load_points, surfaces)
    return matrix


def _cross_class_operator(matrix, wing, rows, points, config, cores, at_load_points, surfaces):
    """OF-0010 P2: overwrite the blocks of points and columns of different classes (module text).

    Each column is one panel's unit rings (as the per-panel reference above); its elements act on the points of the
    other classes with the Rosenhead-Moore kernel and max(AVL, material) radii; at the load points the column keeps its
    bound segments. The diagonal blocks (one class) are left as computed."""
    from .surface_cores import CROSS_MODEL
    surfaces.check(wing)
    if len(surfaces.classes) < 2:
        return
    width = wing.node_count
    target_class = surfaces.panel_class
    for panel in range(wing.panels):
        targets = np.flatnonzero(target_class != target_class[panel])
        if not len(targets):
            continue
        strengths = np.zeros((len(rows)-1, wing.panels))
        strengths[:, panel] = 1.
        nodes, edges, gamma = pack_rings(rows, strengths, wing.panel_edges)
        radii = np.maximum(edge_radii(wing, edges, cores, config.core_radius_m), surfaces.edge_radius(edges, width))
        matrix[targets, panel] = segments_velocity(points[targets], nodes, edges, gamma, model=CROSS_MODEL,
                                                   core_radius=radii, **kernel_options(config))


_RING_LAYOUT = {}


def _ring_layout(depth, width, panel_edges, at_load_points):
    """Per-panel (segment, sign) lists in pack_rings' first-appearance order.

    Identical to calling pack_rings with a unit strength on one panel in every
    row interval: shared edges between the two rings of a panel cancel exactly
    and are omitted; row-zero (bound) edges are dropped at the load points as
    in the per-panel reference. Cached per topology.
    """
    key = (int(depth), int(width), panel_edges.tobytes(), bool(at_load_points))
    layout = _RING_LAYOUT.get(key)
    if layout is not None:
        return layout
    # Global first-appearance order of every undirected edge, exactly as the
    # pack_rings loops (row interval outer, panel inner) insert them. A panel's
    # own edges then keep the positions at which any panel first created them.
    appearance = {}
    per_panel = [dict() for _ in panel_edges]
    for i in range(depth - 1):
        for j, (left, right) in enumerate(panel_edges):
            vertices = [i*width+left, i*width+right, (i+1)*width+right, (i+1)*width+left]
            for a, b in zip(vertices, vertices[1:]+vertices[:1]):
                pair = (min(a, b), max(a, b))
                appearance.setdefault(pair, len(appearance))
                per_panel[j][pair] = per_panel[j].get(pair, 0.) + (1. if a < b else -1.)
    unique = {}
    columns = []
    for edges in per_panel:
        column = []
        for pair in sorted(edges, key=appearance.__getitem__):
            value = edges[pair]
            if value == 0. or (at_load_points and pair[0] < width and pair[1] < width):
                continue
            column.append((unique.setdefault(pair, len(unique)), value))
        columns.append(column)
    segments = np.asarray(list(unique), dtype=np.int64).reshape(-1, 2)
    pointer = np.zeros(len(columns)+1, dtype=np.int64)
    pointer[1:] = np.cumsum([len(column) for column in columns])
    layout = (segments, pointer,
              np.asarray([index for column in columns for index, _ in column], dtype=np.int64),
              np.asarray([sign for column in columns for _, sign in column], dtype=np.float64))
    _RING_LAYOUT[key] = layout
    return layout


def _batched_ring_operator(wing, rows, points, config, cores, at_load_points):
    """Attached-ring operator in one kernel pass; None if a pair would raise.

    Reference: one pack_rings + segments_velocity call per panel. The per-pair
    arithmetic and the per-column summation order are reproduced exactly, so
    the matrix is bitwise identical (tests.test_vsm_wake_batched).
    """
    from .vortex_numba import assemble_columns, pair_matrix
    from .vortex import _norm, _model
    nodes = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    segments, pointer, column_segment, column_sign = _ring_layout(
        len(rows), wing.node_count, wing.panel_edges, at_load_points)
    starts = nodes[segments[:, 0]]
    delta = nodes[segments[:, 1]] - starts
    lengths = _norm(delta)
    if np.any(lengths == 0):
        return None
    directions = delta / lengths[:, None]
    radii = edge_radii(wing, segments, cores, config.core_radius_m)
    points = np.ascontiguousarray(points, dtype=np.float64)
    pairs = np.empty((len(points), len(segments), 3))
    flags = np.zeros(len(points), dtype=np.int8)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        pair_matrix(points, np.ascontiguousarray(starts), np.ascontiguousarray(directions),
                    np.ascontiguousarray(lengths), np.ascontiguousarray(radii), int(_model(config.core_model)),
                    True, pairs, flags)
    if np.any(flags) or not np.isfinite(pairs).all():
        return None
    matrix = np.empty((len(points), len(pointer)-1, 3))
    assemble_columns(pairs, pointer, column_segment, column_sign, matrix)
    return matrix


def history_vsm_induction(wing, rows, strengths, cores, config, surfaces=None):
    """Known historical induction at the distinct closure and load point sets.

    With ``surfaces`` (OF-0010 P2) the points of each class take their own class's rows with the material kernel and
    the other classes' rows with the Rosenhead-Moore kernel and max(AVL, material) radii, in two sums; with one class it
    is the single sum below."""
    check_cores(cores, config)
    points = np.vstack((geometry(wing)[0], wing.cp))
    if len(rows) <= 1:
        zeros = np.zeros((wing.panels, 3))
        return zeros, zeros.copy()
    nodes, edges, gamma = pack_rings(rows, strengths, wing.panel_edges,
                                     compiled=compiled_network(config))
    # The local history's newest row is row two in the complete wake network.
    radii = edge_radii(wing, edges+2*wing.node_count, cores, config.core_radius_m)
    if surfaces is not None:
        surfaces.check(wing)
    if surfaces is None or len(surfaces.classes) < 2:
        velocity = segments_velocity(points, nodes, edges, gamma,
            model=config.core_model, core_radius=radii, **control_options(config))
        return velocity[:wing.panels], velocity[wing.panels:]
    same, cross = history_vsm_parts(wing, points, nodes, edges, gamma, radii, config, surfaces)
    velocity = same + cross
    return velocity[:wing.panels], velocity[wing.panels:]


def history_vsm_parts(wing, points, nodes, edges, gamma, radii, config, surfaces):
    """OF-0010 P2: the history's induction at ``points`` (closure points then load points) split into the part of each
    point's own class (material kernel and radii) and the part of the other classes (Rosenhead-Moore, max(AVL,
    material) radii); a point of a class without other-class rows gets zeros in the second part."""
    from .surface_cores import CROSS_MODEL
    width = wing.node_count
    source = surfaces.edge_class(edges, width)
    cross_radii = np.maximum(radii, surfaces.edge_radius(edges, width))
    point_class = np.concatenate((surfaces.panel_class, surfaces.panel_class))
    same_part, cross_part = np.zeros_like(points), np.zeros_like(points)
    for c in surfaces.classes:
        target = np.flatnonzero(point_class == c)
        same = source == c
        if np.any(same):
            same_part[target] = segments_velocity(points[target], nodes, edges[same], gamma[same],
                model=config.core_model, core_radius=radii[same], **control_options(config))
        if np.any(~same):
            cross_part[target] = segments_velocity(points[target], nodes, edges[~same], gamma[~same],
                model=CROSS_MODEL, core_radius=cross_radii[~same], **control_options(config))
    return same_part, cross_part


def control_options(config):
    """The engine of the wake's induction at control and load points: ``kernel_options(config)`` (exact, CPU), or with
    ``CudaSettings.control='gpu'`` under the 'cuda_tree' velocity backend the GPU direct sum of the same precision and
    block (whole-step GPU contract revision 4 addendum 1, S4 (a))."""
    cuda = getattr(config, 'cuda', None)
    if cuda is None or getattr(cuda, 'control', 'cpu') == 'cpu':
        return kernel_options(config)
    if getattr(config, 'velocity_backend', 'numpy') != 'cuda_tree':
        raise ValueError("cuda control='gpu' needs the cuda_tree velocity backend")
    return dict(backend='cuda', cuda=replace(cuda, suspect='cpu', control='cpu'))


def _solve(wing, polar, config, inflow, time, structural_velocity,
           rows=(), cores=None, known=(0.,0.,0.), known_load=(0.,0.,0.), initial=None, surfaces=None):
    extra = {} if surfaces is None else dict(surfaces=surfaces)      # surfaces=None: the calls exactly as before P2
    operator = vsm_wake_influence(wing, rows, config=config, cores=cores, **extra)
    load_operator = vsm_wake_influence(wing, rows, config=config, cores=cores,
                                     at_load_points=True, **extra)
    return solve_vsm_operator(wing, polar,
        ambient_velocity(inflow, geometry(wing)[0], time), operator, load_operator,
        known_induction=known, known_load_induction=known_load,
        structural_velocity=ambient_velocity(structural_velocity, geometry(wing)[0], time), density=config.density_kg_m3,
        load_relative_ambient=(ambient_velocity(inflow, wing.cp, time)
                               - ambient_velocity(structural_velocity, wing.cp, time)),
        kinematic_viscosity=config.kinematic_viscosity_m2_s,
        initial=initial, settings=config.solve)


def solve_finite_vsm(wing, polar, inflow, downstream_rows=(), *,
                     config=WakeConfig(), cores=None, time_s=0.,
                     structural_velocity=(0.,0.,0.), known_induction=(0.,0.,0.),
                     known_load_induction=(0.,0.,0.), initial=None):
    """Declared finite-wake VSM solve for verification, without creating history.

    Each supplied downstream ring carries the current circulation; callers can
    compare a long straight network against the semi-infinite VSM limit. This
    does not label prescribed rows as a dynamically evolved wake.
    """
    if not isinstance(config, WakeConfig):
        raise ValueError('config must be WakeConfig')
    time = _real(time_s, 'finite VSM evaluation time')
    if time.shape:
        raise ValueError('finite VSM evaluation time must be scalar')
    return _solve(wing, polar, config, inflow, time.item(), structural_velocity,
        downstream_rows, cores, known_induction, known_load_induction, initial)


def start_vsm_wake(wing, polar, config, inflow, *, time_s=0.,
                   structural_velocity=(0.,0.,0.), surfaces=None):
    """Impulsive start with the closed bound-to-TE ring; no prefilled wake."""
    if not isinstance(config, WakeConfig):
        raise ValueError('config must be WakeConfig')
    time = _real(time_s, 'initial VSM free-wake time')
    if time.shape:
        raise ValueError('initial time must be scalar')
    cores = initial_cores(wing, config)
    solution = _solve(wing, polar, config, inflow, time.item(),
                      structural_velocity, cores=cores, **({} if surfaces is None else dict(surfaces=surfaces)))
    return VsmWakeState(time.item(), 0, wing, solution,
        np.empty((0, wing.node_count, 3)), np.empty(0),
        np.empty((0, wing.panels)), ambient_velocity(structural_velocity, geometry(wing)[0], time.item()), cores, config=config)


def advance_vsm_wake(state, polar, config, inflow, *, next_wing=None,
                     structural_velocity=(0.,0.,0.), next_time_s=None, initial_circulation=None,
                     chooser=None, surfaces=None):
    """One physical wake step; failures never modify the old state.

    ``initial_circulation`` (default None: the previous step's circulation, the
    unchanged behaviour) is an explicit starting iterate of the nonlinear solve
    for callers that declare another seeding rule (aero.handover). It changes
    the path of the root finder, never the equations or the tolerances.

    ``chooser`` (default None: unchanged behaviour) lets a caller pick the
    section data and the solve settings of this step from the flow that reaches
    the panels from outside the lifting surfaces' own circulation system, which
    is only known after the wake and the rings have been convected to the new
    time (aero.handover, ``weight_basis='panel'``). It is called once, after the
    convection, as ``chooser(time_s, control_points, ring_control, load_points,
    ring_load)`` with the new time, the two points at which the model evaluates
    the flow on every panel of the new geometry (the 3/4-chord control points of
    the closure and the quarter-chord load points) and the induction of the
    rotor rings at each (zeros without rings), and returns ``(polar, solve,
    initial)``: the section data, the ``SolveSettings`` and the starting iterate
    (None: the previous circulation). ``polar`` and ``initial_circulation`` must
    then be None. The convection, the wake rows and the known induction are the
    same with and without a chooser; the returned state is bound to ``config``
    with the chosen solve settings.
    """
    if type(state) is not VsmWakeState:
        raise ValueError('advance_vsm_wake requires its own VsmWakeState')
    if not isinstance(config, WakeConfig):
        raise ValueError('config must be WakeConfig')
    validate_state_config(state, config)
    if row_limit_reached(state, config):
        raise ValueError('wake row limit reached; choose a bounded run or explicit later far-wake treatment')
    new_time = state.time_s+config.dt_s
    if not np.isfinite(new_time) or new_time <= state.time_s:
        raise ValueError('time step cannot advance the finite state time')
    if next_time_s is not None:
        requested = _real(next_time_s, 'next VSM free-wake time')
        if requested.shape or not np.isclose(requested.item(), new_time, rtol=0,
                atol=8*np.finfo(float).eps*max(1, abs(new_time))):
            raise ValueError('next time must equal old time plus configured physical dt')
        if requested.item() <= state.time_s:
            raise ValueError('requested time must advance the finite state time')
        new_time = requested.item()
    wing = state.wing if next_wing is None else next_wing
    if (wing.node_count != state.wing.node_count or
            not np.array_equal(wing.panel_edges, state.wing.panel_edges)):
        raise ValueError('wing panel count cannot change during a wake run')
    if chooser is not None and (polar is not None or initial_circulation is not None):
        raise ValueError('with a chooser pass polar=None and no initial circulation')
    if initial_circulation is None:
        seed = state.solution.circulation
    else:
        seed = _real(initial_circulation, 'initial circulation')
        if seed.shape != (wing.panels,):
            raise ValueError('initial circulation must have one value per panel')
    rows, ring_nodes = convect_geometry(state, config, inflow, next_wing=wing)
    rings = None if state.rings is None else state.rings.moved(ring_nodes)
    cores = advance_cores(state, rows, config)
    history = (np.vstack([state.solution.circulation, state.older_circulation])
               if state.step else np.empty((0, wing.panels)))
    rows, release_times, history, cores = retain_newest(
        rows, np.r_[state.time_s, state.release_times_s], history, cores, config)
    if surfaces is None:                               # the call exactly as before P2 (callers may wrap this function)
        known, known_load = history_vsm_induction(wing, rows, history, cores, config)
    else:
        known, known_load = history_vsm_induction(wing, rows, history, cores, config, surfaces)
    ring_known = ring_known_load = None
    if rings is not None:
        ring_known, ring_known_load = ring_induction_pair(rings, geometry(wing)[0], wing.cp, config)
        known = known + ring_known
        known_load = known_load + ring_known_load
    if chooser is not None:
        points = geometry(wing)[0]
        polar, settings, chosen = chooser(new_time, points, np.zeros_like(points) if ring_known is None else ring_known,
                                          wing.cp, np.zeros_like(wing.cp) if ring_known_load is None else ring_known_load)
        config = replace(config, solve=settings)        # validates the settings; nothing else of the configuration changes
        if chosen is not None:
            seed = _real(chosen, 'initial circulation')
            if seed.shape != (wing.panels,):
                raise ValueError('initial circulation must have one value per panel')
    solution = _solve(wing, polar, config, inflow, new_time,
        structural_velocity, rows[:1], cores, known, known_load, seed, **({} if surfaces is None else dict(surfaces=surfaces)))
    return VsmWakeState(new_time, state.step+1, wing, solution, rows,
        release_times, history,
        ambient_velocity(structural_velocity, geometry(wing)[0], new_time), cores, config=config,
        rings=rings)
