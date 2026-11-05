# SPDX-License-Identifier: Apache-2.0
# Source-derived update order: OpenFAST v5.0.0 FVW_UpdateStates/FVW_Euler1/FVW_RK4,
# Map_LL_NW/PropagateWake. See third_party/OPENFAST-WING-WAKE-NOTICE.md.
"""Finite free-wake runs with Euler, Heun or RK4 geometry convection.

No ground image, flight or line dynamics. By default no row is ever deleted
and there is no far-wake model. Two opt-in far-wake treatments exist:
``WakeConfig.free_convection_rows`` (old rows drift with the ambient flow) and
``WakeConfig.retained_rows`` (rows older than the limit are discarded, which
bounds the cost per step of the wing wake; rotor rings are not truncated). Pure
state transitions let a caller discard/repeat a coupling attempt.
"""

from dataclasses import dataclass, field
import operator
import numpy as np

from .vortex import CoreModel, CudaSettings, TreeSettings, _real, _model, segments_velocity, cuda_backend_available
from .wing import (Wing, WingSolution, SectionPolar, SolveSettings, owned, positive,
                   vector_field, influence_matrix, solve_circulation)
from .cores import WakeCores, initial_cores, check_cores, advance_cores, edge_radii


def kernel_options(config, large=False):
    """Biot-Savart engine for one call site.

    Only the quadratic wake-on-wake sums (geometry convection and off-body
    probes) use the tree approximation. Wing-point induction (few targets)
    and the attached-ring operators stay direct, so loads see the exact
    induced velocity of the current wake geometry.
    """
    backend = getattr(config, 'velocity_backend', 'numpy')
    if backend == 'tree' and large:
        return dict(backend='tree', tree=config.tree)
    if backend == 'cuda' and large:            # GPU direct-sum contract D1: the same large sums as the tree
        return dict(backend='cuda', cuda=config.cuda)
    if backend == 'cuda_tree' and large:       # whole-step GPU contract S1: CPU-built tree, GPU evaluation
        return dict(backend='cuda_tree', tree=config.tree, cuda=config.cuda)
    return dict(backend='numpy' if backend == 'numpy' else 'numba')


def count(value, name, minimum=0):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError()
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class WakeConfig:
    dt_s: float = .05
    core_radius_m: float = .15
    core_model: CoreModel = CoreModel.LAMB_OSEEN
    max_rows: int = 128
    free_wake: bool = True
    density_kg_m3: float = 1.225
    kinematic_viscosity_m2_s: float = 1.4639e-5
    solve: SolveSettings = SolveSettings()
    core_evolution: str = 'constant'
    eddy_viscosity_factor: float = 1.
    convection_integrator: str = 'euler'
    geometry_substeps: int = 1
    surface_guard: str = 'off'
    guard_chordwise_panels: int = 16
    wake_core_chord_fraction: float | None = None
    velocity_backend: str = 'numpy'
    tree: TreeSettings | None = None
    # Far-wake treatment (opt-in; None, the default, leaves every path and result unchanged): only this
    # many newest wake rows (counting the row leaving the trailing edge) are convected with the induced
    # velocity; older rows are convected with the ambient velocity at their own position and stage time
    # (uniform wind, Euler: a rigid drift). They keep their circulation and still induce velocity on the
    # wing and on the younger rows; no row is discarded, so the cost per step still grows with the wake.
    # Supported by the free-wake methods built on this module (free_wake, vsm_free_wake) only.
    # Analogous to the frozen buffer zone of OLAF, where "the markers convect at the average ambient
    # velocity" (Shaler, Branlard and Platt 2020, NREL/TP-5000-75959, Introduction p. 4 and section
    # 3.2.3), with two differences: each node here uses the local ambient velocity, not an average, and
    # nothing is truncated behind the zone. The OpenFAST source itself has not been compared.
    free_convection_rows: int | None = None
    # Far-wake truncation (opt-in; None, the default, keeps every released row and every existing result):
    # after each step only this many newest released rows are kept; the vortex rings between older rows
    # are removed whole, so the remaining lattice stays a set of closed rings (the spanwise filaments of
    # the last kept row then carry the full circulation of the last ring, a starting-vortex-like end).
    # The circulation solve of a step sees exactly the rows stored in the state it returns. With a
    # limit the number of stored rows is min(step, retained_rows) and ``max_rows`` no longer ends a
    # run. Precedent: the wake length limit of OLAF (WakeLength; Shaler, Branlard and Platt 2020,
    # NREL/TP-5000-75959, Introduction p. 4 and section 3.2.3), where the root and tip vortices "are
    # truncated after a specified distance" behind a frozen buffer zone. Rotor rings are not truncated,
    # so with rotors the cost per step is not bounded by this option alone.
    # Row counting: ``retained_rows`` counts released rows; ``free_convection_rows`` counts the row at the
    # trailing edge as well. Frozen rows therefore exist only while free_convection_rows < retained_rows.
    # The limit is a number of rows, not a length: the kept wake is retained_rows * dt * relative speed.
    retained_rows: int | None = None
    # Opt-in GPU sums: CudaSettings exactly when velocity_backend is 'cuda' (direct sum, GPU direct-sum contract D1) or
    # 'cuda_tree' (the CPU-built tree evaluated on the GPU, which also needs tree settings; whole-step GPU contract S1); the last
    # field, and left out of wake_config_metadata when None, so every other configuration's metadata is unchanged.
    cuda: CudaSettings | None = None

    def __post_init__(self):
        for name in ("dt_s", "core_radius_m", "density_kg_m3", "kinematic_viscosity_m2_s"):
            positive(getattr(self,name),name)
        count(self.max_rows,"max_rows",1)
        if self.free_convection_rows is not None:
            object.__setattr__(self, 'free_convection_rows', count(self.free_convection_rows, 'free_convection_rows', 1))
            if self.free_wake is False:
                raise ValueError('the far-wake limit needs a free wake')
        if self.retained_rows is not None:
            object.__setattr__(self, 'retained_rows', count(self.retained_rows, 'retained_rows', 1))
            if self.retained_rows > self.max_rows:
                raise ValueError('retained_rows cannot exceed max_rows')
        count(self.geometry_substeps, 'geometry_substeps', 1)
        count(self.guard_chordwise_panels, 'guard chordwise panels', 1)
        if self.surface_guard not in ('off', 'reject_static'):
            raise ValueError('surface guard must be off or reject_static')
        if self.surface_guard == 'off' and self.guard_chordwise_panels != 16:
            raise ValueError('guard mesh settings require an enabled surface guard')
        if self.convection_integrator not in ('euler', 'heun', 'rk4'):
            raise ValueError('unknown wake convection integrator')
        _model(self.core_model)
        if self.velocity_backend not in ('numpy', 'numba', 'tree', 'cuda', 'cuda_tree'):
            raise ValueError("velocity backend must be 'numpy', 'numba', 'tree', 'cuda' or 'cuda_tree'")
        if self.velocity_backend == 'cuda_tree':
            if self.tree is None or self.cuda is None:
                raise ValueError('the cuda_tree velocity backend needs both tree and cuda settings')
        else:
            if (self.tree is not None) != (self.velocity_backend == 'tree'):
                raise ValueError('tree settings are required exactly for the tree velocity backend')
            if (self.cuda is not None) != (self.velocity_backend == 'cuda'):
                raise ValueError('cuda settings are required exactly for the cuda velocity backend')
        if self.cuda is not None:
            if not isinstance(self.cuda, CudaSettings):
                raise ValueError('cuda must be CudaSettings')
            if self.cuda.suspect != 'cpu' and self.velocity_backend != 'cuda_tree':
                raise ValueError("cuda suspect='gpu64' is for the cuda_tree velocity backend only")
            if getattr(self.cuda, 'control', 'cpu') != 'cpu' and self.velocity_backend != 'cuda_tree':
                raise ValueError("cuda control='gpu' needs the cuda_tree velocity backend")    # GPU contract revision 4 addendum 2
            if not cuda_backend_available():
                from .vortex import cuda_unavailable_reason
                raise ValueError(f'the cuda velocity backend needs a CUDA device ({cuda_unavailable_reason()})')
        if self.tree is not None and not isinstance(self.tree, TreeSettings):
            raise ValueError('tree must be TreeSettings')
        if self.tree is not None and self.core_model not in (CoreModel.NONE, CoreModel.RANKINE,
                                                             CoreModel.LAMB_OSEEN):
            raise ValueError('the tree backend supports NONE, RANKINE and LAMB_OSEEN cores only')
        if self.wake_core_chord_fraction is not None:
            positive(self.wake_core_chord_fraction, 'wake core chord fraction')
            if self.core_model == CoreModel.NONE:
                raise ValueError('local wake birth cores require a regularized core model')
        if not isinstance(self.free_wake, bool) or not isinstance(self.solve, SolveSettings):
            raise ValueError("invalid wake switch or solver settings")
        if self.solve.jacobian != 'finite_difference':
            raise ValueError('wake circulation does not support an active-segment Jacobian')
        positive(self.eddy_viscosity_factor, 'eddy viscosity factor', zero=True)
        if self.core_evolution not in ('constant', 'age', 'age_stretch'):
            raise ValueError('unknown core evolution policy')
        if self.core_evolution != 'constant' and self.core_model != CoreModel.LAMB_OSEEN:
            raise ValueError('age growth is qualified only with the Lamb-Oseen core')


def stored_rows(step, config):
    """Released rows a state holds after ``step`` steps: the step count, or less with ``retained_rows``."""
    limit=getattr(config,'retained_rows',None)
    return step if limit is None else min(step,limit)


def retain_newest(rows, release_times, history, cores, config):
    """Opt-in far-wake truncation: keep the newest ``config.retained_rows`` released rows.

    ``rows`` (r,nodes,3), ``release_times`` (r,), ``history`` (r-1,panels) and the core history are
    cut together, so whole rings are removed. Without a limit, or below it, the inputs are returned
    as they are (the default path).
    """
    limit=getattr(config,'retained_rows',None)
    if limit is None or len(rows) <= limit:
        return rows, release_times, history, cores
    kept=None if cores is None else WakeCores(cores.span[:limit], cores.stream[:limit], cores.policy,
                                              cores.initial_radius, cores.diffusion_rate, chord_fraction=cores.chord_fraction)
    return rows[:limit], release_times[:limit], history[:limit-1], kept


def row_limit_reached(state, config):
    """True when a run without truncation has used its ``max_rows``; never with ``retained_rows``."""
    return getattr(config,'retained_rows',None) is None and state.step >= config.max_rows


@dataclass(frozen=True)
class WakeState:
    time_s: float
    step: int
    wing: Wing
    solution: WingSolution
    free_rows: np.ndarray
    release_times_s: np.ndarray
    older_circulation: np.ndarray
    structural_velocity: np.ndarray
    cores: WakeCores | None = None
    config: WakeConfig = field(kw_only=True)
    # Optional closed vortex loops that are not wing rows (rotor rings, G3r):
    # an object with nodes (k,3), edges (e,2), circulation (e,) and moved(nodes).
    # None (the default) leaves every existing code path and result unchanged.
    rings: object | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.config, WakeConfig):
            raise ValueError('wake state requires its complete configuration')
        check_cores(self.cores, self.config)
        check_rings(self.rings)
        time = _real(self.time_s,"state time")
        if time.shape:
            raise ValueError("time must be a finite scalar")
        count(self.step,"step")
        for name in ("free_rows", "release_times_s", "older_circulation"):
            object.__setattr__(self,name,owned(getattr(self,name)))
        n=self.wing.panels
        object.__setattr__(self,'structural_velocity',owned(vector_field(self.structural_velocity,n,'structural velocity')))
        rows=stored_rows(self.step,self.config)   # equals the step unless the opt-in truncation is set
        if self.free_rows.shape != (rows,self.wing.node_count,3) or self.release_times_s.shape != (rows,) or self.older_circulation.shape != (max(rows-1,0),n):
            raise ValueError("wake row/strength/step shape mismatch")
        if np.any(self.release_times_s >= self.time_s) or np.any(np.diff(self.release_times_s) >= 0):
            raise ValueError("wake release times must be past and newest-first")
        if self.solution.circulation.shape != (n,):
            raise ValueError("wing solution shape mismatch")
        if self.cores is not None and (not isinstance(self.cores, WakeCores) or
                self.cores.span.shape != (rows, n) or
                self.cores.stream.shape != (rows, self.wing.node_count)):
            raise ValueError('core history shape/type mismatch')


def pack_rings(rows, strengths, panel_edges=None, *, compiled=False):
    """Aggregate shared directed edges; circulation differences shed vorticity.

    A ring runs (row i, span j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j).
    Exact-zero strengths are omitted; no tolerance discards small physical edges.
    Shared edges have one core radius; use edge_radii for material core history.
    ``compiled=True`` (accelerated velocity backends only) runs the same loop
    compiled (network_numba.pack_ring_edges); the result is bitwise identical.
    """
    rows, strengths = _real(rows,"rows"), _real(strengths,"ring strengths")
    if rows.ndim != 3 or rows.shape[2] != 3 or rows.shape[1] < 2 or len(rows) < 2:
        raise ValueError("rows must have shape (depth+1,span+1,3)")
    depth,width=rows.shape[:2]
    if panel_edges is None:panel_edges=np.column_stack([np.arange(width-1),np.arange(1,width)])
    panel_edges=np.asarray(panel_edges)
    if panel_edges.ndim!=2 or panel_edges.shape[1]!=2 or panel_edges.dtype.kind not in 'iu' or np.any(panel_edges<0) or np.any(panel_edges>=width) or np.any(panel_edges[:,0]==panel_edges[:,1]):
        raise ValueError('invalid spanwise panel connectivity')
    if strengths.shape != (depth-1,len(panel_edges)):
        raise ValueError("ring strength shape mismatch")
    if compiled:
        from .network_numba import pack_ring_edges
        keys, values = pack_ring_edges(depth, width, panel_edges, strengths)
        return rows.reshape(-1,3).copy(), keys, values
    edges={}
    for i in range(depth-1):
        for j,(left,right) in enumerate(panel_edges):
            vertices=[i*width+left,i*width+right,(i+1)*width+right,(i+1)*width+left]
            gamma=strengths[i,j]
            for a,b in zip(vertices,vertices[1:]+vertices[:1]):
                key=(min(a,b),max(a,b))
                edges[key]=edges.get(key,0.)+(gamma if a<b else -gamma)
    keys=[key for key,value in edges.items() if value != 0.]
    return rows.reshape(-1,3).copy(), np.asarray(keys,dtype=np.int64).reshape(-1,2), np.array([edges[key] for key in keys])


def wake_network(state):
    rows=np.concatenate(([state.wing.bound,state.wing.trailing_edge],state.free_rows),axis=0)
    gamma=state.solution.circulation
    strengths=np.vstack([gamma,gamma,state.older_circulation]) if state.step else gamma[None,:]
    return pack_rings(rows,strengths,state.wing.panel_edges,compiled=compiled_network(state.config))


def check_rings(rings):
    """Validate the optional closed-loop set carried by a state (duck-typed, see WakeState)."""
    if rings is None:
        return
    for name in ('nodes', 'edges', 'circulation', 'moved'):
        if not hasattr(rings, name):
            raise ValueError(f'rings must provide {name}')
    nodes, edges, gamma = np.asarray(rings.nodes), np.asarray(rings.edges), np.asarray(rings.circulation)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or edges.ndim != 2 or edges.shape[1] != 2 or gamma.shape != (len(edges),):
        raise ValueError('rings must have nodes (k,3), edges (e,2) and circulation (e,)')
    if edges.dtype.kind not in 'iu' or (len(edges) and (edges.min() < 0 or edges.max() >= len(nodes))):
        raise ValueError('ring edges must index ring nodes')
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(gamma))):
        raise ValueError('ring geometry and circulation must be finite')


def extended_network(state):
    """Wing network plus the state's rings; returns (nodes, edges, gamma, ring_first).

    Ring nodes follow every wing node (bound, TE, free rows), ring edges follow
    every wing edge, so wing-only quantities keep their indices and summation
    order; ring_first == len(nodes) when there are no rings.
    """
    nodes, edges, gamma = wake_network(state)
    rings = state.rings
    if rings is None or len(rings.edges) == 0:
        return nodes, edges, gamma, len(nodes)
    ring_first = len(nodes)
    ring_nodes = np.asarray(rings.nodes, dtype=float)
    return (np.concatenate((nodes, ring_nodes)), np.concatenate((edges, np.asarray(rings.edges, dtype=np.int64) + ring_first)),
            np.concatenate((gamma, np.asarray(rings.circulation, dtype=float))), ring_first)


def ring_induction(rings, points, config, *, large=False, sheet_points=None):
    """Velocity of a state's rings at lifting-surface points (zero without rings).

    The core is the rings' ``surface_core_m`` when they declare one (a model
    parameter for filaments crossing the surface), else the configured core.
    Convection uses the rings' ``core_m`` when declared, else the configured core.
    Rings that declare ``surface_overlap`` induce through their refined sheet at these points
    (``RingWake.surface_sheet``; the rings to refine are chosen at ``sheet_points`` when given, else at ``points``);
    without it the stored rings are used unchanged.
    """
    points = np.asarray(points, dtype=float)
    if rings is None or len(rings.edges) == 0:
        return np.zeros_like(points)
    core = getattr(rings, 'surface_core_m', None)
    core = config.core_radius_m if core is None else float(core)
    nodes, edges, gamma = rings.nodes, rings.edges, rings.circulation
    if getattr(rings, 'surface_overlap', None) is not None:
        chosen = points if sheet_points is None else np.asarray(sheet_points, dtype=float)
        nodes, edges, gamma, _ = rings.surface_sheet(chosen.reshape(-1, 3), core)
    return segments_velocity(points, np.asarray(nodes, dtype=float), np.asarray(edges, dtype=np.int64),
                             np.asarray(gamma, dtype=float), model=config.core_model,
                             core_radius=core,
                             **kernel_options(config, large=large))


def ring_induction_pair(rings, first_points, second_points, config):
    """``ring_induction`` at two point sets of the lifting surfaces (closure and load points) from one sheet.

    Without ``surface_overlap`` these are the two plain calls. With it, the sheet is refined once at the union of the
    two sets and evaluated at both in one sum (per-point sums do not depend on the point set), so the closure and the
    loads see the same sheet (contract section 8, 2026-10-22 18:55 BST).
    """
    first_points, second_points = np.asarray(first_points, dtype=float), np.asarray(second_points, dtype=float)
    if rings is None or len(rings.edges) == 0 or getattr(rings, 'surface_overlap', None) is None:
        return ring_induction(rings, first_points, config), ring_induction(rings, second_points, config)
    both = np.concatenate((first_points.reshape(-1, 3), second_points.reshape(-1, 3)))
    velocity = ring_induction(rings, both, config)
    count = len(first_points.reshape(-1, 3))
    return velocity[:count].reshape(first_points.shape), velocity[count:].reshape(second_points.shape)


def compiled_network(config):
    """Accelerated backends also run the O(wake) bookkeeping compiled (bitwise identical)."""
    return getattr(config, 'velocity_backend', 'numpy') != 'numpy'


def ambient_velocity(inflow, points, time):
    values=inflow(points.copy(),float(time)) if callable(inflow) else inflow
    return vector_field(values,len(points),"ambient velocity")


def _network_convection(nodes, edges, gamma, first, radii, config, inflow, time, ring_first=None, ring_free=None,
                        ring_radii=None):
    """Evaluate one geometry stage on the same frozen circulation graph.

    Points are nodes[first:]. With ``ring_first`` (index of the first ring
    node), the wing rows nodes[first:ring_first] keep the 'raise' singularity
    rule while ring nodes nodes[ring_first:] use 'zero': identical rings can
    stack exactly onto one another (starting-ring cluster) and the regularised
    velocity on a filament is zero. Per-point sums do not depend on the point
    set, so the row velocities are the same as without rings. ``ring_free``
    (None: all) is the number of leading ring nodes that take the induced
    velocity; the ring nodes after them keep the ambient velocity (the frozen
    far wake of the rings, ``RingWake.free_convection_steps``). ``ring_radii``
    (None: ``radii``) are the per-edge cores used for the ring targets only
    (``RingWake.bound_core_m``); the row targets always use ``radii``.
    """
    points=nodes[first:]
    target_radii=radii if ring_radii is None else ring_radii
    velocity=ambient_velocity(inflow,points,time).copy()
    if config.free_wake and len(edges):
        # Distinct surface IDs can share the same physical root node. Every
        # coincident endpoint is incident for convection, irrespective of ID.
        # Exact positions only: nearby filaments/interior crossings still raise.
        if compiled_network(config):
            from .network_numba import incident_pairs
            exclusions=incident_pairs(nodes,edges,first)
        else:
            locations={}
            for index,point in enumerate(points):
                locations.setdefault(tuple(point),[]).append(index)
            exclusions=[]
            for segment,(a,b) in enumerate(edges):
                for node in (a,b):
                    exclusions.extend((index,segment) for index in locations.get(tuple(nodes[node]),()))
            exclusions=np.asarray(exclusions,dtype=np.int64).reshape(-1,2)
        rows_end=len(points) if ring_first is None or ring_first >= len(nodes) else ring_first-first
        # The ring targets: all ring nodes, or the leading ``ring_free`` of them (older rings keep the ambient velocity).
        ring_end=len(points) if ring_free is None else min(len(points), rows_end+int(ring_free))
        limit=getattr(config,'free_convection_rows',None)
        if limit is not None and limit*first < rows_end:
            # Far-wake treatment: the newest ``limit`` rows are the only row targets; the older rows keep
            # the ambient velocity set above. Per-point sums do not depend on the point set, so the
            # free rows get the velocity they would get without the limit. Ring nodes stay free unless
            # the rings declare their own limit (``ring_free``).
            # Older rows are no longer targets, so a point-on-filament or overflow condition that the
            # complete evaluation would raise for an old row is not raised here.
            free=limit*first
            velocity[:free] += segments_velocity(points[:free],nodes,edges,gamma,model=config.core_model,
                                                 core_radius=radii, exclusions=exclusions[exclusions[:,0]<free],
                                                 **kernel_options(config, large=True))
            if rows_end < ring_end:
                ring_pairs=exclusions[(exclusions[:,0]>=rows_end)&(exclusions[:,0]<ring_end)]-np.array([rows_end,0])
                velocity[rows_end:ring_end] += segments_velocity(points[rows_end:ring_end],nodes,edges,gamma,model=config.core_model,
                                                                 core_radius=target_radii, exclusions=ring_pairs, singularity='zero',
                                                                 **kernel_options(config, large=True))
        elif ring_first is None or ring_first >= len(nodes):
            velocity += segments_velocity(points,nodes,edges,gamma,model=config.core_model,
                                          core_radius=radii,
                                          exclusions=exclusions,
                                          **kernel_options(config, large=True))
        else:
            split=ring_first-first
            rows_pairs=exclusions[exclusions[:,0]<split]
            ring_pairs=exclusions[(exclusions[:,0]>=split)&(exclusions[:,0]<ring_end)]-np.array([split,0])
            velocity[:split] += segments_velocity(points[:split],nodes,edges,gamma,model=config.core_model,
                                                  core_radius=radii, exclusions=rows_pairs,
                                                  **kernel_options(config, large=True))
            if split < ring_end:
                velocity[split:ring_end] += segments_velocity(points[split:ring_end],nodes,edges,gamma,model=config.core_model,
                                                              core_radius=target_radii, exclusions=ring_pairs, singularity='zero',
                                                              **kernel_options(config, large=True))
    return velocity


def _edge_radii_extended(state, edges, ring_first, config):
    """Material cores for the wing edges; for ring edges the rings' own core (``RingWake.core_m``) when they declare
    one, else the constant configured core."""
    wing_count = int(np.count_nonzero(np.all(edges < ring_first, axis=1))) if len(edges) else 0
    if wing_count == len(edges):
        return edge_radii(state.wing, edges, state.cores, config.core_radius_m)
    radii = edge_radii(state.wing, edges[:wing_count], state.cores, config.core_radius_m)
    ring_core = getattr(state.rings, 'core_m', None)
    ring_core = config.core_radius_m if ring_core is None else float(ring_core)
    return np.concatenate((np.broadcast_to(radii, (wing_count,)), np.full(len(edges) - wing_count, ring_core)))


def _ring_target_radii(state, edges, first, radii):
    """Per-edge cores for the ring targets (``RingWake.bound_core_m``), or None when the rings declare none.

    The bound system of the lifting surfaces is every wing edge whose two nodes lie on the bound row or the
    trailing-edge row (node ids below ``2*first``): the bound vortex and the chordwise and trailing-edge segments of
    the bound rings. Those edges take ``bound_core_m``; every other edge keeps its core.
    """
    bound_core = None if state.rings is None else getattr(state.rings, 'bound_core_m', None)
    if bound_core is None or not len(edges):
        return None
    ring_radii = np.array(np.broadcast_to(radii, (len(edges),)), dtype=float)
    ring_radii[np.all(edges < 2*first, axis=1)] = float(bound_core)
    return ring_radii


def node_convection_velocity(state, config, inflow):
    """Velocity on old TE + free rows; bound nodes never fluid-convect."""
    validate_state_config(state, config)
    nodes,edges,gamma,ring_first=extended_network(state)
    check_cores(state.cores, config)
    first=state.wing.node_count
    radii=_edge_radii_extended(state, edges, ring_first, config)
    velocity=_network_convection(nodes, edges, gamma, first, radii, config, inflow, state.time_s, ring_first)
    return velocity[:ring_first-first].reshape(len(state.free_rows)+1,first,3)


def convect_rows(state, config, inflow, *, next_wing=None):
    """Advance only old TE/free geometry (see convect_geometry); rings, if any, are moved too but not returned."""
    return convect_geometry(state, config, inflow, next_wing=next_wing)[0]


def convect_geometry(state, config, inflow, *, next_wing=None):
    """Advance old TE/free geometry (and ring nodes); release no intermediate wake rows.

    Returns (rows, ring_nodes): rows shaped (stored rows + 1, node_count, 3) and the
    moved ring nodes (empty without rings). Bound nodes never fluid-convect.

    All stages retain the old circulation graph. Bound nodes interpolate
    linearly between endpoint geometries; the old TE convects in every stage.
    This is exact for translation, not intermediate rigid rotation.
    Ambient fields are evaluated
    at stage positions/times. Material core radii are frozen during this step;
    advance_cores applies the existing split core update after geometry moves.
    RK4 therefore describes this frozen-strength geometry ODE, not fourth-order
    accuracy of the full circulation/core/aircraft coupled algorithm.
    Heun is the explicit second-order predictor-corrector (one endpoint
    prediction and trapezoidal correction), not an iterated implicit solve.

    This frozen-graph geometry primitive permits integrator/substep probes.
    Full state advances enforce configuration ownership before calling it.
    """
    nodes,edges,gamma,ring_first=extended_network(state)
    check_cores(state.cores, config)
    first=state.wing.node_count
    if next_wing is not None and (next_wing.node_count != first or
            not np.array_equal(next_wing.panel_edges, state.wing.panel_edges)):
        raise ValueError('wing topology cannot change during geometry convection')
    radii=_edge_radii_extended(state, edges, ring_first, config)
    ring_radii=_ring_target_radii(state, edges, first, radii)
    bound=nodes[:first]
    displacement = None if next_wing is None else next_wing.bound-bound
    if displacement is not None and not np.any(displacement):
        displacement = None
    points=nodes[first:].copy()
    split=ring_first-first  # row points precede ring points
    # Frozen far wake of the rings (opt-in on the ring set): the leading ring nodes that convect freely.
    ring_free=None
    if state.rings is not None and getattr(state.rings, 'free_convection_steps', None) is not None:
        ring_free=state.rings.free_node_count(state.time_s, config.dt_s)
    dt=config.dt_s/config.geometry_substeps
    guard = make_surface_guard(state.wing, config, next_wing)

    def checked(candidate, previous, label, time):
        if guard is not None:
            guard.check_rows(previous[:split].reshape(-1, first, 3), candidate[:split].reshape(-1, first, 3),
                             state.wing, stage=label, time_s=time)
        return candidate

    # Opt-in sheet drift of the frozen rings (RingWake.sheet_drift, model A''): a constant velocity per ring node.
    drift = None
    if state.rings is not None and getattr(state.rings, 'sheet_drift', False) and ring_free is not None:
        node_drift = getattr(state.rings, 'drift_m_s', None)
        if node_drift is not None and len(node_drift) > ring_free:
            drift = np.zeros_like(points)
            drift[split+ring_free:] = node_drift[ring_free:]

    def rhs(current, time):
        # Avoid arithmetic on the legacy fixed path and the first Euler stage.
        stage_bound = bound
        if displacement is not None and time != state.time_s:
            stage_bound = bound + ((time-state.time_s)/config.dt_s)*displacement
        velocity = _network_convection(np.concatenate((stage_bound,current)), edges, gamma,
                                       first, radii, config, inflow, time, ring_first, ring_free, ring_radii)
        return velocity if drift is None else velocity + drift

    for substep in range(config.geometry_substeps):
        time=state.time_s+substep*dt
        k1=rhs(points,time)
        if config.convection_integrator == 'euler':
            candidate=points+dt*k1
        elif config.convection_integrator == 'heun':
            predicted=checked(points+dt*k1, points, 'heun_predictor', time+dt)
            k2=rhs(predicted,time+dt)
            candidate=points+dt*(k1+k2)/2
        else:
            k2=rhs(checked(points+dt*k1/2, points, 'rk4_stage2', time+dt/2),time+dt/2)
            k3=rhs(checked(points+dt*k2/2, points, 'rk4_stage3', time+dt/2),time+dt/2)
            k4=rhs(checked(points+dt*k3, points, 'rk4_stage4', time+dt),time+dt)
            candidate=points+dt*(k1+2*k2+2*k3+k4)/6
        points=checked(candidate, points, 'accepted_geometry', time+dt)
    return points[:split].reshape(len(state.free_rows)+1,first,3), points[split:]


def make_surface_guard(wing, config, next_wing=None):
    """Build a common physical surface; changing solid geometry is unsupported."""
    if config.surface_guard == 'off':
        return None
    if next_wing is not None:
        for name in ('leading_edge', 'trailing_edge', 'panel_edges', 'camber_x', 'camber_offset_m'):
            first, last = getattr(wing, name, None), getattr(next_wing, name, None)
            if (first is None) != (last is None) or (first is not None and not np.array_equal(first, last)):
                raise NotImplementedError('surface guard supports stationary physical surfaces only')
    from .geometry_guard import StaticSurfaceGuard
    from .lifting_surface import LiftingSurfaceConfig, build_lifting_surface_mesh
    mesh = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(
        chordwise_panels=config.guard_chordwise_panels, camber_normal='cell_secant'))
    return StaticSurfaceGuard.from_mesh(wing, mesh)


def check_straight_surface(wing, row, config, time):
    """Inspect a prescribed wake against the common stationary physical mesh."""
    guard = make_surface_guard(wing, config)
    if guard is not None:
        guard.check_rows(wing.trailing_edge[None], np.asarray(row)[None], wing,
                         stage='prescribed_straight', time_s=time)


def start_wake(wing, polar, config, inflow, *, time_s=0., structural_velocity=(0.,0.,0.)):
    """Impulsive start: bound wing ring only, no prefilled artificial wake."""
    time=_real(time_s,"initial time")
    if time.shape:
        raise ValueError("initial time must be scalar")
    structural_velocity=ambient_velocity(structural_velocity,wing.cp,time.item())
    matrix=influence_matrix(wing,model=config.core_model,core_radius=config.core_radius_m)
    result=solve_circulation(wing,polar,ambient_velocity(inflow,wing.cp,time.item()),matrix,
                             structural_velocity=structural_velocity,density=config.density_kg_m3,
                             kinematic_viscosity=config.kinematic_viscosity_m2_s,settings=config.solve)
    return WakeState(time.item(),0,wing,result,np.empty((0,wing.node_count,3)),np.empty(0),np.empty((0,wing.panels)),structural_velocity,initial_cores(wing, config),config=config)


def validate_state_config(state, config):
    if not isinstance(state, WakeState) or state.config != config:
        raise ValueError('state and wake configuration differ; start a new run')


def current_influence(wing, first_row, cores, config):
    """Unknown attached circulation with exactly the shared historical edge cores."""
    if cores is None:
        return influence_matrix(wing, [first_row], model=config.core_model, core_radius=config.core_radius_m)
    rows = np.array([wing.bound, wing.trailing_edge, first_row])
    matrix = np.empty((wing.panels, wing.panels, 3))
    for panel in range(wing.panels):
        strengths = np.zeros((2, wing.panels)); strengths[:, panel] = 1.
        nodes, edges, gamma = pack_rings(rows, strengths, wing.panel_edges)
        matrix[:, panel] = segments_velocity(wing.cp, nodes, edges, gamma,
            model=config.core_model, core_radius=edge_radii(wing, edges, cores, config.core_radius_m),
            singularity='zero', **kernel_options(config))
    return matrix


def advance_wake(state, polar, config, inflow, *, next_wing=None,
                 structural_velocity=(0.,0.,0.), next_time_s=None):
    """Evaluate one new state. Failure/repeated calls never alter the old state."""
    validate_state_config(state, config)
    if type(state) is not WakeState:
        raise ValueError('advance_wake requires its own WakeState')
    if row_limit_reached(state, config):
        raise ValueError("wake row limit reached; choose a bounded run or explicit later far-wake treatment")
    new_time=state.time_s+config.dt_s
    if not np.isfinite(new_time) or new_time <= state.time_s:
        raise ValueError("time step cannot advance the finite state time")
    if next_time_s is not None:
        requested=_real(next_time_s,"next time")
        if requested.shape or not np.isclose(requested.item(),new_time,rtol=0,atol=8*np.finfo(float).eps*max(1,abs(new_time))):
            raise ValueError("next time must equal old time plus configured physical dt")
        # Retain the caller's accepted clock to avoid accumulating addition error
        # against an independently evaluated step*dt motion schedule.
        if requested.item() <= state.time_s:
            raise ValueError("requested time must advance the finite state time")
        new_time=requested.item()
    wing=state.wing if next_wing is None else next_wing
    if wing.node_count != state.wing.node_count or not np.array_equal(wing.panel_edges,state.wing.panel_edges):
        raise ValueError("wing panel count cannot change during a wake run")
    structural_velocity=ambient_velocity(structural_velocity,wing.cp,new_time)
    new_rows,ring_nodes=convect_geometry(state,config,inflow,next_wing=wing)
    rings=None if state.rings is None else state.rings.moved(ring_nodes)
    cores=advance_cores(state, new_rows, config)
    history=np.vstack([state.solution.circulation,state.older_circulation]) if state.step else np.empty((0,wing.panels))
    new_rows,release_times,history,cores=retain_newest(new_rows,np.r_[state.time_s,state.release_times_s],history,cores,config)
    known=np.zeros_like(wing.cp)
    if len(new_rows)>1:
        nodes,edges,gamma=pack_rings(new_rows,history,wing.panel_edges,compiled=compiled_network(config))
        radii=edge_radii(wing, edges+2*wing.node_count, cores, config.core_radius_m)
        known=segments_velocity(wing.cp,nodes,edges,gamma,model=config.core_model,core_radius=radii,
                                **kernel_options(config))
    if rings is not None:
        known=known+ring_induction(rings,wing.cp,config)
    matrix=current_influence(wing,new_rows[0],cores,config)
    solution=solve_circulation(wing,polar,ambient_velocity(inflow,wing.cp,new_time),matrix,
                               structural_velocity=structural_velocity,known_induction=known,
                               density=config.density_kg_m3,kinematic_viscosity=config.kinematic_viscosity_m2_s,
                               initial=state.solution.circulation,settings=config.solve)
    return WakeState(new_time,state.step+1,wing,solution,new_rows,
                     release_times,history,structural_velocity,cores,config=config,
                     rings=rings)
