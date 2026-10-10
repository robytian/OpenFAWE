# SPDX-License-Identifier: Apache-2.0
"""Chordwise ring VLM and explicit material-wake UVLM for thin rigid surfaces.

SI Cartesian geometry follows Wing: downstream chord, ordered span, normal
``chord cross span``. Each geometric chord/span panel has an independent ring
strength, a quarter-panel-chord front vortex and a three-quarter-panel-chord
collocation point. Optional explicit mean-surface offsets add camber; LE/TE
alone define straight chords. The default is inviscid. Optional grid_section
coupling uses supplied Cl/Cd/Cm tables; no thickness or dynamic stall is inferred.

The boundary equation is ``n dot (U - V_body + u_bound + u_wake) = 0``.
Straight wake rings share the last chord-row circulation (steady Kutta). The
transient method defaults to PREVIOUS accepted trailing-edge circulation.
Optional current or endpoint-averaged release is coupled into the boundary
solve. Released material strengths subsequently stay fixed. The reported TE
jump depends on this convention and is not a cross-scheme accuracy metric.

Loads use bound-edge Joukowski forces and rho*dGamma/dt*A*n as described in
Lambert & Dimitriadis, "Induced Drag Calculations ... Cambered Wings", AIAA
Journal (2017), equations 1--3, author manuscript:
https://orbi.uliege.be/bitstream/2268/200767/1/AIAAjournal_InducedDrag_accepted.pdf
The TE closure carries no surface traction (Kutta cut). The finite-step TE
circulation mismatch remains in the induced field and is never hidden. Local
edge self induction is excluded explicitly. This is not a pressure-only load
formula; edge forces retain leading-edge suction. Initial Gamma rate is set
to zero by default. Optional bdf2_consistent requires an explicit initial rate;
neither option resolves a physical start impulse.
"""

from dataclasses import dataclass, replace
import numpy as np

from .wing import WingSolution, owned, positive, vector_field
from .wake import WakeConfig, ambient_velocity, count, pack_rings
from .vortex import _real, _norm, segments_velocity, ring_velocity
from .cores import initial_cores, check_cores, advance_cores, WakeCores
from .polar_coupling import (ClosureProblem, PolarCouplingResult, SectionReferences,
    encode_polars, polar_tables)
from .wake_potential import (potential_network, wake_potential_network, potential_edge_radii,
    material_row_coordinates, near_junction_distance)
from .initial_rate import InitialGammaRate, check_initial_rate


def _indices(value):
    values = np.asarray(value, dtype=np.int64)
    return np.frombuffer(values.tobytes(), dtype=np.int64).reshape(values.shape)


@dataclass(frozen=True)
class LiftingSurfaceConfig:
    """Thin-surface mesh and explicit optional conditioning limit.

    Cosine LE/TE cell lengths scale as O(Nc**-2). At fixed chord and prescribed
    relative speed, keeping U*dt/min_cell_length bounded therefore requires
    dt = O(Nc**-2), not O(Nc**-1). This is a resolution diagnostic, not a CFL
    stability or accuracy guarantee. No time step is selected automatically.
    Archive diagnostics use local |wind-body velocity| and cell-center chord
    lengths. The heave map uses freestream U and the TE cell x-width; those
    ratios coincide only where their geometry and velocity definitions agree.
    See docs/design/lifting-surface-spacing.md and the retained two-parameter
    map in docs/tasks/OF-0006-spacing20260922/evidence/source-limits.json.
    """
    chordwise_panels: int = 4
    wake_length_m: float = 1000.
    linear_residual_tolerance_m_s: float = 1e-9
    wake_direction: tuple | None = None
    max_condition_number: float | None = None
    chordwise_spacing: str = 'uniform'
    camber_normal: str = 'cell_secant'
    wake_release: str = 'previous'
    gamma_derivative: str = 'backward_euler'
    steady_mirror_projection: bool = False
    mirror_projection_rtol: float = 1e-8
    mirror_projection_atol_m2_s: float = 1e-10
    polar_coupling: str = 'none'
    polar_max_virtual_incidence_deg: float = 15.
    polar_max_iterations: int = 40
    polar_cl_tolerance: float = 1e-8
    polar_initial_virtual_incidence_deg: float = 0.
    wake_basis: str = 'constant'
    wake_quadrature_order: int = 4
    wake_quadrature: str = 'gauss'
    initial_cap: str = 'legacy_zero'
    bound_convection_core: str = 'legacy_zero'
    bound_convection_chord_fraction: float = .1

    def __post_init__(self):
        if self.bound_convection_core not in ('legacy_zero', 'te_incident_birth', 'all_bound_chord'):
            raise ValueError('unknown bound convection core mode')
        positive(self.bound_convection_chord_fraction, 'bound convection chord fraction')
        if self.bound_convection_core != 'all_bound_chord' and self.bound_convection_chord_fraction != .1:
            raise ValueError('bound chord fraction requires all_bound_chord mode')
        if self.bound_convection_core == 'te_incident_birth' and (
                self.wake_basis != 'linear_potential' or self.wake_release != 'current'):
            raise ValueError('te_incident_birth requires linear_potential/current')
        if self.initial_cap not in ('legacy_zero', 'wake_core'):
            raise ValueError('initial cap must be legacy_zero or wake_core')
        count(self.chordwise_panels, 'chordwise panels', 1)
        if self.wake_basis not in ('constant', 'linear_potential'):
            raise ValueError('wake basis must be constant or linear_potential')
        if count(self.wake_quadrature_order, 'wake quadrature order', 1) not in (4, 8):
            raise ValueError('wake quadrature order must be 4 or 8')
        if self.wake_quadrature not in ('gauss', 'te_graded'):
            raise ValueError('wake quadrature must be gauss or te_graded')
        if self.wake_basis == 'constant' and (self.wake_quadrature_order != 4 or
                                              self.wake_quadrature != 'gauss'):
            raise ValueError('wake quadrature settings apply only to linear_potential')
        if self.wake_basis == 'linear_potential' and (
                self.wake_release != 'current' or self.steady_mirror_projection):
            raise ValueError('linear_potential requires current release and no mirror projection')
        if self.polar_coupling not in ('none', 'grid_section'):
            raise ValueError('polar coupling must be none or grid_section')
        count(self.polar_max_iterations, 'polar iterations', 1)
        positive(self.polar_cl_tolerance, 'polar Cl tolerance')
        positive(self.polar_max_virtual_incidence_deg, 'virtual incidence limit')
        initial = _real(self.polar_initial_virtual_incidence_deg, 'initial virtual incidence')
        if initial.shape or abs(initial) > self.polar_max_virtual_incidence_deg:
            raise ValueError('initial virtual incidence must be scalar and inside the declared limit')
        if self.polar_max_virtual_incidence_deg >= 90:
            raise ValueError('virtual incidence limit must be below 90 degrees')
        if self.polar_coupling != 'none' and self.steady_mirror_projection:
            raise ValueError('polar coupling and steady mirror projection cannot be combined')
        positive(self.wake_length_m, 'prescribed wake length')
        positive(self.linear_residual_tolerance_m_s, 'normal-velocity tolerance')
        if self.chordwise_spacing not in ('uniform', 'cosine'):
            raise ValueError('chordwise spacing must be uniform or cosine')
        if self.camber_normal not in ('cell_secant', 'reconstructed_slope'):
            raise ValueError('camber normal must be cell_secant or reconstructed_slope')
        if self.camber_normal == 'reconstructed_slope' and self.chordwise_panels < 2:
            raise ValueError('reconstructed camber normal requires at least two chordwise panels')
        if self.wake_release not in ('previous', 'current', 'trapezoidal'):
            raise ValueError('wake release must be previous, current or trapezoidal')
        if self.gamma_derivative not in ('backward_euler', 'bdf2', 'bdf2_consistent'):
            raise ValueError('Gamma derivative must be backward_euler, bdf2 or bdf2_consistent')
        if self.gamma_derivative == 'bdf2_consistent' and (
                self.wake_basis != 'linear_potential' or self.initial_cap != 'wake_core'):
            raise ValueError('bdf2_consistent requires linear_potential/current and wake_core initial cap')
        if not isinstance(self.steady_mirror_projection, bool):
            raise ValueError('steady mirror projection must be boolean')
        positive(self.mirror_projection_rtol, 'mirror projection relative limit')
        positive(self.mirror_projection_atol_m2_s, 'mirror projection absolute limit')
        if self.max_condition_number is not None:
            positive(self.max_condition_number, 'condition number limit')
        if self.wake_direction is not None:
            direction = _real(self.wake_direction, 'prescribed wake direction')
            if direction.shape != (3,) or _norm(direction) == 0:
                raise ValueError('prescribed wake direction must be a nonzero (3,) vector')
            object.__setattr__(self, 'wake_direction', tuple(direction/_norm(direction)))


@dataclass(frozen=True)
class LiftingSurfaceMesh:
    vertices: np.ndarray
    cells: np.ndarray
    vortex_rows: np.ndarray
    collocation_points: np.ndarray
    normals: np.ndarray
    area: np.ndarray
    area_normals: np.ndarray
    centroids: np.ndarray
    strip_index: np.ndarray
    chord_index: np.ndarray

    def __post_init__(self):
        for name in ('vertices', 'vortex_rows', 'collocation_points', 'normals',
                     'area', 'area_normals', 'centroids'):
            object.__setattr__(self, name, owned(getattr(self, name)))
        for name in ('cells', 'strip_index', 'chord_index'):
            object.__setattr__(self, name, _indices(getattr(self, name)))

    @property
    def panels(self):
        return len(self.area)


def build_lifting_surface_mesh(wing, settings=LiftingSurfaceConfig()):
    """Bilinear cells on a supplied mean surface, or original flat-chord strips."""
    if not isinstance(settings, LiftingSurfaceConfig):
        raise ValueError('settings must be LiftingSurfaceConfig')
    nc, width = settings.chordwise_panels, wing.node_count
    uniform = settings.chordwise_spacing == 'uniform' or nc == 1
    fraction = (np.linspace(0., 1., nc+1) if uniform else
                .5*(1.-np.cos(np.pi*np.arange(nc+1)/nc)))
    le, chord = wing.leading_edge, wing.trailing_edge-wing.leading_edge
    rows = le[None] + fraction[:, None, None]*chord[None]
    vortex_fractions = (np.r_[(np.arange(nc)+.25)/nc, 1.] if uniform else
                        np.r_[fraction[:-1]+.25*np.diff(fraction), 1.])
    vortex_rows = le[None] + vortex_fractions[:, None, None]*chord[None]
    camber_x = getattr(wing, 'camber_x', None)
    cambered = camber_x is not None
    if cambered:
        from .wing import camber_data
        camber_x, offsets = camber_data(camber_x, wing.camber_offset_m, width)
        sampled = np.array([[np.interp(fraction, camber_x, station[:, j])
                             for j in range(3)] for station in offsets]).transpose(2,0,1)
        rows += sampled
        # Vortex positions follow the same piecewise-bilinear surface as the
        # cell secant boundary condition, rather than a second smooth surface.
        vortex_rows = np.concatenate((.75*rows[:-1]+.25*rows[1:], rows[-1:]))
    reconstructed = cambered and settings.camber_normal == 'reconstructed_slope'
    if reconstructed:
        # Cell secants approximate slopes at cell midpoints. Reconstruct the
        # 3/4-cell boundary slope without changing ring or load geometry.
        # The trailing cell uses one-sided extrapolation from its neighbors.
        lengths = np.diff(fraction)
        midpoints = .5*(fraction[:-1]+fraction[1:])
        secants = np.diff(rows, axis=0)/lengths[:, None, None]
        reconstructed_slopes = np.empty_like(secants)
        for i in range(nc):
            left, right = (i, i+1) if i < nc-1 else (i-1, i)
            target = fraction[i]+.75*lengths[i]
            weight = (target-midpoints[left])/(midpoints[right]-midpoints[left])
            reconstructed_slopes[i] = (1-weight)*secants[left]+weight*secants[right]
    cells, points, normals, areas, area_normals, centers = [], [], [], [], [], []
    for i in range(nc):
        for a, b in wing.panel_edges:
            cell = [i*width+a, i*width+b, (i+1)*width+b, (i+1)*width+a]
            p = rows.reshape(-1, 3)[cell]
            # A bilinear strip is sampled at half span. This differs deliberately
            # from the legacy full-cosine approximate lifting-line CP convention.
            # Keep the original arithmetic for uniform spacing, including Nc1.
            cp_fraction = (i+.75)/nc if uniform else fraction[i]+.75*(fraction[i+1]-fraction[i])
            cp = (le[a]+le[b])/2 + cp_fraction*(chord[a]+chord[b])/2
            tangent = ((chord[a]+chord[b])/(2*nc) if uniform else
                       (chord[a]+chord[b])/2*(fraction[i+1]-fraction[i]))
            if cambered:
                cp = .125*(p[0]+p[1])+.375*(p[3]+p[2])
                tangent = (p[3]+p[2]-p[0]-p[1])/2
            span = p[1]+p[2]-p[0]-p[3]
            area_vector = np.cross(tangent, span/2)
            area = float(_norm(area_vector))
            span_cp = le[b]-le[a]+cp_fraction*(chord[b]-chord[a])
            if cambered:
                span_cp = .25*(p[1]-p[0])+.75*(p[2]-p[3])
            boundary_tangent = (.5*(reconstructed_slopes[i, a]+reconstructed_slopes[i, b])
                                if reconstructed else tangent)
            normal_cp = np.cross(boundary_tangent, span_cp)
            if area <= 0 or _norm(normal_cp) <= 0:
                raise ValueError('degenerate lifting-surface panel')
            cells.append(cell); points.append(cp); normals.append(normal_cp/_norm(normal_cp))
            area_normals.append(area_vector/area)
            areas.append(area); centers.append(p.mean(axis=0))
    return LiftingSurfaceMesh(rows.reshape(-1, 3), cells, vortex_rows, points,
        normals, areas, area_normals, centers, np.tile(np.arange(wing.panels), nc),
        np.repeat(np.arange(nc), wing.panels))


@dataclass(frozen=True)
class LiftingSurfaceSolution(WingSolution):
    lattice_circulation: np.ndarray
    lattice_force: np.ndarray
    lattice_couple: np.ndarray
    lattice_relative_velocity: np.ndarray
    lattice_induced_velocity: np.ndarray
    lattice_gamma_rate: np.ndarray
    lattice_boundary_residual_m_s: np.ndarray
    matrix_condition_number: float
    trailing_edge_jump_m2_s: np.ndarray
    mirror_preprojection_max_abs_m2_s: float | None = None
    unprojected_boundary_residual_m_s: float | None = None
    polar_coupling: PolarCouplingResult | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ('lattice_circulation', 'lattice_force', 'lattice_couple',
                     'lattice_relative_velocity', 'lattice_induced_velocity',
                     'lattice_gamma_rate', 'lattice_boundary_residual_m_s',
                     'trailing_edge_jump_m2_s'):
            object.__setattr__(self, name, owned(getattr(self, name)))

    @property
    def boundary_residual_m_s(self):
        return self.residual

    @property
    def boundary_residual_limit_m_s(self):
        return self.residual_limit

    residual_m_s = boundary_residual_m_s
    residual_limit_m_s = boundary_residual_limit_m_s


@dataclass(frozen=True)
class _LatticeState:
    time_s: float
    step: int
    wing: object
    solution: LiftingSurfaceSolution
    mesh: LiftingSurfaceMesh
    settings: LiftingSurfaceConfig
    config: WakeConfig
    structural_velocity: np.ndarray
    lattice_structural_velocity: np.ndarray

    def __post_init__(self):
        time = _real(self.time_s, 'state time')
        if time.shape:
            raise ValueError('state time must be scalar')
        count(self.step, 'state step')
        if not isinstance(self.solution, LiftingSurfaceSolution):
            raise ValueError('lifting-surface state requires its own solution')
        for name, n in [('structural_velocity', self.wing.panels),
                        ('lattice_structural_velocity', self.mesh.panels)]:
            object.__setattr__(self, name, owned(vector_field(getattr(self, name), n, name)))
        if self.solution.lattice_circulation.shape != (self.settings.chordwise_panels, self.wing.panels):
            raise ValueError('lattice solution and geometry disagree')


@dataclass(frozen=True)
class VlmState(_LatticeState):
    prescribed_row: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        validate_lifting_surface_wake(self.settings, self.config, False)
        object.__setattr__(self, 'prescribed_row', owned(self.prescribed_row))
        if self.prescribed_row.shape != (self.wing.node_count, 3):
            raise ValueError('prescribed wake row shape mismatch')

    @property
    def free_rows(self):
        return owned(np.empty((0, self.wing.node_count, 3)))

    @property
    def release_times_s(self):
        return owned(np.empty(0))

    @property
    def older_circulation(self):
        return owned(np.empty((0, self.wing.panels)))

    @property
    def cores(self):
        return None


@dataclass(frozen=True)
class VlmWakeState(_LatticeState):
    free_rows: np.ndarray
    release_times_s: np.ndarray
    wake_circulation: np.ndarray
    cores: object
    previous_lattice_circulation: np.ndarray | None = None
    wake_node_potential_m2_s: np.ndarray | None = None
    initial_gamma_rate: InitialGammaRate | None = None

    def __post_init__(self):
        super().__post_init__()
        validate_lifting_surface_wake(self.settings, self.config, True)
        if self.previous_lattice_circulation is not None:
            previous = owned(self.previous_lattice_circulation)
            if self.step == 0 or previous.shape != self.solution.lattice_circulation.shape:
                raise ValueError('invalid previous lattice circulation history')
            object.__setattr__(self, 'previous_lattice_circulation', previous)
        elif self.step and self.settings.gamma_derivative in ('bdf2', 'bdf2_consistent'):
            raise ValueError('BDF2 requires previous lattice circulation history')
        for name in ('free_rows', 'release_times_s', 'wake_circulation'):
            object.__setattr__(self, name, owned(getattr(self, name)))
        if (self.free_rows.shape != (self.step, self.wing.node_count, 3) or
                self.release_times_s.shape != (self.step,) or
                self.wake_circulation.shape != (self.step, self.wing.panels)):
            raise ValueError('material wake history shape mismatch')
        if np.any(self.release_times_s >= self.time_s) or np.any(np.diff(self.release_times_s) >= 0):
            raise ValueError('wake releases must be past and newest first')
        # Histories are not truncated: the oldest material release is the start time.
        check_initial_rate(self.initial_gamma_rate, self.settings,
            self.solution.lattice_circulation.shape,
            self.time_s if self.step == 0 else self.release_times_s[-1])
        if self.settings.wake_basis == 'linear_potential':
            if self.wake_node_potential_m2_s is None:
                raise ValueError('linear_potential requires explicit nodal material history')
            phi = owned(self.wake_node_potential_m2_s)
            if (phi.shape != (self.step+1, self.wing.panels) or
                    not np.array_equal(phi[0], self.solution.circulation) or
                    not np.array_equal(self.wake_circulation, .5*(phi[:-1]+phi[1:]))):
                raise ValueError('nodal potential TE trace or derived cell mean mismatch')
            object.__setattr__(self, 'wake_node_potential_m2_s', phi)
        elif self.wake_node_potential_m2_s is not None:
            raise ValueError('nodal potential history requires linear_potential')
        check_cores(self.cores, self.config)
        if self.cores is not None and (self.cores.span.shape != (self.step, self.wing.panels) or
                                      self.cores.stream.shape != (self.step, self.wing.node_count)):
            raise ValueError('lattice wake core history shape mismatch')

    @property
    def older_circulation(self):
        """Cell strengths, or derived cell means for an explicit nodal potential."""
        return self.wake_circulation


def validate_lifting_surface_wake(settings, config, transient):
    """Reject unsupported combinations instead of silently changing physics."""
    if getattr(config, 'wake_core_chord_fraction', None) is not None:
        if not transient or settings.wake_basis != 'constant':
            raise ValueError('local material birth cores require a constant-basis released wake')
        if settings.initial_cap != 'legacy_zero':
            raise ValueError('local material birth cores currently require the legacy zero-core initial cap')
    if settings.bound_convection_core == 'all_bound_chord':
        if not transient or not config.free_wake or int(config.core_model) not in (1, 2, 3, 5):
            raise ValueError('all_bound_chord requires free convection and a positive additive core')
    if settings.bound_convection_core == 'te_incident_birth':
        if not transient or not config.free_wake:
            raise ValueError('te_incident_birth requires free material-wake convection')
        if int(config.core_model) not in (1, 2, 3) or config.core_radius_m <= 0:
            raise ValueError('te_incident_birth requires a positive additive birth core')
    if settings.initial_cap == 'wake_core':
        if not transient:
            raise ValueError('wake_core initial cap requires a released lattice wake')
        if int(config.core_model) == 4:
            raise ValueError('wake_core initial cap does not support nonadditive OFFSET cores')
    if settings.wake_basis == 'linear_potential':
        if not transient:
            raise ValueError('linear_potential requires a transient material wake')
        if int(config.core_model) == 4:
            raise ValueError('linear_potential does not support nonadditive OFFSET segment cores')


def linear_wake_convection_contract(settings):
    """Version the stage field independently of the saved load/probe network."""
    if settings.bound_convection_core == 'all_bound_chord':
        return 'attached_frozen_nodes_all_bound_chord_v3'
    return ('attached_frozen_nodes_te_incident_birth_v2'
            if settings.bound_convection_core == 'te_incident_birth'
            else 'attached_frozen_nodes_v1')


def _sample(field, points, time, name):
    """Spatial fields must be callable; CP-only arrays cannot define edge loads."""
    value = field(points.copy(), float(time)) if callable(field) else field
    raw = _real(value, name)
    if not callable(field) and raw.shape != (3,):
        raise ValueError(f'{name} must be a uniform (3,) vector or spatial callable')
    return vector_field(raw, len(points), name)


def _initial_cap_enabled(settings, config, free_rows):
    """A zero-history closing cap uses the same additive birth kernel as release."""
    return (settings is not None and settings.initial_cap == 'wake_core'
            and int(config.core_model) != 0 and not len(free_rows))


def _radii(wing, edges, nc, cores, initial, *, initial_cap=False):
    # Bound edges are inviscid except the explicit zero-history birth cap.
    # Never regularize the remaining narrow chordwise panel or tip operator.
    radius = np.zeros(len(edges), dtype=float)
    width = wing.node_count
    lookup = {tuple(sorted(pair)): j for j, pair in enumerate(wing.panel_edges)}
    for k, (first, last) in enumerate(edges):
        ra, a = divmod(int(first), width); rb, b = divmod(int(last), width)
        if initial_cap and ra == rb == nc:
            radius[k] = initial
        elif ra == rb and ra > nc:
            radius[k] = initial if cores is None else cores.span[ra-nc-1, lookup[tuple(sorted((a, b)))]]
        elif ra != rb and min(ra, rb) >= nc:
            radius[k] = initial if cores is None else cores.stream[min(ra, rb)-nc, a]
    return radius


def _network(wing, mesh, gamma, free_rows, wake_gamma):
    rows = np.concatenate((mesh.vortex_rows, free_rows), axis=0)
    strengths = np.vstack((gamma, wake_gamma))
    return pack_rings(rows, strengths, wing.panel_edges)


def lifting_surface_network(state):
    """Exact closed graph used by this solver, including the prescribed far cap."""
    if type(state) is VlmState:
        rows = state.prescribed_row[None]
        gamma = state.solution.circulation[None]
    elif type(state) is VlmWakeState:
        if state.settings.wake_basis == 'linear_potential':
            return potential_network(state.wing, state.mesh,
                state.solution.lattice_circulation, state.free_rows,
                state.wake_node_potential_m2_s, state.settings.wake_quadrature_order,
                state.settings.wake_quadrature)
        rows, gamma = state.free_rows, state.wake_circulation
    else:
        raise ValueError('state is not a lifting-surface state')
    return _network(state.wing, state.mesh, state.solution.lattice_circulation, rows, gamma)


def lifting_surface_edge_radii(state, config=None):
    config = state.config if config is None else config
    if config != state.config:
        raise ValueError('state and wake configuration differ; start a new run')
    if state.settings.wake_basis == 'linear_potential':
        edges = lifting_surface_network(state)[1]
        if int(config.core_model) == 0:
            return np.zeros(len(edges))
        return potential_edge_radii(state.wing, state.mesh, state.free_rows, edges,
            state.cores, config.core_radius_m, state.settings.wake_quadrature_order,
            state.settings.wake_quadrature,
            initial_cap=_initial_cap_enabled(state.settings, config, state.free_rows))
    return _radii(state.wing, lifting_surface_network(state)[1],
                  state.settings.chordwise_panels, state.cores, config.core_radius_m,
                  initial_cap=_initial_cap_enabled(state.settings, config,
                      state.free_rows if type(state) is VlmWakeState else state.prescribed_row[None]))


def lifting_surface_induced_velocity(state, points, config=None):
    """Read-only probes; observation on a filament raises, even with a core."""
    config = state.config if config is None else config
    nodes, edges, gamma = lifting_surface_network(state)
    return segments_velocity(points, nodes, edges, gamma, model=config.core_model,
        core_radius=lifting_surface_edge_radii(state, config))


def _operator(wing, mesh, config, prescribed_row=None, *, initial_cap=False):
    points, nc = mesh.collocation_points, len(mesh.vortex_rows)-1
    influence = np.empty((len(points), len(points), 3))
    for i in range(nc):
        for j, (a, b) in enumerate(wing.panel_edges):
            corners = [mesh.vortex_rows[i, a], mesh.vortex_rows[i, b],
                       mesh.vortex_rows[i+1, b], mesh.vortex_rows[i+1, a]]
            value = ring_velocity(corners, points, 1., model=config.core_model,
                                  core_radius=0.)
            if initial_cap and i == nc-1:
                cap = np.array([corners[2], corners[3]])
                value += (segments_velocity(points, cap, [[0, 1]], 1., model=config.core_model,
                                            core_radius=config.core_radius_m)
                          - segments_velocity(points, cap, [[0, 1]], 1., model=config.core_model,
                                              core_radius=0.))
            if prescribed_row is not None and i == nc-1:
                value += segments_velocity(points, [wing.trailing_edge[a], wing.trailing_edge[b],
                    prescribed_row[b], prescribed_row[a]], [[0, 1], [1, 2], [2, 3], [3, 0]], 1.,
                    model=config.core_model, core_radius=[0., *([config.core_radius_m]*3)])
            influence[:, i*wing.panels+j] = value
    return influence


def lifting_surface_influence(wing, mesh, config, prescribed_row=None, *, initial_cap='legacy_zero'):
    """Unit-ring vector operator, also usable for a circulation-only comparison.

    A valid boundary solution does not ensure nonsingular load-point induction;
    retain that distinction when independently comparing source circulation.
    The optional initial cap represents a zero-history released wake, not a
    regularized bound sheet or a prescribed straight-wake setting.
    """
    if initial_cap not in ('legacy_zero', 'wake_core'):
        raise ValueError('initial cap must be legacy_zero or wake_core')
    if (getattr(config, 'wake_core_chord_fraction', None) is not None
            and (prescribed_row is not None or initial_cap != 'legacy_zero')):
        raise ValueError('local birth cores require released material history and the legacy initial cap')
    if initial_cap == 'wake_core' and (prescribed_row is not None or int(config.core_model) == 4):
        raise ValueError('wake_core cap requires a zero-history released wake and an additive core')
    return _operator(wing, mesh, config, prescribed_row,
        initial_cap=initial_cap == 'wake_core' and int(config.core_model) != 0)


def _wake_induction(wing, mesh, free_rows, wake_gamma, points, cores, config):
    if not len(free_rows):
        return np.zeros_like(points)
    rows = np.concatenate(([wing.trailing_edge], free_rows))
    nodes, edges, gamma = pack_rings(rows, wake_gamma, wing.panel_edges)
    radius = _radii(wing, edges+(len(mesh.vortex_rows)-1)*wing.node_count,
                    len(mesh.vortex_rows)-1, cores, config.core_radius_m)
    return segments_velocity(points, nodes, edges, gamma, model=config.core_model,
                             core_radius=radius)


def _linear_induction(wing, mesh, free_rows, phi, settings, points, cores, config, gamma=None):
    """The same regularized linear-potential field for unknown, history and loads."""
    include_bound = gamma is not None
    network = (potential_network(wing, mesh, gamma, free_rows, phi,
        settings.wake_quadrature_order, settings.wake_quadrature) if include_bound else
        wake_potential_network(wing, mesh, free_rows, phi,
            settings.wake_quadrature_order, settings.wake_quadrature))
    radius = (np.zeros(len(network[1])) if int(config.core_model) == 0 else
        potential_edge_radii(wing, mesh, free_rows, network[1], cores, config.core_radius_m,
            settings.wake_quadrature_order, settings.wake_quadrature, include_bound=include_bound,
            initial_cap=include_bound and _initial_cap_enabled(settings, config, free_rows)))
    return segments_velocity(points, *network, model=config.core_model, core_radius=radius)


def _loads(wing, mesh, gamma, free_rows, wake_gamma, cores, config, inflow,
           structural_velocity, time, rate, nodal_phi=None, settings=None):
    """All unique bound edges, TE traction cut, exact force/moment reductions."""
    nc, width = gamma.shape[0], wing.node_count
    # Keep all geometric edges even at zero strength, so loads remain a smooth
    # function of Gamma and a callback sees a stable evaluation-point set.
    ownership = {}
    strength = {}
    for i in range(nc):
        for j, (a, b) in enumerate(wing.panel_edges):
            vertices = [i*width+a, i*width+b, (i+1)*width+b, (i+1)*width+a]
            for p, q in zip(vertices, vertices[1:]+vertices[:1]):
                if p//width == q//width == nc:
                    continue
                key = tuple(sorted((p, q)))
                ownership.setdefault(key, []).append(i*wing.panels+j)
                strength[key] = strength.get(key, 0.) + (gamma[i, j] if p < q else -gamma[i, j])
    edges = np.asarray(list(ownership), dtype=np.int64)
    bound_nodes = mesh.vortex_rows.reshape(-1, 3)
    points = bound_nodes[edges].mean(axis=1)
    if nodal_phi is None:
        nodes, graph_edges, graph_gamma = _network(wing, mesh, gamma, free_rows, wake_gamma)
        radius = _radii(wing, graph_edges, nc, cores, config.core_radius_m,
            initial_cap=_initial_cap_enabled(settings, config, free_rows))
    else:
        nodes, graph_edges, graph_gamma = potential_network(wing, mesh, gamma,
            free_rows, nodal_phi, settings.wake_quadrature_order, settings.wake_quadrature)
        radius = (np.zeros(len(graph_edges)) if int(config.core_model) == 0 else
            potential_edge_radii(wing, mesh, free_rows, graph_edges, cores, config.core_radius_m,
                settings.wake_quadrature_order, settings.wake_quadrature,
                initial_cap=_initial_cap_enabled(settings, config, free_rows)))
    # Exclude the receiving bound edge itself, also if a duplicated physical
    # root edge has distinct topology IDs. Other filament crossings still fail.
    source_lookup = {}
    for k, (a, b) in enumerate(graph_edges):
        key = tuple(sorted((tuple(nodes[a]), tuple(nodes[b]))))
        source_lookup.setdefault(key, []).append(k)
    excluded = []
    for k, (a, b) in enumerate(edges):
        key = tuple(sorted((tuple(bound_nodes[a]), tuple(bound_nodes[b]))))
        excluded.extend((k, source) for source in source_lookup.get(key, ()))
    induced = segments_velocity(points, nodes, graph_edges, graph_gamma,
        model=config.core_model, core_radius=radius,
        exclusions=np.asarray(excluded, dtype=np.int64).reshape(-1, 2))
    velocity = _sample(inflow, points, time, 'inflow') - _sample(
        structural_velocity, points, time, 'structural velocity') + induced
    dl = bound_nodes[edges[:, 1]]-bound_nodes[edges[:, 0]]
    edge_force = config.density_kg_m3*np.asarray(list(strength.values()))[:, None]*np.cross(velocity, dl)
    force, couple = np.zeros((mesh.panels, 3)), np.zeros((mesh.panels, 3))
    for k, key in enumerate(ownership):
        owners = ownership[key]
        for panel in owners:
            share = edge_force[k]/len(owners)
            force[panel] += share
            couple[panel] += np.cross(points[k]-mesh.collocation_points[panel], share)
    added = config.density_kg_m3*rate.reshape(-1, 1)*mesh.area[:, None]*mesh.area_normals
    force += added
    couple += np.cross(mesh.centroids-mesh.collocation_points, added)
    return force.reshape(nc, wing.panels, 3), couple.reshape(nc, wing.panels, 3)


def _steady_mirror_permutation(wing, mesh, prescribed_row, inflow, structural_velocity):
    """Check self-mirrored y=0 steady surfaces before pairing their rings.

    The coordinate allowance only covers float64 representation of a declared
    symmetric geometry. It is not an aerodynamic residual or vortex-strength
    cutoff. The caller still checks the original nonpenetration tolerance.
    """
    wind = _real(inflow, 'symmetric steady inflow')
    body = _real(structural_velocity, 'symmetric steady structural velocity')
    if wind.shape != (3,) or body.shape != (3,) or wind[1] != 0 or body[1] != 0:
        raise ValueError('steady mirror projection requires constant y-symmetric inflow and body velocity')
    # A Wing and each member of a WingAssembly form an ordered chain with
    # disjoint node IDs. Recover the same blocks from saved topology on replay.
    edges = np.asarray(wing.panel_edges)
    boundaries = np.r_[0, 1+np.flatnonzero(edges[1:, 0] != edges[:-1, 1]), len(edges)]
    nodes, panels = [], []
    next_node = 0
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        block = edges[first:last]
        n = len(block)
        expected = np.column_stack((np.arange(next_node, next_node+n),
                                    np.arange(next_node+1, next_node+n+1)))
        if not np.array_equal(block, expected):
            raise ValueError('steady mirror projection requires ordered chain surfaces')
        nodes.extend(np.arange(next_node+n, next_node-1, -1))
        panels.extend(np.arange(last-1, first-1, -1))
        next_node += n+1
    if next_node != wing.node_count:
        raise ValueError('steady mirror projection topology does not cover wing nodes')
    node_permutation = np.asarray(nodes, dtype=int)
    panel_permutation = np.asarray(panels, dtype=int)
    mirror = np.array([1., -1., 1.])
    local_geometry = (wing.leading_edge, wing.trailing_edge,
                      mesh.vertices.reshape(-1, wing.node_count, 3),
                      mesh.vortex_rows)
    # A very long prescribed leg must not weaken the local mirror check.
    scale = max(1., *(float(np.max(np.abs(points))) for points in local_geometry))
    tolerance_m = 1e-12*scale
    for points in local_geometry:
        if np.max(np.abs(points - points[..., node_permutation, :]*mirror)) > tolerance_m:
            raise ValueError('steady mirror projection requires y=0 mirrored geometry and wake')
    # The far coordinates can lose a local ulp when a long leg is added.
    # Keep the lateral symmetry check local, while allowing only roundoff in
    # the far x/z coordinates. Every node must use the same downstream vector.
    downstream = prescribed_row-wing.trailing_edge
    if np.max(np.abs(downstream[:, 1])) > tolerance_m:
        raise ValueError('steady mirror projection requires a y=0 prescribed wake')
    far_tolerance_m = max(tolerance_m, 4.*float(np.max(
        np.spacing(np.maximum(1., np.abs(prescribed_row[:, [0, 2]]))))))
    if np.max(np.abs(downstream-downstream[0])) > far_tolerance_m:
        raise ValueError('steady mirror projection requires equal prescribed wake legs')
    difference = prescribed_row-prescribed_row[node_permutation]*mirror
    if (np.max(np.abs(difference[:, [0, 2]])) > far_tolerance_m or
            np.max(np.abs(difference[:, 1])) > tolerance_m):
        raise ValueError('steady mirror projection requires y=0 mirrored wake')
    return np.concatenate([i*wing.panels + panel_permutation
                           for i in range(mesh.panels//wing.panels)])


def _solve(wing, mesh, settings, config, inflow, structural_velocity, time,
           free_rows, wake_gamma, cores, prescribed_row=None, previous=None,
           previous_previous=None, polar=None, previous_closure=None, nodal_history=None,
           initial_gamma_rate=None):
    nc, n = settings.chordwise_panels, wing.panels
    points = mesh.collocation_points
    body = _sample(structural_velocity, points, time, 'structural velocity')
    ambient = _sample(inflow, points, time, 'inflow')-body
    influence = _operator(wing, mesh, config, prescribed_row,
        initial_cap=prescribed_row is None and _initial_cap_enabled(settings, config, free_rows))
    bound_influence = influence
    linear = settings.wake_basis == 'linear_potential'
    nodal_phi = None
    theta = {'previous':0., 'current':1., 'trapezoidal':.5}[settings.wake_release]
    coupled = not linear and prescribed_row is None and len(free_rows) and theta != 0
    if linear and len(free_rows):
        if nodal_history is None or np.shape(nodal_history) != (len(free_rows), n):
            raise ValueError('linear wake solve requires accepted nodal history')
        influence = influence.copy()
        for j in range(n):
            unit = np.zeros((2, n)); unit[0, j] = 1.
            influence[:, (nc-1)*n+j] += _linear_induction(wing, mesh, free_rows[:1],
                unit, settings, points, cores, config)
    if coupled:
        # The first material row depends linearly on the unknown TE strength.
        # Use the same full vector kernel/radii as the final physical network.
        influence = influence.copy()
        for j in range(n):
            unit = np.zeros((1, n)); unit[0, j] = 1.
            influence[:, (nc-1)*n+j] += theta*_wake_induction(
                wing, mesh, free_rows[:1], unit, points, cores, config)
        wake_gamma = wake_gamma.copy()
        wake_gamma[0] *= 1-theta
    if linear:
        known = (_linear_induction(wing, mesh, free_rows,
            np.vstack((np.zeros((1, n)), nodal_history)), settings, points, cores, config)
            if len(free_rows) else np.zeros_like(points))
    else:
        known = (np.zeros_like(points) if prescribed_row is not None else
                 _wake_induction(wing, mesh, free_rows, wake_gamma, points, cores, config))
    closure = None
    boundary_normals = mesh.normals
    if settings.polar_coupling == 'grid_section':
        reference = SectionReferences(wing, mesh)
        strip_flow = _sample(inflow, reference.quarter, time, 'inflow')-_sample(
            structural_velocity, reference.quarter, time, 'structural velocity')
        problem = ClosureProblem(wing, mesh, influence, ambient+known,
                                 strip_flow, polar, settings)
        seed = (np.full(n, np.deg2rad(settings.polar_initial_virtual_incidence_deg))
                if previous_closure is None else previous_closure.delta_rad)
        closure, condition = problem.solve(seed)
        gamma, boundary_normals = closure.gamma, closure.virtual_normals
    else:
        matrix = np.einsum('ijk,ik->ij', influence, mesh.normals)
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition) or (settings.max_condition_number is not None
                                         and condition > settings.max_condition_number):
            raise ValueError('lifting-surface matrix condition exceeds declared limit')
        rhs = -np.sum((ambient+known)*mesh.normals, axis=1)
        try:
            gamma = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError('singular lifting-surface nonpenetration matrix') from error
    mirror_difference = unprojected_residual = None
    if settings.steady_mirror_projection:
        if prescribed_row is None:
            raise ValueError('steady mirror projection requires a prescribed straight wake')
        permutation = _steady_mirror_permutation(wing, mesh, prescribed_row,
                                                  inflow, structural_velocity)
        mirror_difference = float(np.max(np.abs(gamma-gamma[permutation])))
        scale = float(np.max(np.abs(gamma)))
        mirror_limit = settings.mirror_projection_atol_m2_s + settings.mirror_projection_rtol*scale
        if mirror_difference > mirror_limit:
            raise ValueError('steady mirror circulation asymmetry exceeds declared limit')
        unprojected_residual = float(np.max(np.abs(matrix@gamma-rhs)))
        gamma = .5*(gamma + gamma[permutation])
    induced = known+np.einsum('ijk,j->ik', influence, gamma)
    if linear:
        nodal_phi = np.vstack((gamma[-n:], nodal_history if len(free_rows) else np.empty((0, n))))
        induced = _linear_induction(wing, mesh, free_rows, nodal_phi, settings,
            points, cores, config, gamma.reshape(nc, n))
    if coupled:
        wake_gamma[0] += theta*gamma[-n:]
        # Check physical nonpenetration, not just the assembled matrix residual.
        induced = _wake_induction(wing, mesh, free_rows, wake_gamma, points, cores, config)
        induced += np.einsum('ijk,j->ik', bound_influence, gamma)
    velocity = ambient+induced
    signed_residual = np.sum(velocity*boundary_normals, axis=1)
    if closure is not None:
        closure = replace(closure, geometric_residual_m_s=np.sum(velocity*mesh.normals, axis=1))
    residual = float(np.max(np.abs(signed_residual)))
    limit = settings.linear_residual_tolerance_m_s
    if not np.isfinite(gamma).all() or residual > limit:
        raise ValueError(f'lifting-surface normal-velocity residual {residual:g} > {limit:g} m/s')
    gamma = gamma.reshape(nc, n)
    rate = np.zeros_like(gamma) if previous is None else (gamma-previous)/config.dt_s
    if settings.gamma_derivative == 'bdf2_consistent':
        if not isinstance(initial_gamma_rate, InitialGammaRate) or initial_gamma_rate.rate_m2_s2.shape != gamma.shape:
            raise ValueError('consistent BDF2 solve requires matching explicit initial Gamma rate')
        if previous is None:
            rate = initial_gamma_rate.rate_m2_s2
        elif previous_previous is None:
            rate = 2*(gamma-previous)/config.dt_s-initial_gamma_rate.rate_m2_s2
    if previous_previous is not None and settings.gamma_derivative in ('bdf2', 'bdf2_consistent'):
        rate = (3*gamma-4*previous+previous_previous)/(2*config.dt_s)
    if prescribed_row is not None:
        free_rows, wake_gamma = prescribed_row[None], gamma[-1:]
    force, couple = _loads(wing, mesh, gamma, free_rows, wake_gamma, cores, config,
                          inflow, structural_velocity, time, rate, nodal_phi, settings)
    if closure is not None:
        added_force, added_couple = closure.corrections(config.density_kg_m3)
        weights = mesh.area.reshape(nc, n)/mesh.area.reshape(nc, n).sum(axis=0)
        correction_force = weights[:,:,None]*added_force
        correction_couple = weights[:,:,None]*added_couple+np.cross(
            closure.reference_points_m-mesh.collocation_points.reshape(nc,n,3), correction_force)
        closure = replace(closure, potential_force_N=force, potential_couple_N_m=couple,
                          correction_force_N=correction_force, correction_couple_N_m=correction_couple)
        force, couple = force+correction_force, couple+correction_couple
    strip_force = force.sum(axis=0)
    strip_couple = (couple+np.cross(mesh.collocation_points.reshape(nc, n, 3)-wing.cp,
                                   force)).sum(axis=0)
    # Strip coefficient axes use prescribed relative flow at the original strip
    # CP. They are derived loads, not evaluated airfoil-polar coefficients.
    strip_ambient = _sample(inflow, wing.cp, time, 'inflow')-_sample(
        structural_velocity, wing.cp, time, 'structural velocity')
    speed = _norm(strip_ambient)
    direction = np.divide(strip_ambient, speed[:, None], out=np.zeros_like(strip_ambient),
                          where=speed[:, None] > 0)
    lift = wing.normal-np.sum(wing.normal*direction, axis=1)[:, None]*direction
    lift_norm = _norm(lift)
    lift = np.divide(lift, lift_norm[:, None], out=np.zeros_like(lift), where=lift_norm[:, None] > 0)
    qarea = .5*config.density_kg_m3*speed**2*wing.area
    divide = lambda value: np.divide(value, qarea, out=np.zeros(n), where=qarea > 0)
    cl = divide(np.sum(strip_force*lift, axis=1))
    cd = divide(np.sum(strip_force*direction, axis=1))
    cm = divide(np.sum(strip_couple*wing.orthogonal, axis=1))/wing.chord_cp
    alpha = np.arctan2(np.sum(strip_ambient*wing.normal, axis=1),
                       np.sum(strip_ambient*wing.tangent, axis=1))
    weights = mesh.area.reshape(nc, n)/mesh.area.reshape(nc, n).sum(axis=0)
    strip_induced = np.sum(induced.reshape(nc, n, 3)*weights[:, :, None], axis=0)
    jump = (np.zeros(n) if linear and len(free_rows) else
            gamma[-1]-(wake_gamma[0] if len(wake_gamma) else np.zeros(n)))
    solution = LiftingSurfaceSolution(gamma[-1], strip_ambient+strip_induced,
        strip_induced, alpha, wing.chord_cp*speed/config.kinematic_viscosity_m2_s,
        cl, cd, cm, strip_force, strip_couple, 1 if closure is None else closure.iterations+1, residual, limit, gamma, force,
        couple, velocity.reshape(nc, n, 3), induced.reshape(nc, n, 3), rate,
        signed_residual.reshape(nc, n), condition, jump,
        mirror_difference, unprojected_residual, closure)
    return solution, body


def start_lifting_surface(wing, config, inflow, *, settings=LiftingSurfaceConfig(),
                         free_wake=False, time_s=0., structural_velocity=(0., 0., 0.), polar=None,
                         initial_gamma_rate=None):
    if not isinstance(config, WakeConfig) or not isinstance(free_wake, bool):
        raise ValueError('WakeConfig and boolean wake selection required')
    validate_lifting_surface_wake(settings, config, free_wake)
    if free_wake and settings.steady_mirror_projection:
        raise ValueError('steady mirror projection is unavailable for free wakes')
    time = _real(time_s, 'initial time')
    if time.shape:
        raise ValueError('initial time must be scalar')
    time = time.item()
    mesh = build_lifting_surface_mesh(wing, settings)
    check_initial_rate(initial_gamma_rate, settings, (settings.chordwise_panels, wing.panels), time)
    rows, history = np.empty((0, wing.node_count, 3)), np.empty((0, wing.panels))
    cores = initial_cores(wing, config) if free_wake else None
    downstream = None
    if not free_wake:
        if settings.wake_release != 'previous' or settings.gamma_derivative != 'backward_euler':
            raise ValueError('transient time settings do not apply to straight VLM')
        if config.core_evolution != 'constant':
            raise ValueError('material core evolution applies only to a released wake')
        wind = _sample(inflow, mesh.collocation_points, time, 'inflow')-_sample(
            structural_velocity, mesh.collocation_points, time, 'structural velocity')
        mean = wind.mean(axis=0)
        if _norm(mean) == 0:
            raise ValueError('straight wake direction requires nonzero mean relative inflow')
        direction = mean/_norm(mean) if settings.wake_direction is None else np.asarray(settings.wake_direction)
        downstream = wing.trailing_edge+settings.wake_length_m*direction
        from .wake import check_straight_surface
        check_straight_surface(wing, downstream, config, time)
    solution, body = _solve(wing, mesh, settings, config, inflow, structural_velocity,
                            time, rows, history, cores, downstream, polar=polar,
                            initial_gamma_rate=initial_gamma_rate)
    common = (time, 0, wing, solution, mesh, settings, config,
              _sample(structural_velocity, wing.cp, time, 'structural velocity'), body)
    if free_wake:
        phi = solution.circulation[None] if settings.wake_basis == 'linear_potential' else None
        return VlmWakeState(*common, rows, np.empty(0), history, cores,
                            wake_node_potential_m2_s=phi, initial_gamma_rate=initial_gamma_rate)
    return VlmState(*common, downstream)


def _linear_convection_network(state, config, moving, fraction, next_mesh=None):
    """Frozen-potential stage field with the full physical TE on the body.

    Moving rows start with the released old TE. A new attached interval joins
    the current body TE to that row with the same accepted potential at both
    endpoints. Only its fully coincident initial state is collapsed. Existing
    material cores stay frozen; the new interval uses the initial core. The
    returned canonical endpoints are for explicit endpoint association only,
    not changes to induction coordinates or accepted archive geometry.
    """
    from types import SimpleNamespace
    moving = np.asarray(moving).reshape(state.step+1, state.wing.node_count, 3)
    end = state.mesh if next_mesh is None else next_mesh
    vortex_rows = state.mesh.vortex_rows+fraction*(end.vortex_rows-state.mesh.vortex_rows)
    mesh = SimpleNamespace(vortex_rows=vortex_rows,
        collocation_points=state.mesh.collocation_points+fraction*(end.collocation_points-state.mesh.collocation_points),
        strip_index=state.mesh.strip_index)
    leading = state.mesh.vertices[:state.wing.node_count]+fraction*(
        end.vertices[:state.wing.node_count]-state.mesh.vertices[:state.wing.node_count])
    wing = SimpleNamespace(leading_edge=leading, trailing_edge=vortex_rows[-1], panel_edges=state.wing.panel_edges,
                           panels=state.wing.panels, node_count=state.wing.node_count)
    gamma = state.solution.lattice_circulation
    if np.array_equal(wing.trailing_edge, moving[0]):
        free, phi, cores = moving[1:], state.wake_node_potential_m2_s, state.cores
    else:
        free = moving
        phi = np.vstack((gamma[-1], state.wake_node_potential_m2_s))
        cores = None if state.cores is None else WakeCores(
            np.vstack((np.full((1, wing.panels), config.core_radius_m), state.cores.span)),
            np.vstack((np.full((1, wing.node_count), config.core_radius_m), state.cores.stream)),
            state.cores.policy, state.cores.initial_radius, state.cores.diffusion_rate)
    network = potential_network(wing, mesh, gamma, free, phi,
                                state.settings.wake_quadrature_order, state.settings.wake_quadrature)
    radii = (np.zeros(len(network[1])) if int(config.core_model) == 0 else
        potential_edge_radii(wing, mesh, free, network[1], cores, config.core_radius_m,
                             state.settings.wake_quadrature_order, state.settings.wake_quadrature,
                             initial_cap=_initial_cap_enabled(state.settings, config, free)))
    physical = np.concatenate((mesh.vortex_rows[-1:], free))
    distance = near_junction_distance(wing, mesh) if state.settings.wake_quadrature == 'te_graded' else None
    coordinate = np.r_[np.arange(1-len(mesh.vortex_rows), 0),
                       material_row_coordinates(physical, state.settings.wake_quadrature_order, distance)]
    canonical = network[0].copy().reshape(-1, wing.node_count, 3)
    for row, s in enumerate(coordinate):
        if s >= 0 and s == int(s):
            canonical[row] = physical[int(s)]
    if state.settings.bound_convection_core == 'te_incident_birth':
        # Only the convection field changes. Exact physical positions include
        # duplicate roots; nonincident bound edges and material cores stay put.
        bound_count = len(mesh.vortex_rows)*wing.node_count
        te_locations = set(map(tuple, wing.trailing_edge))
        on_te = np.fromiter((tuple(p) in te_locations for p in canonical.reshape(-1, 3)[:bound_count]),
                            dtype=bool, count=bound_count)
        bound_edges = np.flatnonzero(np.all(network[1] < bound_count, axis=1))
        incident = bound_edges[np.any(on_te[network[1][bound_edges]], axis=1)]
        incident = incident[radii[incident] == 0]
        radii[incident] = config.core_radius_m
    elif state.settings.bound_convection_core == 'all_bound_chord':
        radii = _bound_convection_radii(wing, network[1], radii,
            len(mesh.vortex_rows)*wing.node_count, state.settings, config)
    return network, radii, canonical.reshape(-1, 3)


def _bound_convection_radii(wing, edges, load_radii, bound_count, settings, config):
    """Regularize only the convection field, with per-edge local chord scale."""
    if settings.bound_convection_core != 'all_bound_chord':
        return load_radii
    result = load_radii.copy()
    selected = np.all(edges < bound_count, axis=1)
    te_row = bound_count//wing.node_count-1
    te_span = np.all(edges//wing.node_count == te_row, axis=1)
    selected &= ~te_span
    chord = _norm(wing.trailing_edge-wing.leading_edge)
    local = edges[selected] % wing.node_count
    result[selected] = settings.bound_convection_chord_fraction * chord[local].mean(axis=1)
    # The old TE span already fluid-convects during this step. It belongs to
    # the wake birth model, not to the fixed attached-edge regularization.
    if getattr(config, 'wake_core_chord_fraction', None) is None:
        result[te_span] = config.core_radius_m
    else:
        from .cores import birth_radii
        span_birth, _ = birth_radii(wing, config)
        lookup = {tuple(sorted(pair)): i for i, pair in enumerate(wing.panel_edges)}
        for index in np.flatnonzero(te_span):
            result[index] = span_birth[lookup[tuple(sorted(edges[index] % wing.node_count))]]
    return result


def _endpoint_exclusions(points, edges, canonical):
    """Exact source-endpoint association, including coincident physical nodes."""
    locations = {}
    for j, point in enumerate(points):
        locations.setdefault(tuple(point), []).append(j)
    exclusions = []
    for k, pair in enumerate(edges):
        for index in pair:
            exclusions.extend((j, k) for j in locations.get(tuple(canonical[index]), ()))
    return np.asarray(exclusions, dtype=np.int64).reshape(-1, 2)


def _convect(state, config, inflow, next_wing=None):
    nc, width = state.settings.chordwise_panels, state.wing.node_count
    from .wake import make_surface_guard
    guard = make_surface_guard(state.wing, config, next_wing)

    def checked(candidate, previous, label, time):
        if guard is not None:
            guard.check_rows(previous.reshape(-1, width, 3), candidate.reshape(-1, width, 3),
                             state.wing, stage=label, time_s=time)
        return candidate
    if state.settings.wake_basis == 'linear_potential':
        # Advance physical material nodes only; quadrature vertices are rebuilt.
        moving = np.concatenate((state.mesh.vortex_rows[-1:], state.free_rows)).reshape(-1, 3).copy()
        next_mesh = state.mesh if next_wing is None else build_lifting_surface_mesh(next_wing, state.settings)
        def rhs(points, time):
            velocity = _sample(inflow, points, time, 'inflow').copy()
            if config.free_wake:
                fraction = (time-state.time_s)/config.dt_s
                network, radii, canonical = _linear_convection_network(state, config, points, fraction, next_mesh)
                velocity += segments_velocity(points, *network, model=config.core_model, core_radius=radii,
                    exclusions=_endpoint_exclusions(points, network[1], canonical))
            return velocity
        dt = config.dt_s/config.geometry_substeps
        for substep in range(config.geometry_substeps):
            time = state.time_s+substep*dt
            k1 = rhs(moving, time)
            if config.convection_integrator == 'euler':
                candidate = moving+dt*k1
            elif config.convection_integrator == 'heun':
                k2 = rhs(checked(moving+dt*k1, moving, 'heun_predictor', time+dt), time+dt)
                candidate = moving+dt*(k1+k2)/2
            else:
                k2 = rhs(checked(moving+dt*k1/2, moving, 'rk4_stage2', time+dt/2), time+dt/2)
                k3 = rhs(checked(moving+dt*k2/2, moving, 'rk4_stage3', time+dt/2), time+dt/2)
                k4 = rhs(checked(moving+dt*k3, moving, 'rk4_stage4', time+dt), time+dt)
                candidate = moving+dt*(k1+2*k2+2*k3+k4)/6
            moving = checked(candidate, moving, 'accepted_geometry', time+dt)
        return moving.reshape(state.step+1, width, 3)
    nodes, edges, gamma = lifting_surface_network(state)
    first = nc*width
    fixed, moving = nodes[:first], nodes[first:].copy()
    next_fixed = (fixed if next_wing is None else
        build_lifting_surface_mesh(next_wing, state.settings).vortex_rows.reshape(-1, 3)[:first])
    radii = _bound_convection_radii(state.wing, edges,
        lifting_surface_edge_radii(state, config), (nc+1)*width, state.settings, config)

    def rhs(points, time):
        # Endpoint geometry supplies a declared linear-in-time bound trajectory.
        # Keeping it stationary in a translating frame breaks Heun/RK4 Galilean
        # covariance even when the user supplies an exact translated next_wing.
        fraction = (time-state.time_s)/config.dt_s
        stage_bound = fixed+fraction*(next_fixed-fixed)
        stage = np.concatenate((stage_bound, points))
        velocity = _sample(inflow, points, time, 'inflow').copy()
        if config.free_wake and len(edges):
            locations = {}
            for j, point in enumerate(points):
                locations.setdefault(tuple(point), []).append(j)
            exclusions = []
            for k, pair in enumerate(edges):
                for index in pair:
                    exclusions.extend((j, k) for j in locations.get(tuple(stage[index]), ()))
            velocity += segments_velocity(points, stage, edges, gamma,
                model=config.core_model, core_radius=radii,
                exclusions=np.asarray(exclusions, dtype=np.int64).reshape(-1, 2))
        return velocity

    dt = config.dt_s/config.geometry_substeps
    for substep in range(config.geometry_substeps):
        time = state.time_s+substep*dt
        k1 = rhs(moving, time)
        if config.convection_integrator == 'euler':
            candidate = moving+dt*k1
        elif config.convection_integrator == 'heun':
            k2 = rhs(checked(moving+dt*k1, moving, 'heun_predictor', time+dt), time+dt)
            candidate = moving+dt*(k1+k2)/2
        else:
            k2 = rhs(checked(moving+dt*k1/2, moving, 'rk4_stage2', time+dt/2), time+dt/2)
            k3 = rhs(checked(moving+dt*k2/2, moving, 'rk4_stage3', time+dt/2), time+dt/2)
            k4 = rhs(checked(moving+dt*k3, moving, 'rk4_stage4', time+dt), time+dt)
            candidate = moving+dt*(k1+2*k2+2*k3+k4)/6
        moving = checked(candidate, moving, 'accepted_geometry', time+dt)
    return moving.reshape(state.step+1, width, 3)


def advance_lifting_surface(state, config, inflow, *, next_wing=None,
                            structural_velocity=(0., 0., 0.), next_time_s=None, polar=None):
    """Pure state transition; geometry RK order does not imply coupled RK order."""
    if type(state) not in (VlmState, VlmWakeState):
        raise ValueError('state does not belong to a lifting-surface method')
    if config != state.config:
        raise ValueError('initialize a new run to change wake configuration')
    time = state.time_s+config.dt_s
    if not np.isfinite(time) or time <= state.time_s:
        raise ValueError('time step cannot advance')
    if next_time_s is not None:
        requested = _real(next_time_s, 'next time')
        if requested.shape or requested <= state.time_s or not np.isclose(
                requested, time, rtol=0, atol=8*np.finfo(float).eps*max(1., abs(time))):
            raise ValueError('next time must equal old time plus physical dt')
        time = requested.item()
    wing = state.wing if next_wing is None else next_wing
    if wing.node_count != state.wing.node_count or not np.array_equal(wing.panel_edges, state.wing.panel_edges):
        raise ValueError('lattice connectivity cannot change during a run')
    if state.solution.polar_coupling is not None:
        if encode_polars(polar_tables(polar, wing.panels)) != state.solution.polar_coupling.source_polars_json:
            raise ValueError('initialize a new run to change coupled source polars')
    if type(state) is VlmState:
        started = start_lifting_surface(wing, config, inflow, settings=state.settings,
            time_s=time, structural_velocity=structural_velocity, polar=polar)
        return VlmState(time, state.step+1, wing, started.solution, started.mesh,
            state.settings, config, started.structural_velocity,
            started.lattice_structural_velocity, started.prescribed_row)
    if state.step >= config.max_rows:
        raise ValueError('wake row limit reached; no implicit truncation is implemented')
    mesh = build_lifting_surface_mesh(wing, state.settings)
    rows = _convect(state, config, inflow, wing)
    cores = advance_cores(state, rows, config)
    history = np.vstack((state.solution.circulation, state.wake_circulation))
    linear = state.settings.wake_basis == 'linear_potential'
    solution, body = _solve(wing, mesh, state.settings, config, inflow,
        structural_velocity, time, rows, history, cores,
        previous=state.solution.lattice_circulation,
        previous_previous=state.previous_lattice_circulation, polar=polar,
        previous_closure=state.solution.polar_coupling,
        nodal_history=state.wake_node_potential_m2_s if linear else None,
        initial_gamma_rate=state.initial_gamma_rate)
    theta = {'previous':0., 'current':1., 'trapezoidal':.5}[state.settings.wake_release]
    phi = None
    if linear:
        phi = np.vstack((solution.circulation, state.wake_node_potential_m2_s))
        history = .5*(phi[:-1]+phi[1:])
    elif theta:
        history[0] = (1-theta)*history[0]+theta*solution.circulation
    return VlmWakeState(time, state.step+1, wing, solution, mesh, state.settings,
        config, _sample(structural_velocity, wing.cp, time, 'structural velocity'),
        body, rows, np.r_[state.time_s, state.release_times_s], history, cores,
        state.solution.lattice_circulation, phi, state.initial_gamma_rate)
