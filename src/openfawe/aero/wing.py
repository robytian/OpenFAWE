# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenFAST v5.0.0 FVW_Wings.f90 (Meshing, Wings_Panelling,
# Wings_ComputeCirculationPolarData, CirculationFromPolarData) and
# FVW_VortexTools.f90 (interpextrap_cp2node).
# See third_party/OPENFAST-WING-WAKE-NOTICE.md.
"""One rigid lifting line with a quasi-steady, single-table section polar.

Geometry is SI Cartesian. Bound nodes and CPs are on quarter chord. This is not
a three-quarter-chord collocation VLM or a six-DOF aircraft/added-mass solver.
"""

from dataclasses import dataclass, field, asdict
import operator
import numpy as np

from .vortex import CoreModel, _real, _norm, ring_velocity


def owned(value):
    """Immutable float64 array with no writable caller-owned backing storage."""
    array = _real(value, "array")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def positive(value, name, *, zero=False):
    array = _real(value, name)
    if array.shape or (array.item() < 0 if zero else array.item() <= 0):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} scalar")
    return array.item()


def vector_field(value, count, name):
    array = _real(value, name)
    if array.shape == (3,):
        array = np.broadcast_to(array, (count, 3))
    if array.shape != (count, 3):
        raise ValueError(f"{name} must have shape (3,) or ({count},3)")
    return array


def full_cosine_control_points(nodes):
    """OLAF/van Garrel fullcosineapprox on strictly increasing abscissae."""
    nodes = _real(nodes, "span nodes")
    if nodes.ndim != 1 or len(nodes) < 2 or np.any(np.diff(nodes) <= 0):
        raise ValueError("span nodes must increase strictly")
    widths = np.diff(nodes)
    if len(widths) == 1:
        return np.array([nodes.mean()])
    points = np.empty(len(widths))
    points[0] = nodes[0] + widths[0]**2/(widths[0]+widths[1])
    points[-1] = nodes[-2] + widths[-1]*widths[-2]/(widths[-2]+widths[-1])
    for j in range(1, len(widths)-1):
        points[j] = nodes[j] + .25*(widths[j-1]/(widths[j-1]+widths[j]) +
                                    widths[j]/(widths[j]+widths[j+1])+1)*widths[j]
    return points


def cp_to_nodes(abscissa, values, nodes):
    """Linear interpolation inside and linear extrapolation at both tips."""
    x, y, target = _real(abscissa, "CP coordinates"), _real(values, "CP values"), _real(nodes, "nodes")
    if x.ndim != 1 or not len(x) or np.any(np.diff(x) <= 0) or target.ndim != 1:
        raise ValueError("invalid interpolation coordinates")
    if y.shape != (len(x),) and y.shape != (len(x), 3):
        raise ValueError("CP values must be scalar or vector values per CP")
    if len(x) == 1:
        return np.broadcast_to(y[0], (len(target),)+y.shape[1:]).copy()
    index = np.clip(np.searchsorted(x, target)-1, 0, len(x)-2)
    fraction = (target-x[index])/(x[index+1]-x[index])
    if y.ndim == 2:
        fraction = fraction[:, None]
    return y[index] + fraction*(y[index+1]-y[index])


@dataclass(frozen=True)
class SectionPolar:
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    name: str = "single-table polar"
    bounds: str = "error"
    # Full-circle tables (stage 4, decision 0020): the table spans exactly
    # [-180, 180] deg and the angle is wrapped to [-pi, pi) before lookup, as
    # Makani's Airfoil.GetCoeffs does. ``attached_deg`` is a declared
    # (low, high) range used only to label panel regimes; it changes no value.
    periodic: bool = field(default=False, kw_only=True)
    attached_deg: tuple | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.bounds not in ("error", "clamp"):
            raise ValueError("polar bounds must be 'error' or explicit source-compatibility 'clamp'")
        for name in ("alpha_deg", "cl", "cd", "cm"):
            object.__setattr__(self, name, owned(getattr(self, name)))
        if self.alpha_deg.ndim != 1 or len(self.alpha_deg) < 2 or np.any(np.diff(self.alpha_deg) <= 0):
            raise ValueError("polar alpha must increase strictly")
        if any(a.shape != self.alpha_deg.shape for a in (self.cl, self.cd, self.cm)) or np.any(self.cd < 0):
            raise ValueError("polar arrays must agree and Cd must be nonnegative")
        if not isinstance(self.periodic, bool):
            raise ValueError("polar periodic flag must be a bool")
        if self.periodic and (self.alpha_deg[0] != -180. or self.alpha_deg[-1] != 180.):
            raise ValueError("a periodic polar must span exactly [-180, 180] deg")
        if self.attached_deg is not None:
            attached = _real(self.attached_deg, "attached range")
            if attached.shape != (2,) or attached[0] >= attached[1] or attached[0] < self.alpha_deg[0] or attached[1] > self.alpha_deg[-1]:
                raise ValueError("attached_deg must be (low, high) inside the table")
            object.__setattr__(self, "attached_deg", (float(attached[0]), float(attached[1])))

    def _wrapped_deg(self, alpha_rad):
        radians = _real(alpha_rad, "angle of attack")
        if self.periodic:
            radians = np.mod(radians+np.pi, 2*np.pi)-np.pi
        return radians, np.rad2deg(radians)

    def coefficients(self, alpha_rad):
        radians, alpha = self._wrapped_deg(alpha_rad)
        # Compare in the input unit: deg -> rad -> deg can move an exact knot
        # outside by one ulp. This admits exact endpoints without widening bounds.
        if not self.periodic and self.bounds == "error" and (np.any(radians < np.deg2rad(self.alpha_deg[0])) or np.any(radians > np.deg2rad(self.alpha_deg[-1]))):
            raise ValueError("angle of attack outside the declared polar table")
        return tuple(np.interp(alpha, self.alpha_deg, values) for values in (self.cl, self.cd, self.cm))

    def cl_slope(self, alpha_rad):
        """Lift derivative per radian; right segment at an interior knot.

        The upper endpoint uses the last segment. Outside a clamped table the
        coefficient is constant. Bounds match ``coefficients`` exactly.
        """
        self.coefficients(alpha_rad)
        alpha = self._wrapped_deg(alpha_rad)[1]
        index = np.clip(np.searchsorted(self.alpha_deg, alpha, side='right')-1,
                        0, len(self.alpha_deg)-2)
        slope = np.diff(self.cl)[index]/np.deg2rad(np.diff(self.alpha_deg)[index])
        if self.bounds == 'clamp':
            slope = np.where((alpha < self.alpha_deg[0]) | (alpha > self.alpha_deg[-1]), 0., slope)
        return slope

    def attached(self, alpha_rad):
        """True where the (wrapped) angle lies inside the declared attached range.

        Without a declaration every angle is reported attached; this is a label
        for records, not a model switch.
        """
        alpha = self._wrapped_deg(alpha_rad)[1]
        if self.attached_deg is None:
            return np.ones(np.shape(alpha), dtype=bool)
        return (alpha >= self.attached_deg[0]) & (alpha <= self.attached_deg[1])

    @classmethod
    def linear(cls, slope=2*np.pi, *, limit_deg=20.):
        slope, limit = positive(slope, "lift slope"), positive(limit_deg, "alpha limit")
        alpha = np.array([-limit, 0., limit])
        return cls(alpha, slope*np.deg2rad(alpha), np.zeros(3), np.zeros(3), "analytic linear polar")


def camber_data(x, offsets, nodes):
    """Validate explicit mean-surface offsets without inferring a local frame."""
    if x is None and offsets is None:
        return None, None
    if x is None or offsets is None:
        raise ValueError('camber coordinates and offsets must be supplied together')
    x, offsets = owned(x), owned(offsets)
    if (x.ndim != 1 or len(x) < 2 or x[0] != 0. or x[-1] != 1.
            or np.any(np.diff(x) <= 0)):
        raise ValueError('camber x must increase strictly from 0 to 1')
    if offsets.shape != (nodes, len(x), 3) or np.any(offsets[:, [0, -1]] != 0.):
        raise ValueError('camber offsets require (nodes,samples,3) and zero endpoints')
    return x, offsets


@dataclass(frozen=True)
class Wing:
    leading_edge: np.ndarray
    trailing_edge: np.ndarray
    camber_x: np.ndarray | None = field(default=None, repr=False)
    camber_offset_m: np.ndarray | None = field(default=None, repr=False)
    bound: np.ndarray = field(init=False, repr=False)
    span: np.ndarray = field(init=False, repr=False)
    cp_span: np.ndarray = field(init=False, repr=False)
    cp: np.ndarray = field(init=False, repr=False)
    chord_cp: np.ndarray = field(init=False, repr=False)
    tangent: np.ndarray = field(init=False, repr=False)
    normal: np.ndarray = field(init=False, repr=False)
    orthogonal: np.ndarray = field(init=False, repr=False)
    dl: np.ndarray = field(init=False, repr=False)
    area: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        le, te = owned(self.leading_edge), owned(self.trailing_edge)
        if le.ndim != 2 or le.shape[1] != 3 or len(le) < 2 or te.shape != le.shape:
            raise ValueError("LE and TE must have matching (n+1,3) shape")
        x, offsets = camber_data(self.camber_x, self.camber_offset_m, len(le))
        object.__setattr__(self, 'camber_x', x)
        object.__setattr__(self, 'camber_offset_m', offsets)
        bound = le + .25*(te-le)
        chord, width = _norm(te-le), _norm(np.diff(bound, axis=0))
        if np.any(chord <= 0) or np.any(width <= 0):
            raise ValueError("zero chord or repeated bound nodes")
        span = np.r_[0., np.cumsum(width)]
        cp_span = full_cosine_control_points(span)
        cp = np.column_stack([np.interp(cp_span, span, bound[:, j]) for j in range(3)])
        tangent = .5*((te[:-1]+te[1:])-(le[:-1]+le[1:]))
        dl = np.diff(bound, axis=0)
        normal = np.cross(tangent, dl)
        span_middle = np.diff(.5*(le+te), axis=0)
        area = _norm(np.cross(tangent, span_middle))
        if np.any(_norm(tangent) <= 0) or np.any(_norm(normal) <= 0) or np.any(area <= 0):
            raise ValueError("degenerate wing panel")
        tangent /= _norm(tangent)[:, None]; normal /= _norm(normal)[:, None]
        data = dict(leading_edge=le, trailing_edge=te, bound=bound, span=span, cp_span=cp_span,
                    cp=cp, chord_cp=np.interp(cp_span, span, chord), tangent=tangent,
                    normal=normal, orthogonal=np.cross(normal, tangent), dl=dl, area=area)
        for name, value in data.items():
            object.__setattr__(self, name, owned(value))

    @property
    def panels(self):
        return len(self.area)

    @property
    def node_count(self):
        return len(self.bound)

    @property
    def panel_edges(self):
        return np.column_stack([np.arange(self.panels),np.arange(1,self.panels+1)])

    def transformed(self, rotation, translation):
        rotation, translation = _real(rotation, "rotation"), _real(translation, "translation")
        if rotation.shape != (3,3) or translation.shape != (3,) or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-12) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-12):
            raise ValueError("a proper rotation and a (3,) translation are required")
        offsets = None if self.camber_offset_m is None else self.camber_offset_m@rotation.T
        return Wing(self.leading_edge@rotation.T+translation, self.trailing_edge@rotation.T+translation,
                    self.camber_x, offsets)

    @classmethod
    def ellipse(cls, panels=20, span=5., root_chord=1.):
        if isinstance(panels, (bool, np.bool_)) or operator.index(panels) < 2:
            raise ValueError("at least two panels required")
        span, root_chord = positive(span, "span"), positive(root_chord, "root chord")
        theta = np.linspace(0, np.pi, panels+1)
        y = -.5*span*np.cos(theta)
        # Small, explicitly declared positive tip avoids a degenerate bound->TE
        # segment. Original source uses 1e-5 m tips for its 1 m root chord.
        chord = np.maximum(root_chord*np.sin(theta), root_chord*1e-5)
        le = np.column_stack([-.25*chord, y, np.zeros_like(y)])
        return cls(le, le+np.column_stack([chord, np.zeros_like(y), np.zeros_like(y)]))


@dataclass(frozen=True)
class SolveSettings:
    relaxation: float = .1
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    max_iterations: int = 500
    solver: str = 'fixed_point'
    fallback_relaxations: tuple = (.1, .03)
    fallback_max_iterations: int = 2000
    jacobian: str = 'finite_difference'
    # 'source': the unchanged forward-flow VSM closure. 'full_incidence' (stage 4,
    # decision 0020): Kutta-Joukowski target on the total sectional velocity with
    # per-panel regimes (attached, post-stall, reverse, no inflow); see vsm.py.
    closure: str = 'source'
    # Declared model parameters of the full-incidence closure (rejected with 'source'). They act
    # through a circulation weight evaluated from the onset flow (ambient + history induction,
    # without the current bound/near-wake induction; vsm.onset_weight), fixed during the solve.
    # span_flow_fraction f: a panel whose onset velocity in its section plane (chord tangent
    #   and normal, the components that define the sectional angle) is at most f of its onset
    #   speed carries no bound circulation; the weight rises smoothly to one at 2f. Sectional
    #   (strip) data do not apply to flow across the section plane and the sectional angle is
    #   ill-conditioned there. The strip loads are kept (dynamic pressure of |v x s|).
    #   0 keeps only the round-off test.
    # reverse_flow: 'coupled' keeps reverse-flow panels in the vortex system (wake shed from
    #   the geometric trailing edge, which is the upstream edge there); 'strip' fades their
    #   bound circulation to zero just past 90 deg onset angle and takes the loads from the
    #   section data at the local flow (strip theory).
    # reverse_fade: width of that fade in the onset chordwise fraction cos(alpha0): the weight
    #   is one at cos(alpha0) = 0 and zero at -reverse_fade (0.2: onset section angle 101.5 deg).
    #   Only with reverse_flow='strip'.
    span_flow_fraction: float = 0.
    reverse_flow: str = 'coupled'
    reverse_fade: float = .2

    def __post_init__(self):
        if self.jacobian not in ('finite_difference', 'active_segment'):
            raise ValueError('unknown circulation Jacobian policy')
        if self.closure not in ('source', 'full_incidence'):
            raise ValueError("VSM closure must be 'source' or 'full_incidence'")
        fraction = positive(self.span_flow_fraction, 'span flow fraction', zero=True)
        if fraction >= .5:
            raise ValueError('span flow fraction must be below 0.5')
        if self.reverse_flow not in ('coupled', 'strip'):
            raise ValueError("reverse_flow must be 'coupled' or 'strip'")
        fade = positive(self.reverse_fade, 'reverse fade')
        if fade > 1.:
            raise ValueError('reverse fade is a cosine and must not exceed 1')
        if self.reverse_flow != 'strip' and fade != .2:
            raise ValueError("reverse_fade belongs to reverse_flow='strip'")
        if self.closure == 'source' and (fraction != 0. or self.reverse_flow != 'coupled'):
            raise ValueError('span_flow_fraction and reverse_flow belong to the full-incidence closure')
        object.__setattr__(self, 'span_flow_fraction', float(fraction))
        object.__setattr__(self, 'reverse_fade', float(fade))
        if self.jacobian != 'finite_difference' and self.solver not in ('newton', 'newton_picard'):
            raise ValueError('active-segment Jacobian requires Newton')
        if self.solver not in ('fixed_point', 'newton', 'newton_picard'):
            raise ValueError('circulation solver must be fixed_point, newton or newton_picard')
        relaxations = tuple(positive(value, 'fallback relaxation') for value in self.fallback_relaxations)
        if not relaxations or any(value > .1 for value in relaxations):
            raise ValueError('fallback relaxations must be nonempty and at most 0.1')
        object.__setattr__(self, 'fallback_relaxations', relaxations)
        if (isinstance(self.fallback_max_iterations, (bool, np.bool_))
                or operator.index(self.fallback_max_iterations) < 1):
            raise ValueError('fallback_max_iterations must be a positive integer')
        if positive(self.relaxation, "relaxation") > 1:
            raise ValueError("relaxation must not exceed 1")
        positive(self.relative_tolerance, "relative tolerance")
        positive(self.absolute_tolerance, "absolute tolerance")
        if isinstance(self.max_iterations, (bool, np.bool_)) or operator.index(self.max_iterations) < 1:
            raise ValueError("max_iterations must be a positive integer")


def solve_settings_metadata(settings):
    """Preserve old metadata; new derivative policies are explicit."""
    values = asdict(settings)
    if settings.jacobian == 'finite_difference':
        values.pop('jacobian')
    if settings.closure == 'source':
        values.pop('closure')
    if settings.span_flow_fraction == 0.:
        values.pop('span_flow_fraction')
    if settings.reverse_flow == 'coupled':
        values.pop('reverse_flow')
        values.pop('reverse_fade')
    return values


def wake_config_metadata(config):
    """Keep the unused default derivative out of legacy wake metadata."""
    values = asdict(config)
    values['solve'] = solve_settings_metadata(config.solve)
    if values.get('wake_core_chord_fraction') is None:
        values.pop('wake_core_chord_fraction', None)
    if values.get('surface_guard') == 'off':
        values.pop('surface_guard')
        values.pop('guard_chordwise_panels')
    if values.get('velocity_backend', 'numpy') == 'numpy':
        values.pop('velocity_backend', None)
        values.pop('tree', None)
    elif values.get('tree') is None:
        values.pop('tree', None)
    return values


class CirculationConvergenceError(RuntimeError):
    def __init__(self, iterations, residual, limit):
        super().__init__(f"Circulation failed after {iterations} iterations: residual {residual:g} > {limit:g} m^2/s")
        self.iterations, self.residual, self.limit = iterations, residual, limit


@dataclass(frozen=True)
class WingSolution:
    circulation: np.ndarray
    relative_velocity: np.ndarray
    induced_velocity: np.ndarray
    alpha_rad: np.ndarray
    reynolds: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    panel_force: np.ndarray
    panel_couple: np.ndarray
    iterations: int
    residual: float
    residual_limit: float
    solver_trace_json: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        for name in ("circulation", "relative_velocity", "induced_velocity", "alpha_rad", "reynolds",
                     "cl", "cd", "cm", "panel_force", "panel_couple"):
            object.__setattr__(self, name, owned(getattr(self, name)))

    @property
    def force(self):
        return self.panel_force.sum(axis=0)

    def moment(self, wing, reference=(0.,0.,0.)):
        reference = _real(reference, "moment reference")
        if reference.shape != (3,):
            raise ValueError("moment reference must have shape (3,)")
        return (np.cross(wing.cp-reference, self.panel_force)+self.panel_couple).sum(axis=0)


def influence_matrix(wing, downstream_rows=(), *, model=CoreModel.NONE, core_radius=0.):
    """Unit-Gamma influence of bound and following rings, all sharing Gamma.

    Explicit upstream zero convention applies on bound-line CPs. Older wake is
    excluded from this operator and supplied separately to solve_circulation.
    """
    rows = [wing.bound, wing.trailing_edge]
    for row in downstream_rows:
        row = _real(row, "wake row")
        if row.shape != wing.bound.shape:
            raise ValueError("downstream row shape mismatch")
        rows.append(row)
    influence = np.zeros((wing.panels, wing.panels, 3))
    for left, right in zip(rows, rows[1:]):
        for j,(a,b) in enumerate(wing.panel_edges):
            corners = [left[a], left[b], right[b], right[a]]
            influence[:, j, :] += ring_velocity(corners, wing.cp, 1., model=model,
                                                core_radius=core_radius, singularity="zero")
    return influence


def _section(wing, polar, velocity):
    chordwise = np.sum(velocity*wing.tangent, axis=1)
    normal = np.sum(velocity*wing.normal, axis=1)
    projected = chordwise[:,None]*wing.tangent + normal[:,None]*wing.normal
    speed = _norm(projected)
    jouk = np.cross(velocity, wing.dl)
    jouk = np.sum(jouk*wing.normal, axis=1)[:,None]*wing.normal + np.sum(jouk*wing.tangent, axis=1)[:,None]*wing.tangent
    denominator = _norm(jouk)
    if np.any((speed > 0) & (denominator == 0)):
        raise ValueError("nonzero sectional flow has zero projected Kutta-Joukowski factor")
    alpha = np.arctan2(normal, chordwise)
    cl, cd, cm = polar.coefficients(alpha)
    gamma = np.divide(.5*cl*speed**2*wing.area, denominator, out=np.zeros_like(speed), where=denominator>0)
    return gamma, alpha, cl, cd, cm, speed, projected, jouk


def solve_circulation(wing, polar, inflow, operator_matrix, *, structural_velocity=(0.,0.,0.),
                      known_induction=(0.,0.,0.), density=1.225, kinematic_viscosity=1.4639e-5,
                      initial=None, settings=SolveSettings()):
    """Solve the source-derived polar fixed point; return only a converged state."""
    rho, nu = positive(density, "density"), positive(kinematic_viscosity, "kinematic viscosity")
    operator_matrix = _real(operator_matrix, "influence operator")
    if operator_matrix.shape != (wing.panels, wing.panels, 3):
        raise ValueError("influence operator shape mismatch")
    ambient = vector_field(inflow, wing.panels, "inflow") - vector_field(structural_velocity, wing.panels, "structural velocity")
    known = vector_field(known_induction, wing.panels, "known induction")
    if not isinstance(settings, SolveSettings):
        raise ValueError('settings must be SolveSettings')
    if settings.jacobian != 'finite_difference':
        raise ValueError('lifting-line closure does not support an active-segment Jacobian')
    gamma = (np.zeros(wing.panels) if settings.solver in ('newton','newton_picard') else _section(wing, polar, ambient+known)[0]) if initial is None else _real(initial, "initial circulation").copy()
    if gamma.shape != (wing.panels,):
        raise ValueError("initial circulation shape mismatch")
    newton_iterations = 0
    solver_trace = None
    if settings.solver in ('newton','newton_picard'):
        from .nonlinear import solve_newton, solve_newton_picard, section_solver_trace
        def target(value):
            return _section(wing, polar, ambient+known+np.einsum('ijk,j->ik', operator_matrix, value))[0]
        if settings.solver == 'newton':
            gamma, newton_iterations = solve_newton(target, gamma, settings)
        else:
            gamma, newton_iterations, solver_trace = solve_newton_picard(target, gamma, settings)
    for iteration in range(1, settings.max_iterations+1):
        induced = known + np.einsum("ijk,j->ik", operator_matrix, gamma)
        velocity = ambient+induced
        target, alpha, cl, cd, cm, speed, projected, jouk = _section(wing, polar, velocity)
        residual = float(np.max(np.abs(target-gamma)))
        limit = settings.absolute_tolerance + settings.relative_tolerance*float(np.mean(np.abs(gamma)))
        if residual <= limit:
            drag_direction = np.divide(projected, speed[:,None], out=np.zeros_like(projected), where=speed[:,None]>0)
            force = rho*gamma[:,None]*jouk + (.5*rho*cd*speed**2*wing.area)[:,None]*drag_direction
            couple = (.5*rho*cm*speed**2*wing.area*wing.chord_cp)[:,None]*wing.orthogonal
            if solver_trace is not None:
                solver_trace = section_solver_trace(solver_trace, polar, alpha, residual, limit)
            return WingSolution(gamma, velocity, induced, alpha, wing.chord_cp*_norm(velocity)/nu,
                                cl, cd, cm, force, couple, newton_iterations or iteration, residual, limit,
                                solver_trace_json=solver_trace)
        if solver_trace is not None:
            from .nonlinear import HybridLoadReevaluationError
            raise HybridLoadReevaluationError(solver_trace, residual, limit)
        gamma = gamma + settings.relaxation*(target-gamma)
        if not np.isfinite(gamma).all():
            raise ValueError("nonfinite circulation iterate")
    raise CirculationConvergenceError(iteration, residual, limit)


def solve_steady(wing, polar, inflow, *, wake_length=1000., model=CoreModel.NONE, core_radius=0., **kwargs):
    """Declared finite straight-wake steady limit, not the original transient run."""
    wind = _real(inflow, "steady inflow")
    if wind.shape != (3,) or _norm(wind) == 0:
        raise ValueError("steady-wake direction requires a nonzero uniform inflow vector")
    downstream = wing.trailing_edge + positive(wake_length, "wake length")*wind/_norm(wind)
    matrix = influence_matrix(wing, [downstream], model=model, core_radius=core_radius)
    return solve_circulation(wing, polar, wind, matrix, **kwargs)
