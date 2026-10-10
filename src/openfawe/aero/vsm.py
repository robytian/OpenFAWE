# SPDX-License-Identifier: MIT
# Adapted from awegroup/Vortex-Step-Method, revision 5485b823f33c76be70cf0569204c2e9c7be976c5.
# See third_party/VSM-NOTICE.md and docs/design/of-0006-aero-method-contract.md.
"""Nonlinear VSM with separately identified historical and modern source cores.

The historical thesis kernel is preserved; its inner finite projection is not
azimuthal. ``source_pinned_v1`` replicates modern core/Filament.py away from the
axis, with explicit OpenFAWE axis rejection. Neither is material wake diffusion.
"""

from dataclasses import dataclass, field
import numpy as np

from .vortex import _real, _norm, segments_velocity, FilamentSingularity
from .wing import (owned, positive, vector_field, WingSolution, SolveSettings,
                   CirculationConvergenceError)
from .aircraft import WingAssembly


def semi_infinite_velocity(start, direction, points):
    """Unit circulation from start towards infinity, no viscous core model."""
    start, direction, points = (_real(v, n) for v, n in
                                [(start, 'start'), (direction, 'direction'), (points, 'points')])
    if start.shape != (3,) or direction.shape != (3,) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('ray inputs require (3,), (3,), (n,3) shapes')
    length = _norm(direction)
    if length == 0:
        raise ValueError('ray direction must be nonzero')
    unit = direction / length
    r = points - start
    cross = np.cross(unit, r)
    distance, radius = _norm(r), _norm(cross)
    axial = r @ unit
    axis = radius <= 8*np.finfo(float).eps*np.maximum(1, distance)
    if np.any(axis & (axial >= 0)):
        raise FilamentSingularity('point on semi-infinite vortex ray')
    velocity = np.zeros_like(points)
    mask = ~axis
    # Avoid cancellation behind the ray origin: (1+a/r)/h^2 = 1/(r*(r-a)).
    coefficient = np.empty(np.count_nonzero(mask))
    d, a, h = distance[mask], axial[mask], radius[mask]
    behind = a < 0
    coefficient[behind] = 1/(d[behind]*(d[behind]-a[behind]))
    coefficient[~behind] = (1+a[~behind]/d[~behind])/h[~behind]**2
    velocity[mask] = cross[mask]*coefficient[:, None]/(4*np.pi)
    return velocity


def geometry(wing):
    """Return source 3/4 CPs, midpoint chord and local normal/tangent/span axes."""
    if isinstance(wing, WingAssembly):
        parts = [geometry(surface) for surface in wing.wings]
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(5))
    fraction = (wing.cp_span-wing.span[:-1])/np.diff(wing.span)
    chord_nodes = wing.trailing_edge-wing.leading_edge
    local_chord = chord_nodes[:-1] + fraction[:, None]*np.diff(chord_nodes, axis=0)
    cp = wing.cp + .5*local_chord
    tangent = local_chord/_norm(local_chord)[:, None]
    normal = np.cross(local_chord, np.diff(wing.leading_edge, axis=0))
    if np.any(_norm(normal) == 0):
        raise ValueError('degenerate VSM local normal')
    normal /= _norm(normal)[:, None]
    span = wing.dl/_norm(wing.dl)[:, None]
    chord = _norm(.5*(chord_nodes[:-1]+chord_nodes[1:]))
    return cp, chord, normal, tangent, span


def _source_finite(points, a, b, speed, bound):
    """Legacy source core projection; retained explicitly for reproducibility."""
    r0=b-a;length=_norm(r0);r1=points-a;r2=points-b
    cross1=np.cross(r1,r0);cross2=np.cross(r2,r0)
    radius=_norm(cross1)/length
    axial=np.sum(r1*r0,axis=1)/length
    epsilon=np.full(len(points),.05*length) if bound else np.sqrt(4*1.25643*1.48e-5*np.abs(axial)/speed)
    inside=radius<=epsilon
    if np.any(inside & (radius==0)):
        raise FilamentSingularity('point on legacy VSM filament axis')
    result=segments_velocity(points,[a,b],[[0,1]],[1.])
    if np.any(inside):
        u=r1[inside];v=r2[inside];eps=epsilon[inside,None]
        up=(u@r0)[:,None]*r0/length**2+eps*cross1[inside]/_norm(cross1[inside])[:,None]
        vp=(v@r0)[:,None]*r0/length**2+eps*cross2[inside]/_norm(cross2[inside])[:,None]
        cross=np.cross(up,vp)
        value=cross*((up/_norm(up)[:,None]-vp/_norm(vp)[:,None])@r0/(4*np.pi*_norm(cross)**2))[:,None]
        result[inside]=value*(radius[inside]/epsilon[inside])[:,None]
    return result


def _source_ray(points,start,direction,speed):
    """Original trailing-core projection, transformed to directed-ray sign."""
    unit=direction/_norm(direction);r=points-start
    radius=_norm(np.cross(unit,r));axial=r@unit
    if np.any(radius == 0):
        raise FilamentSingularity('point on legacy VSM ray axis')
    epsilon=np.sqrt(4*1.25643*1.48e-5*np.abs(axial)/speed)
    inside=radius<=epsilon
    result=semi_infinite_velocity(start,direction,points)
    if np.any(inside):
        if np.any(_norm(r[inside])==0):raise FilamentSingularity('point at legacy VSM ray origin')
        perpendicular=r[inside]/_norm(r[inside])[:,None]-unit
        if np.any(_norm(perpendicular)==0):raise FilamentSingularity('point on legacy VSM ray')
        projected=axial[inside,None]*unit+epsilon[inside,None]*perpendicular/_norm(perpendicular)[:,None]
        cross=np.cross(unit,projected)
        result[inside]=cross*((1+projected@unit/_norm(projected))/(4*np.pi*_norm(cross)**2))[:,None]
    return result


def _pinned_finite(points, a, b, speed, bound):
    """Modern source cylinder projection, nonaxis probes only.

    Trailing cutoff uses radial distance, not age. Its interior magnitude need
    not approach zero. Outside points use the existing ideal finite primitive.
    """
    axis = b-a
    length = _norm(axis)
    if length == 0:
        raise ValueError('zero-length source filament')
    unit = axis/length
    r = points-a
    axial = r@unit
    radial = r-axial[:, None]*unit
    radius = _norm(radial)
    if np.any(radius == 0):
        raise FilamentSingularity('point on pinned VSM filament axis')
    epsilon = (np.full(len(points), .05*length) if bound else
               np.sqrt(4*1.25643*1.48e-5*radius/speed))
    if np.any(radius < 1e-12*epsilon):
        raise FilamentSingularity('point at pinned VSM filament axis guard')
    inside = radius <= epsilon
    result = np.zeros_like(points)
    if np.any(~inside):
        result[~inside] = segments_velocity(points[~inside], [a,b], [[0,1]], [1.])
    if np.any(inside):
        x, eps = axial[inside], epsilon[inside]
        angle = x/np.hypot(x,eps)-(x-length)/np.hypot(x-length,eps)
        result[inside] = np.cross(unit,radial[inside])*(angle/(4*np.pi*eps**2))[:, None]
    return result


def _pinned_ray(points, start, direction, speed):
    """Source boundary-only inner ray field, singular as radial distance^-1/2."""
    length = _norm(direction)
    if length == 0:
        raise ValueError('ray direction must be nonzero')
    unit = direction/length
    r = points-start
    axial = r@unit
    radial = r-axial[:, None]*unit
    radius = _norm(radial)
    if np.any(radius <= 1e-16):
        raise FilamentSingularity('point at pinned VSM ray axis guard')
    epsilon = np.sqrt(4*1.25643*1.48e-5*radius/speed)
    inside = radius <= epsilon
    result = np.zeros_like(points)
    if np.any(~inside):
        result[~inside] = semi_infinite_velocity(start,unit,points[~inside])
    if np.any(inside):
        x, eps = axial[inside], epsilon[inside]
        distance = np.hypot(x,eps)
        # Rationalized coefficient retains behind-origin cancellation accuracy.
        coefficient = np.empty(len(x))
        behind = x < 0
        coefficient[behind] = eps[behind]/(distance[behind]*(distance[behind]-x[behind]))
        coefficient[~behind] = (1+x[~behind]/distance[~behind])/eps[~behind]
        result[inside] = np.cross(unit,radial[inside]/radius[inside,None])*(coefficient/(4*np.pi))[:, None]
    return result


VSM_CORE_POLICIES = ('none', 'source_legacy', 'source_pinned_v1')
_SOURCE_KERNELS = {'source_legacy':(_source_finite,_source_ray),
                   'source_pinned_v1':(_pinned_finite,_pinned_ray)}
_CORE_IDENTITIES = {'none':'openfawe:ideal-v1',
    'source_legacy':'openfawe:thesis-axial-legacy-v1',
    'source_pinned_v1':'openfawe:modern-filament-nonaxis-v1@5485b823f33c76be70cf0569204c2e9c7be976c5'}


def core_kernel_identity(policy):
    """Stable semantic identity; actual executed source hashes are also archived."""
    if not isinstance(policy,str) or policy not in VSM_CORE_POLICIES:
        raise ValueError('invalid VSM core policy')
    return _CORE_IDENTITIES[policy]


def _source_kernels(policy, speed):
    if policy is None:
        if speed is not None:
            raise ValueError('source core speed requires an explicit VSM core policy')
        return None
    core_kernel_identity(policy)
    if policy == 'none':
        if speed is not None:
            raise ValueError('ideal VSM core does not accept source core speed')
        return None
    positive(speed, 'source core speed')
    return _SOURCE_KERNELS[policy]


def horseshoe_influence(wing, direction, points, *, include_bound=True,
                        source_core_speed=None, source_core_policy=None):
    """Raw five-filament field, without any sectional closure self correction."""
    cp = _real(points, 'probe points')
    if cp.ndim != 2 or cp.shape[1] != 3:
        raise ValueError('probe points must have shape (n,3)')
    matrix = np.zeros((len(cp), wing.panels, 3))
    kernels = _source_kernels(source_core_policy, source_core_speed)
    def finite(a, b, bound=False):
        if kernels is not None:
            return kernels[0](cp,a,b,source_core_speed,bound)
        return segments_velocity(cp, [a, b], [[0, 1]], [1.])
    def ray(start):
        if kernels is not None:
            return kernels[1](cp,start,direction,source_core_speed)
        return semi_infinite_velocity(start,direction,cp)
    for j,(ia,ib) in enumerate(wing.panel_edges):
        a, b = wing.bound[[ia,ib]]
        left, right = wing.trailing_edge[[ia,ib]]
        value = finite(left, a) + finite(b, right)
        value += ray(right)-ray(left)
        if include_bound:
            value += finite(a, b, True)
        matrix[:, j] = value
    return matrix


def vsm_influence(wing, direction, *, at_load_points=False, source_core_speed=None,
                  source_core_policy=None):
    """Source 3/4 closure correction, or 1/4 trailing-only load-direction pass."""
    cp = wing.cp if at_load_points else geometry(wing)[0]
    matrix = horseshoe_influence(wing, direction, cp, include_bound=not at_load_points,
                                source_core_speed=source_core_speed, source_core_policy=source_core_policy)
    if not at_load_points:
        matrix[np.arange(wing.panels), np.arange(wing.panels)] -= vsm_self_correction(wing, cp)
    return matrix


def vsm_self_correction(wing, collocation_points=None):
    """Unit-Gamma infinite-bound self field removed by the VSM section closure.

    This is the original inviscid two-dimensional closure correction, not a
    material wake filament. It therefore has no evolving viscous core.
    """
    cp = geometry(wing)[0] if collocation_points is None else _real(collocation_points, 'VSM collocation points')
    if cp.shape != (wing.panels, 3):
        raise ValueError('VSM collocation points must have shape (panels,3)')
    r = cp-.5*(wing.bound[wing.panel_edges[:,0]]+wing.bound[wing.panel_edges[:,1]])
    cross = np.cross(wing.dl, r)
    if np.any(_norm(cross) == 0):
        raise ValueError('VSM self correction requires an off-bound collocation point')
    return cross*(_norm(wing.dl)/(2*np.pi*_norm(cross)**2))[:, None]


#: Per-panel regime codes of the full-incidence closure (``VsmSolution.panel_regime``).
REGIMES = ('attached', 'post_stall', 'reverse', 'no_inflow')


@dataclass(frozen=True)
class VsmSolution(WingSolution):
    polar_alpha_rad: np.ndarray
    collocation_points: np.ndarray
    wake_direction: np.ndarray
    closure_ambient_velocity: np.ndarray | None = field(default=None, kw_only=True)
    load_ambient_sampling: str = field(default='legacy_closure_reuse', kw_only=True)
    closure: str = field(default='source', kw_only=True)
    panel_regime: np.ndarray | None = field(default=None, kw_only=True)
    #: Circulation weight per panel of the declared regime parameters (None when neither is set).
    circulation_weight: np.ndarray | None = field(default=None, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.load_ambient_sampling not in ('legacy_closure_reuse', 'separate_load_input_v1'):
            raise ValueError('unknown VSM load ambient sampling')
        if self.closure not in ('source', 'full_incidence'):
            raise ValueError('unknown VSM closure')
        for name in ('polar_alpha_rad', 'collocation_points', 'wake_direction'):
            object.__setattr__(self, name, owned(getattr(self, name)))
        if self.panel_regime is not None:
            regime = np.asarray(self.panel_regime)
            if (regime.shape != self.circulation.shape or regime.dtype.kind not in 'iu'
                    or np.any(regime < 0) or np.any(regime >= len(REGIMES))):
                raise ValueError('panel regimes must be one code per panel')
            object.__setattr__(self, 'panel_regime', np.frombuffer(regime.astype(np.int8).tobytes(), dtype=np.int8))
        if self.circulation_weight is not None:
            weight = _real(self.circulation_weight, 'circulation weight')
            if weight.shape != self.circulation.shape or np.any(weight < 0) or np.any(weight > 1):
                raise ValueError('circulation weights must be one value in [0, 1] per panel')
            object.__setattr__(self, 'circulation_weight', owned(weight))
        ambient = self.closure_ambient_velocity
        if ambient is None:
            # Compatibility for legacy solutions that reused closure ambient at loads.
            ambient = self.relative_velocity-self.induced_velocity
        object.__setattr__(self, 'closure_ambient_velocity', owned(
            vector_field(ambient, len(self.circulation), 'closure ambient velocity')))


def solve_vsm(wing, polar, inflow, *, structural_velocity=(0.,0.,0.),
              load_relative_ambient=None,
              density=1.225, kinematic_viscosity=1.4639e-5, initial=None,
              settings=SolveSettings(relaxation=.03, max_iterations=2000), core_policy='source_legacy',
              diagnostics=None):
    """Converged source VSM state. Input velocities are defined at 3/4 CPs.

    Optional ``load_relative_ambient`` is wind minus body velocity at 1/4 chord.
    When omitted, the supplied closure-point ambient is reused for compatibility.

    The straight wake uses the normalized average nonzero relative-flow
    direction. Attached forward flow on rigid lifting surfaces is the domain.
    """
    core_kernel_identity(core_policy)
    def operator(direction, speed):
        return vsm_influence(wing, direction,
            source_core_speed=speed if core_policy != 'none' else None, source_core_policy=core_policy)
    def load_operator(direction, speed):
        return vsm_influence(wing, direction, at_load_points=True,
            source_core_speed=speed if core_policy != 'none' else None, source_core_policy=core_policy)
    return solve_vsm_operator(wing, polar, inflow, operator, load_operator,
        structural_velocity=structural_velocity, load_relative_ambient=load_relative_ambient, density=density,
        kinematic_viscosity=kinematic_viscosity, initial=initial, settings=settings, diagnostics=diagnostics)


def _vsm_closure(gamma, ambient, matrix, known, chord, tangent, normal, span, cross_speed, polar):
    """One unchanged target for Newton, its derivative and final acceptance."""
    velocity = ambient + np.einsum('ijk,j->ik', matrix, gamma) + known
    tangential = np.sum(velocity*tangent, axis=1)
    if np.any(tangential <= 0):
        raise ValueError('VSM iterate left forward-flow domain')
    alpha = np.arctan2(np.sum(velocity*normal, axis=1), tangential)
    cl, cd, cm = polar.coefficients(alpha)
    speed2 = _norm(np.cross(velocity, span))**2
    target = .5*chord*cl*speed2/cross_speed
    return target, velocity, alpha, cl, cd, cm, speed2


def _vsm_closure_jacobian(gamma, ambient, matrix, known, chord, tangent, normal, span, cross_speed, polar):
    """Exact active-table derivative of target-Gamma, not a new closure.

    Local span and the chord-normal plane need not be orthogonal. The atan2
    denominator and transverse speed squared must therefore remain distinct.
    """
    _, velocity, alpha, cl, _, _, speed2 = _vsm_closure(
        gamma, ambient, matrix, known, chord, tangent, normal, span, cross_speed, polar)
    vt = np.sum(velocity*tangent, axis=1)
    vn = np.sum(velocity*normal, axis=1)
    an = np.einsum('ijk,ik->ij', matrix, normal)
    at = np.einsum('ijk,ik->ij', matrix, tangent)
    da = (vt[:,None]*an-vn[:,None]*at)/(vt**2+vn**2)[:,None]
    dh = 2*np.einsum('ik,ijk->ij', np.cross(velocity,span), np.cross(matrix,span[:,None,:]))
    return ((.5*chord/cross_speed)[:,None]
            *(polar.cl_slope(alpha)[:,None]*speed2[:,None]*da+cl[:,None]*dh)
            - np.eye(len(gamma)))


def _vsm_closure_full(gamma, ambient, matrix, known, chord, tangent, normal, span, polar, onset=None):
    """Full-incidence closure (decision 0020): Gamma = W 1/2 c CL(alpha) |v x s|.

    ``v`` is the total three-quarter-chord velocity (ambient + current + history
    induction), ``alpha = atan2(v.n, v.t)`` over the full circle and the polar
    is a periodic table. Unlike the source closure the Kutta-Joukowski speed is
    the total transverse speed, so no ambient flow is required: Gamma -> 0
    continuously as the transverse speed vanishes. The weight W is one unless a
    declared model parameter is set (``SolveSettings``); ``onset`` is then the
    pair returned by ``onset_weight``: per-panel weights and the mask of panels
    switched off for span-wise onset flow. The weights are evaluated from the
    onset flow (ambient + history induction) before the solve, so they are
    constants of the circulation problem. The section coefficients and speed are
    returned unweighted: loads follow strip theory at the local total flow on
    every panel. Regimes per panel: 0 attached (inside the declared range, or
    undeclared), 1 post-stall, 2 reverse chordwise flow, 3 no sectional inflow
    (|v x s| at round-off of |v|, or switched off by the span-flow fraction; the
    angle is reported as 0 only in the round-off case). The regime is a label of
    the total flow; which panels had their circulation reduced is the weight
    itself (``VsmSolution.circulation_weight``). A panel switched off by the
    span-flow fraction keeps its strip load (dynamic pressure of |v x s|).
    """
    velocity = ambient + np.einsum('ijk,j->ik', matrix, gamma) + known
    tangential = np.sum(velocity*tangent, axis=1)
    vn = np.sum(velocity*normal, axis=1)
    transverse = np.cross(velocity, span)
    speed2 = _norm(transverse)**2
    speed = np.sqrt(speed2)
    total = _norm(velocity)
    no_inflow = speed <= 16*np.finfo(float).eps*total
    alpha = np.where(no_inflow, 0., np.arctan2(vn, tangential))
    cl, cd, cm = polar.coefficients(alpha)
    target = np.where(no_inflow, 0., .5*chord*cl*speed)
    regime = np.where(polar.attached(alpha), 0, 1).astype(np.int8)
    regime[tangential < 0] = 2
    if onset is not None:
        target = onset[0]*target
        regime[onset[1]] = 3
    regime[no_inflow] = 3
    return target, velocity, alpha, cl, cd, cm, speed2, regime


def onset_weight(onset_velocity, tangent, normal, span_fraction, reverse_fade=None):
    """Circulation weights of the declared regime parameters, from the onset flow.

    ``onset_velocity`` is the three-quarter-chord velocity without the current
    bound/near-wake induction (ambient + history). The sectional angle is
    measured in the section plane spanned by the chord tangent and the normal,
    so sigma = hypot(v0.t, v0.n)/|v0| is the fraction of the onset flow that
    defines it (on a swept bound line |v0 x s| can stay finite while both
    components vanish, which leaves the angle to the induction alone). The
    weight is a smoothstep from zero at sigma = f to one at 2f
    (``span_fraction`` f; no fade when f = 0). With ``reverse_fade`` (the
    'strip' option; None leaves reverse flow coupled) the weight is also
    multiplied by a smoothstep in the onset chordwise fraction
    cos(alpha0) = v0.t/hypot(v0.t, v0.n): one at 0, zero at -reverse_fade, so a
    sweep through +-90 deg has no jump. The weight depends on the direction of
    the onset flow only, not on its speed; a panel with exactly zero onset flow
    keeps weight one (its target vanishes with its own transverse speed).
    Returns (weight, span_off), span_off marking sigma <= f.
    """
    total = _norm(onset_velocity)
    vt = np.sum(onset_velocity*tangent, axis=1)
    vn = np.sum(onset_velocity*normal, axis=1)
    plane = np.hypot(vt, vn)
    moving = total > 0
    weight = np.ones(len(total))
    span_off = np.zeros(len(total), dtype=bool)
    if span_fraction > 0.:
        sigma = np.divide(plane, total, out=np.ones_like(plane), where=moving)
        x = np.clip((sigma-span_fraction)/span_fraction, 0., 1.)
        weight = x*x*(3.-2.*x)
        span_off = moving & (sigma <= span_fraction)
    if reverse_fade is not None:
        tau = np.divide(vt, plane, out=np.ones_like(vt), where=plane > 16*np.finfo(float).eps*np.where(moving, total, 1.))
        y = np.clip(1.+tau/reverse_fade, 0., 1.)
        weight = weight*y*y*(3.-2.*y)
    return weight, span_off


def _vsm_closure_full_jacobian(gamma, ambient, matrix, known, chord, tangent, normal, span, polar, onset=None):
    """Exact active-segment derivative of the full-incidence target minus Gamma."""
    velocity = ambient + np.einsum('ijk,j->ik', matrix, gamma) + known
    vt = np.sum(velocity*tangent, axis=1)
    vn = np.sum(velocity*normal, axis=1)
    transverse = np.cross(velocity, span)
    speed2 = _norm(transverse)**2
    active = ~(np.sqrt(speed2) <= 16*np.finfo(float).eps*_norm(velocity))
    alpha = np.where(active, np.arctan2(vn, vt), 0.)
    cl = polar.coefficients(alpha)[0]
    an = np.einsum('ijk,ik->ij', matrix, normal)
    at = np.einsum('ijk,ik->ij', matrix, tangent)
    # On a swept bound line v.t = v.n = 0 can occur with |v x s| > 0 (flow across the section
    # plane): the angle is atan2(0, 0) = 0 there and its derivative is taken as zero.
    plane2 = vt**2+vn**2
    plane2 = np.where(active & (plane2 > 0), plane2, 1.)
    speed = np.where(active, np.sqrt(speed2), 1.)
    da = (vt[:,None]*an-vn[:,None]*at)/plane2[:,None]
    ds = np.einsum('ik,ijk->ij', transverse, np.cross(matrix, span[:,None,:]))/speed[:,None]
    rows = (.5*chord)[:,None]*(polar.cl_slope(alpha)[:,None]*speed[:,None]*da+cl[:,None]*ds)
    if onset is not None:
        rows = onset[0][:,None]*rows
    rows[~active] = 0.
    return rows-np.eye(len(gamma))


def _full_incidence_loads(gamma, load_ambient, load_operator, known_load, wing, tangent, normal, span,
                          chord, cl, cd, cm, speed2, rho):
    """Loads of the full-incidence closure at the quarter-chord points.

    Directions follow the local load-point flow angle; a panel with no load-point
    flow in its chord-normal plane takes the chord direction (its force factor is
    then the closure's transverse speed squared, which is zero for a no-inflow
    panel). Nothing is clipped or rotated out of domain.
    """
    induction = np.einsum('ijk,j->ik', load_operator, gamma) + known_load
    load_velocity = load_ambient+induction
    if not np.isfinite(load_velocity).all():
        raise ValueError('nonfinite VSM load total flow')
    load_alpha = np.arctan2(np.sum(load_velocity*normal, axis=1), np.sum(load_velocity*tangent, axis=1))
    drag = np.cos(load_alpha)[:, None]*tangent+np.sin(load_alpha)[:, None]*normal
    drag /= _norm(drag)[:, None]
    lift = np.cross(drag, span)
    lift_norm = _norm(lift)
    if not np.isfinite(lift_norm).all() or np.any(lift_norm <= 16*np.finfo(float).eps):
        raise ValueError('VSM load direction is degenerate')
    lift /= lift_norm[:, None]
    drag = np.cross(span, lift)
    factor = .5*rho*speed2*chord*_norm(wing.dl)
    force = factor[:, None]*(cl[:, None]*lift+cd[:, None]*drag)
    couple = (factor*chord*cm)[:, None]*span
    return induction, load_velocity, load_alpha, force, couple


def solve_vsm_operator(wing, polar, inflow, operator_matrix, load_operator_matrix, *,
                       known_induction=(0.,0.,0.), known_load_induction=(0.,0.,0.),
                       structural_velocity=(0.,0.,0.), load_relative_ambient=None, density=1.225,
                       kinematic_viscosity=1.4639e-5, initial=None,
                       settings=SolveSettings(relaxation=.03, max_iterations=2000), diagnostics=None):
    """Solve the unchanged VSM closure using explicit current/history operators.

    ``operator_matrix`` maps current Gamma to induction at 3/4 chord and must
    already contain the diagonal two-dimensional self correction. The load
    operator maps Gamma to induction at 1/4 chord with all bound span filaments
    omitted. Known-history induction is provided separately at each point set.
    Operators have shape (panels,panels,3), or may be lazy builders accepting
    the nominal ambient direction and mean speed for the straight-wake wrapper.
    ``load_relative_ambient`` changes only the load-point velocity/direction;
    polar coefficients, force magnitudes and Reynolds number use the unchanged
    three-quarter-chord closure. It must already exclude body velocity.
    ``settings.closure='full_incidence'`` (decision 0020) replaces the forward-
    flow domain checks and the source target by ``_vsm_closure_full`` and
    ``_full_incidence_loads``; the default ``'source'`` path is unchanged.
    """
    rho, nu = positive(density, 'density'), positive(kinematic_viscosity, 'viscosity')
    if not isinstance(settings, SolveSettings):
        raise ValueError('settings must be SolveSettings')
    if settings.jacobian == 'active_segment':
        from .wing import SectionPolar
        from .aircraft import PanelPolars
        if not isinstance(polar, (SectionPolar, PanelPolars)):
            raise ValueError('active-segment VSM Jacobian requires declared section tables')
    cp, chord, normal, tangent, span = geometry(wing)
    ambient = vector_field(inflow, wing.panels, 'inflow')-vector_field(structural_velocity, wing.panels, 'structural velocity')
    load_ambient = (ambient if load_relative_ambient is None else
                    vector_field(load_relative_ambient, wing.panels, 'load relative ambient'))
    load_sampling = 'legacy_closure_reuse' if load_relative_ambient is None else 'separate_load_input_v1'
    full_incidence = settings.closure == 'full_incidence'
    known = vector_field(known_induction, wing.panels, 'known VSM induction')
    known_load = vector_field(known_load_induction, wing.panels, 'known VSM load induction')
    speed = _norm(ambient)
    cross_speed = _norm(np.cross(ambient, span))
    gamma = np.zeros(wing.panels) if initial is None else _real(initial, 'initial circulation').copy()
    if gamma.shape != (wing.panels,):
        raise ValueError('initial circulation shape mismatch')
    if full_incidence:
        # Full circle, any mix of zero and nonzero panel inflow (hover, slipstream
        # edges, reverse flow). The prescribed-wake direction follows the mean
        # ambient flow, else the mean history induction, else +x.
        if not np.isfinite(ambient).all() or not np.isfinite(load_ambient).all() or not np.isfinite(known).all():
            raise ValueError('nonfinite VSM inflow')
        direction = np.sum(ambient[speed > 0]/speed[speed > 0, None], axis=0) if np.any(speed) else np.sum(known, axis=0)
        direction = np.array([1., 0., 0.]) if _norm(direction) == 0 else direction/_norm(direction)
    else:
        _check_load_flow(load_ambient, tangent, span, 'ambient', allow_zero=True)
        if np.any((speed > 0) & (cross_speed <= 16*np.finfo(float).eps*speed)):
            raise ValueError('nonzero pure-span flow is outside this VSM contract')
        if np.any((speed > 0) & (np.sum(ambient*tangent, axis=1) <= 0)):
            raise ValueError('VSM requires forward chordwise flow')
        if np.any(speed == 0) and np.any(speed > 0):
            raise ValueError('mixed zero/nonzero panel inflow is outside this VSM contract')
        direction = np.array([1.,0.,0.]) if not np.any(speed) else np.sum(ambient/speed[:, None], axis=0)
        if _norm(direction) == 0:
            raise ValueError('ambiguous prescribed-wake direction')
        direction = direction/_norm(direction)
        if not np.any(speed):
            if np.any(known) or np.any(known_load):
                raise ValueError('VSM history with zero ambient sectional speed is outside this closure')
            zeros = np.zeros(wing.panels)
            vectors = np.zeros((wing.panels, 3))
            load_alpha = np.arctan2(np.sum(load_ambient*normal, axis=1), np.sum(load_ambient*tangent, axis=1))
            return VsmSolution(zeros, load_ambient, vectors, load_alpha, zeros, zeros, zeros, zeros,
                               vectors, vectors, 1, 0., settings.absolute_tolerance, zeros, cp, direction,
                               closure_ambient_velocity=ambient, load_ambient_sampling=load_sampling)
    def resolved(value, name):
        value = value(direction, float(np.mean(speed))) if callable(value) else value
        matrix = _real(value, name)
        if matrix.shape != (wing.panels, wing.panels, 3):
            raise ValueError(f'{name} must have shape (panels,panels,3)')
        return matrix
    matrix = resolved(operator_matrix, 'VSM closure operator')
    if full_incidence:
        onset = None
        if settings.span_flow_fraction > 0. or settings.reverse_flow == 'strip':
            onset = onset_weight(ambient+known, tangent, normal, settings.span_flow_fraction,
                                 settings.reverse_fade if settings.reverse_flow == 'strip' else None)
        closure_args = (ambient, matrix, known, chord, tangent, normal, span, polar, onset)
        closure_function, closure_jacobian = _vsm_closure_full, _vsm_closure_full_jacobian
    else:
        closure_args = (ambient, matrix, known, chord, tangent, normal, span, cross_speed, polar)
        closure_function, closure_jacobian = _vsm_closure, _vsm_closure_jacobian
    newton_iterations = 0
    solver_trace = None
    if settings.solver in ('newton','newton_picard'):
        from .nonlinear import solve_newton, solve_newton_picard, section_solver_trace
        def target(value):
            return closure_function(value, *closure_args)[0]
        jacobian = None
        if settings.jacobian == 'active_segment':
            def jacobian(value):
                return closure_jacobian(value, *closure_args)
            if diagnostics is None:
                diagnostics = {}
        if settings.solver == 'newton':
            seed = gamma.copy()
            gamma, newton_iterations = solve_newton(target, gamma, settings,
                                                    jacobian=jacobian, diagnostics=diagnostics)
            if jacobian is not None:
                import json
                maximum = float(np.max(abs(target(gamma)-gamma)))
                limit = float(settings.absolute_tolerance+settings.relative_tolerance*np.mean(abs(gamma)))
                solver_trace = json.dumps(dict(schema='newton_v1',policy='newton_residual_root_v1',
                    jacobian=settings.jacobian,updates=diagnostics['updates'],selected_solver='newton',
                    relaxation=0.,fallback_used=False,attempts=[dict(solver='newton',status='accepted',
                    iterations=newton_iterations,residual_m2_s=maximum,residual_limit_m2_s=limit)],
                    delta_seed_linf_m2_s=float(np.max(abs(gamma-seed))),
                    stability='not_enforced; active-segment root does not establish uniqueness'),sort_keys=True)
        else:
            gamma, newton_iterations, solver_trace = solve_newton_picard(target, gamma, settings,
                                                        jacobian=jacobian, diagnostics=diagnostics)
    for iteration in range(1, settings.max_iterations+1):
        target, velocity, alpha, cl, cd, cm, speed2, *regime = closure_function(gamma, *closure_args)
        residual = float(np.max(np.abs(target-gamma)))
        limit = settings.absolute_tolerance+settings.relative_tolerance*float(np.mean(np.abs(gamma)))
        if residual <= limit and full_incidence:
            induction, load_velocity, load_alpha, force, couple = _full_incidence_loads(
                gamma, load_ambient, resolved(load_operator_matrix, 'VSM load operator'), known_load, wing,
                tangent, normal, span, chord, cl, cd, cm, speed2, rho)
            if solver_trace is not None:
                solver_trace = section_solver_trace(solver_trace, polar, alpha, residual, limit)
            return VsmSolution(gamma, load_velocity, induction, load_alpha,
                               chord*np.sqrt(speed2)/nu, cl, cd, cm, force, couple,
                               newton_iterations or iteration, residual, limit, alpha, cp, direction,
                               closure_ambient_velocity=ambient, load_ambient_sampling=load_sampling,
                               solver_trace_json=solver_trace, closure='full_incidence', panel_regime=regime[0],
                               circulation_weight=None if closure_args[-1] is None else closure_args[-1][0])
        if residual <= limit:
            induction = np.einsum('ijk,j->ik', resolved(load_operator_matrix, 'VSM load operator'), gamma) + known_load
            load_velocity = load_ambient+induction
            _check_load_flow(load_velocity, tangent, span, 'total')
            load_alpha = np.arctan2(np.sum(load_velocity*normal, axis=1), np.sum(load_velocity*tangent, axis=1))
            drag = np.cos(load_alpha)[:, None]*tangent+np.sin(load_alpha)[:, None]*normal
            drag /= _norm(drag)[:, None]
            lift = np.cross(drag, span)
            lift_norm = _norm(lift)
            if not np.isfinite(lift_norm).all() or np.any(lift_norm <= 16*np.finfo(float).eps):
                raise ValueError('VSM load direction is degenerate')
            lift /= lift_norm[:, None]
            # Local orthogonal direction replaces the parent's global-y shortcut.
            drag = np.cross(span, lift)
            factor = .5*rho*speed2*chord*_norm(wing.dl)
            force = factor[:, None]*(cl[:, None]*lift+cd[:, None]*drag)
            couple = (factor*chord*cm)[:, None]*span
            if solver_trace is not None:
                solver_trace = section_solver_trace(solver_trace, polar, alpha, residual, limit)
            return VsmSolution(gamma, load_velocity, induction, load_alpha,
                               chord*np.sqrt(speed2)/nu, cl, cd, cm, force, couple,
                               newton_iterations or iteration, residual, limit, alpha, cp, direction,
                               closure_ambient_velocity=ambient, load_ambient_sampling=load_sampling,
                               solver_trace_json=solver_trace)
        if solver_trace is not None:
            from .nonlinear import HybridLoadReevaluationError
            raise HybridLoadReevaluationError(solver_trace, residual, limit, diagnostics=diagnostics)
        if settings.solver == 'newton':
            from .nonlinear import NonlinearSolveError
            raise NonlinearSolveError('Newton root failed load reevaluation',
                                      iterations=newton_iterations, diagnostics=diagnostics)
        gamma += settings.relaxation*(target-gamma)
        if not np.isfinite(gamma).all():
            raise ValueError('nonfinite VSM iterate')
    raise CirculationConvergenceError(iteration, residual, limit)


def _check_load_flow(velocity, tangent, span, kind, *, allow_zero=False):
    """Apply the declared forward-flow domain at the separate load points.

    This guard never clips angles or rotates an out-of-domain force. An all-zero
    ambient is allowed: the zero-closure branch returns zero forces, while an
    active closure must still have a well-defined total load direction.
    """
    speed = _norm(velocity)
    cross_speed = _norm(np.cross(velocity, span))
    if not np.isfinite(velocity).all() or not np.isfinite(speed).all() or not np.isfinite(cross_speed).all():
        raise ValueError(f'nonfinite VSM load {kind} flow')
    if allow_zero and not np.any(speed):
        return
    if np.any(speed == 0):
        raise ValueError(f'zero VSM load {kind} flow has no sectional direction')
    if np.any(cross_speed <= 16*np.finfo(float).eps*speed):
        raise ValueError(f'pure-span VSM load {kind} flow is outside this contract')
    if np.any(np.sum(velocity*tangent, axis=1) <= 0):
        raise ValueError(f'VSM load {kind} requires forward chordwise flow')
