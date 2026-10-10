"""Explicit aerodynamic-method selection with immutable, separate states."""

from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .wing import SolveSettings, owned, vector_field, solve_settings_metadata, wake_config_metadata
from .wake import WakeConfig, WakeState, start_wake, advance_wake, ambient_velocity, count, wake_network, validate_state_config, kernel_options
from .vortex import _real, _norm, segments_velocity
from .vsm import geometry, solve_vsm, horseshoe_influence, core_kernel_identity
from .vsm_wake import VsmWakeState, start_vsm_wake, advance_vsm_wake
from .cores import check_cores, edge_radii
from .lifting_surface import (LiftingSurfaceConfig, VlmState, VlmWakeState,
    build_lifting_surface_mesh, start_lifting_surface, advance_lifting_surface,
    lifting_surface_induced_velocity)
from .lifting_surface import validate_lifting_surface_wake
from .ll_straight import (LLStraightConfig, LLStraightState, steady_state,
    advance_steady, validate_configuration, induced_velocity as straight_ll_induction)


def rigid_velocity(points, translation_velocity, angular_velocity, origin=(0.,0.,0.)):
    """SI rigid-body velocity at the backend's own evaluation points."""
    points = _real(points, 'points')
    vectors = [_real(v, n) for v, n in [(translation_velocity,'translation velocity'),
                                       (angular_velocity,'angular velocity'), (origin,'origin')]]
    if points.ndim != 2 or points.shape[1] != 3 or any(v.shape != (3,) for v in vectors):
        raise ValueError('points (n,3) and velocity/omega/origin (3,) required')
    return vectors[0] + np.cross(vectors[1], points-vectors[2])


def rigid_velocity_field(translation_velocity, angular_velocity, origin=(0.,0.,0.)):
    """Snapshot spatial callable for every backend and every load-point set.

    Supply a new field when the prescribed velocity or rotation origin changes.
    This does not infer a geometry trajectory or interpolate arbitrary CP data.
    """
    translation, angular, center = (owned(v) for v in
                                    (translation_velocity, angular_velocity, origin))
    rigid_velocity(np.empty((0, 3)), translation, angular, center)
    return lambda points, time: rigid_velocity(points, translation, angular, center)


class AeroMethod(str, Enum):
    FREE_WAKE = 'free_wake'
    VSM = 'vsm'
    VSM_FREE_WAKE = 'vsm_free_wake'
    VLM = 'vlm'
    VLM_FREE_WAKE = 'vlm_free_wake'
    LL_STRAIGHT = 'll_straight'


@dataclass(frozen=True)
class VsmState:
    time_s: float
    step: int
    wing: object
    solution: object
    structural_velocity: np.ndarray
    config: WakeConfig
    settings: SolveSettings
    core_policy: str

    def __post_init__(self):
        time = _real(self.time_s, 'state time')
        if time.shape:
            raise ValueError('time must be a scalar')
        count(self.step, 'step')
        object.__setattr__(self, 'structural_velocity', owned(
            vector_field(self.structural_velocity, self.wing.panels, 'structural velocity')))

    @property
    def free_rows(self):
        return np.empty((0, self.wing.node_count, 3))

    @property
    def release_times_s(self):
        return np.empty(0)

    @property
    def older_circulation(self):
        return np.empty((0, self.wing.panels))


@dataclass(frozen=True)
class AeroBackend:
    """Select once per run; initialize a new run to choose another method.

    Uniform vectors and spatial callables are the portable input contract.
    Lattice methods also sample edge-load points and require callables for
    nonuniform fields; collocation-only arrays cannot define those velocities.
    Legacy line methods additionally accept arrays at evaluation_points(wing).
    """
    method: AeroMethod = AeroMethod.FREE_WAKE
    wake: WakeConfig = WakeConfig()
    vsm_solve: SolveSettings = SolveSettings(relaxation=.03, max_iterations=2000)
    vsm_core_policy: str | None = None
    lifting_surface: LiftingSurfaceConfig = LiftingSurfaceConfig()
    ll_straight: LLStraightConfig = LLStraightConfig()

    def __post_init__(self):
        object.__setattr__(self, 'method', AeroMethod(self.method))
        if self.vsm_core_policy is None:
            object.__setattr__(self, 'vsm_core_policy',
                'source_legacy' if self.method == AeroMethod.VSM else 'none')
        if not isinstance(self.wake, WakeConfig) or not isinstance(self.vsm_solve, SolveSettings):
            raise ValueError('invalid aerodynamic backend settings')
        if self.method != AeroMethod.VSM and self.vsm_solve.jacobian != 'finite_difference':
            raise ValueError('active-segment Jacobian is supported only by straight VSM')
        if not isinstance(self.lifting_surface, LiftingSurfaceConfig):
            raise ValueError('lifting_surface must be LiftingSurfaceConfig')
        if (self.method != AeroMethod.VLM_FREE_WAKE
                and self.lifting_surface.bound_convection_core != 'legacy_zero'):
            raise ValueError('attached convection core settings require vlm_free_wake')
        if not isinstance(self.ll_straight, LLStraightConfig):
            raise ValueError('ll_straight must be LLStraightConfig')
        if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE):
            validate_lifting_surface_wake(self.lifting_surface, self.wake,
                                         self.method == AeroMethod.VLM_FREE_WAKE)
        if self.method == AeroMethod.LL_STRAIGHT:
            validate_configuration(self.wake, self.ll_straight)
        if (self.method in (AeroMethod.LL_STRAIGHT, AeroMethod.VSM, AeroMethod.VLM)
                and self.wake.wake_core_chord_fraction is not None):
            raise ValueError('local material birth cores require a released wake')
        if self.method == AeroMethod.VLM and self.wake.core_evolution != 'constant':
            raise ValueError('released material core evolution does not apply to straight VLM')
        if self.method == AeroMethod.VLM and (self.lifting_surface.wake_release != 'previous'
                or self.lifting_surface.gamma_derivative != 'backward_euler'):
            raise ValueError('transient time settings do not apply to straight VLM')
        if self.method == AeroMethod.VLM and not self.wake.free_wake:
            raise ValueError('ambient-only released-wake option does not apply to straight VLM')
        if self.method == AeroMethod.VSM and not self.wake.free_wake:
            raise ValueError('ambient-only released-wake option does not apply to VSM')
        if self.method == AeroMethod.VSM and self.wake.core_evolution != 'constant':
            raise ValueError('released material core evolution does not apply to VSM')
        core_kernel_identity(self.vsm_core_policy)
        if self.method != AeroMethod.VSM and self.vsm_core_policy != 'none':
            raise ValueError('source VSM core policies belong only to straight VSM')

    def evaluation_points(self, wing):
        """Closure points, not a complete spatial field contract for lattices."""
        if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE):
            return build_lifting_surface_mesh(wing, self.lifting_surface).collocation_points.reshape(-1, 3)
        return (owned(geometry(wing)[0])
                if self.method in (AeroMethod.VSM, AeroMethod.VSM_FREE_WAKE) else wing.cp)

    def induced_velocity(self, state, points):
        """Read-only off-body velocity probes; singular filament points raise."""
        points = _real(points, 'probe points')
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('probe points must have shape (n,3)')
        if self.method == AeroMethod.LL_STRAIGHT:
            if type(state) is not LLStraightState or state.config != self.wake or state.settings != self.ll_straight:
                raise ValueError('state does not belong to selected straight LL configuration')
            return straight_ll_induction(state, points)
        if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE):
            expected = VlmState if self.method == AeroMethod.VLM else VlmWakeState
            if type(state) is not expected:
                raise ValueError('state does not belong to selected backend')
            if state.settings != self.lifting_surface:
                raise ValueError('lifting-surface settings changed; start a new run')
            return lifting_surface_induced_velocity(state, points, self.wake)
        if self.method == AeroMethod.VSM:
            if type(state) is not VsmState:
                raise ValueError('state does not belong to selected backend')
            self._check_vsm_config(state)
            speed = float(np.mean(_norm(state.solution.closure_ambient_velocity)))
            if not np.any(state.solution.circulation):
                return np.zeros_like(points)
            matrix = horseshoe_influence(state.wing, state.solution.wake_direction, points,
                source_core_speed=speed if self.vsm_core_policy!='none' else None,
                source_core_policy=self.vsm_core_policy)
            return np.einsum('ijk,j->ik', matrix, state.solution.circulation)
        expected = VsmWakeState if self.method == AeroMethod.VSM_FREE_WAKE else WakeState
        if type(state) is not expected:
            raise ValueError('state does not belong to selected backend')
        validate_state_config(state, self.wake)
        check_cores(state.cores, self.wake)
        nodes, edges, gamma = wake_network(state)
        return segments_velocity(points, nodes, edges, gamma, model=self.wake.core_model,
            core_radius=edge_radii(state.wing, edges, state.cores, self.wake.core_radius_m),
            **kernel_options(self.wake, large=True))

    def metadata(self):
        common = {'method': self.method.value, 'dt_s': self.wake.dt_s,
                  'density_kg_m3': self.wake.density_kg_m3,
                  'kinematic_viscosity_m2_s': self.wake.kinematic_viscosity_m2_s,
                  'spatial_field_contract': 'Uniform (3,) or callable(points_m, time_s); lattice methods require callable for nonuniform fields',
                  'stage_bound_interpolation': 'linear_endpoint_geometry; exact for translation, approximate for rotation',
                  'convection_order_scope': 'Frozen circulation/core geometry ODE only; not coupled time order',
                  'force_model': ('bound_edge_joukowski' if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE)
                                  else 'quasi_steady_sectional_polar'),
                  'unsteady_term': 'rho_dgamma_dt_area_normal' if self.method == AeroMethod.VLM_FREE_WAKE else 'none'}
        if self.method == AeroMethod.LL_STRAIGHT:
            common.update(ll_straight=asdict(self.ll_straight), wake=wake_config_metadata(self.wake),
                closure='Existing nonlinear quarter-chord lifting line with section polars',
                wake_history=False, prescribed_wake='finite straight rings with far closing edge',
                force_model='Existing LL sectional Kutta-Joukowski, profile drag and polar moment',
                collocation_axis_convention='Existing influence_matrix zero-on-axis convention',
                time_meaning='Independent quasi-steady samples; no transient wake history',
                contract='docs/tasks/OF-0006-11/contract.md')
            return common
        if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE):
            free = self.method == AeroMethod.VLM_FREE_WAKE
            common.update(lifting_surface=asdict(self.lifting_surface),
                closure='Linear nonpenetration at each chordwise/spanwise thin-surface panel',
                polar_usage='not used: inviscid thin surface; optional explicit camber, no profile drag or stall',
                wake_history=free, wake=wake_config_metadata(self.wake),
                prescribed_wake=None if free else 'straight finite wake; length is a solver parameter',
                force_model='Bound-edge Joukowski forces; free-wake circulation-rate normal term',
                boundary_residual_unit='m/s',
                contract='docs/tasks/OF-0006-10/algorithm.md')
            if self.lifting_surface.bound_convection_core != 'all_bound_chord':
                common['lifting_surface'].pop('bound_convection_chord_fraction')
            if self.lifting_surface.polar_coupling != 'none':
                common.update(closure='Nonlinear source-polar closure with virtual boundary normals',
                    polar_usage='Cl circulation closure; source Cd and Cm minus grid reference Cm',
                    force_model='Bound-edge and physical unsteady loads plus sectional drag/moment corrections',
                    contract='docs/tasks/OF-0006-polarcoupled20260924/contract.md')
            return common
        if self.method in (AeroMethod.FREE_WAKE, AeroMethod.VSM_FREE_WAKE):
            vsm = self.method == AeroMethod.VSM_FREE_WAKE
            common.update(wake=wake_config_metadata(self.wake),
                closure=('VSM 3/4-chord closure with material finite free wake; 1/4-chord trailing-only load direction'
                         if vsm else 'OLAF-derived quarter-chord lifting line'),
                wake_history=True,
                contract=('docs/tasks/OF-0006-07/vsm-fvw.md' if vsm
                          else 'docs/design/of-0006-wing-wake-contract.md'))
            if vsm:
                common.update(core_policy='WakeConfig material finite-filament cores',
                    self_correction='inviscid infinite-bound sectional correction',
                    method_origin='OpenFAWE combination; not an upstream KiteFAST free-wake implementation')
        else:
            common.update(solve=solve_settings_metadata(self.vsm_solve),
                          closure=('full-incidence VSM 3/4-chord closure (decision 0020); 1/4-chord load direction'
                                   if self.vsm_solve.closure == 'full_incidence' else
                                   'source VSM 3/4-chord closure; 1/4-chord load direction'),
                          wake_history=False, prescribed_wake='five-filament, straight semi-infinite',
                          core_policy=self.vsm_core_policy,
                          source_core_parameters={'bound_radius_length_fraction':.05,'trailing_kinematic_viscosity_m2_s':1.48e-5,'oseen_alpha':1.25643} if self.vsm_core_policy=='source_legacy' else None,
                          contract='docs/design/of-0006-aero-method-contract.md')
            if self.vsm_core_policy == 'source_pinned_v1':
                common['source_core_parameters'] = dict(bound_radius_length_fraction=.05,
                    trailing_kinematic_viscosity_m2_s=1.48e-5,oseen_alpha=1.25643,
                    source_revision='5485b823f33c76be70cf0569204c2e9c7be976c5',
                    source_module='src/VSM/core/Filament.py',
                    cutoff='sqrt(4*alpha*nu*radial_distance/speed)',
                    finite_inner='radial projection with linear radius/epsilon multiplier; trailing magnitude tends to nonzero constant',
                    ray_inner='boundary-only evaluation; can diverge as radial_distance^-1/2',
                    axis='reject finite axis and ray radius <=1e-16 m; native axis conventions not adopted',
                    argument_map='OpenFAWE unweighted mean ambient speed and averaged unit direction; active pinned Body AIC uses one area-weighted RMS speed and area-weighted mean vector direction; uniform relative inflow agrees',
                    qualification='source replication on stated nonaxis domain, not material-age diffusion or physical regularization')
                common['core_kernel_identity'] = core_kernel_identity(self.vsm_core_policy)
                common['contract'] = 'docs/design/vsm-source-core-profiles.md'
        if self.method in (AeroMethod.VSM, AeroMethod.VSM_FREE_WAKE):
            common['load_point_ambient_sampling'] = (
                'Spatial callables sampled separately at quarter chord; legacy panel arrays reused as supplied')
        return common

    def start(self, wing, polar, inflow, *, time_s=0., structural_velocity=(0.,0.,0.),
              initial_gamma_rate=None):
        if initial_gamma_rate is not None and self.method != AeroMethod.VLM_FREE_WAKE:
            raise ValueError('explicit initial Gamma rate requires transient lifting surface')
        if self.method == AeroMethod.LL_STRAIGHT:
            return steady_state(wing, polar, self.wake, inflow, settings=self.ll_straight,
                time_s=time_s, structural_velocity=structural_velocity)
        if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE):
            return start_lifting_surface(wing, self.wake, inflow,
                settings=self.lifting_surface, polar=polar, free_wake=self.method == AeroMethod.VLM_FREE_WAKE,
                time_s=time_s, structural_velocity=structural_velocity, initial_gamma_rate=initial_gamma_rate)
        if self.method == AeroMethod.FREE_WAKE:
            return start_wake(wing, polar, self.wake, inflow, time_s=time_s,
                              structural_velocity=structural_velocity)
        if self.method == AeroMethod.VSM_FREE_WAKE:
            return start_vsm_wake(wing, polar, self.wake, inflow, time_s=time_s,
                                  structural_velocity=structural_velocity)
        return self._vsm_state(wing, polar, inflow, time_s, 0, structural_velocity)

    def _vsm_state(self, wing, polar, inflow, time_s, step, structural_velocity):
        closure_inflow = ambient_velocity(inflow, self.evaluation_points(wing), time_s)
        closure_body = ambient_velocity(structural_velocity, self.evaluation_points(wing), time_s)
        load_relative_ambient = (ambient_velocity(inflow, wing.cp, time_s)
                                 - ambient_velocity(structural_velocity, wing.cp, time_s))
        result = solve_vsm(wing, polar, closure_inflow,
                           structural_velocity=closure_body, load_relative_ambient=load_relative_ambient,
                           density=self.wake.density_kg_m3,
                           kinematic_viscosity=self.wake.kinematic_viscosity_m2_s, settings=self.vsm_solve,
                           core_policy=self.vsm_core_policy)
        if self.wake.surface_guard != 'off':
            from .wake import make_surface_guard
            guard = make_surface_guard(wing, self.wake)
            extent = 4*np.linalg.norm(np.ptp(guard.vertices, axis=0))
            row = wing.trailing_edge+extent*result.wake_direction
            guard.check_rows(wing.trailing_edge[None], row[None], wing,
                             stage='prescribed_straight_rays', time_s=time_s)
        return VsmState(time_s, step, wing, result, closure_body,
                        self.wake, self.vsm_solve, self.vsm_core_policy)

    def _check_vsm_config(self, state):
        if (state.config != self.wake or state.settings != self.vsm_solve
                or state.core_policy != self.vsm_core_policy):
            raise ValueError('VSM state configuration changed; start a new run')

    def advance(self, state, polar, inflow, *, next_wing=None,
                structural_velocity=(0.,0.,0.), next_time_s=None):
        expected = {AeroMethod.FREE_WAKE: WakeState, AeroMethod.VSM: VsmState,
                    AeroMethod.VSM_FREE_WAKE: VsmWakeState, AeroMethod.VLM: VlmState,
                    AeroMethod.VLM_FREE_WAKE: VlmWakeState, AeroMethod.LL_STRAIGHT: LLStraightState}[self.method]
        if type(state) is not expected:
            raise ValueError('cannot pass a live state between aerodynamic methods; start a new run')
        if self.method == AeroMethod.LL_STRAIGHT:
            return advance_steady(state, polar, self.wake, inflow, settings=self.ll_straight,
                next_wing=next_wing, structural_velocity=structural_velocity, next_time_s=next_time_s)
        if self.method in (AeroMethod.VLM, AeroMethod.VLM_FREE_WAKE):
            if state.settings != self.lifting_surface:
                raise ValueError('lifting-surface settings changed; start a new run')
            return advance_lifting_surface(state, self.wake, inflow, polar=polar,
                next_wing=next_wing, structural_velocity=structural_velocity,
                next_time_s=next_time_s)
        if self.method == AeroMethod.FREE_WAKE:
            return advance_wake(state, polar, self.wake, inflow, next_wing=next_wing,
                                structural_velocity=structural_velocity, next_time_s=next_time_s)
        if self.method == AeroMethod.VSM_FREE_WAKE:
            return advance_vsm_wake(state, polar, self.wake, inflow, next_wing=next_wing,
                structural_velocity=structural_velocity, next_time_s=next_time_s)
        self._check_vsm_config(state)
        time = state.time_s+self.wake.dt_s
        if not np.isfinite(time) or time <= state.time_s:
            raise ValueError('time cannot advance')
        if next_time_s is not None:
            requested = _real(next_time_s, 'next time')
            if requested.shape or not np.isclose(requested.item(), time, rtol=0,
                                                  atol=8*np.finfo(float).eps*max(1, abs(time))):
                raise ValueError('next time must equal old time plus configured dt')
            if requested.item() <= state.time_s:
                raise ValueError('requested time must advance the finite state time')
            time=requested.item()
        wing = state.wing if next_wing is None else next_wing
        if wing.node_count != state.wing.node_count or not np.array_equal(wing.panel_edges,state.wing.panel_edges):
            raise ValueError('panel count cannot change during a run')
        return self._vsm_state(wing, polar, inflow, time, state.step+1, structural_velocity)
