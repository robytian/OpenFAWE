"""Finite prescribed-wake lifting-line adapter; unchanged legacy LL physics.

See docs/tasks/OF-0006-11/contract.md. A sample is a fresh steady solution;
there is no wake release, time-history loading, convection or material aging.
"""
from dataclasses import dataclass
import numpy as np

from .wing import owned, positive, vector_field, influence_matrix, solve_circulation
from .wake import WakeConfig, ambient_velocity, count, pack_rings
from .vortex import _real, _norm, segments_velocity, CoreModel


@dataclass(frozen=True)
class LLStraightConfig:
    wake_length_m: float = 1000.
    wake_direction: tuple | None = None

    def __post_init__(self):
        positive(self.wake_length_m, 'straight LL wake length')
        if self.wake_direction is not None:
            value = _real(self.wake_direction, 'straight LL wake direction')
            if value.shape != (3,) or _norm(value) == 0:
                raise ValueError('straight LL wake direction must be a nonzero three-vector')
            object.__setattr__(self, 'wake_direction', tuple(value / _norm(value)))


def validate_configuration(config, settings):
    if not isinstance(config, WakeConfig) or not isinstance(settings, LLStraightConfig):
        raise ValueError('invalid straight LL configuration')
    if config.core_evolution != 'constant' or not config.free_wake:
        raise ValueError('material aging or ambient-only convection does not apply to straight LL')
    if config.wake_core_chord_fraction is not None:
        raise ValueError('local material birth cores do not apply to straight LL')


@dataclass(frozen=True)
class LLStraightState:
    time_s: float
    step: int
    wing: object
    solution: object
    structural_velocity: np.ndarray
    prescribed_wake_row: np.ndarray
    config: WakeConfig
    settings: LLStraightConfig

    def __post_init__(self):
        if _real(self.time_s, 'state time').shape:
            raise ValueError('state time must be scalar')
        count(self.step, 'step')
        validate_configuration(self.config, self.settings)
        object.__setattr__(self, 'structural_velocity', owned(vector_field(
            self.structural_velocity, self.wing.panels, 'structural velocity')))
        row = _real(self.prescribed_wake_row, 'prescribed wake row')
        if row.shape != self.wing.trailing_edge.shape:
            raise ValueError('prescribed wake row topology differs from wing')
        object.__setattr__(self, 'prescribed_wake_row', owned(row))

    @property
    def free_rows(self):
        return np.empty((0, self.wing.node_count, 3))

    @property
    def release_times_s(self):
        return np.empty(0)

    @property
    def older_circulation(self):
        return np.empty((0, self.wing.panels))


def steady_state(wing, polar, config, inflow, *, settings=LLStraightConfig(),
                 time_s=0., step=0, structural_velocity=(0., 0., 0.)):
    validate_configuration(config, settings)
    wind = ambient_velocity(inflow, wing.cp, time_s)
    body = ambient_velocity(structural_velocity, wing.cp, time_s)
    direction = np.mean(wind - body, axis=0) if settings.wake_direction is None else np.array(settings.wake_direction)
    if _norm(direction) == 0:
        raise ValueError('zero mean relative flow needs an explicit straight wake direction')
    row = wing.trailing_edge + settings.wake_length_m * direction / _norm(direction)
    from .wake import check_straight_surface
    check_straight_surface(wing, row, config, time_s)
    matrix = influence_matrix(wing, [row], model=config.core_model, core_radius=config.core_radius_m)
    solution = solve_circulation(wing, polar, wind, matrix, structural_velocity=body,
        density=config.density_kg_m3, kinematic_viscosity=config.kinematic_viscosity_m2_s,
        settings=config.solve)
    return LLStraightState(time_s, step, wing, solution, body, row, config, settings)


def advance_steady(state, polar, config, inflow, *, settings=LLStraightConfig(),
                   next_wing=None, structural_velocity=(0., 0., 0.), next_time_s=None):
    if type(state) is not LLStraightState or config != state.config or settings != state.settings:
        raise ValueError('straight LL state/configuration changed; initialize a new run')
    time = state.time_s + config.dt_s
    if not np.isfinite(time) or time <= state.time_s:
        raise ValueError('time cannot advance')
    if next_time_s is not None:
        value = _real(next_time_s, 'next time')
        if value.shape or not np.isclose(value.item(), time, rtol=0.,
                atol=8*np.finfo(float).eps*max(1., abs(time))) or value.item() <= state.time_s:
            raise ValueError('next time must equal previous time plus sample dt')
        time = value.item()
    wing = state.wing if next_wing is None else next_wing
    if wing.node_count != state.wing.node_count or not np.array_equal(wing.panel_edges, state.wing.panel_edges):
        raise ValueError('wing topology cannot change during a run')
    return steady_state(wing, polar, config, inflow, settings=settings, time_s=time,
        step=state.step+1, structural_velocity=structural_velocity)


def straight_network(state):
    rows = np.array([state.wing.bound, state.wing.trailing_edge, state.prescribed_wake_row])
    strengths = np.repeat(state.solution.circulation[None, :], 2, axis=0)
    return pack_rings(rows, strengths, state.wing.panel_edges)


def network_radii(state):
    value = 0. if state.config.core_model == CoreModel.NONE else state.config.core_radius_m
    return np.full(len(straight_network(state)[1]), value)


def induced_velocity(state, points):
    points = _real(points, 'probe points')
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('probe points must have shape (n,3)')
    nodes, edges, gamma = straight_network(state)
    return segments_velocity(points, nodes, edges, gamma, model=state.config.core_model,
        core_radius=network_radii(state))
