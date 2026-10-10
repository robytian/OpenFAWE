"""Material core spreading and optional constant-log-strain integration.

Age growth follows the OLAF differential law, integrated exactly in rc squared.
Stretch coupling and midpoint birth age are explicit OpenFAWE conventions;
OLAF v5.0.0 leaves its stretching branch unimplemented. See M11-M13 contract.
"""

from dataclasses import dataclass
import numpy as np

from .vortex import _real, _norm
from .wing import owned, positive

OSEEN_ALPHA = 1.25643


def evolve_radius(radius, old_length, new_length, dt, diffusion_rate, *, stretching=False):
    """Integrate q'=D-q*d(log L)/dt, with q=radius squared, D in m2/s."""
    r, old, new = np.broadcast_arrays(*[_real(v, n) for v, n in
                                       [(radius, 'radius'), (old_length, 'old length'),
                                        (new_length, 'new length')]])
    dt = positive(dt, 'core dt', zero=True)
    rate = positive(diffusion_rate, 'diffusion rate', zero=True)
    if not isinstance(stretching, bool) or np.any(r <= 0) or np.any(old <= 0) or np.any(new <= 0):
        raise ValueError('positive cores/lengths and a boolean stretching switch required')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            strain = np.log(new)-np.log(old) if stretching else np.zeros_like(r)
            factor = np.ones_like(strain)
            np.divide(-np.expm1(-strain), strain, out=factor, where=strain != 0)
            result = np.sqrt(r*r*np.exp(-strain)+rate*dt*factor)
            if not np.isfinite(result).all() or np.any(result <= 0):
                raise ValueError('core evolution exceeds finite positive range')
            return result
    except FloatingPointError as error:
        raise ValueError('core evolution exceeds finite positive range') from error


@dataclass(frozen=True)
class WakeCores:
    """Cores on free span edges and TE/free streamwise edges, including zero Gamma."""
    span: np.ndarray
    stream: np.ndarray
    policy: str
    initial_radius: float
    diffusion_rate: float
    chord_fraction: float | None = None

    def __post_init__(self):
        for name in ('span', 'stream'):
            array = owned(getattr(self, name))
            if array.ndim != 2 or np.any(array <= 0):
                raise ValueError('core history must be positive two-dimensional arrays')
            object.__setattr__(self, name, array)
        if (self.policy not in ('constant', 'age', 'age_stretch')
                or (self.policy == 'constant' and self.chord_fraction is None)
                or len(self.span) != len(self.stream)):
            raise ValueError('invalid core history policy/rows')
        positive(self.initial_radius, 'initial radius')
        positive(self.diffusion_rate, 'diffusion rate', zero=True)
        if self.chord_fraction is not None:
            positive(self.chord_fraction, 'wake core chord fraction')


def core_identity(config):
    return (config.core_evolution, config.core_radius_m,
            4*OSEEN_ALPHA*config.eddy_viscosity_factor*config.kinematic_viscosity_m2_s)


def initial_cores(wing, config):
    fraction = getattr(config, 'wake_core_chord_fraction', None)
    if config.core_evolution == 'constant' and fraction is None:
        return None
    return WakeCores(np.empty((0, wing.panels)), np.empty((0, wing.node_count)),
                     *core_identity(config), chord_fraction=fraction)


def birth_radii(wing, config):
    """Birth cores on material span edges and nodes, without changing bound AIC."""
    fraction = getattr(config, 'wake_core_chord_fraction', None)
    if fraction is None:
        return (np.full(wing.panels, config.core_radius_m),
                np.full(wing.node_count, config.core_radius_m))
    chord = _norm(wing.trailing_edge-wing.leading_edge)
    if np.any(chord <= 0):
        raise ValueError('local wake birth cores require positive node chords')
    a, b = wing.panel_edges.T
    return fraction*(chord[a]+chord[b])/2, fraction*chord


def check_cores(cores, config):
    fraction = getattr(config, 'wake_core_chord_fraction', None)
    if cores is None:
        if config.core_evolution != 'constant' or fraction is not None:
            raise ValueError('initialize a new run to change core evolution')
    elif ((cores.policy, cores.initial_radius, cores.diffusion_rate) != core_identity(config)
          or cores.chord_fraction != fraction):
        raise ValueError('initialize a new run to change core evolution parameters')


def advance_cores(state, new_rows, config):
    check_cores(state.cores, config)
    if state.cores is None:
        return None
    span_birth, stream_birth = birth_radii(state.wing, config)
    if state.cores.policy == 'constant':
        return WakeCores(np.vstack([span_birth, state.cores.span]),
                         np.vstack([stream_birth, state.cores.stream]),
                         *core_identity(config), chord_fraction=state.cores.chord_fraction)
    old_rows = np.concatenate(([state.wing.trailing_edge], state.free_rows))
    a, b = state.wing.panel_edges.T
    old_span = _norm(old_rows[:, b]-old_rows[:, a])
    new_span = _norm(new_rows[:, b]-new_rows[:, a])
    initial, rate = state.cores.initial_radius, state.cores.diffusion_rate
    previous = np.vstack([span_birth, state.cores.span])
    stretch = state.cores.policy == 'age_stretch'
    span = evolve_radius(previous, old_span, new_span, config.dt_s, rate, stretching=stretch)
    # Preserve scalar arithmetic exactly for legacy fixed-radius births.
    first_stream = (np.full(state.wing.node_count, np.sqrt(initial**2+rate*config.dt_s/2))
                    if state.cores.chord_fraction is None else
                    np.sqrt(stream_birth**2+rate*config.dt_s/2))
    stream = evolve_radius(state.cores.stream, _norm(np.diff(old_rows, axis=0)),
                           _norm(np.diff(new_rows, axis=0)), config.dt_s, rate,
                           stretching=stretch)
    return WakeCores(span, np.vstack([first_stream, stream]), *core_identity(config),
                     chord_fraction=state.cores.chord_fraction)


def edge_radii(wing, edges, cores, initial_radius):
    """Map globally indexed bound/TE/free edges to their unique material cores."""
    edges = np.asarray(edges)
    result = np.full(len(edges), initial_radius, dtype=float)
    if cores is None:
        return result
    width = wing.node_count
    lookup = {tuple(sorted(pair)): j for j, pair in enumerate(wing.panel_edges)}
    for i, (first, last) in enumerate(edges):
        ra, a = divmod(int(first), width)
        rb, b = divmod(int(last), width)
        if ra == rb:
            if ra >= 2:
                result[i] = cores.span[ra-2, lookup[tuple(sorted((a, b)))]]
        elif a == b and abs(rb-ra) == 1:
            row = min(ra, rb)
            if row >= 1:
                result[i] = cores.stream[row-1, a]
        else:
            raise ValueError('edge is not a spanwise or streamwise wake segment')
    return result
