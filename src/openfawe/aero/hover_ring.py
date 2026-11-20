"""Hover-mode aerodynamics with rotor vortex rings: the strip model H-ring-S.

Contract: ``docs/tasks/OF-0008-g5control20261006/hover-ring-contract.zh.md`` (revision 2, decision 0045). Opt-in;
nothing here changes an existing default.

H-ring-S (contract M2): one load point per panel at the midpoint of its quarter-chord (bound) segment, or ``sub_strips``
equal parts of that segment; at each point u = wind - point motion + ring-induced velocity, u_perp = u - (u.s)s,
alpha = atan2(u.n, u.t) over the full circle, the section coefficients at alpha, force = 1/2 rho |u_perp|^2 c dl
(CL l + CD d) and couple = 1/2 rho |u_perp|^2 c^2 dl CM s, with d0 = (cos a t + sin a n)/|.|, l = (d0 x s)/|.| and
d = s x l (re-orthogonalised against the span as in ``vsm._full_incidence_loads``; contract revision 3); the
chord, tangent, normal and span directions of each panel are those of ``vsm.geometry`` (the VSM's panel axes). There
is no circulation solve: the wing's own induction is not included, and the coupling is one-way (rings -> sections).

Rotor rings (contract M3): eight actuator discs with prescribed thrust along +axis; each step every disc sheds one
ring of ``ring_nodes`` nodes at its rim with Gamma = T dt / (rho A) (the identity of ``ActuatorDisc.shed_ring``);
only the rings of the newest K = retention_s/dt steps are kept (older rings removed whole); loads, and any probe
points, are taken with the rings of that instant (after shedding
and removal, before convection); then the rings convect for dt with RK4 under the uniform wind plus their own
induction (Lamb-Oseen physical core, ``singularity='zero'`` as ``RingWake.self_velocity`` documents). At load
points the rings induce with the surface core and ``singularity='raise'``.

Nothing is clipped, smoothed or retried; a non-finite value raises.
"""

from dataclasses import dataclass, field

import numpy as np

from .vortex import CoreModel, segments_velocity
from .vsm import geometry
from ..rotor.actuator_disc import ShedRing, ring_edges, ring_nodes, sheet_strength
from ..rotor.ring_wake import RingWake, convect_rings

#: Makani body axes from OpenFAWE body axes (and back): diag(-1, 1, -1).
FLIP = np.array([-1.0, 1.0, -1.0])


def _unit_rows(a):
    n = np.linalg.norm(a, axis=1)
    if np.any(~np.isfinite(n)) or np.any(n <= 16*np.finfo(float).eps):
        raise ValueError("degenerate direction")
    return a/n[:, None]


@dataclass(frozen=True)
class Sections:
    """Load points of a wing assembly with their panel axes (contract M2)."""
    points: np.ndarray      # (k, 3) OpenFAWE body frame
    panel: np.ndarray       # (k,) panel index of each point
    chord: np.ndarray       # (k,)
    length: np.ndarray      # (k,) bound length carried by the point
    tangent: np.ndarray     # (k, 3)
    normal: np.ndarray      # (k, 3)
    span: np.ndarray        # (k, 3)
    surface: np.ndarray     # (k,) surface index (order of wing.wings)
    names: tuple            # surface names
    polar: object           # PanelPolars (one polar per panel)

    @classmethod
    def from_aircraft(cls, aircraft, sub_strips=1):
        """Points on each panel's bound segment at fractions (2j + 1)/(2 sub_strips), j = 0 .. sub_strips-1."""
        if int(sub_strips) != sub_strips or sub_strips < 1:
            raise ValueError("sub_strips must be a positive integer")
        wing = aircraft.wing
        surfaces = getattr(wing, "wings", (wing,))
        _, chord, normal, tangent, span = geometry(wing)
        starts, ends, owner = [], [], []
        for index, surface in enumerate(surfaces):
            bound = np.asarray(surface.bound, dtype=float)
            starts.append(bound[:-1])
            ends.append(bound[1:])
            owner.append(np.full(len(bound)-1, index))
        starts, ends, owner = np.vstack(starts), np.vstack(ends), np.concatenate(owner)
        if len(starts) != len(chord):
            raise ValueError("bound segments and panels disagree")
        fractions = (2.0*np.arange(sub_strips)+1.0)/(2.0*sub_strips)
        panels = np.repeat(np.arange(len(chord)), sub_strips)
        f = np.tile(fractions, len(chord))
        points = starts[panels]+f[:, None]*(ends[panels]-starts[panels])
        length = np.linalg.norm(ends-starts, axis=1)[panels]/sub_strips
        names = tuple(getattr(surface, "name", f"surface {i}") for i, surface in enumerate(surfaces))
        names = tuple(getattr(wing, "names", names))
        return cls(points, panels, chord[panels], length, tangent[panels], normal[panels], span[panels], owner[panels],
                   names, aircraft.polar)

    def loads(self, velocity, density):
        """Per-point force and couple (OpenFAWE body frame) and section angle for the air velocity ``velocity`` (k, 3)."""
        u = np.asarray(velocity, dtype=float).reshape(len(self.points), 3)
        if not np.all(np.isfinite(u)):
            raise ValueError("non-finite section inflow")
        across = u-np.sum(u*self.span, axis=1)[:, None]*self.span
        speed2 = np.sum(across*across, axis=1)
        alpha = np.arctan2(np.sum(u*self.normal, axis=1), np.sum(u*self.tangent, axis=1))
        cl, cd, cm = self.coefficients(alpha)
        drag = np.cos(alpha)[:, None]*self.tangent+np.sin(alpha)[:, None]*self.normal
        drag = _unit_rows(drag)
        lift = _unit_rows(np.cross(drag, self.span))
        drag = np.cross(self.span, lift)                     # as vsm._full_incidence_loads (contract revision 3)
        factor = 0.5*float(density)*speed2*self.chord*self.length
        force = factor[:, None]*(cl[:, None]*lift+cd[:, None]*drag)
        couple = (factor*self.chord*cm)[:, None]*self.span
        if not (np.all(np.isfinite(force)) and np.all(np.isfinite(couple))):
            raise ValueError("non-finite section loads")
        return force, couple, alpha

    def coefficients(self, alpha):
        polars = self.polar.polars
        values = np.array([polars[p].coefficients(a) for p, a in zip(self.panel, alpha)])
        return values[:, 0], values[:, 1], values[:, 2]

    def totals(self, force, couple, origin=np.zeros(3), mask=None):
        """Total force and moment about ``origin`` (OpenFAWE body frame) over the points in ``mask``."""
        keep = np.ones(len(self.points), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        f = force[keep].sum(axis=0)
        m = (np.cross(self.points[keep]-np.asarray(origin, dtype=float), force[keep])+couple[keep]).sum(axis=0)
        return f, m

    def surface_mask(self, prefix):
        return np.array([self.names[s].startswith(prefix) for s in self.surface])


def wind_from_makani(speed, alpha_rad, beta_rad):
    """Air velocity relative to the body in OpenFAWE axes for Makani's (alpha, beta) (contract M7)."""
    from .makani_hover_strip import apparent_wind_sph_to_cart
    return FLIP*apparent_wind_sph_to_cart(float(speed), float(alpha_rad), float(beta_rad))


def makani_coefficients(force, moment, speed, density, reference):
    """[CX, CY, CZ, Cl, Cm, Cn] in Makani axes from an OpenFAWE force and moment about the body origin."""
    q = 0.5*float(density)*float(speed)**2
    f = FLIP*np.asarray(force, dtype=float)/(q*reference["S"])
    m = FLIP*np.asarray(moment, dtype=float)/(q*reference["S"])
    return np.array([f[0], f[1], f[2], m[0]/reference["b"], m[1]/reference["c"], m[2]/reference["b"]])


@dataclass(frozen=True)
class PrescribedDisc:
    """One rotor with a prescribed thrust along +axis (contract M3); no rotor table."""
    label: str
    centre_m: np.ndarray
    axis: np.ndarray
    radius_m: float
    thrust_N: float

    def __post_init__(self):
        axis = np.asarray(self.axis, dtype=float).reshape(3)
        object.__setattr__(self, "centre_m", np.asarray(self.centre_m, dtype=float).reshape(3))
        object.__setattr__(self, "axis", axis/np.linalg.norm(axis))

    @property
    def area_m2(self):
        return np.pi*self.radius_m**2

    def ring(self, dt_s, density, count, time_s, axial_freestream_m_s=0.0):
        """The ring shed during one step: Gamma = T dt / (rho A); gamma from momentum theory (recorded only)."""
        gamma = sheet_strength(self.thrust_N, axial_freestream_m_s, density, self.area_m2)
        circulation = self.thrust_N*dt_s/(density*self.area_m2)
        nodes = ring_nodes(self.centre_m, self.axis, self.radius_m, count)
        return ShedRing(nodes, ring_edges(count), float(circulation), float(time_s), float(gamma),
                        float((axial_freestream_m_s+0.5*gamma)*dt_s))


def discs_from_case(rotor_case, thrust_by_row):
    """Prescribed discs from ``cases/rotors/m600_rotors_makani.json`` and a thrust per row (``{'bottom': T, 'top': T}``)."""
    radius = 0.5*float(rotor_case["diameter_m"])
    return tuple(PrescribedDisc(r["label"], r["centre_m"], r["axis"], radius, float(thrust_by_row[r["row"]]))
                 for r in rotor_case["rotors"])


@dataclass(frozen=True)
class RingSettings:
    """Numerical settings of the ring wake (contract M3, M4); defaults are the contract's base settings."""
    dt_s: float = 0.01
    end_s: float = 3.0
    average_from_s: float = 1.0
    ring_nodes: int = 16
    physical_core_m: float = 0.15
    surface_core_m: float = 0.3
    retention_s: float = 0.5
    density: float = 1.2
    core_model: CoreModel = CoreModel.LAMB_OSEEN
    backend: str = "numba"


def segment_point_distances(points, nodes, edges):
    """Smallest distance of each segment to any of ``points`` (for the filament-near-load-point count)."""
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    ab = b-a
    length2 = np.maximum(np.sum(ab*ab, axis=1), np.finfo(float).tiny)
    best = np.full(len(edges), np.inf)
    for start in range(0, len(points), 32):
        p = points[start:start+32]
        t = np.clip(np.einsum("pk,sk->ps", p, ab)-np.sum(a*ab, axis=1)[None, :], 0.0, None)/length2[None, :]
        t = np.minimum(t, 1.0)
        closest = a[None, :, :]+t[:, :, None]*ab[None, :, :]
        d = np.linalg.norm(p[:, None, :]-closest, axis=2)
        best = np.minimum(best, d.min(axis=0))
    return best


def ring_induction(wake, points, settings, surface_core_m=None, singularity="raise"):
    """Ring-induced velocity at ``points``; default: the surface core and ``singularity='raise'`` (load points)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not wake.count:
        return np.zeros_like(points)
    core = settings.surface_core_m if surface_core_m is None else surface_core_m
    return segments_velocity(points, wake.nodes, wake.edges, wake.circulation, model=settings.core_model,
                             core_radius=core, singularity=singularity, backend=settings.backend)


def disc_sample_points(disc, rings=6, azimuths=16):
    """Six equal-area annuli x 16 azimuths offset by half a step (the D3 sampling of ``hover_closure.py``)."""
    from ..rotor.actuator_disc import disc_basis
    e1, e2, _ = disc_basis(disc.axis)
    radii = disc.radius_m*np.sqrt((np.arange(rings)+0.5)/rings)
    angles = 2*np.pi*(np.arange(azimuths)+0.5)/azimuths
    return np.array([disc.centre_m+r*(np.cos(a)*e1+np.sin(a)*e2) for r in radii for a in angles])


@dataclass
class HoverRun:
    """Time-marched H-ring-S with prescribed rotor thrust at a fixed attitude (contract H2)."""
    sections: Sections
    discs: tuple
    wind_body: np.ndarray
    settings: RingSettings = field(default_factory=RingSettings)

    def __post_init__(self):
        self.wind_body = np.asarray(self.wind_body, dtype=float).reshape(3)
        self.wake = RingWake(surface_core_m=self.settings.surface_core_m)
        self.time_s = 0.0
        self.step = 0

    def axial_freestream(self, disc):
        return float(-np.dot(self.wind_body, disc.axis))

    def advance(self, probes=None, post_probes=None):
        """One step (contract M3 order): shed, remove old rings, loads with these rings, convect. Returns a record.

        ``probes`` maps a name to ``(points, core_m, singularity)``; their ring-induced velocity is evaluated with the
        same rings as the loads (before convection) and returned under ``record['probes']``. ``post_probes`` are
        evaluated after the convection (the state the next step starts from, before its shed; D3's instant; contract
        revision 3 items 4-5) and returned under ``record['post_probes']``. Both are reports: an error there is
        recorded under ``record['probe_errors']`` and does not fail the step.
        """
        s = self.settings
        for index, disc in enumerate(self.discs):
            self.wake = self.wake.shed(disc.ring(s.dt_s, s.density, s.ring_nodes, self.time_s, self.axial_freestream(disc)), index)
        cut = self.time_s-s.retention_s+0.5*s.dt_s
        self.wake = self.wake.released_since(cut)
        induced = ring_induction(self.wake, self.sections.points, s)
        force, couple, alpha = self.sections.loads(self.wind_body+induced, s.density)
        near = int(np.count_nonzero(segment_point_distances(self.sections.points, self.wake.nodes, self.wake.edges)
                                    <= s.surface_core_m)) if self.wake.count else 0
        record = dict(time_s=self.time_s, step=self.step, rings=int(self.wake.count//s.ring_nodes),
                      force=force, couple=couple, alpha=alpha, induced=induced, segments_within_surface_core=near,
                      probes={}, probe_errors={})
        for name, (points, core, singularity) in (probes or {}).items():
            try:
                record["probes"][name] = ring_induction(self.wake, points, s, core, singularity)
            except (ValueError, FloatingPointError) as error:
                record["probe_errors"][name] = f"{type(error).__name__}: {str(error)[:200]}"
        wind = self.wind_body
        self.wake = convect_rings(self.wake, s.dt_s, s.core_model, s.physical_core_m,
                                  ambient=lambda nodes: np.broadcast_to(wind, nodes.shape).copy(), integrator="rk4",
                                  backend=s.backend)
        if not np.all(np.isfinite(self.wake.nodes)):
            raise FloatingPointError("non-finite ring nodes")
        record["post_probes"] = {}
        for name, (points, core, singularity) in (post_probes or {}).items():
            try:
                record["post_probes"][name] = ring_induction(self.wake, points, s, core, singularity)
            except (ValueError, FloatingPointError) as error:
                record["probe_errors"][name] = f"{type(error).__name__}: {str(error)[:200]}"
        self.step += 1
        self.time_s = self.step*s.dt_s
        return record

    def disc_probes(self):
        """Probe points of every disc for the D3 metric (contract V3, revision 3): disc plane and 0.7 D downstream."""
        out = {}
        for index, disc in enumerate(self.discs):
            points = disc_sample_points(disc)
            out[f"disc{index}"] = (points, self.settings.physical_core_m, "zero")
            out[f"disc{index}_0.7D"] = (points-1.4*disc.radius_m*disc.axis, self.settings.physical_core_m, "zero")
        return out


def axial_mean(disc, induced):
    """Mean of the induced velocity along -axis over equal-area sample points (the D3 metric)."""
    return float(np.mean(np.asarray(induced)@(-disc.axis)))
