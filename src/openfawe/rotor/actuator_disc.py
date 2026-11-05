"""Actuator disc with a momentum-theory vortex-ring wake (G3r contract §3).

Loads come from the rotor table (Makani conventions, see ``table.py``); the
disc's effect on the flow is the semi-infinite vortex tube of a uniformly
loaded disc, released as one vortex ring per time step into the free wake.

Momentum theory (Froude/Rankine; Glauert 1935): a tube of sheet strength
``gamma`` (circulation per unit axial length) induces gamma/2 at the disc and
gamma far downstream, so with axial freestream V into the disc
    T = rho A (V + gamma/2) gamma   ->   gamma = -V + sqrt(V^2 + 2 T / (rho A)).
A negative radicand (turbine state beyond the momentum-theory limit, C_T > 1
in wind-turbine notation) has no solution and raises; nothing is extrapolated
(contract T8, owner choice §7.1 pending).

Conventions: ``axis`` is the unit thrust direction; the apparent wind is the
air velocity relative to the hub; the axial freestream is V = -(u_app . axis)
as in rotor.cc CalcVFreestream (positive for air entering the disc from the
front). The shed ring is oriented so that its slipstream flows along -axis.
"""

from dataclasses import dataclass, replace

import numpy as np

from .table import RotorTable


def _unit(vector, name):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a finite non-zero vector")
    return vector / norm


def disc_basis(axis, reference=None):
    """Right-handed orthonormal (e1, e2, n) with n = -axis (ring normal = slipstream direction).

    ``e1`` is the part of ``reference`` normal to the axis; without a reference it is taken from a
    coordinate axis of the frame the disc is given in (x, or y when the axis is within 26 deg of x).
    """
    normal = -_unit(axis, "axis")
    if reference is None:
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        helper = np.asarray(reference, dtype=float).reshape(3)
    e1 = helper - normal * np.dot(helper, normal)
    norm = np.linalg.norm(e1)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("the rim reference must not be parallel to the disc axis")
    e1 /= norm
    e2 = np.cross(normal, e1)
    return e1, e2, normal


def ring_nodes(centre, axis, radius, count, phase=0.0, reference=None):
    """Regular polygon on the rim, ordered right-handed about -axis (see disc_basis)."""
    if count < 3:
        raise ValueError("a vortex ring needs at least three nodes")
    e1, e2, _ = disc_basis(axis, reference)
    angles = phase + 2.0 * np.pi * np.arange(count) / count
    centre = np.asarray(centre, dtype=float).reshape(3)
    return centre + radius * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)


def ring_edges(count, offset=0):
    """Closed directed loop node i -> i+1 (wrapping), node ids shifted by offset."""
    index = np.arange(count) + offset
    return np.column_stack([index, np.roll(index, -1)]).astype(np.int64)


def ring_axis_velocity(circulation, radius, distance):
    """Exact axial velocity of a circular ring on its own axis (Lamb §161): Gamma R^2 / (2 (R^2+z^2)^{3/2})."""
    return circulation * radius ** 2 / (2.0 * (radius ** 2 + np.asarray(distance, dtype=float) ** 2) ** 1.5)


GLAUERT_INDUCTION_LIMIT = 0.4  # a_c: momentum theory is used below, Buhl's relation above


def _buhl_induction(c_t):
    """Axial induction a for a wind-turbine thrust coefficient c_t > 0.96 (Buhl 2005, F = 1).

    C_T = 8/9 + (4F - 40/9) a + (50/9 - 4F) a^2, tangent to 4a(1-a) at a = 0.4; verify the
    reference before citing (Buhl, NREL/TP-500-36834). Returns the root in [0.4, 1].
    """
    a2, a1, a0 = 50.0 / 9.0 - 4.0, 4.0 - 40.0 / 9.0, 8.0 / 9.0 - c_t
    disc = a1 * a1 - 4.0 * a2 * a0
    if disc < 0.0:
        raise ValueError(f"Buhl relation has no real induction for C_T={c_t:.4g}")
    return (-a1 + np.sqrt(disc)) / (2.0 * a2)


def sheet_strength(thrust_N, axial_freestream_m_s, density_kg_m3, area_m2, *, turbine_extension=None):
    """Momentum-theory tube strength gamma (m/s).

    gamma = -V + sqrt(V^2 + 2T/(rho A)); for V > 0 this equals -2 a V with the axial
    induction a of C_T = 4a(1-a) (wind-turbine notation, T < 0 extracting power).
    Without ``turbine_extension`` a negative radicand (C_T > 1, no momentum solution)
    raises and nothing is extrapolated (contract T8, decision 0020). With
    ``turbine_extension='glauert'`` the turbulent-windmill branch uses Buhl's empirical
    relation above a = 0.4 (C_T > 0.96), the standard BEM practice; this option exists
    for comparison and is off by default.
    """
    if turbine_extension not in (None, "glauert"):
        raise ValueError("turbine_extension must be None or 'glauert'")
    if turbine_extension == "glauert" and thrust_N < 0.0 and axial_freestream_m_s > 0.0:
        c_t = -thrust_N / (0.5 * density_kg_m3 * area_m2 * axial_freestream_m_s ** 2)
        if c_t > 4.0 * GLAUERT_INDUCTION_LIMIT * (1.0 - GLAUERT_INDUCTION_LIMIT):
            return -2.0 * _buhl_induction(c_t) * axial_freestream_m_s
    radicand = axial_freestream_m_s ** 2 + 2.0 * thrust_N / (density_kg_m3 * area_m2)
    if not np.isfinite(radicand):
        raise ValueError("non-finite momentum-theory radicand")
    if radicand < 0.0:
        raise ValueError(
            f"momentum theory has no solution: V={axial_freestream_m_s:.4g} m/s, T={thrust_N:.4g} N "
            f"(turbine state beyond C_T=1); not extrapolated")
    return -axial_freestream_m_s + np.sqrt(radicand)


@dataclass(frozen=True)
class ShedRing:
    """One vortex ring released from the disc rim at a given time."""
    nodes: np.ndarray        # (count, 3)
    edges: np.ndarray        # (count, 2) directed, closed
    circulation: float       # Gamma per edge (m^2/s); all edges of the ring carry it
    release_time_s: float
    sheet_strength: float    # gamma (m/s)
    spacing_m: float         # nominal convection distance (V + gamma/2) dt

    @property
    def count(self):
        return len(self.nodes)


@dataclass(frozen=True)
class ActuatorDisc:
    """One rotor as a uniformly loaded actuator disc."""
    label: str
    centre_m: np.ndarray
    axis: np.ndarray             # unit thrust direction
    direction: int               # +1 right-handed rotation about axis, -1 opposite (Makani dir)
    table: RotorTable
    diameter_m: float | None = None
    # Direction from which the rim polygon is counted (its part normal to the axis); None: a coordinate
    # axis of the frame the disc is given in (see disc_basis). A placed disc carries the body's direction.
    rim_reference: np.ndarray | None = None

    def __post_init__(self):
        centre = np.asarray(self.centre_m, dtype=float).reshape(3)
        if not np.all(np.isfinite(centre)):
            raise ValueError("rotor centre must be finite")
        if self.direction not in (1, -1):
            raise ValueError("rotor direction must be +1 or -1")
        if not isinstance(self.table, RotorTable):
            raise ValueError("rotor needs a RotorTable")
        diameter = self.table.diameter_m if self.diameter_m is None else float(self.diameter_m)
        if diameter != self.table.diameter_m:
            raise ValueError("disc diameter must equal the table diameter (the coefficients are for that rotor)")
        object.__setattr__(self, "centre_m", centre)
        object.__setattr__(self, "axis", _unit(self.axis, "axis"))
        object.__setattr__(self, "diameter_m", diameter)

    def placed(self, rotation, position_m):
        """The disc of a body whose axes are turned by ``rotation`` (body to inertial) with its origin at ``position_m``.

        ``centre_m`` and ``axis`` of this disc are then read as body-axis values; the table, the
        rotation sense about the axis and the label are unchanged (a proper rotation keeps the sense).
        The rim polygon turns with the body: its nodes keep their places on the body at every attitude.
        """
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        position = np.asarray(position_m, dtype=float).reshape(3)
        reference = disc_basis(self.axis, self.rim_reference)[0]
        return replace(self, centre_m=position + rotation @ self.centre_m, axis=rotation @ self.axis,
                       rim_reference=rotation @ reference)

    @property
    def radius_m(self):
        return 0.5 * self.diameter_m

    @property
    def area_m2(self):
        return np.pi * self.radius_m ** 2

    def axial_freestream(self, apparent_wind_m_s):
        """V = -(u_app . axis): positive when air enters the disc from the front (rotor.cc)."""
        wind = np.asarray(apparent_wind_m_s, dtype=float).reshape(3)
        return float(-np.dot(wind, self.axis))

    def loads(self, omega_rad_s, apparent_wind_m_s, density_kg_m3):
        """Thrust, power, torque and the body force/moment at the hub from the table.

        ``omega_rad_s`` is the unsigned angular speed (table axis). The moment
        about the axis is direction * (power / omega), opposing the rotation
        while thrusting (rotor.cc). Gyroscopic and reaction moments belong to
        the airframe dynamics (G4) and are not included here.
        """
        speed = float(omega_rad_s)
        if not np.isfinite(speed) or speed < 0.0:
            raise ValueError("rotor angular speed must be finite and non-negative")
        freestream = self.axial_freestream(apparent_wind_m_s)
        thrust = self.table.thrust_N(speed, freestream, density_kg_m3)
        power = self.table.power_W(speed, freestream, density_kg_m3)
        torque = self.table.torque_Nm(speed, freestream, density_kg_m3)
        return dict(axial_freestream_m_s=freestream, thrust_N=thrust, power_W=power,
                    torque_Nm=self.direction * torque,
                    force_N=thrust * self.axis, moment_Nm=self.direction * torque * self.axis,
                    saturated=self.table.saturated(speed, freestream))

    def sheet_strength(self, thrust_N, axial_freestream_m_s, density_kg_m3, *, turbine_extension=None):
        return sheet_strength(thrust_N, axial_freestream_m_s, density_kg_m3, self.area_m2,
                              turbine_extension=turbine_extension)

    def induced_velocity_m_s(self, thrust_N, axial_freestream_m_s, density_kg_m3, *, turbine_extension=None):
        """Momentum-theory induced velocity at the disc, v_i = gamma / 2."""
        return 0.5 * self.sheet_strength(thrust_N, axial_freestream_m_s, density_kg_m3,
                                         turbine_extension=turbine_extension)

    def rim(self, count, phase=0.0):
        return ring_nodes(self.centre_m, self.axis, self.radius_m, count, phase, self.rim_reference)

    def shed_ring(self, omega_rad_s, apparent_wind_m_s, density_kg_m3, dt_s, count, *, time_s=0.0,
                  phase=0.0, turbine_extension=None):
        """The vortex ring released during one step: Gamma = gamma * (V + gamma/2) dt."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("time step must be positive")
        loads = self.loads(omega_rad_s, apparent_wind_m_s, density_kg_m3)
        gamma = self.sheet_strength(loads["thrust_N"], loads["axial_freestream_m_s"], density_kg_m3,
                                    turbine_extension=turbine_extension)
        spacing = (loads["axial_freestream_m_s"] + 0.5 * gamma) * dt_s
        return ShedRing(self.rim(count, phase), ring_edges(count), float(gamma * spacing), float(time_s),
                        float(gamma), float(spacing))


def tube_rings(disc, gamma, spacing, count_rings, nodes_per_ring, *, start_offset=0.5):
    """Rings approximating the semi-infinite tube behind the disc (verification D2).

    Ring k sits at (k + start_offset) * spacing along -axis with circulation
    gamma * spacing. Returns (nodes, edges, circulation per edge).
    """
    nodes, edges, strengths = [], [], []
    for k in range(count_rings):
        centre = disc.centre_m - disc.axis * (k + start_offset) * spacing
        nodes.append(ring_nodes(centre, disc.axis, disc.radius_m, nodes_per_ring))
        edges.append(ring_edges(nodes_per_ring, k * nodes_per_ring))
        strengths.append(np.full(nodes_per_ring, gamma * spacing))
    return np.vstack(nodes), np.vstack(edges), np.concatenate(strengths)


def load_rotor_inertia(path):
    """Moment of inertia of one rotor about its axis [kg m^2] from an ``openfawe.rotors.v1`` file (propeller plus motor rotor)."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "openfawe.rotors.v1":
        raise ValueError(f"{path}: unsupported rotor set schema {data.get('schema')!r}")
    inertia = float(data["rotor_inertia_kg_m2"])
    if not np.isfinite(inertia) or inertia <= 0.0:
        raise ValueError(f"{path}: rotor inertia must be positive")
    return inertia


def load_rotor_set(path, table):
    """Read an ``openfawe.rotors.v1`` JSON file into ActuatorDisc objects (same table for all)."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "openfawe.rotors.v1":
        raise ValueError(f"{path}: unsupported rotor set schema {data.get('schema')!r}")
    if data.get("frame") != "x_downstream_y_span_z_up":
        raise ValueError(f"{path}: rotor set must be in the x_downstream_y_span_z_up frame")
    if float(data["diameter_m"]) != table.diameter_m:
        raise ValueError(f"{path}: diameter {data['diameter_m']} differs from the table's {table.diameter_m}")
    return tuple(ActuatorDisc(str(r["label"]), r["centre_m"], r["axis"], int(r["direction"]), table)
                 for r in data["rotors"])
