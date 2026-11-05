# SPDX-License-Identifier: Apache-2.0
# Formulation transcribed from the Makani simulator tether model,
# google/makani sim/models/tether.{h,cc} (commit 1d66923, Apache-2.0), with the
# parameter conventions of config/common/sim/tether_sim.py and the M600 values
# of config/m600/tether.py. Modifications (2026): NumPy implementation, constant
# length only (no staging node, no ground contact), explicit energy accounting.
"""Lumped-mass tether between two prescribed end points (OF-0007-tether20261001).

Model (Makani sim/models/tether.cc):

* ``N`` interior nodes of mass ``mu L / N``; natural segment length ``L / N``
  between nodes and ``L / (2N)`` between an end point and its first node.
* Longitudinal spring: tension-only, ``T = EA * strain`` for positive strain.
* Longitudinal damping along each segment: ``c * d|seg|/dt`` with
  ``c = zeta_long * sqrt(2 EA mu)``.
* Bending: node curvature from the circle through the node and its two
  neighbours; bending moment ``EI * kappa`` plus a damping moment
  ``L0^2 zeta_bend sqrt(EI mu) |dkappa/dt|`` along the curvature direction; the
  shear force in a segment is ``seg_hat x (M_f - M_i) / |seg|`` and a node
  receives ``F_post - F_pre``.
* Aerodynamics: each node carries a cylinder from the midpoint of its previous
  segment to the midpoint of its next segment (end points for the first and
  last node); only the normal component of the apparent velocity acts:
  ``F = -1/2 rho Cd d Lcyl |v_n| v_n``.
* Gravity ``m g`` on every node.

Forces on the end bodies are the segment forces of the first and last
segments (spring + damping + bending). The source's ``CalcEndForceMomentPos``
has all three terms; its ``CalcStartForceMomentPos`` has spring and damping
only, so the start force here carries a (small) bending shear the source
leaves out. Time integration is explicit RK4 with a fixed step; the
longitudinal modes set the stability limit (``rk4_step_bound_s`` gives a safe
bound; ``longitudinal_period`` is not the chain's shortest period, see the
task README's correction of 2026-10-03).
"""

from dataclasses import dataclass
import numpy as np

G_M_S2 = 9.81


@dataclass(frozen=True)
class TetherProperties:
    """Physical tether properties (defaults: Makani M600 FC1-02, config/m600/tether.py)."""
    length_m: float = 425.8
    linear_density_kg_m: float = 0.917
    outer_diameter_m: float = 0.0294
    tensile_stiffness_N: float = 18e6
    bending_stiffness_Nm2: float = 35.0
    section_drag_coeff: float = 0.7

    def __post_init__(self):
        for name in ("length_m", "linear_density_kg_m", "outer_diameter_m", "tensile_stiffness_N"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("bending_stiffness_Nm2", "section_drag_coeff"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


@dataclass(frozen=True)
class TetherSettings:
    """Discretisation and damping (defaults: config/common/sim/tether_sim.py)."""
    num_nodes: int = 19                       # round(425.8 / 22)
    longitudinal_damping_ratio: float = 0.1
    bending_damping_ratio: float = 1.0
    air_density_kg_m3: float = 1.225
    gravity_m_s2: float = G_M_S2
    curvature_rate_step_s: float = 1e-6       # finite-difference step of the source

    def __post_init__(self):
        if not isinstance(self.num_nodes, (int, np.integer)) or isinstance(self.num_nodes, bool) or self.num_nodes < 1:
            raise ValueError("num_nodes must be a positive integer")
        for name in ("longitudinal_damping_ratio", "bending_damping_ratio", "air_density_kg_m3", "gravity_m_s2"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not np.isfinite(self.curvature_rate_step_s) or self.curvature_rate_step_s <= 0:
            raise ValueError("curvature_rate_step_s must be positive")


class LumpedTether:
    """Constant-length lumped-mass tether; end points are supplied per evaluation."""

    def __init__(self, properties=TetherProperties(), settings=TetherSettings()):
        self.p, self.s = properties, settings
        n = settings.num_nodes
        self.segment_length = properties.length_m / n
        self.node_mass = properties.linear_density_kg_m * properties.length_m / n
        # Natural lengths of the N+1 segments: half segments at both ends.
        self.natural = np.full(n + 1, self.segment_length)
        self.natural[0] = self.natural[-1] = 0.5 * self.segment_length
        self.c_long = settings.longitudinal_damping_ratio * np.sqrt(
            2.0 * properties.tensile_stiffness_N * properties.linear_density_kg_m)
        self.d_bend = settings.bending_damping_ratio * np.sqrt(
            properties.bending_stiffness_Nm2 * properties.linear_density_kg_m)

    # ----- geometry helpers --------------------------------------------------
    def straight_state(self, start, end):
        """Nodes on the straight line between the end points (natural spacing), at rest."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        n = self.s.num_nodes
        fractions = (np.arange(n) + 0.5) / n
        positions = start + fractions[:, None] * (end - start)
        return positions, np.zeros_like(positions)

    def chain(self, positions, start, end):
        """All N+2 points start, nodes..., end."""
        return np.vstack([start, positions, end])

    @staticmethod
    def _curvature(v1, v2):
        """Curvature vectors of the circles through (p, p+v1, p+v1+v2) for stacked rows.

        Magnitude 1/R with R the circumradius |v1||v2||v1+v2| / (2|v1 x v2|),
        direction v1 x v2 (source: internal::CalcCurvature). Parallel segments in
        the same direction give zero; degenerate cases give zero as the source
        caller does.
        """
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross, axis=1)
        n1, n2, n12 = (np.linalg.norm(v, axis=1) for v in (v1, v2, v1 + v2))
        kappa = np.zeros_like(v1)
        ok = (norm_cross >= 1e-6) & (n1 >= 1e-9) & (n2 >= 1e-9) & (n12 > 0)
        inv_r = 2.0 * norm_cross[ok] / (n1[ok] * n2[ok] * n12[ok])
        kappa[ok] = cross[ok] / norm_cross[ok, None] * inv_r[:, None]
        return kappa

    def curvature_and_rate(self, positions, velocities, start, start_vel, end, end_vel):
        points = self.chain(positions, start, end)
        speeds = self.chain(velocities, start_vel, end_vel)
        v1, v2 = points[1:-1] - points[:-2], points[2:] - points[1:-1]
        kappa = self._curvature(v1, v2)
        dt = self.s.curvature_rate_step_s
        later = points + dt * speeds
        kappa_later = self._curvature(later[1:-1] - later[:-2], later[2:] - later[1:-1])
        rate = (kappa_later - kappa) / dt
        # The source zeroes the rate when either evaluation fails; a zero
        # curvature from a degenerate case therefore also zeroes its rate.
        rate[np.all(kappa == 0, axis=1) | np.all(kappa_later == 0, axis=1)] = 0.0
        return kappa, rate

    # ----- forces -------------------------------------------------------------
    def segment_forces(self, positions, velocities, start, start_vel, end, end_vel):
        """Per-segment spring + damping force vectors (pointing from segment start to end) and tensions."""
        points = self.chain(positions, start, end)
        speeds = self.chain(velocities, start_vel, end_vel)
        seg = points[1:] - points[:-1]
        length = np.linalg.norm(seg, axis=1)
        unit = seg / np.maximum(length, 1e-9)[:, None]
        strain = (length - self.natural) / self.natural
        tension = self.p.tensile_stiffness_N * np.maximum(strain, 0.0)
        rate = np.einsum("ij,ij->i", seg, speeds[1:] - speeds[:-1]) / np.maximum(length, 1e-9)
        damping = self.c_long * rate
        return unit * (tension + damping)[:, None], tension, damping, length

    def bending_segment_forces(self, kappa, rate, seg, natural):
        """Shear force of each segment from the moment difference between its end nodes.

        kappa/rate: (N+2, 3) with zeros at the end points (the source uses
        kVec3Zero for start and end). seg: (N+1, 3).
        """
        ei = self.p.bending_stiffness_Nm2
        d = natural**2 * self.d_bend
        kappa_norm = np.maximum(np.linalg.norm(kappa, axis=1), 1e-9)
        kappa_hat = kappa / kappa_norm[:, None]
        rate_norm = np.maximum(np.linalg.norm(rate, axis=1), 1e-9)
        moment = ei * kappa                           # (N+2, 3) elastic part
        # Damping moment uses the segment's own natural length in the source
        # (d_damp_active = L_seg_0^2 * coefficient); apply per segment end.
        length = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
        unit = seg / length[:, None]
        m_i = moment[:-1] + (d[:, None] * rate_norm[:-1, None]) * kappa_hat[:-1]
        m_f = moment[1:] + (d[:, None] * rate_norm[1:, None]) * kappa_hat[1:]
        return np.cross(unit, (m_f - m_i) / length[:, None])

    def aero_forces(self, positions, velocities, start, end, wind):
        """Normal drag on the cylinder between the adjacent segment midpoints of each node."""
        points = self.chain(positions, start, end)
        mids = 0.5 * (points[1:] + points[:-1])        # N+1 segment midpoints
        cyl_start = mids[:-1].copy()
        cyl_end = mids[1:].copy()
        cyl_start[0] = start
        cyl_end[-1] = end
        axis = cyl_end - cyl_start
        length = np.linalg.norm(axis, axis=1)
        unit = axis / np.maximum(length, 1e-9)[:, None]
        apparent = velocities - wind
        normal = apparent - np.einsum("ij,ij->i", apparent, unit)[:, None] * unit
        speed = np.linalg.norm(normal, axis=1)
        factor = -0.5 * self.s.air_density_kg_m3 * self.p.section_drag_coeff * self.p.outer_diameter_m
        return factor * (length * speed)[:, None] * normal

    def forces(self, positions, velocities, start, start_vel, end, end_vel, wind=None):
        """Node forces (N,3), end-body forces and diagnostics."""
        positions = np.asarray(positions, float)
        velocities = np.asarray(velocities, float)
        wind = np.zeros_like(positions) if wind is None else np.asarray(wind, float)
        seg_force, tension, damping, length = self.segment_forces(positions, velocities, start, start_vel, end, end_vel)
        node = seg_force[1:] - seg_force[:-1]
        bend_seg = np.zeros_like(seg_force)
        if self.p.bending_stiffness_Nm2 > 0:
            kappa, rate = self.curvature_and_rate(positions, velocities, start, start_vel, end, end_vel)
            zero = np.zeros((1, 3))
            kappa_all = np.vstack([zero, kappa, zero])
            rate_all = np.vstack([zero, rate, zero])
            points = self.chain(positions, start, end)
            bend_seg = self.bending_segment_forces(kappa_all, rate_all, points[1:] - points[:-1], self.natural)
            node = node + bend_seg[1:] - bend_seg[:-1]
        aero = self.aero_forces(positions, velocities, start, end, wind)
        gravity = np.zeros_like(positions)
        gravity[:, 2] = -self.node_mass * self.s.gravity_m_s2
        total = node + aero + gravity
        # seg_force[k] pulls the start point of segment k towards its end point.
        # Start body: pulled by segment 0 towards node 0, plus the bending shear
        # of segment 0 (a point with a post segment only). End body: pulled by
        # the last segment towards the last node, minus the bending shear of
        # that segment (a point with a pre segment only), as in the source's
        # CalcStartForceMomentPos / CalcEndForceMomentPos.
        start_force = seg_force[0] + bend_seg[0]
        end_force = -seg_force[-1] - bend_seg[-1]
        return dict(node=total, start=start_force, end=end_force, tension=tension, damping=damping,
                    segment_length=length, aero=aero, bending=bend_seg)

    # ----- energy -------------------------------------------------------------
    def energy(self, positions, velocities, start, end):
        points = self.chain(positions, start, end)
        seg = points[1:] - points[:-1]
        length = np.linalg.norm(seg, axis=1)
        strain = np.maximum((length - self.natural) / self.natural, 0.0)
        elastic = float(np.sum(0.5 * self.p.tensile_stiffness_N * strain**2 * self.natural))
        kinetic = float(0.5 * self.node_mass * np.sum(velocities**2))
        potential = float(self.node_mass * self.s.gravity_m_s2 * np.sum(positions[:, 2]))
        return dict(kinetic=kinetic, elastic=elastic, gravity=potential, total=kinetic + elastic + potential)

    def longitudinal_period(self):
        """Shortest longitudinal period of the chain (two-node stiffness bound)."""
        k = self.p.tensile_stiffness_N / self.segment_length
        return 2 * np.pi / np.sqrt(2.0 * k / self.node_mass) if k > 0 else np.inf

    def rk4_step_bound_s(self):
        """A time step below which RK4 is stable for the undamped longitudinal modes of the chain.

        The squared frequencies of the chain are bounded by the largest row sum of its
        mass-normalised stiffness (Gershgorin): ``2 (k_left + k_right) / m`` over the nodes, where the
        half segments at the two ends are twice as stiff as the others. RK4 is stable on the
        imaginary axis up to ``omega dt = 2 sqrt(2)``. The bound is sufficient, not sharp, and it
        ignores the damping and the bending terms (both small here).
        """
        stiffness = self.p.tensile_stiffness_N / self.natural            # one per segment
        largest = np.max(2.0 * (stiffness[:-1] + stiffness[1:]) / self.node_mass)
        return float(2.0 * np.sqrt(2.0) / np.sqrt(largest))

    # ----- time integration ---------------------------------------------------
    def rhs(self, time, positions, velocities, ends, wind_field):
        # An end that depends on the line itself (the kite's bridle point follows the last node) is given
        # by a callable marked ``uses_state``; it is then evaluated at every stage with the stage's state.
        if getattr(ends, "uses_state", False):
            start, start_vel, end, end_vel = ends(time, positions, velocities)
        else:
            start, start_vel, end, end_vel = ends(time)
        wind = None if wind_field is None else wind_field(positions, time)
        out = self.forces(positions, velocities, start, start_vel, end, end_vel, wind)
        return velocities, out["node"] / self.node_mass

    def step(self, time, positions, velocities, dt, ends, wind_field=None):
        """One explicit RK4 step; returns (positions, velocities)."""
        k1x, k1v = self.rhs(time, positions, velocities, ends, wind_field)
        k2x, k2v = self.rhs(time + dt / 2, positions + dt / 2 * k1x, velocities + dt / 2 * k1v, ends, wind_field)
        k3x, k3v = self.rhs(time + dt / 2, positions + dt / 2 * k2x, velocities + dt / 2 * k2v, ends, wind_field)
        k4x, k4v = self.rhs(time + dt, positions + dt * k3x, velocities + dt * k3v, ends, wind_field)
        positions = positions + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        velocities = velocities + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if not (np.isfinite(positions).all() and np.isfinite(velocities).all()):
            raise FloatingPointError("tether state left the finite range (time step too large?)")
        return positions, velocities


def fixed_ends(start, end):
    """End-point function for two fixed points."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    zero = np.zeros(3)

    def ends(time):
        return start, zero, end, zero
    return ends


def relax(line, positions, velocities, ends, dt, steps, wind_field=None, velocity_tolerance=None):
    """March to rest with the model's own damping (dynamic relaxation)."""
    time = 0.0
    for _ in range(steps):
        positions, velocities = line.step(time, positions, velocities, dt, ends, wind_field)
        time += dt
        if velocity_tolerance is not None and np.max(np.linalg.norm(velocities, axis=1)) < velocity_tolerance:
            break
    return positions, velocities, time

