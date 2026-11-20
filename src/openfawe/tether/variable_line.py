# SPDX-License-Identifier: Apache-2.0
# Formulation transcribed from the Makani simulator tether model, google/makani sim/models/tether.{h,cc}
# (commit 1d66923, Apache-2.0): node activation (GetStartIndex/GetEndIndex, 505-532), node masses (534-541), the
# staging and next-staging nodes (439-485, 602-645, 713-753, 792-842, 903-924), node placement at initialisation
# (197-215, 489-501), the start direction (238-249); parameters of config/common/sim/tether_sim.py.
# Modifications (2026): NumPy; the physics of the active part is that of ``tether.lumped_line.LumpedTether``
# (same expressions, same conventions); ground contact of the nodes only as an option, off by default (``ground``:
# Tether::CalcGroundForceNode, tether.cc:364-377, added in UpdateTetherForces at 418-420 before the saturation at 426; the contactor template of tether_sim.py:26-49, 67-72;
# Contactor, sim/physics/contactor.{h,cc}; the ground surface, sim/physics/ground_frame.cc).
"""Lumped-mass tether of variable free length (plan 3.7 part 2, criterion G4).

A separate class: ``LumpedTether`` (constant length) is not changed and is reused for the active part. Makani keeps
``N`` interior nodes (19 for the M600) as states all the time; the free length decides which of them are in the air:

* ``indices(free_length)`` (GetStartIndex/GetEndIndex): the last node is always node ``N - 1`` with a last segment of
  ``min(L0/2, L_free)``; going towards the anchor, segments of ``L0 = L/N`` are added while they fit, the remainder is
  the first segment ``L_start``; a first segment shorter than ``stiff_len_lim`` (1 m) is merged with the next one (one
  node fewer). Node ``k`` is therefore in the air once ``L_free >= L0/2 + (N - 1 - k) L0 + stiff_len_lim``; below
  ``L0/2 + stiff_len_lim`` (12.205 m for the M600) there is no node in the air and the tether is one segment from the
  anchor to the bridle point.
* The first active node ``s`` carries the mass of half a segment plus the first segment, ``(1/2 + L_start/L0) m0``,
  the others ``m0 = rho L / N``; the airborne mass is then ``rho L_free`` whenever a node is active.
* The active nodes ``s .. N-1`` move with the forces of ``LumpedTether`` on the chain anchor, nodes, bridle point,
  with natural lengths ``L_start, L0, ..., L0, L_end``.
* The staging node ``s - 1`` (the next to be paid out) moves with forces that act on no other node: springs from the
  anchor (natural length ``max(stiff_len_lim/2, L_first - L_first0)``) and to the first active node or the bridle
  point (the rest of ``L_first``), each with ``EA`` scaled by ``(L/stiff_len_lim)^3`` when shorter than
  ``stiff_len_lim``; a damping force ``-c_staged V`` (damping ratio 0.14) instead of the segment damping; no bending;
  drag on the cylinder from the anchor to the middle of its post segment; gravity.
* The next staging node ``s - 2`` is driven, position and velocity separately, towards the midpoint of the staging
  node and the anchor with the rate ``20/tau``, ``tau = L0/(3 m/s)``.
* Nodes below ``s - 2`` are frozen where they are (zero derivatives).

The free length enters only through these indices and natural lengths: there is no pay-out term, and a node that
becomes active keeps the position and velocity it had as the staging node.

Conventions kept from ``LumpedTether`` (so that at full free length the two give the same forces, criterion G4(c)):
z-up frame with gravity ``-g`` on z; the start force carries the bending shear of the first segment (Makani's
``CalcStartForceMomentPos`` has spring and damping only; with no active node the shear is zero). Kept from Makani and
not in ``LumpedTether``: the saturation of each node force component to +-1e9 N (tether.cc:426).

Ground contact of the nodes (Makani's ``kSimOptGroundContact``) is an option, off by default (``ground=None``: every
force and every result as before; full-cycle contract revision 7, option 甲). With a ``GroundContact`` the active and
the staging node (Makani: ``start_ind - 2 < i <= end_ind``) each get the force of one contactor of
``ground_contactor_template`` placed at the node: the ground is the plane ``z = ground_height_m`` of the z-up frame
(ground_frame.cc: collision below ``ground_z``, the collision point straight above/below the node); the force is the
spring ``k * penetration`` along +z, added before the +-1e9 N saturation as in tether.cc:418-426. Makani builds the
contactor in the ground frame at the node's position (contactor.h:73-77, tether.cc:371-377), so the contactor's velocity
relative to the ground is zero by construction and the damping (approach only) and friction terms of
``Contactor::CalcContactForce`` (contactor.cc:60-97) vanish - kept as written there: spring only. ``k`` is the template's
``m0 g / 0.01 m`` with the template's node mass ``rho L / N`` (the same for every node, also the first active one).
"""

from dataclasses import dataclass

import numpy as np

from .lumped_line import LumpedTether, TetherProperties, TetherSettings

NODE_FORCE_LIMIT_N = 1e9                 # SaturateVec3ByScalar(&F_teth, -1e9, 1e9) (tether.cc:426)
TYPICAL_PAY_OUT_SPEED_M_S = 3.0          # tether.cc:469


@dataclass(frozen=True)
class StagingSettings:
    """Makani's staging parameters (config/common/sim/tether_sim.py)."""
    stiff_len_lim_m: float = 1.0
    longitudinal_damping_ratio_staged: float = 0.14

    def __post_init__(self):
        if not (np.isfinite(self.stiff_len_lim_m) and self.stiff_len_lim_m > 0):
            raise ValueError("stiff_len_lim_m must be positive")
        if not (np.isfinite(self.longitudinal_damping_ratio_staged) and self.longitudinal_damping_ratio_staged >= 0):
            raise ValueError("longitudinal_damping_ratio_staged must be nonnegative")


@dataclass(frozen=True)
class GroundContact:
    """The ground plane (height in the z-up frame) and the contactor's spring constant (N/m) of the tether nodes."""
    ground_height_m: float
    spring_const_N_m: float

    def __post_init__(self):
        if not np.isfinite(self.ground_height_m):
            raise ValueError("ground_height_m must be finite")
        if not (np.isfinite(self.spring_const_N_m) and self.spring_const_N_m > 0):
            raise ValueError("spring_const_N_m must be positive")


def makani_ground_contact(properties, settings, ground_height_m, penetration_m=0.01):
    """``ground_contactor_template`` of config/common/sim/tether_sim.py:26-49, 67-72: node mass ``rho L / N`` and a
    spring that holds it at ``penetration_m`` (0.01 m) under gravity."""
    node_mass = properties.linear_density_kg_m*properties.length_m/float(settings.num_nodes)
    return GroundContact(float(ground_height_m), node_mass*settings.gravity_m_s2/float(penetration_m))


@dataclass(frozen=True)
class Indices:
    start: int          # first active node; == num_nodes when no node is active
    L_start: float      # natural length of the first segment
    end: int            # last node (always num_nodes - 1)
    L_end: float        # natural length of the last segment

    @property
    def active(self):
        return self.start <= self.end

    @property
    def L_first(self):
        return self.L_start + self.L_end if self.start > self.end else self.L_start


class VariableLengthTether:
    """Makani's variable-length tether on top of ``LumpedTether``'s physics; end points supplied per evaluation."""

    def __init__(self, properties=TetherProperties(), settings=TetherSettings(), staging=StagingSettings(), ground=None,
                 start_force_bending=True):
        if ground is not None and not isinstance(ground, GroundContact):
            raise ValueError("ground must be a GroundContact or None")
        if not isinstance(start_force_bending, bool):
            raise ValueError("start_force_bending must be True or False")
        # OF-0011 revision 6 addendum 62 (F2-Ma): False gives the start force as Makani's CalcStartForceMomentPos
        # (tether.cc:543-556: spring and damping, no bending shear); True (the default) keeps the shear, as before.
        self.start_force_bending = start_force_bending
        self.p, self.s, self.staging = properties, settings, staging
        self.ground = ground
        self.line = LumpedTether(properties, settings)
        self.n = settings.num_nodes
        self.L0 = properties.length_m/self.n                                       # L_seg_0 (tether.cc:105)
        self.m0 = properties.linear_density_kg_m*properties.length_m/self.n         # as LumpedTether.node_mass
        self.c_staged = staging.longitudinal_damping_ratio_staged*np.sqrt(
            2.0*properties.tensile_stiffness_N*properties.linear_density_kg_m)
        self.next_staging_rate = 20.0/(self.L0/TYPICAL_PAY_OUT_SPEED_M_S)

    # ----- indices and masses -------------------------------------------------------------------------------------
    def indices(self, free_length):
        """GetStartIndex and GetEndIndex (tether.cc:505-532)."""
        length = float(free_length)
        L0, stiff = self.L0, self.staging.stiff_len_lim_m
        L_end = min(L0/2.0, length)
        end = self.n - 1
        cum_len = L_end
        start = end
        while cum_len + L0 < length:
            start -= 1
            cum_len += L0
        L_start = length - cum_len
        if L_start < stiff:
            if cum_len > L0/2.0 + stiff:      # no extra segment length when the last node is deleted
                L_start += L0
            start += 1
        if start < 0:
            raise ValueError("the tether node start index must be non-negative")              # DCHECK
        return Indices(start, L_start, end, L_end)

    def held_indices(self, free_length, start):
        """The indices with the first active node held at ``start`` (OF-0011 revision 6 addendum 81, event location
        within a Runge-Kutta step; not used by default): ``indices(free_length)`` itself when its first active node is
        ``start``; otherwise the same ``end`` and ``L_end`` and the first segment's natural length from the expression
        of ``indices``, ``L_free - L_end - (end - start) L0``, continued past the threshold where the first active node
        changes."""
        idx = self.indices(free_length)
        if idx.start == start:
            return idx
        start = int(start)
        if start == idx.end + 1 and idx.start >= idx.end:
            # OF-0011 revision 7 addendum 4 supplement 2: "no active node" held while the first node is paid out - the
            # expression of ``indices`` without a node, ``L_start = L_free - L_end`` (one segment of natural length
            # L_free), continued past the change
            return Indices(start, float(free_length) - idx.L_end, idx.end, idx.L_end)
        if not 0 <= start <= idx.end:
            raise ValueError("a held first active node must be an active node index")
        return Indices(start, float(free_length) - idx.L_end - (idx.end - start)*self.L0, idx.end, idx.L_end)

    def node_masses(self, indices):
        """CalcNodeMass (tether.cc:534-541) for every node."""
        masses = np.full(self.n, self.m0)
        if indices.start <= indices.end:
            masses[indices.start] = (0.5 + indices.L_start/self.L0)*(self.p.linear_density_kg_m*self.p.length_m/self.n)
        return masses

    def airborne_mass(self, indices):
        """Mass of the active nodes (rho L_free when a node is active)."""
        return float(np.sum(self.node_masses(indices)[indices.start:])) if indices.active else 0.0

    # ----- forces -----------------------------------------------------------------------------------------------
    def _segment_forces(self, points, speeds, natural):
        """LumpedTether.segment_forces with given natural lengths."""
        seg = points[1:] - points[:-1]
        length = np.linalg.norm(seg, axis=1)
        unit = seg/np.maximum(length, 1e-9)[:, None]
        strain = (length - natural)/natural
        tension = self.p.tensile_stiffness_N*np.maximum(strain, 0.0)
        rate = np.einsum("ij,ij->i", seg, speeds[1:] - speeds[:-1])/np.maximum(length, 1e-9)
        damping = self.line.c_long*rate
        return unit*(tension + damping)[:, None], tension, damping, length

    def _staged_spring(self, x_i, x_f, natural):
        """CalcLongSpringForceSeg with artificial dynamics allowed (tether.cc:665-708)."""
        if not natural > 0.0:
            raise ValueError("the unstretched tether segment length must be positive")         # DCHECK
        seg = np.asarray(x_f, dtype=float) - np.asarray(x_i, dtype=float)
        length = float(np.linalg.norm(seg))
        if not length > natural:
            return np.zeros(3)
        strain = (length - natural)/natural
        ea = self.p.tensile_stiffness_N
        if natural < self.staging.stiff_len_lim_m:
            ea *= (natural/self.staging.stiff_len_lim_m)**3.0
        return seg*(ea*strain/length)

    def ground_forces(self, positions):
        """Contact force of each node (N,3) against the ground plane: ``k * penetration`` along +z where the node is below
        the plane, zero elsewhere (spring only; see the module text). Needs ``self.ground``."""
        positions = np.asarray(positions, dtype=float)
        penetration = self.ground.ground_height_m - positions[:, 2]
        out = np.zeros_like(positions)
        below = penetration > 0.0                                   # ground_frame.cc: X_contactor.z > ground_z
        out[below, 2] = self.ground.spring_const_N_m*penetration[below]
        return out

    def _aero_one(self, cyl_start, cyl_end, velocity, wind):
        """CalcAeroForceSeg (tether.cc:926-948), the expressions of LumpedTether.aero_forces."""
        axis = np.asarray(cyl_end, dtype=float) - np.asarray(cyl_start, dtype=float)
        length = float(np.linalg.norm(axis))
        unit = axis/max(length, 1e-9)
        apparent = np.asarray(velocity, dtype=float) - np.asarray(wind, dtype=float)
        normal = apparent - float(np.dot(apparent, unit))*unit
        factor = -0.5*self.s.air_density_kg_m3*self.p.section_drag_coeff*self.p.outer_diameter_m
        return factor*(length*float(np.linalg.norm(normal)))*normal

    def forces(self, free_length, positions, velocities, start, start_vel, end, end_vel, wind=None, indices=None):
        """Node forces (N,3; zero for nodes whose forces Makani zeroes, i <= s-2), end-body forces and diagnostics.

        ``indices``: an ``Indices`` to use instead of ``indices(free_length)`` (a diagnostic hook; ``free_length`` is
        then not used)."""
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        start, start_vel = np.asarray(start, dtype=float), np.asarray(start_vel, dtype=float)
        end, end_vel = np.asarray(end, dtype=float), np.asarray(end_vel, dtype=float)
        wind = np.zeros_like(positions) if wind is None else np.asarray(wind, dtype=float)
        idx = self.indices(free_length) if indices is None else indices
        s, e = idx.start, idx.end
        masses = self.node_masses(idx)
        node = np.zeros_like(positions)
        g = self.s.gravity_m_s2
        if idx.active:
            act_x, act_v = positions[s:], velocities[s:]
            points = self.line.chain(act_x, start, end)
            speeds = self.line.chain(act_v, start_vel, end_vel)
            natural = np.full(len(act_x) + 1, self.L0)
            natural[0], natural[-1] = idx.L_start, idx.L_end
            seg_force, tension, damping, length = self._segment_forces(points, speeds, natural)
            total = seg_force[1:] - seg_force[:-1]
            bend_seg = np.zeros_like(seg_force)
            if self.p.bending_stiffness_Nm2 > 0:
                kappa, rate = self.line.curvature_and_rate(act_x, act_v, start, start_vel, end, end_vel)
                zero = np.zeros((1, 3))
                bend_seg = self.line.bending_segment_forces(np.vstack([zero, kappa, zero]), np.vstack([zero, rate, zero]),
                                                            points[1:] - points[:-1], natural)
                total = total + bend_seg[1:] - bend_seg[:-1]
            aero = self.line.aero_forces(act_x, act_v, start, end, wind[s:])
            gravity = np.zeros_like(act_x)
            gravity[:, 2] = -masses[s:]*g
            if self.ground is None:
                node[s:] = np.clip(total + aero + gravity, -NODE_FORCE_LIMIT_N, NODE_FORCE_LIMIT_N)
            else:                                   # tether.cc:413-426: the ground force joins before the saturation
                node[s:] = np.clip(total + aero + gravity + self.ground_forces(act_x), -NODE_FORCE_LIMIT_N, NODE_FORCE_LIMIT_N)
            start_force = seg_force[0] + bend_seg[0] if self.start_force_bending else seg_force[0].copy()
            end_force = -seg_force[-1] - bend_seg[-1]
        else:
            # One segment from the anchor to the bridle point, natural length L_start + L_end (no curvature, so no
            # bending shear).
            points, speeds = np.vstack([start, end]), np.vstack([start_vel, end_vel])
            seg_force, tension, damping, length = self._segment_forces(points, speeds, np.array([idx.L_first]))
            start_force, end_force = seg_force[0].copy(), -seg_force[0]
        staging = s - 1
        if staging >= 0:
            L_first = idx.L_first
            L_first0 = self.L0 if s <= e else 0.5*self.L0
            L_pre = max(0.5*self.staging.stiff_len_lim_m, L_first - L_first0)
            L_post = L_first - L_pre
            x_i, v_i = positions[staging], velocities[staging]
            x_post = positions[staging + 1] if staging < e else end
            spring = self._staged_spring(x_i, x_post, L_post) - self._staged_spring(start, x_i, L_pre)
            damp = -self.c_staged*v_i
            cyl_end = 0.5*positions[staging] + 0.5*positions[staging + 1] if staging < e else end
            aero = self._aero_one(start, cyl_end, v_i, wind[staging])
            force = spring + damp + aero + np.array([0.0, 0.0, -masses[staging]*g])
            if self.ground is not None:
                force = force + self.ground_forces(x_i[None, :])[0]
            node[staging] = np.clip(force, -NODE_FORCE_LIMIT_N, NODE_FORCE_LIMIT_N)
        out = dict(node=node, start=start_force, end=end_force, indices=idx, masses=masses, tension=tension,
                   damping=damping, segment_length=length)
        if self.ground is not None:                 # diagnostics only with the option (the default output is unchanged)
            first = max(s - 1, 0)
            contact = np.zeros_like(positions)
            contact[first:] = self.ground_forces(positions[first:])
            out["ground_force"] = contact
        return out

    def derivatives(self, free_length, positions, velocities, start, start_vel, end, end_vel, wind=None, indices=None):
        """CalcDerivHelper (tether.cc:439-485): (dX, dV, forces). ``indices`` as in ``forces`` (default: from the free
        length; OF-0011 revision 6 addendum 81 passes ``held_indices``)."""
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        out = self.forces(free_length, positions, velocities, start, start_vel, end, end_vel, wind, indices=indices)
        idx, masses = out["indices"], out["masses"]
        dx, dv = np.zeros_like(positions), np.zeros_like(velocities)
        first = max(idx.start - 1, 0)
        dx[first:] = velocities[first:]
        dv[first:] = out["node"][first:]/masses[first:, None]
        next_staging = idx.start - 2
        if next_staging >= 0:
            target_x = 0.5*positions[idx.start - 1] + 0.5*np.asarray(start, dtype=float)
            target_v = 0.5*velocities[idx.start - 1] + 0.5*np.asarray(start_vel, dtype=float)
            dx[next_staging] = (target_x - positions[next_staging])*self.next_staging_rate
            dv[next_staging] = (target_v - velocities[next_staging])*self.next_staging_rate
        return dx, dv, out

    # ----- initialisation and outputs ------------------------------------------------------------------------------
    def init_nodes(self, start, end, free_length):
        """InitTetherNodes (tether.cc:489-501): on the line end -> start at distances L_end, L_end + L0, ...;
        nodes beyond the anchor continue the line outward. Velocities zero."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        length = float(free_length)
        cum_len = min(self.L0/2.0, length)
        positions = np.zeros((self.n, 3))
        for i in range(self.n - 1, -1, -1):
            c = cum_len/length
            positions[i] = (1.0 - c)*end + c*start
            cum_len += self.L0
        return positions, np.zeros_like(positions)

    def last_node(self, indices, positions, velocities, start, start_vel):
        """Xg_last_node / Vg_last_node (tether.h:99-104): the anchor when no node is active."""
        if indices.start > indices.end:
            return np.asarray(start, dtype=float), np.asarray(start_vel, dtype=float)
        return positions[indices.end], velocities[indices.end]

    def start_direction(self, indices, positions, start, end):
        """UpdateTetherStartDirG (tether.cc:238-249): first active node (or the bridle point) minus the anchor."""
        node = positions[indices.start] if indices.start < self.n else np.asarray(end, dtype=float)
        return node - np.asarray(start, dtype=float)

    def init(self, start, start_vel, bridle_point, free_length, positions=None, velocities=None):
        """Tether::Init (tether.cc:197-215): two passes of (indices, bridle point from the last node, forces, node
        placement). ``bridle_point(node_pos, node_vel) -> (end, end_vel)``; ``positions``/``velocities``: the node
        states before the call (Makani's previous values; zeros by default). Returns (positions, velocities, end,
        end_vel, start force of the second pass)."""
        positions = np.zeros((self.n, 3)) if positions is None else np.asarray(positions, dtype=float).copy()
        velocities = np.zeros((self.n, 3)) if velocities is None else np.asarray(velocities, dtype=float).copy()
        start_force = None
        for _ in range(2):
            idx = self.indices(free_length)
            node_x, node_v = self.last_node(idx, positions, velocities, start, start_vel)
            end, end_vel = bridle_point(node_x, node_v)
            start_force = self.forces(free_length, positions, velocities, start, start_vel, end, end_vel)["start"]
            positions, velocities = self.init_nodes(start, end, free_length)
        return positions, velocities, end, end_vel, start_force
