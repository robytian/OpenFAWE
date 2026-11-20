"""The plant over one aerodynamic step: rigid body and tether advanced together (G5 plant contract, step 1b part one).

Contract: ``docs/tasks/OF-0008-g5control20261006/plant-contract.zh.md``, section "第 1b 步（其一）"; coupling
scheme: decision 0030. There is no aerodynamics and no rotor in this module: it takes the loads of the
start of the step as given (``HeldLoads``) and returns the state at the end of the step.

Over one aerodynamic step ``[t, t + step]``:

* **held loads**: a force and a moment about the centre of mass, stored in **body axes** at the start of
  the step and unchanged during it, so that they turn with the body (later: the vortex-wake aerodynamics
  and the rotors at the start of the step; first order in the step);
* the rigid body (``flight.rigid_body``: centre-of-mass state, attitude by rotation-vector increments) and
  the nodes of the lumped-mass tether (``tether.lumped_line``) are advanced **together** by ``substeps``
  classical fourth-order Runge-Kutta steps. At every stage the pose of the stage gives the bridle point
  (``tether.bridle``, with the last node of the stage), the line gives the node forces and its force on
  the body, which acts at the bridle point; the line sees the ambient wind only;
* an **external load** callback ``external(time_s, state, model)`` is called at every stage with the model
  loads of that stage (``model``: ``force_inertial_N`` and ``moment_cm_inertial_Nm`` of held loads plus
  tether, without gravity, and their parts) and returns a force and a moment about the centre of mass in
  inertial components. It is the entry for a controller's actuation and for the checks.

Accounts are integrated with the state by the same scheme (``PlantAccount``): the work of every load on
the body, and the impulse of every force that is external to the system of body and line. The total
linear momentum is a linear function of the state, so its change equals the summed impulses to rounding
whenever action and reaction are equal at every stage (criterion C2); the kinetic energy is quadratic,
so the work account closes to the accuracy of the scheme only.

Without a tether a sub-step is the step of ``flight.rigid_body`` (same expressions). No adaptive step,
no retry: a state outside the finite range raises ``FloatingPointError``; the line's sub-step is checked
against its stability bound before the first sub-step (``ValueError``, a set-up mistake).

Nothing in the package uses this module: no existing path changes.
"""

from dataclasses import dataclass, field

import numpy as np

from ..aero.wake import ambient_velocity
from ..tether.bridle import Bridle
from ..tether.lumped_line import LumpedTether
from .kinematics import rotation_from_vector
from .residual import RigidBody
from .rigid_body import GRAVITY_M_S2, RigidState, _theta_rate

WORKS = ("held", "tether", "gravity", "external")
IMPULSES = ("held", "external", "gravity_body", "gravity_line", "line_aero", "anchor")


@dataclass(frozen=True)
class HeldLoads:
    """Force and moment about the centre of mass in body axes, held over an aerodynamic step."""
    force_body_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment_cm_body_Nm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        for name in ("force_body_N", "moment_cm_body_Nm"):
            value = np.array(getattr(self, name), dtype=float).reshape(3)
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{name} left the finite range")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlantTether:
    """The line between a fixed anchor (the ground point) and the bridle point of the body."""
    line: LumpedTether
    anchor_m: np.ndarray
    bridle: Bridle

    def __post_init__(self):
        if not isinstance(self.line, LumpedTether) or not isinstance(self.bridle, Bridle):
            raise ValueError("a plant tether needs a LumpedTether and a Bridle")
        anchor = np.array(self.anchor_m, dtype=float).reshape(3)
        if not np.isfinite(anchor).all():
            raise ValueError("the anchor must be finite")
        anchor.setflags(write=False)
        object.__setattr__(self, "anchor_m", anchor)

    def loads(self, body, state, positions, velocities, wind):
        """Line forces at one state: ``(out, bridle_point, force_on_body, moment_cm)`` (inertial components)."""
        pose = state.pose(body)
        point, point_velocity = self.bridle.point(pose, positions[-1], velocities[-1])
        air = None if wind is None else ambient_velocity(wind, positions, state.time_s)
        out = self.line.forces(positions, velocities, self.anchor_m, np.zeros(3), point, point_velocity, air)
        force = np.asarray(out["end"], dtype=float)
        return out, point, force, np.cross(point-state.position_cm_m, force)


@dataclass(frozen=True)
class PlantAccount:
    """Work of the loads on the body [J] and impulses of the external forces on body and line [N s] over one call."""
    work_J: dict
    impulse_N_s: dict

    def total_work_J(self):
        return float(sum(self.work_J.values()))

    def total_impulse_N_s(self):
        return np.sum([self.impulse_N_s[name] for name in IMPULSES], axis=0)


def _derivatives(body, inverse, gravity, tether, held, external, wind, time, position, rotation, velocity, omega, nodes, speeds):
    """All rates at one stage: body, nodes, work rates and impulse rates."""
    state = RigidState(time, position, rotation, velocity, omega)
    inertial_rate = rotation@omega
    held_force, held_moment = rotation@held.force_body_N, rotation@held.moment_cm_body_Nm
    zero = np.zeros(3)
    if tether is None:
        tether_force, tether_moment, node_rates, out = zero, zero, None, None
    else:
        out, _, tether_force, tether_moment = tether.loads(body, state, nodes, speeds, wind)
        node_rates = out["node"]/tether.line.node_mass
    model_force, model_moment = held_force+tether_force, held_moment+tether_moment
    if external is None:
        external_force, external_moment = zero, zero
    else:
        model = dict(force_inertial_N=model_force, moment_cm_inertial_Nm=model_moment,
                     held_force_inertial_N=held_force, held_moment_cm_inertial_Nm=held_moment,
                     tether_force_inertial_N=tether_force, tether_moment_cm_inertial_Nm=tether_moment)
        external_force, external_moment = external(time, state, model)
        external_force = np.asarray(external_force, dtype=float).reshape(3)
        external_moment = np.asarray(external_moment, dtype=float).reshape(3)
    weight = body.mass_kg*gravity
    inertia = body.inertia_body_kg_m2
    acceleration = (model_force+external_force)/body.mass_kg+gravity
    spin = inverse@(rotation.T@(model_moment+external_moment)-np.cross(omega, inertia@omega))
    work = np.array([held_force@velocity+held_moment@inertial_rate, tether_force@velocity+tether_moment@inertial_rate,
                     weight@velocity, external_force@velocity+external_moment@inertial_rate])
    if out is None:
        line_gravity = line_aero = anchor = zero
    else:
        line_gravity = np.array([0.0, 0.0, -tether.line.node_mass*tether.line.s.gravity_m_s2*len(nodes)])
        line_aero = np.asarray(out["aero"], dtype=float).sum(axis=0)
        anchor = -np.asarray(out["start"], dtype=float)            # force of the anchor on the line
    impulse = np.array([held_force, external_force, weight, line_gravity, line_aero, anchor])
    return velocity, acceleration, spin, speeds, node_rates, work, impulse


def _substep(body, inverse, gravity, tether, held, external, wind, state, nodes, speeds, h):
    """One fourth-order step of body, line and accounts; returns the new values and the increments of the accounts."""
    t0, p0, r0, v0, w0 = (state.time_s, state.position_cm_m, state.rotation, state.velocity_cm_m_s,
                          state.angular_velocity_body_rad_s)
    with_line = tether is not None
    zero = np.zeros(3)

    def stage(time, dp, dv, theta, dw, dx, du):
        rotation = r0 if theta is zero else r0@rotation_from_vector(theta)
        omega = w0+dw
        rates = _derivatives(body, inverse, gravity, tether, held, external, wind, time, p0+dp, rotation, v0+dv, omega,
                             nodes+dx if with_line else None, speeds+du if with_line else None)
        velocity, acceleration, spin, node_velocity, node_rates, work, impulse = rates
        return (velocity, acceleration, _theta_rate(theta, omega), spin, node_velocity if with_line else 0.0,
                node_rates if with_line else 0.0, work, impulse)

    k1 = stage(t0, zero, zero, zero, zero, 0.0, 0.0)
    k2 = stage(t0+0.5*h, *(0.5*h*k for k in k1[:6]))
    k3 = stage(t0+0.5*h, *(0.5*h*k for k in k2[:6]))
    k4 = stage(t0+h, *(h*k for k in k3[:6]))
    dp, dv, theta, dw, dx, du, work, impulse = (h/6.0*(a+2.0*b+2.0*c+d) for a, b, c, d in zip(k1, k2, k3, k4))
    new = RigidState(t0+h, p0+dp, r0@rotation_from_vector(theta), v0+dv, w0+dw)
    if with_line:
        nodes, speeds = nodes+dx, speeds+du
        if not (np.isfinite(nodes).all() and np.isfinite(speeds).all()):
            raise FloatingPointError("tether state left the finite range (time step too large?)")
    return new, nodes, speeds, work, impulse


def plant_step(body, state, *, step_s, substeps=10, tether=None, nodes=None, node_velocities=None, held=None,
               external=None, wind=None, gravity_m_s2=GRAVITY_M_S2):
    """Advance body and line over one aerodynamic step; returns ``(state, nodes, node_velocities, account)``.

    ``nodes`` and ``node_velocities`` are the line's node state (``(N, 3)`` each; required with a ``tether``,
    refused without). ``held`` are the loads held in body axes over the step (default: none), ``external`` the
    per-stage callback of the module text, ``wind`` the ambient wind on the line (a uniform vector or a callable
    ``(points, time)``; default: still air). The line's own gravity and air density are the line's settings; the
    line's gravity must agree with ``gravity_m_s2`` (checked). The returned time is ``state.time_s + step_s``: a
    caller that marches many steps sets the times from its own step count if it needs them free of accumulated rounding.
    """
    if not isinstance(body, RigidBody) or not isinstance(state, RigidState):
        raise ValueError("plant_step needs a RigidBody and a RigidState")
    count = int(substeps)
    step = float(step_s)
    if count < 1 or not (np.isfinite(step) and step > 0.0):
        raise ValueError("the step must be positive and finite and substeps at least one")
    gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
    held = HeldLoads() if held is None else held
    if not isinstance(held, HeldLoads):
        raise ValueError("held must be a HeldLoads")
    h = step/count
    if tether is None:
        if nodes is not None or node_velocities is not None:
            raise ValueError("a node state was given without a tether")
    else:
        if not isinstance(tether, PlantTether):
            raise ValueError("tether must be a PlantTether")
        if nodes is None or node_velocities is None:
            raise ValueError("a tether needs its node positions and velocities")
        nodes = np.array(nodes, dtype=float)
        node_velocities = np.array(node_velocities, dtype=float)
        shape = (tether.line.s.num_nodes, 3)
        if nodes.shape != shape or node_velocities.shape != shape or not (np.isfinite(nodes).all() and np.isfinite(node_velocities).all()):
            raise ValueError("the node state must be finite with one row per node")
        bound = tether.line.rk4_step_bound_s()
        if h > bound:
            raise ValueError(f"the line's time step {h:.4g} s is above its stability bound {bound:.4g} s: use more substeps")
        if not np.array_equal(gravity, [0.0, 0.0, -tether.line.s.gravity_m_s2]):
            raise ValueError("the line and the body use different gravity")
    inverse = np.linalg.inv(body.inertia_body_kg_m2)
    start = state.time_s
    work, impulse = np.zeros(len(WORKS)), np.zeros((len(IMPULSES), 3))
    for k in range(count):
        state, nodes, node_velocities, dw, di = _substep(body, inverse, gravity, tether, held, external, wind, state, nodes,
                                                         node_velocities, h)
        # Sub-times from the start of the step, the last one ending exactly on start + step (no accumulated sum).
        state = RigidState(start+step if k == count-1 else start+(k+1)*h, state.position_cm_m, state.rotation,
                           state.velocity_cm_m_s, state.angular_velocity_body_rad_s)
        work, impulse = work+dw, impulse+di
    account = PlantAccount({name: float(value) for name, value in zip(WORKS, work)},
                           {name: value.copy() for name, value in zip(IMPULSES, impulse)})
    return state, nodes, node_velocities, account


def rates(body, state, *, tether=None, nodes=None, node_velocities=None, held=None, external=None, wind=None,
          gravity_m_s2=GRAVITY_M_S2):
    """Accelerations of the body at one state, as the first stage of a sub-step would compute them.

    Returns ``(acceleration_cm_m_s2, angular_acceleration_body_rad_s2)``: the inertial acceleration of the
    centre of mass and the rate of the body-axis angular velocity under the held loads, the line (evaluated at
    the given node state), gravity and the external callback. Nothing is advanced. A check of how loads enter
    the equations of motion (plant contract, step 1b part two, criterion P2a).
    """
    if not isinstance(body, RigidBody) or not isinstance(state, RigidState):
        raise ValueError("rates needs a RigidBody and a RigidState")
    held = HeldLoads() if held is None else held
    if not isinstance(held, HeldLoads):
        raise ValueError("held must be a HeldLoads")
    if (tether is None) != (nodes is None) or (tether is None) != (node_velocities is None):
        raise ValueError("give the tether together with its node positions and velocities, or none of them")
    gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
    out = _derivatives(body, np.linalg.inv(body.inertia_body_kg_m2), gravity, tether, held, external, wind, state.time_s,
                       state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s,
                       None if nodes is None else np.asarray(nodes, dtype=float),
                       None if node_velocities is None else np.asarray(node_velocities, dtype=float))
    return out[1], out[2]


def system_momentum(body, state, tether=None, node_velocities=None):
    """Linear momentum of the body and of the line's nodes."""
    momentum = body.mass_kg*state.velocity_cm_m_s
    if tether is not None:
        momentum = momentum+tether.line.node_mass*np.asarray(node_velocities, dtype=float).sum(axis=0)
    return momentum


def system_energy(body, state, tether=None, nodes=None, node_velocities=None, gravity_m_s2=GRAVITY_M_S2):
    """Mechanical energy: body kinetic (translation, rotation) and gravity potential, and the line's own energy.

    The line's part is ``LumpedTether.energy`` (kinetic, stretching, gravity; no bending energy), evaluated with the
    bridle point of this state as the line's end.
    """
    omega = state.angular_velocity_body_rad_s
    gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
    parts = dict(body_translation=0.5*body.mass_kg*float(state.velocity_cm_m_s@state.velocity_cm_m_s),
                 body_rotation=0.5*float(omega@(body.inertia_body_kg_m2@omega)),
                 body_gravity=-body.mass_kg*float(gravity@state.position_cm_m))
    if tether is not None:
        point, _ = tether.bridle.point(state.pose(body), nodes[-1], node_velocities[-1])
        line = tether.line.energy(nodes, node_velocities, tether.anchor_m, point)
        parts.update(line_kinetic=line["kinetic"], line_elastic=line["elastic"], line_gravity=line["gravity"])
    parts["total"] = float(sum(parts.values()))
    return parts
