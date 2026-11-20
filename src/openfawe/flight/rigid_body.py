"""Time propagation of one rigid body under given loads (G5 plant contract, step 1a).

Contract: ``docs/tasks/OF-0008-g5control20261006/plant-contract.zh.md``, section "第 1a 步". This is
the rigid body alone: the loads are whatever function the caller supplies; coupling with the
aerodynamics, the tether and the rotors is not here and is not decided.

State (``RigidState``): time, position and velocity of the **centre of mass** in the inertial frame,
attitude ``R`` (body to inertial) and the angular velocity in **body** components. Equations:

    dp/dt = v                         m dv/dt = F + m g
    dR/dt = R [w_b]x                  I dw_b/dt = R^T M - w_b x (I w_b)

``F`` is the resultant external force in inertial components without gravity, ``M`` the resultant
external moment about the centre of mass in inertial components, ``I`` the inertia about the centre
of mass in body axes (``flight.residual.RigidBody``). The signs are those of the residual account
(``flight.residual``): the required force and moment of a motion, given back as loads, reproduce
the motion.

Method: classical explicit fourth-order Runge-Kutta with a fixed step. The attitude is advanced
through a rotation-vector increment (Runge-Kutta-Munthe-Kaas): within a step ``R = R_n exp([theta]x)``
with

    d(theta)/dt = w_b + theta x w_b / 2 + theta x (theta x w_b) / 12

(the inverse of the differential of the exponential map, truncated after the second-order term,
which is enough for a fourth-order scheme), and ``R_{n+1} = R_n exp([theta]x)`` by the Rodrigues
formula. ``R`` is therefore a proper rotation by construction at every step; nothing is normalised
afterwards, so rounding accumulates over a long run (2.5e-14 after the 10 000 steps of the checks;
no longer run has been looked at). The truncated series assumes a small rotation per step
(``step * |w|``; the checks go to 0.025 rad) and the code does not guard it.
The loads are evaluated at every stage with the state of that stage. No adaptive step, no retry, no
limiter: a state that leaves the finite range raises ``FloatingPointError``.

The scheme is the common one (Munthe-Kaas 1998; Hairer, Lubich, Wanner, Geometric Numerical
Integration, section IV.8); it was implemented from the usual statement of it, not from the
sources, and is checked against analytic solutions and invariants (criteria P1a-P1k of the
contract, ``docs/tasks/OF-0008-g5control20261006/p1_rigid_body_checks.py``).

Nothing in the package uses this module: no existing path changes.
"""

from dataclasses import dataclass

import numpy as np

from openfawe.vec3 import cross3   # OF-0014: numpy.cross for 3-vectors, bit for bit

from .kinematics import Pose, rotation_from_vector
from .residual import RigidBody

GRAVITY_M_S2 = (0.0, 0.0, -9.81)
#: An attitude handed to ``RigidState`` must be orthogonal to this (the scheme itself keeps it to rounding per step).
ORTHOGONALITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RigidState:
    """State of the rigid body: centre-of-mass position and velocity (inertial), attitude, body angular velocity."""
    time_s: float
    position_cm_m: np.ndarray
    rotation: np.ndarray
    velocity_cm_m_s: np.ndarray
    angular_velocity_body_rad_s: np.ndarray

    def __post_init__(self):
        shapes = dict(position_cm_m=(3,), rotation=(3, 3), velocity_cm_m_s=(3,), angular_velocity_body_rad_s=(3,))
        for name, shape in shapes.items():
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{name} left the finite range")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.abs(self.rotation.T@self.rotation-np.eye(3)).max() > ORTHOGONALITY_TOLERANCE or np.linalg.det(self.rotation) <= 0.0:
            raise ValueError("the attitude must be a proper rotation")
        time = float(self.time_s)
        if not np.isfinite(time):
            raise FloatingPointError("time left the finite range")
        object.__setattr__(self, "time_s", time)

    @property
    def angular_velocity_rad_s(self):
        """Angular velocity in inertial components."""
        return self.rotation@self.angular_velocity_body_rad_s

    def pose(self, body):
        """The ``Pose`` of the G4 conventions: position and velocity of the body origin, inertial angular velocity."""
        arm = self.rotation@body.centre_of_mass_body_m          # origin -> centre of mass, inertial components
        omega = self.angular_velocity_rad_s
        return Pose(self.time_s, self.position_cm_m-arm, self.rotation, self.velocity_cm_m_s-cross3(omega, arm), omega)


def state_from_pose(pose, body):
    """The state whose body origin has the position, velocity and angular velocity of ``pose``."""
    rotation = np.asarray(pose.rotation, dtype=float)
    arm = rotation@body.centre_of_mass_body_m
    omega = np.asarray(pose.angular_velocity_rad_s, dtype=float)
    return RigidState(pose.time_s, np.asarray(pose.position_m, dtype=float)+arm, rotation,
                      np.asarray(pose.velocity_m_s, dtype=float)+cross3(omega, arm), rotation.T@omega)


def angular_momentum(body, state):
    """Angular momentum about the centre of mass, inertial components."""
    return state.rotation@(body.inertia_body_kg_m2@state.angular_velocity_body_rad_s)


def kinetic_energy(body, state):
    """Translational and rotational kinetic energy ``(E_trans, E_rot)``."""
    omega = state.angular_velocity_body_rad_s
    return (0.5*body.mass_kg*float(state.velocity_cm_m_s@state.velocity_cm_m_s),
            0.5*float(omega@(body.inertia_body_kg_m2@omega)))


def _rates(body, inverse_inertia, gravity, loads, time, position, rotation, velocity, omega):
    """Time derivatives of velocity and body angular velocity at one stage."""
    force, moment = loads(time, RigidState(time, position, rotation, velocity, omega))
    force = np.asarray(force, dtype=float).reshape(3)
    moment = np.asarray(moment, dtype=float).reshape(3)
    inertia = body.inertia_body_kg_m2
    return force/body.mass_kg+gravity, inverse_inertia@(rotation.T@moment-cross3(omega, inertia@omega))


def _theta_rate(theta, omega):
    """d(theta)/dt for the rotation-vector increment (inverse differential of the exponential map, second order)."""
    cross = cross3(theta, omega)
    return omega+0.5*cross+cross3(theta, cross)/12.0


def no_loads(time_s, state):
    """Zero force and zero moment (a body under gravity alone, or free of everything with zero gravity)."""
    return np.zeros(3), np.zeros(3)


def rigid_step(body, state, loads, step_s, *, gravity_m_s2=GRAVITY_M_S2):
    """One fixed step of the fourth-order scheme; returns the new ``RigidState``.

    ``loads(time_s, state)`` returns the external force (inertial, without gravity) and the external
    moment about the centre of mass (inertial) for a ``RigidState``; it is called four times, with
    the stage states.
    """
    if not isinstance(body, RigidBody) or not isinstance(state, RigidState):
        raise ValueError("rigid_step needs a RigidBody and a RigidState")
    h = float(step_s)
    if not (np.isfinite(h) and h > 0.0):
        raise ValueError("the step must be positive and finite")
    gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
    inverse = np.linalg.inv(body.inertia_body_kg_m2)
    t0, p0, r0, v0, w0 = (state.time_s, state.position_cm_m, state.rotation, state.velocity_cm_m_s,
                          state.angular_velocity_body_rad_s)
    zero = np.zeros(3)

    def stage(time, dp, dv, theta, dw):
        """Derivatives (dp, dv, dtheta, dw) at the state displaced by the given increments."""
        rotation = r0 if theta is zero else r0@rotation_from_vector(theta)
        velocity, omega = v0+dv, w0+dw
        acceleration, spin = _rates(body, inverse, gravity, loads, time, p0+dp, rotation, velocity, omega)
        return velocity, acceleration, _theta_rate(theta, omega), spin

    k1 = stage(t0, zero, zero, zero, zero)
    k2 = stage(t0+0.5*h, *(0.5*h*k for k in k1))
    k3 = stage(t0+0.5*h, *(0.5*h*k for k in k2))
    k4 = stage(t0+h, *(h*k for k in k3))
    dp, dv, theta, dw = (h/6.0*(a+2.0*b+2.0*c+d) for a, b, c, d in zip(k1, k2, k3, k4))
    return RigidState(t0+h, p0+dp, r0@rotation_from_vector(theta), v0+dv, w0+dw)


def propagate(body, state, loads, step_s, steps, *, gravity_m_s2=GRAVITY_M_S2):
    """``steps`` fixed steps from ``state``; returns the list of the ``steps + 1`` states, the first being ``state``.

    The time of state ``k`` is ``state.time_s + k * step_s`` computed as such (no accumulated sum), so
    that a long run ends on the time it was asked for.
    """
    count = int(steps)
    if count < 0:
        raise ValueError("the number of steps cannot be negative")
    h = float(step_s)
    states = [state]
    for k in range(count):
        new = rigid_step(body, states[-1], loads, h, gravity_m_s2=gravity_m_s2)
        states.append(RigidState(state.time_s+(k+1)*h, new.position_cm_m, new.rotation, new.velocity_cm_m_s,
                                 new.angular_velocity_body_rad_s))
    return states
