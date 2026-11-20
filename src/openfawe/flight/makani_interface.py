# SPDX-License-Identifier: Apache-2.0
# Definitions of the quantities exchanged with the Makani flight controller follow google/makani commit 1d66923
# (Apache-2.0): control/estimator/estimator_types.h and estimator_nav.c (state estimate), control/sensor_util.c
# (apparent wind and tether force in spherical form), control/ground_frame.c (wind direction),
# sim/models/actuators/servo.cc with config/m600/sim/servos_sim.py and config/m600/servos.py (servo reference model),
# sim/models/power_sys.cc (rate limit of the rotor speed command). Modifications (2026): NumPy implementation, truth
# in place of estimates, only the first stage of the servo model.
"""The plant side of the Makani controller interface: frames, the truth "estimate", servos and rotor speed commands.

Nothing here needs the controller library; the functions return plain numbers that ``flight.closed_loop`` writes
into the library's structures. Closed-loop contract (G5 step 2), items D3 to D5.

Frames. Makani's ground frame g (x along the ground-frame heading, z down) and the OpenFAWE inertial frame W are
related by ``W = diag(1, -1, -1) g``; Makani's body axes (x forward, y starboard, z down) and the case-file body axes
(x towards the trailing edge, y spanwise, z up) by ``diag(-1, 1, -1)``. Both maps are their own inverses. With ``R``
the OpenFAWE attitude (body to inertial), Makani's direction-cosine matrix from g to body is
``dcm_g2b = B R^T G`` (``G``, ``B`` the two diagonal maps).

The estimate is the truth of the plant, unfiltered, in Makani's definitions:

* ``Xg``, ``Vg``: position and velocity of the **body origin** in g; ``Vb = dcm_g2b Vg``; ``pqr``: body rates in
  Makani's body axes (also used for ``pqr_f``);
* ``Ag``: kinematic acceleration of the body origin in g; ``Ab_f = dcm_g2b Ag``; ``acc_norm_f = |Ag - g_g|`` (the
  norm of the specific force);
* apparent wind: the air's velocity relative to the body origin in Makani's body axes, **without** induced
  velocities (Makani's estimate calibrates the wing's upwash out);
* tether force: the force of the line **on the kite** at the bridle point, Makani's body axes;
* wind: the vector the air moves along, in g.
"""

from dataclasses import dataclass
import math

import numpy as np

G_TO_WORLD = np.diag([1.0, -1.0, -1.0])
MAKANI_BODY_TO_BODY = np.diag([-1.0, 1.0, -1.0])
FLAPS = ("A1", "A2", "A4", "A5", "A7", "A8", "Ele", "Rud")
RUDDER = 7


@dataclass(frozen=True)
class MakaniTruth:
    """The plant's state in Makani's frames (see the module text)."""
    Xg: np.ndarray
    Vg: np.ndarray
    dcm_g2b: np.ndarray
    pqr: np.ndarray
    Ag: np.ndarray
    apparent_wind_b: np.ndarray
    tether_force_b: np.ndarray
    wind_g: np.ndarray

    @property
    def Vb(self):
        return self.dcm_g2b@self.Vg

    @property
    def Ab(self):
        return self.dcm_g2b@self.Ag


def origin_acceleration(body, state, acceleration_cm_m_s2, angular_acceleration_body_rad_s2):
    """Inertial acceleration of the body origin from that of the centre of mass and the body-axis angular acceleration."""
    arm = state.rotation@body.centre_of_mass_body_m                     # origin -> centre of mass
    omega = state.rotation@state.angular_velocity_body_rad_s
    spin = state.rotation@np.asarray(angular_acceleration_body_rad_s2, dtype=float)
    return np.asarray(acceleration_cm_m_s2, dtype=float)-np.cross(spin, arm)-np.cross(omega, np.cross(omega, arm))


def makani_truth(body, state, wind_m_s, *, acceleration_origin_m_s2, tether_force_inertial_N=None):
    """``MakaniTruth`` of a rigid state: ``wind_m_s`` and the forces are inertial (W) components."""
    pose = state.pose(body)
    dcm_g2b = MAKANI_BODY_TO_BODY@pose.rotation.T@G_TO_WORLD
    wind = np.asarray(wind_m_s, dtype=float).reshape(3)
    tether = np.zeros(3) if tether_force_inertial_N is None else np.asarray(tether_force_inertial_N, dtype=float).reshape(3)
    return MakaniTruth(Xg=G_TO_WORLD@pose.position_m, Vg=G_TO_WORLD@pose.velocity_m_s, dcm_g2b=dcm_g2b,
                       pqr=MAKANI_BODY_TO_BODY@state.angular_velocity_body_rad_s,
                       Ag=G_TO_WORLD@np.asarray(acceleration_origin_m_s2, dtype=float).reshape(3),
                       apparent_wind_b=MAKANI_BODY_TO_BODY@(pose.rotation.T@(wind-pose.velocity_m_s)),
                       tether_force_b=MAKANI_BODY_TO_BODY@(pose.rotation.T@tether), wind_g=G_TO_WORLD@wind)


def plant_from_makani(body, time_s, Xg, Vg, dcm_g2b, pqr):
    """The inverse map for the kinematic part: the ``RigidState`` whose Makani quantities are the arguments."""
    from .kinematics import Pose
    from .rigid_body import state_from_pose
    rotation = G_TO_WORLD@np.asarray(dcm_g2b, dtype=float).T@MAKANI_BODY_TO_BODY
    omega_body = MAKANI_BODY_TO_BODY@np.asarray(pqr, dtype=float)
    pose = Pose(float(time_s), G_TO_WORLD@np.asarray(Xg, dtype=float), rotation, G_TO_WORLD@np.asarray(Vg, dtype=float),
                rotation@omega_body)
    return state_from_pose(pose, body)


def apparent_wind_sph(apparent_wind_b):
    """``ApparentWindCartToSph`` (control/sensor_util.c lines 246-252): ``(v, alpha, beta)``."""
    x, y, z = (float(c) for c in apparent_wind_b)
    return math.sqrt(x*x+y*y+z*z), math.atan2(-z, -x), math.atan2(-y, math.hypot(x, z))


def tether_force_sph(force_b):
    """``TetherForceCartToSph`` (control/sensor_util.c lines 97-101): ``(tension, roll, pitch)``."""
    x, y, z = (float(c) for c in force_b)
    return math.sqrt(x*x+y*y+z*z), math.atan2(-y, math.hypot(x, z)), math.atan2(x, z)


def wind_direction(wind_g, azi_ref_offset=0.0):
    """``VecGToAzimuth`` of the wind vector (control/ground_frame.c lines 92-96): wrapped to [-pi, pi)."""
    angle = math.atan2(float(wind_g[1]), float(wind_g[0]))+float(azi_ref_offset)
    return (angle+math.pi) % (2.0*math.pi)-math.pi


class ServoReference:
    """First stage of Makani's servo model: a first-order reference model with a rate limit, per flap.

    ``d(ref)/dt = sat(2 pi f_c (cmd - ref), +-rate)`` on the servo shaft angle (``servo.cc`` lines 172-186; cut-off
    15.91 Hz, rate limit 0.85 rad/s), the command saturated to the position limits of the servo first (lines 62-69;
    limits of ``servos_sim.py`` lines 104-152). The ailerons and the elevator are direct drive; the rudder shaft
    angle is ``asin(flap / 0.401)`` and the rudder angle ``0.401 sin(shaft)`` (``config/m600/servos.py``). The
    motor, current and hinge-moment stages of the source are not modelled: the flap follows the reference model.
    A step is integrated exactly (the equation is piecewise linear).
    """
    LOWER_RAD = np.radians([-81.0, -81.0, -97.0, -97.0, -81.0, -81.0, -90.0, -78.0])     # shaft angles
    UPPER_RAD = np.radians([10.0, 10.0, 46.0, 46.0, 10.0, 10.0, 16.0, 78.0])
    RUDDER_RATIO = 0.401

    def __init__(self, flaps_rad, *, cutoff_hz=15.91, rate_limit_rad_s=0.85):
        self.gain = 2.0*math.pi*float(cutoff_hz)
        self.rate = float(rate_limit_rad_s)
        if not (self.gain > 0.0 and self.rate > 0.0):
            raise ValueError("the cut-off frequency and the rate limit must be positive")
        self.shaft = self._shaft(flaps_rad)
        self.rate_limited = np.zeros(len(FLAPS), dtype=bool)
        self.at_limit = np.zeros(len(FLAPS), dtype=bool)

    def _shaft(self, flaps_rad):
        flaps = np.array(flaps_rad, dtype=float).reshape(len(FLAPS))
        if not np.isfinite(flaps).all():
            raise FloatingPointError("flap commands left the finite range")
        shaft = flaps.copy()
        shaft[RUDDER] = math.asin(min(max(flaps[RUDDER]/self.RUDDER_RATIO, -1.0), 1.0))
        return shaft

    @property
    def flaps_rad(self):
        flaps = self.shaft.copy()
        flaps[RUDDER] = self.RUDDER_RATIO*math.sin(self.shaft[RUDDER])
        return flaps

    def step(self, command_flaps_rad, step_s):
        """Advance the reference model by ``step_s`` under a constant command; returns the new flap angles."""
        dt = float(step_s)
        if not dt > 0.0:
            raise ValueError("the step must be positive")
        wanted = self._shaft(command_flaps_rad)
        command = np.minimum(np.maximum(wanted, self.LOWER_RAD), self.UPPER_RAD)
        self.at_limit = wanted != command
        knee = self.rate/self.gain                          # error below which the model is linear
        for k in range(len(FLAPS)):
            error = command[k]-self.shaft[k]
            ramp = max((abs(error)-knee)/self.rate, 0.0)     # time spent on the rate limit
            self.rate_limited[k] = ramp > 0.0
            if ramp >= dt:
                self.shaft[k] += math.copysign(self.rate*dt, error)
                continue
            if ramp > 0.0:
                self.shaft[k] += math.copysign(self.rate*ramp, error)
                error = command[k]-self.shaft[k]
            self.shaft[k] += error*(1.0-math.exp(-self.gain*(dt-ramp)))
        return self.flaps_rad


def rate_limited_speeds(command_rad_s, previous_rad_s, step_s, rate_limit_rad_s2=500.0):
    """Rotor speeds held over a step: the command, moved from the previous speeds by at most ``rate * step``.

    The rate limit is the one Makani's simulator applies to the speed command (``sim/models/power_sys.cc`` lines
    91-115, 500 rad/s^2); the motor's own speed loop and the rotor's equation of motion are not modelled.
    """
    command = np.asarray(command_rad_s, dtype=float)
    if previous_rad_s is None:
        return command.copy()
    previous = np.asarray(previous_rad_s, dtype=float)
    reach = float(rate_limit_rad_s2)*float(step_s)
    return previous+np.clip(command-previous, -reach, reach)
