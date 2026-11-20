# SPDX-License-Identifier: Apache-2.0
# Transcribed from google/makani commit 1d66923 (Apache-2.0):
#   sim/models/actuators/ground_station_v2.{h,cc}  (GroundStationV2, ReelController, TransformController,
#                                                   HighTensionController, AttenuateAndApplyDeadZone),
#   sim/models/ground_station_util.cc              (RacetrackTetherIntegrator),
#   sim/models/sensors/ground_station_v2_sensors.{h,cc} (GS02 sensor gating),
#   sim/state.h, sim/models/model.cc               (DiscreteState and Model::DiscreteUpdate semantics),
#   sim/physics/reference_frame.cc                 (frame chain, position and velocity transforms),
#   common/c_math/{util,filter,geometry}.c         (Wrap, Saturate, Sign, Sqrt, RateLimit, Lpf2, AngleToDcm, CartToSph),
#   control/vessel_frame.c, control/sensor_util.c   (VecVToElevation/Azimuth, CalcDcmWdToGsg0).
# Modifications (2026): Python/NumPy; the vessel frame is the ground frame (an onshore, fixed platform); the
# simulator's connection system is replaced by explicit calls (see the module text).
"""Makani's GS02 ground station, transcribed as coded (plan 3.7 part 2; full-cycle contract revision 4).

Everything here is in Makani's units and frames: the root frame is NED, which Makani requires to be the ground
frame (full_system.cc:718), and the platform's parent (the vessel frame) is taken equal to it - the onshore, fixed
platform of the contract. A caller that works in the plant's z-up frame converts with
``flight.makani_interface.G_TO_WORLD``.

What the model is (ground_station_v2.cc; the line numbers below are of that file unless stated):

* Continuous states (960-966, 980-993): platform azimuth and its rate (first-order velocity servo, gain
  ``azi_accel_kp``), drum angle, rate and acceleration (second-order velocity servo, natural frequency
  ``winch_drive_natural_freq`` and damping ratio ``winch_drive_damping_ratio``) and the detwist angle (rate given
  directly). The tether force does not act on the drum or the platform (ideal servos); only the HighTension brake
  logic uses its moment.
* The discrete step (995-1077): the mode machine Reel -> Transform -> HighTension and back, the inputs of the three
  mode controllers, the proximity sensor. The mode controllers (108-592) run after it within the same step, each at
  its own sample time (0.01 s).
* Geometry: free tether length against drum angle (766-799 with the racetrack integral of ground_station_util.cc),
  the tether anchor (812-877), the GSG angles of a tether direction (879-916), the levelwind engagement (642-667).
* The GS02 sensors (ground_station_v2_sensors.cc): which of the GSG or levelwind readings are zeroed.

Makani's discrete-state semantics are reproduced exactly (``DiscreteState``): a state with a positive sample time
accepts only the first write in each sample period, a state with sample time 0 accepts every write; a write takes
effect at once. This is what makes the order of calls matter, and the order here is Makani's.

How a caller drives it (the simulator's step, kite_sim.cc:173-204, and the connections of full_system.cc:696-732):

1. ``discrete_update(t)`` once per simulator step (0.01 s), before integrating the step;
2. continuous integration over the step with ``derivatives(x)`` (``mode_controller_commands`` is part of it, as the
   connection at level 1 is re-run at every right-hand-side evaluation);
3. the controller's command for the next step through ``set_from_command`` (the simulator receives it after
   integrating); the tether force through ``set_tether_force`` and the levelwind engagement through
   ``set_levelwind_engaged`` from the tether direction (the simulator refreshes both at every right-hand-side
   evaluation; the discrete step uses the last values).

As-coded behaviour that is reproduced and not repaired (the survey's list B1-B9, full-cycle contract revision 4),
each marked in the code with its tag:

* B1 ``SetCommonInputs`` writes ``drum_pos_`` twice (78-79); the second write (the drum rate) is ignored, so the
  controllers' ``drum_vel_`` stays at its initial 0.
* B2 on the main wrap the anchor velocity is ``dx_dtheta_main_wrap`` without the factor drum rate (867).
* B3 the Reel azimuth unwrap handles only ``d > pi`` (126-132); the command filter starts at 0 and is held while
  the Reel controller is disabled.
* B4 the HighTension controller runs whether enabled or not (371-380), and its state-machine variable is a
  function-static (386): shared by every instance in the process (``HighTensionController.reset_function_static``
  is a test hook that sets it back to its start value).
* B5 the HighTension azimuth command is ``total_torque / Iz * ts`` (589-590), a velocity increment per sample used
  as a velocity set point.
* B6 the Transform controller's detwist targets are computed and not used (the detwist law of 732-739 is used).
* B7 the sensors stop the process if the mode is Transform and the mode command is neither Reel nor HighTension
  (ground_station_v2_sensors.cc:80-82); here a ``RuntimeError``.
* B8 a misplaced parenthesis in the panel contact (ground_station_util.cc:95-96): in ``ground.perch``.
* B9 ``winch_drive_natural_freq`` is labelled Hz in the configuration and used as rad/s (985-989).

Debug checks of the source (``DCHECK``) are kept as exceptions: the gs_azi_target range (1022) and a non-negative
free length (797). Left out: the single-precision cast of the sensor message fields (as in part 1's
``flight.makani_full.gs02_sensors``); the GS02 telemetry.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from openfawe.vec3 import cross3   # OF-0014: numpy.cross for 3-vectors, bit for bit

PI = math.pi
EPS = float(np.finfo(float).eps)
DBL_MAX = float(np.finfo(float).max)
DBL_TOL = 1e-9                         # common/c_math/util.h:36

# GroundStationMode (avionics/common/plc_messages.h).
MODE_MANUAL, MODE_HIGH_TENSION, MODE_REEL, MODE_TRANSFORM = 0, 1, 2, 3


# ----- common/c_math helpers, with the source's operation order ------------------------------------------------------
def saturate(x, low, high):
    """Saturate (util.c:211): fmin(fmax(x, low), high)."""
    if not low <= high:
        raise ValueError("saturate: low > high")
    return min(max(x, low), high)


def wrap(x, left, right):
    """Wrap (util.c:556-564), with C's fmod."""
    if not left < right:
        raise ValueError("wrap: left >= right")
    wrap0 = math.fmod(x - left, right - left)
    return wrap0 + left if wrap0 >= 0.0 else wrap0 + right


def sign(x):
    """Sign (util.c:74): -1, 0 or +1 (an int, as the source)."""
    return int(0.0 < x) - int(x < 0.0)


def sqrt_pos(x):
    """Sqrt (util.c:605): sqrt(fmax(x, 0))."""
    return math.sqrt(max(x, 0.0))


def rate_limit(u, low, high, ts, y_z1):
    """RateLimit (filter.c:103-110); returns (output, new state)."""
    if not (high >= low and ts > 0.0):
        raise ValueError("rate_limit: bad arguments")
    y_z1 = y_z1 + saturate(u - y_z1, low*ts, high*ts)
    return y_z1, y_z1


def second_order_lowpass_coefficients(fc, zeta, ts):
    """SecondOrderFilterCoeff with kFilterTypeLowPass (filter.c:323-347): Tustin, no prewarp; (a, b)."""
    if not (ts > DBL_TOL and fc >= 0.0 and zeta > 0.0):
        raise ValueError("second_order_lowpass_coefficients: bad arguments")
    w = 2.0*PI*fc
    w2 = w*w
    ts2 = ts*ts
    a0 = 4.0/ts2 + w2 + (4.0*zeta*w)/ts
    a0 = max(a0, DBL_TOL)
    a = (1.0, (2.0*w2 - 8.0/ts2)/a0, (4.0/ts2 + w2 - (4.0*zeta*w)/ts)/a0)
    w2_a0 = w2/a0
    b = (w2_a0, 2.0*w2_a0, w2_a0)
    return a, b


def filter3(u, a, b, z):
    """Filter (filter.c:150-171) for n = 3, direct form II; returns (y, new z)."""
    y = 0.0
    z_tmp = u
    for i in (1, 2):
        z_tmp -= a[i]*z[i - 1]
        y += b[i]*z[i - 1]
    y += b[0]*z_tmp
    return y, (z_tmp, z[0])


def lpf2(u, fc, zeta, ts, z):
    """Lpf2 (filter.c:417-421); returns (y, new z)."""
    a, b = second_order_lowpass_coefficients(fc, zeta, ts)
    return filter3(u, a, b, z)


def angle_to_dcm(r1, r2, r3, order="zyx"):
    """AngleToDcm (geometry.c:111-235) for the two orders used here; a passive rotation (parent to local)."""
    s1, c1, s2, c2, s3, c3 = math.sin(r1), math.cos(r1), math.sin(r2), math.cos(r2), math.sin(r3), math.cos(r3)
    if order == "zyx":
        return np.array([[c1*c2, s1*c2, -s2],
                         [c1*s2*s3 - s1*c3, s1*s2*s3 + c1*c3, c2*s3],
                         [c1*s2*c3 + s1*s3, s1*s2*c3 - c1*s3, c2*c3]])
    if order == "xyz":
        return np.array([[c2*c3, s1*s2*c3 + c1*s3, s1*s3 - c1*s2*c3],
                         [-c2*s3, c1*c3 - s1*s2*s3, c1*s2*s3 + s1*c3],
                         [s2, -s1*c2, c1*c2]])
    raise ValueError(order)


def calc_dcm_wd_to_gsg0(detwist_ele, detwist_pos):
    """CalcDcmWdToGsg0 (sensor_util.c:383-387)."""
    return angle_to_dcm(0.0, PI*0.5 - detwist_ele, detwist_pos + PI, "xyz")


def attenuate_and_apply_dead_zone(vel_cmd, pos_error, max_accel, dead_zone_half_width):
    """AttenuateAndApplyDeadZone (34-58)."""
    v_max = sqrt_pos(abs(pos_error)*2.0*max_accel)
    modified_cmd = saturate(vel_cmd, -v_max, v_max)
    dead_zone_upper = sqrt_pos(abs(dead_zone_half_width)*2.0*max_accel)
    if modified_cmd < -dead_zone_upper:
        modified_cmd += dead_zone_upper
    elif modified_cmd < dead_zone_upper:
        modified_cmd = 0.0
    else:
        modified_cmd -= dead_zone_upper
    return modified_cmd


def vec_v_to_elevation(v):
    """VecVToElevation (vessel_frame.c:29-31)."""
    return math.atan2(-v[2], math.hypot(v[0], v[1]))


def vec_v_to_azimuth(v, azi_ref_offset=0.0):
    """VecVToAzimuth (vessel_frame.c:23-27)."""
    return wrap(math.atan2(v[1], v[0]) + azi_ref_offset, -PI, PI)


# ----- sim/state.h and sim/models/model.cc ---------------------------------------------------------------------------
class DiscreteState:
    """Makani's ``DiscreteState<T>`` (state.h:489-500): with ts > 0 the first write of each sample period wins."""
    __slots__ = ("val", "ts", "t_z1")

    def __init__(self, ts, val):
        self.ts, self.val, self.t_z1 = float(ts), val, -DBL_MAX

    def update(self, t, value):
        t_cmp = t*(1.0 + EPS)
        if t_cmp - self.t_z1 >= self.ts:
            self.val = value
            self.t_z1 = t_cmp - math.fmod(t_cmp, self.ts) if self.ts > 0.0 else t


class _Model:
    """The sample-time gate of ``Model::DiscreteUpdate`` (model.cc:134-148)."""

    def __init__(self, ts):
        self.ts, self._t_z1 = float(ts), -DBL_MAX

    def discrete_update(self, t):
        t_cmp = t*(1.0 + EPS)
        if t_cmp - self._t_z1 >= self.ts:
            self.discrete_step(t)
            self._t_z1 = t_cmp - math.fmod(t_cmp, self.ts) if self.ts > 0.0 else t

    def discrete_step(self, t):
        raise NotImplementedError


# ----- sim/physics/reference_frame.cc ---------------------------------------------------------------------------------
class Frame:
    """ReferenceFrame (reference_frame.cc:20-62, 120-140) for positions, velocities and rotations.

    ``dcm`` is the root-to-local DCM; ``pos``, ``vel`` and ``omega`` are the origin's position and velocity and the
    frame's angular velocity, in root coordinates. A velocity transform adds only the origin's velocity (no
    omega x r term): the source obtains the velocity of a point by making a frame at that point.
    """
    __slots__ = ("dcm", "pos", "vel", "omega")

    def __init__(self, dcm, pos, vel, omega):
        self.dcm, self.pos, self.vel, self.omega = dcm, pos, vel, omega

    @classmethod
    def root(cls):
        zero = np.zeros(3)
        return cls(np.eye(3), zero, zero, zero)

    def child(self, origin_pos_parent, dcm_parent_to_local=None, origin_vel_parent=None, omega_parent=None):
        origin_pos_parent = np.asarray(origin_pos_parent, dtype=float)
        dcm_local = np.eye(3) if dcm_parent_to_local is None else np.asarray(dcm_parent_to_local, dtype=float)
        vel_parent = np.zeros(3) if origin_vel_parent is None else np.asarray(origin_vel_parent, dtype=float)
        omega_local = np.zeros(3) if omega_parent is None else np.asarray(omega_parent, dtype=float)
        dcm = dcm_local@self.dcm
        offset_root = self.dcm.T@origin_pos_parent
        pos = offset_root + self.pos
        omega = self.dcm.T@omega_local + self.omega
        vel = self.dcm.T@vel_parent + self.vel + cross3(self.omega, offset_root)
        return Frame(dcm, pos, vel, omega)

    def _origin(self, kind):
        return {"position": self.pos, "velocity": self.vel, "vector": np.zeros(3)}[kind]

    def to_root(self, kind, vec):
        return self.dcm.T@np.asarray(vec, dtype=float) + self._origin(kind)

    def from_root(self, kind, vec):
        return self.dcm@(np.asarray(vec, dtype=float) - self._origin(kind))

    def transform_to(self, frame, kind, vec):
        return frame.from_root(kind, self.to_root(kind, vec))

    def rotate_to(self, frame, vec):
        return frame.dcm@(self.dcm.T@np.asarray(vec, dtype=float))


# ----- parameters -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gs02Params:
    """The parameters the model reads, from the file ``makani_lib/gen_gs02_params.py`` writes (Makani's own
    configuration, land test site). ``raw`` keeps the whole file."""
    raw: dict

    @property
    def gs02(self):
        return self.raw["ground_station"]["gs02"]

    @property
    def sim(self):
        return self.raw["gs02_sim"]

    @property
    def mclaren(self):
        return self.raw["gs02_sim"]["mclaren"]

    @property
    def tether_length(self):
        return float(self.raw["tether"]["length"])

    @property
    def azi_ref_offset(self):
        return float(self.raw["ground_station"]["azi_ref_offset"])


def load_gs02_params(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "openfawe.makani_gs02_params.v1":
        raise ValueError(f"{path}: not an openfawe.makani_gs02_params.v1 file")
    if data["ground_station"]["azi_ref_offset"] != 0.0:
        raise ValueError("azi_ref_offset must be 0 (ground_station_v2.cc:689)")
    return Gs02Params(data)


# ----- sim/models/ground_station_util.cc ------------------------------------------------------------------------------
class RacetrackTetherIntegrator:
    """Length of tether wrapped on the racetrack (ground_station_util.cc:119-151)."""

    def __init__(self, params):
        gs02, angles = params.gs02, params.gs02["drum_angles"]
        self.theta_high = float(angles["racetrack_high"])
        self.tau_max = float(angles["racetrack_high"]) - float(angles["racetrack_low"])
        r0 = float(gs02["gsg_pos_drum"][2])
        dr_dtau = (float(gs02["drum_radius"]) - float(gs02["gsg_pos_drum"][2]))/self.tau_max
        dx_dtau = (float(params.sim["wrap_start_posx_drum"]) - float(gs02["gsg_pos_drum"][0]))/self.tau_max
        self.a = dr_dtau*dr_dtau
        self.b = 2.0*r0*dr_dtau
        self.c = r0*r0 + dx_dtau*dx_dtau + dr_dtau*dr_dtau

    def integrand(self, tau):
        """d(length)/d(tau) = sqrt(a tau^2 + b tau + c) (the helix-on-a-cone arc length the antiderivative is of)."""
        return math.sqrt(self.a*tau*tau + self.b*tau + self.c)

    def antiderivative(self, tau):
        a, b, c = self.a, self.b, self.c
        return ((tau/2.0 + b/(4.0*a))*sqrt_pos(a*tau*tau + b*tau + c)
                + (c/(2.0*sqrt_pos(a)) - b*b/(8.0*a**1.5))*math.asinh((2.0*a*tau + b)/math.sqrt(4.0*a*c - b*b)))

    def wrapped_length(self, drum_angle):
        tau = saturate(self.theta_high - drum_angle, 0.0, self.tau_max)
        return self.antiderivative(tau) - self.antiderivative(0.0)


# ----- the mode controllers -------------------------------------------------------------------------------------------
class _ModeController(_Model):
    """Gs02ModeController (62-81)."""

    def __init__(self, ts):
        super().__init__(ts)
        self.azi_pos, self.azi_vel = DiscreteState(ts, 0.0), DiscreteState(ts, 0.0)
        self.drum_pos, self.drum_vel = DiscreteState(ts, 0.0), DiscreteState(ts, 0.0)
        self.wing_azi, self.enabled = DiscreteState(ts, 0.0), DiscreteState(ts, False)
        self.azi_vel_cmd, self.drum_vel_cmd = DiscreteState(ts, 0.0), DiscreteState(ts, 0.0)

    def set_common_inputs(self, t, azi_pos, azi_vel, drum_pos, drum_vel, wing_azi):
        self.azi_pos.update(t, azi_pos)
        self.azi_vel.update(t, azi_vel)
        self.drum_pos.update(t, drum_pos)
        self.drum_pos.update(t, drum_vel)      # B1: as coded (78-79); ignored (first write wins), drum_vel_ stays 0
        self.wing_azi.update(t, wing_azi)

    def enable(self, t):
        self.enabled.update(t, True)

    def disable(self, t):
        self.enabled.update(t, False)


class ReelController(_ModeController):
    """ReelController (83-208)."""

    def __init__(self, params):
        mc = params.mclaren
        super().__init__(mc["ts"])
        self.p, self.ts_c = mc["reel"], float(mc["ts"])
        self.drum_radius = float(params.gs02["drum_radius"])
        ts = self.ts_c
        self.drum_linear_vel_cmd_from_wing = DiscreteState(ts, 0.0)
        self.azi_cmd_z = DiscreteState(ts, (0.0, 0.0))
        self.azi_cmd_delay_state = DiscreteState(ts, 0.0)
        self.azi_cmd_dead_zone = DiscreteState(ts, 0.0)
        self.azi_in_dead_zone_z1 = DiscreteState(ts, False)
        self.azi_delay_state = DiscreteState(ts, 0.0)
        self.azi_rate_limit_state = DiscreteState(ts, 0.0)
        self.drum_vel_rate_limit_state = DiscreteState(ts, 0.0)
        self.levelwind_engaged = DiscreteState(ts, True)

    def discrete_step(self, t):
        p, ts = self.p, self.ts_c
        if not self.enabled.val:
            self.azi_cmd_z.update(t, self.azi_cmd_z.val)
            self.azi_cmd_delay_state.update(t, self.azi_cmd_delay_state.val)
            self.azi_delay_state.update(t, self.azi_vel.val)
            self.azi_rate_limit_state.update(t, self.azi_vel.val)
            self.drum_vel_rate_limit_state.update(t, self.drum_vel.val)
            return
        # Azimuth velocity command (118-176).
        target_pos = self.wing_azi.val + p["azi_offset_from_wing"]
        unwrapped_target_pos = self.azi_cmd_delay_state.val
        twopimoddiff = math.fmod(self.azi_cmd_delay_state.val - target_pos, 2.0*PI)
        if twopimoddiff > PI:                     # B3: as coded, no branch for a difference below -pi
            unwrapped_target_pos -= twopimoddiff - 2.0*PI
        else:
            unwrapped_target_pos -= twopimoddiff
        azi_cmd, azi_cmd_z = lpf2(unwrapped_target_pos, p["azi_cmd_filter_omega"]/2.0/PI, p["azi_cmd_filter_zeta"], ts,
                                  self.azi_cmd_z.val)
        self.azi_cmd_z.update(t, azi_cmd_z)
        output_cmd = self.azi_delay_state.val
        azi_vel_cmd = (azi_cmd - self.azi_cmd_delay_state.val)/ts
        self.azi_cmd_delay_state.update(t, azi_cmd)
        pos_error = wrap(azi_cmd - self.azi_pos.val, -PI, PI)
        effective_dead_zone = (self.azi_cmd_dead_zone.val if self.azi_in_dead_zone_z1.val
                               else self.azi_cmd_dead_zone.val*p["little_dead_zone"])
        if abs(pos_error) <= effective_dead_zone:
            self.azi_in_dead_zone_z1.update(t, True)
            pos_error = 0.0
        else:
            self.azi_in_dead_zone_z1.update(t, False)
        pos_error_saturated = saturate(pos_error, -p["azi_error_max"], p["azi_error_max"])
        self.azi_delay_state.update(t, p["azi_vel_cmd_kp"]*pos_error_saturated + p["azi_vel_cmd_ff_gain"]*azi_vel_cmd)
        output_cmd, cmd_z1 = rate_limit(output_cmd, -p["azi_vel_cmd_rate_limit"], p["azi_vel_cmd_rate_limit"], ts,
                                        self.azi_rate_limit_state.val)
        self.azi_rate_limit_state.update(t, cmd_z1)
        self.azi_vel_cmd.update(t, output_cmd)
        # Drum velocity command (178-207).
        cmd = self.drum_linear_vel_cmd_from_wing.val/self.drum_radius
        if not self.levelwind_engaged.val:
            cmd = 0.0
        if cmd > 0.0:
            pos_error = p["drum_angle_upper_limit"] - self.drum_pos.val
        else:
            pos_error = -math.inf
        if cmd*pos_error < 0:                     # 0 * -inf is NaN, not < 0, as in C
            cmd = 0.0
        cmd = attenuate_and_apply_dead_zone(cmd, p["drum_angle_upper_limit"] - self.drum_pos.val, p["max_drum_accel"], 0.0)
        cmd, cmd_z1 = rate_limit(cmd, -p["max_drum_accel"], p["max_drum_accel"], ts, self.drum_vel_rate_limit_state.val)
        self.drum_vel_rate_limit_state.update(t, cmd_z1)
        self.drum_vel_cmd.update(t, cmd)


class TransformController(_ModeController):
    """TransformController (210-335)."""

    def __init__(self, params):
        mc = params.mclaren
        super().__init__(mc["ts"])
        self.p = mc["transform"]
        ts = float(mc["ts"])
        self.azi_rate_limit_state = DiscreteState(ts, 0.0)
        self.winch_rate_limit_state = DiscreteState(ts, 0.0)
        self.detwist_pos_cmd = DiscreteState(ts, 0.0)
        self.unpause_transform = DiscreteState(ts, False)
        self.transform_stage = DiscreteState(ts, 0)
        self.ht2reel = DiscreteState(ts, False)
        self.transform_complete = DiscreteState(ts, False)

    @property
    def detwist_max_vel(self):
        return float(self.p["detwist_max_vel"])

    @property
    def stage(self):
        return self.transform_stage.val

    def start_transform(self, t, ht2reel):
        self.transform_stage.update(t, 0)
        self.ht2reel.update(t, bool(ht2reel))
        self.transform_complete.update(t, False)

    def discrete_step(self, t):
        p, ts = self.p, self.ts
        if not self.enabled.val:
            self.azi_rate_limit_state.update(t, self.azi_vel.val)
            self.winch_rate_limit_state.update(t, self.drum_vel.val)      # drum_vel_ is always 0 (B1)
            self.transform_complete.update(t, False)
            self.transform_stage.update(t, 0)
            return
        s = self.transform_stage.val
        # Azimuth (244-268).
        target = self.wing_azi.val + p["azi_offset_from_wing"]
        if self.ht2reel.val:
            target += p["azi_targets_ht2reel"][s]
            tol = p["azi_tols_ht2reel"][s]
        else:
            target += p["azi_targets_reel2ht"][s]
            tol = p["azi_tols_reel2ht"][s]
        pos_error = wrap(target - self.azi_pos.val, -PI, PI)
        vel_cmd = sign(pos_error)*p["azi_nominal_vel"]
        vel_cmd = attenuate_and_apply_dead_zone(vel_cmd, pos_error, p["azi_decel_ratio"]*p["azi_max_accel"],
                                                p["azi_dead_zone_half_width"])
        vel_cmd, _ = rate_limit(vel_cmd, -p["azi_max_accel"], p["azi_max_accel"], ts, self.azi_rate_limit_state.val)
        self.azi_rate_limit_state.update(t, vel_cmd)
        self.azi_vel_cmd.update(t, vel_cmd)
        azi_target_satisfied = -tol < pos_error < tol
        # Winch (270-295).
        if self.ht2reel.val:
            target, tol = p["winch_targets_ht2reel"][s], p["winch_tols_ht2reel"][s]
        else:
            target, tol = p["winch_targets_reel2ht"][s], p["winch_tols_reel2ht"][s]
        pos_error = target - self.drum_pos.val
        vel_cmd = sign(pos_error)*p["winch_nominal_vel"]
        vel_cmd = attenuate_and_apply_dead_zone(vel_cmd, pos_error, p["winch_decel_ratio"]*p["winch_max_accel"],
                                                p["winch_dead_zone_half_width"])
        vel_cmd, _ = rate_limit(vel_cmd, -p["winch_max_accel"], p["winch_max_accel"], ts, self.winch_rate_limit_state.val)
        self.winch_rate_limit_state.update(t, vel_cmd)
        self.drum_vel_cmd.update(t, vel_cmd)
        winch_target_satisfied = target - tol < self.drum_pos.val < target + tol
        # Detwist position command (297-305): computed, not used by the ground station (B6).
        self.detwist_pos_cmd.update(t, p["detwist_targets_ht2reel"][s] if self.ht2reel.val else p["detwist_targets_reel2ht"][s])
        # Stage progression (307-328).
        stage_complete = azi_target_satisfied and winch_target_satisfied
        if self.ht2reel.val:
            last_stage = 4
            next_stage = min(s + 1, last_stage)
            if not self.unpause_transform.val and s in (1, 2):
                stage_complete = False
        else:
            last_stage = 1
            next_stage = 4 if s == 0 else max(s - 1, last_stage)
            if not self.unpause_transform.val and s == 3:
                stage_complete = False
        self.transform_complete.update(t, s == last_stage and stage_complete)
        self.transform_stage.update(t, next_stage if stage_complete else s)


class HighTensionController(_ModeController):
    """HighTensionController (337-592)."""

    BASE, WAIT_POS_OVERLOAD, WAIT_HPU_POS, AZI_POS_ROTATION = 0, 1, 2, 3
    ACTIVE_BRAKING, WAIT_NEG_OVERLOAD, WAIT_HPU_NEG, AZI_NEG_ROTATION = 5, 6, 7, 8
    BRAKES_ON, BRAKES_REGULATED, BRAKES_OFF = 1, 2, 3
    _n_state_machine_val = 0          # B4: the function-static of RunControlSwitchingLogic (386), one per process

    @classmethod
    def reset_function_static(cls):
        """Test hook: the function-static back to its start value (kBaseCase), as at the start of a process."""
        cls._n_state_machine_val = cls.BASE

    def __init__(self, params):
        mc = params.mclaren
        super().__init__(mc["ts"])
        self.p = mc["high_tension"]
        ts = float(mc["ts"])
        self.active_braking_first_entry = True
        self.active_braking_t0 = 0.0
        self.angular_rate_tol = 1.7e-4
        self.hpu_cmd_change_t0 = 0.0
        self.hpu_delay = 0.4
        self.ht_ts = ts
        self.brake_torque = DiscreteState(ts, 0.0)
        self.tether_torque = DiscreteState(ts, 0.0)
        self.brake_torque_cmd = DiscreteState(ts, 0.0)
        self.n_hpu_mode_demand = DiscreteState(ts, self.BRAKES_ON)
        self.azi_velocity_dir = DiscreteState(ts, 0.0)
        self.n_state_machine = DiscreteState(ts, self.BASE)
        self.total_torque = DiscreteState(ts, 0.0)
        self.n_hpu_mode = DiscreteState(ts, self.BRAKES_ON)
        self.n_hpu_mode_last = DiscreteState(ts, self.BRAKES_ON)
        self.a_error = DiscreteState(ts, 0.0)
        self.detwist_pos_cmd = DiscreteState(ts, 0.0)
        self.detwist_pos_cmd_from_wing = DiscreteState(ts, 0.0)

    @property
    def detwist_max_vel(self):
        return float(self.p["detwist_max_vel"])

    def set_tether_torque(self, t, gsg_pos_p, tether_force_p):
        self.tether_torque.update(t, float(cross3(gsg_pos_p, tether_force_p)[2]))

    def discrete_step(self, t):
        # B4: no check of enabled_ (371-380).
        self._switching_logic(t)
        self._control_law(t)
        self.detwist_pos_cmd.update(t, self.detwist_pos_cmd_from_wing.val)

    def _switching_logic(self, t):
        p, cls = self.p, type(self)
        self.a_error.update(t, wrap(self.wing_azi.val - self.azi_pos.val - PI, -PI, PI))
        torque, a_error, state = self.tether_torque.val, self.a_error.val, self.n_state_machine.val
        if state == self.BASE:
            self.azi_velocity_dir.update(t, 0.0)
            self.n_hpu_mode_demand.update(t, self.BRAKES_ON)
            if torque > p["m_max_azi_ht"] and a_error > p["a_control_threshold_azi_ht"]:
                cls._n_state_machine_val = self.WAIT_POS_OVERLOAD
            if torque < -p["m_max_azi_ht"] and a_error < -p["a_control_threshold_azi_ht"]:
                cls._n_state_machine_val = self.WAIT_NEG_OVERLOAD
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.WAIT_POS_OVERLOAD:
            self.azi_velocity_dir.update(t, 0.0)
            self.n_hpu_mode_demand.update(t, self.BRAKES_ON)
            if torque > p["m_control_threshold_azi_ht"]:
                cls._n_state_machine_val = self.WAIT_HPU_POS
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.WAIT_HPU_POS:
            self.azi_velocity_dir.update(t, 0.0)
            self.n_hpu_mode_demand.update(t, self.BRAKES_REGULATED)
            if self.n_hpu_mode.val == self.BRAKES_REGULATED:
                cls._n_state_machine_val = self.AZI_POS_ROTATION
            if torque < p["m_control_threshold_azi_ht"]:
                cls._n_state_machine_val = self.WAIT_POS_OVERLOAD
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.AZI_POS_ROTATION:
            self.azi_velocity_dir.update(t, p["n_demand_azi_ht"])
            if abs(a_error) < p["a_control_tolerance_azi_ht"] or a_error < 0.0:
                cls._n_state_machine_val = self.ACTIVE_BRAKING
                self.active_braking_first_entry = True
            elif a_error < 0.0:                   # unreachable, as coded (438-440)
                cls._n_state_machine_val = self.AZI_NEG_ROTATION
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.ACTIVE_BRAKING:
            self.azi_velocity_dir.update(t, 0.0)
            if self.active_braking_first_entry:
                self.active_braking_first_entry = False
                self.active_braking_t0 = t
            if abs(self.azi_vel.val) < p["n_control_tolerance_azi_ht"]:
                cls._n_state_machine_val = self.BASE
            if t > self.active_braking_t0 + p["t_threshold_wait_azi_ht"]:
                cls._n_state_machine_val = self.BASE
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.WAIT_NEG_OVERLOAD:
            self.azi_velocity_dir.update(t, 0.0)
            self.n_hpu_mode_demand.update(t, self.BRAKES_ON)
            if torque < -p["m_control_threshold_azi_ht"]:
                cls._n_state_machine_val = self.WAIT_HPU_NEG
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.WAIT_HPU_NEG:
            self.azi_velocity_dir.update(t, 0.0)
            self.n_hpu_mode_demand.update(t, self.BRAKES_REGULATED)
            if torque > -p["m_control_threshold_azi_ht"]:
                cls._n_state_machine_val = self.WAIT_NEG_OVERLOAD
            if self.n_hpu_mode.val == self.BRAKES_REGULATED:
                cls._n_state_machine_val = self.AZI_NEG_ROTATION
            self.n_state_machine.update(t, cls._n_state_machine_val)
        elif state == self.AZI_NEG_ROTATION:
            self.azi_velocity_dir.update(t, -p["n_demand_azi_ht"])
            if abs(a_error) < p["a_control_tolerance_azi_ht"] or a_error > 0.0:
                cls._n_state_machine_val = self.ACTIVE_BRAKING
                self.active_braking_first_entry = True
            elif a_error > 0.0:                   # unreachable, as coded (492-494)
                cls._n_state_machine_val = self.AZI_POS_ROTATION
            self.n_state_machine.update(t, cls._n_state_machine_val)

    def _brake_model(self, t):
        brake_torque_val = self.brake_torque_cmd.val
        total_torque_val = 0.0
        torque = self.tether_torque.val
        if torque*brake_torque_val > 0.0:
            brake_torque_val = 0.0
        elif abs(brake_torque_val) > abs(torque):
            brake_torque_val = -torque
        if self.n_state_machine.val == self.BASE and self.n_hpu_mode.val == self.BRAKES_REGULATED:
            if abs(self.azi_vel.val) > self.angular_rate_tol:
                brake_torque_val = self.total_torque.val - torque
            else:
                brake_torque_val = -torque
        else:
            total_torque_val = torque + brake_torque_val
        self.brake_torque.update(t, brake_torque_val)
        self.total_torque.update(t, total_torque_val)

    def _hpu_model(self, t):
        if self.n_hpu_mode_last.val != self.n_hpu_mode_demand.val:
            self.hpu_cmd_change_t0 = t
        if t > self.hpu_cmd_change_t0 + self.hpu_delay:
            self.n_hpu_mode.update(t, self.n_hpu_mode_demand.val)
        self.n_hpu_mode_last.update(t, self.n_hpu_mode_demand.val)

    def _control_law(self, t):
        p = self.p
        if self.n_hpu_mode_demand.val == self.BRAKES_REGULATED:
            direction = self.azi_velocity_dir.val
            if abs(direction) < p["test_threshold"]:
                azi_omega_cmd, k_omega = 0.0, p["k_stop"]
            elif abs(direction - p["n_demand_azi_ht"]) < p["test_threshold"]:
                azi_omega_cmd, k_omega = p["omega_nom"], p["k_spin"]
            elif abs(direction + p["n_demand_azi_ht"]) < p["test_threshold"]:
                azi_omega_cmd, k_omega = -p["omega_nom"], p["k_spin"]
            else:
                azi_omega_cmd, k_omega = 0.0, p["k_stop"]
            total_torque_cmd = k_omega*(azi_omega_cmd - self.azi_vel.val)
            brake_torque_cmd_val = total_torque_cmd - self.tether_torque.val
        else:
            brake_torque_cmd_val = -self.tether_torque.val
        self.brake_torque_cmd.update(t, brake_torque_cmd_val)
        self._brake_model(t)
        self._hpu_model(t)
        self.azi_vel_cmd.update(t, self.total_torque.val/p["Iz_gndstation"]*self.ht_ts)     # B5
        self.drum_vel_cmd.update(t, 0.0)


# ----- the ground station ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gs02Frames:
    platform: Frame
    wd0: Frame
    wd: Frame
    gsg0: Frame
    panel: Frame


class GroundStationV2:
    """GroundStationV2 and GroundStationV2Base (594-1077) with an onshore, fixed vessel frame.

    Option (OF-0011 revision 6 addenda 60, 62; F2-Ma): ``vessel_frame``, a ``Frame`` (root NED) for the platform's
    parent - Makani's ``buoy.vessel_frame()`` (buoy.cc:204-205: origin Xg, velocity Vg, DCM g2v, angular velocity in g)
    on a floating base. ``None`` (the default) is the onshore, fixed vessel frame equal to NED, and every result is as
    before. ``set_vessel_frame`` changes it between right-hand-side evaluations, as the buoy's state changes.
    """

    # Continuous state vector: platform azimuth, its rate, drum angle, detwist angle, drum rate, drum acceleration.
    PSI, PSI_DOT, THETA, PHI, OMEGA, ALPHA = range(6)

    def __init__(self, params, vessel_frame=None):
        self.params = params
        self.vessel_frame = vessel_frame
        gs02, sim = params.gs02, params.sim
        self.racetrack = RacetrackTetherIntegrator(params)
        self.reel_controller = ReelController(params)
        self.transform_controller = TransformController(params)
        self.high_tension_controller = HighTensionController(params)
        self.sub_models = (self.reel_controller, self.transform_controller, self.high_tension_controller)
        self.x = np.zeros(6)
        self.x[self.PHI] = float(params.mclaren["detwist_setpoint"])
        # Inputs from the controller command (SetFromAvionicsPackets, 669-682); a default ControllerCommandMessage
        # is all zero (gs_mode_request 0 = Manual leaves Reel unchanged).
        self.azi_cmd_target, self.azi_cmd_dead_zone = 0.0, 0.0
        self.drum_linear_vel_cmd_from_wing, self.detwist_pos_cmd_from_wing = 0.0, 0.0
        self.mode_cmd, self.unpause_transform = MODE_MANUAL, False
        self.tether_force_g = np.zeros(3)
        self.tether_force_p = np.zeros(3)
        self.levelwind_engaged = None          # a derived value; must be set before the first discrete step in Reel
        # Discrete states, ts = 0 (the GS02 model is an Actuator with ts 0, 598).
        self.mode = DiscreteState(0.0, MODE_REEL)
        self.transform_stage = DiscreteState(0.0, 0)
        self.mode_z1 = DiscreteState(0.0, MODE_REEL)
        self.ht2reel = DiscreteState(0.0, False)
        self.prox_sensor_active = DiscreteState(0.0, False)
        self._angles = gs02["drum_angles"]
        self._k_wide = sqrt_pos(sim["dx_dtheta_wide_wrap"]**2 + gs02["drum_radius"]**2)
        self._k_main = sqrt_pos(sim["dx_dtheta_main_wrap"]**2 + gs02["drum_radius"]**2)

    # ----- state access --------------------------------------------------------------------------------------------
    def init(self, azimuth, drum_angle):
        """Init (633-640): the platform azimuth and the drum angle; the other states keep their values."""
        self.x[self.PSI], self.x[self.THETA] = float(azimuth), float(drum_angle)

    @property
    def platform_azi(self):
        return float(self.x[self.PSI])

    @property
    def platform_azi_vel(self):
        return float(self.x[self.PSI_DOT])

    @property
    def drum_angle(self):
        return float(self.x[self.THETA])

    @property
    def drum_omega(self):
        return float(self.x[self.OMEGA])

    @property
    def detwist_angle(self):
        return float(self.x[self.PHI])

    def set_vessel_frame(self, frame):
        """The platform's parent frame (a ``Frame``, root NED), or ``None`` for the fixed onshore one (= NED)."""
        self.vessel_frame = frame

    def _vessel(self):
        return Frame.root() if self.vessel_frame is None else self.vessel_frame

    # ----- inputs ---------------------------------------------------------------------------------------------------
    def set_from_command(self, *, gs_azi_target, gs_azi_dead_zone, winch_velocity, detwist_position, gs_mode_request,
                         gs_unpause_transform):
        """SetFromAvionicsPackets (669-682), from the fields of the controller's ControllerCommandMessage."""
        self.azi_cmd_target = float(gs_azi_target)
        self.azi_cmd_dead_zone = float(gs_azi_dead_zone)
        self.drum_linear_vel_cmd_from_wing = float(winch_velocity)
        self.detwist_pos_cmd_from_wing = float(detwist_position)
        self.mode_cmd = int(gs_mode_request)
        self.unpause_transform = bool(gs_unpause_transform)

    def set_tether_force(self, tether_start_force_g, x=None):
        """SetTetherForce (GSV2.h:262-268): the tether's start force, NED, and its platform-frame copy."""
        self.tether_force_g = np.asarray(tether_start_force_g, dtype=float).copy()
        self.tether_force_p = Frame.root().rotate_to(self.frames(x).platform, self.tether_force_g)

    def set_levelwind_engaged(self, tether_elevation):
        """SetLevelwindEngaged (642-667)."""
        mode = self.mode.val
        if mode in (MODE_REEL, MODE_MANUAL):
            in_use = True
        elif mode == MODE_TRANSFORM:
            in_use = (self.transform_stage.val & 0xFF) == 4
        elif mode == MODE_HIGH_TENSION:
            in_use = False
        else:
            raise RuntimeError(f"bad GS02 mode {mode}")
        self.levelwind_engaged = bool(in_use and tether_elevation >= self.params.sim["min_levelwind_angle_for_tether_engagement"])

    # ----- frames and geometry ----------------------------------------------------------------------------------------
    def frames(self, x=None):
        """UpdateDerivedStates (684-720) for the continuous state ``x`` (default: the current one)."""
        x = self.x if x is None else x
        gs02, sim = self.params.gs02, self.params.sim
        root = self._vessel()                     # the vessel frame: NED (onshore, fixed) unless the option is set
        platform = root.child(np.zeros(3), angle_to_dcm(float(x[self.PSI]), 0.0, 0.0, "zyx"), None,
                              np.array([0.0, 0.0, float(x[self.PSI_DOT])]))
        wd0 = platform.child(gs02["drum_origin_p"], np.eye(3))
        wd = wd0.child(np.zeros(3), angle_to_dcm(0.0, 0.0, float(x[self.THETA]), "zyx"), np.zeros(3),
                       np.array([float(x[self.OMEGA]), 0.0, 0.0]))
        gsg0 = wd.child(gs02["gsg_pos_drum"], calc_dcm_wd_to_gsg0(gs02["detwist_elevation"], float(x[self.PHI])))
        panel = platform.child(sim["panel"]["origin_pos_p"], np.array(sim["panel"]["dcm_p2panel"]["d"], dtype=float))
        return Gs02Frames(platform, wd0, wd, gsg0, panel)

    def free_length(self, drum_angle=None):
        """CalcTetherFreeLengthInternal (770-799)."""
        theta = self.drum_angle if drum_angle is None else float(drum_angle)
        angles = self._angles
        length = self.params.tether_length
        if theta < angles["racetrack_high"]:
            length -= self.racetrack.wrapped_length(theta)
        wide_wrap_high = angles["racetrack_low"]
        if theta < wide_wrap_high:
            length -= (wide_wrap_high - max(angles["wide_wrap_low"], theta))*self._k_wide
        main_wrap_high = angles["wide_wrap_low"]
        if theta < main_wrap_high:
            length -= (main_wrap_high - theta)*self._k_main
        if not length >= 0.0:
            raise ValueError(f"negative free length {length} at drum angle {theta}")     # DCHECK (797)
        return length

    def min_drum_angle(self):
        """MinDrumAngle (801-810)."""
        angles = self._angles
        main_wrap_length = self.free_length(angles["wide_wrap_low"])
        return angles["wide_wrap_low"] - main_wrap_length/self._k_main

    def anchor_pos_vel_ned(self, x=None):
        """CalcTetherAnchorPosVelNed (812-877)."""
        x = self.x if x is None else x
        frames = self.frames(x)
        gs02, sim, angles = self.params.gs02, self.params.sim, self._angles
        theta, omega = float(x[self.THETA]), float(x[self.OMEGA])
        gsg = np.asarray(gs02["gsg_pos_drum"], dtype=float)
        if theta > angles["racetrack_high"]:
            pos = frames.wd.to_root("position", gsg)
            at_gsg = frames.wd.child(gsg)
            return pos, at_gsg.to_root("velocity", np.zeros(3))
        inner_radius = math.hypot(gsg[1], gsg[2])
        pos_wd0 = np.array([gsg[0], 0.0, -inner_radius])
        vel_wd0 = np.zeros(3)
        if theta < angles["racetrack_high"]:
            span = angles["racetrack_high"] - angles["racetrack_low"]
            dx_dtheta = (gsg[0] - sim["wrap_start_posx_drum"])/span
            dz_dtheta = (-inner_radius + gs02["drum_radius"])/span
            dtheta = max(theta, angles["racetrack_low"]) - angles["racetrack_high"]
            pos_wd0[0] += dx_dtheta*dtheta
            pos_wd0[2] += dz_dtheta*dtheta
            vel_wd0[0] = dx_dtheta*omega
            vel_wd0[2] = dz_dtheta*omega
        wide_wrap_high = angles["racetrack_low"]
        if theta < wide_wrap_high:
            dtheta = max(theta, angles["wide_wrap_low"]) - wide_wrap_high
            pos_wd0[0] += sim["dx_dtheta_wide_wrap"]*dtheta
            vel_wd0[0] = sim["dx_dtheta_wide_wrap"]*omega
            vel_wd0[2] = 0.0
        main_wrap_high = angles["wide_wrap_low"]
        if theta < main_wrap_high:
            pos_wd0[0] += sim["dx_dtheta_main_wrap"]*(theta - main_wrap_high)
            vel_wd0[0] = sim["dx_dtheta_main_wrap"]          # B2: as coded (867), no factor drum rate
            vel_wd0[2] = 0.0
        pos = frames.wd0.to_root("position", pos_wd0)
        anchor = frames.wd0.child(pos_wd0)
        return pos, anchor.to_root("velocity", vel_wd0)

    def gsg_pos_p(self, x=None):
        """gsg_pos_p (GSV2.h:270-275): the GSG in the platform frame."""
        frames = self.frames(x)
        return frames.wd.transform_to(frames.platform, "position", self.params.gs02["gsg_pos_drum"])

    def tether_dir_to_angles(self, tether_dir_v, x=None):
        """TetherDirToAngles (879-916): (elevation, azimuth in the vessel frame, GSG yoke, GSG termination)."""
        tether_dir_v = np.asarray(tether_dir_v, dtype=float)
        ele = vec_v_to_elevation(tether_dir_v)
        azi = vec_v_to_azimuth(tether_dir_v, self.params.azi_ref_offset)
        gsg0 = self.frames(x).gsg0
        local = gsg0.child(np.zeros(3), angle_to_dcm(0.0, 0.5*PI, 0.0, "zyx"))
        tether_dir_local = self._vessel().rotate_to(local, tether_dir_v)       # parent_frame_.RotateTo (908)
        yoke = math.atan2(tether_dir_local[1], tether_dir_local[0])                       # CartToSph azimuth
        termination = -math.atan2(tether_dir_local[2], math.hypot(tether_dir_local[0], tether_dir_local[1]))
        return ele, azi, yoke, termination

    # ----- dynamics -----------------------------------------------------------------------------------------------------
    def mode_controller_commands(self, x=None):
        """UpdateModeControllerCommands (722-757): (azi_vel_cmd, drum_vel_cmd, detwist_vel)."""
        x = self.x if x is None else x
        sim, mode = self.params.sim, self.mode.val
        theta, phi = float(x[self.THETA]), float(x[self.PHI])
        if mode == MODE_REEL:
            return self.reel_controller.azi_vel_cmd.val, self.reel_controller.drum_vel_cmd.val, 0.0
        if mode == MODE_TRANSFORM:
            tc = self.transform_controller
            detwist = saturate(sim["detwist_angle_kp"]*wrap(saturate(-theta, 0.0, self.params.mclaren["detwist_setpoint"]) - phi,
                                                            -PI, PI), -tc.detwist_max_vel, tc.detwist_max_vel)
            return tc.azi_vel_cmd.val, tc.drum_vel_cmd.val, detwist
        if mode == MODE_HIGH_TENSION:
            ht = self.high_tension_controller
            detwist = saturate(sim["detwist_angle_kp"]*wrap(ht.detwist_pos_cmd.val - phi, -PI, PI),
                               -ht.detwist_max_vel, ht.detwist_max_vel)
            return ht.azi_vel_cmd.val, ht.drum_vel_cmd.val, detwist
        raise RuntimeError(f"bad GS02 mode {mode}")

    def derivatives(self, x=None):
        """CalcDerivHelper (980-993)."""
        x = self.x if x is None else x
        sim = self.params.sim
        azi_vel_cmd, drum_vel_cmd, detwist_vel = self.mode_controller_commands(x)
        omega_n = sim["winch_drive_natural_freq"]          # B9: labelled Hz, used as rad/s
        zeta = sim["winch_drive_damping_ratio"]
        d = np.zeros(6)
        d[self.PSI] = x[self.PSI_DOT]
        d[self.PSI_DOT] = sim["azi_accel_kp"]*(azi_vel_cmd - x[self.PSI_DOT])
        d[self.ALPHA] = omega_n*omega_n*(drum_vel_cmd - x[self.OMEGA]) - 2.0*zeta*omega_n*x[self.ALPHA]
        d[self.OMEGA] = x[self.ALPHA]
        d[self.THETA] = x[self.OMEGA]
        d[self.PHI] = detwist_vel
        return d

    def discrete_update(self, t):
        """Model::DiscreteUpdate of the GS02 model (ts 0: every call) and then of its three controllers."""
        self.discrete_step(t)
        for model in self.sub_models:
            model.discrete_update(t)

    def discrete_step(self, t):
        """DiscreteStepHelper (995-1077)."""
        mode = self.mode.val
        if mode == MODE_REEL:
            if self.mode_cmd == MODE_HIGH_TENSION:
                self.mode.update(t, MODE_TRANSFORM)
                self.ht2reel.update(t, False)
        elif mode == MODE_HIGH_TENSION:
            if self.mode_cmd == MODE_REEL:
                self.mode.update(t, MODE_TRANSFORM)
                self.ht2reel.update(t, True)
        elif mode == MODE_TRANSFORM:
            if self.transform_controller.transform_complete.val:
                self.mode.update(t, MODE_REEL if self.ht2reel.val else MODE_HIGH_TENSION)
        else:
            raise RuntimeError(f"bad GS02 mode {mode}")
        if not 0.0 <= self.azi_cmd_target <= 2.0*PI:
            raise ValueError(f"gs_azi_target {self.azi_cmd_target} outside [0, 2 pi]")     # DCHECK (1022)
        common = (self.platform_azi, self.platform_azi_vel, self.drum_angle, self.drum_omega, self.azi_cmd_target)
        for controller in self.sub_models:
            controller.set_common_inputs(t, *common)
        mode = self.mode.val
        reel, transform, high_tension = self.sub_models
        if mode == MODE_REEL:
            reel.enable(t)
            transform.disable(t)
            high_tension.disable(t)
            reel.azi_cmd_dead_zone.update(t, self.azi_cmd_dead_zone)
            reel.drum_linear_vel_cmd_from_wing.update(t, self.drum_linear_vel_cmd_from_wing)
            if self.levelwind_engaged is None:
                raise RuntimeError("levelwind_engaged read before it was set (set_levelwind_engaged)")
            reel.levelwind_engaged.update(t, self.levelwind_engaged)
            self.transform_stage.update(t, 0)
        elif mode == MODE_TRANSFORM:
            transform.enable(t)
            reel.disable(t)
            high_tension.disable(t)
            self.transform_stage.update(t, transform.stage)
            transform.unpause_transform.update(t, self.unpause_transform)
            if mode != self.mode_z1.val:
                transform.start_transform(t, self.ht2reel.val)
        elif mode == MODE_HIGH_TENSION:
            high_tension.set_tether_torque(t, self.gsg_pos_p(), self.tether_force_p)
            high_tension.enable(t)
            transform.disable(t)
            reel.disable(t)
            high_tension.detwist_pos_cmd_from_wing.update(t, wrap(self.detwist_pos_cmd_from_wing, -PI, PI))
            self.transform_stage.update(t, 0)
        else:
            raise RuntimeError(f"bad GS02 mode {mode}")
        self.mode_z1.update(t, mode)
        self.prox_sensor_active.update(t, self.free_length() < self.params.sim["prox_sensor_tether_free_length"])


# ----- sim/models/sensors/ground_station_v2_sensors.cc ------------------------------------------------------------------
def tether_engaged(mode, mode_cmd, transform_stage):
    """The sensors' ``tether_engaged`` (ground_station_v2_sensors.cc:67-84)."""
    if mode in (MODE_REEL, MODE_MANUAL):
        return True
    if mode == MODE_HIGH_TENSION:
        return False
    if mode_cmd == MODE_REEL:
        return transform_stage == 4
    if mode_cmd == MODE_HIGH_TENSION:
        return transform_stage in (0, 4)
    raise RuntimeError("Invalid GS mode command.")          # B7: CHECK(false) in the source


def sensor_readings(gs, nominal_levelwind_ele, nominal_gsg_yoke, nominal_gsg_termination):
    """What the GS02 sensors report (ground_station_v2_sensors.cc:47-130), without the single-precision cast.

    The mode, mode command and stage are those the sensor model was given (the ground station's values at the time
    of the call); the nominal angles are ``tether_dir_to_angles`` of the tether's start direction. Returns the
    readings in the units of ``flight.makani_full.gs02_sensors``'s arguments.
    """
    engaged = tether_engaged(gs.mode.val, gs.mode_cmd, gs.transform_stage.val & 0xFF)
    angle = lambda value: wrap(value, -PI, PI)     # noqa: E731  (WrapAngle)
    return dict(
        tether_engaged=engaged,
        gs_mode=gs.mode.val, transform_stage=gs.transform_stage.val & 0xFF,
        drum_angle_rad=gs.drum_angle,                                     # multi-turn
        platform_azimuth_rad=angle(gs.platform_azi),
        detwist_angle_rad=angle(wrap(gs.detwist_angle, -PI, PI)),        # wrapped on set and on output
        proximity=bool(gs.prox_sensor_active.val),
        levelwind_elevation_rad=angle(nominal_levelwind_ele if engaged else 0.0),
        gsg_yoke_rad=angle(0.0 if engaged else nominal_gsg_yoke),
        gsg_termination_rad=angle(0.0 if engaged else nominal_gsg_termination))
