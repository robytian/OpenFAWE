"""Makani's stacked power system with the rotors' equation of motion (borrowed part, transcribed; opt-in, used by nothing
by default).

Source: Makani snapshot 1d66923 (Apache License 2.0, Makani Technologies LLC):

- ``sim/models/stacked_power_sys.cc`` (``UpdateDerivedStates`` 140-268, ``CalcDerivHelper`` 290-381, ``IsBlockFaulted``,
  ``GetNumActiveBlocks``, ``CalcWingVoltage``, ``CalcMeanVoltageCorrection``, ``CalcTetherCurrent`` 383-426,
  ``CalcMotorTorqueCommand`` 431-532, ``CalcGroundVoltageCommand`` 534-550, ``CalcMotorSpeedCommand`` 556-575); the
  blocks as ``avionics/common/motor_util.c:22-30`` (bottom motor i, top motor i + 4);
- ``sim/models/power_sys.cc`` (``DiscreteStepHelper`` 100-115: the rate limit of the upper and lower speed commands,
  ``RateLimit`` = ``common/c_math/filter.c:103-110``; the speed-error integrators); ``Model::DiscreteUpdate``
  (``sim/models/model.cc:134-143``, ``ts_`` 0: the step's ``ts`` is ``t - t_z1``, the first one after ``t_z1 = -DBL_MAX``);
- the rotor (``sim/models/actuators/rotor.{h,cc}``): ``SetFromAvionicsPackets`` (rotor.cc:190-219), ``AngularUpperSpeedCmd``,
  ``AngularLowerSpeedCmd``, ``TorqueCmd`` (rotor.h:83-105), ``AngularSpeed`` = omega * dir, ``SetMotorTorque`` (value * dir,
  rotor.cc:367-369), ``CalcAeroTorque`` (rotor.cc:262-268), ``RotorAcc`` (388-390); the aerodynamic power from the 2-D
  rotor table at (AngularSpeed, axial freestream, density) (rotor.cc:339-345; no 3-D table);
- the controller side: ``control/avionics/avionics_interface.c`` ``ConvertRotors`` (776-806: direction, ``InvertCal``,
  conversion to float32, ordering) and the run condition of ``motor_command`` (828-832); ``ApplyCal``/``InvertCal``
  (``common/c_math/util.c:512-523``).

Parameters: ``cases/powertrain/makani_m600_power_sys.json`` and ``cases/powertrain/makani_m600_motors.json``. Faults are
not modelled (every motor connected) and the tether is never released, but both branches are transcribed.

Power-energy contract (plan 3.9), revision 1 with addenda 1-4 (decision 0047): what is ours is the integration - the
42 states of the power system and the 8 signed rotor speeds over one 0.01 s period by classical RK4 with a fixed
sub-step (0.25 ms), the discrete inputs and the rotors' axial freestream held over the period; a period in which any
stage takes the excess-tether-current branch or the upper saturation of the average block voltage is integrated again
with sub-steps 64 times smaller. Auxiliary integrals (energy account, means) are integrated with the same RK4 stages.
The right-hand side is piecewise smooth (controller states, clamps, the table's bilinear cells; the speed controller's
anti-windup at a torque limit makes a sliding mode), which a fixed step resolves only to first or second order there
(addenda 3-4).

Arithmetic as the source: scalar, in the source's order, C's fmin/fmax as glibc behaves (``makani_motors.fmin``), the
command message's values rounded to float32.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from .makani_motors import (LIMIT_PHASE_CURRENT, LIMIT_POWER, MotorParams, calc_motor_power, calc_torque_limits, fmax,
                            fmin, saturate)

NUM_MOTORS, NUM_BLOCKS = 8, 4
DBL_MAX = 1.7976931348623157e308
# SimMotorLimit (sim/sim_types.h:156-161) and the speed controller's states (stacked_power_sys.h:124-128)
LIMIT_NONE, LIMIT_GROUND_POWER, LIMIT_PHASE_CURRENT_ID, LIMIT_POWER_ID = 0, 1, 2, 3
LIMIT_IDS = {LIMIT_PHASE_CURRENT: LIMIT_PHASE_CURRENT_ID, LIMIT_POWER: LIMIT_POWER_ID}
SPEED_TORQUE, SPEED_UPPER, SPEED_LOWER = 0, 1, 2

# state vector layout (42 power-system states, then the 8 signed rotor speeds omega of Rotor::omega_)
FTC, VG = 0, 1
V0, SC0, VCX0, SFU0, SFL0, IERR0, OMEGA0 = 2, 6, 10, 18, 26, 34, 42
STATE_SIZE = 50
SUB_STEP_S = 0.25e-3
STIFF_DIVISOR = 64


def float32(x):
    """C's (float) conversion of a double (round to nearest even; x86-64 SSE2 gives +-inf where the rounded value
    overflows, which Python's ``struct`` reports as ``OverflowError`` - addendum 3 item 9)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


@dataclass(frozen=True)
class Cal:
    scale: float
    bias: float
    bias_count: int

    def apply(self, raw):
        """``ApplyCal`` (util.c:512-514)."""
        return self.scale*(raw + float(self.bias_count)) + self.bias

    def invert(self, value):
        """``InvertCal`` (util.c:521-523)."""
        return (value - self.bias)/self.scale - float(self.bias_count)


@dataclass(frozen=True)
class PowerSysParams:
    use_ground_voltage_compensation: bool
    R_tether: float
    R_source: float
    v_source_0: float
    C_block: float


@dataclass(frozen=True)
class PowerSysSimParams:
    current_filter_cutoff_freq: float
    ground_voltage_pole: float
    min_ground_voltage_compensation: float
    max_ground_voltage_compensation: float
    kp_rotor_vel_err: float
    ki_rotor_vel_err: float
    rotor_vel_err_pole: float
    rotor_vel_err_torque_pole: float
    kp_voltage_err: float
    voltage_control_pole: float
    voltage_control_zero: float
    fc_stacking_speed_correction: float
    kp_stacking_speed_correction: float
    cap_drain_conductance: float
    omega_cmd_rate_limit: float
    speed_cmd_pole: float
    min_tether_current: float
    kp_excess_tether_current: float
    voltage_average_upper_sat: float


@dataclass(frozen=True)
class PowerSystemConfig:
    power_sys: PowerSysParams
    sim: PowerSysSimParams
    motor: MotorParams
    omega_cal: Cal
    torque_cal: Cal
    dirs: tuple                  # the rotors' directions (+1 / -1), Makani's RotorParams.dir
    inertia_kg_m2: tuple         # the rotors' moments of inertia (Makani's RotorParams.I)
    compiled: bool = False       # OF-0014 revision 2: integrate on the compiled path (makani_power_jit; opt-in)

    @classmethod
    def from_files(cls, power_path, motor_path, dirs, inertia_kg_m2):
        data = json.loads(Path(power_path).read_text(encoding="utf-8"))
        if data.get("schema") != "openfawe.makani_power_sys_params.v1":
            raise ValueError(f"{power_path}: unsupported schema {data.get('schema')!r}")
        ps = data["power_sys"]
        power_sys = PowerSysParams(bool(ps["use_ground_voltage_compensation"]), float(ps["R_tether"]), float(ps["R_source"]),
                                   float(ps["v_source_0"]), float(ps["C_block"]))
        sim = PowerSysSimParams(**{name: float(data["power_sys_sim"][name]) for name in PowerSysSimParams.__dataclass_fields__})
        cals = [Cal(float(data[k]["scale"]), float(data[k]["bias"]), int(data[k]["bias_count"])) for k in ("omega_cal", "torque_cal")]
        dirs = tuple(int(d) for d in dirs)
        if len(dirs) != NUM_MOTORS or any(d not in (1, -1) for d in dirs):
            raise ValueError("eight rotor directions of +1 or -1 are required")
        inertia = tuple(float(i) for i in inertia_kg_m2)
        if len(inertia) != NUM_MOTORS:
            raise ValueError("eight rotor inertias are required")
        return cls(power_sys, sim, MotorParams.from_json(motor_path), cals[0], cals[1], dirs, inertia)


# ------------------------------------------------------------------------------------------------------------ commands
def convert_rotors(upper, lower, torque, config, run):
    """The controller side, ``ConvertRotors`` (avionics_interface.c:776-806) and the run bit (828-832): the command
    message's motor speed limits and torques (float32 values) and whether the motors run."""
    msg_upper, msg_lower, msg_torque = [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS
    for i in range(NUM_MOTORS):
        d = float(config.dirs[i])
        torque_command = d*float(torque[i])
        msg_torque[i] = float32(config.torque_cal.invert(torque_command))
        upper_vel = float32(config.omega_cal.invert(d*float(upper[i])))
        lower_vel = float32(config.omega_cal.invert(d*float(lower[i])))
        if upper_vel > lower_vel:
            msg_upper[i], msg_lower[i] = upper_vel, lower_vel
        else:
            msg_upper[i], msg_lower[i] = lower_vel, upper_vel
    return dict(upper=msg_upper, lower=msg_lower, torque=msg_torque, run=bool(run))


ZERO_MESSAGE = dict(upper=[0.0]*NUM_MOTORS, lower=[0.0]*NUM_MOTORS, torque=[0.0]*NUM_MOTORS, run=False)


class CommandPipeline:
    """Which command message acts when (contract revision 1 addendum 1 item 1, addendum 2 item 9; Makani's loop,
    kite_sim.cc:173-201): in period k the rate limiter reads the rotors' speed commands from the message of period k-2
    and the torque commands come from the message of period k-1; both start as the zero message
    (``ControllerCommandMessage()``, kite_sim.cc:160, 462). ``push`` the message made from period k's controller output
    at the end of period k."""

    def __init__(self):
        self.for_rate, self.for_torque = ZERO_MESSAGE, ZERO_MESSAGE

    def push(self, message):
        self.for_rate, self.for_torque = self.for_torque, message


def angular_commands(omega_upper_cmd, omega_lower_cmd, torque_cmd, direction):
    """``AngularUpperSpeedCmd``, ``AngularLowerSpeedCmd``, ``TorqueCmd`` (rotor.h:83-105) of one rotor."""
    d = float(direction)
    if d > 0:
        upper, lower = omega_upper_cmd*d, omega_lower_cmd*d
    else:
        upper, lower = omega_lower_cmd*d, omega_upper_cmd*d
    return upper, lower, torque_cmd*d


def rotor_commands(message, config):
    """The simulator side: ``SetFromAvionicsPackets`` (rotor.cc:190-219), then ``AngularUpperSpeedCmd``,
    ``AngularLowerSpeedCmd`` and ``TorqueCmd`` (rotor.h:83-105) for each rotor: (upper, lower, torque) lists."""
    upper, lower, torque = [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS
    for i in range(NUM_MOTORS):
        omega_upper_cmd__, omega_lower_cmd__, torque_cmd__ = 0.0, 0.0, 0.0
        if message["run"]:
            omega_upper_cmd__ = message["upper"][i]
            omega_lower_cmd__ = message["lower"][i]
            torque_cmd__ = message["torque"][i]
        omega_upper_cmd__ = config.omega_cal.apply(omega_upper_cmd__)
        omega_lower_cmd__ = config.omega_cal.apply(omega_lower_cmd__)
        if omega_upper_cmd__ >= omega_lower_cmd__:
            omega_upper_cmd, omega_lower_cmd = omega_upper_cmd__, omega_lower_cmd__
        else:
            omega_upper_cmd, omega_lower_cmd = omega_lower_cmd__, omega_upper_cmd__
        torque_cmd = config.torque_cal.apply(torque_cmd__)
        upper[i], lower[i], torque[i] = angular_commands(omega_upper_cmd, omega_lower_cmd, torque_cmd, config.dirs[i])
    return upper, lower, torque


def rate_limit(u, low, high, ts, y_z1):
    """``RateLimit`` (filter.c:103-110): returns the new y_z1."""
    return y_z1 + saturate(u - y_z1, low*ts, high*ts)


class TablePower:
    """The rotors' aerodynamic power from their tables at a density, ``tables[i].power_W(s, v, density)`` - what
    ``turnkey`` passes as ``aero_power``; it carries the tables so that the compiled path (OF-0014 revision 2) can make the
    same look-up."""

    def __init__(self, tables, density):
        self.tables, self.density = list(tables), density

    def __call__(self, i, s, v):
        return self.tables[i].power_W(s, v, self.density)


def calc_aero_torque(direction, rotor_vel, aero_power):
    """``CalcAeroTorque`` (rotor.cc:262-268), ``rotor_vel`` the signed omega."""
    if float(direction)*rotor_vel < 10.0:
        return 0.0
    return aero_power/rotor_vel


# -------------------------------------------------------------------------------------------------------- the system
class StackedPowerSystem:
    """The transcribed model. ``x`` is the 50-state vector (layout above); the discrete states are the rate-limited
    upper and lower speed commands, ``t_z1``, the motor connections and the tether-release flag."""

    def __init__(self, config):
        self.config = config
        p = config.power_sys
        self.x = [0.0]*STATE_SIZE
        self.x[VG] = p.v_source_0                                  # constructor (stacked_power_sys.cc:39-42, 98-100)
        for i in range(NUM_BLOCKS):
            self.x[V0 + i] = p.v_source_0/NUM_BLOCKS
        self.rl_upper, self.rl_lower = [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS
        self.t_z1 = -DBL_MAX
        self.connections = [True]*NUM_MOTORS
        self.tether_released = False
        self.trace = None                                # a dict to count the branches taken (E2b), or None

    def _hit(self, name):
        if self.trace is not None:
            self.trace[name] = self.trace.get(name, 0) + 1

    # -- discrete step ---------------------------------------------------------------------------------------------
    def discrete_step(self, t, upper_cmds, lower_cmds):
        """``PowerSys::DiscreteStepHelper`` within ``Model::DiscreteUpdate`` (ts_ 0): the rate limit with ts = t - t_z1."""
        s = self.config.sim
        ts = t - self.t_z1
        for i in range(NUM_MOTORS):
            self.rl_upper[i] = rate_limit(upper_cmds[i], -s.omega_cmd_rate_limit, s.omega_cmd_rate_limit, ts, self.rl_upper[i])
            self.rl_lower[i] = rate_limit(lower_cmds[i], -s.omega_cmd_rate_limit, s.omega_cmd_rate_limit, ts, self.rl_lower[i])
        self.t_z1 = t

    # -- helpers as the source -------------------------------------------------------------------------------------
    def is_block_faulted(self, i):
        return not (self.connections[i] and self.connections[i + NUM_MOTORS//2])

    def num_active_blocks(self):
        n = 0
        for i in range(NUM_BLOCKS):
            if not self.is_block_faulted(i):
                n += 1
        return n

    def angular_speed(self, x, i):
        return x[OMEGA0 + i]*float(self.config.dirs[i])

    def ground_voltage_command(self, x):
        """``CalcGroundVoltageCommand`` (534-550)."""
        p, s = self.config.power_sys, self.config.sim
        if self.tether_released:
            self._hit("535-537 ground command, tether released")
            return 0.0
        compensation = -p.R_tether*x[FTC] if p.use_ground_voltage_compensation else 0.0
        if self.trace is not None:
            self._hit("540-542 ground compensation " + ("on" if p.use_ground_voltage_compensation else "off"))
            self._hit("544-547 compensation " + ("at the minimum" if compensation < s.min_ground_voltage_compensation else
                                                  "at the maximum" if compensation > s.max_ground_voltage_compensation else "inside"))
        compensation = saturate(compensation, s.min_ground_voltage_compensation, s.max_ground_voltage_compensation)
        return p.v_source_0 + compensation

    def motor_speed_command(self, x, i, tether_current, torque_cmds):
        """``CalcMotorSpeedCommand`` (556-575): (omega_upper_cmd, omega_lower_cmd, torque_cmd)."""
        s = self.config.sim
        current_excess = s.min_tether_current - tether_current
        upper = fmax(0.0, x[SFU0 + i] - s.kp_excess_tether_current*fmax(0.0, current_excess) + x[SC0 + i % NUM_BLOCKS])
        lower = fmax(0.0, x[SFL0 + i] - s.kp_excess_tether_current*fmax(0.0, current_excess) + x[SC0 + i % NUM_BLOCKS])
        if self.trace is not None:
            self._hit("563-570 excess tether current " + ("positive" if current_excess > 0.0 else "not positive"))
            raw_upper = x[SFU0 + i] - s.kp_excess_tether_current*fmax(0.0, current_excess) + x[SC0 + i % NUM_BLOCKS]
            raw_lower = x[SFL0 + i] - s.kp_excess_tether_current*fmax(0.0, current_excess) + x[SC0 + i % NUM_BLOCKS]
            self._hit("561-565 upper speed command " + ("floored at 0" if raw_upper < 0.0 else "not floored"))
            self._hit("567-571 lower speed command " + ("floored at 0" if raw_lower < 0.0 else "not floored"))
        return upper, lower, torque_cmds[i]

    def motor_torque_command(self, limits, motor_cmd, rotor_vel, int_rotor_vel_err, local_voltage_correction):
        """``CalcMotorTorqueCommand`` (431-532): (torque_cmd, constraint, local_voltage_correction, controller_state,
        int_deriv); ``int_deriv`` starts at 0.0 as the caller's variable does."""
        s = self.config.sim
        int_deriv = 0.0
        omega_upper_cmd, omega_lower_cmd, torque_request = motor_cmd
        omega_upper_error = omega_upper_cmd - rotor_vel
        omega_lower_error = omega_lower_cmd - rotor_vel
        if omega_lower_cmd > omega_upper_cmd:
            omega_lower_error = omega_upper_error
            self._hit("442-444 inverted speed limits")
        torque_upper_cmd = s.kp_rotor_vel_err*(omega_upper_error) + int_rotor_vel_err
        torque_lower_cmd = s.kp_rotor_vel_err*(omega_lower_error) + int_rotor_vel_err
        if torque_request >= torque_upper_cmd:
            torque_cmd, controller_state = torque_upper_cmd, SPEED_UPPER
            self._hit("457-459 speed control upper")
        elif torque_request <= torque_lower_cmd:
            torque_cmd, controller_state = torque_lower_cmd, SPEED_LOWER
            self._hit("460-462 speed control lower")
        else:
            torque_cmd, controller_state = torque_request, SPEED_TORQUE
            self._hit("463-473 torque mode")
            int_deriv = s.rotor_vel_err_torque_pole*(int_rotor_vel_err - saturate(torque_request, limits.lower_limit,
                                                                                    limits.upper_limit))
        if torque_cmd >= limits.upper_limit:
            torque_cmd = limits.upper_limit
            if controller_state != SPEED_TORQUE:
                int_deriv = 0.0
            self._hit("492-494 first limit: upper, " + ("torque mode" if controller_state == SPEED_TORQUE else "speed mode"))
        elif torque_cmd <= limits.lower_limit:
            torque_cmd = limits.lower_limit
            if controller_state != SPEED_TORQUE:
                int_deriv = 0.0
            self._hit("495-497 first limit: lower, " + ("torque mode" if controller_state == SPEED_TORQUE else "speed mode"))
        elif controller_state == SPEED_UPPER:
            int_deriv = (omega_upper_cmd - rotor_vel)*s.ki_rotor_vel_err + s.rotor_vel_err_pole*int_rotor_vel_err
            self._hit("498-501 integrator, upper speed loop")
        elif controller_state == SPEED_LOWER:
            int_deriv = (omega_lower_cmd - rotor_vel)*s.ki_rotor_vel_err + s.rotor_vel_err_pole*int_rotor_vel_err
            self._hit("502-505 integrator, lower speed loop")
        if self.trace is not None:
            raised = fmin(0.0, -torque_cmd)
            self._hit("512-513 voltage correction " + ("raised to fmin(0, -torque)" if raised > local_voltage_correction else "not raised"))
        local_voltage_correction = fmax(local_voltage_correction, fmin(0.0, -torque_cmd))
        if self.trace is not None:
            self._hit("514-515 voltage correction " + ("cut to upper - torque" if limits.upper_limit - torque_cmd < local_voltage_correction
                                                       else "not cut"))
        local_voltage_correction = fmin(local_voltage_correction, limits.upper_limit - torque_cmd)
        torque_cmd += local_voltage_correction
        if torque_cmd >= limits.upper_limit:
            torque_cmd, constraint = limits.upper_limit, LIMIT_IDS[limits.upper_constraint]
            self._hit("521-523 final: at the upper limit")
        elif torque_cmd <= limits.lower_limit:
            torque_cmd, constraint = limits.lower_limit, LIMIT_IDS[limits.lower_constraint]
            self._hit("524-526 final: at the lower limit")
        else:
            constraint = LIMIT_NONE
            self._hit("527-529 final: inside the limits")
        return torque_cmd, constraint, local_voltage_correction, controller_state, int_deriv

    # -- derived values and derivatives ----------------------------------------------------------------------------
    def derived(self, x, torque_cmds, *, parts=False):
        """``UpdateDerivedStates`` (140-268) at the state ``x`` with the rotors' ``TorqueCmd`` values. Returns a dict of
        the derived values (with ``parts`` also each motor's power terms)."""
        p, s, motor = self.config.power_sys, self.config.sim, self.config.motor
        limits = [calc_torque_limits(abs(x[V0 + i % NUM_BLOCKS]), self.angular_speed(x, i), motor) for i in range(NUM_MOTORS)]
        if self.trace is not None:                       # motors.cc:55-56, 63, 67, 85, 93: which constraint sets each limit
            for limit in limits:
                self._hit("motors.cc:55-93 upper torque limit: " + ("power" if limit.upper_constraint == LIMIT_POWER else "phase current"))
                self._hit("motors.cc:55-93 lower torque limit: " + ("power" if limit.lower_constraint == LIMIT_POWER else "phase current"))
        wing_voltage = 0.0
        for i in range(NUM_BLOCKS):
            if not self.is_block_faulted(i):
                wing_voltage += x[V0 + i]
        tether_current = 0.0 if self.tether_released else (wing_voltage - x[VG])/(p.R_tether + p.R_source)
        self._hit("420-425 tether current, " + ("released" if self.tether_released else "attached"))
        torques, constraints, states = [0.0]*NUM_MOTORS, [LIMIT_NONE]*NUM_MOTORS, [SPEED_TORQUE]*NUM_MOTORS
        corrections, int_derivs, currents = [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS, [0.0]*NUM_MOTORS
        block_powers, terms = [0.0]*NUM_BLOCKS, [None]*NUM_MOTORS
        for i in range(NUM_BLOCKS):
            r_bot, r_top = i, i + NUM_MOTORS//2
            if self.is_block_faulted(i):
                if self.trace is not None:
                    self._hit("383-386 block faulted by " + {(False, True): "the bottom motor", (True, False): "the top motor",
                                                             (False, False): "both motors"}[(self.connections[r_bot], self.connections[r_top])])
                for r in (r_bot, r_top):
                    limit = limits[r]
                    torque, constraint = 0.0, LIMIT_NONE
                    if torque > limit.upper_limit:
                        torque, constraint = limit.upper_limit, LIMIT_IDS[limit.upper_constraint]
                        self._hit("159-161/171-173 faulted: zero torque above the upper limit")
                    elif torque < limit.lower_limit:
                        torque, constraint = limit.lower_limit, LIMIT_IDS[limit.lower_constraint]
                        self._hit("162-164/174-176 faulted: zero torque below the lower limit")
                    else:
                        self._hit("faulted: zero torque inside the limits")
                    torques[r], constraints[r] = torque, constraint
                block_powers[i] = 0.0
                for r in (r_bot, r_top):
                    currents[r], corrections[r], states[r], int_derivs[r] = 0.0, 0.0, SPEED_TORQUE, 0.0
            else:
                voltage_err = x[V0 + i] - fmin(wing_voltage/self.num_active_blocks(), s.voltage_average_upper_sat)
                if self.trace is not None:
                    self._hit("194 average block voltage " + ("saturated" if wing_voltage/self.num_active_blocks() > s.voltage_average_upper_sat
                                                              else "not saturated"))
                powers = []
                for r in (r_bot, r_top):
                    correction = s.kp_voltage_err*s.voltage_control_pole/s.voltage_control_zero*(voltage_err + x[VCX0 + r])
                    cmd = self.motor_speed_command(x, r, tether_current, torque_cmds)
                    torque, constraint, correction, state, int_deriv = self.motor_torque_command(
                        limits[r], cmd, self.angular_speed(x, r), x[IERR0 + r], correction)
                    if tether_current < s.min_tether_current:
                        constraint = LIMIT_GROUND_POWER
                        self._hit("217-219/247-249 tether current below the minimum")
                    torques[r], states[r] = torque, state
                    if parts:
                        power, terms[r] = calc_motor_power(x[V0 + i], torques[r], self.angular_speed(x, r), motor, parts=True)
                    else:
                        power = calc_motor_power(x[V0 + i], torques[r], self.angular_speed(x, r), motor)
                    int_derivs[r] = int_deriv
                    constraints[r], corrections[r] = constraint, correction
                    powers.append(power)
                block_powers[i] = powers[0] + powers[1]
                if self.trace is not None:
                    self._hit("263-265 motor current " + ("over 1 V (block below 1 V)" if x[V0 + i] < 1.0 else "over the block voltage"))
                currents[r_bot] = powers[0]/fmax(1.0, x[V0 + i])
                currents[r_top] = powers[1]/fmax(1.0, x[V0 + i])
        out = dict(limits=limits, wing_voltage=wing_voltage, tether_current=tether_current, motor_torque_cmds=torques,
                   motor_constraints=constraints, speed_controller_state=states, voltage_correction=corrections,
                   speed_controller_int_deriv=int_derivs, motor_currents=currents, block_powers=block_powers)
        if parts:
            out["motor_terms"] = terms
        return out

    def mean_voltage_correction(self, x):
        """``CalcMeanVoltageCorrection`` (407-417)."""
        corr_sum, num_active = 0.0, 0.0
        for i in range(NUM_BLOCKS):
            if not self.is_block_faulted(i):
                corr_sum += x[SC0 + i]
                num_active += 1.0
        return corr_sum/num_active if num_active > 0.0 else 0.0

    def derivatives(self, x, d, aero_powers):
        """``CalcDerivHelper`` (290-381) and ``Rotor::CalcDerivHelper`` (``RotorAcc``) with the derived values ``d`` and
        the rotors' aerodynamic powers. Returns (dx, applied aerodynamic torques)."""
        p, s = self.config.power_sys, self.config.sim
        dx = [0.0]*STATE_SIZE
        dx[FTC] = 2.0*math.pi*s.current_filter_cutoff_freq*(d["tether_current"] - x[FTC])
        dx[VG] = -s.ground_voltage_pole*(self.ground_voltage_command(x) - x[VG])
        mean_wing_voltage_correction = self.mean_voltage_correction(x)
        for i in range(NUM_BLOCKS):
            if self.is_block_faulted(i):
                block_cap_current = -x[V0 + i]*s.cap_drain_conductance
                dx[SC0 + i] = 0.0
                self._hit("305-310 capacitor of a faulted block drains")
            else:
                if x[V0 + i] <= 0.0:
                    block_cap_current = -x[V0 + i]*s.cap_drain_conductance
                    self._hit("312-315 block voltage <= 0: the diode conducts")
                elif not self.tether_released:
                    block_cap_current = -d["tether_current"] + d["block_powers"][i]/fmax(x[V0 + i], 1.0)
                    self._hit("316-318 capacitor current" + (", block below 1 V" if x[V0 + i] < 1.0 else ""))
                else:
                    block_cap_current = d["block_powers"][i]/fmax(x[V0 + i], 1.0)
                    self._hit("319-322 capacitor current, tether released" + (", block below 1 V" if x[V0 + i] < 1.0 else ""))
                mean_pair_voltage_correction = (d["voltage_correction"][i] + d["voltage_correction"][i + NUM_MOTORS//2])*0.5
                dx[SC0 + i] = 2.0*math.pi*s.fc_stacking_speed_correction*(
                    s.kp_stacking_speed_correction*(mean_pair_voltage_correction - mean_wing_voltage_correction) - x[SC0 + i])
            dx[V0 + i] = block_cap_current/p.C_block
        for i in range(NUM_MOTORS):
            dx[SFU0 + i] = -s.speed_cmd_pole*(self.rl_upper[i] - x[SFU0 + i])
            dx[SFL0 + i] = -s.speed_cmd_pole*(self.rl_lower[i] - x[SFL0 + i])
            dx[IERR0 + i] = d["speed_controller_int_deriv"][i]
            temp_gain = s.kp_voltage_err*s.voltage_control_pole/s.voltage_control_zero
            voltage_err = x[V0 + i % NUM_BLOCKS] - fmin(d["wing_voltage"]/self.num_active_blocks(), s.voltage_average_upper_sat)
            if self.trace is not None:
                self._hit("362-364 anti-windup, average " + ("saturated" if d["wing_voltage"]/self.num_active_blocks() > s.voltage_average_upper_sat
                                                             else "not saturated"))
            output = temp_gain*(voltage_err + x[VCX0 + i])
            overage = output - d["voltage_correction"][i]
            voltage_err -= overage/temp_gain
            dx[VCX0 + i] = ((s.voltage_control_pole - s.voltage_control_zero)*voltage_err + s.voltage_control_pole*x[VCX0 + i])
        aero_torques = [0.0]*NUM_MOTORS
        for i in range(NUM_MOTORS):
            direction = float(self.config.dirs[i])
            motor_torque = d["motor_torque_cmds"][i]*direction                    # SetMotorTorque (rotor.cc:367-369)
            aero_torques[i] = calc_aero_torque(direction, x[OMEGA0 + i], aero_powers[i])
            if self.trace is not None:
                self._hit("rotor.cc:262-268 aero torque " + ("zero below 10 rad/s" if direction*x[OMEGA0 + i] < 10.0 else "power / omega"))
            dx[OMEGA0 + i] = (motor_torque + aero_torques[i])/self.config.inertia_kg_m2[i]   # RotorAcc (388-390)
        return dx, aero_torques

    def stiff_branch(self, d):
        """True when the excess-tether-current branch (CalcMotorSpeedCommand) or the upper saturation of the average block
        voltage is active (contract revision 1 addendum 1 item 4, addendum 2 item 7)."""
        low_current, saturated = self.stiff_branches(d)
        return bool(low_current or saturated)

    def stiff_branches(self, d):
        """The two branches of ``stiff_branch`` separately: (tether current below the minimum, average block voltage above
        the upper saturation) - which one made a period's integration stiff (addendum 1 item 4, review P39D-03)."""
        s = self.config.sim
        if self.num_active_blocks() == 0:
            return False, False
        return bool(d["tether_current"] < s.min_tether_current), bool(d["wing_voltage"]/self.num_active_blocks() > s.voltage_average_upper_sat)


# ------------------------------------------------------------------------------------------------------- integration
AUX_NAMES = ("ground_energy_J", "tether_loss_J", "storage_rate_integral_J")
MOTOR_AUX = ("electrical_J", "mechanical_J", "resistive_J", "speed_J", "hysteresis_J", "controller_J", "conduction_J",
             "aero_applied_J", "aero_torque_Nms", "peak_current_sq_A2s")


def stage(system, x, torque_cmds, aero_power_fn, tau=None):
    """One evaluation: derived values, the rotors' aerodynamic power at (AngularSpeed, held freestream), derivatives and
    the integrands of the auxiliary quantities. Returns (dx, aux_rates, info)."""
    p, motor = system.config.power_sys, system.config.motor
    d = system.derived(x, torque_cmds, parts=True)
    speeds = [system.angular_speed(x, i) for i in range(NUM_MOTORS)]
    aero_powers = [aero_power_fn(i, speeds[i]) if tau is None else aero_power_fn(i, speeds[i], tau) for i in range(NUM_MOTORS)]
    dx, aero_torques = system.derivatives(x, d, aero_powers)
    current = d["tether_current"]
    ground = x[VG]*current
    tether = current*current*(p.R_tether + p.R_source)
    storage = 0.0
    for i in range(NUM_BLOCKS):
        if not system.is_block_faulted(i):
            storage += p.C_block*x[V0 + i]*dx[V0 + i]
    for i in range(NUM_MOTORS):
        storage += system.config.inertia_kg_m2[i]*x[OMEGA0 + i]*dx[OMEGA0 + i]
    motor_rates = []
    flows = 0.0
    scale = abs(ground) + abs(tether)
    for i in range(NUM_MOTORS):
        terms = d["motor_terms"][i]
        aero_applied = aero_torques[i]*x[OMEGA0 + i]
        if terms is None:                                     # a faulted block: no motor power (not modelled)
            row = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, aero_applied, aero_torques[i], 0.0)
        else:
            total = terms["mechanical_W"] + terms["resistive_W"] + terms["speed_W"] + terms["hysteresis_W"] + terms["controller_W"]
            conduction = -1.0*terms["peak_phase_current_sq_A2"]*motor.rds_on             # loss_factor as coded
            row = (total, terms["mechanical_W"], terms["resistive_W"], terms["speed_W"], terms["hysteresis_W"], terms["controller_W"],
                   conduction, aero_applied, aero_torques[i], terms["peak_phase_current_sq_A2"])
            losses = terms["resistive_W"] + terms["speed_W"] + terms["hysteresis_W"] + terms["controller_W"]
            flows += losses
            scale += abs(terms["resistive_W"]) + abs(terms["speed_W"]) + abs(terms["hysteresis_W"]) + abs(terms["controller_W"])
        flows += aero_applied
        scale += abs(aero_applied)
        motor_rates.append(row)
    flows += -ground - tether
    residual = storage - flows                                # E3a: the instantaneous identity at this stage
    low_current, saturated = system.stiff_branches(d)
    info = dict(stiff=bool(low_current or saturated), stiff_low_current=low_current, stiff_saturated=saturated,
                residual=residual, scale=scale + abs(storage),
                low_voltage=any(x[V0 + i] < 1.0 for i in range(NUM_BLOCKS)), derived=d)
    return dx, (ground, tether, storage, motor_rates), info


def integrate_period(system, t, torque_cmds, aero_power_fn, *, sub_step_s=SUB_STEP_S, period_s=0.01, allow_rerun=True,
                     time_aware=False):
    """Integrate the 50 states over one period from ``system.x`` (the discrete step must have been taken). Returns a
    dict: the new state (also stored in ``system.x``), the auxiliary integrals, the period means, the stage checks.
    ``time_aware``: ``aero_power_fn(i, s, tau)`` with the stage's time since the period start (E2c's split report only)."""
    steps = int(round(period_s/sub_step_s))
    if abs(steps*sub_step_s - period_s) > 1e-15:
        raise ValueError("the sub-step must divide the period")
    start = list(system.x)
    result = _rk4(system, start, torque_cmds, aero_power_fn, sub_step_s, steps, time_aware)
    result["sub_step_s"], result["rerun"] = sub_step_s, False
    first = dict(stiff_stages=result["stiff_stages"], low_current_stages=result["low_current_stages"],
                 saturated_stages=result["saturated_stages"])
    if allow_rerun and result["stiff_stages"] > 0:
        small = _rk4(system, start, torque_cmds, aero_power_fn, sub_step_s/STIFF_DIVISOR, steps*STIFF_DIVISOR, time_aware)
        small["sub_step_s"], small["rerun"] = sub_step_s/STIFF_DIVISOR, True
        result = small
    # The first attempt's counts (review P39D-03): which branch made the period stiff, also when it was integrated again.
    result["first_attempt_stiff_stages"] = first["stiff_stages"]
    result["first_attempt_low_current_stages"] = first["low_current_stages"]
    result["first_attempt_saturated_stages"] = first["saturated_stages"]
    for value in result["x"]:
        if not math.isfinite(value):
            raise _power_failure("the power system's state is not finite")
    system.x = list(result["x"])
    return result


def _power_failure(message, kind=FloatingPointError):
    error = kind(message)
    error.failed_module = "power system"
    return error


def _rk4(system, x0, torque_cmds, aero_power_fn, h, steps, time_aware=False):
    n = STATE_SIZE
    x = list(x0)
    ground = tether = storage = 0.0
    motor = [[0.0]*len(MOTOR_AUX) for _ in range(NUM_MOTORS)]
    worst_ratio, stiff_stages, low_voltage, low_current_stages, saturated_stages = 0.0, 0, 0, 0, 0
    lowest = min(x[V0 + i] for i in range(NUM_BLOCKS))
    for step in range(steps):
        taus = (step*h, step*h + 0.5*h, step*h + 0.5*h, step*h + h) if time_aware else (None,)*4
        k1, a1, i1 = stage(system, x, torque_cmds, aero_power_fn, taus[0])
        x2 = [x[j] + 0.5*h*k1[j] for j in range(n)]
        k2, a2, i2 = stage(system, x2, torque_cmds, aero_power_fn, taus[1])
        x3 = [x[j] + 0.5*h*k2[j] for j in range(n)]
        k3, a3, i3 = stage(system, x3, torque_cmds, aero_power_fn, taus[2])
        x4 = [x[j] + h*k3[j] for j in range(n)]
        k4, a4, i4 = stage(system, x4, torque_cmds, aero_power_fn, taus[3])
        x = [x[j] + h/6.0*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]) for j in range(n)]
        ground += h/6.0*(a1[0] + 2.0*a2[0] + 2.0*a3[0] + a4[0])
        tether += h/6.0*(a1[1] + 2.0*a2[1] + 2.0*a3[1] + a4[1])
        storage += h/6.0*(a1[2] + 2.0*a2[2] + 2.0*a3[2] + a4[2])
        for i in range(NUM_MOTORS):
            for q in range(len(MOTOR_AUX)):
                motor[i][q] += h/6.0*(a1[3][i][q] + 2.0*a2[3][i][q] + 2.0*a3[3][i][q] + a4[3][i][q])
        for stage_x in (x2, x3, x4, x):
            for i in range(NUM_BLOCKS):
                lowest = min(lowest, stage_x[V0 + i])
        for info in (i1, i2, i3, i4):
            if info["scale"] > 0.0:
                worst_ratio = max(worst_ratio, abs(info["residual"])/info["scale"])
            stiff_stages += int(info["stiff"])
            low_current_stages += int(info["stiff_low_current"])
            saturated_stages += int(info["stiff_saturated"])
            low_voltage += int(info["low_voltage"])
    return dict(x=x, ground_energy_J=ground, tether_loss_J=tether, storage_integral_J=storage,
                motor={name: [motor[i][q] for i in range(NUM_MOTORS)] for q, name in enumerate(MOTOR_AUX)},
                e3a_worst_residual_over_scale=worst_ratio, stiff_stages=stiff_stages, low_current_stages=low_current_stages,
                saturated_stages=saturated_stages, low_voltage_stages=low_voltage, steps=steps,
                min_block_voltage_V=lowest)


def stored_energy_J(system, x):
    """Block capacitors (non-faulted) and rotors: 1/2 C V^2 + 1/2 I omega^2 (E3b)."""
    p = system.config.power_sys
    e = 0.0
    for i in range(NUM_BLOCKS):
        if not system.is_block_faulted(i):
            e += 0.5*p.C_block*x[V0 + i]*x[V0 + i]
    for i in range(NUM_MOTORS):
        e += 0.5*system.config.inertia_kg_m2[i]*x[OMEGA0 + i]*x[OMEGA0 + i]
    return e


def stored_energy_parts_J(system, x):
    """The rotors' kinetic energy and the block capacitors' (non-faulted) energy separately (R4's split, addendum 3
    item 8); their sum in this order is not ``stored_energy_J``'s running sum and is not used for E3b."""
    p = system.config.power_sys
    rotors = capacitors = 0.0
    for i in range(NUM_MOTORS):
        rotors += 0.5*system.config.inertia_kg_m2[i]*x[OMEGA0 + i]*x[OMEGA0 + i]
    for i in range(NUM_BLOCKS):
        if not system.is_block_faulted(i):
            capacitors += 0.5*p.C_block*x[V0 + i]*x[V0 + i]
    return rotors, capacitors
