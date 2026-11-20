"""Makani's motor and motor-controller model (borrowed part, transcribed; opt-in, used by nothing by default).

Source: Makani snapshot 1d66923, ``sim/physics/motors.cc`` (Apache License 2.0, Makani Technologies LLC):
``CalcTorqueLimits`` (lines 26-101), ``CalcMotorPower`` (110-184), ``CalcMotorControllerLoss`` (189-217); the parameters
are ``MotorParams`` (``sim/sim_types.h:286-305``) with the values of ``config/powertrain/motors.py:21-106`` (here
``cases/powertrain/makani_m600_motors.json``). ``Saturate`` is ``common/c_math/util.c:211-214`` without its
``assert(low <= high)`` (the bounds here are constants with low < high). Power-energy contract (plan 3.9): written as
development code before revision 1; judged by E2a (``docs/tasks/OF-0008-g5control20261006/power_e2_checks.py``) and,
inside the stacked power system, by E2b/E2d.

Conventions as in the source: ``voltage`` [V] is the motor's bus voltage (clamped at zero), ``rotor_vel`` [rad/s] the
rotor's speed, ``torque`` [N m] the motor torque on the rotor; power is positive for generation, losses negative.
The inputs Makani passes (``sim/models/stacked_power_sys.cc``; review B-R04): ``CalcTorqueLimits`` gets the magnitude of
the motor block's voltage (``fabs(GetMotorBlockVoltage(...))``, lines 142-144), ``CalcMotorPower`` the signed block
voltage (``block_voltages(i)``, lines 222-224 and 252-254); both get ``AngularSpeed()`` = omega times the rotor's
direction (``sim/models/actuators/rotor.h:107-109``), the speed in the rotor's own sense, non-negative in normal
operation. The speed-loss polynomial is not odd in the speed: at -150 rad/s it gives +411 W (a "loss" of the wrong
sign); a caller should pass the speed as Makani does.

Scope of the transcription (review B-R03): it equals the source bit for bit for finite inputs (criterion E2a). C's
``fmin``/``fmax`` are copied as glibc behaves (``fmin``/``fmax`` below: on ties the second argument, with a NaN the
other one; contract addendum 2 item 2), so a NaN input is treated as in C there; Python still raises
``ZeroDivisionError`` or ``OverflowError`` where C returns an infinity.

The source writes the factor of the resistive and conduction losses as ``-3 / 2`` (``motors.cc:163, 193``), which in C++
is integer division and equals -1. The comment at line 162 says "Loss [W} for 3 phases.  Divide by two to convert peak to
rms", i.e. 1.5 for the resistive loss; the conduction loss (lines 191-193) has the same expression and its comment does
not mention the division - that 1.5 is meant there too is our inference by analogy (review B-R07). ``loss_factor``
selects the factor for both: ``LOSS_FACTOR_AS_CODED`` (1.0, what Makani's simulator computes; default) or
``LOSS_FACTOR_AS_COMMENTED`` (1.5); the power-energy contract's choice (2) decided: the runs use the factor as coded,
and R4 reports the extra loss the factor 1.5 would add.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LOSS_FACTOR_AS_CODED = 1.0          # C++ integer division: -3 / 2 == -1 (motors.cc:163, 193)
LOSS_FACTOR_AS_COMMENTED = 1.5
DBL_EPSILON = 2.220446049250313e-16
LIMIT_PHASE_CURRENT, LIMIT_POWER = "phase_current", "power"     # kSimMotorLimitPhaseCurrent, kSimMotorLimitPower


@dataclass(frozen=True)
class MotorParams:
    """``MotorParams`` of ``sim/sim_types.h:286-305``."""
    modulation_limit: float
    phase_current_cmd_limit: float
    iq_cmd_lower_limit: float
    iq_cmd_upper_limit: float
    Lq: float
    Ld: float
    Rs: float
    flux_linkage: float
    num_pole_pairs: int
    hysteresis_loss_coefficient: float
    omega_loss_coefficient_cubic: float
    omega_loss_coefficient_sq: float
    omega_loss_coefficient_lin: float
    rds_on: float
    specific_switching_loss: float
    fixed_loss_sq_coeff: float
    fixed_loss_lin_coeff: float
    switching_frequency: float

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != "openfawe.makani_motor_params.v1":
            raise ValueError(f"{path}: unsupported schema {data.get('schema')!r}")
        values = {name: data[name] for name in cls.__dataclass_fields__}
        values["num_pole_pairs"] = int(values["num_pole_pairs"])
        return cls(**values)


@dataclass(frozen=True)
class TorqueLimits:
    lower_limit: float
    lower_constraint: str
    upper_limit: float
    upper_constraint: str


def _check_non_salient(params):
    if not DBL_EPSILON >= abs(params.Ld - params.Lq):       # DCHECK_GE(DBL_EPSILON, fabs(Ld - Lq))
        raise ValueError("the motor model does not support saliency (Ld != Lq)")


def fmin(x, y):
    """C's ``fmin`` as glibc 2.39 on x86-64 behaves (probed; power-energy contract revision 1 addendum 2 item 2): with equal
    arguments (+0 and -0 included) the second one, with one NaN the other argument. Python's ``min`` returns the first."""
    if x != x:
        return y
    if y != y:
        return x
    return x if x < y else y


def fmax(x, y):
    """C's ``fmax`` as glibc 2.39 on x86-64 behaves (see ``fmin``)."""
    if x != x:
        return y
    if y != y:
        return x
    return x if x > y else y


def saturate(x, low, high):
    """``Saturate`` (common/c_math/util.c:211-214, without its assert): fmin(fmax(x, low), high)."""
    return fmin(fmax(x, low), high)


_saturate = saturate


def calc_torque_limits(voltage, rotor_vel, params, *, parts=False):
    """``CalcTorqueLimits`` (motors.cc:26-101): lower and upper torque limits and which constraint sets each. With
    ``parts`` also the branches taken (for E2's coverage count; review B-R01): whether the power circle set each side,
    whether ``cos_idq`` was clamped (below, inside, above) and which ``theta_ref`` branch ran, and whether each
    phase-current test passed."""
    if voltage < 0.0:
        voltage = 0.0
    _check_non_salient(params)
    L, Rs, lam = params.Lq, params.Rs, params.flux_linkage
    i_phase_lim = params.phase_current_cmd_limit
    npp = params.num_pole_pairs
    omega_e = rotor_vel*npp
    vdq_max = voltage*(1.0/math.sqrt(3.0))*params.modulation_limit
    z2 = Rs*Rs + L*L*omega_e*omega_e
    iq_lower, iq_upper = params.iq_cmd_lower_limit, params.iq_cmd_upper_limit
    lower_constraint = upper_constraint = LIMIT_PHASE_CURRENT
    id_center = -omega_e*omega_e*L*lam/z2
    iq_center = -Rs*omega_e*lam/z2
    iq_radius = vdq_max/math.sqrt(z2)
    if iq_lower < iq_center - iq_radius:
        lower_constraint, iq_lower = LIMIT_POWER, iq_center - iq_radius
    if iq_upper > iq_center + iq_radius:
        upper_constraint, iq_upper = LIMIT_POWER, iq_center + iq_radius
    cos_idq = ((vdq_max*vdq_max - z2*i_phase_lim*i_phase_lim - lam*lam*omega_e*omega_e)
               / (2.0*fmax(abs(omega_e), 1.0)*lam*i_phase_lim*math.sqrt(z2)))
    branches = dict(power_lower=lower_constraint == LIMIT_POWER, power_upper=upper_constraint == LIMIT_POWER,
                    cos_idq="below" if cos_idq < -1.0 else ("above" if cos_idq > 1.0 else "inside"),
                    theta_ref="atan" if abs(omega_e) > DBL_EPSILON else "zero") if parts else None
    cos_idq = _saturate(cos_idq, -1.0, 1.0)
    theta_delta = math.acos(cos_idq)
    theta_ref = math.atan(Rs/(omega_e*L)) if abs(omega_e) > DBL_EPSILON else 0.0
    theta = fmin(theta_ref - theta_delta, -0.5*math.pi)
    phase_lower = id_center < i_phase_lim*math.cos(theta) and i_phase_lim*math.sin(theta) > iq_lower
    if phase_lower:
        lower_constraint, iq_lower = LIMIT_PHASE_CURRENT, i_phase_lim*math.sin(theta)
    theta = fmax(theta_ref + theta_delta, 0.5*math.pi)
    phase_upper = id_center < i_phase_lim*math.cos(theta) and i_phase_lim*math.sin(theta) < iq_upper
    if phase_upper:
        upper_constraint, iq_upper = LIMIT_PHASE_CURRENT, i_phase_lim*math.sin(theta)
    limits = TorqueLimits(1.5*npp*lam*iq_lower, lower_constraint, 1.5*npp*lam*iq_upper, upper_constraint)
    if parts:
        return limits, dict(branches, phase_lower=bool(phase_lower), phase_upper=bool(phase_upper))
    return limits


def calc_motor_controller_loss(voltage, peak_phase_current_sq, params, *, loss_factor=LOSS_FACTOR_AS_CODED):
    """``CalcMotorControllerLoss`` (motors.cc:189-217); negative for loss."""
    conduction_loss = -loss_factor*peak_phase_current_sq*params.rds_on
    variable = -(3.0*2.0/math.pi*voltage*math.sqrt(peak_phase_current_sq)*params.specific_switching_loss)
    fixed = -3.0*(params.fixed_loss_sq_coeff*voltage + params.fixed_loss_lin_coeff)*voltage
    return conduction_loss + params.switching_frequency*(variable + fixed)


def calc_motor_power(voltage, torque, rotor_vel, params, *, loss_factor=LOSS_FACTOR_AS_CODED, parts=False):
    """``CalcMotorPower`` (motors.cc:110-184): the motor's electrical terminal power, positive for generation. With
    ``parts`` also the terms (mechanical, resistive, speed, hysteresis, controller), the peak phase current squared and the
    branch of the peak current (``unreachable``: the torque is outside the voltage circle; ``id_negative``,
    ``id_non_negative``: the sign of the d-axis current on the circle; review B-R01)."""
    if voltage < 0.0:
        voltage = 0.0
    _check_non_salient(params)
    L, Rs, lam = params.Lq, params.Rs, params.flux_linkage
    npp = params.num_pole_pairs
    omega_e = rotor_vel*npp
    vdq_max = voltage*(1.0/math.sqrt(3.0))*params.modulation_limit
    z2 = Rs*Rs + L*L*omega_e*omega_e
    iq = torque/(1.5*npp*lam)
    id_center = -omega_e*omega_e*L*lam/z2
    iq_center = -Rs*omega_e*lam/z2
    iq_radius = vdq_max/math.sqrt(z2)
    peak_sq = iq*iq
    iq_height = iq - iq_center
    if abs(iq_height) > iq_radius:
        peak_sq = peak_sq + id_center*id_center
        branch = "unreachable"
    else:
        id_ = id_center + math.sqrt(iq_radius*iq_radius - iq_height*iq_height)
        branch = "id_negative" if id_ < 0 else "id_non_negative"
        if id_ < 0:
            peak_sq = peak_sq + id_*id_
    mechanical = -torque*rotor_vel
    resistive = -loss_factor*peak_sq*Rs
    speed = -(params.omega_loss_coefficient_cubic*rotor_vel*rotor_vel + params.omega_loss_coefficient_sq*rotor_vel
              + params.omega_loss_coefficient_lin)*rotor_vel
    hysteresis = -0.5*params.hysteresis_loss_coefficient*peak_sq*rotor_vel*rotor_vel
    controller = calc_motor_controller_loss(voltage, peak_sq, params, loss_factor=loss_factor)
    total = mechanical + resistive + speed + hysteresis + controller
    if parts:
        return total, dict(mechanical_W=mechanical, resistive_W=resistive, speed_W=speed, hysteresis_W=hysteresis,
                           controller_W=controller, peak_phase_current_sq_A2=peak_sq, branch=branch)
    return total
