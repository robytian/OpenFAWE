# SPDX-License-Identifier: Apache-2.0
"""The compiled path of the stacked power system's period integration (OF-0014 revision 2; opt-in).

``makani_power_sys.integrate_period`` with the rotors' aerodynamic power from their tables
(``makani_power_sys.TablePower``) transcribed once more with numba (nopython, no fastmath): the same operations in the same
order on the same doubles, so the result is the Python path's bit for bit (criteria PJ-U, PJ-1..PJ-3). What changes is
only where the arithmetic runs:

- ``StackedPowerSystem.derived`` (parts), ``derivatives``, ``makani_motors.calc_torque_limits`` and ``calc_motor_power``,
  ``stage``, ``_rk4`` and the rerun with the smaller sub-step of ``integrate_period``: every expression is copied from
  those functions (left to right as Python evaluates it); the state is a float64 array;
- the rotor table (``rotor.table.RotorTable.power_W``): ``bisect_right`` on the inner grid points, the fraction, the
  saturated bilinear form of ``_bilinear_list`` and ``density * n**3 * diameter**5 * cp``. CPython's float ``**`` is C's
  ``pow``; here it is ``math.pow`` with the exponent passed in as a value (so that the compiler cannot turn it into
  products), and ``diameter**5`` is computed by Python beforehand (the same ``pow`` on the same double);
- ``fmin``, ``fmax`` and ``saturate`` as ``makani_motors`` copies glibc; Python's two-argument ``min`` and ``max`` (the first
  argument unless the second is strictly smaller or larger) where the Python path uses them.

Not here: the branch trace (``StackedPowerSystem.trace``; the caller keeps the Python path when a trace is set), the
time-aware look-up of E2c and any ``aero_power_fn`` other than the tables. A NaN or infinite table index raises (Python's
``math.floor`` raises there too). The motors must be non-salient (checked in Python, as ``_check_non_salient``).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import makani_motors as MM
from . import makani_power_sys as MPS

NUM_MOTORS, NUM_BLOCKS = MPS.NUM_MOTORS, MPS.NUM_BLOCKS
FTC, VG = MPS.FTC, MPS.VG
V0, SC0, VCX0, SFU0, SFL0, IERR0, OMEGA0 = MPS.V0, MPS.SC0, MPS.VCX0, MPS.SFU0, MPS.SFL0, MPS.IERR0, MPS.OMEGA0
STATE_SIZE = MPS.STATE_SIZE
NAUX = len(MPS.MOTOR_AUX)
DBL_EPSILON = MM.DBL_EPSILON
LIMIT_NONE, LIMIT_GROUND_POWER = MPS.LIMIT_NONE, MPS.LIMIT_GROUND_POWER
LIMIT_PHASE_CURRENT_ID, LIMIT_POWER_ID = MPS.LIMIT_PHASE_CURRENT_ID, MPS.LIMIT_POWER_ID
SPEED_TORQUE, SPEED_UPPER, SPEED_LOWER = MPS.SPEED_TORQUE, MPS.SPEED_UPPER, MPS.SPEED_LOWER

# indices into the packed parameter arrays (the dataclasses' field order)
P_USE_COMP, P_R_TETHER, P_R_SOURCE, P_V_SOURCE_0, P_C_BLOCK = range(5)
(S_CUTOFF, S_GV_POLE, S_MIN_GV_COMP, S_MAX_GV_COMP, S_KP_RV, S_KI_RV, S_RV_POLE, S_RV_TORQUE_POLE, S_KP_V, S_V_POLE,
 S_V_ZERO, S_FC_STACK, S_KP_STACK, S_DRAIN, S_RATE_LIMIT, S_SPEED_POLE, S_MIN_TC, S_KP_EXCESS, S_V_AVG_SAT) = range(19)
(M_MOD, M_IPH, M_IQ_LO, M_IQ_HI, M_LQ, M_LD, M_RS, M_LAM, M_NPP, M_HYST, M_C3, M_C2, M_C1, M_RDS, M_SSL, M_FSQ, M_FLIN,
 M_FSW) = range(18)
SIM_FIELDS = tuple(MPS.PowerSysSimParams.__dataclass_fields__)
MOTOR_FIELDS = tuple(MM.MotorParams.__dataclass_fields__)
assert SIM_FIELDS[S_V_AVG_SAT] == "voltage_average_upper_sat" and MOTOR_FIELDS[M_FSW] == "switching_frequency"


# ------------------------------------------------------------------------------------------------- scalar helpers
@njit(cache=True, inline="always")
def _fmin(x, y):
    if x != x:
        return y
    if y != y:
        return x
    return x if x < y else y


@njit(cache=True, inline="always")
def _fmax(x, y):
    if x != x:
        return y
    if y != y:
        return x
    return x if x > y else y


@njit(cache=True, inline="always")
def _saturate(x, low, high):
    return _fmin(_fmax(x, low), high)


@njit(cache=True, inline="always")
def _pymin(a, b):
    return b if b < a else a


@njit(cache=True, inline="always")
def _pymax(a, b):
    return b if b > a else a


@njit(cache=True, inline="always")
def _faulted(conn, i):
    return not (conn[i] and conn[i + NUM_MOTORS//2])


@njit(cache=True, inline="always")
def _num_active(conn):
    n = 0
    for i in range(NUM_BLOCKS):
        if not _faulted(conn, i):
            n += 1
    return n


# ------------------------------------------------------------------------------------------------------ the rotor table
@njit(cache=True, inline="always")
def _fractional_index(grid, inner, value):
    lo, hi = 0, inner.shape[0]
    while lo < hi:                                   # bisect.bisect_right
        mid = (lo + hi)//2
        if value < inner[mid]:
            hi = mid
        else:
            lo = mid + 1
    low, high = grid[lo], grid[lo + 1]
    return lo + (value - low) / (high - low)


@njit(cache=True, inline="always")
def _floor_index(v):
    if v != v:
        raise ValueError("cannot convert float NaN to integer")
    if v == np.inf or v == -np.inf:
        raise OverflowError("cannot convert float infinity to integer")
    return int(math.floor(v))


@njit(cache=True)
def _power_W(r, omega, freestream, density, omegas, om_inner, speeds, sp_inner, kp, diam5, three):
    """``RotorTable.power_W(omega, freestream, density)`` of rotor ``r``."""
    n = omega / (2.0 * np.pi)
    s = _fractional_index(omegas[r], om_inner[r], omega)
    t = _fractional_index(speeds[r], sp_inner[r], freestream)
    size1, size2 = kp.shape[1], kp.shape[2]
    s0 = min(max(_floor_index(s), 0), size1 - 2)
    t0 = min(max(_floor_index(t), 0), size2 - 2)
    s = _pymin(_pymax(s, 0.0), float(size1 - 1))
    t = _pymin(_pymax(t, 0.0), float(size2 - 1))
    ds, dt = s - s0, t - t0
    z00, z10, z01, z11 = kp[r, s0, t0], kp[r, s0 + 1, t0], kp[r, s0, t0 + 1], kp[r, s0 + 1, t0 + 1]
    cp = z00 + (z10 - z00) * ds + (z01 - z00) * dt + (z00 - z10 - z01 + z11) * ds * dt
    return density * math.pow(n, three) * diam5[r] * cp


# --------------------------------------------------------------------------------------------------------- the motors
@njit(cache=True)
def _torque_limits(voltage, rotor_vel, M):
    """``calc_torque_limits``: (lower limit, lower constraint id, upper limit, upper constraint id)."""
    if voltage < 0.0:
        voltage = 0.0
    L, Rs, lam = M[M_LQ], M[M_RS], M[M_LAM]
    i_phase_lim = M[M_IPH]
    npp = M[M_NPP]
    omega_e = rotor_vel*npp
    vdq_max = voltage*(1.0/math.sqrt(3.0))*M[M_MOD]
    z2 = Rs*Rs + L*L*omega_e*omega_e
    iq_lower, iq_upper = M[M_IQ_LO], M[M_IQ_HI]
    lower_constraint = LIMIT_PHASE_CURRENT_ID
    upper_constraint = LIMIT_PHASE_CURRENT_ID
    id_center = -omega_e*omega_e*L*lam/z2
    iq_center = -Rs*omega_e*lam/z2
    iq_radius = vdq_max/math.sqrt(z2)
    if iq_lower < iq_center - iq_radius:
        lower_constraint, iq_lower = LIMIT_POWER_ID, iq_center - iq_radius
    if iq_upper > iq_center + iq_radius:
        upper_constraint, iq_upper = LIMIT_POWER_ID, iq_center + iq_radius
    cos_idq = ((vdq_max*vdq_max - z2*i_phase_lim*i_phase_lim - lam*lam*omega_e*omega_e)
               / (2.0*_fmax(abs(omega_e), 1.0)*lam*i_phase_lim*math.sqrt(z2)))
    cos_idq = _saturate(cos_idq, -1.0, 1.0)
    theta_delta = math.acos(cos_idq)
    theta_ref = math.atan(Rs/(omega_e*L)) if abs(omega_e) > DBL_EPSILON else 0.0
    theta = _fmin(theta_ref - theta_delta, -0.5*math.pi)
    if id_center < i_phase_lim*math.cos(theta) and i_phase_lim*math.sin(theta) > iq_lower:
        lower_constraint, iq_lower = LIMIT_PHASE_CURRENT_ID, i_phase_lim*math.sin(theta)
    theta = _fmax(theta_ref + theta_delta, 0.5*math.pi)
    if id_center < i_phase_lim*math.cos(theta) and i_phase_lim*math.sin(theta) < iq_upper:
        upper_constraint, iq_upper = LIMIT_PHASE_CURRENT_ID, i_phase_lim*math.sin(theta)
    return 1.5*npp*lam*iq_lower, lower_constraint, 1.5*npp*lam*iq_upper, upper_constraint


@njit(cache=True)
def _motor_power(voltage, torque, rotor_vel, M, loss_factor, terms, r):
    """``calc_motor_power(..., parts=True)``: the total; ``terms[r]`` gets (mechanical, resistive, speed, hysteresis,
    controller, peak phase current squared)."""
    if voltage < 0.0:
        voltage = 0.0
    L, Rs, lam = M[M_LQ], M[M_RS], M[M_LAM]
    npp = M[M_NPP]
    omega_e = rotor_vel*npp
    vdq_max = voltage*(1.0/math.sqrt(3.0))*M[M_MOD]
    z2 = Rs*Rs + L*L*omega_e*omega_e
    iq = torque/(1.5*npp*lam)
    id_center = -omega_e*omega_e*L*lam/z2
    iq_center = -Rs*omega_e*lam/z2
    iq_radius = vdq_max/math.sqrt(z2)
    peak_sq = iq*iq
    iq_height = iq - iq_center
    if abs(iq_height) > iq_radius:
        peak_sq = peak_sq + id_center*id_center
    else:
        id_ = id_center + math.sqrt(iq_radius*iq_radius - iq_height*iq_height)
        if id_ < 0:
            peak_sq = peak_sq + id_*id_
    mechanical = -torque*rotor_vel
    resistive = -loss_factor*peak_sq*Rs
    speed = -(M[M_C3]*rotor_vel*rotor_vel + M[M_C2]*rotor_vel + M[M_C1])*rotor_vel
    hysteresis = -0.5*M[M_HYST]*peak_sq*rotor_vel*rotor_vel
    # calc_motor_controller_loss
    conduction_loss = -loss_factor*peak_sq*M[M_RDS]
    variable = -(3.0*2.0/math.pi*voltage*math.sqrt(peak_sq)*M[M_SSL])
    fixed = -3.0*(M[M_FSQ]*voltage + M[M_FLIN])*voltage
    controller = conduction_loss + M[M_FSW]*(variable + fixed)
    total = mechanical + resistive + speed + hysteresis + controller
    terms[r, 0] = mechanical
    terms[r, 1] = resistive
    terms[r, 2] = speed
    terms[r, 3] = hysteresis
    terms[r, 4] = controller
    terms[r, 5] = peak_sq
    return total


# ---------------------------------------------------------------------------------------------- derived and derivatives
@njit(cache=True)
def _derived(x, torque_cmds, conn, released, P, S, M, dirs, loss_factor, d_scalars, torques, corrections, int_derivs,
             block_powers, terms, has_terms):
    """``StackedPowerSystem.derived(x, torque_cmds, parts=True)``: fills the arrays; ``d_scalars`` = (wing voltage,
    tether current)."""
    lo_lim = np.empty(NUM_MOTORS)
    hi_lim = np.empty(NUM_MOTORS)
    lo_con = np.empty(NUM_MOTORS, np.int64)
    hi_con = np.empty(NUM_MOTORS, np.int64)
    for i in range(NUM_MOTORS):
        lo_lim[i], lo_con[i], hi_lim[i], hi_con[i] = _torque_limits(abs(x[V0 + i % NUM_BLOCKS]), x[OMEGA0 + i]*dirs[i], M)
    wing_voltage = 0.0
    for i in range(NUM_BLOCKS):
        if not _faulted(conn, i):
            wing_voltage += x[V0 + i]
    tether_current = 0.0 if released else (wing_voltage - x[VG])/(P[P_R_TETHER] + P[P_R_SOURCE])
    for i in range(NUM_BLOCKS):
        r_bot, r_top = i, i + NUM_MOTORS//2
        if _faulted(conn, i):
            for k in range(2):
                r = r_bot if k == 0 else r_top
                torque = 0.0
                if torque > hi_lim[r]:
                    torque = hi_lim[r]
                elif torque < lo_lim[r]:
                    torque = lo_lim[r]
                torques[r] = torque
                corrections[r] = 0.0
                int_derivs[r] = 0.0
                has_terms[r] = False
            block_powers[i] = 0.0
        else:
            voltage_err = x[V0 + i] - _fmin(wing_voltage/_num_active(conn), S[S_V_AVG_SAT])
            p0 = 0.0
            p1 = 0.0
            for k in range(2):
                r = r_bot if k == 0 else r_top
                correction = S[S_KP_V]*S[S_V_POLE]/S[S_V_ZERO]*(voltage_err + x[VCX0 + r])
                # CalcMotorSpeedCommand
                current_excess = S[S_MIN_TC] - tether_current
                omega_upper_cmd = _fmax(0.0, x[SFU0 + r] - S[S_KP_EXCESS]*_fmax(0.0, current_excess) + x[SC0 + r % NUM_BLOCKS])
                omega_lower_cmd = _fmax(0.0, x[SFL0 + r] - S[S_KP_EXCESS]*_fmax(0.0, current_excess) + x[SC0 + r % NUM_BLOCKS])
                torque_request = torque_cmds[r]
                # CalcMotorTorqueCommand
                rotor_vel = x[OMEGA0 + r]*dirs[r]
                int_rotor_vel_err = x[IERR0 + r]
                int_deriv = 0.0
                omega_upper_error = omega_upper_cmd - rotor_vel
                omega_lower_error = omega_lower_cmd - rotor_vel
                if omega_lower_cmd > omega_upper_cmd:
                    omega_lower_error = omega_upper_error
                torque_upper_cmd = S[S_KP_RV]*(omega_upper_error) + int_rotor_vel_err
                torque_lower_cmd = S[S_KP_RV]*(omega_lower_error) + int_rotor_vel_err
                if torque_request >= torque_upper_cmd:
                    torque_cmd, controller_state = torque_upper_cmd, SPEED_UPPER
                elif torque_request <= torque_lower_cmd:
                    torque_cmd, controller_state = torque_lower_cmd, SPEED_LOWER
                else:
                    torque_cmd, controller_state = torque_request, SPEED_TORQUE
                    int_deriv = S[S_RV_TORQUE_POLE]*(int_rotor_vel_err - _saturate(torque_request, lo_lim[r], hi_lim[r]))
                if torque_cmd >= hi_lim[r]:
                    torque_cmd = hi_lim[r]
                    if controller_state != SPEED_TORQUE:
                        int_deriv = 0.0
                elif torque_cmd <= lo_lim[r]:
                    torque_cmd = lo_lim[r]
                    if controller_state != SPEED_TORQUE:
                        int_deriv = 0.0
                elif controller_state == SPEED_UPPER:
                    int_deriv = (omega_upper_cmd - rotor_vel)*S[S_KI_RV] + S[S_RV_POLE]*int_rotor_vel_err
                elif controller_state == SPEED_LOWER:
                    int_deriv = (omega_lower_cmd - rotor_vel)*S[S_KI_RV] + S[S_RV_POLE]*int_rotor_vel_err
                correction = _fmax(correction, _fmin(0.0, -torque_cmd))
                correction = _fmin(correction, hi_lim[r] - torque_cmd)
                torque_cmd += correction
                if torque_cmd >= hi_lim[r]:
                    torque_cmd = hi_lim[r]
                elif torque_cmd <= lo_lim[r]:
                    torque_cmd = lo_lim[r]
                torques[r] = torque_cmd
                power = _motor_power(x[V0 + i], torques[r], x[OMEGA0 + r]*dirs[r], M, loss_factor, terms, r)
                has_terms[r] = True
                int_derivs[r] = int_deriv
                corrections[r] = correction
                if k == 0:
                    p0 = power
                else:
                    p1 = power
            block_powers[i] = p0 + p1
    d_scalars[0] = wing_voltage
    d_scalars[1] = tether_current


@njit(cache=True)
def _derivatives(x, conn, released, P, S, dirs, inertia, rl_upper, rl_lower, d_scalars, torques, corrections, int_derivs,
                 block_powers, aero_powers, dx, aero_torques):
    wing_voltage, tether_current = d_scalars[0], d_scalars[1]
    for j in range(STATE_SIZE):
        dx[j] = 0.0
    dx[FTC] = 2.0*math.pi*S[S_CUTOFF]*(tether_current - x[FTC])
    # CalcGroundVoltageCommand
    if released:
        command = 0.0
    else:
        compensation = -P[P_R_TETHER]*x[FTC] if P[P_USE_COMP] != 0.0 else 0.0
        compensation = _saturate(compensation, S[S_MIN_GV_COMP], S[S_MAX_GV_COMP])
        command = P[P_V_SOURCE_0] + compensation
    dx[VG] = -S[S_GV_POLE]*(command - x[VG])
    # CalcMeanVoltageCorrection
    corr_sum, num_active = 0.0, 0.0
    for i in range(NUM_BLOCKS):
        if not _faulted(conn, i):
            corr_sum += x[SC0 + i]
            num_active += 1.0
    mean_wing_voltage_correction = corr_sum/num_active if num_active > 0.0 else 0.0
    for i in range(NUM_BLOCKS):
        if _faulted(conn, i):
            block_cap_current = -x[V0 + i]*S[S_DRAIN]
            dx[SC0 + i] = 0.0
        else:
            if x[V0 + i] <= 0.0:
                block_cap_current = -x[V0 + i]*S[S_DRAIN]
            elif not released:
                block_cap_current = -tether_current + block_powers[i]/_fmax(x[V0 + i], 1.0)
            else:
                block_cap_current = block_powers[i]/_fmax(x[V0 + i], 1.0)
            mean_pair_voltage_correction = (corrections[i] + corrections[i + NUM_MOTORS//2])*0.5
            dx[SC0 + i] = 2.0*math.pi*S[S_FC_STACK]*(
                S[S_KP_STACK]*(mean_pair_voltage_correction - mean_wing_voltage_correction) - x[SC0 + i])
        dx[V0 + i] = block_cap_current/P[P_C_BLOCK]
    for i in range(NUM_MOTORS):
        dx[SFU0 + i] = -S[S_SPEED_POLE]*(rl_upper[i] - x[SFU0 + i])
        dx[SFL0 + i] = -S[S_SPEED_POLE]*(rl_lower[i] - x[SFL0 + i])
        dx[IERR0 + i] = int_derivs[i]
        temp_gain = S[S_KP_V]*S[S_V_POLE]/S[S_V_ZERO]
        voltage_err = x[V0 + i % NUM_BLOCKS] - _fmin(wing_voltage/_num_active(conn), S[S_V_AVG_SAT])
        output = temp_gain*(voltage_err + x[VCX0 + i])
        overage = output - corrections[i]
        voltage_err -= overage/temp_gain
        dx[VCX0 + i] = ((S[S_V_POLE] - S[S_V_ZERO])*voltage_err + S[S_V_POLE]*x[VCX0 + i])
    for i in range(NUM_MOTORS):
        direction = dirs[i]
        motor_torque = torques[i]*direction
        if direction*x[OMEGA0 + i] < 10.0:                   # calc_aero_torque
            aero_torques[i] = 0.0
        else:
            aero_torques[i] = aero_powers[i]/x[OMEGA0 + i]
        dx[OMEGA0 + i] = (motor_torque + aero_torques[i])/inertia[i]


# ------------------------------------------------------------------------------------------------------------ the stage
@njit(cache=True)
def _stage(x, torque_cmds, conn, released, P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, freestreams, density,
           omegas, om_inner, speeds, sp_inner, kp, diam5, three, dx, aux, rows, flags):
    """``stage``: ``dx``; ``aux`` = (ground, tether, storage, residual, scale); ``rows`` the motor rates; ``flags`` =
    (stiff, low current, saturated, low voltage)."""
    d_scalars = np.empty(2)
    torques = np.empty(NUM_MOTORS)
    corrections = np.empty(NUM_MOTORS)
    int_derivs = np.empty(NUM_MOTORS)
    block_powers = np.empty(NUM_BLOCKS)
    terms = np.zeros((NUM_MOTORS, 6))
    has_terms = np.zeros(NUM_MOTORS, np.bool_)
    _derived(x, torque_cmds, conn, released, P, S, M, dirs, loss_factor, d_scalars, torques, corrections, int_derivs,
             block_powers, terms, has_terms)
    aero_powers = np.empty(NUM_MOTORS)
    for i in range(NUM_MOTORS):
        aero_powers[i] = _power_W(i, x[OMEGA0 + i]*dirs[i], freestreams[i], density, omegas, om_inner, speeds, sp_inner,
                                  kp, diam5, three)
    aero_torques = np.empty(NUM_MOTORS)
    _derivatives(x, conn, released, P, S, dirs, inertia, rl_upper, rl_lower, d_scalars, torques, corrections, int_derivs,
                 block_powers, aero_powers, dx, aero_torques)
    current = d_scalars[1]
    ground = x[VG]*current
    tether = current*current*(P[P_R_TETHER] + P[P_R_SOURCE])
    storage = 0.0
    for i in range(NUM_BLOCKS):
        if not _faulted(conn, i):
            storage += P[P_C_BLOCK]*x[V0 + i]*dx[V0 + i]
    for i in range(NUM_MOTORS):
        storage += inertia[i]*x[OMEGA0 + i]*dx[OMEGA0 + i]
    flows = 0.0
    scale = abs(ground) + abs(tether)
    for i in range(NUM_MOTORS):
        aero_applied = aero_torques[i]*x[OMEGA0 + i]
        if not has_terms[i]:
            for q in range(NAUX):
                rows[i, q] = 0.0
            rows[i, 7] = aero_applied
            rows[i, 8] = aero_torques[i]
        else:
            mech, res, spd, hyst, ctrl, peak = terms[i, 0], terms[i, 1], terms[i, 2], terms[i, 3], terms[i, 4], terms[i, 5]
            rows[i, 0] = mech + res + spd + hyst + ctrl
            rows[i, 1] = mech
            rows[i, 2] = res
            rows[i, 3] = spd
            rows[i, 4] = hyst
            rows[i, 5] = ctrl
            rows[i, 6] = -1.0*peak*M[M_RDS]
            rows[i, 7] = aero_applied
            rows[i, 8] = aero_torques[i]
            rows[i, 9] = peak
            losses = res + spd + hyst + ctrl
            flows += losses
            scale += abs(res) + abs(spd) + abs(hyst) + abs(ctrl)
        flows += aero_applied
        scale += abs(aero_applied)
    flows += -ground - tether
    residual = storage - flows
    if _num_active(conn) == 0:
        low_current, saturated = False, False
    else:
        low_current = d_scalars[1] < S[S_MIN_TC]
        saturated = d_scalars[0]/_num_active(conn) > S[S_V_AVG_SAT]
    low_voltage = False
    for i in range(NUM_BLOCKS):
        if x[V0 + i] < 1.0:
            low_voltage = True
    aux[0], aux[1], aux[2], aux[3], aux[4] = ground, tether, storage, residual, scale + abs(storage)
    flags[0] = low_current or saturated
    flags[1] = low_current
    flags[2] = saturated
    flags[3] = low_voltage


@njit(cache=True)
def _rk4(x0, torque_cmds, conn, released, P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, freestreams, density,
         omegas, om_inner, speeds, sp_inner, kp, diam5, three, h, steps, x_out, sums, motor, counts):
    """``_rk4``: the end state to ``x_out``; ``sums`` = (ground, tether, storage, worst ratio, lowest block voltage);
    ``motor`` the motor integrals; ``counts`` = (stiff, low current, saturated, low voltage) stages."""
    n = STATE_SIZE
    x = x0.copy()
    ground = 0.0
    tether = 0.0
    storage = 0.0
    for i in range(NUM_MOTORS):
        for q in range(NAUX):
            motor[i, q] = 0.0
    worst_ratio = 0.0
    stiff_stages, low_voltage, low_current_stages, saturated_stages = 0, 0, 0, 0
    lowest = x[V0]
    for i in range(1, NUM_BLOCKS):
        lowest = _pymin(lowest, x[V0 + i])
    k = np.empty((4, n))
    a = np.empty((4, 5))
    rows = np.empty((4, NUM_MOTORS, NAUX))
    flags = np.zeros((4, 4), np.bool_)
    x2 = np.empty(n)
    x3 = np.empty(n)
    x4 = np.empty(n)
    for step in range(steps):
        _stage(x, torque_cmds, conn, released, P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, freestreams, density,
               omegas, om_inner, speeds, sp_inner, kp, diam5, three, k[0], a[0], rows[0], flags[0])
        for j in range(n):
            x2[j] = x[j] + 0.5*h*k[0, j]
        _stage(x2, torque_cmds, conn, released, P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, freestreams, density,
               omegas, om_inner, speeds, sp_inner, kp, diam5, three, k[1], a[1], rows[1], flags[1])
        for j in range(n):
            x3[j] = x[j] + 0.5*h*k[1, j]
        _stage(x3, torque_cmds, conn, released, P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, freestreams, density,
               omegas, om_inner, speeds, sp_inner, kp, diam5, three, k[2], a[2], rows[2], flags[2])
        for j in range(n):
            x4[j] = x[j] + h*k[2, j]
        _stage(x4, torque_cmds, conn, released, P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, freestreams, density,
               omegas, om_inner, speeds, sp_inner, kp, diam5, three, k[3], a[3], rows[3], flags[3])
        for j in range(n):
            x[j] = x[j] + h/6.0*(k[0, j] + 2.0*k[1, j] + 2.0*k[2, j] + k[3, j])
        ground += h/6.0*(a[0, 0] + 2.0*a[1, 0] + 2.0*a[2, 0] + a[3, 0])
        tether += h/6.0*(a[0, 1] + 2.0*a[1, 1] + 2.0*a[2, 1] + a[3, 1])
        storage += h/6.0*(a[0, 2] + 2.0*a[1, 2] + 2.0*a[2, 2] + a[3, 2])
        for i in range(NUM_MOTORS):
            for q in range(NAUX):
                motor[i, q] += h/6.0*(rows[0, i, q] + 2.0*rows[1, i, q] + 2.0*rows[2, i, q] + rows[3, i, q])
        for which in range(4):
            for i in range(NUM_BLOCKS):
                if which == 0:
                    v = x2[V0 + i]
                elif which == 1:
                    v = x3[V0 + i]
                elif which == 2:
                    v = x4[V0 + i]
                else:
                    v = x[V0 + i]
                lowest = _pymin(lowest, v)
        for which in range(4):
            if a[which, 4] > 0.0:
                worst_ratio = _pymax(worst_ratio, abs(a[which, 3])/a[which, 4])
            stiff_stages += int(flags[which, 0])
            low_current_stages += int(flags[which, 1])
            saturated_stages += int(flags[which, 2])
            low_voltage += int(flags[which, 3])
    for j in range(n):
        x_out[j] = x[j]
    sums[0], sums[1], sums[2], sums[3], sums[4] = ground, tether, storage, worst_ratio, lowest
    counts[0], counts[1], counts[2], counts[3] = stiff_stages, low_current_stages, saturated_stages, low_voltage


# ------------------------------------------------------------------------------------------------------------- Python
def _pack(system, tables, density):
    config = system.config
    MM._check_non_salient(config.motor)
    p = config.power_sys
    P = np.array([1.0 if p.use_ground_voltage_compensation else 0.0, p.R_tether, p.R_source, p.v_source_0, p.C_block],
                 dtype=np.float64)
    S = np.array([float(getattr(config.sim, name)) for name in SIM_FIELDS], dtype=np.float64)
    M = np.array([float(getattr(config.motor, name)) for name in MOTOR_FIELDS], dtype=np.float64)
    if float(int(config.motor.num_pole_pairs)) != M[M_NPP]:
        raise ValueError("the number of pole pairs must be an integer")
    dirs = np.array([float(d) for d in config.dirs], dtype=np.float64)
    inertia = np.array([float(i) for i in config.inertia_kg_m2], dtype=np.float64)
    if len(tables) != NUM_MOTORS:
        raise ValueError("eight rotor tables are required")
    shapes = {t.power_coeffs.shape for t in tables}
    if len(shapes) != 1:
        raise NotImplementedError("the compiled power system needs rotor tables of one shape")
    omegas = np.stack([np.asarray(t.omegas_rad_s, dtype=np.float64) for t in tables])
    speeds = np.stack([np.asarray(t.freestreams_m_s, dtype=np.float64) for t in tables])
    om_inner = np.ascontiguousarray(omegas[:, 1:omegas.shape[1] - 1])
    sp_inner = np.ascontiguousarray(speeds[:, 1:speeds.shape[1] - 1])
    kp = np.stack([np.asarray(t.power_coeffs, dtype=np.float64) for t in tables])
    diam5 = np.array([t.diameter_m ** 5 for t in tables], dtype=np.float64)       # the Python path's pow, once
    return (P, S, M, dirs, inertia, np.array(system.rl_upper, dtype=np.float64), np.array(system.rl_lower, dtype=np.float64),
            float(MM.LOSS_FACTOR_AS_CODED), omegas, om_inner, speeds, sp_inner, kp, diam5, float(density))


def _run(system, x0, torque_cmds, packed, freestreams, h, steps):
    P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor, omegas, om_inner, speeds, sp_inner, kp, diam5, density = packed
    x_out = np.empty(STATE_SIZE)
    sums = np.empty(5)
    motor = np.empty((NUM_MOTORS, NAUX))
    counts = np.zeros(4, np.int64)
    _rk4(np.array(x0, dtype=np.float64), np.array(torque_cmds, dtype=np.float64), np.array(system.connections, dtype=np.bool_),
         bool(system.tether_released), P, S, M, dirs, inertia, rl_upper, rl_lower, loss_factor,
         np.array(freestreams, dtype=np.float64), density, omegas, om_inner, speeds, sp_inner, kp, diam5, 3.0, float(h),
         int(steps), x_out, sums, motor, counts)
    return dict(x=[float(v) for v in x_out], ground_energy_J=float(sums[0]), tether_loss_J=float(sums[1]),
                storage_integral_J=float(sums[2]),
                motor={name: [float(motor[i, q]) for i in range(NUM_MOTORS)] for q, name in enumerate(MPS.MOTOR_AUX)},
                e3a_worst_residual_over_scale=float(sums[3]), stiff_stages=int(counts[0]), low_current_stages=int(counts[1]),
                saturated_stages=int(counts[2]), low_voltage_stages=int(counts[3]), steps=int(steps),
                min_block_voltage_V=float(sums[4]))


def integrate_period(system, t, torque_cmds, tables, freestreams, density, *, sub_step_s=MPS.SUB_STEP_S, period_s=0.01,
                     allow_rerun=True):
    """``makani_power_sys.integrate_period(system, t, torque_cmds, lambda i, s: tables[i].power_W(s, freestreams[i],
    density), ...)`` on the compiled path: the same result, bit for bit. ``system.trace`` must be None."""
    if system.trace is not None:
        raise ValueError("the compiled power system does not record the branch trace; use the Python path")
    steps = int(round(period_s/sub_step_s))
    if abs(steps*sub_step_s - period_s) > 1e-15:
        raise ValueError("the sub-step must divide the period")
    start = list(system.x)
    packed = _pack(system, tables, density)
    result = _run(system, start, torque_cmds, packed, freestreams, sub_step_s, steps)
    result["sub_step_s"], result["rerun"] = sub_step_s, False
    first = dict(stiff_stages=result["stiff_stages"], low_current_stages=result["low_current_stages"],
                 saturated_stages=result["saturated_stages"])
    if allow_rerun and result["stiff_stages"] > 0:
        small = _run(system, start, torque_cmds, packed, freestreams, sub_step_s/MPS.STIFF_DIVISOR,
                     steps*MPS.STIFF_DIVISOR)
        small["sub_step_s"], small["rerun"] = sub_step_s/MPS.STIFF_DIVISOR, True
        result = small
    result["first_attempt_stiff_stages"] = first["stiff_stages"]
    result["first_attempt_low_current_stages"] = first["low_current_stages"]
    result["first_attempt_saturated_stages"] = first["saturated_stages"]
    for value in result["x"]:
        if not math.isfinite(value):
            raise MPS._power_failure("the power system's state is not finite")
    system.x = list(result["x"])
    return result
