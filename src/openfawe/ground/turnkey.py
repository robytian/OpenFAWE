# SPDX-License-Identifier: Apache-2.0
# The order of one period, the joystick schedule, the motor gating and the blown-wing force follow google/makani
# commit 1d66923 (Apache-2.0): sim/kite_sim.cc lines 173-204 (one simulator step), config/common/sim/joystick_sim.py
# lines 110-131 (TurnKey), control/control_output.c lines 87-105 and control/avionics/avionics_interface.c lines
# 828-832 (run_motors latch and motor command), sim/models/actuators/rotor.cc lines 144-168 and 186-203 (blown-wing
# force; speed commands only while the run command is set) with config/m600/sim/rotor_sim.py. Modifications (2026):
# Python driver of the plant of ``ground.perch`` with held loads of the reference plant (``flight.reference_flight``)
# and the rotor tables of the crosswind closed loop (``flight.closed_loop``).
"""TurnKey on the reference plant (plan 3.7 part 3; full-cycle contract, revision 6, K1-K7).

The plant of part 2 (``ground.perch.PerchedSystem``: rigid wing, 19-node variable-length tether, GS02, perch contact)
flies under Makani's whole controller (part 1, ``flight.makani_full``) with the loads part 2 left out:

* **aerodynamics** of the reference plant (``aero.makani_database.MakaniDatabase``, a borrowed model, not the vortex
  method): force and moment about the body origin from the state at the start of each controller period, with the
  flap angles of the servo reference model (``flight.makani_interface.ServoReference``) and the mean rotor thrust
  coefficient of the period before (as the crosswind closed loop: ``RotorThrustCoeff`` of ``rotor.cc`` lines 76-87,
  axial inflow bounded below by 30 m/s);
* **rotors**: the Rev4 two-dimensional table (S0) at the kinematic inflow of each hub, thrust along the axis and the
  axis torque, for the speeds held over the period; the gyroscopic moment ``L x omega`` and the reaction moment
  ``-I dOmega/dt`` at every Runge-Kutta stage (as ``CrosswindLoop.external``);
* **blown wing** (only with ``blown_wing=True``; K1 has it off): per rotor ``R_y(theta) T a - T a`` in body axes at the
  body origin, ``theta = Crossfade(0.3, 0, V_axial, 6, 10)`` (``rotor.makani_corrections.blown_wing_force``);
  ``blown_wing='tables'`` (OF-0010 revision 11, a segment hook that declares ``BLOWN_WING_TABLES``): the same term only
  in the periods whose acting aerodynamics is the database term (no segment loads take its place);
* **tether drag** with the uniform wind on every node (part 2 had none), or with ``wind_shear`` (plan 3.9b) the wind at
  each node's height.

The aerodynamic, rotor and blown-wing loads are computed once per period and held in body axes about the centre of
mass (K2); tether, contact and ground station are evaluated at every stage of the 1 ms RK4 steps.

One period (K3; the order of the simulator's step and of part 2's G7):

1. the rotor speeds of this period: the speed command of the period before (zero unless that period's output had
   ``run_motors`` and not ``stop_motors``), reached from the speeds of the period before by at most 500 rad/s^2;
   the flap angles of this period: the servo reference model's state;
2. the held loads at the state of the period's start;
3. the derivatives at that state (they give the acceleration of the estimate, K4: this period's held loads);
4. the ground station's derived inputs (levelwind engagement, tether force) and sensors from that state; its discrete
   step; then the controller's ground-station command of the period before;
5. the estimate (truth, ``ground.perch.kite_estimate``) and the ground-side estimator, the controller step;
6. the controller's flap command of the period before moves the servo reference model over this period, and this
   period's commands are kept for the next period (the simulator receives them after integrating, so they act from
   the next period on; the crosswind closed loop of CL1 uses its rotor command in the same step, which differs);
7. ten RK4 steps of 1 ms.

Differences from the simulator (declared in the contract, K1): by default no motor speed loop, rotor equation of
motion, torque limit or power system (plan 3.9a's opt-in ``power_system`` adds Makani's, power-energy contract); no three-dimensional rotor table and no bound-vortex induction of the wing at the hubs; the
public SN02 high-incidence database (no wake model) instead of the simulator's SN03 set; uniform steady wind, no
turbulence or shear (plan 3.9b's opt-in ``wind_shear`` adds Makani's power-law shear, ``ground.wind_shear``); no ground
contact of the wing, and of the tether nodes only with ``tether_ground_contact`` (off by default; full-cycle contract
revision 7, option 甲); perfect sensors (truth), no estimator of the kite;
the servos are their reference model only; fixed-step RK4 instead of RKF45; an onshore fixed platform.

Nothing in the package imports this module.
"""

import ctypes
import math
import time

import numpy as np

from openfawe.vec3 import cross3   # OF-0014: numpy.cross for 3-vectors, bit for bit

from openfawe.flight.free_flight import gyroscopic_moment, held_loads
from openfawe.flight.makani_interface import G_TO_WORLD, ServoReference, rate_limited_speeds
from openfawe.flight.plant import HeldLoads
from openfawe.flight.reference_flight import reference_loads_body
from openfawe.rotor.makani_corrections import blown_wing_force

from .perch import PerchedSystem, kite_estimate, makani_wind_g, perched_initialisation

PERIOD_S = 0.01
# TurnKey's joystick (joystick_sim.py lines 110-131), in controller periods: switch up and throttle off from 0 s,
# switch to the middle at 10.1 s, throttle 'EnterCrosswind' (0.81) at 11.0 s, 'ReturnToPerch' (0.39) at 820 s.
SWITCH_MIDDLE_PERIOD, THROTTLE_CROSSWIND_PERIOD, THROTTLE_RETURN_PERIOD = 1010, 1100, 82000
THROTTLE_CROSSWIND, THROTTLE_RETURN = 0.81, 0.39
GROUND_Z_G = 6.122                    # Parker Ranch: the ground is at z_g = 6.122 m (config/base_station/ground_frame.py)
# K6: end 5 s after the hover gain ramp is first below 1e-8 in HoverDescend; at most 120 s in HoverDescend; a stall of
# 450 s in one mode; 1600 s of simulated time.
GAIN_DONE, GAIN_DONE_HOLD_PERIODS, DESCEND_PERIODS, STALL_PERIODS, END_PERIODS = 1e-8, 500, 12000, 45000, 160000
R1_KEYS = ("load_source", "flight_mode_time", "cw_loop_angle", "cw_current_pos_cw", "cw_path_radius_target", "cw_airspeed_cmd",
           "cw_alpha_cmd", "cw_path_type", "cw_int_crosstrack", "est_airspeed_sph_m_s", "origin_g_m", "held_aero_force_body_N",
           "held_aero_moment_cm_body_Nm", "rotor_moment_cm_body_Nm")      # plan 3.8a, R1: in every record when a hook is given
MODE_ORDER = ("PilotHover", "Perched", "HoverAscend", "HoverPayOut", "HoverPrepTransformGsUp", "HoverTransformGsUp",
              "HoverFullLength", "HoverAccel", "TransIn", "CrosswindNormal", "CrosswindPrepTransOut", "HoverTransOut",
              "HoverPrepTransformGsDown", "HoverTransformGsDown", "HoverReelIn", "HoverDescend")


STOP_RULES = ("throttle_return_period", "stall_periods", "end_periods", "descend_periods", "gain_done_hold_periods")


def stop_rules_of(stop_rules=None):
    """OF-0010 revision 19 addendum 22 (ST-05): the five stop rules a run uses - those given, the rest read from the module
    constants at the call (so a caller that sets the constants keeps working)."""
    rules = dict(throttle_return_period=THROTTLE_RETURN_PERIOD, stall_periods=STALL_PERIODS, end_periods=END_PERIODS,
                 descend_periods=DESCEND_PERIODS, gain_done_hold_periods=GAIN_DONE_HOLD_PERIODS)
    for key, value in (stop_rules or {}).items():
        if key not in STOP_RULES:
            raise ValueError(f"stop_rules: unknown rule {key!r}; the rules are {STOP_RULES}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"stop_rules: {key} must be a positive integer, not {value!r}")
        rules[key] = int(value)
    return rules


def joystick_at(api, period, return_period=None):
    """TurnKey's joystick in controller period ``period`` (``return_period``: ST-05; None is the module constant)."""
    from openfawe.flight.makani_full import Joystick
    back = THROTTLE_RETURN_PERIOD if return_period is None else return_period
    switch = api.kJoystickSwitchPositionUp if period < SWITCH_MIDDLE_PERIOD else api.kJoystickSwitchPositionMiddle
    throttle = 0.0 if period < THROTTLE_CROSSWIND_PERIOD else THROTTLE_CROSSWIND if period < back \
        else THROTTLE_RETURN
    return Joystick(switch_position=switch, throttle=throttle)


def mode_names(api):
    """Flight-mode number -> name (without ``kFlightMode``), from the binding."""
    return {int(getattr(api, f"kFlightMode{name}")): name for name in MODE_ORDER}


def _finite_names(value, prefix=""):
    from openfawe.flight.closed_loop import _finite
    return _finite(value, prefix)


class FlyingSystem(PerchedSystem):
    """``PerchedSystem`` with the held loads of one period, the rotors' gyroscopic and reaction moments and tether wind.

    ``set_period`` stores the loads held over the period (about the centre of mass, body axes) and the rotor speeds
    and their rates; ``rates`` adds them to part 2's loads (contact, tether, gravity). The tether's aerodynamic drag
    sees the uniform wind ``wind_world_m_s`` at every node, or with ``wind_shear`` (plan 3.9b, a
    ``ground.wind_shear.ShearWind``) the wind at each node's height at every evaluation (tether.cc:907).
    """

    held_start = None          # OF-0011 revision 7 addendum 4: the tether's first active node held in a step (None: as before)
    events_located = 0         # supplement 3 (report): the events located so far
    event_kinds = "first"      # addendum 4 supplement 3: "all" also locates the segments' slack/taut and the contacts;
                               # supplement 4: "branches" as "all" with each contactor's branch (perch.contact_branch);
                               # supplement 5: "held" as "branches", the contactors' branches held within each step;
                               # supplement 6: "depth" as "held", with each contactor's depth >= CONTACT_DEPTH_M located
    EVENT_BISECTIONS, EVENTS_PER_STEP = 50, 100                # F2-S's (f2s.py, addendum 81)
    CONTACT_DEPTH_M = 1e-8     # supplement 6: ten times Makani's 1e-9 floor of the contact normal (contactor.cc:60-97)

    def __init__(self, params, discs, rotor_inertia_kg_m2, wind_world_m_s, wind_shear=None, **kwargs):
        super().__init__(params, **kwargs)
        self.shear = wind_shear
        self.discs = tuple(discs)
        self.rotor_inertia = float(rotor_inertia_kg_m2)
        self.sense = np.array([disc.direction for disc in self.discs], dtype=float)
        self.axes = np.array([disc.axis for disc in self.discs], dtype=float)        # body axes (discs not placed)
        self.wind = np.asarray(wind_world_m_s, dtype=float).reshape(3)
        self.node_wind = np.tile(self.wind, (self.line.n, 1))
        self.held = HeldLoads()
        self.omegas = np.zeros(len(self.discs))
        self.omega_rates = np.zeros(len(self.discs))

    def set_period(self, held, omegas, omega_rates):
        self.held = held
        self.omegas = np.asarray(omegas, dtype=float).copy()
        self.omega_rates = np.asarray(omega_rates, dtype=float).copy()

    def rotor_moments(self, state):
        """Gyroscopic and reaction moments of the rotors at a stage (inertial components)."""
        momentum = (self.rotor_inertia*self.omegas*self.sense)@self.axes
        reaction = state.rotation@((-self.rotor_inertia*self.omega_rates*self.sense)@self.axes)
        return gyroscopic_moment(state, momentum) + reaction

    def tether_evaluation(self, state, nodes, speeds, gsx):
        pose = state.pose(self.body)
        start, start_vel = self.anchor_world(gsx)
        free_length = self.gs.free_length(float(gsx[self.gs.THETA]))
        if self.held_start is None:
            idx = self.line.indices(free_length)
        else:                                           # addendum 4: within an event-located step (step_events)
            idx = self.line.held_indices(free_length, self.held_start)
        last_x, last_v = self.line.last_node(idx, nodes, speeds, start, start_vel)
        end, end_vel = self.bridle.point(pose, last_x, last_v)
        wind = self.node_wind if self.shear is None else self.shear.node_winds_world(nodes)
        dx, dv, out = self.line.derivatives(free_length, nodes, speeds, start, start_vel, end, end_vel, wind=wind,
                                            **({} if self.held_start is None else dict(indices=idx)))
        out.update(start_point=start, start_velocity=start_vel, end_point=end, end_velocity=end_vel, free_length=free_length)
        return dx, dv, out

    def first_node(self):
        """The tether's first active node from the ground station's present state (addendum 4)."""
        return self.line.indices(self.gs.free_length(float(self.gs.x[self.gs.THETA]))).start

    def event_signature(self, state, nodes, speeds):
        """What an event-located step watches (addendum 4 and supplement 3): the first active node; with
        ``event_kinds == 'all'`` also each active segment's taut (tension > 0) and each perch contactor's contact, from
        the natural indices at the present ground-station and base states; with ``event_kinds == 'branches'``
        (supplement 4) the contacts' part is each contactor's branch (``perch.contact_branch``), which also changes
        where the collision point jumps without the contact changing."""
        first = self.first_node()
        if (self.event_kinds != "all" and self.event_kinds != "branches" and self.event_kinds != "held"
                and self.event_kinds != "depth"):
            return (first,)
        held, self.held_start = self.held_start, None
        try:
            if self.floating is not None:
                self.set_base(self.base_x)
            _, _, out = self.tether_evaluation(state, nodes, speeds, self.gs.x)
            if self.event_kinds == "branches" or self.event_kinds == "held":
                contacts = self.contact_branches(state, self.gs.x)
            elif self.event_kinds == "depth":                   # supplement 6: (branch, deep) per contactor
                contacts = self.contact_branches(state, self.gs.x, depth_m=self.CONTACT_DEPTH_M)
            else:
                _, _, each = self.contact_loads(state, self.gs.x)
                contacts = tuple(bool(np.any(hit)) for _, hit in each)
        finally:
            self.held_start = held
        return (first, tuple(bool(v > 0.0) for v in np.atleast_1d(out["tension"])), contacts)

    def _restore(self, gs_x, base_x):
        self.gs.x = gs_x.copy()
        if base_x is not None:
            self.base_x = base_x.copy()

    def step_events(self, state, nodes, speeds, h):
        """``step`` with event location of the first active node (OF-0011 revision 7 addendum 4; F2-S's ``sub_step``):
        the stages use the node of the step's start (``held_start``); if the node at the end differs, the change time
        is found by bisection (from the start, the node held; the later side kept), the step goes to it and the rest
        continues with the new node. A step without a change is ``step``'s, bit for bit (``held_indices`` returns
        ``indices`` itself and ``derivatives`` gets the same indices)."""
        done = 0.0
        try:
            for _ in range(self.EVENTS_PER_STEP):
                rest = h - done
                s0 = self.first_node()
                sig0 = self.event_signature(state, nodes, speeds)
                g0 = self.gs.x.copy()
                b0 = None if self.floating is None else self.base_x.copy()
                self.held_start = s0
                if self.event_kinds == "held":           # supplement 5: the contact forces keep the start's branches
                    self.held_branches = sig0[2]
                elif self.event_kinds == "depth":        # supplement 6: the same, the depth flags aside
                    self.held_branches = tuple(branch for branch, _ in sig0[2])
                new = self.step(state, nodes, speeds, rest)
                if self.event_signature(*new) == sig0:
                    return new
                lo, hi = 0.0, rest
                for _ in range(self.EVENT_BISECTIONS):
                    mid = 0.5*(lo + hi)
                    self._restore(g0, b0)
                    trial = self.step(state, nodes, speeds, mid)
                    if self.event_signature(*trial) == sig0:
                        lo = mid
                    else:
                        hi = mid
                self._restore(g0, b0)
                state, nodes, speeds = self.step(state, nodes, speeds, hi)
                self.events_located += 1
                done += hi
            raise RuntimeError(f"more than {self.EVENTS_PER_STEP} events (first active node, slack or contact) in one step")
        finally:
            self.held_start = None
            self.held_branches = None

    def rates(self, state, nodes, speeds, gsx):
        if self.floating is not None:                   # OF-0011 rev 6 add 73: the vessel frame of this stage
            self.set_base(self.base_x)
        force, moment, each = self.contact_loads(state, gsx)
        dx, dv, out = self.tether_evaluation(state, nodes, speeds, gsx)
        end_force = out["end"]
        force = force + end_force + state.rotation@self.held.force_body_N
        moment = (moment + cross3(out["end_point"] - state.position_cm_m, end_force)
                  + state.rotation@self.held.moment_cm_body_Nm + self.rotor_moments(state))
        omega = state.angular_velocity_body_rad_s
        inertia = self.body.inertia_body_kg_m2
        acceleration = force/self.body.mass_kg + self.gravity
        spin = self.inverse_inertia@(state.rotation.T@moment - cross3(omega, inertia@omega))
        diag = dict(tether=out, contacts=each, contact_force=force)
        if self.floating is not None:
            diag["base_rate"] = self.floating.rates(state.time_s, self.base_x, out)
        return state.velocity_cm_m_s, acceleration, spin, dx, dv, self.gs.derivatives(gsx), diag


def rotor_freestreams(system, state, hub_induced=None):
    """The axial freestream of each rotor at ``state``, computed as ``period_loads`` computes it (plan 3.9a: held over the
    period for the power system's integration). ``hub_induced`` (OF-0010 revision 15, B'): per disc the velocity the
    vortex system induces at its hub (world axes), added to the wind; None: the arithmetic as before."""
    pose = state.pose(system.body)
    out = []
    for index, disc in enumerate(d.placed(pose.rotation, pose.position_m) for d in system.discs):
        apparent = system.wind - pose.point_velocity(disc.centre_m[None, :])[0]
        if hub_induced is not None:
            apparent = apparent + np.asarray(hub_induced[index], dtype=float)
        out.append(float(disc.axial_freestream(apparent)))
    return out


def power_record(pss, x_start, derived, period, speed_upper_in, speed_lower_in, torque_cmds, message_for_rate,
                 message_for_torque, freestreams, axial_torques):
    """The ``power_*`` fields of one period's record (plan 3.9a; contract revision 1 R2 and addendum 1 item 9): the
    derived values at the period start, the 50 states at the start (E2c), the discrete inputs, the period's integrals
    and the checks of the integration."""
    from openfawe.power import makani_power_sys as mps
    motor = period["motor"]
    sums = {}
    for name in ("resistive_J", "speed_J", "hysteresis_J", "controller_J", "conduction_J", "peak_current_sq_A2s"):     # sums (E3, R4)
        total = 0.0
        for value in motor[name]:
            total += value
        sums[name] = total
    return dict(
        power_state_start=[float(v) for v in x_start],
        power_block_voltage_V=[float(x_start[mps.V0 + i]) for i in range(mps.NUM_BLOCKS)],
        power_wing_voltage_V=float(derived["wing_voltage"]), power_ground_voltage_V=float(x_start[mps.VG]),
        power_tether_current_A=float(derived["tether_current"]), power_filtered_tether_current_A=float(x_start[mps.FTC]),
        power_motor_torque_cmd_Nm=[float(v) for v in derived["motor_torque_cmds"]],
        power_torque_upper_limit_Nm=[float(l.upper_limit) for l in derived["limits"]],
        power_torque_lower_limit_Nm=[float(l.lower_limit) for l in derived["limits"]],
        power_motor_constraint=[int(v) for v in derived["motor_constraints"]],
        power_speed_controller_state=[int(v) for v in derived["speed_controller_state"]],
        power_block_power_W=[float(v) for v in derived["block_powers"]],
        power_motor_current_A=[float(v) for v in derived["motor_currents"]],
        power_speed_cmd_upper_rad_s=[float(v) for v in speed_upper_in], power_speed_cmd_lower_rad_s=[float(v) for v in speed_lower_in],
        power_rate_limited_upper_rad_s=[float(v) for v in pss.rl_upper], power_rate_limited_lower_rad_s=[float(v) for v in pss.rl_lower],
        power_torque_cmd_Nm=[float(v) for v in torque_cmds],
        power_message_for_rate=message_for_rate, power_message_for_torque=message_for_torque,
        power_freestream_m_s=[float(v) for v in freestreams], power_aero_torque_mean_Nm=[float(v) for v in axial_torques],
        power_ground_energy_J=float(period["ground_energy_J"]), power_tether_loss_J=float(period["tether_loss_J"]),
        power_storage_integral_J=float(period["storage_integral_J"]),
        power_stored_energy_start_J=float(mps.stored_energy_J(pss, x_start)), power_stored_energy_end_J=float(mps.stored_energy_J(pss, pss.x)),
        power_electrical_J=[float(v) for v in motor["electrical_J"]], power_mechanical_J=[float(v) for v in motor["mechanical_J"]],
        power_aero_applied_J=[float(v) for v in motor["aero_applied_J"]],
        power_resistive_J=sums["resistive_J"], power_speed_loss_J=sums["speed_J"], power_hysteresis_J=sums["hysteresis_J"],
        power_controller_J=sums["controller_J"], power_conduction_J=sums["conduction_J"], power_peak_current_sq_A2s=sums["peak_current_sq_A2s"],
        power_e3a_worst=float(period["e3a_worst_residual_over_scale"]), power_stiff_stages=int(period["stiff_stages"]),
        power_low_voltage_stages=int(period["low_voltage_stages"]), power_min_block_voltage_V=float(period["min_block_voltage_V"]),
        power_rerun=bool(period["rerun"]), power_sub_step_s=float(period["sub_step_s"]),
        power_low_current_stages=int(period["low_current_stages"]), power_saturated_stages=int(period["saturated_stages"]),
        power_first_attempt_stiff_stages=int(period["first_attempt_stiff_stages"]),
        power_first_attempt_low_current_stages=int(period["first_attempt_low_current_stages"]),
        power_first_attempt_saturated_stages=int(period["first_attempt_saturated_stages"]),
        # per motor (R2, addendum 1 item 9; review P39D-04): the loss terms and the peak current squared of each motor
        power_resistive_each_J=[float(v) for v in motor["resistive_J"]], power_speed_loss_each_J=[float(v) for v in motor["speed_J"]],
        power_hysteresis_each_J=[float(v) for v in motor["hysteresis_J"]], power_controller_each_J=[float(v) for v in motor["controller_J"]],
        power_conduction_each_J=[float(v) for v in motor["conduction_J"]],
        power_peak_current_sq_each_A2s=[float(v) for v in motor["peak_current_sq_A2s"]],
        power_stored_rotor_start_J=float(mps.stored_energy_parts_J(pss, x_start)[0]),
        power_stored_capacitor_start_J=float(mps.stored_energy_parts_J(pss, x_start)[1]),
        power_stored_rotor_end_J=float(mps.stored_energy_parts_J(pss, pss.x)[0]),
        power_stored_capacitor_end_J=float(mps.stored_energy_parts_J(pss, pss.x)[1]))


def power_period(pss, pipeline, config, t, freestreams, aero_power, period_s):
    """Plan 3.9a, the power system over one period (power-energy contract addendum 3 item 7; addendum 1 items 1, 3, 10):
    the rate limiter takes the speed commands of the message pushed two periods before, the torque commands come from
    the message pushed one period before; the 50 states are integrated over the period with the rotors' aerodynamic
    power ``aero_power(i, s, freestream)`` at the held axial ``freestreams``. Returns the period-start speeds (Makani's
    AngularSpeed), the mean angular accelerations, the period-mean aerodynamic torques and the record's ``power_*``
    fields."""
    try:
        return _power_period(pss, pipeline, config, t, freestreams, aero_power, period_s)
    except Exception as error:                          # review P39D-08: any failure inside is the power system's
        if getattr(error, "failed_module", None) is None:
            try:
                error.failed_module = "power system"
            except AttributeError:
                pass
        raise


def _power_period(pss, pipeline, config, t, freestreams, aero_power, period_s):
    from openfawe.power import makani_power_sys as mps
    rate_upper, rate_lower, _ = mps.rotor_commands(pipeline.for_rate, config)
    pss.discrete_step(t, rate_upper, rate_lower)
    _, _, torque_cmds = mps.rotor_commands(pipeline.for_torque, config)
    x_start = list(pss.x)
    count = len(config.dirs)
    omegas = np.array([pss.angular_speed(x_start, i) for i in range(count)], dtype=float)
    if getattr(config, "compiled", False) and pss.trace is None:       # OF-0014 revision 2 (opt-in): the same, compiled
        if not isinstance(aero_power, mps.TablePower):
            raise ValueError("the compiled power system needs the rotors' tables (makani_power_sys.TablePower)")
        from openfawe.power import makani_power_jit as mpj
        period = mpj.integrate_period(pss, t, torque_cmds, aero_power.tables, freestreams, aero_power.density,
                                      period_s=period_s)
    else:
        period = mps.integrate_period(pss, t, torque_cmds, lambda i, s: aero_power(i, s, freestreams[i]), period_s=period_s)
    omega_rates = (np.array([pss.angular_speed(pss.x, i) for i in range(count)], dtype=float) - omegas)/period_s
    axial_torques = [period["motor"]["aero_torque_Nms"][i]/period_s for i in range(count)]
    start_derived = pss.derived(x_start, torque_cmds)
    fields = power_record(pss, x_start, start_derived, period, rate_upper, rate_lower, torque_cmds, pipeline.for_rate,
                          pipeline.for_torque, freestreams, axial_torques)
    return omegas, omega_rates, axial_torques, fields


def _power_system_disc_loads(disc, speed, apparent, density_kg_m3, axial_torque_Nm):
    """``disc.loads`` for plan 3.9a's power system: the speed (Makani's AngularSpeed) may be negative - the table's
    coefficients saturate as Makani's rotor database does (rotor_database.cc:73-118) - and the moment about the axis is
    the given period-mean aerodynamic torque (power-energy contract revision 1 addendum 1 item 10)."""
    freestream = disc.axial_freestream(apparent)
    thrust = disc.table.thrust_N(speed, freestream, density_kg_m3)
    power = disc.table.power_W(speed, freestream, density_kg_m3)
    return dict(axial_freestream_m_s=freestream, thrust_N=thrust, power_W=power, torque_Nm=float(axial_torque_Nm),
                force_N=thrust*disc.axis, moment_Nm=float(axial_torque_Nm)*disc.axis, saturated=disc.table.saturated(speed, freestream))


def period_loads(system, aero, state, flaps_rad, thrust_coeff, density_kg_m3, *, blown_wing=False, aero_override=None,
                 parts=False, rotor_axial_torques=None, wind_rates_body=None, hub_induced=None):
    """The loads held over one period at ``state``: aerodynamics, rotors (speeds ``system.omegas``), blown wing.

    Returns (HeldLoads about the centre of mass in body axes, the mean thrust coefficient for the next period, fields).
    ``aero_override`` (plan 3.8a, a segment hook): HeldLoads that take the place of the database term in the sum; the
    database term is computed all the same and its fields are unchanged. ``parts=True`` returns a fourth item, the
    acting aerodynamic, rotor and blown-wing HeldLoads of the sum. ``rotor_axial_torques`` (plan 3.9a, the power system):
    the rotors' aerodynamic torques about their axes (Makani's signed convention) that take the place of the table's
    torque at ``system.omegas``; the speeds may then be negative. ``wind_rates_body`` (plan 3.9b): the wind-induced
    apparent rates added to the body rates for the database's lookup (``reference_loads_body``); the wind is
    ``system.wind`` (with the shear, the wind at the body origin of the period's start). ``hub_induced`` (OF-0010
    revision 15, B'): per disc the velocity induced at its hub (world axes), added to the rotor's apparent wind; None:
    the arithmetic as before.
    """
    body = system.body
    aero_held, aero_moment_origin, condition = reference_loads_body(
        aero, body, state, system.wind, density_kg_m3, flaps_rad, thrust_coeff,
        **({} if wind_rates_body is None else dict(wind_rates_body=wind_rates_body)))
    pose = state.pose(body)
    rows, force, moment = [], np.zeros(3), np.zeros(3)
    for index, (disc, omega) in enumerate(zip((d.placed(pose.rotation, pose.position_m) for d in system.discs), system.omegas)):
        apparent = system.wind - pose.point_velocity(disc.centre_m[None, :])[0]
        if hub_induced is not None:
            apparent = apparent + np.asarray(hub_induced[index], dtype=float)
        if rotor_axial_torques is None:
            row = disc.loads(float(omega), apparent, density_kg_m3)
        else:
            row = _power_system_disc_loads(disc, float(omega), apparent, density_kg_m3, rotor_axial_torques[index])
        row["area_m2"] = disc.area_m2
        rows.append(row)
        force += row["force_N"]
        moment += cross3(disc.centre_m - pose.position_m, row["force_N"]) + row["moment_Nm"]
    rotor = held_loads(body, pose.rotation, force, moment)
    coefficient = float(np.mean([row["thrust_N"]/(0.5*density_kg_m3*row["area_m2"]*max(abs(row["axial_freestream_m_s"]), 30.0)**2)
                                 for row in rows]))
    blown = np.zeros(3)
    if blown_wing and not (isinstance(blown_wing, str) and blown_wing == "tables" and aero_override is not None):
        for disc, row in zip(system.discs, rows):            # body axes; Makani's thrust_vectoring_pos_b = 0: the origin
            blown += blown_wing_force(row["thrust_N"]*disc.axis, row["axial_freestream_m_s"])
    blown_held = held_loads(body, pose.rotation, pose.rotation@blown, np.zeros(3))
    acting = aero_held if aero_override is None else aero_override
    total = HeldLoads(acting.force_body_N + rotor.force_body_N + blown_held.force_body_N,
                      acting.moment_cm_body_Nm + rotor.moment_cm_body_Nm + blown_held.moment_cm_body_Nm)
    fields = dict(aero_force_body_N=aero_held.force_body_N.tolist(), aero_moment_origin_body_Nm=np.asarray(aero_moment_origin).tolist(),
                  airspeed_m_s=float(condition["airspeed_m_s"]), alpha_rad=float(condition["alpha_rad"]),
                  beta_rad=float(condition["beta_rad"]),
                  rotor_thrust_N=[float(row["thrust_N"]) for row in rows], rotor_power_W=[float(row["power_W"]) for row in rows],
                  rotor_axial_freestream_m_s=[float(row["axial_freestream_m_s"]) for row in rows],
                  rotor_table_saturated=[bool(row["saturated"]) for row in rows],
                  rotor_force_body_N=rotor.force_body_N.tolist(), blown_wing_force_body_N=blown.tolist())
    for key in ("blend_weight",):
        if key in condition:
            fields[key] = float(condition[key])
    if parts:
        return total, coefficient, fields, dict(aero=acting, rotor=rotor, blown=blown_held)
    return total, coefficient, fields


def _vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def run_turnkey(params, api, aero, discs, rotor_inertia_kg_m2, *, blown_wing=False, step_s=1e-3, max_periods=None,
                on_record=None, keep_records=True, tether_ground_contact=False, segment=None, power_system=None,
                wind_shear=None, checkpoint=None, resume=None, record_nodes_every=None, stop_rules=None, floating=None,
                record_terminal=False, tether_events=False, couple_every=None):
    """TurnKey from the perch (see the module text). Returns dict(init, records, end, settings, wall_s).

    ``floating`` (OF-0011 revision 6 addendum 73, F3a; default None: nothing changes): a floating base of
    ``floating.turnkey_base`` under GS02 (``PerchedSystem(floating=...)``). The controller's ground estimate is the
    base's ``message`` (when it gives one), the tether direction reaches GS02 in the vessel frame, the ground height is
    the base's ``ground_z_g_m`` (the controller's must equal it) and every record carries the base's ``record``. Not with
    the tether's ground contact (refused); since OF-0011 revision 7 addendum 6 (P2) with a wind field, whose ground
    height must be the base's (the ground wind sensor stays at its nominal place). With ``checkpoint``/``resume`` (addendum 74) the
    base object is one of the shared objects (not pickled); its states are saved with the system (``base_x``). A base
    with ``period_start``/``period_end`` (the spar, addendum 83) gets them called around each period's integration,
    ``period_end`` with the base's derivative at the period's end; one with ``RESUMABLE = False`` is refused with
    ``checkpoint``/``resume``.

    ``on_record(record)`` is called once per period, after the period's integration (the child process writes the
    records with it); ``keep_records=False`` does not keep them in the result. ``max_periods`` stops after that many
    periods (end reason ``max_periods``; criterion P5 runs 4000). A Python exception ends the run with end reason
    ``failure``, the exception's type and text (K6(e)); the record of the failing period is passed on with
    ``incomplete=True`` when it was already made. ``tether_ground_contact`` (opt-in, default off; full-cycle contract
    revision 7, option 甲): Makani's ground contact of the tether nodes (``tether.variable_line``); with it the settings
    and every record carry the contact quantities, without it settings and records are as before.

    ``segment`` (plan 3.8a, ``vortex-crosswind-segment-contract.zh.md``; default None): a segment-load hook asked once at
    the start of every period, ``segment.loads(k, state, flight_mode, flaps_rad, thrust_coeff_used)`` with the mode the
    controller will run in this period, returning ``(source, held, fields)``: ``source`` 'table', 'shadow' or
    'vortex'; with 'vortex' the HeldLoads ``held`` (about the centre of mass, body axes) take the place of the database
    term; ``fields`` go into the record. With a hook every record also carries R1 (``load_source``, crosswind telemetry,
    the acting held loads by part) and ``segment.settings()`` goes into the settings; a failure ends the run with the
    exception's ``failed_module`` as ``module``. Without a hook nothing changes. A hook may declare (plan 3.8b,
    ``whole-flight-vortex-contract.zh.md``): ``SOURCES`` (the sources it may return; default 'table', 'shadow', 'vortex')
    and ``APPLIED`` (those whose loads take the place of the database term; default 'vortex'); ``BLOWN_WING`` (the blown
    wing setting it needs; the run is refused otherwise); ``after_loads(k, state, fields)``, called right after the
    period's loads with ``period_loads``' fields (the rotor thrusts of the period). A hook without them works as before.

    ``power_system`` (plan 3.9a, ``power-energy-contract.zh.md`` revision 1 with addenda 1-4 and 6; default None): a
    ``power.makani_power_sys.PowerSystemConfig``; Makani's stacked power system and the rotors' equation of motion then
    give the rotor speeds (in place of the rate-limited speed command): each period's discrete step takes the speed
    commands of the controller output two periods back and the torque commands of the one before (Makani's loop), the
    50 states are integrated over the period by RK4 with the rotors' axial freestream held, the thrust and the blown wing
    use the period-start speed, the axial aerodynamic moment the period-mean torque and the reaction moment the mean
    acceleration; the settings and every record carry ``power_*`` quantities. Without it nothing changes.

    ``wind_shear`` (plan 3.9b, ``power-energy-contract.zh.md`` revision 1 addendum 5; default None): a
    ``ground.wind_shear.ShearWind``, Makani's no-turbulence wind with the power-law shear in place of the uniform wind:
    at the start of each period the wind at the body origin (held over the period, as the loads) for the aerodynamics,
    the rotors (thrust, blown wing, the power system's axial freestream) and the estimate's apparent wind and wind
    aloft, with the wind-induced apparent rates added to the body rates for the database's lookup; every tether node at
    its own height at every evaluation; the estimate's ground wind at the ground wind sensor. The settings carry
    ``wind_shear``; the records are as without it (E5a: the exponent 0 through this path gives the same records). With
    a segment hook only if the hook declares ``ACCEPTS_WIND_FIELD`` (OF-0010 revision 19 addendum 6, B1): each period,
    before ``segment.loads``, ``segment.set_period_wind(origin wind, wind rates, the ShearWind)``; the database term of
    the segment periods gets the same wind rates; any other hook with the shear is refused. OF-0010 revision 23 (B2):
    a ``ground.turbulent_wind.TurbulentWind`` (IEC turbulence from a TurbSim file) is taken the same way; when the wind
    object has ``set_time`` it is called with the period's start time before the period's first evaluation, so every
    evaluation of that period (the body origin, the tether nodes, the hook's wake points, the sensor) sees one instant.

    ``checkpoint`` and ``resume`` (OF-0010 revision 12, opt-in; default None: nothing changes): with ``checkpoint`` (an
    object with an integer ``every`` >= 1 and a method ``save(k, make_blob)``) the state at the start of every period k
    that is a multiple of ``every``, other than the run's first, is offered to ``save``: ``make_blob()`` returns it
    (``ground.resume.turnkey_blob``), and an error there or in ``save`` is the checkpoint's to handle; with
    ``resume`` (such a blob) the run initialises as without it, puts the saved state back (the hook's attributes into
    ``segment``) and continues the loop at the saved period. The caller passes the same arguments as the saved run had.
    ``record_nodes_every`` (OF-0010 revision 15, B'; default None: nothing changes): every that many periods the record gets
    ``tether_nodes_Xg_m``, the airborne tether nodes in the frame of ``Xg``, and (OF-0010 revision 19, Studio review STR-04)
    ``tether_ends_Xg_m``, the ground anchor and the bridle point of the same evaluation. With a segment hook that has ``hub_induced``
    (B') the hook's ``loads`` runs before the power system's integration and ``hub_induced(k, state)`` gives the
    velocity induced at each hub (world axes, or None), added to the rotors' apparent wind in ``rotor_freestreams`` and
    ``period_loads``; hooks without it run as before. ``stop_rules`` (OF-0010 revision 19 addendum 22, ST-05; default None:
    nothing changes): a dict of some of ``STOP_RULES`` - the return to perch, the stall, the end, the descent and the hold
    after the gain ramp, in periods - used instead of the module constants and recorded in ``settings["stop_rules"]``.
    """
    from openfawe.flight.makani_full import FullController, gs02_sensors
    from .gs02 import sensor_readings
    rules = stop_rules_of(stop_rules)                   # OF-0010 revision 19 addendum 22 (ST-05)
    if floating is not None and tether_ground_contact:      # OF-0011 rev 7 add 6 (P2): a wind field is now taken with it
        raise ValueError("a floating base is not taken with the tether's ground contact (OF-0011 rev 6 add 73)")
    ground_z = GROUND_Z_G if floating is None else float(floating.ground_z_g_m)
    base_hooks = floating is not None and hasattr(floating, "period_start")      # OF-0011 rev 6 add 83 (the spar)
    if floating is not None and not getattr(floating, "RESUMABLE", True) and (checkpoint is not None or resume is not None):
        raise ValueError(f"the base {floating.name!r} cannot be saved and resumed (its library keeps state)")
    density = float(params.raw["phys_sim"]["air_density"])
    wind_g = makani_wind_g(params)
    wind_world = G_TO_WORLD@wind_g
    if wind_shear is not None:                          # plan 3.9b
        from .wind_shear import ShearWind
        from .turbulent_wind import TurbulentWind
        from .iec_wind import IecEventWind
        if not isinstance(wind_shear, (ShearWind, TurbulentWind, IecEventWind)):  # OF-0010 rev 23 (B2), OF-0011 rev 7 add 6
            raise ValueError("wind_shear must be a ground.wind_shear.ShearWind, a ground.turbulent_wind.TurbulentWind or a "
                             "ground.iec_wind.IecEventWind")
        if segment is not None and not getattr(segment, "ACCEPTS_WIND_FIELD", False):     # OF-0010 revision 19 add. 6 (B1)
            raise ValueError("the wind shear is not defined with a segment hook (its vortex method takes the uniform wind)")
        if wind_shear.ground_z_g_m != ground_z:          # the base's ground with a floating base (OF-0011 rev 7 add 6)
            raise ValueError(f"the wind shear's ground_z {wind_shear.ground_z_g_m} differs from {ground_z}")
        if wind_shear.speed_m_s != float(params.raw["phys_sim"]["wind_speed"]) \
                or wind_shear.direction_rad != float(params.raw["phys_sim"]["wind_direction"]):
            raise ValueError("the wind shear's reference speed or direction differs from the parameters' wind")
        shear_kw = dict(wind_shear=wind_shear)
    else:
        shear_kw = {}
    tables_only = isinstance(blown_wing, str) and blown_wing == "tables"     # OF-0010 revision 11
    if isinstance(blown_wing, str) and not tables_only:
        raise ValueError("blown_wing must be False, True or 'tables'")
    if tables_only:
        if segment is None or not getattr(segment, "BLOWN_WING_TABLES", False):
            raise ValueError("blown_wing='tables' needs a segment hook that declares BLOWN_WING_TABLES")
    elif segment is not None and getattr(segment, "BLOWN_WING", None) is not None and bool(segment.BLOWN_WING) != bool(blown_wing):
        raise ValueError(f"the segment hook needs blown_wing={bool(segment.BLOWN_WING)}")     # plan 3.8b
    if tether_ground_contact:
        system = FlyingSystem(params, discs, rotor_inertia_kg_m2, wind_world, tether_ground_contact=True,
                              ground_z_g_m=GROUND_Z_G, **shear_kw)
    elif floating is not None:                          # addendum 73; OF-0011 rev 7 add 6 (P2): with a wind field
        system = FlyingSystem(params, discs, rotor_inertia_kg_m2, wind_world, floating=floating, **shear_kw)
    else:
        system = FlyingSystem(params, discs, rotor_inertia_kg_m2, wind_world, **shear_kw)
    init, system = perched_initialisation(params, system)
    gs, line = system.gs, system.line
    state, nodes, speeds = init["wing_state"], init["nodes"], init["node_velocities"]
    fc = FullController(api)
    cal = fc.system.winch.velocity_cmd_cal
    if (cal.scale, cal.bias) != (1.0, 0.0):
        raise ValueError("the winch velocity command calibration is expected to be the identity")
    if len(discs) != api.kNumMotors:
        raise ValueError("one disc per motor of the controller is required")
    per = int(round(PERIOD_S/step_s))
    if abs(per*step_s - PERIOD_S) > 1e-15:
        raise ValueError("the step must divide the controller period")
    if couple_every is not None:                        # OF-0011 revision 7 addendum 4: the base's hooks every m steps
        if not base_hooks or isinstance(couple_every, bool) or int(couple_every) != couple_every or couple_every < 1 \
                or per % int(couple_every):
            raise ValueError("couple_every takes a whole number of steps dividing the period, with a base that has period hooks")
        couple_every = int(couple_every)
        hd_dt = getattr(getattr(floating, "hd", None), "dt", None)
        if hd_dt is not None and abs(float(hd_dt) - couple_every*step_s) > 1e-12:
            raise ValueError(f"the base's HydroDyn step {hd_dt} differs from couple_every * step_s")
    names = mode_names(api)
    descend = int(api.kFlightModeHoverDescend)
    control = fc.control
    if float(fc.system.ground_frame.ground_z) != ground_z:
        raise ValueError(f"the controller's ground_z {fc.system.ground_frame.ground_z} differs from {ground_z}")
    settings = dict(blown_wing="tables" if tables_only else bool(blown_wing), step_s=float(step_s), period_s=PERIOD_S, density_kg_m3=density,
                    wind_g_m_s=wind_g.tolist(), rotor_inertia_kg_m2=float(rotor_inertia_kg_m2),
                    controller_rho_kg_m3=float(fc.system.phys.rho), mass_controller_kg=float(fc.system.wing.m),
                    mass_plant_kg=float(system.body.mass_kg), e_stop_throttle=float(control.joystick_control.e_stop_throttle),
                    transout_airspeed_m_s=float(control.crosswind.mode.transout_airspeed),
                    transout_alpha_rad=float(control.crosswind.mode.transout_alpha),
                    perched_wing_pos_p=_vec(control.hover.path.perched_wing_pos_p),
                    hover_limits=dict(max_thrust=float(control.hover.altitude.max_thrust),
                                      min_moment_y=float(control.hover.angles.min_moment.y),
                                      max_moment_y=float(control.hover.angles.max_moment.y),
                                      min_moment_z=float(control.hover.angles.min_moment.z),
                                      max_moment_z=float(control.hover.angles.max_moment.z)),
                    mode_names={str(k): v for k, v in names.items()}, rotor_area_m2=[float(d.area_m2) for d in discs],
                    ground_z_g_m=ground_z)
    if tether_events not in (False, True, "all", "branches", "held", "depth"):
        raise ValueError("tether_events: False, True, 'all', 'branches', 'held' or 'depth'")
    if tether_events:                                   # OF-0011 revision 7 addendum 4: recorded only when given
        settings["tether_events"] = True if tether_events is True else tether_events  # supplements 3-6

    if couple_every is not None:
        settings["couple_every"] = couple_every
    if floating is not None:                            # addendum 73: recorded only when given
        settings["floating"] = dict(base=floating.name, states=int(floating.STATES), x0=[float(v) for v in floating.x0],
                                    ground_z_g_m=ground_z)
    if stop_rules is not None:                          # ST-05: recorded only when given (the default changes nothing)
        settings["stop_rules"] = dict(rules)
    if tether_ground_contact:
        settings["tether_ground_contact"] = dict(ground_height_m=float(line.ground.ground_height_m),
                                                 spring_const_N_m=float(line.ground.spring_const_N_m))
    if power_system is not None:                        # plan 3.9a
        from openfawe.power import makani_power_sys as mps
        if len(power_system.dirs) != len(discs) or any(int(d.direction) != int(dd) for d, dd in zip(discs, power_system.dirs)):
            raise ValueError("the power system's rotor directions differ from the discs'")
        if any(float(i) != float(rotor_inertia_kg_m2) for i in power_system.inertia_kg_m2):
            raise ValueError("the power system's rotor inertias differ from the plant's")
        pss = mps.StackedPowerSystem(power_system)
        pipeline = mps.CommandPipeline()               # the messages of periods k-2 (speeds) and k-1 (torques)
        settings["power_system"] = dict(integration="classical RK4", sub_step_s=mps.SUB_STEP_S, stiff_divisor=mps.STIFF_DIVISOR,
                                        omega_cal=dict(vars(power_system.omega_cal)), torque_cal=dict(vars(power_system.torque_cal)),
                                        power_sys=dict(vars(power_system.power_sys)), sim=dict(vars(power_system.sim)),
                                        motor=dict(vars(power_system.motor)), dirs=list(power_system.dirs),
                                        inertia_kg_m2=list(power_system.inertia_kg_m2),
                                        **(dict(compiled=True) if getattr(power_system, "compiled", False) else {}))
    if segment is not None:
        settings["segment"] = segment.settings()
        segment_sources = tuple(getattr(segment, "SOURCES", ("table", "shadow", "vortex")))       # plan 3.8b
        segment_applied = tuple(getattr(segment, "APPLIED", ("vortex",)))
        segment_after = getattr(segment, "after_loads", None)
        segment_hub = getattr(segment, "hub_induced", None)                                     # OF-0010 revision 15 (B')
    if wind_shear is not None:
        settings["wind_shear"] = wind_shear.settings()
    servo = ServoReference(np.zeros(8))                 # Makani's servo reference models start at a zero shaft angle
    flap_command = np.zeros(8)                          # the zero-initialised command message of the simulator
    speed_command = np.zeros(len(discs))
    omegas = np.zeros(len(discs))
    thrust_coeff = 0.0
    pending = None
    records, end = [], None
    mode_entry, last_mode, gain_done_period, descend_entry = 0, None, None, None
    wall = time.time()
    total = rules["end_periods"] if max_periods is None else min(int(max_periods), rules["end_periods"])
    k, record, emitted = 0, None, True
    start = 0
    if checkpoint is not None or resume is not None:   # OF-0010 revision 12: the objects shared with the caller
        if checkpoint is not None and (isinstance(checkpoint.every, bool) or not isinstance(checkpoint.every, int)
                                       or checkpoint.every < 1):
            raise ValueError("checkpoint.every must be an integer of at least 1")
        from . import resume as RS
        shared = dict(params=params, aero=aero, discs=discs, api=api, fc_system=fc.system, fc_control=fc.control,
                      power_system=power_system, wind_shear=wind_shear)
        if floating is not None:                        # OF-0011 rev 6 add 74: the base object is not pickled (its
            shared["floating"] = floating               # states are the system's base_x)
        if resume is not None:
            start, carried = RS.turnkey_restore(resume, segment, fc, api, shared)
            (state, nodes, speeds, system, servo, flap_command, speed_command, omegas, thrust_coeff, pending, mode_entry,
             last_mode, gain_done_period, descend_entry, records) = (carried[name] for name in RS.LOOP_NAMES[:15])
            gs, line = system.gs, system.line
            if power_system is not None:
                pss, pipeline = carried["pss"], carried["pipeline"]
            k = start

    # OF-0011 rev 7 addendum 4; bound here, after a resume has put its own system in place (bound before the resume it
    # stepped the discarded system: the R1 suite's test_resumed_equals_uninterrupted, 2026-11-15)
    advance = system.step_events if tether_events else system.step
    if tether_events in ("all", "branches", "held", "depth"):    # supplements 3-6
        system.event_kinds = tether_events

    def emit(r):
        nonlocal emitted
        emitted = True                                  # also when on_record raises: the record is not passed on twice
        if keep_records:
            records.append(r)
        if on_record is not None:
            on_record(r)

    try:
        for k in range(start, total):
            if checkpoint is not None and k > start and k % checkpoint.every == 0:     # OF-0010 revision 12
                loop = dict(state=state, nodes=nodes, speeds=speeds, system=system, servo=servo, flap_command=flap_command,
                            speed_command=speed_command, omegas=omegas, thrust_coeff=thrust_coeff, pending=pending,
                            mode_entry=mode_entry, last_mode=last_mode, gain_done_period=gain_done_period,
                            descend_entry=descend_entry, records=records, pss=pss if power_system is not None else None,
                            pipeline=pipeline if power_system is not None else None)
                checkpoint.save(k, lambda: RS.turnkey_blob(k, loop, segment, fc, api, shared))
            record, emitted = None, False
            t = k*PERIOD_S
            if base_hooks:                              # OF-0011 rev 6 add 83: the base's work at the period's start (F2-S's
                floating.period_start(t, system.base_x)  # hold before the sample's integration), before any evaluation
            if wind_shear is None:
                kite_wind_g, estimate_kw, loads_kw = wind_g, {}, {}
            else:                                       # plan 3.9b: the wind at the body origin of the period's start
                if hasattr(wind_shear, "set_time"):     # OF-0010 revision 23 (B2): the field's instant for this period
                    wind_shear.set_time(t)
                start_pose = state.pose(system.body)
                kite_wind_g = wind_shear.origin_wind_g(start_pose)
                system.wind = G_TO_WORLD@kite_wind_g
                estimate_kw = dict(sensor_wind_g=wind_shear.sensor_wind_g())
                loads_kw = dict(wind_rates_body=wind_shear.wind_rates_body(start_pose))
                if segment is not None:                 # OF-0010 revision 19 addendum 6 (B1): the hook's wind
                    segment.set_period_wind(system.wind, loads_kw["wind_rates_body"], wind_shear)
            hub = None
            if segment is not None and segment_hub is not None:   # OF-0010 revision 15 addendum 1 (BP-01): this period's
                flaps = servo.flaps_rad                            # solve first, then the hub inflow of the solved state
                source, segment_held, segment_fields = segment.loads(k, state, fc.flight_mode, flaps, thrust_coeff)
                hub = segment_hub(k, state)                        # None where the rotors see the kinematic inflow
            if power_system is None:
                previous = omegas
                omegas = rate_limited_speeds(speed_command, previous, PERIOD_S)
                omega_rates = (omegas - previous)/PERIOD_S
                axial_torques, power_fields = None, {}
            else:                                       # plan 3.9a: discrete step, then the period's integration
                omegas, omega_rates, axial_torques, power_fields = power_period(
                    pss, pipeline, power_system, t, rotor_freestreams(system, state, hub),
                    mps.TablePower([d.table for d in discs], density), PERIOD_S)
            flaps = servo.flaps_rad
            system.set_period(HeldLoads(), omegas, omega_rates)          # period_loads reads the rotor speeds
            if segment is None:
                held, next_coeff, loads = period_loads(system, aero, state, flaps, thrust_coeff, density, blown_wing=blown_wing,
                                                       **({} if axial_torques is None else dict(rotor_axial_torques=axial_torques)),
                                                       **loads_kw)
            else:                                       # plan 3.8a: the mode at the period's start is fc.flight_mode (V3)
                if segment_hub is None:
                    source, segment_held, segment_fields = segment.loads(k, state, fc.flight_mode, flaps, thrust_coeff)
                if source not in segment_sources or (source in segment_applied and not isinstance(segment_held, HeldLoads)):
                    raise RuntimeError(f"the segment hook returned the source {source!r} with loads {type(segment_held).__name__}")
                held, next_coeff, loads, held_parts = period_loads(
                    system, aero, state, flaps, thrust_coeff, density, blown_wing=blown_wing,
                    aero_override=segment_held if source in segment_applied else None, parts=True,
                    **({} if axial_torques is None else dict(rotor_axial_torques=axial_torques)),
                    **({} if hub is None else dict(hub_induced=hub)),
                    **loads_kw)                        # OF-0010 revision 19 addendum 6 (B1): empty without the shear
                if segment_after is not None:          # plan 3.8b: e.g. the rotor rings of this period's thrust
                    segment_after(k, state, loads)
            system.set_period(held, omegas, omega_rates)
            thrust_used, thrust_coeff = thrust_coeff, next_coeff
            velocity, acceleration, spin, _, _, _, diag = system.rates(state, nodes, speeds, gs.x)
            out = diag["tether"]
            direction_ned = G_TO_WORLD@line.start_direction(out["indices"], nodes, out["start_point"], out["end_point"])
            if floating is not None:                    # addendum 73: TetherDirToAngles takes the vessel frame's direction
                vessel = floating.frame(system.base_x)
                if vessel is not None:
                    direction_ned = vessel.dcm@direction_ned
            ele, azi, yoke, termination = gs.tether_dir_to_angles(direction_ned)
            gs.set_levelwind_engaged(ele)
            gs.set_tether_force(G_TO_WORLD@out["start"])
            readings = sensor_readings(gs, ele, yoke, termination)
            gs.discrete_update(t)
            if pending is not None:
                gs.set_from_command(**pending)
            sensors = gs02_sensors(api, fc.system, drum_angle_rad=readings["drum_angle_rad"],
                                   platform_azimuth_rad=readings["platform_azimuth_rad"], gs_mode=readings["gs_mode"],
                                   transform_stage=readings["transform_stage"], proximity=readings["proximity"],
                                   detwist_angle_rad=readings["detwist_angle_rad"],
                                   levelwind_elevation_rad=readings["levelwind_elevation_rad"],
                                   gsg_yoke_rad=readings["gsg_yoke_rad"], gsg_termination_rad=readings["gsg_termination_rad"],
                                   platform_azimuth_rate_rad_s=gs.platform_azi_vel)
            joystick = joystick_at(api, k, rules["throttle_return_period"])
            est = kite_estimate(api, system, state, acceleration, spin, out["end"], kite_wind_g, **estimate_kw)
            if floating is None:
                est, _, _ = fc.ground_estimate(est, sensors, joystick)
            else:                                       # addendum 73: the base's true state as the ground estimate
                message = floating.message(api, system.base_x, sensors)
                est, _, _ = fc.ground_estimate(est, sensors, joystick,
                                               **({} if message is None else dict(ground_estimate_message=message)))
            ran, output = fc.step(est)
            telemetry = api.GetControlTelemetry().contents
            bad = (_finite_names(output, "output") + _finite_names(telemetry.thrust_moment, "thrust_moment")
                   + _finite_names(telemetry.thrust_moment_avail, "thrust_moment_avail")
                   + _finite_names(telemetry.deltas, "deltas") + _finite_names(telemetry.hover, "hover")
                   + _finite_names(telemetry.trans_in, "trans_in") + _finite_names(telemetry.crosswind, "crosswind"))
            mode = fc.flight_mode
            gates = [int(v) for v in telemetry.flight_mode_gates]
            command_flaps = np.array(list(output.flaps), dtype=float)
            upper = np.array(list(output.motor_speed_upper_limit), dtype=float)
            run_motors, stop_motors = bool(output.run_motors), bool(output.stop_motors)
            pending = dict(gs_azi_target=float(output.gs_azi_cmd.target), gs_azi_dead_zone=float(output.gs_azi_cmd.dead_zone),
                           winch_velocity=float(output.winch_vel_cmd), detwist_position=float(output.detwist_cmd),
                           gs_mode_request=int(output.gs_mode_request), gs_unpause_transform=bool(output.gs_unpause_transform))
            # The flap command of the period before moves the servos over this period; this period's commands act next.
            servo.step(flap_command, PERIOD_S)
            flap_command = command_flaps
            speed_command = upper if (run_motors and not stop_motors) else np.zeros(len(discs))
            if power_system is not None:                # this period's packets act from the next period (torque) and the one after (speeds)
                controller_lower, controller_torque = list(output.motor_speed_lower_limit), list(output.motor_torque)
                pushed = mps.convert_rotors(upper, controller_lower, controller_torque, power_system, run_motors and not stop_motors)
                pipeline.push(pushed)
            pose = state.pose(system.body)
            active = nodes[out["indices"].start:] if out["indices"].active else np.zeros((0, 3))
            tension_segments = np.asarray(out["tension"], dtype=float)
            hover = telemetry.hover
            # flight_mode: the planner's mode after this step (the mode of the next period; P2, P3, K6);
            # controller_flight_mode: ControlTelemetry.flight_mode, the mode the controller ran in this step, which
            # PlannerStep writes before it switches (control_planner.c:546-568) and Makani's scores select by (P4, P7).
            record = dict(
                k=k, t=t, ran=bool(ran), flight_mode=mode, controller_flight_mode=int(telemetry.flight_mode), gates=gates,
                gs_mode=int(gs.mode.val),
                transform_stage=int(gs.transform_stage.val), gs_mode_request=pending["gs_mode_request"],
                winch_velocity_cmd=pending["winch_velocity"], free_length=float(out["free_length"]),
                start_index=int(out["indices"].start), tension_bridle_N=float(np.linalg.norm(out["end"])),
                tension_anchor_N=float(np.linalg.norm(out["start"])), drum_angle=gs.drum_angle, drum_omega=gs.drum_omega,
                platform_azi=gs.platform_azi, detwist_angle=gs.detwist_angle, levelwind_engaged=gs.levelwind_engaged,
                tether_elevation=float(ele), proximity=bool(gs.prox_sensor_active.val),
                rotor_omega_rad_s=omegas.tolist(), rotor_command_rad_s=upper.tolist(), run_motors=run_motors,
                stop_motors=stop_motors, tether_release=bool(output.tether_release),
                operator_flags=[bool(est.tether_released), bool(est.force_high_tension), bool(est.force_reel),
                                bool(est.gs_unpause_transform), bool(est.force_detwist_turn_once), bool(joystick.release)],
                flaps_acting_rad=flaps.tolist(), flaps_command_rad=command_flaps.tolist(), thrust_coeff_used=float(thrust_used),
                position_cm_m=state.position_cm_m.tolist(), velocity_cm_m_s=state.velocity_cm_m_s.tolist(),
                rotation=state.rotation.tolist(), angular_velocity_body_rad_s=state.angular_velocity_body_rad_s.tolist(),
                Xg=(G_TO_WORLD@pose.position_m).tolist(), origin_height_m=float(pose.position_m[2] + ground_z),
                min_node_height_m=float(np.min(active[:, 2]) + ground_z) if len(active) else None,
                slack_segments=int(np.sum(tension_segments <= 0.0)), segments=int(len(tension_segments)),
                **({} if system.event_kinds == "first" else dict(events_located=int(system.events_located))),  # supplements 3, 4
                contacts=[bool(c[1]) for c in diag["contacts"]], held_force_body_N=held.force_body_N.tolist(),
                held_moment_cm_body_Nm=held.moment_cm_body_Nm.tolist(),
                gain_ramp_scale=float(hover.gain_ramp_scale), tension_cmd_N=float(hover.tension_cmd),
                wing_pos_g_cmd=_vec(hover.wing_pos_g_cmd),
                thrust_moment=[float(telemetry.thrust_moment.thrust)] + _vec(telemetry.thrust_moment.moment),
                thrust_moment_avail=[float(telemetry.thrust_moment_avail.thrust)] + _vec(telemetry.thrust_moment_avail.moment),
                est_airspeed_m_s=float(est.apparent_wind.sph_f.v), est_alpha_rad=float(est.apparent_wind.sph.alpha),
                est_tension_N=float(est.tether_force_b.sph.tension), payout_m=float(est.winch.payout),
                joystick_throttle=float(joystick.throttle), non_finite=bad, **loads, **power_fields)
            if floating is not None:                    # addendum 73: the base at the period's start
                record.update(floating.record(system.base_x))
                if record_terminal:                     # OF-0011 revision 7 addendum 3 (C1-V3): the tether's start
                    record.update(terminal_force_Xg_N=(G_TO_WORLD@np.asarray(out["start"], dtype=float)).tolist(),
                                  terminal_point_Xg_m=(G_TO_WORLD@np.asarray(out["start_point"], dtype=float)).tolist(),
                                  terminal_velocity_Xg_m_s=(G_TO_WORLD@np.asarray(out["start_velocity"], dtype=float)).tolist())
            if record_nodes_every and k % int(record_nodes_every) == 0:   # OF-0010 revision 15 (B'): the airborne tether
                record["tether_nodes_Xg_m"] = (G_TO_WORLD@active.T).T.tolist()   # nodes, in the frame of Xg
                record["tether_ends_Xg_m"] = [(G_TO_WORLD@np.asarray(out["start_point"], dtype=float)).tolist(),
                                              (G_TO_WORLD@np.asarray(out["end_point"], dtype=float)).tolist()]
            if power_system is not None:                # the crosswind loop angle of Makani's scoring (R4); the read-back (addendum 3 item 7)
                record.update(power_cw_loop_angle=float(telemetry.crosswind.loop_angle), power_state_end=[float(v) for v in pss.x],
                              power_controller_lower_rad_s=[float(v) for v in controller_lower],
                              power_controller_torque_Nm=[float(v) for v in controller_torque], power_message_pushed=pushed)
            if tether_ground_contact:                   # nodes at the start of the period: active and staging (Makani's set)
                idx = out["indices"]
                first = max(idx.start - 1, 0)
                penetration = line.ground.ground_height_m - nodes[:, 2]
                considered = penetration[first:]
                active_pen = penetration[idx.start:] if idx.active else np.zeros(0)
                deepest = first + int(np.argmax(considered)) if len(considered) else None
                record.update(ground_contact_nodes=int(np.sum(considered > 0.0)),
                              ground_contact_nodes_active=int(np.sum(active_pen > 0.0)),
                              max_penetration_m=float(max(np.max(considered), 0.0)) if len(considered) else 0.0,
                              max_penetration_active_m=float(max(np.max(active_pen), 0.0)) if len(active_pen) else 0.0,
                              max_penetration_staging_m=float(max(penetration[idx.start - 1], 0.0)) if idx.start >= 1 else 0.0,
                              deepest_node_index=deepest,
                              deepest_node_xy_m=None if deepest is None else (G_TO_WORLD@nodes[deepest])[:2].tolist(),
                              ground_force_N=float(np.sum(out["ground_force"][:, 2])))
            if segment is not None:                     # R1 of plan 3.8a (both modes of the hook, every period)
                clash = set(segment_fields) & (set(record) | set(R1_KEYS))
                if clash:
                    raise RuntimeError(f"the segment hook would overwrite record fields: {sorted(clash)}")
                cw = telemetry.crosswind
                record.update(load_source=source, flight_mode_time=float(telemetry.flight_mode_time),
                              cw_loop_angle=float(cw.loop_angle), cw_current_pos_cw=[float(cw.current_pos_cw.x), float(cw.current_pos_cw.y)],
                              cw_path_radius_target=float(cw.path_radius_target), cw_airspeed_cmd=float(cw.airspeed_cmd),
                              cw_alpha_cmd=float(cw.alpha_cmd), cw_path_type=int(cw.path_type), cw_int_crosstrack=float(cw.int_crosstrack),
                              est_airspeed_sph_m_s=float(est.apparent_wind.sph.v), origin_g_m=(G_TO_WORLD@pose.position_m).tolist(),
                              held_aero_force_body_N=held_parts["aero"].force_body_N.tolist(),
                              held_aero_moment_cm_body_Nm=held_parts["aero"].moment_cm_body_Nm.tolist(),
                              rotor_moment_cm_body_Nm=held_parts["rotor"].moment_cm_body_Nm.tolist(), **segment_fields)
            if bad:
                end = dict(reason="failure", type="non-finite", period=k, detail=bad[:12])
                emit(record)
                break
            if mode != last_mode:
                mode_entry, last_mode = k, mode
                if mode == descend:
                    descend_entry = k
            if mode == descend and gain_done_period is None and record["gain_ramp_scale"] < GAIN_DONE:
                gain_done_period = k
            if couple_every is None:
                for _ in range(per):
                    state, nodes, speeds = advance(state, nodes, speeds, step_s)
                if base_hooks:                          # the base's derivative at the period's end (F2-S's d1)
                    floating.period_end(t, PERIOD_S, system.base_x, system.rates(state, nodes, speeds, gs.x)[-1]["base_rate"])
            else:                                       # addendum 4: the base's hooks every couple_every steps
                for j in range(per//couple_every):
                    tj = t + j*couple_every*step_s
                    if j:
                        floating.period_start(tj, system.base_x)
                    for _ in range(couple_every):
                        state, nodes, speeds = advance(state, nodes, speeds, step_s)
                    floating.period_end(tj, couple_every*step_s, system.base_x,
                                        system.rates(state, nodes, speeds, gs.x)[-1]["base_rate"])
            if floating is not None and getattr(floating, "balance", False):   # OF-0011 rev 7 add 10 (C1-V3)
                record["base_balance"] = floating.balance_fields()
            record["start_index_end"] = int(line.indices(gs.free_length()).start)
            record["free_length_end"] = float(gs.free_length())
            emit(record)
            if not (np.isfinite(state.position_cm_m).all() and np.isfinite(state.velocity_cm_m_s).all()
                    and np.isfinite(state.rotation).all() and np.isfinite(state.angular_velocity_body_rad_s).all()):
                end = dict(reason="failure", type="non-finite", period=k, detail=["wing state"])
                break
            if gain_done_period is not None and k - gain_done_period >= rules["gain_done_hold_periods"]:
                end = dict(reason="gain_ramp_done", period=k, gain_done_period=gain_done_period)
                break
            if descend_entry is not None and k - descend_entry >= rules["descend_periods"]:
                end = dict(reason="descend_timeout", period=k)
                break
            if k - mode_entry >= rules["stall_periods"]:
                end = dict(reason="stall", period=k, mode=names.get(mode, mode), gates=gates)
                break
        else:
            end = dict(reason="max_periods" if max_periods is not None and total < rules["end_periods"] else "time_limit", period=k)
    except Exception as error:                                   # noqa: BLE001  (K6(e): a Python exception is a failure)
        end = dict(reason="failure", type=type(error).__name__, period=k, detail=str(error)[:400])
        if segment is not None or power_system is not None:      # plan 3.8a V6, 3.9a: the module that failed
            end["module"] = getattr(error, "failed_module", None)
        if segment is not None:
            failure = getattr(error, "zero_seed_failure", None)  # the first, zero-seeded attempt of a retried solve
            if failure is not None:
                end["zero_seed_failure"] = {key: (value if isinstance(value, (int, float, str, type(None))) else str(value))
                                            for key, value in dict(failure).items()} if isinstance(failure, dict) else str(failure)
            first = getattr(error, "first_seed_failure", None)   # OF-0010 P5a: the first, mixed-seeded attempt
            if first is not None:
                end["first_seed_failure"] = {key: (value if isinstance(value, (int, float, str, type(None))) else str(value))
                                             for key, value in dict(first).items()} if isinstance(first, dict) else str(first)
        if record is not None and not emitted:
            record["incomplete"] = True
            emit(record)
    return dict(init={key: value for key, value in init.items() if key not in ("nodes", "node_velocities", "wing_state")},
                records=records, end=end, settings=settings, wall_s=time.time() - wall)
