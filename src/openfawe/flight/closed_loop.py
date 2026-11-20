"""The Makani crosswind controller on the OpenFAWE plant: hooks of ``run_free`` (G5 step 2, closed-loop contract).

``CrosswindLoop`` holds the controller's state (the structures of the controller library, passed in as its Python
binding) and stands between the plant and the controller at every aerodynamic step, which is also the controller's
period (0.01 s):

1. ``before_aero`` (called by ``run_free`` before the lifting surfaces are solved): the plant's state is written
   into a ``StateEstimate`` as truth (``flight.makani_interface``), ``CrosswindStep`` and Makani's output stage
   ``ControlOutputStep`` are called once, the flap commands go into the servo reference model and the rotor speed
   commands through the rate limit of the speed command.
2. ``extra_loads`` (called after the lifting surfaces are solved): the loads the lifting-surface solution does not
   hold are returned as a force and a moment about the body origin: the control-surface increments of Makani's
   low-incidence tables (``aero.makani_flap_increments``, a declared stand-in; left out with
   ``flap_increments=None``, for a driver whose aerodynamic model holds the control surfaces itself, as
   ``flight.reference_flight`` with ``aero.makani_database``) and, in the variant without rings, the rotor forces
   and axis torques from the rotor tables at the kinematic inflow of each hub. With a positive
   ``parasitic_drag_coefficient`` a drag force of that coefficient on the reference area is added at the body origin,
   against the origin's velocity through the air. The constant is borrowed: Makani's simulator adds 0.075 to the drag
   coefficient of its low-incidence database, "to account for parasitic drag components on the kite. Includes bridle
   drag. Value was determined from analysis of the glide portion of flight RPX-07" (``aero_sim.py``); the simulator
   weights it, with the rest of that database, by one minus the blend weight above an incidence of 5 deg, which is not
   done here (drag contract, M2 and revision 1). It must not be combined with an aerodynamic model that already holds
   that offset (``aero.makani_database`` with its default settings does). The default is zero.
3. ``external`` (the plant's per-stage callback): the gyroscopic moment ``L x omega`` and the reaction moment
   ``-I dOmega/dt`` of the rotors.

Timing. The flap angles that act in step n are the state of the servo reference model at ``t_n``; the command
computed at ``t_n`` moves the model over the step and acts from step n+1 on. The rotor speeds held over step n are
the command of ``t_n``, moved from the speeds of step n-1 by at most the rate limit; the first command is taken as it
is (no rate limit, no reaction moment: there is no speed before it). The acceleration in the
estimate is the plant's acceleration at ``t_n`` under the loads held over the step before (``run_free`` supplies
it); in the first step of the controller this is the acceleration of the pinned reference motion.

Before ``start_step`` the controller is not called and nothing is added (the lead-in of a run is pinned to a
reference motion; the increments shed no vorticity, so the wake does not depend on them). With rings the lead-in
releases them at the idle speed, and the speed jumps to the first command at the hand-over.

The entry is not one Makani makes. ``CrosswindInit`` is written for the entry from the transition-in: it sets the
integrators of the inner loops so that the first surface commands match ``flaps_z1`` under a zero tether-roll and
sideslip command and zero rate commands (``crosswind_inner.c``, ``InitLateralIntegrators``), and it reads the aileron
from the absolute angles of A1 and A8, whose crosswind offsets differ. Entered on the loop, with the nominal roll and
rates, the first commands therefore do not equal ``flaps_z1`` whichever mode is given as the one before: a start
transient of the controller's own states is part of every run made with this class (review finding CL-01).

The control-surface increments are applied in full at every incidence. Makani's simulator weights its low-incidence
database, these terms included, by one minus a blend weight that leaves zero above an incidence of 5 deg (0.1 at
6 deg, 0.5 at 10 deg; ``sim/physics/aero.cc``, ``aero_sim.py``); that weight is not applied here (review finding CL-03).

Not here: the planner and the other flight modes, the estimator (and its filters), the winch, electrical power,
rotor rings (those go through ``PrescribedRotors`` with ``rotor_speeds`` as its schedule). Nothing in the package
imports this module.
"""

import ctypes
import math

import numpy as np

from .free_flight import gyroscopic_moment
from .makani_interface import (FLAPS, ServoReference, apparent_wind_sph, makani_truth, origin_acceleration,
                               rate_limited_speeds, tether_force_sph, wind_direction)
from .rigid_body import GRAVITY_M_S2


def _vec(api, values):
    return api.Vec3(float(values[0]), float(values[1]), float(values[2]))


def _set_mat(target, values):
    for i in range(3):
        for j in range(3):
            target.d[i][j] = float(values[i][j])


def _finite(value, prefix=""):
    """Names of the floating-point leaves of a ctypes value that are not finite."""
    bad = []
    if isinstance(value, (ctypes.Structure, ctypes.Union)):
        for name, *_ in value._fields_:
            bad += _finite(getattr(value, name), f"{prefix}.{name}" if prefix else name)
    elif isinstance(value, ctypes.Array):
        for index in range(len(value)):
            bad += _finite(value[index], f"{prefix}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        bad.append(prefix)
    return bad


def _numbers(value, prefix=""):
    """``(name, number)`` of every float in a nested dict or list of the log."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _numbers(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _numbers(item, f"{prefix}[{index}]")
    elif isinstance(value, float):
        yield prefix, value


class CrosswindLoop:
    """Controller, actuators and the added loads of one closed-loop run; see the module text."""

    def __init__(self, api, body, wind_m_s, *, start_step, step_s, density_kg_m3, flap_increments, tether=None,
                 discs=None, rotor_inertia_kg_m2=None, rotor_tables=True, throttle=0.81, last_flight_mode=None,
                 gravity_m_s2=GRAVITY_M_S2, parasitic_drag_coefficient=0.0, reference_area_m2=32.9):
        self.api, self.body = api, body
        self.wind = np.array(wind_m_s, dtype=float).reshape(3)
        self.start_step, self.step_s, self.density = int(start_step), float(step_s), float(density_kg_m3)
        self.flap_increments, self.tether = flap_increments, tether
        self.discs = None if discs is None else tuple(discs)
        self.rotor_inertia = None if rotor_inertia_kg_m2 is None else float(rotor_inertia_kg_m2)
        self.rotor_tables = bool(rotor_tables)                 # False: the rotor loads come from rings (PrescribedRotors)
        self.throttle = float(throttle)
        self.gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
        # A constant drag coefficient on the reference area: a force against the velocity of the body origin through
        # the air, applied at the body origin. Borrowed: Makani's simulator adds 0.075 to the drag coefficient of its
        # database for "parasitic drag components on the kite", bridles included (aero_sim.py, coeff_offsets). Zero
        # (the default) adds nothing and leaves every record as it was.
        self.parasitic_drag = float(parasitic_drag_coefficient)
        self.reference_area = float(reference_area_m2)
        if not (math.isfinite(self.parasitic_drag) and self.parasitic_drag >= 0.0 and math.isfinite(self.reference_area)
                and self.reference_area > 0.0):
            raise ValueError("the parasitic drag coefficient must be finite and not negative, the reference area positive")
        system = api.GetSystemParams().contents
        ts = ctypes.cast(api.GlobalSystemParams.in_dll(api._lib, "g_sys").ts, ctypes.POINTER(ctypes.c_double)).contents.value
        if ts != self.step_s:
            raise ValueError(f"the controller's period is {ts!r} s; the aerodynamic step must equal it")
        if self.discs is not None and len(self.discs) != api.kNumMotors:
            raise ValueError("one disc per motor of the controller is required")
        self.azi_ref_offset = float(system.ground_station.azi_ref_offset)
        self.rho_controller = float(system.phys.rho)
        # The controller's own constants beside those of the plant: recorded with a run, so that a mismatch is seen.
        self.consistency = dict(density_plant_kg_m3=self.density, density_controller_kg_m3=self.rho_controller,
                                mass_plant_kg=float(body.mass_kg), mass_controller_kg=float(system.wing.m),
                                tether_length_controller_m=float(system.tether.length),
                                tether_length_plant_m=None if tether is None else float(tether.line.p.length_m),
                                motors_controller=int(api.kNumMotors), discs=None if self.discs is None else len(self.discs))
        self.control = api.GetControlParams().contents
        self.last_flight_mode = api.kFlightModeCrosswindPrepTransOut if last_flight_mode is None else int(last_flight_mode)
        self.active, self.failed = False, None
        self.state, self.output_state, self.hover = api.CrosswindState(), api.ControlOutputState(), api.HoverState()
        self.servo = None
        self.flaps_now = np.zeros(len(FLAPS))                   # the angles acting in the current step
        self.omegas = self.previous_omegas = None
        self.omega_rates = None
        self.thrust_coeff = 0.0                                # mean rotor thrust coefficient of the step before
        self.log = []

    # ----- the estimate ---------------------------------------------------------------------------------------------
    def estimate(self, truth):
        """A ``StateEstimate`` filled with the truth (closed-loop contract, D5); fields not named stay zero."""
        api, est = self.api, self.api.StateEstimate()
        est.stacking_state = api.kStackingStateNormal
        _set_mat(est.dcm_g2b, truth.dcm_g2b)
        est.pqr, est.pqr_f = _vec(api, truth.pqr), _vec(api, truth.pqr)
        est.Xg, est.Vg, est.Vg_f, est.Ag = (_vec(api, v) for v in (truth.Xg, truth.Vg, truth.Vg, truth.Ag))
        est.Vb, est.Vb_f, est.Ab_f = _vec(api, truth.Vb), _vec(api, truth.Vb), _vec(api, truth.Ab)
        gravity_g = np.array([0.0, 0.0, -self.gravity[2]])        # g_g of the plant: z of g points down
        est.acc_norm_f = float(np.linalg.norm(truth.Ag-gravity_g))
        est.gps_active = True
        est.rho = self.rho_controller
        wind_b = _vec(api, truth.apparent_wind_b)
        est.apparent_wind.solution_type = api.kApparentWindSolutionTypePitot
        est.apparent_wind.vector = wind_b
        api.ApparentWindCartToSph(ctypes.byref(wind_b), ctypes.byref(est.apparent_wind.sph))
        api.ApparentWindCartToSph(ctypes.byref(wind_b), ctypes.byref(est.apparent_wind.sph_f))
        est.joystick.valid = True
        est.joystick.throttle_f = est.joystick.data.throttle = self.throttle
        force_b = _vec(api, truth.tether_force_b)
        est.tether_force_b.valid = True
        est.tether_force_b.vector = est.tether_force_b.vector_f = force_b
        api.TetherForceCartToSph(ctypes.byref(force_b), ctypes.byref(est.tether_force_b.sph))
        est.tether_force_b.tension_f = est.tether_force_b.sph.tension
        direction = wind_direction(truth.wind_g, self.azi_ref_offset)
        for target in (est.wind_g, est.wind_aloft_g):
            target.valid = True
            target.vector, target.vector_f, target.vector_f_slow = (_vec(api, truth.wind_g) for _ in range(3))
            target.dir_f = target.dir_f_playbook = direction
            target.speed_f = target.speed_f_playbook = float(np.linalg.norm(truth.wind_g))
        _set_mat(est.vessel.dcm_g2v, np.eye(3))
        est.experiment.active_type = est.experiment.staged_type = api.kExperimentTypeNoTest
        return est

    def truth(self, state, nodes, node_velocities, acceleration):
        tether_force = None
        if self.tether is not None:
            tether_force = self.tether.loads(self.body, state, nodes, node_velocities, self.wind)[2]
        origin = origin_acceleration(self.body, state, *acceleration)
        return makani_truth(self.body, state, self.wind, acceleration_origin_m_s2=origin, tether_force_inertial_N=tether_force)

    # ----- hooks of run_free ----------------------------------------------------------------------------------------
    def before_aero(self, index, time_s, state, nodes, node_velocities, acceleration):
        if index < self.start_step:
            return
        if acceleration is None:
            raise ValueError("the controller cannot start at the first step of a run: it needs the acceleration of a step before")
        api, ref = self.api, ctypes.byref
        truth = self.truth(state, nodes, node_velocities, acceleration)
        est = self.estimate(truth)
        params = self.control
        if not self.active:
            offsets = np.array(list(params.crosswind.output.flap_offsets), dtype=float)
            self.servo = ServoReference(offsets)
            start = (ctypes.c_double*api.kNumFlaps)(*self.servo.flaps_rad)
            # Detwist angle, revolution count and accumulated loop angle as Makani's RunController passes them the first
            # time the crosswind controller is entered (control_system.c lines 180-181 and 211-219): zeros.
            api.CrosswindInit(ref(est), start, 0.0, 0, 0.0, ref(params.crosswind), ref(self.hover), ref(self.state))
            api.ControlOutputInit(ref(params.control_output), ref(self.output_state))
            for k in range(api.kNumFlaps):                   # the output stage's rate limiter starts from the flaps as they are
                self.output_state.flaps_z1[k] = start[k]
            self.active = True
            self.first_step = index
        status = api.FlightStatus()
        # The mode before: not trans-in. After trans-in the controller fades its tether-roll command in from zero
        # over 2 s (crosswind_curvature.c lines 200-205), which suits a kite that arrives with no roll; this run is
        # entered on the loop with the nominal roll. The other mode from which Makani enters this one is the
        # preparation of the trans-out; ``last_flight_mode`` is read nowhere else in the crosswind controller.
        status.flight_mode, status.last_flight_mode = api.kFlightModeCrosswindNormal, self.last_flight_mode
        status.flight_mode_first_entry = index == self.first_step
        status.flight_mode_time = (index-self.first_step)*self.step_s
        raw, out = api.ControlOutput(), api.ControlOutput()
        api.CrosswindStep(ref(status), ref(est), ref(params.crosswind), ref(self.state), ref(raw))
        api.ControlOutputStep(ref(raw), ref(est.joystick), ref(est.perch_azi), ref(params.control_output), ref(self.output_state), ref(out))
        # Both structures are judged: Makani's Saturate is fmin(fmax(x, low), high), which turns a NaN into the lower
        # limit, so the output stage can hide a non-finite command of CrosswindStep (review finding CL-02).
        bad = [f"raw.{name}" for name in _finite(raw)]+_finite(out)
        self.flaps_now = self.servo.flaps_rad
        command_flaps = np.array(list(out.flaps), dtype=float)
        upper, lower = np.array(list(out.motor_speed_upper_limit)), np.array(list(out.motor_speed_lower_limit))
        telemetry = api.GetCrosswindTelemetry().contents
        entry = dict(step=int(index), time_s=float(time_s), non_finite_outputs=bad,
                     flaps_acting_rad=self.flaps_now.tolist(), flaps_command_rad=command_flaps.tolist(),
                     flaps_raw_command_rad=[float(v) for v in raw.flaps],
                     motor_speed_upper_rad_s=upper.tolist(), motor_speed_lower_rad_s=lower.tolist(),
                     motor_torque_Nm=[float(v) for v in out.motor_torque],
                     loop_angle_rad=float(telemetry.loop_angle), path_center_g=[telemetry.path_center_g.x, telemetry.path_center_g.y, telemetry.path_center_g.z],
                     path_radius_target_m=float(telemetry.path_radius_target),
                     alpha_cmd_rad=float(telemetry.alpha_cmd), beta_cmd_rad=float(telemetry.beta_cmd),
                     tether_roll_cmd_rad=float(telemetry.tether_roll_cmd), airspeed_cmd_m_s=float(telemetry.airspeed_cmd),
                     k_aero_cmd=float(telemetry.k_aero_cmd), k_geom_cmd=float(telemetry.k_geom_cmd), k_geom_curr=float(telemetry.k_geom_curr),
                     current_pos_cw=[telemetry.current_pos_cw.x, telemetry.current_pos_cw.y], target_pos_cw=[telemetry.target_pos_cw.x, telemetry.target_pos_cw.y],
                     thrust_ff_N=float(telemetry.thrust_ff), thrust_fb_N=float(telemetry.thrust_fb),
                     integrators=dict(tether_roll_error=float(telemetry.int_tether_roll_error), beta_error=float(telemetry.int_beta_error),
                                      aileron=float(telemetry.int_aileron), elevator=float(telemetry.int_elevator),
                                      rudder=float(telemetry.int_rudder), thrust=float(telemetry.int_thrust),
                                      crosstrack=float(telemetry.int_crosstrack)),
                     pqr_cmd=[telemetry.pqr_cmd.x, telemetry.pqr_cmd.y, telemetry.pqr_cmd.z], k_aero_curr=float(telemetry.k_aero_curr),
                     estimate=dict(Xg=truth.Xg.tolist(), Vg=truth.Vg.tolist(), pqr=truth.pqr.tolist(), Ag=truth.Ag.tolist(),
                                   airspeed_m_s=float(est.apparent_wind.sph.v), alpha_rad=float(est.apparent_wind.sph.alpha),
                                   beta_rad=float(est.apparent_wind.sph.beta), tether_tension_N=float(est.tether_force_b.sph.tension),
                                   tether_roll_rad=float(est.tether_force_b.sph.roll), tether_pitch_rad=float(est.tether_force_b.sph.pitch)))
        entry["non_finite_telemetry"] = sorted(name for name, value in _numbers(entry) if not math.isfinite(value))   # report only
        if bad:
            self.failed = f"non-finite controller outputs at step {index}: {bad[:6]}"
            self.log.append(entry)
            raise FloatingPointError(self.failed)
        self.servo.step(command_flaps, self.step_s)
        entry.update(servo_rate_limited=self.servo.rate_limited.tolist(), servo_at_limit=self.servo.at_limit.tolist())
        if self.discs is not None:
            self.previous_omegas = self.omegas
            self.omegas = rate_limited_speeds(upper, self.previous_omegas, self.step_s)
            self.omega_rates = (np.zeros(len(self.discs)) if self.previous_omegas is None
                                else (self.omegas-self.previous_omegas)/self.step_s)
            entry.update(rotor_omega_rad_s=self.omegas.tolist(), rotor_speed_rate_limited=bool(np.any(self.omegas != upper)),
                         motor_limits_differ=bool(np.any(upper != lower)))
        self.log.append(entry)

    def extra_loads(self, pose, record):
        """Force and moment about the body origin (inertial) that the lifting-surface solution does not hold, and record fields."""
        zero = np.zeros(3)
        if not self.active:
            return zero, zero, dict(controller_active=False)
        fields = dict(controller_active=True)
        force, moment = zero.copy(), zero.copy()
        thrust_coeff = self.thrust_coeff
        if self.discs is not None and self.rotor_tables:
            rows, placed = [], [disc.placed(pose.rotation, pose.position_m) for disc in self.discs]
            for disc, omega in zip(placed, self.omegas):
                apparent = self.wind-pose.point_velocity(disc.centre_m[None, :])[0]
                rows.append(disc.loads(float(omega), apparent, self.density))
                force += rows[-1]["force_N"]
                moment += np.cross(disc.centre_m-pose.position_m, rows[-1]["force_N"])+rows[-1]["moment_Nm"]
            # Makani's RotorThrustCoeff (rotor.cc lines 76-87), mean over the rotors; it acts on the tail in the next step.
            self.thrust_coeff = float(np.mean([row["thrust_N"]/(0.5*self.density*disc.area_m2*max(abs(row["axial_freestream_m_s"]), 30.0)**2)
                                               for row, disc in zip(rows, placed)]))
            fields.update(rotor_force_inertial_N=force.tolist(), rotor_moment_origin_inertial_Nm=moment.tolist(),
                          rotor_thrust_N=[float(row["thrust_N"]) for row in rows], rotor_power_W=[float(row["power_W"]) for row in rows],
                          rotor_shaft_power_W=float(sum(row["power_W"] for row in rows)),
                          rotor_axial_freestream_m_s=[float(row["axial_freestream_m_s"]) for row in rows],
                          rotor_omega_rad_s=self.omegas.tolist(), rotor_table_saturated=[bool(row["saturated"]) for row in rows],
                          rotor_thrust_coeff=self.thrust_coeff, rotor_rings=False)
        elif self.discs is not None and "rotor_thrust_N" not in record:
            raise ValueError("discs were given without rotor tables, and the step has no rotor record: the rotors would carry no load")
        elif self.discs is not None:                                      # rings: the rotor record of this step (rotor_step)
            self.thrust_coeff = float(np.mean([thrust/(0.5*self.density*disc.area_m2*max(abs(freestream), 30.0)**2)
                                               for thrust, freestream, disc in zip(record["rotor_thrust_N"], record["rotor_axial_freestream_m_s"],
                                                                                   self.discs)]))
            fields.update(rotor_thrust_coeff=self.thrust_coeff, rotor_rings=True)
        if self.parasitic_drag > 0.0:
            relative = pose.velocity_m_s-self.wind                         # velocity of the body origin through the air
            drag = -self.parasitic_drag*0.5*self.density*self.reference_area*float(np.linalg.norm(relative))*relative
            force = force+drag                                             # at the body origin: no moment about it
            fields.update(parasitic_drag_inertial_N=drag.tolist(), parasitic_drag_coefficient=self.parasitic_drag)
        if self.flap_increments is None:                # the control surfaces are in the aerodynamic model of the driver
            return force, moment, fields
        through_air = pose.rotation.T@(pose.velocity_m_s-self.wind)
        flap_force, flap_moment, condition = self.flap_increments.loads(through_air, self.density, self.flaps_now, thrust_coeff)
        flap_force, flap_moment = pose.rotation@flap_force, pose.rotation@flap_moment
        fields.update(flap_force_inertial_N=flap_force.tolist(), flap_moment_origin_inertial_Nm=flap_moment.tolist(),
                      flaps_acting_rad=self.flaps_now.tolist(), flap_condition=condition, flap_thrust_coeff_used=thrust_coeff)
        return force+flap_force, moment+flap_moment, fields

    def external(self, time_s, stage, model):
        """Gyroscopic and reaction moments of the rotors on the body (inertial components); no force."""
        zero = np.zeros(3)
        if not self.active or self.discs is None or self.rotor_inertia is None:
            return zero, zero
        sense = np.array([disc.direction for disc in self.discs], dtype=float)
        axes = np.array([disc.axis for disc in self.discs], dtype=float)
        momentum = (self.rotor_inertia*self.omegas*sense)@axes
        reaction = stage.rotation@((-self.rotor_inertia*self.omega_rates*sense)@axes)
        return zero, gyroscopic_moment(stage, momentum)+reaction

    def rotor_speeds(self, time_s):
        """Schedule for ``PrescribedRotors`` (rings): the speeds held over the current step; idle before the start."""
        if self.omegas is None:
            return np.full(len(self.discs), float(self.control.rotor_control.idle_speed))
        return self.omegas


class Externals:
    """Sum of external-load callbacks; ``begin_step`` is passed on to those that have it."""

    def __init__(self, *parts):
        self.parts = [part for part in parts if part is not None]

    def begin_step(self, time_s):
        for part in self.parts:
            if hasattr(part, "begin_step"):
                part.begin_step(time_s)

    def __call__(self, time_s, state, model):
        force, moment = np.zeros(3), np.zeros(3)
        for part in self.parts:
            f, m = part(time_s, state, model)
            force, moment = force+np.asarray(f, dtype=float), moment+np.asarray(m, dtype=float)
        return force, moment
