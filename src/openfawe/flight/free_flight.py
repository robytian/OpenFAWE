"""Free flight with frozen controls: vortex-wake aerodynamics, rotors and tether in the loop (G5 plant contract, step 1b part two).

Contract: ``docs/tasks/OF-0008-g5control20261006/plant-contract.zh.md``, section "第 1b 步（其二）"; coupling
scheme: decision 0030 (the loads of the start of a step are held over it). There is no controller here: the
control surfaces do not move and the rotor speeds are whatever the caller prescribes. A kite without its
controller is not expected to stay on its loop; the module is the coupling of the existing parts, to be checked
as such.

One aerodynamic step ``t_n -> t_n+1`` (``run_free``):

1. the rigid state gives the pose of the body origin (``RigidState.pose``) and the velocity of every body point,
   ``v_cm + omega x (x - x_cm)``, which the aerodynamics receives as its structural velocity;
2. the lifting surfaces are solved at ``t_n`` with the wake convected from the previous step: the same calls, in
   the same order, as ``flight.prescribed.run_prescribed`` (``start`` / ``advance``, or ``handover_step`` under a
   section hand-over with the weight of the body origin);
3. with ``rotors`` the discs are placed at the pose, their loads come from the rotor table and one ring per disc
   is released (``prescribed.rotor_step``); the rings act on the lifting surfaces from the next step on;
4. force and moment of 2 and 3 are taken to the centre of mass, ``M_cm = M_O - (R c) x F``, and stored in body
   axes (``held_loads``): these are the held loads of the step;
5. ``flight.plant.plant_step`` advances the body and the line together over the step (sub-steps of the fourth-order
   scheme, the line's forces evaluated at every stage). The gyroscopic moment of the spinning rotors,
   ``L x omega`` with ``L = R sum(I_r Omega_i dir_i a_i)`` (the convention of ``flight.residual``), depends on the
   angular velocity and is therefore evaluated at every stage (through the external-load entry of the plant), not
   held; the speeds are constant within a step, so there is no reaction moment.

``scheme='heun'`` (a comparison, not the default) repeats the step: the state reached with the loads of ``t_n``
is a predictor; the aerodynamics (and rotors) are solved once more at the predicted state from the wake of
``t_n`` - that solution and its rings are thrown away -, the two sets of held loads are averaged component by
component in body axes, and the step is taken again from the state of ``t_n``. Two aerodynamic solves per step.

``pin_to`` (a check device, not a model): a reference motion whose samples the body state is set back to at the
end of every step up to ``pin_until`` (a step number; None: every step). With an external load that makes the
body follow the reference within the step (``Release`` before its release time) the aerodynamics then sees exactly
the states of the prescribed-motion driver, which is what criterion P2b compares. A reference given by samples
cannot be followed to rounding by a cancelling load alone (its accelerations are differences), hence the pinning.

A failed step (the failures of ``prescribed.FAILURES`` in any part) stops the run; the records up to it are kept
and the failure is recorded with its step and module. Nothing is smoothed, replaced or retried, except the one
retry from the previous circulation that the hand-over rule itself makes under zero seeding.

Not in this module: controller, control surfaces, rotor-speed dynamics and electrical power, variable tether
length, ground contact, fuselage aerodynamics, added mass, flexibility. The pylons of decision 0028 (a group of
their own whose loads are added) are not in the loop either: a caller that wants them has to add their loads.

Nothing in the package uses this module: no existing path changes.
"""

import time as _time
from dataclasses import dataclass, field

import numpy as np

from ..aero.handover import SectionHandover
from ..aero.methods import AeroMethod
from ..aero.wake import ambient_velocity
from .kinematics import RigidMotion, relative_wind_body
from .plant import HeldLoads, PlantTether, plant_step, rates
from .prescribed import (FAILURES, PrescribedRotors, PrescribedTether, handover_record, handover_step, rotor_step,
                         step_record, tether_fields)
from .residual import RigidBody, accelerations
from .rigid_body import GRAVITY_M_S2, RigidState, kinetic_energy, state_from_pose

SCHEMES = ("hold", "heun")


def held_loads(body, rotation, force_inertial_N, moment_origin_inertial_Nm):
    """Loads given in inertial components about the body origin, as held loads about the centre of mass in body axes."""
    rotation = np.asarray(rotation, dtype=float)
    force = np.asarray(force_inertial_N, dtype=float).reshape(3)
    arm = rotation@body.centre_of_mass_body_m                  # origin -> centre of mass, inertial components
    moment = np.asarray(moment_origin_inertial_Nm, dtype=float).reshape(3)-np.cross(arm, force)
    return HeldLoads(rotation.T@force, rotation.T@moment)


def rotor_momentum_body(rotors, omegas, inertia_kg_m2):
    """Sum of the rotors' angular momenta in body axes: ``sum(I_r Omega_i dir_i a_i)``."""
    sense = np.array([disc.direction for disc in rotors.discs], dtype=float)
    axes = np.array([disc.axis for disc in rotors.discs], dtype=float)
    return (float(inertia_kg_m2)*np.asarray(omegas, dtype=float)*sense)@axes


def gyroscopic_moment(state, momentum_body):
    """Gyroscopic moment of the spinning rotors on the body, inertial components: ``L x omega``."""
    return state.rotation@np.cross(momentum_body, state.angular_velocity_body_rad_s)


class Release:
    """External load that carries the body along a reference motion and, from the release time on, holds a trim load.

    Before ``release_time_s`` (None: never released) the load returned at a stage is what makes the body's
    accelerations those of the reference: ``m a_cm,ref - m g - F_model`` and ``dH_ref/dt - M_model``, the model
    loads being those the plant hands to the callback (held loads plus line). The reference accelerations are
    those of ``flight.residual.accelerations`` at the samples (stored with the motion, or central differences of
    the stored velocities), linearly interpolated between samples.

    The release is decided per **aerodynamic step**, not per stage: the driver tells the callback the start time of
    every step (``begin_step``; ``run_free`` does), and the body is released from the first step that starts at or
    after the release time. The stage times of the plant reach the end of a step (the fourth stage of its last
    sub-step), so a decision by stage time would release one stage early, with the held loads of the step before
    (review finding PF-01 of the first run of this part). At the first stage of the release step the expression
    above is evaluated once more, with the loads of that step at the state the step starts from, and kept in
    **body axes**: the trim load (plant contract 1c.3). From then on it is returned unchanged, turning with the
    body. With ``trim=False`` nothing is applied after the release. ``trim_body`` holds the kept load (force and
    moment about the centre of mass) once it exists. A release time without a ``begin_step`` call is refused.
    """

    def __init__(self, body, motion, *, release_time_s=None, trim=True, gravity_m_s2=GRAVITY_M_S2):
        if not isinstance(body, RigidBody) or not isinstance(motion, RigidMotion):
            raise ValueError("Release needs a RigidBody and a RigidMotion")
        self.body, self.motion, self.trim = body, motion, bool(trim)
        self.release_time_s = None if release_time_s is None else float(release_time_s)
        self.gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
        linear, angular, self.acceleration_source = accelerations(motion)
        inertia_force, momentum_rate = np.empty_like(linear), np.empty_like(angular)
        for k in range(len(motion.time_s)):
            rotation, omega = motion.rotation[k], motion.angular_velocity_rad_s[k]
            offset = rotation@body.centre_of_mass_body_m
            inertia = rotation@body.inertia_body_kg_m2@rotation.T
            inertia_force[k] = body.mass_kg*(linear[k]+np.cross(angular[k], offset)+np.cross(omega, np.cross(omega, offset)))
            momentum_rate[k] = inertia@angular[k]+np.cross(omega, inertia@omega)
        self._inertia_force, self._momentum_rate = inertia_force, momentum_rate
        self.trim_body = None
        self._step_start = None

    def begin_step(self, time_s):
        """Called by the driver with the start time of the aerodynamic step the plant is about to advance over."""
        self._step_start = float(time_s)

    def reference(self, time_s):
        """``(m a_cm, dH/dt)`` of the reference at a time inside the samples (linear between samples)."""
        times = self.motion.time_s
        time = float(time_s)
        if time < times[0]-1e-12 or time > times[-1]+1e-12:
            raise ValueError("time outside the reference motion")
        index = int(np.clip(np.searchsorted(times, time, side="right")-1, 0, len(times)-2))
        s = (time-times[index])/(times[index+1]-times[index])
        return ((1.0-s)*self._inertia_force[index]+s*self._inertia_force[index+1],
                (1.0-s)*self._momentum_rate[index]+s*self._momentum_rate[index+1])

    def _cancel(self, time_s, model):
        inertia_force, momentum_rate = self.reference(time_s)
        return (inertia_force-self.body.mass_kg*self.gravity-model["force_inertial_N"],
                momentum_rate-model["moment_cm_inertial_Nm"])

    def __call__(self, time_s, state, model):
        released = False
        if self.release_time_s is not None:
            if self._step_start is None:
                raise ValueError("a Release with a release time must be told the start of each step (begin_step)")
            released = self._step_start >= self.release_time_s-1e-12
        if not released:
            return self._cancel(time_s, model)
        if not self.trim:
            return np.zeros(3), np.zeros(3)
        if self.trim_body is None:
            force, moment = self._cancel(time_s, model)
            self.trim_body = (state.rotation.T@force, state.rotation.T@moment)
        return state.rotation@self.trim_body[0], state.rotation@self.trim_body[1]


@dataclass
class FreeRun:
    """Records of one free-flight run; ``status`` is ``complete`` or ``failed`` (then ``error`` says why)."""
    records: list = field(default_factory=list)
    status: str = "complete"
    error: str | None = None
    error_type: str | None = None
    failed_step: int | None = None
    failed_module: str | None = None           # 'aerodynamics', 'rotors', 'plant', 'predictor', 'controller', or a hook's own name
    state: RigidState | None = None            # rigid state at the last recorded step
    nodes: np.ndarray | None = None            # line node positions at the last recorded step
    node_velocities: np.ndarray | None = None
    aero_state: object = None                  # aerodynamic state of the last recorded step (rings of that step released)
    step_seconds: list = field(default_factory=list)
    aero_solves: int = 0
    settings: dict = field(default_factory=dict)


def _aero(previous, backend, wing, polar, handover, wind, pose):
    """The lifting surfaces at one pose: the calls of ``run_prescribed``; ``previous`` is None at the first step."""
    placed = wing.transformed(pose.rotation, pose.position_m)

    def velocity(points, time_s):                         # the body's velocity field of this step
        return pose.point_velocity(points)
    choice = seeding = failure = None
    if handover is not None:
        origin_wind = ambient_velocity(wind, pose.position_m[None, :], pose.time_s)[0]
        _, _, alpha, beta = relative_wind_body(pose, origin_wind)
        if alpha is None:
            raise ValueError("no relative wind at the body origin: the hand-over weight is undefined")
        choice = handover.select(alpha, beta)
        state, seeding, failure = handover_step(previous, choice, backend.wake, placed, wind, velocity, pose.time_s)
    elif previous is None:
        state = backend.start(placed, polar, wind, time_s=pose.time_s, structural_velocity=velocity)
    else:
        state = backend.advance(previous, polar, wind, next_wing=placed, structural_velocity=velocity, next_time_s=pose.time_s)
    return state, placed, choice, seeding, failure


def _step_loads(body, pose, solved, placed, rotor_fields):
    """Held loads of a step from the solved lifting surfaces and, if present, the rotor record."""
    force = np.array(solved.solution.force, dtype=float)
    moment = np.array(solved.solution.moment(placed, reference=pose.position_m), dtype=float)
    if rotor_fields is not None:
        force = force+np.asarray(rotor_fields["rotor_force_inertial_N"], dtype=float)
        moment = moment+np.asarray(rotor_fields["rotor_moment_origin_inertial_Nm"], dtype=float)
    return held_loads(body, pose.rotation, force, moment)


def run_free(backend, wing, polar, body, state, wind, *, times, handover=None, rotors=None, rotor_inertia_kg_m2=None,
             tether=None, nodes=None, node_velocities=None, substeps=10, external=None, scheme="hold", pin_to=None,
             pin_until=None, gravity_m_s2=GRAVITY_M_S2, progress=None, before_aero=None, extra_loads=None):
    """March the coupled system from ``state`` over the steps ``times[0] -> times[1] -> ...``; returns a ``FreeRun``.

    ``times`` are the step times, spaced by the wake time step of the backend; ``state`` (``RigidState``) is the
    body at ``times[0]`` (its time is set to it). ``wing`` is the geometry in body axes, ``polar`` the section data
    (None under a ``handover``, which must be a ``SectionHandover`` with the weight of the body origin).
    ``rotors`` is a ``PrescribedRotors`` (given speeds); with ``rotor_inertia_kg_m2`` the gyroscopic moment is
    applied at every stage. ``tether`` is a ``PlantTether`` with its node state. ``external`` is the plant's
    external-load callback ``(time_s, state, model) -> (force, moment about the centre of mass)``, inertial
    components; if it has a method ``begin_step(time_s)``, it is called with the start time of every step before the
    plant advances over it (a callback that switches at a step boundary must use this: the plant's stage times
    reach the end of the step). ``wind`` is a uniform vector or a callable ``(points, time)``. Rotor speeds given
    as a schedule are taken at the start of each step and held over it; the reaction moment of a change of speed
    is not applied (frozen controls).

    One record per step time: the fields of the prescribed-motion driver (loads about the body origin, flight
    condition, hand-over, rotors, tether) with the rigid state, the held loads, the kinetic energy and, for every
    step but the last, the work and impulse account of the plant over the step that follows the record
    (``plant_work_J``, ``plant_impulse_N_s``). ``pinned`` says whether the state of that record was set to the
    reference sample.

    Two opt-in hooks let a controller stand in the loop (``flight.closed_loop``); without them nothing changes.
    ``before_aero(index, time_s, state, nodes, node_velocities, acceleration)`` is called at the start of every step,
    before the lifting surfaces are solved; ``acceleration`` is ``plant.rates`` at the state of this step under the
    held loads and the external callback of the step before (None at the first step). ``extra_loads(pose, record)``
    is called after the solve and returns ``(force, moment about the body origin, fields)`` in inertial components:
    loads that are added to the held loads of the step, and fields that are added to its record. Both hooks are
    refused with the predictor-corrector (the extra loads would have to be evaluated at the predicted state as well,
    and the acceleration handed to ``before_aero`` would not be that of the averaged loads the step was taken with).
    A failure raised inside a hook is recorded under the hook's own name for the failed part when the exception
    carries one (attribute ``failed_module``; ``flight.companion`` sets it), else under "controller".
    """
    if not isinstance(body, RigidBody) or not isinstance(state, RigidState):
        raise ValueError("run_free needs a RigidBody and a RigidState")
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if (extra_loads is not None or before_aero is not None) and scheme != "hold":
        raise ValueError("the controller hooks are only taken with the hold scheme")
    times = np.array(times, dtype=float)
    dt = backend.wake.dt_s
    if times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all():
        raise ValueError("times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.abs(times[:-1]+dt-times[1:]) > 8*np.finfo(float).eps*np.maximum(1.0, np.abs(times[1:]))):
        raise ValueError("the step times must be spaced by the wake time step")
    if len(times)-1 > backend.wake.max_rows and backend.wake.retained_rows is None:
        raise ValueError("the wake row limit of the backend is below the number of steps")
    if handover is not None:
        if not isinstance(handover, SectionHandover) or polar is not None:
            raise ValueError("with a hand-over pass a SectionHandover and polar=None")
        if handover.weight_basis != "origin":
            raise ValueError("free flight takes the hand-over weight of the body origin only")
        if backend.method != AeroMethod.VSM_FREE_WAKE:
            raise ValueError("the hand-over needs the VSM free-wake method")
        if backend.wake.solve != handover.low_solve:
            raise ValueError("the backend's solve settings must equal the crosswind settings of the hand-over")
    momentum_body = None
    if rotors is not None:
        if not isinstance(rotors, PrescribedRotors):
            raise ValueError("rotors must be a PrescribedRotors")
        if backend.method not in (AeroMethod.VSM_FREE_WAKE, AeroMethod.FREE_WAKE):
            raise ValueError("rotors need a free-wake method (their rings join the wake network)")
        for time in times:
            rotors.omegas(time)
    elif rotor_inertia_kg_m2 is not None:
        raise ValueError("a rotor inertia was given without rotors")
    if tether is not None:
        if not isinstance(tether, PlantTether):
            raise ValueError("tether must be a PlantTether")
        if nodes is None or node_velocities is None:
            raise ValueError("a tether needs its node positions and velocities")
        if tether.line.s.air_density_kg_m3 != backend.wake.density_kg_m3:
            raise ValueError("the line and the aerodynamics use different air densities")
        nodes, node_velocities = np.array(nodes, dtype=float), np.array(node_velocities, dtype=float)
        bound = tether.line.rk4_step_bound_s()
        if dt/int(substeps) > bound:
            raise ValueError(f"the line's time step {dt/int(substeps):.4g} s is above its stability bound {bound:.4g} s: use more substeps")
    elif nodes is not None or node_velocities is not None:
        raise ValueError("a node state was given without a tether")
    if pin_to is not None and not isinstance(pin_to, RigidMotion):
        raise ValueError("pin_to must be a RigidMotion")
    last_pinned = len(times)-1 if pin_until is None else int(pin_until)
    record_line = None if tether is None else PrescribedTether(tether.line, tether.anchor_m, tether.bridle, substeps=int(substeps))
    density = backend.wake.density_kg_m3
    state = RigidState(times[0], state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s)
    first_pose = state.pose(body)
    ambient_velocity(wind, wing.transformed(first_pose.rotation, first_pose.position_m).cp, first_pose.time_s)   # a set-up check
    run = FreeRun(settings=dict(scheme=scheme, substeps=int(substeps), step_s=float(dt), steps=int(len(times)-1),
                                gravity_m_s2=np.asarray(gravity_m_s2, dtype=float).tolist(),
                                pinned_to_reference_until_step=None if pin_to is None else last_pinned,
                                rotor_gyroscopic_moment=bool(rotors is not None and rotor_inertia_kg_m2 is not None),
                                handover=None if handover is None else "origin weight", tether=tether is not None,
                                rotors=rotors is not None))
    aero_state = None
    before = None                                   # held loads and external callback of the step before (for the hook)

    def loads_of(previous, pose, module):
        """Aerodynamics and rotors at a pose from the aerodynamic state ``previous``; returns the pieces of a record."""
        module[0] = "aerodynamics"
        solved, placed, choice, seeding, failure = _aero(previous, backend, wing, polar, handover, wind, pose)
        run.aero_solves += 1
        after, rotor_fields = solved, None
        if rotors is not None:
            module[0] = "rotors"
            after, rotor_fields = rotor_step(solved, backend.wake, rotors, pose, wind)
        return solved, after, placed, choice, seeding, failure, rotor_fields

    def external_at(time_s):
        """The external callback of a step: the caller's, plus the gyroscopic moment when the rotor inertia is given."""
        if rotors is None or rotor_inertia_kg_m2 is None:
            return external
        momentum = rotor_momentum_body(rotors, rotors.omegas(time_s), rotor_inertia_kg_m2)

        def combined(time, stage, model):
            force, moment = (np.zeros(3), np.zeros(3)) if external is None else external(time, stage, model)
            return np.asarray(force, dtype=float), np.asarray(moment, dtype=float)+gyroscopic_moment(stage, momentum)
        return combined

    for index in range(len(times)):
        tick = _time.perf_counter()
        module = ["aerodynamics"]
        pose = state.pose(body)
        try:
            if before_aero is not None:
                acceleration = None
                if before is not None:
                    module[0] = "plant"
                    acceleration = rates(body, state, tether=tether, nodes=nodes, node_velocities=node_velocities, held=before[0],
                                         external=before[1], wind=wind if tether is not None else None, gravity_m_s2=gravity_m_s2)
                module[0] = "controller"
                before_aero(index, float(times[index]), state, nodes, node_velocities, acceleration)
                module[0] = "aerodynamics"
            solved, after, placed, choice, seeding, failure, rotor_fields = loads_of(aero_state, pose, module)
            held = _step_loads(body, pose, solved, placed, rotor_fields)
            record = step_record(index, pose, solved, placed, wind, density)
            if choice is not None:
                record.update(handover_record(solved, placed, choice, seeding, failure))
            if rotor_fields is not None:
                record.update(rotor_fields)
            if tether is not None:
                module[0] = "plant"
                record.update(tether_fields(record_line, pose, nodes, node_velocities, wind))
            if extra_loads is not None:
                module[0] = "controller"
                added_force, added_moment, fields = extra_loads(pose, record)
                added = held_loads(body, pose.rotation, added_force, added_moment)
                held = HeldLoads(held.force_body_N+added.force_body_N, held.moment_cm_body_Nm+added.moment_cm_body_Nm)
                record.update(fields)
            energy = kinetic_energy(body, state)
            record.update(position_cm_m=state.position_cm_m.tolist(), velocity_cm_m_s=state.velocity_cm_m_s.tolist(),
                          rotation=state.rotation.tolist(), angular_velocity_body_rad_s=state.angular_velocity_body_rad_s.tolist(),
                          held_force_body_N=held.force_body_N.tolist(), held_moment_cm_body_Nm=held.moment_cm_body_Nm.tolist(),
                          kinetic_energy_J=float(energy[0]+energy[1]),
                          pinned=bool(pin_to is not None and index <= last_pinned))
            new = None
            if index < len(times)-1:
                step = float(times[index+1]-times[index])
                callback = external_at(times[index])
                module[0] = "plant"
                if external is not None and hasattr(external, "begin_step"):
                    external.begin_step(float(times[index]))
                args = dict(step_s=step, substeps=int(substeps), tether=tether, nodes=nodes, node_velocities=node_velocities,
                            external=callback, wind=wind if tether is not None else None, gravity_m_s2=gravity_m_s2)
                new = plant_step(body, state, held=held, **args)
                before = (held, callback)
                if scheme == "heun":
                    module[0] = "predictor"
                    trial = RigidState(times[index+1], new[0].position_cm_m, new[0].rotation, new[0].velocity_cm_m_s,
                                       new[0].angular_velocity_body_rad_s)
                    trial_pose = trial.pose(body)
                    trial_solved, _, trial_placed, _, _, _, trial_rotor = loads_of(after, trial_pose, ["predictor"])
                    ahead = _step_loads(body, trial_pose, trial_solved, trial_placed, trial_rotor)
                    mean = HeldLoads(0.5*(held.force_body_N+ahead.force_body_N), 0.5*(held.moment_cm_body_Nm+ahead.moment_cm_body_Nm))
                    record.update(predictor_force_body_N=ahead.force_body_N.tolist(),
                                  predictor_moment_cm_body_Nm=ahead.moment_cm_body_Nm.tolist())
                    module[0] = "plant"
                    new = plant_step(body, state, held=mean, **args)
                account = new[3]
                record.update(plant_work_J=dict(account.work_J),
                              plant_impulse_N_s={name: value.tolist() for name, value in account.impulse_N_s.items()})
        except FAILURES as error:
            run.status, run.failed_step, run.error_type = "failed", index, type(error).__name__
            run.failed_module = getattr(error, "failed_module", module[0])   # a hook may name the part that failed
            run.error = f"{type(error).__name__}: {str(error)[:300]}"
            break
        run.step_seconds.append(_time.perf_counter()-tick)
        run.records.append(record)
        run.state, run.nodes, run.node_velocities, run.aero_state = state, nodes, node_velocities, after
        aero_state = after
        if progress is not None:
            progress(record)
        if new is not None:
            moved, nodes, node_velocities, _ = new
            if pin_to is not None and index+1 <= last_pinned:
                state = state_from_pose(pin_to.pose(times[index+1]), body)
                state = RigidState(times[index+1], state.position_cm_m, state.rotation, state.velocity_cm_m_s,
                                   state.angular_velocity_body_rad_s)
            else:                                           # the time from the step list, free of accumulated rounding
                state = RigidState(times[index+1], moved.position_cm_m, moved.rotation, moved.velocity_cm_m_s,
                                   moved.angular_velocity_body_rad_s)
    return run
