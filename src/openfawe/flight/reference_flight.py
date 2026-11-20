"""Flight on a reference aerodynamic model that needs no flow solution (G5 task, reference-plant contract).

``run_reference`` marches the same rigid body, line and controller hooks as ``flight.free_flight.run_free``, with
the lifting-surface solution replaced by a model that returns the loads of the whole kite from its state alone
(``aero.makani_database.MakaniDatabase``: the databases of the Makani simulator, a borrowed and declared model).
There is no wake, so a run needs no lead-in.

One step ``t_n -> t_n+1``:

1. the controller hook (``before_step``, the ``before_aero`` of ``flight.closed_loop.CrosswindLoop``) is called with
   the state and the acceleration of the body under the loads of the step before (``start_acceleration`` at the
   first step);
2. the reference model gives force and moment about the body origin in body axes at the state of ``t_n``, with the
   flap angles and the rotor thrust coefficient the controller holds for this step; the rotor loads come from the
   hook ``extra_loads`` as in ``run_free``;
3. ``plant_step`` advances body and line. With ``per_stage=False`` (the default, and what the vortex-wake plant
   does) the reference loads of ``t_n`` are held in body axes over the step. With ``per_stage=True`` they are
   evaluated again at every stage of the body's integration, at the state of that stage, with the flap angles,
   thrust coefficient and rotor loads of the step still held: the difference between the two runs is the size of
   the "hold" approximation for these loads.

A failed step stops the run; the records up to it are kept, with the step and the module. Nothing is smoothed or
retried. Nothing in the package uses this module.
"""

import time as _time

import numpy as np

from .free_flight import FreeRun, held_loads
from .kinematics import relative_wind_body
from .plant import HeldLoads, PlantTether, plant_step, rates
from .prescribed import FAILURES, PrescribedTether, tether_fields
from .residual import RigidBody
from .rigid_body import GRAVITY_M_S2, RigidState, kinetic_energy


def reference_loads_body(aero, body, state, wind_m_s, density_kg_m3, flaps_rad, thrust_coeff, *, wind_rates_body=None):
    """Loads of the reference model at a state: ``HeldLoads`` about the centre of mass in body axes, and the condition.

    ``wind_rates_body`` (plan 3.9b, default None): the wind-induced apparent rates in body axes, added to the body's
    angular velocity for the coefficient lookup only, as Makani's ``Wing`` does (wing.cc:553-558)."""
    pose = state.pose(body)
    through_air = pose.rotation.T@(pose.velocity_m_s-np.asarray(wind_m_s, dtype=float))
    omega = state.angular_velocity_body_rad_s if wind_rates_body is None else state.angular_velocity_body_rad_s + wind_rates_body
    force, moment_origin, condition = aero.loads(through_air, omega, density_kg_m3, flaps_rad, thrust_coeff)
    moment_cm = moment_origin-np.cross(body.centre_of_mass_body_m, force)
    return HeldLoads(force, moment_cm), moment_origin, condition


def run_reference(aero, body, state, wind_m_s, *, times, density_kg_m3, tether=None, nodes=None, node_velocities=None,
                  substeps=5, external=None, per_stage=False, before_step=None, extra_loads=None, flaps=None,
                  thrust_coeff=None, start_acceleration=None, gravity_m_s2=GRAVITY_M_S2, progress=None):
    """March body and line on the reference model over ``times``; returns a ``FreeRun``.

    ``aero`` has ``loads(velocity_through_air_body, angular_velocity_body, density, flaps, thrust_coeff)`` returning
    force and moment about the body origin in body axes and a condition dict. ``wind_m_s`` is a uniform vector.
    ``before_step(index, time_s, state, nodes, node_velocities, acceleration)`` and ``extra_loads(pose, record)`` are
    the controller hooks of ``run_free``; ``flaps()`` and ``thrust_coeff()`` return the flap angles and the rotor
    thrust coefficient held over the step (called after ``before_step`` and before ``extra_loads``; default: zeros).
    ``external`` is the plant's per-stage callback; a ``begin_step`` method is honoured as in ``run_free``.
    """
    if not isinstance(body, RigidBody) or not isinstance(state, RigidState):
        raise ValueError("run_reference needs a RigidBody and a RigidState")
    times = np.array(times, dtype=float)
    if times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be a finite increasing one-dimensional array")
    wind = np.array(wind_m_s, dtype=float).reshape(3)
    if tether is not None:
        if not isinstance(tether, PlantTether) or nodes is None or node_velocities is None:
            raise ValueError("a tether must be a PlantTether with its node positions and velocities")
        nodes, node_velocities = np.array(nodes, dtype=float), np.array(node_velocities, dtype=float)
    elif nodes is not None or node_velocities is not None:
        raise ValueError("a node state was given without a tether")
    record_line = None if tether is None else PrescribedTether(tether.line, tether.anchor_m, tether.bridle, substeps=int(substeps))
    state = RigidState(times[0], state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s)
    run = FreeRun(settings=dict(model="reference", per_stage=bool(per_stage), substeps=int(substeps), steps=int(len(times)-1),
                                density_kg_m3=float(density_kg_m3), gravity_m_s2=np.asarray(gravity_m_s2, dtype=float).tolist(),
                                tether=tether is not None))
    before = None
    for index in range(len(times)):
        tick = _time.perf_counter()
        module = ["plant"]
        pose = state.pose(body)
        try:
            if before_step is not None:
                acceleration = start_acceleration
                if before is not None:
                    acceleration = rates(body, state, tether=tether, nodes=nodes, node_velocities=node_velocities, held=before[0],
                                         external=before[1], wind=wind if tether is not None else None, gravity_m_s2=gravity_m_s2)
                module[0] = "controller"
                before_step(index, float(times[index]), state, nodes, node_velocities, acceleration)
            flap_angles = np.zeros(8) if flaps is None else np.array(flaps(), dtype=float)
            coefficient = 0.0 if thrust_coeff is None else float(thrust_coeff())
            module[0] = "aerodynamics"
            reference, moment_origin, condition = reference_loads_body(aero, body, state, wind, density_kg_m3, flap_angles, coefficient)
            _, _, alpha_deg, beta_deg = relative_wind_body(pose, wind)
            record = dict(step=int(index), time_s=float(times[index]), position_m=pose.position_m.tolist(),
                          alpha_deg=alpha_deg, beta_deg=beta_deg, airspeed_m_s=condition["airspeed_m_s"],
                          aero_force_body_N=reference.force_body_N.tolist(), aero_moment_origin_body_Nm=np.asarray(moment_origin).tolist(),
                          aero_condition={key: value for key, value in condition.items()}, flaps_rad=flap_angles.tolist(),
                          thrust_coeff_used=coefficient)
            held = HeldLoads() if per_stage else reference
            if extra_loads is not None:
                module[0] = "controller"
                added_force, added_moment, fields = extra_loads(pose, record)
                added = held_loads(body, pose.rotation, added_force, added_moment)
                held = HeldLoads(held.force_body_N+added.force_body_N, held.moment_cm_body_Nm+added.moment_cm_body_Nm)
                record.update(fields)
            if tether is not None:
                module[0] = "plant"
                record.update(tether_fields(record_line, pose, nodes, node_velocities, wind))
            energy = kinetic_energy(body, state)
            record.update(position_cm_m=state.position_cm_m.tolist(), velocity_cm_m_s=state.velocity_cm_m_s.tolist(),
                          rotation=state.rotation.tolist(), angular_velocity_body_rad_s=state.angular_velocity_body_rad_s.tolist(),
                          held_force_body_N=held.force_body_N.tolist(), held_moment_cm_body_Nm=held.moment_cm_body_Nm.tolist(),
                          kinetic_energy_J=float(energy[0]+energy[1]))
            new = None
            if index < len(times)-1:
                callback = external
                if per_stage:
                    def callback(time_s, stage, model, flap_angles=flap_angles, coefficient=coefficient, module=module):
                        force, moment = (np.zeros(3), np.zeros(3)) if external is None else external(time_s, stage, model)
                        module[0] = "aerodynamics"          # a failure of the model inside the plant's stages is the model's
                        loads, _, _ = reference_loads_body(aero, body, stage, wind, density_kg_m3, flap_angles, coefficient)
                        module[0] = "plant"
                        return (np.asarray(force, dtype=float)+stage.rotation@loads.force_body_N,
                                np.asarray(moment, dtype=float)+stage.rotation@loads.moment_cm_body_Nm)
                module[0] = "plant"
                if external is not None and hasattr(external, "begin_step"):
                    external.begin_step(float(times[index]))
                new = plant_step(body, state, step_s=float(times[index+1]-times[index]), substeps=int(substeps), tether=tether,
                                 nodes=nodes, node_velocities=node_velocities, held=held, external=callback,
                                 wind=wind if tether is not None else None, gravity_m_s2=gravity_m_s2)
                before = (held, callback)
                account = new[3]
                record.update(plant_work_J=dict(account.work_J),
                              plant_impulse_N_s={name: value.tolist() for name, value in account.impulse_N_s.items()})
        except FAILURES as error:
            run.status, run.failed_step, run.error_type = "failed", index, type(error).__name__
            run.failed_module = module[0]
            run.error = f"{type(error).__name__}: {str(error)[:300]}"
            break
        run.step_seconds.append(_time.perf_counter()-tick)
        run.records.append(record)
        run.state, run.nodes, run.node_velocities = state, nodes, node_velocities
        if progress is not None:
            progress(record)
        if new is not None:
            moved, nodes, node_velocities, _ = new
            state = RigidState(times[index+1], moved.position_cm_m, moved.rotation, moved.velocity_cm_m_s,
                               moved.angular_velocity_body_rad_s)
    return run
