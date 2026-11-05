"""Several rigid bodies on prescribed motions in one vortex-wake system (OF-0009 contract, case C3).

Each body is a lifting-surface geometry in its own body axes with its own panel polars, its own
``RigidMotion`` and, optionally, its own tether (``PrescribedTether``, one line from its ground
point to its bridle point). At every step the bodies are placed at their poses and joined into one
``WingAssembly`` (surface names prefixed with the body name), so that all surfaces share one
circulation solve and one free-wake state, as the surfaces of one aircraft do: the wake of each
body induces velocities on every other body, and the wakes convect together. The structural
velocity of a panel point is that of the body it belongs to; the bodies are told apart by the
nearest body origin, which is unambiguous while the bodies stay farther apart than their own size
(checked at every step against ``separation_m``: a step at which two origins come closer fails
the run). The aerodynamic settings are those of the supplied backend, which must be a free-wake
method (``vsm_free_wake`` or ``free_wake``); hand-over and panel-onset diagnostics of
``run_prescribed`` are not offered here.

With ``rotors`` on a body (``PrescribedRotors``, as in ``run_prescribed``) its discs follow it: after the
lifting surfaces of a step are solved, every disc of every body is placed at its body's pose, its loads
come from its table at the apparent wind of its hub (ambient wind, the induction of the whole
lifting-surface network and of the other discs' rings, minus the hub velocity) and one ring per disc
is released into the one ring set of the wake, where it convects with everything else and induces on
every body from the next step on (``openfawe.rotor.coupled.shed_rotor_rings`` with the discs of all
bodies in one call). The ring settings (nodes per ring, surface core, turbine extension, ring phase,
the retained and the freely convected ring steps) must be the same for every body with rotors, because
the wake carries one ring set. The records of a body with rotors hold the rotor fields of
``openfawe.flight.prescribed.rotor_step`` for that body (its own discs; ``ring_nodes`` is the size of the whole ring
set, ``ring_nodes_within_surface_core`` counts the ring nodes near this body's own load points and
``ring_nodes_within_surface_core_all_bodies`` those near the load points of all bodies).

The records of a step hold the loads of every body in inertial and body axes about its own origin
(the sums of its panel forces and couples), its flight condition at its origin and, with a tether,
the tether fields of ``openfawe.flight.prescribed.tether_fields``. With one body and no tether the
run reproduces ``run_prescribed`` bit for bit (test ``test_multi_prescribed.py``, also with rotors);
with two bodies far apart it reproduces two single runs to the tree code's error budget
(superposition check S1 of the contract).
"""

import time as _time
from dataclasses import dataclass, field, replace

import numpy as np

from ..aero.aircraft import PanelPolars, WingAssembly
from ..aero.methods import AeroMethod
from ..aero.wake import ambient_velocity
from ..aero.wing import SectionPolar, Wing
from ..rotor.coupled import rotor_totals, shed_rotor_rings
from .kinematics import RigidMotion, relative_wind_body
from .prescribed import FAILURES, PrescribedRotors, PrescribedTether, failure_info, ring_sheet_fields, tether_fields, wind_field

RING_SETTINGS = ("nodes_per_ring", "surface_core_m", "turbine_extension", "ring_phase_rad", "retained_ring_steps",
                 "free_convection_ring_steps", "ring_core_m", "ring_bound_core_m", "ring_surface_overlap", "ring_sheet_drift",
                 "ring_ambient_spacing")


@dataclass(frozen=True)
class PrescribedBody:
    """One body of a multi-body run: geometry in body axes, panel polars, motion, optional tether and rotors."""
    name: str
    wing: object
    polar: PanelPolars
    motion: RigidMotion
    tether: PrescribedTether | None = None
    rotors: PrescribedRotors | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a body needs a name")
        if not isinstance(self.wing, (Wing, WingAssembly)):
            raise ValueError("wing must be a Wing or a WingAssembly")
        if isinstance(self.polar, SectionPolar):
            object.__setattr__(self, "polar", PanelPolars((self.polar,)*self.wing.panels))
        if not isinstance(self.polar, PanelPolars) or len(self.polar.polars) != self.wing.panels:
            raise ValueError("polar must be a PanelPolars with one polar per panel of the wing (or one SectionPolar for all)")
        if not isinstance(self.motion, RigidMotion):
            raise ValueError("motion must be a RigidMotion")
        if self.tether is not None and not isinstance(self.tether, PrescribedTether):
            raise ValueError("tether must be a PrescribedTether")
        if self.rotors is not None and not isinstance(self.rotors, PrescribedRotors):
            raise ValueError("rotors must be a PrescribedRotors")

    @property
    def surfaces(self):
        """``(wings, names)`` of the body's surfaces, names prefixed with the body name."""
        if isinstance(self.wing, WingAssembly):
            return tuple(self.wing.wings), tuple(f"{self.name}/{n}" for n in self.wing.names)
        return (self.wing,), (f"{self.name}/wing",)


@dataclass
class MultiPrescribedRun:
    """Records of one run; ``status`` is ``complete`` or ``failed`` (then ``error`` says why)."""
    body_names: tuple
    motion_consistency: dict
    panel_slices: dict
    status: str = "complete"
    error: str | None = None
    error_type: str | None = None
    failed_step: int | None = None
    failed_module: str | None = None
    failed_iterations: int | None = None
    records: list = field(default_factory=list)
    step_seconds: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    final_state: object = None
    tether_states: dict = field(default_factory=dict)
    tether_settings: dict = field(default_factory=dict)


def place_bodies(bodies, time_s):
    """Poses, the joint placed assembly and the panel index slice of every body at ``time_s``."""
    poses, wings, names, slices = [], [], [], {}
    offset = 0
    for body in bodies:
        pose = body.motion.pose(time_s)
        poses.append(pose)
        own, own_names = body.surfaces
        for wing, name in zip(own, own_names):
            wings.append(wing.transformed(pose.rotation, pose.position_m))
            names.append(name)
        slices[body.name] = slice(offset, offset+body.wing.panels)
        offset += body.wing.panels
    return poses, WingAssembly(tuple(wings), tuple(names)), slices


def joint_polars(bodies):
    return PanelPolars(tuple(p for body in bodies for p in body.polar.polars))


def structural_velocity_field(bodies, separation_m):
    """The ``(points, time)`` callable: the velocity of the body whose origin is nearest to each point."""
    def velocity(points, time_s):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        poses = [body.motion.pose(time_s) for body in bodies]
        origins = np.array([pose.position_m for pose in poses])
        if len(poses) > 1:
            gaps = np.linalg.norm(origins[:, None, :]-origins[None, :, :], axis=2)
            if np.min(gaps[np.triu_indices(len(poses), 1)]) < separation_m:
                raise ValueError(f"two body origins are closer than the declared separation {separation_m:g} m")
        owner = np.argmin(np.linalg.norm(points[:, None, :]-origins[None, :, :], axis=2), axis=1)
        out = np.empty_like(points)
        for k, pose in enumerate(poses):
            mine = owner == k
            if np.any(mine):
                out[mine] = pose.point_velocity(points[mine])
        return out
    return velocity


def body_record(index, body, pose, state, placed, panel_slice, wind, density):
    """Loads of one body at one step in inertial and body axes about its own origin, with its flight condition."""
    solution = state.solution
    force = solution.panel_force[panel_slice].sum(axis=0)
    moment = (np.cross(placed.cp[panel_slice]-pose.position_m, solution.panel_force[panel_slice])
              + solution.panel_couple[panel_slice]).sum(axis=0)
    origin_wind = ambient_velocity(wind, pose.position_m[None, :], pose.time_s)[0]
    relative, speed, alpha, beta = relative_wind_body(pose, origin_wind)
    return dict(position_m=pose.position_m.tolist(), velocity_m_s=pose.velocity_m_s.tolist(),
                force_inertial_N=force.tolist(), moment_origin_inertial_Nm=moment.tolist(),
                force_body_N=(pose.rotation.T@force).tolist(), moment_origin_body_Nm=(pose.rotation.T@moment).tolist(),
                airspeed_m_s=speed, alpha_deg=alpha, beta_deg=beta, relative_wind_body_m_s=relative.tolist(),
                aero_power_W=float(force@pose.velocity_m_s+moment@pose.angular_velocity_rad_s),
                dynamic_pressure_Pa=float(.5*density*speed*speed),
                panel_alpha_deg=dict(min=float(np.degrees(solution.alpha_rad[panel_slice].min())),
                                     max=float(np.degrees(solution.alpha_rad[panel_slice].max()))))


def ring_settings(bodies):
    """The one set of ring settings of the bodies with rotors (None without rotors); unequal settings are refused."""
    sets = [body.rotors for body in bodies if body.rotors is not None]
    if not sets:
        return None
    for other in sets[1:]:
        for name in RING_SETTINGS:
            if getattr(other, name) != getattr(sets[0], name):
                raise ValueError(f"all bodies with rotors must share the ring setting {name} (one ring set in the wake)")
    return {name: getattr(sets[0], name) for name in RING_SETTINGS}


def rotor_step_multi(state, config, bodies, poses, wind, settings, panel_slices=None):
    """The rotor step of all bodies with rotors at their poses: ``(state with this step's rings, {name: fields})``.

    Same quantities per body as ``openfawe.flight.prescribed.rotor_step`` (loads of its own discs about its
    own origin, their rate of work on it, shaft power, per-disc rows); the discs of all bodies are placed
    and released in one call so that every hub sees the rings of every other disc. ``ring_nodes`` is the
    size of the whole ring set. ``ring_nodes_within_surface_core`` counts the nodes of all rings (of every
    body) within the surface core of a load point of this body's own lifting surfaces (its panels of
    ``panel_slices``; without the slices, of all lifting surfaces of the step), and
    ``ring_nodes_within_surface_core_all_bodies`` the same for the load points of all bodies together.
    """
    discs, hub_velocity, omegas, owners = [], [], [], []
    for body, pose in zip(bodies, poses):
        if body.rotors is None:
            continue
        placed = [disc.placed(pose.rotation, pose.position_m) for disc in body.rotors.discs]
        hubs = np.array([disc.centre_m for disc in placed])
        discs.extend(placed)
        hub_velocity.append(pose.point_velocity(hubs))
        omegas.append(body.rotors.omegas(pose.time_s))
        owners.append((body, pose, slice(len(discs)-len(placed), len(discs))))
    hub_velocity = np.vstack(hub_velocity)
    omegas = np.concatenate(omegas)
    core = settings["surface_core_m"] or config.core_radius_m
    near_all = 0 if state.rings is None else state.rings.near_surface_count(state.wing.cp, core)
    near = {}
    for body, _, _ in owners:
        if state.rings is None or panel_slices is None:
            near[body.name] = near_all
        else:
            near[body.name] = state.rings.near_surface_count(state.wing.cp[panel_slices[body.name]], core)
    sheet = ring_sheet_fields(state, core) if settings["ring_surface_overlap"] is not None else {}
    state, rows = shed_rotor_rings(state, config, discs, omegas, wind, nodes_per_ring=settings["nodes_per_ring"],
                                   hub_velocities=hub_velocity, phase=settings["ring_phase_rad"],
                                   surface_core_m=settings["surface_core_m"], turbine_extension=settings["turbine_extension"],
                                   free_convection_steps=settings["free_convection_ring_steps"], ring_core_m=settings["ring_core_m"],
                                   ring_bound_core_m=settings["ring_bound_core_m"], ring_surface_overlap=settings["ring_surface_overlap"],
                                   ring_sheet_drift=settings["ring_sheet_drift"], ring_ambient_spacing=settings["ring_ambient_spacing"])
    if settings["retained_ring_steps"] is not None:
        horizon = state.time_s-(int(settings["retained_ring_steps"])-0.5)*config.dt_s
        state = replace(state, rings=state.rings.released_since(horizon))
    fields = {}
    for body, pose, part in owners:
        own_rows, own_discs, own_velocity = rows[part], discs[part], hub_velocity[part]
        force, moment = rotor_totals(own_rows, own_discs, about=pose.position_m)
        work = sum(float(np.dot(row["force_N"], velocity)+np.dot(row["moment_Nm"], pose.angular_velocity_rad_s))
                   for row, velocity in zip(own_rows, own_velocity))
        entry = dict(rotor_force_inertial_N=force.tolist(), rotor_moment_origin_inertial_Nm=moment.tolist(),
                     rotor_force_body_N=(pose.rotation.T@force).tolist(), rotor_moment_origin_body_Nm=(pose.rotation.T@moment).tolist(),
                     rotor_power_on_body_W=work, rotor_shaft_power_W=float(sum(row["power_W"] for row in own_rows)),
                     rotor_thrust_N=[float(row["thrust_N"]) for row in own_rows], rotor_power_W=[float(row["power_W"]) for row in own_rows],
                     rotor_axial_freestream_m_s=[float(row["axial_freestream_m_s"]) for row in own_rows],
                     rotor_omega_rad_s=omegas[part].tolist(), rotor_table_saturated=[bool(row["saturated"]) for row in own_rows],
                     ring_nodes=int(state.rings.count), ring_nodes_within_surface_core=int(near[body.name]),
                     ring_nodes_within_surface_core_all_bodies=int(near_all), ring_surface_core_m=float(core))
        if settings["retained_ring_steps"] is not None:
            entry["ring_retained_steps"] = int(settings["retained_ring_steps"])
        if settings["free_convection_ring_steps"] is not None:
            entry["ring_free_convection_steps"] = int(settings["free_convection_ring_steps"])
        if settings["ring_core_m"] is not None:
            entry["ring_core_m"] = float(settings["ring_core_m"])
        if settings["ring_bound_core_m"] is not None:
            entry["ring_bound_core_m"] = float(settings["ring_bound_core_m"])
        if settings["ring_surface_overlap"] is not None:
            entry["ring_surface_overlap"] = float(settings["ring_surface_overlap"])
            entry.update(sheet)
        if settings["ring_sheet_drift"]:
            entry["ring_sheet_drift"] = True
        if settings["ring_ambient_spacing"]:
            entry["ring_ambient_spacing"] = True
        if settings["ring_sheet_drift"]:                  # review AS-04: the released rings per disc of this body
            entry["rotor_ring_circulation_m2_s"] = [float(row["ring_circulation_m2_s"]) for row in own_rows]
            entry["rotor_ring_spacing_m"] = [float(row["ring_spacing_m"]) for row in own_rows]
            entry["rotor_sheet_strength_m_s"] = [float(row["sheet_strength_m_s"]) for row in own_rows]
        fields[body.name] = entry
    return state, fields


def run_prescribed_multi(backend, bodies, wind, *, steps=None, keep_states=(), progress=None, attitude_scheme="interval",
                         keep_tether_states=(), separation_m=50., state_sink=None):
    """March ``backend`` along the motions of ``bodies`` (a sequence of ``PrescribedBody``) in one wake.

    All motions must share the time samples (spaced by the wake time step); step ``k`` uses sample ``k``.
    ``wind`` is a uniform (3,) vector or a callable ``(points_m, time_s)``. ``separation_m`` is the
    smallest distance between two body origins at which the ownership of a panel point by the nearest
    origin is still unambiguous; it should exceed the largest body extent. The self-consistency reports
    of the motions are stored per body (``motion_consistency``); judging them is the caller's task.
    ``keep_states`` and ``keep_tether_states`` list step numbers whose wake state and tether node states
    (per body) are kept. With ``state_sink`` (a callable ``(step, state)``) the wake states of ``keep_states``
    are handed to it as they are reached instead of being kept in ``states`` (frames of a long run without holding
    them all in memory; OF-0009, 2026-10-22).
    """
    bodies = tuple(bodies)
    if not bodies or any(not isinstance(b, PrescribedBody) for b in bodies):
        raise ValueError("bodies must be a non-empty sequence of PrescribedBody")
    if len({b.name for b in bodies}) != len(bodies):
        raise ValueError("body names must be unique")
    if backend.method not in (AeroMethod.VSM_FREE_WAKE, AeroMethod.FREE_WAKE):
        raise ValueError("the multi-body run needs a free-wake method (one wake for all bodies)")
    times = bodies[0].motion.time_s
    for body in bodies[1:]:
        if len(body.motion.time_s) != len(times) or not np.allclose(body.motion.time_s, times, rtol=0., atol=8*np.finfo(float).eps*np.maximum(1., np.abs(times))):
            raise ValueError("all motions must share the time samples")
    dt = backend.wake.dt_s
    count = len(times)-1 if steps is None else int(steps)
    if count < 0 or count > len(times)-1:
        raise ValueError("more steps requested than motion samples")
    times = times[:count+1]
    if count and np.any(np.abs(times[:-1]+dt-times[1:]) > 8*np.finfo(float).eps*np.maximum(1., np.abs(times[1:]))):
        raise ValueError("motion samples must be spaced by the wake time step")
    if count > backend.wake.max_rows and backend.wake.retained_rows is None:
        raise ValueError("the wake row limit of the backend is below the number of steps")
    for body in bodies:
        if body.tether is not None:
            bound = body.tether.line.rk4_step_bound_s()
            if dt/body.tether.substeps > bound:
                raise ValueError(f"the line's time step of body {body.name} is above its stability bound {bound:.4g} s: use more substeps")
            if body.tether.line.s.air_density_kg_m3 != backend.wake.density_kg_m3:
                raise ValueError("the line and the aerodynamics use different air densities")
    rings = ring_settings(bodies)
    if rings is not None:
        if backend.method is not AeroMethod.VSM_FREE_WAKE and backend.method is not AeroMethod.FREE_WAKE:
            raise ValueError("rotors need a free-wake method (their rings join the wake network)")
        for body in bodies:
            if body.rotors is not None:
                for time_s in times:
                    body.rotors.omegas(time_s)                     # a speed schedule is checked before the first step
    polar = joint_polars(bodies)
    velocity = structural_velocity_field(bodies, float(separation_m))
    poses, placed, slices = place_bodies(bodies, times[0])
    ambient_velocity(wind, placed.cp, times[0])
    velocity(placed.cp, times[0])                                   # the set-up check of the separation and the field
    density = backend.wake.density_kg_m3
    keep, keep_lines = {int(k) for k in keep_states}, {int(k) for k in keep_tether_states}
    run = MultiPrescribedRun(tuple(b.name for b in bodies), {b.name: b.motion.consistency(attitude_scheme) for b in bodies}, slices)
    lines = {}
    line_wind = wind_field(wind)
    for body in bodies:
        if body.tether is not None:
            lines[body.name] = (body.tether.start(body.motion), body.tether.ends(body.motion))
            ambient_velocity(wind, lines[body.name][0][0], times[0])
            run.tether_settings[body.name] = body.tether.settings(dt)
    state = None
    for index in range(count+1):
        tick = _time.perf_counter()
        time_s = times[index]
        poses, placed, _ = place_bodies(bodies, time_s)
        module, before = "aerodynamics", state
        tether_records, rotor_records = {}, {}
        try:
            if index == 0:
                state = backend.start(placed, polar, wind, time_s=time_s, structural_velocity=velocity)
            else:
                state = backend.advance(state, polar, wind, next_wing=placed, structural_velocity=velocity, next_time_s=time_s)
            solved = state                                        # the loads of this step come from the state before the release
            if rings is not None:
                module = "rotors"
                state, rotor_records = rotor_step_multi(state, backend.wake, bodies, poses, wind, rings, slices)
            for body, pose in zip(bodies, poses):
                if body.tether is None:
                    continue
                module = f"tether:{body.name}"
                (positions, velocities), ends = lines[body.name]
                if index:
                    begin, end = times[index-1], times[index]
                    for part in range(body.tether.substeps):
                        start = begin+part*(end-begin)/body.tether.substeps
                        until = end if part == body.tether.substeps-1 else begin+(part+1)*(end-begin)/body.tether.substeps
                        positions, velocities = body.tether.line.step(start, positions, velocities, until-start, ends, line_wind)
                    lines[body.name] = ((positions, velocities), ends)
                tether_records[body.name] = tether_fields(body.tether, pose, positions, velocities, wind)
        except FAILURES as error:
            run.status, run.failed_step, run.failed_module = "failed", index, module
            info = failure_info(error)
            run.error_type, run.error, run.failed_iterations = info["type"], f"{info['type']}: {info['message']}", info["iterations"]
            state = before
            break
        run.step_seconds.append(_time.perf_counter()-tick)
        record = dict(step=int(index), time_s=float(time_s), iterations=int(solved.solution.iterations),
                      residual=float(solved.solution.residual), residual_limit=float(solved.solution.residual_limit),
                      force_inertial_N=solved.solution.force.tolist(), bodies={})
        for body, pose in zip(bodies, poses):
            entry = body_record(index, body, pose, solved, placed, slices[body.name], wind, density)
            if body.name in rotor_records:
                entry.update(rotor_records[body.name])
            if body.name in tether_records:
                entry.update(tether_records[body.name])
                if index in keep_lines:
                    (positions, velocities), _ = lines[body.name]
                    run.tether_states.setdefault(body.name, {})[index] = (positions.copy(), velocities.copy())
            record["bodies"][body.name] = entry
        run.records.append(record)
        if index in keep:
            if state_sink is None:
                run.states[index] = state
            else:
                state_sink(index, state)
        if progress is not None:
            progress(record)
    run.final_state = state
    return run
