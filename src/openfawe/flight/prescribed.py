"""Vortex-wake aerodynamics marched along a prescribed rigid-body motion (G4 contract, section 3 item 2).

The body-frame geometry is placed at every motion sample, the panel velocity of the body is
``velocity + angular_velocity x (point - position)`` and the wake stays in the inertial frame, where
the existing ``advance`` interface convects it. The aerodynamic settings are whatever the supplied
backend carries; this module adds no closure, no smoothing and no retry. A step whose solve fails
(``FAILURES``: the solver's non-convergence errors and the domain errors the aerodynamics raises as
``ValueError``, such as an incidence outside the polar table or reversed chordwise flow) stops the
run: the records and the last good state are kept and the failure is recorded with its type. The
set-up (time grid, wake row limit, wind and body-velocity fields) is checked before the first step,
so that a configuration mistake raises instead of being booked as a failed aerodynamic step.

With a ``handover`` (``openfawe.aero.handover.SectionHandover``) the section data, the closure and
the seeding of each step follow the hand-over rule, evaluated from the incidence and sideslip of the
relative wind at the body origin: ``w = 0`` uses the crosswind data with the production closure,
``w > 0`` the blended data with the take-off/landing setting. With zero seeding a failed zero-seeded
solve is retried once from the previous circulation and the retry is recorded together with the
error of the zero-seeded attempt; the step fails only if both fail (then both errors are kept). The
wake history is common to all steps. A state of a hand-over run carries the configuration of its own
step (the solve settings differ between the two sides), so it can be passed back to the backend's
own ``advance`` or probes only on the crosswind side. A step at which the relative wind at the body
origin vanishes has no incidence and therefore no weight: it is a recorded failed step.

With ``SectionHandover(weight_basis='panel')`` (option (c) of the owner item (乙) of the G4 contract, an
opt-in trial) every panel has its own weight, from the flow that reaches it from outside the lifting
surfaces' own circulation system: the ambient wind minus the panel's velocity plus the induction of the
rotor rings, in body axes (``panel_onset_body``), at the panel's 3/4-chord control point and at its
quarter-chord load point; the weight is the larger of the two. The rings are those already in the wake,
at their positions of this step, so the weights are known only after the wake has been convected: the
step function hands the wake advance a chooser for that (``panel_handover_step``). The weight of the body origin is still computed and recorded
(``handover_weight``) and a step without relative wind at the origin still fails; the panel weights are
recorded next to it (``panel_weights`` and the counts derived from them). Seeding, retry and the
re-binding of the state are those of the origin basis, with "the blend is active" read as "some panel
weight is above zero".

With ``rotors`` (``PrescribedRotors``; contract section 3 item 3) the discs follow the body: at every
step, after the lifting surfaces are solved, each disc is placed at the pose, its hub velocity is the
velocity of the body point at the hub, the loads come from the rotor table at the apparent wind of
the hub (ambient wind, the induction of the lifting-surface network and of the other discs' rings,
minus the hub velocity) and one vortex ring per disc is released at the rim into the inertial frame,
where it convects with the wake (``openfawe.rotor.coupled``). The rings released at step ``k`` act on
the lifting surfaces from step ``k+1`` on. The angular speeds are given, not solved. A rotor look-up
or ring release without a solution (for example a turbine state beyond the momentum-theory limit)
fails the step like an aerodynamic failure, with ``failed_module`` saying which part failed. The loads
of the lifting surfaces under ring filaments that cross them are not quantitative (rotor contract,
section 7 item 5); the records count the ring nodes inside the surface core at every step.
What the rotor part leaves out: the gyroscopic moment of the spinning rotors and the reaction moment
of a changing rotor speed are not in the rotor moment (the residual account adds them:
``openfawe.flight.residual``; with equal constant speeds and four rotors of each sense they
cancel, otherwise not); the table is the axial one, also where the stream meets the discs edge-on
or from behind (an axial freestream below the table's lowest value is looked up at that value and
flagged; momentum theory is applied to it as it is); the rings are not truncated by the far-wake
options of the wake configuration, which bound the rows of the lifting surfaces only (their own opt-in
limit is ``PrescribedRotors.retained_ring_steps``).

With ``tether`` (``PrescribedTether``; contract section 3 item 4) a lumped-mass line runs from a
fixed ground point to the bridle point of the body. Between two steps of the aerodynamics the line is
advanced with its own, smaller time step; the body's pose at those times is the interpolation of the
motion samples, and the bridle point follows the last node of the line at every stage of the line's
integrator (``openfawe.tether.bridle``). The interpolation is linear in position and in velocity, so
on a curved path the two are not exactly consistent between samples (on a 130 m circle at 47 m/s with
0.02 s samples: a chord error below a millimetre and a velocity mismatch up to about 0.2 m/s, repeating
at the sample rate, which the line feels as a small forcing of its end). The line sees the ambient
wind only (not the induction of the wakes), and its own gravity and air density, which must agree
with the run's (the density is checked). The tether force acts on the body at the bridle point. The
line starts from the state it is given; without one it starts straight and at rest between its end
points, which is not an equilibrium: the caller settles it first where that matters
(``settle_tether``). The line's time step is checked before the first step against a bound for the
stability of its integrator; a line whose state nevertheless leaves the finite range fails the step
with ``failed_module='tether'`` (the steps recorded before such a failure are then suspect too).

Scope so far: lifting-surface aerodynamics, rotors, tether and bridle. The force residual of the
contract is computed from the records afterwards (``openfawe.flight.residual``).
"""

import time as _time
from dataclasses import dataclass, field, replace

import numpy as np

from ..aero.handover import SectionHandover, incidence_steps, makani_weight, stall_switches
from ..aero.vsm import geometry
from ..aero.methods import AeroMethod
from ..aero.nonlinear import NonlinearSolveError
from ..aero.vsm_wake import advance_vsm_wake, start_vsm_wake
from ..aero.wake import ambient_velocity, kernel_options, ring_induction
from ..aero.vortex import segments_velocity
from ..aero.wing import CirculationConvergenceError
from ..rotor.coupled import rotor_totals, shed_rotor_rings
from ..tether.bridle import Bridle
from ..tether.lumped_line import LumpedTether
from .kinematics import RigidMotion, relative_wind_body

FAILURES = (NonlinearSolveError, CirculationConvergenceError, ValueError, FloatingPointError)


@dataclass(frozen=True)
class PrescribedRotors:
    """Rotor discs carried by the body, with prescribed angular speeds.

    ``discs`` are ``ActuatorDisc`` objects whose centre and axis are in body axes. ``omega_rad_s`` is
    one unsigned angular speed for all discs, one per disc, or a callable ``time_s -> (n,)``.
    ``surface_core_m`` is the core the rings use when they induce at lifting-surface points (None: the
    wake core); ``turbine_extension`` is passed to the ring release (None: momentum theory only);
    ``ring_phase_rad`` turns the rim polygon about the disc axis (the polygon is fixed to the body).
    A speed schedule is checked at every time of the run before the first step.
    ``retained_ring_steps`` (opt-in; None, the default, keeps every ring and every existing result) is a far-wake
    truncation of the ring set: after the release of a step only the rings of the newest that many steps stay, older
    rings are removed whole. The kept slipstream is that many steps of relative travel long; on the axis of a uniform
    vortex cylinder of radius R cut at a length L the axial induction at the disc is L/sqrt(L^2+R^2) of the
    untruncated one. It bounds the cost per step of a run with rotors, as ``WakeConfig.retained_rows`` does for the
    rows of the lifting surfaces.
    ``free_convection_ring_steps`` (opt-in; None, the default, convects every ring freely) is the frozen far wake
    of the rings: only the rings released within the newest that many steps convect with the induced velocity,
    older rings drift with the ambient flow and nothing is deleted (``RingWake.free_convection_steps``), the ring
    counterpart of ``WakeConfig.free_convection_rows`` for the rows of the lifting surfaces. All rings keep inducing.
    0 lets every ring drift with the ambient flow from its release (model A' of the OF-0009 contract, section 8,
    2026-10-22 18:55 BST).
    ``ring_core_m`` (opt-in; None, the default, uses the wake core and keeps every existing result) is the core of the
    ring filaments away from the lifting surfaces: convection of rows and rings and the inflow at the hubs
    (``RingWake.core_m``).
    ``ring_bound_core_m`` (opt-in; None, the default, keeps every existing result) is the core of the bound system of
    the lifting surfaces (the bound vortex and the chordwise and trailing-edge segments of the bound rings) where it
    induces at the ring nodes only (``RingWake.bound_core_m``; model B of the OF-0009 contract, section 8,
    2026-10-22). One absolute radius serves every lifting surface. The rows, the closure and load points, the hub
    inflow and the probes keep their cores.
    ``ring_surface_overlap`` (opt-in; None, the default, keeps every existing result) is the core overlap lambda of the
    ring sheet where the rings induce at lifting-surface points (``RingWake.surface_overlap``, ``surface_sheet``;
    "S" of the OF-0009 contract, section 8, 2026-10-22 18:22 BST): stretches of the sheet longer than the surface core
    over lambda are read as sub-rings near those points. Convection and the hub inflow keep the stored rings.
    ``ring_sheet_drift`` (opt-in; False keeps every existing result) lets the rings that do not convect freely drift
    with the ambient flow plus the sheet's own speed gamma/2 along the flow through their disc at release
    (``RingWake.sheet_drift``; model A'' of the OF-0009 contract, section 8, 2026-10-22 19:16 BST); it needs
    ``free_convection_ring_steps``. ``ring_ambient_spacing`` (opt-in, with ``ring_sheet_drift``) releases the
    circulation over the drift spacing (V_ambient + gamma/2) dt, so that the drifting tube carries the momentum
    strength gamma (``RingWake.ambient_spacing``; review RA-01 of 2026-10-22).
    """
    discs: tuple
    omega_rad_s: object
    nodes_per_ring: int = 16
    surface_core_m: float | None = None
    turbine_extension: str | None = None
    ring_phase_rad: float = 0.0
    retained_ring_steps: int | None = None
    free_convection_ring_steps: int | None = None
    ring_core_m: float | None = None
    ring_bound_core_m: float | None = None
    ring_surface_overlap: float | None = None
    ring_sheet_drift: bool = False
    ring_ambient_spacing: bool = False

    def __post_init__(self):
        discs = tuple(self.discs)
        if not discs or not all(hasattr(disc, "placed") and hasattr(disc, "loads") for disc in discs):
            raise ValueError("rotors need at least one actuator disc")
        if int(self.nodes_per_ring) < 3:
            raise ValueError("a vortex ring needs at least three nodes")
        core = self.surface_core_m
        if core is not None and not (np.isfinite(core) and core > 0.):
            raise ValueError("surface_core_m must be a positive finite radius or None")
        if self.turbine_extension not in (None, "glauert"):
            raise ValueError("turbine_extension must be None or 'glauert'")
        if not np.isfinite(self.ring_phase_rad):
            raise ValueError("ring_phase_rad must be finite")
        if self.ring_core_m is not None and not (np.isfinite(self.ring_core_m) and self.ring_core_m > 0.):
            raise ValueError("ring_core_m must be a positive finite radius or None")
        if self.ring_bound_core_m is not None and not (np.isfinite(self.ring_bound_core_m) and self.ring_bound_core_m > 0.):
            raise ValueError("ring_bound_core_m must be a positive finite radius or None")
        if self.ring_surface_overlap is not None and not (np.isfinite(self.ring_surface_overlap) and self.ring_surface_overlap > 0.):
            raise ValueError("ring_surface_overlap must be a positive finite number or None")
        if not isinstance(self.ring_sheet_drift, (bool, np.bool_)):
            raise ValueError("ring_sheet_drift must be True or False")
        if not isinstance(self.ring_ambient_spacing, (bool, np.bool_)):
            raise ValueError("ring_ambient_spacing must be True or False")
        if self.ring_sheet_drift and self.free_convection_ring_steps is None:
            raise ValueError("ring_sheet_drift needs free_convection_ring_steps (0: every ring drifts from its release)")
        if self.ring_ambient_spacing and not self.ring_sheet_drift:
            raise ValueError("ring_ambient_spacing needs ring_sheet_drift")
        keep = self.retained_ring_steps
        if keep is not None and (isinstance(keep, bool) or not isinstance(keep, (int, np.integer)) or keep < 1):
            raise ValueError("retained_ring_steps must be a positive whole number of steps or None")
        free = self.free_convection_ring_steps
        if free is not None and (isinstance(free, bool) or not isinstance(free, (int, np.integer)) or free < 0):
            raise ValueError("free_convection_ring_steps must be a non-negative whole number of steps or None")
        if free is not None and keep is not None and free > keep:
            raise ValueError("free_convection_ring_steps cannot exceed retained_ring_steps")
        if free is not None:
            object.__setattr__(self, "free_convection_ring_steps", int(free))
        object.__setattr__(self, "discs", discs)
        object.__setattr__(self, "nodes_per_ring", int(self.nodes_per_ring))
        if not callable(self.omega_rad_s):
            self.omegas(0.)                               # a constant: shape and sign are checked here

    def omegas(self, time_s):
        """Angular speed of every disc at ``time_s`` [rad/s], unsigned."""
        value = self.omega_rad_s(float(time_s)) if callable(self.omega_rad_s) else self.omega_rad_s
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = np.full(len(self.discs), float(value))
        if value.shape != (len(self.discs),) or not np.all(np.isfinite(value)) or np.any(value < 0.):
            raise ValueError("one finite non-negative angular speed per disc is required")
        return value


@dataclass
class PrescribedRun:
    """Records of one run; ``status`` is ``complete`` or ``failed`` (then ``error`` says why)."""
    records: list = field(default_factory=list)
    status: str = "complete"
    error: str | None = None
    error_type: str | None = None
    failed_step: int | None = None
    failed_iterations: int | None = None
    failed_module: str | None = None          # 'aerodynamics', 'rotors' or 'tether' on a failed step
    failed_step_aerodynamics: dict | None = None   # rotors or tether failed: the record of that step's solved lifting surfaces
    failed_panels_outside_crosswind_table: list | None = None   # out_of_range='hover' inside the blend: the panels at the failed evaluation
    failed_panel_weights: list | None = None  # weight_basis='panel': the panel weights of the failed step, if they were reached
    tether_state: tuple | None = None         # node positions and velocities at the last recorded step
    tether_states: dict = field(default_factory=dict)   # keep_tether_states: step -> (positions, velocities) copies
    tether_settings: dict | None = None       # what the line was run with (time step, stability bound, gravity, density, ...)
    zero_seed_failure: dict | None = None     # on a failed hand-over step: the failure of the zero-seeded attempt, if any
    motion_consistency: dict | None = None
    step_seconds: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    final_state: object = None


def step_record(index, pose, state, placed, wind, density):
    """Loads of one step in inertial and body axes, about the body origin, with the flight condition."""
    solution = state.solution
    force = solution.force
    moment = solution.moment(placed, reference=pose.position_m)
    origin_wind = ambient_velocity(wind, pose.position_m[None, :], pose.time_s)[0]
    relative, speed, alpha, beta = relative_wind_body(pose, origin_wind)
    record = dict(step=int(index), time_s=float(pose.time_s), position_m=pose.position_m.tolist(),
                  force_inertial_N=force.tolist(), moment_origin_inertial_Nm=moment.tolist(),
                  force_body_N=(pose.rotation.T@force).tolist(), moment_origin_body_Nm=(pose.rotation.T@moment).tolist(),
                  airspeed_m_s=speed, alpha_deg=alpha, beta_deg=beta, relative_wind_body_m_s=relative.tolist(),
                  aero_power_W=float(force@pose.velocity_m_s+moment@pose.angular_velocity_rad_s),
                  dynamic_pressure_Pa=float(.5*density*speed*speed),
                  iterations=int(solution.iterations), residual=float(solution.residual),
                  residual_limit=float(solution.residual_limit))
    return record


@dataclass(frozen=True)
class PrescribedTether:
    """A lumped-mass tether from a fixed ground point to the bridle point of the moving body.

    ``substeps`` line steps are taken per step of the aerodynamics: a count, so the line's time step
    follows the run's (10 gives the contract's 2 ms with the 0.02 s step, 1 ms when that step is
    halved). ``positions`` and ``velocities`` are the node state at the first motion sample (None:
    straight and at rest between the end points).
    """
    line: LumpedTether
    ground_point_m: np.ndarray
    bridle: Bridle
    substeps: int = 10
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.line, LumpedTether) or not isinstance(self.bridle, Bridle):
            raise ValueError("a tether needs a LumpedTether and a Bridle")
        ground = np.array(self.ground_point_m, dtype=float).reshape(3)
        if not np.all(np.isfinite(ground)) or int(self.substeps) < 1:
            raise ValueError("the ground point must be finite and substeps at least one")
        object.__setattr__(self, "ground_point_m", ground)
        object.__setattr__(self, "substeps", int(self.substeps))
        if (self.positions is None) != (self.velocities is None):
            raise ValueError("give both the node positions and velocities, or neither")
        if self.positions is not None:
            nodes = self.line.s.num_nodes
            for name in ("positions", "velocities"):
                value = np.array(getattr(self, name), dtype=float)
                if value.shape != (nodes, 3) or not np.all(np.isfinite(value)):
                    raise ValueError(f"tether {name} must be finite with one row per node")
                object.__setattr__(self, name, value)

    def ends(self, motion):
        """End points of the line as a function of time and of the line's own state."""
        zero = np.zeros(3)

        def ends(time_s, positions, velocities):
            point, velocity = self.bridle.point(motion.pose(time_s), positions[-1], velocities[-1])
            return self.ground_point_m, zero, point, velocity
        ends.uses_state = True
        return ends

    def start(self, motion):
        """Node state at the first motion sample."""
        if self.positions is not None:
            return self.positions.copy(), self.velocities.copy()
        pose = motion.pose(motion.time_s[0])
        pivot, _ = self.bridle.pivot(pose)                # first guess of the line direction: ground to pivot
        towards = self.ground_point_m-pivot
        towards = towards-np.dot(pose.rotation[:, 1], towards)*pose.rotation[:, 1]
        norm = np.linalg.norm(towards)
        if norm < 1e-6:
            raise ValueError("the ground point lies on the span line through the bridle pivot: give the line's start state")
        return self.line.straight_state(self.ground_point_m, pivot+self.bridle.radius_m*towards/norm)

    def settings(self, dt_s):
        """What the line was run with, for the receipts; ``dt_s`` is the step of the aerodynamics."""
        line = self.line
        return dict(substeps=self.substeps, line_step_s=float(dt_s/self.substeps), rk4_step_bound_s=line.rk4_step_bound_s(),
                    nodes=int(line.s.num_nodes), length_m=float(line.p.length_m), gravity_m_s2=float(line.s.gravity_m_s2),
                    gravity_direction="inertial -z", air_density_kg_m3=float(line.s.air_density_kg_m3),
                    ground_point_m=self.ground_point_m.tolist(), bridle_pivot_body_m=self.bridle.pivot_body_m.tolist(),
                    bridle_radius_m=self.bridle.radius_m, started_from="given state" if self.positions is not None else "straight line at rest",
                    wind_on_the_line="ambient wind only", ground_contact="not modelled", length="constant")


def wind_field(wind):
    """The ambient wind as the ``(points, time)`` callable the tether takes."""
    return lambda points, time_s: ambient_velocity(wind, points, time_s)


def settle_tether(tether, motion, wind, *, duration_s):
    """Node state after the line has run for ``duration_s`` with the body held at the first pose of the motion.

    The body is frozen at the pose of the first sample with its velocity and angular velocity set to
    zero; the line relaxes with its own damping. The transverse swing of the line
    is damped by its air drag only and dies out slowly (tens of seconds for a 60 m line in a 12 m/s
    wind); the node speeds of the returned state say how far it got. Returns a ``PrescribedTether``
    that starts from the settled state. For a body that is moving at the start this is still not the
    state of the moving system: a starting transient remains and is the caller's to judge.
    """
    time = float(motion.time_s[0])
    pose = motion.pose(time)
    held = replace(pose, velocity_m_s=np.zeros(3), angular_velocity_rad_s=np.zeros(3))
    zero = np.zeros(3)

    def ends(_, positions, velocities):
        point, velocity = tether.bridle.point(held, positions[-1], velocities[-1])
        return tether.ground_point_m, zero, point, velocity
    ends.uses_state = True
    positions, velocities = tether.start(motion)
    dt = (motion.time_s[1]-motion.time_s[0])/tether.substeps
    field = wind_field(wind)
    for _ in range(int(round(float(duration_s)/dt))):
        positions, velocities = tether.line.step(time, positions, velocities, dt, ends, lambda points, _t: field(points, time))
    return replace(tether, positions=positions, velocities=velocities)


def run_tether_alone(tether, motion, wind, *, record_every=1):
    """Node state after the line alone has followed the whole motion (no aerodynamics of the body).

    The body runs through every sample of ``motion`` as in ``run_prescribed`` (the same sub-steps and
    sub-times); only the line is integrated. Meant for a periodic motion: run the line over some laps,
    then start the coupled run from the returned state, whose body pose at the first sample equals the
    one at the end of the warm-up. Returns ``(PrescribedTether starting from the final state, report)``;
    the report lists the tension at the kite end and the largest node speed at every ``record_every``-th
    sample, so that the caller can judge how far the starting transient has decayed (nothing here
    decides that). The returned tether's start state refers to the first sample of the motion that the
    coupled run will use; the caller must make sure that pose is the one the warm-up ended at.
    """
    if int(record_every) < 1:
        raise ValueError("record_every must be a positive integer")
    bound = tether.line.rk4_step_bound_s()
    step = float(np.max(np.diff(motion.time_s)))/tether.substeps
    if step > bound:
        raise ValueError(f"the line's time step {step:.4g} s is above its stability bound {bound:.4g} s: use more substeps")
    positions, velocities = tether.start(motion)
    ends, field = tether.ends(motion), wind_field(wind)
    report = []
    times = motion.time_s
    for k in range(len(times)-1):
        begin, end = float(times[k]), float(times[k+1])
        for part in range(tether.substeps):
            time = begin+part*(end-begin)/tether.substeps
            until = end if part == tether.substeps-1 else begin+(part+1)*(end-begin)/tether.substeps
            positions, velocities = tether.line.step(time, positions, velocities, until-time, ends, field)
        if (k+1) % int(record_every) == 0 or k == len(times)-2:
            fields = tether_fields(tether, motion.pose(end), positions, velocities, wind)
            report.append(dict(sample=k+1, time_s=end, tension_kite_N=fields["tether_tension_kite_N"],
                               tension_ground_N=fields["tether_tension_ground_N"], slack_segments=fields["tether_slack_segments"],
                               largest_node_speed_m_s=float(np.max(np.linalg.norm(velocities, axis=1))),
                               lowest_node_height_m=fields["tether_lowest_node_height_m"]))
    return replace(tether, positions=positions, velocities=velocities), report


def tether_fields(tether, pose, positions, velocities, wind):
    """Tether loads on the body at one pose, from the node state at that time.

    The force of the line on the body acts at the bridle point; the moment is about the body origin.
    ``tether_power_on_body_W`` is the rate of work of this load on the rigid body (force times origin
    velocity plus moment times angular velocity). Tensions are the spring tensions of the last and of
    the first segment; the force on the body also holds the damping force of the last segment
    (``tether_damping_kite_N``, along the segment, positive while it lengthens) and its bending shear.
    ``tether_ground_force_inertial_N`` is the force of the line on the ground point.
    """
    point, point_velocity = tether.bridle.point(pose, positions[-1], velocities[-1])
    air = ambient_velocity(wind, positions, pose.time_s)
    out = tether.line.forces(positions, velocities, tether.ground_point_m, np.zeros(3), point, point_velocity, air)
    force = np.asarray(out["end"], dtype=float)
    moment = np.cross(point-pose.position_m, force)
    energy = tether.line.energy(positions, velocities, tether.ground_point_m, point)
    return dict(tether_force_inertial_N=force.tolist(), tether_moment_origin_inertial_Nm=moment.tolist(),
                tether_force_body_N=(pose.rotation.T@force).tolist(), tether_moment_origin_body_Nm=(pose.rotation.T@moment).tolist(),
                tether_power_on_body_W=float(force@pose.velocity_m_s+moment@pose.angular_velocity_rad_s),
                tether_tension_kite_N=float(out["tension"][-1]), tether_tension_ground_N=float(out["tension"][0]),
                tether_damping_kite_N=float(out["damping"][-1]),
                tether_tension_largest_N=float(np.max(out["tension"])), tether_slack_segments=int(np.sum(out["tension"] <= 0.)),
                tether_ground_force_inertial_N=np.asarray(out["start"], dtype=float).tolist(),
                tether_aero_force_inertial_N=np.asarray(out["aero"], dtype=float).sum(axis=0).tolist(),
                tether_energy_J=energy, bridle_point_body_m=(pose.rotation.T@(point-pose.position_m)).tolist(),
                tether_lowest_node_height_m=float(positions[:, 2].min()))


def rotor_step(state, config, rotors, pose, wind):
    """Loads of the discs at the pose and the state with this step's rings released; returns ``(state, fields)``.

    The fields give the total rotor force and moment about the body origin (hub forces at their hubs
    plus the axis torques) in inertial and body axes, the rate of work of these loads on the moving body
    (sum of hub force times hub velocity plus axis torque times the body's angular velocity), the shaft
    power of the table (negative while the motors drive the rotors) and, per disc, thrust, power, axial
    freestream, angular speed and whether the look-up was clamped to the table. ``ring_nodes`` counts
    the ring nodes after the release; ``ring_nodes_within_surface_core`` the nodes of the rings already in
    the wake that lie within the surface core of a load point of the lifting surfaces at this step.
    That count is a lower bound on filament crossings: it looks at ring nodes, not at the segments
    between them (0.45 m long for a 16-node ring of radius 1.15 m), and at the load points, not at the
    closure points of the solve; a zero does not show that no filament passed through a core.
    The moment holds the hub forces and the axis torques only (no gyroscopic or reaction moment).
    """
    state_before = state                                  # the rings the solve of this step saw
    discs = [disc.placed(pose.rotation, pose.position_m) for disc in rotors.discs]
    hubs = np.array([disc.centre_m for disc in discs])
    hub_velocity = pose.point_velocity(hubs)
    omegas = rotors.omegas(pose.time_s)
    core = rotors.surface_core_m or config.core_radius_m
    near = 0 if state.rings is None else state.rings.near_surface_count(state.wing.cp, core)
    state, rows = shed_rotor_rings(state, config, discs, omegas, wind, nodes_per_ring=rotors.nodes_per_ring,
                                   hub_velocities=hub_velocity, phase=rotors.ring_phase_rad,
                                   surface_core_m=rotors.surface_core_m, turbine_extension=rotors.turbine_extension,
                                   free_convection_steps=rotors.free_convection_ring_steps, ring_core_m=rotors.ring_core_m,
                                   ring_bound_core_m=rotors.ring_bound_core_m, ring_surface_overlap=rotors.ring_surface_overlap,
                                   ring_sheet_drift=rotors.ring_sheet_drift, ring_ambient_spacing=rotors.ring_ambient_spacing)
    if rotors.retained_ring_steps is not None:
        # The rings of the newest K steps stay: those released at t, t-dt, ..., t-(K-1) dt (half a step of margin).
        horizon = state.time_s-(int(rotors.retained_ring_steps)-0.5)*config.dt_s
        state = replace(state, rings=state.rings.released_since(horizon))
    force, moment = rotor_totals(rows, discs, about=pose.position_m)
    work = sum(float(np.dot(row["force_N"], velocity)+np.dot(row["moment_Nm"], pose.angular_velocity_rad_s))
               for row, velocity in zip(rows, hub_velocity))
    fields = dict(rotor_force_inertial_N=force.tolist(), rotor_moment_origin_inertial_Nm=moment.tolist(),
                  rotor_force_body_N=(pose.rotation.T@force).tolist(), rotor_moment_origin_body_Nm=(pose.rotation.T@moment).tolist(),
                  rotor_power_on_body_W=work, rotor_shaft_power_W=float(sum(row["power_W"] for row in rows)),
                  rotor_thrust_N=[float(row["thrust_N"]) for row in rows], rotor_power_W=[float(row["power_W"]) for row in rows],
                  rotor_axial_freestream_m_s=[float(row["axial_freestream_m_s"]) for row in rows],
                  rotor_omega_rad_s=omegas.tolist(), rotor_table_saturated=[bool(row["saturated"]) for row in rows],
                  ring_nodes=int(state.rings.count), ring_nodes_within_surface_core=int(near), ring_surface_core_m=float(core))
    if rotors.retained_ring_steps is not None:
        fields["ring_retained_steps"] = int(rotors.retained_ring_steps)
    if rotors.free_convection_ring_steps is not None:
        fields["ring_free_convection_steps"] = int(rotors.free_convection_ring_steps)
    if rotors.ring_core_m is not None:
        fields["ring_core_m"] = float(rotors.ring_core_m)
    if rotors.ring_bound_core_m is not None:
        fields["ring_bound_core_m"] = float(rotors.ring_bound_core_m)
    if rotors.ring_surface_overlap is not None:
        fields["ring_surface_overlap"] = float(rotors.ring_surface_overlap)
        fields.update(ring_sheet_fields(state_before, core))
    if rotors.ring_sheet_drift:
        fields["ring_sheet_drift"] = True
    if rotors.ring_ambient_spacing:
        fields["ring_ambient_spacing"] = True
    if rotors.ring_sheet_drift:                       # review AS-04: the released rings per disc
        fields["rotor_ring_circulation_m2_s"] = [float(row["ring_circulation_m2_s"]) for row in rows]
        fields["rotor_ring_spacing_m"] = [float(row["ring_spacing_m"]) for row in rows]
        fields["rotor_sheet_strength_m_s"] = [float(row["sheet_strength_m_s"]) for row in rows]
    return state, fields


def failure_info(error):
    """Type, message and iteration count of a caught step failure, for the records."""
    return dict(type=type(error).__name__, message=str(error)[:300], iterations=getattr(error, "iterations", None))


def handover_step(state, choice, base_config, placed, wind, structural_velocity, time_s):
    """One step under a hand-over choice; returns ``(state, seeding_used, zero_seed_failure)``.

    ``state`` is None for the starting step, which has no previous circulation and always starts
    from zero (``'start'``). Otherwise ``seeding_used`` is ``'previous'``, ``'zero'`` or
    ``'retry_previous'`` (zero seeding failed, the retry from the previous circulation solved);
    ``zero_seed_failure`` is then the ``failure_info`` of the zero-seeded attempt, else None. If the
    retry fails too, its exception is raised with the first failure attached as
    ``zero_seed_failure``. The state is re-bound to the configuration that carries the solve
    settings of this step; the two configurations may differ in the solve settings only, and the
    wake rows, strengths and cores are untouched by the re-binding.
    """
    config = replace(base_config, solve=choice.solve)
    if state is None:
        return start_vsm_wake(placed, choice.polar, config, wind, time_s=time_s,
                              structural_velocity=structural_velocity), "start", None
    if state.config != config:
        if replace(state.config, solve=config.solve) != config:
            raise RuntimeError("hand-over re-binding would change more than the solve settings")
        state = replace(state, config=config)

    def advance(seed):
        return advance_vsm_wake(state, choice.polar, config, wind, next_wing=placed,
                                structural_velocity=structural_velocity, next_time_s=time_s, initial_circulation=seed)
    if not choice.zero_seed:
        return advance(None), "previous", None
    try:
        return advance(np.zeros(placed.panels)), "zero", None
    except FAILURES as first:
        info = failure_info(first)
    try:
        return advance(None), "retry_previous", info
    except FAILURES as second:
        second.zero_seed_failure = info
        raise


def panel_onset_body(pose, points, wind, structural_velocity, ring_velocity, time_s):
    """Flow reaching each panel from outside the lifting surfaces' own circulation system, in body axes.

    Ambient wind minus the velocity of the body point plus the induction of the rotor rings, at
    ``points`` (inertial), rotated into the case-file body axes with the expression
    ``relative_wind_body`` uses for the body origin, one panel at a time, so that a uniform stream past a
    body that does not rotate gives every panel the very numbers of the origin.
    """
    relative = (ambient_velocity(wind, points, time_s)-ambient_velocity(structural_velocity, points, time_s)
                + np.asarray(ring_velocity, dtype=float))
    return np.array([pose.rotation.T@row for row in relative])


def ring_sheet_fields(state, core):
    """Record fields of the sheet refinement (review RS-02 of contract section 8, 2026-10-22): the rings refined at the
    closure and load points of the state's lifting surfaces, their largest sub-ring count and largest stretch."""
    rings = getattr(state, "rings", None)
    summary = dict(refined_rings=0, max_subrings=0, max_stretch_m=0.0)
    if rings is not None and getattr(rings, "surface_overlap", None) is not None:
        points = np.concatenate((np.asarray(geometry(state.wing)[0]).reshape(-1, 3), np.asarray(state.wing.cp).reshape(-1, 3)))
        summary = rings.surface_sheet_summary(points, core)
    return dict(ring_sheet_refined_rings=int(summary["refined_rings"]), ring_sheet_max_subrings=int(summary["max_subrings"]),
                ring_sheet_max_stretch_m=float(summary["max_stretch_m"]))


def ring_induction_by_disc(rings, points, config, discs, sheet_points=None):
    """The induction of ``ring_induction`` split by the disc that released each ring: ``(discs, n, 3)``.

    Same kernel, same core (the rings' surface core when they declare one) and same options as
    ``aero.wake.ring_induction``, one call per disc on that disc's ring segments; the parts add up to
    the whole to rounding. Diagnostic only: nothing in a step uses it. With the sheet refinement the rings to refine
    are chosen at ``sheet_points`` when given (as in the step: closure and load points together).
    """
    points = np.asarray(points, dtype=float)
    parts = np.zeros((int(discs),)+points.shape)
    if rings is None or len(rings.edges) == 0:
        return parts
    core = getattr(rings, "surface_core_m", None)
    core = config.core_radius_m if core is None else float(core)
    nodes, edges, gamma = rings.nodes, rings.edges, rings.circulation
    source = np.asarray(rings.source)[np.asarray(rings.edges)[:, 0]]
    if getattr(rings, "surface_overlap", None) is not None:
        chosen = points if sheet_points is None else np.asarray(sheet_points, dtype=float)
        nodes, edges, gamma, source = rings.surface_sheet(chosen.reshape(-1, 3), core)
    for disc in range(int(discs)):
        keep = source == disc
        if np.any(keep):
            parts[disc] = segments_velocity(points, np.asarray(nodes, dtype=float), np.asarray(edges, dtype=np.int64)[keep],
                                            np.asarray(gamma, dtype=float)[keep], model=config.core_model,
                                            core_radius=core, **kernel_options(config))
    return parts


def panel_handover_step(state, handover, origin_weight, base_config, placed, pose, wind, structural_velocity, onset_record=None):
    """One step with per-panel hand-over weights; returns ``(state, choice, seeding_used, zero_seed_failure)``.

    A panel's weight is the larger of the weights at its control point and at its load point.
    The starting step (``state`` None) has no rings yet: its weights come from the ambient flow alone.
    Later steps let the wake advance call back once the rings are at their new positions. Seeding and
    retry as in ``handover_step``; the retry repeats the convection and therefore the choice, which is
    the same. A failure carries the choice of the step, when it was reached, as ``handover_choice``.

    ``onset_record`` (default None: nothing extra is computed) is a dictionary that receives the flows
    the weights of this step were taken from, in body axes: ``control`` and ``load`` (what
    ``select_panels`` was given) and the same without the rings' induction.
    """
    def select(time_s, control_points, ring_control, load_points, ring_load):
        control = panel_onset_body(pose, control_points, wind, structural_velocity, ring_control, time_s)
        load = panel_onset_body(pose, load_points, wind, structural_velocity, ring_load, time_s)
        if onset_record is not None:
            onset_record.update(
                control=control, load=load,
                control_without_rings=panel_onset_body(pose, control_points, wind, structural_velocity,
                                                       np.zeros_like(control_points), time_s),
                load_without_rings=panel_onset_body(pose, load_points, wind, structural_velocity, np.zeros_like(load_points), time_s))
        return handover.select_panels(control, origin_weight, load)
    if state is None:
        points = geometry(placed)[0]
        choice = select(pose.time_s, points, np.zeros_like(points), placed.cp, np.zeros_like(placed.cp))
        try:
            return (start_vsm_wake(placed, choice.polar, replace(base_config, solve=choice.solve), wind, time_s=pose.time_s,
                                   structural_velocity=structural_velocity), choice, "start", None)
        except FAILURES as error:
            error.handover_choice = choice
            raise
    if state.config != base_config:
        if replace(state.config, solve=base_config.solve) != base_config:
            raise RuntimeError("hand-over re-binding would change more than the solve settings")
        state = replace(state, config=base_config)
    seen = {}

    def advance(zero_allowed):
        def chooser(*onset):
            choice = seen["choice"] = select(*onset)
            return choice.polar, choice.solve, np.zeros(placed.panels) if zero_allowed and choice.zero_seed else None
        return advance_vsm_wake(state, None, base_config, wind, next_wing=placed, structural_velocity=structural_velocity,
                                next_time_s=pose.time_s, chooser=chooser)
    try:
        new = advance(True)
        return new, seen["choice"], "zero" if seen["choice"].zero_seed else "previous", None
    except FAILURES as first:
        choice = seen.get("choice")
        first.handover_choice = choice
        if choice is None or not choice.zero_seed:
            raise
        info = failure_info(first)
    first_choice, first_most = choice, getattr(choice.polar, "most_out_of_range", 0)
    seen.clear()

    def carry(again):                                     # the retry builds its section data anew: keep the largest count
        if hasattr(again.polar, "most_out_of_range"):
            again.polar.most_out_of_range = max(again.polar.most_out_of_range, first_most)
        return again
    try:
        new = advance(False)
        return new, carry(seen["choice"]), "retry_previous", info
    except FAILURES as second:
        second.handover_choice = carry(seen["choice"]) if "choice" in seen else first_choice
        second.zero_seed_failure = info
        raise


def panel_onset_record(onset, solved, placed, pose, rotors):
    """The record entry of ``panel_onset_steps``: flows in body axes, one row per panel, and the rings by disc.

    ``solved`` is the state of the step before this step's rings are released: its rings are the ones
    whose induction the weights saw (convected to this step's time).
    """
    entry = {name: np.asarray(onset[name], dtype=float).tolist() for name in ("control", "load", "control_without_rings", "load_without_rings")}
    entry["axes"] = "case-file body axes (x chordwise towards the trailing edge, y spanwise, z up); m/s; one row per panel"
    rings = getattr(solved, "rings", None)
    if rotors is not None and rings is not None:
        discs = len(rotors.discs)
        to_body = pose.rotation.T
        both = np.concatenate((np.asarray(geometry(placed)[0]).reshape(-1, 3), np.asarray(placed.cp).reshape(-1, 3)))
        for name, points in (("control", geometry(placed)[0]), ("load", placed.cp)):
            parts = ring_induction_by_disc(rings, points, solved.config, discs, sheet_points=both)
            entry[f"rings_by_disc_{name}"] = np.einsum("ij,dnj->dni", to_body, parts).tolist()
        entry["rings_by_disc_are"] = "induction of the rings released by each disc (disc order of the rotor set), same core and rings as the weights saw"
    return entry


def blend_active(choice):
    """Whether a step on the hover side still consults the crosswind table (so that panels can leave its range).

    Origin basis: the one weight lies strictly between 0 and 1. Panel basis: some weight is above zero (the
    step is not the production model) and some weight is below one (that panel uses the crosswind table,
    alone or in a mixture) - weights of zeros and ones only are included.
    """
    if choice.panel_weights is None:
        return 0. < choice.weight < 1.
    weights = np.asarray(choice.panel_weights)
    return bool(np.any(weights > 0.) and np.any(weights < 1.))


def handover_record(state, placed, choice, seeding_used, zero_seed_failure=None):
    """Hand-over fields of a step record: weight, closure, seeding and the roughness indicators.

    ``closure`` is read from the solution, not from the choice. ``main_panels_not_attached`` counts the
    main-wing panels in any regime other than attached (post-stall, reverse, no inflow).
    With ``BlendedPolars(out_of_range='hover')`` inside the blend, ``panels_outside_crosswind_table`` lists
    the panels the hover data stood in for at the last closure evaluation of the step (the one at the
    converged circulation) and ``most_panels_outside_crosswind_table`` the largest count over every
    evaluation of the step, rejected line-search trials and a zero-seed attempt before a retry included;
    both are None when the option is off or the step did not consult the crosswind table on the hover side
    (origin basis: weight 0 or 1; see ``blend_active``).
    With ``weight_basis='panel'`` that condition is "some panel weight above zero and some below one"
    (a panel of weight zero outside the table is then listed too: the option gives it the hover data), and on
    a step solved by the retry the largest count includes the zero-seeded attempt, as with the origin basis.
    ``handover_weight`` stays the weight of the body origin, and ``panel_weights`` (one per
    panel, in panel order) with their smallest and largest value and the counts of panels on the crosswind
    data alone (weight 0), in the blend, and on the hover data alone (weight 1) are added; with the origin
    basis those fields are None.
    """
    solution = state.solution
    main = placed.wings[0].panels if hasattr(placed, "wings") else placed.panels
    switches, not_attached = stall_switches(solution, main)
    regime = getattr(solution, "panel_regime", None)
    weight = getattr(solution, "circulation_weight", None)
    polar = choice.polar
    outside = getattr(polar, "last_out_of_range", None)          # BlendedPolars with out_of_range='hover' only
    if outside is not None and (getattr(polar, "out_of_range", "error") != "hover" or not blend_active(choice)):
        outside = None
    per_panel = None if choice.panel_weights is None else np.asarray(choice.panel_weights, dtype=float)
    return dict(handover_weight=float(choice.weight), closure=solution.closure, seeding=seeding_used,
                zero_seed_failure=zero_seed_failure,
                stall_switches=switches, main_panels_not_attached=not_attached, incidence_steps=incidence_steps(solution, placed),
                regime_counts=None if regime is None else [int(np.sum(np.asarray(regime) == k)) for k in range(4)],
                panels_without_circulation=None if weight is None else int(np.sum(np.asarray(weight) == 0.)),
                panels_outside_crosswind_table=None if outside is None else [int(k) for k in outside],
                most_panels_outside_crosswind_table=None if outside is None else int(polar.most_out_of_range),
                panel_weights=None if per_panel is None else per_panel.tolist(),
                panel_weight_range=None if per_panel is None else [float(per_panel.min()), float(per_panel.max())],
                panels_by_weight=None if per_panel is None else dict(
                    crosswind_only=int(np.sum(per_panel == 0.)), blend=int(np.sum((per_panel > 0.) & (per_panel < 1.))),
                    hover_only=int(np.sum(per_panel == 1.))))


def run_prescribed(backend, wing, polar, motion, wind, *, handover=None, rotors=None, tether=None, steps=None, keep_states=(),
                   progress=None, attitude_scheme="interval", panel_onset_steps=None, keep_tether_states=()):
    """March ``backend`` along ``motion``; ``wing`` is the geometry in body axes.

    With ``handover`` the section data come from it and ``polar`` must be None; the backend must be the
    VSM free-wake method, and its solve settings must be the crosswind settings of the hand-over, so
    that a step with weight zero is the backend's own step.

    With ``rotors`` (``PrescribedRotors``) the discs follow the body and release their rings at every
    step; see the module text. A stored state (``keep_states``, ``final_state``) then carries the rings
    released up to and including its own step (with ``retained_ring_steps``, those of the newest that many steps).

    With ``tether`` (``PrescribedTether``) the line is advanced between the steps and its load on the
    body is recorded; the node state of the last solved step is in ``tether_state``.

    ``wind`` is a uniform (3,) vector or a callable ``(points_m, time_s)`` in the inertial frame.
    The motion samples must be spaced by the wake time step; step ``k`` uses sample ``k``.
    ``progress`` is an optional callable receiving each record. The self-consistency report of the
    motion samples is stored with the run (``motion_consistency``, with the attitude check of
    ``attitude_scheme``, see ``RigidMotion.consistency``); judging it is the caller's task.

    ``panel_onset_steps`` (default None: off, nothing extra computed or recorded) is a pair
    ``(first, last)`` of step numbers; on those steps, with ``weight_basis='panel'``, the record gets
    ``panel_onset``: the flow the panel weights were taken from at the control and at the load points in
    body axes, the same without the rings, and the rings' induction split by releasing disc (G4 contract
    section 8, 2026-10-07 11:09). A diagnostic: it changes no step.

    ``keep_tether_states`` (default empty: nothing kept) lists step numbers whose node state of the line is
    copied into ``tether_states`` (``step -> (positions, velocities)``); it changes no step either.
    """
    if not isinstance(motion, RigidMotion):
        raise ValueError("motion must be a RigidMotion")
    if handover is not None:
        if not isinstance(handover, SectionHandover) or polar is not None:
            raise ValueError("with a hand-over pass a SectionHandover and polar=None")
        if backend.method != AeroMethod.VSM_FREE_WAKE:
            raise ValueError("the hand-over needs the VSM free-wake method")
        if backend.wake.solve != handover.low_solve:
            raise ValueError("the backend's solve settings must equal the crosswind settings of the hand-over")
    if panel_onset_steps is not None:
        if handover is None or handover.weight_basis != "panel":
            raise ValueError("the panel onset record needs a hand-over with weight_basis='panel'")
        first_onset, last_onset = (int(k) for k in panel_onset_steps)
        if not 0 <= first_onset <= last_onset:
            raise ValueError("panel_onset_steps must be (first, last) with 0 <= first <= last")
    if rotors is not None:
        if not isinstance(rotors, PrescribedRotors):
            raise ValueError("rotors must be a PrescribedRotors")
        if backend.method not in (AeroMethod.VSM_FREE_WAKE, AeroMethod.FREE_WAKE):
            raise ValueError("rotors need a free-wake method (their rings join the wake network)")
    dt = backend.wake.dt_s
    if tether is not None:
        if not isinstance(tether, PrescribedTether):
            raise ValueError("tether must be a PrescribedTether")
        bound = tether.line.rk4_step_bound_s()
        if dt/tether.substeps > bound:
            raise ValueError(f"the line's time step {dt/tether.substeps:.4g} s is above its stability bound {bound:.4g} s: use more substeps")
        if tether.line.s.air_density_kg_m3 != backend.wake.density_kg_m3:
            raise ValueError("the line and the aerodynamics use different air densities")
    count = len(motion.time_s)-1 if steps is None else int(steps)
    if count < 0 or count > len(motion.time_s)-1:
        raise ValueError("more steps requested than motion samples")
    # Set-up checks, outside the per-step failure handling. The time grid uses the tolerance the wake
    # advance itself applies to the requested time.
    times = motion.time_s[:count+1]
    if count and np.any(np.abs(times[:-1]+dt-times[1:]) > 8*np.finfo(float).eps*np.maximum(1., np.abs(times[1:]))):
        raise ValueError("motion samples must be spaced by the wake time step")
    if count > backend.wake.max_rows and backend.wake.retained_rows is None:
        raise ValueError("the wake row limit of the backend is below the number of steps")
    first = motion.pose(times[0])
    placed = wing.transformed(first.rotation, first.position_m)
    ambient_velocity(wind, placed.cp, first.time_s)
    ambient_velocity(motion.structural_velocity, placed.cp, first.time_s)
    if rotors is not None:
        for time in times:                                # a bad speed schedule is a set-up mistake, not a failed step
            rotors.omegas(time)
    density = backend.wake.density_kg_m3
    keep = {int(k) for k in keep_states}
    keep_lines = {int(k) for k in keep_tether_states}
    run = PrescribedRun(motion_consistency=motion.consistency(attitude_scheme))
    state = None
    line_state = line_ends = line_wind = None
    if tether is not None:
        line_state, line_ends, line_wind = tether.start(motion), tether.ends(motion), wind_field(wind)
        ambient_velocity(wind, line_state[0], first.time_s)
        run.tether_settings = tether.settings(dt)
    for index in range(count+1):
        tick = _time.perf_counter()
        pose = motion.pose(motion.time_s[index])
        placed = wing.transformed(pose.rotation, pose.position_m)
        choice = seeding_used = zero_seed_failure = None
        onset = {} if panel_onset_steps is not None and first_onset <= index <= last_onset else None
        module, before = "aerodynamics", state
        try:
            if handover is not None:
                origin_wind = ambient_velocity(wind, pose.position_m[None, :], pose.time_s)[0]
                _, speed, alpha, beta = relative_wind_body(pose, origin_wind)
                if alpha is None:
                    raise ValueError("no relative wind at the body origin: the hand-over weight is undefined")
                if handover.weight_basis == "panel":
                    state, choice, seeding_used, zero_seed_failure = panel_handover_step(
                        state if index else None, handover, makani_weight(alpha, beta, handover.stall), backend.wake, placed, pose,
                        wind, motion.structural_velocity, onset)
                else:
                    choice = handover.select(alpha, beta)
                    state, seeding_used, zero_seed_failure = handover_step(
                        state if index else None, choice, backend.wake, placed, wind, motion.structural_velocity, pose.time_s)
            elif index == 0:
                state = backend.start(placed, polar, wind, time_s=pose.time_s,
                                      structural_velocity=motion.structural_velocity)
            else:
                state = backend.advance(state, polar, wind, next_wing=placed,
                                        structural_velocity=motion.structural_velocity, next_time_s=pose.time_s)
            solved, rotor_fields, tether_record = state, None, None    # the loads of this step come from the state before the release
            if rotors is not None:
                module = "rotors"
                state, rotor_fields = rotor_step(state, backend.wake, rotors, pose, wind)
            if tether is not None:
                module = "tether"
                if index:
                    positions, velocities = line_state
                    begin, end = motion.time_s[index-1], motion.time_s[index]
                    for part in range(tether.substeps):
                        # Sub-times from the sample itself, the last one ending exactly on the next sample: no drift
                        # of the stage times past the motion's last sample.
                        time = begin+part*(end-begin)/tether.substeps
                        until = end if part == tether.substeps-1 else begin+(part+1)*(end-begin)/tether.substeps
                        positions, velocities = tether.line.step(time, positions, velocities, until-time, line_ends, line_wind)
                    line_state = (positions, velocities)
                tether_record = tether_fields(tether, pose, *line_state, wind)
        except FAILURES as error:
            run.status, run.failed_step, run.error_type = "failed", index, type(error).__name__
            run.failed_module = module
            if choice is None:
                choice = getattr(error, "handover_choice", None)      # panel basis: the choice is made inside the step
            if choice is not None and getattr(choice.polar, "out_of_range", "error") == "hover" and blend_active(choice):
                run.failed_panels_outside_crosswind_table = [int(k) for k in choice.polar.last_out_of_range]
            if choice is not None and choice.panel_weights is not None:
                run.failed_panel_weights = np.asarray(choice.panel_weights, dtype=float).tolist()
            if module in ("rotors", "tether"):            # the lifting surfaces of this step did solve: keep what they gave
                kept = step_record(index, pose, state, placed, wind, density)
                if choice is not None:
                    kept.update(handover_record(state, placed, choice, seeding_used, zero_seed_failure))
                run.failed_step_aerodynamics = kept
            state = before                                # the last good state, also when the rotors failed after the solve
            run.error = f"{type(error).__name__}: {str(error)[:300]}"
            run.failed_iterations = getattr(error, "iterations", None)
            run.zero_seed_failure = getattr(error, "zero_seed_failure", None)
            break
        run.step_seconds.append(_time.perf_counter()-tick)
        record = step_record(index, pose, solved, placed, wind, density)
        if choice is not None:
            record.update(handover_record(solved, placed, choice, seeding_used, zero_seed_failure))
        if onset:
            record["panel_onset"] = panel_onset_record(onset, solved, placed, pose, rotors)
        if rotor_fields is not None:
            record.update(rotor_fields)
        if tether_record is not None:
            record.update(tether_record)
            run.tether_state = line_state
            if index in keep_lines:
                run.tether_states[index] = (line_state[0].copy(), line_state[1].copy())
        run.records.append(record)
        if index in keep:
            run.states[index] = state
        if progress is not None:
            progress(record)
    run.final_state = state
    return run
