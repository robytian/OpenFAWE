"""From a ``Spec`` to OpenFAWE objects, and the step-by-step march.

``march`` is the loop of ``openfawe.flight.prescribed.run_prescribed`` for the options the studio offers (no section
hand-over, no tether), written with the library's own public functions - ``AeroBackend.start``/``advance``,
``rotor_step``, ``step_record`` - so that every step is the library's step and can be shown as soon as it is solved;
``tests/test_engine.py`` checks that its records equal ``run_prescribed``'s bit for bit. Nothing here computes
aerodynamics of its own: ``coefficients`` only divides recorded loads by the dynamic pressure and the reference values.
"""
import time as _time
from dataclasses import dataclass, replace

import numpy as np

from . import resources
from .inflow import Inflow, uniform_wind
from .spec import AIRCRAFT, COMPUTE, TREE

resources.ensure_library()


@dataclass
class Built:
    spec: object
    wing: object                # body axes
    polar: object
    backend: object
    motion: object
    wind: object                # (3,) vector or Inflow
    rotors: object              # PrescribedRotors or None
    area_m2: float
    chord_m: float
    case_file: str
    density_kg_m3: float
    handover: object = None     # SectionHandover (Makani loop) or None
    tether: object = None       # PrescribedTether or None
    ground_point: object = None # where the ground station is drawn (the tether's ground point when it is computed)

    @property
    def count(self):
        return len(self.motion.time_s)-1


@dataclass
class Step:
    index: int
    record: dict
    seconds: float
    pose: object
    placed: object              # the wing at this step's pose
    solved: object              # the state whose loads the record holds
    state: object               # the state after this step's ring release (what the next step starts from)


@dataclass
class Failure:
    index: int
    module: str
    error: str


def rotation_for(alpha_deg, beta_deg):
    """Body-to-inertial rotation of the fixed aircraft: the relative wind along +x has the body components of
    ``openfawe.aero.reference.inflow(1, alpha, beta)`` (Rz(-beta) Ry(alpha))."""
    a, b = np.radians(float(alpha_deg)), np.radians(float(beta_deg))
    ry = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])
    rz = np.array([[np.cos(-b), -np.sin(-b), 0.], [np.sin(-b), np.cos(-b), 0.], [0., 0., 1.]])
    return rz@ry


def wake_options(spec):
    from openfawe.aero.vortex import CudaSettings, TreeSettings
    backend = COMPUTE[spec.compute]["backend"]
    options = dict(velocity_backend=backend)
    if backend in ("tree", "cuda_tree"):
        options["tree"] = TreeSettings(**TREE)
    if backend == "cuda":
        options["cuda"] = CudaSettings("mixed", 128)
    if backend == "cuda_tree":
        options["cuda"] = CudaSettings("mixed", 128, suspect="gpu64")
    return options


def build_wind(spec):
    if not spec.shear_exponent and spec.gust == "none":
        return uniform_wind(spec.wind_m_s)
    return Inflow(spec.wind_m_s, shear_exponent=spec.shear_exponent, reference_height_m=spec.shear_reference_height_m,
                  gust=spec.gust, gust_amplitude_m_s=spec.gust_amplitude_m_s, gust_start_s=spec.gust_start_s,
                  gust_duration_s=spec.gust_duration_s)


def makani_loop(spec):
    """Makani's nominal crosswind loop for the wind speed (openfawe.flight.crosswind_loop, as OF-0007 A5 / OF-0009 C1)."""
    from openfawe.flight import crosswind_loop as cl
    entries, _, playbook = cl.load_playbook(resources.path(resources.PLAYBOOK))
    constants = cl.LoopConstants.from_case(playbook)
    return cl.NominalLoop(cl.playbook_entry(entries, float(spec.wind_m_s)), [float(spec.wind_m_s), 0.0, 0.0], constants)


def loop_steps_per_lap(loop, dt_s):
    return max(8, int(round(loop.period_s()/float(dt_s))))


def build_motion(spec, loop=None):
    from openfawe.flight.kinematics import circular_motion, uniform_translation
    if spec.motion == "makani_loop":
        steps_per_lap = loop_steps_per_lap(loop, spec.dt_s)
        laps = max(1, int(np.ceil(int(spec.steps)/steps_per_lap)))
        motion, _ = loop.motion(laps, steps_per_lap)
        return motion
    times = np.arange(int(spec.steps)+1)*float(spec.dt_s)
    if spec.motion == "fixed":
        return uniform_translation(times, (0., 0., 0.), rotation=rotation_for(spec.alpha_deg, spec.beta_deg),
                                   origin=(0., 0., float(spec.height_m)))
    base = circular_motion(times, spec.loop_radius_m, spec.loop_speed_m_s, spec.wind_m_s, spec.alpha_deg)
    return replace(base, position_m=base.position_m+np.array([0., 0., float(spec.height_m)]),
                   description=base.description+f", centre {float(spec.height_m):g} m above the ground")


def build(spec):
    """The library objects of one run (raises ValueError on a choice the library refuses)."""
    from openfawe.aero.aircraft import load_aircraft
    from openfawe.aero.handover import CROSSWIND_SOLVE
    from openfawe.aero.lifting_surface import LiftingSurfaceConfig
    from openfawe.aero.ll_straight import LLStraightConfig
    from openfawe.aero.methods import AeroBackend
    from openfawe.aero.wake import WakeConfig
    spec.validate()
    case = AIRCRAFT[spec.aircraft]
    definition = load_aircraft(resources.path(case["file"]))
    steps = int(spec.steps)
    loop = makani_loop(spec) if spec.motion == "makani_loop" else None
    dt = float(spec.dt_s) if loop is None else loop.period_s()/loop_steps_per_lap(loop, spec.dt_s)
    config = WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=float(spec.core_radius_m),
                        convection_integrator="euler", core_evolution="constant", solve=CROSSWIND_SOLVE,
                        free_convection_rows=spec.free_convection_rows, retained_rows=spec.retained_rows,
                        **wake_options(spec))
    # The lattice and straight-line settings are those of the six-method comparison (OF-0006-body20260930,
    # run_body_methods.run_one); the other methods ignore them.
    backend = AeroBackend(spec.method, wake=config, lifting_surface=LiftingSurfaceConfig(
        chordwise_panels=8, chordwise_spacing="cosine", camber_normal="reconstructed_slope", wake_length_m=1000.,
        wake_release="previous", gamma_derivative="backward_euler"), ll_straight=LLStraightConfig(wake_length_m=1000.))
    rotors = None
    if spec.rotors:
        from openfawe.flight.prescribed import PrescribedRotors
        from openfawe.rotor.actuator_disc import load_rotor_set
        from openfawe.rotor.table import RotorTable
        discs = load_rotor_set(resources.path(resources.ROTOR_SET),
                               RotorTable.from_makani_json(resources.path(resources.ROTOR_TABLE)))
        rotors = PrescribedRotors(discs, float(spec.rotor_omega_rad_s), nodes_per_ring=int(spec.ring_nodes),
                                  surface_core_m=float(spec.ring_surface_core_m),
                                  retained_ring_steps=spec.retained_ring_steps,
                                  free_convection_ring_steps=spec.free_convection_ring_steps)
    motion = build_motion(spec, loop)
    motion = replace(motion, time_s=motion.time_s[:steps+1], position_m=motion.position_m[:steps+1],
                     rotation=motion.rotation[:steps+1], velocity_m_s=motion.velocity_m_s[:steps+1],
                     angular_velocity_rad_s=motion.angular_velocity_rad_s[:steps+1],
                     acceleration_m_s2=None if motion.acceleration_m_s2 is None else motion.acceleration_m_s2[:steps+1],
                     angular_acceleration_rad_s2=None if motion.angular_acceleration_rad_s2 is None
                     else motion.angular_acceleration_rad_s2[:steps+1]) if loop is not None else motion
    wind = build_wind(spec) if loop is None else loop.wind_world()
    polar, handover = definition.polar, None
    if loop is not None:                    # the section hand-over of A5/C1: crosswind tables, XFOIL at high incidence
        from openfawe.aero.handover import SectionHandover
        high = load_aircraft(resources.path(AIRCRAFT["m600_xfoil"]["file"]))
        if not (np.array_equal(high.wing.cp, definition.wing.cp) and np.array_equal(high.wing.dl, definition.wing.dl)):
            raise ValueError("the Makani loop needs an aircraft with the XFOIL partner's geometry (wing and tails)")
        handover, polar = SectionHandover(definition.polar, high.polar), None
    tether, ground = None, None
    if spec.motion == "fixed":
        ground = fixed_ground_point(spec)
    elif loop is not None:
        ground = np.asarray(loop.ground_point_world(), dtype=float)
    if spec.tether and ground is not None:
        from openfawe.flight.prescribed import PrescribedTether
        from openfawe.tether.bridle import load_bridle
        from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings
        line = LumpedTether(TetherProperties(), TetherSettings(air_density_kg_m3=config.density_kg_m3))
        tether = PrescribedTether(line, ground, load_bridle(resources.path(resources.RIGID_BODY)), substeps=int(spec.tether_substeps))
    return Built(spec=spec, wing=definition.wing, polar=polar, backend=backend, motion=motion,
                 wind=wind, rotors=rotors, area_m2=case["area_m2"], chord_m=case["chord_m"],
                 case_file=case["file"], density_kg_m3=config.density_kg_m3, handover=handover, tether=tether,
                 ground_point=ground)


TETHER_ELEVATION_DEG = 30.0


def fixed_ground_point(spec):
    """The ground point of the fixed aircraft's line: upwind and below, at the line's length plus the bridle radius
    from the body origin along the elevation 30 deg (or lower, when the aircraft flies lower than that allows)."""
    from openfawe.tether.lumped_line import TetherProperties
    reach = TetherProperties().length_m+4.786
    elevation = np.radians(TETHER_ELEVATION_DEG)
    height = float(spec.height_m)
    if height < reach*np.sin(elevation):
        elevation = np.arcsin(max(0.0, height)/reach)
    return np.array([-reach*np.cos(elevation), 0.0, height-reach*np.sin(elevation)])


def march(built, *, should_stop=None):
    """Yield one ``Step`` per solved step (a ``Failure`` ends the march), as ``run_prescribed`` makes them (its
    hand-over branch with the body-origin weight, its rotor and tether branches)."""
    from openfawe.aero.wake import ambient_velocity
    from openfawe.flight.kinematics import relative_wind_body
    from openfawe.flight.prescribed import (FAILURES, handover_record, handover_step, rotor_step, step_record,
                                            tether_fields, wind_field)
    backend, wing, polar, motion, wind, rotors = built.backend, built.wing, built.polar, built.motion, built.wind, built.rotors
    handover, tether = built.handover, built.tether
    dt = backend.wake.dt_s
    count = built.count
    if tether is not None:
        bound = tether.line.rk4_step_bound_s()
        if dt/tether.substeps > bound:
            raise ValueError(f"the line's time step {dt/tether.substeps:.4g} s is above its stability bound {bound:.4g} s: use more substeps")
        if tether.line.s.air_density_kg_m3 != backend.wake.density_kg_m3:
            raise ValueError("the line and the aerodynamics use different air densities")
    times = motion.time_s[:count+1]
    # the set-up checks of run_prescribed
    if count and np.any(np.abs(times[:-1]+dt-times[1:]) > 8*np.finfo(float).eps*np.maximum(1., np.abs(times[1:]))):
        raise ValueError("motion samples must be spaced by the wake time step")
    if count > backend.wake.max_rows and backend.wake.retained_rows is None:
        raise ValueError("the wake row limit of the backend is below the number of steps")
    first = motion.pose(times[0])
    placed = wing.transformed(first.rotation, first.position_m)
    ambient_velocity(wind, placed.cp, first.time_s)
    ambient_velocity(motion.structural_velocity, placed.cp, first.time_s)
    if rotors is not None:
        for time in times:
            rotors.omegas(time)
    density = backend.wake.density_kg_m3
    state = None
    line_state = line_ends = line_wind = None
    if tether is not None:
        line_state, line_ends, line_wind = tether.start(motion), tether.ends(motion), wind_field(wind)
        ambient_velocity(wind, line_state[0], first.time_s)
    for index in range(count+1):
        if should_stop is not None and should_stop():
            return
        tick = _time.perf_counter()
        pose = motion.pose(motion.time_s[index])
        placed = wing.transformed(pose.rotation, pose.position_m)
        choice = seeding_used = zero_seed_failure = None
        module = "aerodynamics"
        try:
            if handover is not None:
                origin_wind = ambient_velocity(wind, pose.position_m[None, :], pose.time_s)[0]
                _, speed, alpha, beta = relative_wind_body(pose, origin_wind)
                if alpha is None:
                    raise ValueError("no relative wind at the body origin: the hand-over weight is undefined")
                choice = handover.select(alpha, beta)
                state, seeding_used, zero_seed_failure = handover_step(
                    state if index else None, choice, backend.wake, placed, wind, motion.structural_velocity, pose.time_s)
            elif index == 0:
                state = backend.start(placed, polar, wind, time_s=pose.time_s, structural_velocity=motion.structural_velocity)
            else:
                state = backend.advance(state, polar, wind, next_wing=placed,
                                        structural_velocity=motion.structural_velocity, next_time_s=pose.time_s)
            solved, rotor_fields, tether_record = state, None, None
            if rotors is not None:
                module = "rotors"
                state, rotor_fields = rotor_step(state, backend.wake, rotors, pose, wind)
            if tether is not None:
                module = "tether"
                if index:
                    positions, velocities = line_state
                    begin, end = motion.time_s[index-1], motion.time_s[index]
                    for part in range(tether.substeps):
                        time = begin+part*(end-begin)/tether.substeps
                        until = end if part == tether.substeps-1 else begin+(part+1)*(end-begin)/tether.substeps
                        positions, velocities = tether.line.step(time, positions, velocities, until-time, line_ends, line_wind)
                    line_state = (positions, velocities)
                tether_record = tether_fields(tether, pose, *line_state, wind)
        except FAILURES as error:
            yield Failure(index, module, f"{type(error).__name__}: {str(error)[:300]}")
            return
        seconds = _time.perf_counter()-tick
        record = step_record(index, pose, solved, placed, wind, density)
        if choice is not None:
            record.update(handover_record(solved, placed, choice, seeding_used, zero_seed_failure))
        if rotor_fields is not None:
            record.update(rotor_fields)
        if tether_record is not None:
            record.update(tether_record)
        step = Step(index, record, seconds, pose, placed, solved, state)
        step.line = None if line_state is None else np.asarray(line_state[0], dtype=float)
        yield step


def coefficients(record, area_m2, chord_m):
    """Wind-axis force and body-axis moment coefficients of a step record (the formulas of
    ``openfawe.aero.reference.load_coefficients``; moments about the body origin)."""
    q = float(record["dynamic_pressure_Pa"])
    relative = np.asarray(record["relative_wind_body_m_s"], dtype=float)
    speed = float(np.linalg.norm(relative))
    out = dict(CL=np.nan, CD=np.nan, CY=np.nan, Cl=np.nan, Cm=np.nan, Cn=np.nan)
    if not (q > 0 and speed > 0):
        return out
    drag = relative/speed
    lift = np.cross(drag, [0., 1., 0.])
    norm = np.linalg.norm(lift)
    if norm == 0:
        return out
    lift /= norm
    side = np.cross(lift, drag)
    force = np.asarray(record["force_body_N"], dtype=float)
    moment = np.asarray(record["moment_origin_body_Nm"], dtype=float)
    qs = q*area_m2
    out.update(CL=float(force@lift/qs), CD=float(force@drag/qs), CY=float(force@side/qs),
               Cl=float(moment[0]/(qs*chord_m)), Cm=float(moment[1]/(qs*chord_m)), Cn=float(moment[2]/(qs*chord_m)))
    return out


def flat_record(record, area_m2, chord_m):
    """Scalar columns of one record (vectors split into _x, _y, _z) plus the coefficients, for plots and CSV."""
    out = {}
    for key, value in record.items():
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)) or value is None:
            out[key] = np.nan if value is None else float(value)
        elif isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
            for axis, v in zip("xyz", value):
                out[f"{key}_{axis}"] = float(v)
    if "dynamic_pressure_Pa" in record and "relative_wind_body_m_s" in record:
        out.update(coefficients(record, area_m2, chord_m))
    return out


def network(built, step):
    """Nodes, edges, circulation per edge, the first ring node, and the number of nodes on the surface rows."""
    method = built.backend.method.value
    state = step.state
    width = built.wing.node_count
    if method in ("free_wake", "vsm_free_wake"):
        from openfawe.aero.wake import extended_network
        nodes, edges, gamma, ring_first = extended_network(state)
        return nodes, edges, gamma, ring_first, 2*width
    if method == "ll_straight":
        from openfawe.aero.ll_straight import straight_network
        nodes, edges, gamma = straight_network(state)
        return nodes, edges, gamma, len(nodes), 2*width
    if method in ("vlm", "vlm_free_wake"):
        from openfawe.aero.lifting_surface import lifting_surface_network
        nodes, edges, gamma = lifting_surface_network(state)
        return nodes, edges, gamma, len(nodes), 0
    # straight VSM: horseshoes along the prescribed wake direction, drawn 20 chords long
    wing = step.placed
    direction = np.asarray(state.solution.wake_direction, dtype=float)
    te = np.asarray(wing.trailing_edge, dtype=float)
    far = te+20.0*built.chord_m*(direction if direction.ndim == 2 else direction[None, :])
    nodes = np.vstack([te, far])
    gamma_panel = np.asarray(state.solution.circulation, dtype=float)
    edges, gamma = [], []
    shed = np.zeros(len(te))
    for (a, b), g in zip(np.asarray(wing.panel_edges), gamma_panel):
        shed[a] -= g
        shed[b] += g
    for k, g in enumerate(shed):
        if g:
            edges.append((k, len(te)+k))
            gamma.append(g)
    return nodes, np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(gamma, dtype=float), len(nodes), 0


def frame(built, step):
    """Everything the 3D view draws for one step, as plain arrays (it pickles to the interface process)."""
    nodes, edges, gamma, ring_first, surface_nodes = network(built, step)
    rings = getattr(step.state, "rings", None)
    ring_source = np.zeros(0, dtype=np.int64)
    if rings is not None and len(rings.edges):
        ring_source = np.asarray(rings.source, dtype=np.int64)
    placed = step.placed
    return dict(step=int(step.index), time_s=float(step.record["time_s"]),
                rotation=np.asarray(step.pose.rotation, dtype=float), position=np.asarray(step.pose.position_m, dtype=float),
                nodes=np.asarray(nodes, dtype=float), edges=np.asarray(edges, dtype=np.int64),
                gamma=np.asarray(gamma, dtype=float), ring_first=int(ring_first), surface_nodes=int(surface_nodes),
                ring_source=ring_source,
                leading_edge=np.asarray(placed.leading_edge, dtype=float),
                trailing_edge=np.asarray(placed.trailing_edge, dtype=float),
                panel_edges=np.asarray(placed.panel_edges, dtype=np.int64),
                bound_gamma=np.asarray(step.solved.solution.circulation, dtype=float),
                force=np.asarray(step.record["force_inertial_N"], dtype=float),
                line=np.zeros((0, 3)) if getattr(step, "line", None) is None else step.line,
                ground=np.full(3, np.nan) if built.ground_point is None else np.asarray(built.ground_point, dtype=float))
