"""Checks of G5 step 2: the Makani crosswind controller (library) on the OpenFAWE plant.

Contract: ``closed-loop-contract.zh.md`` (criteria B1 to B4 and the report items, written 2026-10-07 14:26, before any
of the code below or of ``flight/closed_loop.py`` existed). B0 (the library's own checks) is ``makani_lib/l_checks.py``.
Nothing here sets a threshold: they are those of the contract.

Case CL1: the settings of case A5 of the G4 contract (nominal loop of the 9 m/s land playbook on the tether sphere,
spline tables, wing and tails with the tether, hand-over by the body origin with zero seeding, default far wake),
with the differences the closed-loop contract names: the wind points to the azimuth -110 deg (inside the allowed
sector of the test site), the strain that sets the sphere is 0.0054375, the air density is 1.026 kg/m^3, the time
step is exactly 0.01 s (the controller's period). The line is run alone for two laps of the periodic motion first;
the coupled run then follows the same law, sampled at multiples of 0.01 s, for the lead-in and is handed to the
controller at step N0.

Parts (``--parts``):
  b1   interface checks (need the controller library);
  b2   the hooks of ``run_free`` with a controller that adds nothing: bit-identical records;
  b3   the first closed loop, CL1-S0 (rotor loads from the tables, no rings): one lap, or as far as it gets;
  b4   the first 50 closed-loop steps in two processes: bit-identical.
``--steps N`` shortens the closed-loop part of b3 (a trial that judges nothing and says so). ``--rings`` runs b3 with
rotor rings (CL1-S1, report only).

Writes ``checks.json`` (criteria, report items) and ``history.json`` (records and the controller's log) into OUT_DIR.

Usage: closed_loop_checks.py OUT_DIR BUILD [--parts b1,b2,b3,b4] [--steps N] [--lead-in 200] [--rings] [--backend tree]
  BUILD is the folder of ``makani_lib/build_makani.py`` with ``lib/makani_controller_api.py``.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
G4 = REPO/"docs/tasks/OF-0007-g4motion20261002"
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.handover import CROSSWIND_SOLVE, SectionHandover  # noqa: E402
from openfawe.aero.makani_flap_increments import ELEVATOR, RUDDER, FlapIncrements, FlapSettings  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.flight import crosswind_loop as cl  # noqa: E402
from openfawe.flight import makani_interface as mi  # noqa: E402
from openfawe.flight.closed_loop import CrosswindLoop, Externals  # noqa: E402
from openfawe.flight.free_flight import Release, run_free  # noqa: E402
from openfawe.flight.kinematics import Pose, RigidMotion, relative_wind_body, rotation_from_vector, rotation_vector  # noqa: E402
from openfawe.flight.plant import PlantTether  # noqa: E402
from openfawe.flight.prescribed import PrescribedRotors, PrescribedTether, run_tether_alone  # noqa: E402
from openfawe.flight.residual import load_rigid_body  # noqa: E402
from openfawe.flight.rigid_body import RigidState, state_from_pose  # noqa: E402
from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set  # noqa: E402
from openfawe.rotor.table import RotorTable  # noqa: E402
from openfawe.tether.bridle import load_bridle  # noqa: E402
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(f"g4_{name}_for_cl", G4/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A5 = _script("a5_loop")
A3 = A5.A3

SEED = 20261007
STEP_S = 0.01
STRAIN = 0.0054375
WIND_SPEED, WIND_AZIMUTH_G = 9.0, math.radians(-110.0)
DENSITY = 1.026
FLAP_DATABASE = "cases/polars/makani/m600_aswing_baseline.json"
LEAD_IN, REPEAT_STEPS = 200, 50
PERIODS_ALLOWED = 1.5


def load_api(build):
    spec = importlib.util.spec_from_file_location("makani_controller_api", Path(build)/"lib"/"makani_controller_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wake_options(backend):
    options = dict(velocity_backend=backend)
    if backend == "tree":
        options["tree"] = TreeSettings(theta=0.2, order=2, leaf_size=32)
    return options


def sampled_motion(loop, times, start_angle=0.0, h=1e-3):
    """The nominal loop at arbitrary times: the stencils of ``NominalLoop.motion`` (fourth-order differences in time)."""
    bounds, knot_times = loop._knot_times(start_angle)
    period = float(knot_times[-1])

    def angle_at(time):
        if time >= 0.0:
            return loop.loop_angle_at(time, start_angle, bounds, knot_times)
        return loop.loop_angle_at(time+period, start_angle, bounds, knot_times)+2.0*np.pi
    rows = []
    for time in times:
        poses = [loop.pose_world(angle_at(time+m*h)) for m in range(-2, 3)]
        p = [pose[0] for pose in poses]
        phi1 = rotation_vector(poses[3][1]@poses[1][1].T)
        phi2 = rotation_vector(poses[4][1]@poses[0][1].T)
        rows.append((p[2], poses[2][1], (-p[4]+8.0*p[3]-8.0*p[1]+p[0])/(12.0*h), (8.0*phi1-phi2)/(12.0*h),
                     (-p[4]+16.0*p[3]-30.0*p[2]+16.0*p[1]-p[0])/(12.0*h*h)))
    return RigidMotion(np.asarray(times, dtype=float), *(np.array([row[k] for row in rows]) for k in range(5)),
                       description="Makani nominal crosswind loop sampled at the controller's period")


def cl_base(closed_steps, backend, lead_in):
    """Case CL1 (see the module text); ``closed_steps`` closed-loop steps follow the lead-in (None: 1.5 nominal periods)."""
    contract = A5.CONTRACT
    entries, _, playbook = cl.load_playbook(REPO/A5.PLAYBOOK)
    constants = cl.LoopConstants.from_case(playbook)
    entry = cl.playbook_entry(entries, WIND_SPEED)
    bridle = load_bridle(REPO/A5.RIGID_BODY)
    sphere = constants.tether_length_m*(1.0+STRAIN)+constants.bridle_radius_m
    centre = cl.NominalLoop.sphere_centre_distance(sphere, entry["path_radius_target"])
    wind_g = WIND_SPEED*np.array([math.cos(WIND_AZIMUTH_G), math.sin(WIND_AZIMUTH_G), 0.0])
    loop = cl.NominalLoop(entry, wind_g, constants, centre_distance_m=centre, interpolation=contract["interpolation"],
                          roll_limit_tension_N=contract["roll_limit_tension_N"])
    period = loop.period_s()
    steps_per_lap = int(round(period/STEP_S))
    full_steps = int(math.ceil(PERIODS_ALLOWED*period/STEP_S))
    closed_steps = full_steps if closed_steps is None else int(closed_steps)
    steps = lead_in+closed_steps
    wind = loop.wind_world()
    low, high = load_aircraft(REPO/A5.LOW_CASE), load_aircraft(REPO/A5.HIGH_CASE)
    handover = SectionHandover(low.polar, high.polar, seeding=contract["seeding"])
    free_rows = int(round(contract["far_wake_free_lap_fraction"]*steps_per_lap))
    keep_rows = contract["far_wake_keep_laps"]*steps_per_lap

    def aero():
        config = WakeConfig(dt_s=STEP_S, max_rows=max(steps, keep_rows, 1), core_radius_m=0.15, convection_integrator="euler",
                            core_evolution="constant", solve=CROSSWIND_SOLVE, free_convection_rows=free_rows,
                            retained_rows=keep_rows, density_kg_m3=DENSITY, **wake_options(backend))
        return AeroBackend("vsm_free_wake", wake=config)
    line = LumpedTether(TetherProperties(), TetherSettings(air_density_kg_m3=DENSITY))
    ground = loop.ground_point_world()
    tick = time.perf_counter()
    # The line alone over two laps of the periodic motion (as in a5_loop.py); it ends at the pose of time zero.
    warm_motion, _ = loop.motion(max(contract["warmup_laps"], 1), steps_per_lap)
    substeps = 5
    tether, report = run_tether_alone(PrescribedTether(line, ground, bridle, substeps=substeps), warm_motion, wind,
                                      record_every=steps_per_lap)
    motion = sampled_motion(loop, np.arange(lead_in+3)*STEP_S)
    start_gap = float(np.linalg.norm(motion.position_m[0]-warm_motion.position_m[0]))
    return SimpleNamespace(aero=aero, wing=low.wing, handover=handover, motion=motion, wind=wind, wind_g=wind_g, loop=loop,
                           line=line, ground=ground, bridle=bridle, tether=tether, substeps=substeps, period=period,
                           closed_steps=closed_steps, full_steps=full_steps,
                           rigid=load_rigid_body(REPO/A5.RIGID_BODY), constants=constants, entry=entry,
                           settings=dict(step_s=STEP_S, period_s=period, steps_per_nominal_lap=period/STEP_S, wind_m_s=WIND_SPEED,
                                         wind_azimuth_g_deg=math.degrees(WIND_AZIMUTH_G), wind_world_m_s=wind.tolist(),
                                         strain=STRAIN, sphere_radius_m=sphere, density_kg_m3=DENSITY,
                                         far_wake=dict(free_rows=free_rows, keep_rows=keep_rows), line_substeps=substeps,
                                         warmup=dict(laps=contract["warmup_laps"], seconds=time.perf_counter()-tick,
                                                     tension_at_lap_ends_N=[r["tension_kite_N"] for r in report],
                                                     start_pose_gap_m=start_gap),
                                         low_case=A5.LOW_CASE, high_case=A5.HIGH_CASE, playbook=A5.PLAYBOOK,
                                         rigid_body=A5.RIGID_BODY, flap_database=FLAP_DATABASE, rotor_set=A3.ROTOR_SET,
                                         rotor_table=A3.ROTOR_TABLE))


# ---------------------------------------------------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------------------------------------------------
def b1(api, case, flaps):
    out, ref = {}, ctypes.byref
    rng = np.random.default_rng(SEED)
    body = case.rigid
    # (a) frames: round trip, and the nominal loop in Makani's axes
    worst = dict(position=0.0, velocity=0.0, rotation=0.0, omega=0.0)
    for _ in range(100):
        state = RigidState(float(rng.uniform(0, 10)), rng.uniform(-500, 500, 3), rotation_from_vector(rng.uniform(-2, 2, 3)),
                           rng.uniform(-60, 60, 3), rng.uniform(-1, 1, 3))
        truth = mi.makani_truth(body, state, np.zeros(3), acceleration_origin_m_s2=np.zeros(3))
        back = mi.plant_from_makani(body, state.time_s, truth.Xg, truth.Vg, truth.dcm_g2b, truth.pqr)
        worst["position"] = max(worst["position"], np.linalg.norm(back.position_cm_m-state.position_cm_m)/max(1.0, np.linalg.norm(state.position_cm_m)))
        worst["velocity"] = max(worst["velocity"], np.linalg.norm(back.velocity_cm_m_s-state.velocity_cm_m_s)/max(1.0, np.linalg.norm(state.velocity_cm_m_s)))
        worst["rotation"] = max(worst["rotation"], float(np.abs(back.rotation-state.rotation).max()))
        worst["omega"] = max(worst["omega"], np.linalg.norm(back.angular_velocity_body_rad_s-state.angular_velocity_body_rad_s)
                             / max(1.0, np.linalg.norm(state.angular_velocity_body_rad_s)))
    loop, gaps, angles_gap = case.loop, dict(Xg=0.0, Vg=0.0, dcm_g2b=0.0), dict(alpha=0.0, beta=0.0)
    for k in range(64):
        angle = 2.0*math.pi*k/64.0
        position, rotation = loop.pose_world(angle)
        position_g, vel_hat, _ = loop.loop_geometry(angle)
        velocity_g = loop.kite_speed(angle)*vel_hat
        # A plant state with the origin on the loop: the attitude and origin velocity of the law, any body rate.
        pose = Pose(0.0, position, rotation, cl.MAKANI_G_TO_WORLD@velocity_g, np.zeros(3))
        truth = mi.makani_truth(body, state_from_pose(pose, body), case.wind, acceleration_origin_m_s2=np.zeros(3))
        gaps["Xg"] = max(gaps["Xg"], np.linalg.norm(truth.Xg-position_g)/max(1.0, np.linalg.norm(position_g)))
        gaps["Vg"] = max(gaps["Vg"], np.linalg.norm(truth.Vg-velocity_g)/max(1.0, np.linalg.norm(velocity_g)))
        gaps["dcm_g2b"] = max(gaps["dcm_g2b"], float(np.abs(truth.dcm_g2b-loop.attitude_g2b(angle)).max()))
        # (b) the library's spherical apparent wind against the OpenFAWE angles (sideslip of the other sign)
        sph = api.ApparentWindSph()
        api.ApparentWindCartToSph(ref(api.Vec3(*truth.apparent_wind_b)), ref(sph))
        _, _, alpha_deg, beta_deg = relative_wind_body(pose, case.wind)
        angles_gap["alpha"] = max(angles_gap["alpha"], abs(sph.alpha-math.radians(alpha_deg)))
        angles_gap["beta"] = max(angles_gap["beta"], abs(sph.beta+math.radians(beta_deg)))
    out["a"] = dict(round_trip_worst=worst, nominal_loop_worst=gaps,
                    cannot_show="both halves compose a map with its inverse, or apply the same two diagonal maps on both sides: they "
                                "show that the modules agree with each other, not that the convention is right (review finding CL-06); "
                                "the convention is pinned by the hard-coded expectation of tests/test_closed_loop.py and by the "
                                "library's own loop angle at the hand-over",
                    passed=bool(all(v <= 1e-12 for v in worst.values()) and all(v <= 1e-9 for v in gaps.values())))
    out["b"] = dict(worst_gap_rad=angles_gap, passed=bool(all(v <= 1e-12 for v in angles_gap.values())))
    # (c) tether force in spherical form: three hand-made vectors, the library against the formulas of sensor_util.c
    rows = []
    for vector in ([0.0, 0.0, 1000.0], [0.0, -300.0, 1000.0], [200.0, 0.0, 1000.0]):
        sph = api.TetherForceSph()
        api.TetherForceCartToSph(ref(api.Vec3(*vector)), ref(sph))
        x, y, z = vector
        hand = (math.sqrt(x*x+y*y+z*z), math.atan2(-y, math.hypot(x, z)), math.atan2(x, z))
        rows.append(dict(vector=vector, library=[sph.tension, sph.roll, sph.pitch], by_hand=list(hand),
                         python=list(mi.tether_force_sph(vector)),
                         gap=float(max(abs(a-b) for a, b in zip((sph.tension, sph.roll, sph.pitch), hand)))))
    state = state_from_pose(case.motion.pose(0.0), body)
    nodes, speeds = case.tether.start(case.motion)
    force = PlantTether(case.line, case.ground, case.bridle).loads(body, state, nodes, speeds, case.wind)[2]
    truth = mi.makani_truth(body, state, case.wind, acceleration_origin_m_s2=np.zeros(3), tether_force_inertial_N=force)
    out["c"] = dict(vectors=rows, plant_tether_force_makani_body_N=truth.tether_force_b.tolist(),
                    plant_tether_sph=list(mi.tether_force_sph(truth.tether_force_b)),
                    passed=bool(all(row["gap"] <= 1e-12 for row in rows) and truth.tether_force_b[2] > 0.0))
    # (d) flap increments
    from scipy.interpolate import RegularGridInterpolator
    # At the nodes the interpolation is asked in the table's own unit (degrees): bit for bit. Through the module's
    # interface in radians a node is reached only to the rounding of deg -> rad -> deg; that gap is reported.
    from openfawe.aero.makani_flap_increments import _saturating_bilinear
    node_gap = node_gap_radians = 0.0
    for i, alpha in enumerate(flaps.alphas_deg):
        for j, beta in enumerate(flaps.betas_deg):
            node_gap = max(node_gap, float(np.abs(_saturating_bilinear(flaps.alphas_deg, flaps.betas_deg, flaps.derivatives, alpha, beta)
                                                  - flaps.derivatives[:, :, i, j]).max()))
            node_gap_radians = max(node_gap_radians, float(np.abs(flaps.derivatives_at(math.radians(alpha), math.radians(beta))
                                                                  - flaps.derivatives[:, :, i, j]).max()))
    reference = RegularGridInterpolator((flaps.alphas_deg, flaps.betas_deg), np.moveaxis(flaps.derivatives, (0, 1), (2, 3)), method="linear")
    between = 0.0
    for _ in range(200):
        alpha, beta = rng.uniform(flaps.alphas_deg[0], flaps.alphas_deg[-1]), rng.uniform(flaps.betas_deg[0], flaps.betas_deg[-1])
        between = max(between, float(np.abs(flaps.derivatives_at(math.radians(alpha), math.radians(beta))-reference([[alpha, beta]])[0]).max()))
    outside = float(np.abs(flaps.derivatives_at(math.radians(40.0), math.radians(-40.0))-flaps.derivatives[:, :, -1, 0]).max())
    signs = []
    for alpha_deg in (0.0, 3.0, 6.0):
        table = flaps.derivatives_at(math.radians(alpha_deg), 0.0)        # [flap, (CX, CY, CZ, Cl, Cm, Cn)]
        signs.append(dict(alpha_deg=alpha_deg, Cl_A1=float(table[0, 3]), Cl_A2=float(table[1, 3]), Cl_A7=float(table[4, 3]),
                          Cl_A8=float(table[5, 3]), Cm_Ele=float(table[ELEVATOR, 4]), Cn_Rud=float(table[RUDDER, 5]),
                          holds=bool(table[0, 3] >= 0 and table[1, 3] >= 0 and table[4, 3] <= 0 and table[5, 3] <= 0
                                     and table[ELEVATOR, 4] <= 0 and table[RUDDER, 5] <= 0)))
    s, deg = flaps.settings, math.radians
    offset = float(mi_bilinear(flaps, 2.0, 1.0))
    hand = [
        # (flaps in rad, expected effective deflections in degrees), from the source's formulas by hand
        (np.array([0.0]*7+[deg(20.0)]), ("rudder above the threshold", RUDDER, 10.0+0.4692*10.0)),
        (np.array([0.0]*7+[deg(-25.0)]), ("rudder below the threshold", RUDDER, -10.0-0.5374*15.0)),
        (np.array([0.0]*7+[deg(5.0)]), ("rudder inside the thresholds", RUDDER, 5.0)),
        (np.array([deg(8.0)]+[0.0]*7), ("wing flap above +5 deg", 0, 5.0)),
        (np.array([-0.7]+[0.0]*7), ("wing flap below -0.5 rad", 0, math.degrees(-0.5))),
        (np.array([0.0]*6+[0.6, 0.0]), ("elevator above +0.5 rad, minus de1", ELEVATOR, math.degrees(0.5)-offset)),
        (np.array([0.0]*6+[deg(-3.0), 0.0]), ("elevator minus de1", ELEVATOR, -3.0-offset)),
        (np.array([0.0]*6+[deg(4.0), 0.0]), ("elevator minus de1", ELEVATOR, 4.0-offset)),
        (np.array([0.0]*7+[deg(40.0)]), ("rudder scaled, then saturated at 0.5 rad", RUDDER, min(10.0+0.4692*30.0, math.degrees(0.5)))),
    ]
    treatment = []
    for flap_values, (what, index, expected) in hand:
        got = float(flaps.effective_deflections_deg(flap_values, deg(2.0), deg(1.0))[index])
        treatment.append(dict(what=what, expected_deg=expected, got_deg=got, gap=abs(got-expected)))
    # dimensional loads and axes at one hand-made point
    through_air = np.array([-40.0, 1.5, -3.0])                             # OpenFAWE body axes: flying towards -x (forward)
    deflection = np.array([deg(2.0), deg(-3.0), 0.0, 0.0, deg(1.0), deg(-2.0), deg(-4.0), deg(6.0)])
    force, moment, condition = flaps.loads(through_air, DENSITY, deflection, thrust_coeff=-0.1)
    v = np.array([40.0, 1.5, 3.0])                                         # the same velocity in Makani's body axes
    alpha, beta = math.atan2(v[2], v[0]), math.asin(v[1]/np.linalg.norm(v))
    q = 0.5*DENSITY*float(v@v)
    table = np.array(flaps.derivatives_at(alpha, beta))
    table[[ELEVATOR, RUDDER]] *= 0.9
    degrees = np.degrees(deflection)
    degrees[ELEVATOR] -= float(mi_bilinear(flaps, math.degrees(alpha), math.degrees(beta)))
    total = degrees@table
    expected_force = q*s.area_m2*total[:3]*np.array([-1.0, 1.0, -1.0])
    expected_moment = q*s.area_m2*np.array([s.span_m, s.chord_m, s.span_m])*total[3:]*np.array([-1.0, 1.0, -1.0])
    dimensional = dict(force_gap=float(np.linalg.norm(force-expected_force)/np.linalg.norm(expected_force)),
                       moment_gap=float(np.linalg.norm(moment-expected_moment)/np.linalg.norm(expected_moment)),
                       alpha_gap=abs(condition["alpha_rad"]-alpha), beta_gap=abs(condition["beta_rad"]-beta),
                       force_N=force.tolist(), moment_Nm=moment.tolist())
    out["d"] = dict(largest_gap_at_table_nodes=node_gap, largest_gap_at_table_nodes_through_radians=node_gap_radians,
                    largest_gap_between_nodes_against_scipy=between,
                    gap_outside_the_grid_against_the_corner_value=outside, signs_of_the_database_check=signs,
                    treatment_of_the_deflections=treatment, dimensional=dimensional,
                    database=FLAP_DATABASE, database_sha256=hashlib.sha256((REPO/FLAP_DATABASE).read_bytes()).hexdigest(),
                    note="no independent numerical reference: Makani's C++ simulator is not built; the transcription is "
                         "checked against the source's formulas",
                    cannot_show="de1 is identically zero in this database, so (iii) cannot see the sign of the elevator offset here (the "
                                "unit test with a synthetic table does); the expectation of (iv) is built with the module's own "
                                "interpolation (review finding CL-06)",
                    passed=bool(node_gap == 0.0 and between <= 1e-12 and outside == 0.0 and all(row["holds"] for row in signs)
                                and all(row["gap"] <= 1e-12 for row in treatment)
                                and dimensional["force_gap"] <= 1e-12 and dimensional["moment_gap"] <= 1e-12))
    # (e) servo reference model against a fine integration
    servo_rows = []
    for step in (0.3, 0.005):
        servo = mi.ServoReference(np.zeros(8))
        command = np.zeros(8)
        command[0] = -step                                    # A1, inside its limits (-81 deg .. +10 deg)
        exact = [servo.step(command, STEP_S)[0] for _ in range(60)]
        gain, rate, h = 2.0*math.pi*15.91, 0.85, 1e-5

        def slope(angle):
            return min(max(gain*(-step-angle), -rate), rate)
        # The reference of the criterion is the classical fourth-order scheme at 1e-5 s (contract revision 1); the
        # explicit Euler scheme at the same step is reported beside it: its own error is above the threshold.
        ref_angle, euler_angle, fine, euler = 0.0, 0.0, [], []
        for k in range(60*1000):
            k1 = slope(ref_angle)
            k2 = slope(ref_angle+0.5*h*k1)
            k3 = slope(ref_angle+0.5*h*k2)
            k4 = slope(ref_angle+h*k3)
            ref_angle += h*(k1+2.0*k2+2.0*k3+k4)/6.0
            euler_angle += h*slope(euler_angle)
            if (k+1) % 1000 == 0:
                fine.append(ref_angle)
                euler.append(euler_angle)
        servo_rows.append(dict(step_rad=-step, largest_gap_rad=float(np.abs(np.array(exact)-np.array(fine)).max()),
                               largest_gap_against_explicit_euler_rad=float(np.abs(np.array(exact)-np.array(euler)).max()),
                               final_rad=exact[-1]))
    servo = mi.ServoReference(np.zeros(8))
    command = np.zeros(8)
    command[0] = math.radians(30.0)                           # above the +10 deg limit of A1
    for _ in range(200):
        held = servo.step(command, STEP_S)[0]
    out["e"] = dict(steps=servo_rows, held_at_limit_rad=float(held), limit_rad=math.radians(10.0),
                    passed=bool(all(row["largest_gap_rad"] <= 1e-6 for row in servo_rows) and abs(held-math.radians(10.0)) <= 1e-9))
    out["passed"] = bool(all(out[k]["passed"] for k in "abcde"))
    return out


def mi_bilinear(flaps, alpha_deg, beta_deg):
    """The elevator offset table at a point, by the same interpolation (zero in the baseline database)."""
    from openfawe.aero.makani_flap_increments import _saturating_bilinear
    return _saturating_bilinear(flaps.alphas_deg, flaps.betas_deg, flaps.elevator_offset_deg, alpha_deg, beta_deg)


# ---------------------------------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------------------------------
def start_of(case):
    nodes, speeds = case.tether.start(case.motion)
    return (state_from_pose(case.motion.pose(case.motion.time_s[0]), case.rigid),
            dict(tether=PlantTether(case.line, case.ground, case.bridle), nodes=nodes, node_velocities=speeds, substeps=case.substeps))


def b2(backend, api, flaps, lead_in=20, free=20):
    """A controller that adds nothing: the records must be those of ``run_free`` without the hooks, bit for bit.

    The case is the A5 base at half its time step of the step-1b checks (``p_free_flight_checks.a5_base(2, ...)``),
    released with the trim load, as the contract words it. Two forms of "nothing": two empty functions as hooks, and
    the closed-loop driver itself (``CrosswindLoop`` with a start step beyond the run, its ``external`` through
    ``Externals``; review finding CL-04). The driver adds the field ``controller_active`` to every record, which
    must be False everywhere and is left out of the comparison; it is given the controller's period as its step
    although this case runs at 0.0099988 s, which it never uses because it never starts.
    """
    spec = importlib.util.spec_from_file_location("p_checks_for_cl", TASK/"p_free_flight_checks.py")
    p_checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p_checks)
    case = p_checks.a5_base(2, lead_in+free, backend, 5)
    times = case.motion.time_s[:lead_in+free+1]

    def run(hooks):
        state, line = start_of(case)
        release = Release(case.rigid, case.motion, release_time_s=times[lead_in], trim=True)
        return run_free(case.aero(), case.wing, None, case.rigid, state, case.wind, times=times, handover=case.handover,
                        pin_to=case.motion, pin_until=lead_in, external=release, **line, **hooks)
    def run_driver():
        state, line = start_of(case)
        release = Release(case.rigid, case.motion, release_time_s=times[lead_in], trim=True)
        loop = CrosswindLoop(api, case.rigid, case.wind, start_step=10**9, step_s=STEP_S, density_kg_m3=DENSITY, flap_increments=flaps,
                             tether=line["tether"])
        return run_free(case.aero(), case.wing, None, case.rigid, state, case.wind, times=times, handover=case.handover,
                        pin_to=case.motion, pin_until=lead_in, external=Externals(release, loop.external),
                        before_aero=loop.before_aero, extra_loads=loop.extra_loads, **line), loop

    def text(record, drop=()):
        return json.dumps({key: value for key, value in record.items() if key not in drop}, sort_keys=True, default=str)
    plain = run({})
    hooked = run(dict(before_aero=lambda *args: None, extra_loads=lambda pose, record: (np.zeros(3), np.zeros(3), {})))
    driven, loop = run_driver()
    same = plain.status == hooked.status == "complete" and len(plain.records) == len(hooked.records) and all(
        text(a) == text(b) for a, b in zip(plain.records, hooked.records))
    inert = all(record.get("controller_active") is False for record in driven.records) and not loop.log
    same_driver = plain.status == driven.status == "complete" and len(plain.records) == len(driven.records) and inert and all(
        text(a) == text(b, drop=("controller_active",)) for a, b in zip(plain.records, driven.records))
    return dict(steps=len(times)-1, status=[plain.status, hooked.status, driven.status], errors=[plain.error, hooked.error, driven.error],
                records_bit_identical=bool(same), records_bit_identical_through_the_driver=bool(same_driver),
                driver_stayed_inert=bool(inert), field_left_out_of_the_driver_comparison="controller_active",
                passed=bool(same and same_driver))


def closed_loop(api, case, flaps, lead_in, steps, *, rings=False, progress=None, entry="prep-trans-out"):
    """Lead-in pinned to the nominal loop, then the controller; returns ``(run, controller, seconds)``.

    ``progress(record, controller)`` is called after every step.
    """
    discs = load_rotor_set(REPO/A3.ROTOR_SET, RotorTable.from_makani_json(REPO/A3.ROTOR_TABLE))
    inertia = load_rotor_inertia(REPO/A3.ROTOR_SET)
    times = np.arange(lead_in+steps+1)*STEP_S
    state, line = start_of(case)
    controller = CrosswindLoop(api, case.rigid, case.wind, start_step=lead_in, step_s=STEP_S, density_kg_m3=DENSITY,
                               flap_increments=flaps, tether=line["tether"], discs=discs, rotor_inertia_kg_m2=inertia,
                               rotor_tables=not rings,
                               last_flight_mode=dict([("prep-trans-out", api.kFlightModeCrosswindPrepTransOut),
                                                      ("trans-in", api.kFlightModeTransIn)])[entry])
    release = Release(case.rigid, case.motion, release_time_s=times[lead_in], trim=False)
    extra = {}
    if rings:
        extra["rotors"] = PrescribedRotors(discs, controller.rotor_speeds, nodes_per_ring=A3.A4_RING_NODES,
                                           surface_core_m=A3.A4_SURFACE_CORE_M)
    tick = time.perf_counter()
    run = run_free(case.aero(), case.wing, None, case.rigid, state, case.wind, times=times, handover=case.handover,
                   pin_to=case.motion, pin_until=lead_in, external=Externals(release, controller.external),
                   before_aero=controller.before_aero, extra_loads=controller.extra_loads,
                   progress=None if progress is None else (lambda record: progress(record, controller)), **line, **extra)
    return run, controller, time.perf_counter()-tick


def summarise(api, case, run, controller, lead_in):
    """Criterion B3 and the report items from the records after the hand-over."""
    records, log = run.records[lead_in:], controller.log
    out = dict(status=run.status, error=run.error, failed_step=run.failed_step, failed_module=run.failed_module,
               records=len(run.records), closed_loop_steps=max(len(records)-1, 0), controller_calls=len(log),
               controller_failure=controller.failed)
    if not log:
        return dict(out, lap_completed=False, passed=False)
    angle = np.unwrap(np.array([entry["loop_angle_rad"] for entry in log]))
    travelled = float(angle[0]-angle[-1])                                   # the loop angle decreases in flight
    lap_step = next((k for k in range(len(angle)) if angle[0]-angle[k] >= 2.0*math.pi), None)
    lap_time = None if lap_step is None else lap_step*STEP_S
    non_finite = sum(1 for entry in log if entry["non_finite_outputs"])
    slack = [int(r["tether_slack_segments"]) for r in records]
    height = [float(r["position_cm_m"][2]) for r in records]
    lap = lap_step is not None and lap_time <= PERIODS_ALLOWED*case.period
    upto = len(records) if lap_step is None else min(lap_step+1, len(records))
    out.update(loop_angle_travelled_rad=travelled, lap_completed=bool(lap), lap_time_s=lap_time, nominal_period_s=case.period,
               steps_with_non_finite_outputs=non_finite, largest_slack_segments=max(slack) if slack else None,
               lowest_height_m=min(height) if height else None,
               largest_slack_segments_up_to_the_lap=max(slack[:upto]) if slack else None,
               lowest_height_up_to_the_lap_m=min(height[:upto]) if height else None,
               height_is="that of the centre of mass", failed_after_the_lap=bool(lap and run.status != "complete"),
               judged_window="(i), (iv): up to the completed lap; (ii), (iii): the whole run of 1.5 nominal periods (stricter than "
                             "the contract's 'during the lap'; a failure after the lap is reported as such, not re-judged)",
               passed=bool(run.status == "complete" and controller.failed is None and lap and non_finite == 0
                           and bool(slack) and max(slack[:upto]) == 0 and min(height[:upto]) > 0.0))

    def span(values):
        values = np.asarray(values, dtype=float)
        return [float(values.min()), float(values.max())]
    n = min(len(log), len(records))
    if n == 0:                                  # failed in the first closed-loop step: nothing to report on (CL-07)
        return out
    log, records = log[:n], records[:n]
    current = np.array([entry["current_pos_cw"] for entry in log])
    radius = np.array([entry["path_radius_target_m"] for entry in log])
    centre = np.array([entry["path_center_g"] for entry in log])
    flaps_cmd = np.array([entry["flaps_command_rad"] for entry in log])
    flaps_raw = np.array([entry["flaps_raw_command_rad"] for entry in log])
    acting = np.array([entry["flaps_acting_rad"] for entry in log])
    est = {key: np.array([entry["estimate"][key] for entry in log]) for key in ("airspeed_m_s", "alpha_rad", "beta_rad",
                                                                                 "tether_tension_N", "tether_roll_rad")}
    upper, lower = api.GetControlParams().contents.crosswind.output.upper_flap_limits, api.GetControlParams().contents.crosswind.output.lower_flap_limits
    report = dict(
        radial_error_of_the_path_m=span(np.linalg.norm(current, axis=1)-radius),
        airspeed_minus_command_m_s=span(est["airspeed_m_s"]-np.array([entry["airspeed_cmd_m_s"] for entry in log])),
        airspeed_m_s=span(est["airspeed_m_s"]), alpha_deg=span(np.degrees(est["alpha_rad"])), beta_deg=span(np.degrees(est["beta_rad"])),
        alpha_minus_command_deg=span(np.degrees(est["alpha_rad"]-np.array([entry["alpha_cmd_rad"] for entry in log]))),
        beta_minus_command_deg=span(np.degrees(est["beta_rad"]-np.array([entry["beta_cmd_rad"] for entry in log]))),
        tether_tension_N=span(est["tether_tension_N"]),
        tether_roll_minus_command_deg=span(np.degrees(est["tether_roll_rad"]-np.array([entry["tether_roll_cmd_rad"] for entry in log]))),
        flap_command_deg={name: span(np.degrees(flaps_cmd[:, k])) for k, name in enumerate(mi.FLAPS)},
        flap_acting_deg={name: span(np.degrees(acting[:, k])) for k, name in enumerate(mi.FLAPS)},
        share_of_steps_flap_at_crosswind_limit={name: float(np.mean((flaps_raw[:, k] >= upper[k]-1e-12) | (flaps_raw[:, k] <= lower[k]+1e-12)))
                                                for k, name in enumerate(mi.FLAPS)},
        share_of_steps_servo_rate_limited={name: float(np.mean([entry["servo_rate_limited"][k] for entry in log]))
                                           for k, name in enumerate(mi.FLAPS)},
        path_centre_minus_nominal_m=span(np.linalg.norm(centre-case.loop.path_centre_g, axis=1)),
        nominal_path_centre_note="the nominal centre is the transcription's, on the tether sphere; the controller's is at the tether length",
        seconds_per_step=float(np.mean(run.step_seconds[lead_in:])) if len(run.step_seconds) > lead_in else None,
        handover_weight=span([r["handover_weight"] for r in records]),
        share_of_steps_with_handover_weight_above_zero=float(np.mean([r["handover_weight"] > 0.0 for r in records])),
        share_of_steps_with_incidence_above_5_deg=float(np.mean(np.degrees(est["alpha_rad"]) > 5.0)),
        first_step=dict(flap_command_minus_acting_deg=np.degrees(flaps_cmd[0]-acting[0]).tolist(),
                        tether_roll_command_deg=math.degrees(log[0]["tether_roll_cmd_rad"]),
                        tether_roll_deg=math.degrees(log[0]["estimate"]["tether_roll_rad"]),
                        integrators=log[0].get("integrators"),
                        note="the controller's own initialisation is made for the entry from the transition-in (review finding CL-01)"),
        integrators={name: span([entry["integrators"][name] for entry in log]) for name in log[0]["integrators"]}
        if "integrators" in log[0] else None,
        steps_with_non_finite_telemetry=sum(1 for entry in log if entry.get("non_finite_telemetry")),
        closure_counts={name: [r["closure"] for r in records].count(name) for name in sorted({r["closure"] for r in records})})
    if "rotor_omega_rad_s" in log[0]:
        omega = np.array([entry["rotor_omega_rad_s"] for entry in log])
        rotor = api.GetControlParams().contents.rotor_control
        power = np.array([r["rotor_shaft_power_W"] for r in records if "rotor_shaft_power_W" in r])
        report.update(rotor_speed_rad_s=span(omega),
                      share_of_steps_rotor_speed_at_a_limit=float(np.mean(np.any((omega <= rotor.idle_speed+1e-9) | (omega >= np.array(list(rotor.max_speeds))-1e-9), axis=1))),
                      share_of_steps_rotor_speed_rate_limited=float(np.mean([entry["rotor_speed_rate_limited"] for entry in log])),
                      motor_limits_ever_differ=bool(any(entry["motor_limits_differ"] for entry in log)),
                      rotor_shaft_power_W=None if not len(power) else dict(mean=float(power.mean()), range=span(power),
                                                                           note="table shaft power, negative while the motors drive the rotors"),
                      share_of_steps_rotor_table_saturated=float(np.mean([any(r.get("rotor_table_saturated", [False])) for r in records])))
    out["report"] = report
    return out


def b3_child(build, folder, backend, lead_in, steps, rings, entry):
    """The closed-loop run in a process of its own: every record and controller entry is written as it is made.

    A failed assertion of the controller library ends the process; the lines written so far are what the parent reads.
    """
    folder = Path(folder)
    api = load_api(build)
    case = cl_base(steps, backend, lead_in)
    flaps = FlapIncrements.from_makani_json(REPO/FLAP_DATABASE)
    tick = time.perf_counter()
    written = [0]
    with open(folder/"b3-records.jsonl", "w", encoding="utf-8") as records, open(folder/"b3-log.jsonl", "w", encoding="utf-8") as log:
        def progress(record, controller):
            records.write(json.dumps(record, default=str)+"\n")
            records.flush()
            for entry in controller.log[written[0]:]:
                log.write(json.dumps(entry, default=str)+"\n")
            log.flush()
            written[0] = len(controller.log)
            if record["step"] % 25 == 0:
                last = controller.log[-1] if controller.log else None
                extra = "" if last is None else (f" loop {math.degrees(last['loop_angle_rad']):6.1f} deg  v {last['estimate']['airspeed_m_s']:.1f}"
                                                 f" T {last['estimate']['tether_tension_N']/1e3:.1f} kN  ele {math.degrees(last['flaps_command_rad'][6]):+.1f}"
                                                 f" rotors {np.mean(last.get('rotor_omega_rad_s', [0.0])):.0f}")
                print(f"   step {record['step']} alpha={record['alpha_deg']:.2f} beta={record['beta_deg']:.2f} w={record['handover_weight']:.2f}"
                      f"{extra} elapsed {time.perf_counter()-tick:.0f}s", flush=True)
        run, controller, seconds = closed_loop(api, case, flaps, lead_in, case.closed_steps, rings=rings, progress=progress, entry=entry)
        for entry in controller.log[written[0]:]:              # the entry of a step that failed after the controller call
            log.write(json.dumps(entry, default=str)+"\n")
    (folder/"b3-final.json").write_text(json.dumps(dict(status=run.status, error=run.error, failed_step=run.failed_step,
                                                        failed_module=run.failed_module, step_seconds=run.step_seconds,
                                                        settings=run.settings, controller_failure=controller.failed,
                                                        controller_settings=dict(last_flight_mode=int(controller.last_flight_mode),
                                                                                 throttle=controller.throttle, start_step=controller.start_step),
                                                        consistency=controller.consistency,
                                                        seconds=seconds, records=len(run.records)), default=str), encoding="utf-8")


def b4_child(build, out_path, backend, lead_in):
    api = load_api(build)
    case = cl_base(REPEAT_STEPS, backend, lead_in)
    flaps = FlapIncrements.from_makani_json(REPO/FLAP_DATABASE)
    run, controller, _ = closed_loop(api, case, flaps, lead_in, REPEAT_STEPS)
    text = json.dumps(dict(status=run.status, error=run.error, records=run.records, log=controller.log), sort_keys=True, default=str)
    Path(out_path).write_text(text, encoding="utf-8")


def provenance():
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()
    import numba
    import scipy
    return dict(git_commit=git("rev-parse", "HEAD"), src_clean=git("status", "--porcelain", "--", "src", "cases") == "",
                src_tree=git("rev-parse", "HEAD:src"), cases_tree=git("rev-parse", "HEAD:cases"),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                threads=numba.get_num_threads(), platform=platform.platform(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(out, argv):
    parser = argparse.ArgumentParser(prog="closed_loop_checks.py")
    parser.add_argument("build")
    parser.add_argument("--parts", default="b1,b2,b3,b4")
    parser.add_argument("--steps", type=int, default=None, help="closed-loop steps of b3 (default: 1.5 nominal periods); a shorter run judges nothing")
    parser.add_argument("--lead-in", type=int, default=LEAD_IN)
    parser.add_argument("--rings", action="store_true", help="b3 with rotor rings (CL1-S1, report only)")
    parser.add_argument("--entry", default="prep-trans-out", choices=("prep-trans-out", "trans-in"),
                        help="the flight mode given as the one before (contract revision 2: prep-trans-out; the other is report only)")
    parser.add_argument("--backend", default="tree", choices=("tree", "numba", "numpy"))
    parser.add_argument("--b4-child", default=None)
    parser.add_argument("--b3-child", action="store_true")
    args = parser.parse_args(argv)
    if args.b4_child:
        b4_child(args.build, args.b4_child, args.backend, args.lead_in)
        return
    if args.b3_child:
        b3_child(args.build, out, args.backend, args.lead_in, args.steps, args.rings, args.entry)
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started, tick = time.time(), time.perf_counter()
    at_start = provenance()
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]

    def note(text):
        print(f"[{time.perf_counter()-tick:7.0f}s] {text}", flush=True)
    api = load_api(args.build)
    library = Path(args.build)/"lib"/"libmakani_controller.so"
    flaps = FlapIncrements.from_makani_json(REPO/FLAP_DATABASE)
    criteria, history = {}, {}
    case = None
    if any(part in parts for part in ("b1", "b3")):
        note("case CL1: the line alone over the warm-up laps")
        case = cl_base(args.steps if "b3" in parts else 1, args.backend, args.lead_in)     # the parent only needs the law and the start
        steps, full = case.closed_steps, case.full_steps
    if "b1" in parts:
        criteria["B1"] = b1(api, case, flaps)
        note(f"B1: {criteria['B1']['passed']} {dict((k, criteria['B1'][k]['passed']) for k in 'abcde')}")
    if "b2" in parts:
        criteria["B2"] = b2(args.backend, api, flaps)
        note(f"B2: {criteria['B2']['passed']}")
    if "b3" in parts:
        trial = args.steps is not None and args.steps < full
        name = "CL1-S1" if args.rings else "CL1-S0"
        note(f"B3: {name}, lead-in {args.lead_in}, closed loop {steps} steps{' (a shortened trial: judges nothing)' if trial else ''}")

        command = [sys.executable, str(Path(__file__).resolve()), str(out), args.build, "--b3-child", "--backend", args.backend,
                   "--lead-in", str(args.lead_in), "--entry", args.entry]+(["--steps", str(args.steps)] if args.steps is not None else [])+(["--rings"] if args.rings else [])
        child_tick = time.perf_counter()
        done = subprocess.run(command, stderr=subprocess.PIPE, text=True)          # its progress lines go to our output
        seconds = time.perf_counter()-child_tick

        def lines(name):
            path = out/name
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
        final = json.loads((out/"b3-final.json").read_text(encoding="utf-8")) if (out/"b3-final.json").exists() else None
        if final is None:
            final = dict(status="aborted", error=f"the process ended with return code {done.returncode}: {done.stderr.strip()[-600:]}",
                         failed_step=None, failed_module="process", step_seconds=[], settings={}, controller_failure=None)
        run = SimpleNamespace(records=lines("b3-records.jsonl"), status=final["status"], error=final["error"],
                              failed_step=final["failed_step"], failed_module=final["failed_module"],
                              step_seconds=final["step_seconds"], settings=final["settings"])
        controller = SimpleNamespace(log=lines("b3-log.jsonl"), failed=final["controller_failure"])
        if final["status"] == "aborted" and controller.log:
            run.failed_step = controller.log[-1]["step"]+(0 if len(controller.log)+args.lead_in > len(run.records) else 1)
        summary = summarise(api, case, run, controller, args.lead_in)
        summary.update(case=name, seconds=seconds, shortened_trial=bool(trial), rings=bool(args.rings),
                       controller_settings=final.get("controller_settings"), consistency=final.get("consistency"))
        other_settings = args.lead_in != LEAD_IN or args.backend != "tree" or args.entry != "prep-trans-out"
        if trial or args.rings or other_settings:           # nothing but the contract's case CL1-S0 is judged
            summary["judged"] = False
            summary["would_have_passed"] = None if trial else summary["passed"]
            summary["passed"] = None
        history[name] = dict(records=run.records, controller_log=controller.log, step_seconds=run.step_seconds, settings=run.settings)
        criteria["B3" if not args.rings else "CL1-S1 (report only)"] = summary
        note(f"B3: status {run.status} {run.error or ''}; lap {summary.get('lap_completed')}; "
             f"travelled {math.degrees(summary.get('loop_angle_travelled_rad', 0.0)):.1f} deg; passed {summary['passed']}")
    if "b4" in parts:
        texts = []
        for index in range(2):
            target = out/f"b4-run{index}.json"
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(out), args.build, "--b4-child", str(target),
                                   "--backend", args.backend, "--lead-in", str(args.lead_in)], capture_output=True, text=True)
            texts.append(target.read_text(encoding="utf-8") if done.returncode == 0 and target.exists() else None)
            if done.returncode:
                note(f"B4 child {index} failed: {done.stderr.strip()[-400:]}")
        same = texts[0] is not None and texts[0] == texts[1]
        status = None if texts[0] is None else json.loads(texts[0])["status"]
        criteria["B4"] = dict(steps=REPEAT_STEPS, both_finished=bool(all(t is not None for t in texts)), status=status,
                              bit_identical=bool(same), sha256=[None if t is None else hashlib.sha256(t.encode()).hexdigest() for t in texts],
                              passed=bool(same and status == "complete"))
        note(f"B4: {criteria['B4']['passed']}")
    end = provenance()
    result = dict(what="G5 step 2: the Makani crosswind controller on the plant (closed-loop contract, B1-B4)",
                  parts=parts, criteria=criteria, case=None if case is None else case.settings,
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  binding_sha256=hashlib.sha256((Path(args.build)/"lib"/"makani_controller_api.py").read_bytes()).hexdigest(),
                  elapsed_s=time.perf_counter()-tick, started_unix=started, backend=args.backend, at_start=at_start,
                  code_unchanged_during_the_run=bool(all(at_start[k] == end[k] for k in ("src_tree", "cases_tree", "script_sha256"))
                                                     and at_start["src_clean"] and end["src_clean"]), **end)
    text = json.dumps(dict(result, history=history), indent=1, default=str)+"\n"
    (out/"history.json").write_text(text, encoding="utf-8")
    result["history_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out/"checks.json").write_text(json.dumps(result, indent=1, default=str)+"\n", encoding="utf-8")
    print(json.dumps({name: value.get("passed") for name, value in criteria.items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
