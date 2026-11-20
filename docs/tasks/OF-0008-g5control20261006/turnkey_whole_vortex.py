"""Plan 3.8b (``whole-flight-vortex-contract.zh.md`` revision 1, decision 0049 with its addendum 1): the segment-load hook
of ``ground.turnkey.run_turnkey`` that gives every flying period of a TurnKey flight a vortex-method load, with the rotors'
vortex rings from period 0 on.

W2: the mode at the period's start picks the method - Makani's tables (the database term, the rings carried on) while the
kite sits on the perch (PilotHover, Perched: ``TABLE_MODES``); VSM + free wake with the rings in every other mode
(``VSM_MODES``: hover, transitions, crosswind); the strip H-ring-S only as the fallback of a VSM mode (W7).

W3, the rotor rings: after the period's loads (``after_loads``, called by ``run_turnkey`` with ``period_loads``' fields)
each rotor releases one ring of 16 nodes on its rim at the period-start pose with Gamma = T dt / (rho A), T the table
thrust TurnKey applies in this period (a disc with T exactly zero releases nothing: such a ring induces nothing);
only the rings of the newest 80 periods are kept. Physical core 0.15 m (Lamb-Oseen) for their convection, surface core
0.3 m at load and control points. In a table or fallback period the rings are standalone and convect one Euler step
with their own induction and the uniform wind (tree code) - with the B1 wind field (OF-0010 revision 19 addendum 6) the
field's wind at each node; in a VSM period they live in the main lifting surfaces' state and convect with the wake rows.

W4, the strip, only as the fallback of a VSM mode (configuration F of the hover-ring contract): one load point per panel of the case
``m600_makani_xfoil_polars_pylons.json`` (wing, tails, pylons), inflow = wind - rigid-body velocity of the point + the
rings' induction (surface core, ``singularity='raise'``), section loads by ``aero.hover_ring.Sections.loads``; the
fuselage's cross-flow drag (C_dc 1.2, the stations of the hover-ring contract's fuselage mesh, rings' induction with
``singularity='zero'``); the database's flap increments (W6). No circulation solve, no parasitic drag.

W5, VSM: plan 3.8a's main lifting surfaces (VSM + free wake under the section hand-over) with the rings in their state,
the pylons as their own vortex system (no ring induction on them), the parasitic drag, the database's flap increments.

W6: the flap increments are ``MakaniDatabase.loads`` with the acting flaps minus the same with zero flaps (same apparent
wind, rates and thrust coefficient).

W7, hand-overs: entering the VSM from the perch or after a fallback, both groups start from an empty wake (the start
solve without the rings; the rings are handed to the main state right after it); from one VSM mode to the next the wake
and the rings go on; leaving the VSM, its states are dropped and the rings convect one standalone step to the period's
time before they are used. A VSM period whose solve fails (``FAILURES`` after the hand-over rule's
own retry) takes the strip loads (``fallback``) and so does the rest of that mode; the next VSM mode starts afresh. Any
failure of the strip, of the flap increments or of the rings raises (the run ends as a failure with the module named);
the failure record of a fallback carries its ``cause`` (revision 2 B5: 'attitude' for ``Wing.transformed``'s proper-rotation
check, 'solver' for the nonlinear or circulation solve, 'other').

``orthonormalise=True`` (revision 2 B2, opt-in, only with ``mode='vortex'``; the judged run uses it only with the owner's
consent): as plan 3.8a's option, the attitude of the pose the hook starts from is replaced by the nearest rotation (U V^T of
its SVD) for every computation of the hook - both VSM groups, the strip and the fuselage, the flap and parasitic items'
axis changes, the rings' placement and the transfer to the centre of mass; the plant's state is not touched, and the pose's
position, velocity and angular velocity stay as the plant computed them. Every period then records
``whole_attitude_orthogonality_error`` (max |R^T R - I| of the plant's attitude) and ``whole_attitude_correction`` (max
|nearest - R|); without the option neither field is written. In a flight child the option is the segment ``'whole+orth'``.

``gpu=True`` (revision 3, opt-in, only with ``mode='vortex'`` and a CUDA device; segment ``'+gpu'``): the two large sums of
the hook - the VSM groups' wake advance and the standalone rings' convection - go to the whole-step GPU contract's
``'cuda_tree'`` backend (``GPU_SETTINGS``: 'mixed', suspect pairs in float64 on the GPU; the trees as without it); the
rings' induction at load and control points stays on the CPU (numba). The model and every rule are unchanged. Every period
then records ``whole_gpu_calls`` and ``whole_gpu_suspect_pairs`` (the GPU tree code's call and suspect-pair counts in it).

``mode='record'`` solves nothing and marks every period 'table' (the control flight of G2(b)).

``platform='p1'`` (OF-0010 contract revision 1 and its addendum 1, opt-in, only with ``mode='vortex'``; segment ``'+p1'``):
the platform model's step P1 in every hook period that computes loads (the perch's table periods are unchanged). Each
period, from the acting flaps (``flight.platform_surfaces.PlatformP1``): one full-circle section table per panel of the
case ``m600_makani_xfoil_polars_pylons.json`` at its flap's deflection (Makani's XFOIL tables, every flap row, the hover
model's post-stall blend - since OF-0010 revision 2 the main wing's only outside its XFOIL range [0, 25] deg; clamped at
the table edge, the excess recorded), the horizontal tail turned by the elevator about
body +y through its middle leading-edge point. Both VSM groups (main lifting surfaces with that geometry and those tables;
the pylons with the mp6d table) are ``PlatformGroup``s: ``TAKEOFF_LANDING_SOLVE`` in every mode, zero seeding with one
retry from the previous circulation. The strip fallback is rebuilt from the turned assembly with the same tables. The
database's flap increments are not computed (``MakaniDatabase.loads`` is not called by the hook); the parasitic drag stays
(until P4). Every such period records ``platform_flap_excess_rad``, ``platform_flap_clamped``, ``platform_tail_fin_overlap``
and ``platform_elevator_rad``. Without the option nothing changes.

``platform='p2'`` (OF-0010 revision 3 with its addenda 1-4; segment ``'+p2'``): P1, and the main lifting surfaces and the
pylons in **one** vortex system (one ``PlatformGroup`` 'whole_main' of 81 panels with ``surfaces``: AVL's finite core
between the class of the main surfaces and each pylon, decision 0053 addendum 5); the pylons then also join the state
that holds the rotor rings (both ways: the rings induce on them, their vortices move the rings). Every such period also
records ``platform_p2_surface_cores`` (VSM periods only).

``seeding='mixed'`` (OF-0010 revision 5 with its addenda 1-3, P5a; with ``platform`` 'p1' or 'p2'; segments ``'+p1m'``,
``'+p2m'``): the platform groups start each step from the mixed iterate (after Piszkin & Levinsky, adapted)
(``platform_surfaces.mixed_choice_step``) instead of zero; every VSM period records ``platform_seeding`` = 'mixed'. The
default ``seeding='zero'`` changes nothing. ``seeding='strip'`` (OF-0010 revision 6, P5a'; segments ``'+p1s'``, ``'+p2s'``):
the strip-theory start on stalled panels (``platform_surfaces.strip_choice_step``); records as for 'mixed'.

``local_strip=True`` (OF-0010 revision 9, P5d; only with ``platform='p2'``, not with the viscosity; segments end in 'l',
e.g. ``'+p2sl'``): the one group's local strip fallback (``platform_surfaces.PlatformGroup(local_strip=True)``): a step
that does not solve is solved again with the panels whose residual does not converge on strip theory for that step, at
most three rounds, before W7. Every VSM period records ``platform_local_strip`` = True and the group's
``whole_main_local_strip_*``; a W7 record carries the group's ``local_strip`` attempt. The default changes nothing.

``keep_rows=K`` (OF-0010 revision 10, the kept-wake check WK; only with mode 'vortex'): the lifting surfaces' free wake
keeps at most K rows (``WakeConfig.max_rows`` and ``retained_rows`` both K, K > plan 3.8a's 187 free rows) instead of
plan 3.8a's 1872; the free rows and the rotor rings are unchanged; the settings record K. None (the default) changes
nothing.

``free_rows=N`` (OF-0010 revision 22 addendum 3, B3 candidate H4; only with mode 'vortex'): only the N newest rows of the
lifting surfaces' free wake are convected with the induced velocity (``WakeConfig.free_convection_rows`` N instead of plan
3.8a's 187); older rows drift with the ambient velocity at their own positions, OLAF's frozen buffer zone (Shaler,
Branlard and Platt 2020). The kept rows, the rings and everything else are unchanged; the settings record N. None (the
default) changes nothing. ``frozen_drift='rings'`` (revision 22 addendum 4, candidate H5; only with ``free_rows``): the older
rows also take the rotor rings' induction (``WakeConfig.frozen_rows_drift='ambient_rings'``), so the slipstream carries them
while the sheet's self-induction is left out of their motion; 'ambient' (the default) is H4's drift.
``stall_lag=k`` (revision 22 addendum 8, candidate H6; only with mode 'vortex'): the Oye-type first-order lag of the stalled
sections' circulation and loads between steps, ``WakeConfig.stall_lag`` k (tau = k c / |U_out|; Oye's constant is 4); None
(the default) changes nothing.

``hover_tables=True`` (OF-0010 revision 11, B; decision 0053 addendum 16): the hover modes (``HOVER_TABLE_MODES``) join
the perch in ``TABLE_MODES`` - Makani's database acts there, the rotor rings are carried on - and the VSM modes are
HoverAccel, TransIn, CrosswindNormal, CrosswindPrepTransOut and HoverTransOut. The hook then declares
``BLOWN_WING_TABLES`` (``run_turnkey(blown_wing='tables')`` adds Makani's blown-wing term only where the database acts).
False (the default) changes nothing.
"""

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(TASK))

from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.handover import SectionHandover  # noqa: E402
from openfawe.aero.hover_ring import Sections, segment_point_distances  # noqa: E402
from openfawe.aero.vortex import CoreModel, CudaSettings, TreeSettings, segments_velocity  # noqa: E402
from openfawe.flight.companion import CompanionSurfaces  # noqa: E402
from openfawe.flight.platform_surfaces import PlatformGroup, default_p1, p2_surfaces  # noqa: E402
from openfawe.flight.free_flight import held_loads  # noqa: E402
from openfawe.flight.kinematics import Pose  # noqa: E402
from openfawe.flight.prescribed import FAILURES  # noqa: E402
from openfawe.aero.wing import CirculationConvergenceError  # noqa: E402
from openfawe.aero.nonlinear import NonlinearSolveError  # noqa: E402
from openfawe.flight.rigid_body import RigidState  # noqa: E402
from openfawe.rotor.actuator_disc import ShedRing, ring_edges  # noqa: E402
from openfawe.rotor.ring_wake import RingWake, convect_rings  # noqa: E402

import turnkey_vortex as TV  # noqa: E402

PERIOD_S = TV.PERIOD_S
DENSITY_KG_M3 = TV.DENSITY_KG_M3
TABLE_MODES = ("PilotHover", "Perched")                       # not flying: on the perch (decision 0049 addendum 1)
HOVER_TABLE_MODES = ("HoverAscend", "HoverPayOut", "HoverPrepTransformGsUp", "HoverTransformGsUp", "HoverFullLength",
                     "HoverPrepTransformGsDown", "HoverTransformGsDown", "HoverReelIn", "HoverDescend")   # OF-0010 rev. 11


def _flying_modes():
    from openfawe.ground.turnkey import MODE_ORDER
    return tuple(m for m in MODE_ORDER if m not in TABLE_MODES)


VSM_MODES = _flying_modes()
STRIP_CASE = "cases/qualification/m600_makani_xfoil_polars_pylons.json"
FUSELAGE_MESH = "docs/tasks/OF-0006-body20260930/mesh/fuselage-s030-r16.npz"
CROSSFLOW_CDC = 1.2
RING_NODES, RING_RETAINED_PERIODS = 16, 80
PHYSICAL_CORE_M, SURFACE_CORE_M = 0.15, 0.3
RING_TREE = dict(theta=0.2, order=2, leaf_size=32)
SOURCES = ("table", "vsm", "fallback")
GPU_SETTINGS = dict(precision="mixed", suspect="gpu64")      # revision 3 B1 (whole-step GPU contract S1, S2)


def cuda_settings(gpu_settings=None):
    """The hook's ``CudaSettings``: GPU_SETTINGS with the caller's choices. GPU whole-step contract revision 12: with
    precision 'fp64' and no suspect of the caller's, the suspect is 'cpu' (gpu64 is for the single precisions; before,
    fp64 + the default gpu64 was refused and the double-precision whole flight never started)."""
    merged = dict(GPU_SETTINGS, **(gpu_settings or {}))
    if merged["precision"] == "fp64" and "suspect" not in (gpu_settings or {}):
        merged["suspect"] = "cpu"
    return CudaSettings(**merged)
ATTITUDE_MESSAGE = "a proper rotation"                         # Wing.transformed's check (revision 2 B5)
PLATFORMS = (None, "p1", "p2")                                 # OF-0010: the platform model's steps (opt-in)
# OF-0010 revision 15 with addendum 1 (B'): the rotor slipstream and the hub inflow (opt-in; 'free' and 'kinematic' are B)
SLIPSTREAMS = ("free", "t3", "t3_4s", "model_d", "none")
HUB_INFLOWS = ("kinematic", "induced")
RING_SETTINGS = dict(
    free=dict(retained_periods=RING_RETAINED_PERIODS, surface_core_m=SURFACE_CORE_M, physical_core_m=PHYSICAL_CORE_M, ring={}),
    # paper 1's tube T3 (A'' with the ambient spacing): frozen rings drifting with the wind - gamma/2 along the axis at release
    t3=dict(retained_periods=200, surface_core_m=0.1, physical_core_m=0.45,
            ring=dict(free_convection_steps=0, core_m=0.45, surface_overlap=2.0, sheet_drift=True, ambient_spacing=True)),
    # T3 with the rings kept 4 s (the sensitivity S4 of revision 15 addendum 1, item 11)
    t3_4s=dict(retained_periods=400, surface_core_m=0.1, physical_core_m=0.45,
               ring=dict(free_convection_steps=0, core_m=0.45, surface_overlap=2.0, sheet_drift=True, ambient_spacing=True)),
    # paper 1's model D: free rings whose convection sees the bound system regularised over r_b = 0.64 m
    model_d=dict(retained_periods=200, surface_core_m=0.1, physical_core_m=0.45,
                 ring=dict(core_m=0.45, bound_core_m=0.64, surface_overlap=2.0)),
    none=dict(retained_periods=200, surface_core_m=0.1, physical_core_m=0.45, ring={}))
STATE_OFFSETS = dict(HoverAccel=100, TransIn=100, CrosswindNormal=5000, CrosswindPrepTransOut=200, HoverTransOut=100)


def failure_cause(error):
    """Revision 2 B5: 'attitude', 'solver' or 'other'."""
    if isinstance(error, ValueError) and ATTITUDE_MESSAGE in str(error):
        return "attitude"
    if isinstance(error, (NonlinearSolveError, CirculationConvergenceError)):
        return "solver"
    return "other"


def nearest_rotation(rotation):
    """U V^T of the SVD (plan 3.8a's option); raises when it is not a proper rotation."""
    rotation = np.asarray(rotation, dtype=float)
    u, _, vt = np.linalg.svd(rotation)
    nearest = u@vt
    if np.linalg.det(nearest) <= 0.0:
        error = ValueError("the attitude is not near a proper rotation")
        error.failed_module = "aerodynamics (attitude orthonormalisation)"
        raise error
    return nearest


def fuselage_stations(path=REPO/FUSELAGE_MESH):
    """Stations of the structured fuselage mesh (as ``hover_checks.fuselage_model``): x, centre and equivalent diameter
    of each ring of nodes, with the station lengths of ``hover_checks.crossflow_drag``."""
    nodes = np.load(path)["nodes"]
    xs = np.unique(np.round(nodes[:, 0], 9))
    centres, diameters = [], []
    for x in xs:
        ring = nodes[np.abs(nodes[:, 0] - x) < 1e-9]
        centre = ring.mean(axis=0)
        centres.append(centre)
        diameters.append(2.0*float(np.mean(np.linalg.norm(ring[:, 1:] - centre[1:], axis=1))))
    edges = np.r_[xs[0], 0.5*(xs[1:] + xs[:-1]), xs[-1]]
    return np.array(centres), np.array(diameters), np.diff(edges)


def crossflow_drag_body(centres, diameters, lengths, velocities_body, density, cdc=CROSSFLOW_CDC):
    """Sum of 1/2 rho C_dc d |u_n| u_n dx over the stations (body axes, axis = body x); force and moment about the origin."""
    u = np.asarray(velocities_body, dtype=float).reshape(len(centres), 3)
    un = u.copy()
    un[:, 0] = 0.0
    f = (0.5*float(density)*cdc*diameters*lengths*np.linalg.norm(un, axis=1))[:, None]*un
    return f.sum(axis=0), np.cross(centres, f).sum(axis=0)


class FieldInflow:
    """OF-0010 revision 19 addendum 6 (B1): the wake library's inflow from a wind field (``ShearWind.winds_world``), a
    module-level class so that the hook, which holds it inside its groups, pickles (checkpoints; review B1R-01)."""

    def __init__(self, field):
        self.field = field

    def __call__(self, points, time_s):
        return self.field.winds_world(points)


class WholeFlightVortex:
    """The segment-load hook of plan 3.8b (see the module text).

    ``body`` is the plant's rigid body, ``wind_world_m_s`` its uniform wind, ``discs`` the rotors (body axes, as given to
    ``run_turnkey``), ``mode_numbers`` the controller's flight-mode name -> number, ``database`` the reference plant's
    ``MakaniDatabase`` (for the flap increments)."""

    SOURCES = SOURCES
    APPLIED = ("vsm", "fallback")
    BLOWN_WING = False

    def __init__(self, body, wind_world_m_s, discs, mode_numbers, database, *, mode="vortex", density_kg_m3=DENSITY_KG_M3,
                 orthonormalise=False, gpu=False, platform=None, seeding="zero", viscosity=False, local_strip=False,
                 keep_rows=None, hover_tables=False, slipstream="free", hub_inflow="kinematic", state_folder=None,
                 free_rows=None, frozen_drift="ambient", stall_lag=None, wake_frames=None, gpu_settings=None,
                 ring_inflow=False, quasi_steady=False):
        if mode not in ("vortex", "record"):
            raise ValueError("mode must be 'vortex' or 'record'")
        if orthonormalise and mode != "vortex":
            raise ValueError("orthonormalise only with mode 'vortex'")
        if gpu and mode != "vortex":
            raise ValueError("gpu only with mode 'vortex'")
        if platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}")
        if platform is not None and mode != "vortex":
            raise ValueError("platform only with mode 'vortex'")
        if seeding not in ("zero", "mixed", "strip") or (seeding != "zero" and platform is None):
            raise ValueError("seeding 'mixed' or 'strip' only with a platform step (OF-0010 P5a, P5a')")
        if viscosity not in (False, True) or (viscosity and platform is None):
            raise ValueError("viscosity only with a platform step (OF-0010 P5c)")
        if viscosity and platform == "p1" and seeding == "zero":  # review P5cR-07: rejected here, not by a fallback each period
            raise ValueError("viscosity with platform 'p1' needs seeding 'mixed' or 'strip' (P1's zero seeding goes through "
                             "prescribed.handover_step, which is not edited)")
        if not isinstance(local_strip, bool) or (local_strip and platform != "p2"):
            raise ValueError("local_strip only with platform 'p2' (OF-0010 P5d)")
        if local_strip and viscosity:
            raise ValueError("local_strip is not combined with the viscosity (OF-0010 P5d)")
        if keep_rows is not None and (isinstance(keep_rows, bool) or not isinstance(keep_rows, int) or mode != "vortex"
                                      or keep_rows <= TV.FREE_ROWS):
            raise ValueError(f"keep_rows must be an integer above {TV.FREE_ROWS} rows, with mode 'vortex' (OF-0010 revision 10)")
        self.keep_rows = keep_rows
        if free_rows is not None and (isinstance(free_rows, bool) or not isinstance(free_rows, int) or mode != "vortex"
                                      or not 1 <= free_rows < TV.FREE_ROWS):
            raise ValueError(f"free_rows must be an integer from 1 to {TV.FREE_ROWS - 1}, with mode 'vortex' "
                             "(OF-0010 revision 22 addendum 3)")
        self.free_rows = free_rows
        if frozen_drift not in ("ambient", "rings") or (frozen_drift != "ambient" and free_rows is None):
            raise ValueError("frozen_drift must be 'ambient' or 'rings', 'rings' with free_rows (OF-0010 revision 22 addendum 4)")
        self.frozen_drift = frozen_drift
        if stall_lag is not None and (isinstance(stall_lag, bool) or not isinstance(stall_lag, (int, float))
                                      or not np.isfinite(stall_lag) or stall_lag <= 0.0 or mode != "vortex"):
            raise ValueError("stall_lag must be a finite number > 0, with mode 'vortex' (OF-0010 revision 22 addendum 8)")
        self.stall_lag = None if stall_lag is None else float(stall_lag)
        if not isinstance(hover_tables, bool) or (hover_tables and mode != "vortex"):
            raise ValueError("hover_tables must be True or False, True with mode 'vortex' (OF-0010 revision 11)")
        self.hover_tables = hover_tables
        if slipstream not in SLIPSTREAMS or hub_inflow not in HUB_INFLOWS:
            raise ValueError(f"slipstream must be one of {SLIPSTREAMS} and hub_inflow one of {HUB_INFLOWS}")
        if (slipstream != "free" or hub_inflow != "kinematic") and (mode != "vortex" or platform != "p2"):
            raise ValueError("the slipstream and hub-inflow options need mode 'vortex' and platform 'p2' (OF-0010 revision 15)")
        if slipstream == "free" and hub_inflow != "kinematic":
            raise ValueError("the induced hub inflow is defined with the slipstream 't3', 'model_d' or 'none' (OF-0010 revision 15)")
        self.slipstream, self.hub_inflow = slipstream, hub_inflow
        ring_settings = RING_SETTINGS[slipstream]
        self.ring_retained_periods = int(ring_settings["retained_periods"])
        self.ring_surface_core_m = float(ring_settings["surface_core_m"])
        self.ring_physical_core_m = float(ring_settings["physical_core_m"])
        self.state_folder = None if state_folder is None else str(state_folder)
        self.states_saved = []
        # OF-0010 revision 27: a wake frame every ``every`` periods of the vortex method (float32/int32, npz); None: none
        if wake_frames is not None and (set(wake_frames) != {"folder", "every"} or isinstance(wake_frames["every"], bool)
                                        or not isinstance(wake_frames["every"], int) or wake_frames["every"] < 1
                                        or mode != "vortex" or slipstream == "free"):
            raise ValueError("wake_frames must be dict(folder=..., every=N >= 1), with mode 'vortex' and a B' slipstream "
                             "(OF-0010 revision 27)")
        self.wake_frames = None if wake_frames is None else dict(folder=str(wake_frames["folder"]), every=int(wake_frames["every"]))
        self.ring_inflow = bool(ring_inflow)            # OF-0011 revision 7 addendum 11 (C1-R7): read only
        if quasi_steady and (mode != "vortex" or platform != "p2" or gpu):
            raise ValueError("quasi_steady only with mode 'vortex', platform 'p2' and the CPU (OF-0011 rev 7 add 12)")
        self.quasi_steady = bool(quasi_steady)          # OF-0011 revision 7 addendum 12 (C1-R6)
        self.frames_saved = dict(count=0, bytes=0, first_k=None, last_k=None)
        self.mode_entry_k = None
        self.source = None
        self.ring_skips = []                           # B': the periods and discs that released no ring (addendum 1, 3 and 8)
        self.BLOWN_WING_TABLES = hover_tables          # run_turnkey(blown_wing='tables') only with the hover on the tables
        self.table_modes = TABLE_MODES + (HOVER_TABLE_MODES if hover_tables else ())
        self.vsm_modes = tuple(m for m in VSM_MODES if m not in self.table_modes)
        self.platform = platform
        self.seeding = seeding
        self.viscosity = bool(viscosity)
        self.local_strip = bool(local_strip)
        self.orthonormalise = bool(orthonormalise)
        self.gpu = bool(gpu)
        # GPU whole-step contract revision 5 (PM-C): ``gpu_settings`` replaces keys of GPU_SETTINGS (precision, suspect,
        # control; revisions 6 and 7 add moments and build, revision 10 rings); None (the default) keeps GPU_SETTINGS and
        # every result as before.
        if gpu_settings is not None and (not self.gpu or not isinstance(gpu_settings, dict) or not set(gpu_settings) <=
                                         {"precision", "suspect", "control", "moments", "build", "rings"}):
            raise ValueError("gpu_settings must be a dict of precision, suspect, control, moments, build and rings, with "
                             "gpu=True (GPU whole-step contract revisions 5-7, 10)")
        self.gpu_settings = None if gpu_settings is None else dict(gpu_settings)
        self.cuda = cuda_settings(gpu_settings) if self.gpu else None
        if self.gpu:
            from openfawe.aero import vortex_cuda as VC
            if not VC.available():
                raise ValueError(f"the gpu option needs a CUDA device ({VC.probe_error()})")
        # OF-0011 revision 6 addendum 77 (F3c): another air density (offshore 1.22 kg/m^3) is taken, and goes into the wake's
        # own settings too (below); with the land density 1.026 kg/m^3 everything is as before (it used to be refused).
        if not (math.isfinite(float(density_kg_m3)) and float(density_kg_m3) > 0.0):
            raise ValueError("the air density must be finite and positive")
        missing = [name for name in TV_MODE_ORDER() if name not in mode_numbers]
        if missing:
            raise ValueError(f"mode numbers missing: {missing}")
        self.body, self.mode, self.database = body, mode, database
        self.wind = np.asarray(wind_world_m_s, dtype=float).reshape(3).copy()
        self.wind_field, self.origin_wind = None, None     # OF-0010 revision 19 addendum 6 (B1): set_period_wind
        self.discs = tuple(discs)
        self.density = float(density_kg_m3)
        self.vsm_numbers = {int(mode_numbers[name]) for name in self.vsm_modes}
        self.table_numbers = {int(mode_numbers[name]) for name in self.table_modes}
        self.names = {int(v): k for k, v in mode_numbers.items()}
        self.tree = TreeSettings(**RING_TREE)
        self.files, pylon = TV.case_files()
        self.rings = (RingWake(surface_core_m=SURFACE_CORE_M) if self.slipstream == "free" else        # B as before
                      RingWake(surface_core_m=self.ring_surface_core_m, **RING_SETTINGS[self.slipstream]["ring"]))
        self.main = self.pylons = None
        self.kind = None                                              # where the rings are: 'vsm' (main state) or 'strip' (standalone)
        self.rings_time_s = None                                      # the time the standalone rings are at
        self.fallback_mode = None                                     # the mode in which the strip took over
        self.last_mode = None
        self.pose = None
        self.fields = None
        self.periods = {name: 0 for name in SOURCES}
        self.fallbacks = []
        if mode == "vortex":
            self.sections = Sections.from_aircraft(load_aircraft(REPO/STRIP_CASE))
            self.stations = fuselage_stations()
            self.backend = TV.wake_backend()
            if self.keep_rows is not None:             # OF-0010 revision 10 (WK): the kept rows only
                self.backend = replace(self.backend, wake=replace(self.backend.wake, max_rows=self.keep_rows,
                                                                  retained_rows=self.keep_rows))
            if self.free_rows is not None:             # OF-0010 revision 22 addendum 3 (B3, H4): fewer free rows
                self.backend = replace(self.backend, wake=replace(self.backend.wake, free_convection_rows=self.free_rows))
            if self.frozen_drift == "rings":           # revision 22 addendum 4 (B3, H5): the frozen rows ride the slipstream
                self.backend = replace(self.backend, wake=replace(self.backend.wake, frozen_rows_drift="ambient_rings"))
            if self.stall_lag is not None:             # revision 22 addendum 8 (B3, H6): the stalled sections' lag
                self.backend = replace(self.backend, wake=replace(self.backend.wake, stall_lag=self.stall_lag))
            if self.gpu:                               # revision 3 B1: the same wake on the GPU tree code
                self.backend = replace(self.backend, wake=replace(self.backend.wake, velocity_backend="cuda_tree", cuda=self.cuda))
            if self.density != DENSITY_KG_M3:          # OF-0011 rev 6 add 77 (F3c): the wake's forces at the plant's density
                self.backend = replace(self.backend, wake=replace(self.backend.wake, density_kg_m3=self.density))
            low, high = load_aircraft(REPO/self.files["main_low"]), load_aircraft(REPO/self.files["main_high"])
            if low.wing.panels != high.wing.panels or not np.array_equal(low.wing.bound, high.wing.bound):
                raise ValueError("the two main-surface cases do not share one geometry")
            self.main_wing, self.main_handover = low.wing, SectionHandover(low.polar, high.polar, seeding=self.files["seeding"])
            self.pylon_body, _, self.pylon_handover = pylon.pylon_parts()
            if platform in ("p1", "p2"):               # OF-0010 P1: one geometry, checked against the 3.8b groups
                self.p1 = default_p1(REPO)
                p1_main = self.p1.period(np.zeros(8))
                for mine, theirs, label in ((p1_main.main_wing, self.main_wing, "main lifting surfaces"),
                                            (p1_main.pylon_wing, self.pylon_body, "pylons")):
                    same = mine.names == theirs.names and all(
                        np.array_equal(a.leading_edge, b.leading_edge) and np.array_equal(a.trailing_edge, b.trailing_edge)
                        for a, b in zip(mine.wings, theirs.wings))
                    if not same:
                        raise ValueError(f"the P1 case's {label} differ from plan 3.8b's")
                self.surfaces = p2_surfaces(self.p1) if platform == "p2" else None

    def settings(self):
        out = dict(model="plan 3.8b (revisions 1-3): VSM + free wake with the rotor rings in every flying mode, the strip H-ring-S "
                          "(configuration F) as the per-mode fallback, Makani's tables with the rings carried on the perch",
                    mode=self.mode, vsm_modes=list(self.vsm_modes), table_modes=list(self.table_modes), density_kg_m3=self.density,
                    **({} if not self.hover_tables else dict(hover_tables="OF-0010 revision 11 (B): the hover modes on Makani's "
                                                                          "database, the rings carried on; blown wing only there")),
                    wind_world_m_s=self.wind.tolist(), **({} if self.wind_field is None else dict(
                        wind_field=dict(self.wind_field.settings(), note="OF-0010 revision 19 addendum 6 (B1): sections, wake, "
                                        "rings, strip and fuselage stations at their own heights; database flap increments, "
                                        "rotors (T3's ambient inflow) and parasitic drag at the body origin"))),
                    strip_case=STRIP_CASE, fuselage_mesh=FUSELAGE_MESH, crossflow_cdc=CROSSFLOW_CDC,
                    rings=(dict(nodes=RING_NODES, retained_periods=RING_RETAINED_PERIODS, physical_core_m=PHYSICAL_CORE_M,
                                surface_core_m=SURFACE_CORE_M, convection_strip="euler, cuda_tree" if self.gpu else "euler, tree", tree=RING_TREE,
                                circulation="T dt / (rho A) of the applied table thrust") if self.slipstream == "free" else
                           self._b_prime_ring_settings()),
                    vsm=dict(files=self.files, wake=dict(dt_s=PERIOD_S, core_radius_m=TV.CORE_RADIUS_M,
                                                         free_rows=TV.FREE_ROWS if self.free_rows is None else self.free_rows,
                                                         keep_rows=TV.KEEP_ROWS if self.keep_rows is None else self.keep_rows,
                                                         tree=TV.TREE, convection="euler",
                                                         **({} if self.keep_rows is None else
                                                            dict(keep_rows_option="OF-0010 revision 10 (WK)")),
                                                         **({} if self.free_rows is None else
                                                            dict(free_rows_option="OF-0010 revision 22 addendum 3 (B3, H4): "
                                                                 f"{self.free_rows} free rows instead of {TV.FREE_ROWS}")),
                                                         **({} if self.frozen_drift == "ambient" else
                                                            dict(frozen_drift="rings", frozen_drift_option="OF-0010 revision 22 "
                                                                 "addendum 4 (B3, H5): the frozen rows take the ambient velocity "
                                                                 "plus the rotor rings' induction")),
                                                         **({} if self.stall_lag is None else
                                                            dict(stall_lag=self.stall_lag, stall_lag_option="OF-0010 revision 22 "
                                                                 "addendum 8 (B3, H6): Oye-type first-order lag of the stalled "
                                                                 "sections' circulation and loads, tau = k c / |U_out|"))),
                             parasitic=dict(coefficient=TV.PARASITIC_COEFFICIENT, area_m2=TV.REFERENCE_AREA_M2)),
                    flap_increments="MakaniDatabase.loads(acting flaps) - MakaniDatabase.loads(zero flaps)", blown_wing=False,
                    orthonormalise=self.orthonormalise,
                    gpu=None if not self.gpu else self._gpu_used())
        if self.platform is not None:                  # OF-0010 P1 (the keys above are those of plan 3.8b)
            platform = dict(step=self.platform, **self.p1.settings(),
                            **({} if self.platform != "p2" else dict(p2=self.surfaces.settings())))
            if self.seeding == "strip":                # OF-0010 P5a' (revision 6): replaces P1's text on the seeding
                platform["seeding"] = ("strip: after Piszkin & Levinsky, adapted (OF-0010 revision 6; decision 0053 addendum 9): "
                                       "the strip-theory circulation at the geometric angle on panels stalled at the previous "
                                       "step, the previous circulation elsewhere; 'start' on a state's first step; one retry "
                                       "from zero")
            elif self.seeding != "zero":               # OF-0010 P5a: replaces P1's text on the seeding
                platform["seeding"] = ("mixed: after Piszkin & Levinsky, adapted (OF-0010 revision 5 addenda 2-3; decision 0053 "
                                       "addenda 7-8): the previous circulation on panels not stalled at the previous step, "
                                       "zero on stalled ones; 'start' on a state's first step; one retry from zero")
            if self.viscosity:                         # OF-0010 P5c (revision 7 addendum 1)
                platform["viscosity"] = ("spanwise artificial viscosity (OF-0010 revision 7 addendum 1): mu_i = max(0, -k S CL'/dz^2), "
                                         "k 0.035, per lifting surface, on while a panel is beyond its stall angles; after Li, "
                                         "Gaunaa, Pirrung & Lonbaek (2026) as in the TU Delft Vortex-Step-Method")
            if self.local_strip:                       # OF-0010 P5d (revision 9)
                platform["local_strip"] = ("local strip fallback (OF-0010 revision 9; decision 0053 addendum 11): on a step "
                                           "that does not solve, the panels with |r_i| >= 1/2 max|r| of the last exhausted "
                                           "Picard attempt carry no bound circulation for that step (strip theory at the "
                                           "local flow), re-solved by the same rule; at most 3 rounds, then W7")
            out.update(platform=platform,
                       flap_increments="none (P1: the control surfaces are in the section tables and the tail's geometry)",
                       vsm_solve=("every mode and " + ("both groups" if self.platform == "p1" else "the one group (P2)") +
                                  (": TAKEOFF_LANDING_SOLVE, zero seeding with one retry from the previous circulation "
                                   if self.seeding == "zero" else
                                   ": TAKEOFF_LANDING_SOLVE, the mixed start (after Piszkin & Levinsky) with one retry from zero "
                                   if self.seeding == "mixed" else
                                   ": TAKEOFF_LANDING_SOLVE, the strip-theory start on stalled panels (after Piszkin & Levinsky) "
                                   "with one retry from zero ") +
                                  "(PlatformGroup); the section hand-over is not used"))
        return out

    def _b_prime_ring_settings(self):
        """OF-0010 revision 15 with addendum 1 (B'): the slipstream's and the hub inflow's settings."""
        model = dict(t3="paper 1's tube T3: frozen rings drifting with the wind - (gamma/2) axis at release, "
                        "Gamma = gamma (V_inf + gamma/2) dt (ambient spacing)",
                     t3_4s="tube T3 with the rings kept 4 s (sensitivity)",
                     model_d="paper 1's model D: free rings, their convection seeing the bound system with r_b = 0.64 m, "
                             "Gamma = gamma (V + gamma/2) dt",
                     none="no rings")[self.slipstream]
        return dict(slipstream=self.slipstream, model=model, nodes=RING_NODES, retained_periods=self.ring_retained_periods,
                    surface_core_m=self.ring_surface_core_m, ring_core_m=self.ring_physical_core_m,
                    ring_options={k: v for k, v in RING_SETTINGS[self.slipstream]["ring"].items()},
                    gamma="-V + sqrt(V^2 + 2T/(rho A)), T the applied thrust, V its axial inflow; no ring where the radicand < 0, "
                          "V + gamma/2 <= 0 or (t3) V_inf + gamma/2 <= 0 (recorded)",
                    standalone="drift only" if self.slipstream in ("t3", "t3_4s") else ("none" if self.slipstream == "none" else
                                                                             "free, Lamb-Oseen 0.45 m, euler"),
                    hub_inflow=self.hub_inflow if self.hub_inflow == "kinematic" else
                    "induced: after this period's solve, the solved state's bound, trailing and free rows plus the other "
                    "discs' rings (core 0.45 m, singularity zero) at the hubs of this period's pose; kinematic in the "
                    "table modes and in strip periods", strip_ring_core_m=self.ring_surface_core_m,
                    states=None if self.state_folder is None else dict(folder=self.state_folder, offsets=STATE_OFFSETS),
                    **({} if self.wake_frames is None else dict(wake_frames=dict(
                        self.wake_frames, note="OF-0010 revision 27: the fields of the states, float32/int32"))))

    def _gpu_used(self):
        """The GPU choice as the wake configuration in use holds it (review P38B3-08)."""
        wake = self.backend.wake
        return dict(backend=wake.velocity_backend, settings=dict(precision=wake.cuda.precision, suspect=wake.cuda.suspect,
                                                                  **({} if wake.cuda.control == "cpu" else
                                                                     dict(control=wake.cuda.control))),
                    block=wake.cuda.block)

    # ----- the wind (OF-0010 revision 19 addendum 6, B1) --------------------------------------------------------------
    ACCEPTS_WIND_FIELD = True
    wind_rates_body = None                             # B1R-04: the table path's wind-gradient rates of the period

    def set_period_wind(self, origin_wind_world, wind_rates_body, field):
        """B1: called by ``run_turnkey`` at the start of every period when it flies with a ``ShearWind``: the wind at the
        body origin of the period's start (Makani's convention for the database, the rotors and the parasitic drag, as
        the table path) and the field itself (the sections, the wake, the rings and the strip take it point by point).
        ``wind_rates_body`` is the table path's: the hook adds it to the body rates where it looks up Makani's database
        (the flap increments, B1R-04); the hook's table periods leave the loads to the plant, which adds it itself."""
        self.wind_field = field
        self.origin_wind = np.asarray(origin_wind_world, dtype=float).reshape(3).copy()
        self.wind_rates_body = np.asarray(wind_rates_body, dtype=float).reshape(3).copy()

    def b3_diagnostics(self):
        """OF-0010 revision 22 addendum 8: with any of the B3 options (H4, H5, H6) the main group records the induced
        velocity's maximum at the load points and the newest shed row's maximum strength; without them nothing is added."""
        return self.free_rows is not None or self.frozen_drift != "ambient" or self.stall_lag is not None

    def origin_or_uniform_wind(self):
        return self.wind if self.wind_field is None else self.origin_wind

    def wind_at(self, points):
        """B1: the field's wind at world points (one row per point)."""
        return self.wind_field.winds_world(points)

    def inflow(self):
        """The wake library's inflow: the uniform vector as before, or with the field a function of points and time."""
        if self.wind_field is None:
            return self.wind
        return FieldInflow(self.wind_field)                # picklable (checkpoints, review B1R-01)

    # ----- the rings ------------------------------------------------------------------------------------------------
    def _convect_standalone(self, to_time_s):
        """One Euler step of the standalone rings with their own induction and the uniform wind (W3); with the slipstream
        't3' the drift only (OF-0010 revision 15 addendum 1, BP-06): x + dt (wind + drift)."""
        if self.slipstream in ("t3", "t3_4s"):
            if self.rings.count:
                if self.wind_field is None:
                    self.rings = self.rings.moved(self.rings.nodes + PERIOD_S*(self.wind[None, :] + self.rings.drift_m_s))
                else:                                  # B1: the shear's wind at each ring node
                    self.rings = self.rings.moved(self.rings.nodes + PERIOD_S*(self.wind_at(self.rings.nodes) + self.rings.drift_m_s))
            self.rings_time_s = float(to_time_s)
            return
        if self.rings.count:
            wind = self.wind
            backend = dict(backend="cuda_tree", tree=self.tree, cuda=self.cuda) if self.gpu else dict(backend="tree", tree=self.tree)
            ambient = (lambda nodes: np.broadcast_to(wind, nodes.shape).copy()) if self.wind_field is None else self.wind_at
            try:
                self.rings = convect_rings(self.rings, PERIOD_S, CoreModel.LAMB_OSEEN, self.ring_physical_core_m,
                                           ambient=ambient, integrator="euler",
                                           **backend)
            except (ValueError, FloatingPointError) as error:          # revision 2 D1 (review P38B-06)
                error.failed_module = "rotor rings"
                raise
            except Exception as error:                                 # revision 3 addendum 2 (P38B3-06)
                if self.gpu and getattr(error, "failed_module", None) is None:
                    error.failed_module = "rotor rings (GPU)"
                raise
        self.rings_time_s = float(to_time_s)

    def _take_rings_from_vsm(self, time_s):
        """Leaving the VSM: the rings of the main state; convected to ``time_s`` when they are a period behind."""
        state = None if self.main is None else self.main.state
        rings = None if state is None else state.rings
        self.rings = rings if rings is not None else self.rings
        behind = state is not None and self.main.last_time_s is not None and self.main.last_time_s < time_s - 0.5*PERIOD_S
        self.main = self.pylons = None
        self.kind = "strip"
        if behind:
            self._convect_standalone(time_s)
        else:
            self.rings_time_s = float(time_s)

    def _ring_inflow_fields(self):
        """Addendum 11 (C1-R7): the rings' induced velocity at the strip points, body axes, mean and largest size by
        surface - the rings that acted in this period (the main state's in a VSM period, the standalone ones otherwise)."""
        rings = self.main.state.rings if (self.kind == "vsm" and self.main is not None and self.main.state is not None
                                          and self.main.state.rings is not None) else self.rings
        sections = self.sections
        R, origin = np.asarray(self.pose.rotation, dtype=float), np.asarray(self.pose.position_m, dtype=float)
        points = origin + sections.points@R.T
        if rings is None or not rings.count:
            v_body = np.zeros_like(points)
        else:
            host = lambda a: a.get() if hasattr(a, "get") else a                     # noqa: E731 (device arrays)
            v = segments_velocity(points, np.asarray(host(rings.nodes), dtype=float), np.asarray(host(rings.edges)),
                                  np.asarray(host(rings.circulation), dtype=float),
                                  model=CoreModel.LAMB_OSEEN, core_radius=self.ring_surface_core_m, singularity="zero",
                                  backend="numba")
            v_body = v@R
        mean, largest = {}, {}
        for index, name in enumerate(sections.names):
            mask = np.asarray(sections.surface) == index
            if mask.any():
                mean[str(name)] = v_body[mask].mean(axis=0).tolist()
                largest[str(name)] = float(np.linalg.norm(v_body[mask], axis=1).max())
        return dict(whole_ring_inflow_body_m_s=mean, whole_ring_inflow_max_m_s=largest)

    def _qs_ring_induction(self, points):
        """Addendum 12: the standalone rings' induction at the quasi-steady closure and load points (surface core, as
        the strip's ``ring_induction(points, 'raise')``)."""
        return self.ring_induction(points, "raise")

    def ring_induction(self, points, singularity):
        if not self.rings.count:
            return np.zeros_like(points)
        return segments_velocity(points, self.rings.nodes, self.rings.edges, self.rings.circulation, model=CoreModel.LAMB_OSEEN,
                                 core_radius=self.ring_surface_core_m, singularity=singularity, backend="numba")

    def shed(self, rings, pose, thrusts, time_s):
        """The rings of one period (W3): Gamma = T dt/(rho A) per rotor, the newest 80 periods kept. Returns (rings, Gammas)."""
        gammas = []
        for index, (disc, thrust) in enumerate(zip(self.discs, thrusts)):
            placed = disc.placed(pose.rotation, pose.position_m)
            gamma = float(thrust)*PERIOD_S/(self.density*placed.area_m2)
            gammas.append(gamma)
            if gamma != 0.0:
                rings = rings.shed(ShedRing(placed.rim(RING_NODES), ring_edges(RING_NODES), gamma, float(time_s), 0.0, 0.0), index)
        horizon = float(time_s) - (RING_RETAINED_PERIODS - 0.5)*PERIOD_S
        return rings.released_since(horizon), gammas

    def shed_b_prime(self, rings, pose, thrusts, freestreams, time_s):
        """OF-0010 revision 15 with addendum 1 (B'): one ring per disc from the applied thrust T and its axial inflow V:
        gamma = -V + sqrt(V^2 + 2T/(rho A)); 't3': Gamma = gamma (V_inf + gamma/2) dt with V_inf the ambient axial inflow,
        the drift -(gamma/2) axis; 'model_d': Gamma = gamma (V + gamma/2) dt; no ring where the radicand is negative,
        V + gamma/2 <= 0 or ('t3') V_inf + gamma/2 <= 0 (recorded); a non-finite radicand raises. Returns (rings, fields)."""
        fields = dict(whole_ring_V_m_s=[], whole_ring_V_ambient_m_s=[], whole_ring_sheet_strength_m_s=[],
                      whole_ring_circulation_m2_s=[], whole_ring_spacing_m=[], whole_ring_skipped=[])
        if self.slipstream != "none":
            for index, (disc, thrust, inflow) in enumerate(zip(self.discs, thrusts, freestreams)):
                placed = disc.placed(pose.rotation, pose.position_m)
                ambient = float(placed.axial_freestream(self.origin_or_uniform_wind() - pose.point_velocity(placed.centre_m[None, :])[0]))
                thrust, inflow = float(thrust), float(inflow)
                radicand = inflow*inflow + 2.0*thrust/(self.density*placed.area_m2)
                if not np.isfinite(radicand):
                    error = FloatingPointError(f"non-finite momentum radicand at disc {index}: V={inflow!r}, T={thrust!r}")
                    error.failed_module = "rotor rings"
                    raise error
                gamma = spacing = circulation = None
                reason = None
                if radicand < 0.0:
                    reason = "no real root"
                else:
                    gamma = -inflow + float(np.sqrt(radicand))
                    if not inflow + 0.5*gamma > 0.0:
                        reason = "V + gamma/2 <= 0"
                    elif self.slipstream in ("t3", "t3_4s"):
                        spacing = (ambient + 0.5*gamma)*PERIOD_S
                        if not spacing > 0.0:
                            reason = "V_inf + gamma/2 <= 0"
                        else:
                            circulation = gamma*spacing
                    else:
                        spacing = (inflow + 0.5*gamma)*PERIOD_S
                        circulation = gamma*spacing
                fields["whole_ring_V_m_s"].append(inflow)
                fields["whole_ring_V_ambient_m_s"].append(ambient)
                fields["whole_ring_sheet_strength_m_s"].append(gamma)
                fields["whole_ring_circulation_m2_s"].append(circulation)
                fields["whole_ring_spacing_m"].append(spacing)
                if reason is not None:
                    fields["whole_ring_skipped"].append([index, reason])
                    self.ring_skips.append(dict(k=int(round(time_s/PERIOD_S)), disc=index, reason=reason, V=inflow, T=thrust,
                                                C_T=thrust/(0.5*self.density*placed.area_m2*max(inflow*inflow, 1e-12)),
                                                skipped_gamma_dt=abs(thrust)*PERIOD_S/(self.density*placed.area_m2)))
                    continue
                ring = ShedRing(placed.rim(RING_NODES), ring_edges(RING_NODES), float(circulation), float(time_s), float(gamma),
                                float(spacing))
                rings = rings.shed(ring, index, axis=placed.axis if self.slipstream in ("t3", "t3_4s") else None)
        horizon = float(time_s) - (self.ring_retained_periods - 0.5)*PERIOD_S
        return rings.released_since(horizon), fields

    def hub_induced(self, k, state):
        """OF-0010 revision 15 addendum 1 (BP-01, B'): after this period's solve, the velocity induced at each hub of this
        period's pose (world axes) by the solved state's bound, trailing and free rows and by the other discs' rings;
        None where the rotors keep the kinematic inflow (the option 'kinematic', table modes, strip periods)."""
        if self.hub_inflow == "induced" and self.source == "vsm" and self.quasi_steady and self.main is not None:
            return self._qs_hub_induced()                # OF-0011 revision 7 addendum 12 (C1-R6)
        if self.hub_inflow != "induced" or self.source != "vsm" or self.kind != "vsm":
            return None
        from openfawe.aero.wake import kernel_options
        from openfawe.rotor.coupled import hub_wing_winds
        current = self.main.state
        config = current.config
        wing_only = replace(current, rings=None)
        rings = current.rings
        ring_core = config.core_radius_m if rings is None or getattr(rings, "core_m", None) is None else rings.core_m
        pose = self.pose
        placed_discs = [disc.placed(pose.rotation, pose.position_m) for disc in self.discs]
        # node 3 contract addendum 9 supplement 4: the network's part at all hubs in one sum, bit for bit the per-disc
        # hub_apparent_wind (one network packing per period instead of one per disc)
        wing_part = list(hub_wing_winds(wing_only, config, [placed.centre_m for placed in placed_discs]))
        ring_part = []
        for index, placed in enumerate(placed_discs):
            if rings is not None and len(rings.edges):
                ring_part.append(rings.induced_velocity(placed.centre_m[None, :], config.core_model, ring_core,
                                                        exclude_source=index, singularity="zero", **kernel_options(config))[0])
            else:
                ring_part.append(np.zeros(3))
        wing_part, ring_part = np.asarray(wing_part, dtype=float), np.asarray(ring_part, dtype=float)
        self.fields.update(whole_hub_induced_wing_m_s=wing_part.tolist(), whole_hub_induced_rings_m_s=ring_part.tolist())
        return wing_part + ring_part

    def _qs_hub_induced(self):
        """Addendum 12: the quasi-steady solution's bound vortices and straight trailing legs at each hub, plus the
        standalone rings of the other discs (as ``hub_induced``)."""
        from openfawe.aero.wake import kernel_options
        config = self.backend.wake
        rings = self.rings
        ring_core = config.core_radius_m if getattr(rings, "core_m", None) is None else rings.core_m
        pose = self.pose
        centres = np.array([disc.placed(pose.rotation, pose.position_m).centre_m for disc in self.discs], dtype=float)
        wing_part = self.main.induced_at(centres)
        ring_part = []
        for index, centre in enumerate(centres):
            if rings is not None and len(rings.edges):
                ring_part.append(rings.induced_velocity(centre[None, :], config.core_model, ring_core, exclude_source=index,
                                                        singularity="zero", **kernel_options(config))[0])
            else:
                ring_part.append(np.zeros(3))
        wing_part, ring_part = np.asarray(wing_part, dtype=float), np.asarray(ring_part, dtype=float)
        self.fields.update(whole_hub_induced_wing_m_s=wing_part.tolist(), whole_hub_induced_rings_m_s=ring_part.tolist())
        return wing_part + ring_part

    def _state_arrays(self, k):
        """The fields of a wake state (revision 15 item 5, addendum 1 item 9), as ``_save_state`` writes them."""
        from openfawe.aero.cores import edge_radii
        from openfawe.aero.wake import wake_network
        current = self.main.state
        nodes, edges, gamma = wake_network(current)
        radii = edge_radii(current.wing, edges, current.cores, current.config.core_radius_m)
        ring_core = current.config.core_radius_m if current.rings is None or getattr(current.rings, "core_m", None) is None \
            else current.rings.core_m
        rings = current.rings
        name = self.names.get(self.last_mode, self.last_mode)
        empty = np.zeros((0, 3))
        return name, dict(k=int(k), time_s=float(current.time_s), mode=str(name), wake_nodes=nodes, wake_edges=edges,
                            wake_gamma=gamma, wake_radii=np.asarray(radii, dtype=float), core_model=int(current.config.core_model),
                            ring_core_m=float(ring_core), bound_circulation=np.asarray(current.solution.circulation, dtype=float),
                            ring_nodes=empty if rings is None else rings.nodes,
                            ring_edges=np.zeros((0, 2), dtype=int) if rings is None else rings.edges,
                            ring_circulation=np.zeros(0) if rings is None else rings.circulation,
                            ring_source=np.zeros(0, dtype=int) if rings is None else rings.source,
                            ring_release_times_s=np.zeros(0) if rings is None else rings.release_times_s,
                            ring_drift_m_s=empty if rings is None or rings.drift_m_s is None else rings.drift_m_s,
                            rotation=np.asarray(self.pose.rotation, dtype=float), position_m=np.asarray(self.pose.position_m, dtype=float),
                            velocity_m_s=np.asarray(self.pose.velocity_m_s, dtype=float),
                            angular_velocity_rad_s=np.asarray(self.pose.angular_velocity_rad_s, dtype=float))

    def _save_state(self, k):
        """OF-0010 revision 15 item 5 with addendum 1 item 9: one wake state of a vortex mode (float64, npz)."""
        name, arrays = self._state_arrays(k)
        folder = Path(self.state_folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f"state-{name}-k{int(k)}.npz"
        np.savez_compressed(path, **arrays)
        self.states_saved.append(dict(k=int(k), mode=str(name), file=path.name))

    @staticmethod
    def frame_arrays(arrays):
        """OF-0010 revision 27: a state's fields for a wake frame - floats as float32, integers as int32, the rest as is."""
        out = {}
        for key, value in arrays.items():
            a = np.asarray(value)
            if a.dtype.kind == "f":
                out[key] = a.astype(np.float32)
            elif a.dtype.kind in "iu":
                out[key] = a.astype(np.int32)
            else:
                out[key] = a
        return out

    def _save_frame(self, k):
        """OF-0010 revision 27: one wake frame (``frame-k{k:07d}.npz`` in the frames' folder)."""
        _, arrays = self._state_arrays(k)
        folder = Path(self.wake_frames["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f"frame-k{int(k):07d}.npz"
        np.savez_compressed(path, **self.frame_arrays(arrays))
        saved = self.frames_saved
        saved["count"] += 1
        saved["bytes"] += path.stat().st_size
        saved["first_k"] = int(k) if saved["first_k"] is None else saved["first_k"]
        saved["last_k"] = int(k)

    # ----- the loads ------------------------------------------------------------------------------------------------
    def _flap_increment(self, pose, flaps_rad, thrust_coeff):
        """W6 in body axes about the origin."""
        through_air = pose.rotation.T@(pose.velocity_m_s - self.origin_or_uniform_wind())
        omega = pose.rotation.T@pose.angular_velocity_rad_s
        if self.wind_field is not None:                # B1R-04: Makani adds the wind-gradient rates to the body rates
            omega = omega + self.wind_rates_body
        try:
            f1, m1, _ = self.database.loads(through_air, omega, self.density, np.asarray(flaps_rad, dtype=float), float(thrust_coeff))
            f0, m0, _ = self.database.loads(through_air, omega, self.density, np.zeros(8), float(thrust_coeff))
        except (ValueError, FloatingPointError) as error:
            error.failed_module = "aerodynamics (flap increments)"
            raise
        return np.asarray(f1) - np.asarray(f0), np.asarray(m1) - np.asarray(m0)

    def strip_loads(self, pose, flap_increment):
        """W4: force and moment about the body origin in body axes, and fields; ``flap_increment`` from ``_flap_increment``
        (with ``platform='p1'``: the period's ``P1Period``, whose sections replace the case's and which adds no increment)."""
        R, origin = np.asarray(pose.rotation, dtype=float), np.asarray(pose.position_m, dtype=float)
        sections = self.sections if self.platform is None else flap_increment.sections
        try:
            points = origin + sections.points@R.T
            u_world = (self.wind if self.wind_field is None else self.wind_at(points)) - pose.point_velocity(points) \
                + self.ring_induction(points, "raise")
            force, couple, alpha = sections.loads(u_world@R, self.density)
            f_sections, m_sections = sections.totals(force, couple)
            centres, diameters, lengths = self.stations
            station_points = origin + centres@R.T
            u_stations = (self.wind if self.wind_field is None else self.wind_at(station_points)) \
                - pose.point_velocity(station_points) + self.ring_induction(station_points, "zero")
            f_fuselage, m_fuselage = crossflow_drag_body(centres, diameters, lengths, u_stations@R, self.density)
        except (ValueError, FloatingPointError) as error:
            error.failed_module = "aerodynamics (strip)"
            raise
        near = int(np.count_nonzero(segment_point_distances(points, self.rings.nodes, self.rings.edges) <= self.ring_surface_core_m)) \
            if self.rings.count else 0
        if self.platform is not None:                  # OF-0010 P1: no database increment
            fields = dict(whole_strip_near_segments=near, whole_strip_sections_force_body_N=f_sections.tolist(),
                          whole_strip_sections_moment_origin_body_Nm=m_sections.tolist(),
                          whole_strip_fuselage_force_body_N=f_fuselage.tolist(), whole_strip_fuselage_moment_origin_body_Nm=m_fuselage.tolist(),
                          whole_strip_alpha_range_rad=[float(np.min(alpha)), float(np.max(alpha))])
            return f_sections + f_fuselage, m_sections + m_fuselage, fields
        f_flap, m_flap = flap_increment
        fields = dict(whole_strip_near_segments=near, whole_strip_sections_force_body_N=f_sections.tolist(), whole_strip_sections_moment_origin_body_Nm=m_sections.tolist(),
                      whole_strip_fuselage_force_body_N=f_fuselage.tolist(), whole_strip_fuselage_moment_origin_body_Nm=m_fuselage.tolist(),
                      whole_flap_force_body_N=f_flap.tolist(), whole_flap_moment_origin_body_Nm=m_flap.tolist(),
                      whole_strip_alpha_range_rad=[float(np.min(alpha)), float(np.max(alpha))])
        return f_sections + f_fuselage + f_flap, m_sections + m_fuselage + m_flap, fields

    def vsm_loads(self, pose, flap_increment, fields):
        """W5: force and moment about the body origin in body axes; raises the groups' FAILURES (the flap increment,
        ``_flap_increment``'s, is computed by the caller outside the capture of those failures)."""
        starting = self.main is None
        p1 = self.platform is not None
        if starting and not p1:
            self.main = CompanionSurfaces("whole_main", self.backend, self.main_wing, self.inflow(), handover=self.main_handover)
            self.pylons = CompanionSurfaces("whole_pylons", self.backend, self.pylon_body, self.inflow(), handover=self.pylon_handover)
        elif starting and self.platform == "p2" and self.quasi_steady:   # OF-0011 rev 7 add 12 (C1-R6)
            from openfawe.flight.quasi_steady import QuasiSteadyGroup
            self.main = QuasiSteadyGroup("whole_main", self.inflow(), density_kg_m3=self.density,
                                         induction=self._qs_ring_induction)
            self.pylons = None
        elif starting and self.platform == "p2":       # OF-0010 P2: one vortex system of 81 panels
            self.main = PlatformGroup("whole_main", self.backend, self.inflow(), surfaces=self.surfaces, seeding=self.seeding,
                                      **({} if not self.viscosity else dict(viscosity=True)),
                                      **({} if not self.local_strip else dict(local_strip=True)),
                                      **({} if not self.b3_diagnostics() else dict(diagnostics=True)))  # revision 22 add. 8
            self.pylons = None
        elif starting:                                 # OF-0010 P1: the fixed choice, the period's geometry and tables
            self.main = PlatformGroup("whole_main", self.backend, self.inflow(), seeding=self.seeding,
                                      **({} if not self.viscosity else dict(viscosity=True)))
            self.pylons = PlatformGroup("whole_pylons", self.backend, self.inflow(), seeding=self.seeding,
                                        **({} if not self.viscosity else dict(viscosity=True)))
        if self.platform == "p2":
            return self._p2_vsm_loads(pose, flap_increment, fields, starting)
        main_args = () if not p1 else (flap_increment.main_wing, flap_increment.main_polar)
        pylon_args = () if not p1 else (flap_increment.pylon_wing, flap_increment.pylon_polar)
        try:
            force_main, moment_main, main_fields = self.main.extra_loads(pose, fields, *main_args)
        except Exception as error:                     # revision 3 addendum 2 (P38B3-06): attribution only, not a fallback
            if self.gpu and not isinstance(error, FAILURES) and getattr(error, "failed_module", None) is None:
                error.failed_module = "aerodynamics (VSM on the GPU)"
            raise
        if starting:                                   # W7: the rings join the main state after its start solve
            self.main.state = replace(self.main.state, rings=self.rings)
            self.kind = "vsm"
        fields.update(main_fields)
        try:
            force_pylons, moment_pylons, pylon_fields = self.pylons.extra_loads(pose, fields, *pylon_args)
        except Exception as error:                     # revision 3 addendum 2 (P38B3-06)
            if self.gpu and not isinstance(error, FAILURES) and getattr(error, "failed_module", None) is None:
                error.failed_module = "aerodynamics (VSM on the GPU)"
            raise
        fields.update(pylon_fields)
        relative = pose.velocity_m_s - self.origin_or_uniform_wind()
        drag = -TV.PARASITIC_COEFFICIENT*0.5*self.density*TV.REFERENCE_AREA_M2*float(np.linalg.norm(relative))*relative
        R = pose.rotation
        if p1:                                         # OF-0010 P1: no database increment
            fields.update(whole_parasitic_force_body_N=(R.T@drag).tolist())
            return R.T@(force_main + force_pylons + drag), R.T@(moment_main + moment_pylons)
        f_flap, m_flap = flap_increment
        fields.update(whole_parasitic_force_body_N=(R.T@drag).tolist(), whole_flap_force_body_N=f_flap.tolist(),
                      whole_flap_moment_origin_body_Nm=m_flap.tolist())
        force = R.T@(force_main + force_pylons + drag) + f_flap
        moment = R.T@(moment_main + moment_pylons) + m_flap
        return force, moment

    def _p2_vsm_loads(self, pose, period, fields, starting):
        """OF-0010 P2: the one vortex system (81 panels, the period's turned geometry and tables); the parasitic drag."""
        try:
            force_main, moment_main, main_fields = self.main.extra_loads(pose, fields, period.assembly, period.polar)
        except Exception as error:                     # revision 3 addendum 2 (P38B3-06): attribution only, not a fallback
            if self.gpu and not isinstance(error, FAILURES) and getattr(error, "failed_module", None) is None:
                error.failed_module = "aerodynamics (VSM on the GPU)"
            raise
        if starting and self.quasi_steady:             # add 12: no wake state; the rings stay standalone
            self.kind = "qs"
        elif starting:                                 # W7: the rings join the main state after its start solve
            self.main.state = replace(self.main.state, rings=self.rings)
            self.kind = "vsm"
        fields.update(main_fields)
        relative = pose.velocity_m_s - self.origin_or_uniform_wind()
        drag = -TV.PARASITIC_COEFFICIENT*0.5*self.density*TV.REFERENCE_AREA_M2*float(np.linalg.norm(relative))*relative
        R = pose.rotation
        fields.update(whole_parasitic_force_body_N=(R.T@drag).tolist())
        return R.T@(force_main + drag), R.T@moment_main

    def loads(self, k, state, flight_mode, flaps_rad, thrust_coeff_used):
        if self.mode == "record":
            self.periods["table"] += 1
            return "table", None, {}
        if not isinstance(state, RigidState):
            raise TypeError("the plant state must be a RigidState")
        tick = time.perf_counter()
        self.gpu_mark = self._gpu_totals()
        mode, t = int(flight_mode), k*PERIOD_S
        pose = replace(state, time_s=t).pose(self.body)
        if mode != self.last_mode:
            self.fallback_mode = None
            self.mode_entry_k = int(k)
        self.last_mode = mode
        fields = {}
        if self.orthonormalise:                        # revision 2 B2: every computation of the hook uses this pose
            rotation = np.asarray(pose.rotation, dtype=float)
            nearest = nearest_rotation(rotation)
            orth = dict(whole_attitude_orthogonality_error=float(np.max(np.abs(rotation.T@rotation - np.eye(3)))),
                        whole_attitude_correction=float(np.max(np.abs(nearest - rotation))))
            fields.update(orth)
            pose = Pose(pose.time_s, pose.position_m, nearest, pose.velocity_m_s, pose.angular_velocity_rad_s)
        else:
            orth = {}
        wants_vsm = mode in self.vsm_numbers and self.fallback_mode != mode
        if mode in self.table_numbers:                 # W2: on the perch, the database term acts; the rings are carried on
            if self.kind == "vsm":
                self._take_rings_from_vsm(t)
            elif self.rings_time_s is not None and self.rings_time_s < t - 0.5*PERIOD_S:
                self._convect_standalone(t)
            self.kind = "strip"
            self.periods["table"] += 1
            fields.update(whole_model="table", whole_mode=self.names.get(mode, mode), whole_ring_count=int(self.rings.count//RING_NODES),
                          whole_ring_nodes=int(self.rings.count), whole_solve_s=time.perf_counter() - tick)
            self.pose, self.fields = pose, fields
            self.source = "table"
            return "table", None, fields
        if self.platform is None:
            flap_increment = self._flap_increment(pose, flaps_rad, thrust_coeff_used)   # revision 2 D1: outside the capture
            platform_fields = {}
        else:                                          # OF-0010 P1: the period's surfaces, also outside the capture
            try:
                flap_increment = self.p1.period(flaps_rad)
            except (ValueError, FloatingPointError) as error:
                error.failed_module = "aerodynamics (platform sections)"
                raise
            platform_fields = flap_increment.fields()
        if wants_vsm:
            if self.kind != "vsm" and self.rings_time_s is not None and self.rings_time_s < t - 0.5*PERIOD_S:
                self._convect_standalone(t)
            try:
                force, moment = self.vsm_loads(pose, flap_increment, fields)
                source = "vsm"
            except FAILURES as error:                  # W7: this period and the rest of the mode on the strip
                self.fallbacks.append(dict(k=int(k), mode=self.names.get(mode, mode), type=type(error).__name__,
                                           message=str(error)[:300], module=getattr(error, "failed_module", None),
                                           cause=failure_cause(error)))
                if self.seeding != "zero":             # OF-0010 P5a (review P5aC-09): the first, mixed-seeded attempt
                    first = getattr(error, "first_seed_failure", None)
                    self.fallbacks[-1]["first_seed_failure"] = None if first is None else {
                        key: (value if isinstance(value, (int, float, str, type(None))) else str(value)) for key, value in dict(first).items()}
                if self.local_strip:                   # OF-0010 P5d: the rounds tried before W7
                    self.fallbacks[-1]["local_strip"] = getattr(error, "local_strip", None)
                fields = dict(orth, whole_vsm_failure=self.fallbacks[-1])
                self._take_rings_from_vsm(t)
                self.fallback_mode = mode
                force, moment, strip_fields = self.strip_loads(pose, flap_increment)
                fields.update(strip_fields)
                source = "fallback"
        else:
            if self.kind == "vsm":
                self._take_rings_from_vsm(t)
            elif self.rings_time_s is not None and self.rings_time_s < t - 0.5*PERIOD_S:
                self._convect_standalone(t)
            self.kind = "strip"
            force, moment, strip_fields = self.strip_loads(pose, flap_increment)
            fields.update(strip_fields)
            source = "fallback"                       # every flying mode is a VSM mode: the strip only falls back
        held = held_loads(self.body, pose.rotation, pose.rotation@force, pose.rotation@moment)
        rings = self.main.state.rings if self.kind == "vsm" else self.rings
        fields.update(platform_fields)
        if self.platform == "p2" and source == "vsm":    # OF-0010 P2 (review P2C-04): the rule's flag on solved periods only
            fields["platform_p2_surface_cores"] = True
        if self.seeding != "zero" and source == "vsm":   # OF-0010 P5a
            fields["platform_seeding"] = self.seeding
        if self.viscosity and source == "vsm":           # OF-0010 P5c
            fields["platform_viscosity"] = True
        if self.local_strip and source == "vsm":         # OF-0010 P5d
            fields["platform_local_strip"] = True
        fields.update(whole_model=source, whole_mode=self.names.get(mode, mode), whole_force_body_N=held.force_body_N.tolist(),
                      whole_moment_cm_body_Nm=held.moment_cm_body_Nm.tolist(),
                      whole_ring_count=int(0 if rings is None else rings.count//RING_NODES),
                      whole_ring_nodes=int(0 if rings is None else rings.count), whole_solve_s=time.perf_counter() - tick)
        self.periods[source] += 1
        self.pose, self.fields = pose, fields
        self.source = source
        return source, held, fields

    def after_loads(self, k, state, loads):
        """W3: this period's rings from the applied thrusts; in a strip period they convect one step."""
        if self.mode == "record":
            return
        t = k*PERIOD_S
        thrusts = [float(v) for v in loads["rotor_thrust_N"]]
        if self.ring_inflow:                            # OF-0011 rev 7 addendum 11: before this period's rings are shed
            self.fields.update(self._ring_inflow_fields())
        if self.slipstream != "free":                  # OF-0010 revision 15 with addendum 1 (B')
            freestreams = [float(v) for v in loads["rotor_axial_freestream_m_s"]]
            name = self.names.get(self.last_mode, self.last_mode)
            if (self.state_folder is not None and self.kind == "vsm" and self.source == "vsm" and name in STATE_OFFSETS
                    and self.mode_entry_k is not None and int(k) - self.mode_entry_k == STATE_OFFSETS[name]):
                self._save_state(k)                    # revision 15 addendum 7: the state the hub inflow was taken from,
                                                       # before this period's rings are shed and the oldest ones dropped
            if (self.wake_frames is not None and self.kind == "vsm" and self.source == "vsm"
                    and int(k) % self.wake_frames["every"] == 0):
                self._save_frame(k)                    # revision 27: at the same point as the states
            try:
                if self.kind == "vsm":
                    rings, ring_fields = self.shed_b_prime(self.main.state.rings, self.pose, thrusts, freestreams, t)
                    self.main.state = replace(self.main.state, rings=rings)
                else:
                    self.rings, ring_fields = self.shed_b_prime(self.rings, self.pose, thrusts, freestreams, t)
                    self._convect_standalone(t + PERIOD_S)
            except (ValueError, FloatingPointError) as error:
                error.failed_module = "rotor rings"
                raise
            self.fields.update(ring_fields)
        else:
            try:
                if self.kind == "vsm":
                    rings, gammas = self.shed(self.main.state.rings, self.pose, thrusts, t)
                    self.main.state = replace(self.main.state, rings=rings)
                else:
                    self.rings, gammas = self.shed(self.rings, self.pose, thrusts, t)
                    self._convect_standalone(t + PERIOD_S)
            except (ValueError, FloatingPointError) as error:
                error.failed_module = "rotor rings"
                raise
            self.fields.update(whole_ring_circulation_m2_s=gammas)
        if self.gpu:                                   # revision 3 B3: the GPU calls of this period (loads and after_loads)
            now = self._gpu_totals()
            self.fields.update(whole_gpu_calls=now[0] - self.gpu_mark[0], whole_gpu_suspect_pairs=now[1] - self.gpu_mark[1])

    def _gpu_totals(self):
        if not self.gpu:
            return (0, 0)
        from openfawe.aero.vortex_cuda_tree import Statistics
        return (int(Statistics.totals.get("calls", 0)), int(Statistics.totals.get("suspect_pairs", 0)))


def TV_MODE_ORDER():
    from openfawe.ground.turnkey import MODE_ORDER
    return MODE_ORDER
