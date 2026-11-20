"""Plan 3.8a (``vortex-crosswind-segment-contract.zh.md``, section 2): the segment-load hook of ``ground.turnkey.run_turnkey``
that gives the first CrosswindNormal stretch of a TurnKey flight the whole-aircraft aerodynamic load of the vortex method
of M1 (case CL1-DP of the multi-lap contract), V2-V6.

V2, the load (about the centre of mass, body axes, held over the period; computed once at the period's start) is the sum
of four items, M1's aerodynamic components unchanged:

1. the main lifting surfaces, VSM + free wake, under the section hand-over (``SectionHandover`` of the profile-drag case
   and the high-incidence XFOIL case, zero seeding with the recorded retry from the previous step) - a
   ``CompanionSurfaces`` group of its own (the pylon contract's E1 judged this bit-identical to ``run_free``'s solve);
2. the pylons as their own vortex system (``pylon_loop_checks.pylon_group``: no mutual induction with wing and tails);
3. parasitic drag ``-0.075 * q * 32.9 m^2`` at the body origin, along the origin's velocity through the air
   (``CrosswindLoop.extra_loads``);
4. the flap increments of Makani's low-incidence table (``FlapIncrements``) at this period's acting flap angles and the
   thrust coefficient recorded for this period (``thrust_coeff_used``) - the same pair the database term uses.

Wake (M1): 0.01 s steps, Lamb-Oseen core 0.15 m constant, Euler convection, ``CROSSWIND_SOLVE``, tree code (theta 0.2,
order 2, leaf 32), the latest 187 rows free, 1872 rows retained, density 1.026. Time base: step j of the segment is at
``(k_in + j) * 0.01`` s (the integer period number times the period; the plant's accumulated time is not used).

V3: the segment is the first unbroken run of periods whose mode at the period's start is CrosswindNormal; V4: its first
1872 periods are a shadow lead-in (solved along the flown trajectory from an empty wake, recorded, not applied); V5: from
the first period after it the database again and the vortex state is dropped; V6: any failure of a solve raises (the run
ends as a failure with the module named), nothing is retried beyond the hand-over rule's own retry.

``mode='record'`` solves nothing and marks every period 'table' (the control flight of G2(b)): R1 only.

``orthonormalise=True`` (opt-in, off by default, only with ``mode='vortex'``; contract revision 1 draft, item 1, option
丙 - only with the owner's consent): the attitude of the pose V2 starts from is replaced by the nearest rotation (polar
decomposition, U V^T of its SVD) for the whole V2 load - both vortex groups, the parasitic and flap items' body-axis
projections and the transfer to the centre of mass (review A-R07); the plant's state is not touched, and the pose's
position, velocity and angular velocity stay as the plant computed them. The blown-wing-on flight's attitude drifts to
|R^T R - I| = 1.2-1.44e-12 in CrosswindNormal (``attitude_orthogonality.py``), beyond ``Wing.transformed``'s 1e-12;
without the option such a pose ends the run (V6). With it, every shadow and vortex period also records
``vortex_attitude_orthogonality_error`` (max |R^T R - I| of the plant's attitude) and ``vortex_attitude_correction``
(max |nearest - R|); without it neither field is written. In a flight child the option is the segment string
``'vortex+orth'`` (``parse_segment``).

Records (R2, shadow and vortex periods only; every name starts with ``vortex_``): the sum ``vortex_force_body_N``,
``vortex_moment_cm_body_Nm``; per item the force and moment (main surfaces and pylons: the companion fields, prefixed
``vortex_main_`` and ``vortex_pylons_``; parasitic drag and flap increments in body axes about the origin); closure, weight,
iterations, seeding and retries of both groups (in the companion fields); the thrust coefficient of the flap increments;
the wall time of the period's solve (``vortex_solve_s``, a report; it differs from run to run and is compared nowhere).
"""

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
from openfawe.aero.handover import CROSSWIND_SOLVE, SectionHandover  # noqa: E402
from openfawe.aero.makani_flap_increments import FlapIncrements  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.flight.companion import CompanionSurfaces  # noqa: E402
from openfawe.flight.free_flight import held_loads  # noqa: E402
from openfawe.flight.kinematics import Pose  # noqa: E402
from openfawe.flight.rigid_body import RigidState  # noqa: E402

PERIOD_S = 0.01
SHADOW_PERIODS = 1872                                  # V4: the retained wake is full at 1872 rows
FREE_ROWS, KEEP_ROWS = 187, 1872                       # V2: M1's far wake (0.1 and 1 nominal lap of 18.72 s)
CORE_RADIUS_M, DENSITY_KG_M3 = 0.15, 1.026
TREE = dict(theta=0.2, order=2, leaf_size=32)
PARASITIC_COEFFICIENT, REFERENCE_AREA_M2 = 0.075, 32.9
SOURCES = ("table", "shadow", "vortex")


def parse_segment(segment):
    """The flight child's segment string: 'record', 'vortex' or 'vortex+orth' -> (mode, orthonormalise) (review A-R09:
    the option only with the vortex mode)."""
    mode, plus, option = str(segment).partition("+")
    if mode not in ("vortex", "record") or (plus and option != "orth") or (option == "orth" and mode != "vortex"):
        raise ValueError(f"unknown segment {segment!r} (record, vortex or vortex+orth)")
    return mode, option == "orth"


def case_files():
    """The files of V2, taken from the task scripts that define M1's case (one source of truth)."""
    import drag_checks as drag
    import pylon_loop_checks as pylon
    from drag_checks import checks
    return dict(main_low=drag.DRAG_CASE, main_high=checks.A5.HIGH_CASE, seeding=checks.A5.CONTRACT["seeding"],
                pylon_low=pylon.PYLON_LOW_CASE, pylon_high=pylon.PYLON_HIGH_CASE, flap_database=checks.FLAP_DATABASE,
                parasitic=drag.PARASITIC), pylon


def wake_backend():
    config = WakeConfig(dt_s=PERIOD_S, max_rows=KEEP_ROWS, core_radius_m=CORE_RADIUS_M, convection_integrator="euler",
                        core_evolution="constant", solve=CROSSWIND_SOLVE, free_convection_rows=FREE_ROWS,
                        retained_rows=KEEP_ROWS, density_kg_m3=DENSITY_KG_M3, velocity_backend="tree", tree=TreeSettings(**TREE))
    return AeroBackend("vsm_free_wake", wake=config)


class VortexSegment:
    """The segment-load hook (see the module text). ``body`` is the plant's rigid body, ``wind_world_m_s`` its uniform
    wind, ``crosswind_normal`` the controller's flight-mode number of CrosswindNormal."""

    def __init__(self, body, wind_world_m_s, crosswind_normal, *, mode="vortex", density_kg_m3=DENSITY_KG_M3,
                 shadow_periods=SHADOW_PERIODS, orthonormalise=False):
        if mode not in ("vortex", "record"):
            raise ValueError("mode must be 'vortex' or 'record'")
        if orthonormalise and mode != "vortex":
            raise ValueError("orthonormalise only with mode 'vortex'")
        if float(density_kg_m3) != DENSITY_KG_M3:
            raise ValueError(f"V2 uses the density {DENSITY_KG_M3} kg/m^3 of the wake settings")
        self.body, self.mode = body, mode
        self.wind = np.asarray(wind_world_m_s, dtype=float).reshape(3).copy()
        self.crosswind_normal = int(crosswind_normal)
        self.density = float(density_kg_m3)
        self.shadow_periods = int(shadow_periods)
        self.orthonormalise = bool(orthonormalise)
        self.k_in = self.k_last = None
        self.ended = False
        self.main = self.pylons = None
        self.files, pylon = case_files()
        if self.files["parasitic"] != PARASITIC_COEFFICIENT:
            raise ValueError("the parasitic drag coefficient of M1's case changed")
        if mode == "vortex":
            backend = wake_backend()
            low, high = load_aircraft(REPO/self.files["main_low"]), load_aircraft(REPO/self.files["main_high"])
            if low.wing.panels != high.wing.panels or not np.array_equal(low.wing.bound, high.wing.bound):
                raise ValueError("the two main-surface cases do not share one geometry")
            handover = SectionHandover(low.polar, high.polar, seeding=self.files["seeding"])
            self.main = CompanionSurfaces("vortex_main", backend, low.wing, self.wind, handover=handover)
            pylon_body, _, pylon_handover = pylon.pylon_parts()
            self.pylons = CompanionSurfaces("vortex_pylons", backend, pylon_body, self.wind, handover=pylon_handover)
            self.flaps = FlapIncrements.from_makani_json(REPO/self.files["flap_database"])
        self.periods = dict(table=0, shadow=0, vortex=0)

    def settings(self):
        out = dict(mode=self.mode, shadow_periods=self.shadow_periods, crosswind_normal=self.crosswind_normal,
                    wind_world_m_s=self.wind.tolist(), density_kg_m3=self.density, files=self.files,
                    wake=dict(dt_s=PERIOD_S, core_radius_m=CORE_RADIUS_M, free_rows=FREE_ROWS, keep_rows=KEEP_ROWS, tree=TREE,
                              convection="euler", core_evolution="constant", solve="CROSSWIND_SOLVE"),
                    parasitic=dict(coefficient=PARASITIC_COEFFICIENT, area_m2=REFERENCE_AREA_M2))
        if self.orthonormalise:                          # only when on: the settings of every earlier run stay as they were
            out["orthonormalise"] = True
        return out

    def _source(self, k, flight_mode):
        """V3-V5 from the mode at the start of period ``k``."""
        if self.ended:
            return "table"
        if int(flight_mode) == self.crosswind_normal:
            if self.k_in is None:
                self.k_in = int(k)
            self.k_last = int(k)
            return "shadow" if k < self.k_in + self.shadow_periods else "vortex"
        if self.k_in is not None:                       # V5: the segment ended at the period before; drop the state
            self.ended, self.main, self.pylons = True, None, None
        return "table"

    def loads(self, k, state, flight_mode, flaps_rad, thrust_coeff_used):
        source = self._source(k, flight_mode)
        if self.mode == "record":
            source = "table"
        self.periods[source] += 1
        if source == "table":
            return source, None, {}
        # V2: the pose at the period's start, on the integer time base
        if not isinstance(state, RigidState):
            raise TypeError("the plant state must be a RigidState")
        tick = time.perf_counter()
        pose = replace(state, time_s=k*PERIOD_S).pose(self.body)
        fields = {}
        if self.orthonormalise:
            rotation = np.asarray(pose.rotation, dtype=float)
            u, _, vt = np.linalg.svd(rotation)
            nearest = u@vt
            if np.linalg.det(nearest) <= 0.0:          # cannot happen for a RigidState (det > 0 is checked there)
                error = ValueError("the attitude is not near a proper rotation")
                error.failed_module = "aerodynamics (attitude orthonormalisation)"     # review A-R05: V6 names the module
                raise error
            fields.update(vortex_attitude_orthogonality_error=float(np.max(np.abs(rotation.T@rotation - np.eye(3)))),
                          vortex_attitude_correction=float(np.max(np.abs(nearest - rotation))))
            pose = Pose(pose.time_s, pose.position_m, nearest, pose.velocity_m_s, pose.angular_velocity_rad_s)
        force_main, moment_main, main_fields = self.main.extra_loads(pose, fields)
        fields.update(main_fields)
        force_pylons, moment_pylons, pylon_fields = self.pylons.extra_loads(pose, fields)
        fields.update(pylon_fields)
        relative = pose.velocity_m_s - self.wind                  # the origin's velocity through the air (inertial)
        drag = -PARASITIC_COEFFICIENT*0.5*self.density*REFERENCE_AREA_M2*float(np.linalg.norm(relative))*relative
        try:
            flap_force_b, flap_moment_b, condition = self.flaps.loads(pose.rotation.T@relative, self.density,
                                                                       np.asarray(flaps_rad, dtype=float), float(thrust_coeff_used))
        except (ValueError, FloatingPointError) as error:
            error.failed_module = "aerodynamics (flap increments)"
            raise
        flap_force, flap_moment = pose.rotation@flap_force_b, pose.rotation@flap_moment_b
        force = force_main + force_pylons + drag + flap_force                   # inertial, about the body origin
        moment = moment_main + moment_pylons + flap_moment
        held = held_loads(self.body, pose.rotation, force, moment)
        fields.update(vortex_force_body_N=held.force_body_N.tolist(), vortex_moment_cm_body_Nm=held.moment_cm_body_Nm.tolist(),
                      vortex_parasitic_force_body_N=(pose.rotation.T@drag).tolist(),
                      vortex_flap_force_body_N=np.asarray(flap_force_b).tolist(),
                      vortex_flap_moment_origin_body_Nm=np.asarray(flap_moment_b).tolist(),
                      vortex_flap_thrust_coeff=float(thrust_coeff_used),
                      vortex_flap_condition={key: float(value) for key, value in condition.items()},
                      vortex_step=int(k) - self.k_in, vortex_solve_s=time.perf_counter() - tick)
        return source, held, fields
