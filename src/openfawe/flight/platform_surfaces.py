"""Platform model P1 on the body (OF-0010 contract, revision 1 and its addendum 1): the lifting surfaces of one step with
the control surfaces of that step, and the groups that solve them under one fixed choice.

``PlatformP1`` holds the M600's panels with one airfoil table per panel (``aero.platform_sections``) and gives, for the
acting flap deflections of a period, a ``P1Period``: the assembly with the horizontal tail turned by the elevator, one
section table per panel at its flap's deflection (clamped at the table edge, the excess recorded), the main lifting
surfaces (wing and tails) and the pylons as two assemblies with their tables, the tail-fin overlap flag, and the strip
model's sections rebuilt from the turned assembly (on first use).

``PlatformGroup`` is one group of lifting surfaces with its own free-wake state whose geometry and section data are those
of the step (the call passes them). Every step is solved by ``prescribed.handover_step`` with the fixed choice
``HandoverChoice(nan, polar, TAKEOFF_LANDING_SOLVE, zero_seed=True)`` (addendum 1, item B3): the first step of a state is
``'start'``; every later step starts from zero circulation and, when that solve fails with one of ``FAILURES``, is
solved once more from the previous step's circulation (``'retry_previous'``); a failed retry raises the solver's error
with ``failed_module`` set to ``"aerodynamics (<name>)"``. The weight of a hand-over has no meaning on this path: the
record's ``handover_weight`` is None. Every step's record carries the solve settings (``<name>_solve_settings``), the
closure read from the solution and the seeding used.

There is no mutual induction between two groups (the pylons are their own vortex system until P2), as in
``flight.companion``.

P2 (OF-0010 revision 3 with its addenda 1-4): a ``PlatformGroup`` built with ``surfaces``
(``aero.surface_cores.SurfaceCores``) solves the main lifting surfaces and the pylons in one vortex system, with AVL's
finite core between the classes; it steps with ``fixed_choice_step`` (``prescribed.handover_step``'s three seedings,
copied so that ``surfaces`` reaches ``aero.vsm_wake``; with ``surfaces=None`` the two are the same bit for bit) and
records ``<name>_surface_cores``. ``p2_surfaces`` gives the M600's classes under the owner's option (b).

P5a (OF-0010 revision 5 with its addenda 1-2; decision 0053 addendum 7): ``PlatformGroup(seeding='mixed')`` starts each
step after a state's first from the previous step's circulation on every panel that was not stalled at the previous step
and from zero on every stalled panel, seeding ``'mixed_first'``; when that solve fails with one of ``FAILURES`` it is
solved once more from zero (``'retry_zero'``, the first failure's ``failure_info`` in ``<name>_first_seed_failure``).
This is after Piszkin & Levinsky 1976 (CASD-NSC-76-001), adapted: they restart the induced angle of a stalled section
from zero in a nonlinear lifting line; here the panel's circulation is seeded with zero in the VSM. "Stalled" (this
module's rule, revision 5 addendum 3): the previous step's closure angle (``polar_alpha_rad``, wrapped; reverse flow
counts) beyond the angles of the largest or the smallest lift coefficient of the panel's table of this step within
-30...+30 deg (``stall_angles_deg``; knots only, ties to the smaller angle, the extremes interior). With
``'mixed'`` every step after a state's first also records the counts of addendum 3 (``<name>_seed_*``) and, for a group
whose first surface is the wing, the wing's panels on the stalled branch (above 25 deg), its ``stall_switches`` and
not-attached panels; none of these fields is written under ``'zero'``.
P5a' (OF-0010 revision 6; decision 0053 addendum 9): ``seeding='strip'`` seeds the panels stalled at the previous step
with the strip-theory circulation at their geometric angle (``strip_circulation``: wind minus structural velocity at the
three-quarter-chord point, no induction) instead of zero ('strip_first'), the others with the previous circulation; one
retry from zero. The record fields of 'mixed' are written too, the seed counts as ``<name>_seed_stalled_seeded...``.
The default ``seeding='zero'`` is the behaviour above, unchanged bit for bit.
P5d (OF-0010 revision 9; decision 0053 addendum 11, the owner's option A): ``PlatformGroup(local_strip=True)`` solves a
step as above; when it fails with one of ``FAILURES`` and the error carries the residual of its last exhausted Picard
attempt (``aero.nonlinear.HybridSolveError.last_residual``), the panels with |r_i| >= 1/2 max|r| are marked and the step
is solved again by the same rule with their circulation weight multiplied by zero (``circulation_mask``: no bound
circulation on them in this step, their loads by strip theory at the local total flow; every seed of the step is zero
on them); at most ``LOCAL_STRIP_ROUNDS`` rounds, each adding to the marks of the last, fewer when a round would add no
panel; then the last round's error is raised, with ``local_strip`` attached (rounds, the panels added by each round, the
``failure_info`` of every failed solve, round 0 first). The marks hold for that step only (a state's first step
included). With the option on every step records ``<name>_local_strip_panels`` (sorted), ``<name>_local_strip_rounds``,
``<name>_local_strip_round_panels``, ``<name>_local_strip_round_residuals_m2_s`` (max|r| that started each round),
``<name>_local_strip_failures``, the marked panels' regime and closure angle in the solved step and two measures of how
much of the vortex system is left (revision 9 addendum 1): ``<name>_local_strip_weighted_panels`` (panels of weight > 0
after the marks) and ``<name>_local_strip_share`` (the marked panels' strip circulation 1/2 c CL |v x s| at the solved
flow over that plus the unmarked panels' |Gamma|, summed; None without marks). With the option off (the default)
nothing changes, bit for bit. Not with the spanwise viscosity, and not on ``prescribed.handover_step``'s path (seeding
'zero' without surfaces).

Nothing in the package imports this module; with it unused every result is unchanged.
"""
import hashlib
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..aero.aircraft import PanelPolars, WingAssembly, load_aircraft
from ..aero.handover import TAKEOFF_LANDING_SOLVE, HandoverChoice
from ..aero.hover_ring import Sections
from ..aero.methods import AeroMethod
from ..aero.platform_sections import (ELEVATOR, FLAPS, FlapTables, hinge_angles_deg, overlap_onset_deg, panel_airfoils,
                                      panel_polars, tail_axis_point, tail_fin_overlap, turn_horizontal_tail)
from ..aero.surface_cores import SurfaceCores
from ..aero.vsm import geometry as vsm_geometry
from ..aero.vsm_wake import advance_vsm_wake, start_vsm_wake
from ..aero.wing import solve_settings_metadata
from .prescribed import FAILURES, failure_info, handover_record, handover_step

MAIN_SURFACES = ("Wing", "Horizontal tail", "Vertical tail")
SEEDINGS_USED = ("start", "zero", "retry_previous")
SEEDINGS = dict(zero=SEEDINGS_USED, mixed=("start", "mixed_first", "retry_zero"),      # P5a: the seedings of each option
                strip=("start", "strip_first", "retry_zero"))                           # P5a' (revision 6)
STALL_WINDOW_DEG = (-30.0, 30.0)
STALLED_BRANCH_DEG = 25.0           # P5a criteria: a main-wing panel "on the stalled branch" above its tables' attached range
LOCAL_STRIP_ROUNDS = 3              # P5d (revision 9): the rounds of marking and re-solving before the error is raised
LOCAL_STRIP_FRACTION = 0.5          # P5d: a panel is marked when |r_i| >= LOCAL_STRIP_FRACTION max|r|


@dataclass
class P1Period:
    """The surfaces of one period (body axes)."""
    flaps_rad: np.ndarray
    delta_e_rad: float
    assembly: WingAssembly
    polar: PanelPolars
    main_wing: WingAssembly
    main_polar: PanelPolars
    pylon_wing: WingAssembly
    pylon_polar: PanelPolars
    excess_rad: np.ndarray
    tail_fin_overlap: bool

    @property
    def clamped(self):
        return self.excess_rad > 0.0

    @cached_property
    def sections(self):
        """The strip model's load points and axes from the turned assembly, with this period's tables."""
        return Sections.from_aircraft(SimpleNamespace(wing=self.assembly, polar=self.polar))

    def fields(self):
        return dict(platform_flap_excess_rad=[float(v) for v in self.excess_rad],
                    platform_flap_clamped=[bool(v) for v in self.clamped],
                    platform_tail_fin_overlap=bool(self.tail_fin_overlap),
                    platform_elevator_rad=float(self.delta_e_rad))


class PlatformP1:
    """The M600's panels with their airfoil tables and control surfaces (see the module text).

    ``case_path`` is a case whose ``polar_ids`` are the Makani airfoil names (``m600_makani_xfoil_polars_pylons.json``),
    its surfaces the main lifting surfaces first and then the pylons; ``tables_dir`` holds Makani's airfoil tables;
    ``avl_path`` Makani's AVL file."""

    def __init__(self, case_path, tables_dir, avl_path):
        aircraft = load_aircraft(case_path)
        self.case_path, self.tables_dir, self.avl_path = str(case_path), str(tables_dir), str(avl_path)
        self.case_sha256 = aircraft.source_sha256
        self.assembly = aircraft.wing
        names = self.assembly.names
        if names[:len(MAIN_SURFACES)] != MAIN_SURFACES or not all(n.startswith("Pylon") for n in names[len(MAIN_SURFACES):]):
            raise ValueError(f"the case's surfaces must be {MAIN_SURFACES} and then the pylons, not {names}")
        self.main_count = len(MAIN_SURFACES)
        self.main_panels = sum(w.panels for w in self.assembly.wings[:self.main_count])
        self.airfoils = panel_airfoils(case_path, self.assembly, avl_path)
        self.tables = FlapTables(tables_dir)
        self.axis_point = tail_axis_point(self.assembly)
        self.onset_deg = overlap_onset_deg(self.assembly, self.axis_point)
        self.hinge_deg = hinge_angles_deg(self.assembly, self.axis_point)
        self.avl_sha256 = hashlib.sha256(Path(avl_path).read_bytes()).hexdigest()
        self.pylon_wing = WingAssembly(self.assembly.wings[self.main_count:], names[self.main_count:])

    def settings(self):
        tables = {name: dict(stall_deg=[float(np.degrees(v)) for v in table.stall_rad], flap_rows_rad=table.flap_rad.tolist(),
                             source_sha256=table.source_sha256)
                  for name, table in self.tables.blended.items()}      # as built (review P1C-10)
        return dict(model="OF-0010 P1: one full-circle section table per panel at its flap's deflection (Makani's XFOIL tables "
                          "with the hover model's post-stall blend, every flap row), the elevator as the whole horizontal tail "
                          "turned, no database flap increments",
                    case=self.case_path, case_sha256=self.case_sha256, tables_dir=self.tables_dir, avl=self.avl_path,
                    avl_sha256=self.avl_sha256, high_aoa_sha256=self.tables.high_aoa_sha256,
                    tables=tables, flaps=list(FLAPS), panel_flaps=self.airfoils.flaps.tolist(),
                    panel_airfoils=list(self.airfoils.names),
                    out_of_table="clamped at the table edge; excess max(0, d - dmax, dmin - d) per flap recorded",
                    elevator=dict(axis_point_m=self.axis_point.tolist(), axis="body +y", positive="trailing edge down",
                                  hinge_line_angle_deg=self.hinge_deg, tail_fin_overlap_onset_deg=self.onset_deg),
                    solve=solve_settings_metadata(TAKEOFF_LANDING_SOLVE), solve_name="TAKEOFF_LANDING_SOLVE",
                    seeding="'start' on a state's first step; then zero, one retry from the previous circulation on FAILURES",
                    handover_weight="None (no hand-over on this path)")

    def period(self, flaps_rad):
        flaps = np.asarray(flaps_rad, dtype=float).reshape(8)
        if not np.all(np.isfinite(flaps)):
            raise ValueError("non-finite flap deflection")
        delta_e = float(flaps[ELEVATOR])
        assembly = turn_horizontal_tail(self.assembly, delta_e, self.axis_point)
        polar, excess = panel_polars(self.tables, self.airfoils, flaps)
        n = self.main_panels
        main = WingAssembly(assembly.wings[:self.main_count], assembly.names[:self.main_count])
        overlap = tail_fin_overlap(assembly, self.assembly) if delta_e != 0.0 else False
        return P1Period(flaps.copy(), delta_e, assembly, polar, main, PanelPolars(polar.polars[:n]), self.pylon_wing,
                        PanelPolars(polar.polars[n:]), excess, overlap)


class PlatformGroup:
    """One group of lifting surfaces with its own wake, solved under the fixed choice of the module text; ``extra_loads``
    takes the step's geometry (body axes) and section data."""

    MAIN_WING_ONLY = ("stall_switches", "main_panels_not_attached")

    def __init__(self, name, backend, wind, *, solve=TAKEOFF_LANDING_SOLVE, surfaces=None, seeding="zero", viscosity=False,
                 local_strip=False, diagnostics=False):
        if not isinstance(diagnostics, bool):
            raise ValueError("diagnostics must be True or False")
        if surfaces is not None and not isinstance(surfaces, SurfaceCores):
            raise ValueError("surfaces must be SurfaceCores")
        if seeding not in SEEDINGS:
            raise ValueError(f"seeding must be one of {sorted(SEEDINGS)}")
        if viscosity not in (False, True):
            raise ValueError("viscosity must be True or False")
        if viscosity and seeding == "zero" and surfaces is None:
            raise ValueError("the spanwise viscosity with zero seeding needs P2's surfaces (prescribed.handover_step is not edited)")
        if not isinstance(local_strip, bool):
            raise ValueError("local_strip must be True or False")
        if local_strip and viscosity:
            raise ValueError("the local strip fallback is not combined with the spanwise viscosity")
        if local_strip and seeding == "zero" and surfaces is None:
            raise ValueError("the local strip fallback with zero seeding needs P2's surfaces (prescribed.handover_step is not edited)")
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("the group needs a name of letters, digits and underscores (it prefixes its record fields)")
        if backend.method != AeroMethod.VSM_FREE_WAKE:
            raise ValueError("the platform group needs the VSM free-wake method")
        self.name, self.backend, self.wind, self.solve = name, backend, wind, solve
        self.surfaces = surfaces
        self.seeding = seeding
        self.state = None
        self.solves = 0
        self.last_time_s = None
        self.seedings = {key: 0 for key in SEEDINGS[seeding]}
        self.previous_polar = None                     # P5a: the tables of the previous step (reports only)
        self.use_viscosity = bool(viscosity)           # P5c (revision 7 addendum 1): built from the first wing passed
        self.viscosity = None
        self.local_strip = bool(local_strip)           # P5d (revision 9)
        self.diagnostics = diagnostics                 # OF-0010 revision 22 addendum 8: two extra record fields (opt-in)

    def extra_loads(self, pose, record, wing, polar):
        """Solve the group at the pose of this step with ``wing`` (body axes) and ``polar``; force and moment about the
        body origin (inertial), and fields."""
        if self.last_time_s is not None and not pose.time_s > self.last_time_s:
            raise RuntimeError(f"the platform group '{self.name}' was called twice for one step, or out of order")
        placed = wing.transformed(pose.rotation, pose.position_m)

        def velocity(points, time_s):
            return pose.point_velocity(points)
        choice = HandoverChoice(float("nan"), polar, self.solve, True)
        built = False
        if self.use_viscosity and self.viscosity is None:
            from ..aero.spanwise_viscosity import SpanwiseViscosity
            self.viscosity, built = SpanwiseViscosity(wing), True
        if self.viscosity is not None:                 # review P5cR-05: count this step's calls and gate switches
            self.viscosity.reset_tally()
        reg = {} if self.viscosity is None else dict(regularization=self.viscosity)
        seed_report = None
        if self.seeding in ("mixed", "strip") and self.state is not None:   # P5a reports (revision 5 addendum 3), before the step
            previous = self.state.solution
            stalled = stalled_panels(previous.polar_alpha_rad, polar)
            regime = getattr(previous, "panel_regime", None)
            attached = None if regime is None else np.asarray(regime) == 0
            changed = (np.flatnonzero(stalled != stalled_panels(previous.polar_alpha_rad, self.previous_polar)).tolist()
                       if self.previous_polar is not None and self.previous_polar is not polar else [])
            given = "zero" if self.seeding == "mixed" else "stalled"   # 'strip': the stalled panels get the strip seed
            seed_report = {f"{given}_seeded": int(stalled.sum()),
                           f"{given}_seeded_regime_attached": None if attached is None else int(np.sum(stalled & attached)),
                           "previous_seeded_regime_not_attached": None if attached is None else int(np.sum(~stalled & ~attached)),
                           "stalled_table_change": [int(k) for k in changed]}
        marked = np.zeros(placed.panels, dtype=bool)  # P5d: this step's marks (none unless the option is on)
        rounds, round_residuals, round_panels, failures = 0, [], [], []
        while True:
            options = dict(reg)
            if rounds:
                options["circulation_mask"] = np.where(marked, 0.0, 1.0)
            try:
                if self.seeding == "mixed":            # P5a: Piszkin & Levinsky's start
                    solved, seeding, failure = mixed_choice_step(self.state, choice, self.backend.wake, placed, self.wind,
                                                                 velocity, pose.time_s, self.surfaces, **options)
                elif self.seeding == "strip":          # P5a' (revision 6): the strip-theory seed on stalled panels
                    solved, seeding, failure = strip_choice_step(self.state, choice, self.backend.wake, placed, self.wind,
                                                                 velocity, pose.time_s, self.surfaces, **options)
                elif self.surfaces is None:
                    solved, seeding, failure = handover_step(self.state, choice, self.backend.wake, placed, self.wind,
                                                             velocity, pose.time_s)
                else:
                    solved, seeding, failure = fixed_choice_step(self.state, choice, self.backend.wake, placed, self.wind,
                                                                 velocity, pose.time_s, self.surfaces, **options)
                break
            except FAILURES as error:
                residual = getattr(error, "last_residual", None) if self.local_strip else None
                size = None if residual is None else np.abs(np.asarray(residual, dtype=float))
                new = None
                if (size is not None and rounds < LOCAL_STRIP_ROUNDS and size.shape == (placed.panels,)
                        and np.isfinite(size).all() and size.max() > 0.0):
                    new = (size >= LOCAL_STRIP_FRACTION*size.max()) & ~marked
                if self.local_strip:                   # review P5dR-04: every failed solve of the step, round 0 first
                    failures.append(dict(failure_info(error), first_seed_failure=getattr(error, "first_seed_failure", None),
                                         residual_attached=residual is not None,
                                         residual_relaxation=getattr(error, "last_residual_relaxation", None)))
                if new is None or not new.any():       # review P5dR-02: no residual, the last round, or nothing new
                    error.failed_module = f"aerodynamics ({self.name})"
                    if self.viscosity is not None:
                        error.viscosity_tally = dict(self.viscosity.tally)
                    if self.local_strip:               # P5d: what was tried before the error is raised
                        error.local_strip = dict(rounds=rounds, panels=[int(k) for k in np.flatnonzero(marked)],
                                                 round_panels=list(round_panels), round_residuals_m2_s=list(round_residuals),
                                                 failures=list(failures))
                    raise
                marked |= new
                round_panels.append([int(k) for k in np.flatnonzero(new)])
                round_residuals.append(float(size.max()))
                rounds += 1
        self.state, self.solves, self.last_time_s = solved, self.solves+1, float(pose.time_s)
        self.seedings[seeding] += 1
        force = np.array(solved.solution.force, dtype=float)
        moment = np.array(solved.solution.moment(placed, reference=pose.position_m), dtype=float)
        prefix = self.name
        fields = {f"{prefix}_force_inertial_N": force.tolist(), f"{prefix}_moment_origin_inertial_Nm": moment.tolist(),
                  f"{prefix}_force_body_N": (pose.rotation.T@force).tolist(),
                  f"{prefix}_moment_origin_body_Nm": (pose.rotation.T@moment).tolist(),
                  f"{prefix}_iterations": int(solved.solution.iterations)}
        if self.diagnostics:                            # revision 22 addendum 8: the induced velocity at the load points
            induced = np.asarray(solved.solution.induced_velocity, dtype=float)   # (all of it: own wake, rings, bound)
            fields[f"{prefix}_induced_max_m_s"] = float(np.linalg.norm(induced, axis=1).max())
            older = np.asarray(solved.older_circulation)
            fields[f"{prefix}_shed_max_m2_s"] = (None if not len(older) else     # the newest row's shed strength
                                                 float(np.abs(np.asarray(solved.solution.circulation)-older[0]).max()))
        handed = handover_record(solved, placed, choice, seeding, failure)
        handed["handover_weight"] = None
        fields.update({f"{prefix}_{key}": value for key, value in handed.items() if key not in self.MAIN_WING_ONLY})
        fields[f"{prefix}_solve_settings"] = solve_settings_metadata(self.solve)
        if self.surfaces is not None:
            fields[f"{prefix}_surface_cores"] = True
        if self.seeding in ("mixed", "strip"):         # P5a (addendum 2 part B): the label of the failure, the wing's branch
            fields[f"{prefix}_first_seed_failure"] = fields.pop(f"{prefix}_zero_seed_failure")
            fields[f"{prefix}_seeding_option"] = ("mixed (after Piszkin & Levinsky, adapted)" if self.seeding == "mixed" else
                                                  "strip (after Piszkin & Levinsky, adapted: the stalled section restarts "
                                                  "from its geometric angle)")
            for key, value in (seed_report or {}).items():
                fields[f"{prefix}_seed_{key}"] = value
            self.previous_polar = polar
            if placed.names[0] == "Wing":
                alpha = np.degrees(np.asarray(solved.solution.polar_alpha_rad)[:placed.wings[0].panels])
                fields[f"{prefix}_wing_panels_stalled_branch"] = [int(k) for k in np.flatnonzero(alpha > STALLED_BRANCH_DEG)]
                fields[f"{prefix}_wing_stall_switches"] = handed["stall_switches"]
                fields[f"{prefix}_wing_panels_not_attached"] = handed["main_panels_not_attached"]
        if self.local_strip:                           # P5d: the marks of this step
            fields.update(local_strip_fields(prefix, solved.solution, placed, self.backend.wake, marked, rounds,
                                             round_panels, round_residuals, failures))
        if self.viscosity is not None:                 # P5c: the regularisation at the converged iterate
            from ..aero.spanwise_viscosity import sawtooth
            last = self.viscosity.last
            fields[f"{prefix}_viscosity_on"] = last["on"]
            fields[f"{prefix}_viscosity_active"] = last["active"]
            fields[f"{prefix}_viscosity_mu_max"] = last["mu_max"]
            fields[f"{prefix}_viscosity_deviation_percent"] = last["deviation_percent"]
            fields[f"{prefix}_viscosity_tally"] = dict(self.viscosity.tally)
            if built:                                  # review P5cR-06: the surfaces and their ends, once
                fields[f"{prefix}_viscosity_settings"] = self.viscosity.settings()
            if placed.names[0] == "Wing":
                fields[f"{prefix}_wing_sawtooth"] = sawtooth(solved.solution.circulation, placed.wings[0].panels)
        repeated = set(fields) & set(record)
        if repeated:
            raise RuntimeError(f"the platform group '{self.name}' would overwrite record fields: {sorted(repeated)}")
        return force, moment, fields


def local_strip_fields(prefix, solution, placed, config, marked, rounds, round_panels, round_residuals, failures):
    """P5d (revision 9 with its addendum 1): the record fields of one solved step under ``local_strip``."""
    k = np.flatnonzero(marked)
    regime = getattr(solution, "panel_regime", None)
    weight = getattr(solution, "circulation_weight", None)
    share = None
    if k.size:
        chord = vsm_geometry(placed)[1]
        speed = np.asarray(solution.reynolds, dtype=float)*config.kinematic_viscosity_m2_s/chord   # |v x s| at 3/4 chord
        strip = np.abs(0.5*chord*np.asarray(solution.cl, dtype=float)*speed)[marked]
        rest = np.abs(np.asarray(solution.circulation, dtype=float))[~marked]
        share = float(strip.sum()/(strip.sum()+rest.sum())) if strip.sum()+rest.sum() > 0.0 else None
    return {f"{prefix}_local_strip_panels": [int(j) for j in k], f"{prefix}_local_strip_rounds": int(rounds),
            f"{prefix}_local_strip_round_panels": list(round_panels),
            f"{prefix}_local_strip_round_residuals_m2_s": list(round_residuals),
            f"{prefix}_local_strip_failures": list(failures),
            f"{prefix}_local_strip_panel_regime": [] if regime is None else [int(regime[j]) for j in k],
            f"{prefix}_local_strip_panel_alpha_deg": [float(np.degrees(solution.polar_alpha_rad[j])) for j in k],
            f"{prefix}_local_strip_weighted_panels": None if weight is None else int(np.sum(np.asarray(weight) > 0.0)),
            f"{prefix}_local_strip_share": share}


def _masked_seed(seed, state, circulation_mask):
    """P5d (review P5dR-06 v): zero on the masked panels, whose target is zero; ``seed`` itself without a mask (None: the
    previous circulation)."""
    if circulation_mask is None:
        return seed
    base = np.asarray(state.solution.circulation if seed is None else seed, dtype=float)
    return np.where(np.asarray(circulation_mask) == 0.0, 0.0, base)


def _solver_options(regularization, circulation_mask):
    """The optional solver arguments of P5c and P5d, passed only when given (None: the calls exactly as before)."""
    extra = {} if regularization is None else dict(regularization=regularization)
    if circulation_mask is not None:
        extra["circulation_mask"] = circulation_mask
    return extra


def fixed_choice_step(state, choice, base_config, placed, wind, structural_velocity, time_s, surfaces, regularization=None,
                      circulation_mask=None):
    """``prescribed.handover_step`` (that file is not edited) with ``surfaces`` passed to ``aero.vsm_wake``: 'start' on
    the first step, then 'previous' or zero with one retry from the previous circulation ('zero', 'retry_previous'),
    the state re-bound to the step's solve settings - the same statements in the same order."""
    config = replace(base_config, solve=choice.solve)
    if state is None:
        return start_vsm_wake(placed, choice.polar, config, wind, time_s=time_s,
                              structural_velocity=structural_velocity, surfaces=surfaces,
                              **_solver_options(regularization, circulation_mask)), "start", None
    if state.config != config:
        if replace(state.config, solve=config.solve) != config:
            raise RuntimeError("hand-over re-binding would change more than the solve settings")
        state = replace(state, config=config)

    def advance(seed):
        return advance_vsm_wake(state, choice.polar, config, wind, next_wing=placed,
                                structural_velocity=structural_velocity, next_time_s=time_s,
                                initial_circulation=_masked_seed(seed, state, circulation_mask),
                                surfaces=surfaces, **_solver_options(regularization, circulation_mask))
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


def stall_angles_deg(polar):
    """(angle of the smallest, angle of the largest lift coefficient) of a section table within STALL_WINDOW_DEG, deg.

    The extremes are taken over the table's knots inside the window (the table is piecewise linear between them); a tie
    gives the smallest such angle. Either extreme must be interior: strictly beyond the table's (interpolated) value at
    both window edges, else ValueError (an extreme at the edge is an artefact of the window, not a stall angle)."""
    alpha = np.asarray(polar.alpha_deg, dtype=float)
    cl_all = np.asarray(polar.cl, dtype=float)
    inside = (alpha >= STALL_WINDOW_DEG[0]) & (alpha <= STALL_WINDOW_DEG[1])
    if not np.any(inside):
        raise ValueError("a section table without knots in the stall window")
    a, cl = alpha[inside], cl_all[inside]
    edges = np.interp(STALL_WINDOW_DEG, alpha, cl_all)
    low, high = int(np.argmin(cl)), int(np.argmax(cl))
    if not (cl[high] > edges.max() and cl[low] < edges.min()):
        raise ValueError("the largest or the smallest lift coefficient lies at the edge of the stall window")
    return float(a[low]), float(a[high])


def stalled_panels(alpha_rad, panel_polars):
    """P5a: panels whose section angle lies beyond the angles of the smallest or the largest lift coefficient of their
    table (``stall_angles_deg``); the angle is wrapped to [-180, 180) deg as a periodic table looks it up."""
    alpha = np.asarray(alpha_rad, dtype=float)
    if alpha.shape != (len(panel_polars.polars),):
        raise ValueError("one angle per panel table")
    wrapped = np.degrees(np.mod(alpha+np.pi, 2*np.pi)-np.pi)
    cache, out = {}, np.zeros(len(alpha), dtype=bool)
    for j, polar in enumerate(panel_polars.polars):
        key = id(polar)
        if key not in cache:
            cache[key] = stall_angles_deg(polar)
        low, high = cache[key]
        out[j] = wrapped[j] > high or wrapped[j] < low
    return out


def mixed_choice_step(state, choice, base_config, placed, wind, structural_velocity, time_s, surfaces, regularization=None,
                      circulation_mask=None):
    """P5a (module text): 'start' on a state's first step; then the previous circulation on the panels not stalled at the
    previous step and zero on the stalled ones ('mixed_first'); on FAILURES once more from zero ('retry_zero'). The
    re-binding of the state to the step's solve settings is that of ``fixed_choice_step``."""
    config = replace(base_config, solve=choice.solve)
    extra = {} if surfaces is None else dict(surfaces=surfaces)
    extra.update(_solver_options(regularization, circulation_mask))   # OF-0010 P5c, P5d
    if state is None:
        return start_vsm_wake(placed, choice.polar, config, wind, time_s=time_s,
                              structural_velocity=structural_velocity, **extra), "start", None
    if state.config != config:
        if replace(state.config, solve=config.solve) != config:
            raise RuntimeError("hand-over re-binding would change more than the solve settings")
        state = replace(state, config=config)
    stalled = stalled_panels(state.solution.polar_alpha_rad, choice.polar)
    seed = np.where(stalled, 0.0, np.asarray(state.solution.circulation, dtype=float))

    def advance(start):
        return advance_vsm_wake(state, choice.polar, config, wind, next_wing=placed,
                                structural_velocity=structural_velocity, next_time_s=time_s,
                                initial_circulation=_masked_seed(start, state, circulation_mask), **extra)
    try:
        return advance(seed), "mixed_first", None
    except FAILURES as first:
        info = failure_info(first)
    try:
        return advance(np.zeros(placed.panels)), "retry_zero", info
    except FAILURES as second:
        second.first_seed_failure = info
        raise


def strip_circulation(placed, polar, wind, structural_velocity, time_s):
    """P5a' (revision 6): per panel 1/2 c CL(alpha_g) |v_g x s|, v_g = wind - structural velocity at the three-quarter-chord
    point (``aero.vsm.geometry``), no induction; alpha_g = atan2(v_g.n, v_g.t); CL from the panel's table."""
    cp, chord, normal, tangent, span = vsm_geometry(placed)
    if callable(wind):                                 # OF-0010 revision 19 addendum 6 (B1): a wind field
        from ..aero.wake import ambient_velocity
        v = ambient_velocity(wind, cp, time_s)-np.asarray(structural_velocity(cp, time_s), dtype=float)
    else:
        v = np.asarray(wind, dtype=float)[None, :]-np.asarray(structural_velocity(cp, time_s), dtype=float)
    alpha = np.arctan2(np.sum(v*normal, axis=1), np.sum(v*tangent, axis=1))
    cl = polar.coefficients(alpha)[0]
    return 0.5*chord*cl*np.linalg.norm(np.cross(v, span), axis=1)


def strip_choice_step(state, choice, base_config, placed, wind, structural_velocity, time_s, surfaces, regularization=None,
                      circulation_mask=None):
    """P5a' (revision 6): as ``mixed_choice_step`` with the strip-theory circulation (``strip_circulation``) instead of zero
    on the panels stalled at the previous step ('strip_first'); on FAILURES once more from zero ('retry_zero')."""
    config = replace(base_config, solve=choice.solve)
    extra = {} if surfaces is None else dict(surfaces=surfaces)
    extra.update(_solver_options(regularization, circulation_mask))   # OF-0010 P5c, P5d
    if state is None:
        return start_vsm_wake(placed, choice.polar, config, wind, time_s=time_s,
                              structural_velocity=structural_velocity, **extra), "start", None
    if state.config != config:
        if replace(state.config, solve=config.solve) != config:
            raise RuntimeError("hand-over re-binding would change more than the solve settings")
        state = replace(state, config=config)
    stalled = stalled_panels(state.solution.polar_alpha_rad, choice.polar)
    seed = np.where(stalled, strip_circulation(placed, choice.polar, wind, structural_velocity, time_s),
                    np.asarray(state.solution.circulation, dtype=float))

    def advance(start):
        return advance_vsm_wake(state, choice.polar, config, wind, next_wing=placed,
                                structural_velocity=structural_velocity, next_time_s=time_s,
                                initial_circulation=_masked_seed(start, state, circulation_mask), **extra)
    try:
        return advance(seed), "strip_first", None
    except FAILURES as first:
        info = failure_info(first)
    try:
        return advance(np.zeros(placed.panels)), "retry_zero", info
    except FAILURES as second:
        second.first_seed_failure = info
        raise


P2_CLASSES = (0, 0, 0, 1, 2, 3, 4)       # wing, horizontal tail, vertical tail one class; each pylon its own (option (b))


def p2_surfaces(p1):
    """The M600's surface classes and AVL strip radii for P2, from P1's body-axis geometry (not turned)."""
    return SurfaceCores.from_assembly(p1.assembly, P2_CLASSES)


def default_p1(repo):
    """The P1 model of the M600 from the repository's files."""
    repo = Path(repo)
    return PlatformP1(repo/"cases/qualification/m600_makani_xfoil_polars_pylons.json", repo/"cases/polars/makani",
                      repo/"cases/polars/makani/m600_low_tail_no_winglets.avl")
