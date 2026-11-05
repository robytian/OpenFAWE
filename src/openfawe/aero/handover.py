"""Hand-over between crosswind and hover section data (decision 0025 with amendments 1 and 2).

The rule is the Makani simulator's own (source S8 of the contract of OF-0007-wingfull20261001; the
G4 contract uses the label S8 for something else): sim/physics/aero.cc,
``Aero::CalcForceMomentCoeff`` (snapshot 1d66923, lines 216-237), with the stall angles of
config/m600/sim/aero_sim.py lines 262-270. The simulator mixes its low-incidence and its
high-incidence whole-aircraft databases with

    c_a = 0.5 + max(sat((alpha - high_a)/width, -0.5, 0.5), sat((low_a - alpha)/width, -0.5, 0.5))
    c_b = 0.5 + max(sat((beta  - high_b)/width, -0.5, 0.5), sat((low_b - beta )/width, -0.5, 0.5))
    w   = 1 - (1 - c_a)(1 - c_b)

evaluated from the incidence and sideslip of the body. Here the same weight is applied to the
*section* data instead: every panel uses ``(1 - w) crosswind + w hover`` coefficients in one
circulation solve with one wake (decision 0025 amendment 1). With ``w = 0`` the model is the
production crosswind model (source closure, crosswind section data); with ``w > 0`` it is the
full-incidence closure with the prescribed take-off/landing setting of decision 0025 item 2.

Seeding of the nonlinear solve when ``w > 0`` is a declared choice, not settled by the owner
(decision 0025 amendment 2):

``'previous'``  the previous step's circulation (continuation; contract of the full-incidence
                task, section 3 item 2). Past stall onset this follows a branch on which
                neighbouring panels alternate between attached and post-stall states.
``'zero'``      zero circulation first; if that solve fails, one retry from the previous
                circulation, recorded; a failure only if both fail (amendment 1).

Weight basis (``SectionHandover.weight_basis``). ``'origin'`` (default) is the rule above: one weight
for all panels, from the relative wind at the body origin. ``'panel'`` is option (c) of the owner item
(乙) of the G4 contract, an opt-in trial (decision 0029): every panel gets its own weight from the same
formula, evaluated with the incidence and sideslip, in body axes, of the flow that reaches that panel
from outside the lifting surfaces' own circulation system: the ambient wind minus the panel's own
velocity, plus the induction of the rotor rings (``panel_weights``). The lifting surfaces' own wake is
left out on purpose: it belongs to the solution, not to the oncoming stream the simulator's rule looks
at. The model evaluates the flow on a panel at two points, the 3/4-chord control point of the closure
and the quarter-chord load point; the panel's weight is the larger of the weights of the two, so that
a step is on the production crosswind model only when the oncoming flow is inside the crosswind range
at every point where that model reads it. In a uniform stream without body rotation and without rings
every panel weight equals the origin weight and the two bases are the same model, bit for bit. The
step uses the production crosswind model when all panel weights are zero, else the full-incidence
closure with ``(1 - w_i) crosswind + w_i hover`` per panel; the weights are fixed before the
circulation solve, so they are constants of it (like the onset weights of the full-incidence closure).
On a rotating body the two bases differ also without rotors, because a panel's own velocity is not the
origin's.

Nothing in this module is used unless it is called; no default path of the solvers changes.
Known limits recorded with the decisions: the solution in the stall-onset ranges is not unique
under either seeding; W4 (attached-flow lift against the ASWING table) is not met at 5 deg.
"""

from dataclasses import dataclass, field

import numpy as np

from .wing import SolveSettings, solve_settings_metadata

#: Stall angles and blend width of the M600 simulator configuration (aero_sim.py, 1d66923), degrees.
MAKANI_M600_STALL = dict(low_alpha_deg=-20.0, high_alpha_deg=10.0, low_beta_deg=-20.0, high_beta_deg=20.0,
                         blending_deg=10.0)

#: Production crosswind closure as used by the convergence runs.
CROSSWIND_SOLVE = SolveSettings(solver="newton", max_iterations=60)

#: Prescribed take-off/landing setting (decision 0025 item 2).
TAKEOFF_LANDING_SOLVE = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence",
                                      span_flow_fraction=0.05, reverse_flow="strip", reverse_fade=0.2)

SEEDINGS = ("previous", "zero")


def makani_weight(alpha_deg, beta_deg, stall=MAKANI_M600_STALL):
    """High-incidence weight of the simulator rule; angles in degrees, result in [0, 1]."""
    width = stall["blending_deg"]

    def one(angle, low, high):
        return 0.5+max(np.clip((angle-high)/width, -0.5, 0.5), np.clip((low-angle)/width, -0.5, 0.5))
    c_alpha = one(float(alpha_deg), stall["low_alpha_deg"], stall["high_alpha_deg"])
    c_beta = one(float(beta_deg), stall["low_beta_deg"], stall["high_beta_deg"])
    return float(1.0-(1.0-c_alpha)*(1.0-c_beta))


def panel_weights(relative_body, stall=MAKANI_M600_STALL, fallback=0.0):
    """Hand-over weight of every panel from its own oncoming flow in body axes (``weight_basis='panel'``).

    ``relative_body`` (n, 3): velocity of the air relative to each panel in the case-file body axes
    (x chordwise towards the trailing edge, y spanwise, z up), i.e. what ``relative_wind_body`` gives for
    the body origin, taken at the panel instead. Incidence ``atan2(w, u)`` and sideslip ``asin(v / V)`` as
    there; a panel with no oncoming flow at all gets ``fallback`` (the weight of the body origin).
    """
    relative = np.asarray(relative_body, dtype=float).reshape(-1, 3)
    if not np.isfinite(relative).all():
        raise ValueError("panel onset flow must be finite")
    weights = np.empty(len(relative))
    for k, row in enumerate(relative):
        # The very expressions of ``flight.kinematics.relative_wind_body`` (speed, incidence, sideslip), so that a panel
        # whose flow is the origin's gets the origin's weight to the last bit, with sideslip as without.
        speed = float(np.linalg.norm(row))
        if speed == 0.0:
            weights[k] = float(fallback)
            continue
        weights[k] = makani_weight(float(np.degrees(np.arctan2(row[2], row[0]))),
                                   float(np.degrees(np.arcsin(np.clip(row[1]/speed, -1.0, 1.0)))), stall)
    return weights


OUT_OF_RANGE = ("error", "hover")
WEIGHT_BASES = ("origin", "panel")


def _outside_low_table(low, alpha):
    """Mask of the angles outside the declared range of the low (crosswind) table(s), and the angles clamped into it.

    ``low`` is a ``PanelPolars`` (one table per panel) or a single ``SectionPolar``; a periodic table has no
    outside. The clamped copy only keeps the table evaluation legal where the result is discarded. The
    range test is the one ``SectionPolar.coefficients`` applies (and the one ``PanelPolars.clipped`` makes,
    which ignores periodic tables and is therefore not used here). A table with ``bounds='clamp'`` would not
    raise under 'error' either; all M600 crosswind tables declare 'error'.
    """
    alpha = np.array(alpha, dtype=float, copy=True).reshape(-1)
    tables = getattr(low, "polars", None)
    if tables is None:
        tables = [low]*len(alpha)
    outside = np.zeros(len(alpha), dtype=bool)
    for k, table in enumerate(tables):
        if getattr(table, "periodic", False):
            continue
        lower, upper = np.deg2rad(table.alpha_deg[0]), np.deg2rad(table.alpha_deg[-1])
        if alpha[k] < lower or alpha[k] > upper:
            outside[k] = True
            alpha[k] = min(max(alpha[k], lower), upper)
    return outside, alpha


class BlendedPolars:
    """Section data ``(1 - w) low + w high`` for every panel; ``w`` of 0 or 1 never evaluates the other set.

    ``out_of_range`` says what a panel gets when its incidence lies outside the declared range of the
    crosswind (low) table while the blend is active (``0 < w < 1``): ``'error'`` (default) lets the
    table raise as before (the failure of case A4 at 10 m/s, G4 contract section 8, 2026-10-04 04:20);
    ``'hover'`` gives that panel the hover (high) data alone, i.e. weight 1 for that panel only — option
    (a) of the owner item (乙) of the G4 README, an opt-in trial, not a decision. The panels so treated
    at the last evaluation are in ``last_out_of_range`` and the largest count seen in
    ``most_out_of_range`` (diagnostics for the receipts; the solver evaluates the tables many times per
    step, so the counts refer to evaluations, not steps; the last evaluation of a step is the closure at
    the converged circulation).

    The option makes the section data discontinuous at the table edge: a panel crossing the edge has its
    weight jump from ``w`` to 1, i.e. its coefficients jump by ``(1 - w)`` times the difference between
    the two tables there (for the M600 crosswind tables, ideal linear tables declared to +-80 deg, that
    difference is of order 10 in CL and 2 in CD). A solve that fails near the edge may therefore fail
    because of this jump rather than because of the table range; the receipts' diagnostics are what tell
    the two apart, and a trial with this option is not evidence about option (a) without them.
    """

    def __init__(self, low, high, weight, out_of_range="error"):
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("blend weight must lie in [0, 1]")
        if out_of_range not in OUT_OF_RANGE:
            raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE}")
        self.low, self.high, self.weight, self.out_of_range = low, high, weight, out_of_range
        self.last_out_of_range = np.zeros(0, dtype=np.int64)
        self.most_out_of_range = 0

    def _mix(self, method, alpha):
        w = self.weight
        if w == 0.0:
            return getattr(self.low, method)(alpha)
        if w == 1.0:
            return getattr(self.high, method)(alpha)
        if self.out_of_range == "hover":
            outside, clamped = _outside_low_table(self.low, alpha)
            self.last_out_of_range = np.flatnonzero(outside)
            self.most_out_of_range = max(self.most_out_of_range, int(outside.sum()))
            if outside.any():
                # The low table is evaluated at a clamped copy of the angles only to keep its shape; those
                # entries are replaced by the hover values below and never enter the result.
                a, b = getattr(self.low, method)(clamped), getattr(self.high, method)(alpha)
                weights = np.where(outside, 1.0, w)
                if isinstance(a, tuple):
                    return tuple((1.0-weights)*x+weights*y for x, y in zip(a, b))
                return (1.0-weights)*a+weights*b
        a, b = getattr(self.low, method)(alpha), getattr(self.high, method)(alpha)
        if isinstance(a, tuple):
            return tuple((1.0-w)*x+w*y for x, y in zip(a, b))
        return (1.0-w)*a+w*b

    def coefficients(self, alpha):
        return self._mix("coefficients", alpha)

    def cl_slope(self, alpha):
        return self._mix("cl_slope", alpha)

    def attached(self, alpha):
        """Attached-range label used only for the panel regimes; taken from the hover set once it takes part."""
        return self.low.attached(alpha) if self.weight == 0.0 else self.high.attached(alpha)


class PanelBlendedPolars:
    """Section data ``(1 - w_i) low + w_i high`` with one constant weight per panel (``weight_basis='panel'``).

    A panel of weight 1 gets the hover (high) values alone and a panel of weight 0 the crosswind (low)
    values alone, exactly (no arithmetic on them). Both tables are evaluated for all panels in one call;
    what a panel does not use is discarded, and the crosswind table is asked at an angle clamped into its
    range for a panel of weight 1, so that such a panel cannot make it raise. For a panel that does use
    the crosswind table (weight below 1) and whose incidence lies outside its declared range,
    ``out_of_range`` has the meaning it has in ``BlendedPolars``: ``'error'`` lets the table raise,
    ``'hover'`` gives that panel the hover data alone; ``last_out_of_range`` and ``most_out_of_range`` are
    the same diagnostics. (A panel of weight 0 is treated like a blend panel here: under ``'error'`` it
    raises outside the table as the production model would, under ``'hover'`` it gets the hover data.)
    """

    def __init__(self, low, high, weights, out_of_range="error"):
        weights = np.array(weights, dtype=float, copy=True).reshape(-1)
        if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("panel blend weights must lie in [0, 1]")
        if out_of_range not in OUT_OF_RANGE:
            raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE}")
        weights.setflags(write=False)
        self.low, self.high, self.weights, self.out_of_range = low, high, weights, out_of_range
        self.weight = float(weights.max()) if len(weights) else 0.0       # largest panel weight (for records that ask for one number)
        self.last_out_of_range = np.zeros(0, dtype=np.int64)
        self.most_out_of_range = 0

    def _low(self, method, alpha):
        """The low table where it is consulted; angles of panels that do not consult it are clamped (results discarded)."""
        alpha = np.asarray(alpha, dtype=float)
        if len(alpha) != len(self.weights):
            raise ValueError("panel weight count and angles differ")
        unused = self.weights == 1.0
        outside, clamped = _outside_low_table(self.low, alpha)
        blend_outside = outside & ~unused
        self.last_out_of_range = np.flatnonzero(blend_outside)
        self.most_out_of_range = max(self.most_out_of_range, int(blend_outside.sum()))
        if blend_outside.any() and self.out_of_range != "hover":
            return getattr(self.low, method)(alpha), self.weights     # the table raises for the offending panel, as declared
        use = np.where(unused | blend_outside, clamped, alpha)
        weights = np.where(blend_outside, 1.0, self.weights)
        return getattr(self.low, method)(use), weights

    def _mix(self, method, alpha):
        a, weights = self._low(method, alpha)
        b = getattr(self.high, method)(alpha)
        if isinstance(a, tuple):
            return tuple(np.where(weights == 0.0, x, np.where(weights == 1.0, y, (1.0-weights)*x+weights*y)) for x, y in zip(a, b))
        return np.where(weights == 0.0, a, np.where(weights == 1.0, b, (1.0-weights)*a+weights*b))

    def coefficients(self, alpha):
        return self._mix("coefficients", alpha)

    def cl_slope(self, alpha):
        return self._mix("cl_slope", alpha)

    def attached(self, alpha):
        """Attached-range label per panel: from the crosswind set where that set alone is used, else from the hover set.

        A panel of weight zero that the ``'hover'`` option moved to the hover data (outside the crosswind table) is
        labelled from the hover set, like the data it got.
        """
        alpha = np.asarray(alpha, dtype=float)
        outside, clamped = _outside_low_table(self.low, alpha)
        low_alone = self.weights == 0.0
        if self.out_of_range == "hover":
            low_alone = low_alone & ~outside
        return np.where(low_alone, self.low.attached(clamped), self.high.attached(alpha))


def stall_switches(solution, panels):
    """Attached/not-attached label changes between neighbouring panels among the first ``panels``, and the not-attached count.

    Not attached is every regime other than attached (post-stall, reverse flow, no inflow).
    ``(None, None)`` for a solution without panel regimes (source closure). The count is blind when
    every panel carries the same label; ``incidence_steps`` is not.
    """
    if getattr(solution, "panel_regime", None) is None:
        return None, None
    stalled = np.asarray(solution.panel_regime)[:panels] != 0
    return int(np.sum(stalled[1:] != stalled[:-1])), int(stalled.sum())


def incidence_steps(solution, wing, limit_deg=6.0):
    """Per lifting surface: largest neighbour-to-neighbour step of effective incidence and the count above the limit.

    An indicator of spanwise roughness, not a criterion.
    """
    alpha = np.rad2deg(np.asarray(solution.polar_alpha_rad))
    names = getattr(wing, "names", None)
    parts = getattr(wing, "wings", None)
    if parts is None:
        names, parts = ("wing",), (wing,)
    out, offset = {}, 0
    for name, part in zip(names, parts):
        steps = np.abs((np.diff(alpha[offset:offset+part.panels])+180.0) % 360.0-180.0)
        out[name] = dict(max_step_deg=float(steps.max()) if len(steps) else 0.0,
                         steps_over_limit=int(np.sum(steps > limit_deg)))
        offset += part.panels
    out["total_steps_over_limit"] = int(sum(v["steps_over_limit"] for v in out.values()))
    out["limit_deg"] = float(limit_deg)
    return out


@dataclass(frozen=True)
class HandoverChoice:
    """What one step uses: the weight, the section data, the solve settings and the seeding rule."""
    weight: float
    polar: object
    solve: SolveSettings
    zero_seed: bool
    # per-panel weights (weight_basis='panel'); None with the origin basis. An array: left out of equality.
    panel_weights: object = field(default=None, compare=False)


@dataclass(frozen=True)
class SectionHandover:
    """The hand-over rule bound to two sets of section data for one geometry."""
    low_polar: object
    high_polar: object
    seeding: str = "previous"
    stall: dict = field(default_factory=lambda: dict(MAKANI_M600_STALL))
    low_solve: SolveSettings = CROSSWIND_SOLVE
    high_solve: SolveSettings = TAKEOFF_LANDING_SOLVE
    out_of_range: str = "error"            # what a panel outside the crosswind table gets inside the blend (BlendedPolars)
    weight_basis: str = "origin"           # 'origin': one weight from the body origin; 'panel': one per panel (module text)

    def __post_init__(self):
        if self.seeding not in SEEDINGS:
            raise ValueError(f"seeding must be one of {SEEDINGS}")
        if self.out_of_range not in OUT_OF_RANGE:
            raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE}")
        if self.weight_basis not in WEIGHT_BASES:
            raise ValueError(f"weight_basis must be one of {WEIGHT_BASES}")
        if self.low_solve.closure != "source" or self.high_solve.closure != "full_incidence":
            raise ValueError("the crosswind side uses the source closure and the hover side the full-incidence closure")
        missing = set(MAKANI_M600_STALL)-set(self.stall)
        if missing or self.stall["blending_deg"] <= 0:
            raise ValueError("stall parameters incomplete or blend width not positive")

    def select(self, alpha_deg, beta_deg):
        """Choice for a body incidence and sideslip (degrees) of the relative wind at the body origin."""
        weight = makani_weight(alpha_deg, beta_deg, self.stall)
        if weight == 0.0:
            return HandoverChoice(0.0, self.low_polar, self.low_solve, False)
        return HandoverChoice(weight, BlendedPolars(self.low_polar, self.high_polar, weight, self.out_of_range), self.high_solve,
                              self.seeding == "zero")

    def select_panels(self, relative_body, origin_weight, relative_body_load=None):
        """Choice for the per-panel basis from every panel's oncoming flow in body axes (``panel_weights``).

        ``relative_body`` is the flow at the control points; with ``relative_body_load`` (the flow at the
        load points) a panel's weight is the larger of the two. ``origin_weight`` is the weight of the
        body origin: it stands in for a point without oncoming flow and is what the record keeps as
        ``handover_weight``. All panel weights zero gives the production crosswind choice; all equal gives
        a single-weight ``BlendedPolars`` at that common weight (which is the origin's weight in a uniform
        stream past a body that does not rotate, so that such a run reproduces the origin basis bit for
        bit; elsewhere the common weight need not be the origin's); otherwise ``PanelBlendedPolars``.
        """
        weights = panel_weights(relative_body, self.stall, fallback=origin_weight)
        if relative_body_load is not None:
            at_load = panel_weights(relative_body_load, self.stall, fallback=origin_weight)
            if at_load.shape != weights.shape:
                raise ValueError("one flow per panel at the control points and at the load points")
            weights = np.maximum(weights, at_load)
        if not np.any(weights > 0.0):
            return HandoverChoice(float(origin_weight), self.low_polar, self.low_solve, False, weights)
        if np.all(weights == weights[0]):
            polar = BlendedPolars(self.low_polar, self.high_polar, float(weights[0]), self.out_of_range)
        else:
            polar = PanelBlendedPolars(self.low_polar, self.high_polar, weights, self.out_of_range)
        return HandoverChoice(float(origin_weight), polar, self.high_solve, self.seeding == "zero", weights)

    def metadata(self):
        return dict(rule="Makani simulator weights on section data (decision 0025 amendment 1)", stall=dict(self.stall),
                    seeding=self.seeding, out_of_range=self.out_of_range, weight_basis=self.weight_basis,
                    low_closure=self.low_solve.closure, high_closure=self.high_solve.closure,
                    span_flow_fraction=self.high_solve.span_flow_fraction, reverse_flow=self.high_solve.reverse_flow,
                    reverse_fade=self.high_solve.reverse_fade, low_solver=self.low_solve.solver, high_solver=self.high_solve.solver,
                    low_solve=solve_settings_metadata(self.low_solve), high_solve=solve_settings_metadata(self.high_solve))
