"""D3: whole-kite alpha sweep of the full-incidence VSM closure against Makani (contract §4-5).

Case cases/qualification/m600_makani_xfoil_polars.json (archived geometry, Makani XFOIL
sections with the hover model's post-stall blend), 10 m/s, no rotors, ``closure=
'full_incidence'`` with Newton-Picard seeded from the previous angle (continuation).
Straight semi-infinite wake over the full circle at beta = 0, +-15, +-30 deg; free
wake (>= 20 chords) at selected angles. Compared with: the public hover table (S5),
the transcribed strip model (S4; with the 1d66923 2D-to-3D factor and without it,
with and without the pylons OpenFAWE does not model) and the ASWING baseline (S6).

Coefficients are reported in Makani body axes (x forward, y right, z down; moments
about the body origin) with Makani's reference values S 32.9 m^2, b 25.66 m, c 1.28 m.
OpenFAWE's frame is x downstream, y right, z up, i.e. diag(-1, 1, -1) of Makani's.
W3 and W4 of the contract are evaluated here; W2 (continuity) in transition_sweep.py.
Usage: alpha_sweep.py OUT_DIR [--step 5] [--free-wake 0,45,90,-90,135,180] [--chords 25] [--no-free-wake]
                      [--span-fraction F] [--reverse coupled|strip]   (declared regime parameters of the closure)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero import makani_hover_strip as H  # noqa: E402
from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.methods import AeroBackend  # noqa: E402
from openfawe.aero.nonlinear import NonlinearSolveError  # noqa: E402
from openfawe.aero.vsm import REGIMES, solve_vsm  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.aero.wing import SolveSettings  # noqa: E402

CASE = REPO/"cases/qualification/m600_makani_xfoil_polars.json"
TABLE = REPO/"cases/polars/makani/hover_model_sn02_wind_10_no_wake_model.json"
AIRFOILS = REPO/"cases/polars/makani"
ASWING = REPO/"cases/polars/makani/m600_aswing_baseline_zero_angular_rate.json"   # verbatim copy of the Makani database
SPEED = 10.0
RHO = 1.225
MAKANI_REF = dict(S=32.9, b=25.66, c=1.28)
FLIP = np.array([-1.0, 1.0, -1.0])
FULL = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence")


def configure(argv):
    """Set the declared regime parameters from --span-fraction F, --reverse coupled|strip, --reverse-fade W."""
    global FULL
    fraction = float(argv[argv.index("--span-fraction")+1]) if "--span-fraction" in argv else 0.0
    reverse = argv[argv.index("--reverse")+1] if "--reverse" in argv else "coupled"
    fade = float(argv[argv.index("--reverse-fade")+1]) if "--reverse-fade" in argv else 0.2
    FULL = SolveSettings(solver="newton_picard", max_iterations=60, closure="full_incidence",
                         span_flow_fraction=fraction, reverse_flow=reverse, reverse_fade=fade)
    return FULL


def settings_record():
    """The closure settings of this run (the Newton Jacobian is the solver's finite-difference default)."""
    return dict(closure=FULL.closure, solver=FULL.solver, jacobian=FULL.jacobian, max_iterations=FULL.max_iterations,
                span_flow_fraction=FULL.span_flow_fraction, reverse_flow=FULL.reverse_flow,
                reverse_fade=FULL.reverse_fade if FULL.reverse_flow == "strip" else None)


def provenance():
    """Commit of the checkout and whether src/ differs from it (receipts are only as good as this)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return dict(git_commit=git("rev-parse", "HEAD"), src_clean=git("status", "--porcelain", "--", "src") == "",
                src_tree=git("rev-parse", "HEAD:src"))


def weight_counts(solution, panels=slice(None)):
    """Panels by circulation weight (None when no declared parameter is set): zero, partial, one."""
    weight = solution.circulation_weight
    if weight is None:
        return None
    weight = np.asarray(weight)[panels]
    return dict(zero=int(np.sum(weight == 0.)), partial=int(np.sum((weight > 0.) & (weight < 1.))), one=int(np.sum(weight == 1.)))
BETAS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)


def inflow_makani(alpha_rad, beta_rad, speed=SPEED):
    """Apparent wind of Makani's (alpha, beta) in OpenFAWE axes: diag(-1,1,-1) of apparent_wind_b."""
    wind_b = H.apparent_wind_sph_to_cart(speed, np.asarray(alpha_rad), np.asarray(beta_rad))
    return wind_b*FLIP


def makani_coefficients(force, moment_origin, speed=SPEED):
    """[CX, CY, CZ, Cl, Cm, Cn] in Makani axes from OpenFAWE force/moment about the origin."""
    q = 0.5*RHO*speed**2
    f = FLIP*np.asarray(force)/(q*MAKANI_REF["S"])
    m = FLIP*np.asarray(moment_origin)/(q*MAKANI_REF["S"])
    return np.array([f[0], f[1], f[2], m[0]/MAKANI_REF["b"], m[1]/MAKANI_REF["c"], m[2]/MAKANI_REF["b"]])


def solution_record(solution, wing, alpha_deg, beta_deg):
    cfm = makani_coefficients(solution.force, solution.moment(wing))
    regime = np.asarray(solution.panel_regime)
    return dict(alpha_deg=float(alpha_deg), beta_deg=float(beta_deg), status="ok", cfm=cfm.tolist(),
                CL_wind=None, iterations=int(solution.iterations), residual=float(solution.residual),
                regimes={name: int(np.sum(regime == k)) for k, name in enumerate(REGIMES)},
                weights=weight_counts(solution), circulation_max=float(np.max(np.abs(solution.circulation))))


def solve_continued(wing, polar, alpha_from, seed, alpha_to, beta_deg, depth=0, trace=None):
    """Solve at alpha_to seeded from alpha_from; on failure bisect the increment (<= 5 levels).

    Returns (solution, trace) where trace lists the intermediate angles that were
    needed. A failure that survives the bisection is raised to the caller.
    """
    trace = [] if trace is None else trace
    wind = inflow_makani(np.deg2rad(alpha_to), np.deg2rad(beta_deg))
    try:
        return solve_vsm(wing, polar, wind, settings=FULL, initial=seed, density=RHO), trace
    except (NonlinearSolveError, ValueError, FloatingPointError):
        if depth >= 5 or seed is None or alpha_from is None:
            raise
        middle = 0.5*(alpha_from+alpha_to)
        trace.append(float(middle))
        mid_solution, trace = solve_continued(wing, polar, alpha_from, seed, middle, beta_deg, depth+1, trace)
        return solve_continued(wing, polar, middle, mid_solution.circulation, alpha_to, beta_deg, depth+1, trace)


def straight_sweep(aircraft, alphas_deg, beta_deg):
    """Continuation in alpha from 0 outwards (0 -> 180, then 0 -> -180); failures recorded, not hidden.

    Each angle is seeded with the previous converged circulation; if the solver
    fails, the increment is bisected (recorded in ``continuation``). A failure
    after bisection is recorded as failed and the next angle starts from zero.
    """
    wing, polar = aircraft.wing, aircraft.polar
    records = {}
    for branch in (sorted(a for a in alphas_deg if a >= 0), sorted((a for a in alphas_deg if a < 0), reverse=True)):
        seed, previous = None, None
        for alpha in branch:
            tick = time.perf_counter()
            try:
                solution, trace = solve_continued(wing, polar, previous, seed, alpha, beta_deg)
            except (NonlinearSolveError, ValueError, FloatingPointError) as error:
                records[alpha] = dict(alpha_deg=float(alpha), beta_deg=float(beta_deg), status="failed",
                                      error=f"{type(error).__name__}: {str(error)[:300]}", seconds=time.perf_counter()-tick)
                seed, previous = None, None
                continue
            row = solution_record(solution, wing, alpha, beta_deg)
            row["seconds"] = time.perf_counter()-tick
            row["continuation"] = trace
            records[alpha] = row
            seed, previous = solution.circulation, alpha
    return [records[a] for a in sorted(records)]


def free_wake_run(aircraft, alpha_deg, beta_deg, chords, dt_s=0.04, core_radius=0.15):
    """Impulsive-start free wake to the requested wake length with the full-incidence closure."""
    wing, polar = aircraft.wing, aircraft.polar
    chord = MAKANI_REF["c"]
    steps = int(np.ceil(chords*chord/(SPEED*dt_s)))
    config = WakeConfig(dt_s=dt_s, max_rows=steps, core_radius_m=core_radius, convection_integrator="euler",
                        core_evolution="constant", solve=FULL, velocity_backend="tree",
                        tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
    backend = AeroBackend("vsm_free_wake", wake=config)
    wind = inflow_makani(np.deg2rad(alpha_deg), np.deg2rad(beta_deg))
    history, status, error = [], "complete", None
    tick = time.perf_counter()
    try:
        state = backend.start(wing, polar, wind)
        history.append(dict(step=0, time_s=0.0, cfm=makani_coefficients(state.solution.force, state.solution.moment(wing)).tolist()))
        for index in range(1, steps+1):
            state = backend.advance(state, polar, wind, next_time_s=index*dt_s)
            history.append(dict(step=index, time_s=index*dt_s,
                                cfm=makani_coefficients(state.solution.force, state.solution.moment(wing)).tolist(),
                                iterations=int(state.solution.iterations)))
    except (NonlinearSolveError, ValueError, FloatingPointError, RuntimeError) as exc:
        status, error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
    last = history[-1] if history else None
    regime, weights = None, None
    if status == "complete":
        counts = np.asarray(state.solution.panel_regime)
        regime = {name: int(np.sum(counts == k)) for k, name in enumerate(REGIMES)}
        weights = weight_counts(state.solution)
    return dict(alpha_deg=float(alpha_deg), beta_deg=float(beta_deg), chords=chords, dt_s=dt_s, steps=steps, status=status,
                error=error, seconds=time.perf_counter()-tick, final=last, regimes=regime, weights=weights,
                history=[dict(step=h["step"], CZ=h["cfm"][2], CX=h["cfm"][0], Cm=h["cfm"][4]) for h in history[::max(1, steps//25)]+history[-1:]])


def strip_models():
    full = H.HoverStripModel.from_table_params(TABLE, AIRFOILS)
    no3d = H.HoverStripModel.from_table_params(TABLE, AIRFOILS, convert_2d_to_3d=False)
    from dataclasses import replace
    keep = tuple(p for p in full.panels if not p.is_pylon)
    return {"strip_1d66923": full, "strip_no2d3d_(table_like)": no3d,
            "strip_1d66923_no_pylons": replace(full, panels=keep), "strip_no2d3d_no_pylons": replace(no3d, panels=keep)}


def aswing_rows():
    if not ASWING.exists():
        return None
    data = json.loads(ASWING.read_text(encoding="utf-8"))
    alphas, betas = np.array(data["alphads"]), np.array(data["betads"])
    rows = []
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            rows.append(dict(alpha_deg=float(a), beta_deg=float(b), CLtot=float(np.array(data["CLtot"])[0, i, j]),
                             CDtot=float(np.array(data["CDtot"])[0, i, j]),
                             cfm=[float(np.array(data[k])[0, i, j]) for k in ("CXtot", "CYtot", "CZtot", "Cltot", "Cmtot", "Cntot")]))
    return dict(file=str(ASWING.relative_to(REPO)), Sref=data["Sref"], Cref=data["Cref"], Bref=data["Bref"], reynolds=data["reynolds_number"], rows=rows)


def main(out, step=5.0, free_alphas=(0., 45., 90., -90., 135., 180.), chords=25.0, free_wake=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    aircraft = load_aircraft(CASE)
    table_alphas, table_betas, public = H.public_table_cfm(TABLE)
    alphas = np.unique(np.round(np.concatenate([np.arange(-180., 180.+step/2, step), np.rad2deg(table_alphas)]), 6))
    models = strip_models()
    result = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), case=str(CASE.relative_to(REPO)), speed_m_s=SPEED, rho=RHO,
                  makani_reference=MAKANI_REF, settings=settings_record(), **provenance(),
                  axes="Makani body (x forward, y right, z down); moments about the body origin; OpenFAWE = diag(-1,1,-1)",
                  straight={}, strip={}, public_table={}, aswing=aswing_rows(), free_wake=[])
    for beta in BETAS_DEG:
        tick = time.perf_counter()
        rows = straight_sweep(aircraft, alphas, beta)
        result["straight"][f"{beta:g}"] = rows
        failed = [r["alpha_deg"] for r in rows if r["status"] != "ok"]
        print(f"beta {beta:g}: {len(rows)} angles, {len(failed)} failed {failed[:12]}, {time.perf_counter()-tick:.0f} s", flush=True)
        a_rad, b_rad = np.deg2rad(alphas), np.full(alphas.shape, np.deg2rad(beta))
        result["strip"][f"{beta:g}"] = {name: model.force_moment_coefficients(a_rad, b_rad, np.zeros((8,)+alphas.shape), [0., 0., 0.], SPEED).tolist()
                                        for name, model in models.items()}
        j = int(np.argmin(np.abs(np.rad2deg(table_betas)-beta)))
        if abs(np.rad2deg(table_betas[j])-beta) < 1e-6:
            result["public_table"][f"{beta:g}"] = dict(alpha_deg=np.rad2deg(table_alphas).tolist(), cfm=public[:, j, :].tolist())
    result["alphas_deg"] = alphas.tolist()
    if free_wake:
        for alpha in free_alphas:
            run = free_wake_run(aircraft, alpha, 0.0, chords)
            result["free_wake"].append(run)
            print(f"free wake alpha {alpha:g}: {run['status']} {run['steps']} steps {run['seconds']:.0f} s",
                  "CZ %.4f" % run["final"]["cfm"][2] if run["final"] else "", run["error"] or "", flush=True)
            (out/"alpha-sweep.json").write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    result["criteria"] = evaluate(result)
    (out/"alpha-sweep.json").write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(json.dumps(result["criteria"], indent=1))


def evaluate(result):
    """W3 (normal force at +-90 deg vs the public table, beta 0..+-30) and W4 (attached CL vs ASWING)."""
    out = dict(W3=[], W4=[])
    q_S_c = 0.5*RHO*SPEED**2*MAKANI_REF["S"]*MAKANI_REF["c"]
    for beta_key, rows in result["straight"].items():
        table = result["public_table"].get(beta_key)
        if table is None:
            continue
        for alpha in (90.0, -90.0):
            ours = [r for r in rows if abs(r["alpha_deg"]-alpha) < 1e-6]
            # The table has no +-90 deg node: interpolate linearly between its +-67.5 and +-105 deg rows.
            ta = np.array(table["alpha_deg"])
            tc = np.array(table["cfm"])
            cz_table = float(np.interp(alpha, ta, tc[:, 2]))
            cm_table = float(np.interp(alpha, ta, tc[:, 4]))
            if not ours or ours[0]["status"] != "ok":
                out["W3"].append(dict(beta_deg=float(beta_key), alpha_deg=alpha, status="no solution"))
                continue
            cz, cm = ours[0]["cfm"][2], ours[0]["cfm"][4]
            out["W3"].append(dict(beta_deg=float(beta_key), alpha_deg=alpha, CZ_ours=cz, CZ_table=cz_table,
                                  CZ_relative=abs(cz-cz_table)/abs(cz_table), CZ_pass=bool(abs(cz-cz_table) <= 0.15*abs(cz_table)),
                                  Cm_ours=cm, Cm_table=cm_table, Cm_same_sign=bool(np.sign(cm) == np.sign(cm_table)),
                                  Cm_difference_Nm=abs(cm-cm_table)*q_S_c, Cm_pass=bool(np.sign(cm) == np.sign(cm_table) and abs(cm-cm_table)*q_S_c <= 3500.)))
    aswing = result.get("aswing")
    if aswing:
        rows = result["straight"]["0"]
        for ref in aswing["rows"]:
            if ref["beta_deg"] != 0.0:
                continue
            ours = [r for r in rows if abs(r["alpha_deg"]-ref["alpha_deg"]) < 1e-6 and r["status"] == "ok"]
            if not ours:
                continue
            cz = ours[0]["cfm"][2]
            cx = ours[0]["cfm"][0]
            a = np.deg2rad(ref["alpha_deg"])
            cl_ours = (-cz*np.cos(a)+cx*np.sin(a))*MAKANI_REF["S"]/aswing["Sref"]   # wind-axis lift from body coefficients
            out["W4"].append(dict(alpha_deg=ref["alpha_deg"], CL_ours=float(cl_ours), CL_aswing=ref["CLtot"],
                                  relative=float(abs(cl_ours-ref["CLtot"])/abs(ref["CLtot"])),
                                  CL_pass=bool(abs(cl_ours-ref["CLtot"]) <= 0.10*abs(ref["CLtot"]))))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    configure(args)
    step = float(args[args.index("--step")+1]) if "--step" in args else 5.0
    free = tuple(float(x) for x in args[args.index("--free-wake")+1].split(",")) if "--free-wake" in args else (0., 45., 90., -90., 135., 180.)
    chords = float(args[args.index("--chords")+1]) if "--chords" in args else 25.0
    main(args[0], step=step, free_alphas=free, chords=chords, free_wake="--no-free-wake" not in args)
