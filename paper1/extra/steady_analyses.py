"""Steady analyses for the discussion chapter (paper session, 2026-10-08), unchanged repository code (e58677c).

1. Wake length with a prescribed straight wake: the VSM closure with straight finite rings of length L
   (closed by a starting vortex at L), against L -> 1000 chords. This isolates the kinematic effect of
   the starting vortex from the deformation of a free wake.
2. Well-posedness of the closure: Jacobian of the closure residual R = G(Gamma) - Gamma
   (vsm._vsm_closure_jacobian, the exact derivative used by the code) for refined panellings and core
   radii, evaluated (a) at the converged solution where it exists and (b) at a fixed reference
   circulation, the converged solution with the smallest core. Reports the smallest singular value,
   the condition number and the largest diagonal self-coefficient beta_i = dG_i/dGamma_i.
Usage: steady_analyses.py OUT.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper1/scripts/convergence"))
import accel_bench as ab  # noqa: E402
from openfawe.aero.reference import refine_aircraft  # noqa: E402
from openfawe.aero.wake import WakeConfig  # noqa: E402
from openfawe.aero.wing import SolveSettings  # noqa: E402
from openfawe.aero.vsm import geometry, _vsm_closure_jacobian  # noqa: E402
from openfawe.aero.vsm_wake import solve_finite_vsm, vsm_wake_influence  # noqa: E402

CASES = {"V3": ("V3", None), "V3-7.35": ("V3", 7.350324457982788), "M600": ("M600", None)}


def setup(case, refine=1):
    model, alpha = CASES[case]
    definition = ab.load_aircraft(ab.REPO / ab.CASES[model])
    if refine != 1:
        definition = refine_aircraft(definition, refine)
    area, chord, speed = ab.reference_values(model)
    wind = ab.inflow(speed, ab.ALPHA[model] if alpha is None else alpha, 0.0)
    return definition.wing, definition.polar, np.asarray(wind, float), area, chord


def config_for(core):
    return WakeConfig(dt_s=0.02, max_rows=10, core_radius_m=core, convection_integrator="euler",
                      core_evolution="constant", solve=SolveSettings(solver="newton", max_iterations=60))


def straight_rows(wing, wind, length):
    direction = wind / np.linalg.norm(wind)
    return [wing.trailing_edge + length * direction]


def wake_length(case, chords=(5, 10, 20, 40, 80, 160, 320, 1000), core=0.15):
    wing, polar, wind, area, chord = setup(case)
    rows = []
    for n in chords:
        sol = solve_finite_vsm(wing, polar, wind, straight_rows(wing, wind, n * chord), config=config_for(core))
        coeff = ab.load_coefficients(sol, wing, wind, area, chord)
        rows.append(dict(chords=n, CL=coeff["CL"], CD=coeff["CD"], CMy=coeff["CMy"]))
    return rows


def jacobian(wing, polar, wind, rows, config, gamma):
    cp, chord, normal, tangent, span = geometry(wing)
    ambient = np.broadcast_to(wind, (wing.panels, 3)).astype(float)
    cross_speed = np.linalg.norm(np.cross(ambient, span), axis=1)
    matrix = vsm_wake_influence(wing, rows, config=config)
    J = _vsm_closure_jacobian(gamma, ambient, matrix, np.zeros((wing.panels, 3)), chord, tangent, normal, span,
                              cross_speed, polar)
    s = np.linalg.svd(J, compute_uv=False)
    beta = np.diag(J) + 1.0
    return dict(sigma_min=float(s[-1]), sigma_max=float(s[0]), condition=float(s[0] / s[-1]),
                beta_max=float(beta.max()), beta_min=float(beta.min()), argmax=int(beta.argmax()))


def closure(case, refines=(1, 2, 3), cores=(0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4), wake_chords=100):
    out = []
    for refine in refines:
        wing, polar, wind, area, chord = setup(case, refine)
        width = np.linalg.norm(wing.dl, axis=1)
        geometries = {"start": [], "straight": straight_rows(wing, wind, wake_chords * chord)}
        for name, rows in geometries.items():
            reference = None
            for core in cores:
                entry = dict(case=case, refine=refine, panels=int(wing.panels), w_min=float(width.min()), core=core,
                             ratio=core / float(width.min()), geometry=name)
                try:
                    sol = solve_finite_vsm(wing, polar, wind, rows, config=config_for(core))
                    gamma = sol.circulation
                    entry["status"] = "solved"
                    entry["CL"] = ab.load_coefficients(sol, wing, wind, area, chord)["CL"]
                    entry["at_solution"] = jacobian(wing, polar, wind, rows, config_for(core), gamma)
                    if reference is None:
                        reference = gamma.copy()
                except Exception as exc:  # noqa: BLE001 - outcome is data
                    entry["status"] = f"failed: {type(exc).__name__}"
                if reference is not None:
                    try:
                        entry["at_reference"] = jacobian(wing, polar, wind, rows, config_for(core), reference)
                    except Exception as exc:  # noqa: BLE001
                        entry["at_reference"] = f"{type(exc).__name__}: {exc}"[:120]
                out.append(entry)
                print(case, refine, name, core, entry["status"], entry.get("at_solution", {}).get("sigma_min"),
                      flush=True)
    return out


if __name__ == "__main__":
    tick = time.perf_counter()
    result = dict(created="2026-10-08", git_commit="e58677c", note=__doc__.strip().splitlines()[0])
    result["wake_length"] = {case: wake_length(case) for case in ("V3", "M600")}
    print(json.dumps(result["wake_length"], indent=0)[:1500], flush=True)
    result["closure"] = {case: closure(case) for case in ("V3", "V3-7.35", "M600")}
    result["elapsed_s"] = time.perf_counter() - tick
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1), encoding="utf-8")
