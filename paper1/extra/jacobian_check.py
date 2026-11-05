"""Finite-difference Jacobian of the Newton iteration (nonlinear.solve_newton rule) against the exact active-segment
Jacobian vsm._vsm_closure_jacobian, at converged steady solutions (paper session, 2026-10-08; code e58677c)."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import steady_analyses as sa
from openfawe.aero.vsm import geometry, _vsm_closure, _vsm_closure_jacobian
from openfawe.aero.vsm_wake import solve_finite_vsm, vsm_wake_influence
out = {}
for case in ("V3", "M600"):
    wing, polar, wind, area, chord = sa.setup(case)
    rows = sa.straight_rows(wing, wind, 100 * chord)
    cfg = sa.config_for(0.15)
    sol = solve_finite_vsm(wing, polar, wind, rows, config=cfg)
    g = np.asarray(sol.circulation)
    cp, ch, normal, tangent, span = geometry(wing)
    amb = np.broadcast_to(wind, (wing.panels, 3)).astype(float)
    cross = np.linalg.norm(np.cross(amb, span), axis=1)
    A = vsm_wake_influence(wing, rows, config=cfg)
    known = np.zeros((wing.panels, 3))
    R = lambda x: _vsm_closure(x, amb, A, known, ch, tangent, normal, span, cross, polar)[0] - x
    J = _vsm_closure_jacobian(g, amb, A, known, ch, tangent, normal, span, cross, polar)
    e0 = R(g); F = np.empty_like(J)
    for j in range(len(g)):
        h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(g[j]))
        d = np.zeros_like(g); d[j] = h
        F[:, j] = (R(g + d) - e0) / h
    rel = float(np.abs(F - J).max() / np.abs(J).max())
    out[case] = dict(max_abs_diff=float(np.abs(F - J).max()), max_abs_J=float(np.abs(J).max()), rel=rel)
    print(case, out[case])
json.dump(out, open("jacobian-check.json", "w"), indent=1)
