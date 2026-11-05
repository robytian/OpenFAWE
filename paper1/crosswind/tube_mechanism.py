"""Mechanism of the slipstream tube near a lifting surface (Sec. III E of the paper, OF-0009): what a load point a
distance d outside the wall of an actuator-disc vortex tube sees when the tube is a row of discrete rings h apart with the
Lamb-Oseen core sigma of the surface points, against the continuous regularised sheet and the ideal (singular) sheet.

Geometry in units of the core radius sigma: a straight tube of radius R = 11.5 sigma (the M600 disc radius 1.15 m over
the surface core 0.1 m), sheet strength gamma = 1 (ring circulation gamma h), rings of 64 straight segments, extending
L = 400 sigma upstream and downstream of the evaluation points so that the ends do not matter; the rings h apart
lie within 40 sigma of the points and both tubes share 0.25 sigma rings beyond, where the discreteness of the near rings
no longer reaches the points. For an infinite tube the
ideal sheet induces an axial velocity gamma inside and zero outside.

Two quantities, both with the kernel of the production code (``segments_velocity``, Lamb-Oseen, numba backend):
  discreteness  the axial velocity at distance d outside the wall over one ring spacing (24 axial positions), its largest
                deviation from the same tube resolved as a continuous sheet (sub-rings h/64 apart), divided by gamma, against
                h/sigma, for d/sigma = 1, 2 and 3. The production code subdivides the rings at the surface points so that
                the spacing seen there is at most sigma/lambda (lambda = 2, Sec. II E).
  smoothing     the axial velocity of the continuous regularised sheet against the distance from the wall, inside and
                outside, divided by gamma, against d/sigma: the jump from gamma to 0 that the ideal sheet has at the wall is
                spread over about two core radii.

Usage: tube_mechanism.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero.vortex import CoreModel, segments_velocity  # noqa: E402

R, L, NSEG = 11.5, 400.0, 64


NEAR = 40.0   # the rings with the true spacing h lie within this distance of the points; beyond it both tubes are the same


def tube(h):
    """Rings h apart within NEAR of the origin (one of them at x = 0) and 0.25 sigma apart beyond, out to L; every ring
    carries gamma times the length of tube it stands for, so both parts are the same sheet of strength gamma = 1."""
    near = np.arange(-np.floor(NEAR/h)*h, NEAR+1e-9, h)
    lo, hi = near[0]-h/2, near[-1]+h/2
    far_lo = np.arange(lo-0.125, -L, -0.25)[::-1]
    far_hi = np.arange(hi+0.125, L, 0.25)
    centres = np.concatenate([far_lo, near, far_hi])
    weights = np.concatenate([np.full(len(far_lo), 0.25), np.full(len(near), h), np.full(len(far_hi), 0.25)])
    phi = np.linspace(0.0, 2*np.pi, NSEG, endpoint=False)
    ring = np.stack([np.zeros(NSEG), R*np.cos(phi), R*np.sin(phi)], axis=1)
    nodes = np.concatenate([ring+np.array([x, 0.0, 0.0]) for x in centres])
    base = np.stack([np.arange(NSEG), (np.arange(NSEG)+1) % NSEG], axis=1)
    edges = np.concatenate([base+k*NSEG for k in range(len(centres))])
    return nodes, edges, np.repeat(weights, NSEG)


def axial(points, h):
    nodes, edges, gamma = tube(h)
    u = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=1.0, backend="numba",
                          singularity="zero")
    return u[:, 0]


def main(out):
    report = dict(script="tube_mechanism.py", units="lengths in core radii sigma, velocities in sheet strengths gamma",
                  R=R, half_length=L, ring_segments=NSEG, discreteness={}, smoothing={})
    ratios = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
    for d in (1.0, 2.0, 3.0):
        rows = []
        for hs in ratios:
            xs = np.linspace(0.0, hs, 24, endpoint=False)
            pts = np.stack([xs, np.full_like(xs, R+d), np.zeros_like(xs)], axis=1)
            discrete = axial(pts, hs)
            fine = axial(pts, hs/64.0)
            rows.append(dict(h_over_sigma=hs, largest_deviation=float(np.abs(discrete-fine).max()),
                             mean_discrete=float(discrete.mean()), mean_continuous=float(fine.mean())))
        report["discreteness"][f"d={d:g}"] = rows
    ds = np.concatenate([np.linspace(-4.0, -0.25, 16), np.linspace(0.25, 4.0, 16)])
    pts = np.stack([np.zeros_like(ds), R+ds, np.zeros_like(ds)], axis=1)
    report["smoothing"] = dict(d_over_sigma=ds.tolist(), axial_over_gamma=axial(pts, 1.0/64.0).tolist(),
                               note="negative d: inside the tube; the ideal sheet gives 1 inside and 0 outside")
    Path(out).write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    r = main(sys.argv[1])
    for k, rows in r["discreteness"].items():
        print(k, [(x["h_over_sigma"], f"{x['largest_deviation']:.2e}") for x in rows])
    print("smoothing", [(round(d, 2), round(u, 4)) for d, u in zip(r["smoothing"]["d_over_sigma"], r["smoothing"]["axial_over_gamma"])])
