"""Q criterion of the free-wake velocity field (paper session, 2026-10-08; visualisation for Chapter 4).

The velocity induced by the stored filament network of a free-wake state (nodes, edges, circulation, core radii
of the run; Lamb-Oseen, Eq. (1)) is summed with the tree code of the unchanged code (e58677c, theta 0.3, order 2, leaf 32;
segments_velocity backend "tree", singularity "zero" for grid points on a filament axis) on a uniform grid.
The velocity gradient is taken by second-order central differences and Q = (|Omega|^2 - |S|^2)/2 [Hunt et al. 1988];
the streamwise vorticity is stored for colouring. Grid axes: x along the free stream, y spanwise, n normal to the
stream (lift direction), origin at the trailing-edge centroid of the main wing.
Usage: q_field.py KEY STEP SPACING_M X0 X1 Y0 Y1 N0 N1 OUT.npz   (bounds in metres)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "fig"))
import wakelib as wl  # noqa: E402
from openfawe.aero.vortex import CoreModel, TreeSettings, segments_velocity  # noqa: E402


BACKEND = dict(backend="tree", tree=TreeSettings(theta=0.3, order=2, leaf_size=32))


def main(key, step, h, x0, x1, y0, y1, n0, n1, out):
    state = wl.read_state(key, step)
    wing = state["wing"]
    main_nodes = wl.surfaces(wing)[0].node_count
    stream = wl.freestream(key, 1.0)
    ex = stream / np.linalg.norm(stream)
    ey = np.array([0.0, 1.0, 0.0])
    en = np.cross(ex, ey)
    origin = state["rows"][1, :main_nodes].mean(axis=0)
    xs, ys, ns = np.arange(x0, x1 + h / 2, h), np.arange(y0, y1 + h / 2, h), np.arange(n0, n1 + h / 2, h)
    shape = (len(xs), len(ys), len(ns))
    print(key, step, "grid", shape, f"{np.prod(shape) / 1e6:.2f}M points", flush=True)
    vel = np.empty(shape + (3,), dtype=np.float32)
    tick = time.perf_counter()
    gy, gn = np.meshgrid(ys, ns, indexing="ij")
    plane = np.stack([gy.ravel(), gn.ravel()], axis=1)
    for i, x in enumerate(xs):
        pts = origin + x * ex + plane[:, :1] * ey + plane[:, 1:] * en
        v = segments_velocity(pts, state["nodes"], state["edges"], state["gamma"], model=CoreModel(state["core_model"]),
                              core_radius=state["radii"], singularity="zero", **BACKEND)
        vel[i] = np.stack([v @ ex, v @ ey, v @ en], axis=1).reshape(shape[1], shape[2], 3)
        if i % 50 == 0:
            print(f"  plane {i}/{len(xs)} {time.perf_counter() - tick:.0f} s", flush=True)
    grad = np.empty(shape + (3, 3), dtype=np.float32)          # grad[..., i, j] = d u_i / d x_j
    for i in range(3):
        for j, g in enumerate(np.gradient(vel[..., i], h, h, h)):
            grad[..., i, j] = g
    del vel
    q = np.zeros(shape, dtype=np.float32)                       # Q = -tr(G G)/2 = (|Omega|^2 - |S|^2)/2
    for i in range(3):
        for j in range(3):
            q -= 0.5 * grad[..., i, j] * grad[..., j, i]
    omega_x = grad[..., 2, 1] - grad[..., 1, 2]                # dw/dy - dv/dn
    np.savez_compressed(out, q=q.astype(np.float32), omega_x=omega_x.astype(np.float32), x=xs, y=ys, n=ns,
                        origin=origin, ex=ex, ey=ey, en=en, chord=state["chord"], speed=state["speed"], spacing=h,
                        state=str(state["path"]), elapsed=time.perf_counter() - tick)
    print("done", f"{time.perf_counter() - tick:.0f} s", "Q range", float(q.min()), float(q.max()), flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), float(a[3]), *map(float, a[4:10]), a[10])
