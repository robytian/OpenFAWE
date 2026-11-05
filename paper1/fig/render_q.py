"""Q-criterion iso-surfaces of the free wake (paper session, 2026-10-08). Display only: no text in the image.
The M600 fuselage (OF-0006-body20260930 mesh) is drawn for orientation only; it is not part of the computations.
Reads ../extra/q/<key>.npz (q_field.py) and the wing geometry; the wing is transformed into the wake frame
(x along the free stream, y spanwise, n normal to the stream) used by q_field.py.
Usage: xvfb-run -a python3 render_q.py KEY NPZ OUT.png THRESHOLD [azimuth elevation distance_factor [LOW_THRESHOLD]]
THRESHOLD (opaque) and LOW_THRESHOLD (translucent, optional) in units of (U/c)^2.
"""
import sys
import numpy as np
import pyvista as pv
import render3d as r3
import wakelib as wl

key, path, out, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
az, el, dist = (float(v) for v in (sys.argv[5:8] if len(sys.argv) > 5 else (-135, 28, 1.0)))
low = float(sys.argv[8]) if len(sys.argv) > 8 else None
d = np.load(path)
x, y, n = d["x"], d["y"], d["n"]
scale = (float(d["speed"]) / float(d["chord"])) ** 2
grid = pv.ImageData(dimensions=(len(x), len(y), len(n)), spacing=(float(d["spacing"]),) * 3, origin=(x[0], y[0], n[0]))
grid.point_data["q"] = (d["q"] / scale).ravel(order="F")
grid.point_data["wx"] = (d["omega_x"] * float(d["chord"]) / float(d["speed"])).ravel(order="F")
iso = grid.contour([thr], scalars="q")
print("iso points", iso.n_points, flush=True)
wing = wl.aircraft(key).wing
origin, ex, ey, en = d["origin"], d["ex"], d["ey"], d["en"]
mesh = wl.surface_mesh(wing, smooth=(key == "v3"))
pts = np.asarray(mesh.points) - origin
mesh.points = np.stack([pts @ ex, pts @ ey, pts @ en], axis=1)
p = r3.plotter((2600, 1500))
lim = 4.0
p.add_mesh(iso, scalars="wx", cmap="RdBu_r", clim=(-lim, lim), smooth_shading=True, show_scalar_bar=False,
           ambient=0.25, diffuse=0.75, specular=0.15)
if low is not None:
    weak = grid.contour([low], scalars="q")
    print("low iso points", weak.n_points, flush=True)
    p.add_mesh(weak, scalars="wx", cmap="RdBu_r", clim=(-lim, lim), smooth_shading=True, show_scalar_bar=False,
               opacity=0.35, ambient=0.3, diffuse=0.7, specular=0.0)
p.add_mesh(mesh, color=wl.SURFACE, smooth_shading=True, ambient=0.35, diffuse=0.7)
if key == "m600":
    # M600 fuselage of OF-0006-body20260930: display only, not part of the computations (as in render3d.add_fuselage)
    with np.load(r3.FUSELAGE) as body:
        nodes, quads = np.asarray(body["nodes"], dtype=float) - origin, body["quads"]
    nodes = np.stack([nodes @ ex, nodes @ ey, nodes @ en], axis=1)
    faces = np.column_stack([np.full(len(quads), 4), quads]).ravel()
    p.add_mesh(pv.PolyData(nodes, faces=faces), color="#aeb5bf", smooth_shading=True, ambient=0.35, diffuse=0.7,
               specular=0.1)
centre = np.array([(x[0] + x[-1]) / 2, 0.0, (n[0] + n[-1]) / 2])
size = float(np.ptp(x))
direction = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                      np.sin(np.radians(el))])
r3.look(p, centre, direction, dist * 1.25 * size, up=(0, 0, 1), angle=30)
p.screenshot(out)
r3.trim(out)
print("written", out)
