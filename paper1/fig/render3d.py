"""Off-screen 3-D renders for the paper-1 figures (white background, no text).

Each function writes a trimmed PNG; make_model_figures.py assembles them with
panel letters, legends and colour bars. Run under xvfb-run.
"""
import numpy as np
import pyvista as pv
from PIL import Image

import wakelib as wl

pv.OFF_SCREEN = True
EDGE = "#3d4654"
def _green_blue():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("green_blue", ["#b6e3a8", "#5cc0a0", "#2f96c0", "#1f5fa8", "#0b2c6b"])


WAKE_CMAP = _green_blue()


def plotter(size=(2400, 1500)):
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    return p


def trim(path, pad=20):
    image = Image.open(path).convert("RGB")
    data = np.asarray(image)
    mask = np.any(data < 250, axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    box = (max(cols[0] - pad, 0), max(rows[0] - pad, 0), min(cols[-1] + pad, data.shape[1]), min(rows[-1] + pad, data.shape[0]))
    image.crop(box).save(path)


def look(p, focus, direction, distance, up=(0, 0, 1), angle=24.0):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    focus = np.asarray(focus, dtype=float)
    p.camera.focal_point = focus
    p.camera.position = focus + distance * direction
    p.camera.up = up
    p.camera.view_angle = angle
    p.camera.clipping_range = (0.01 * distance, 20 * distance)


def tube(points, radius, sides=16):
    return wl.polyline([points]).tube(radius=radius, n_sides=sides)


def add_wing(p, wing, color=wl.SURFACE, edge_width=1.6, opacity=1.0, outline=True, smooth=False):
    """Mean surfaces with the spanwise panel boundaries; edge_width is in thousandths of the span.

    smooth=True draws the surface and its leading and trailing edges with a spline through the nodes (V3 kite).
    """
    p.add_mesh(wl.surface_mesh(wing, smooth=smooth), color=color, smooth_shading=True, opacity=opacity, ambient=0.35,
               diffuse=0.7, specular=0.1, show_edges=False)
    if outline:
        span = float(np.ptp(wing.bound[:, 1]))
        p.add_mesh(wl.polyline(wl.strip_outline(wing, smooth=smooth)).tube(radius=0.0006 * edge_width * span, n_sides=8),
                   color=wl.INK, lighting=False)


def bridle_points(key, wing):
    """Schematic bridle: attachment points on the wing and the bridle point (not part of the aerodynamic model)."""
    if key == "v3":
        part = wl.surfaces(wing)[0]
        picks = [0, 6, 12, 18, 24, 30, 36]
        attach = [part.leading_edge[i] for i in picks] + [part.trailing_edge[i] for i in picks]
        return attach, np.zeros(3)
    part = wl.surfaces(wing)[0]
    quarter = part.bound
    span = quarter[:, 1]
    attach = [quarter[np.argmin(np.abs(span - y))] for y in (-3.6, 3.6)]
    centre = quarter[np.argmin(np.abs(span))]
    return attach, centre + np.array([0.3, 0.0, -4.5])


FUSELAGE = wl.REPO / "paper1/fig/assets/fuselage-s030-r16.npz"
ELEVATION_DEG = 54.0          # as in the ground-unit preview scene of the platform (wing at x 22 m, z 30 m)
TETHER_SPANS = {"v3": 1.1, "m600": 0.42}
BOX_SCALE = {"v3": 1.0, "m600": 0.5}          # box of the platform preview (6 x 4 x 2 m at 26 m span), scaled


def add_fuselage(p):
    """M600 fuselage of OF-0006-body20260930: display only, not part of the computations of the paper."""
    import numpy as _np
    with _np.load(FUSELAGE) as data:
        nodes, quads = data["nodes"], data["quads"]
    faces = _np.column_stack([_np.full(len(quads), 4), quads]).ravel()
    p.add_mesh(pv.PolyData(_np.asarray(nodes, dtype=float), faces=faces), color="#aeb5bf", smooth_shading=True,
               ambient=0.35, diffuse=0.7, specular=0.1)


def add_ground_unit(p, key, wing, ground=True, bridle_width=1.4, stub=None):
    """Bridle, straight tether and ground station (box with drum) below and upwind of the kite.

    Schematic and not to scale: the tether is about one span long; none of it enters the aerodynamic model.
    With stub (m) only the first part of the tether is drawn and the ground station is omitted.
    """
    span = float(np.ptp(wing.bound[:, 1]))
    attach, knot = bridle_points(key, wing)
    p.add_mesh(wl.polyline([[a, knot] for a in attach]), color="#555b66", line_width=bridle_width, lighting=False)
    beta = np.radians(ELEVATION_DEG)
    direction = np.array([-np.cos(beta), 0.0, -np.sin(beta)])
    if stub is not None:
        p.add_mesh(tube(np.array([knot, knot + stub * direction]), 0.004 * span), color="#2b2f36", smooth_shading=True)
        return knot
    scale = BOX_SCALE[key] * span / 26.0
    top = knot + TETHER_SPANS[key] * span * direction
    base = top - np.array([0.0, 0.0, 2.0 * scale])
    p.add_mesh(tube(np.array([knot, top + [0.0, 0.0, 1.0 * scale]]), 0.004 * span), color="#2b2f36", smooth_shading=True)
    p.add_mesh(pv.Cube(center=base + [0.0, 0.0, 1.0 * scale], x_length=6 * scale, y_length=4 * scale, z_length=2 * scale),
               color="#6c7684")
    p.add_mesh(pv.Cylinder(center=top, direction=(0, 1, 0), radius=1.2 * scale, height=2.6 * scale, resolution=48),
               color="#3f4651", smooth_shading=True)
    if ground:
        patch = pv.Plane(center=base + [3.0 * scale, 0.0, -0.01 * scale], direction=(0, 0, 1), i_size=24 * scale,
                         j_size=18 * scale)
        p.add_mesh(patch, color="#e9ecef", lighting=False)
    return knot


def add_wake(p, filaments, clim, radius):
    """Trailing vortex filaments as tubes coloured by the magnitude of their circulation (green to blue)."""
    p.add_mesh(filaments.tube(radius=radius, n_sides=12), scalars="gamma", cmap=WAKE_CMAP, clim=clim,
               smooth_shading=True, ambient=0.55, diffuse=0.5, specular=0.0, show_scalar_bar=False)


def shot(p, path):
    p.screenshot(str(path))
    p.close()
    trim(path)
