"""Fig. 7 of the English manuscript v10 (paper session, 2026-10-10): free wakes as vortex lattice and as Q-criterion
iso-surfaces side by side, with the downwash on the cross-flow planes below.

Owner 2026-10-10: 16:29 the two kites at a similar size; 16:32 the Q-criterion figure (Chinese v5 Fig. 9) next to the
vortex-lattice wake of Fig. 7, Fig. 11 (cross-flow planes) below; 16:33 no tether, bridle or ground station, only the kite.

All four renders use one camera rule in span units (direction, distance, view angle, focus one and a half spans behind
the kite) and show the wake over the same number of spans, so that both kites appear at the same size; the images are
placed at one common scale. No solver is called beyond the plane fields of make_model_figures.plane_fields (cached):
  (a), (c) vortex lattice of the stored states (M600 step 513, V3 step 180), drawn as make_model_figures.scene_render
           without the ground unit; filament colour |Gamma_f|/(U c), planes coloured by w/U
  (b), (d) Q = 1.0 (U/c)^2 opaque and 0.015 (U/c)^2 translucent iso-surfaces of ../extra/q/m600-0128.npz and
           v3-0090.npz (q_field.py), drawn as render_q.py; colour omega_x c/U
  (e)-(h) downwash on the planes of (a) and (c) with the crossings of the trailing filaments, as
           make_model_figures.fig7_8 (Chinese v5 Fig. 11)
Usage: python3 make_fig07_wake_q_planes.py [--render]      (--render redoes the four 3-D renders)
Outputs: fig07_wake_q_planes.png/.pdf, parts/f7c_*.png, fig07-wake-q-planes-receipt.json
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import make_draft_figures as mdf
import make_model_figures as mmf
import render3d as r3
import wakelib as wl

HERE = Path(__file__).resolve().parent
PARTS = HERE / "parts"
EXTRA = HERE.parent / "extra"
PLANES = {"m600": dict(step=513, stations=(5.0, 40.0), drop=(0.0, 1.5), half_height=0.32),
          "v3": dict(step=180, stations=(2.0, 10.0), drop=(0.0, 0.25), half_height=0.42)}
Q = {"m600": dict(npz=EXTRA / "q/m600-0128.npz"), "v3": dict(npz=EXTRA / "q/v3-0090.npz")}
SPANS_SHOWN = 3.5            # wake shown downstream of the trailing edge, in spans, in all four renders
CAMERA = dict(azimuth=-128.0, elevation=27.0, distance_spans=5.2, focus_spans=1.5, angle=30.0)
WINDOW = (2600, 1700)
receipt = {"created": "2026-10-10", "inputs": {}, "camera": CAMERA, "spans_shown": SPANS_SHOWN, "values": {}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(key, wing):
    """Unit vectors along the free stream, the span and the normal (upwards), as make_model_figures.plane_fields."""
    stream = wl.freestream(key, 1.0)
    ex = stream / np.linalg.norm(stream)
    en = np.cross(ex, [0.0, 1.0, 0.0])
    en = -en / np.linalg.norm(en) if en[2] < 0 else en / np.linalg.norm(en)
    ey = np.cross(en, ex)
    return ex, ey / np.linalg.norm(ey), en


def camera(p, centre, ex, ey, en, span):
    az, el = np.radians(CAMERA["azimuth"]), np.radians(CAMERA["elevation"])
    direction = np.cos(el) * np.cos(az) * ex + np.cos(el) * np.sin(az) * ey + np.sin(el) * en
    focus = centre + CAMERA["focus_spans"] * span * ex
    r3.look(p, focus, direction, CAMERA["distance_spans"] * span, up=tuple(en), angle=CAMERA["angle"])


def plane_data(key):
    cfg = PLANES[key]
    state = wl.read_state(key, cfg["step"])
    cache = PARTS / f"f7_fields_{key}_{cfg['step']}.pkl"
    if cache.exists():
        fields = pickle.loads(cache.read_bytes())
    else:
        fields = [mmf.plane_fields(state, key, s, half_height_factor=cfg["half_height"], drop_chords=d)
                  for s, d in zip(cfg["stations"], cfg["drop"])]
        cache.write_bytes(pickle.dumps(fields))
    return state, fields


def render_lattice(key):
    import pyvista as pv
    state, fields = plane_data(key)
    wing, chord, speed = state["wing"], state["chord"], state["speed"]
    span = float(np.ptp(wing.bound[:, 1]))
    step = float(np.linalg.norm(state["rows"][3] - state["rows"][2], axis=1).mean())
    count = int(round(SPANS_SHOWN * span / step))
    p = r3.plotter(WINDOW)
    filaments = wl.trailing_filaments(state["rows"], state["strengths"] / (speed * chord), wing.panel_edges, 1, 1 + count)
    r3.add_wake(p, filaments, mmf.FILAMENT_CLIM[key], mmf.FILAMENT_RADIUS[key] * span)
    shed = wl.shed_filaments(state["rows"], state["strengths"] / (speed * chord), wing.panel_edges, 1, 1 + count,
                             every=2 * mmf.SHED_EVERY[key])
    r3.add_wake(p, shed, mmf.FILAMENT_CLIM[key], 0.55 * mmf.FILAMENT_RADIUS[key] * span)
    for field in fields:
        points = field["points"]
        nv, nu = points.shape[:2]
        grid = pv.StructuredGrid()
        grid.points = points.reshape(-1, 3)
        grid.dimensions = [nu, nv, 1]
        grid.point_data["w"] = field["downwash"].reshape(-1)
        p.add_mesh(grid, scalars="w", cmap="RdBu_r", clim=(-0.2, 0.2), opacity=0.88, lighting=False, show_scalar_bar=False)
        corners = points[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0]]
        p.add_mesh(wl.polyline([corners]), color="#3d4654", line_width=1.5, lighting=False)
    r3.add_wing(p, wing, edge_width=1.4, smooth=(key == "v3"))       # the kite only: no bridle, tether or ground unit
    if key == "m600":
        r3.add_fuselage(p)
    ex, ey, en = frame(key, wing)
    camera(p, wing.bound.mean(axis=0), ex, ey, en, span)
    out = PARTS / f"f7c_{key}_lattice.png"
    r3.shot(p, out)
    receipt["values"][f"{key}_lattice_rows_shown"] = count
    return out


def render_q(key):
    import pyvista as pv
    d = np.load(Q[key]["npz"])
    x, y, n = d["x"], d["y"], d["n"]
    scale = (float(d["speed"]) / float(d["chord"])) ** 2
    grid = pv.ImageData(dimensions=(len(x), len(y), len(n)), spacing=(float(d["spacing"]),) * 3,
                        origin=(x[0], y[0], n[0]))
    grid.point_data["q"] = (d["q"] / scale).ravel(order="F")
    grid.point_data["wx"] = (d["omega_x"] * float(d["chord"]) / float(d["speed"])).ravel(order="F")
    wing = wl.aircraft(key).wing
    span = float(np.ptp(wing.bound[:, 1]))
    origin, ex, ey, en = d["origin"], d["ex"], d["ey"], d["en"]
    mesh = wl.surface_mesh(wing, smooth=(key == "v3"))
    pts = np.asarray(mesh.points) - origin
    mesh.points = np.stack([pts @ ex, pts @ ey, pts @ en], axis=1)
    te = (np.asarray(wing.trailing_edge) - origin)
    te_x = float((te @ ex).max())
    x_end = te_x + SPANS_SHOWN * span
    p = r3.plotter(WINDOW)
    for thr, kw in ((1.0, dict(ambient=0.25, diffuse=0.75, specular=0.15)),
                    (0.015, dict(opacity=0.35, ambient=0.3, diffuse=0.7, specular=0.0))):
        iso = grid.contour([thr], scalars="q")
        iso = iso.clip(normal=(1.0, 0.0, 0.0), origin=(x_end, 0.0, 0.0))     # keep x < x_end
        p.add_mesh(iso, scalars="wx", cmap="RdBu_r", clim=(-4, 4), smooth_shading=True, show_scalar_bar=False, **kw)
    p.add_mesh(mesh, color=wl.SURFACE, smooth_shading=True, ambient=0.35, diffuse=0.7)
    if key == "m600":
        with np.load(r3.FUSELAGE) as body:
            nodes, quads = np.asarray(body["nodes"], dtype=float) - origin, body["quads"]
        nodes = np.stack([nodes @ ex, nodes @ ey, nodes @ en], axis=1)
        faces = np.column_stack([np.full(len(quads), 4), quads]).ravel()
        p.add_mesh(pv.PolyData(nodes, faces=faces), color="#aeb5bf", smooth_shading=True, ambient=0.35, diffuse=0.7,
                   specular=0.1)
    e = np.eye(3)
    camera(p, np.asarray(mesh.points).mean(axis=0), e[0], e[1], e[2], span)
    out = PARTS / f"f7c_{key}_q.png"
    p.screenshot(str(out))
    p.close()
    r3.trim(out)
    receipt["inputs"][Q[key]["npz"].name] = sha(Q[key]["npz"])
    receipt["values"][f"{key}_q_wake_end_spans"] = SPANS_SHOWN
    return out


def letter(fig, ax, text):
    box = ax.get_position()
    fig.text(box.x0, box.y1 + 0.003, text, fontweight="bold", fontsize=11, va="bottom", ha="left")


def hbar(fig, rect, norm, cmap, label, ticks=None):
    cax = fig.add_axes(rect)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    bar.set_label(label)
    bar.outline.set_linewidth(0.6)
    if ticks is None:
        bar.locator = matplotlib.ticker.MaxNLocator(4)
        bar.update_ticks()
    return bar


def figure():
    names = {(r, c): PARTS / f"f7c_{k}_{kind}.png" for r, k in enumerate(("m600", "v3"))
             for c, kind in enumerate(("lattice", "q"))}
    img = {rc: mpimg.imread(p) for rc, p in names.items()}
    for p in names.values():
        receipt["inputs"][p.name] = sha(p)
    W = 6.5
    gap = 0.12
    # one common scale (inch per pixel) for all four renders, set by the widest row
    s = min((W - 0.1 - gap) / (img[(r, 0)].shape[1] + img[(r, 1)].shape[1]) for r in (0, 1))
    rows_h = [max(img[(r, 0)].shape[0], img[(r, 1)].shape[0]) * s for r in (0, 1)]
    top_block = rows_h[0] + rows_h[1] + 0.25
    bars = 0.75
    planes = 3.55
    H = 0.2 + top_block + bars + planes + 0.8
    fig = plt.figure(figsize=(W, H))
    y = H - 0.2
    axes_r = {}
    for r in (0, 1):
        y -= rows_h[r]
        x = (W - (img[(r, 0)].shape[1] + img[(r, 1)].shape[1]) * s - gap) / 2
        for c in (0, 1):
            w, h = img[(r, c)].shape[1] * s, img[(r, c)].shape[0] * s
            ax = fig.add_axes([x / W, (y + (rows_h[r] - h) / 2) / H, w / W, h / H])
            ax.imshow(img[(r, c)])
            ax.axis("off")
            axes_r[(r, c)] = ax
            x += w + gap
        y -= 0.25
    # colour bars: filament circulation of (a) and (c), streamwise vorticity of (b) and (d)
    yb = (y - 0.12) / H
    hbar(fig, [0.04, yb, 0.27, 0.085 / H], Normalize(*mmf.FILAMENT_CLIM["m600"]), r3.WAKE_CMAP,
         r"$|\Gamma_f|/(U_\infty c)$, (a)")
    hbar(fig, [0.37, yb, 0.27, 0.085 / H], Normalize(*mmf.FILAMENT_CLIM["v3"]), r3.WAKE_CMAP,
         r"$|\Gamma_f|/(U_\infty c)$, (c)")
    hbar(fig, [0.70, yb, 0.27, 0.085 / H], Normalize(-4, 4), "RdBu_r", r"$\omega_x c/U_\infty$, (b), (d)",
         ticks=[-4, -2, 0, 2, 4])
    # planes
    top = (y - bars) / H
    grid = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=1.05 / H, top=top, hspace=0.32, wspace=0.22)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)])
    levels = np.linspace(-0.2, 0.2, 21)
    image = None
    for col, key in enumerate(("m600", "v3")):
        state, fields = plane_data(key)
        receipt["inputs"][state["path"].name] = wl.sha(state["path"])
        chord = state["chord"]
        for row, (station, field) in enumerate(zip(PLANES[key]["stations"], fields)):
            ax = axes[row, col]
            image = ax.contourf(field["u"] / chord, (field["v"] - field["drop"]) / chord, field["downwash"],
                                levels=levels, cmap="RdBu_r", extend="both")
            trace = mmf.sheet_trace(state, field, station) / chord
            ax.plot(trace[:, 0], trace[:, 1], "o", color="k", markersize=1.5, markeredgewidth=0)
            ax.set_aspect("equal")
            ax.set_ylabel(r"$n/c$")
            if row == 1:
                ax.set_xlabel(r"$y/c$")
    wax = fig.add_axes([(W / 2 - 1.6) / W, 0.42 / H, 3.2 / W, 0.085 / H])
    bar = fig.colorbar(image, cax=wax, orientation="horizontal", ticks=[-0.2, -0.1, 0, 0.1, 0.2])
    bar.set_label(r"$w\,/\,U_\infty$, (a), (c), (e)–(h)")
    bar.outline.set_linewidth(0.6)
    fig.canvas.draw()
    for (rc, ax), t in zip(sorted(axes_r.items()), "abcd"):
        letter(fig, ax, f"({t})")
    for ax, t in zip(axes.flat, "efgh"):
        letter(fig, ax, f"({t})")
    fig.savefig(HERE / "fig07_wake_q_planes.png")
    fig.savefig(HERE / "fig07_wake_q_planes.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt["outputs"] = {"fig07_wake_q_planes.png": sha(HERE / "fig07_wake_q_planes.png"), "size_in": [W, H]}
    for key in ("m600", "v3"):                                   # wake length drawn in (a) and (c)
        state = wl.read_state(key, PLANES[key]["step"])
        span = float(np.ptp(state["wing"].bound[:, 1]))
        step = float(np.linalg.norm(state["rows"][3] - state["rows"][2], axis=1).mean())
        count = int(round(SPANS_SHOWN * span / step))
        receipt["values"][f"{key}_lattice_shown"] = dict(rows=count, chords=round(count * step / state["chord"], 2))


if __name__ == "__main__":
    mdf.style()
    if "--render" in sys.argv:
        for key in ("m600", "v3"):
            render_lattice(key)
            render_q(key)
    figure()
    (HERE / "fig07-wake-q-planes-receipt.json").write_text(json.dumps(receipt, indent=1, default=str), encoding="utf-8")
    print("written fig07_wake_q_planes", receipt["outputs"]["size_in"])
