"""Figures 1-3 and the wake-structure figures of paper 1 (paper session, 2026-10-02).

Figure 1  the two configurations tethered to a ground station (schematic) and their panelled models
Figure 2  treatment of the wing by the lifting line, the vortex step method and the lifting surface
Figure 3  prescribed straight wake and free vortex wake of the two configurations (vortex lattice)
Figure 7  wake of the M600 and of the V3: vortex lattice coloured by ring circulation with cross-flow planes
Figure 8  cross-flow planes: downwash contours with the trace of the wake sheet

Inputs: the aircraft cases of the repository and the stored VSM free-wake states of the
workstation runs p1-*-planes-wakelength (see wakelib.STATE). The straight-wake circulation
is computed here with the prescribed-wake VSM of the repository at its archived settings.
Style: no text inside the figures apart from panel letters, axis labels, legends and colour bars.
Usage: xvfb-run -a python3 make_model_figures.py [fig1 fig2 fig3 fig7 fig8]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import render3d as r3
import wakelib as wl

HERE = Path(__file__).resolve().parent
PARTS = HERE / "parts"
PARTS.mkdir(exist_ok=True)
STEP = {"v3": 180, "m600": 257}                 # 40 chords (V3), 80 chords (M600)
RECEIPT = {}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 4.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.frameon": False, "mathtext.fontset": "stix", "savefig.dpi": 300})


def letter(ax, text, x=0.0, y=1.0):
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom", ha="left")


def show(ax, path):
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def view(p, direction, up=(0, 0, 1), zoom=1.0):
    p.view_vector(direction, viewup=up)
    p.reset_camera()
    p.camera.zoom(zoom)


# ---------------------------------------------------------------- Figure 1
def system_render(key, path):
    """Kite, bridle, tether and ground station (schematic, as in the ground-unit preview of the platform)."""
    wing = wl.aircraft(key).wing
    span = float(np.ptp(wing.bound[:, 1]))
    p = r3.plotter((2400, 1900))
    r3.add_wing(p, wing, edge_width=1.3, smooth=(key == "v3"))
    if key == "m600":
        r3.add_fuselage(p)
    knot = r3.add_ground_unit(p, key, wing)
    stream = wl.freestream(key, 1.0)
    reach = r3.TETHER_SPANS[key] * span
    start = knot + np.array([-1.7 * reach if key == "m600" else -0.95 * reach, -0.35 * span, -0.35 * reach])
    p.add_mesh(pv.Arrow(start=start, direction=stream, scale=0.4 * reach, tip_length=0.25, tip_radius=0.06,
                        shaft_radius=0.02), color=wl.TRAIL, lighting=False)
    view(p, (-0.42, -1.0, 0.34), zoom=1.2)
    r3.shot(p, path)


def model_render(key, path):
    wing = wl.aircraft(key).wing
    span = float(np.ptp(wing.bound[:, 1]))
    p = r3.plotter((2400, 1500))
    r3.add_wing(p, wing, edge_width=2.0, smooth=(key == "v3"))
    if key == "m600":
        r3.add_fuselage(p)
    r3.add_ground_unit(p, key, wing, stub=0.25 * span if key == "v3" else 0.12 * span, bridle_width=1.6)
    view(p, (-0.55, -0.72, 0.42) if key == "v3" else (-0.62, -0.58, 0.52), zoom=1.2)
    r3.shot(p, path)


def fig1():
    for key in ("v3", "m600"):
        system_render(key, PARTS / f"f1_{key}_system.png")
        model_render(key, PARTS / f"f1_{key}_model.png")
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.6), gridspec_kw=dict(height_ratios=[1.15, 1]))
    for ax, name, tag in zip(axes.ravel(), ("f1_v3_system", "f1_m600_system", "f1_v3_model", "f1_m600_model"), "abcd"):
        show(ax, PARTS / f"{name}.png")
        ax.set_anchor("C")
    for tag, (x, y) in zip("abcd", ((0.01, 0.97), (0.51, 0.97), (0.01, 0.44), (0.51, 0.44))):
        fig.text(x, y, f"({tag})", fontweight="bold", fontsize=11, va="top", ha="left")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.01, wspace=0.06, hspace=0.04)
    fig.savefig(HERE / "fig01_models.png")
    plt.close(fig)
    RECEIPT["fig01_models.png"] = dict(inputs=[str(wl.CASES[k].relative_to(wl.REPO)) for k in ("v3", "m600")],
                                       fuselage=str(r3.FUSELAGE.relative_to(wl.REPO)),
                                       schematic=dict(tether_length_spans=r3.TETHER_SPANS, elevation_deg=r3.ELEVATION_DEG),
                                       note="tether, bridle and ground station are schematic; fuselage is display only")


# ---------------------------------------------------------------- Figure 2
def bound_render(key, method, path):
    wing = wl.aircraft(key).wing
    chord = 2.5675 if key == "v3" else 1.2831
    stream = wl.freestream(key, 1.0)
    p = r3.plotter((2000, 1300))
    # The line models see the flat chord strips; the lifting surface uses the mean (camber) surface.
    smooth = key == "v3"                                     # display spline through the nodes of the curved kite
    surface = wl.surface_mesh(wing, smooth=smooth) if method == "ls" else wl.flat_mesh(wing, smooth=smooth)
    outline = wl.strip_outline(wing, smooth=smooth) if method == "ls" else wl.flat_outline(wing, smooth=smooth)
    p.add_mesh(surface, color="#e3e6ea", smooth_shading=True, ambient=0.6, diffuse=0.45, specular=0.0, opacity=0.55)
    p.add_mesh(wl.polyline(outline).tube(radius=0.004 * chord, n_sides=8), color="#8b93a0", lighting=False)
    radius, ball = 0.012 * chord, 0.024 * chord
    bound, trail, closure, load = [], [], [], []
    for part in wl.surfaces(wing):
        if method == "ls":
            mesh = wl.build_lifting_surface_mesh(part, wl.LiftingSurfaceConfig(chordwise_panels=8, chordwise_spacing="cosine"))
            rows = mesh.vortex_rows                                  # (9, nodes, 3): ring fronts and the trailing edge
            bound += [rows[i] for i in range(len(rows) - 1)]
            for j in range(part.node_count):
                trail.append(np.vstack([rows[:, j], rows[-1, j] + 0.8 * chord * stream]))
            closure.append(mesh.collocation_points.reshape(-1, 3))
        else:
            bound.append(part.bound)
            for j in range(part.node_count):
                trail.append(np.array([part.bound[j], part.trailing_edge[j], part.trailing_edge[j] + 0.8 * chord * stream]))
            if method == "ll":
                closure.append(part.cp)
            else:
                closure.append(wl.vsm_geometry(part)[0])
                load.append(part.cp)
    scale = 0.6 if method == "ls" else 1.0
    for line in bound:
        p.add_mesh(r3.tube(wl.smooth_line(line) if smooth else line, radius * scale), color=wl.BOUND, smooth_shading=True)
    for line in trail:
        p.add_mesh(r3.tube(line, radius * scale * 0.8), color=wl.TRAIL, smooth_shading=True)
    for points, colour, size in ((closure, wl.POINT, ball * (0.6 if method == "ls" else 1.0)), (load, wl.LOAD, ball)):
        if points:
            cloud = pv.PolyData(np.vstack(points))
            p.add_mesh(cloud.glyph(geom=pv.Sphere(radius=size, theta_resolution=16, phi_resolution=16), scale=False,
                                   orient=False), color=colour, smooth_shading=True)
    if key == "v3":
        view(p, (-0.55, -0.72, 0.42), zoom=1.3)
    else:
        # starboard outer wing of the M600: the last seven panels up to the tip
        part = wl.surfaces(wing)[0]
        tip = 0.5 * (part.leading_edge[-1] + part.trailing_edge[-1])
        focus = tip + np.array([0.5 * chord, -2.05 * chord, 0.0])
        r3.look(p, focus, (0.38, 0.0, 0.92), 6.9 * chord, up=(-1, 0, 0))
    r3.shot(p, path)


def fig2():
    for key in ("v3", "m600"):
        for method in ("ll", "vsm", "ls"):
            bound_render(key, method, PARTS / f"f2_{key}_{method}.png")
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.3), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for ax, (key, method) in zip(axes.ravel(), [(k, m) for k in ("v3", "m600") for m in ("ll", "vsm", "ls")]):
        show(ax, PARTS / f"f2_{key}_{method}.png")
    for tag, (x, y) in zip("abcdef", [(0.01 + 0.335 * i, top) for top in (0.985, 0.50) for i in range(3)]):
        fig.text(x, y, f"({tag})", fontweight="bold", fontsize=11, va="top", ha="left")
    handles = [Line2D([], [], color=wl.BOUND, linewidth=3, label="Bound vortex"),
               Line2D([], [], color=wl.TRAIL, linewidth=3, label="Trailing vortex"),
               Line2D([], [], color=wl.POINT, marker="o", linestyle="", markersize=4.5, label="Closure point"),
               Line2D([], [], color=wl.LOAD, marker="o", linestyle="", markersize=4.5, label="Load point")]
    fig.legend(handles=handles, loc="lower center", ncol=4, columnspacing=1.4, handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.09, wspace=0.05, hspace=0.05)
    fig.savefig(HERE / "fig02_bound_models.png")
    plt.close(fig)
    RECEIPT["fig02_bound_models.png"] = dict(inputs=[str(wl.CASES[k].relative_to(wl.REPO)) for k in ("v3", "m600")],
                                             chordwise_panels=8, trailing_length_chords=0.8)


# ---------------------------------------------------------------- Figure 3
FILAMENT_CLIM = {"v3": (0.0, 0.065), "m600": (0.0, 0.41)}
FILAMENT_RADIUS = {"v3": 0.0085, "m600": 0.0062}          # in spans
SHED_EVERY = {"v3": 3, "m600": 4}                         # every n-th row of shed filaments is drawn
FIG3_STEP = {"v3": 90, "m600": 128}                       # complete wakes of 20 chords (V3) and 40 chords (M600)


def straight_gamma(key):
    import accel_bench as ab
    from openfawe.aero.methods import AeroBackend
    from openfawe.aero.reference import inflow, load_coefficients
    model = key.upper()
    definition = wl.aircraft(key)
    area, chord, speed = ab.reference_values(model)
    wind = inflow(speed, ab.ALPHA[model], 0.0)
    state = AeroBackend("vsm").start(definition.wing, definition.polar, wind)
    coefficients = load_coefficients(state.solution, definition.wing, wind, area, chord)
    return state.solution.circulation, coefficients


def add_kite(p, key, wing):
    r3.add_wing(p, wing, edge_width=1.4, smooth=(key == "v3"))
    if key == "m600":
        r3.add_fuselage(p)
    r3.add_ground_unit(p, key, wing)


def wake_render(key, kind, path, chords, state, gamma_straight=None):
    wing, chord, speed = state["wing"], state["chord"], state["speed"]
    span = float(np.ptp(wing.bound[:, 1]))
    stream = wl.freestream(key, speed)
    direction = stream / np.linalg.norm(stream)
    step = float(np.linalg.norm(state["rows"][3] - state["rows"][2], axis=1).mean())
    count = int(round(chords * chord / step))
    p = r3.plotter((2400, 1500))
    if kind == "free":
        rows, strengths = state["rows"], state["strengths"] / (speed * chord)
        shed = wl.shed_filaments(rows, strengths, wing.panel_edges, 1, 1 + count, every=SHED_EVERY[key])
        r3.add_wake(p, shed, FILAMENT_CLIM[key], 0.55 * FILAMENT_RADIUS[key] * span)
    else:
        rows = wl.straight_rows(wing, direction, chords * chord, count)
        strengths = np.tile(gamma_straight / (speed * chord), (count + 1, 1))
    filaments = wl.trailing_filaments(rows, strengths, wing.panel_edges, 1, 1 + count)
    r3.add_wake(p, filaments, FILAMENT_CLIM[key], FILAMENT_RADIUS[key] * span)
    add_kite(p, key, wing)
    view(p, (-0.50, -0.70, 0.52) if key == "v3" else (-0.52, -0.64, 0.57), zoom=1.25)
    r3.shot(p, path)
    return count


def colourbar(fig, cax, key, label=r"$|\Gamma_f|\,/\,(U_\infty c)$"):
    mappable = ScalarMappable(norm=Normalize(*FILAMENT_CLIM[key]), cmap=r3.WAKE_CMAP)
    bar = fig.colorbar(mappable, cax=cax)
    bar.set_label(label)
    bar.outline.set_linewidth(0.6)
    bar.locator = matplotlib.ticker.MaxNLocator(4)
    bar.update_ticks()
    return bar


def fig3():
    lengths = {"v3": 20.5, "m600": 40.5}                    # the complete free wake including the starting vortex
    info = {}
    for key in ("v3", "m600"):
        state = wl.read_state(key, FIG3_STEP[key])
        gamma, coefficients = straight_gamma(key)
        np.save(PARTS / f"gamma_straight_{key}.npy", gamma)
        rows = wake_render(key, "straight", PARTS / f"f3_{key}_straight.png", lengths[key], state, gamma)
        wake_render(key, "free", PARTS / f"f3_{key}_free.png", lengths[key], state)
        info[key] = dict(state=str(state["path"].name), state_sha256=wl.sha(state["path"]), shown_chords=lengths[key],
                         shown_rows=rows, straight_CL=coefficients["CL"], straight_CD=coefficients["CD"],
                         filament_clim=list(FILAMENT_CLIM[key]), shed_rows_drawn_every=SHED_EVERY[key])
    fig = plt.figure(figsize=(6.5, 4.9))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035], wspace=0.04, hspace=0.04, left=0.03, right=0.89, top=0.99,
                            bottom=0.01)
    for row, key in enumerate(("v3", "m600")):
        for col, kind in enumerate(("straight", "free")):
            ax = fig.add_subplot(grid[row, col])
            show(ax, PARTS / f"f3_{key}_{kind}.png")
            fig.text(0.01 + 0.445 * col, 0.985 - 0.495 * row, f"({'abcd'[2 * row + col]})", fontweight="bold", fontsize=11,
                     va="top", ha="left")
        cax = fig.add_subplot(grid[row, 2])
        box = cax.get_position()
        cax.set_position([box.x0 + 0.01, box.y0 + 0.08, box.width, box.height - 0.16])
        colourbar(fig, cax, key)
    fig.savefig(HERE / "fig03_wake_models.png")
    plt.close(fig)
    RECEIPT["fig03_wake_models.png"] = info


# ---------------------------------------------------------------- Figures 7 and 8
def plane_fields(state, key, station_chords, half_span_factor=0.62, half_height_factor=0.32, drop_chords=0.0,
                 resolution=(241, 121)):
    from openfawe.aero import wake_probe as wp
    wing, chord, speed = state["wing"], state["chord"], state["speed"]
    stream = wl.freestream(key, speed)
    direction = stream / np.linalg.norm(stream)
    te_centre = wing.trailing_edge.mean(axis=0)
    axis_v = np.cross(direction, [0.0, 1.0, 0.0])
    axis_v = -axis_v / np.linalg.norm(axis_v) if axis_v[2] < 0 else axis_v / np.linalg.norm(axis_v)
    span = float(np.ptp(wing.bound[:, 1]))
    origin = te_centre + station_chords * chord * direction - drop_chords * chord * axis_v
    plane = wp.Plane(origin, [0.0, 1.0, 0.0], axis_v, half_span_factor * span, half_height_factor * span, *resolution)
    fields = wp.plane_fields(plane, state["nodes"], state["edges"], state["gamma"], model=state["core_model"],
                             core_radius=state["radii"], freestream=stream, backend="numba")
    fields["downwash"] = fields["in_plane"][..., 1] / speed
    fields["axis_v"], fields["direction"], fields["te_centre"] = axis_v, direction, te_centre
    fields["drop"] = drop_chords * chord
    return fields


def sheet_trace(state, fields, station_chords):
    """Intersection of the trailing filaments with a cross-flow plane: (y, n) per node column, in metres."""
    rows, chord = state["rows"][1:], state["chord"]
    direction, axis_v, te_centre = fields["direction"], fields["axis_v"], fields["te_centre"]
    distance = (rows - te_centre) @ direction - station_chords * chord          # (depth, width)
    trace = np.full((rows.shape[1], 2), np.nan)
    for j in range(rows.shape[1]):
        d = distance[:, j]
        crossing = np.where(d[:-1] * d[1:] <= 0.0)[0]
        if len(crossing):
            i = crossing[0]
            weight = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0.0
            point = rows[i, j] + weight * (rows[i + 1, j] - rows[i, j])
            trace[j] = [point[1], (point - te_centre) @ axis_v]
    return trace


def scene_render(key, state, path, chords, fields, clim_w):
    wing, chord, speed = state["wing"], state["chord"], state["speed"]
    span = float(np.ptp(wing.bound[:, 1]))
    step = float(np.linalg.norm(state["rows"][3] - state["rows"][2], axis=1).mean())
    count = int(round(chords * chord / step))
    p = r3.plotter((2600, 1500))
    filaments = wl.trailing_filaments(state["rows"], state["strengths"] / (speed * chord), wing.panel_edges, 1, 1 + count)
    r3.add_wake(p, filaments, FILAMENT_CLIM[key], FILAMENT_RADIUS[key] * span)
    shed = wl.shed_filaments(state["rows"], state["strengths"] / (speed * chord), wing.panel_edges, 1, 1 + count,
                             every=2 * SHED_EVERY[key])
    r3.add_wake(p, shed, FILAMENT_CLIM[key], 0.55 * FILAMENT_RADIUS[key] * span)
    for field in fields:
        points = field["points"]
        nv, nu = points.shape[:2]
        grid = pv.StructuredGrid()
        grid.points = points.reshape(-1, 3)
        grid.dimensions = [nu, nv, 1]
        grid.point_data["w"] = field["downwash"].reshape(-1)
        p.add_mesh(grid, scalars="w", cmap="RdBu_r", clim=clim_w, opacity=0.88, lighting=False, show_scalar_bar=False)
        corners = points[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0]]
        p.add_mesh(wl.polyline([corners]), color="#3d4654", line_width=1.5, lighting=False)
    add_kite(p, key, wing)
    view(p, (-0.56, -0.62, 0.55) if key == "v3" else (-0.60, -0.58, 0.55), zoom=1.3)
    r3.shot(p, path)
    return count


def fig7_8():
    setup = {"m600": dict(step=513, shown=70.0, stations=(5.0, 40.0), drop=(0.0, 1.5), half_height=0.32,
                          clim_w=(-0.2, 0.2)),
             "v3": dict(step=180, shown=16.0, stations=(2.0, 10.0), drop=(0.0, 0.25), half_height=0.42,
                        clim_w=(-0.2, 0.2))}
    info, stored = {}, {}
    for key, cfg in setup.items():
        state = wl.read_state(key, cfg["step"])
        fields = [plane_fields(state, key, s, half_height_factor=cfg["half_height"], drop_chords=d)
                  for s, d in zip(cfg["stations"], cfg["drop"])]
        stored[key] = (state, fields)
        rows = scene_render(key, state, PARTS / f"f7_{key}_scene.png", cfg["shown"], fields, cfg["clim_w"])
        info[key] = dict(state=str(state["path"].name), state_sha256=wl.sha(state["path"]), shown_chords=cfg["shown"],
                         shown_rows=rows, stations_chords=list(cfg["stations"]), plane_centre_drop_chords=list(cfg["drop"]),
                         plane_grid=[241, 121], kernel="numba direct sum, the computation's own core model and radii")
    # Figure 7: the two scenes with their colour bars
    fig = plt.figure(figsize=(6.5, 6.9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 0.03], hspace=0.04, left=0.03, right=0.87, top=0.99, bottom=0.11)
    for row, key in enumerate(("m600", "v3")):
        ax = fig.add_subplot(grid[row, 0])
        show(ax, PARTS / f"f7_{key}_scene.png")
        fig.text(0.01, 0.985 - 0.445 * row, f"({'ab'[row]})", fontweight="bold", fontsize=11, va="top", ha="left")
        cax = fig.add_subplot(grid[row, 1])
        box = cax.get_position()
        cax.set_position([box.x0 + 0.015, box.y0 + 0.07, box.width, box.height - 0.14])
        colourbar(fig, cax, key)
    wax = fig.add_axes([0.26, 0.065, 0.42, 0.016])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-0.2, 0.2), cmap="RdBu_r"), cax=wax, orientation="horizontal",
                       ticks=[-0.2, -0.1, 0, 0.1, 0.2])
    bar.set_label(r"$w\,/\,U_\infty$")
    bar.outline.set_linewidth(0.6)
    fig.savefig(HERE / "fig07_wake_structure.png")
    plt.close(fig)
    RECEIPT["fig07_wake_structure.png"] = info
    # Figure 8: contour maps on the planes with the positions of the trailing filaments
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.9), constrained_layout=True)
    levels = np.linspace(-0.2, 0.2, 21)
    traces = {}
    for col, key in enumerate(("m600", "v3")):
        state, fields = stored[key]
        chord = state["chord"]
        for row, (station, field) in enumerate(zip(setup[key]["stations"], fields)):
            ax = axes[row, col]
            image = ax.contourf(field["u"] / chord, (field["v"] - field["drop"]) / chord, field["downwash"], levels=levels,
                                cmap="RdBu_r", extend="both")
            trace = sheet_trace(state, field, station) / chord
            traces[f"{key}_{station:g}c"] = trace.tolist()
            ax.plot(trace[:, 0], trace[:, 1], "o", color="k", markersize=1.5, markeredgewidth=0)
            ax.set_aspect("equal")
            ax.set_ylabel(r"$n/c$")
            if row == 1:
                ax.set_xlabel(r"$y/c$")
            letter(ax, f"({'abcd'[2 * row + col]})", y=1.02)
    bar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.045, pad=0.03, aspect=40,
                       ticks=[-0.2, -0.1, 0, 0.1, 0.2])
    bar.set_label(r"$w\,/\,U_\infty$")
    fig.savefig(HERE / "fig08_wake_planes.png")
    plt.close(fig)
    (PARTS / "fig08_sheet_traces.json").write_text(json.dumps(traces), encoding="utf-8")
    RECEIPT["fig08_wake_planes.png"] = info


# ---------------------------------------------------------------- Figure 13
def fig13():
    """Bound circulation along the span: prescribed straight wake against the free wake (VSM)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1))
    info = {}
    for ax, key, tag in zip(axes, ("v3", "m600"), "ab"):
        state = wl.read_state(key, STEP[key])
        wing, chord, speed = state["wing"], state["chord"], state["speed"]
        gamma, _ = straight_gamma(key)
        part = wl.surfaces(wing)[0]
        count = part.panels                                   # main wing only
        span = float(np.ptp(part.bound[:, 1]))
        station = part.cp[:, 1] / span
        free = state["strengths"][0][:count]
        ax.plot(station, gamma[:count] / (speed * chord), "-", color="#1f5fa8", label="Straight wake")
        ax.plot(station, free / (speed * chord), linestyle=(0, (3.2, 1.9)), color="#D55E00", label="Free wake")
        ax.set_xlabel(r"$y/b$")
        ax.set_ylabel(r"$\Gamma\,/\,(U_\infty c)$")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
        letter(ax, f"({tag})", y=1.02)
        info[key] = dict(state=str(state["path"].name), panels=int(count),
                         max_relative_difference=float(np.abs(free - gamma[:count]).max() / np.abs(gamma[:count]).max()),
                         mean_circulation_ratio_free_over_straight=float(free.sum() / gamma[:count].sum()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    fig.savefig(HERE / "fig13_circulation.png")
    plt.close(fig)
    RECEIPT["fig13_circulation.png"] = info
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    style()
    wanted = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig7", "fig13"]
    if "fig1" in wanted:
        fig1()
    if "fig2" in wanted:
        fig2()
    if "fig3" in wanted:
        fig3()
    if "fig7" in wanted or "fig8" in wanted:
        fig7_8()
    if "fig13" in wanted:
        fig13()
    receipt_path = HERE / "model-figure-receipt.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    previous.setdefault("figures", {}).update(RECEIPT)
    previous.update(producer="make_model_figures.py", producer_sha256=wl.sha(__file__),
                    helpers={n: wl.sha(HERE / n) for n in ("wakelib.py", "render3d.py")}, created="2026-10-02",
                    pyvista=pv.__version__, matplotlib=matplotlib.__version__,
                    repository_commit="e58677c (task/of-0006-accel-20261001)",
                    outputs={p.name: wl.sha(p) for p in sorted(HERE.glob("fig0[1-3]_*.png")) + sorted(HERE.glob("fig0[78]_wake_*.png")) + sorted(HERE.glob("fig13_*.png"))})
    receipt_path.write_text(json.dumps(previous, indent=1), encoding="utf-8")
    print("done", wanted)
