"""Lit three-dimensional rendering of aerodynamic states with PyVista (VTK).

Optional module (requires ``pyvista``). It draws exactly the archived network
(``aero_vtk.frame_network``), the wing mean surfaces and an optional fuselage
mesh, with the colour style of the public hero image: dark gradient
background, wake filaments as tubes coloured by |Gamma|, light lit surfaces.
Nothing here changes a computation; PolyData built here is also what the
ParaView export writes (see ``aero_vtk``).
"""

from pathlib import Path
import numpy as np

BG_TOP, BG_BOTTOM = "#050c14", "#0f2738"
INK, MUTED = "#eaf4f9", "#8eaebf"
WAKE_COLOURS = ["#173d55", "#1f7ea2", "#4ccbe8", "#c8f2ff", "#ffffff"]
WING_COLOUR, FUSELAGE_COLOUR = "#dde6ee", "#9aa7b2"
VIEWS = {"v3": (21.0, -126.0), "m600": (25.0, -128.0)}


def wake_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("openfawe_wake", WAKE_COLOURS)


def wake_polydata(nodes, edges, gamma, *, max_length=None):
    """Line PolyData of the filament network with |Gamma| as cell data."""
    import pyvista as pv
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    gamma = np.asarray(gamma, dtype=float)
    keep = np.ones(len(edges), dtype=bool)
    if max_length is not None:
        keep &= np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1) < max_length
    edges, gamma = edges[keep], gamma[keep]
    cells = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    poly = pv.PolyData(nodes, lines=cells)
    poly.cell_data["circulation_m2_s"] = gamma
    poly.cell_data["abs_circulation_m2_s"] = np.abs(gamma)
    return poly


def wing_polydata(leading_edge, trailing_edge, panel_edges):
    """Quad PolyData of the mean surface strips between leading and trailing edges."""
    import pyvista as pv
    le, te = np.asarray(leading_edge, dtype=float), np.asarray(trailing_edge, dtype=float)
    width = len(le)
    faces = np.column_stack([np.full(len(panel_edges), 4), panel_edges[:, 0], panel_edges[:, 1],
                             panel_edges[:, 1] + width, panel_edges[:, 0] + width]).ravel()
    return pv.PolyData(np.vstack([le, te]), faces=faces)


def fuselage_polydata(path):
    """Quad PolyData of a structured fuselage mesh (OF-0006-body20260930 .npz)."""
    import pyvista as pv
    with np.load(path) as data:
        nodes, quads = data["nodes"], data["quads"]
    faces = np.column_stack([np.full(len(quads), 4), quads]).ravel()
    return pv.PolyData(np.asarray(nodes, dtype=float), faces=faces)


def gradient_background(plotter):
    plotter.set_background(BG_BOTTOM, top=BG_TOP)


def add_scene(plotter, *, wake=None, wing=None, fuselage=None, line_width=3.0,
              scalar_limit=None, show_scalar_bar=True):
    """Add the lit scene; returns the actors by name."""
    actors = {}
    if wing is not None:
        actors["wing"] = plotter.add_mesh(wing, color=WING_COLOUR, smooth_shading=False,
                                          specular=0.35, specular_power=18, ambient=0.25,
                                          diffuse=0.8, show_edges=True, edge_color="#ffffff",
                                          edge_opacity=0.35, line_width=0.6)
    if fuselage is not None:
        actors["fuselage"] = plotter.add_mesh(fuselage.compute_normals(auto_orient_normals=True),
                                              color=FUSELAGE_COLOUR, smooth_shading=True,
                                              specular=0.5, specular_power=25, ambient=0.2, diffuse=0.85)
    if wake is not None and wake.n_cells:
        values = wake.cell_data["abs_circulation_m2_s"]
        limit = float(np.percentile(values, 99)) if scalar_limit is None else float(scalar_limit)
        actors["wake"] = plotter.add_mesh(
            wake, scalars="abs_circulation_m2_s", cmap=wake_colormap(), clim=[0.0, max(limit, 1e-12)],
            render_lines_as_tubes=True, line_width=line_width, lighting=True, ambient=0.35,
            show_scalar_bar=False)
        if show_scalar_bar:
            plotter.add_scalar_bar(title="circulation  m²/s", n_labels=2, color=MUTED, title_font_size=14,
                                   label_font_size=11, width=0.22, height=0.05, position_x=0.74,
                                   position_y=0.06, fmt="%.2f")
    return actors


def frame_bounds(wing, chord, frame_chords=12.0, wake_axis=0):
    """Bounds of the wing plus ``frame_chords`` chords downstream along ``wake_axis``."""
    bounds = list(wing.bounds)
    bounds[2 * wake_axis + 1] += frame_chords * chord
    return bounds


def set_view(plotter, bounds, elevation_deg, azimuth_deg, zoom=1.15):
    """Camera like the hero figure: azimuth about z, elevation above the x-y plane."""
    centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    e, a = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    plotter.camera_position = [centre + 2.6 * size * direction, centre, (0.0, 0.0, 1.0)]
    plotter.camera.zoom(zoom)


FONT_FILE = None  # applications may point this at a bundled TTF


def text_font():
    """A Unicode-capable TTF (Greek letters, degree sign): FONT_FILE, else matplotlib's DejaVu."""
    if FONT_FILE and Path(FONT_FILE).exists():
        return str(FONT_FILE)
    try:
        import matplotlib
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return str(path) if path.exists() else None
    except ImportError:
        return None


def render_state(output_png, *, nodes, edges, gamma, leading_edge, trailing_edge, panel_edges,
                 fuselage_path=None, view="v3", title=None, subtitle=None, size=(1800, 1000),
                 max_length=None, line_width=3.0, off_screen=True, frame_chords=12.0, chord=None):
    """Render one state to a PNG; returns the PolyData used (for export)."""
    import pyvista as pv
    plotter = pv.Plotter(off_screen=off_screen, window_size=list(size))
    gradient_background(plotter)
    wake = wake_polydata(nodes, edges, gamma, max_length=max_length)
    wing = wing_polydata(leading_edge, trailing_edge, panel_edges)
    fuselage = fuselage_polydata(fuselage_path) if fuselage_path else None
    add_scene(plotter, wake=wake, wing=wing, fuselage=fuselage, line_width=line_width)
    elevation, azimuth = VIEWS.get(view, VIEWS["v3"])
    if chord is None:
        chord = float(np.median(np.linalg.norm(np.asarray(trailing_edge) - np.asarray(leading_edge), axis=1)))
    set_view(plotter, frame_bounds(wing, chord, frame_chords), elevation, azimuth)
    font = text_font()
    extra = dict(font_file=font) if font else {}
    if title:
        plotter.add_text(title, position=(0.03, 0.91), viewport=True, font_size=16, color=INK, **extra)
    if subtitle:
        plotter.add_text(subtitle, position=(0.03, 0.86), viewport=True, font_size=10, color=MUTED, **extra)
    plotter.screenshot(str(output_png))
    plotter.close()
    return dict(wake=wake, wing=wing, fuselage=fuselage)


def write_polydata(path, poly):
    """Binary VTP for ParaView (the ascii exporter in aero_vtk stays the archive route)."""
    poly.save(str(Path(path)), binary=True)


class ReplayScene:
    """Frames of an aerodynamic archive in one PyVista plotter (window or off-screen).

    The plotter may be ``pyvista.Plotter`` or ``pyvistaqt.QtInteractor``. The wing
    strips and the optional fuselage are added once; the wake actor is replaced
    per frame with the exact archived network of that frame.
    """

    def __init__(self, plotter, reader, *, fuselage_path=None, view=None, line_width=3.0,
                 frame_chords=12.0, scalar_limit=None):
        from .aero_vtk import frame_network
        self.plotter, self.reader = plotter, reader
        self.line_width = line_width
        self._network = frame_network
        first = reader.frame(0)
        le, te = first["leading_edge"], first["trailing_edge"]
        width = len(le)
        panel_edges = first.get("panel_edges", np.column_stack([np.arange(width - 1), np.arange(1, width)]))
        self.chord = float(np.median(np.linalg.norm(te - le, axis=1)))
        self.wing = wing_polydata(le, te, panel_edges)
        self.fuselage = fuselage_polydata(fuselage_path) if fuselage_path else None
        gradient_background(plotter)
        if scalar_limit is None:
            # One colour scale for the whole replay: 99th percentile of the last frame.
            _, _, gamma = frame_network(reader.frame(reader.count - 1))
            scalar_limit = float(np.percentile(np.abs(gamma), 99)) if len(gamma) else 1.0
        self.scalar_limit = max(scalar_limit, 1e-12)
        self.actors = add_scene(plotter, wing=self.wing, fuselage=self.fuselage)
        self.wake_actor = None
        self.index = None
        model_key = "m600" if any("m600" in str(v).lower() for v in (reader.case.get("model"), reader.case.get("name"))) else "v3"
        elevation, azimuth = VIEWS[view or model_key]
        # Frame the wing plus the near wake of the last frame, at most frame_chords downstream.
        last_nodes, last_edges, last_gamma = frame_network(reader.frame(reader.count - 1))
        last = wake_polydata(last_nodes, last_edges, last_gamma, max_length=6.0 * self.chord)
        bounds = list(self.wing.bounds)
        if last.n_cells:
            for axis in range(3):
                bounds[2 * axis] = min(bounds[2 * axis], last.bounds[2 * axis])
                bounds[2 * axis + 1] = max(bounds[2 * axis + 1], last.bounds[2 * axis + 1])
        bounds[1] = min(bounds[1], self.wing.bounds[1] + frame_chords * self.chord)
        set_view(plotter, bounds, elevation, azimuth)
        font = text_font()
        self._font = dict(font_file=font) if font else {}
        self.caption = plotter.add_text("", position=(0.03, 0.9), viewport=True, font_size=10, color=MUTED,
                                        **self._font)
        self.show_frame(0)

    def show_frame(self, index):
        frame = self.reader.frame(int(index))
        nodes, edges, gamma = self._network(frame)
        wake = wake_polydata(nodes, edges, gamma, max_length=6.0 * self.chord)
        if self.wake_actor is not None:
            self.plotter.remove_actor(self.wake_actor, render=False)
        actors = add_scene(self.plotter, wake=wake, line_width=self.line_width, scalar_limit=self.scalar_limit,
                           show_scalar_bar=self.index is None)
        self.wake_actor = actors.get("wake")
        self.index = int(index)
        self.caption.SetInput(f"t = {float(frame['time_s']):.3f} s   step {self.index} of {self.reader.count - 1}"
                              f"   {frame.get('aero_method', self.reader.method)}")
        self.plotter.render()
        return wake

    def set_fuselage_visible(self, visible):
        actor = self.actors.get("fuselage")
        if actor is not None:
            actor.SetVisibility(bool(visible))
            self.plotter.render()


# ---------------------------------------------------------------------------
# Wake-plane probes (aero.wake_probe): slices coloured by a sampled field,
# drawn in the same lit scene, and 2-D figures for publication.
# ---------------------------------------------------------------------------

PLANE_COLORMAPS = {"axial_over_Uinf": "viridis", "speed_over_Uinf": "viridis",
                   "normal_vorticity_1_s": "PuOr_r", "downwash_over_Uinf": "RdBu_r"}


def add_plane(plotter, grid, scalars="axial_over_Uinf", clim=None, opacity=0.92, cmap=None,
              show_scalar_bar=True, title=None):
    """Add one probe plane (StructuredGrid from wake_probe.structured_grid) to the scene."""
    values = grid.point_data[scalars]
    if clim is None:
        low, high = np.percentile(values, [1, 99])
        if scalars.startswith(("normal_vorticity", "downwash")):
            limit = max(abs(low), abs(high), 1e-12)
            clim = [-limit, limit]
        else:
            clim = [float(low), float(high)]
    actor = plotter.add_mesh(grid, scalars=scalars, cmap=cmap or PLANE_COLORMAPS.get(scalars, "viridis"),
                             clim=clim, opacity=opacity, lighting=False, show_scalar_bar=False)
    if show_scalar_bar:
        plotter.add_scalar_bar(title=title or scalars.replace("_", " "), n_labels=3, color=MUTED,
                               title_font_size=13, label_font_size=11, width=0.22, height=0.05,
                               position_x=0.74, position_y=0.14, fmt="%.2f")
    return actor


def plane_figure(output, fields_list, *, scalar="axial", chord=1.0, labels=None, vectors=True,
                 reference=None, title=None, cmap=None, symmetric=False, dpi=200, ncols=None,
                 clip_percentiles=(1, 99)):
    """Matplotlib panels of probe planes (one per entry): filled contours + in-plane vectors.

    ``scalar`` is a key of wake_probe.plane_fields ("axial", "speed", "normal_vorticity") or
    "downwash" (the axis_v component). Values are normalised by the freestream speed except
    the vorticity (1/s). Axes are in chords.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    n = len(fields_list)
    ncols = n if ncols is None else max(1, min(ncols, n))
    nrows = -(-n // ncols)
    aspect = max(1e-6, float(np.ptp(fields_list[0]["v"]) / max(np.ptp(fields_list[0]["u"]), 1e-12)))
    # Row height follows the equal-aspect axes (about 4.6 in of the 5.4 in column is axes width)
    # plus tick labels, so rows pack without the large vertical gaps of the first version.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, max(2.0, 4.6 * aspect + 0.75) * nrows),
                             squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    values_all = []
    for fields in fields_list:
        if scalar == "downwash":
            data = fields["in_plane"][..., 1] / fields["freestream_speed"]
        elif scalar == "normal_vorticity":
            data = fields["normal_vorticity"]
        else:
            data = fields[scalar] / fields["freestream_speed"]
        values_all.append(data)
    stack = np.concatenate([v.ravel() for v in values_all])
    low, high = np.percentile(stack, list(clip_percentiles))
    if symmetric or scalar in ("downwash", "normal_vorticity"):
        limit = max(abs(low), abs(high), 1e-12)
        norm = colors.Normalize(-limit, limit)
        cmap = cmap or ("PuOr_r" if scalar == "normal_vorticity" else "RdBu_r")
    else:
        norm = colors.Normalize(low, high)
        cmap = cmap or "viridis"
    mappable = None
    for ax, fields, data, index in zip(axes_flat, fields_list, values_all, range(n)):
        u, v = fields["u"] / chord, fields["v"] / chord
        mappable = ax.contourf(u, v, data, levels=41, cmap=cmap, norm=norm, extend="both")
        if vectors:
            step = max(1, len(u) // 18)
            ax.quiver(u[::step], v[::step], fields["in_plane"][::step, ::step, 0], fields["in_plane"][::step, ::step, 1],
                      color="#ffffff", alpha=0.75, width=0.003, scale=None)
        if reference is not None:
            ax.plot(*reference[index], color="#ffffff", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{'y' if abs(fields['plane'].axis_u[1]) > 0.5 else 's (along U∞)'} / c")
        if index % ncols == 0:
            ax.set_ylabel("n (normal to U∞) / c")
        if labels:
            ax.set_title(labels[index], fontsize=10)
        ax.tick_params(labelsize=8)
    unit = "1/s" if scalar == "normal_vorticity" else "/ U∞"
    label = {"axial": "u", "speed": "|V|", "downwash": "w", "normal_vorticity": "ω_n"}.get(scalar, scalar)
    fig.colorbar(mappable, ax=axes_flat[:n].tolist(), shrink=0.85, pad=0.02, label=f"{label} {unit}")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output
