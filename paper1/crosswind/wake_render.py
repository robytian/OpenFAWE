"""Three-dimensional wake figures of the paper in the style the author asked for on 2026-10-20 12:48 (the earlier OpenFAWE
renders: vortex filaments as tubes coloured by their circulation with a colour bar, rotor rings in a colour of their own,
and in the overview the tether, the loop path, the wind and the laps of the wake). Display only; the data are the stored
wake networks of the runs (``state_*.npz`` with ``history.json``).

Two renders per run, composed into one figure with matplotlib colour bars (pof_style, no text in the images):
  overview   ground grid, ground point and tether (straight line to the bridle, schematic), the kite with its complete
             airframe, the loop path of the last lap (dashed), wind arrows, and the trailing filaments of the main wing
             (tips and mid-span) coloured by their age in laps, blue to white (``--colour age``, the default, author
             2026-10-20 16:41; ``--colour gamma`` colours them by circulation with one bar shared with the close-up); rows
             older than the free part of a frozen far wake are drawn thinner; no rotor rings (they are not on the age scale,
             author 2026-10-20 16:46; ``--ring-every N`` draws those of every N-th step in orange); ``--zoom-frame`` frames the
             part the close-up magnifies and joins it to the close-up by an arrow (not in the paper, author 16:46). ``--station`` adds
             a true-scale inset of the ground station (airframe.add_ground_station, schematic after the Makani GS02
             dimensions; not used in the paper, author 16:41)
  close-up   the kite and the newest part of its wake: every trailing filament of wing and tails as a tube coloured by
             its circulation (diverging map, symmetric limits), and the rotor rings up to 60 steps old coloured by their
             circulation on the same map, with the disc outlines in black (author 2026-10-20 16:50: everything by
             circulation, the same data and mapping in both panels)

Usage: wake_render.py RUN_DIR OUT_STEM [--step N] [--closeup-steps 120] [--ring-every 0] [--zoom-frame] [--station] [--free-laps 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parents[1]/"src"))
import pof_style as ps  # noqa: E402
import wake_analysis as wa  # noqa: E402
import wake_figures as wf  # noqa: E402

LAP_COLOURS = ["#08306B", "#2171B5", "#6BAED6", "#B3D3EA", "#D6E6F4", "#E8F1FA", "#F2F7FC"]   # blue to white, newest lap darkest (author 10-20 16:41)
RING_COLOUR = "#E6730F"


def ring_cmap():
    """Colour map of the ring circulation (own bar, author 2026-10-20 16:58): oranges, the strongest (most negative) ring
    darkest; the light end of Oranges is cut so that the weakest ring still shows on white."""
    from matplotlib import colormaps, colors
    base = colormaps["Oranges"]
    return colors.ListedColormap(base(np.linspace(1.0, 0.35, 256)), name="rings")


def ring_limits(run):
    """Range of the ring circulation of the stored state, widened to whole m^2/s (None without rings)."""
    rings, per = _rings(run)
    if rings is None:
        return None
    g = _ring_gamma(run, len(rings), per)
    return float(np.floor(g.min())), float(np.ceil(g.max()))
WIND_COLOUR = "#5B8C2A"
PATH_COLOUR = "#6E6E6E"
TETHER_COLOUR = "#2b2f36"
GRID_COLOUR = "#D9D9D9"


def _rows(run):
    nodes, first, rf = run["nodes"], run["first"], run["ring_first"]
    rows = nodes[first:rf].reshape(-1, first, 3)
    wing_nodes, wing_edges, wing_gamma = wa.wing_part(run["nodes"], run["edges"], run["gamma"], rf)
    _, circulation = wa.wake_rows(wing_nodes, wing_edges, wing_gamma, first)
    circulation[0] = circulation[1]
    return rows, circulation


def _ring_gamma(run, count, per):
    """Circulation of each ring (its edges are stored newest first like its nodes, one value per ring)."""
    edges, rf = run["edges"], run["ring_first"]
    g = np.asarray(run["gamma"], dtype=float)[np.all(edges >= rf, axis=1)]
    return g[:count*per].reshape(count, per).mean(axis=1)


def _ring_mesh(pick, gamma, per):
    """Closed ring polygons with their circulation as point data."""
    import pyvista as pv
    cells = np.hstack([[per+1]+list(range(q*per, (q+1)*per))+[q*per] for q in range(len(pick))])
    mesh = pv.PolyData(pick.reshape(-1, 3), lines=cells)
    mesh["gamma"] = np.repeat(gamma, per)
    return mesh


def _rings(run):
    """Ring polygons as (n_rings, nodes_per_ring, 3), newest first, and the nodes per ring."""
    nodes, edges, rf = run["nodes"], run["edges"], run["ring_first"]
    if rf >= len(nodes):
        return None, 0
    # rings are closed polygons of consecutive nodes (k -> k+1, last -> first), newest ring first; the node count per
    # ring is the run's setting, checked against the closing edge of the first ring
    per = int((run["settings"].get("rotors") or {}).get("nodes_per_ring", 16))
    ring_edges = edges[np.all(edges >= rf, axis=1)]
    if len(ring_edges) < per or ring_edges[per-1, 0]-rf != per-1 or ring_edges[per-1, 1]-rf != 0:
        raise ValueError("ring layout differs from consecutive closed polygons of nodes_per_ring nodes")
    count = (len(nodes)-rf)//per
    return nodes[rf:rf+count*per].reshape(count, per, 3), per


def _polyline(points):
    import pyvista as pv
    return pv.lines_from_points(np.asarray(points, dtype=float))


def _dashed(points, on=6, off=5):
    import pyvista as pv
    points = np.asarray(points, dtype=float)
    cells = []
    k = 0
    while k+1 < len(points):
        stop = min(k+on, len(points)-1)
        idx = list(range(k, stop+1))
        cells.append([len(idx)]+idx)
        k = stop+off
    return pv.PolyData(points, lines=np.hstack(cells))


def _arrow(start, direction, length):
    import pyvista as pv
    return pv.Arrow(start=start, direction=direction, scale=length, tip_length=0.22, tip_radius=0.06, shaft_radius=0.018)


def overview(run, out_png, size=(3000, 1500), ring_every=12, free_laps=None, columns=None, camera=None, colour="gamma", clim=None,
             zoom_steps=120, ring_clim=None):
    import pyvista as pv
    import airframe
    pv.OFF_SCREEN = True
    rows, circulation = _rows(run)
    n, S = run["steps_per_lap"], run["step"]
    motion = run["motions"]["A"]
    wn = wf.WING_NODES_M600
    if columns is None:
        columns = [0, wn//4, 3*wn//4, wn-1] if colour == "gamma" else [0, wn//2, wn-1]
    laps = int(np.ceil((rows.shape[0]-1)/n))
    free_rows = None if free_laps is None else int(free_laps*n)
    plot = pv.Plotter(off_screen=True, window_size=size)
    plot.set_background("white")
    # filaments by lap
    for lap in range(laps):
        j0, j1 = lap*n, min((lap+1)*n, rows.shape[0]-1)
        lap_colour = LAP_COLOURS[min(lap, len(LAP_COLOURS)-1)]
        frozen = free_rows is not None and j0 >= free_rows
        for c in columns:
            seg = rows[j0:j1+1, c]
            tip = c in (0, wn-1)
            width = (6.0 if tip else 3.0)*(0.6 if frozen else 1.0)
            if colour == "gamma":                          # coloured by circulation with the limits of the close-up
                line = _polyline(seg)
                line["gamma"] = circulation[j0:j1+1, c]
                plot.add_mesh(line, scalars="gamma", cmap="RdBu_r", clim=[-clim, clim], show_scalar_bar=False, line_width=width,
                              render_lines_as_tubes=True, opacity=1.0, lighting=True)   # same colours as the close-up
            else:
                plot.add_mesh(_polyline(seg), color=lap_colour, line_width=width, render_lines_as_tubes=True,
                              opacity=0.8 if frozen else 1.0, lighting=True)
    # rotor rings
    rings, per = _rings(run)
    if rings is not None and ring_every > 0:
        # every disc's ring of every ring_every-th step (the rings of one step are stored in reverse disc order, so a plain
        # stride over rings would show one disc only)
        discs = int((run["settings"].get("rotors") or {}).get("discs", 8))
        keep = (np.arange(len(rings))//discs) % ring_every == 0
        if colour == "gamma":                              # by circulation on their own scale, as in the close-up
            plot.add_mesh(_ring_mesh(rings[keep], _ring_gamma(run, len(rings), per)[keep], per), scalars="gamma", cmap=ring_cmap(),
                          clim=list(ring_clim), show_scalar_bar=False, line_width=1.0, lighting=False)
        else:
            pick = rings[keep]
            cells = np.hstack([[per+1]+list(range(k*per, (k+1)*per))+[k*per] for k in range(len(pick))])
            plot.add_mesh(pv.PolyData(pick.reshape(-1, 3), lines=cells), color=RING_COLOUR, line_width=1.0, opacity=0.95, lighting=False)
    # kite, tether, path, ground, wind
    airframe.add_airframe(plot, motion["rotation"][S], motion["position_m"][S], ground_point=np.zeros(3), scale_lines=6.0)
    plot.add_mesh(pv.Sphere(radius=5.0, center=motion["position_m"][S]), color=TETHER_COLOUR)

    path = motion["position_m"][max(0, S-n):S+1]
    plot.add_mesh(_dashed(path, on=10, off=7), color=PATH_COLOUR, line_width=3.0, lighting=False)
    pts = rows.reshape(-1, 3)
    lo, hi = np.minimum(pts.min(axis=0), 0.0), pts.max(axis=0)
    xs = np.arange(np.floor(lo[0]/100)*100-100, np.ceil(hi[0]/100)*100+101, 100.0)
    ys = np.arange(np.floor(min(lo[1], -150)/100)*100, np.ceil(max(hi[1], 150)/100)*100+1, 100.0)
    grid = [np.array([[x, ys[0], 0.], [x, ys[-1], 0.]]) for x in xs]+[np.array([[xs[0], y, 0.], [xs[-1], y, 0.]]) for y in ys]
    for g in grid:
        plot.add_mesh(_polyline(g), color=GRID_COLOUR, line_width=1.0, lighting=False)
    plot.add_mesh(pv.Sphere(radius=4.0, center=(0., 0., 0.)), color=TETHER_COLOUR)
    for z in (120., 200., 280., 360.):
        plot.add_mesh(_arrow(np.array([xs[0]+20., ys[0]+40., z]), np.array([1., 0., 0.]), 70.), color=WIND_COLOUR)
    if camera is None:
        focus = 0.5*(np.array([0., 0., 0.])+hi*np.array([1., 0., 1.]))+np.array([0., 0.5*(lo[1]+hi[1]), 0.])
        extent = float(max(hi-lo))
        camera = (focus+np.array([-0.75, -1.05, 0.62])*extent, focus, (0., 0., 1.))
    plot.camera_position = camera
    plot.enable_anti_aliasing("ssaa")
    plot.render()
    # pixel box (left, top, right, bottom) of the part shown in the close-up: the kite and its newest zoom_steps rows
    ren = plot.renderer
    k = min(zoom_steps, rows.shape[0]-1)
    corners = np.vstack([rows[:k+1].reshape(-1, 3), motion["position_m"][S][None]+np.array([[-15., -15., -15.], [15., 15., 15.]])])
    xy = []
    for q in corners:
        ren.SetWorldPoint(float(q[0]), float(q[1]), float(q[2]), 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        xy.append((x, size[1]-y))
    xy = np.asarray(xy)
    image = plot.screenshot(str(out_png), return_img=True)
    scale = image.shape[1]/size[0]
    plot.close()
    zoom = (float(xy[:, 0].min()*scale), float(xy[:, 1].min()*scale), float(xy[:, 0].max()*scale), float(xy[:, 1].max()*scale))
    return laps, zoom


def closeup(run, out_png, size=(3000, 1500), age_steps=120, clim=None, camera=None, ring_clim=None, ring_steps=60):
    import pyvista as pv
    import airframe
    pv.OFF_SCREEN = True
    rows, circulation = _rows(run)
    S = run["step"]
    motion = run["motions"]["A"]
    R, P = motion["rotation"][S], motion["position_m"][S]
    k = min(age_steps, rows.shape[0]-1)
    window = circulation[:k+1]
    limit = float(np.percentile(np.abs(window[:, [0, wf.WING_NODES_M600-1]]), 90)) if clim is None else float(clim)
    plot = pv.Plotter(off_screen=True, window_size=size)
    plot.set_background("white")
    for c in range(rows.shape[1]):
        line = _polyline(rows[:k+1, c])
        line["gamma"] = window[:, c]
        tube = line.tube(radius=0.11, n_sides=12)
        plot.add_mesh(tube, scalars="gamma", cmap="RdBu_r", clim=[-limit, limit], show_scalar_bar=False, smooth_shading=True,
                      ambient=0.3, diffuse=0.75, specular=0.1)
    rings, per = _rings(run)
    if rings is not None:
        count = 8*(min(k, ring_steps)+1)                     # eight discs shed one ring each per step; rings up to ring_steps old (60 by default)
        plot.add_mesh(_ring_mesh(rings[:count], _ring_gamma(run, len(rings), per)[:count], per), scalars="gamma", cmap=ring_cmap(),
                      clim=list(ring_clim or ring_limits(run)), show_scalar_bar=False, line_width=2.4, lighting=False)   # own scale
    airframe.add_airframe(plot, R, P, ground_point=np.zeros(3), rotors=True, disc_outline=rings is not None)
    if camera is None:
        camera = (P+R@np.array([-42., -40., 32.]), P+R@np.array([32., 0., -2.]), tuple(R@np.array([0., 0., 1.])))
    plot.camera_position = camera
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(str(out_png))
    plot.close()
    return limit


def station_inset(run, out_png, size=(1400, 1100)):
    """The ground station at the foot of the tether, at true scale (display only), for the inset of panel (a)."""
    import pyvista as pv
    import airframe
    pv.OFF_SCREEN = True
    S = run["step"]
    kite = np.asarray(run["motions"]["A"]["position_m"][S], dtype=float)
    plot = pv.Plotter(off_screen=True, window_size=size)
    plot.set_background("white")
    depth = airframe.add_ground_station(plot, np.zeros(3), toward=(1.0, 0.0, 0.0))
    direction = kite/np.linalg.norm(kite)
    plot.add_mesh(_polyline(np.vstack([np.zeros(3), 40.0*direction])), color=TETHER_COLOUR, line_width=6.0, lighting=False)
    zg, half = -depth, 16.0
    for v in np.arange(-half, half+0.1, 4.0):
        plot.add_mesh(_polyline([[v, -half, zg], [v, half, zg]]), color=GRID_COLOUR, line_width=1.0, lighting=False)
        plot.add_mesh(_polyline([[-half, v, zg], [half, v, zg]]), color=GRID_COLOUR, line_width=1.0, lighting=False)
    focus = np.array([0.0, 2.0, 0.5*zg])
    plot.camera_position = (focus+np.array([-22.0, -30.0, 14.0]), focus, (0.0, 0.0, 1.0))
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(str(out_png))
    plot.close()


def compose(out_stem, overview_png, closeup_png, laps, limit, free_laps=None, colour="age", inset_png=None, zoom=None, ring_clim=None):
    """Two panels with matplotlib colour bars: (a) wake age in laps (discrete, blue to white) and (b) circulation (red to
    blue); with ``colour="gamma"`` one circulation bar shared by both panels. ``zoom`` (pixel box of the overview render)
    is framed in (a) and joined to (b), which magnifies it (author 2026-10-20 16:41)."""
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    from matplotlib.patches import ConnectionPatch, Rectangle
    import compose_renders as cr
    ps.style()
    (a, abox), b = cr.trimmed_box(overview_png), cr.trimmed(closeup_png)
    ra, rb = a.size[1]/a.size[0], b.size[1]/b.size[0]
    width = ps.WIDTH_DOUBLE_IN
    fig = plt.figure(figsize=(width, width*0.86*(ra+rb)+0.45))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.025], height_ratios=[ra, rb], wspace=0.03, hspace=0.12)
    axes = []
    for row, (img, letter) in enumerate(((a, "(a)"), (b, "(b)"))):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(np.asarray(img))
        ax.set_axis_off()
        ps.letter(ax, letter)
        axes.append(ax)
    if inset_png is not None:                              # optional true-scale inset, lower right of (a)
        c = cr.trimmed(inset_png)
        box = axes[0].inset_axes([0.70, 0.0, 0.30, 0.30*c.size[1]/c.size[0]*a.size[0]/a.size[1]])
        box.imshow(np.asarray(c))
        box.set_xticks([])
        box.set_yticks([])
    if zoom is not None:
        x0, y0, x1, y1 = zoom[0]-abox[0], zoom[1]-abox[1], zoom[2]-abox[0], zoom[3]-abox[1]
        pad = 0.04*max(x1-x0, y1-y0)
        x0, y0, x1, y1 = x0-pad, y0-pad, x1+pad, y1+pad
        axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#4D4D4D", linewidth=0.8))
        fig.add_artist(ConnectionPatch(xyA=(0.5*(x0+x1), y1), coordsA=axes[0].transData, xyB=(0.32, 1.0), coordsB=axes[1].transAxes,
                                       color="#4D4D4D", linewidth=0.8, arrowstyle="-|>", mutation_scale=9, shrinkA=1, shrinkB=2))
    gamma = cm.ScalarMappable(norm=colors.Normalize(-limit, limit), cmap="RdBu_r")
    if colour == "gamma" and ring_clim is not None:       # filaments and rings on their own scales, both panels
        bar = fig.colorbar(gamma, cax=fig.add_subplot(gs[0, 1]))
        bar.set_label(r"$\Gamma$ of the filaments (m$^2$/s)")
        bar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*ring_clim), cmap=ring_cmap()), cax=fig.add_subplot(gs[1, 1]))
        bar2.set_label(r"$\Gamma$ of the rings (m$^2$/s)")
    elif colour == "gamma":
        bar = fig.colorbar(gamma, cax=fig.add_subplot(gs[:, 1]))
        bar.set_label(r"$\Gamma$ (m$^2$/s)")
    else:
        cmap = colors.ListedColormap(LAP_COLOURS[:laps])
        norm = colors.BoundaryNorm(np.arange(0, laps+1), laps)
        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(gs[0, 1]), ticks=np.arange(0, laps+1))
        bar.set_label("wake age (laps)")
        bar.ax.invert_yaxis()
        bar2 = fig.colorbar(gamma, cax=fig.add_subplot(gs[1, 1]))
        bar2.set_label(r"$\Gamma$ (m$^2$/s)")
    fig.savefig(str(out_stem)+".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_stem)+".pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv):
    parser = argparse.ArgumentParser(prog="wake_render.py")
    parser.add_argument("run")
    parser.add_argument("out_stem")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--closeup-steps", type=int, default=120)
    parser.add_argument("--ring-every", type=int, default=48, help="draw the rings of all discs of every N-th step in the overview (0: none)")
    parser.add_argument("--zoom-frame", action="store_true", help="frame the part magnified in (b) and join it to (b) by an arrow (not in the paper; author 16:46)")
    parser.add_argument("--free-laps", type=float, default=None, help="laps of free wake of a frozen-far-wake run (drawn solid)")
    parser.add_argument("--clim", type=float, default=None)
    parser.add_argument("--colour", choices=("gamma", "age"), default="gamma", help="overview filaments by circulation with one bar shared with the close-up (default, author 16:50) or by age")
    parser.add_argument("--station", action="store_true", help="add the true-scale ground-station inset (not used in the paper)")
    args = parser.parse_args(argv)
    run = wf.read(args.run, args.step)
    stem = Path(args.out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if args.clim is None:                                  # one limit for both panels: 90th percentile of the tip circulation
        _, circulation = _rows(run)
        args.clim = float(np.percentile(np.abs(circulation[:, [0, wf.WING_NODES_M600-1]]), 90))
    ring_clim = ring_limits(run)
    laps, zoom = overview(run, str(stem)+"_overview.png", ring_every=args.ring_every, free_laps=args.free_laps, colour=args.colour,
                          clim=args.clim, zoom_steps=args.closeup_steps, ring_clim=ring_clim)
    limit = closeup(run, str(stem)+"_closeup.png", age_steps=args.closeup_steps, clim=args.clim, ring_clim=ring_clim)
    inset = None
    if args.station:
        inset = str(stem)+"_station.png"
        station_inset(run, inset)
    compose(stem, str(stem)+"_overview.png", str(stem)+"_closeup.png", laps, limit, args.free_laps, colour=args.colour,
            inset_png=inset, zoom=zoom if args.zoom_frame else None, ring_clim=ring_clim)
    print(dict(laps=laps, circulation_limit=limit))


if __name__ == "__main__":
    main(sys.argv[1:])
