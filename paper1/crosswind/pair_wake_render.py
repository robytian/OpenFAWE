"""Three-dimensional wake of two M600 units at the end of a two-unit run (OF-0009, manuscript, author request
2026-10-26 16:31: the images of the tandem and the offset pair, with and without the rotor rings). Display only; the data
are the stored wake networks of the runs (``state_*.npz`` with ``history.json`` of c3_pair.py).

Drawn per run: the ground grid, the ground points and the tethers (straight lines, schematic), both kites with their
airframes, the loop path of the last lap of each kite (dashed), the wind (arrows), the trailing filaments of the main
wing of each unit at the two tips (thick) and at mid-span (thin), coloured by unit, and, with rotors, every rotor ring of
both units (orange). Rows older than the free part of the wake (one lap; drifting with the wind) are drawn lighter. The
filaments are drawn up to ``x_behind`` loop diameters downwind of the loop centre of the downstream unit B.

Close-up of unit B (``--closeup``): every trailing filament of the main wings of both units inside a cube about B (tips
thick, the others thin and light), the rotor rings in the sphere and the airframe of B with its tether.

Usage: pair_wake_render.py OUT_STEM --overview RUN_DIR [RUN_DIR] --closeup RUN_DIR [RUN_DIR ...] [--x-behind 1.0] [--half 70]
       writes the panels OUT_STEM_overview_<run>.png and OUT_STEM_closeup_<run>.png, OUT_STEM_overviews.png/.pdf (2 x 2,
       across the page), OUT_STEM_closeups.png/.pdf (one column) and OUT_STEM_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parents[1]/"src"))
import pof_style as ps  # noqa: E402
import wake_analysis as wa  # noqa: E402
import wake_figures as wf  # noqa: E402
import wake_render as wr  # noqa: E402

UNIT_COLOUR = {"A": ps.PALETTE["blue"], "B": ps.PALETTE["pink"]}       # paper palette (author 2026-10-26 17:16)
UNIT_LIGHT = {"A": "#A6CEE3", "B": "#E9C6DD"}
RING_COLOUR = ps.PALETTE["orange"]
VIEW = np.array([-0.62, -1.0, 0.55])


def _clip(points, xmax):
    """Leading part of a polyline (newest first) up to the first point beyond x = xmax."""
    beyond = np.nonzero(points[:, 0] > xmax)[0]
    return points if len(beyond) == 0 else points[:max(beyond[0], 1)]


def render(run, out_png, x_behind=1.0, size=(3000, 1700), extent=None):
    import pyvista as pv
    import airframe
    pv.OFF_SCREEN = True
    s = run["settings"]
    nodes, edges, gamma = wa.wing_part(run["nodes"], run["edges"], run["gamma"], run["ring_first"])
    rows, _ = wa.wake_rows(nodes, edges, gamma, run["first"])
    bodies = list(run["motions"])
    per_body = run["first"]//len(bodies)
    n, S = run["steps_per_lap"], run["step"]
    free_rows = int(round(float(s.get("free_laps", 1.0))*n))
    wn = wf.WING_NODES_M600
    ground = {"A": np.zeros(3), "B": np.asarray(s.get("ground_point_b_m", [0., 0., 0.]), dtype=float)}
    centre = {k: run["motions"][k]["position_m"][max(0, S-n+1):S+1].mean(axis=0) for k in bodies}
    diameter = float(s.get("loop_diameter_m", 270.0))
    xmax = centre["B"][0]+x_behind*diameter
    plot = pv.Plotter(off_screen=True, window_size=size)
    plot.set_background("white")
    drawn = {}
    for b, body in enumerate(bodies):
        offset = b*per_body
        for column, width in ((offset, 4.5), (offset+wn//2, 2.0), (offset+wn-1, 4.5)):
            line = rows[:, column]
            fresh = _clip(line[:free_rows+1], xmax)
            old = _clip(line[free_rows:], xmax) if len(fresh) == min(free_rows+1, len(line)) else line[:0]
            if len(fresh) > 1:
                plot.add_mesh(wr._polyline(fresh), color=UNIT_COLOUR[body], line_width=width, render_lines_as_tubes=True, lighting=True)
            if len(old) > 1:
                plot.add_mesh(wr._polyline(old), color=UNIT_LIGHT[body], line_width=0.7*width, render_lines_as_tubes=True, lighting=True)
            drawn[f"{body}_{column-offset}"] = int(len(fresh)+len(old))
    rings, per = wr._rings(run)
    ring_count = 0
    if rings is not None:
        # at the scale of the loops a ring is a dot: each slipstream tube is drawn as the line through its ring centres
        discs = int((s.get("rotors") or {}).get("discs", 8)) if isinstance(s.get("rotors"), dict) else 8
        centres = rings.mean(axis=1)
        steps = len(centres)//(discs*len(bodies))
        for k in range(discs*len(bodies)):
            line = centres[k::discs*len(bodies)][:steps]
            if len(line) > 1:
                plot.add_mesh(wr._polyline(line), color=RING_COLOUR, line_width=6.0, render_lines_as_tubes=True, lighting=False)
        ring_count = int(len(rings))
    for body in bodies:
        motion = run["motions"][body]
        airframe.add_airframe(plot, motion["rotation"][S], motion["position_m"][S], ground_point=ground[body], scale_lines=5.0,
                              rotors=rings is not None)
        plot.add_mesh(pv.Sphere(radius=6.0, center=motion["position_m"][S]), color=wr.TETHER_COLOUR)
        plot.add_mesh(wr._dashed(motion["position_m"][max(0, S-n):S+1], on=10, off=7), color=wr.PATH_COLOUR, line_width=3.0,
                      lighting=False)
        plot.add_mesh(pv.Sphere(radius=4.0, center=ground[body]), color=wr.TETHER_COLOUR)
    x0 = min(ground["A"][0], centre["A"][0]-0.7*diameter)
    xs = np.arange(np.floor(x0/100)*100-100, np.ceil(xmax/100)*100+101, 100.0)
    ylo = min(-200.0, min(c[1] for c in centre.values())-200.0)
    yhi = max(200.0, max(c[1] for c in centre.values())+200.0)
    ys = np.arange(np.floor(ylo/100)*100, np.ceil(yhi/100)*100+1, 100.0)
    for g in [np.array([[x, ys[0], 0.], [x, ys[-1], 0.]]) for x in xs]+[np.array([[xs[0], y, 0.], [xs[-1], y, 0.]]) for y in ys]:
        plot.add_mesh(wr._polyline(g), color=wr.GRID_COLOUR, line_width=1.0, lighting=False)
    for z in (60., 130., 200.):
        plot.add_mesh(wr._arrow(np.array([ground["A"][0]+20., ground["A"][1]-120., z]), np.array([1., 0., 0.]), 80.), color=wr.WIND_COLOUR)
    focus = np.array([0.5*(xs[0]+xs[-1]), 0.5*(centre["A"][1]+centre["B"][1]), 0.42*centre["A"][2]])
    span = float(xs[-1]-xs[0]) if extent is None else float(extent)
    plot.camera_position = (focus+VIEW/np.linalg.norm(VIEW)*1.08*span, focus, (0., 0., 1.))
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(str(out_png))
    plot.close()
    return dict(step=int(S), laps=float(S/n), free_rows=free_rows, x_max_m=float(xmax), rings=ring_count, drawn_rows=drawn,
                ground_b_m=ground["B"].tolist(), scene_length_m=span)


def _inside_segments(points, centre, half):
    """Contiguous runs of a polyline whose points lie within the sphere of radius ``half`` about ``centre``."""
    inside = np.linalg.norm(points-centre, axis=1) <= half
    out, start = [], None
    for k, flag in enumerate(inside):
        if flag and start is None:
            start = k
        if (not flag or k == len(inside)-1) and start is not None:
            stop = k+1 if flag else k
            if stop-start > 1:
                out.append(points[start:stop])
            start = None
    return out


def closeup(run, out_png, half=70.0, size=(2400, 1800), view=(-0.62, -0.62, 0.48)):
    """Unit B and its surroundings: the filaments of both units within a sphere of radius ``half`` about B (every
    trailing filament of the main wings, tips thicker), the rotor rings of both units in the sphere and the airframe of B."""
    import pyvista as pv
    import airframe
    pv.OFF_SCREEN = True
    s = run["settings"]
    nodes, edges, gamma = wa.wing_part(run["nodes"], run["edges"], run["gamma"], run["ring_first"])
    rows, _ = wa.wake_rows(nodes, edges, gamma, run["first"])
    bodies = list(run["motions"])
    per_body = run["first"]//len(bodies)
    S = run["step"]
    wn = wf.WING_NODES_M600
    motion = run["motions"]["B"]
    centre = np.asarray(motion["position_m"][S], dtype=float)
    ground_b = np.asarray(s.get("ground_point_b_m", [0., 0., 0.]), dtype=float)
    plot = pv.Plotter(off_screen=True, window_size=size)
    plot.set_background("white")
    count = {}
    for b, body in enumerate(bodies):
        offset = b*per_body
        count[body] = 0
        for column in range(offset, offset+wn):
            tip = column in (offset, offset+wn-1)
            for seg in _inside_segments(rows[:, column], centre, half):
                plot.add_mesh(wr._polyline(seg), color=UNIT_COLOUR[body] if tip else UNIT_LIGHT[body], line_width=3.0 if tip else 1.0,
                              render_lines_as_tubes=tip, lighting=tip, opacity=1.0 if tip else 0.6)
                count[body] += len(seg)
    rings, per = wr._rings(run)
    shown = 0
    if rings is not None:
        keep = np.linalg.norm(rings.mean(axis=1)-centre, axis=1) <= half
        pick = rings[keep]
        if len(pick):
            cells = np.hstack([[per+1]+list(range(k*per, (k+1)*per))+[k*per] for k in range(len(pick))])
            plot.add_mesh(pv.PolyData(pick.reshape(-1, 3), lines=cells), color=RING_COLOUR, line_width=1.6, lighting=False)
        shown = int(keep.sum())
    airframe.add_airframe(plot, motion["rotation"][S], centre, ground_point=ground_b, scale_lines=1.5, rotors=rings is not None,
                          disc_outline=rings is not None)
    v = np.asarray(view, dtype=float)
    plot.camera_position = (centre+v/np.linalg.norm(v)*3.1*half, centre, (0., 0., 1.))
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(str(out_png))
    plot.close()
    return dict(step=int(S), half_m=half, filament_points=count, rings_shown=shown, centre_m=centre.tolist())


def compose(out_stem, overview_pngs, closeup_pngs):
    """Two figures for the two-column layout (author 2026-10-26 17:14, 17:19): the overviews as a 2 x 2 grid across the
    page (OUT_STEM_overviews) and the close-ups of unit B stacked in one column (OUT_STEM_closeups)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import compose_renders as cr
    ps.style()
    handles = [Line2D([], [], color=UNIT_COLOUR["A"], lw=2.4, label="unit A"),
               Line2D([], [], color=UNIT_COLOUR["B"], lw=2.4, label="unit B"),
               Line2D([], [], color=UNIT_LIGHT["A"], lw=1.8, label="A, frozen far wake"),
               Line2D([], [], color=UNIT_LIGHT["B"], lw=1.8, label="B, frozen far wake"),
               Line2D([], [], color=RING_COLOUR, lw=1.6, label="rotor rings")]
    if overview_pngs:
        top = [cr.trimmed(p) for p in overview_pngs]
        cols = 2
        rows = int(np.ceil(len(top)/cols))
        r = max(im.size[1]/im.size[0] for im in top)
        width = ps.WIDTH_DOUBLE_IN
        panel = width/cols
        legend, gap = 0.36, 0.20
        height = panel*rows*r+legend+gap*rows
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(rows, cols, wspace=0.03, hspace=0.12, left=0.0, right=1.0, top=1-gap/height, bottom=legend/height)
        for k, (im, tag) in enumerate(zip(top, "abcdefgh")):
            ax = fig.add_subplot(gs[k//cols, k % cols])
            ax.imshow(np.asarray(im))
            ax.set_axis_off()
            ps.letter(ax, f"({tag})")
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, handlelength=1.6, columnspacing=1.0, frameon=False)
        fig.savefig(str(out_stem)+"_overviews.png", dpi=300)
        fig.savefig(str(out_stem)+"_overviews.pdf")
        plt.close(fig)
    if closeup_pngs:
        ps.column_style()
        low = [cr.trimmed(p) for p in closeup_pngs]
        r = max(im.size[1]/im.size[0] for im in low)
        width = ps.WIDTH_DOUBLE_IN                          # side by side across the page (author 2026-10-26 17:16)
        legend, gap = 0.30, 0.20
        height = width/len(low)*r+legend+gap
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(1, len(low), wspace=0.04, left=0.0, right=1.0, top=1-gap/height, bottom=legend/height)
        for k, (im, tag) in enumerate(zip(low, "abcdefgh")):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(np.asarray(im))
            ax.set_axis_off()
            ps.letter(ax, f"({tag})")
        fig.legend(handles=[handles[0], handles[1], handles[4]], loc="lower center", ncol=3, handlelength=1.6, columnspacing=1.0,
                   frameon=False)
        fig.savefig(str(out_stem)+"_closeups.png", dpi=300)
        fig.savefig(str(out_stem)+"_closeups.pdf")
        plt.close(fig)


def main(argv):
    p = argparse.ArgumentParser(prog="pair_wake_render.py")
    p.add_argument("out_stem")
    p.add_argument("--overview", nargs="*", default=[], help="runs drawn as overviews in the first row")
    p.add_argument("--closeup", nargs="*", default=[], help="runs drawn as close-ups of unit B in the rows below")
    p.add_argument("--x-behind", type=float, default=1.0)
    p.add_argument("--half", type=float, default=70.0)
    args = p.parse_args(argv)
    stem = Path(args.out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    report, top, low = dict(overview={}, closeup={}), [], []
    runs = {d: wf.read(d) for d in dict.fromkeys(args.overview+args.closeup)}
    if args.overview:
        for d in args.overview:
            png = f"{stem}_overview_{Path(d).name.split('-')[0]}.png"
            report["overview"][Path(d).name] = render(runs[d], png, x_behind=args.x_behind)
            top.append(png)
    for d in args.closeup:
        png = f"{stem}_closeup_{Path(d).name.split('-')[0]}.png"
        report["closeup"][Path(d).name] = closeup(runs[d], png, half=args.half)
        low.append(png)
    compose(stem, top, low)
    (stem.parent/(stem.name+"_report.json")).write_text(json.dumps(dict(script="pair_wake_render.py", **report), indent=1)+"\n",
                                                       encoding="utf-8")
    print(json.dumps(report, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1:])
