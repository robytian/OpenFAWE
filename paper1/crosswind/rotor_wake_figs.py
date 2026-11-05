"""Wake figures of the paper with rotors (OF-0009, Sec. III E FIG. 18 and Sec. III F FIG. 20). Display and read-back
only; the data are the stored wake networks of the runs (``state_*.npz`` with ``history.json``).

FIG. 18, the slipstream near the kite, undeformed tube A'' against model B, both at the end of the last lap (the same
loop angle and pose of the kite):
  (a), (b)  close-up renders (wake_render.closeup): trailing filaments of wing and tails up to 1.2 s old coloured by their
            circulation, rotor rings up to about 0.6 s old coloured by theirs, one scale per quantity for both panels.
  (c)       the rings of the two inboard rotors of the starboard side (top Sti and bottom Sbi, y = 1.213 m) in the body
            x-z plane: the highest and the lowest node of every ring, with the sections of the main wing and the horizontal
            tail at that span station and the two discs.
  (d)       the centre of every ring of the eight rotors against its distance behind its disc along the body x axis:
            height above the hub along the body z axis, mean of the four top and of the four bottom rotors.
  (e)       mean radius of the rings over the disc radius, same abscissa and means.
  Numbers of (d) and (e) at the trailing edge of the main wing and at the horizontal tail go to the JSON report.

FIG. 20, the wake of A at B (two units with rotors, 24 stored networks over the last lap):
  (a), (b)  vorticity of the wake of A in the plane that B flies through (normal to the body x axis of B through its body
            origin, body y-z axes of B) at the stored step where the lift of B is lowest relative to A: flux of the
            lifting-surface filaments of A crossing the plane, spread with a Gaussian of 3 m (wake_figures.flux_field),
            with the bound line of the main wing of B; tandem at two diameters and one diameter downwind with half a
            diameter to the side.
  (c)       change of the lift of B relative to A along the last lap (every step) against the controller's loop angle,
            also for runs without stored networks (--loss-only).
  (d)       velocity induced by the wake of A (lifting-surface filaments of A only, Lamb-Oseen with the run's core, the
            production kernel) at the bound nodes of the main wing of B, mean over the span, along the body z axis of B
            over the airspeed of B (an angle-of-attack change) at the 24 stored steps.

Usage: rotor_wake_figs.py fig18 OUT_DIR TUBE_RUN MODELB_RUN
       rotor_wake_figs.py fig20 OUT_DIR PAIR_RUN [PAIR_RUN ...] --cache CACHE_DIR [--loss-only RUN ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(REPO/"src"))
import pof_style as ps  # noqa: E402
import wake_figures as wf  # noqa: E402
import wake_render as wr  # noqa: E402

TUBE, MODEL_B = "#0072B2", "#D55E00"        # paper palette (author 2026-10-26 17:16): tube T3 blue, model D vermilion
ROTORS = json.loads((REPO/"cases/rotors/m600_rotors_makani.json").read_text(encoding="utf-8"))
HUBS = np.array([r["centre_m"] for r in ROTORS["rotors"]])
DISC_R = 0.5*float(ROTORS["diameter_m"])
TOP = [k for k, r in enumerate(ROTORS["rotors"]) if r["row"] == "top"]
BOTTOM = [k for k, r in enumerate(ROTORS["rotors"]) if r["row"] == "bottom"]


# ----------------------------------------------------------------------------------------------------------------------
# FIG. 18
# ----------------------------------------------------------------------------------------------------------------------


def unit_rings(run, body="A"):
    """Rings of unit ``body`` in its body frame at the stored step: (n_steps, 8, nodes, 3) newest first, the disc index of
    every one of the 8 (by the nearest hub of the newest rings), and the ring age in s."""
    rings, per = wr._rings(run)
    bodies = list(run["motions"])
    nb = len(bodies)
    S = run["step"]
    R = np.asarray(run["motions"][body]["rotation"][S], float)
    P = np.asarray(run["motions"][body]["position_m"][S], float)
    steps = len(rings)//(8*nb)
    rings = rings[:steps*8*nb].reshape(steps, nb, 8, per, 3)
    # the block of this body: the one whose newest rings sit at the hubs
    for b in range(nb):
        c = ((rings[0, b].mean(axis=1)-P)@R)
        d = np.linalg.norm(c[:, None, :]-HUBS[None, :, :], axis=2)
        if d.min(axis=1).max() < 0.05:
            order = d.argmin(axis=1)
            body_rings = (rings[:, b]-P)@R
            return body_rings, order, np.arange(steps)*run["dt"]
    raise SystemExit("no ring block at the hubs of this body")


def section(surface, y):
    """Leading and trailing edge of a lifting surface at span station y (body frame), or None outside its span."""
    le, te = np.asarray(surface.leading_edge, float), np.asarray(surface.trailing_edge, float)
    ys = le[:, 1]
    if ys.max()-ys.min() < 1e-6 or not (min(ys) <= y <= max(ys)):
        return None
    k = np.argsort(ys)
    lo = np.array([np.interp(y, ys[k], le[k, j]) for j in range(3)])
    tr = np.array([np.interp(y, ys[k], te[k, j]) for j in range(3)])
    return lo, tr


def _closure_points(surface):
    """Three-quarter-chord points at the panel mid-spans of a lifting surface (body frame)."""
    le, te = np.asarray(surface.leading_edge, float), np.asarray(surface.trailing_edge, float)
    m_le, m_te = 0.5*(le[1:]+le[:-1]), 0.5*(te[1:]+te[:-1])
    return m_le+0.75*(m_te-m_le)


def _segment_distance(points, a, b):
    """Distance of every point from the nearest of the segments (a, b)."""
    ab = b-a
    t = np.clip(((points[:, None, :]-a[None])*ab[None]).sum(-1)/(ab*ab).sum(-1)[None], 0.0, 1.0)
    q = a[None]+t[..., None]*ab[None]
    return np.linalg.norm(points[:, None, :]-q, axis=-1).min(axis=1)


def centre_track(body_rings, order, discs):
    """Ring centres and mean radii of the discs ``discs`` against the distance behind the hub along body x."""
    out = []
    for slot in range(8):
        d = int(order[slot])
        if d not in discs:
            continue
        r = body_rings[:, slot]
        c = r.mean(axis=1)
        rad = np.linalg.norm(r-c[:, None, :], axis=2).mean(axis=1)
        out.append(dict(disc=d, x=c[:, 0]-HUBS[d, 0], dz=c[:, 2]-HUBS[d, 2], dy=c[:, 1]-HUBS[d, 1], r=rad/DISC_R))
    return out


def mean_track(tracks, grid):
    keep = lambda t, key: np.interp(grid, t["x"], t[key], left=np.nan, right=np.nan)   # noqa: E731
    ok = [t for t in tracks if np.all(np.diff(t["x"]) > 0)]
    if len(ok) != len(tracks):                      # x must increase with age near the kite; trim where it does not
        ok = []
        for t in tracks:
            stop = int(np.argmax(np.diff(t["x"]) <= 0)) if np.any(np.diff(t["x"]) <= 0) else len(t["x"])-1
            ok.append({k: (v[:stop+1] if isinstance(v, np.ndarray) else v) for k, v in t.items()})
    return {key: np.nanmean([keep(t, key) for t in ok], axis=0) for key in ("dz", "r", "dy")}


def fig18(out_dir, tube_dir, modelb_dir):
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    import airframe
    import compose_renders as cr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tube, modelb = wf.read(tube_dir), wf.read(modelb_dir)
    report = dict(script="rotor_wake_figs.py fig18", tube=dict(run=Path(tube_dir).name, step=tube["step"], dt=tube["dt"]),
                  model_b=dict(run=Path(modelb_dir).name, step=modelb["step"], dt=modelb["dt"]))
    # (a), (b): close-ups on shared scales
    _, ct = wr._rows(tube)
    _, cb = wr._rows(modelb)
    clim = float(np.percentile(np.abs(np.concatenate([ct[:int(1.2/tube["dt"])+1, [0, wf.WING_NODES_M600-1]].ravel(),
                                                      cb[:int(1.2/modelb["dt"])+1, [0, wf.WING_NODES_M600-1]].ravel()])), 90))
    lt, lb = wr.ring_limits(tube), wr.ring_limits(modelb)
    ring_clim = (min(lt[0], lb[0]), max(lt[1], lb[1]))
    wr.closeup(tube, out_dir/"fig18_tube.png", age_steps=int(round(1.2/tube["dt"])), clim=clim, ring_clim=ring_clim,
               ring_steps=int(round(0.6/tube["dt"])))
    wr.closeup(modelb, out_dir/"fig18_modelb.png", age_steps=int(round(1.2/modelb["dt"])), clim=clim, ring_clim=ring_clim,
               ring_steps=int(round(0.6/modelb["dt"])))
    # (c)-(e)
    wing, _, _, _ = airframe.parts()
    y_station = float(HUBS[[k for k in range(8) if ROTORS["rotors"][k]["label"] == "Sti"][0], 1])
    side = {}
    tracks = {}
    for label, run in (("tube", tube), ("model_b", modelb)):
        body_rings, order, age = unit_rings(run)
        side[label] = dict(body_rings=body_rings, order=order)
        tracks[label] = dict(top=centre_track(body_rings, order, TOP), bottom=centre_track(body_rings, order, BOTTOM))
    # split into two figures for the two-column layout (author 2026-10-26 17:14): the renders across the page and the
    # shape of the slipstream in one column
    ps.style()
    a_img, b_img = cr.trimmed(out_dir/"fig18_tube.png"), cr.trimmed(out_dir/"fig18_modelb.png")
    width = ps.WIDTH_DOUBLE_IN
    ra = a_img.size[1]/a_img.size[0]
    fig = plt.figure(figsize=(width, width*0.47*ra+0.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.06)
    for col, (img, letter) in enumerate(((a_img, "(a)"), (b_img, "(b)"))):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.asarray(img))
        ax.set_axis_off()
        ps.letter(ax, letter)
    bars = gs[0, 2].subgridspec(2, 1, hspace=0.35)
    bar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(-clim, clim), cmap="RdBu_r"), cax=fig.add_subplot(bars[0]))
    bar.set_label(r"$\Gamma$ (m$^2$/s)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*ring_clim), cmap=wr.ring_cmap()), cax=fig.add_subplot(bars[1]))
    bar2.set_label(r"$\Gamma_r$ (m$^2$/s)", fontsize=8)
    bar2.ax.tick_params(labelsize=7)
    ps.save(fig, out_dir/"fig_rotor_wake_renders")
    ps.style()
    ps.column_style()
    fig = plt.figure(figsize=(ps.WIDTH_SINGLE_IN, 4.7))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.62, 1, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    for label, colour, marker in (("tube", TUBE, "o"), ("model_b", MODEL_B, "s")):
        body_rings, order = side[label]["body_rings"], side[label]["order"]
        for slot in range(8):
            d = int(order[slot])
            if abs(HUBS[d, 1]-y_station) > 0.01:
                continue
            r = body_rings[:, slot]
            hi = r[np.arange(len(r)), r[:, :, 2].argmax(axis=1)]
            lo = r[np.arange(len(r)), r[:, :, 2].argmin(axis=1)]
            sel = (hi[:, 0] < 11.0)
            ax.plot(hi[sel, 0], hi[sel, 2], marker, color=colour, ms=1.6, mew=0, alpha=0.85)
            ax.plot(lo[sel, 0], lo[sel, 2], marker, color=colour, ms=1.6, mew=0, alpha=0.85)
    for surface in wing.wings[:2]:                      # main wing and horizontal tail
        s = section(surface, y_station)
        if s is not None:
            ax.plot([s[0][0], s[1][0]], [s[0][2], s[1][2]], color="0.2", lw=2.0, solid_capstyle="round")
    for d in range(8):
        if abs(HUBS[d, 1]-y_station) < 0.01:
            axis = np.asarray(ROTORS["rotors"][d]["axis"], float)
            n = np.array([-axis[2], axis[0]])            # in the x-z plane, normal to the axis
            n /= np.linalg.norm(n)
            ends = np.array([HUBS[d, [0, 2]]-DISC_R*n, HUBS[d, [0, 2]]+DISC_R*n])
            ax.plot(ends[:, 0], ends[:, 1], color="0.45", lw=1.4)
    ax.plot([], [], "o", color=TUBE, ms=3, label=r"tube T3")
    ax.plot([], [], "s", color=MODEL_B, ms=3, label="model D")
    ax.set_xlim(-3.0, 11.0)
    ax.set_ylim(-3.2, 2.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$z$ (m)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
    ps.letter(ax, "(a)")
    grid_x = np.linspace(0.0, 12.0, 241)
    means = {lab: {row: mean_track(tracks[lab][row], grid_x) for row in ("top", "bottom")} for lab in tracks}
    axd = fig.add_subplot(gs[1])
    axe = fig.add_subplot(gs[2], sharex=axd)
    for lab, colour in (("tube", TUBE), ("model_b", MODEL_B)):
        for row, ls in (("top", "-"), ("bottom", ps.DASH)):
            m = means[lab][row]
            axd.plot(grid_x, m["dz"], color=colour, ls=ls)
            axe.plot(grid_x, m["r"], color=colour, ls=ls)
    axd.plot([], [], color="0.3", ls="-", label="upper rotors")
    axd.plot([], [], color="0.3", ls=ps.DASH, label="lower rotors")
    axd.legend(loc="upper left")
    for a, lab, letter in ((axd, r"$\Delta z_c$ (m)", "(b)"), (axe, r"$r/R$", "(c)")):
        a.set_ylabel(lab)
        a.set_xlim(0, 12)
        ps.grid(a)
        ps.letter(a, letter)
    axe.set_xlabel(r"$x - x_{\mathrm{hub}}$ (m)")
    ps.save(fig, out_dir/"fig_rotor_wake_shape")
    # numbers at the trailing edge of the main wing and at the horizontal tail (body x of the sections at y_station)
    te_wing = section(wing.wings[0], y_station)[1][0]
    tail = section(wing.wings[1], y_station)
    stations = dict(wing_trailing_edge=float(te_wing), horizontal_tail_leading_edge=float(tail[0][0]))
    report["stations_body_x_m"] = stations
    report["ring_centres"] = {}
    for lab in means:
        report["ring_centres"][lab] = {}
        for row in ("top", "bottom"):
            hub_x = float(np.mean(HUBS[TOP if row == "top" else BOTTOM, 0]))
            vals = {}
            for name, xs in stations.items():
                g = xs-hub_x
                vals[name] = dict(x_behind_hub_m=g, dz_m=float(np.interp(g, grid_x, means[lab][row]["dz"])),
                                  r_over_R=float(np.interp(g, grid_x, means[lab][row]["r"])),
                                  dy_m=float(np.interp(g, grid_x, means[lab][row]["dy"])))
            report["ring_centres"][lab][row] = vals
    report["colour_limits"] = dict(filaments=clim, rings=list(ring_clim))
    # distance of the closure points of wing and tails from the nearest ring filament (rings up to 1 s old), and the
    # extremes of the mean ring-centre height and radius of each row
    closure = {name: _closure_points(wing.wings[k]) for k, name in enumerate(("main wing", "horizontal tail", "vertical tail"))}
    report["wall_distance_m"] = {}
    report["track_extremes"] = {}
    for lab, run in (("tube", tube), ("model_b", modelb)):
        body_rings, order, age = unit_rings(run)
        r = body_rings[age <= 1.0].reshape(-1, body_rings.shape[2], 3)
        a_, b_ = r.reshape(-1, 3), np.roll(r, -1, axis=1).reshape(-1, 3)
        rows = {}
        for name, pts in list(closure.items())+[("all", np.vstack(list(closure.values())))]:
            d = _segment_distance(pts, a_, b_)
            rows[name] = dict(min=float(d.min()), median=float(np.median(d)), below_0p2=int((d < 0.2).sum()), points=len(d))
        report["wall_distance_m"][lab] = rows
        report["track_extremes"][lab] = {row: dict(dz_max=float(np.nanmax(means[lab][row]["dz"])), dz_max_x=float(grid_x[np.nanargmax(means[lab][row]["dz"])]),
                                                   r_min=float(np.nanmin(means[lab][row]["r"])), r_min_x=float(grid_x[np.nanargmin(means[lab][row]["r"])]),
                                                   r_max=float(np.nanmax(means[lab][row]["r"])), r_max_x=float(grid_x[np.nanargmax(means[lab][row]["r"])]),
                                                   slope_dz_per_m=float(np.polyfit(grid_x[:200], means[lab][row]["dz"][:200], 1)[0]))
                                             for row in ("top", "bottom")}
    (out_dir/"fig18_report.json").write_text(json.dumps(report, indent=1, default=float)+"\n", encoding="utf-8")
    return report


# ----------------------------------------------------------------------------------------------------------------------
# FIG. 20
# ----------------------------------------------------------------------------------------------------------------------


def body_columns(run):
    first = run["first"]
    per_body = first//len(run["motions"])
    return first, per_body


def a_network(run):
    """Edges of the lifting-surface network of unit A (bound, trailing and shed filaments; no rings)."""
    nodes, edges, gamma, rf = run["nodes"], run["edges"], run["gamma"], run["ring_first"]
    first, per_body = body_columns(run)
    wing = np.all(edges < rf, axis=1)
    col = edges % first
    mine = wing & np.all(col < per_body, axis=1)
    return nodes, edges[mine], gamma[mine]


def b_wing_points(run):
    first, per_body = body_columns(run)
    return run["nodes"][per_body:per_body+wf.WING_NODES_M600]


def loop_plane(run, body="B"):
    centre, radius, diameter = wf.geometry(run, body)
    ground = np.asarray(run["settings"]["ground_point_b_m"], float) if body == "B" else np.zeros(3)
    n = centre-ground
    n /= np.linalg.norm(n)
    e1 = np.cross([0., 0., 1.], n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return centre, n, e1, e2, radius, diameter


def plane_flux(run, half_m=70.0, count=141, sigma=3.0):
    """Vorticity of the wake of A through the plane normal to the body x axis of B through its body origin (the plane
    that B flies through), in the body y-z axes of B: flux of the filaments of A crossing it, spread with a Gaussian."""
    nodes, edges, gamma = a_network(run)
    S = run["step"]
    R = np.asarray(run["motions"]["B"]["rotation"][S], float)
    P = np.asarray(run["motions"]["B"]["position_m"][S], float)
    local = (nodes-P)@R
    pts, flux = wf.crossings(local, edges, gamma, 0.0)
    v = np.linspace(-half_m, half_m, count)
    yy, zz = np.meshgrid(v, v)
    field = wf.flux_field(pts, flux, yy, zz, sigma)
    wing = (b_wing_points(run)-P)@R
    centre, radius, diameter = wf.geometry(run, "B")
    return dict(y=yy, z=zz, omega=field*diameter/run["wind"], wing=wing[:, 1:], crossings=int(np.sum(np.abs(pts).max(axis=1) < half_m)),
                diameter=diameter)


def induced_at_b(run):
    from openfawe.aero.vortex import CoreModel, segments_velocity
    nodes, edges, gamma = a_network(run)
    pts = b_wing_points(run)
    u = segments_velocity(pts, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=float(run["settings"]["core_radius_m"]),
                          backend="numba", singularity="zero")
    S = run["step"]
    R = np.asarray(run["motions"]["B"]["rotation"][S], float)
    ub = u@R                                                # body components of B
    return ub.mean(axis=0)


def fig20(out_dir, pair_dirs, cache, loss_only=()):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import c3_figures as cf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(script="rotor_wake_figs.py fig20", runs=[])
    series = []
    for run_dir in pair_dirs:
        run_dir = Path(run_dir)
        data = json.loads((run_dir/"checks.json").read_text(encoding="utf-8"))
        s = data["settings"]
        n, laps = int(s["steps_per_lap"]), int(s["laps"])
        angle = cf.loop_angle_deg(s, cache)
        d = {k: data["per_step"][k] for k in ("A", "B")}
        steps = np.asarray(d["A"]["step"])
        keep = (steps > (laps-1)*n) & (steps <= laps*n)
        la, lb = np.asarray(d["A"]["force_body_N"], float)[keep, 2], np.asarray(d["B"]["force_body_N"], float)[keep, 2]
        vb = np.asarray(d["B"]["airspeed_m_s"], float)[keep]
        loss = lb/la-1.
        stored = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("state_*.npz"))
        stored = [k for k in stored if (laps-1)*n < k <= laps*n]
        frames = []
        base = wf.read(run_dir, stored[-1])
        import wake_analysis as wa
        for k in stored:
            run = dict(base)
            run["nodes"], run["edges"], run["gamma"], run["first"], run["ring_first"] = wa.load_state(run_dir/f"state_{k:05d}.npz")
            run["step"] = k
            ub = induced_at_b(run)
            i = int(np.flatnonzero(steps[keep] == k)[0])
            frames.append(dict(step=k, angle_deg=float(angle[k]), u_body=ub.tolist(), d_alpha_rad=float(ub[2]/vb[i]),
                               d_airspeed_rel=float(ub[0]/vb[i]), lift_change=float(loss[i])))
        worst = min(frames, key=lambda f: f["lift_change"])
        run = dict(base)
        run["nodes"], run["edges"], run["gamma"], run["first"], run["ring_first"] = wa.load_state(run_dir/f"state_{worst['step']:05d}.npz")
        run["step"] = worst["step"]
        plane = plane_flux(run)
        series.append(dict(run=run_dir.name, spacing=float(s["spacing_diameters"]), offset=float(s["offset_diameters"]),
                           angle=angle[steps[keep]], loss=loss, frames=frames, plane=plane, worst=worst))
        report["runs"].append(dict(run=run_dir.name, spacing_D=float(s["spacing_diameters"]), offset_D=float(s["offset_diameters"]),
                                   frames=frames, plane_step=worst["step"], plane_angle_deg=worst["angle_deg"],
                                   plane_crossings=plane["crossings"],
                                   lift_change_min_percent=100*float(loss.min()), lift_change_mean_percent=100*float(lb.mean()/la.mean()-1.),
                                   corr_lift_change_vs_d_alpha=float(np.corrcoef([f["lift_change"] for f in frames],
                                                                                 [f["d_alpha_rad"] for f in frames])[0, 1])))
    ps.style()
    fig = plt.figure(figsize=(ps.WIDTH_DOUBLE_IN, 6.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1.0, 0.72], hspace=0.38, wspace=0.42)
    vmax = max(np.percentile(np.abs(x["plane"]["omega"]), 99.7) for x in series)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
    for c, x in enumerate(series[:2]):
        ax = fig.add_subplot(gs[0, c])
        p = x["plane"]
        im = ax.pcolormesh(p["y"], p["z"], p["omega"], cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
        ax.plot(p["wing"][:, 0], p["wing"][:, 1], color="k", lw=2.0, solid_capstyle="round")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$y_B$ (m)")
        if c == 0:
            ax.set_ylabel(r"$z_B$ (m)")
        ps.letter(ax, f"({'ab'[c]})")
    bar = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    bar.set_label(r"$\omega_x D/U_\infty$")
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for run_dir in loss_only:                       # per-step loss along the loop without stored networks
        data = json.loads((Path(run_dir)/"checks.json").read_text(encoding="utf-8"))
        s = data["settings"]
        n, laps = int(s["steps_per_lap"]), int(s["laps"])
        angle = cf.loop_angle_deg(s, cache)
        steps = np.asarray(data["per_step"]["A"]["step"])
        keep = (steps > (laps-1)*n) & (steps <= laps*n)
        la = np.asarray(data["per_step"]["A"]["force_body_N"], float)[keep, 2]
        lb = np.asarray(data["per_step"]["B"]["force_body_N"], float)[keep, 2]
        a = angle[steps[keep]]
        o = np.argsort(a)
        axc.plot(a[o], 100*(lb/la-1.)[o], color="0.35", lw=1.0, label=f"{float(s['spacing_diameters']):g} $D$")
        report.setdefault("loss_only", []).append(dict(run=Path(run_dir).name, lift_change_min_percent=100*float((lb/la-1.).min()),
                                                       min_angle_deg=float(a[np.argmin(lb/la)])))
    for x, colour in zip(series, colours):
        lab = f"{x['spacing']:g} $D$" + (f", offset {x['offset']:g} $D$" if x["offset"] else "")
        o = np.argsort(x["angle"])
        axc.plot(x["angle"][o], 100*x["loss"][o], color=colour, label=lab)
        fr = sorted(x["frames"], key=lambda f: f["angle_deg"])
        axd.plot([f["angle_deg"] for f in fr], [np.degrees(f["d_alpha_rad"]) for f in fr], "o-", color=colour, ms=3, lw=0.9)
    axc.set_ylabel(r"$L_B/L_A - 1$ (%)")
    axd.set_ylabel(r"$\Delta\alpha_B$ (deg)")
    axc.legend(loc="lower right", fontsize=8)
    for a, letter in ((axc, "(c)"), (axd, "(d)")):
        a.set_xlabel("loop angle (deg)")
        a.set_xlim(0, 360)
        a.set_xticks(range(0, 361, 90))
        ps.grid(a)
        ps.letter(a, letter)
    ps.save(fig, out_dir/"fig_wake_at_b")
    (out_dir/"fig20_report.json").write_text(json.dumps(report, indent=1, default=float)+"\n", encoding="utf-8")
    return report


def planes(out_json, run_dirs, stations=(0.5, 1.0, 2.0), half_d=0.75, count=61, sigma=3.0):
    """Wake of a single unit with and without rotors in cross-flow planes downstream of the path centre (the planes of
    Sec. III D): circulation and centroid radius of the lifting-surface filaments crossing each plane, ring crossings,
    and the velocity deficit induced by the whole network (wake_analysis.plane_at, tree code, the run's core)."""
    report = dict(script="rotor_wake_figs.py planes", stations_D=list(stations), half_D=half_d, count=count, sigma_m=sigma, runs={})
    for run_dir in run_dirs:
        run = wf.read(run_dir)
        rows, centre, radius, diameter = wf.plane_vorticity(run, stations, half_d, count, sigma, True)
        report["runs"][Path(run_dir).name] = dict(step=run["step"], dt=run["dt"], diameter_m=diameter, planes=[
            {k: (v if not isinstance(v, np.ndarray) else None) for k, v in p.items()
             if k not in ("y", "z", "omega_wing", "omega_rings", "deficit", "in_plane", "profile", "radius_bins_m")} for p in rows])
    Path(out_json).write_text(json.dumps(report, indent=1, default=float)+"\n", encoding="utf-8")
    return report


def main(argv):
    p = argparse.ArgumentParser(prog="rotor_wake_figs.py")
    p.add_argument("which", choices=("fig18", "fig20", "planes"))
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--cache", default=None)
    p.add_argument("--loss-only", nargs="*", default=[])
    a = p.parse_args(argv)
    if a.which == "fig18":
        r = fig18(a.out, a.runs[0], a.runs[1])
    elif a.which == "planes":
        r = planes(a.out, a.runs)
    else:
        r = fig20(a.out, a.runs, a.cache, a.loss_only)
    print(json.dumps(r, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main(sys.argv[1:])
