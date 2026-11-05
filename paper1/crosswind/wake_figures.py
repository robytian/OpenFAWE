"""Wake structure and vorticity of the OF-0009 crosswind runs (M600 on its loop, with and without rotors): figures and
numbers for Secs. III D to III F of the paper. Reads a run directory (``history.json`` and ``state_<step>.npz`` of
``c1_loop.py`` or ``c3_pair.py``); nothing is recomputed except velocities induced by the stored network.

Quantities (definitions for the captions):
  tips        the trailing nodes of the two wing tips in every wake row (row j released j steps ago): their distance r from
              the axis of the wake, the horizontal line through the path centre along the wind, and their height, against
              the downstream distance x - x_c of the row; r/R with R the loop radius. The tip with the larger r in the
              newest row is the outer tip. The circulation of the tip trailing filaments is taken from the network.
  crossing    the streamwise vorticity on a cross-flow plane x = x0 represented by the filaments that cross it: every segment
              (a, b) with (x_a - x0)(x_b - x0) < 0 carries the flux Gamma sign(x_b - x_a) through the plane at its
              intersection point; the field omega_x(y, z) is that flux spread with a Gaussian of standard deviation sigma
              (default 3 m), so that its integral over the plane is the net circulation of the crossing filaments. Totals
              of the positive and the negative flux and their centroids are exact (no smoothing). Rotor rings are counted
              separately.
  deficit     1 - u/U_inf of the velocity induced by the whole network (the run's core, Lamb-Oseen) plus the wind, on the
              same planes (CPU tree code).
  wake3d      PyVista rendering: trailing filaments of the lifting surfaces coloured by the sign and magnitude of their
              streamwise vorticity (circulation times the x component of the unit tangent), shed filaments light grey, rotor
              rings green; the complete airframe (wing, tails, pylons, fuselage, rotor discs, bridle and a straight tether to the
              ground point; ``airframe.py``) at the pose of the stored step. Display only.

Usage: wake_figures.py RUN_DIR OUT_DIR [--state N] [--label L] [--planes 0.5,1,2,3] [--sigma 3] [--count 121]
                       [--half 0.75] [--no-3d] [--body A]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(TASK))
import pof_style as ps  # noqa: E402
import wake_analysis as wa  # noqa: E402

WING_NODES_M600 = 35          # nodes of the main wing of the M600 assembly (34 panels); htail 9, vtail 8 follow


# ----------------------------------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------------------------------


def read(run_dir, step=None):
    run_dir = Path(run_dir)
    history = json.loads((run_dir/"history.json").read_text(encoding="utf-8"))
    states = sorted(run_dir.glob("state_*.npz"))
    if not states:
        raise SystemExit(f"{run_dir}: no stored wake state")
    state = states[-1] if step is None else run_dir/f"state_{int(step):05d}.npz"
    nodes, edges, gamma, first, ring_first = wa.load_state(state)
    s = history["settings"]
    if "motions" in history:                                   # c3_pair
        motions = {k: {kk: np.asarray(vv) for kk, vv in m.items()} for k, m in history["motions"].items()}
        case = s["case"]
    else:                                                       # c1_loop
        motions = {"A": {k: np.asarray(v) for k, v in history["motion"].items()}}
        case = s.get("low_case", s.get("case"))
    stored = int(Path(state).stem.split("_")[1])
    return dict(history=history, settings=s, nodes=nodes, edges=edges, gamma=gamma, first=first, ring_first=ring_first,
                step=stored, motions=motions, case=case, dt=float(s["dt_s"]), wind=float(s["wind_m_s"]),
                steps_per_lap=int(s["steps_per_lap"]))


def geometry(run, body="A"):
    """Path centre (mean position over the last complete lap before the stored step), loop radius and diameter."""
    n, step = run["steps_per_lap"], run["step"]
    position = run["motions"][body]["position_m"]
    last = position[max(0, step-n+1):step+1]
    centre = last.mean(axis=0)
    loop = run["settings"].get("loop", {})
    radius = float(loop.get("radius_m", np.mean(np.linalg.norm(last-centre, axis=1))))
    return centre, radius, 2.*radius


# ----------------------------------------------------------------------------------------------------------------------
# tip vortices
# ----------------------------------------------------------------------------------------------------------------------


def tips(run, body_index=0, wing_nodes=WING_NODES_M600):
    nodes, edges, gamma = wa.wing_part(run["nodes"], run["edges"], run["gamma"], run["ring_first"])
    first = run["first"]
    rows, circulation = wa.wake_rows(nodes, edges, gamma, first)
    per_body = first//len(run["motions"])
    offset = body_index*per_body
    centre, radius, diameter = geometry(run, list(run["motions"])[body_index])
    out = {}
    for name, column in (("left", offset), ("right", offset+wing_nodes-1)):
        p = rows[1:, column]
        r = np.hypot(p[:, 1]-centre[1], p[:, 2]-centre[2])
        out[name] = dict(x=p[:, 0]-centre[0], y=p[:, 1]-centre[1], z=p[:, 2]-centre[2], r=r, gamma=circulation[1:, column],
                         age_s=np.arange(1, len(p)+1)*run["dt"])
    outer, inner = ("left", "right") if out["left"]["r"][0] > out["right"]["r"][0] else ("right", "left")
    return dict(outer=out[outer], inner=out[inner], centre=centre, radius=radius, diameter=diameter)


def binned(x, values, edges):
    index = np.digitize(x, edges)-1
    mean = np.full(len(edges)-1, np.nan)
    for k in range(len(edges)-1):
        sel = index == k
        if np.any(sel):
            mean[k] = values[sel].mean()
    return 0.5*(edges[:-1]+edges[1:]), mean


def tip_summary(t, stations):
    D, R = t["diameter"], t["radius"]
    rows = []
    for name in ("outer", "inner"):
        d = t[name]
        xd = d["x"]/D
        for s in stations:
            sel = np.abs(xd-s) < 0.05
            if np.any(sel):
                rows.append(dict(tip=name, station_D=s, r_over_R=float(d["r"][sel].mean()/R), z_over_D=float(d["z"][sel].mean()/D),
                                 gamma_m2_s=float(np.abs(d["gamma"][sel]).mean()), samples=int(sel.sum())))
    return rows


def figure_tips(t, path):
    import matplotlib.pyplot as plt
    ps.style()
    D, R = t["diameter"], t["radius"]
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_DOUBLE_IN, 2.35), constrained_layout=True)
    edges = np.linspace(-0.3, max(t["outer"]["x"].max(), t["inner"]["x"].max())/D+0.05, 60)
    for name in ("outer", "inner"):
        d = t[name]
        xd = d["x"]/D
        axes[0].plot(xd, d["r"]/R, ".", ms=0.8, color=ps.COL[name], alpha=0.25, rasterized=True)
        xc, rm = binned(xd, d["r"]/R, edges)
        axes[0].plot(xc, rm, color=ps.COL[name], lw=1.3, label=f"{name} tip")
        axes[1].plot(xd, d["z"]/D, ".", ms=0.8, color=ps.COL[name], alpha=0.25, rasterized=True)
        xc, zm = binned(xd, d["z"]/D, edges)
        axes[1].plot(xc, zm, color=ps.COL[name], lw=1.3)
        xc, gm = binned(xd, np.abs(d["gamma"]), edges)
        axes[2].plot(xc, gm, color=ps.COL[name], lw=1.3)
    axes[0].set_xlabel(r"$(x-x_c)/D$"); axes[0].set_ylabel(r"$r/R$")
    axes[1].set_xlabel(r"$(x-x_c)/D$"); axes[1].set_ylabel(r"$(z-z_c)/D$")
    axes[2].set_xlabel(r"$(x-x_c)/D$"); axes[2].set_ylabel(r"$|\Gamma_{\mathrm{tip}}|$ (m$^2$ s$^{-1}$)")
    axes[0].legend(loc="best")
    for ax, k in zip(axes, "abc"):
        ps.grid(ax); ps.letter(ax, f"({k})")
    ps.save(fig, path)


def tip_displacement(run, wing_nodes=WING_NODES_M600, body="A"):
    """Displacement of the wing-tip trailing nodes from pure convection with the wind, against their age.

    Row j of the stored state was the trailing-edge row j steps before the stored step. Its tip node was released at the
    trailing-edge tip of the wing at that pose (``rotation @ te + position`` with the wing of the run's case) and, carried
    by the wind alone, would be displaced by U_inf * age along x. The difference is the motion induced by the wake itself
    (and by the kite), decomposed along the outward radial direction of the loop at the release point (normal to the loop
    axis, the line from the ground point to the path centre), along the loop axis and along the vertical.
    """
    from openfawe.aero.aircraft import load_aircraft
    nodes, edges, gamma = wa.wing_part(run["nodes"], run["edges"], run["gamma"], run["ring_first"])
    rows, circulation = wa.wake_rows(nodes, edges, gamma, run["first"])
    definition = load_aircraft(REPO/run["case"])
    surface = definition.wing.wings[0] if hasattr(definition.wing, "wings") else definition.wing
    te = surface.trailing_edge
    motion = run["motions"][body]
    S, dt, U = run["step"], run["dt"], run["wind"]
    centre, radius, diameter = geometry(run, body)
    axis = centre-np.zeros(3)
    axis /= np.linalg.norm(axis)
    out = {}
    count = min(rows.shape[0]-1, S)
    for name, column in (("left", 0), ("right", wing_nodes-1)):
        ages, dr, dn, dz, dx, gam = [], [], [], [], [], []
        for j in range(1, count+1):
            k = S-j
            release = motion["rotation"][k]@te[column]+motion["position_m"][k]
            delta = rows[j, column]-release-np.array([U*j*dt, 0., 0.])
            rel = release-centre
            radial = rel-np.dot(rel, axis)*axis
            radial /= np.linalg.norm(radial)
            ages.append(j*dt); dr.append(delta@radial); dn.append(delta@axis); dz.append(delta[2]); dx.append(delta[0])
            gam.append(circulation[j, column])
        out[name] = {k: np.asarray(v) for k, v in dict(age_s=ages, dr=dr, dn=dn, dz=dz, dx=dx, gamma=gam).items()}
    # the tip farther from the loop axis at the newest row is the outer tip
    rel = [motion["rotation"][S-1]@te[c]+motion["position_m"][S-1]-centre for c in (0, wing_nodes-1)]
    dist = [np.linalg.norm(r-np.dot(r, axis)*axis) for r in rel]
    outer, inner = ("left", "right") if dist[0] > dist[1] else ("right", "left")
    period = run["steps_per_lap"]*dt
    return dict(outer=out[outer], inner=out[inner], period_s=period, radius=radius, diameter=diameter)


def figure_tip_displacement(t, path):
    import matplotlib.pyplot as plt
    ps.style()
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_DOUBLE_IN, 2.35), constrained_layout=True)
    for name in ("outer", "inner"):
        d = t[name]
        laps = d["age_s"]/t["period_s"]
        edges = np.linspace(0., laps.max(), 49)
        for ax, key, scale in ((axes[0], "dr", t["radius"]), (axes[1], "dz", t["diameter"]), (axes[2], "dx", t["diameter"])):
            ax.plot(laps, d[key]/scale, ".", ms=0.7, color=ps.COL[name], alpha=0.2, rasterized=True)
            xc, m = binned(laps, d[key]/scale, edges)
            ax.plot(xc, m, color=ps.COL[name], lw=1.3, label=f"{name} tip")
    axes[0].set_ylabel(r"radial displacement $\Delta r/R$")
    axes[1].set_ylabel(r"vertical displacement $\Delta z/D$")
    axes[2].set_ylabel(r"streamwise displacement $\Delta x/D$")
    for ax, k in zip(axes, "abc"):
        ax.set_xlabel("age (loops)")
        ps.grid(ax); ps.letter(ax, f"({k})")
    axes[0].legend(loc="best")
    ps.save(fig, path)


def meridional(run, vis_core, spacing=2.0, x_range=(-0.5, 2.6), z_half=0.75, body="A"):
    """Velocity and out-of-plane vorticity on the vertical plane through the path centre along the wind (y = y_c).

    The velocity is induced by the whole network with a Lamb-Oseen core ``vis_core`` on every segment (a display
    regularisation that smooths the field at the scale of the filament spacing; away from the filaments it equals the
    field of the run's core) plus the wind; omega_y by central differences of it.
    """
    from openfawe.aero import wake_probe as wp
    from openfawe.aero.vortex import CoreModel, TreeSettings
    centre, radius, diameter = geometry(run, body)
    x0, x1 = x_range
    half_x = 0.5*(x1-x0)*diameter
    cx = centre[0]+0.5*(x0+x1)*diameter
    nx, nz = int(2*half_x/spacing)+1, int(2*z_half*diameter/spacing)+1
    plane = wp.symmetry_plane(centre[1], cx, half_x, centre[2], z_half*diameter, nx, nz)
    fields = wp.plane_fields(plane, run["nodes"], run["edges"], run["gamma"], model=CoreModel.LAMB_OSEEN, core_radius=float(vis_core),
                             freestream=np.array([run["wind"], 0., 0.]), backend="tree", tree=TreeSettings(theta=0.3, order=2, leaf_size=32))
    pts = fields["points"]
    path = run["motions"][body]["position_m"][max(0, run["step"]-run["steps_per_lap"]):run["step"]+1]
    return dict(x=(pts[..., 0]-centre[0])/diameter, z=(pts[..., 2]-centre[2])/diameter, omega_y=-fields["normal_vorticity"],
                deficit=1.-fields["axial"]/run["wind"], w=fields["velocity"][..., 2], vis_core=float(vis_core),
                path_xz=np.stack([(path[:, 0]-centre[0])/diameter, (path[:, 2]-centre[2])/diameter], axis=1),
                kite_xz=((path[-1, 0]-centre[0])/diameter, (path[-1, 2]-centre[2])/diameter), diameter=diameter, wind=run["wind"])


def track_pairs(m, window_d=0.2):
    """The crossings of the helical wake through the meridional plane: per crossing (top or bottom of a turn), the
    circulation of the positive and the negative omega_y (integral of omega_y dA over a window of +-window_d D about the
    crossing), their centroids and separation. Crossings are the local maxima along x of the z-integral of |omega_y|,
    separately above and below the path centre."""
    x, z, w = m["x"], m["z"], m["omega_y"]
    D = m["diameter"]
    dx = (x[0, 1]-x[0, 0])*D
    dz = (z[1, 0]-z[0, 0])*D
    area = abs(dx*dz)
    out = []
    for half, sel in (("top", z[:, 0] > 0), ("bottom", z[:, 0] <= 0)):
        profile = np.abs(w[sel]).sum(axis=0)
        if profile.max() <= 0:
            continue
        xs = x[0]
        peaks = [k for k in range(1, len(xs)-1) if profile[k] >= profile[k-1] and profile[k] > profile[k+1] and profile[k] > 0.15*profile.max()]
        merged = []
        for k in peaks:                                   # keep the strongest peak within one window
            if merged and xs[k]-xs[merged[-1]] < 2*window_d:
                if profile[k] > profile[merged[-1]]:
                    merged[-1] = k
                continue
            merged.append(k)
        for k in merged:
            box = sel[:, None] & (np.abs(x-xs[k]) <= window_d)
            pos, neg = box & (w > 0), box & (w < 0)
            entry = dict(half=half, x_D=float(xs[k]))
            for name, mask in (("positive", pos), ("negative", neg)):
                if np.any(mask):
                    weight = np.abs(w[mask])
                    entry[f"{name}_circulation_m2_s"] = float(w[mask].sum()*area)
                    entry[f"{name}_centroid_xz_D"] = [float((weight*x[mask]).sum()/weight.sum()), float((weight*z[mask]).sum()/weight.sum())]
            if "positive_centroid_xz_D" in entry and "negative_centroid_xz_D" in entry:
                a, b = np.array(entry["positive_centroid_xz_D"]), np.array(entry["negative_centroid_xz_D"])
                entry["separation_D"] = float(np.linalg.norm(a-b))
                entry["pair_centroid_xz_D"] = (0.5*(a+b)).tolist()
            out.append(entry)
    return sorted(out, key=lambda e: (e["half"], e["x_D"]))


def figure_meridional(panels, path, labels=None, clip=99.0):
    """Per run two full-width panels: omega_y D/U (saturated at the ``clip`` percentile) and 1 - u/U with the in-plane
    induced velocity (u - U, w) as arrows; the last loop projected on the plane (dashed)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    ps.style()
    n = len(panels)
    p0 = panels[0]
    aspect = (p0["z"].max()-p0["z"].min())/(p0["x"].max()-p0["x"].min())
    fig, axes = plt.subplots(2*n, 1, figsize=(ps.WIDTH_DOUBLE_IN, 2*n*(ps.WIDTH_DOUBLE_IN*0.82*aspect)+0.5), squeeze=False,
                             constrained_layout=True)
    wmax = max(np.percentile(np.abs(p["omega_y"]*p["diameter"]/p["wind"]), clip) for p in panels)
    letters = iter("abcdefgh")
    for r, p in enumerate(panels):
        ax = axes[2*r, 0]
        im0 = ax.pcolormesh(p["x"], p["z"], p["omega_y"]*p["diameter"]/p["wind"], cmap="RdBu_r",
                            norm=TwoSlopeNorm(vmin=-wmax, vcenter=0., vmax=wmax), shading="auto", rasterized=True)
        ax.plot(p["path_xz"][:, 0], p["path_xz"][:, 1], color="0.2", lw=0.7, ls=ps.DASH)
        ax.set_aspect("equal"); ax.set_ylabel(r"$(z-z_c)/D$"); ps.letter(ax, f"({next(letters)})")
        fig.colorbar(im0, ax=ax, shrink=0.95, pad=0.01, label=r"$\omega_y D/U_\infty$")
        ax = axes[2*r+1, 0]
        im1 = ax.pcolormesh(p["x"], p["z"], p["deficit"], cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.1, vcenter=0., vmax=0.3),
                            shading="auto", rasterized=True)
        sx, sz = max(1, p["x"].shape[1]//60), max(1, p["x"].shape[0]//18)
        uu = -p["deficit"]*p["wind"]
        ww = p["w"]
        cap = 0.1*p["wind"]
        mag = np.hypot(uu, ww)
        f = np.minimum(1., cap/np.maximum(mag, 1e-12))
        ax.quiver(p["x"][::sz, ::sx], p["z"][::sz, ::sx], (uu*f)[::sz, ::sx], (ww*f)[::sz, ::sx], color="0.15", scale=40*cap,
                  width=0.0018, headwidth=3.5, alpha=0.8)
        ax.plot(p["path_xz"][:, 0], p["path_xz"][:, 1], color="0.2", lw=0.7, ls=ps.DASH)
        ax.set_aspect("equal"); ax.set_ylabel(r"$(z-z_c)/D$"); ps.letter(ax, f"({next(letters)})")
        fig.colorbar(im1, ax=ax, shrink=0.95, pad=0.01, label=r"$1-u/U_\infty$")
    axes[-1, 0].set_xlabel(r"$(x-x_c)/D$")
    ps.save(fig, path)


# ----------------------------------------------------------------------------------------------------------------------
# vorticity flux through cross-flow planes
# ----------------------------------------------------------------------------------------------------------------------


def crossings(nodes, edges, gamma, x0):
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    da, db = a[:, 0]-x0, b[:, 0]-x0
    sel = (da*db) < 0.
    if not np.any(sel):
        return np.empty((0, 2)), np.empty(0)
    t = da[sel]/(da[sel]-db[sel])
    point = a[sel]+t[:, None]*(b[sel]-a[sel])
    flux = gamma[sel]*np.sign(db[sel]-da[sel])
    return point[:, 1:], flux


def flux_field(points, flux, yy, zz, sigma):
    field = np.zeros_like(yy)
    norm = 1./(2.*np.pi*sigma*sigma)
    for (y, z), f in zip(points, flux):
        field += f*norm*np.exp(-((yy-y)**2+(zz-z)**2)/(2.*sigma*sigma))
    return field


def plane_vorticity(run, stations, half_d, count, sigma, with_deficit=True):
    centre, radius, diameter = geometry(run)
    nodes, edges, gamma, rf = run["nodes"], run["edges"], run["gamma"], run["ring_first"]
    wing_edges = np.all(edges < rf, axis=1)
    out = []
    for s in stations:
        x0 = centre[0]+s*diameter
        half = half_d*diameter
        yv = np.linspace(-half, half, count)
        yy, zz = np.meshgrid(centre[1]+yv, centre[2]+yv)
        pw, fw = crossings(nodes, edges[wing_edges], gamma[wing_edges], x0)
        pr, fr = crossings(nodes, edges[~wing_edges], gamma[~wing_edges], x0) if np.any(~wing_edges) else (np.empty((0, 2)), np.empty(0))
        entry = dict(station_D=s, x_m=float(x0), y=yy-centre[1], z=zz-centre[2],
                     omega_wing=flux_field(pw, fw, yy, zz, sigma), omega_rings=flux_field(pr, fr, yy, zz, sigma) if len(fr) else None)
        for label, (pts, fl) in (("wing", (pw, fw)), ("rings", (pr, fr))):
            pos, neg = fl > 0, fl < 0
            entry[f"{label}_positive_m2_s"] = float(fl[pos].sum()) if np.any(pos) else 0.
            entry[f"{label}_negative_m2_s"] = float(fl[neg].sum()) if np.any(neg) else 0.
            entry[f"{label}_crossings"] = int(len(fl))
            for sign, sel in (("positive", pos), ("negative", neg)):
                if np.any(sel):
                    w = np.abs(fl[sel])
                    c = (w[:, None]*pts[sel]).sum(axis=0)/w.sum()
                    entry[f"{label}_{sign}_centroid_yz_m"] = (c-centre[1:]).tolist()
                    entry[f"{label}_{sign}_centroid_r_over_R"] = float(np.hypot(*(c-centre[1:]))/radius)
        if with_deficit:
            p = wa.plane_at(x0, centre[1:], half, count, nodes, edges, gamma, run["wind"], run["settings"]["core_radius_m"], "tree")
            entry["deficit"] = p["deficit"]
            entry["in_plane"] = p["in_plane"]
            entry["deficit_max"] = p["deficit_max"]
            entry["profile"] = p["profile"]
            entry["radius_bins_m"] = p["radius_bins_m"]
            entry["momentum_deficit_m2_s"] = p["momentum_deficit_m2_s"]
        out.append(entry)
    return out, centre, radius, diameter


def figure_planes(planes, diameter, path, rows_label=None):
    """omega_x (colour) with deficit contours; one row per run when ``planes`` is a list of lists."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    ps.style()
    runs = planes if isinstance(planes[0], list) else [planes]
    ncol = len(runs[0])
    fig, axes = plt.subplots(len(runs), ncol, figsize=(ps.WIDTH_DOUBLE_IN, 1.85*len(runs)+0.45), squeeze=False, constrained_layout=True)
    vmax = max(np.percentile(np.abs(p["omega_wing"]+(0 if p["omega_rings"] is None else p["omega_rings"])), 99.7) for run in runs for p in run) or 1.
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
    letters = iter("abcdefghijklmnop")
    for r, run in enumerate(runs):
        for c, p in enumerate(run):
            ax = axes[r, c]
            omega = p["omega_wing"]+(0 if p["omega_rings"] is None else p["omega_rings"])
            im = ax.pcolormesh(p["y"]/diameter, p["z"]/diameter, omega, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
            if "deficit" in p:
                levels = [0.02, 0.05, 0.1, 0.2]
                ax.contour(p["y"]/diameter, p["z"]/diameter, p["deficit"], levels=levels, colors="k", linewidths=0.5)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$(y-y_c)/D$")
            if c == 0:
                ax.set_ylabel(r"$(z-z_c)/D$")
            ps.letter(ax, f"({next(letters)})")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label=r"$\omega_x$ (s$^{-1}$)")
    ps.save(fig, path)


def figure_budget(planes, diameter, radius, path):
    import matplotlib.pyplot as plt
    ps.style()
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE_IN, 2.5), constrained_layout=True)
    x = [p["station_D"] for p in planes]
    axes[0].plot(x, [p["wing_positive_m2_s"] for p in planes], "o-", color=ps.COL["outer"], label=r"$\Gamma^+$, wing")
    axes[0].plot(x, [-p["wing_negative_m2_s"] for p in planes], "s-", color=ps.COL["inner"], label=r"$-\Gamma^-$, wing")
    if any(p["rings_crossings"] for p in planes):
        axes[0].plot(x, [p["rings_positive_m2_s"] for p in planes], "o--", color=ps.COL["rotors"], label=r"$\Gamma^+$, rings")
        axes[0].plot(x, [-p["rings_negative_m2_s"] for p in planes], "s--", color="#66C2A4", label=r"$-\Gamma^-$, rings")
    axes[0].set_xlabel(r"$(x-x_c)/D$"); axes[0].set_ylabel(r"circulation through the plane (m$^2$ s$^{-1}$)")
    axes[0].legend(loc="best")
    for sign, col, mark in (("positive", ps.COL["outer"], "o-"), ("negative", ps.COL["inner"], "s-")):
        axes[1].plot(x, [p.get(f"wing_{sign}_centroid_r_over_R", np.nan) for p in planes], mark, color=col, label=f"{sign} flux, wing")
    axes[1].set_xlabel(r"$(x-x_c)/D$"); axes[1].set_ylabel(r"centroid distance from the axis, $r/R$")
    axes[1].legend(loc="best")
    for ax, k in zip(axes, "ab"):
        ps.grid(ax); ps.letter(ax, f"({k})")
    ps.save(fig, path)


# ----------------------------------------------------------------------------------------------------------------------
# 3D rendering (PyVista, offscreen; display only)
# ----------------------------------------------------------------------------------------------------------------------


def render3d(run, path, size=(2600, 1500), views=None, closeup_d=0.45):
    """Two views (overview of the whole stored wake, close-up behind the kite); one plotter per view."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    nodes, edges, gamma, first, rf = run["nodes"], run["edges"], run["gamma"], run["first"], run["ring_first"]
    wing_e = np.all(edges < rf, axis=1)
    free = wing_e & ~np.all(edges < first, axis=1)                  # wake segments (not the bound row)
    a, b = edges[:, 0], edges[:, 1]
    trailing = free & (np.abs(a-b) % first == 0) & (np.abs(a-b) > 0)
    shed = free & ~trailing
    tangent = nodes[b]-nodes[a]
    length = np.linalg.norm(tangent, axis=1)
    omega_x = gamma*tangent[:, 0]/np.maximum(length, 1e-12)      # circulation times the x component of the unit tangent

    def lines(mask, scalars=None):
        idx = np.flatnonzero(mask)
        cells = np.empty((len(idx), 3), dtype=np.int64)
        cells[:, 0] = 2
        cells[:, 1:] = edges[idx]
        mesh = pv.PolyData(nodes, lines=cells.ravel())
        if scalars is not None:
            mesh.cell_data["s"] = scalars[idx]
        return mesh

    scale = float(np.percentile(np.abs(omega_x[trailing]), 99)) or 1.
    strong = trailing & (np.abs(omega_x) > 0.12*scale)
    weak = trailing & ~strong
    layers = []
    if np.any(shed):
        layers.append((lines(shed), dict(color=(0.80, 0.80, 0.80), opacity=0.10, line_width=0.5, lighting=False)))
    if np.any(weak):
        layers.append((lines(weak, omega_x), dict(scalars="s", cmap="RdBu_r", clim=(-scale, scale), opacity=0.25, line_width=0.8,
                                                   show_scalar_bar=False, lighting=False)))
    layers.append((lines(strong, omega_x), dict(scalars="s", cmap="RdBu_r", clim=(-scale, scale), line_width=2.2,
                                                show_scalar_bar=False, lighting=False)))
    if np.any(~wing_e):
        layers.append((lines(~wing_e), dict(color="#1b9e77", opacity=0.35, line_width=0.9, lighting=False)))
    import airframe
    grounds = {"A": np.zeros(3)}
    if "B" in run["motions"]:
        grounds["B"] = np.asarray(run["settings"].get("ground_point_b_m", [0., 0., 0.]), dtype=float)
    centre, radius, diameter = geometry(run)
    kite = run["motions"]["A"]["position_m"][run["step"]]
    lo, hi = nodes[:rf].min(axis=0), nodes[:rf].max(axis=0)
    focus = 0.5*(lo+hi)
    extent = float(max(hi-lo))
    if views is None:
        views = dict(overview=(focus+np.array([-0.9, -1.5, 0.75])*extent*0.95, focus, (0., 0., 1.)),
                     closeup=(kite+np.array([-0.45, -0.75, 0.40])*closeup_d*diameter, kite+np.array([0.25, 0., -0.05])*closeup_d*diameter,
                              (0., 0., 1.)),
                     detail=(kite+run["motions"]["A"]["rotation"][run["step"]]@np.array([-34., -46., 24.]),
                             kite+run["motions"]["A"]["rotation"][run["step"]]@np.array([6., 0., 0.]), (0., 0., 1.)))
    for name, camera in views.items():
        plot = pv.Plotter(off_screen=True, window_size=size)
        plot.set_background("white")
        for mesh, options in layers:
            plot.add_mesh(mesh, **options)
        for body, motion in run["motions"].items():
            airframe.add_airframe(plot, motion["rotation"][run["step"]], motion["position_m"][run["step"]],
                                  ground_point=grounds.get(body))
        plot.camera_position = camera
        plot.enable_anti_aliasing("ssaa")
        plot.screenshot(str(path)+f"_{name}.png")
        plot.close()


def render3d_sheet(run, path, size=(2800, 1500), wing_nodes=WING_NODES_M600, tube_radius=0.8, sheet_opacity=0.18,
                   max_age_laps=None, views=None):
    """Publication render: the two wing-tip vortex lines of the wake as tubes (outer tip red, inner tip blue), the wake
    sheet of the main wing as a translucent surface, rotor rings (if any) as thin green lines, and the complete airframe
    with bridle and tether at the stored pose. Display only."""
    import pyvista as pv
    import airframe
    pv.OFF_SCREEN = True
    nodes, edges, gamma, first, rf = run["nodes"], run["edges"], run["gamma"], run["first"], run["ring_first"]
    rows = nodes[first:rf].reshape(-1, first, 3)          # row 0 = trailing-edge row, then the free rows (newest first)
    if max_age_laps is not None:
        rows = rows[:int(max_age_laps*run["steps_per_lap"])+1]
    td = tip_displacement(run, wing_nodes)
    outer_left = td["outer"] is not None
    S = run["step"]
    motion = run["motions"]["A"]
    centre, radius, diameter = geometry(run)
    # which column is the outer tip: compare the distance from the loop axis at the newest row
    axis = centre/np.linalg.norm(centre)
    def dist(point):
        rel = point-centre
        return np.linalg.norm(rel-np.dot(rel, axis)*axis)
    outer_col, inner_col = (0, wing_nodes-1) if dist(rows[1, 0]) > dist(rows[1, wing_nodes-1]) else (wing_nodes-1, 0)
    layers = []
    for col, colour in ((outer_col, "#B2182B"), (inner_col, "#2166AC")):
        line = pv.lines_from_points(rows[:, col])
        layers.append((line.tube(radius=tube_radius, n_sides=12), dict(color=colour, smooth_shading=True, ambient=0.35, diffuse=0.7)))
    # wing sheet (main wing columns only)
    sheet_pts = rows[:, :wing_nodes].reshape(-1, 3)
    nr, nc = rows.shape[0], wing_nodes
    quads = []
    for j in range(nr-1):
        base = j*nc
        for i in range(nc-1):
            quads.append([4, base+i, base+i+1, base+nc+i+1, base+nc+i])
    sheet = pv.PolyData(sheet_pts, faces=np.asarray(quads, dtype=np.int64).ravel())
    layers.append((sheet, dict(color="#8a95a5", opacity=sheet_opacity, smooth_shading=False, lighting=False)))
    if rf < len(nodes):
        ring_edges = edges[np.all(edges >= rf, axis=1)]
        cells = np.column_stack([np.full(len(ring_edges), 2), ring_edges]).ravel()
        layers.append((pv.PolyData(nodes, lines=cells), dict(color="#1b9e77", opacity=0.35, line_width=0.8, lighting=False)))
    kite = motion["position_m"][S]
    lo, hi = rows.reshape(-1, 3).min(axis=0), rows.reshape(-1, 3).max(axis=0)
    focus = 0.5*(lo+hi)
    extent = float(max(hi-lo))
    if views is None:
        views = dict(side=(focus+np.array([0.05, -1.45, 0.25])*extent, focus, (0., 0., 1.)),
                     oblique=(focus+np.array([-0.95, -1.25, 0.65])*extent, focus, (0., 0., 1.)),
                     kite=(kite+motion["rotation"][S]@np.array([-30., -40., 22.]), kite+motion["rotation"][S]@np.array([8., 0., 0.]), (0., 0., 1.)))
    for name, camera in views.items():
        plot = pv.Plotter(off_screen=True, window_size=size)
        plot.set_background("white")
        for mesh, options in layers:
            plot.add_mesh(mesh, **options)
        airframe.add_airframe(plot, motion["rotation"][S], motion["position_m"][S], ground_point=np.zeros(3))
        # wind direction (green) and, at the kite, the lift direction (body z, dark grey): author's figure rule
        if name == "kite":
            plot.add_mesh(pv.Arrow(start=kite-motion["rotation"][S]@np.array([0., 0., -2.5]), direction=np.array([1., 0., 0.]), scale=8.0,
                                   tip_length=0.25, tip_radius=0.08, shaft_radius=0.03), color="#2a9d4a")
            plot.add_mesh(pv.Arrow(start=kite, direction=motion["rotation"][S]@np.array([0., 0., 1.]), scale=6.0, tip_length=0.25,
                                   tip_radius=0.08, shaft_radius=0.03), color="#3d4654")
        else:
            anchor = lo+np.array([0.02, 0.0, 0.05])*extent
            plot.add_mesh(pv.Arrow(start=anchor, direction=np.array([1., 0., 0.]), scale=0.12*extent, tip_length=0.25,
                                   tip_radius=0.08, shaft_radius=0.03), color="#2a9d4a")
        plot.camera_position = camera
        plot.enable_anti_aliasing("ssaa")
        plot.screenshot(str(path)+f"_{name}.png")
        plot.close()


# ----------------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------------


def to_json(planes):
    keep = ("station_D", "x_m", "deficit_max", "momentum_deficit_m2_s", "profile", "radius_bins_m")
    return [{k: v for k, v in p.items() if k in keep or k.startswith(("wing_", "rings_"))} for p in planes]


def main(argv):
    parser = argparse.ArgumentParser(prog="wake_figures.py")
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--state", type=int, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--planes", default="0.5,1,2,3")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--count", type=int, default=121)
    parser.add_argument("--half", type=float, default=0.75, help="half-width of the planes in loop diameters")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--no-deficit", action="store_true")
    parser.add_argument("--no-meridional", action="store_true")
    parser.add_argument("--vis-core", type=float, default=3.0, help="display core of the meridional field [m]")
    parser.add_argument("--mer-spacing", type=float, default=2.0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = read(args.run, args.state)
    label = args.label or Path(args.run).name
    stations = [float(v) for v in args.planes.split(",") if v.strip()]
    t = tips(run)
    figure_tips(t, out/f"{label}_tips")
    td = tip_displacement(run)
    figure_tip_displacement(td, out/f"{label}_tip_displacement")
    if not args.no_meridional:
        mer = meridional(run, args.vis_core, args.mer_spacing)
        figure_meridional([mer], out/f"{label}_meridional")
        pairs = track_pairs(mer)
        np.savez_compressed(out/f"{label}_meridional.npz", **{k: np.asarray(v) for k, v in mer.items()})
    planes, centre, radius, diameter = plane_vorticity(run, stations, args.half, args.count, args.sigma, not args.no_deficit)
    figure_planes(planes, diameter, out/f"{label}_omega_planes")
    figure_budget(planes, diameter, radius, out/f"{label}_circulation")
    summary = dict(run=str(args.run), step=run["step"], path_centre_m=centre.tolist(), radius_m=radius, diameter_m=diameter,
                   sigma_m=args.sigma, tips=tip_summary(t, stations), planes=to_json(planes),
                   tip_displacement={name: {f"lap{k}": dict(dr_over_R=float(np.mean(td[name]["dr"][(td[name]["age_s"]/td["period_s"] > k-0.5) & (td[name]["age_s"]/td["period_s"] <= k)])/td["radius"]),
                                                       dz_over_D=float(np.mean(td[name]["dz"][(td[name]["age_s"]/td["period_s"] > k-0.5) & (td[name]["age_s"]/td["period_s"] <= k)])/td["diameter"]),
                                                       dx_over_D=float(np.mean(td[name]["dx"][(td[name]["age_s"]/td["period_s"] > k-0.5) & (td[name]["age_s"]/td["period_s"] <= k)])/td["diameter"]))
                                            for k in range(1, int(td[name]["age_s"].max()/td["period_s"])+1)} for name in ("outer", "inner")})
    if not args.no_meridional:
        summary["meridional_pairs"] = pairs
        summary["meridional_vis_core_m"] = args.vis_core
    (out/f"{label}_wake_figures.json").write_text(json.dumps(summary, indent=1, default=float)+"\n", encoding="utf-8")
    np.savez_compressed(out/f"{label}_planes.npz", **{f"p{i}_{k}": np.asarray(v) for i, p in enumerate(planes)
                                                      for k, v in p.items() if isinstance(v, np.ndarray)})
    if not args.no_3d:
        render3d(run, out/f"{label}_wake3d_pv")
        render3d_sheet(run, out/f"{label}_sheet3d")
    print(json.dumps(dict(tips=summary["tips"], planes=[{k: p[k] for k in p if k in ("station_D", "deficit_max", "wing_positive_m2_s", "wing_negative_m2_s", "rings_positive_m2_s", "rings_negative_m2_s")} for p in summary["planes"]]), indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
