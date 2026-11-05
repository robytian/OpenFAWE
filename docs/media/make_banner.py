"""Frames of the animation and the still in docs/media: the two M600 units of the tandem at one loop diameter with their
rotors (run p2a-m600-pair-d1-rotors-adp-5laps of paper1/crosswind, the 24 wake networks stored over the last lap with
--frames 24 and the final one), drawn from the stored filaments on a dark background. Illustration only: the airframes
and the young rotor rings are drawn three times their size about the body origin and the youngest 70 wake rows of the
main wing are moved along their own span so that the tip filaments leave the enlarged wing tips; the filaments are
otherwise the stored data, coloured by their circulation (unit A blue and cyan, unit B magenta and orange).
The wake and rings (--layer wake) and the airframes (--layer kites, transparent background) are rendered separately at
the same camera and composited so that the kites stay on top.
Usage, for every stored step S of RUN (a run folder with history.json and state_<step>.npz) and LAYER in wake, kites:
  python docs/media/make_banner.py RUN . LAYER_S.png --layer LAYER --dark --bg "#0D1117" --step S --size 1800 900
      --scale 3 --ring-lines --ring-steps 20 --campos -23.9 452.5 323.1 --focus 600 -7 232 --angle 32 --widths 3 1.6 1"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def radial_map(x, centres, s, r0, r1):
    """magnify by s inside r0 about each centre, blend linearly in r to the identity at r1 (monotone in r)"""
    y = x.copy()
    for c in centres:
        d = x - c; r = np.linalg.norm(d, axis=1)
        g = np.where(r < r0, s * r, np.where(r < r1, s * r0 + (r - r0) * (r1 - s * r0) / (r1 - r0), r))
        inside = r < r1
        y[inside] = c + d[inside] * (g[inside] / np.maximum(r[inside], 1e-9))[:, None]
    return y


def field(X, Ew, Gw, wt, sel, lo, shape, h, sig):
    """vorticity vector of the edges sel (weights wt) with a Gaussian core sig, on the grid (lo, h, shape)"""
    p0, d = X[Ew[sel, 0]], X[Ew[sel, 1]] - X[Ew[sel, 0]]
    L = np.linalg.norm(d, axis=1); ns = np.maximum(1, np.ceil(L / (0.5 * h)).astype(int))
    rep = np.repeat(np.arange(len(sel)), ns)
    offs = np.concatenate([(np.arange(n) + 0.5) / n for n in ns])
    pts = p0[rep] + d[rep] * offs[:, None]
    q = ((Gw[sel] * wt[sel])[rep] / ns[rep])[:, None] * d[rep]
    idx = np.clip(np.rint((pts - lo) / h).astype(int), 0, shape - 1)
    flat = np.ravel_multi_index(idx.T, shape); n = int(np.prod(shape))
    return np.stack([gaussian_filter(np.bincount(flat, weights=q[:, c], minlength=n).reshape(shape).astype(np.float32),
                                     sig / h, mode="constant") / h ** 3 for c in range(3)])


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("run"); p.add_argument("repo"); p.add_argument("out")
    p.add_argument("--size", type=int, nargs=2, default=[2600, 2600])
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--shift-rows", type=float, nargs=2, default=[3, 70])
    p.add_argument("--shift-w", type=float, default=1.5, help="span width (m) over which the shift changes sign")
    p.add_argument("--h", type=float, default=2.0)
    p.add_argument("--cores", type=float, nargs="+", default=[1.4, 2.4, 3.4, 4.5])
    p.add_argument("--row-edges", type=float, nargs="+", default=[40, 160, 480], help="row ages between the cores")
    p.add_argument("--iso", type=float, default=0.1, help="iso level, fraction of the 99.9 percentile of |w| of the old wake")
    p.add_argument("--eye", type=float, nargs=3, default=[-0.35, -1.0, 0.45])
    p.add_argument("--up", type=float, nargs=3, default=[0, 0, 0], help="0 0 0: roll so that the wind points right")
    p.add_argument("--focus", type=float, nargs=3, default=None)
    p.add_argument("--campos", type=float, nargs=3, default=None)
    p.add_argument("--dist", type=float, default=2200.0); p.add_argument("--angle", type=float, default=30.0)
    p.add_argument("--bg", default="white")
    p.add_argument("--ring-steps", type=int, default=14)
    p.add_argument("--ring-tube", type=float, default=0.10)
    p.add_argument("--ring-r", type=float, nargs=2, default=[14, 60], help="ring map: full scale within, identity beyond")
    p.add_argument("--widths", type=float, nargs=3, default=[5.0, 2.6, 1.4], help="line widths (px) tip, strong, weak")
    p.add_argument("--mid-gamma", type=float, default=4.0)
    p.add_argument("--gclim", type=float, default=20.0)
    p.add_argument("--smooth-rows", type=float, default=0.0)
    p.add_argument("--smooth-from", type=float, nargs=2, default=[80, 300])
    p.add_argument("--ring-lines", action="store_true", help="rings as thin lines as in Fig. 12")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--dark", action="store_true", help="colours for a dark background")
    p.add_argument("--layer", default="all", choices=["all", "wake", "kites"],
                   help="wake: filaments and rings only; kites: airframes only on a transparent background (top layer)")
    p.add_argument("--clip-upstream", action="store_true", help="end the wake of the first unit at the loop plane of the second")
    p.add_argument("--frame", default="#8A9099", help="colour of wing and fuselage")
    p.add_argument("--box", type=float, nargs=6, default=None)
    p.add_argument("--marks", default=None)
    a = p.parse_args(argv)
    repo = Path(a.repo); task = repo/"paper1/crosswind"
    sys.path.insert(0, str(repo/"src")); sys.path.insert(0, str(task))
    import pyvista as pv
    import wake_figures as wf, wake_render as wr, airframe
    from matplotlib.colors import LinearSegmentedColormap as LSC
    pv.OFF_SCREEN = True
    run = wf.read(a.run, a.step); S, n = run["step"], run["steps_per_lap"]
    bodies = list(run["motions"])
    kites = {k: np.asarray(run["motions"][k]["position_m"][S], float) for k in bodies}
    rot = {k: np.asarray(run["motions"][k]["rotation"][S], float) for k in bodies}
    centre = {k: np.asarray(run["motions"][k]["position_m"][S-n+1:S+1], float).mean(axis=0) for k in bodies}
    E, G = run["edges"], run["gamma"]; first, ring_first = run["first"], run["ring_first"]
    nb = len(bodies); per = first // nb; wn = wf.WING_NODES_M600
    # wing nodes as node rows (row 0 bound, then the wake rows, newest first) and the span shift of the young rows
    X = run["nodes"].astype(float).copy()
    Xr = X[:ring_first].reshape(-1, first, 3)
    j0, j1 = a.shift_rows; J = int(np.ceil(j1)) + 1
    for b in range(nb):
        cols = np.arange(b * per, b * per + wn)
        tl, tr = Xr[:J, cols[0]], Xr[:J, cols[-1]]
        mid = 0.5 * (tl + tr); half = 0.5 * np.linalg.norm(tr - tl, axis=1); e = (tr - tl) / (2 * half[:, None])
        f = 1 - smooth((np.arange(J) - j0) / (j1 - j0))
        yb = np.einsum("jcx,jx->jc", Xr[:J, cols] - mid[:, None], e)
        Xr[:J, cols] += ((a.scale - 1) * half[:, None] * np.tanh(yb / a.shift_w) * f[:, None])[..., None] * e[:, None, :]
    X[:ring_first] = Xr.reshape(-1, 3)
    # trailing filaments of the main wing of each unit as polylines over all wake rows, coloured by their circulation
    tr = (E[:, 1] - E[:, 0] == first) | (E[:, 0] - E[:, 1] == first)
    lo_n = np.minimum(E[tr, 0], E[tr, 1]); g = np.where(E[tr, 1] > E[tr, 0], G[tr], -G[tr])
    gam = np.zeros(ring_first); gam[lo_n] = g
    gr = gam.reshape(-1, first)
    plot = pv.Plotter(off_screen=True, window_size=a.size); plot.set_background(a.bg)
    cmaps = {"A": LSC.from_list("A", ["#0A1E8C", "#2F6BFF", "#C9D6F2", "#E6E9EF", "#BFEFF7", "#13C8EC", "#007C99"]),
             "B": LSC.from_list("B", ["#8C0050", "#E8007D", "#F4C2DA", "#EFE9EC", "#FFE0B8", "#FF9A1F", "#C25E00"])}
    if a.dark:
        cmaps = {"A": LSC.from_list("A", ["#3D7BFF", "#2A5FD6", "#1C2A44", "#161D2B", "#16394A", "#1FB8E0", "#5FF0FF"]),
                 "B": LSC.from_list("B", ["#FF3FA4", "#C2187A", "#3A1B2E", "#22171F", "#4A2A12", "#F08A1C", "#FFC24A"])}
    nrow = Xr.shape[0]
    if a.smooth_rows > 0:                     # row-to-row jitter of the old rolled-up filaments smoothed for the picture
        from scipy.ndimage import gaussian_filter1d
        Xs = gaussian_filter1d(Xr, a.smooth_rows, axis=0, mode="nearest")
        bl = smooth((np.arange(nrow) - a.smooth_from[0]) / (a.smooth_from[1] - a.smooth_from[0]))[:, None, None]
        Xr = (1 - bl) * Xr + bl * Xs
    for k, body in enumerate(bodies if a.layer != "kites" else []):
        for group, width, op in (("thin", a.widths[2], 0.55), ("mid", a.widths[1], 0.85), ("tip", a.widths[0], 1.0)):
            cols = []
            for c in range(wn):
                gm = np.abs(gr[1:, k * per + c]).mean()
                kind = "tip" if c in (0, wn - 1) else ("mid" if gm > a.mid_gamma else "thin")
                if kind == group: cols.append(k * per + c)
            if not cols: continue
            ends = [nrow] * len(cols)
            if a.clip_upstream and k == 0 and nb > 1:
                # owner 2026-11-05 00:05: the wake of the first unit behind the second one removed; each filament of A
                # ends where it first crosses the loop plane of B (through the loop centre of B, normal along the
                # fitted loop normal pointing downstream)
                P = np.asarray(run["motions"][bodies[1]]["position_m"][S-n+1:S+1], float); cB = P.mean(axis=0)
                nrm = np.linalg.svd(P - cB)[2][-1]; nrm = nrm if nrm[0] > 0 else -nrm
                for i, c in enumerate(cols):
                    beyond = np.nonzero((Xr[1:, c] - cB) @ nrm > 0)[0]
                    ends[i] = 1 + (beyond[0] if len(beyond) else nrow - 1)
            segs = [Xr[1:e, c] for c, e in zip(cols, ends)]
            pts = np.vstack(segs); sc = np.concatenate([gr[1:e, c] for c, e in zip(cols, ends)])
            cells = []; o = 0
            for sg in segs:
                cells += [len(sg)] + list(range(o, o + len(sg))); o += len(sg)
            cells = np.array(cells)
            poly = pv.PolyData(pts, lines=cells); poly.point_data["g"] = sc
            plot.add_mesh(poly, scalars="g", cmap=cmaps[body], clim=[-a.gclim, a.gclim], line_width=width,
                          render_lines_as_tubes=True, opacity=op, show_scalar_bar=False, specular=0.3, ambient=0.3)
    lo = X[:ring_first].min(axis=0); hi = X[:ring_first].max(axis=0)
    rings, nper = wr._rings(run)
    if rings is not None and a.ring_steps > 0 and a.layer != "kites":
        m = min(len(rings), 2 * 8 * a.ring_steps)
        rr_ = radial_map(rings[:m].reshape(-1, 3), [kites[k] for k in bodies], a.scale, *a.ring_r).reshape(m, nper, 3)
        cells = np.hstack([[nper + 1] + list(range(k * nper, (k + 1) * nper)) + [k * nper] for k in range(m)])
        poly = pv.PolyData(rr_.reshape(-1, 3), lines=cells)
        poly.point_data["age"] = np.repeat(np.arange(m) // 16, nper) / max(a.ring_steps - 1, 1)
        rcm = LSC.from_list("ring", ["#FFD000", "#FF9500", "#FF5A00"])
        if a.ring_lines:
            plot.add_mesh(poly, scalars="age", cmap=rcm, clim=[0, 1], line_width=a.widths[2], opacity=0.8,
                          lighting=False, show_scalar_bar=False)
        else:
            plot.add_mesh(poly.tube(radius=a.ring_tube * a.scale, n_sides=10), scalars="age", cmap=rcm, clim=[0, 1],
                          smooth_shading=True, specular=0.4, ambient=0.3, show_scalar_bar=False)
    add = plot.add_mesh; tether = airframe.COLOURS["tether"]; cur = {}

    def scaled(mesh, *x, **kw):
        if kw.get("color") == tether:
            return None
        mesh = mesh.copy(); mesh.points = cur["o"] + a.scale * (np.asarray(mesh.points) - cur["o"])
        return add(mesh, *x, **kw)
    plot.add_mesh = scaled
    saved = dict(airframe.COLOURS)
    airframe.COLOURS.update(surface=a.frame, fuselage=a.frame, edge="#3A3F46", disc="#D9DDE3", nacelle="#3A3F46")
    if a.dark:
        airframe.COLOURS.update(surface="#D7DCE3", fuselage="#C9CFD8", edge="#F5F7FA", disc="#FFE08A", nacelle="#9AA3AF")
    for body in (bodies if a.layer != "wake" else []):
        cur["o"] = kites[body]
        airframe.add_airframe(plot, rot[body], kites[body], ground_point=None, rotors=True, scale_lines=1.6, disc_outline=True)
    airframe.COLOURS.clear(); airframe.COLOURS.update(saved)
    plot.add_mesh = add
    focus = np.array(a.focus) if a.focus else 0.5 * (lo + hi)
    eye = np.asarray(a.eye, float); eye /= np.linalg.norm(eye)
    cam = np.array(a.campos) if a.campos else focus + a.dist * eye
    up = np.asarray(a.up, float)
    if not up.any():                                   # roll the camera so that the wind (x) points to the right
        f = (focus - cam) / np.linalg.norm(focus - cam); r = np.array([1.0, 0, 0]) - f[0] * f; r /= np.linalg.norm(r)
        up = np.cross(r, f)
    plot.camera_position = (cam, focus, tuple(up)); plot.camera.view_angle = a.angle
    plot.enable_anti_aliasing("ssaa"); plot.screenshot(a.out, transparent_background=(a.layer == "kites"))
    if a.marks:
        ren = plot.renderer; marks = {}
        for key in bodies:
            ren.SetWorldPoint(*kites[key], 1.0); ren.WorldToDisplay(); x, y, _ = ren.GetDisplayPoint()
            marks[key] = [x, a.size[1] - y]
        Path(a.marks).write_text(json.dumps(marks))
    plot.close()
    print(json.dumps({"lo": lo.round(1).tolist(), "hi": hi.round(1).tolist()}))


if __name__ == "__main__":
    main(sys.argv[1:])
