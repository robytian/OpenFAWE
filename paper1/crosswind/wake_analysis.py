"""Read-back and figures of the OF-0009 crosswind-flight runs (C1 loops, C2 patterns, C3 pairs).

Quantities (all from the run's own records and stored wake networks; nothing is interpolated from elsewhere):
  loads     per-step body-axis force and moment coefficients, tether tension, airspeed, incidence, aerodynamic power,
            against the loop angle (C1) or the time in the cycle (C2), lap by lap; lap/cycle means;
  centre    the wake centre line: for every wake row (one per time step, newest first) the mean downstream position and the
            centroid of the node positions weighted by the trailing circulation magnitude, split into the positive and the
            negative trailing-vorticity halves (the two tip-vortex centres of Betz); their descent and lateral drift
            against the downstream distance, and the spread (rms radius) of the trailing vorticity about the centroid;
  planes    axial velocity deficit (1 - u/U) and in-plane velocity on cross-flow planes normal to the wind at stations
            downstream of the path centre, from the complete network with the run's own kernel and core (``wake_probe``),
            and the azimuthally averaged deficit profile about the wake centre of each plane;
  pair      (C3) the loads of unit B against unit A lap by lap and step by step.

Usage: wake_analysis.py RUN_DIR OUT_DIR [--planes 0.5,1,2,3] [--plane-half 250] [--plane-count 121] [--backend tree]
       The stations are in loop diameters (C1/C3: 2 R of the loop; C2: the circle's path diameter or the eight's width)
       downstream of the path centre. Figures carry no text inside the plot except axes and legends (author rule).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
sys.path.insert(0, str(REPO/"src"))
from openfawe.aero import wake_probe as wp  # noqa: E402
from openfawe.aero.vortex import CoreModel, TreeSettings  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------------------------------


def read_run(run_dir):
    run_dir = Path(run_dir)
    history = json.loads((run_dir/"history.json").read_text(encoding="utf-8"))
    states = sorted(run_dir.glob("state_*.npz"))
    kind = "pair" if "per_lap" in history and isinstance(history.get("per_lap"), dict) else ("pattern" if "pattern" in history["settings"] else "loop")
    return history, states, kind


def period_steps(history, kind):
    s = history["settings"]
    return int(s["steps_per_cycle"] if kind == "pattern" else s["steps_per_lap"])


def body_records(history, kind, name=None):
    """The per-step records of one body as a dict of arrays."""
    records = history["records"]
    rows = [r["bodies"][name] for r in records] if kind == "pair" else records
    out = dict(step=np.array([r["step"] for r in records]), time_s=np.array([r["time_s"] for r in records]),
               force_body_N=np.array([r["force_body_N"] for r in rows]), moment_body_Nm=np.array([r["moment_origin_body_Nm"] for r in rows]),
               force_inertial_N=np.array([r["force_inertial_N"] for r in rows]), airspeed_m_s=np.array([r["airspeed_m_s"] for r in rows]),
               alpha_deg=np.array([r["alpha_deg"] for r in rows]), q_Pa=np.array([r["dynamic_pressure_Pa"] for r in rows]),
               power_W=np.array([r["aero_power_W"] for r in rows]), iterations=np.array([r["iterations"] for r in records]),
               position_m=np.array([r["position_m"] for r in rows]))
    if "tether_tension_kite_N" in rows[0]:
        out["tension_N"] = np.array([r["tether_tension_kite_N"] for r in rows])
        out["tether_force_body_N"] = np.array([r["tether_force_body_N"] for r in rows])
    if kind == "loop":
        angle = np.array(history["commands"]["loop_angle_rad"])
        out["loop_angle_deg"] = np.degrees(angle[:len(records)])
    return out


def coefficients(body, area, chord):
    q = body["q_Pa"][:, None]
    return body["force_body_N"]/(q*area), body["moment_body_Nm"]/(q*area*chord)


def per_period(body, n, area, chord):
    """Means over each complete period (lap or cycle)."""
    cf, cm = coefficients(body, area, chord)
    rows = []
    k = 1
    while k*n < len(body["step"]):
        sl = slice((k-1)*n+1, k*n+1)
        row = dict(period=k, CX=float(cf[sl, 0].mean()), CY=float(cf[sl, 1].mean()), CZ=float(cf[sl, 2].mean()),
                   Cl=float(cm[sl, 0].mean()), Cm=float(cm[sl, 1].mean()), Cn=float(cm[sl, 2].mean()),
                   lift_N=float(body["force_body_N"][sl, 2].mean()), power_kW=float(body["power_W"][sl].mean()/1e3),
                   airspeed_m_s=float(body["airspeed_m_s"][sl].mean()), alpha_deg=float(body["alpha_deg"][sl].mean()))
        if "tension_N" in body:
            t = body["tension_N"][sl]
            row.update(tension_mean_kN=float(t.mean()/1e3), tension_min_kN=float(t.min()/1e3), tension_max_kN=float(t.max()/1e3))
        rows.append(row)
        k += 1
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# wake rows and the centre line
# ----------------------------------------------------------------------------------------------------------------------


def load_state(file):
    """``(nodes, edges, gamma, first, ring_first)``; the rotor rings, when stored (``c3_pair.py``), follow every wing node."""
    with np.load(file) as d:
        ring_first = int(d["ring_first"]) if "ring_first" in d.files else len(d["nodes"])
        return d["nodes"], d["edges"].astype(np.int64), d["gamma"], int(d["first"]), ring_first


def wing_part(nodes, edges, gamma, ring_first):
    """The network of the lifting surfaces alone (bound, trailing-edge and free rows), without the rotor rings."""
    keep = np.all(edges < ring_first, axis=1)
    return nodes[:ring_first], edges[keep], gamma[keep]


def wake_rows(nodes, edges, gamma, first):
    """Wake rows (newest first, row 0 = trailing edge) and, per row, the trailing circulation of each spanwise node.

    The network packs the bound row, the trailing-edge row and the free rows with the same node count ``n``; a trailing
    edge joins node i of row j-1 to node i of row j (index difference n), a shed edge joins neighbours within a row.
    """
    n = first
    wake = nodes[first:]
    rows = wake.reshape(-1, n, 3)
    a, b = edges[:, 0], edges[:, 1]
    trailing = (a >= first) & (b >= first) & (np.abs(a-b) == n)
    low = np.minimum(a[trailing], b[trailing])-first
    j = low//n+1                                           # the row the edge ends on (the older row)
    i = low % n
    sign = np.where(a[trailing] < b[trailing], 1.0, -1.0)   # circulation taken from the newer to the older row
    circulation = np.zeros((rows.shape[0], n))
    np.add.at(circulation, (j, i), gamma[trailing]*sign)
    return rows, circulation


def centre_line(rows, circulation, dt, wind_m_s):
    """Per wake row: age, mean x, |Γ|-weighted centroid, the centroids of the positive and negative trailing vorticity, spread."""
    out = []
    for j in range(1, rows.shape[0]):
        g, p = circulation[j], rows[j]
        weight = np.abs(g)
        if weight.sum() == 0.:
            continue
        centroid = (weight[:, None]*p).sum(axis=0)/weight.sum()
        spread = float(np.sqrt((weight*np.sum((p-centroid)**2, axis=1)).sum()/weight.sum()))
        halves = {}
        for name, mask in (("positive", g > 0.), ("negative", g < 0.)):
            if mask.any():
                w = np.abs(g[mask])
                halves[name] = ((w[:, None]*p[mask]).sum(axis=0)/w.sum()).tolist()
                halves[name+"_circulation_m2_s"] = float(g[mask].sum())
        out.append(dict(row=j, age_s=j*dt, x_mean_m=float(p[:, 0].mean()), centroid_m=centroid.tolist(), spread_m=spread,
                        circulation_abs_sum_m2_s=float(weight.sum()), **halves))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# cross-flow planes
# ----------------------------------------------------------------------------------------------------------------------


def plane_at(x, centre_yz, half, count, nodes, edges, gamma, wind_m_s, core_radius, backend="tree"):
    plane = wp.crossflow_plane(x, half, half, count, count, centre_y=centre_yz[0], centre_z=centre_yz[1])
    kernel = dict(backend=backend) if backend != "tree" else dict(backend="tree", tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
    fields = wp.plane_fields(plane, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=core_radius,
                             freestream=np.array([wind_m_s, 0., 0.]), **kernel)
    deficit = 1.-fields["axial"]/wind_m_s
    # wake centre of the plane: deficit-weighted centroid of the positive deficit
    w = np.clip(deficit, 0., None)
    yy, zz = fields["points"][..., 1], fields["points"][..., 2]
    yc, zc = (float((w*yy).sum()/w.sum()), float((w*zz).sum()/w.sum())) if w.sum() > 0 else (float(centre_yz[0]), float(centre_yz[1]))
    r = np.hypot(yy-yc, zz-zc)
    edges_r = np.linspace(0., half, 26)
    profile = [float(deficit[(r >= a) & (r < b)].mean()) if np.any((r >= a) & (r < b)) else None for a, b in zip(edges_r[:-1], edges_r[1:])]
    return dict(x_m=float(x), u=fields["u"].tolist(), v=fields["v"].tolist(), y=yy, z=zz, deficit=deficit, in_plane=fields["in_plane"],
                vorticity=fields["normal_vorticity"], centre=(yc, zc), path_centre_yz=(float(centre_yz[0]), float(centre_yz[1])), wind_m_s=float(wind_m_s),
                radius_bins_m=(0.5*(edges_r[:-1]+edges_r[1:])).tolist(),
                profile=profile, deficit_max=float(deficit.max()), deficit_mean_positive=float(w[w > 0].mean()) if (w > 0).any() else 0.,
                momentum_deficit_m2_s=float((deficit*wind_m_s*(1.-deficit)).sum()*(fields["u"][1]-fields["u"][0])*(fields["v"][1]-fields["v"][0])))


# ----------------------------------------------------------------------------------------------------------------------
# figures (matplotlib; no text inside the plot area, captions carry the wording)
# ----------------------------------------------------------------------------------------------------------------------


def figure_loads(body, n, kind, area, chord, out, label):
    """Loads per lap against the loop angle in the paper's style (pof_style): C_Z, C_m, aerodynamic power, tension."""
    sys.path.insert(0, str(TASK))
    import pof_style as ps
    import matplotlib.pyplot as plt
    ps.style()
    cf, cm = coefficients(body, area, chord)
    x = body["loop_angle_deg"] if kind == "loop" else (body["time_s"] % (n*(body["time_s"][1]-body["time_s"][0])))
    xlabel = "loop angle (deg)" if kind == "loop" else "time in the cycle (s)"
    periods = (len(body["step"])-1)//n
    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH_DOUBLE_IN, 4.6), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, max(periods, 1)))
    for k in range(periods):
        sl = slice(k*n+1, (k+1)*n+1)
        order = np.argsort(x[sl])
        kw = dict(color=colours[k], lw=1.1, label=f"{'lap' if kind == 'loop' else 'cycle'} {k+1}")
        axes[0, 0].plot(x[sl][order], cf[sl, 2][order], **kw)
        axes[0, 1].plot(x[sl][order], cm[sl, 1][order], **kw)
        axes[1, 0].plot(x[sl][order], body["power_W"][sl][order]/1e3, **kw)
        if "tension_N" in body:
            axes[1, 1].plot(x[sl][order], body["tension_N"][sl][order]/1e3, **kw)
    axes[0, 0].set_ylabel(r"$C_Z$"); axes[0, 1].set_ylabel(r"$C_m$"); axes[1, 0].set_ylabel("aerodynamic power (kW)")
    axes[1, 1].set_ylabel("tether tension at the kite (kN)")
    for ax, k in zip(axes.flat, "abcd"):
        ax.set_xlabel(xlabel); ps.grid(ax); ps.letter(ax, f"({k})")
        if kind == "loop":
            ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
    axes[0, 0].legend(loc="lower left")
    fig.savefig(out/f"{label}_loads.png", dpi=300); fig.savefig(out/f"{label}_loads.pdf"); plt.close(fig)


def figure_centre(centre, diameter, out, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([c["x_mean_m"] for c in centre])
    cz = np.array([c["centroid_m"][2] for c in centre]); cy = np.array([c["centroid_m"][1] for c in centre])
    spread = np.array([c["spread_m"] for c in centre])
    x0 = x[0]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    axes[0].plot((x-x0)/diameter, (cz-cz[0])/diameter, lw=1.2)
    axes[0].set_ylabel("descent of the wake centroid / D"); axes[0].set_xlabel("downstream distance / D")
    axes[1].plot((x-x0)/diameter, (cy-cy[0])/diameter, lw=1.2)
    axes[1].set_ylabel("lateral drift of the wake centroid / D"); axes[1].set_xlabel("downstream distance / D")
    axes[2].plot((x-x0)/diameter, spread/diameter, lw=1.2)
    axes[2].set_ylabel("rms radius of the trailing vorticity / D"); axes[2].set_xlabel("downstream distance / D")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(out/f"{label}_centreline.png", dpi=200); fig.savefig(out/f"{label}_centreline.pdf"); plt.close(fig)


def figure_planes(planes, out, label, diameter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = len(planes)
    fig, axes = plt.subplots(1, count, figsize=(3.6*count+1.2, 3.8), constrained_layout=True, squeeze=False)
    vmax = max(p["deficit_max"] for p in planes)
    vmin = -0.25*vmax
    for ax, p in zip(axes[0], planes):
        yc, zc = p["path_centre_yz"]
        im = ax.pcolormesh((p["y"]-yc)/diameter, (p["z"]-zc)/diameter, p["deficit"], cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        skip = max(1, p["y"].shape[0]//18)
        v = p["in_plane"][::skip, ::skip]
        speed = np.linalg.norm(v, axis=-1)
        cap = 0.15*p["wind_m_s"]                                        # vectors longer than this are drawn at this length
        v = v*np.minimum(1., cap/np.maximum(speed, 1e-12))[..., None]
        ax.quiver((p["y"][::skip, ::skip]-yc)/diameter, (p["z"][::skip, ::skip]-zc)/diameter, v[..., 0], v[..., 1],
                  color="k", scale=3.*cap, scale_units="width", width=0.004, alpha=0.6)
        ax.set_aspect("equal"); ax.set_xlabel("(y - y_c) / D"); ax.set_ylabel("(z - z_c) / D")
    fig.colorbar(im, ax=axes[0].tolist(), label=r"$1 - u/U_\infty$", shrink=0.9)
    fig.savefig(out/f"{label}_planes.png", dpi=200); fig.savefig(out/f"{label}_planes.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    for p in planes:
        ax.plot(np.array(p["radius_bins_m"])/diameter, p["profile"], lw=1.2, label=f"x/D = {p['x_m']/diameter:.2g}")
    ax.set_xlabel("distance from the wake centre / D"); ax.set_ylabel(r"azimuthal mean of $1 - u/U_\infty$"); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out/f"{label}_profiles.png", dpi=200); fig.savefig(out/f"{label}_profiles.pdf"); plt.close(fig)


def placed_surfaces(history, kind, state_file):
    """Quads of the lifting surfaces of every body at the pose of the stored step (from the case file and the motion samples)."""
    from openfawe.aero.aircraft import load_aircraft
    s = history["settings"]
    step = int(Path(state_file).stem.split("_")[1])
    quads = []
    if kind == "pair":
        motions = {name: (np.array(m["rotation"]), np.array(m["position_m"])) for name, m in history["motions"].items()}
        case = REPO/s["case"]
        bodies = list(motions)
    else:
        case = REPO/(s["case"] if kind == "pattern" else s["low_case"])
        motions = {"kite": (np.array(history["motion"]["rotation"]), np.array(history["motion"]["position_m"]))}
        bodies = ["kite"]
    definition = load_aircraft(case)
    wings = definition.wing.wings if hasattr(definition.wing, "wings") else (definition.wing,)
    for name in bodies:
        rotation, position = motions[name][0][step], motions[name][1][step]
        for wing in wings:
            le = wing.leading_edge@rotation.T+position
            te = wing.trailing_edge@rotation.T+position
            quads.append([[le[k], le[k+1], te[k+1], te[k]] for k in range(len(le)-1)])
    return quads


def figure_wake3d(nodes, edges, gamma, first, surfaces_placed, out, label, wind_m_s, view=(22., -62.), ring_first=None):
    """Quick matplotlib check of the wake (lifting surfaces only). The paper figures use wake_figures.render3d, which draws the
    complete airframe (author 2026-10-19 17:46)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    ring_first = len(nodes) if ring_first is None else ring_first
    rings = np.all(edges >= ring_first, axis=1)
    keep = ~((edges[:, 0] < first) & (edges[:, 1] < first)) & ~rings
    segments = nodes[edges[keep]]
    strength = np.abs(gamma[keep])
    scale = float(np.percentile(strength, 98)) or 1.
    colours = cm.Blues(0.3+0.7*np.clip(strength/scale, 0., 1.)); colours[:, 3] = 0.55
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.add_collection3d(Line3DCollection(segments, colors=colours, linewidths=0.35, zorder=3))
    if np.any(rings):                                   # rotor rings: thin grey lines, every second ring edge
        ax.add_collection3d(Line3DCollection(nodes[edges[rings][::2]], colors=(0.35, 0.35, 0.35, 0.35), linewidths=0.2, zorder=2))
    for quads in surfaces_placed:
        ax.add_collection3d(Poly3DCollection(quads, facecolor="#c23b22", edgecolor="#c23b22", lw=0.3, alpha=0.95, zorder=6))
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    mid, span = 0.5*(lo+hi), 0.55*max(hi-lo)
    ax.set_xlim(mid[0]-span, mid[0]+span); ax.set_ylim(mid[1]-span, mid[1]+span); ax.set_zlim(mid[2]-span, mid[2]+span)
    ax.quiver(mid[0]-span, mid[1]-0.8*span, mid[2]-0.8*span, 0.25*span, 0., 0., color="#2a9d4a", arrow_length_ratio=0.15, lw=1.5)
    ax.view_init(*view); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    fig.savefig(out/f"{label}_wake3d.png", dpi=220, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------------


def analyse(run_dir, out_dir, stations, plane_half, plane_count, backend, bodies=None, diameter=None):
    history, states, kind = read_run(run_dir)
    s = history["settings"]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n = period_steps(history, kind)
    area, chord = s["reference_area_m2"], s["reference_chord_m"]
    wind = s["wind_m_s"]
    dt = s["dt_s"]
    names = history["per_lap"].keys() if kind == "pair" else [None]
    summary = dict(run=str(run_dir), kind=kind, status=history["status"], settings={k: s[k] for k in s if k not in ("warmup", "aerodynamics", "loop", "motion")},
                   period_steps=n, bodies={})
    if diameter is None:
        if kind == "loop":
            diameter = 2.*float(s["loop"]["radius_m"]) if "loop" in s and "radius_m" in s["loop"] else 270.
        elif kind == "pair":
            diameter = float(s["loop_diameter_m"])
        else:
            p = s["pattern"]
            diameter = (2.*s["sphere_radius_m"]*np.sin(np.radians(p["angular_radius"])) if p["pattern"] == "circle"
                        else 2.*s["sphere_radius_m"]*np.cos(np.radians(p["elevation"]))*np.sin(np.radians(p["azimuth_amplitude"])))
    summary["diameter_m"] = float(diameter)
    for name in names:
        body = body_records(history, kind, name)
        label = Path(run_dir).name+("" if name is None else f"_{name}")
        rows = per_period(body, n, area, chord)
        summary["bodies"][name or "kite"] = dict(per_period=rows)
        figure_loads(body, n, kind, area, chord, out, label)
    # path centre (for the plane stations): mean position of the body over the last period
    body = body_records(history, kind, None if kind != "pair" else "A")
    centre_path = body["position_m"][-n:].mean(axis=0)
    summary["path_centre_m"] = centre_path.tolist()
    if states:
        nodes, edges, gamma, first, ring_first = load_state(states[-1])
        rows, circulation = wake_rows(*wing_part(nodes, edges, gamma, ring_first), first)
        centre = centre_line(rows, circulation, dt, wind)
        summary["centre_line"] = centre[::max(1, len(centre)//400)]
        figure_centre(centre, diameter, out, Path(run_dir).name)
        planes = []
        for station in stations:
            x = centre_path[0]+station*diameter
            yz = centre_path[1:]
            planes.append(plane_at(x, yz, plane_half, plane_count, nodes, edges, gamma, wind, s["core_radius_m"], backend))
        summary["planes"] = [{k: v for k, v in p.items() if k not in ("y", "z", "deficit", "in_plane", "vorticity", "u", "v")} for p in planes]
        np.savez_compressed(out/f"{Path(run_dir).name}_planes.npz", **{f"plane{i}_{k}": np.asarray(v) for i, p in enumerate(planes) for k, v in p.items() if k in ("y", "z", "deficit", "in_plane", "vorticity")})
        figure_planes(planes, out, Path(run_dir).name, diameter)
        figure_wake3d(nodes, edges, gamma, first, placed_surfaces(history, kind, states[-1]), out, Path(run_dir).name, wind, ring_first=ring_first)
    (out/f"{Path(run_dir).name}_summary.json").write_text(json.dumps(summary, indent=1, default=float)+"\n", encoding="utf-8")
    return summary


def main(argv):
    parser = argparse.ArgumentParser(prog="wake_analysis.py")
    parser.add_argument("run_dir"); parser.add_argument("out_dir")
    parser.add_argument("--planes", default="0.5,1,2,3", help="stations downstream of the path centre in diameters")
    parser.add_argument("--plane-half", type=float, default=250.)
    parser.add_argument("--plane-count", type=int, default=121)
    parser.add_argument("--backend", default="tree", choices=("tree", "numba", "numpy"))
    parser.add_argument("--diameter", type=float, default=None)
    args = parser.parse_args(argv)
    stations = [float(x) for x in args.planes.split(",") if x]
    summary = analyse(args.run_dir, args.out_dir, stations, args.plane_half, args.plane_count, args.backend, diameter=args.diameter)
    print(json.dumps(dict(kind=summary["kind"], status=summary["status"], diameter_m=summary["diameter_m"],
                          periods={k: len(v["per_period"]) for k, v in summary["bodies"].items()},
                          planes=[(p["x_m"], round(p["deficit_max"], 4)) for p in summary.get("planes", [])])))


if __name__ == "__main__":
    main(sys.argv[1:])
