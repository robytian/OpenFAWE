"""Paper figures and tables of Sec. III E (the eight rotors on one M600) and III F (two M600 units) from the receipts of
``c3_pair.py`` (``checks.json``: settings, verdicts, per-lap table, per-step loads of the last three laps and the wake
summary with the cross-flow deficit planes of the final wake). Style of ``pof_style.py``: no text inside the plot area
apart from axes, ticks, legends and panel letters; wording goes into the captions.

Written:
  rotor_loads.png/pdf     single unit without and with rotors over the last lap both runs completed (the same lap
                          number) against the loop angle: (a) C_Z, (b) the
                          change of the wing and tail coefficients C_X, C_Z (with minus without), (c) aerodynamic power of
                          wing and tails and the rotor shaft power, (d) total rotor thrust
  rotor_planes.png/pdf    axial deficit 1 - u/U on the cross-flow planes of the final wake, without (top row) and with
                          rotors (bottom row), and the azimuthal mean profiles
  array_ratios.png/pdf    pairs: lap means of unit B over unit A lap by lap (aerodynamic power, lift), and C_Z of B minus
                          C_Z of A over the last lap against the loop angle, without (solid) and with rotors (dashed)
  array_planes.png/pdf    pairs: deficit planes of the final wake, one row per run
  c3_table.json / .md     per run: K3, K7 (last pair), last-lap means of each unit, ratios B/A, ring proximity counts,
                          plane deficits, elapsed time

The loop angle is recomputed from the run settings with the loop of ``c3_pair.py`` (cached in OUT/loop_angle_*.npy; the
receipts hold steps, not angles). Angle convention of the paper: 0 deg at the +y side, 90 deg at the bottom; the kite
flies towards decreasing angle.

Usage: c3_figures.py OUT_DIR RECEIPT_DIR [RECEIPT_DIR ...] [--stations 0.5,1,1.5,2]
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

AREA_M2, CHORD_M = 32.9285, 1.2831


def load(directory):
    directory = Path(directory)
    data = json.loads((directory/"checks.json").read_text(encoding="utf-8"))
    data["_name"] = directory.name
    return data


def kind(data):
    """(single, spacing, offset, rotors) of a run."""
    s = data["settings"]
    rotors = bool(s.get("rotors"))
    if s["single"]:
        return ("single", None, None, rotors)
    return ("pair", float(s["spacing_diameters"]), float(s["offset_diameters"]), rotors)


def loop_angle_deg(settings, cache_dir):
    """Loop angle [deg, 0..360) of every step of the run, with the loop of c3_pair.py (cached)."""
    n, laps = int(settings["steps_per_lap"]), int(settings["laps"])
    key = f"loop_angle_w{settings['wind_m_s']:g}_s{settings['strain']:g}_n{n}_l{laps}.npy"
    path = Path(cache_dir)/key
    if path.exists():
        return np.load(path)
    import c3_pair as cp
    from openfawe.flight import crosswind_loop as cl
    entries, _, playbook = cl.load_playbook(REPO/cp.PLAYBOOK)
    constants = cl.LoopConstants.from_case(playbook)
    entry = cl.playbook_entry(entries, float(settings["wind_m_s"]))
    sphere = constants.tether_length_m*(1.0+float(settings["strain"]))+constants.bridle_radius_m
    centre = cl.NominalLoop.sphere_centre_distance(sphere, entry["path_radius_target"])
    loop = cl.NominalLoop(entry, [float(settings["wind_m_s"]), 0.0, 0.0], constants, centre_distance_m=centre, interpolation="spline",
                          roll_limit_tension_N=20000.0)
    _, table = loop.motion(laps+1, n)
    angle = np.degrees(np.asarray(table["loop_angle_rad"], dtype=float)) % 360.0
    np.save(path, angle)
    return angle


def complete_laps(data):
    return (int(data["frames"])-1)//int(data["settings"]["steps_per_lap"])


def last_lap(data, cache_dir, lap=None):
    """Per-step series of every unit over lap ``lap`` (default the last complete lap; all steps when no lap is complete:
    smoke runs)."""
    s = data["settings"]
    n = int(s["steps_per_lap"])
    done = int(data["frames"])-1
    laps_done = done//n if lap is None else min(int(lap), done//n)
    first, last = ((laps_done-1)*n+1, laps_done*n) if laps_done >= 1 else (0, done)
    angle = loop_angle_deg(s, cache_dir)
    out = {}
    for name, d in data["per_step"].items():
        steps = np.asarray(d["step"])
        keep = (steps >= first) & (steps <= last)
        unit = {}
        for k, v in d.items():
            if k == "step" or len(v) != len(steps) or not v or v[0] is None:
                continue
            unit[k] = np.asarray(v, dtype=float)[keep]
        unit["step"] = steps[keep]
        unit["loop_angle_deg"] = angle[np.minimum(steps[keep], len(angle)-1)]
        q = 0.5*float(s["density_kg_m3"])*unit["airspeed_m_s"]**2
        unit["CF"] = unit["force_body_N"]/(q[:, None]*AREA_M2)
        unit["Cm"] = unit["moment_origin_body_Nm"]/(q[:, None]*AREA_M2*CHORD_M)
        out[name] = unit
    return out


def ordered(unit, key, column=None):
    order = np.argsort(unit["loop_angle_deg"])
    values = unit[key] if column is None else unit[key][:, column]
    return unit["loop_angle_deg"][order], values[order]


def angle_axis(ax):
    ax.set_xlabel("loop angle (deg)")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    ps.grid(ax)


def figure_rotor_loads(without, with_rotors, cache_dir, out):
    import matplotlib.pyplot as plt
    ps.style()
    lap = min(complete_laps(without), complete_laps(with_rotors)) or None   # the same lap of both runs
    a, b = last_lap(without, cache_dir, lap)["A"], last_lap(with_rotors, cache_dir, lap)["A"]
    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH_DOUBLE_IN, 4.6), constrained_layout=True)
    no_kw = dict(color=ps.COL["no rotors"], lw=1.2, label="without rotors")
    ro_kw = dict(color=ps.COL["rotors"], lw=1.2, label="with rotors")
    axes[0, 0].plot(*ordered(a, "CF", 2), **no_kw)
    axes[0, 0].plot(*ordered(b, "CF", 2), **ro_kw)
    axes[0, 0].set_ylabel(r"$C_Z$")
    axes[0, 0].legend(loc="best")
    common = np.intersect1d(a["step"], b["step"])
    ia, ib = np.searchsorted(a["step"], common), np.searchsorted(b["step"], common)
    angle = a["loop_angle_deg"][ia]
    order = np.argsort(angle)
    axes[0, 1].plot(angle[order], (b["CF"][ib, 2]-a["CF"][ia, 2])[order], color="k", lw=1.2, label=r"$\Delta C_Z$")
    axes[0, 1].plot(angle[order], (b["CF"][ib, 0]-a["CF"][ia, 0])[order], color=ps.COL["grey"], lw=1.2, ls=ps.DASH, label=r"$\Delta C_X$")
    axes[0, 1].axhline(0.0, color="0.6", lw=0.6)
    axes[0, 1].set_ylabel("change with rotors")
    axes[0, 1].legend(loc="best")
    for unit, kw, label in ((a, no_kw, "wing and tails, without rotors"), (b, ro_kw, "wing and tails, with rotors")):
        x, y = ordered(unit, "aero_power_W")
        axes[1, 0].plot(x, y/1e3, **dict(kw, label=label))
    if "rotor_shaft_power_W" in b:
        x, y = ordered(b, "rotor_shaft_power_W")
        axes[1, 0].plot(x, y/1e3, color=ps.COL["rotors"], lw=1.2, ls=ps.DASH, label="rotor shaft")
        x, y = ordered(b, "rotor_thrust_total_N")
        axes[1, 1].plot(x, y/1e3, color=ps.COL["rotors"], lw=1.2)
    axes[1, 0].set_ylabel("power (kW)")
    axes[1, 0].legend(loc="best", fontsize=8)
    axes[1, 1].set_ylabel("total rotor thrust (kN)")
    for ax, k in zip(axes.flat, "abcd"):
        angle_axis(ax)
        ps.letter(ax, f"({k})")
    ps.save(fig, Path(out)/"rotor_loads")


def plane_panels(axes_row, data, stations, vmax, diameter_key="diameter_m"):
    """Deficit planes of one run at the given stations into a row of axes (the last axis: azimuthal profiles)."""
    wake = data["wake"]
    diameter = wake[diameter_key]
    mesh = None
    by_station = {round(p["station_diameters"], 3): p for p in wake["planes"]}
    for ax, station in zip(axes_row[:-1], stations):
        p = by_station.get(round(station, 3))
        if p is None:
            ax.set_axis_off()
            continue
        # u, v: in-plane coordinates about the plane's centre, the mean position of A's path over the last lap
        u, v = np.asarray(p["u"])/diameter, np.asarray(p["v"])/diameter
        mesh = ax.pcolormesh(u, v, np.asarray(p["deficit"]), vmin=-0.25*vmax, vmax=vmax, cmap="viridis",
                             shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(r"$(y-y_c)/D$")
    prof = axes_row[-1]
    import matplotlib.pyplot as plt
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(stations), 1)))
    for colour, station in zip(colours, stations):
        p = by_station.get(round(station, 3))
        if p is None:
            continue
        radius = np.asarray(p["radius_bins_m"], dtype=float)/diameter
        values = np.array([np.nan if x is None else x for x in p["profile"]], dtype=float)
        prof.plot(radius[:len(values)], values, color=colour, lw=1.1, label=f"{station:g} D")
    prof.set_xlabel(r"$r/D$")
    prof.set_ylabel(r"$\langle 1-u/U\rangle_\theta$")
    ps.grid(prof)
    return mesh


def figure_planes(rows, stations, out, stem):
    import matplotlib.pyplot as plt
    ps.style()
    rows = [d for d in rows if d.get("wake") and not d["wake"].get("error")]
    if not rows:
        return
    # colour limit: the 99.5th percentile of the deficit over all planes shown (the planes use the run's core of 0.15 m,
    # so a filament passing close to a grid point gives an isolated spike that would otherwise set the scale)
    shown = [np.asarray(p["deficit"], dtype=float) for d in rows for p in d["wake"]["planes"]
             if round(p["station_diameters"], 3) in {round(x, 3) for x in stations}]
    vmax = float(np.percentile(np.concatenate([a.ravel() for a in shown]), 99.5)) if shown else 1.0
    vmax = vmax if vmax > 0 else 1.0
    count = len(stations)
    fig, axes = plt.subplots(len(rows), count+1, figsize=(ps.WIDTH_DOUBLE_IN, 1.75*len(rows)+0.5), squeeze=False,
                             constrained_layout=True, gridspec_kw=dict(width_ratios=[1]*count+[1.25]))
    mesh = None
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, d in enumerate(rows):
        mesh = plane_panels(axes[r], d, stations, vmax) or mesh
        axes[r, 0].set_ylabel(r"$(z-z_c)/D$")
        for ax in axes[r]:
            if ax.axison:
                ps.letter(ax, f"({next(letters)})")
        if r == 0:
            axes[r, -1].legend(loc="best", fontsize=7)
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=axes[:, :-1], shrink=0.9, pad=0.01)
        bar.set_label(r"$1-u/U$")
    ps.save(fig, Path(out)/stem)


def figure_array(pairs, cache_dir, out):
    import matplotlib.pyplot as plt
    ps.style()
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_DOUBLE_IN, 2.5), constrained_layout=True)
    colours = {}
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for d in pairs:
        _, spacing, offset, rotors = kind(d)
        config = (spacing, offset)
        colour = colours.setdefault(config, palette[len(colours) % len(palette)])
        label = f"{spacing:g} D" + (f", {offset:g} D sideways" if offset else "") + (", rotors" if rotors else "")
        kw = dict(color=colour, lw=1.2, ls=ps.DASH if rotors else "-", label=label)
        rel = d.get("b_relative_to_a") or []
        if rel:
            laps = [r["lap"] for r in rel]
            axes[0].plot(laps, [r["power_ratio_b_over_a"] for r in rel], marker="o", ms=3, **kw)
            axes[1].plot(laps, [r["lift_ratio_b_over_a"] for r in rel], marker="o", ms=3, **kw)
        lap = last_lap(d, cache_dir)
        if "A" in lap and "B" in lap and float(d["settings"].get("phase_deg", 0.0)) == 0.0:   # equal loop angle at equal step
            a, b = lap["A"], lap["B"]
            common = np.intersect1d(a["step"], b["step"])
            ia, ib = np.searchsorted(a["step"], common), np.searchsorted(b["step"], common)
            angle = a["loop_angle_deg"][ia]
            order = np.argsort(angle)
            axes[2].plot(angle[order], (b["CF"][ib, 2]-a["CF"][ia, 2])[order], **kw)
    axes[0].set_ylabel(r"$\bar P_B/\bar P_A$")
    axes[1].set_ylabel(r"$\bar L_B/\bar L_A$")
    for ax in axes[:2]:
        ax.set_xlabel("lap")
        ps.grid(ax)
    axes[2].set_ylabel(r"$C_{Z,B}-C_{Z,A}$")
    angle_axis(axes[2])
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="best", fontsize=7)
    else:
        axes[2].legend(loc="best", fontsize=7)
    for ax, k in zip(axes, "abc"):
        ps.letter(ax, f"({k})")
    ps.save(fig, Path(out)/"array_ratios")


def table_row(data):
    single, spacing, offset, rotors = kind(data)
    row = dict(run=data["_name"], kind=single, spacing_D=spacing, offset_D=offset, rotors=rotors, status=data["status"],
               laps=data["settings"]["laps"], frames=data["frames"], backend=data["settings"]["backend"],
               elapsed_h=round(data["elapsed_s"]/3600.0, 2),
               K3={k: v["worst_relative"] for k, v in data["K3"].items()},
               K7={k: v["verdict"] for k, v in data["K7"].items()},
               K7_last={k: (v[-1] if v else None) for k, v in data["K7_pairs"].items()})
    units = {}
    for name, laps in data["per_lap"].items():
        if not laps:
            continue
        last = laps[-1]
        units[name] = dict(lap=last["lap"], CZ=last["force_body_coefficient_mean"][2], CX=last["force_body_coefficient_mean"][0],
                           Cm=last["moment_body_coefficient_mean"][1], lift_kN=last["lift_body_z_mean_N"]/1e3,
                           aero_power_kW=last["aero_power_mean_W"]/1e3,
                           tension_kN=None if "tether_tension_kite_N" not in last else last["tether_tension_kite_N"]["mean"]/1e3,
                           shaft_power_kW=None if "rotor_shaft_power_W" not in last else last["rotor_shaft_power_W"]["mean"]/1e3,
                           thrust_kN=None if "rotor_thrust_total_N" not in last else last["rotor_thrust_total_N"]["mean"]/1e3,
                           rotor_power_on_body_kW=None if "rotor_power_on_body_mean_W" not in last else last["rotor_power_on_body_mean_W"]/1e3,
                           ring_core=last.get("ring_nodes_within_surface_core"))
    row["units"] = units
    if data.get("b_relative_to_a"):
        row["b_over_a"] = data["b_relative_to_a"]
    wake = data.get("wake")
    if wake and not wake.get("error"):
        row["planes"] = [dict(station_D=p["station_diameters"], deficit_max=p["deficit_max"], deficit_mean_positive=p["deficit_mean_positive"],
                              momentum_deficit_m2_s=p["momentum_deficit_m2_s"]) for p in wake["planes"]]
        row["ring_nodes"] = wake.get("ring_nodes")
    elif wake:
        row["wake_error"] = wake.get("error")
    return row


def markdown(rows):
    f = lambda x, fmt="{:.4f}": "–" if x is None else fmt.format(x)   # noqa: E731
    lines = ["| run | rotors | spacing | offset | laps | K7 last pair | C_Z A | C_Z B | P A (kW) | P_B/P_A | L_B/L_A | shaft A (kW) | shaft_B/shaft_A | elapsed (h) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = r["units"].get("A", {}), r["units"].get("B", {})
        rel = (r.get("b_over_a") or [{}])[-1]
        k7 = ", ".join(f"{k}: {v['verdict'] if isinstance(v, dict) and 'verdict' in v else v}" for k, v in r["K7"].items())
        lines.append(f"| {r['run']} | {'yes' if r['rotors'] else 'no'} | {f(r['spacing_D'], '{:g}')} | {f(r['offset_D'], '{:g}')} | {r['laps']} | {k7} | "
                     f"{f(a.get('CZ'))} | {f(b.get('CZ'))} | {f(a.get('aero_power_kW'), '{:.1f}')} | {f(rel.get('power_ratio_b_over_a'))} | "
                     f"{f(rel.get('lift_ratio_b_over_a'))} | {f(a.get('shaft_power_kW'), '{:.1f}')} | {f(rel.get('shaft_power_ratio_b_over_a'))} | "
                     f"{r['elapsed_h']:.1f} |")
    return "\n".join(lines)+"\n"


def main(argv):
    parser = argparse.ArgumentParser(prog="c3_figures.py")
    parser.add_argument("out")
    parser.add_argument("receipts", nargs="+")
    parser.add_argument("--stations", default="0.5,1,1.5,2")
    parser.add_argument("--pair-stations", default=None, help="stations of the pair planes (default: those of each receipt)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datas = [load(d) for d in args.receipts]
    rows = [table_row(d) for d in datas]
    (out/"c3_table.json").write_text(json.dumps(rows, indent=1, default=float)+"\n", encoding="utf-8")
    (out/"c3_table.md").write_text(markdown(rows), encoding="utf-8")
    singles = {kind(d)[3]: d for d in datas if kind(d)[0] == "single"}
    stations = [float(v) for v in args.stations.split(",") if v.strip()]
    if False in singles and True in singles:
        figure_rotor_loads(singles[False], singles[True], out, out)
        figure_planes([singles[False], singles[True]], stations, out, "rotor_planes")
    pairs = sorted((d for d in datas if kind(d)[0] == "pair"), key=lambda d: (kind(d)[1], kind(d)[2], kind(d)[3]))
    if pairs:
        figure_array(pairs, out, out)
        if args.pair_stations:
            pair_stations = [float(v) for v in args.pair_stations.split(",")]
        else:
            pair_stations = sorted({p["station_diameters"] for d in pairs if d.get("wake") and not d["wake"].get("error")
                                    for p in d["wake"]["planes"]})[:4]
        figure_planes(pairs, pair_stations, out, "array_planes")
    print(markdown(rows))


if __name__ == "__main__":
    main(sys.argv[1:])
