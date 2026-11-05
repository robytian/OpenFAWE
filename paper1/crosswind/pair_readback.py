"""Read-back of the OF-0009 runs of ``c3_pair.py`` (single units and pairs, with and without rotors) from their receipts.

Node 2 commits only ``checks.json`` (settings, verdicts, per-lap table, per-step loads of the last three laps and every fourth
step before, and the ``wake`` summary: centre lines per unit and cross-flow deficit planes of the final wake). This script
reads any number of such receipts and writes

  table.json / table.md   per run: K3, K7, lap-averaged loads of the last lap per unit, rotor shaft power and thrust, ring
                          crossings, and for pairs the ratios of unit B to unit A (lift, aerodynamic power, tension, shaft power);
  <run>_loads.png         lift coefficient, aerodynamic power, tension (and shaft power with rotors) of each unit over the
                          last lap against the loop angle;
  <run>_planes.png        axial deficit on the cross-flow planes of the final wake (from the receipt), coordinates from A's
                          path centre in loop diameters, with the azimuthal profiles;
  compare_<a>_<b>.png     two runs side by side (for example without and with rotors): the planes and the profiles.

Figures carry no text inside the plot area apart from axes, ticks and legends (author rule); captions carry the wording.

Usage: pair_readback.py OUT_DIR RECEIPT_DIR [RECEIPT_DIR ...] [--compare NAME_A,NAME_B ...]
       A receipt directory holds checks.json (a node-2 results folder or a run folder).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AREA_M2 = 32.9285


def load(directory):
    directory = Path(directory)
    data = json.loads((directory/"checks.json").read_text(encoding="utf-8"))
    data["_name"] = directory.name
    return data


def last_lap(data):
    s = data["settings"]
    n, laps = s["steps_per_lap"], s["laps"]
    first = (laps-1)*n+1
    out = {}
    for name, d in data["per_step"].items():
        steps = np.array(d["step"])
        keep = (steps >= first) & (steps <= laps*n)
        unit = {k: np.asarray(v, dtype=float)[keep] for k, v in d.items() if k != "step" and len(v) == len(steps) and v and v[0] is not None}
        unit["step"] = steps[keep]
        unit["phase"] = (steps[keep]-first)/n
        out[name] = unit
    return out


def summary_row(data):
    s = data["settings"]
    row = dict(run=data["_name"], status=data["status"], spacing=None if s["single"] else s["spacing_diameters"],
               offset=None if s["single"] else s["offset_diameters"], rotors=None if not s.get("rotors") else s["rotors"]["omega_rad_s"],
               laps=s["laps"], backend=s["backend"], free_rows=s["free_convection_rows"],
               K3={k: v["worst_relative"] for k, v in data["K3"].items()}, K7={k: v["verdict"] for k, v in data["K7"].items()},
               K7_last={k: (v[-1] if v else None) for k, v in data["K7_pairs"].items()}, elapsed_h=data["elapsed_s"]/3600.)
    units = {}
    for name, laps in data["per_lap"].items():
        if not laps:
            continue
        last = laps[-1]
        units[name] = dict(lap=last["lap"], lift_N=last["lift_body_z_mean_N"], aero_power_W=last["aero_power_mean_W"],
                           tension_N=last.get("tether_tension_kite_N", {}).get("mean"), CZ=last["force_body_coefficient_mean"][2],
                           shaft_power_W=last.get("rotor_shaft_power_W", {}).get("mean"), thrust_N=last.get("rotor_thrust_total_N", {}).get("mean"),
                           ring_steps_in_core=last.get("ring_nodes_within_surface_core", {}).get("steps_nonzero"))
    row["units"] = units
    if data.get("b_relative_to_a"):
        row["b_over_a_last_lap"] = data["b_relative_to_a"][-1]
    if data.get("wake"):
        row["planes"] = [dict(station=p["station_diameters"], deficit_max=p["deficit_max"], deficit_mean_positive=p["deficit_mean_positive"],
                              momentum_deficit_m2_s=p["momentum_deficit_m2_s"]) for p in data["wake"]["planes"]]
    return row


def figure_loads(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lap = last_lap(data)
    rotors = any("rotor_shaft_power_W" in u for u in lap.values())
    panels = 4 if rotors else 3
    fig, axes = plt.subplots(1, panels, figsize=(3.2*panels, 2.8), constrained_layout=True)
    for name, unit in lap.items():
        q = None
        if "airspeed_m_s" in unit:
            q = 0.5*1.225*unit["airspeed_m_s"]**2
        force = unit["force_body_N"]
        phase = 360.*unit["phase"]
        axes[0].plot(phase, -force[:, 2]/(q*AREA_M2) if q is not None else -force[:, 2], label=f"unit {name}")
        axes[1].plot(phase, unit["aero_power_W"]/1e3, label=f"unit {name}")
        if "tension_N" in unit:
            axes[2].plot(phase, unit["tension_N"]/1e3, label=f"unit {name}")
        if rotors and "rotor_shaft_power_W" in unit:
            axes[3].plot(phase, unit["rotor_shaft_power_W"]/1e3, label=f"unit {name}")
    labels = ["lift coefficient", "aerodynamic power [kW]", "tether tension [kN]", "rotor shaft power [kW]"]
    for ax, label in zip(axes, labels):
        ax.set_xlabel("loop angle [deg]")
        ax.set_ylabel(label)
        ax.set_xlim(0, 360)
    axes[0].legend(frameon=False)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_planes(datas, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [d for d in datas if d.get("wake")]
    if not rows:
        return
    count = max(len(d["wake"]["planes"]) for d in rows)
    fig, axes = plt.subplots(len(rows), count+1, figsize=(2.6*(count+1), 2.5*len(rows)), squeeze=False, constrained_layout=True)
    vmax = max(max(p["deficit_max"] for p in d["wake"]["planes"]) for d in rows)
    for r, d in enumerate(rows):
        diameter = d["wake"]["diameter_m"]
        centre = np.asarray(d["wake"]["path_centre_m"])
        for c, p in enumerate(d["wake"]["planes"]):
            ax = axes[r, c]
            u, v = np.asarray(p["u"]), np.asarray(p["v"])
            deficit = np.asarray(p["deficit"])
            ax.pcolormesh(u/diameter, v/diameter, deficit, vmin=-0.1*vmax, vmax=vmax, cmap="viridis", shading="auto")
            ax.set_aspect("equal")
            ax.set_xlabel("y / D")
            if c == 0:
                ax.set_ylabel("z / D")
            ax.set_title(f"x = {p['station_diameters']:g} D", fontsize=9)
            prof = axes[r, -1]
            radius = np.asarray(p["radius_bins_m"])/diameter
            values = np.array([np.nan if x is None else x for x in p["profile"]])
            prof.plot(radius, values, label=f"{p['station_diameters']:g} D")
        axes[r, -1].set_xlabel("r / D")
        axes[r, -1].set_ylabel("azimuthal mean deficit")
        axes[r, -1].legend(frameon=False, fontsize=7)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def markdown(rows):
    lines = ["| run | rotors | spacing | offset | laps | K7 | CZ A | CZ B | L_B/L_A | P_B/P_A | T_B/T_A | Pshaft_B/Pshaft_A | shaft A [kW] | elapsed [h] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = r["units"].get("A", {}), r["units"].get("B", {})
        rel = r.get("b_over_a_last_lap") or {}
        f = lambda x, fmt="{:.4f}": "–" if x is None else fmt.format(x)   # noqa: E731
        lines.append(f"| {r['run']} | {r['rotors'] or 'no'} | {r['spacing'] or '–'} | {r['offset'] or '–'} | {r['laps']} | "
                     f"{', '.join(f'{k}: {v}' for k, v in r['K7'].items())} | {f(a.get('CZ'))} | {f(b.get('CZ'))} | "
                     f"{f(rel.get('lift_ratio_b_over_a'))} | {f(rel.get('power_ratio_b_over_a'))} | {f(rel.get('tension_ratio_b_over_a'))} | "
                     f"{f(rel.get('shaft_power_ratio_b_over_a'))} | {f(None if a.get('shaft_power_W') is None else a['shaft_power_W']/1e3, '{:.0f}')} | {r['elapsed_h']:.1f} |")
    return "\n".join(lines)+"\n"


def main(argv):
    parser = argparse.ArgumentParser(prog="pair_readback.py")
    parser.add_argument("out")
    parser.add_argument("receipts", nargs="+")
    parser.add_argument("--compare", action="append", default=[], help="NAME_A,NAME_B: planes of two runs one above the other")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datas = [load(d) for d in args.receipts]
    rows = [summary_row(d) for d in datas]
    (out/"table.json").write_text(json.dumps(rows, indent=1, default=float)+"\n", encoding="utf-8")
    (out/"table.md").write_text(markdown(rows), encoding="utf-8")
    for d in datas:
        if d["status"] == "complete" and d["per_step"]:
            figure_loads(d, out/f"{d['_name']}_loads.png")
        figure_planes([d], out/f"{d['_name']}_planes.png")
    by_name = {d["_name"]: d for d in datas}
    for pair in args.compare:
        a, b = pair.split(",")
        figure_planes([by_name[a], by_name[b]], out/f"compare_{a}_{b}.png")
    print(markdown(rows))


if __name__ == "__main__":
    main(sys.argv[1:])
