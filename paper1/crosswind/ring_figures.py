"""Shape of the rotor vortex rings against their age (Sec. III E): the mean radius of each ring about its centroid and the
spread of its node radii (largest minus smallest), for the discs above and below the wing, from a stored wake state of a
``c3_pair.py`` run with rotors. Rings are stored newest first, eight discs per step in reverse disc order, ``nodes_per_ring`` nodes per ring.

Written: OUT/rings_age.json (per disc row: age in steps, median radius and spread over the discs of the row) and
OUT/rings_age.png/pdf (pof_style; the newest lap, ages up to ``--max-steps``).

Usage: ring_figures.py RUN_DIR OUT_DIR [--step N] [--max-steps 936] [--compare RUN_DIR ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
sys.path.insert(0, str(TASK))
import pof_style as ps  # noqa: E402
import wake_figures as wf  # noqa: E402


def ring_shapes(run, max_steps):
    s = run["settings"]["rotors"]
    per, discs = int(s["nodes_per_ring"]), int(s["discs"])
    nodes, rf = run["nodes"], run["ring_first"]
    count = (len(nodes)-rf)//per
    rings = nodes[rf:rf+count*per].reshape(count, per, 3)
    centre = rings.mean(axis=1)
    radius = np.linalg.norm(rings-centre[:, None, :], axis=2)
    rotor_set = json.loads((REPO/s["rotor_set"]).read_text(encoding="utf-8"))["rotors"]
    upper = np.array([d["centre_m"][2] > 0 for d in rotor_set])          # frame x downstream, y span, z up
    age = np.arange(count)//discs
    # RingWake.shed puts every new ring first and the discs are shed in rotor-set order (rotor/coupled.py), so within the
    # rings of one step the order is reversed: slot s holds disc discs-1-s (independent check 2026-10-20, state_03744)
    disc = discs-1-np.arange(count) % discs
    # check of that order: the newest rings sit at their disc centres in the body frame of the stored step
    motion = run["motions"]["A"]
    rot, pos = np.asarray(motion["rotation"][run["step"]]), np.asarray(motion["position_m"][run["step"]])
    body = (centre[:discs]-pos) @ rot                                   # inertial -> body (rotation maps body to inertial)
    expected = np.array([rotor_set[k]["centre_m"] for k in disc[:discs]])
    if np.max(np.abs(body-expected)) > 0.05:
        raise ValueError(f"newest rings are not at the disc centres in the assumed order (largest offset {np.max(np.abs(body-expected)):.3f} m)")
    out = {}
    for name, rows in (("upper", upper), ("lower", ~upper)):
        ages, mean_r, spread = [], [], []
        # medians over the discs of the row and over logarithmic bins of age (1, 2, 3, 4, 5, 6-7, 8-9, ...), age 0 excluded
        top = min(max_steps, int(age.max()))
        edges = np.unique(np.round(np.logspace(0, np.log10(top+1), 40)).astype(int))
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (age >= lo) & (age < hi) & rows[disc]
            if not sel.any():
                continue
            ages.append(float(np.sqrt(lo*(hi-1)) if hi-1 > lo else lo))
            mean_r.append(float(np.median(radius[sel].mean(axis=1))))
            spread.append(float(np.median(radius[sel].max(axis=1)-radius[sel].min(axis=1))))
        out[name] = dict(age_steps=ages, radius_m=mean_r, spread_m=spread)
    out["disc_radius_m"] = float(radius[:discs].mean())
    out["nodes_per_ring"], out["dt_s"] = per, float(run["dt"])
    out["ring_core_m"] = s.get("ring_core_m")
    out["ring_bound_core_m"] = s.get("ring_bound_core_m")
    out["ring_surface_overlap"] = s.get("ring_surface_overlap")
    return out


def figure(curves, out):
    import matplotlib.pyplot as plt
    ps.style()
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE_IN, 2.6), constrained_layout=True)
    styles = ["-", ps.DASH, ":", "-."]
    for k, (label, data) in enumerate(curves.items()):
        for name, colour in (("upper", ps.COL["outer"]), ("lower", ps.COL["inner"])):
            c = data[name]
            t = np.asarray(c["age_steps"])*data["dt_s"]
            kw = dict(color=colour, ls=styles[k % len(styles)], lw=1.2, label=f"{name} discs" + (f", {label}" if len(curves) > 1 else ""))
            axes[0].plot(t, np.asarray(c["radius_m"])/data["disc_radius_m"], **kw)
            axes[1].plot(t, np.asarray(c["spread_m"])/data["disc_radius_m"], **kw)
    axes[0].set_ylabel(r"mean ring radius / $R_{\mathrm{disc}}$")
    axes[1].set_ylabel(r"spread of node radii / $R_{\mathrm{disc}}$")
    for ax, k in zip(axes, "ab"):
        ax.set_xlabel("ring age (s)")
        ax.set_xscale("log")
        ps.grid(ax)
        ps.letter(ax, f"({k})")
    axes[1].set_yscale("log")
    axes[0].legend(loc="upper left", fontsize=8)
    ps.save(fig, Path(out)/"rings_age")


def main(argv):
    parser = argparse.ArgumentParser(prog="ring_figures.py")
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=936)
    parser.add_argument("--label", default="r1")
    parser.add_argument("--compare", nargs="*", default=[], help="LABEL=RUN_DIR pairs drawn with the first run")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curves = {args.label: ring_shapes(wf.read(args.run, args.step), args.max_steps)}
    for item in args.compare:
        label, path = item.split("=", 1)
        curves[label] = ring_shapes(wf.read(path), args.max_steps)
    (out/"rings_age.json").write_text(json.dumps(curves, indent=1)+"\n", encoding="utf-8")
    figure(curves, out)
    for label, data in curves.items():
        for name in ("upper", "lower"):
            c = data[name]
            pick = {round(a, 1): (round(r, 2), round(s, 2)) for a, r, s in zip(c["age_steps"], c["radius_m"], c["spread_m"])}
            print(label, name, pick)


if __name__ == "__main__":
    main(sys.argv[1:])
