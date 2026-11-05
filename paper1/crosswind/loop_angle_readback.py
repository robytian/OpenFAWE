"""Loop-angle positions of the along-the-loop results of Secs. III E and III F (OF-0009), with the controller's loop
angle of every step (the motion table of c3_pair.py, ``c3_figures.loop_angle_deg``) instead of the fraction of the lap
in time.

Why: ``rotor_figures_pof.last_lap`` drew FIG. 17 against 360 (1 - k/n) and ``k7_readback`` reported the loop angle of a
largest value as 360 k/n, both with k the step within the lap. The loop angle decreases in time on this loop (clockwise
seen from the ground station) and does not advance uniformly: it differs from 360 (1 - k/n) by up to 23 deg (checked
2026-10-26 on the 9 m/s loop, 936 steps per lap). The magnitudes read back before are unchanged; only the angles at which
they occur are recomputed here.

Per single-unit pair (with rotors, without rotors; same time step, last lap of each): the change of C_Z (and of the lift
in per cent) and of C_m along the loop, with the loop angles of their extremes, and the shaft power and thrust of the
rotors with the angles of theirs. Per two-unit run: the change of C_Z of B against A along the last lap (extremes and
the angle ranges where the lift of B is lower by more than a given fraction), and the step of the largest lap-to-lap
change of the total load and of the wing load (K7, K7a) with its loop angle and the total and aerodynamic moments there.

Usage: loop_angle_readback.py OUT.json CACHE_DIR --single WITH WITHOUT [--single ...] --pair RUN [--pair ...]
       (WITH, WITHOUT and RUN are directories holding checks.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parents[1]/"src"))
import c3_figures as cf  # noqa: E402

RHO, AREA, CHORD = 1.225, 32.9285, 1.2831


def load(run_dir):
    return json.loads((Path(run_dir)/"checks.json").read_text(encoding="utf-8"))


def lap(data, name, cache, which=None):
    """Per-step series of unit ``name`` over the last lap (or lap ``which``) with the controller's loop angle."""
    s = data["settings"]
    n, laps = int(s["steps_per_lap"]), int(s["laps"]) if which is None else int(which)
    d = data["per_step"][name]
    steps = np.asarray(d["step"])
    keep = (steps > (laps-1)*n) & (steps <= laps*n)
    angle = cf.loop_angle_deg(s, cache)
    q = 0.5*RHO*np.asarray(d["airspeed_m_s"], float)[keep]**2
    f = np.asarray(d["force_body_N"], float)[keep]
    m = np.asarray(d["moment_origin_body_Nm"], float)[keep]
    out = dict(step=steps[keep], angle=angle[steps[keep]], CZ=f[:, 2]/(q*AREA), Cm=m[:, 1]/(q*AREA*CHORD),
               lift=f[:, 2], P=np.asarray(d["aero_power_W"], float)[keep], moment=m)
    if d.get("rotor_shaft_power_W") and len(d["rotor_shaft_power_W"]) == len(steps):
        out["shaft"] = np.asarray(d["rotor_shaft_power_W"], float)[keep]
        out["thrust"] = np.asarray(d["rotor_thrust_total_N"], float)[keep]
    return out


def at(series, key, index):
    return dict(value=float(series[key][index]), angle_deg=round(float(series["angle"][index]), 1))


def single(with_dir, without_dir, cache):
    a, b = load(with_dir), load(without_dir)
    la, lb = lap(a, "A", cache), lap(b, "A", cache)
    if len(la["angle"]) != len(lb["angle"]) or np.abs(la["angle"]-lb["angle"]).max() > 1e-6:
        raise SystemExit(f"{with_dir}: steps or loop angles differ from {without_dir}")
    dCZ, dCm = la["CZ"]-lb["CZ"], la["Cm"]-lb["Cm"]
    lift = la["lift"]/lb["lift"]-1.
    out = dict(run=Path(with_dir).name, reference=Path(without_dir).name, laps=a["settings"]["laps"], dt_s=a["settings"]["dt_s"],
               mean_dCZ=float(dCZ.mean()), mean_dCm=float(dCm.mean()),
               lift_change_percent=dict(min=100*float(lift.min()), min_angle_deg=round(float(la["angle"][lift.argmin()]), 1),
                                        max=100*float(lift.max()), max_angle_deg=round(float(la["angle"][lift.argmax()]), 1),
                                        mean_of_lap_lift=100*float(la["lift"].mean()/lb["lift"].mean()-1.)),
               dCm=dict(min=float(dCm.min()), min_angle_deg=round(float(la["angle"][dCm.argmin()]), 1),
                        max=float(dCm.max()), max_angle_deg=round(float(la["angle"][dCm.argmax()]), 1)),
               dP_kW=float((la["P"]-lb["P"]).mean()/1e3))
    if "shaft" in la:
        out["shaft_kW"] = dict(mean=float(la["shaft"].mean()/1e3), min=at(dict(v=la["shaft"]/1e3, angle=la["angle"]), "v", int(la["shaft"].argmin())),
                               max=at(dict(v=la["shaft"]/1e3, angle=la["angle"]), "v", int(la["shaft"].argmax())))
        out["thrust_kN"] = dict(mean=float(la["thrust"].mean()/1e3), smallest_magnitude=at(dict(v=la["thrust"]/1e3, angle=la["angle"]), "v", int(np.abs(la["thrust"]).argmin())),
                                largest_magnitude=at(dict(v=la["thrust"]/1e3, angle=la["angle"]), "v", int(np.abs(la["thrust"]).argmax())))
    return out


def ranges(angle, mask):
    """Contiguous loop-angle ranges (in the order of the steps) where ``mask`` holds, as (first, last) pairs."""
    out, start = [], None
    for k, flag in enumerate(mask):
        if flag and start is None:
            start = k
        if (not flag or k == len(mask)-1) and start is not None:
            stop = k if flag else k-1
            out.append((round(float(angle[start]), 1), round(float(angle[stop]), 1)))
            start = None
    return out


def pair(run_dir, cache):
    data = load(run_dir)
    s = data["settings"]
    n, laps = int(s["steps_per_lap"]), int(s["laps"])
    a, b = lap(data, "A", cache), lap(data, "B", cache)
    loss = b["lift"]/a["lift"]-1.
    out = dict(run=Path(run_dir).name, laps=laps, steps_per_lap=n,
               lift_ratio_change_percent=dict(min=100*float(loss.min()), min_angle_deg=round(float(a["angle"][loss.argmin()]), 1),
                                              max=100*float(loss.max()), max_angle_deg=round(float(a["angle"][loss.argmax()]), 1)),
               loss_above_1p2_percent=ranges(a["angle"], loss < -0.012),
               loss_above_0p5_percent=ranges(a["angle"], loss < -0.005),
               loss_below_0p05_percent=ranges(a["angle"], loss > -0.0005))
    # local maxima of the loss (the deepest step of every contiguous range above 0.3 %)
    deep = []
    for lo, hi in ranges(np.arange(len(loss)), loss < -0.003):
        k = int(lo)+int(np.argmin(loss[int(lo):int(hi)+1]))
        deep.append(dict(angle_deg=round(float(a["angle"][k]), 1), loss_percent=-100*float(loss[k])))
    out["loss_peaks_above_0p3_percent"] = deep
    # largest lap-to-lap change of the total and of the wing load of B (K7, K7a) with its loop angle
    records = {}
    hist = Path(run_dir)/"history.json"
    if hist.exists():
        h = json.loads(hist.read_text(encoding="utf-8"))
        for r in h["records"]:
            records[int(r["step"])] = r
    angle = cf.loop_angle_deg(s, cache)
    for name in ("A", "B"):
        d = data["per_step"][name]
        index = {k: i for i, k in enumerate(d["step"])}
        last = [(laps-1)*n+k for k in range(1, n+1)]
        before = [(laps-2)*n+k for k in range(1, n+1)]
        pick = lambda key, steps: np.array([d[key][index[k]] for k in steps], dtype=float)   # noqa: E731
        row = {}
        for kind, key in (("force", "force_body_N"), ("moment", "moment_origin_body_Nm")):
            x0, x1 = pick(key, before), pick(key, last)
            rel = np.linalg.norm(x1-x0, axis=1)/np.linalg.norm(x0, axis=1)
            k = int(rel.argmax())
            row[f"K7a_{kind}"] = dict(max=float(rel.max()), angle_deg=round(float(angle[last[k]]), 1))
        if records:
            for kind, aero_key, teth_key, rot_key in (("force", "force_body_N", "tether_force_body_N", "rotor_force_body_N"),
                                                      ("moment", "moment_origin_body_Nm", "tether_moment_origin_body_Nm", "rotor_moment_origin_body_Nm")):
                def total(steps):
                    t = []
                    for st in steps:
                        body = records[st]["bodies"][name]
                        v = np.asarray(body[aero_key], float)+np.asarray(body[teth_key], float)
                        if rot_key in body:
                            v = v+np.asarray(body[rot_key], float)
                        t.append(v)
                    return np.array(t)
                x0, x1 = total(before), total(last)
                rel = np.linalg.norm(x1-x0, axis=1)/np.linalg.norm(x0, axis=1)
                k = int(rel.argmax())
                aero = np.array([records[st]["bodies"][name][aero_key] for st in before], float)
                row[f"K7_{kind}"] = dict(max=float(rel.max()), angle_deg=round(float(angle[last[k]]), 1),
                                         total_norm_at_max=float(np.linalg.norm(x0[k])), aero_norm_at_max=float(np.linalg.norm(aero[k])),
                                         steps_above_1_percent_angles=ranges(angle[np.array(last)], rel > 0.01))
        out[name] = row
    return out


def main(argv):
    p = argparse.ArgumentParser(prog="loop_angle_readback.py")
    p.add_argument("out")
    p.add_argument("cache")
    p.add_argument("--single", nargs=2, action="append", default=[])
    p.add_argument("--pair", action="append", default=[])
    args = p.parse_args(argv)
    report = dict(script="loop_angle_readback.py", angle="controller's loop angle of every step (c3_figures.loop_angle_deg)",
                  singles=[single(w, wo, args.cache) for w, wo in args.single], pairs=[pair(r, args.cache) for r in args.pair])
    Path(args.out).write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    r = main(sys.argv[1:])
    print(json.dumps(r, indent=1)[:12000])
