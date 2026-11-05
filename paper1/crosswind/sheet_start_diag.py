"""Where does the ring sheet start? Static diagnostic of review RS-01 of the sheet refinement S (contract section 8,
2026-10-22 18:5x BST). Report only, no criterion.

The rings are released at the rim after the solve of a step and convected before the solve of the next, so at a solve
the newest ring is about one ring spacing h behind the current rim. The definition of S (18:22) lets ring 0 stand for
[0, 1/2] of the ring index: the refined sheet then starts at ring 0, about h behind the rim, with ring 0's circulation
on half a stretch. The stored rings themselves stand, by the midpoint rule, for a sheet that starts h/2 behind the rim.
Variants of the sheet at the closure points (3/4 chord) of the final wake of a run:

  stored     the stored rings
  S          the refinement as defined at 18:22 (RingWake.surface_sheet)
  anchored   the current rim is ring index u = -1 (carrying nothing): ring 0 stands for [-1/2, 1/2], interpolated
             towards the rim; the sheet starts h/2 behind the rim, as the stored rings do
  attached   anchored, plus the stretch [-1, -1/2] at ring 0's density: the sheet starts at the rim (the vorticity shed
             since the middle of the last step, not yet released, estimated with the newest ring's strength)

For each variant the ring-induced velocity at the closure points along the panel normal, divided by the airspeed (an
incidence change in radians), with the surface core of the run and sub-rings at overlap 4 (the stored variant
excepted). The stored wake of a run is the one after the release of its last step, so the rings on the rim (just
released) are set aside: the variants are those the solve of that step saw. Two runs at different dt give the
time-step dependence of each variant at the same instant.

Usage: sheet_start_diag.py OUT.json RUN_DIR [RUN_DIR ...] [--core 0.3] [--overlap 4]
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
import wake_figures as wf                                            # noqa: E402
from ring_spacing_diag import closure_points, rings_of               # noqa: E402
from openfawe.aero.vortex import CoreModel, segments_velocity        # noqa: E402
from openfawe.rotor.actuator_disc import load_rotor_set, ring_edges  # noqa: E402
from openfawe.rotor.ring_wake import RingWake, sheet_subrings         # noqa: E402
from openfawe.rotor.table import RotorTable                          # noqa: E402


def velocity(points, rings, circ, core):
    """Velocity of closed rings (count, per, 3) with one circulation each."""
    count, per = rings.shape[:2]
    edges = np.vstack([ring_edges(per, k*per) for k in range(count)])
    return segments_velocity(points, rings.reshape(-1, 3), edges, np.repeat(circ, per), model=CoreModel.LAMB_OSEEN,
                             core_radius=core, singularity="zero", backend="numpy")


def anchored_sheet(seq, circ, rim, core, overlap, attached):
    """Sub-rings of one disc's sheet with the current rim as ring index -1 (see the module docstring)."""
    full = np.concatenate([rim[None], seq])                         # index 0 is the rim (u = -1)
    K = len(full)-1
    out_r, out_c = [], []
    for k in range(1, K+1):                                          # stored ring k-1 sits at full index k
        lo, hi = k-0.5, min(float(K), k+0.5)
        if k == 1 and attached:
            lo = 0.0
        gap = max(np.max(np.linalg.norm(full[k]-full[k-1], axis=1)),
                  np.max(np.linalg.norm(full[min(k+1, K)]-full[k], axis=1)) if k < K else 0.0)
        m = max(1, int(np.ceil(overlap*gap*(hi-lo)/core)))
        density = circ[k-1]                                           # circulation per unit of the ring index
        for j in range(m):
            a, b = lo+(hi-lo)*j/m, lo+(hi-lo)*(j+1)/m
            u = 0.5*(a+b)
            i0 = min(int(np.floor(u)), K-1)
            t = u-i0
            out_r.append(full[i0]*(1-t)+full[i0+1]*t)
            out_c.append(density*(b-a))
    return np.array(out_r), np.array(out_c)


def main(argv):
    ap = argparse.ArgumentParser(prog="sheet_start_diag.py")
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--core", type=float, default=None)
    ap.add_argument("--overlap", type=float, default=4.0)
    a = ap.parse_args(argv)
    result = dict(overlap=a.overlap, runs={})
    for path in a.runs:
        run = wf.read(path)
        s = run["settings"]["rotors"]
        core = float(s["surface_core_m"]) if a.core is None else a.core
        motion = run["motions"]["A"]
        rot, pos = np.asarray(motion["rotation"][run["step"]]), np.asarray(motion["position_m"][run["step"]])
        vel = np.asarray(motion["velocity_m_s"][run["step"]])
        airspeed = float(np.linalg.norm(np.array([run["wind"], 0., 0.])-vel))
        pts, nrm, names = closure_points(run["case"], rot, pos)
        rings, circ, disc, age, per, discs = rings_of(run)
        placed = [d.placed(rot, pos) for d in load_rotor_set(REPO/s["rotor_set"], RotorTable.from_makani_json(REPO/s["table"]))]
        phase = float(s.get("ring_phase_rad", 0.0) or 0.0)
        # The stored state is the one after this step's release: its newest rings sit on the rim. The solve of this
        # step saw the older rings only, so the newest rings are set aside and the rim is where they sit.
        rim_check = float(max(np.max(np.linalg.norm(placed[d].rim(per, phase)-rings[(disc == d) & (age == 0)][0], axis=1))
                              for d in range(discs)))
        keep = age > 0
        rings, circ, disc, age = rings[keep], circ[keep], disc[keep], age[keep]-1
        variants = {}
        variants["stored"] = velocity(pts, rings, circ, core)
        # S as defined at 18:22: through RingWake.surface_sheet on the same rings (times from the age, newest first)
        order = np.lexsort((-disc, age))
        wake = RingWake(rings[order].reshape(-1, 3), np.vstack([ring_edges(per, k*per) for k in range(len(order))]),
                        np.repeat(circ[order], per), np.repeat(-age[order]*run["dt"], per), np.repeat(disc[order], per),
                        surface_core_m=core, surface_overlap=a.overlap)
        n, e, g, _ = wake.surface_sheet(pts, core)
        variants["S"] = segments_velocity(pts, n, e, g, model=CoreModel.LAMB_OSEEN, core_radius=core, singularity="zero",
                                          backend="numpy")
        for name, attached in (("anchored", False), ("attached", True)):
            total = np.zeros_like(pts)
            for d in range(discs):
                sel = np.where(disc == d)[0]
                sel = sel[np.argsort(age[sel])]
                rim = placed[d].rim(per, phase)
                sub, sub_c = anchored_sheet(rings[sel], circ[sel], rim, core, a.overlap, attached)
                total += velocity(pts, sub, sub_c, core)
            variants[name] = total
        gap = float(np.median([np.min(np.linalg.norm(placed[d].rim(per, phase)[None]-rings[(disc == d) & (age == 0)], axis=2))
                               for d in range(discs)]))
        entry = dict(dt_s=run["dt"], step=run["step"], airspeed_m_s=airspeed, surface_core_m=core, rim_to_newest_ring_m=gap,
                     released_ring_off_rim_m=rim_check,
                     incidence_rad={k: (np.einsum("ij,ij->i", v, nrm)/airspeed).tolist() for k, v in variants.items()})
        result["runs"][str(path)] = entry
    keys = list(result["runs"])
    print("| run | dt (s) | rim to newest ring (m) | variant | rms incidence (deg) |")
    print("|---|---|---|---|---|")
    for k in keys:
        r = result["runs"][k]
        for v, inc in r["incidence_rad"].items():
            print(f"| {Path(k).name} | {r['dt_s']:.4f} | {r['rim_to_newest_ring_m']:.2f} | {v} | {np.degrees(np.sqrt(np.mean(np.square(inc)))):.3f} |")
    if len(keys) >= 2:
        fine = result["runs"][keys[-1]]["incidence_rad"]
        result["difference_to_last"] = {}
        print(f"\n| run | variant | rms difference to {Path(keys[-1]).name}, same variant (deg) | to its 'attached' (deg) |")
        print("|---|---|---|---|")
        for k in keys[:-1]:
            row = {}
            for v, inc in result["runs"][k]["incidence_rad"].items():
                d_same = np.degrees(np.sqrt(np.mean((np.asarray(inc)-np.asarray(fine[v]))**2)))
                d_att = np.degrees(np.sqrt(np.mean((np.asarray(inc)-np.asarray(fine["attached"]))**2)))
                row[v] = dict(rms_deg_same_variant=d_same, rms_deg_to_fine_attached=d_att)
                print(f"| {Path(k).name} | {v} | {d_same:.4f} | {d_att:.4f} |")
            result["difference_to_last"][k] = row
    Path(a.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
