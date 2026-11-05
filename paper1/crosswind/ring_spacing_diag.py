"""Diagnostic of the rotor-ring ladder (contract section 8, 2026-10-22): is the velocity that the rings induce at the closure
points of the wing resolved by rings one time step apart? Report only, no criterion.

The stored wake of a run with rotors is frozen at one step. Its rings are kept in place and the slipstream sheet of every
disc is re-discretised along the path of its rings: ring k stands for the stretch u in [k-1/2, k+1/2] of the ring index u
(the newest ring for [0, 1/2]), and each stretch is split into m sub-rings, each at the linear interpolation of the ring
nodes at the middle of its sub-interval, with the ring circulation shared in proportion to the length of the sub-interval.
m = 1 is the stored discretisation. The velocity of the rings at the 3/4-chord point of every panel (the closure points of
VSM) is summed with the Lamb-Oseen core used there in the runs (the surface core) and with other cores, and its component
along the panel normal is divided by the airspeed, a change of incidence in radians.

If the incidence change of the stored rings (m = 1) differs from that of the refined sheet (m = 8) by a fraction of
itself, the rings one time step apart do not resolve the slipstream at the wing, and the loads depend on the time step.

Usage: ring_spacing_diag.py RUN_DIR OUT.json [--step N] [--subdivide 1,2,4,8] [--cores 0.15,0.3,0.6]
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
from openfawe.aero.aircraft import load_aircraft            # noqa: E402
from openfawe.aero.vortex import CoreModel, segments_velocity   # noqa: E402


def closure_points(case, rotation, position):
    """3/4-chord midpoints and unit normals (lift side) of all panels of all lifting surfaces, inertial frame."""
    wing = load_aircraft(REPO/case).wing
    points, normals, names = [], [], []
    for surface in wing.wings:
        le, te = np.asarray(surface.leading_edge, float), np.asarray(surface.trailing_edge, float)
        q = le+0.75*(te-le)
        mid = 0.5*(q[1:]+q[:-1])
        chord = 0.5*((te-le)[1:]+(te-le)[:-1])
        span = q[1:]-q[:-1]
        n = np.cross(span, chord)
        n /= np.linalg.norm(n, axis=1)[:, None]
        n *= np.sign(n[:, 2:3]+1e-12) if abs(np.mean(n[:, 2])) > 0.3 else 1.0       # lift side up for the main surfaces
        points.append(mid@rotation.T+position)
        normals.append(n@rotation.T)
        names += [getattr(surface, "name", "surface")]*len(mid)
    return np.vstack(points), np.vstack(normals), names


def rings_of(run):
    s = run["settings"]["rotors"]
    per, discs = int(s["nodes_per_ring"]), int(s["discs"])
    nodes, rf = run["nodes"], run["ring_first"]
    count = (len(nodes)-rf)//per
    rings = nodes[rf:rf+count*per].reshape(count, per, 3)
    edges, gamma = run["edges"], run["gamma"]
    ring_edge = (edges[:, 0] >= rf) & (edges[:, 1] >= rf)
    e, g = edges[ring_edge], gamma[ring_edge]
    # circulation of each ring from its first edge (a closed ring has one circulation)
    ring_index = (e[:, 0]-rf)//per
    circ = np.zeros(count)
    circ[ring_index] = g
    disc = discs-1-np.arange(count) % discs                   # RingWake.shed prepends; slot s within a step is disc 7-s
    age = np.arange(count)//discs
    return rings, circ, disc, age, per, discs


def sheet(rings, circ, disc, age, discs, m):
    """Sub-rings of every disc's sheet, m per ring stretch (see the module docstring)."""
    out_nodes, out_circ = [], []
    for d in range(discs):
        sel = np.where(disc == d)[0]
        sel = sel[np.argsort(age[sel])]
        P, G = rings[sel], circ[sel]
        K = len(P)-1
        for k in range(len(P)):
            lo, hi = max(0.0, k-0.5), min(float(K), k+0.5)
            if hi <= lo:
                continue
            for j in range(m):
                a, b = lo+(hi-lo)*j/m, lo+(hi-lo)*(j+1)/m
                u = 0.5*(a+b)
                i0 = min(int(np.floor(u)), K-1) if K > 0 else 0
                t = u-i0
                pos = P[i0]*(1-t)+P[min(i0+1, K)]*t if K > 0 else P[0]
                out_nodes.append(pos)
                out_circ.append(G[k]*(b-a)/(hi-lo))
    return np.array(out_nodes), np.array(out_circ)


def ring_velocity(points, ring_nodes, ring_circ, core):
    count, per = ring_nodes.shape[:2]
    nodes = ring_nodes.reshape(-1, 3)
    base = np.arange(count)[:, None]*per
    k = np.arange(per)[None, :]
    conn = np.stack([base+k, base+(k+1) % per], axis=-1).reshape(-1, 2)
    circ = np.repeat(ring_circ, per)
    return segments_velocity(points, nodes, conn, circ, model=CoreModel.LAMB_OSEEN, core_radius=core, singularity="zero",
                             backend="numpy")


def main(argv):
    ap = argparse.ArgumentParser(prog="ring_spacing_diag.py")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--subdivide", default="1,2,4,8")
    ap.add_argument("--cores", default="0.15,0.3,0.6")
    a = ap.parse_args(argv)
    import wake_figures as wf     # here, not at the top: the library tests import sheet() on nodes without matplotlib
    run = wf.read(a.run, a.step)
    motion = run["motions"]["A"]
    rot, pos = np.asarray(motion["rotation"][run["step"]]), np.asarray(motion["position_m"][run["step"]])
    vel = np.asarray(motion["velocity_m_s"][run["step"]])
    airspeed = float(np.linalg.norm(np.array([run["wind"], 0., 0.])-vel))
    pts, nrm, names = closure_points(run["case"], rot, pos)
    rings, circ, disc, age, per, discs = rings_of(run)
    result = dict(run=str(a.run), step=run["step"], airspeed_m_s=airspeed, ring_steps=int(age.max()+1), points=len(pts),
                  surfaces=sorted(set(names)), dt_s=run["dt"], cases=[])
    for core in [float(c) for c in a.cores.split(",")]:
        ref = None
        for m in [int(x) for x in a.subdivide.split(",")]:
            rn, rc = (rings, circ) if m == 1 else (None, None)
            if m != 1:
                rn, rc = sheet(rings, circ, disc, age, discs, m)
                rn = rn.reshape(-1, per, 3)
            u = ring_velocity(pts, rn, rc, core)
            dalpha = np.einsum("ij,ij->i", u, nrm)/airspeed
            row = dict(core_m=core, subdivide=m, rings=int(len(rc)), incidence_rad_rms=float(np.sqrt(np.mean(dalpha**2))),
                       incidence_rad_max=float(np.abs(dalpha).max()), incidence_rad=dalpha.tolist())
            result["cases"].append(row)
    finest = {}
    for row in result["cases"]:
        finest[row["core_m"]] = row if row["subdivide"] >= finest.get(row["core_m"], {"subdivide": 0})["subdivide"] else finest[row["core_m"]]
    for row in result["cases"]:
        f = np.asarray(finest[row["core_m"]]["incidence_rad"])
        d = np.asarray(row["incidence_rad"])-f
        row["difference_to_finest_rms_rad"] = float(np.sqrt(np.mean(d**2)))
        row["difference_to_finest_max_rad"] = float(np.abs(d).max())
        row["relative_to_finest_rms"] = float(np.sqrt(np.mean(d**2))/max(np.sqrt(np.mean(f**2)), 1e-30))
    Path(a.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(f"step {run['step']}, airspeed {airspeed:.2f} m/s, {len(pts)} closure points, rings of {result['ring_steps']} steps")
    print("| surface core (m) | sub-rings per ring | rms incidence from rings (deg) | max (deg) | rms difference to finest (deg) | relative |")
    print("|---|---|---|---|---|---|")
    for row in result["cases"]:
        print(f"| {row['core_m']} | {row['subdivide']} | {np.degrees(row['incidence_rad_rms']):.3f} | {np.degrees(row['incidence_rad_max']):.3f} | "
              f"{np.degrees(row['difference_to_finest_rms_rad']):.3f} | {row['relative_to_finest_rms']:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
