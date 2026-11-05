"""Figure 14(b)(c) of paper 1: wake-plane downwash versus wake length and versus core radius.

Inputs are the states of jobs-paper1-20261002.txt (run on the owner's PC):
  M600  p1-m600-planes-wakelength/state_{0032,0064,0128,0257,0513}.npz  (10/20/40/80/160 chords)
        p1-m600-planes-core{0075,0300}/state_0257.npz + the 0.15 m state_0257 above (80 chords)
  V3    p1-v3-planes-wakelength/state_{0045,0090,0180,0360}.npz          (10/20/40/80 chords)
        p1-v3-planes-core{0075,0300}/state_0180.npz + the 0.15 m state_0180 above (40 chords)
For each ladder: one row of cross-flow planes at TE + STATION_CHORDS (downwash w/U with in-plane
vectors, shared colour scale, decision 0024 style: no text inside the figure), a spanwise profile
of w/U along the plane's horizontal centre line (legend only), VTS files, and a receipt with the
states' SHA-256, their CL and the probe kernel. Same plane construction as render_wake_planes.py.
Usage: render_wake_ladder.py OUT_DIR RUNS_DIR [--direct]
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper1/scripts/convergence"))
import accel_bench as ab  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from openfawe import aero_render as ar  # noqa: E402
from openfawe.aero import wake_probe as wp  # noqa: E402
from openfawe.aero.aircraft import load_aircraft  # noqa: E402
from openfawe.aero.vortex import TreeSettings  # noqa: E402

CASES = {"v3": REPO / "cases/aircraft/v3_cad_camber_integral.json",
         "m600": REPO / "cases/qualification/m600_makani_reconstructed.json"}
STATION_CHORDS = 5.0
LADDERS = {
    "m600": dict(wakelength=[("10 c", "p1-m600-planes-wakelength/state_0032.npz"), ("20 c", "p1-m600-planes-wakelength/state_0064.npz"),
                             ("40 c", "p1-m600-planes-wakelength/state_0128.npz"), ("80 c", "p1-m600-planes-wakelength/state_0257.npz"),
                             ("160 c", "p1-m600-planes-wakelength/state_0513.npz")],
                 core=[("0.075 m", "p1-m600-planes-core0075/state_0257.npz"), ("0.15 m", "p1-m600-planes-wakelength/state_0257.npz"),
                       ("0.30 m", "p1-m600-planes-core0300/state_0257.npz")]),
    "v3": dict(wakelength=[("10 c", "p1-v3-planes-wakelength/state_0045.npz"), ("20 c", "p1-v3-planes-wakelength/state_0090.npz"),
                           ("40 c", "p1-v3-planes-wakelength/state_0180.npz"), ("80 c", "p1-v3-planes-wakelength/state_0360.npz")],
               core=[("0.075 m", "p1-v3-planes-core0075/state_0180.npz"), ("0.15 m", "p1-v3-planes-wakelength/state_0180.npz"),
                     ("0.30 m", "p1-v3-planes-core0300/state_0180.npz")]),
}
DT_S = {"m600": 0.02, "v3": 0.2}   # archived time steps of the campaign (jobs-paper1-20261002.txt)
COLOURS = ["#1b4f72", "#2e86c1", "#17a589", "#d68910", "#922b21"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_cl(state_path):
    history = Path(state_path).parent / "history.json"
    if not history.exists():
        return None
    records = json.loads(history.read_text(encoding="utf-8"))["records"]
    step = int(Path(state_path).stem.split("_")[1])
    match = [r for r in records if r["step"] == step]
    return match[0]["CL"] if match else None


def main(out, runs, direct=False):
    out, runs = Path(out), Path(runs)
    out.mkdir(parents=True, exist_ok=True)
    kernel = dict(backend="numba") if direct else dict(tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
    receipt = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), station_chords=STATION_CHORDS,
                   figure_style="paper (no in-figure text, decision 0024)",
                   kernel="numba direct" if direct else "tree theta 0.2 order 2 leaf 32",
                   git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
                   ladders={}, outputs=[])
    for key, ladders in LADDERS.items():
        wing = load_aircraft(CASES[key]).wing
        te_centre = wing.trailing_edge.mean(axis=0)
        span = float(np.ptp(wing.bound[:, 1]))
        half_y, half_z = 0.62 * span, 0.32 * span
        for kind, entries in ladders.items():
            tick = time.perf_counter()
            fields_list, profiles, meta = [], [], []
            for label, relative in entries:
                path = runs / relative
                data = np.load(path)
                nodes, edges, gamma, radii = data["nodes"], data["edges"], data["gamma"], data["radii"]
                chord, speed, model = float(data["chord"]), float(data["speed"]), int(data["core_model"])
                freestream = np.asarray(ab.inflow(speed, ab.ALPHA[key.upper()], 0.0), dtype=float)
                direction = freestream / np.linalg.norm(freestream)
                axis_v = np.cross(direction, [0.0, 1.0, 0.0])
                axis_v = -axis_v / np.linalg.norm(axis_v) if axis_v[2] < 0 else axis_v / np.linalg.norm(axis_v)
                plane = wp.Plane(te_centre + STATION_CHORDS * chord * direction, [0.0, 1.0, 0.0], axis_v, half_y, half_z, 181, 65)
                fields = wp.plane_fields(plane, nodes, edges, gamma, model=model, core_radius=radii, freestream=freestream, **kernel)
                fields_list.append(fields)
                grid = wp.structured_grid(fields)
                grid.point_data["downwash_over_Uinf"] = fields["in_plane"][..., 1].reshape(-1) / speed
                vts = out / f"{key}_{kind}_{label.replace(' ', '').replace('.', 'p')}_cross_{STATION_CHORDS:g}c.vts"
                grid.save(str(vts), binary=True)
                receipt["outputs"].append(vts.name)
                # Spanwise profile along the plane's horizontal centre line (v = 0).
                row = int(np.argmin(np.abs(fields["v"])))   # u, v are 1-D plane coordinates (nu,), (nv,)
                profiles.append((label, fields["u"] / chord, fields["in_plane"][row, :, 1] / speed, fields["axial"][row] / speed))
                meta.append(dict(label=label, state=str(relative), sha256=sha(path), steps=int(path.stem.split("_")[1]),
                                 wake_chords=int(path.stem.split("_")[1]) * DT_S[key] * speed / chord,
                                 core_radius_m=[float(np.min(radii)), float(np.max(radii))], CL_history=history_cl(path),
                                 downwash_profile_min=float(np.min(profiles[-1][2])), downwash_profile_max=float(np.max(profiles[-1][2]))))
            wing_trace = [(wing.trailing_edge[:, 1] / chord, ((wing.trailing_edge - te_centre) @ axis_v) / chord)] * len(fields_list)
            for suffix in (".png", ".pdf"):
                figure = out / f"{key}_{kind}_cross_downwash{suffix}"
                ar.plane_figure(figure, fields_list, scalar="downwash", chord=chord, labels=None, vectors=True,
                                reference=wing_trace, title=None, ncols=len(fields_list), clip_percentiles=(2, 98),
                                dpi=300 if suffix == ".png" else 200)
                receipt["outputs"].append(figure.name)
            fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True, constrained_layout=True)
            for (label, u, w, ax_u), colour in zip(profiles, COLOURS):
                axes[0].plot(u, w, color=colour, lw=1.6, label=label)
                axes[1].plot(u, ax_u, color=colour, lw=1.6, label=label)
            axes[0].set_ylabel("w / U∞")
            axes[1].set_ylabel("u / U∞")
            axes[1].set_xlabel("y / c")
            for ax in axes:
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=9)
            axes[0].legend(frameon=False, fontsize=9, ncol=min(5, len(profiles)))
            for suffix in (".png", ".pdf"):
                figure = out / f"{key}_{kind}_profile_{STATION_CHORDS:g}c{suffix}"
                fig.savefig(figure, dpi=300 if suffix == ".png" else 200)
                receipt["outputs"].append(figure.name)
            plt.close(fig)
            (out / f"{key}_{kind}_profile_{STATION_CHORDS:g}c.json").write_text(json.dumps(
                dict(station_chords=STATION_CHORDS, entries=[dict(label=l, y_over_c=u.tolist(), w_over_U=w.tolist(), u_over_U=a.tolist())
                                                            for l, u, w, a in profiles]), indent=1) + "\n", encoding="utf-8")
            receipt["ladders"][f"{key}_{kind}"] = dict(states=meta, seconds=time.perf_counter() - tick,
                                                      plane=dict(te_centre=te_centre.tolist(), half_y=half_y, half_z=half_z, grid=[181, 65]))
            print(key, kind, "done in %.0f s" % receipt["ladders"][f"{key}_{kind}"]["seconds"], flush=True)
    (out / "wake-ladder-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], direct="--direct" in sys.argv)
