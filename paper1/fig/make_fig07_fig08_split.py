"""Figs. 7 and 8 of the English manuscript v13 (paper session, 2026-10-10): the former Fig. 7 split in two.

Owner 2026-10-10 21:07: "图7拆成 图7和图8 abcd为图7 ac在上 bd在下, ac下分别一个颜色条 bd下一个颜色条 然后efgh为图8
一个颜色条".
  Fig. 7  top row (a) M600 and (c) V3 vortex lattice, each with its colour bar of |Gamma_f|/(U c) below;
          bottom row (b) M600 and (d) V3 Q-criterion iso-surfaces with one colour bar of omega_x c/U below.
  Fig. 8  the four cross-flow planes, (a), (c) M600 at 5 and 40 chords, (b), (d) V3 at 2 and 10 chords, one colour bar
          of w/U (the same scale colours the planes in Fig. 7(a), (c)).
The renders and plane fields are those of make_fig07_wake_q_planes.py (parts/f7c_*.png, parts/f7_fields_*.pkl);
nothing is recomputed. Outputs: fig07_wake_lattice_q.png/.pdf, fig08_wake_planes.png/.pdf, fig07-08-split-receipt.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_draft_figures as mdf  # noqa: E402
import make_fig07_wake_q_planes as f7  # noqa: E402
import make_model_figures as mmf  # noqa: E402
import render3d as r3  # noqa: E402
import wakelib as wl  # noqa: E402

receipt = {"created": "2026-10-10", "inputs": {}, "outputs": {}}


def fig7():
    keys = ("m600", "v3")
    img = {(r, c): mpimg.imread(f7.PARTS / f"f7c_{k}_{kind}.png")
           for r, kind in enumerate(("lattice", "q")) for c, k in enumerate(keys)}
    for r, kind in enumerate(("lattice", "q")):
        for k in keys:
            p = f7.PARTS / f"f7c_{k}_{kind}.png"
            receipt["inputs"][p.name] = f7.sha(p)
    W, gap = 6.5, 0.15
    s = min((W - 0.1 - gap) / (img[(r, 0)].shape[1] + img[(r, 1)].shape[1]) for r in (0, 1))
    rows_h = [max(img[(r, 0)].shape[0], img[(r, 1)].shape[0]) * s for r in (0, 1)]
    bar_block = 0.78
    H = 0.25 + rows_h[0] + bar_block + rows_h[1] + bar_block + 0.05
    fig = plt.figure(figsize=(W, H))
    y = H - 0.25
    axes = {}
    boxes = {}
    for r in (0, 1):
        y -= rows_h[r]
        x = (W - (img[(r, 0)].shape[1] + img[(r, 1)].shape[1]) * s - gap) / 2
        for c in (0, 1):
            w, h = img[(r, c)].shape[1] * s, img[(r, c)].shape[0] * s
            ax = fig.add_axes([x / W, (y + (rows_h[r] - h) / 2) / H, w / W, h / H])
            ax.imshow(img[(r, c)])
            ax.axis("off")
            axes[(r, c)] = ax
            boxes[(r, c)] = (x, w)
            x += w + gap
        yb = y - 0.3
        if r == 0:            # one colour bar under each lattice: filament circulation of (a) and of (c)
            for c, key in enumerate(keys):
                x0, w = boxes[(0, c)]
                bw = min(0.8 * w, 2.6)
                f7.hbar(fig, [(x0 + (w - bw) / 2) / W, yb / H, bw / W, 0.085 / H], Normalize(*mmf.FILAMENT_CLIM[key]),
                        r3.WAKE_CMAP, r"$|\Gamma_f|/(U_\infty c)$")
        else:                 # one colour bar under (b) and (d): streamwise vorticity
            f7.hbar(fig, [(W / 2 - 1.6) / W, yb / H, 3.2 / W, 0.085 / H], Normalize(-4, 4), "RdBu_r",
                    r"$\omega_x c/U_\infty$", ticks=[-4, -2, 0, 2, 4])
        y -= bar_block
    fig.canvas.draw()
    for (r, c), t in (((0, 0), "a"), ((0, 1), "c"), ((1, 0), "b"), ((1, 1), "d")):
        f7.letter(fig, axes[(r, c)], f"({t})")
    fig.savefig(HERE / "fig07_wake_lattice_q.png")
    fig.savefig(HERE / "fig07_wake_lattice_q.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt["outputs"]["fig07_wake_lattice_q.png"] = dict(sha256=f7.sha(HERE / "fig07_wake_lattice_q.png"), size_in=[W, H])


def fig8():
    W, H = 6.5, 4.55
    fig = plt.figure(figsize=(W, H))
    grid = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=1.05 / H, top=1 - 0.22 / H, hspace=0.32, wspace=0.22)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)])
    levels = np.linspace(-0.2, 0.2, 21)
    image = None
    for col, key in enumerate(("m600", "v3")):
        state, fields = f7.plane_data(key)
        receipt["inputs"][state["path"].name + f" ({key})"] = wl.sha(state["path"])
        chord = state["chord"]
        for row, (station, field) in enumerate(zip(f7.PLANES[key]["stations"], fields)):
            ax = axes[row, col]
            image = ax.contourf(field["u"] / chord, (field["v"] - field["drop"]) / chord, field["downwash"],
                                levels=levels, cmap="RdBu_r", extend="both")
            trace = mmf.sheet_trace(state, field, station) / chord
            ax.plot(trace[:, 0], trace[:, 1], "o", color="k", markersize=1.5, markeredgewidth=0)
            ax.set_aspect("equal")
            ax.set_ylabel(r"$n/c$")
            if row == 1:
                ax.set_xlabel(r"$y/c$")
    wax = fig.add_axes([(W / 2 - 1.6) / W, 0.42 / H, 3.2 / W, 0.085 / H])
    bar = fig.colorbar(image, cax=wax, orientation="horizontal", ticks=[-0.2, -0.1, 0, 0.1, 0.2])
    bar.set_label(r"$w\,/\,U_\infty$")
    bar.outline.set_linewidth(0.6)
    fig.canvas.draw()
    for ax, t in zip(axes.flat, "abcd"):
        f7.letter(fig, ax, f"({t})")
    fig.savefig(HERE / "fig08_wake_planes.png")
    fig.savefig(HERE / "fig08_wake_planes.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt["outputs"]["fig08_wake_planes.png"] = dict(sha256=f7.sha(HERE / "fig08_wake_planes.png"), size_in=[W, H])


if __name__ == "__main__":
    mdf.style()
    fig7()
    fig8()
    (HERE / "fig07-08-split-receipt.json").write_text(json.dumps(receipt, indent=1, default=str), encoding="utf-8")
    print(json.dumps(receipt["outputs"]))
