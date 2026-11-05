"""Assemble the Q-criterion figure (paper session, 2026-10-08): renders from render_q.py plus panel letters and a
colour bar. No other text in the figure."""
import hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import make_draft_figures as mdf

HERE = Path(__file__).resolve().parent
mdf.style()
a, b = mpimg.imread(HERE / "parts/q_m600.png"), mpimg.imread(HERE / "parts/q_v3.png")
ha, hb = a.shape[0] / a.shape[1], b.shape[0] / b.shape[1]
width = 6.5
fig = plt.figure(figsize=(width, width * (ha + hb) + 1.0))
total = width * (ha + hb) + 1.0
axa = fig.add_axes([0.0, 1 - width * ha / total, 1.0, width * ha / total])
axb = fig.add_axes([0.0, 1.0 / total, 1.0, width * hb / total])
for ax, img, letter in ((axa, a, "(a)"), (axb, b, "(b)")):
    ax.imshow(img)
    ax.set_axis_off()
    ax.text(0.0, 0.98, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top", ha="left")
cax = fig.add_axes([0.3, 0.55 / total, 0.4, 0.12 / total])
bar = fig.colorbar(ScalarMappable(Normalize(-4, 4), cmap="RdBu_r"), cax=cax, orientation="horizontal")
bar.set_label(r"$\omega_x c/U_\infty$", fontsize=10)
bar.ax.tick_params(labelsize=9)
fig.savefig(HERE / "figD0_q_criterion.png", dpi=300)
receipt = {"created": "2026-10-08", "parts": {p: hashlib.sha256((HERE / "parts" / p).read_bytes()).hexdigest()
                                              for p in ("q_m600.png", "q_v3.png")},
           "fields": {"m600": "../extra/q/m600-0128.npz (state_0128, 40 chords, h = 0.10 m)",
                      "v3": "../extra/q/v3-0090.npz (state_0090, 20 chords, h = 0.08 m)"},
           "threshold": "Q = 1.0 (U/c)^2 opaque, Q = 0.015 (U/c)^2 translucent (opacity 0.35); render_q.py ... 1.0 -130 30 1.1 0.015 (M600), 1.0 -130 28 1.0 0.015 (V3)", "colour": "omega_x c / U, -4..4, RdBu_r",
           "figure_sha256": hashlib.sha256((HERE / "figD0_q_criterion.png").read_bytes()).hexdigest()}
(HERE / "q-figure-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
print("ok")
