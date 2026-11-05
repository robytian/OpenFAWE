"""Combined result figures for an earlier journal version (paper session, 2026-10-08, author's comments on the
Chinese draft v2: one figure or one combined figure per type of result).

Reuses the plotting functions and typed numbers of make_draft_figures.py (same evidence files, same style), only
the panel layout changes. No solver is called.
  figR1_comparison.png     V3 (a-c) and M600 (d-e) against the reference data   [was Figs. 5 and 6]
  figR2_wake_dt.png        wake-length (a-c) and time-step (d-f) ladders          [was Figs. 7 and 9]
  figR3_spanwise_core.png  spanwise refinement (a), outcome map (b), core sensitivity (c-e) [was Figs. 10 and 11]
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_draft_figures as D  # noqa: E402

COL, MARK, DASH, LABEL = D.COL, D.MARK, D.DASH, D.LABEL


def comparison_panel(ax, data, free_data, fig_key, q, yl, tag):
    for m in ("LL", "VSM", "LSi", "LSp"):
        s = data.get((fig_key, f"OpenFAWE {m}", "straight"), {}).get(q)
        if s is not None:
            ax.plot(s[:, 0], s[:, 1], "-", color=COL[m], marker=MARK[m])
        free = free_data.get((fig_key, m), {}).get(q)
        if free is not None:
            ax.plot(free[:, 0], free[:, 1], linestyle=DASH, color=COL[m], marker=MARK[m], markerfacecolor="white")
    if fig_key == "Fig1":
        rans = data.get(("Fig1", "rans", ""), {}).get(q)
        if rans is not None:
            ax.plot(rans[:, 0], rans[:, 1], "-", color="0.65", linewidth=2.4, zorder=1)
        tun = data.get(("Fig1", "tunnel", ""), {}).get(q)
        if tun is not None:
            ax.errorbar(tun[:, 0], tun[:, 1], yerr=tun[:, 2], fmt="D", color="k", markerfacecolor="white",
                        markersize=4.5, capsize=2.5, linewidth=0.9, zorder=5)
    ref = data.get((fig_key, "vsm", ""), {}).get(q)
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], "x", color="k", markersize=5, markeredgewidth=1.0, zorder=6)
    ax.set_xlabel(r"$\alpha$ (°)")
    ax.set_ylabel(yl)
    D.grid(ax)
    D.letter(ax, f"({tag})")


def fig_comparison():
    data, free_data = D.series(), D.free_series()
    fig = plt.figure(figsize=(6.5, 6.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    top = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    bottom = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    for ax, q, yl, tag in zip(top, ("CL", "CD", "CMy"), (r"$C_L$", r"$C_D$", r"$C_{M_y}$"), "abc"):
        comparison_panel(ax, data, free_data, "Fig1", q, yl, tag)
    for ax, q, yl, tag in zip(bottom, ("CL", "CD"), (r"$C_L$", r"$C_D$"), "de"):
        comparison_panel(ax, data, free_data, "Fig2", q, yl, tag)
    handles = [Line2D([], [], color=COL[m], marker=MARK[m], label=LABEL[m]) for m in ("LL", "VSM", "LSi", "LSp")]
    handles += [Line2D([], [], color="k", linestyle="-", label="Straight wake"),
                Line2D([], [], color="k", linestyle=DASH, label="Free wake"),
                Line2D([], [], color="0.65", linewidth=3, label=r"RANS ($Re = 5\times10^5$)"),
                Line2D([], [], color="k", marker="D", markerfacecolor="white", linestyle="", label="Wind tunnel (99 % CI)"),
                Line2D([], [], color="k", marker="x", linestyle="", markeredgewidth=1.2, label="Open-source VSM, same input")]
    fig.legend(handles=handles, loc="lower center", ncol=3, columnspacing=1.2, handlelength=2.2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=1.0, h_pad=1.6)
    fig.savefig(HERE / "figR1_comparison.png")
    if os.environ.get("PDF_OUT"):
        fig.savefig(HERE / "figR1_comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def fig_wake_dt():
    chords = {"V3": [10, 20, 40, 80], "M600": [10, 20, 40, 80, 160]}
    val = {"V3": {"CL": [0.570975, 0.572362, 0.572674, 0.572748], "CD": [0.051670, 0.051574, 0.051550, 0.051544],
                  "CM": [0.037934, 0.037180, 0.037000, 0.036956]},
           "M600": {"CL": [1.881342, 1.926112, 1.938629, 1.941550, 1.942216],
                    "CD": [0.068691, 0.062206, 0.059881, 0.059276, 0.059129],
                    "CM": [-0.038768, -0.078333, -0.085493, -0.086732, -0.086982]}}
    cases = {
        "V3 kite, 5.41°": (2.851 / 2.5675, [0.4, 0.2, 0.1, 0.05], COL["V3"], "o",
                           {"CL": [0.572102, 0.572249, 0.572324, 0.572360], "CD": [0.051577, 0.051583, 0.051587, 0.051589],
                            "CM": [0.037264, 0.037244, 0.037238, 0.037237]}),
        "V3 kite, 7.35°": (2.851 / 2.5675, [0.4, 0.2, 0.1, 0.05], COL["V3b"], "^",
                           {"CL": [0.717365, 0.717658, 0.717813, 0.717878], "CD": [0.058310, 0.058322, 0.058330, 0.058335],
                            "CM": [-0.072031, -0.072102, -0.072136, -0.072143]}),
        "M600, 0°": (20 / 1.2831, [0.04, 0.02, 0.01, 0.005], COL["M600"], "s",
                     {"CL": [1.931834, 1.932147, 1.932231, 1.932222], "CD": [0.061045, 0.061127, 0.061183, 0.061211],
                      "CM": [-0.082074, -0.082099, -0.081775, -0.081660]}),
    }
    ylabels = (r"$|\Delta C_L|/C_L$ (%)", r"$|\Delta C_D|/C_D$ (%)", r"$|\Delta C_{M_y}|/|C_{M_y}|$ (%)")
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 5.6))
    for ax, q, yl, tag in zip(axes[0], ("CL", "CD", "CM"), ylabels, "abc"):
        for model, marker in (("V3", "o"), ("M600", "s")):
            v = np.array(val[model][q])
            dev = np.abs(v[:-1] - v[-1]) / abs(v[-1]) * 100
            ax.loglog(chords[model][:-1], dev, "-", color=COL[model], marker=marker,
                      label=("V3 kite, 5.41°" if model == "V3" else "M600, 0°"))
        ax.set_xlabel("Wake length (chords)")
        ax.set_ylabel(yl)
        ax.set_xticks([10, 20, 40, 80])
        ax.set_xticklabels(["10", "20", "40", "80"])
        ax.minorticks_off()
        D.grid(ax)
        D.letter(ax, f"({tag})")
    for ax, q, yl, tag in zip(axes[1], ("CL", "CD", "CM"), ylabels, "def"):
        for label, (scale, dts, colour, marker, values) in cases.items():
            v = np.array(values[q])
            diff = np.abs(np.diff(v)) / np.abs(v[1:]) * 100
            ax.loglog(scale * np.array(dts[1:]), diff, "-", color=colour, marker=marker, label=label)
        x = np.array([0.05, 0.35])
        lo, hi = ax.get_ylim()
        anchor = np.sqrt(lo * hi) if q != "CL" else 0.01
        ax.loglog(x, anchor * x / x[0] * 0.4, linestyle=(0, (4, 3)), color="0.4", linewidth=0.9, label="Slope 1")
        ax.set_xlabel(r"$U_\infty\Delta t/c$")
        ax.set_ylabel(yl)
        ax.set_xticks([0.05, 0.1, 0.2, 0.4])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
        ax.tick_params(axis="x", pad=5)
        ax.minorticks_off()
        ax.set_xlim(0.045, 0.42)
        D.grid(ax)
        D.letter(ax, f"({tag})")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8, h_pad=1.4)
    fig.savefig(HERE / "figR2_wake_dt.png")
    if os.environ.get("PDF_OUT"):
        fig.savefig(HERE / "figR2_wake_dt.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def fig_spanwise_core():
    fig = plt.figure(figsize=(6.5, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1])
    a = fig.add_subplot(gs[0, 0:3])
    b = fig.add_subplot(gs[0, 3:6])
    cs = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]
    # (a) spanwise refinement at fixed core (numbers as in make_draft_figures.spanwise_core)
    a.plot([1, 2, 3], [0, 0.14, 0.14 + 0.09], "-", color=COL["V3"], marker="o", label="V3 kite, 5.41°, 0.075 m")
    v = np.array([0.707104, 0.706744, 0.707958])
    a.plot([1, 2, 3], (v / v[0] - 1) * 100, "-", color=COL["V3b"], marker="^", label="V3 kite, 7.35°, 0.05 m")
    a.plot([1, 2, 3], [0, -0.36, -0.36 - 0.13], "-", color=COL["M600"], marker="s", label="M600, 0°, 0.075 m")
    a.set_xticks([1, 2, 3])
    a.set_xlabel("Sub-panels per panel")
    a.set_ylabel(r"$C_L/C_L(\times 1) - 1$ (%)")
    a.set_ylim(-0.95, 0.3)
    a.legend(loc="lower left", fontsize=8.5)
    D.grid(a)
    D.letter(a, "(a)")
    # (b) outcome map
    ok = [(0.203, 0.075), (0.203, 0.15), (0.203, 0.30), (0.101, 0.075), (0.101, 0.15), (0.068, 0.025), (0.068, 0.05),
          (0.068, 0.075), (0.203, 0.05), (0.101, 0.05), (0.235, 0.075), (0.235, 0.15), (0.235, 0.30), (0.118, 0.075),
          (0.118, 0.15), (0.078, 0.075)]
    fail = [(0.068, 0.15), (0.068, 0.075), (0.078, 0.15), (0.078, 0.30)]
    other = [(0.078, 0.05)]
    b.loglog(*zip(*ok), "o", color="0.15", markersize=4, label="Completed")
    b.loglog(*zip(*fail), "x", color="#D55E00", markersize=6, markeredgewidth=1.4, label="Closure failed")
    b.loglog(*zip(*other), "s", color="#0072B2", markerfacecolor="white", markersize=5, label="Reversed flow at tail")
    line = np.array([0.02, 0.4])
    b.loglog(line, line, linestyle=(0, (4, 3)), color="0.4", linewidth=0.9, label=r"$r_c = w_{\min}$")
    b.set_xlim(0.05, 0.3)
    b.set_ylim(0.02, 0.4)
    b.set_xticks([0.06, 0.1, 0.2])
    b.set_xticklabels(["0.06", "0.1", "0.2"])
    b.set_yticks([0.025, 0.05, 0.1, 0.2, 0.4])
    b.set_yticklabels(["0.025", "0.05", "0.1", "0.2", "0.4"])
    b.minorticks_off()
    b.set_xlabel(r"Smallest panel width $w_{\min}$ (m)")
    b.set_ylabel(r"Core radius $r_c$ (m)")
    b.legend(loc="lower right", fontsize=8.5, handletextpad=0.3)
    D.grid(b)
    D.letter(b, "(b)")
    # (c)-(e) core sensitivity
    rc = np.array([0.075, 0.15, 0.30])
    val = {"V3": (0.318, {"CL": [0.565516, 0.572249, 0.587436], "CD": [0.051002, 0.051583, 0.052165],
                          "CM": [0.037154, 0.037244, 0.035123]}),
           "M600": (0.755, {"CL": [1.932251, 1.932147, 1.944221], "CD": [0.061143, 0.061127, 0.061444],
                            "CM": [-0.082946, -0.082099, -0.047805]})}
    for ax, q, yl, tag in zip(cs, ("CL", "CD", "CM"), (r"$\Delta C_L/C_L$ (%)", r"$\Delta C_D/C_D$ (%)",
                                                      r"$\Delta |C_{M_y}|/|C_{M_y}|$ (%)"), "cde"):
        for model, marker in (("V3", "o"), ("M600", "s")):
            width, values = val[model]
            vv = np.abs(np.array(values[q]))
            ax.plot(rc / width, (vv / vv[1] - 1) * 100, "-", color=COL[model], marker=marker,
                    label=("V3 kite, 5.41°" if model == "V3" else "M600, 0°"))
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_xlabel(r"$r_c\,/\,\bar{w}$")
        ax.set_ylabel(yl)
        D.grid(ax)
        D.letter(ax, f"({tag})")
    handles, labels = cs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.2, h_pad=1.6)
    fig.savefig(HERE / "figR3_spanwise_core.png")
    if os.environ.get("PDF_OUT"):
        fig.savefig(HERE / "figR3_spanwise_core.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    D.style()
    fig_comparison()
    fig_wake_dt()
    fig_spanwise_core()
    names = ["figR1_comparison.png", "figR2_wake_dt.png", "figR3_spanwise_core.png"]
    receipt = {"producer": "make_re_figures.py", "producer_sha256": D.sha(__file__), "created": "2026-10-09",
               "status": "combined layouts of the draft-4 figures (same data, same typed numbers); figR1 now also shows the "
                         "lifting-surface free wakes at the converged length (extra/ls_freewake, 2026-10-09)",
               "inputs": {n: D.sha(HERE / n) for n in ("plotted-series.csv", "plotted-series-freewake-converged.csv",
                                                       "make_draft_figures.py")},
               "outputs": {n: D.sha(HERE / n) for n in names}, "matplotlib": matplotlib.__version__}
    (HERE / "re-figure-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("\n".join(names))
