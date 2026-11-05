"""Result figures 4-6, 9-12 and 14 of the paper-1 draft (paper session, 2026-10-02; fonts enlarged in draft 4).

No solver is called. Inputs:
  plotted-series.csv                        (copy of OF-0006-paperfig20260930/figures-v1/plotted-series.csv)
  plotted-series-freewake-converged.csv     (copy of OF-0006-viz20261001/paper1-combos, commit e58677c)
  m600_wakelength_profile_5c.json, m600_core_profile_5c.json (copies of OF-0006-viz20261001/wake-ladder-paper)
  v1-m600-40-200.csv, v1-v3-40-200.csv      (OF-0006-accel20261001/evidence)
  numbers typed below, each from the evidence file named in the comment
Style: decision 0024 - no text inside the figures apart from axis labels, legends and panel letters.
Figure width 6.5 in = printed width, so the font sizes below are the printed sizes.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
EVID = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
COL = {"LL": "#0072B2", "VSM": "#D55E00", "LSi": "#009E73", "LSp": "#CC79A7", "V3": "#0072B2", "V3b": "#56B4E9",
       "M600": "#D55E00"}
MARK = {"LL": "o", "VSM": "s", "LSi": "^", "LSp": "v"}
DASH = (0, (3.2, 1.9))


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 4.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.frameon": False, "legend.handlelength": 1.8,
        "mathtext.fontset": "stix", "savefig.dpi": 300, "figure.dpi": 100})


def letter(ax, text):
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom", ha="left")


def grid(ax):
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)


def series():
    out = {}
    with (HERE / "plotted-series.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["figure"], row["series"], row["wake"])
            entry = out.setdefault(key, {}).setdefault(row["quantity"], [])
            entry.append((float(row["alpha_deg"]), float(row["value"]),
                          float(row["ci99_halfwidth"]) if row["ci99_halfwidth"] else 0.0))
    return {k: {q: np.array(sorted(v)) for q, v in d.items()} for k, d in out.items()}


# Free wake at converged wake length (VSM, LL; V3 40 chords, M600 103 chords, tree theta 0.2, order 2, leaf 32,
# core 0.15 m): plotted-series-freewake-converged.csv of the simulation session (paper1_combos.py, commit e58677c).
def free_series():
    out = {}
    with (HERE / "plotted-series-freewake-converged.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["figure"], row["series"].replace("OpenFAWE ", ""))
            out.setdefault(key, {}).setdefault(row["quantity"], []).append((float(row["alpha_deg"]), float(row["value"])))
    return {k: {q: np.array(sorted(v)) for q, v in d.items()} for k, d in out.items()}


LABEL = {"LL": "LL", "VSM": "VSM", "LSi": "LS, inviscid", "LSp": "LS, polar-coupled"}


def comparison(fig_key, quantities, ylabels, name, width, height):
    data = series()
    free_data = free_series()
    fig, axes = plt.subplots(1, len(quantities), figsize=(width, height))
    for ax, q, yl, tag in zip(axes, quantities, ylabels, "abc"):
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
        grid(ax)
        letter(ax, f"({tag})")
    handles = [Line2D([], [], color=COL[m], marker=MARK[m], label=LABEL[m]) for m in ("LL", "VSM", "LSi", "LSp")]
    handles += [Line2D([], [], color="k", linestyle="-", label="Straight wake"),
                Line2D([], [], color="k", linestyle=DASH, label="Free wake")]
    if fig_key == "Fig1":
        handles += [Line2D([], [], color="0.65", linewidth=3, label=r"RANS ($Re = 5\times10^5$)"),
                    Line2D([], [], color="k", marker="D", markerfacecolor="white", linestyle="", label="Wind tunnel (99 % CI)")]
    handles += [Line2D([], [], color="k", marker="x", linestyle="", markeredgewidth=1.2, label="Open-source VSM, same input")]
    rows = 3
    fig.legend(handles=handles, loc="lower center", ncol=3, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.075 * rows * 2.9 / height, 1, 1), w_pad=1.0)
    fig.savefig(HERE / name)
    plt.close(fig)


def wake_length():
    # convergence-results.zh.md: C2 (V3, 400-step run) and workstation 800-step run (M600)
    chords = {"V3": [10, 20, 40, 80], "M600": [10, 20, 40, 80, 160]}
    val = {"V3": {"CL": [0.570975, 0.572362, 0.572674, 0.572748], "CD": [0.051670, 0.051574, 0.051550, 0.051544],
                  "CM": [0.037934, 0.037180, 0.037000, 0.036956]},
           "M600": {"CL": [1.881342, 1.926112, 1.938629, 1.941550, 1.942216],
                    "CD": [0.068691, 0.062206, 0.059881, 0.059276, 0.059129],
                    "CM": [-0.038768, -0.078333, -0.085493, -0.086732, -0.086982]}}
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.75))
    for ax, q, yl, tag in zip(axes, ("CL", "CD", "CM"), (r"$|\Delta C_L|/C_L$ (%)", r"$|\Delta C_D|/C_D$ (%)",
                                                        r"$|\Delta C_{M_y}|/|C_{M_y}|$ (%)"), "abc"):
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
        grid(ax)
        letter(ax, f"({tag})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    fig.savefig(HERE / "fig06_wake_length.png")
    plt.close(fig)


def time_step():
    # convergence-results.zh.md: C1 with direct-sum finest levels; V3 7.35 deg from jobs 20261002b
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
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9))
    for ax, q, yl, tag in zip(axes, ("CL", "CD", "CM"), (r"$|\Delta C_L|/C_L$ (%)", r"$|\Delta C_D|/C_D$ (%)",
                                                        r"$|\Delta C_{M_y}|/|C_{M_y}|$ (%)"), "abc"):
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
        grid(ax)
        letter(ax, f"({tag})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    fig.savefig(HERE / "fig10_time_step.png")
    plt.close(fig)


def spanwise_core():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.7))
    # (a) convergence-results.zh.md C3 (fixed 0.075 m: cumulative level changes); V3 7.35 deg core 0.05 m (jobs 20261002b)
    a.plot([1, 2, 3], [0, 0.14, 0.14 + 0.09], "-", color=COL["V3"], marker="o", label="V3 kite, 5.41°, 0.075 m")
    v = np.array([0.707104, 0.706744, 0.707958])
    a.plot([1, 2, 3], (v / v[0] - 1) * 100, "-", color=COL["V3b"], marker="^", label="V3 kite, 7.35°, 0.05 m")
    a.plot([1, 2, 3], [0, -0.36, -0.36 - 0.13], "-", color=COL["M600"], marker="s", label="M600, 0°, 0.075 m")
    a.set_xticks([1, 2, 3])
    a.set_xlabel("Sub-panels per panel")
    a.set_ylabel(r"$C_L/C_L(\times 1) - 1$ (%)")
    a.set_ylim(-0.95, 0.3)
    a.legend(loc="lower left", fontsize=9)
    grid(a)
    letter(a, "(a)")
    # (b) outcome map: Table 9 (w_min from the case files, V3 0.203 m and M600 0.235 m at x1)
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
    handles, names = b.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower right", ncol=2, fontsize=9.5, columnspacing=1.0, handletextpad=0.3,
               bbox_to_anchor=(1.0, 0.0))
    grid(b)
    letter(b, "(b)")
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.6)
    fig.savefig(HERE / "fig11_spanwise_core.png")
    plt.close(fig)


def core():
    # convergence-results.zh.md C4; mean wing panel widths 0.318 m (V3) and 0.755 m (M600) from the case files
    rc = np.array([0.075, 0.15, 0.30])
    val = {"V3": (0.318, {"CL": [0.565516, 0.572249, 0.587436], "CD": [0.051002, 0.051583, 0.052165],
                          "CM": [0.037154, 0.037244, 0.035123]}),
           "M600": (0.755, {"CL": [1.932251, 1.932147, 1.944221], "CD": [0.061143, 0.061127, 0.061444],
                            "CM": [-0.082946, -0.082099, -0.047805]})}
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.75))
    for ax, q, yl, tag in zip(axes, ("CL", "CD", "CM"), (r"$\Delta C_L/C_L$ (%)", r"$\Delta C_D/C_D$ (%)",
                                                        r"$\Delta |C_{M_y}|/|C_{M_y}|$ (%)"), "abc"):
        for model, marker in (("V3", "o"), ("M600", "s")):
            width, values = val[model]
            v = np.abs(np.array(values[q]))
            ax.plot(rc / width, (v / v[1] - 1) * 100, "-", color=COL[model], marker=marker,
                    label=("V3 kite, 5.41°" if model == "V3" else "M600, 0°"))
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_xlabel(r"$r_c\,/\,\bar{w}$")
        ax.set_ylabel(yl)
        grid(ax)
        letter(ax, f"({tag})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    fig.savefig(HERE / "fig12_core.png")
    plt.close(fig)


def tree():
    def read(path, state):
        rows = []
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["state"] == state and row["targets"] == "wake" and int(row["leaf"]) == 32:
                    rows.append((int(row["order"]), float(row["theta"]), float(row["max_err_over_U"]),
                                 float(row["rms_err_over_U"])))
        return rows
    sets = [(EVID / "v1-m600-40-200.csv", "state_0200.npz"), (EVID / "v1-v3-40-200.csv", None)]
    with sets[1][0].open(encoding="utf-8", newline="") as stream:
        states = sorted({row["state"] for row in csv.DictReader(stream)})
    v3_state = [s for s in states if "200" in s][0]
    sets[1] = (sets[1][0], v3_state)
    colours = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.3))
    for ax, (path, state), tag in zip(axes[:2], sets, "ab"):
        rows = read(path, state)
        for order in (0, 1, 2):
            sel = sorted([r for r in rows if r[0] == order], key=lambda r: -r[1])
            theta = [r[1] for r in sel]
            ax.semilogy(theta, [r[3] for r in sel], "-", color=colours[order], marker="o")
            ax.semilogy(theta, [r[2] for r in sel], linestyle=DASH, color=colours[order], marker="o", markerfacecolor="white")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$|\Delta \mathbf{u}|/U_\infty$")
        grid(ax)
        letter(ax, f"({tag})")
    handles = [Line2D([], [], color=colours[o], marker="o", label=f"Order {o}") for o in (0, 1, 2)]
    handles += [Line2D([], [], color="k", linestyle="-", marker="o", label="RMS"),
                Line2D([], [], color="k", linestyle=DASH, marker="o", markerfacecolor="white", label="Maximum")]
    first = handles
    # (c) local-benchmark-threads-run2.zh.txt
    threads = [1, 4, 16]
    times = {("M600", "direct"): [38.701, 12.385, 4.243], ("M600", 0.3): [1.231, 0.647, 0.321], ("M600", 0.2): [2.216, 1.190, 0.493],
             ("V3", "direct"): [18.781, 4.640, 1.981], ("V3", 0.3): [0.831, 0.318, 0.201], ("V3", 0.2): [1.625, 0.551, 0.319]}
    ax = axes[2]
    line = {"direct": "-", 0.3: (0, (1, 1.5)), 0.2: DASH}
    for (model, kind), t in times.items():
        ax.loglog(threads, t, linestyle=line[kind], color=COL[model], marker="s" if model == "M600" else "o",
                  markerfacecolor=COL[model] if kind == "direct" else "white")
    ax.set_xticks(threads)
    ax.set_xticklabels(["1", "4", "16"])
    ax.minorticks_off()
    ax.set_xlabel("Threads")
    ax.set_ylabel("Wall time (s)")
    grid(ax)
    letter(ax, "(c)")
    handles = [Line2D([], [], color=COL["M600"], marker="s", label="M600"), Line2D([], [], color=COL["V3"], marker="o", label="V3 kite"),
               Line2D([], [], color="k", linestyle="-", label="Direct sum"),
               Line2D([], [], color="k", linestyle=line[0.3], label=r"Tree, $\theta = 0.3$"),
               Line2D([], [], color="k", linestyle=DASH, label=r"Tree, $\theta = 0.2$")]
    ax.set_ylim(0.1, 100)
    fig.legend(handles=first, loc="lower left", ncol=3, columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.0, 0.0),
               fontsize=9)
    fig.legend(handles=handles, loc="lower right", ncol=3, columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(1.0, 0.0),
               fontsize=9)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.8)
    fig.savefig(HERE / "fig14_tree.png")
    plt.close(fig)
    return v3_state


def profiles():
    """Spanwise profiles in the cross-flow plane 5 chords behind the trailing-edge centroid of the M600."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True)
    colours = ["#b6e3a8", "#5cc0a0", "#2f96c0", "#1f5fa8", "#0b2c6b"]
    for col, (name, pick, labels) in enumerate((("m600_wakelength_profile_5c.json", colours,
                                                 ["10 chords", "20 chords", "40 chords", "80 chords", "160 chords"]),
                                                ("m600_core_profile_5c.json", ["#5cc0a0", "#1f5fa8", "#D55E00"],
                                                 ["0.075 m", "0.15 m", "0.30 m"]))):
        entries = json.loads((HERE / name).read_text(encoding="utf-8"))["entries"]
        for entry, colour, label in zip(entries, pick, labels):
            axes[0, col].plot(entry["y_over_c"], entry["w_over_U"], color=colour, label=label)
            axes[1, col].plot(entry["y_over_c"], entry["u_over_U"], color=colour, label=label)
        axes[0, col].set_ylabel(r"$w/U_\infty$")
        axes[1, col].set_ylabel(r"$u/U_\infty$")
        axes[1, col].set_xlabel(r"$y/c$")
        handles, names = axes[1, col].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.29 + 0.49 * col, 0.0), ncol=3, fontsize=9.5,
                   columnspacing=0.9, handlelength=1.3, handletextpad=0.4)
        for row in range(2):
            grid(axes[row, col])
            letter(axes[row, col], f"({'abcd'[2 * row + col]})")
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2, h_pad=0.8)
    fig.savefig(HERE / "fig09_wake_profiles.png")
    plt.close(fig)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    style()
    comparison("Fig1", ("CL", "CD", "CMy"), (r"$C_L$", r"$C_D$", r"$C_{M_y}$"), "fig04_v3_comparison.png", 6.5, 3.5)
    comparison("Fig2", ("CL", "CD"), (r"$C_L$", r"$C_D$"), "fig05_m600_comparison.png", 5.2, 3.6)
    wake_length()
    profiles()
    time_step()
    spanwise_core()
    core()
    v3_state = tree()
    names = ["fig04_v3_comparison.png", "fig05_m600_comparison.png", "fig06_wake_length.png", "fig09_wake_profiles.png",
             "fig10_time_step.png", "fig11_spanwise_core.png", "fig12_core.png", "fig14_tree.png"]
    inputs = ["plotted-series.csv", "plotted-series-freewake-converged.csv", "m600_wakelength_profile_5c.json",
              "m600_core_profile_5c.json"]
    receipt = {"producer": "make_draft_figures.py", "producer_sha256": sha(__file__), "created": "2026-10-02",
               "status": "figures drawn by the paper session from the evidence files; draft 4 (fonts enlarged, renumbered)",
               "style": "decision 0024: no in-figure text apart from axis labels, legends and panel letters",
               "inputs": {n: sha(HERE / n) for n in inputs} | {n: sha(EVID / n) for n in ("v1-m600-40-200.csv", "v1-v3-40-200.csv")},
               "tree_states": {"m600": "state_0200.npz", "v3": v3_state},
               "typed_numbers": "see comments in the script: convergence-results.zh.md, local-benchmark-threads-run2.zh.txt, jobs 20261002b",
               "outputs": {n: sha(HERE / n) for n in names}, "matplotlib": matplotlib.__version__}
    (HERE / "draft-figure-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("\n".join(names))
