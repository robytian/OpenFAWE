"""Combined Figs. 8 and 9 of the English manuscript v10 (paper session, 2026-10-10).

Owner 2026-10-10 15:58: chapter-4 physics cut in v7 restored; 16:06: several figures may be merged and explained in one
paragraph. The profiles behind the tail of the M600 (Chinese v5 Fig. 14) return split by topic. No solver is called:
  Fig. 8  (a), (b) the wake-length law of make_discussion_figures.wake_length_law (../extra/*.json)
          (c), (d) downwash and axial velocity for wake lengths, make_draft_figures.profiles (m600_wakelength_profile_5c.json)
  Fig. 9  (a), (b) the closure Jacobian of make_discussion_figures.closure (../extra/steady-analyses.json)
          (c), (d) downwash and axial velocity for core radii, make_draft_figures.profiles (m600_core_profile_5c.json)
The plotting code repeats those functions with the axes passed in and the panel letters (a)-(d), so that the plotted
data are the same as in figD1_wake_length_law.png, figD3_closure.png and fig09_wake_profiles.png, which stay unchanged
for the Chinese draft v5. Fig. 7 of v10 is made by make_fig07_wake_q_planes.py.
Outputs: fig08_law_profiles, fig09_closure_profiles (.png and .pdf), combined-figure-receipt.json.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import make_draft_figures as mdf
import make_discussion_figures as mdi
import make_model_figures as mmf
import render3d as r3

HERE = Path(__file__).resolve().parent
PARTS = HERE / "parts"
COL = mdi.COL
receipt = {"created": "2026-10-10", "inputs": {}, "values": {}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def letter(fig, ax, text):
    box = ax.get_position()
    fig.text(box.x0, box.y1 + 0.004, text, fontweight="bold", fontsize=11, va="bottom", ha="left")


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png")
    fig.savefig(HERE / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- Figs. 8 and 9: law, closure and tail profiles
def profile_row(fig, grid, name, pick, labels, legend_y):
    """w and u profiles 5 chords behind the trailing-edge centroid of the M600, as make_draft_figures.profiles."""
    receipt["inputs"][name] = sha(HERE / name)
    entries = json.loads((HERE / name).read_text(encoding="utf-8"))["entries"]
    axes = [fig.add_subplot(grid[0, j]) for j in range(2)]
    for entry, colour, label in zip(entries, pick, labels):
        axes[0].plot(entry["y_over_c"], entry["w_over_U"], color=colour, label=label)
        axes[1].plot(entry["y_over_c"], entry["u_over_U"], color=colour, label=label)
    axes[0].set_ylabel(r"$w/U_\infty$")
    axes[1].set_ylabel(r"$u/U_\infty$")
    for ax in axes:
        ax.set_xlabel(r"$y/c$")
        mdf.grid(ax)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="center", bbox_to_anchor=(0.5, legend_y), ncol=len(labels), fontsize=9.5,
               columnspacing=1.2, handlelength=1.6, handletextpad=0.4)
    return axes


def fig8():
    fig = plt.figure(figsize=(6.5, 6.0))
    g1 = fig.add_gridspec(1, 2, left=0.105, right=0.985, top=0.955, bottom=0.62, wspace=0.32)
    la = [fig.add_subplot(g1[0, j]) for j in range(2)]
    wll, duh, resp = mdi.load("wake-length-law.json")["models"], mdi.load("duhamel.json")["models"], \
        mdi.load("response-test.json")
    law = mdi.law
    for model in ("V3", "M600"):
        m, d = wll[model], duh[model]
        area, chord, span = m["area"], m["chord"], m["betz"]["span"]
        cw = resp[model]["straight"]["CLa_eff"]
        f_inf = d["f_inf_straight"]["CL"]
        pts = {round(r["chords"], 6): r["CL"] for r in d["response"]}
        pts.update({float(r["chords"]): r["CL"] for r in m["straight"] if r["chords"] < 3000})
        chords = np.array(sorted(pts))
        eps_s = np.array([(f_inf - pts[c]) / f_inf for c in chords])
        x_s = chords * chord + 0.25 * chord
        keep = chords >= 2.0
        free = d["rows"]
        cf = np.array([r["chords"] for r in free])
        x_f = cf * chord + 0.25 * chord
        eps_f = np.array([r["eps_free"] for r in free])
        eps_d = np.array([r["eps_duhamel"] for r in free])
        thin = np.unique(np.round(np.geomspace(2.0, chords[keep].max(), 22), 1))
        idx = [int(np.argmin(np.abs(chords - t))) for t in thin]
        ax = la[0]
        ax.loglog(chords[idx], eps_s[idx], ls="none", marker="o" if model == "V3" else "s", mfc="white", color=COL[model],
                  ms=4.0)
        good = cf >= 1.5
        ax.loglog(cf[good], eps_f[good], color=COL[model], lw=1.3)
        xx = np.geomspace(1.5, 3000, 200)
        ax.loglog(xx, law(xx * chord + 0.25 * chord, cw, area), color=COL[model], lw=0.9, ls=mdf.DASH)
        ax = la[1]
        ratio_s = eps_s / law(x_s, cw, area)
        first = 7.0 if model == "M600" else 3.5
        ok = (chords >= first) & (chords <= 320)
        idx2 = [i for i in idx if ok[i]]
        ax.semilogx(x_s[idx2] / span, ratio_s[idx2], ls="none", marker="o" if model == "V3" else "s", mfc="white",
                    color=COL[model], ms=4.0)
        good = cf >= (10.0 if model == "M600" else 3.5)
        ax.semilogx(x_f[good] / span, eps_f[good] / law(x_f[good], cw, area), color=COL[model], lw=1.3)
        ax.semilogx(x_f[good] / span, eps_d[good] / law(x_f[good], cw, area), color=COL[model], lw=1.0, ls=(0, (1, 1.2)))
    ax = la[0]
    ax.set_xlabel(r"Distance of the starting vortex, $L/c$")
    ax.set_ylabel(r"Lift deficit $\varepsilon_L$")
    ax.set_xlim(1.5, 3000)
    ax.set_ylim(1e-8, 0.2)
    ax = la[1]
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"Distance from the closure points, $x/b$")
    ax.set_ylabel(r"$\varepsilon_L\,/\,(C_w S/16\pi x^2)$")
    ax.set_ylim(0.6, 1.6)
    for ax in la:
        mdf.grid(ax)
    handles = [Line2D([], [], color=COL["V3"], lw=1.3, label="V3"), Line2D([], [], color=COL["M600"], lw=1.3, label="M600"),
               Line2D([], [], color="0.3", ls="none", marker="o", mfc="white", label="straight finite wake"),
               Line2D([], [], color="0.3", lw=1.3, label="free wake"),
               Line2D([], [], color="0.3", lw=1.0, ls=(0, (1, 1.2)), label="superposition"),
               Line2D([], [], color="0.3", lw=0.9, ls=mdf.DASH, label="far-field law")]
    fig.legend(handles=handles, loc="center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 0.515), handlelength=1.6,
               columnspacing=1.0)
    g2 = fig.add_gridspec(1, 2, left=0.105, right=0.985, top=0.43, bottom=0.135, wspace=0.32)
    pa = profile_row(fig, g2, "m600_wakelength_profile_5c.json", ["#b6e3a8", "#5cc0a0", "#2f96c0", "#1f5fa8", "#0b2c6b"],
                     ["10 chords", "20 chords", "40 chords", "80 chords", "160 chords"], 0.03)
    fig.canvas.draw()
    for ax, t in zip(la + pa, "abcd"):
        letter(fig, ax, f"({t})")
    save(fig, "fig08_law_profiles")


def fig9():
    data = mdi.load("steady-analyses.json")["closure"]
    fig = plt.figure(figsize=(6.5, 6.3))
    g1 = fig.add_gridspec(1, 2, left=0.105, right=0.985, top=0.955, bottom=0.63, wspace=0.32)
    axes = [fig.add_subplot(g1[0, j]) for j in range(2)]
    floor = 3e-3
    dashes = {1: "-", 2: mdf.DASH, 3: (0, (1, 1.2))}
    table = {}
    for case, rows in data.items():
        for refine in (1, 2, 3):
            sel = [e for e in rows if e["refine"] == refine and e["geometry"] == "start"]
            r = np.array([e["ratio"] for e in sel])
            sig = np.array([e["at_solution"]["sigma_min"] if "at_solution" in e else np.nan for e in sel])
            beta = np.array([e["at_solution"]["beta_max"] if "at_solution" in e else np.nan for e in sel])
            axes[0].semilogy(r, sig, color=COL[case], ls=dashes[refine], lw=1.0)
            axes[1].plot(r, beta, color=COL[case], ls=dashes[refine], lw=1.0)
            for e in sel:
                table[(case, refine, round(e["core"], 3))] = e
    for case, refine, core, outcome in mdi.TABLE7:
        e = table[(case, refine, round(core, 3))]
        sig = e["at_solution"]["sigma_min"] if "at_solution" in e else None
        y = sig if sig is not None else floor
        marker = {"c": "o", "f": "x", "r": "^"}[outcome]
        axes[0].plot(e["ratio"], y, marker=marker, color=COL[case], mfc="white" if outcome != "f" else COL[case],
                     ms=5.5 if outcome != "f" else 6.5, mew=1.3, ls="none", zorder=5)
    axes[0].axhline(0.1, color="0.4", lw=0.8, ls=mdf.DASH)
    axes[0].set_ylim(2e-3, 8)
    axes[0].set_xlim(0, 6.2)
    axes[0].set_xlabel(r"$r_c/w_{\min}$")
    axes[0].set_ylabel(r"$\sigma_{\min}(\mathbf{J})$ at the start step")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlim(0, 6.2)
    axes[1].set_ylim(-35, 5)
    axes[1].set_xlabel(r"$r_c/w_{\min}$")
    axes[1].set_ylabel(r"$\max_i\,\partial G_i/\partial\Gamma_i$ at the start step")
    for ax in axes:
        mdf.grid(ax)
    handles = [Line2D([], [], color=COL["V3"], lw=1.0, label=r"V3, 5.41$^\circ$"),
               Line2D([], [], color=COL["V3-7.35"], lw=1.0, label=r"V3, 7.35$^\circ$"),
               Line2D([], [], color=COL["M600"], lw=1.0, label="M600"),
               Line2D([], [], color="0.3", ls="-", lw=1.0, label="$n$ = 1"),
               Line2D([], [], color="0.3", ls=mdf.DASH, lw=1.0, label="$n$ = 2"),
               Line2D([], [], color="0.3", ls=(0, (1, 1.2)), lw=1.0, label="$n$ = 3"),
               Line2D([], [], color="0.3", marker="o", mfc="white", ls="none", label="completed"),
               Line2D([], [], color="0.3", marker="x", ls="none", label="failed"),
               Line2D([], [], color="0.3", marker="^", mfc="white", ls="none", label="reversed flow")]
    fig.legend(handles=handles, loc="center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 0.505), handlelength=1.6,
               columnspacing=1.0)
    g2 = fig.add_gridspec(1, 2, left=0.105, right=0.985, top=0.40, bottom=0.13, wspace=0.32)
    pa = profile_row(fig, g2, "m600_core_profile_5c.json", ["#5cc0a0", "#1f5fa8", "#D55E00"],
                     ["0.075 m", "0.15 m", "0.30 m"], 0.03)
    fig.canvas.draw()
    for ax, t in zip(axes + pa, "abcd"):
        letter(fig, ax, f"({t})")
    save(fig, "fig09_closure_profiles")


if __name__ == "__main__":
    mdf.style()
    fig8()
    fig9()
    receipt["inputs"].update(mdi.receipt["inputs"])
    receipt["outputs"] = {n: sha(HERE / n) for n in ("fig08_law_profiles.png", "fig09_closure_profiles.png")}
    (HERE / "combined-figure-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("written fig08_law_profiles, fig09_closure_profiles")
