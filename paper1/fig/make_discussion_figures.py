"""Chapter 4 (discussion) figures of an earlier version of paper 1 (paper session, 2026-10-08).

No solver is called here except reading stored states. Inputs (all in ../extra, written by the scripts there with
the unchanged code at e58677c):
  wake-length-law.json, duhamel.json, response-test.json   wake-length law (straight finite wake, free wake, Duhamel)
  steady-analyses.json                                     closure Jacobian (sigma_min, self-coefficient)
  stored free-wake states (wakelib.STATE)                  roll-up and descent
Style as make_draft_figures.py (Times New Roman, no text inside the plots apart from axis labels, legends, letters).
Outputs: figD1_wake_length_law.png, figD2_rollup.png, figD3_closure.png and discussion-figure-receipt.json.
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_draft_figures as mdf
import wakelib as wl

HERE = Path(__file__).resolve().parent
EXTRA = HERE.parent / "extra"
COL = {"V3": "#0072B2", "M600": "#D55E00", "V3-7.35": "#56B4E9"}
KEY = {"V3": "v3", "M600": "m600"}
receipt = {"created": "2026-10-08", "inputs": {}, "figures": {}, "values": {}}


def load(name):
    path = EXTRA / name
    receipt["inputs"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def law(x, cw, area):
    return cw * area / (16 * np.pi * x ** 2)


def wake_length_law():
    wll, duh, resp = load("wake-length-law.json")["models"], load("duhamel.json")["models"], load("response-test.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    out = {}
    for model in ("V3", "M600"):
        m, d = wll[model], duh[model]
        area, chord, span = m["area"], m["chord"], m["betz"]["span"]
        cw = resp[model]["straight"]["CLa_eff"]
        f_inf = d["f_inf_straight"]["CL"]
        # straight finite wake: row grid (duhamel.json) and long wakes (wake-length-law.json), common reference
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
        # (a) deficit against distance in chords
        ax = axes[0]
        sel = keep & (np.isin(np.round(chords, 3), np.round(chords, 3)))
        thin = np.unique(np.round(np.geomspace(2.0, chords[keep].max(), 22), 1))
        idx = [int(np.argmin(np.abs(chords - t))) for t in thin]
        ax.loglog(chords[idx], eps_s[idx], ls="none", marker="o" if model == "V3" else "s", mfc="white", color=COL[model],
                  ms=4.0)
        good = cf >= 1.5
        ax.loglog(cf[good], eps_f[good], color=COL[model], lw=1.3)
        xx = np.geomspace(1.5, 3000, 200)
        ax.loglog(xx, law(xx * chord + 0.25 * chord, cw, area), color=COL[model], lw=0.9, ls=mdf.DASH)
        # (b) ratio to the law against distance in spans
        ax = axes[1]
        ratio_s = eps_s / law(x_s, cw, area)
        first = 7.0 if model == "M600" else 3.5
        ok = (chords >= first) & (chords <= 320)
        idx2 = [i for i in idx if ok[i]]
        ax.semilogx(x_s[idx2] / span, ratio_s[idx2], ls="none", marker="o" if model == "V3" else "s", mfc="white",
                    color=COL[model], ms=4.0)
        good = cf >= (10.0 if model == "M600" else 3.5)
        ax.semilogx(x_f[good] / span, eps_f[good] / law(x_f[good], cw, area), color=COL[model], lw=1.3)
        ax.semilogx(x_f[good] / span, eps_d[good] / law(x_f[good], cw, area), color=COL[model], lw=1.0, ls=(0, (1, 1.2)))
        # values for the text
        def at(c, arr, grid):
            return float(arr[int(np.argmin(np.abs(grid - c)))])
        out[model] = dict(
            Cw=cw, area=area, chord=chord, span=span, bprime=m["betz"]["bprime"],
            L_0p1pct_chords=float(np.sqrt(cw * area / (16 * np.pi * 1e-3)) / chord - 0.25),
            L_0p1pct_spans=float(np.sqrt(cw * area / (16 * np.pi * 1e-3)) / span),
            straight_ratio_range_5_320=[float(ratio_s[(chords >= 5) & (chords <= 320)].min()),
                                        float(ratio_s[(chords >= 5) & (chords <= 320)].max())],
            straight_ratio_range_7_320=[float(ratio_s[(chords >= 7) & (chords <= 320)].min()),
                                        float(ratio_s[(chords >= 7) & (chords <= 320)].max())],
            free_ratio={f"{c}c": at(c, eps_f / law(x_f, cw, area), cf) for c in (10, 20, 40, 80) if c <= cf.max() + 0.5},
            duhamel_over_free={f"{c}c": at(c, eps_d / eps_f, cf) for c in (7, 10, 14, 20, 40, 80) if c <= cf.max() + 0.5})
    ax = axes[0]
    ax.set_xlabel(r"Distance of the starting vortex, $L/c$")
    ax.set_ylabel(r"Lift deficit $\varepsilon_L$")
    ax.set_xlim(1.5, 3000)
    ax.set_ylim(1e-8, 0.2)
    mdf.letter(ax, "(a)")
    ax = axes[1]
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"Distance from the closure points, $x/b$")
    ax.set_ylabel(r"$\varepsilon_L\,/\,(C_w S/16\pi x^2)$")
    ax.set_ylim(0.6, 1.6)
    mdf.letter(ax, "(b)")
    for ax in axes:
        mdf.grid(ax)
    handles = [Line2D([], [], color=COL["V3"], lw=1.3, label="V3"), Line2D([], [], color=COL["M600"], lw=1.3, label="M600"),
               Line2D([], [], color="0.3", ls="none", marker="o", mfc="white", label="straight finite wake"),
               Line2D([], [], color="0.3", lw=1.3, label="free wake"),
               Line2D([], [], color="0.3", lw=1.0, ls=(0, (1, 1.2)), label="superposition"),
               Line2D([], [], color="0.3", lw=0.9, ls=mdf.DASH, label="far-field law")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.02), handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "figD1_wake_length_law.png")
    if os.environ.get("PDF_OUT"):
        fig.savefig(HERE / "figD1_wake_length_law.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt["values"]["wake_length_law"] = out


def sections(key, step, stations):
    state = wl.read_state(key, step)
    rows, strengths, wing = state["rows"], state["strengths"], state["wing"]
    main = wl.surfaces(wing)[0].node_count
    trailing = wl.trailing_strength(strengths, wing.panel_edges, rows.shape[1])
    stream = wl.freestream(key, 1.0)
    ex = stream / np.linalg.norm(stream)
    en = np.cross(ex, [0.0, 1.0, 0.0])
    te = rows[1, :main]
    origin = te.mean(axis=0)
    right = np.where(te[:, 1] > 1e-9)[0]
    right = right[np.argsort(te[right, 1])]                     # root to tip
    half = te[:, 1].max()
    span = 2 * half
    mids = 0.5 * (rows[1:-1, right] + rows[2:, right])         # interval k = 1.. (rows k..k+1)
    s = ((mids - origin) @ ex).mean(axis=1)
    out = []
    for st in stations:
        k = int(np.argmin(np.abs(s - st * span)))
        p = mids[k] - origin
        g = trailing[k + 1, right]
        y, n = mids[k][:, 1], p @ en
        out.append(dict(x_b=float(s[k] / span), y=y / half, n=n / half, g=g,
                        yc=float((g * y).sum() / g.sum() / half), nc=float((g * n).sum() / g.sum() / half)))
    return out, span


def rollup():
    roll = load("rollup.json")["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.9))
    shades = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]
    oranges = ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"]
    info = {}
    for ax, key, step, stations, cmap in ((axes[0, 0], "m600", 513, (0.25, 1.0, 2.0, 4.0), oranges),
                                          (axes[0, 1], "v3", 360, (0.25, 1.0, 3.0, 6.0), shades)):
        secs, span = sections(key, step, stations)
        gmax = max(np.abs(sc["g"]).max() for sc in secs)
        for sec, color in zip(secs, cmap):
            ax.scatter(sec["y"], sec["n"], s=4 + 60 * np.abs(sec["g"]) / gmax, color=color, edgecolors="none", zorder=3)
            ax.plot(sec["yc"], sec["nc"], marker="x", color=color, ms=7, mew=1.6, ls="none", zorder=4)
        ax.set_xlabel(r"$y/(b/2)$")
        ax.set_ylabel(r"$n/(b/2)$")
        ax.set_aspect("equal", adjustable="datalim")
        mdf.grid(ax)
        info[key] = [dict(x_b=sc["x_b"], yc=sc["yc"], nc=sc["nc"], tip_filament_share=float(np.abs(sc["g"]).max() / np.abs(sc["g"]).sum()))
                     for sc in secs]
        ax.legend(handles=[Line2D([], [], color=c, ls="none", marker="o", ms=4, label=f"$x/b$ = {sc['x_b']:.2g}")
                           for sc, c in zip(secs, cmap)], fontsize=8.5, loc="upper left" if key == "m600" else "lower right",
                  handlelength=1.0)
    mdf.letter(axes[0, 0], "(a)")
    mdf.letter(axes[0, 1], "(b)")
    for model, key, name in (("M600", "m600", "m600-513"), ("V3", "v3", "v3-360")):
        e = roll[name]
        b, half = e["span"], e["span"] / 2
        s = np.array([r["s"] for r in e["rows"]]) / b
        length = s[-1]
        sel = s < length - 2.0
        yc = np.array([r["yc"] for r in e["rows"]]) / half
        nc = np.array([r["nc"] for r in e["rows"]]) / b
        axes[1, 0].plot(s[sel], yc[sel] / (e["betz_yc"] / half), color=COL[model], lw=1.2)
        axes[1, 1].plot(s[sel], nc[sel], color=COL[model], lw=1.2)
        # vortex-pair descent Gamma_h / (2 pi b') through the centroid at x = b
        gh = e["rows"][0]["gamma_h"]
        bprime = 2 * e["betz_yc"]
        slope = -abs(gh) / (2 * np.pi * bprime) / e["speed"]
        i1 = int(np.argmin(np.abs(s - 1.0)))
        xx = np.linspace(1.0, min(length - 2.0, 10.0), 50)
        axes[1, 1].plot(xx, nc[i1] + slope * (xx - s[i1]) * b / b, color=COL[model], lw=0.9, ls=mdf.DASH)
        info[model + "_pair_slope_per_span"] = float(slope)
    axes[1, 0].axhline(1.0, color="0.5", lw=0.8)
    axes[1, 0].set_ylim(0.97, 1.03)
    axes[1, 0].set_xlabel(r"$x/b$")
    axes[1, 0].set_ylabel(r"$y_c\,/\,y_{c,\mathrm{Betz}}$")
    axes[1, 1].set_xlabel(r"$x/b$")
    axes[1, 1].set_ylabel(r"$n_c/b$")
    axes[1, 0].set_xlim(0, 10)
    axes[1, 1].set_xlim(0, 10)
    mdf.letter(axes[1, 0], "(c)")
    mdf.letter(axes[1, 1], "(d)")
    for ax in axes[1]:
        mdf.grid(ax)
    axes[1, 1].legend(handles=[Line2D([], [], color=COL["V3"], lw=1.2, label="V3"),
                               Line2D([], [], color=COL["M600"], lw=1.2, label="M600"),
                               Line2D([], [], color="0.3", lw=0.9, ls=mdf.DASH, label=r"$\Gamma_h/(2\pi b')$")],
                      fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "figD2_rollup.png")
    if os.environ.get("PDF_OUT"):
        fig.savefig(HERE / "figD2_rollup.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt["values"]["rollup_sections"] = info


TABLE7 = [  # case, refine, core, outcome (c: completed, f: closure failure, r: reversed flow)  -- Table 7
    ("V3", 1, 0.075, "c"), ("V3", 1, 0.15, "c"), ("V3", 1, 0.30, "c"), ("V3", 2, 0.075, "c"), ("V3", 2, 0.15, "c"),
    ("V3", 3, 0.025, "c"), ("V3", 3, 0.05, "c"), ("V3", 3, 0.075, "c"), ("V3", 3, 0.15, "f"),
    ("V3-7.35", 1, 0.05, "c"), ("V3-7.35", 2, 0.05, "c"), ("V3-7.35", 3, 0.05, "c"), ("V3-7.35", 1, 0.075, "c"),
    ("V3-7.35", 2, 0.075, "c"), ("V3-7.35", 3, 0.075, "f"),
    ("M600", 1, 0.075, "c"), ("M600", 1, 0.15, "c"), ("M600", 1, 0.30, "c"), ("M600", 2, 0.075, "c"),
    ("M600", 2, 0.15, "c"), ("M600", 3, 0.075, "c"), ("M600", 3, 0.15, "f"), ("M600", 3, 0.30, "f"),
    ("M600", 3, 0.05, "r")]


def closure():
    data = load("steady-analyses.json")["closure"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
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
    out = []
    for case, refine, core, outcome in TABLE7:
        e = table[(case, refine, round(core, 3))]
        sig = e["at_solution"]["sigma_min"] if "at_solution" in e else None
        y = sig if sig is not None else floor
        marker = {"c": "o", "f": "x", "r": "^"}[outcome]
        axes[0].plot(e["ratio"], y, marker=marker, color=COL[case], mfc="white" if outcome != "f" else COL[case],
                     ms=5.5 if outcome != "f" else 6.5, mew=1.3, ls="none", zorder=5)
        out.append(dict(case=case, refine=refine, core=core, ratio=e["ratio"], outcome=outcome, sigma_min_start=sig))
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
    mdf.letter(axes[0], "(a)")
    mdf.letter(axes[1], "(b)")
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
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.02), handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(HERE / "figD3_closure.png")
    if os.environ.get("PDF_OUT"):
        fig.savefig(HERE / "figD3_closure.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt["values"]["closure_table7"] = out


if __name__ == "__main__":
    mdf.style()
    wake_length_law()
    rollup()
    closure()
    for name in ("figD1_wake_length_law.png", "figD2_rollup.png", "figD3_closure.png"):
        receipt["figures"][name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    (HERE / "discussion-figure-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt["values"], indent=1)[:4000])
