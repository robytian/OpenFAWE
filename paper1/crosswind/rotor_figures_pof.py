"""Figures of Secs. III E and III F of the paper (OF-0009, 2026-10-26) from the result files and the receipts.

  fig_tube_mechanism   (a) largest deviation of the axial velocity of a row of discrete rings from the continuous sheet at
                       distance d outside the wall, against h/sigma (results/tube_mechanism.json); (b) axial velocity of the
                       continuous regularised sheet against the distance from the wall; (c) effect of the rotors on C_Z over
                       the first second against the surface core for the tube with refinement at dt 0.02 and 0.01 s, the
                       tube without refinement at fine dt, and model B (results/rotor_evidence_1s.json).
  fig_dt_treatments    effect of the rotors on C_Z and C_m over the first second against dt for free rings, tube A, tube A'
                       with refinement and model B (results/rotor_evidence_1s.json).
  fig_thrust           effect of the rotors on C_Z and C_m over the first second against the total rotor thrust (tube A'
                       with refinement, rotor speeds 60, 97.4, 160 and 200 rad/s), and with only the newest ring kept.
  fig_loop_loads       last lap along the loop: change of C_Z and C_m by the rotors for the undeformed tube (p1a minus p1n,
                       dt 0.01 s) and model B (mbl5 minus mbn, dt 0.005 s), shaft power and thrust, aerodynamic power.
  fig_array_ratios     lift and aerodynamic-power ratios of unit B to unit A over the last lap for every two-unit run.

Figures carry no text inside the plot area apart from axes, ticks, legends and panel letters (author rule).

The abscissa of fig_loop_loads is the controller's loop angle of every step (c3_figures.loop_angle_deg, cached in
CACHE_DIR); until 2026-10-26 it was the fraction of the lap in time, 360 (1 - k/n), which differs from the loop angle by up
to 23 deg on this loop (loop_angle_readback.py).

Usage: rotor_figures_pof.py OUT_DIR RECEIPTS_DIR [PC_RUNS_DIR] [CACHE_DIR] [--column]
"""
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import pof_style as ps  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RHO, AREA, CHORD = 1.225, 32.9285, 1.2831
CACHE = {"dir": None}
# Names of the slipstream treatments in the legends (2026-10-26 restructure of the manuscript): the tubes A, A' and A''
# become T1, T2 and T3 and model B becomes model D (deflected), so that they are not confused with the units A and B of
# the two-unit runs. Only the legend text changes; the data keys below keep the old names.
RENAME = [(r"A$''$", "T3"), (r"A$'$", "T2"), ("tube A,", "tube T1,"), ("tube A ", "tube T1 "), ("model B", "model D")]


# Single-column layout for the two-column version of the paper (author 2026-10-26 17:13, 17:14): panels stacked in one
# column of 3.37 in with fonts of 8 to 9 pt. Set by the option --column; the default keeps the double-width layout.
COLUMN = {"on": False}
COLUMN_IN = 3.37


def column_style():
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "lines.markersize": 3.5})


def relabel(fig):
    for legend in [ax.get_legend() for ax in fig.axes]+list(fig.legends):
        if legend is None:
            continue
        for text in legend.get_texts():
            label = text.get_text()
            for old, new in RENAME:
                label = label.replace(old, new)
            if label.endswith("tube A"):
                label = label[:-1]+"T1"
            text.set_text(label)


def save(fig, path):
    relabel(fig)
    ps.save(fig, path)
P = ps.PALETTE                                             # one palette for the paper (author 2026-10-26 17:16)
COLOURS = {"free rings": P["grey"], "tube A": P["green"], "tube A'": P["sky"], "tube A' 0.1": P["blue"], "tube A''": P["blue"],
           "model B": P["vermilion"], "none": P["light grey"]}


def evidence():
    rows = json.loads((TASK/"results/rotor_evidence_1s.json").read_text())["rows"]
    seen, out = set(), []
    for r in rows:                                   # one row per distinct setting (drop duplicate diagnostic runs)
        if "missing_reference" in r:
            continue
        key = (r["treatment"], round(r["dt_s"], 6), r["surface_core_m"], r["overlap"], r["nodes"], r["kept_steps"], r["ring_core_m"],
               r["omega_rad_s"], r["bound_core_m"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def select(rows, **kw):
    sel = [r for r in rows if all((r.get(k) == v if not isinstance(v, float) else (r.get(k) is not None and abs(r[k]-v) < 1e-9))
                                  for k, v in kw.items())]
    return sorted(sel, key=lambda r: r["dt_s"])


def fig_tube_mechanism(out):
    m = json.loads((TASK/"results/tube_mechanism.json").read_text())
    rows = evidence()
    if COLUMN["on"]:
        fig, axes = plt.subplots(3, 1, figsize=(COLUMN_IN, 5.1))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_DOUBLE_IN, 2.35))
    ax = axes[0]
    for (key, data), c in zip(m["discreteness"].items(), (P["vermilion"], P["green"], P["blue"])):
        h = [r["h_over_sigma"] for r in data]
        e = [max(r["largest_deviation"], 1e-9) for r in data]
        ax.loglog(h, e, "o-", color=c, ms=3, label=rf"$d/\sigma = {key.split('=')[1]}$")
    ax.axvline(0.5, color="0.6", lw=0.8, ls=ps.DASH)
    ax.set_xlabel(r"$h/\sigma$")
    ax.set_ylabel(r"$\max|u - u_{\mathrm{sheet}}|/\gamma$")
    ax.set_ylim(1e-8, 1)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="lower right")
    ps.letter(ax, "(a)")
    ax = axes[1]
    d = np.asarray(m["smoothing"]["d_over_sigma"])
    u = np.asarray(m["smoothing"]["axial_over_gamma"])
    ax.plot(d, u, "o-", color=P["blue"], ms=3, label="regularised sheet")
    ax.plot([-4, 0, 0, 4], [1, 1, 0, 0], color="0.4", lw=0.9, ls=ps.DASH, label="ideal sheet")
    ax.set_xlabel(r"distance from the wall $/\sigma$")
    ax.set_ylabel(r"$u/\gamma$")
    ax.legend(loc="center right")
    ps.letter(ax, "(b)")
    ps.grid(ax)
    ax = axes[2]
    for dt, mk, lab in ((0.02, "s", r"tube A$'$, refined, $\Delta t = 0.02$ s"), (0.01, "o", r"tube A$'$, refined, $\Delta t = 0.01$ s")):
        sel = sorted([r for r in rows if r["treatment"] == "tube A'" and r["overlap"] and abs(r["dt_s"]/dt-1.) < 1e-3 and r["nodes"] == 16
                      and r["kept_steps"] in (100, 200) and r["ring_core_m"] == 0.45 and r["omega_rad_s"] == 97.4], key=lambda r: r["surface_core_m"])
        ax.plot([r["surface_core_m"] for r in sel], [r["dCZ"] for r in sel], mk+"-", color=COLOURS["tube A' 0.1"], ms=4, label=lab,
                mfc="white" if dt == 0.02 else COLOURS["tube A' 0.1"])
    fine = [r for r in rows if r["treatment"] == "tube A" and not r["overlap"] and r["dt_s"] < 0.0013]
    fine = sorted({r["surface_core_m"]: r for r in fine if r["nodes"] in (16, 64)}.values(), key=lambda r: r["surface_core_m"])
    ax.plot([r["surface_core_m"] for r in fine], [r["dCZ"] for r in fine], "^", color=COLOURS["tube A"], ms=5,
            label=r"tube A, $\Delta t = 0.00125$ s")
    mb = [r for r in rows if r["treatment"] == "model B" and abs(r["dt_s"]/0.0025-1.) < 1e-3]
    ax.plot([r["surface_core_m"] for r in mb], [r["dCZ"] for r in mb], "D", color=COLOURS["model B"], ms=4, label=r"model B, $\Delta t = 0.0025$ s")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"surface core $\sigma$ (m)")
    ax.set_ylabel(r"$\Delta C_Z$")
    ax.set_xlim(0.0, 0.33)
    ax.legend(loc="center left", fontsize=7, bbox_to_anchor=(0.0, 0.42) if not COLUMN["on"] else (0.25, 0.45))
    ax.set_ylim(-0.11, 0.06)
    ps.letter(ax, "(c)")
    ps.grid(ax)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, out/"fig_tube_mechanism")


def fig_dt_treatments(out):
    rows = evidence()
    series = [("free rings", dict(treatment="free rings"), r"free rings, $\sigma = 0.3$ m"),
              ("tube A", dict(treatment="tube A", overlap=None, surface_core_m=0.3, nodes=16), r"tube A, $\sigma = 0.3$ m"),
              ("tube A'", dict(treatment="tube A'", overlap=2.0, surface_core_m=0.3), r"tube A$'$, refined, $\sigma = 0.3$ m"),
              ("tube A''", dict(treatment="tube A'", overlap=2.0, surface_core_m=0.1, nodes=16, omega_rad_s=97.4, ring_core_m=0.45),
               r"tube A$'$, refined, $\sigma = 0.1$ m"),
              ("model B", dict(treatment="model B", surface_core_m=0.3), r"model B, $\sigma = 0.3$ m"),
              ("model B2", dict(treatment="model B", surface_core_m=0.1), r"model B, refined, $\sigma = 0.1$ m")]
    if COLUMN["on"]:
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 4.6), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE_IN, 2.7))
    for name, kw, lab in series:
        sel = select(rows, **kw)
        if name == "tube A''":
            sel = [r for r in sel if r["kept_steps"] in (100, 200)]
        if name == "tube A":
            sel = list({r["dt_s"]: r for r in sel}.values())
        sel = sorted(sel, key=lambda r: r["dt_s"])
        colour = COLOURS["tube A' 0.1"] if name == "tube A''" else COLOURS.get(name, COLOURS["model B"])
        filled = name not in ("model B", "tube A'")
        for ax, key in zip(axes, ("dCZ", "dCm")):
            ax.semilogx([r["dt_s"] for r in sel], [r[key] for r in sel], "o-" if filled else "s--", color=colour, ms=4,
                        mfc=colour if filled else "white", label=lab)
    prod = [r for r in rows if r["treatment"] == "tube A'' ambient"]
    for ax, key in zip(axes, ("dCZ", "dCm")):
        ax.plot([r["dt_s"] for r in prod], [r[key] for r in prod], "*", color=COLOURS["tube A''"], ms=9,
                label=r"tube A$''$, refined, $\sigma = 0.1$ m")
    for ax, lab, letter in zip(axes, (r"$\Delta C_Z$", r"$\Delta C_m$"), ("(a)", "(b)")):
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlabel(r"$\Delta t$ (s)")
        ax.set_ylabel(lab)
        ax.set_xlim(0.001, 0.05)
        ax.set_xticks([0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04])
        ax.set_xticklabels(["0.00125", "0.0025", "0.005", "0.01", "0.02", "0.04"], fontsize=8)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ps.letter(ax, letter)
        ps.grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    if COLUMN["on"]:
        axes[0].set_xlabel("")
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, handlelength=2.0, columnspacing=0.8)
        fig.tight_layout(rect=(0, 0.21, 1, 1), h_pad=0.6)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.13))
        fig.tight_layout(w_pad=1.5)
    save(fig, out/"fig_dt_treatments")


def fig_thrust(out):
    rows = evidence()
    sel = sorted([r for r in rows if r["treatment"] == "tube A'" and abs(r["dt_s"]/0.01-1.) < 1e-3 and r["surface_core_m"] == 0.1
                  and r["nodes"] == 16 and r["kept_steps"] == 200 and r["ring_core_m"] == 0.45], key=lambda r: r["thrust_N"])
    one = [r for r in rows if r["treatment"] == "tube A'" and r["kept_steps"] == 1]
    if COLUMN["on"]:
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.6), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE_IN, 2.4))
    for ax, key, lab, letter in zip(axes, ("dCZ", "dCm"), (r"$\Delta C_Z$", r"$\Delta C_m$"), ("(a)", "(b)")):
        t = np.array([r["thrust_N"] for r in sel])/1e3
        y = np.array([r[key] for r in sel])
        ax.plot(t, y, "o-", color=COLOURS["tube A' 0.1"], ms=4, label="rings kept 2 s")
        if one:
            ax.plot([one[0]["thrust_N"]/1e3], [one[0][key]], "x", color="0.2", ms=6, mew=1.2, label="newest ring only")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.axvline(0.0, color="0.6", lw=0.6)
        ax.set_xlabel(r"total rotor thrust $T$ (kN)")
        ax.set_ylabel(lab)
        ps.letter(ax, letter)
        ps.grid(ax)
    axes[0].legend(loc="upper right")
    if COLUMN["on"]:
        axes[0].set_xlabel("")
    fig.tight_layout(w_pad=1.5, h_pad=0.8)
    save(fig, out/"fig_thrust")


def last_lap(data):
    s = data["settings"]
    n, laps = s["steps_per_lap"], s["laps"]
    d = data["per_step"]["A"]
    st = np.asarray(d["step"])
    keep = (st > (laps-1)*n) & (st <= laps*n)
    q = 0.5*RHO*np.asarray(d["airspeed_m_s"])[keep]**2
    f = np.asarray(d["force_body_N"], float)[keep]
    mo = np.asarray(d["moment_origin_body_Nm"], float)[keep]
    import c3_figures as cf
    angle = cf.loop_angle_deg(s, CACHE["dir"])
    out = dict(angle=angle[st[keep]], CZ=f[:, 2]/(q*AREA), Cm=mo[:, 1]/(q*AREA*CHORD), P=np.asarray(d["aero_power_W"], float)[keep])
    if d.get("rotor_shaft_power_W"):
        out["shaft"] = np.asarray(d["rotor_shaft_power_W"], float)[keep]
        out["thrust"] = np.asarray(d["rotor_thrust_total_N"], float)[keep]
    return out


def ordered(x):
    i = np.argsort(x["angle"])
    return {k: v[i] for k, v in x.items()}


def fig_loop_loads(out, receipts):
    load = lambda n: json.loads((receipts/n/"checks.json").read_text())   # noqa: E731
    a2, n1 = ordered(last_lap(load("p1a-m600-single-rotors-adp-dt010-4laps"))), ordered(last_lap(load("p1n-m600-single-dt010-4laps")))
    b5, n5 = ordered(last_lap(load("mbl5-m600-single-rotors-modelb-rb064-sc010-dt005-3laps"))), ordered(last_lap(load("mbn-m600-single-dt005-3laps")))
    if COLUMN["on"]:
        fig, flat = plt.subplots(4, 1, figsize=(COLUMN_IN, 8.0), sharex=True)
        axes = np.array([[flat[0], flat[1]], [flat[2], flat[3]]])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH_DOUBLE_IN, 4.6), sharex=True)
    ax = axes[0, 0]
    ax.plot(a2["angle"], a2["CZ"]-n1["CZ"], color=COLOURS["tube A''"], label=r"undeformed tube A$''$")
    ax.plot(b5["angle"], b5["CZ"]-n5["CZ"], color=COLOURS["model B"], label="model B")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_ylabel(r"$\Delta C_Z$")
    ax.legend(loc="center left")
    ps.letter(ax, "(a)")
    ax = axes[0, 1]
    ax.plot(a2["angle"], a2["Cm"]-n1["Cm"], color=COLOURS["tube A''"])
    ax.plot(b5["angle"], b5["Cm"]-n5["Cm"], color=COLOURS["model B"])
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_ylabel(r"$\Delta C_m$")
    ps.letter(ax, "(b)")
    ax = axes[1, 0]
    ax.plot(a2["angle"], a2["shaft"]/1e3, color=COLOURS["tube A''"], label="shaft power")
    ax.plot(b5["angle"], b5["shaft"]/1e3, color=COLOURS["model B"], ls=ps.DASH)
    ax.set_ylabel("shaft power (kW)")
    ax2 = ax.twinx()
    ax2.plot(a2["angle"], -a2["thrust"]/1e3, color="0.35", lw=1.0, label=r"$-T$")
    ax2.set_ylabel(r"$-T$ (kN)")
    ax2.tick_params(direction="in")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper right")
    ax.set_xlabel("loop angle (deg)")
    ps.letter(ax, "(c)")
    ax = axes[1, 1]
    ax.plot(a2["angle"], (a2["P"]-n1["P"])/1e3, color=COLOURS["tube A''"], label=r"tube A$''$")
    ax.plot(b5["angle"], (b5["P"]-n5["P"])/1e3, color=COLOURS["model B"], label="model B")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_ylabel(r"$\Delta P_{\mathrm{aero}}$ (kW)")
    ax.set_xlabel("loop angle (deg)")
    ax.legend(loc="lower left")
    ps.letter(ax, "(d)")
    for ax in axes.flat:
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 90))
        ps.grid(ax)
    if COLUMN["on"]:
        axes[1, 0].set_xlabel("")
    fig.tight_layout(h_pad=1.0, w_pad=1.6)
    save(fig, out/"fig_loop_loads")


def fig_array_ratios(out, receipts, pc_runs):
    def load(name):
        for base in (receipts, pc_runs):
            if base is not None and (base/name/"checks.json").exists():
                return json.loads((base/name/"checks.json").read_text())
        raise SystemExit(f"no receipt for {name}")
    cases = [("1 D", "c3a-m600-pair-d1-5laps", "p2a-m600-pair-d1-rotors-adp-5laps"),
             ("1 D\nrings 2 laps", "c3a-m600-pair-d1-5laps", "p2b-m600-pair-d1-rotors-adp-keep1872-4laps"),
             ("1 D\n+ 0.5 D", "c3c7-m600-pair-d1-off05-7laps", "p5-m600-pair-d1-off05-rotors-adp-7laps"),
             ("2 D", "c3b-m600-pair-d2-7laps", "p4-m600-pair-d2-rotors-adp-7laps")]
    if COLUMN["on"]:
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.7), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE_IN, 2.5))
    x = np.arange(len(cases))
    for ax, key, lab, letter in zip(axes, ("lift_body_z_mean_N", "aero_power_mean_W"), (r"$1 - L_B/L_A$ (%)", r"$1 - P_B/P_A$ (%)"), ("(a)", "(b)")):
        loss0, loss1 = [], []
        for _, without, with_ in cases:
            lap = len(load(with_)["per_lap"]["A"])          # the last lap of the run with rotors, the same lap without
            for name, store in ((without, loss0), (with_, loss1)):
                d = load(name)
                a, b = d["per_lap"]["A"][lap-1], d["per_lap"]["B"][lap-1]
                store.append(100.*(1.-b[key]/a[key]))
        ax.bar(x-0.18, loss0, 0.34, color=COLOURS["none"], label="without rotors")
        ax.bar(x+0.18, loss1, 0.34, color=COLOURS["tube A''"], label=r"with rotors, tube A$''$")
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in cases], fontsize=8)
        ax.set_ylim(0, (0.95 if key.startswith("lift") else 3.8))
        ax.set_ylabel(lab)
        ps.letter(ax, letter)
        ps.grid(ax)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.5, h_pad=0.8)
    save(fig, out/"fig_array_ratios")


def main(out, receipts, pc_runs=None, cache=None, *flags):
    CACHE["dir"] = cache or str(TASK/".cache")
    COLUMN["on"] = "--column" in flags
    Path(CACHE["dir"]).mkdir(parents=True, exist_ok=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ps.style()
    if COLUMN["on"]:
        column_style()
    receipts = Path(receipts)
    pc_runs = Path(pc_runs) if pc_runs else None
    fig_tube_mechanism(out)
    fig_dt_treatments(out)
    fig_thrust(out)
    fig_loop_loads(out, receipts)
    fig_array_ratios(out, receipts, pc_runs)


if __name__ == "__main__":
    main(*sys.argv[1:])
