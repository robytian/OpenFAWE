"""Single-column figures of Sec. IV of the two-column manuscript (author 2026-10-26 17:13, 17:14 and 17:16: figures
split and enlarged next to their paragraphs, one palette). The same data and quantities as the earlier double-width
figures of wake_analysis.figure_loads (c1a_loads) and wake_figures.figure_tip_displacement (c1a_tip_displacement); only
the layout (panels stacked in one column of 3.37 in) and the colours (pof_style.PALETTE, laps in Blues) change.

  loads_column      C_Z, C_m, aerodynamic power and tether tension of every lap against the controller's loop angle
  tips_column       radial, vertical and streamwise displacement of the wing-tip nodes from pure convection with the wind

Usage: pof_column_figs.py OUT_DIR C1A_RUN_DIR
"""
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parents[1]/"src"))
import pof_style as ps  # noqa: E402
import wake_analysis as wa  # noqa: E402
import wake_figures as wf  # noqa: E402


def loads_column(run_dir, out, double=False):
    import matplotlib.pyplot as plt
    history, states, kind = wa.read_run(run_dir)
    s = history["settings"]
    n = wa.period_steps(history, kind)
    body = wa.body_records(history, kind, None)
    area, chord = float(s.get("reference_area_m2", 32.9285)), float(s.get("reference_chord_m", 1.2831))
    cf, cm = wa.coefficients(body, area, chord)
    x = body["loop_angle_deg"]
    periods = (len(body["step"])-1)//n
    ps.style()
    if double:                      # four panels (a) to (d) as a 2 x 2 grid across the page (author 2026-10-26 17:19)
        fig, grid = plt.subplots(2, 2, figsize=(ps.WIDTH_DOUBLE_IN, 4.6), sharex=True)
        axes = grid.ravel()
    else:
        ps.column_style()
        fig, axes = plt.subplots(4, 1, figsize=(ps.WIDTH_SINGLE_IN, 7.4), sharex=True)
    for k in range(periods):
        sl = slice(k*n+1, (k+1)*n+1)
        order = np.argsort(x[sl])
        kw = dict(color=ps.LAPS[min(k, len(ps.LAPS)-1)], lw=1.1, label=f"lap {k+1}")
        axes[0].plot(x[sl][order], cf[sl, 2][order], **kw)
        axes[1].plot(x[sl][order], cm[sl, 1][order], **kw)
        axes[2].plot(x[sl][order], body["power_W"][sl][order]/1e3, **kw)
        axes[3].plot(x[sl][order], body["tension_N"][sl][order]/1e3, **kw)
    for ax, lab, k in zip(axes, (r"$C_Z$", r"$C_m$", "aerodynamic power (kW)", "tether tension (kN)"), "abcd"):
        ax.set_ylabel(lab)
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 90))
        ps.grid(ax)
        ps.letter(ax, f"({k})")
    for ax in (axes[2:] if double else axes[-1:]):
        ax.set_xlabel("loop angle (deg)")
    axes[0].legend(loc="lower left", ncol=2)
    fig.tight_layout(h_pad=0.8, w_pad=1.6)
    ps.save(fig, Path(out)/("c1a_loads_2x2" if double else "c1a_loads_col"))
    return dict(periods=periods, steps_per_lap=n)


def tips_column(run_dir, out):
    import matplotlib.pyplot as plt
    run = wf.read(run_dir)
    t = wf.tip_displacement(run)
    ps.style()
    ps.column_style()
    fig, axes = plt.subplots(3, 1, figsize=(ps.WIDTH_SINGLE_IN, 4.5), sharex=True)
    for name in ("outer", "inner"):
        d = t[name]
        laps = d["age_s"]/t["period_s"]
        edges = np.linspace(0., laps.max(), 49)
        for ax, key, scale in ((axes[0], "dr", t["radius"]), (axes[1], "dz", t["diameter"]), (axes[2], "dx", t["diameter"])):
            ax.plot(laps, d[key]/scale, ".", ms=0.7, color=ps.COL[name], alpha=0.2, rasterized=True)
            xc, m = wf.binned(laps, d[key]/scale, edges)
            ax.plot(xc, m, color=ps.COL[name], lw=1.3, label=f"{name} tip")
    axes[0].set_ylabel(r"$\Delta r/R$")
    axes[1].set_ylabel(r"$\Delta z/D$")
    axes[2].set_ylabel(r"$\Delta x/D$")
    for ax, k in zip(axes, "abc"):
        ps.grid(ax)
        ps.letter(ax, f"({k})")
    axes[-1].set_xlabel("age (loops)")
    axes[0].legend(loc="best")
    fig.tight_layout(h_pad=0.8)
    ps.save(fig, Path(out)/"c1a_tip_displacement_col")
    return dict(radius=float(t["radius"]), diameter=float(t["diameter"]))


if __name__ == "__main__":
    out, run_dir = sys.argv[1], sys.argv[2]
    Path(out).mkdir(parents=True, exist_ok=True)
    print(loads_column(run_dir, out))
    print(loads_column(run_dir, out, double=True))
    print(tips_column(run_dir, out))
