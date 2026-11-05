"""The frozen far wake against the complete free wake for the single M600 on the 9 m/s loop (Sec. III E, first paragraph):
c3s (``c3_pair.py --single``, rows older than one lap drift with the wind, node 2, GPU tree code) against C1a
(``c1_loop.py``, every row free, node 1, CPU tree code). Same loop, aircraft, closure, core and step; the multi-body path of
c3_pair reproduces the single-body path bit for bit (tests/test_multi_prescribed.py), and the GPU tree code differs from the
CPU tree code by at most 3.9e-6 in the induced velocities (contract section 4 addition), so the difference of the loads is
the effect of freezing the far wake.

Written: frozen_check.json (lap means of C_X, C_Z, C_m, aerodynamic power and tension of both runs for the laps they share,
their relative differences, and the largest step-to-step difference of C_Z within each shared lap that both store densely)
and frozen_check.png/pdf (C_Z of both runs and their difference against the loop angle over the last shared lap).

Usage: frozen_check.py OUT_DIR C3S_RECEIPT_DIR C1A_RUN_DIR      (the C1a run directory holds history.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import pof_style as ps  # noqa: E402
import wake_analysis as wa  # noqa: E402

AREA_M2, CHORD_M = 32.9285, 1.2831


def main(out, c3s_dir, c1a_dir):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c3s = json.loads((Path(c3s_dir)/"checks.json").read_text(encoding="utf-8"))
    history, _, kind = wa.read_run(c1a_dir)
    n = wa.period_steps(history, kind)
    if int(c3s["settings"]["steps_per_lap"]) != n:
        raise SystemExit("the two runs have different steps per lap")
    body = wa.body_records(history, kind)
    free = wa.per_period(body, n, AREA_M2, CHORD_M)
    frozen = c3s["per_lap"]["A"]
    laps = []
    for f, z in zip(free, frozen):
        if f["period"] != z["lap"]:
            raise SystemExit("lap numbering differs")
        cz = z["force_body_coefficient_mean"]
        cm = z["moment_body_coefficient_mean"]
        row = dict(lap=z["lap"], CZ_free=f["CZ"], CZ_frozen=cz[2], CX_free=f["CX"], CX_frozen=cz[0], Cm_free=f["Cm"], Cm_frozen=cm[1],
                   power_kW_free=f["power_kW"], power_kW_frozen=z["aero_power_mean_W"]/1e3,
                   tension_kN_free=f.get("tension_mean_kN"),
                   tension_kN_frozen=None if "tether_tension_kite_N" not in z else z["tether_tension_kite_N"]["mean"]/1e3)
        row["CZ_rel"] = row["CZ_frozen"]/row["CZ_free"]-1.0
        row["power_rel"] = row["power_kW_frozen"]/row["power_kW_free"]-1.0
        row["Cm_diff"] = row["Cm_frozen"]-row["Cm_free"]
        row["CX_diff"] = row["CX_frozen"]-row["CX_free"]
        laps.append(row)
    # per-step comparison over the laps c3s stores densely (its last three) that C1a also has
    cf_free, _ = wa.coefficients(body, AREA_M2, CHORD_M)
    step_free = body["step"]
    per = c3s["per_step"]["A"]
    steps = np.asarray(per["step"])
    force = np.asarray(per["force_body_N"], dtype=float)
    q = 0.5*float(c3s["settings"]["density_kg_m3"])*np.asarray(per["airspeed_m_s"], dtype=float)**2
    cz_frozen = force[:, 2]/(q*AREA_M2)
    dense = []
    for lap in range(1, len(free)+1):
        sl = (steps >= (lap-1)*n+1) & (steps <= lap*n)
        if sl.sum() != n:
            continue
        idx = steps[sl]
        where = np.searchsorted(step_free, idx)
        diff = cz_frozen[sl]-cf_free[where, 2]
        dense.append(dict(lap=lap, CZ_max_abs_diff=float(np.abs(diff).max()), CZ_rms_diff=float(np.sqrt(np.mean(diff**2))),
                          angle=body["loop_angle_deg"][where] % 360.0, free=cf_free[where, 2], frozen=cz_frozen[sl], diff=diff))
    report = dict(free=str(c1a_dir), frozen=str(c3s_dir), laps=laps,
                  dense=[{k: v for k, v in d.items() if not isinstance(v, np.ndarray)} for d in dense])
    (out/"frozen_check.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    if dense:
        import matplotlib.pyplot as plt
        ps.style()
        d = dense[-1]
        order = np.argsort(d["angle"])
        fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE_IN, 2.5), constrained_layout=True)
        axes[0].plot(d["angle"][order], d["free"][order], color=ps.COL["M600"], lw=1.2, label="all rows free")
        axes[0].plot(d["angle"][order], d["frozen"][order], color="k", lw=1.0, ls=ps.DASH, label="frozen beyond one lap")
        axes[0].set_ylabel(r"$C_Z$")
        axes[0].legend(loc="best")
        axes[1].plot(d["angle"][order], d["diff"][order], color="k", lw=1.1)
        axes[1].axhline(0.0, color="0.6", lw=0.6)
        axes[1].set_ylabel(r"$\Delta C_Z$ (frozen $-$ free)")
        for ax, k in zip(axes, "ab"):
            ax.set_xlabel("loop angle (deg)")
            ax.set_xlim(0, 360)
            ax.set_xticks(range(0, 361, 90))
            ps.grid(ax)
            ps.letter(ax, f"({k})")
        ps.save(fig, out/"frozen_check")
    print(json.dumps(report["laps"], indent=1))
    print(json.dumps(report["dense"], indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
