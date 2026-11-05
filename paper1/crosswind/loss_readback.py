"""Self-induced loss of the M600 on its loop (Sec. III D): the free-wake run of ``c1_loop.py`` against the same poses solved with
the straight wake (``c1_straight.py``, quasi-steady, no history of the loop).

Both runs use the same motion samples (same playbook entry, sphere, spline reading and steps per lap), so step k of lap n of
the free-wake run and step k of the straight-wake lap are the same pose. Reported per lap of the free-wake run: the lap means of
the body-axis lift (force along body z), the body-axis force along x, the aerodynamic power and the lift coefficient, each with
its relative difference to the straight wake; and for the last lap the step-by-step relative difference of the lift. The
aerodynamic loads only (no tether): the tether does not change the aerodynamic solution of a prescribed motion.

Usage: loss_readback.py FREE_RUN_DIR STRAIGHT_RUN_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

AREA_M2 = 32.9285


def records(run):
    data = json.loads((Path(run)/"history.json").read_text(encoding="utf-8"))
    return data, data["records"]


def series(rows):
    force = np.array([r["force_body_N"] for r in rows])
    q = np.array([r["dynamic_pressure_Pa"] for r in rows])
    power = np.array([r["aero_power_W"] for r in rows])
    return force, q, power


def main(free_dir, straight_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free, free_rows = records(free_dir)
    straight, straight_rows = records(straight_dir)
    n = straight["settings"]["steps_per_lap"]
    if free["settings"].get("steps_per_lap", n) != n:
        raise SystemExit("the two runs have different steps per lap")
    s_force, s_q, s_power = series(straight_rows[1:n+1])
    reference = dict(lift_N=float(s_force[:, 2].mean()), fx_N=float(s_force[:, 0].mean()), power_W=float(s_power.mean()),
                     CZ=float((s_force[:, 2]/(s_q*AREA_M2)).mean()))
    laps, lap = [], 1
    while lap*n < len(free_rows):
        force, q, power = series(free_rows[(lap-1)*n+1:lap*n+1])
        row = dict(lap=lap, lift_N=float(force[:, 2].mean()), fx_N=float(force[:, 0].mean()), power_W=float(power.mean()),
                   CZ=float((force[:, 2]/(q*AREA_M2)).mean()))
        row["lift_relative_to_straight"] = row["lift_N"]/reference["lift_N"]-1.
        row["power_relative_to_straight"] = row["power_W"]/reference["power_W"]-1.
        row["CZ_relative_to_straight"] = row["CZ"]/reference["CZ"]-1.
        laps.append(row)
        last = (force, q, power)
        lap += 1
    result = dict(free=str(free_dir), straight=str(straight_dir), steps_per_lap=n, straight_lap=reference, free_laps=laps)
    if laps:
        force, q, power = last
        step_lift = force[:, 2]/s_force[:, 2]-1.
        result["last_lap_step_lift_relative"] = dict(min=float(step_lift.min()), max=float(step_lift.max()), mean=float(step_lift.mean()))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            angle = 360.*np.arange(1, n+1)/n
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
            axes[0].plot(angle, s_force[:, 2]/(s_q*AREA_M2), color="0.4", label="straight wake")
            for k in range(1, len(laps)+1):
                f, qq, _ = series(free_rows[(k-1)*n+1:k*n+1])
                axes[0].plot(angle, f[:, 2]/(qq*AREA_M2), label=f"free wake, lap {k}")
            axes[0].set_xlabel("loop angle [deg]"); axes[0].set_ylabel("lift coefficient"); axes[0].legend(frameon=False, fontsize=7)
            axes[1].plot(angle, 100.*step_lift)
            axes[1].set_xlabel("loop angle [deg]"); axes[1].set_ylabel("lift, last lap against straight wake [%]")
            fig.savefig(out/"self_induced_loss.png", dpi=200)
            plt.close(fig)
        except ImportError:
            pass
    (out/"self_induced_loss.json").write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(json.dumps(dict(straight=reference, free_last=laps[-1] if laps else None), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
