# Crosswind loops, rotor slipstreams and two units

Run drivers, read-back and figure scripts, and result files of the crosswind-flight part of

> Y. Tian, S. Wang, *Free-vortex-wake dynamics of airborne wind energy kites in crosswind flight*,
> arXiv:2610.37313 (2026), <https://doi.org/10.48550/arXiv.2610.37313>.

The Makani M600 flies the nominal crosswind loop of the Makani land playbook with its lumped-mass tether,
first alone, then with its eight rotors as actuator discs that release vortex rings, and finally as one of
two units in one free vortex wake.

## Runs

| Script | What it runs |
|---|---|
| `c1_loop.py` | One M600 on the nominal loop at 5, 9 or 13 m/s with the complete free wake, the wake network stored at the end of every lap (runs `c1a`, `c1b`, `c1c`). Uses the pitch-manoeuvre helpers in `g4/`. |
| `c1_straight.py` | The same poses solved one by one with a straight wake (quasi-steady reference of the self-induced loss). |
| `c3_pair.py` | One or two M600 units in one wake, with or without the eight rotors and their vortex rings, the far wake frozen beyond a chosen number of rows (`--single` for one unit). Every run of the rotor and two-unit studies. |

[`RUNS.md`](RUNS.md) lists the command line of every run. A run writes `history.json` (settings,
per-step loads, motions), `checks.json` (settings, verdicts, per-lap table and per-step loads of the last
laps) and, with `--save-laps`, `--save-final` or `--frames`, the wake networks `state_<step>.npz`.

## Results in this folder

| Folder | Content |
|---|---|
| `results/` | Read-back of the runs used in the paper (loads per lap, self-induced loss, wake structure, meridional deficit, rotor-ring studies, two-unit tables, periodicity checks). |
| `receipts/<run>/checks.json` | The compact receipts of eight production runs computed on the GPU computer: the single unit without rotors and with the undeformed slipstream tube at 0.01 s (`p1n`, `p1a`), without rotors and with the deflected slipstream at 0.005 s (`mbn`, `mbl5`), the tandem at one loop diameter without and with rotors (`c3a`, `p2a`, `p2b`) and at two diameters without rotors (`c3b`). |
| `mesh/` | The M600 fuselage surface drawn in the 3D figures (display only, not part of any computation). |

The complete time histories (`history.json`, 26 to 63 MB per run) and the stored wake networks
(several hundred MB per run) are not included. The commands in `RUNS.md` regenerate them.

## Read-back and figures

| Topic | Script | Output in `results/` |
|---|---|---|
| Loads around the loop, wake structure, Trevisi helical-wake comparison | `wake_analysis.py`, `wake_figures.py`, `trevisi_check.py` | `c1*_summary.json`, `c1*_wake_figures.json`, `c1*_trevisi.json` |
| Self-induced loss of the single kite | `loss_readback.py` (with `c1_straight.py`) | `c1*_self_induced_loss.json`, `c1_straight_laps.json` |
| Azimuthal mean velocity deficit behind the loop | `wake_figures.py`, `meridional_compare.py` | `c1a_meridional_deficit.json`, `r1_c3s_meridional.json` |
| Frozen far wake against the complete free wake | `frozen_check.py` | `c3s_frozen_check.json` |
| Rotor-ring treatments, ladders and rules | `ring_study.py`, `ladder_study.py`, `rules_study.py`, `sheet_study.py`, `rotor_evidence.py`, `ring_figures.py`, `ring_spacing_diag.py`, `sheet_start_diag.py` | `rs_study.json`, `ladder_study.json`, `rules_study_r1.json`, `sheet_study_*.json`, `rotor_evidence_1s.json`, `r1_rings_age.json`, `vb_dt_check/` |
| Slipstream tube near the wing, deflected slipstream | `tube_mechanism.py`, `rotor_wake_figs.py`, `singles_readback.py` | `tube_mechanism.json`, `fig18_*.json`, `singles_readback_20261024.json`, `mbl5_readback_20261026.json`, `modelb_laps_readback_20261026.json` |
| Two units | `pair_readback.py`, `c3_figures.py`, `k7_readback.py`, `pair_wake_render.py` | `pairs_table_20261026.json`, `p2a_readback_20261024.json`, `p2b_readback_20261025.json`, `p4_readback_20261024.json`, `p5_readback_20261024.json`, `c3_table_c3s_c3a.json`, `k7*.json`, `fig20_report_20261026.json` |
| Loop angle of the results along the loop | `loop_angle_readback.py` | `loop_angle_readback_20261026.json` |
| 3D renders and figure layout | `wake_render.py`, `rotor_wake_figs.py`, `pair_wake_render.py`, `pof_column_figs.py`, `rotor_figures_pof.py`, `compose_renders.py`, `airframe.py`, `pof_style.py` | figures |

Figures need Matplotlib and PyVista (`pip install -e ".[paper]"`).

## Provenance

The numerical code in `../../src` is the development repository at commit
`901d096ec28258a802f2ffe48b187bbe728dc3ef`, the newest commit of the runs listed here. The scripts and
result files are that repository's task folder at commit `cec9b21d6eeaabad372e54ecb382d4b1f41f5642`.
Every run records the commit it was computed with (`settings.git_commit`); all of them are ancestors of
`901d096e`. Between the runs of 23 to 26 October and `901d096e` the source files used here are
unchanged; earlier runs (C1, `c3a`, `c3b`) were computed before later additions to the code, which are
recorded in [`../../CHANGELOG.md`](../../CHANGELOG.md).

[`MANIFEST.json`](MANIFEST.json) lists every file of this folder with its origin and SHA-256. Edited
files differ from their origin only in file paths (this folder sits one level higher than the task
folder of the development repository) and in the wording of comments; no numerical code was changed.
The scripts keep the comments of the development repository, including references to its internal task
records (contract sections, check names such as K3 and K7, run names), which are not part of this release.
