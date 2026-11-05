# Paper 1: code and data

This folder holds the run drivers, the read-back and figure scripts and the result data of

> Y. Tian, S. Wang, *Free-vortex-wake dynamics of airborne wind energy kites in crosswind flight*,
> arXiv:2610.37313 (2026), <https://doi.org/10.48550/arXiv.2610.37313>.

The numerical code is in [`../src/openfawe`](../src/openfawe) and the kite, rotor, tether and loop models
in [`../cases`](../cases). Everything here is released under the Apache License 2.0 (see
[`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE)); third-party data keep their own licences
(see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)).

| Folder | Part of the paper |
|---|---|
| [`scripts/`](scripts), [`extra/`](extra), [`fig/`](fig) | Method and verification: the vortex step method against the open-source vortex step code and RANS, the tree code, the wake length and the starting vortex, the time step, the spanwise refinement and the vortex core, the conditioning of the circulation closure |
| [`crosswind/`](crosswind) | Crosswind flight: the M600 on its loop from 5 to 13 m/s, the eight rotors and their vortex rings, two units in one wake ([its own README](crosswind/README.md)) |

## Provenance

The verification part was computed with the development repository at commit
`e58677c06f092141cea79ed6fc545fd4c3fd3869`; [`MANIFEST.json`](MANIFEST.json) lists the files of that
part with their origin and SHA-256 as first assembled on 2026-10-10. The crosswind part and the
source code of this release are described in [`crosswind/README.md`](crosswind/README.md) and
[`crosswind/MANIFEST.json`](crosswind/MANIFEST.json); the source code is the development repository at
commit `901d096ec28258a802f2ffe48b187bbe728dc3ef`, and the tests of [`../tests`](../tests) pass with it.
Run records name the commit they were computed with (`git_commit`). The method-comparison values with
straight wakes and the same-input comparison with the open-source vortex step code come from earlier
campaigns of the development repository; their compiled values are included here, their raw run archives
are not.

Raw time histories and wake states (several GB in all) are not in this repository; the run scripts
regenerate them. The section and figure numbers in the scripts and in the tables below refer to an earlier
manuscript of the same work; the tables name the topic as well.

## Installation

Python 3.12 or newer.

```
python -m venv .venv
.venv/bin/pip install -e ".[fast,paper,test]"      # Windows: .venv\Scripts\pip
python -m pytest tests                              # from the repository root
```

`fast` installs Numba, used by the compiled direct sum and the tree code (the paper runs use the tree
code); `paper` installs Matplotlib and PyVista for the figures. The exact versions of the paper runs
are in [`../requirements-paper1.txt`](../requirements-paper1.txt).

## Layout

| Folder | Content |
|---|---|
| `scripts/convergence/` | Free-wake run driver (`run_history.py`), job lists and ladder definitions, the analyses of Section 3.3 and of the tree code (Section 2.3), and their outputs in `evidence/` |
| `scripts/methods/` | Drivers of the six method combinations on the two kites (Section 3.2, Fig. 4) |
| `extra/` | Chapter 4 analyses and their outputs (`*.json`), the lifting-surface free wakes of Fig. 4 (`ls_freewake/`) and the core-growth runs (`core-growth/`) |
| `fig/` | Figure scripts and the data files they plot |

Run outputs go to `../local/` (ignored by Git). `fig/wakelib.py` reads stored wake states from
`../local/accel-20261001/runs` or from the folder named by the environment variable `OPENFAWE_RUNS`.

## Runs

All commands from the repository root. One run, for example the baseline V3 history of the time-step
ladder (40 steps of 0.4 s, tree code θ = 0.2, order 2, 32 segments per leaf):

```
python paper1/scripts/convergence/run_history.py local/accel-20261001/runs/c1-v3-dt0400 V3 40 \
    --backend tree --theta 0.2 --order 2 --leaf 32 --dt 0.4
```

Each line of the `jobs-*.txt` files is one run in this form (name, model, steps, options);
`queue.sh` runs a job list. The ladders of Section 3.3 are defined in `ladders-*.json` and evaluated by
`analyse_convergence.py`, which writes `evidence/convergence-*.json`; `analyse_pc_rechecks.py` adds the
direct-sum rechecks of the finest levels and the 160-chord M600 wake; `paper1_tables.py` recomputes the
time-step numbers quoted in the text.

| Paper | Runs (job list) | Analysis | Output |
|---|---|---|---|
| Sec. 2.3, tree-code error and timing | `jobs-lane-a.txt`, `jobs-lane-b.txt`, `save_states.py` | `check_v1_frozen.py`, `check_v2_histories.py`, `summarise_v2.py`, `local_benchmark.py`, `check_a3_runs.py` | `evidence/v1-*`, `v2-summary.json`, `local-benchmark-*.json`, `a3-run-equivalence*.json` |
| Sec. 3.3, wake length and time step | `jobs-c-v3.txt`, `jobs-c-m600.txt`, `jobs-lane-*.txt` | `analyse_convergence.py ladders-c.json`, `analyse_pc_rechecks.py`, `paper1_tables.py` | `evidence/convergence-c*.json`, `paper1-tables.json` |
| Sec. 3.3, spanwise refinement and core radius | `jobs-c3s*.txt`, `jobs-paper1-b-20261002.txt`, `jobs-paper1-c-20261002.txt` (7.35°) | `analyse_convergence.py ladders-c.json`, `ladders-v3-a735.json` | `evidence/convergence-c.json`, `convergence-v3-a735.json` |
| Sec. 3.3, lifting line ladders | `jobs-paper1-20261002.txt` (runs `ll-*`) | `analyse_convergence.py ladders-ll.json` | `evidence/convergence-ll.json` |
| Sec. 4.1, stored wake states, Figs. 7–10 | `jobs-paper1-20261002.txt` (runs `p1-*`, with `--save`) | `render_wake_ladder.py` (profiles behind the tail) | `../fig/m600_*_profile_5c.json` |
| Sec. 3.2 and Fig. 4, VSM and LL free wakes at the converged length | `jobs-paper1-b-20261002.txt` | `paper1_combos.py` | `../fig/plotted-series-freewake-converged.csv` |
| Fig. 4, LSi and LSp free wakes | `extra/ls_freewake/jobs*.txt` with `run_ls_freewake.py` | `collect_ls_freewake.py` | `ls-freewake-summary.json`, rows of `plotted-series-freewake-converged.csv` |
| Fig. 4, straight wakes | `scripts/methods/run_campaign.py` (uses `run_lattice_comparison.py`, `run_comparison.py`) | | `../fig/plotted-series.csv` |

## Chapter 4 analyses (`extra/`)

These read stored free-wake states (runs `p1-v3-planes-wakelength` and `p1-m600-planes-wakelength`,
made with `run_history.py --save`) or solve steady problems directly.

| Paper | Script | Output |
|---|---|---|
| Sec. 4.1, Betz invariants and descent of the wake | `rollup.py` | `rollup.json` |
| Sec. 4.1, wake-length law, Eqs. (18)–(21), Fig. 9(a) | `steady_analyses.py`, `wake_length_law.py` | `steady-analyses.json`, `wake-length-law.json` |
| Sec. 4.1, response to a uniform downwash ($C_w$, $\kappa_D$) | `response_test.py` | `response-test.json` |
| Sec. 4.1, superposition of the shed history, Eq. (23), Fig. 9(b) | `duhamel.py` | `duhamel.json` |
| Sec. 4.1, shape of the wake, Eq. (24) | `geometry_swap.py`, `shed_centroid.py`, `far_end_velocity.py`, `far_end_consistent.py` | `geometry-swap.json`, `shed-centroid.json`, `far-end-*.json` |
| Sec. 4.1, Q criterion of Fig. 7(b), (d) | `q_field.py` | `q/*.npz` (not included, 190 MB), `q/q-tree-check.json` |
| Sec. 4.2, smallest singular value and self-coefficient, Fig. 10(a), (b) | `steady_analyses.py`, `jacobian_check.py` | `steady-analyses.json`, `jacobian-check.json` |
| Sec. 4.2, growing cores | `core_growth.py` | `core-growth/*/result.json` |
| Sec. 3.3, M600 loads by lifting surface | `surface_loads.py` | `surface-loads-T1p6.json` |

## Figures (`fig/`)

| Figure | Script | Data |
|---|---|---|
| 1, 3 | `make_model_figures.py` (PyVista, off-screen) | `../cases`; Fig. 3 of the paper is the upper row of `fig01_models.png` |
| 2 | `flowchart/make_flowchart_en7.js` (PptxGenJS) | none |
| 4, 5, 6 | `make_re_figures.py` (uses `make_draft_figures.py`) | `plotted-series.csv`, `plotted-series-freewake-converged.csv`; the numbers of Figs. 5 and 6 are typed from `scripts/convergence/evidence/convergence-*.json` with a comment at each |
| 7, 8 | `make_fig07_wake_q_planes.py --render` (3-D renders and plane fields; uses `wakelib.py`, `render3d.py`, `render_q.py`), then `make_fig07_fig08_split.py` (layout of Figs. 7 and 8) | stored wake states and `extra/q/*.npz` (regenerate first) |
| 9, 10 | `make_fig08_fig09_combined.py` (uses `make_discussion_figures.py`; writes `fig08_law_profiles.png` and `fig09_closure_profiles.png`, which are Figs. 9 and 10 of the paper) | `extra/*.json`, `m600_wakelength_profile_5c.json`, `m600_core_profile_5c.json` |

Figures 4, 5, 6, 9 and 10 are drawn from the data in this repository alone:

```
python paper1/fig/make_re_figures.py
python paper1/fig/make_fig08_fig09_combined.py
```

## Notes

- The comparison with the open-source vortex step code (Section 3.2) used
  [awegroup/Vortex-Step-Method](https://github.com/awegroup/Vortex-Step-Method) at revision `5485b82`.
- The scripts keep the comments and identifiers of the development repository, including references
  to its internal task records, which are not part of this release.
- The M600 fuselage mesh in `fig/assets/` is drawn in the figures for orientation only; it is not part
  of any computation.
