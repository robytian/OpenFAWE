# Changelog

All notable changes to OpenFAWE are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from the first code
release the project adheres to [Semantic Versioning](https://semver.org/).

Archived releases are never modified. If a released result is later found to be wrong, a new version
is published that says so and the archived record is annotated; the original is not replaced.

---

## [0.3.0] — 2026-11-05

Code and data of the arXiv paper *Free-vortex-wake dynamics of airborne wind energy kites in crosswind
flight* (Y. Tian, S. Wang, arXiv:2610.37313): the kite on its crosswind loop, the rotor slipstreams and
two kites in one wake, together with the aerodynamic module and the verification material prepared
for the withdrawn release 0.2.0 (see below).

### Added

- `src/openfawe/rotor`: actuator discs with the Makani Rev4 rotor tables and the free vortex-ring wake
  of the rotors in the same wake network as the wing.
- `src/openfawe/flight`: prescribed rigid-body motions (`kinematics`, `prescribed`), the nominal crosswind
  loop of the Makani M600 controller (`crosswind_loop`), several kites in one vortex system
  (`multi_prescribed`) and the load account (`residual`).
- `src/openfawe/tether`: lumped-mass tether and bridle.
- `src/openfawe/aero`: the GPU direct sum and GPU tree code (`vortex_cuda`, `vortex_cuda_tree`, Numba
  CUDA), the section hand-over of the crosswind closure (`handover`), surface cores (`surface_cores`);
  `aero_render.py` (PyVista renders).
- `cases/`: the Makani M600 rigid body, crosswind playbook (land set), rotor set and Rev4 table, GS02
  ground station and the pylon variants of the reconstructed M600.
- `paper1/crosswind/`: run drivers (`c1_loop.py`, `c1_straight.py`, `c3_pair.py`), read-back and figure
  scripts, the read-back files of the runs used in the paper, the receipts of eight production runs,
  the command line of every run (`RUNS.md`) and a manifest with the origin and SHA-256 of every file.
- `tests/`: tests of the new modules (586 tests pass; three optional tests are skipped when PyVista, Qt or
  an internal record of the development repository is absent).
- `docs/media/`: the animation and still of this page and the script that draws them.

### Changed

- `src/openfawe/aero` and `aero_archive.py` are the development repository at commit `901d096e`, the
  newest commit of the paper runs (0.2.0: `e58677c`). The published files changed between the two
  commits with the crosswind, rotor-ring and GPU work; the tests of 0.2.0 pass unchanged.
- `paper1/README.md` refers to the arXiv paper; `third_party/MAKANI-NOTICE.md` lists the Makani routines
  and data added in this release.

---

## [0.2.0] — 2026-10-10 (withdrawn)

Aerodynamic module and the code and data of paper 1. Published and withdrawn the same day; its
content is included in 0.3.0.

### Added

- `src/openfawe/aero` and `src/openfawe/aero_archive.py`, `run_catalog.py`, `aero_demo.py`,
  `aero_vtk.py`, `provenance.py`: lifting line, vortex step method and vortex-lattice lifting surface
  with straight and free vortex wakes; NumPy, Numba and tree-code Biot–Savart backends; run archives.
  Identical to commit `e58677c` of the development repository except the package docstring.
- `cases/`: TU Delft V3 kite and Makani M600 models and section data.
- `tests/`: the tests of the published modules (375 tests pass; four tests that compare with internal records of the development repository are deselected in `tests/conftest.py`).
- `paper1/`: run drivers, job lists, analyses, figure scripts and the data behind every figure and table
  of the paper; `paper1/MANIFEST.json` lists the origin and SHA-256 of every file.
- `pyproject.toml`, `requirements-paper1.txt`, `NOTICE`.

### Changed

- The licence of the Windows research-preview program moved from `LICENSE.md` to
  `LICENSE-BINARIES.md`; the repository content is under the Apache License 2.0 (`LICENSE`).

---

## [0.1.0] — 2026-07-29

First public release. **Specification and roadmap only — no production solver.**

### Added

- `README.md` — scope, the three research questions, the calibrated non-superposition rule,
  planned architecture, reference configuration and provenance.
- `ROADMAP.md` — planned work over 24 relative months: work packages, task windows, effort,
  eight deliverables, three milestones with pass criteria **and stated fallbacks if they are not
  met**, and the release and archiving policy.
- `docs/design-basis.md` — the DB1–DB4 load-case matrix, what is simulated versus prescribed in each
  case, environmental content, record-length and seed rules, the exclusion of fatigue-life
  modelling, and the reference configuration.
- `docs/evidence-policy.md` — the separation of verification from component validation, the
  `R = |y_full − y_superposition| / U95` rule and its two-observable / two-subsystem / two-seed
  support conditions, per-module acceptance thresholds, the uncertainty budget, and the
  computed-versus-assumed split for cost conclusions.
- `docs/architecture.md` — module map across new code, reused open solvers and validation-evidence-only
  sources; the coupling scheme; the physics-to-cost chain; the traceability rule; and the explicit
  separation from MRUT.
- `CONTRIBUTING.md` — governance: maintainer, contribution and review rules, dependency review,
  release-time provenance review, deprecation policy, versioning and conduct.
- `CITATION.cff`, `.zenodo.json`, `LICENSE` (Apache-2.0).

### Not included

- Any solver, model implementation or numerical result.
- Any validated or verified claim. Nothing in this release has been through the acceptance rules it
  describes.

### Notes

This release exists so that the scope, design basis, acceptance rules and release policy are
inspectable *before* any results are produced, and so that later results can be checked against a
specification that was published first.

[0.1.0]: https://doi.org/10.5281/zenodo.21656345
