# Changelog

All notable changes to OpenFAWE are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from the first code
release the project adheres to [Semantic Versioning](https://semver.org/).

Archived releases are never modified. If a released result is later found to be wrong, a new version
is published that says so and the archived record is annotated; the original is not replaced.

---

## [1.0.0] — 2026-11-20

The whole flight of the Makani M600 energy kite and OpenFAWE Studio, the Windows program that runs and
replays it. The program is under Releases (`OpenFAWE-Studio-1.0.0-win64.zip`); this repository gains the
source the program is built from.

### Added

- `src/openfawe/flight` (the interface to Makani's M600 flight controller), `src/openfawe/ground` (the
  GS02 ground station with its winch, the perch, winds, the whole flight), `src/openfawe/power` (motors
  and the power system) and the modules they need: the source of what OpenFAWE Studio runs, 78 modules in
  `src/openfawe`, the same set as in the packaged program.
- `apps/studio`: OpenFAWE Studio - the program, its packaging scripts and its tests.
- `docs/user-guide.md` and `docs/user-guide.zh.md` (the user guide, with the validation chapter),
  `docs/record-keys.md` and `docs/record-keys.zh.md` (the recorded quantities) and the media of the front
  page.
- `docs/tasks`: what the program loads at run time (the whole flight on the vortex method, the power
  settings, the energy account) and the build of Makani's controller without the GNU Scientific Library;
  the folder names there carry the numbers of development records that are not published.
- `third_party/makani-controller-win-nogsl`: Makani's flight controller compiled as a Windows library
  (Apache-2.0; built from Makani's source without the GNU Scientific Library). On Linux it is built with
  `docs/tasks/OF-0013-release-v1-20261112/nogsl/build_nogsl_linux.py`.
- `cases/`: the models and tables of the whole flight and the qualification cases of its tests.
- `tests/`: the tests of the added modules.
- `MANIFEST-1.0.0.json`: every file this release adds or changes, with its SHA-256 and origin.

### Changed

- `README.md`, `LICENSE-BINARIES.md` (a clause for the LGPL libraries of the packaged program),
  `pyproject.toml`, `src/openfawe/__init__.py`, `third_party/MAKANI-NOTICE.md`, `CITATION.cff` and
  `.zenodo.json` describe release 1.0.0.
- The files of release 0.2.0 that the whole flight changed are replaced by the versions the program is
  built from; `MANIFEST-1.0.0.json` lists them. `MANIFEST.json` remains the record of release 0.2.0.

### Known issues

- Two of the 831 tests fail, and are published as they are: the variable-length tether at full length
  against the fixed-length one (`tests/test_variable_line.py`: 7.8e-11 against the bound of 1e-12 written
  first) and the kite at rest on the perch (`tests/test_perch.py`: 40 mm and 0.8° within 10 s against 5 mm
  and 0.1°). Both come from methods taken over from Makani as they are, and the program was not changed
  to pass them; section 11 of the user guide gives the causes. Eight tests of release 0.2.0 compare with
  development records or use modules that are not published and stop with an error; 17 are skipped, each
  with its reason.
- The source is published as it is, with the internal labels of its development records still in the
  comments.

### Not included

- The floating platform, the mooring and two kites on a shared mooring: the next version.

---

## [0.2.0] — 2026-11-05

Aerodynamic module with rotor, flight and tether models, and the code and data of the paper
*Free-vortex-wake dynamics of airborne wind energy kites in crosswind flight* (Y. Tian, S. Wang,
arXiv:2610.37313): verification of the method, the kite on its crosswind loop, the rotor slipstreams and
two kites in one wake.

A first version of this release, with the aerodynamic module and the verification material only, was
published on 2026-10-10 and withdrawn the same day; this release replaces it.

### Added

- `src/openfawe/aero` and `aero_archive.py`, `run_catalog.py`, `aero_demo.py`, `aero_vtk.py`,
  `aero_render.py`, `provenance.py`: lifting line, vortex step method and vortex-lattice lifting surface
  with straight and free vortex wakes; NumPy, Numba, CPU tree-code and GPU tree-code (Numba CUDA)
  Biot–Savart backends; the section hand-over of the crosswind closure; run archives, ParaView export and
  PyVista renders.
- `src/openfawe/rotor`: actuator discs with the Makani Rev4 rotor tables and the free vortex-ring wake
  of the rotors in the same wake network as the wing.
- `src/openfawe/flight`: prescribed rigid-body motions (`kinematics`, `prescribed`), the nominal crosswind
  loop of the Makani M600 controller (`crosswind_loop`), several kites in one vortex system
  (`multi_prescribed`) and the load account (`residual`).
- `src/openfawe/tether`: lumped-mass tether and bridle.
- The source code is the development repository at commit `901d096e`, the newest commit of the paper
  runs, except the package docstring.
- `cases/`: TU Delft V3 kite and Makani M600 models and section data; the M600 rigid body, crosswind
  playbook (land set), rotor set and Rev4 table, GS02 ground station and the pylon variants.
- `tests/`: the tests of the published modules (586 tests pass; three optional tests are skipped when
  PyVista, Qt or an internal record of the development repository is absent, and four tests that compare
  with internal records are deselected in `tests/conftest.py`).
- `paper1/`: run drivers, job lists, analyses, figure scripts and data of the verification part
  (`scripts/`, `extra/`, `fig/`, computed with the development repository at `e58677c`) and of the
  crosswind part (`crosswind/`: the command line of every run, read-back and figure scripts, the
  read-back files of the runs used in the paper and the receipts of eight production runs).
  `MANIFEST.json`, `paper1/MANIFEST.json` and `paper1/crosswind/MANIFEST.json` list the origin and
  SHA-256 of every file.
- `docs/media/`: the animation and still of the README and the script that draws them.
- `pyproject.toml`, `requirements-paper1.txt`, `NOTICE`, `MANIFEST.json`.

### Changed

- The licence of the Windows research-preview program moved from `LICENSE.md` to
  `LICENSE-BINARIES.md`; the repository content is under the Apache License 2.0 (`LICENSE`).
- `README.md`, `CITATION.cff` and `.zenodo.json` describe the released code and cite the arXiv paper.

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

[0.2.0]: https://github.com/robytian/OpenFAWE/releases/tag/v0.2.0
[0.1.0]: https://github.com/robytian/OpenFAWE/releases/tag/v0.1.0-preview
