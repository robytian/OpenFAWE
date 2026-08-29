# Changelog

All notable changes to OpenFAWE are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from the first code
release the project adheres to [Semantic Versioning](https://semver.org/).

Archived releases are never modified. If a released result is later found to be wrong, a new version
is published that says so and the archived record is annotated; the original is not replaced.

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
