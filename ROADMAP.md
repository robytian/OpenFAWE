# OpenFAWE — Development Roadmap

**Project:** OpenFAWE — an open framework for coupled floating airborne wind energy (F-AWE) dynamics and physics-linked cost screening
**Maintainer:** Yinong Tian ([ORCID 0000-0003-1912-0232](https://orcid.org/0000-0003-1912-0232)) · CENTEC, Instituto Superior Técnico, University of Lisbon
**Repository:** https://github.com/robytian/OpenFAWE
**Document version:** 0.1.0

---

## How to read this document

- **Month numbers (M1–M24) are relative to project start**, not calendar dates. M1 is the first month of the 24-month development programme.
- This is a **development plan, contingent on the programme being resourced**. It is published so that the scope, sequence, acceptance rules and release points are inspectable in advance. It is not a delivery commitment, and it does not report completed or funded results.
- Every dated item below is either a **deliverable** (a released artefact) or a **milestone** (a gate with a stated pass criterion and a stated fallback if it is not met).
- Items marked **gated** proceed only if the preceding milestone passes. Items marked **unconditional** are released regardless of gate outcome.

---

## 1. Timeline at a glance

| Month | What happens | Type |
|---|---|---|
| M1–M4 | T1.1 — freeze architecture, controller, interfaces, tolerances and the DB1–DB4 design basis | task |
| M3 | Profile free-vortex-wake / HAMS / MoorDyn kernels over 3,600 s runs | checkpoint |
| M3–M10 | T1.2 — single-unit coupling, open-solver adapters, predictor–corrector macro-step layer | task |
| M6 | **Data Management Plan + repository governance and release policy** | deliverable (D4.2) |
| M7–M12 | T1.3 — verification, component validation, calibrated non-superposition tests, uncertainty assessment | task |
| M7–M13 | T2.1 — feasible-layout map over spacing, flight-envelope phase/altitude offset, tether clearance, non-crossing moorings | task |
| M9 | Re-baseline the integrated case; narrow, reassign or exclude unsupported channels | checkpoint |
| **M10** | **D1.1 — single-unit model and version-controlled code** | **deliverable** |
| **M12** | **D1.2 — verification and component-validation report** | **deliverable** |
| **M12** | **MS1 — single-unit evidence accepted** | **milestone / gate** |
| M12 | Public reproduction clinic | engagement |
| M13 | Cost definitions freeze (T3.1) | checkpoint |
| M13–M17 | T2.2 — two-unit coupled-array production runs (**gated on MS1**) | task |
| M13–M18 | T3.1 — component-sizing-to-CAPEX chain and discounted LCoE baseline | task |
| M15–M20 | T2.3 — Gaussian-process active-learning surrogate and applicability map | task |
| M16 | Interfaces and cost-model schema freeze | checkpoint |
| M16–M21 | T3.2 — global sensitivity and feasibility-preserving cost screening | task |
| **M17** | **D2.1 — two-unit shared-moored array model, code and dataset** | **deliverable** |
| **M18** | **D3.1 — line/anchor-sizing and comparative-LCoE model** | **deliverable** |
| M19 | Regression buffer (no new scope) | checkpoint |
| M20–M24 | T3.3 — integration, visual workflow, documentation and open release | task |
| **M20** | **D2.2 — surrogate, hold-out/interval assessment, applicability map** | **deliverable** |
| **M20** | **MS2 — array surrogate qualified** | **milestone / gate** |
| M20 | Array review with external users | engagement |
| **M21** | **D3.2 — cost-sensitivity benchmark and decision-stability dataset** | **deliverable** |
| M22 | Industrial–academic prototype workshop | engagement |
| **M23** | **D3.3 — Apache-2.0 release: manual, tutorials, software manuscript, archival DOI** | **deliverable** |
| **M23** | **MS3 — independent reproduction achieved** | **milestone / gate** |
| M24 | Corrections, reporting and uptake record; no new scope | closing |

Project management, training, data management, dissemination and communication run continuously from M1 to M24.

---

## 2. Work packages

### WP1 — Single-unit coupled dynamics and evidence · M1–M12 · 9 person-months

| Task | Window | Effort | Content |
|---|---|---|---|
| T1.1 | M1–M4 | 1.5 PM | Convert a VolturnUS-S-derived semi-submersible case to an F-AWE station. Freeze architecture, the published pumping-cycle controller, module interfaces, numerical tolerances and the DB1–DB4 design-basis matrix. |
| T1.2 | M3–M10 | 4.5 PM | Implement the airborne-wind lifting-surface / free-vortex-wake module from published formulations and regression cases. Connect HAMS/NEMOH-v3 hydrodynamics and two lumped-mass line configurations through project adapters. Add the predictor–corrector macro-step coupling layer. |
| T1.3 | M7–M12 | 3 PM | Code verification, component validation, calibrated non-superposition tests, uncertainty assessment. |

**Outputs**

- **D1.1 (M10)** — version-controlled single-unit model and code, with module interfaces, unit tests and a reproducible reference case.
- **D1.2 (M12)** — verification and component-validation report, benchmark data, an explicit applicability statement, and *(unconditional)* a leading-edge-inflatable (LEI) evidence map and interface specification covering deformation inputs, loads, tether attachment and applicability limits.

**MS1 (M12) — pass criterion.** The rigid-wing reference case meets the declared stability, convergence and evidence bands, and D1.2 reports the effect ratio *R* for every pre-registered observable.
**If not met.** WP2 continues on the prescribed-loading branch; the unresolved evidence needed for closure is stated explicitly. Unsupported variables and any unqualified LEI component case are labelled outside the accepted scope rather than described as validated.

---

### WP2 — Wake-coupled shared-moored arrays and qualified surrogate · M7–M20 · 7 person-months

| Task | Window | Effort | Content |
|---|---|---|---|
| T2.1 | M7–M13 | 1.5 PM | Map the geometrically feasible two-unit layout space over platform spacing, flight-envelope phase or altitude offset, tether clearance and non-crossing mooring constraints. |
| T2.2 | M13–M17 | 3.5 PM | **Gated on MS1.** Sample feasible no-overlap, partial-overlap and near-clearance-boundary layouts. Compare full-array, wake-disabled, shared-line-disabled and isolated-unit cases. |
| T2.3 | M15–M20 | 2 PM | Qualify a Gaussian-process active-learning surrogate inside the accepted evidence domain. |

**Outputs**

- **D2.1 (M17)** — two-unit shared-moored array model, code and dataset covering wake-interference and shared-anchor cases.
- **D2.2 (M20)** — Gaussian-process surrogate, hold-out and calibrated-interval assessment, applicability-domain map.

**MS2 (M20) — pass criterion.** Hold-out errors meet the variable-specific thresholds, and uncertain or out-of-domain queries are routed back to the coupled solver rather than answered by the surrogate.
**If not met.** D2.2 is restricted to the qualified sub-domain, and the four-unit array — a stretch case throughout — is dropped.

---

### WP3 — Physics-linked cost screening and reusable workflow · M13–M24 · 5 person-months

| Task | Window | Effort | Content |
|---|---|---|---|
| T3.1 | M13–M18 | 1.5 PM | Freeze cost definitions at M13. Implement the component-sizing-to-CAPEX chain and the discounted comparative-LCoE baseline. |
| T3.2 | M16–M21 | 1.5 PM | Propagate physical and scenario uncertainty through global sensitivity analysis and feasibility-preserving screening. |
| T3.3 | M20–M24 | 2 PM | Integrate physics, cost, visual and release workflows; documentation, tutorials, knowledge base and archival release. |

**Outputs**

- **D3.1 (M18)** — executable line/anchor-sizing and comparative-LCoE model, editable scenario template, frozen baseline and evidence-review record.
- **D3.2 (M21)** — cost-sensitivity benchmark and decision-stability dataset that separates physics-driven from assumption-driven conclusions.
- **D3.3 (M23)** — integrated Apache-2.0 release with visual workflow, user manual, tutorials, software manuscript and archival DOI.

**MS3 (M23) — pass criterion.** An independent user reproduces the benchmark, edits a declared lifecycle scenario, and traces the resulting decision class back to its physics, uncertainty and cost inputs.
**If not met.** M24 is reserved for correction and uptake evidence; no new scope is added.

---

### WP4 — Management, training, data and dissemination · M1–M24 · 3 person-months

Runs continuously. The output relevant to users of this repository is:

- **D4.2 (M6)** — Data Management Plan plus GitHub–Zenodo–OpenAIRE repository governance note, covering metadata, licensing, contribution rules, deprecation policy, dependency review and the release plan.

Remaining WP4 activity covers project management, training, reporting, dissemination and communication.

---

## 3. Deliverable summary

| ID | Deliverable | WP | Month |
|---|---|---|---|
| D1.1 | Version-controlled single-unit model and code, interfaces, unit tests, reference case | WP1 | M10 |
| D1.2 | Verification and validation report, benchmark data, LEI evidence/interface specification, applicability limits | WP1 | M12 |
| D2.1 | Two-unit shared-moored array model, code and wake/shared-anchor dataset | WP2 | M17 |
| D2.2 | Gaussian-process surrogate, hold-out and interval assessment, applicability map | WP2 | M20 |
| D3.1 | Line/anchor-sizing and comparative-LCoE model, frozen baseline, review record | WP3 | M18 |
| D3.2 | Cost-sensitivity benchmark with decision-stability dataset | WP3 | M21 |
| D3.3 | Apache-2.0 release, manual, tutorials, software manuscript, archival DOI | WP3 | M23 |
| D4.2 | Data Management Plan, repository governance and release plan | WP4 | M6 |

---

## 4. Release and archiving policy

| Release window | Content | Archive |
|---|---|---|
| M10–M12 | Single-unit model, code, benchmark data, verification/validation evidence | Zenodo DOI |
| M18–M20 | Array model and dataset, surrogate and applicability map, cost model and baseline | Zenodo DOI |
| M23 | Integrated release: manual, tutorials, knowledge base, software manuscript | Zenodo DOI |
| M24 | Corrections and uptake record | Zenodo DOI |

Each tagged release is archived on Zenodo and exposed through OpenAIRE. Locked environment specifications and container recipes (Docker / Apptainer) accompany each release so that archived results remain executable. Semantic versioning applies from the first code release.

**Licensing.** Project-owned code: Apache-2.0. Documentation and benchmark data: CC BY 4.0. Third-party dependencies retain their own licences and are reused through adapters, not vendored.

---

## 5. Scope boundaries

Stated here so that the roadmap cannot be read as promising more than it plans.

**In scope.** Coupled offshore response of a floating station under a moving, direction-varying airborne load; airborne-tether and station-keeping line mechanics including shared topology; two-unit wake and shared-mooring interaction; a qualified surrogate with an explicit applicability domain; design-basis load envelopes carried into line and anchor sizing, manufacturing-stage CAPEX and comparative LCoE.

**Out of scope.** Airborne-wind flight physics, kite design and flight-control synthesis — these are taken from the published state of the art and frozen in the design basis. Also excluded: full membrane aeroelasticity, fatigue-life modelling (cycle-count-ready channels are archived, but no fatigue model is introduced), transition-control simulation for launch, landing and recovery (prescribed load envelopes are used instead), and any certification assessment. The DB1–DB4 matrix adapts IEC 61400-3-2:2025 and DNV-ST-0119 logic as a traceable *research* design basis; it is not a certification claim.

**Stretch, not planned.** A four-unit array layout, gated behind MS2. An LEI component-transfer case, gated behind MS1 — note that the LEI *evidence map and interface specification* is unconditional and ships with D1.2 regardless.

---

## 6. Provenance

OpenFAWE reimplements its free-vortex-wake kernels from published formulations and regression cases. It does **not** import or call MRUT (Multi-Rotor/Unit Wind Turbines Tool, [doi:10.17632/5jrdc8jb65.1](https://doi.org/10.17632/5jrdc8jb65.1)), the maintainer's earlier rotor-wake code, which remains a separate project under separate versioning. Third-party solvers — potential-flow hydrodynamics, lumped-mass line dynamics, turbulent inflow generation — are reused through project adapters under their own licences.

Traceability rule: one versioned case identifier follows every input, solver run, evidence check and cost result, so that any released number can be traced to the run that produced it.

---

*Month numbers are relative to project start. This roadmap describes planned work and is subject to the gates and fallbacks stated above.*
