# OpenFAWE

**An open framework for coupled floating airborne wind energy (F-AWE) dynamics and physics-linked cost screening.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21656345.svg)](https://doi.org/10.5281/zenodo.21656345)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/Docs%20%26%20data-CC%20BY%204.0-lightgrey.svg)](#licensing)

---

> ### Status: v0.1.0 — specification and roadmap release
>
> **This repository does not yet contain a production solver.** It currently publishes the scope,
> module boundaries, design basis, evidence and acceptance rules, and the release policy, so that
> they can be inspected before any results exist. Solver, array and cost components are added only
> as they pass the documented verification and component-validation checks described in
> [`docs/evidence-policy.md`](docs/evidence-policy.md).
>
> Planned sequence and gates: [`ROADMAP.md`](ROADMAP.md).

---

## What this is

OpenFAWE addresses the offshore engineering side of Floating Airborne Wind Energy: how a floating
station and its lines respond when a crosswind-flying airborne unit acts on them as a **moving,
direction-varying boundary condition** — and what the resulting design envelopes cost.

Power estimates alone cannot show whether an F-AWE concept stays dynamically and structurally
feasible offshore once waves, platform motion, airborne-line loads, mooring response, array
interference, access and replacement are included. OpenFAWE targets that pre-prototype decision gap
by turning coupled offshore physics into reproducible design and cost evidence.

## Three questions it is built to answer

**1 · Coupled single-unit response.** When does the coupled response of a floating station, its
airborne tether and its station-keeping lines depart from the superposition of separately evaluated
aerodynamic, hydrodynamic and mooring models?

For each pre-registered observable OpenFAWE reports a calibrated effect ratio

```
R = | y_full − y_superposition | / U95
```

where `U95` is the combined 95 % numerical and forcing uncertainty. Non-superposition is treated as
established only if `R > 1` for at least two observables from **different subsystems** — including
station motion and line load — and the exceedance recurs under **two forcing seeds**.

**2 · Array coupling.** Within geometrically feasible two-unit layouts, do combined wake and
shared-mooring interactions move energy, motion, tether, fairlead or anchor observables by more than
the combined uncertainty — or is there a defensible negligible-interaction boundary?

**3 · Physics to cost.** Does a configuration favoured by annual energy production alone survive
load-driven line and anchor sizing and bounded lifecycle assumptions?

## Scope — and what is deliberately outside it

**In scope.** Coupled offshore response under a moving, direction-varying airborne load; airborne-tether
and station-keeping line mechanics including shared topology; two-unit wake and shared-mooring
interaction; a surrogate with an explicit applicability domain; design-basis load envelopes carried
into line and anchor sizing, manufacturing-stage CAPEX and comparative levelised cost of energy.

**Out of scope.** OpenFAWE does **not** aim to advance airborne-wind flight physics, kite design or
flight-control synthesis — those inputs are taken from the published state of the art and frozen in
the design basis. Also excluded: full membrane aeroelasticity; fatigue-life modelling
(cycle-count-ready channels are archived, but no fatigue model is introduced); transition-control
simulation for launch, landing and recovery (prescribed load envelopes are used instead); and any
certification assessment.

The DB1–DB4 matrix adapts IEC 61400-3-2:2025 and DNV-ST-0119 logic as a traceable **research** design
basis. It is not a certification claim. See [`docs/design-basis.md`](docs/design-basis.md).

## Planned architecture

| Layer | Component | Origin |
|---|---|---|
| Aerodynamics | Lifting-surface / free-vortex-wake module for the airborne unit; vortex-step screening model | **new OpenFAWE code**, reimplemented from published formulations and regression cases |
| Control | Published pumping-cycle controller, frozen at project start; saturation and tether-force exceedance kept as diagnostics | frozen published input |
| Hydrodynamics | Potential-flow coefficients (HAMS; NEMOH-v3 quadratic-transfer-function checks) assembled into a Cummins model with viscous correction, Newman slow drift and wave-drift damping | reused open solvers + **new** assembly |
| Lines | One lumped-mass core (MoorDyn) in two configurations: a variable-length airborne tether carrying the pumping boundary condition, and submerged station-keeping lines with seabed contact and shared topology | reused open solver + **new** boundary condition, adapters and shared-load routing |
| Inflow | Turbulent wind fields (TurbSim); current treated explicitly | reused open solver |
| Coupling | Predictor–corrector macro-step layer, project adapters, automated evidence checks | **new OpenFAWE code** |
| Surrogate | Gaussian-process active learning with a variable-specific applicability map; constraint-crossing intervals are returned to the coupled solver rather than accepted | **new OpenFAWE code** |
| Cost | Design-basis load envelopes → line and anchor sizing → manufacturing-stage CAPEX → comparative LCoE under declared low / central / high lifecycle scenarios | **new OpenFAWE code** |
| Validation evidence only | Selected CFD (OpenFOAM) comparisons | evidence source, not a runtime dependency |

Architecture detail and interface boundaries: [`docs/architecture.md`](docs/architecture.md).

## Reference configuration

The reference case couples a published multi-megawatt rigid fixed-wing pumping system
(150.3 m² wing; 526–1434 m tether; up to 3.8 MW) to a VolturnUS-S-derived semi-submersible station
(243.3 m water depth; 20 m draft; three 851.55 m lines; 877.3 m anchor radius).

The motivating overlap is spectral: the pumping cycle sits at roughly 133–333 s while the station's
horizontal slow modes sit at roughly 65–210 s. The two bands overlap, which is precisely why
component superposition cannot be assumed.

## Provenance

OpenFAWE reimplements its free-vortex-wake kernels from **published formulations and regression
cases**. It does **not** import or call
[MRUT](https://github.com/robytian/MRUT) (Multi-Rotor/Unit Wind Turbines Tool,
[doi:10.17632/5jrdc8jb65.1](https://doi.org/10.17632/5jrdc8jb65.1)) — the maintainer's earlier
rotor-wake code, which remains a separate project under separate versioning.

Third-party solvers are reused through project adapters and keep their own licences; they are not
vendored into this repository.

**Traceability rule.** One versioned case identifier follows every input, solver run, evidence check
and cost result, so any released number can be traced back to the run that produced it.

## Repository layout

```
README.md              this file
ROADMAP.md             planned work, deliverables, milestones and gates
CHANGELOG.md           released versions
CONTRIBUTING.md        governance: contributions, review, dependencies, deprecation, releases
LICENSE                Apache-2.0
CITATION.cff           how to cite
.zenodo.json           archival metadata for tagged releases
docs/
  architecture.md      module map, interfaces, coupling scheme, physics-to-cost chain
  design-basis.md      DB1-DB4 load-case matrix and environmental content
  evidence-policy.md   verification vs validation, acceptance rules, uncertainty handling
```

## Licensing

- **Code owned by this project:** [Apache-2.0](LICENSE).
- **Documentation and benchmark data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Third-party dependencies:** retain their own licences. Restricted or partner-confidential inputs
  are never placed in this repository.

## Citation

Please cite the archived release rather than the repository URL:

> Tian, Y. *OpenFAWE: an open framework for coupled floating airborne wind energy dynamics and
> physics-linked cost screening.* Zenodo. https://doi.org/10.5281/zenodo.21656345

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## Development

Developed by **Yinong Tian** ([ORCID 0000-0003-1912-0232](https://orcid.org/0000-0003-1912-0232)) at
the Centre for Marine Technology and Ocean Engineering (CENTEC), Instituto Superior Técnico,
University of Lisbon, Portugal, with scientific supervision from
**Dr Shan Wang** ([ORCID 0000-0002-6990-8071](https://orcid.org/0000-0002-6990-8071)).

Questions, corrections and reuse requests are welcome through
[GitHub Issues](https://github.com/robytian/OpenFAWE/issues).
