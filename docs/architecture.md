# Architecture

**Status:** specification, v0.1.0. Module boundaries and interfaces are fixed here before
implementation, so that the boundary between new code, reused solvers and evidence sources cannot
drift silently.

---

## 1. Three categories, kept visually and legally distinct

| Category | What it means | Examples |
|---|---|---|
| **New OpenFAWE code** | Written for this project, Apache-2.0, maintained here | Airborne lifting-surface / free-vortex-wake module; time-domain orchestrator; adapters; evidence checks; surrogate; cost chain |
| **Reused open solver** | Third-party, called through an adapter, under its own licence, **not vendored** | Potential-flow hydrodynamics (HAMS, NEMOH-v3); lumped-mass line dynamics (MoorDyn); turbulent inflow (TurbSim) |
| **Validation evidence only** | Used to produce comparison data; **not** a runtime dependency | Selected CFD (OpenFOAM) results |

A component in the third category never appears in a call graph. If OpenFAWE cannot run without it,
it is not evidence — it is a dependency, and it moves to the second category.

---

## 2. Data flow

```
                    frozen inputs
        (published pumping-cycle controller,
         reference wing geometry, site metocean)
                          │
                          ▼
  ┌───────────────────────────────────────────────┐
  │            NEW OPENFAWE CODE                  │
  │                                               │
  │   ┌─────────────────────────────────────┐     │
  │   │ AWE lifting surface / free-vortex   │     │
  │   │ wake  (+ vortex-step screening)     │     │
  │   └─────────────────────────────────────┘     │
  │   ┌─────────────────────────────────────┐     │      ┌───────────────┐
  │   │ Time-domain orchestrator            │◄────┼──────┤ HAMS /        │
  │   │ (predictor-corrector macro-step)    │     │      │ NEMOH-v3      │
  │   └─────────────────────────────────────┘     │      ├───────────────┤
  │   ┌─────────────────────────────────────┐     │      │ MoorDyn × 2   │
  │   │ Adapters and evidence checks        │◄────┼──────┤ TurbSim       │
  │   └─────────────────────────────────────┘     │      └───────────────┘
  └───────────────────────────────────────────────┘         reused solvers
                          │
                          ▼
              macro-step outputs, per case id
                          │
   ┌──────────────────────┴───────────────────────┐
   │              PHYSICS TO COST                 │
   │  DB1-DB4 load  →  line & anchor  →  mooring  │
   │  envelopes         sizing            CAPEX   │
   │                          │                   │
   │        bounded lifecycle scenarios           │
   │                          ▼                   │
   │        comparative LCoE + decision class     │
   └──────────────────────────────────────────────┘

   OpenFOAM ····► validation evidence only (no runtime link)
```

---

## 3. Module notes

### 3.1 Airborne aerodynamics

Bound circulation on the rigid lifting surface is shed into a regularised free-vortex wake. A
cheaper vortex-step model screens candidate cases before full free-vortex-wake evaluation. Direct
and tree-based Biot–Savart evaluation, wake-age coarsening and far-wake treatment are benchmarked
against each other and against published regression cases.

A flexible leading-edge inflatable (LEI) wing is represented through measured or prescribed
deformation envelopes. **Full membrane aeroelasticity is out of scope.**

### 3.2 Control

The pumping-cycle controller is taken from the published state of the art and **frozen** at project
start. It is an input, not a contribution. Command saturation and tether-force exceedance are
retained as diagnostics — if the frozen controller is being pushed outside its declared envelope by
station motion, that fact is reported rather than tuned away.

### 3.3 Hydrodynamics

Frequency-domain coefficients from HAMS are assembled into a Cummins time-domain model:
hydrostatics, radiation memory, a viscous correction, Newman slow drift and wave-drift damping.
NEMOH-v3 quadratic-transfer-function checks are run on selected survival cases.

### 3.4 Lines — one core, two configurations

A single mature lumped-mass line core is used twice, which is deliberate: it removes an entire class
of inconsistency between the two line systems.

| Configuration | Features used | New in OpenFAWE |
|---|---|---|
| **Airborne tether** | Variable unstretched length, elasticity, mass, gravity, drag, optional bridle | The pumping boundary condition |
| **Station keeping** | Submerged loading, seabed contact, fairlead and anchor outputs, shared topology | Shared-load routing between platforms; adapters and tests |

OpenFAWE contributes the boundary condition, the adapters, the shared-load routing and the tests —
**not** generic line mechanics.

### 3.5 Coupling

A predictor–corrector macro-step layer exchanges forces and motion between the airborne unit, the
lines and the station. Macro-step size is a tested convergence parameter, not a tuning knob.

### 3.6 Surrogate

A Gaussian-process surrogate with active learning is the primary model for scalar energy, motion and
load responses, because each coupled run is expensive and the training set is bounded. Latin-hypercube
seeds are followed by samples that maximise predictive variance or feasibility information gain.

Variable-specific hold-out errors and calibrated intervals define the applicability map. **If an
interval crosses a motion, tether or anchor constraint, the candidate is unresolved and returns to
the coupled solver.** An independent gradient-boosting screen provides a feature-importance
cross-check.

### 3.7 Physics to cost

One pathway, physics-driven end to end:

1. array-level shared-line and anchor load redistribution;
2. governing ULS and accidental envelopes → line diameter and grade, anchor capacity;
3. those components → manufacturing-stage mooring and anchor CAPEX;
4. plus **declared** low / central / high lifecycle scenarios (availability, line-replacement
   interval, installation, O&M);
5. → comparative LCoE = discounted lifecycle cost ÷ discounted net energy;
6. → sensitivity analysis separating computed from assumed drivers, and a stable / conditional /
   indeterminate decision label.

See [`evidence-policy.md`](evidence-policy.md) §6 for the labelling rules.

---

## 4. Traceability

One **versioned case identifier** follows every input, solver run, evidence check and cost result.
Given any released number, the identifier resolves to: the input set, the solver versions, the
environment lock, the convergence record, the evidence class, and the cost scenario in force.

---

## 5. Provenance and separation from MRUT

OpenFAWE's free-vortex-wake kernels are **reimplemented from published formulations and regression
cases**. OpenFAWE does not import, link against, call, or vendor
[MRUT](https://github.com/robytian/MRUT)
([doi:10.17632/5jrdc8jb65.1](https://doi.org/10.17632/5jrdc8jb65.1)), the maintainer's earlier
rotor-wake code, which remains a separate project with separate versioning and its own release
history.

This is checked at release time as part of the provenance review described in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

Restricted, partner-confidential or unpublished third-party inputs are kept outside this repository
and outside any external service.
