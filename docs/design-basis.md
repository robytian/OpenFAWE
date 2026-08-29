# Design basis — DB1 to DB4

**Status:** specification, v0.1.0. This document fixes the load-case matrix before any results exist,
so that the cases cannot be selected after seeing the answers.

> **Not a certification claim.** The DB1–DB4 matrix *adapts* the logic of IEC 61400-3-2:2025
> (floating offshore wind turbines) and DNV-ST-0119 (floating wind turbines) to a system class those
> standards do not cover. It is a traceable **research** design basis for comparative screening. It
> confers no compliance, no class approval and no certification.

---

## 1. The matrix

| Case | Condition | What is simulated | What is prescribed | Limit state |
|---|---|---|---|---|
| **DB1** | Normal pumping production | Full coupled aero–hydro–line response over complete pumping cycles | — | ULS |
| **DB2** | Low wind, launch, landing and controlled return | Station and line response to a prescribed airborne-load envelope | Airborne load envelope. **Transition control is not simulated.** | ULS |
| **DB3a** | Recovered survival | Station and line response with **no crosswind traction** | Airborne unit recovered | ULS |
| **DB3b** | Failed recovery | Station and line response under a conservative no-generation airborne-load envelope | Airborne load envelope. **Recovery control is not simulated.** | ULS |
| **DB4** | Airborne-line, mooring-line or anchor fault | Load redistribution across the remaining lines and shared nodes; safe-state logic | Fault instant and affected member | Accidental |

**Sizing rule.** Governing DB1–DB3b ultimate-limit-state envelopes and DB4 accidental envelopes set
line diameter and grade, and anchor capacity. These components then enter the cost chain
(see [`architecture.md`](architecture.md)).

---

## 2. Environmental content

Every case retains the following, because omitting any of them would bias the comparison that DB1–DB4
exist to support:

- irregular waves, with **difference-frequency slow drift** (Newman approximation; quadratic-transfer-function
  checks on selected DB3 cases);
- **wave-drift damping**;
- **direct wind loading on the station**, separately from the airborne load path;
- **current**, treated explicitly rather than folded into an equivalent offset;
- turbulent inflow for the airborne unit;
- station keeping with seabed contact and, in array cases, shared topology.

DB3 demand explicitly retains slow drift, direct wind, current, wave-drift damping and station
keeping. A DB3 case evaluated as a first-order-only wave problem is **not** a DB3 case.

---

## 3. Record length, seeds and convergence

| Item | Rule |
|---|---|
| Transient | Discarded before any statistic is formed |
| Record length | At least **20 periods of the slowest resolved mode** after transients |
| Seeds | **Six** forcing seeds by default; expanded only when confidence intervals remain unstable |
| Convergence tested | Macro-step, wake discretisation, solver residual, and energy balance |

The two-seed recurrence requirement in the H1 acceptance rule
(see [`evidence-policy.md`](evidence-policy.md)) is a *minimum*, not the seed budget.

---

## 4. Fatigue

Fatigue-life modelling is **outside scope**. Cycle-count-ready channels are archived with each case so
that a fatigue assessment can be performed later by others, but no S–N curve, damage accumulation
rule or design fatigue factor is applied, and no fatigue conclusion is published.

---

## 5. Reference configuration

| Parameter | Value | Source |
|---|---|---|
| Airborne unit | Rigid fixed-wing pumping system, 150.3 m² wing, up to 3.8 MW | published multi-megawatt reference design |
| Tether length range | 526–1434 m | published reference design |
| Generator / winch location | On the floating station | architecture choice |
| Propellers | Launch and landing only — **not** onboard generation | architecture choice |
| Station | VolturnUS-S-derived semi-submersible | published reference platform, site-adapted |
| Water depth | 243.3 m | site adaptation |
| Draft | 20 m | reference platform |
| Station-keeping lines | 3 × 851.55 m | site adaptation |
| Anchor radius | 877.3 m | site adaptation |

**Spectral motivation.** The pumping cycle sits at roughly **133–333 s**; the station's horizontal
slow modes sit at roughly **65–210 s**. The bands overlap. That overlap is the physical reason
component superposition cannot be assumed, and it is what the DB1 cases are built to probe.

> The station geometry is **adapted** from the standard VolturnUS-S configuration (200 m water depth)
> to the study site. Depth, line length and anchor radius therefore differ from the published
> reference values and are stated here explicitly so the difference is not mistaken for an error.

---

## 6. Architecture comparison

The primary demonstrator is the rigid fixed-wing pumping system above. A flexible leading-edge
inflatable (LEI) architecture receives an **unconditional** evidence map and interface specification;
a component-transfer case on the same station class is **conditional** on the first evidence gate.

When architectures are compared, the following are held **common**: site, wind and wave seeds, rated
power, cost year and reporting definitions. The following remain **explicit and different**: geometry,
control bounds and line properties.

Any resulting comparison is **site- and evidence-bounded**. It is not a universal ranking of airborne
wind energy architectures, and must not be quoted as one.
