# Evidence policy

**Status:** specification, v0.1.0. This document states what OpenFAWE will and will not call
*validated*, and fixes the acceptance rules before any results exist.

---

## 1. Two kinds of evidence, never merged

| Term | Means | Reported as |
|---|---|---|
| **Verification** | Code-to-code comparison, method-of-manufactured-solutions checks, convergence and conservation tests | "verified against *X*" |
| **Component validation** | Comparison against experimental or field measurements, for one component or channel at a time | "component-validated against *Y*" |

A code-to-code agreement is **never** reported as validation. A channel with no experimental or field
comparator is **never** described as validated, however well it converges.

**Unsupported channels** are handled in one of three declared ways, decided and recorded at the
integrated re-baseline point rather than at write-up time:

1. **narrowed** — the applicability statement is restricted so the channel is only claimed where a
   comparator exists;
2. **assigned a fallback** — a more conservative, simpler model is used and labelled as such;
3. **excluded** — the channel is removed from the released results entirely.

---

## 2. Pre-registration

Case matrices, acceptance bands and exclusions are registered in the repository **before** production
screening runs. A case matrix committed after the corresponding results is not admissible as
pre-registered, and the released record says so.

---

## 3. The non-superposition rule (H1)

For each pre-registered observable:

```
R = | y_full − y_superposition | / U95
```

- `y_full` — the fully coupled response.
- `y_superposition` — the same quantity assembled from separately evaluated aerodynamic,
  hydrodynamic and mooring models, driven by **identical** long wind–wave records, with the
  environment kept complete (slow drift, direct wind, current, wave-drift damping, station keeping).
  A first-order-only comparator is not a valid superposition baseline.
- `U95` — the combined 95 % uncertainty from numerical sources and forcing variability.

**Support requires all three:**

1. `R > 1` for at least **two** observables,
2. drawn from **different subsystems** — including at minimum one station-motion observable and one
   line-load observable,
3. with the exceedance **recurring under two forcing seeds**.

If any of the three fails, non-superposition is **not** established, and the released report says
that rather than reporting a near-miss as a trend.

---

## 4. Acceptance rules by module

| Module | Observables | Acceptance / decision rule |
|---|---|---|
| **Aerodynamics and fixed controller** | Energy, forces, moments, wake geometry, line tension, command saturation, LEI deformation | Rigid wing: convergence and agreement within declared uncertainty at the first gate. LEI minimum output: evidence map and interface specification, delivered unconditionally. A component-transfer case is added afterwards **only if** the evidence is sufficient. |
| **Hydrodynamics and lines** | Slow-mode periods, drift offset, 6DoF motion, phase, tether / fairlead / anchor loads | Code verification **plus** component agreement within combined source and numerical uncertainty at the first gate. The H1 rule in §3 applies. Unsupported channels remain outside scope. |
| **Shared array** | Tether, fairlead and anchor loads; energy and motion departure from isolated units | Either an effect above the combined uncertainty, **or** a defensible negligible-interaction boundary, mapped across the feasible-layout space. Four-unit scaling only after the two-unit gate passes. |
| **Surrogate and cost** | Energy, line diameter and grade, anchor capacity, component CAPEX, feasibility, LCoE, decision class | Energy error **≤ 5 %** and load error **≤ 10 %** on held-out solver cases. Any calibrated interval that crosses a motion, tether or anchor constraint returns the candidate to the coupled solver. Computed and assumed cost drivers reported **separately**. |

**Asymmetry, stated deliberately.** A small energy error never overrides a missed safety constraint.
A candidate whose interval crosses a constraint is *unresolved*, not *feasible*.

---

## 5. Uncertainty budget

`U95` is assembled from, at minimum:

- discretisation and macro-step contributions, established by the convergence tests in
  [`design-basis.md`](design-basis.md) §3;
- solver-residual and energy-balance contributions;
- forcing variability across seeds;
- declared source uncertainty for any coefficient taken from a published dataset.

Each contribution is reported separately, not only as a total, so a reader can see which term
dominates.

---

## 6. Cost conclusions

The cost chain is split into two clearly separated halves, and every released cost statement declares
which half it came from:

- **Computed** — energy from solver runs; line and anchor sizing from design-basis load envelopes;
  the resulting manufacturing-stage CAPEX.
- **Assumed** — technical availability, line-replacement interval, installation concept, operation and
  maintenance, project life, discount rate. These remain **editable low / central / high scenarios**
  until field evidence exists.

Every design choice reported from the cost model carries one of three labels:

| Label | Meaning |
|---|---|
| **stable** | The choice does not change across the declared scenario range |
| **conditional** | The choice changes, and the report names the assumption that flips it |
| **indeterminate** | The scenario range is too wide to separate the options |

An LCoE number published without its scenario label is not a released result.

---

## 7. Reproducibility

- One versioned case identifier follows every input, solver run, evidence check and cost result.
- Each tagged release carries locked environment specifications and container recipes
  (Docker / Apptainer) so that archived results remain executable.
- Benchmark data are released under CC BY 4.0 with the metadata needed to re-run them.
- A result that cannot be regenerated from the archived case identifier is withdrawn, not footnoted.
