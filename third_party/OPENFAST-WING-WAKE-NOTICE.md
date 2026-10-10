# Source attribution for wing and wake increments

Source: OpenFAST v5.0.0, commit `2895884d2be01862173c88d70f86b358d2f1a50a`.
The original source/license identities are in docs/tasks/OF-0006/upstream-cases.json.
License: Apache-2.0, reproduced in LICENSE-OpenFAST.txt. This does not license
unrelated project materials or imply OpenFAST endorsement.

Source-derived files:

- src/openfawe/aero/wing.py: FVW_Wings.f90 Meshing, Wings_Panelling_Init,
  Wings_Panelling, Wings_ComputeCirculationPolarData and CirculationFromPolarData;
  FVW_VortexTools.f90 interpextrap_cp2node.
- src/openfawe/aero/wake.py: FVW.f90 FVW_UpdateStates/FVW_Euler1/FVW_RK4/wake RHS and
  FVW_Subs.f90 Map_LL_NW/PropagateWake state ordering and circulation ownership.
- tests/fixtures/olaf_ellipse.json: exact blade and polar numeric inputs plus
  last-step values extracted from the paired r-test historical output. Input
  files and the r-test source commit have their own hashes in that fixture.

Changes in 2026: new Python APIs and immutable states, explicit input errors and
nonconvergence failures, direct influence operators, separate constant-radius
short-wake representation, aggregated directed edges and explicit self exclusions.
Single-table polar interpolation is linear rather than the default source cubic
interpolation. Full source interpolation/UA/diffusion/far-wake/tree paths are not
ported. Quasi-steady force/moment reconstruction is explicitly documented.
The finite-wake steady comparison is not a reproduction of the complete source
driver transient. The original reference repositories are unchanged.

OF-0006-08 added selectable frozen-circulation RK4 convection and geometry-only
substeps with fixed bound nodes. OF-0006-fix corrects the moving-aircraft stages
using linear interpolation of bound endpoint coordinates (not exact rigid
rotation); the old trailing edge remains fluid-convected. This motion repair is
an OpenFAWE adaptation, not a claim of identical source transient integration.
A new trailing-edge row and new circulation appear only once per
physical step. Material cores retain the existing split update; this is not a
claim of fourth-order accuracy for the full coupled solver or a full OLAF port.
The additional standard Heun second-order predictor-corrector is an OpenFAWE
option requested by the owner, not attributed to the source FVW_RK4 routine.
