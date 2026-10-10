# VSM source attribution

Source: awegroup/Vortex-Step-Method, revision
5485b823f33c76be70cf0569204c2e9c7be976c5. MIT license reproduced in
LICENSE-VSM.txt. Source/data hashes are in tests/fixtures/vsm_source.json.

src/openfawe/aero/vsm.py derives the five-filament topology, 3/4-chord nonlinear
closure, own 2-D induction subtraction and quarter-chord force-direction pass
from tests/thesis_functions_oriol_cayon.py, specifically create_geometry_general,
velocity_induced_bound_2D, solve_lifting_line_system_matrix_approach_semiinfinite
and output_results. Existing OpenFAST-derived finite-segment kernels are reused
under their own Apache-2.0 notice. See of-0006-aero-method-contract.md for mapping.

Changes: vectorized explicit immutable API, analytic unregularized semi-infinite
segments plus an explicit source_legacy core policy, atan2 with forward-flow guards, strict polar bounds and convergence
failure, covariant force/moment directions, configurable density/viscosity,
separate closure/load alpha diagnostics and method-aware provenance. Legacy core
projections derive from velocity_3D_bound_vortex, velocity_3D_trailing_vortex and
velocity_3D_trailing_vortex_semiinfinite. Artificial stall viscosity and KiteFAST
ad hoc drag offsets are excluded. Endpoint clamping is opt-in for source replay.

Fixture generation executes the original functions without source edits or
OpenFAWE imports. Nonlinear AR12 fixtures preserve the source test's Cl-only
table with zero profile Cd/Cm; they are not experimental drag validation.
