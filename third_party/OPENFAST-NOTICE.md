# OpenFAST-derived vortex components

`src/openfawe/aero/vortex.py` and the K03/K04 portions of `tests/test_vortex.py`
adapt selected mathematical routines and test case structure from OpenFAST/OLAF:

- Upstream: https://github.com/OpenFAST/openfast
- Version: v5.0.0, commit `2895884d2be01862173c88d70f86b358d2f1a50a`
- `modules/aerodyn/src/FVW_BiotSavart.f90`: `ui_seg_11`, `ui_seg`, `ui_quad_n1`.
- `modules/aerodyn/src/FVW_Tests.f90`: `Test_BiotSavart_Sgmt`.
- Source license: Apache License 2.0, reproduced in `LICENSE-OpenFAST.txt`.

Original selected files have no additional per-file copyright/NOTICE text.
This note attributes the source project and does not assert ownership of its work.
Source files, license and checksums are recorded in
`docs/tasks/OF-0006/upstream-cases.json`. No original source file was modified.

OpenFAWE changes (2026): Python/NumPy API and array layouts; explicit errors and
self exclusions; local nondimensional geometry; stable denominator identity;
component-ordered projections; owned output and bounded point chunks; ring edge
summation in A-B-C-D-A order (upstream accumulates 1-2, 3-4, 2-3, 4-1).
These adapted files are provided under Apache-2.0. This statement does not set a
license for unrelated project material. They are not endorsed by OpenFAST.

The welib comparison reuses previously recorded numerical results only; it does
not vendor welib code. Its common lineage and known test-fixture error remain
documented in the source contract. No bitwise Fortran-equivalence claim is made.
