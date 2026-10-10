# Explicit camber source attribution

OpenFAWE's Python mean-surface implementation is independently written. Its
workflow is informed by SHARPy `aero/models/aerogrid.py::generate_strip` at commit
2c9d92b1187e6dd46263cf51b4a973639d8f569e (BSD-3-Clause): interpolate a supplied
mean line, scale by chord and transform using a section frame. OpenFAWE does not
copy SHARPy's beam-frame or quarter-panel grid-shift convention. The source and
full license are retained under the OF-0006-models20260922 task evidence.

The V3 CAD profiles are from the retained Vortex-Step-Method repository, with
its MIT license and exact file hashes in that task's source-manifest.json. The
corresponding existing V3 case records the original section IDs and polar data.

The separate reconstructed M600 wing/tail input uses Google Makani commit
1d66923243f73a69cddda52082fd5415907afc99, its Apache-2.0 license, the
m600_low_tail_no_winglets.avl input, and five named AFILE profiles. Original
bytes, licenses and hashes are retained. Pylons, rotors, body, control deflection,
CDCL and compressibility are excluded from this named ideal comparison variant.
It uses physically rotated sections; AVL uses Ainc to modify tangency normals
without rotating its geometry (MIT AVL User Primer, SECTION entry):
https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf
This is not an AVL-compatible parser or a claim to reproduce its whole solver.

Thin-airfoil companion polars use analytical integrals of the processed mean
line with 2*pi lift slope and zero profile drag. They are labelled as ideal
derived polars, never as Makani/KiteFAST measured or numerical polar tables.
