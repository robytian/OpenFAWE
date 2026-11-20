# Third-party notices

The OpenFAWE source code and research-preview programs contain or are derived
from the following third-party works. Each remains under its own licence. Full
licence texts and the file-level attributions of the source code are in
[`third_party/`](third_party), in the `licenses` folder of the program (copied
from each bundled package when the program is built), or at the links below.

## Methods and data

| Component | Use in OpenFAWE | Licence |
|---|---|---|
| Vortex-Step-Method, © 2022–2024 Oriol Cayon, Jelle Poland, TU Delft ([repository](https://github.com/awegroup/Vortex-Step-Method)) | Vortex step method algorithm, ported to OpenFAWE with attribution | MIT |
| OpenFAST / OLAF, National Renewable Energy Laboratory and contributors ([repository](https://github.com/OpenFAST/openfast)) | Free-vortex-wake algorithm, ported to OpenFAWE with attribution; modified | Apache-2.0 |
| TU Delft V3 kite data, © 2024 Airborne Wind Energy Research Group ([repository](https://github.com/awegroup/TUDELFT_V3_KITE)) | V3 geometry and reference data | MIT (`third_party/LICENSE-TUDELFT-V3.txt`) |
| Makani, Makani Technologies LLC ([repository](https://github.com/google/makani)) | Public M600 data used to reconstruct the M600 wing and tail, rotor positions and Rev4 tables, aerodynamic database, crosswind playbook, rigid body and GS02 ground station; lumped-mass tether, GS02 ground station and winch, motors and power system, rotor-table conventions and nominal crosswind loop transcribed from the simulator and flight controller; the display-only fuselage mesh derived from the M600 display model; modified | Apache-2.0 (`third_party/LICENSE-Makani.txt`, `third_party/MAKANI-NOTICE.md`) |
| Makani flight controller (`third_party/makani-controller-win-nogsl`) | Makani's C controller sources compiled unchanged into a library that OpenFAWE calls; linked against OpenFAWE's own small linear-algebra subset in place of the GNU Scientific Library (no GSL code is in this program) | Apache-2.0 (Makani); the subset is OpenFAWE's own |

## Software bundled in the Windows program

| Component | Licence |
|---|---|
| Python | Python Software Foundation License |
| NumPy, SciPy | BSD-3-Clause |
| Numba and llvmlite | BSD-2-Clause (llvmlite includes LLVM: Apache-2.0 with LLVM exceptions) |
| h5py and the HDF5 library | BSD-3-Clause / HDF5 licence (BSD-style) |
| Qt for Python (PySide6, shiboken6) and Qt 6 libraries | GNU LGPL v3 (`licenses/LGPL-3.0.txt`, `licenses/GPL-3.0.txt`) |
| VTK (The Visualization Toolkit), Kitware Inc. | BSD-3-Clause |
| PyVista and pyvistaqt | MIT |
| pyqtgraph, QtPy | MIT |
| matplotlib (colour maps) | matplotlib licence (PSF-based) |
| Pillow | MIT-CMU (HPND) |
| psutil | BSD-3-Clause |
| numba-cuda, used only when an NVIDIA GPU is present | BSD-2-Clause |
| NVIDIA CUDA 12 runtime components: CUDA runtime, NVRTC, nvJitLink, NVVM and libdevice (from NVIDIA's `nvidia-cuda-*-cu12` and `nvidia-nvjitlink-cu12` packages; their developer tools, static and import libraries and the CCCL headers are left out) | NVIDIA CUDA Toolkit End User License Agreement, as shipped in the packages (`licenses/nvidia-*`) |
| cuda-bindings, cuda-core, cuda-pathfinder (NVIDIA's Python access to CUDA) | Apache-2.0 |
| certifi (CA certificates) | MPL-2.0; source at <https://github.com/certifi/python-certifi> |
| MinGW-w64 runtime and libgcc, statically linked into the controller library | MinGW-w64 runtime licence (permissive); libgcc: GPL v3 with the GCC Runtime Library Exception (no obligations for the program) |
| PyInstaller bootloader | GPL v2 with the bootloader exception (no obligations for the bundled program) |
| Every other bundled Python package (for example contourpy, cycler, fonttools, kiwisolver, Pygments, requests, rich, urllib3) | as declared by each package: its licence text is in `licenses/<package>/`, the declared licence in `licenses/INDEX.json` |

Qt and Qt for Python are used under the GNU Lesser General Public License
version 3. The Windows program is distributed as a folder in which the Qt
libraries and Qt for Python's own modules (the `PySide6` and `shiboken6`
folders) are separate, replaceable files; the program's licence does not
restrict your rights under the LGPL (see LICENSE-BINARIES.md). The
corresponding source code is available from the Qt Project at
<https://download.qt.io/> and <https://code.qt.io/>.

The file-level notices in `third_party/` describe the OpenFAWE source code as a
whole; the Windows program contains only the modules the program reaches
(`bundle-manifest.json` in the program's data folder lists them), so some files
named there, such as the mooring and floating-platform code, are not in it.
