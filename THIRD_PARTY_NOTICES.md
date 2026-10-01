# Third-party notices

The OpenFAWE research-preview programs contain or are derived from the
following third-party works. Each remains under its own licence. Full licence
texts are in the `licenses` folder of the release, or at the links below.

## Methods and data

| Component | Use in OpenFAWE | Licence |
|---|---|---|
| Vortex-Step-Method, © 2022–2024 Oriol Cayon, Jelle Poland, TU Delft ([repository](https://github.com/awegroup/Vortex-Step-Method)) | Vortex step method algorithm, ported to OpenFAWE with attribution | MIT |
| OpenFAST / OLAF, National Renewable Energy Laboratory and contributors ([repository](https://github.com/OpenFAST/openfast)) | Free-vortex-wake algorithm, ported to OpenFAWE with attribution; modified | Apache-2.0 |
| TU Delft V3 kite data, © 2024 Airborne Wind Energy Research Group ([repository](https://github.com/awegroup/TUDELFT_V3_KITE)) | V3 geometry and reference data | MIT |
| Makani, Makani Technologies LLC ([repository](https://github.com/google/makani)) | Public M600 data used to reconstruct the M600 wing and tail; modified | Apache-2.0 |

## Software bundled in the Windows program

| Component | Licence |
|---|---|
| Python | Python Software Foundation License |
| NumPy | BSD-3-Clause |
| h5py and the HDF5 library | BSD-3-Clause / HDF5 licence (BSD-style) |
| Qt for Python (PySide6, shiboken6) and Qt 6 libraries | GNU LGPL v3 |
| pyqtgraph | MIT |
| PyInstaller bootloader | GPL v2 with the bootloader exception (no obligations for the bundled program) |

Qt and Qt for Python are used under the GNU Lesser General Public License
version 3. The Windows release is distributed as a folder in which the Qt
libraries are separate, replaceable files. The corresponding source code is
available from the Qt Project at <https://download.qt.io/> and
<https://code.qt.io/>.
