# OpenFAWE

**Simulation of a land-based airborne wind energy system: free-vortex-wake aerodynamics and whole flights of the
Makani M600 energy kite with its ground station, winch, tether, rotors and power system.**

```{image} _generated/media/openfawe-land-system.png
:alt: The land-based airborne wind energy system in OpenFAWE - the Makani M600 on its crosswind loop with the free vortex wake of its wing and rotors, the tether and the GS02 ground station
:width: 100%
```

```{rst-class} hero-caption
```
The land-based system as OpenFAWE flies it, to scale: the M600 on its crosswind loop at 12 m/s with the free vortex
wake of its wing (blue) and rotors (orange), the tether and the GS02 ground station (example flight `vortex-12ms`; the
ground station is drawn after GS02's dimensions).

OpenFAWE Studio 1.0 is a Windows program that flies the Makani M600 energy kite with Makani's own flight controller,
from the perch to crosswind power generation and back to the perch, and shows it in 3-D while it computes. The source
it is built from is published under Apache-2.0.

| | |
|---|---|
| **Whole flight** | Perch, hover, payout, transition, crosswind loops, transition out, reel-in and landing on the perch, flown by Makani's M600 flight controller (compiled from Makani's source) |
| **Ground system** | GS02 ground station with its winch, drum and levelwind; variable-length lumped-mass tether with bridle |
| **Power** | Eight rotors on Makani's Rev4 tables, motors and the stacked power system; the ground energy of every flight mode |
| **Aerodynamics** | Makani's aerodynamic database, or the vortex step method with a free vortex wake and rotor rings in the flying modes |
| **Short windows** | Six vortex models (VSM, lifting line, vortex lattice; free or straight wake), fixed in the wind, on a circular crosswind loop or on Makani's nominal loop, with gusts and wind shear |
| **Hardware** | Tree code on all CPU cores, or an NVIDIA GPU for the wake sums; chosen automatically |
| **Output** | Replay with a time line and live plots of any recorded quantity; ParaView (VTP/PVD), CSV, screenshots, GIF and MP4 videos |

## Where to start

- **Use the program**: [Getting started](getting-started.md) - download, check, open an example flight.
- **Learn every panel and file**: the [user guide](guide/index.md) ([中文](zh/index.md)).
- **See how it was checked**: [Validation](guide/validation.md) - each part against its reference, then the whole flight.
- **Work with the source**: [Source code](source-code.md) - install, run the tests, reproduce the paper.
- **Cite**: [Citing OpenFAWE](citing.md).

```{toctree}
:caption: Getting started
:maxdepth: 1
:hidden:

getting-started
```

```{toctree}
:caption: User guide
:maxdepth: 2
:hidden:

guide/index
guide/what-it-computes
guide/installation
guide/window
guide/examples
guide/replay
guide/export
guide/running
guide/output-files
guide/flight-modes
guide/validation
guide/limitations
guide/faq
guide/references
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

reference/record-keys
source-code
```

```{toctree}
:caption: Project
:maxdepth: 1
:hidden:

changelog
citing
licence
```

```{toctree}
:caption: 中文
:maxdepth: 2
:hidden:

zh/index
zh/what-it-computes
zh/installation
zh/window
zh/examples
zh/replay
zh/export
zh/running
zh/output-files
zh/flight-modes
zh/validation
zh/limitations
zh/faq
zh/references
zh/record-keys
```
