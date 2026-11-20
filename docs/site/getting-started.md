# Getting started

## Download

Download `OpenFAWE-Studio-1.0.0-win64.zip` from the
[latest release](https://github.com/robytian/OpenFAWE/releases/latest) (749 828 041 bytes) and check it in a command
prompt:

```bat
certutil -hashfile OpenFAWE-Studio-1.0.0-win64.zip SHA256
```

The result must be `9147bf97ae1038df23e23159c903c3365107e4b2a89f45f127890991b3b68b54`.

**Needs:** Windows 10 or 11, 64-bit; 8 GB of memory (16 GB for long vortex flights); about 3 GB of disk for the
program and the examples. An NVIDIA GPU with a current driver is optional; the program finds out what the computer has
and uses the CPU tree code otherwise.

## Run

1. Unzip, then start `OpenFAWE Studio.exe`. The program is not code-signed: if Windows shows "Windows protected your
   PC", choose **More info** and then **Run anyway**.
2. **Open run** opens the examples folder; choose `e4-tables-10ms\run` and **Play replay**.
3. Or press **Run** to fly yourself (see [Running a simulation](guide/running.md)).

To check an installation (about a minute):

```bat
start /wait "" "OpenFAWE Studio.exe" --self-test
```

## Example flights

| Example | Aerodynamics | Wind | Shows |
|---|---|---|---|
| `e4-tables-10ms` | Makani's aerodynamic database | uniform 10 m/s | the whole flight with the power system, mode by mode |
| `bprime6-vortex-10ms` | free vortex wake in the flying modes, tables in hover | uniform 10 m/s | the wake of a whole flight |
| `vortex-8ms` | as above | uniform 8 m/s | crosswind loops that still use more energy than they make |
| `vortex-12ms` | as above | uniform 12 m/s | the flight of the picture on the front page |
| `vortex-shear-10ms` | as above | wind shear 1/7, 10 m/s at 21 m | more power from the faster wind at the kite's height |
| `f3-vortex-turbulent` | as above | IEC 61400-1 normal turbulence, mean 10 m/s | the whole flight in turbulent wind |
| `cycle-power-curve` | tables; vortex | 6–14 m/s | break-even crosswind time of a whole flight (CSV) |

The [example flights chapter](guide/examples.md) explains each one.

**Data use.** The example flights distributed with OpenFAWE Studio 1.0.0 and their results – the program's `examples` folder, the power-curve table, the whole-flight video and the pictures made from them – belong to a manuscript in preparation by Y. Tian and S. Wang. Until that paper is published they may be used to check an installation, to learn the program and to compare with your own simulations, citing this repository (see `CITATION.cff`). They may not be republished, or used as the data of a publication or a thesis, without the author's written permission. Contact: yinong.tian@tecnico.ulisboa.pt. The full terms are in [DATA-USE.md](https://github.com/robytian/OpenFAWE/blob/main/DATA-USE.md).
