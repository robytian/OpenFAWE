<p align="center">
  <img src="docs/figures/openfawe-hero.png" alt="Free vortex wakes of the TU Delft V3 kite and the Makani M600 computed with OpenFAWE" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20preview-0f2738" alt="status: research preview">
  <img src="https://img.shields.io/badge/platform-Windows%2064--bit-1f7ea2" alt="platform: Windows 64-bit">
  <img src="https://img.shields.io/badge/source%20licence-Apache--2.0-4a5560" alt="source licence: Apache-2.0">
</p>

# OpenFAWE

Aerodynamic simulation of airborne wind energy kites — research preview.

> **Status.** OpenFAWE is independent research carried out and funded by the author.
> The source code of the aerodynamic module is released here under the Apache License 2.0
> ([`src/`](src), [`tests/`](tests)). Kite models, examples and the other modules will be added
> progressively. The repository also provides a method comparison report and a Windows
> research-preview program.

## Results at a glance

<p align="center">
  <img src="docs/figures/Fig1_V3_methods_vs_public_600dpi.png" alt="TU Delft V3 kite: lift, drag and pitching moment of eight OpenFAWE method combinations compared with wind-tunnel, RANS and open-source VSM results" width="100%">
</p>

TU Delft V3 kite: eight OpenFAWE method combinations compared with wind-tunnel measurements
(Poland et al., 2026), RANS results (Viré et al., 2020) and the open-source vortex-step-method
code (Cayon et al., 2023).

- The open-source VSM code, run with the same input, reproduces OpenFAWE VSM: lift within 0.3%,
  drag within 0.9%.
- At 5.4° and 7.4° the lift of the polar-coupled methods is within −12% to +5% of the wind tunnel.
  The low-angle lift and the pitching moment differ from the measurements for all codes,
  including RANS and the open-source VSM; this is still being studied.
- For the Makani M600, the straight-wake methods are within −1.6% to +0.8% of the open-source
  VSM lift ([Figure 2](docs/figures/Fig2_M600_methods_vs_public_VSM_600dpi.png)). The effect of a
  fuselage is shown separately and is not a validation ([Figure 3](docs/figures/Fig3_M600_fuselage_effect_600dpi.png)).

Full details, limitations and references: **[method comparison report](docs/comparison-report.md)**.

## OpenFAWE Wake Viewer (Windows)

Choose a model (V3 kite or M600), a wind speed and a simulated time; the program computes the
free vortex wake and replays it. Download it from
[Releases](https://github.com/robytian/OpenFAWE/releases).

1. Download `OpenFAWE-WakeViewer-0.1.0-win64.zip` and check its SHA-256 against the value on the
   release page.
2. Unzip it and run `OpenFAWE-WakeViewer.exe`. The program is not code-signed, so Windows may show
   "Windows protected your PC"; choose "More info" and then "Run anyway".
3. Choose the model, wind speed and simulated time, then click **Run and view wake**.
   Runs are saved in `%USERPROFILE%\OpenFAWE-WakeViewer\runs`.

What this preview does not do: the angle of attack is fixed per model; only the lifting-line
free-wake method is offered; section polars are fixed (no Reynolds-number correction); at most
80 time steps are computed; there is no tether, control or power model. The results are
research output and are not validated for design use.

## Methods in brief

Eight aerodynamic method combinations: lifting line (LL), vortex step method (VSM;
Cayon et al., 2023), inviscid vortex-lattice lifting surface (LS) and polar-coupled LS, each
with a prescribed straight wake or a free vortex wake. The free-wake formulation is derived
from OpenFAST/OLAF (Branlard et al., 2022). The wake images above are archived 40-step
free-wake states from an impulsive start; they illustrate the method and are not a
convergence result.

## Source code: aerodynamic module

[`src/openfawe/aero`](src/openfawe/aero) contains the lifting line, the vortex step method and the
vortex-lattice lifting surface, each with a prescribed straight wake or a time-marching free vortex
wake, the Lamb–Oseen regularised Biot–Savart kernel with a NumPy reference, a compiled (Numba) direct
sum and a Barnes–Hut tree code, and the Newton solution of the circulation closure. Kite models and
section data are not included yet; tests that need them are deselected in
[`tests/conftest.py`](tests/conftest.py).

```
python -m venv .venv
.venv/bin/pip install -e ".[fast,test]"      # Windows: .venv\Scripts\pip
python -m pytest tests
```

Python 3.12 or newer.

## Licence

Source code (`src/`, `tests/`): Copyright (c) 2026 Yinong Tian, Apache License 2.0; see
[LICENSE](LICENSE) and [NOTICE](NOTICE). The Windows research-preview program, the text, figures
and data summaries: see [LICENSE-BINARIES.md](LICENSE-BINARIES.md). Third-party components remain
under their own licences; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[`third_party/`](third_party).

Questions and bug reports: please open an issue.

## 中文说明

OpenFAWE 是作者自费开展的独立研究，用于空中风能风筝的气动仿真。气动模块的源代码已按 Apache-2.0 许可公开（src、tests 文件夹），风筝模型、算例和其他模块将逐步补充。本仓库另提供方法对比报告和一个 Windows 研究预览程序（可选择模型、风速和仿真时间，计算并回放自由尾迹），程序请在 Releases 下载。计算结果为研究结果，不能用于设计。
