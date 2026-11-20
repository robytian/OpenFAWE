# Source code

Python 3.12 or newer.

```bash
git clone https://github.com/robytian/OpenFAWE.git && cd OpenFAWE
python -m venv .venv
.venv/bin/pip install -e ".[fast,paper,test]"          # Windows: .venv\Scripts\pip
.venv/bin/python -m unittest discover -s tests -t .      # 831 tests
```

Two of the 831 tests fail, and are published as they are: the variable-length tether at full length against the
fixed-length one (`tests/test_variable_line.py`: 7.8e-11 against the bound of 1e-12 written first) and the kite at rest
on the perch (`tests/test_perch.py`: 40 mm and 0.8° within 10 s against 5 mm and 0.1°). Both come from methods taken
over from Makani as they are; [Scope and limitations](guide/limitations.md) gives the causes.

## Repository layout

| Folder | What is in it |
|---|---|
| [`src/openfawe`](https://github.com/robytian/OpenFAWE/tree/main/src/openfawe) | The package: aerodynamics (`aero`), flight and control interface (`flight`), ground station and winds (`ground`), tether (`tether`), rotors (`rotor`), power system (`power`) |
| [`apps/studio`](https://github.com/robytian/OpenFAWE/tree/main/apps/studio) | OpenFAWE Studio, its packaging scripts and tests |
| [`cases`](https://github.com/robytian/OpenFAWE/tree/main/cases) | Aircraft, polars, rotors, power train, ground station and tether data |
| [`tests`](https://github.com/robytian/OpenFAWE/tree/main/tests) | Unit tests |
| [`third_party`](https://github.com/robytian/OpenFAWE/tree/main/third_party) | Licences and notices; Makani's controller built without the GSL |
| [`paper1`](https://github.com/robytian/OpenFAWE/tree/main/paper1) | Code and data of the arXiv paper |

## Reproduce the paper

Everything behind the figures and tables of
[arXiv:2610.37313](https://doi.org/10.48550/arXiv.2610.37313) is in
[`paper1/`](https://github.com/robytian/OpenFAWE/tree/main/paper1), with the command line of every run in
[`RUNS.md`](https://github.com/robytian/OpenFAWE/blob/main/paper1/crosswind/RUNS.md) and the versions pinned in
[`requirements-paper1.txt`](https://github.com/robytian/OpenFAWE/blob/main/requirements-paper1.txt).

The source is published as it is, with the internal labels of its development records still in the comments.
