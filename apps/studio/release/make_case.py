"""OF-0013 revision 1 section 5 and addendum 5: one example of the case package from a studio run folder.

  python apps/studio/release/make_case.py RUN OUT --name e4-tables-10ms --title "..." [--commit SHA] [--machine TEXT]
         [--flight-every 10] [--hover-every 100] [--note TEXT]

OUT/ gets:
- ``run/``: a studio run folder that the studio replays (``RunFolder``): ``spec.json`` without local paths or git status
  (the library's file name and SHA-256 and the commit stay), ``status.json``, ``settings.json``, every stored frame, and
  the records thinned: the flying modes (names starting ``Trans`` or ``Crosswind``) every ``flight-every`` periods, the
  other modes (hover, perch) every ``hover-every``, plus every period where the mode changes, every period with a stored
  frame and the last period. Records are kept whole (every field can be plotted).
- ``case.json``: what the case is (name, title, aircraft, aerodynamics, wind, controller library, power system), the
  periods, the duration, the net ground energy, the end, the thinning, the source run's name, commit and machine.
- ``energy-by-mode.csv``: the status's mode-by-mode energy account (when the run has the power system).
- ``trajectory.csv``: from the thinned records: period, time, mode, the position in the TurnKey world frame (x, y, z up,
  m), the rotation matrix as recorded (row by row), airspeed, angle of attack and sideslip, bridle tension, free length.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

STUDIO = Path(__file__).resolve().parents[1]
REPO = STUDIO.parents[1]
for p in (REPO/"src", STUDIO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FLYING = ("Trans", "Crosswind")
LOCAL_KEYS = ("makani_build",)


def mode_of(record, names):
    number = record.get("flight_mode", record.get("controller_flight_mode"))
    return names.get(str(number), str(number))


def keep_periods(records, names, frames, flight_every, hover_every):
    """The periods to keep (see the module text)."""
    keep, previous = set(), None
    for record in records:
        k = int(record["k"])
        mode = mode_of(record, names)
        every = flight_every if mode.startswith(FLYING) else hover_every
        if k % every == 0 or mode != previous or k in frames:
            keep.add(k)
        previous = mode
    if records:
        keep.add(int(records[-1]["k"]))
    return keep


def sanitised_spec(meta):
    meta = json.loads(json.dumps(meta))
    if meta.get("makani_build"):
        meta["makani_build"] = Path(str(meta["makani_build"]).replace("\\", "/")).name
    spec = meta.get("spec")
    if isinstance(spec, dict) and spec.get("makani_build"):
        # OF-0013 addendum 14: the run's own spec carries the controller folder as given on the machine that ran it;
        # empty is the bundled library (``StudioSpec.makani_build``), the folder's name stays at the top and in case.json
        spec["makani_build"] = ""
    library = meta.get("library")
    if isinstance(library, dict):
        git = (library.get("source") or {}).get("git") or {}
        environment = library.get("environment") or {}
        meta["library"] = dict(git_head=git.get("head"), python=environment.get("python"),
                               packages=environment.get("packages"))
    source = meta.get("source")
    if isinstance(source, dict) and source.get("folder"):
        source["folder"] = Path(str(source["folder"]).replace("\\", "/")).name
    return meta


def make_case(run, out, *, name, title, commit=None, machine=None, flight_every=10, hover_every=100, note=None):
    from openfawe.flight.makani_interface import G_TO_WORLD
    run, out = Path(run), Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    meta = json.loads((run/"spec.json").read_text(encoding="utf-8"))
    status = json.loads((run/"status.json").read_text(encoding="utf-8"))
    names = {str(k): v for k, v in (meta.get("mode_names") or {}).items()}
    frames = sorted(int(p.stem.split("_")[1]) for p in (run/"frames").glob("frame_*.npz"))
    records = [json.loads(line) for line in (run/"records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = keep_periods(records, names, set(frames), flight_every, hover_every)
    target = out/"run"
    (target/"frames").mkdir(parents=True)
    for k in frames:
        shutil.copy2(run/"frames"/f"frame_{k:06d}.npz", target/"frames"/f"frame_{k:06d}.npz")
    kept = [r for r in records if int(r["k"]) in keep]
    with (target/"records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in kept:
            stream.write(json.dumps(record) + "\n")
    (target/"spec.json").write_text(json.dumps(sanitised_spec(meta), indent=1, default=str) + "\n", encoding="utf-8")
    (target/"status.json").write_text(json.dumps(status, indent=1, default=str) + "\n", encoding="utf-8")
    if (run/"settings.json").exists():
        shutil.copy2(run/"settings.json", target/"settings.json")
    energy = status.get("energy") or {}
    by_mode = energy.get("by_mode") or {}
    if by_mode:
        fields = sorted({key for row in by_mode.values() for key, value in row.items() if not isinstance(value, dict)})
        with (out/"energy-by-mode.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["mode"] + fields)
            for mode, row in by_mode.items():
                writer.writerow([mode] + [row.get(f, "") for f in fields])
    with (out/"trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["period", "time_s", "mode", "x_m", "y_m", "z_m"] + [f"R{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
                        + ["airspeed_m_s", "alpha_rad", "beta_rad", "tension_bridle_N", "free_length_m"])
        for record in kept:
            position = G_TO_WORLD@np.asarray(record["Xg"], float)
            rotation = np.asarray(record["rotation"], float).reshape(3, 3)
            writer.writerow([int(record["k"]), repr(float(record["t"])), mode_of(record, names)]
                            + [repr(float(v)) for v in position] + [repr(float(v)) for v in rotation.ravel()]
                            + [repr(float(record[key])) if isinstance(record.get(key), (int, float)) else ""
                               for key in ("airspeed_m_s", "alpha_rad", "beta_rad", "tension_bridle_N", "free_length")])
    first = {}
    for record in records:
        first.setdefault(mode_of(record, names), int(record["k"]))
    library = meta.get("makani_library_sha256") or {}
    case = dict(
        name=name, title=title, aircraft="Makani M600 (OpenFAWE's reconstruction)",
        aerodynamics=meta.get("aerodynamics"), wind=meta.get("wind"),
        controller=dict(what="Makani's M600 flight controller, compiled from Makani's source", library=library.get("file"),
                        sha256=library.get("sha256"), folder=Path(str(meta.get("makani_build", ""))).name or None),
        power_system=bool(meta.get("power_system")), periods=int(status.get("periods", len(records))),
        period_s=0.01, duration_s=float(records[-1]["t"]) + 0.01 if records else 0.0, end=status.get("end"),
        first_period_by_mode=first, net_ground_energy_J=energy.get("net_ground_energy_J"),
        net_mean_ground_power_W=energy.get("net_mean_ground_power_W"),
        energy_signs=energy.get("signs"),
        thinning=dict(flying_modes_every_periods=flight_every, other_modes_every_periods=hover_every,
                      also_kept="every mode change, every period with a stored frame, the last period",
                      records_kept=len(kept), records_in_run=len(records), frames=len(frames)),
        source=dict(run=run.name, commit=commit, machine=machine, studio_version=meta.get("version")), note=note)
    (out/"case.json").write_text(json.dumps(case, indent=1, default=str) + "\n", encoding="utf-8")
    return case


def main(argv=None):
    ap = argparse.ArgumentParser(prog="make_case.py")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--flight-every", type=int, default=10)
    ap.add_argument("--hover-every", type=int, default=100)
    ap.add_argument("--note", default=None)
    a = ap.parse_args(argv)
    case = make_case(a.run, a.out, name=a.name, title=a.title, commit=a.commit, machine=a.machine,
                     flight_every=a.flight_every, hover_every=a.hover_every, note=a.note)
    print(json.dumps(dict(periods=case["periods"], kept=case["thinning"]["records_kept"], frames=case["thinning"]["frames"],
                          net_ground_energy_J=case["net_ground_energy_J"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
