"""Rerun actual V3/M600 production methods using unchanged retained drivers.

This is orchestration, not a new aerodynamic algorithm. Each worker invokes
the existing run_one function on freshly frozen specifications and new paths.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO / "src"))
from openfawe.aero_archive import aero_source_identity

DRIVERS = {"none": REPO / "paper1/scripts/methods/run_lattice_comparison.py",
           "grid_section": REPO / "paper1/scripts/methods/run_comparison.py"}
METHODS = ("ll_straight", "vsm", "free_wake", "vsm_free_wake", "vlm", "vlm_free_wake")
CASES = {"V3": "cases/aircraft/v3_cad_camber_integral.json",
         "M600": "cases/qualification/m600_makani_reconstructed.json"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
utc = lambda: datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def jobs():
    rows = []
    for model, angles, duration, dt in (("V3", (3.0810784665258444, 5.412846420261046, 7.350324457982788), 8., .2),
                                       ("M600", (-4., 0., 4.), .8, .02)):
        for alpha in angles:
            for coupling, methods in (("none", METHODS), ("grid_section", METHODS[-2:])):
                for method in methods:
                    index = len(rows)
                    rows.append(dict(model=model, variant="v3-integral" if model == "V3" else "makani-camber",
                        aircraft_file=CASES[model], method=method, alpha=alpha, beta=0., duration=duration,
                        dt=duration if method in ("ll_straight", "vsm", "vlm") else dt,
                        chordwise_panels=8, chordwise_spacing="cosine", normal_scheme="reconstructed_slope",
                        core="constant", integrator="euler", wake_length_m=1000., coupling=coupling,
                        initial_virtual_incidence_deg=2. if model == "V3" and method == "vlm_free_wake" and coupling == "grid_section" else 0.,
                        name=f"{index:03d}-{model.lower()}-{method}-{coupling}-a{alpha:.8f}"))
    assert len(rows) == 48
    return rows


def worker(specification):
    spec = read(specification)
    driver = DRIVERS[spec["job"]["coupling"]]
    assert sha(driver) == spec["driver_sha256"]
    assert sha(REPO / spec["job"]["aircraft_file"]) == spec["case_sha256"]
    module_spec = importlib.util.spec_from_file_location("retained_kite_driver", driver)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.ROOT == REPO
    # Existing archived driver records its own source; wrapper/spec hashes are separate.
    return module.run_one(spec)


def campaign(output, workers):
    output.mkdir(parents=True, exist_ok=False)
    specifications = output / "jobs"
    specifications.mkdir()
    source = aero_source_identity()
    plan = jobs()
    hashes = {str(p.relative_to(REPO)): sha(p) for p in [*DRIVERS.values(), *(REPO / n for n in CASES.values())]}
    save(output / "plan.json", dict(started_utc=utc(), source=source, producer_sha256=sha(__file__),
        python=sys.version, executable=sys.executable, jobs=plan, inputs_and_drivers_sha256=hashes,
        case_timeout_s=900, workers=workers, no_changed_solver_or_kite_geometry=True))

    def execute(job):
        path = specifications / (job["name"] + ".json")
        destination = output / job["name"]
        save(path, dict(job=job, output=str(destination), driver_sha256=sha(DRIVERS[job["coupling"]]),
            case_sha256=sha(REPO / job["aircraft_file"]), wrapper_sha256=sha(__file__)))
        code, error = None, None
        with path.with_suffix(".log").open("xb") as stream:
            try:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                    cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                code = process.returncode
            except subprocess.TimeoutExpired:
                error = "Worker900s timeout; partial archive retained, no automatic retry"
        result_path = destination / "result.json"
        result = read(result_path) if result_path.exists() else dict(job=job, status="process_failed", error=error)
        result.update(process_exit_code=code, worker_log_sha256=sha(path.with_suffix(".log")))
        return result

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, j) for j in plan]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            j = result["job"]
            # Versioned progress receipts avoid overwriting any completed receipt.
            save(output / f"progress-{len(results):02d}.json", dict(recorded_utc=utc(), results=results))
            print(json.dumps(dict(name=j["name"], status=result["status"], frames=result.get("frames"),
                final=result.get("last"), error=result.get("error"))), flush=True)
    current = aero_source_identity()
    unchanged = source["files_sha256"] == current["files_sha256"] and all(sha(REPO / p) == h for p, h in hashes.items())
    save(output / "summary.json", dict(finished_utc=utc(), runs=results, planned=48,
        source_before=source, source_after=current, inputs_and_drivers_sha256=hashes, source_unchanged=unchanged,
        no_new_physical_qualification=True, original_solver_and_case_inputs=True))
    assert unchanged and len(results) == 48


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        if args.workers not in (1, 2): raise ValueError("Use one or two bounded workers")
        campaign(args.output.resolve(), args.workers)
