"""V1: tree accuracy on frozen wake geometries (OF-0006-accel20261001).

For each stored VSM free-wake state, compares the tree backend with the exact
Numba direct sum at (a) the wake convection targets (old TE + free nodes, with
the incident-segment exclusions used by wake._network_convection) and (b) the
wing closure/load points against the historical wake (rows from the first
released row onward, as in vsm_wake.history_vsm_induction). Errors are
normalised by the free-stream speed. Sweep: theta x order x leaf size.
"""
import csv
import itertools
import json
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import numpy as np  # noqa: E402
import accel_bench as ab  # noqa: E402
from openfawe.aero.vortex import TreeSettings, segments_velocity  # noqa: E402
from openfawe.aero.vortex_tree import TreeStatistics  # noqa: E402

THETAS = (0.8, 0.6, 0.5, 0.4, 0.3, 0.2)
ORDERS = (0, 1, 2)
LEAVES = (16, 32, 64)


def evaluate(state_path):
    data = np.load(state_path)
    nodes, edges, gamma, radii = data["nodes"], data["edges"], data["gamma"], data["radii"]
    first, speed, model = int(data["first"]), float(data["speed"]), int(data["core_model"])
    targets, exclusions = nodes[first:], data["exclusions"]
    history = np.all(edges >= 2 * first, axis=1)
    wing_points = data["wing_points"]
    cases = dict(wake=(targets, edges, gamma, radii, exclusions, "raise"),
                 wing=(wing_points, edges[history], gamma[history], radii[history], None, "raise"))
    rows = []
    for name, (points, e, g, r, ex, singular) in cases.items():
        tick = time.perf_counter()
        direct = segments_velocity(points, nodes, e, g, model=model, core_radius=r, exclusions=ex,
                                   singularity=singular, backend="numba")
        direct_s = time.perf_counter() - tick
        for theta, order, leaf in itertools.product(THETAS, ORDERS, LEAVES):
            settings = TreeSettings(theta=theta, order=order, leaf_size=leaf)
            tick = time.perf_counter()
            tree = segments_velocity(points, nodes, e, g, model=model, core_radius=r, exclusions=ex,
                                     singularity=singular, backend="tree", tree=settings)
            tree_s = time.perf_counter() - tick
            error = np.linalg.norm(tree - direct, axis=1) / speed
            rows.append(dict(state=Path(state_path).name, targets=name, points=len(points), segments=len(e),
                             theta=theta, order=order, leaf=leaf, max_err_over_U=float(error.max()),
                             rms_err_over_U=float(np.sqrt(np.mean(error**2))),
                             max_induced_over_U=float(np.max(np.linalg.norm(direct, axis=1)) / speed),
                             direct_s=direct_s, tree_s=tree_s,
                             mean_far_nodes=TreeStatistics.last.get("mean_far_nodes")))
    return rows


def main(output, states):
    all_rows = []
    for state in states:
        rows = evaluate(state)
        all_rows.extend(rows)
        best = [r for r in rows if r["theta"] == 0.3 and r["order"] == 2 and r["leaf"] == 32]
        print(json.dumps(dict(state=str(state), sample=best)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    ab.dump(output.with_suffix(".json"), dict(environment=ab.environment(), states=[str(s) for s in states],
                                             note="Timings from the shared 2-core cloud machine; indicative only."))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
