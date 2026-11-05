"""Save wake networks of archived-configuration VSM free-wake runs for kernel checks.

Uses the exact Numba backend (A1/A3 equivalent to NumPy). Output: one .npz per
(model, step) with nodes, edges, gamma, radii, first free node index, targets
(old TE + free nodes), wing evaluation points (closure and load points),
exclusions as built by wake._network_convection, and reference values.
"""
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import numpy as np  # noqa: E402
import accel_bench as ab  # noqa: E402
from openfawe.aero.vsm import geometry  # noqa: E402

OUT = ab.REPO / "local" / "accel-20261001" / "states"


def exclusions_for(nodes, edges, first):
    points = nodes[first:]
    locations = {}
    for index, point in enumerate(points):
        locations.setdefault(tuple(point), []).append(index)
    pairs = []
    for segment, (a, b) in enumerate(edges):
        for node in (a, b):
            pairs.extend((index, segment) for index in locations.get(tuple(nodes[node]), ()))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def main(model, steps, keep):
    result = ab.run(model, "vsm_free_wake", steps=steps, keep_states=tuple(keep),
                    wake_options=dict(velocity_backend="numba"), progress=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for step, state in result["states"].items():
        nodes, edges, gamma, radii = ab.network_of(state)
        first = state.wing.node_count
        wing_points = np.vstack((geometry(state.wing)[0], state.wing.cp))
        np.savez(OUT / f"{model.lower()}_vsm_step{step:04d}.npz", nodes=nodes, edges=edges, gamma=gamma,
                 radii=radii, first=first, wing_points=wing_points,
                 exclusions=exclusions_for(nodes, edges, first), chord=result["chord"],
                 speed=result["speed"], core_model=int(state.config.core_model))
        print(model, step, "segments", len(edges), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")])
