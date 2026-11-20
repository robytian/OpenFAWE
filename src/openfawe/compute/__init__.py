"""Compute resources (decision 0056 addendum 3; owner 2026-11-12 18:24): what the machine offers and how a run uses it.

``MODULES`` registers every module with its role, as in ``openfawe.floating`` and ``openfawe.structure``. Opt-in: nothing
in the existing models imports this package.
"""
MODULES = {
    "hardware": "product",          # the machine as a run sees it: usable CPUs (affinity, cgroup quota), memory, GPU
}
