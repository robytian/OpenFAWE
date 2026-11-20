"""The machine as a run sees it (decision 0056 addendum 3): the CPUs a process may really use, the memory, the GPU.

On a rented container the logical CPU count (``os.cpu_count``) is the host's - 192 on node 3 - while the container may
use fewer: its affinity mask (``os.sched_getaffinity``) and its cgroup CPU quota (v2 ``cpu.max``; v1
``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``) limit it. ``usable_cpus`` is the smallest of them (at least 1). Numba's
thread pool and the jobs' thread counts take that number, so that threads do not outnumber the CPUs they get.

Nothing here changes a computed value; it only reads the system. Every function works without a GPU and without
psutil (the result then says what is unknown).
"""
from __future__ import annotations

import math
import os
import platform
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")


def cgroup_quota_cpus(root=CGROUP):
    """The cgroup CPU quota in CPUs (a float), or None when there is none (v2 'max', v1 -1) or no cgroup file."""
    root = Path(root)
    v2 = root / "cpu.max"
    try:
        quota, period = v2.read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return None if quota <= 0 or period <= 0 else quota / period
    except (OSError, ValueError):
        return None


def cpu_counts(root=CGROUP):
    """dict(logical, affinity, cgroup_quota, usable): ``usable`` = min(affinity, floor(quota)), at least 1."""
    logical = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):          # Windows, macOS
        affinity = logical
    quota = cgroup_quota_cpus(root)
    usable = affinity if quota is None else min(affinity, max(1, math.floor(quota + 1e-9)))
    return dict(logical=logical, affinity=affinity, cgroup_quota=quota, usable=max(1, int(usable)))


def usable_cpus(root=CGROUP):
    return cpu_counts(root)["usable"]


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def memory_bytes(root=CGROUP):
    """dict(total, cgroup_limit): the machine's memory and the container's limit (None when unknown or unlimited)."""
    total = None
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
    except Exception:                          # noqa: BLE001 - psutil absent or failing: read /proc
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
        except OSError:
            pass
    limit = None
    for name in ("memory.max", "memory/memory.limit_in_bytes"):
        try:
            text = (Path(root) / name).read_text().strip()
            if text != "max" and int(text) < 1 << 60:
                limit = int(text)
                break
        except (OSError, ValueError):
            continue
    return dict(total=total, cgroup_limit=limit)


def gpu_info():
    """The CUDA devices numba sees: name, compute capability, memory; or why there is none. Never raises."""
    try:
        from numba import cuda
        if not cuda.is_available():
            return dict(available=False, reason="numba.cuda.is_available() is False")
        devices = []
        for d in cuda.list_devices():
            with d:
                dev = cuda.get_current_device()
                free, total = cuda.current_context().get_memory_info()
                name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
                devices.append(dict(id=int(d.id), name=name, compute_capability=list(dev.compute_capability),
                                    memory_total=int(total), memory_free=int(free),
                                    multiprocessors=int(dev.MULTIPROCESSOR_COUNT)))
        driver = None
        try:
            driver = ".".join(str(v) for v in cuda.cudadrv.driver.driver.get_version())
        except Exception:                      # noqa: BLE001 - reported as unknown
            pass
        return dict(available=True, devices=devices, driver_cuda_version=driver)
    except Exception as error:                 # noqa: BLE001 - no numba, no driver: reported
        return dict(available=False, reason=f"{type(error).__name__}: {error}")


def probe(gpu=True, root=CGROUP):
    """Everything above in one dict (for a run's receipt)."""
    return dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(), cpu_model=cpu_model(),
                cpus=cpu_counts(root), memory=memory_bytes(root), gpu=gpu_info() if gpu else dict(available=None, reason="not probed"))


if __name__ == "__main__":
    import json
    print(json.dumps(probe(), indent=1))
