"""Decision 0056 addendum 3: the usable CPUs from the affinity and the cgroup quota (v2 and v1), the memory limit, the
GPU probe without a GPU, and the package's module registry."""
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openfawe import compute
from openfawe.compute import hardware as H


class Cpus(unittest.TestCase):
    def test_cgroup_v2_quota_limits(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cpu.max").write_text("1600000 100000\n")
            self.assertEqual(H.cgroup_quota_cpus(d), 16.0)
            with mock.patch.object(H.os, "sched_getaffinity", lambda pid: set(range(192)), create=True):
                c = H.cpu_counts(d)
            self.assertEqual((c["affinity"], c["cgroup_quota"], c["usable"]), (192, 16.0, 16))

    def test_no_quota_and_fractional_quota(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cpu.max").write_text("max 100000\n")
            self.assertIsNone(H.cgroup_quota_cpus(d))
            with mock.patch.object(H.os, "sched_getaffinity", lambda pid: set(range(8)), create=True):
                self.assertEqual(H.cpu_counts(d)["usable"], 8)
            (Path(d) / "cpu.max").write_text("50000 100000\n")             # half a CPU: still one thread
            with mock.patch.object(H.os, "sched_getaffinity", lambda pid: set(range(8)), create=True):
                self.assertEqual(H.cpu_counts(d)["usable"], 1)

    def test_cgroup_v1(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cpu").mkdir()
            (Path(d) / "cpu" / "cpu.cfs_quota_us").write_text("2400000\n")
            (Path(d) / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
            self.assertEqual(H.cgroup_quota_cpus(d), 24.0)
            (Path(d) / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
            self.assertIsNone(H.cgroup_quota_cpus(d))

    def test_no_cgroup_files(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(H.cgroup_quota_cpus(d))
            self.assertGreaterEqual(H.usable_cpus(d), 1)

    def test_memory_limit(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "memory.max").write_text("68719476736\n")
            self.assertEqual(H.memory_bytes(d)["cgroup_limit"], 68719476736)
            (Path(d) / "memory.max").write_text("max\n")
            self.assertIsNone(H.memory_bytes(d)["cgroup_limit"])


class Probe(unittest.TestCase):
    def test_probe_never_raises(self):
        p = H.probe()
        self.assertIn("usable", p["cpus"])
        self.assertIn("available", p["gpu"])

    def test_registry(self):
        for name, role in compute.MODULES.items():
            self.assertIn(role, ("product", "check"))
            importlib.import_module(f"openfawe.compute.{name}")


if __name__ == "__main__":
    unittest.main()
