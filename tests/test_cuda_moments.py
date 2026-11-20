"""M1 of the GPU whole-step contract, revision 6 (S3'): the tree's node quantities on the GPU (``moments='gpu'``).

Without a device: the option's values and refusals, and that the default stays out of the wake metadata. With a CUDA
device, or numba's simulator (NUMBA_ENABLE_CUDASIM=1; logic only, the simulator rounds nothing to float32): the node
quantities against ``vortex_tree.node_moments`` and ``node_contractions``, repeatability, ``cuda_tree_sum`` with
``moments='gpu'`` against ``moments='cpu'``, and that the option really reaches the GPU code. On a machine with neither,
the device tests skip -- so this module is not in hub/update-tests.txt.
"""
import os
import unittest

import numpy as np

from openfawe.aero.vortex import CudaSettings, TreeSettings, cuda_backend_available
from openfawe.aero import vortex_numba as vn, vortex_tree as vt

AVAILABLE = cuda_backend_available()
SIMULATOR = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
QUANTITIES = ("centre", "bound", "core", "m0", "m1", "m2", "w1", "t2", "w2")


def segments(seed, count=400, zero_every=7):
    rng = np.random.default_rng(seed)
    starts = rng.random((count, 3))*[20.0, 5.0, 2.0]
    d = rng.normal(size=(count, 3))
    directions = d/np.linalg.norm(d, axis=1)[:, None]
    lengths = 0.2 + 0.3*rng.random(count)
    gamma = rng.normal(size=count)
    if zero_every:
        gamma[::zero_every] = 0.0
    radius = rng.uniform(0.0, 0.08, count)
    return starts, directions, lengths, gamma, radius


def cpu_and_gpu(seed, leaf, count=400):
    from openfawe.aero import vortex_cuda_moments as VCM
    starts, directions, lengths, gamma, radius = segments(seed, count)
    lv = directions*lengths[:, None]
    ends = starts + lv
    alpha = gamma[:, None]*lv
    order, lo, hi, _, _, _ = vt.build_tree(0.5*(starts + ends), leaf)
    centre, bound, core, m0, m1, m2 = vt.node_moments(order, lo, hi, starts, ends, alpha, lv, radius, gamma)
    w1, t2, w2 = vt.node_contractions(m1, m2)
    cpu = dict(zip(QUANTITIES, (centre, bound, core, m0, m1, m2, w1, t2, w2)))
    gpu = dict(zip(QUANTITIES, VCM.node_quantities(order, lo, hi, starts, ends, alpha, lv, radius, gamma)))
    return cpu, gpu, (order, lo, hi, starts, ends, alpha, lv, radius, gamma)


def node_relative(a, b):
    """Per node: the largest difference over the node's components, over the largest component's magnitude."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.abs(a).reshape(len(a), -1).max(axis=1)
    diff = np.abs(a - b).reshape(len(a), -1).max(axis=1)
    return np.where(scale > 0, diff/np.where(scale > 0, scale, 1.0), diff)


class Settings(unittest.TestCase):
    def test_values_and_refusals(self):
        self.assertEqual(CudaSettings().moments, "cpu")
        self.assertEqual(CudaSettings(moments="gpu").moments, "gpu")
        with self.assertRaises(ValueError):
            CudaSettings(moments="device")
        from openfawe.aero.vortex import segments_velocity
        nodes = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "moments='gpu' is for the cuda_tree backend only"):
            segments_velocity(np.zeros((1, 3)), nodes, [[0, 1]], [1.0], backend="cuda", cuda=CudaSettings(moments="gpu"))

    def test_wake_config_refuses_other_backends(self):
        from openfawe.aero.wake import WakeConfig
        with self.assertRaisesRegex(ValueError, "moments='gpu' needs the cuda_tree velocity backend"):
            WakeConfig(velocity_backend="cuda", cuda=CudaSettings(moments="gpu"))

    def test_metadata_default_stays_out(self):
        from dataclasses import replace
        from unittest import mock
        from openfawe.aero import wake as W
        from openfawe.aero.wing import wake_config_metadata
        with mock.patch.object(W, "cuda_backend_available", return_value=True):
            base = W.WakeConfig(velocity_backend="cuda_tree", tree=TreeSettings(), cuda=CudaSettings("mixed", 128, "gpu64"))
            gpu = replace(base, cuda=CudaSettings("mixed", 128, "gpu64", moments="gpu"))
        self.assertNotIn("moments", wake_config_metadata(base)["cuda"])
        self.assertEqual(wake_config_metadata(gpu)["cuda"]["moments"], "gpu")
        plain = wake_config_metadata(base)
        plain["cuda"]["moments"] = "gpu"
        self.assertEqual(plain, wake_config_metadata(gpu))          # the option is the only difference


@unittest.skipUnless(AVAILABLE, "no CUDA device and no simulator")
class Device(unittest.TestCase):
    def test_centre_and_core_bit_for_bit_the_rest_to_rounding(self):
        for seed, leaf in ((1, 8), (2, 32), (3, 1)):
            cpu, gpu, _ = cpu_and_gpu(seed, leaf, count=300 if leaf > 1 else 120)
            with self.subTest(seed=seed, leaf=leaf):
                self.assertTrue(np.array_equal(cpu["centre"], gpu["centre"]))
                self.assertTrue(np.array_equal(cpu["core"], gpu["core"]))
                for name in ("bound", "m0", "m1", "m2"):
                    a = cpu[name] if np.ndim(cpu[name]) > 1 else np.asarray(cpu[name])[:, None]
                    b = gpu[name] if np.ndim(gpu[name]) > 1 else np.asarray(gpu[name])[:, None]
                    self.assertLessEqual(float(node_relative(a, b).max()), 1e-12, name)
                # The contractions are differences of moments; where they cancel, their own size is at the moments'
                # rounding, so they are measured against the moments they are made of (code-commit addendum).
                for name, made_of in (("w1", "m1"), ("t2", "m2"), ("w2", "m2")):
                    scale = np.abs(cpu[made_of]).reshape(len(cpu[made_of]), -1).max(axis=1)
                    diff = np.abs(cpu[name] - gpu[name]).reshape(len(cpu[name]), -1).max(axis=1)
                    self.assertTrue(np.all(diff <= 1e-12*scale), name)

    def test_repeatable(self):
        from openfawe.aero import vortex_cuda_moments as VCM
        _, gpu, args = cpu_and_gpu(4, 16)
        again = dict(zip(QUANTITIES, VCM.node_quantities(*args)))
        for name in QUANTITIES:
            self.assertTrue(np.array_equal(gpu[name], again[name]), name)

    def test_sum_with_gpu_moments_is_the_cpu_moments_sum(self):
        from openfawe.aero import vortex_cuda_tree as ct
        from openfawe.aero import vortex_cuda_moments as VCM
        starts, directions, lengths, gamma, radius = segments(5, 500)
        rng = np.random.default_rng(6)
        points = rng.random((60, 3))*[20.0, 5.0, 2.0]
        pairs = np.array([[0, 5], [0, 17], [3, 100], [10, 200], [10, 201], [59, 499]], dtype=np.int64)
        csr = vn.exclusion_csr(len(points), pairs)
        for leaf in (4, 32):
            tree = TreeSettings(theta=0.4, leaf_size=leaf)
            outs = {}
            for moments in ("cpu", "gpu"):
                VCM.Statistics.reset_totals()
                outs[moments] = ct.cuda_tree_sum(points, starts, directions, lengths, gamma, radius, 2, csr, "zero", tree,
                                                 CudaSettings("fp64", 32, moments=moments))
                self.assertEqual(VCM.Statistics.totals["calls"], 1 if moments == "gpu" else 0)
                self.assertEqual(ct.Statistics.last.get("moments"), "gpu" if moments == "gpu" else None)
            (a, fa), (b, fb) = outs["cpu"], outs["gpu"]
            scale = np.linalg.norm(a, axis=1).max()
            self.assertLessEqual(float(np.linalg.norm(a - b, axis=1).max()), 1e-12*scale)
            self.assertEqual(fa, fb)

    def test_single_precisions_run_with_gpu_moments(self):
        from openfawe.aero import vortex_cuda_tree as ct
        starts, directions, lengths, gamma, radius = segments(7, 400)
        rng = np.random.default_rng(8)
        points = rng.random((50, 3))*[20.0, 5.0, 2.0]
        csr = vn.exclusion_csr(len(points), np.zeros((0, 2), dtype=np.int64))
        tree = TreeSettings(theta=0.4, leaf_size=16)
        reference, _ = ct.cuda_tree_sum(points, starts, directions, lengths, gamma, radius, 2, csr, "zero", tree,
                                        CudaSettings("fp64", 32))
        scale = np.linalg.norm(reference, axis=1).max()
        for precision, suspect in (("mixed", "cpu"), ("mixed", "gpu64"), ("fp32", "gpu64")):
            out, _ = ct.cuda_tree_sum(points, starts, directions, lengths, gamma, radius, 2, csr, "zero", tree,
                                      CudaSettings(precision, 32, suspect, moments="gpu"))
            self.assertLessEqual(float(np.linalg.norm(out - reference, axis=1).max()), 1e-5*scale, (precision, suspect))

    def test_empty(self):
        from openfawe.aero import vortex_cuda_tree as ct
        starts, directions, lengths, gamma, radius = segments(9, 50)
        tree = TreeSettings(theta=0.4)
        for precision in ("fp64", "mixed"):
            settings = CudaSettings(precision, 32, moments="gpu")
            out, flag = ct.cuda_tree_sum(np.zeros((0, 3)), starts, directions, lengths, gamma, radius, 2,
                                         vn.exclusion_csr(0, np.zeros((0, 2), dtype=np.int64)), "raise", tree, settings)
            self.assertEqual(out.shape, (0, 3))
            self.assertFalse(flag)


if __name__ == "__main__":
    unittest.main()
