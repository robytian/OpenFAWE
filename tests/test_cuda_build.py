"""B1-B2 of the GPU whole-step contract, revision 7 (S3''): the tree built on the GPU (``build='gpu'``).

Without a device: the option's values and refusals, and that the default stays out of the wake metadata. With a CUDA
device, or numba's simulator (NUMBA_ENABLE_CUDASIM=1): ``vortex_cuda_build.device_build_tree`` against
``vortex_tree.build_tree`` element for element (random midpoints, a regular grid with many equal coordinates, leaf sizes
1 to 32, coincident midpoints, fewer segments than a leaf, nodes longer than one shared-memory tile), and
``cuda_tree_sum`` with ``build='gpu'`` bit for bit the sum with ``build='cpu'``. On a machine with neither the device tests
skip -- so this module is not in hub/update-tests.txt.
"""
import unittest

import numpy as np

from openfawe.aero.vortex import CudaSettings, TreeSettings, cuda_backend_available
from openfawe.aero import vortex_numba as vn, vortex_tree as vt

AVAILABLE = cuda_backend_available()


def midpoint_cases():
    rng = np.random.default_rng(11)
    grid = np.array([[i*0.5, j*0.25, 0.0] for i in range(30) for j in range(20)])
    return [("random", rng.random((600, 3))*[20.0, 5.0, 2.0], 8),
            ("grid", grid, 32),
            ("grid-leaf-1", grid[:200], 1),
            ("coincident", np.ones((40, 3)), 4),
            ("fewer-than-a-leaf", rng.random((5, 3)), 32),
            ("longer-than-a-tile", rng.random((2100, 3))*[30.0, 3.0, 1.0], 16)]


class Settings(unittest.TestCase):
    def test_values_and_refusals(self):
        self.assertEqual(CudaSettings().build, "cpu")
        self.assertEqual(CudaSettings(build="gpu").build, "gpu")
        with self.assertRaises(ValueError):
            CudaSettings(build="device")
        from openfawe.aero.vortex import segments_velocity
        nodes = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "build='gpu' is for the cuda_tree backend only"):
            segments_velocity(np.zeros((1, 3)), nodes, [[0, 1]], [1.0], backend="cuda", cuda=CudaSettings(build="gpu"))

    def test_wake_config_refuses_other_backends(self):
        from openfawe.aero.wake import WakeConfig
        with self.assertRaisesRegex(ValueError, "build='gpu' needs the cuda_tree velocity backend"):
            WakeConfig(velocity_backend="cuda", cuda=CudaSettings(build="gpu"))

    def test_metadata_default_stays_out(self):
        from dataclasses import replace
        from unittest import mock
        from openfawe.aero import wake as W
        from openfawe.aero.wing import wake_config_metadata
        with mock.patch.object(W, "cuda_backend_available", return_value=True):
            base = W.WakeConfig(velocity_backend="cuda_tree", tree=TreeSettings(), cuda=CudaSettings("mixed", 128, "gpu64"))
            gpu = replace(base, cuda=CudaSettings("mixed", 128, "gpu64", build="gpu"))
        self.assertNotIn("build", wake_config_metadata(base)["cuda"])
        plain = wake_config_metadata(base)
        plain["cuda"]["build"] = "gpu"
        self.assertEqual(plain, wake_config_metadata(gpu))


@unittest.skipUnless(AVAILABLE, "no CUDA device and no simulator")
class Device(unittest.TestCase):
    def test_tree_element_for_element(self):
        from openfawe.aero import vortex_cuda_build as VCB
        for name, mid, leaf in midpoint_cases():
            with self.subTest(case=name):
                ref = vt.build_tree(mid, leaf)
                _, order, lo, hi, left, right, count = VCB.device_build_tree(mid, leaf)
                for a, b in zip(ref[:5], (order, lo, hi, left, right)):
                    self.assertTrue(np.array_equal(a, b))
                self.assertEqual(ref[5], count)

    def test_sum_bit_for_bit(self):
        from openfawe.aero import vortex_cuda_tree as ct
        from openfawe.aero import vortex_cuda_build as VCB
        rng = np.random.default_rng(12)
        starts = rng.random((500, 3))*[20.0, 5.0, 2.0]
        d = rng.normal(size=(500, 3))
        directions = d/np.linalg.norm(d, axis=1)[:, None]
        lengths = 0.2 + 0.3*rng.random(500)
        gamma = rng.normal(size=500)
        radius = np.full(500, 0.05)
        points = rng.random((60, 3))*[20.0, 5.0, 2.0]
        csr = vn.exclusion_csr(60, np.array([[0, 5], [3, 100], [59, 499]], dtype=np.int64))
        tree = TreeSettings(theta=0.4, leaf_size=8)
        for precision, suspect, moments in (("fp64", "cpu", "cpu"), ("fp64", "cpu", "gpu"), ("mixed", "gpu64", "gpu")):
            with self.subTest(precision=precision, moments=moments):
                outs = []
                for build in ("cpu", "gpu"):
                    VCB.Statistics.reset_totals()
                    outs.append(ct.cuda_tree_sum(points, starts, directions, lengths, gamma, radius, 2, csr, "zero", tree,
                                                 CudaSettings(precision, 32, suspect, moments=moments, build=build)))
                    self.assertEqual(VCB.Statistics.totals["calls"], 1 if build == "gpu" else 0)
                self.assertTrue(np.array_equal(outs[0][0], outs[1][0]))
                self.assertEqual(outs[0][1], outs[1][1])


if __name__ == "__main__":
    unittest.main()
