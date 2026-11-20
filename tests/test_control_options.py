"""GPU whole-step contract revision 7 addendum 4: the induction at the control points on the GPU (``control='gpu'``, the
direct sum) together with the tree's options of revisions 5-7. ``vsm_wake.control_options`` hands the direct sum the
settings with the tree's options at their defaults (the direct sum reads only the precision and the block), so
``segments_velocity`` no longer refuses them; with the defaults the settings are what the earlier rule gave. No device
needed (the device check is mocked or expected).
"""
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from openfawe.aero import vortex as V
from openfawe.aero import wake as W
from openfawe.aero.vortex import CudaSettings, TreeSettings
from openfawe.aero.vsm_wake import control_options


def config(**cuda):
    with mock.patch.object(W, "cuda_backend_available", return_value=True):
        return W.WakeConfig(velocity_backend="cuda_tree", tree=TreeSettings(),
                            cuda=CudaSettings("mixed", 128, "gpu64", control="gpu", **cuda))


class ControlOptions(unittest.TestCase):
    def test_tree_options_set_back_for_the_direct_sum(self):
        for cuda in (dict(), dict(moments="gpu"), dict(build="gpu"), dict(moments="gpu", build="gpu")):
            with self.subTest(**cuda):
                options = control_options(config(**cuda))
                self.assertEqual(options["backend"], "cuda")
                got = options["cuda"]
                self.assertEqual((got.precision, got.block), ("mixed", 128))
                self.assertEqual((got.suspect, got.control, got.moments, got.build), ("cpu", "cpu", "cpu", "cpu"))

    def test_defaults_as_before(self):
        cfg = config()
        self.assertEqual(control_options(cfg)["cuda"], replace(cfg.cuda, suspect="cpu", control="cpu"))

    def test_segments_velocity_takes_them(self):
        nodes = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        options = control_options(config(moments="gpu", build="gpu"))
        with mock.patch.object(V, "cuda_backend_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "needs a CUDA device"):       # past the option checks
                V.segments_velocity(np.zeros((1, 3)), nodes, [[0, 1]], [1.0], **options)
        with self.assertRaisesRegex(ValueError, "moments='gpu' is for the cuda_tree backend only"):
            V.segments_velocity(np.zeros((1, 3)), nodes, [[0, 1]], [1.0], backend="cuda",
                                cuda=CudaSettings("mixed", 128, moments="gpu"))


if __name__ == "__main__":
    unittest.main()
