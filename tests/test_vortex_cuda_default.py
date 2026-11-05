"""Default paths are untouched by the opt-in cuda backend (GPU direct-sum contract section 4 item 2 (c), (d); review
GPUC-R07). Runs everywhere, without a GPU and without numba.cuda."""
import subprocess
import sys
import unittest

from openfawe.aero.vortex import TreeSettings
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import wake_config_metadata


class DefaultPaths(unittest.TestCase):
    def test_metadata_has_no_cuda_key(self):
        for config in (WakeConfig(), WakeConfig(velocity_backend="numba"),
                       WakeConfig(velocity_backend="tree", tree=TreeSettings(theta=0.2))):
            self.assertNotIn("cuda", wake_config_metadata(config))

    def test_cuda_backend_not_imported(self):
        # Only our module is required to stay out: with numba-cuda installed, numba's extension entry points may import
        # numba_cuda on the first compile (review GPUC2 note), which is not ours to prevent.
        code = ("import sys; from openfawe.aero.vortex import segments_velocity; import openfawe.aero.wake; "
                "segments_velocity([[0., 1., 0.]], [[0., 0., 0.], [1., 0., 0.]], [[0, 1]], 1., backend='numba'); "
                "segments_velocity([[0., 1., 0.]], [[0., 0., 0.], [1., 0., 0.]], [[0, 1]], 1., backend='tree', tree=__import__("
                "'openfawe.aero.vortex', fromlist=['TreeSettings']).TreeSettings()); "
                "print(sorted(m for m in sys.modules if m.startswith('openfawe.aero.vortex_cuda')))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        self.assertEqual(done.returncode, 0, done.stderr[-1000:])
        self.assertEqual(done.stdout.strip().splitlines()[-1], "[]")


if __name__ == "__main__":
    unittest.main()
