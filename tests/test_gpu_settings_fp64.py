"""GPU whole-step contract revision 12: the whole-flight hook's CUDA settings - with precision 'fp64' and no suspect of
the caller's the suspect is 'cpu' (fp64 with the default 'gpu64' was refused); 'mixed' keeps 'gpu64'; a suspect the caller
gives is kept (and fp64 + gpu64 given explicitly is still refused by CudaSettings)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT/"docs/tasks/OF-0008-g5control20261006", ROOT/"src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import turnkey_whole_vortex as W  # noqa: E402


class CudaSettingsMerge(unittest.TestCase):
    def test_default_is_mixed_with_gpu64(self):
        s = W.cuda_settings()
        self.assertEqual((s.precision, s.suspect), ("mixed", "gpu64"))
        self.assertEqual(W.cuda_settings(dict(control="gpu")).suspect, "gpu64")

    def test_fp64_takes_cpu_suspects(self):
        s = W.cuda_settings(dict(precision="fp64"))
        self.assertEqual((s.precision, s.suspect), ("fp64", "cpu"))
        s = W.cuda_settings(dict(precision="fp64", control="gpu", moments="gpu", build="gpu"))
        self.assertEqual((s.suspect, s.control, s.moments, s.build), ("cpu", "gpu", "gpu", "gpu"))

    def test_callers_suspect_is_kept(self):
        self.assertEqual(W.cuda_settings(dict(precision="mixed", suspect="cpu")).suspect, "cpu")
        with self.assertRaises(ValueError):
            W.cuda_settings(dict(precision="fp64", suspect="gpu64"))


if __name__ == "__main__":
    unittest.main()
