"""OF-0010 revision 19 addendum 26 (ARC-01): the task-folder files the studio's whole flight reads are all in
``resources.TASK_FILES`` (what a packaged studio ships at the same relative paths), every listed file exists, and
``resources.task_modules()`` puts the two task folders on sys.path as the worker did before (OF-0010's in front of
G5's, nothing added twice).

The files are recorded with an audit hook in a fresh interpreter: importing ``turnkey_whole_vortex``, ``turnkey_checks``
and ``platform_flight`` after ``task_modules()``, then ``turnkey_vortex.case_files()`` (it imports six check scripts of
four task folders, two of them by file path), ``pylon_loop_checks.pylon_parts()`` and ``fuselage_stations()`` - what
``WholeFlightVortex.__init__`` reads before the flight. A compiled file in ``__pycache__`` counts as its source file;
the library (src/) and the studio's own package are not task files.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT/"apps/studio"
if str(STUDIO) not in sys.path:
    sys.path.insert(0, str(STUDIO))

from openfawe_studio import resources  # noqa: E402

PROBE = r"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
seen = set()
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        name = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
        try:
            p = Path(name).resolve()
        except (OSError, ValueError):
            return
        if p.is_relative_to(root) and not (p.is_relative_to(root/"src") or p.is_relative_to(root/"apps/studio")):
            seen.add(p)
sys.addaudithook(hook)
sys.path.insert(0, str(root/"apps/studio"))
from openfawe_studio import resources
resources.ensure_library()
resources.task_modules()
import turnkey_whole_vortex as W, turnkey_checks, platform_flight  # noqa: F401
files, pylon = W.TV.case_files()
pylon.pylon_parts()
W.fuselage_stations()
out = set()
for p in seen:
    if p.parent.name == "__pycache__":
        p = p.parent.parent/(p.name.split(".")[0]+".py")
    out.add(p.relative_to(root).as_posix())
print(json.dumps(sorted(out)))
"""


class TaskFiles(unittest.TestCase):
    def test_listed_files_exist(self):
        self.assertEqual(len(set(resources.TASK_FILES)), len(resources.TASK_FILES))
        for name in resources.TASK_FILES:
            self.assertTrue((ROOT/name).is_file(), name)
        for folder in resources.TASK_FOLDERS:
            self.assertTrue(any(f.startswith(folder+"/") for f in resources.TASK_FILES), folder)

    def test_task_modules_puts_the_folders_once_in_the_old_order(self):
        saved = list(sys.path)
        try:
            folders = [str(resources.path(f)) for f in resources.TASK_FOLDERS]
            sys.path[:] = [p for p in sys.path if p not in folders]
            before = len(sys.path)
            resources.task_modules()
            self.assertEqual(len(sys.path), before+2)
            self.assertEqual(sys.path[:2], folders[::-1])        # the worker's loop: insert(0) each, in this order
            resources.task_modules()
            self.assertEqual(len(sys.path), before+2)
        finally:
            sys.path[:] = saved

    def test_every_file_read_is_listed(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
        done = subprocess.run([sys.executable, "-c", PROBE, str(ROOT)], cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=900)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        read = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertTrue(read)
        self.assertEqual(sorted(set(read)-set(resources.TASK_FILES)), [])
        self.assertIn("docs/tasks/OF-0007-g4motion20261002/a5_loop.py", read)   # loaded by file path


if __name__ == "__main__":
    unittest.main()
