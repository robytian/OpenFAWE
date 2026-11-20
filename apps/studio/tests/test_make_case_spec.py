"""OF-0013 addendum 14: the example case's spec.json carries no local path (rc2's RS-3 found the run's own spec with the
controller folder of the PC that ran it)."""
import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]/"release"
sys.path.insert(0, str(RELEASE))

import make_case  # noqa: E402


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


class SanitisedSpec(unittest.TestCase):
    def meta(self):
        folder = "C:\\Users\\someone\\Desktop\\work\\third_party\\makani-controller-win-nogsl"
        return dict(app="OpenFAWE Studio", makani_build=folder, spec=dict(motion="free_flight", makani_build=folder),
                    source=dict(folder="/home/someone/runs/e4"),
                    library=dict(source=dict(git=dict(head="abc", status="M x")), environment=dict(python="3.12")))

    def test_no_local_path_left(self):
        out = make_case.sanitised_spec(self.meta())
        for text in strings(out):
            self.assertNotIn("someone", text)
            self.assertNotIn(":\\", text)
            self.assertFalse(text.startswith("/"), text)
        self.assertEqual(out["makani_build"], "makani-controller-win-nogsl")
        self.assertEqual(out["spec"]["makani_build"], "")              # the bundled library
        self.assertEqual(out["spec"]["motion"], "free_flight")
        self.assertEqual(out["source"]["folder"], "e4")
        self.assertEqual(out["library"], dict(git_head="abc", python="3.12", packages=None))

    def test_empty_spec_build_unchanged(self):
        meta = self.meta()
        meta["spec"]["makani_build"] = ""
        self.assertEqual(make_case.sanitised_spec(meta)["spec"]["makani_build"], "")


if __name__ == "__main__":
    unittest.main()
