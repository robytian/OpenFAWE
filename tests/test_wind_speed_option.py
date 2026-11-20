"""OF-0010 revision 18 addendum 8: platform_flight --wind-speed - the options it takes and refuses, the one parameter it
changes (and none without it), and the saved-state fingerprint, which is the same as before when the option is not given."""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT/"docs/tasks/OF-0010-platform20261022", ROOT/"docs/tasks/OF-0008-g5control20261006", ROOT/"src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def old_items(segment, window, periods, ladder, keep, power, hover_tables, blown, every, crosswind, shear, turbulence,
              free_rows, frozen, stall_lag):
    """The fingerprint list as the child built it inline before the option (0253b7b7)."""
    return ([segment, window, periods, ladder, keep, power, hover_tables, blown, every] +
            ([] if crosswind == "-" else [crosswind]) +
            ([] if shear == "-" else ["shear", shear]) +
            ([] if turbulence is None else ["turbulence", turbulence[0], turbulence[1]]) +
            ([] if free_rows == "-" else ["free_rows", free_rows]) +
            ([] if frozen == "-" else ["frozen", frozen]) +
            ([] if stall_lag == "-" else ["stall_lag", stall_lag]))


class WindSpeed(unittest.TestCase):
    BASE = ["--build", "b", "--segment", "whole+orth+gpu+p2sl+t3i", "--window", "whole"]

    def test_parse_args(self):
        import platform_flight as PFL
        self.assertEqual(PFL.parse_args(self.BASE+["--wind-speed", "8"])["--wind-speed"], "8")
        self.assertEqual(PFL.parse_args(self.BASE+["--wind-speed", "12.5", "--wind-shear-exponent", "0.2"])["--wind-speed"], "12.5")
        for bad in ("0", "-3", "nan", "inf", "x"):
            with self.assertRaises(SystemExit):
                PFL.parse_args(self.BASE+["--wind-speed", bad])
        with tempfile.TemporaryDirectory() as d:
            bts = Path(d)/"f.bts"
            bts.write_bytes(b"x")
            with self.assertRaises(SystemExit):
                PFL.parse_args(self.BASE+["--wind-speed", "8", "--turbulence", str(bts)])
        self.assertNotIn("--wind-speed", PFL.parse_args(self.BASE))

    def test_change(self):
        import platform_flight as PFL
        import turnkey_checks as TC
        from openfawe.ground import gs02 as G
        params = G.load_gs02_params(TC.PARAMS)
        before = copy.deepcopy(params.raw)
        self.assertIsNone(PFL.wind_speed_change(params, "-"))
        self.assertEqual(params.raw, before)                            # not given: nothing touched
        old = float(params.raw["phys_sim"]["wind_speed"])
        self.assertTrue(math.isfinite(old))
        self.assertEqual(PFL.wind_speed_change(params, "12"), [old, 12.0])
        after = copy.deepcopy(params.raw)
        self.assertEqual(after["phys_sim"]["wind_speed"], 12.0)
        after["phys_sim"]["wind_speed"] = before["phys_sim"]["wind_speed"]
        self.assertEqual(after, before)                                 # only that one entry changed

    def test_fingerprint(self):
        import platform_flight as PFL
        cases = [("whole+orth+gpu+p2sl+t3i", "whole", 0, "-", "1872", "on", "on", "tables", "500", "-", "-", None, "-", "-", "-"),
                 ("whole+orth+p2sl", "hover", 2100, "-", "-", "on", "-", "-", "100", "-", "-", None, "10", "rings", "-"),
                 ("whole+orth+p2sl", "hover", 2100, "-", "-", "on", "-", "-", "100", "-", "-", None, "-", "-", "4"),
                 ("whole+orth+p2sl", "whole", 0, "-", "-", "on", "on", "tables", "500", "300", "0.2", None, "-", "-", "-"),
                 ("whole+orth+p2sl", "whole", 0, "-", "-", "on", "on", "tables", "500", "-", "-", ("ab"*32, 30.0), "-", "-", "-")]
        for c in cases:
            self.assertEqual(PFL.fingerprint_items(*c), old_items(*c))
            self.assertEqual(PFL.fingerprint_items(*c, "8"), old_items(*c)+["wind_speed", "8"])


if __name__ == "__main__":
    unittest.main()
