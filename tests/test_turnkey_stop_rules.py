"""OF-0010 revision 19 addendum 22 (ST-05): TurnKey's stop rules as a keyword argument - the values a run uses with and
without ``stop_rules``, a module constant set by a caller still used, unknown keys and bad values refused, the joystick's
return period, and (with the library) a run without the option has no new settings key while one with it stops at its
end rule."""
import os
import unittest
from pathlib import Path

from openfawe.ground import gs02 as G
from openfawe.ground import turnkey as T

ROOT = Path(__file__).resolve().parents[1]
BUILD = os.environ.get("OPENFAWE_MAKANI_FULL_BUILD")


class Rules(unittest.TestCase):
    def test_values(self):
        default = T.stop_rules_of()
        self.assertEqual(default, dict(throttle_return_period=82000, stall_periods=45000, end_periods=160000,
                                       descend_periods=12000, gain_done_hold_periods=500))
        self.assertEqual(T.stop_rules_of(None), default)
        self.assertEqual(T.stop_rules_of(dict(end_periods=200000))["end_periods"], 200000)
        self.assertEqual(T.stop_rules_of(dict(end_periods=200000))["stall_periods"], 45000)
        saved = T.STALL_PERIODS
        try:                                               # table_variants sets the module constants before the call
            T.STALL_PERIODS = 600000
            self.assertEqual(T.stop_rules_of()["stall_periods"], 600000)
        finally:
            T.STALL_PERIODS = saved
        for bad in (dict(end=1), dict(end_periods=0), dict(end_periods=-5), dict(end_periods=1.5), dict(end_periods=True)):
            with self.assertRaises(ValueError):
                T.stop_rules_of(bad)

    def test_joystick(self):
        class Api:
            kJoystickSwitchPositionUp, kJoystickSwitchPositionMiddle = 2, 1
        api = Api()
        for k in (0, 1009, 1010, 1100, 81999, 82000, 90000):
            a, b = T.joystick_at(api, k), T.joystick_at(api, k, None)
            c = T.joystick_at(api, k, T.THROTTLE_RETURN_PERIOD)
            self.assertEqual((a.switch_position, a.throttle), (b.switch_position, b.throttle))
            self.assertEqual((a.switch_position, a.throttle), (c.switch_position, c.throttle))
        self.assertEqual(T.joystick_at(api, 85000, 90000).throttle, 0.81)
        self.assertEqual(T.joystick_at(api, 90000, 90000).throttle, 0.39)


@unittest.skipUnless(BUILD, "OPENFAWE_MAKANI_FULL_BUILD not set")
class Run(unittest.TestCase):
    def test_settings_and_end(self):
        from openfawe.aero.makani_database import MakaniDatabase
        from openfawe.flight.makani_full import load_full_api
        from openfawe.rotor.actuator_disc import load_rotor_inertia, load_rotor_set
        from openfawe.rotor.table import RotorTable
        data = ROOT/"cases"/"polars"/"makani"
        api = load_full_api(BUILD)
        params = G.load_gs02_params(ROOT/"cases"/"ground"/"makani_gs02_parker_ranch.json")
        aero = MakaniDatabase.from_files(data/"m600_aswing_baseline.json", spoiler_path=data/"spoiler_offset_tables.json",
                                         high_path=data/"hover_model_sn02_wind_10_no_wake_model.json")
        rotor_set = ROOT/"cases"/"rotors"/"m600_rotors_makani.json"
        discs = load_rotor_set(rotor_set, RotorTable.from_makani_json(ROOT/"cases"/"rotors"/"makani"/"rotor_rev4.json"))
        inertia = load_rotor_inertia(rotor_set)
        plain = T.run_turnkey(params, api, aero, discs, inertia, max_periods=40)
        ruled = T.run_turnkey(params, api, aero, discs, inertia, max_periods=40, stop_rules=dict(end_periods=30))
        self.assertNotIn("stop_rules", plain["settings"])
        self.assertEqual(ruled["settings"]["stop_rules"]["end_periods"], 30)
        self.assertEqual(len(plain["records"]), 40)
        self.assertEqual(len(ruled["records"]), 30)
        self.assertEqual(ruled["end"]["reason"], "time_limit")
        # The records of two runs in one process are not compared: the controller library keeps state between calls
        # (two identical calls differ in the gates at period 0). Bitwise identity of the default is d1/d2 in separate
        # processes (the R1-type check of the addendum).


if __name__ == "__main__":
    unittest.main()
