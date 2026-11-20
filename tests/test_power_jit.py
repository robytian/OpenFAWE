"""OF-0014 revision 2, PJ-U: the stacked power system's period integration on the compiled path
(``makani_power_jit.integrate_period``) equals the Python path (``makani_power_sys.integrate_period`` with the rotors'
tables) bit for bit - every output, floats by their bits - over the branches the criteria list; the option's default, its
record in the settings, and the refusals.
"""
import json
import math
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from openfawe.power import makani_power_sys as mps
from openfawe.rotor.table import RotorTable

try:
    import numba  # noqa: F401
    HAVE_NUMBA = True
except ImportError:                                     # pragma: no cover
    HAVE_NUMBA = False

ROOT = Path(__file__).resolve().parents[1]
POWER = ROOT/"cases"/"powertrain"/"makani_m600_power_sys.json"
MOTORS = ROOT/"cases"/"powertrain"/"makani_m600_motors.json"
TABLE = ROOT/"cases"/"rotors"/"makani"/"rotor_rev4.json"
DIRS = (-1, 1, 1, -1, 1, -1, -1, 1)


def config():
    return mps.PowerSystemConfig.from_files(POWER, MOTORS, DIRS, [1.56]*8)


def bits(v):
    return struct.pack("<d", v)


def same(a, b):
    """Equal outputs: floats by their bits, everything else by value and type; the first difference or None."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"keys {sorted(set(a) ^ set(b))}"
        for k in a:
            d = same(a[k], b[k])
            if d:
                return f"{k}: {d}"
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return "length"
        for i, (x, y) in enumerate(zip(a, b)):
            d = same(x, y)
            if d:
                return f"[{i}] {d}"
        return None
    if isinstance(a, float):
        return None if isinstance(b, float) and bits(a) == bits(b) else f"{a!r} != {b!r}"
    return None if (a == b and type(a) is type(b)) else f"{a!r} != {b!r}"


@unittest.skipUnless(HAVE_NUMBA, "numba is not installed")
class Equal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openfawe.power import makani_power_jit as mpj
        cls.mpj = mpj
        cls.cfg = config()
        cls.tables = [RotorTable.from_makani_json(TABLE)]*8

    def fly(self, name, periods, commands, freestreams, setup=None, density=1.026):
        up, lo, tq = commands
        a, b = mps.StackedPowerSystem(self.cfg), mps.StackedPowerSystem(self.cfg)
        if setup:
            setup(a)
            setup(b)
        power = mps.TablePower(self.tables, density)
        reruns = 0
        for k in range(periods):
            for s in (a, b):
                s.discrete_step(k*0.01, up, lo)
            ra = mps.integrate_period(a, k*0.01, tq, lambda i, s: power(i, s, freestreams[i]), period_s=0.01)
            rb = self.mpj.integrate_period(b, k*0.01, tq, self.tables, freestreams, density, period_s=0.01)
            self.assertIsNone(same(ra, rb), f"{name}, period {k}")
            self.assertIsNone(same(a.x, b.x), f"{name}, period {k}: state")
            reruns += int(ra["rerun"])
        return reruns

    def commands(self, upper, lower, torque):
        return mps.rotor_commands(mps.convert_rotors(upper, lower, torque, self.cfg, True), self.cfg)

    def test_spin_up_and_torque_mode(self):
        self.fly("spin-up", 40, self.commands([160.0]*8, [150.0]*8, [0.0]*8), [5.0]*8)
        self.fly("torque mode", 40, self.commands([220.0]*8, [180.0]*8, [-300.0, 250, 400, -100, 0, 50, -500, 600]),
                 [30.0, 35, 40, 25, 60, 70, 10, -5])

    def test_low_tether_current_reruns(self):
        def low(s):
            for i in range(4):
                s.x[mps.V0 + i] = 500.0
        self.assertGreater(self.fly("low current", 2, ([0.0]*8, [0.0]*8, [0.0]*8), [0.0]*8, low), 0)

    def test_saturated_average_voltage(self):
        def high(s):
            for i in range(4):
                s.x[mps.V0 + i] = 1700.0
        self.fly("saturated", 3, self.commands([160.0]*8, [150.0]*8, [0.0]*8), [5.0]*8, high)

    def test_faults_and_release(self):
        cmd = self.commands([220.0]*8, [180.0]*8, [-300.0, 250, 400, -100, 0, 50, -500, 600])

        def one(s):
            s.connections[5] = False
            s.x[mps.V0 + 1] = 900.0

        def both(s):
            s.connections[2] = s.connections[6] = False

        def released(s):
            s.tether_released = True
        self.fly("one motor off", 10, cmd, [30.0]*8, one)
        self.fly("both motors off", 10, cmd, [30.0]*8, both)
        self.fly("tether released", 10, cmd, [30.0]*8, released)

    def test_low_block_voltages(self):
        def low(s):
            s.x[mps.V0 + 2] = -5.0
            s.x[mps.V0 + 3] = 0.5
        self.fly("block voltage <= 0 and < 1 V", 3, self.commands([220.0]*8, [180.0]*8, [100.0]*8), [30.0]*8, low)

    def test_inverted_limits_slow_negative_speeds_and_table_edges(self):
        def speeds(s):
            for i in range(8):
                s.x[mps.OMEGA0 + i] = (-20.0 if i % 2 else 3.0)*float(DIRS[i])
        self.fly("inverted, slow, negative, beyond the table", 15, self.commands([100.0]*8, [150.0]*8, [100.0]*8),
                 [200.0, -50.0, 0.0, 300.0, 5.0, 5.0, 5.0, 5.0], speeds)

    def test_nan_index_raises_on_both_paths(self):
        system = mps.StackedPowerSystem(self.cfg)
        with self.assertRaises(ValueError):
            self.mpj.integrate_period(system, 0.0, [0.0]*8, self.tables, [math.nan]*8, 1.026)
        system = mps.StackedPowerSystem(self.cfg)
        power = mps.TablePower(self.tables, 1.026)
        with self.assertRaises(ValueError):
            mps.integrate_period(system, 0.0, [0.0]*8, lambda i, s: power(i, s, math.nan))

    def test_refusals(self):
        system = mps.StackedPowerSystem(self.cfg)
        system.trace = {}
        with self.assertRaisesRegex(ValueError, "branch trace"):
            self.mpj.integrate_period(system, 0.0, [0.0]*8, self.tables, [5.0]*8, 1.026)
        with self.assertRaisesRegex(ValueError, "eight rotor tables"):
            self.mpj.integrate_period(mps.StackedPowerSystem(self.cfg), 0.0, [0.0]*8, self.tables[:7], [5.0]*8, 1.026)


@unittest.skipUnless(HAVE_NUMBA, "numba is not installed")
class Turnkey(unittest.TestCase):
    """Through ``turnkey.power_period``: the compiled configuration gives the same fields; a trace keeps the Python path;
    an aerodynamic power that is not the tables is refused on the compiled path."""

    def test_power_period_fields_equal(self):
        from openfawe.ground import turnkey as tk
        tables = [RotorTable.from_makani_json(TABLE)]*8
        plain, fast = config(), replace(config(), compiled=True)
        out = []
        for cfg in (plain, fast):
            system, pipe = mps.StackedPowerSystem(cfg), mps.CommandPipeline()
            rows = []
            for k in range(8):
                omegas, rates, torques, fields = tk.power_period(system, pipe, cfg, k*0.01, [12.0]*8,
                                                                 mps.TablePower(tables, 1.026), 0.01)
                rows.append(json.dumps(dict(o=list(map(float, omegas)), r=list(map(float, rates)), t=torques, f=fields),
                                       sort_keys=True))
                on = k >= 2
                pipe.push(mps.convert_rotors([150.0 if on else 0.0]*8, [140.0 if on else 0.0]*8, [30.0 if on else 0.0]*8,
                                             cfg, on))
            out.append(rows)
        self.assertEqual(out[0], out[1])

    def test_trace_keeps_python_and_lambda_refused(self):
        from openfawe.ground import turnkey as tk
        tables = [RotorTable.from_makani_json(TABLE)]*8
        cfg = replace(config(), compiled=True)
        system, pipe = mps.StackedPowerSystem(cfg), mps.CommandPipeline()
        system.trace = {}
        tk.power_period(system, pipe, cfg, 0.0, [12.0]*8, mps.TablePower(tables, 1.026), 0.01)
        self.assertTrue(system.trace)                      # the Python path counted its branches
        system = mps.StackedPowerSystem(cfg)
        with self.assertRaisesRegex(ValueError, "TablePower"):
            tk.power_period(system, pipe, cfg, 0.0, [12.0]*8, lambda i, s, v: 0.0, 0.01)


class Option(unittest.TestCase):
    def test_default_off_and_settings(self):
        self.assertFalse(config().compiled)
        sys.path.insert(0, str(ROOT/"docs"/"tasks"/"OF-0008-g5control20261006"))
        import turnkey_checks as TC
        from tests.test_turnkey import rotors          # by the package: run alone, tests/ is not on the path
        discs, inertia = rotors()
        self.assertFalse(TC.power_system_config(discs, inertia).compiled)
        self.assertTrue(TC.power_system_config(discs, inertia, compiled=True).compiled)
        self.assertEqual(replace(TC.power_system_config(discs, inertia, compiled=True), compiled=False),
                         TC.power_system_config(discs, inertia))

    def test_platform_flight_power_values(self):
        sys.path.insert(0, str(ROOT/"docs"/"tasks"/"OF-0010-platform20261022"))
        import platform_flight as PFL
        base = ["--build", "x", "--segment", "whole+orth+p2sl+t3i", "--window", "hover", "--periods", "10",
                "--hover-tables", "on", "--blown-wing", "tables"]
        self.assertEqual(PFL.parse_args(base + ["--power", "jit"])["--power"], "jit")
        self.assertEqual(PFL.parse_args(base + ["--power", "on"])["--power"], "on")
        with self.assertRaises(SystemExit):
            PFL.parse_args(base + ["--power", "fast"])


if __name__ == "__main__":
    unittest.main()
