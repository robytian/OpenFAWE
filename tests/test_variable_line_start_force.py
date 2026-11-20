"""OF-0011 revision 6 addenda 62-63 (F2-Ma): ``VariableLengthTether(start_force_bending=False)`` gives the start force as
Makani's CalcStartForceMomentPos (spring and damping only); the default keeps the first segment's bending shear, bit for
bit as before, and nothing else changes."""
import unittest

import numpy as np

from openfawe.tether.variable_line import VariableLengthTether


def bent_state(line, free_length):
    start, end = np.zeros(3), np.array([300.0, 40.0, 280.0])
    positions, velocities = line.init_nodes(start, end, free_length)
    rng = np.random.default_rng(7)
    positions = positions + rng.normal(scale=0.5, size=positions.shape)
    velocities = rng.normal(scale=2.0, size=velocities.shape)
    return positions, velocities, start, np.array([0.1, 0.0, 0.0]), end, np.array([0.0, 30.0, 5.0])


class StartForce(unittest.TestCase):
    def test_option(self):
        plain, makani = VariableLengthTether(), VariableLengthTether(start_force_bending=False)
        for free_length in (425.8, 300.0, 120.0):
            args = bent_state(plain, free_length)
            a = plain.forces(free_length, *args)
            b = makani.forces(free_length, *args)
            for key in ("node", "end", "tension", "damping", "segment_length"):
                self.assertTrue(np.array_equal(a[key], b[key]), key)
            self.assertFalse(np.array_equal(a["start"], b["start"]))     # the bending shear is there by default
            da = plain.derivatives(free_length, *args)
            db = makani.derivatives(free_length, *args)
            self.assertTrue(np.array_equal(da[0], db[0]) and np.array_equal(da[1], db[1]))

    def test_checks(self):
        with self.assertRaises(ValueError):
            VariableLengthTether(start_force_bending=0)
        self.assertTrue(VariableLengthTether().start_force_bending)


if __name__ == "__main__":
    unittest.main()
