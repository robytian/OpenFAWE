"""Regression for independently scheduled long prescribed-motion clocks."""
import unittest
import numpy as np
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing,SectionPolar


class PhysicalClockTests(unittest.TestCase):
    def test_long_independent_clock_and_wrong_step_rejection(self):
        wing=Wing.ellipse(2);polar=SectionPolar.linear()
        for method in ('vsm','free_wake'):
            with self.subTest(method=method):
                backend=AeroBackend(method,wake=WakeConfig(dt_s=.1,max_rows=121))
                state=backend.start(wing,polar,[5.,0.,0.],time_s=.37)
                for i in range(1,121):
                    expected=.37+i*.1
                    state=backend.advance(state,polar,[5.,0.,0.],next_time_s=expected)
                    self.assertEqual(state.time_s,expected)
                with self.assertRaisesRegex(ValueError,'next time'):
                    backend.advance(state,polar,[5.,0.,0.],next_time_s=state.time_s+.101)
                self.assertEqual(state.step,120)
                np.testing.assert_array_equal(state.solution.force,[0.,0.,0.])


if __name__=='__main__':unittest.main()
