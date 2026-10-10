"""Geometrically shared junctions must not depend on disconnected node IDs."""
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from openfawe.aero.aircraft import WingAssembly, PanelPolars, load_aircraft
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings
from openfawe.aero.wake import WakeConfig
from openfawe.aero.methods import AeroBackend
from openfawe.aero.reference import inflow
from openfawe.aero.vortex import FilamentSingularity

ROOT=Path(__file__).resolve().parents[1]


class WakeJunctionTests(unittest.TestCase):
    def test_single_rectangle_matches_two_touching_halves(self):
        le=np.array([[0.,-1.,0.],[0.,0.,0.],[0.,1.,0.]])
        te=le+[1.,0.,0.]
        whole=Wing(le,te)
        split=WingAssembly((Wing(le[:2],te[:2]),Wing(le[1:],te[1:])),('left','right'))
        polar=SectionPolar.linear()
        tables=PanelPolars((polar,polar))
        for core in ('constant','age_stretch'):
            backend=AeroBackend('free_wake',wake=WakeConfig(dt_s=.1,max_rows=3,core_evolution=core,
                solve=SolveSettings(solver='newton')))
            a=backend.start(whole,polar,[2.,0.,.2])
            b=backend.start(split,tables,[2.,0.,.2])
            for _ in range(3):
                a=backend.advance(a,polar,[2.,0.,.2])
                b=backend.advance(b,tables,[2.,0.,.2])
                assert_allclose(b.free_rows[:,[0,1,3]],a.free_rows,atol=1e-12,rtol=1e-12)
                assert_allclose(b.free_rows[:,1],b.free_rows[:,2],atol=0,rtol=0)
                assert_allclose(b.solution.force,a.solution.force,atol=1e-11,rtol=1e-11)
                assert_allclose(b.solution.circulation,a.solution.circulation,atol=1e-11,rtol=1e-11)
            # Arbitrary field probes have not inherited the convection exemption.
            with self.assertRaises(FilamentSingularity):
                backend.induced_velocity(b,b.wing.trailing_edge[1:2])

    def test_actual_m600_tail_root_can_advance(self):
        definition=load_aircraft(ROOT/'cases/aircraft/m600_neutral.json')
        wing=WingAssembly(definition.wing.wings[:5],definition.wing.names[:5])
        polar=PanelPolars(definition.polar.polars[:wing.panels])
        backend=AeroBackend('free_wake',wake=WakeConfig(dt_s=.02,max_rows=2,
            solve=SolveSettings(solver='newton',max_iterations=60)))
        state=backend.start(wing,polar,inflow(20.,4.))
        before=state.solution.circulation.copy()
        first=state
        for _ in range(2):
            state=backend.advance(state,polar,inflow(20.,4.))
            self.assertLessEqual(state.solution.residual,state.solution.residual_limit)
            assert_allclose(state.free_rows[:,36],state.free_rows[:,47],atol=0,rtol=0)
        assert_allclose(first.solution.circulation,before,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
