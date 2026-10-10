"""Whole-state regression of six backends against the frozen ordered kernel."""
from dataclasses import fields, is_dataclass
from enum import Enum
import unittest
from unittest.mock import patch
import numpy as np
from openfawe.aero import vortex
from openfawe.aero.methods import AeroBackend, AeroMethod
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings
from openfawe.aero.lifting_surface import LiftingSurfaceConfig


def assert_bits(a, b, path='state'):
    if is_dataclass(a):
        for f in fields(a): assert_bits(getattr(a,f.name),getattr(b,f.name),path+'.'+f.name)
    elif isinstance(a,np.ndarray):
        if a.dtype.kind=='f':
            np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64),err_msg=path)
        else: np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,float):
        np.testing.assert_array_equal(np.float64(a).view(np.uint64),np.float64(b).view(np.uint64),err_msg=path)
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b): raise AssertionError(path)
        for i,(x,y) in enumerate(zip(a,b)): assert_bits(x,y,f'{path}[{i}]')
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError(path)
        for k in a: assert_bits(a[k],b[k],path+'.'+k)
    elif a!=b: raise AssertionError(path)


class BackendBatchTests(unittest.TestCase):
    def test_six_methods_twenty_steps_full_state_bitwise(self):
        polar=SectionPolar.linear()
        solve=SolveSettings(solver='newton',max_iterations=60)
        for method in AeroMethod:
            released=method in (AeroMethod.FREE_WAKE,AeroMethod.VSM_FREE_WAKE,AeroMethod.VLM_FREE_WAKE)
            config=WakeConfig(dt_s=.02,core_radius_m=.08,max_rows=24,solve=solve,
                core_evolution='age_stretch' if released else 'constant',convection_integrator='heun')
            backend=AeroBackend(method,wake=config,vsm_solve=solve,
                lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
            histories=[]
            for kernel in (vortex._sum_reference,vortex._sum_batched):
                state=None;history=[]
                with patch.object(vortex,'_sum_batched',kernel):
                    for k in range(21):
                        wing=Wing.ellipse(4)
                        shift=np.array([0.,0.,.01*np.sin(k*.02)])
                        wing=Wing(wing.leading_edge+shift,wing.trailing_edge+shift)
                        kwargs={'structural_velocity':[0.,0.,.01*np.cos(k*.02)]}
                        state=(backend.start(wing,polar,[5.,0.,.25],**kwargs) if k==0 else
                            backend.advance(state,polar,[5.,0.,.25],next_wing=wing,next_time_s=k*.02,**kwargs))
                        history.append((state,backend.induced_velocity(state,[[2.,.7,.8],[3.,-.5,-.8]])))
                histories.append(history)
            with self.subTest(method=method): assert_bits(*histories)


if __name__=='__main__': unittest.main()
