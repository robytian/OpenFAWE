"""Independent formulas, strict reference identity and batch error contracts."""
import unittest
from unittest.mock import patch
import numpy as np
from openfawe.aero import vortex as v


def reference(*args, **kwargs):
    with patch.object(v, '_sum_batched', v._sum_reference):
        return v.segments_velocity(*args, **kwargs)


def bits(a, b):
    np.testing.assert_array_equal(np.asarray(a, dtype=np.float64).view(np.uint64),
                                  np.asarray(b, dtype=np.float64).view(np.uint64))


class VortexBatchTests(unittest.TestCase):
    def test_all_models_blocks_chunks_mixed_inputs_bitwise(self):
        rng = np.random.default_rng(20260927)
        points, nodes = rng.normal(size=(35,3)), rng.normal(size=(134,3))
        edges = np.arange(134).reshape(-1,2)
        gamma, radius = rng.normal(size=67), rng.uniform(.01,.8,67)
        gamma[::5] = 0.; radius[::4] = 0.
        exclusions = np.array([[0,j] for j in range(67)]+[[3,31],[3,31],[7,32],[8,63]])
        original = [a.copy() for a in (points,nodes,edges,gamma,radius,exclusions)]
        for a in (points,nodes,edges,gamma,radius,exclusions): a.flags.writeable=False
        for model in v.CoreModel:
            expected = reference(points,nodes,edges,gamma,model=model,core_radius=radius,exclusions=exclusions)
            for chunk in (1,4,4096):
                for singularity in ('raise','zero'):
                    with self.subTest(model=model,chunk=chunk,singularity=singularity):
                        actual = v.segments_velocity(points,nodes,edges,gamma,model=model,
                            core_radius=radius,exclusions=exclusions,chunk_size=chunk,singularity=singularity)
                        bits(actual,expected)
                        self.assertTrue(actual.flags.owndata)
                        bits(actual[0],np.zeros(3))
        for actual,saved in zip((points,nodes,edges,gamma,radius,exclusions),original):
            np.testing.assert_array_equal(actual,saved)

    def test_independent_formula_in_65_segment_network(self):
        for position in (0,31,32,63):
            starts=np.tile([0.,0.,3.],(65,1));ends=starts+[0,0,1.]
            starts[position]=[0.,0.,-1.];ends[position]=[0.,0.,1.]
            excluded_segment=(position+1)%65
            starts[excluded_segment]=[1.,0.,-1.];ends[excluded_segment]=[1.,0.,1.]
            nodes=np.stack((starts,ends),axis=1).reshape(-1,3)
            gamma=np.zeros(65);gamma[position]=1.;gamma[excluded_segment]=1.
            args=([[1.,0.,0.]],nodes,np.arange(130).reshape(-1,2),gamma)
            with patch.object(v,'_evaluate_pairs',wraps=v._evaluate_pairs) as called:
                actual=v.segments_velocity(*args,exclusions=[[0,excluded_segment]])
                self.assertEqual(called.call_count,3)
            np.testing.assert_allclose(actual,[[0,1/(2*np.pi*np.sqrt(2)),0]],rtol=1e-12,atol=1e-14)
            bits(actual,reference(*args,exclusions=[[0,excluded_segment]]))

    def test_scalar_formula_and_quadrature_reach_batch(self):
        a,b,p=np.array([-.7,.3,-.8]),np.array([1.2,-.4,1.]),np.array([.5,1.2,.4])
        x,w=np.polynomial.legendre.leggauss(96)
        delta=b-a;r=p-(a+((x+1)/2)[:,None]*delta)
        expected=1.7/(4*np.pi)*np.sum((w/2)[:,None]*np.cross(delta,r)/np.linalg.norm(r,axis=1)[:,None]**3,axis=0)
        with patch.object(v,'_evaluate_pairs',wraps=v._evaluate_pairs) as called:
            actual=v.segment_velocity(a,b,p,1.7)
            self.assertGreater(called.call_count,0)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)

    def test_zero_strength_and_inactive_pairs_do_not_evaluate_core(self):
        nodes=np.tile([[0.,0.,-1.],[0.,0.,1.]],(65,1));edges=np.arange(130).reshape(-1,2)
        for model in (v.CoreModel.RANKINE,v.CoreModel.LAMB_OSEEN,v.CoreModel.VATISTAS):
            args=([[1.,0.,0.],[0.,0.,0.]],nodes,edges,0.)
            actual=v.segments_velocity(*args,model=model,core_radius=1e-310,singularity='zero')
            bits(actual,np.zeros((2,3)))
            bits(actual,reference(*args,model=model,core_radius=1e-310,singularity='zero'))
            with self.assertRaises(v.FilamentSingularity):
                v.segments_velocity(*args,model=model,core_radius=1e-310)
            with self.assertRaises(ValueError):
                v.segments_velocity([[1.,0.,0.]],nodes,edges,1.,model=model,core_radius=1e-310)

    def test_error_precedence_with_mixed_failures(self):
        starts=np.tile([0.,0.,-1.],(65,1));ends=starts+[0.,0.,2.]
        starts[1]=[1.,0.,-1.];ends[1]=[1.,0.,1.]
        for order in (np.arange(65),np.r_[1,0,np.arange(2,65)]):
            nodes=np.stack((starts[order],ends[order]),axis=1).reshape(-1,3)
            for chunk in (1,4096):
                args=([[1.,0.,0.],[2.,0.,0.]],nodes,np.arange(130).reshape(-1,2),1.)
                kwargs=dict(model=v.CoreModel.RANKINE,core_radius=1e-310,chunk_size=chunk)
                caught=[]
                for function in (reference,v.segments_velocity):
                    try:function(*args,**kwargs)
                    except ValueError as error:caught.append((type(error),str(error)))
                    else:self.fail('Mixed singular/overflow case succeeded')
                self.assertEqual(caught[0],caught[1])

    def test_batch_failure_cannot_silently_fall_back_to_success(self):
        with patch.object(v,'_evaluate_pairs',side_effect=v.FilamentSingularity('injected')):
            with self.assertRaisesRegex(RuntimeError,'reference succeeded'):
                v.segment_velocity([0,0,-1],[0,0,1],[1,0,0],1.)

    def test_excluded_overflow_geometry_is_never_evaluated(self):
        # Segment normalization is finite, while the excluded subtraction is not.
        nodes=np.array([[-1e308,0,0],[-1e308,1,0],[0,0,-1],[0,0,1]])
        args=([[1e308,0,0],[1,0,0]],nodes,[[0,1],[2,3]],[0.,1.])
        exclusions=[[0,0],[0,1],[1,0]]
        bits(v.segments_velocity(*args,exclusions=exclusions),reference(*args,exclusions=exclusions))


if __name__=='__main__':unittest.main()
