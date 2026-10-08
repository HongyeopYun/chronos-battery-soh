import unittest
import numpy as np
from src.c2_interval_calibration import fit_scale,apply_scale


class IntervalCalibrationTests(unittest.TestCase):
    def test_empirical_width_fit_preserves_median_and_asymmetry(self):
        base=np.broadcast_to(np.array([.5,.65,.8,.9,1.,1.2,1.4,1.6,1.8]),(10,16,9)).copy()
        median=np.full((10,16),.96)
        y=median+np.linspace(-1.2,1.5,160).reshape(10,16)
        scale=fit_scale(y,median,base)
        self.assertGreaterEqual(scale,1.)
        q=apply_scale(median,base,scale)
        np.testing.assert_array_equal(q[...,4],median)
        self.assertGreaterEqual(np.mean((y>=q[...,0])&(y<=q[...,-1])),.8)
        np.testing.assert_allclose((q[...,-1]-median)/(median-q[...,0]),1.6)
        # Prediction uses a saved scale and foundation forecast; no target is passed.
        independent=apply_scale(median[:2],base[:2],scale)
        np.testing.assert_array_equal(independent,q[:2])


if __name__=="__main__":
    unittest.main()
