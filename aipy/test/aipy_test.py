import numpy as np
import unittest
from ..core import Gradient_Descent as GD
from ..core import *

class GD_Test(unittest.TestCase):
    def test_sigmoid(self):
        X = np.random.rand(4,3)
        Y = np.random.rand(4,1)
        W = np.random.rand(3,1)
        gd=GD(X, Y, W)
        self.assertEqual(gd.sigmoid(0), 0.5)
        
    def test_gradientDescent(self):
        np.random.seed(1)
        X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
        Y = (np.random.rand(10, 1) > 0.35).astype(float)
        W = np.zeros((3, 1))
        alpha = 1e-8
        gd=GD(X, Y, W, learning_rate=alpha)
        J, W = gd.gradientDescent(700)
        W=W.squeeze()
        self.assertAlmostEqual(J, 0.63154869)
        self.assertAlmostEqual(W[0], 2.53e-06)
        self.assertAlmostEqual(W[1], 0.00173294)
        self.assertAlmostEqual(W[2], -0.00053311)
        
    def test_initialize_with_zeros(self):
        dim=3
        w, b = initialize_with_zeros(dim)
        assert(w.shape[0]==dim and b.shape[0]==dim)

    
if __name__ == '__main__':
    unittest.main()
