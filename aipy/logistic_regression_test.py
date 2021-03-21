import unittest
import numpy as np
from logistic_regression import *

class LR_Test(unittest.TestCase):
    def test_sigmoid(self):
        z = np.array([0,2])
        a = sigmoid(z)
        a_e = np.array([0.5,0.88079708])
        self.assertTrue(np.allclose(a, a_e))
        
    def test_initialize_with_zeros(self):
        dim = 2
        w, b = initialize_with_zeros(dim)
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        self.assertTrue(np.allclose(w, np.array([[0],[0]])))
        self.assertEqual(b, 0)

    def test_initialization(self):
        np.random.seed(1)
        parameters = initialize_parameters(2, 4, 1)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        
        eW1 = np.array([[-0.00416758, -0.00056267],
                        [-0.02136196,  0.01640271],
                        [-0.01793436, -0.00841747],
                        [ 0.00502881, -0.01245288]])
        eb1 = np.array([[ 0.],
                        [ 0.],
                        [ 0.],
                        [ 0.]])
        eW2 = np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]])
        eb2 = np.array([[ 0.]])
        self.assertTrue(np.allclose(W1, eW1))
        self.assertTrue(np.allclose(b1, eb1))
        self.assertTrue(np.allclose(W2, eW2))
        self.assertTrue(np.allclose(b2, eb2))

if __name__ == '__main__':
    unittest.main()
