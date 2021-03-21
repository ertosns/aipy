import unittest
import numpy as np
from neural_network import *

class Utils_Test(unittest.TestCase):
        
    def test_deep_initialization(self):
        parameters = deep_initialization([5,4,3])
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        eW1 = np.array([[ 0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388], [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218], [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034], [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
        eb1 = np.array([[ 0.], [ 0.], [ 0.], [ 0.]])
        eW2 = np.array([[-0.01185047, -0.0020565, 0.01486148, 0.00236716], [-0.01023785, -0.00712993, 0.00625245, -0.00160513], [-0.00768836, -0.00230031, 0.00745056, 0.01976111]])
        eb2 = np.array([[ 0.], [ 0.], [ 0.]])

        self.assertTrue(np.allclose(W1, eW1))
        self.assertTrue(np.allclose(b1, eb1))
        self.assertTrue(np.allclose(W2, eW2))
        self.assertTrue(np.allclose(b2, eb2))
        
    def test_linear_forward(self):
        np.random.seed(1)
        A = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        Z, linear_cache = linear_forward(A, W, b)
        self.assertTrue(np.allclose(Z, np.array([[3.26295337, -1.23429987]])))

    def test_linear_activation_forward(self):
        np.random.seed(2)
        A_prev = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        A, linear_activation_cache = deep_linear_activation_forward(A_prev, W, b, activation = "sigmoid")
        self.assertTrue(np.allclose(A, np.array([[0.96890023, 0.11013289]])))
        A, linear_activation_cache = deep_linear_activation_forward(A_prev, W, b, activation = "relu")
        self.assertTrue(np.allclose(A, np.array([[3.43896131, 0.]])))

    def test_L_model_forward(self):
        np.random.seed(6)
        X = np.random.randn(5,4)
        W1 = np.random.randn(4,5)
        b1 = np.random.randn(4,1)
        W2 = np.random.randn(3,4)
        b2 = np.random.randn(3,1)
        W3 = np.random.randn(1,3)
        b3 = np.random.randn(1,1)
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        
        AL, caches = L_model_forward(X, parameters)
        self.assertEqual(len(caches), 3)
        self.assertTrue(np.allclose(AL, np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])))
    
    def test_linear_backward(self):
        np.random.seed(1)
        dZ = np.random.randn(3,4)
        A = np.random.randn(5,4)
        W = np.random.randn(3,5)
        b = np.random.randn(3,1)
        linear_cache = (A, W, b)
        dA_prev, dW, db = deep_linear_backward(dZ, linear_cache)
        dA_prev_e = np.array(
            [[-1.15171336,  0.06718465, -0.3204696,   2.09812712],
             [ 0.60345879, -3.72508701,  5.81700741, -3.84326836],
             [-0.4319552,  -1.30987417,  1.72354705,  0.05070578],
             [-0.38981415,  0.60811244, -1.25938424,  1.47191593],
             [-2.52214926,  2.67882552, -0.67947465,  1.48119548]])

        dW_e = np.array([
            [0.07313866,-0.0976715, -0.87585828, 0.73763362, 0.00785716],
            [ 0.85508818,  0.37530413, -0.59912655,  0.71278189, -0.58931808],
            [ 0.97913304, -0.24376494, -0.08839671,  0.55151192, -0.10290907]])
        db_e = np.array([[-0.14713786], [-0.11313155],[-0.13209101]])
        self.assertTrue(np.allclose(dA_prev, dA_prev_e))
        self.assertTrue(np.allclose(dW, dW_e))
        self.assertTrue(np.allclose(db, db_e))
        
    def test_linear_activation_backward(self):
        np.random.seed(2)
        dA = np.random.randn(1,2)
        A = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        Z = np.random.randn(1,2)
        linear_cache = (A, W, b)
        activation_cache = Z
        linear_activation_cache = (linear_cache, activation_cache)
        ###
        dA_prev, dW, db = deep_linear_activation_backward(dA, linear_activation_cache, activation = "sigmoid")
        dA_prev_e=np.array([[ 0.11017994, 0.01105339],
                            [ 0.09466817, 0.00949723],
                            [-0.05743092, -0.00576154]]).reshape((3,2))
        dW_e=np.array( 	[[ 0.10266786, 0.09778551, -0.01968084]])
        db_e=np.array( 	[[-0.05729622]])
        self.assertTrue(np.allclose(dA_prev, dA_prev_e))
        self.assertTrue(np.allclose(dW, dW_e))
        self.assertTrue(np.allclose(db, db_e))
        ###
        dA_prev, dW, db = deep_linear_activation_backward(dA, linear_activation_cache, activation = "relu")
        dA_prev_e=np.array([[0.44090989, 0.],
                            [0.37883606, 0.],
                            [-0.2298228, 0.]])
        dW_e=np.array([[ 0.44513824, 0.37371418, -0.10478989]])
        db_e=np.array([[-0.20837892]])
        self.assertTrue(np.allclose(dA_prev, dA_prev_e))
        self.assertTrue(np.allclose(dW, dW_e))
        self.assertTrue(np.allclose(db, db_e))
        
    def test_L_model_backward(self):
        np.random.seed(3)
        AL = np.random.randn(1, 2)
        Y_assess = np.array([[1, 0]])
        #
        A1 = np.random.randn(4,2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        Z1 = np.random.randn(3,2)
        linear_cache_activation_1 = ((A1, W1, b1), Z1)
        #
        A2 = np.random.randn(3,2)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        Z2 = np.random.randn(1,2)
        linear_cache_activation_2 = ((A2, W2, b2), Z2)
        #
        caches = (linear_cache_activation_1, linear_cache_activation_2)
        grads = L_model_backward(AL, Y_assess, caches)
        dW1=grads['dW1']
        db1=grads['db1']
        dA1=grads['dA1']
        dW1_e= np.array([[ 0.41010002, 0.07807203, 0.13798444, 0.10502167],
                         [ 0., 0., 0., 0.],
                         [ 0.05283652, 0.01005865, 0.01777766, 0.0135308 ]])
        db1_e=np.array([[-0.22007063],
                        [ 0. ],
                        [-0.02835349]])
        dA1_e=np.array([[ 0.12913162, -0.44014127],
                        [-0.14175655, 0.48317296],
                        [0.01663708, -0.05670698]])

        self.assertTrue(np.allclose(dW1, dW1_e))
        self.assertTrue(np.allclose(db1, db1_e))
        self.assertTrue(np.allclose(dA1, dA1_e))
        
if __name__ == '__main__':
    unittest.main()
