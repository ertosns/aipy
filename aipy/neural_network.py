import numpy as np

## Randomly initialize weight, and bias parameters of given layer dimensions.
#
# @param layer_dims python array (list) containing the dimensions of each layer in our network
# @return parameters python dictionary of parameters 
#    "W1", "b1", ..., "WL", "bL": 
#    Wl weight matrix of shape(layer_dims[l], layer_dims[l-1])
#    bl bias vector of shape (layer_dims[l], 1)
def deep_initialization(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = \
            np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = \
            np.zeros((layer_dims[l], 1))*0.01
    return parameters


## Implements the sigmoid activation in numpy
#
# @param Z numpy array of any shape
# @return A output of sigmoid(z), same shape as Z
# @return cache returns Z as well, useful during backpropagation
def deep_sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

## Implement the cost function.
#
# @param AL (1, m) probability vector corresponding to the label predictions.
# @param Y 0,1 "label" vector.
# @return cost cross-entropy cost
def deep_compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1.0/m * np.sum(np.multiply(Y, np.log(AL)) + \
                           np.multiply((1-Y), np.log(1-AL)))    
    cost = cost.squeeze()
    assert(cost.shape == ())
    
    return cost


## Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
#
# @param w weights, a numpy array of size (n, 1)
# @param b bias, a scalar
# @param X data of size (n,m)
# @return Y_prediction a numpy array (vector) containing all predictions (0/1) for the examples in X
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    Z = linear(w, X, b)
    A = sigmoid(Z)
    #TODO replace this with pandas apply function
    #RELU-like activation
    for i in range(A.shape[1]):
        Y_prediction[0][i] = 1 if A[0][i]>0.5 else 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


## Implement the RELU function.
#
# @param Z Output of the linear layer, of any shape
# @return A Post-activation parameter, of the same shape as Z
# @return cache a python dictionary containing "A" ; stored for computing the backward pass efficiently
def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

## Implement the backward propagation for a single RELU unit.
#
# @param dA post-activation gradient, of any shape
# @param cache 'Z' where we store for computing backward propagation efficiently
# @return dZ Gradient of the cost with respect to Z
def relu_backward(dA, cache):   
    Z = cache

    dZ = np.array(dA, copy=True)
    assert (dZ.shape == Z.shape)

    dZ[Z <= 0] = 0    
    return dZ

## Implement the backward propagation for a single SIGMOID unit.
# 
# @param dA post-activation gradient, of any shape
# @param cache 'Z' where we store for computing backward propagation efficiently
# @return dZ Gradient of the cost with respect to Z
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ


## Implement the linear part of a layer's forward propagation.
#
# @param A activations from previous layer (or input data): (size of previous layer, number of examples)
# @param W weights matrix: numpy array of shape (size of current layer, size of previous layer)
# @param b bias vector, numpy array of shape (size of the current layer, 1)
# @return Z the input of the activation function, also called pre-activation parameter 
# @return cache a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

## Tan activation function
#
#    @param z is the linear activation
#    @return tan(z) 
def tanh_activation(z):
    return np.tanh(z)

## Implement the forward propagation for the LINEAR->ACTIVATION layer
#
# @param A_prev activations from previous layer (or input data): (size of previous layer, number of examples)
# @param W  weights matrix: numpy array of shape (size of current layer, size of previous layer)
# @param b bias vector, numpy array of shape (size of the current layer, 1)
# @param activation the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
# @return A the output of the activation function, also called the post-activation value 
# @return cache a python tuple containing "linear_cache" and "activation_cache", stored for computing the backward pass efficiently
def deep_linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = deep_sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


## Implement the linear portion of backward propagation for a single layer (layer l).
#
# @param dZ Gradient of the cost with respect to the linear output (of current layer l)
# @param cache tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
# @return dA_prev Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev, from \f$\frac{\partial{L}}{\partial{z}}=\frac{\partial{L}{\partial{A}}\frac{\partial{A}}{\partial{z}}\f$  then \f$A_{i-1}=W_{i}^T\sigma{z_i}\f$ 
# @return dW Gradient of the cost with respect to W (current layer l), same shape as W
# @return db Gradient of the cost with respect to b (current layer l), same shape as b
def deep_linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1.0/m *np.dot(dZ, A_prev.T)
    db = 1.0/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

## Implement the backward propagation for the LINEAR->ACTIVATION layer.
#    
# @param dA post-activation gradient for current layer l 
# @param cache tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
# @param activation the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
# @return dA_prev Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev.
# @return dW Gradient of the cost with respect to W (current layer l), same shape as W.
# @return db Gradient of the cost with respect to b (current layer l), same shape as b.
def deep_linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    assert(dA.shape == activation_cache.shape)
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = deep_linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = deep_linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

## Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
#
# @param X data, numpy array of shape (input size, number of examples).
# @param parameters output of initialize_parameters_deep().
# @return AL last post-activation value.
# @return caches list of caches containing: every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1).
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A, cache = deep_linear_activation_forward(A, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    AL, cache = deep_linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

## Update parameters using gradient descent.
#
# @param parameters dictionary containing parameters 
# @param grads dictionary of gradients.
# @return updated parameters.
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        w=parameters["W" + str(l+1)]
        b=parameters["b" + str(l+1)]
        dw=learning_rate*grads['dW'+str(l+1)]
        db=learning_rate*grads['db'+str(l+1)]
        parameters["W" + str(l+1)] = w - learning_rate * dw
        parameters["b" + str(l+1)] = b - learning_rate * db
    return parameters

## Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group.
#
# @param AL probability vector, output of the forward propagation.
# @param Y true "label" vector (containing 0 if non-cat, 1 if cat)
# @param caches list of caches containing: every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2) the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
# @return grads  A dictionary with the gradients dAi, dWi, dbi for the (i)th layer.
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    # Initializing the backpropagation
    dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #dA_prev =  Y - AL / np.dot(1-AL, AL.T) * 4.734719705992509
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = deep_linear_activation_backward(dA_prev, current_cache, 'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = deep_linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads
