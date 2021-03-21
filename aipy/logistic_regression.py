import numpy as np

#TODO need more test cases.

## creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
#
# @param dim size of the w vector we want (or number of parameters in this case)
# @return w initialized vector of shape (dim, 1)
# @return b initialized scalar (corresponds to the bias)
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

## Randomly initialize the parameters W, b.
#
# @param n_x size of the input layer
# @param n_h size of the hidden layer
# @param n_y size of the output layer
# @return params dictionary of parameters:
#                    W1 -- weight matrix of shape (n_h, n_x)
#                    b1 -- bias vector of shape (n_h, 1)
#                    W2 -- weight matrix of shape (n_y, n_h)
#                    b2 -- bias vector of shape (n_y, 1)
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2) 
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

## linear activation function
#
# @param w (m,1) weight matrix
# @param X (m,n) input matrix
# @param b (1) bias 
# @return z (1,m) linear estimation
def linear(w, X, b):
    z = np.dot(w.T, X) + b
    return z

#** //things are starting to get out of control, even the size is starting to differ! i need more test cases.
## linear activation function
#
# @param W (n_i,n_{i-1}) weight matrix
# @param A (n_{i-1}, m) input matrix
# @param b (1) bias 
# @return z (1,m) linear estimation
#def linear(W, A, b):
#    z = np.dot(W, A) + b
#    return z

## sigmoid activation
#
# @param z is the input (can be a scalar or an array)
# @return h the sigmoid of z
def sigmoid(z):
    return 1/(1+np.exp(-1*z))

## compute log likelihood of the logistic regression
#
# @param Y (1,m) labeled vector Output
# @param h (1,m) estimated output
# @return J (1) scalar cost function output
def compute_cost(Y, h):
    m=Y.shape[1]
    def compute_loss():
        L=np.dot(Y.T, np.log(h)) + np.dot((1-Y).T, np.log((1-h)))
        return L.squeeze()
    J = -1.0/m * compute_loss()
    return J

## Update weight values with single gradient descent step.
#
# @param w (n,1) weight vector .
# @param dw (n,1) weight vector of dJ/dw.
# @param b scalar bias.
# @param db scalar dJ/db.
# @param alpha scalar is the learning reate.
# @return tuple of new (w,b).
def update_weight(w, dw, b, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b
 
## Tan activation function
#
#    @param z is the linear activation
#    @return tan(z) 
def tanh_activation(z):
    return np.tanh(z)


#TODO generalize this, or make a separate file for 2layer propagation.
## Implement Forward propagation.
#
# @param X input data of size (n_x, m)
# @param parameters initialization return value
# @return A2 The sigmoid output of the second activation
# @return cache a dictionary containing "Z1", "A1", "Z2" and "A2"
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = linear(W1, X, b1)
    A1 = tanh_activation(Z1)
    Z2 = linear(W2, A1, b2)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

## predicts a class for each example in X.
#
# @param parameters
# @param X (n_x, m) input data.
# @return predictions vector of predictions.
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = [1 if e > 0.5 else 0 for e in np.squeeze(A2)]
    
    return np.array(predictions)

## Implement Forward, and Backward propagation, the cost function and its gradient .
#
# @param w weights, a numpy array of size (n,1)
# @param b bias, a scalar
# @param X data of size (n,m)
# @param Y true "label" vector {0,1} (1,m)
# @return cost negative log-likelihood cost for logistic regression
# @return dw gradient of the loss with respect to w, thus same shape as w
# @return db gradient of the loss with respect to b, thus same shape as b
def single_iteration_propagation(w, b, X, Y):
    m = X.shape[1]
    z = linear(w, X, b)
    h = sigmoid(z)
    cost = compute_cost(Y, h)
    dw = 1.0/m * (np.dot(X, (h-Y).T))
    db = 1.0/m * np.sum(h-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost


## This function optimizes w and b by running a gradient descent algorithm
#     
# @param w weights, a numpy array of size (n,1)
# @param b bias, a scalar
# @param X data of shape (n,m)
# @param Y true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
# @param num_iterations number of iterations of the optimization loop
# @param learning_rate learning rate of the gradient descent update rule
# @param print_cost True to print the loss every 100 steps
# @return params dictionary containing the weights w and bias b
# @return grads dictionary containing the gradients of the weights and bias with respect to the cost function
# @return costs list of all the costs computed during the optimization, this will be used to plot the learning curve.
def gradient_descent(w, b, X, Y, num_iterations=1000, alpha=0.001, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w,b = update_weight(w, dw, b, db, alpha)
        # track cost
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
