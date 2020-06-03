
# coding: utf-8

# # Gradient Checking
# 
# Goal: Learn to implement and use gradient checking. 

# Backpropagation is challenging to implement, and sometimes has bugs. Want to be very certain that implementation of backpropagation is correct.
# To give this reassurance, going to use "gradient checking".
#

# Import Packages
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector # Gradient checking package


### 1) How does gradient checking work?
# 
# Backpropagation computes the gradients $\frac{\partial J}{\partial \theta}$, where $\theta$ denotes the parameters of the model. $J$ is computed using forward propagation and loss function.
# Can use the code for computing $J$ to verify the code for computing $\frac{\partial J}{\partial \theta}$. 
# 
# Look at the definition of a derivative (or gradient):
# $$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} \tag{1}$$
#
### 2) 1-dimensional gradient checking

#

def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    # Make sure theta is computed correctly with 1D linear function
    J = np.dot(x, theta) # x is the input and theta is a real-valued parameter

    return J


# Evaluate the cost function J(x), implementing forward propagation
x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))


# **Expected Output**:
# 
# <table style=>
#     <tr>
#         <td>  ** J **  </td>
#         <td> 8</td>
#     </tr>
# </table>

#

### backward_propagation ###

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    # Identifying the derivative of J (dtheta) with respect to x
    dtheta = x
    
    return dtheta


# Implement the backpropagation step with dtheta
x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>  ** dtheta **  </td>
#         <td> 2 </td>
#     </tr>
# </table>

# 

#

### gradient_check ###

def gradient_check(x, theta, epsilon = 1e-7):
    """
    Implement the backward propagation presented in Figure 1.
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # Implement gradient checking
    # Compute gradapprox using left side of formula (1). epsilon is small enough, don't need to worry about the limit.
    thetaplus = theta + epsilon                              
    thetaminus = theta - epsilon                             
    J_plus = forward_propagation(x, thetaplus)                                 
    J_minus = forward_propagation(x, thetaminus)                                 
    gradapprox = (J_plus - J_minus) / (2 * epsilon)    # Similar to derivative to surely output the correct result                        
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)                               # compute the numerator using np.linag.norm(...) 
    denominator =  np.linalg.norm(grad) + np.linalg.norm(gradapprox)            # compute the denominator(need to call np.linag.norm(...) twice)
    difference = numerator / denominator                                        # divide both

    # Loop looking at if gradients is small,
    if difference < 1e-7:                              # if small, then gradients is correct
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")               # otherwise, may be a mistake in gradients computation
    
    return difference


#

x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))


# **Expected Output**:
# The gradient is correct!
# <table>
#     <tr>
#         <td>  ** difference **  </td>
#         <td> 2.9193358103083e-10 </td>
#     </tr>
# </table>

# The difference is smaller, so can have high confidence that the the gradient is correctly computed in `backward_propagation()`. 
# 

### 3) N-dimensional gradient checking 

#

def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    
    Returns:
    cost -- the cost function (logistic cost for one example)
    """
    # Run forward propagation
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # Deep Neural Network: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y) # compute logprobabilities to avoid underflowing floats when calculating likelihood function
    cost = 1./m * np.sum(logprobs) # comput cost summation of logprobs
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache


# Now, run backward propagation.

#

def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()
    
    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    # Retrieve parameter, activation, input information from output
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    # Implement backpropagation
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) * 2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

#

# How does gradient checking work?
#  $\theta$ is not a scalar anymore. It is a dictionary called "parameters". Now implemented a function "`dictionary_to_vector()`".
# obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them.
# 
# Instructions: Here is pseudo-code that will help implement the gradient check.
# 
# $$ difference = \frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 } \tag{3}$$

#

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # How to help implement gradient check.
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters) # converts the "parameters" dictionary into a vector called "values"
    grad = gradients_to_vector(gradients)                   # convert gradients dictionary into a vector, "grads"
    num_parameters = parameters_values.shape[0]             # get current shape of an array by assigning a tuple of array dimensions
    J_plus = np.zeros((num_parameters, 1))                  # initialize J_plus with zeros and number of parameter objects
    J_minus = np.zeros((num_parameters, 1))                 # initialize J_minus with zeros and number of parameter objects
    gradapprox = np.zeros((num_parameters, 1))              # initialize gradapprox with zeros and number of parameter objects
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function outputs two parameters but only care about the first one
        thetaplus = np.copy(parameters_values)                                        # Set theta to np.copy
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                   # Set theta_plus
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))   # Calculate J_plus using forward propagation
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                       # Set theta to np.copy
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                 # Set theta_minus     
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus)) # Calculate J_minus using forward propagation
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                                     # compute the numerator using np.linag.norm(...)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   # compute the denominator(need to call np.linag.norm(...) twice)
    difference = numerator / denominator                                              # divide both

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference


# Implement gradient_check_n().
X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)


# **Expected output**:
# 
# <table>
#     <tr>
#         <td>  ** There is a mistake in the backward propagation!**  </td>
#         <td> difference = 0.285093156781 </td>
#     </tr>
# </table>

# It seems that there were errors in the `backward_propagation_n` code give! Good that gradient check implemented.
# Go back to `backward_propagation` and try to find/correct the errors *(Hint: check dW2 and db1)*. Rerun the gradient check when fixed.
# Remember: need to re-execute the cell defining `backward_propagation_n()` if modifying the code. 
# 
# Strongly urge to try to find the bug and re-run gradient check until convinced backprop is now correctly implemented. 
# 
# Note: 
# - Gradient Checking is slow! Approximating the gradient with $\frac{\partial J}{\partial \theta} \approx  \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$ is computationally costly. For this reason, we don't run gradient checking at every iteration during training. Just a few times to check if the gradient is correct. 
# - Gradient Checking, at least presented, doesn't work with dropout. Would usually run the gradient check algorithm without dropout to make sure backprop is correct, then add dropout. 
# 
# Now can be confident that deep learning model for fraud detection is working correctly! :)
#
# Summary:
# - Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
# - Gradient checking is slow, so don't run it in every iteration of training--usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process. 

#



