
# coding: utf-8

### TensorFlow Tutorial
# 
# Goal:
# Go through a deep learning framework to build a neural networks more easily.
# Machine learning frameworks: TensorFlow, PaddlePaddle, Torch, Caffe, Keras, and many others can speed up machine learning development significantly.
# In this respository, will learn to do the following in TensorFlow: 
# 
# - Initialize variables
# - Start own session
# - Train algorithms 
# - Implement a Neural Network
# 
# Programing frameworks can not only shorten coding time, but sometimes also perform optimizations that speed up the code. 
#
### 1) Exploring the Tensorflow Library
# 
# Import packages needed

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf                 # Used for high performance numerical computation easily deployed on various platforms, clusers, servers and devices
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

get_ipython().magic('matplotlib inline')
np.random.seed(1)


# Now that the library is imported, start with an example to compute the loss of one training example. 
# $$loss = \mathcal{L}(\hat{y}, y) = (\hat y^{(i)} - y^{(i)})^2 \tag{1}$$
#

y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss


# Writing and running programs in TensorFlow has the following steps:
# 
# 1. Create Tensors (variables) that are not yet executed/evaluated. 
# 2. Write operations between those Tensors.
# 3. Initialize Tensors. 
# 4. Create a Session. 
# 5. Run the Session. This will run the operations written above. 
# 
# Therefore, when creating a variable for the loss, it defines the loss as a function of other quantities, but did not evaluate its value.
# To evaluate it, had to run `init=tf.global_variables_initializer()`. That initialized the loss variable, and in the last line, finally able to evaluate the value of `loss` and print its value.
# 
# Now, run the cell below:

# 

a = tf.constant(2) # Lines 1-3: put in the 'computation graph', have not run this computation yet.
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)


# As expected, got a tensor saying that the result is a tensor that does not have the shape attribute, and is of type "int32".
# In order to actually multiply the two numbers, will: 

# Create a session and run it.
sess = tf.Session()
print(sess.run(c))


# To initialize variables, create a session and run the operations inside the session. 

# Change the value of x in the feed_dict
x = tf.placeholder(tf.int64, name = 'x')   # Specify values of placeholders for x
print(sess.run(2 * x, feed_dict = {x: 3})) # Pass values using a "feed dictionary"
sess.close()


# When first defined `x`, did not have to specify a value for it. A placeholder is simply a variable that will assign data to only later, when running the session.
#
### 1.1) Linear function
# 
# Compute the following equation: $Y = WX + b$, where $W$ and $X$ are random matrices and b is a random vector. 
# 
# X = tf.constant(np.random.randn(3,1), name = "X")
# 
# ```
# The following functions that help: 
# - tf.matmul(..., ...) to do a matrix multiplication
# - tf.add(..., ...) to do an addition
# - np.random.randn(...) to initialize randomly
# 

# 

### linear_function ###

def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
    
    """
    Note, to ensure that the "random" numbers generated match the expected results,
    create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # Compute WX + b, where W, X and b are drawn from random normal distribution
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add((tf.matmul(W, X)), b)
    
    # Create the session using tf.Session() and run it with sess.run(...) on the variable want to calculate
    sess = tf.Session()
    result = sess.run(Y)
   
    # close the session 
    sess.close()

    return result

# 

print( "result = \n" + str(linear_function()))


# *** Expected Output ***: 
# 
# ```
# result = 
# [[-2.15657382]
#  [ 2.95891446]
#  [-1.08926781]
#  [-0.84538042]]
# ```

### 1.2) Computing the sigmoid for an input
#
# Have just implemented a linear function. Tensorflow offers a variety of commonly used functions like `tf.sigmoid` and `tf.softmax`.
# 
# Use a placeholder variable `x`. When running the session, use the feed dictionary to pass in the input `z`. Will have to:
# (i) create a placeholder `x`,
# (ii) define the operations needed to compute the sigmoid using `tf.sigmoid`
# (iii) run the session. 
# 
# Implement the sigmoid function below using the following: 
# 
# - `tf.placeholder(tf.float32, name = "...")`
# - `tf.sigmoid(...)`
# - `sess.run(..., feed_dict = {x: z})`
# 


### sigmoid ###

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name = "X")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Use the method 2 explained above. 
    # Use a feed_dict to pass z's value to x. 
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {x: z})

    
    return result


# 

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))


# *** Expected Output ***: 
# 
# <table> 
# <tr> 
# <td>
# **sigmoid(0)**
# </td>
# <td>
# 0.5
# </td>
# </tr>
# <tr> 
# <td>
# **sigmoid(12)**
# </td>
# <td>
# 0.999994
# </td>
# </tr> 
# 
# </table> 
#

### 1.3)  Computing the Cost
# 
# Can also use a built-in function to compute the cost of the neural network. Instead of writing code, to compute this as a function of $a^{[2](i)}$ and $y^{(i)}$ for i=1...m: 
# $$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log a^{ [2] (i)} + (1-y^{(i)})\log (1-a^{ [2] (i)} )\large )\small\tag{2}$
# 

### cost ###

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
    # Create the placeholders for "logits" (z) and "labels" (y) 
    z = tf.placeholder(tf.float32, name = "z")
    y = tf.placeholder(tf.float32, name = "y")
    
    # Implement cross entropy loss function (should input z and compute the sigmoid)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    
    # Create a session 
    sess = tf.Session()
    
    # Run the session 
    cost = sess.run(cost, feed_dict = {z: logits, y: labels})
    
    # Close the session 
    sess.close()

        
    return cost


# 

logits = np.array([0.2,0.4,0.7,0.9])

cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))


# ** Expected Output** : 
# 
# ```
# cost = [ 0.79813886  0.91301525  0.40318605  0.34115386]
# ```

### 1.4) Using One Hot encodings
# 
# Implement the function below to take one vector of labels and the total number of classes $C$, and return the one hot encoding. Use `tf.one_hot()` to do this. 
# 

### one_hot_matrix ###

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    
    # Create a tf.constant equal to C (depth), name it 'C'.
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis 
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    
    # Create the session 
    sess = tf.Session()
    
    # Run the session 
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session 
    sess.close()
    
    
    return one_hot

#

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = \n" + str(one_hot))


# **Expected Output**: 
# 
# ```
# one_hot = 
# [[ 0.  0.  0.  1.  0.  0.]
#  [ 1.  0.  0.  0.  0.  1.]
#  [ 0.  1.  0.  0.  1.  0.]
#  [ 0.  0.  1.  0.  0.  0.]]
# ```

### 1.5) Initialize with zeros and ones
# 
# Now, learn how to initialize a vector of zeros and ones. Will call the function `tf.ones()`. To initialize with zeros, use tf.zeros() instead.
#

def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    # Create "ones" tensor using tf.ones(...) to take in a shape and to return an array (of the shape's dimensions of ones)
    ones = tf.ones(shape)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    return ones


#

print ("ones = " + str(ones([3])))


# **Expected Output:**
# 
# <table> 
#     <tr> 
#         <td>
#             **ones**
#         </td>
#         <td>
#         [ 1.  1.  1.]
#         </td>
#     </tr>
# 
# </table>

### 2) Building First Neural Network in tensorflow
# 
# There are two parts to implement a tensorflow model:
# 
# - Create the computation graph
# - Run the graph
# 
# Run the following code to load the dataset.
#
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 

# Change the index below and run the cell to visualize some examples in the dataset.
# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

#

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# Note: 12288 comes from $64 \times 64 \times 3$. Each image is square, 64 by 64 pixels, and 3 is for the RGB colors.
# Make sure all these shapes make sense to before continuing.
# 
# The Model: *LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX*.
# - The SIGMOID output layer has been converted to a SOFTMAX.
# - A SOFTMAX layer generalizes SIGMOID to when there are more than two classes. 

### 2.1) Create placeholders
# 

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    # Create placeholders for X and U to allow for a later pass for training date in when running the session.
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y


# 

X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


# **Expected Output**: 
# 
# <table> 
#     <tr> 
#         <td>
#             **X**
#         </td>
#         <td>
#         Tensor("Placeholder_1:0", shape=(12288, ?), dtype=float32) (not necessarily Placeholder_1)
#         </td>
#     </tr>
#     <tr> 
#         <td>
#             **Y**
#         </td>
#         <td>
#         Tensor("Placeholder_2:0", shape=(6, ?), dtype=float32) (not necessarily Placeholder_2)
#         </td>
#     </tr>
# 
# </table>

### 2.2) Initializing the parameters
# 
# Second task: Initialize the parameters in tensorflow.

#

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    # Initialize the parameters in TensorFlow
    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    # Use the Xavier Initialization for weights and Zero Initialization for biases    
    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# In[22]:

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# **Expected Output**: 
# 
# <table> 
#     <tr> 
#         <td>
#             **W1**
#         </td>
#         <td>
#          < tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref >
#         </td>
#     </tr>
#     <tr> 
#         <td>
#             **b1**
#         </td>
#         <td>
#         < tf.Variable 'b1:0' shape=(25, 1) dtype=float32_ref >
#         </td>
#     </tr>
#     <tr> 
#         <td>
#             **W2**
#         </td>
#         <td>
#         < tf.Variable 'W2:0' shape=(12, 25) dtype=float32_ref >
#         </td>
#     </tr>
#     <tr> 
#         <td>
#             **b2**
#         </td>
#         <td>
#         < tf.Variable 'b2:0' shape=(12, 1) dtype=float32_ref >
#         </td>
#     </tr>
# 
# </table>

# As expected, the parameters haven't been evaluated yet.

### 2.3) Forward propagation in tensorflow 
# 
# Now implement the forward propagation module in tensorflow. The functions being used: 
# 
# - `tf.add(...,...)` to do an addition
# - `tf.matmul(...,...)` to do a matrix multiplication
# - `tf.nn.relu(...)` to apply the ReLU activation
# 

# 

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Implement forward pass of the neural network
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3, A2) + b3
    
    
    return Z3


# In[24]:

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))


# **Expected Output**: 
# 
# <table> 
#     <tr> 
#         <td>
#             **Z3**
#         </td>
#         <td>
#         Tensor("Add_2:0", shape=(6, ?), dtype=float32)
#         </td>
#     </tr>
# 
# </table>

# Here, forward propagation doesn't output any cache. 

### 2.4) Compute cost

# 

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    # Compute the cost function
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)) # Does summation over the examples
    
    return cost


#

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)            # Created placeholder for X and Y
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)         # Have it output forward propagation
    cost = compute_cost(Z3, Y)                      # Tensor for cost function
    print("cost = " + str(cost))


# **Expected Output**: 
# 
# <table> 
#     <tr> 
#         <td>
#             **cost**
#         </td>
#         <td>
#         Tensor("Mean:0", shape=(), dtype=float32)
#         </td>
#     </tr>
# 
# </table>

### 2.5) Backward propagation & parameter updates
# 
# Where programming frameworks become appreciated. All the backpropagation and the parameters update is taken care of in 1 line of code.
# 

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
  
    # Initialize parameters
    parameters = initialize_parameters()
     
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                          # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict= {X: minibatch_X, Y: minibatch_Y})

                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


# Run the following cell to train model! Can takes about 5 minutes. "Cost after epoch 100" should be 1.048222.
# IF NOT,interrupt the training and try to correct the code.
# IF CORRECT cost, take a break and come back in 5 minutes!

# 

parameters = model(X_train, Y_train, X_test, Y_test)


# **Expected Output**:
# 
# <table> 
#     <tr> 
#         <td>
#             **Train Accuracy**
#         </td>
#         <td>
#         0.999074
#         </td>
#     </tr>
#     <tr> 
#         <td>
#             **Test Accuracy**
#         </td>
#         <td>
#         0.716667
#         </td>
#     </tr>
# 
# </table>
# 
# The algorithm can now recognize a sign representing a figure between 0 and 5 with 71.7% accuracy.
# 
#
# Summary:
# - Tensorflow is a programming framework used in deep learning
# - The two main object classes in tensorflow are Tensors and Operators. 
# - When coding in tensorflow, take the following steps:
#     - Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
#     - Create a session
#     - Initialize the session
#     - Run the session to execute the graph
# - Can execute the graph multiple times as seen in model()
# - The backpropagation and optimization is automatically done when running the session on the "optimizer" object.
