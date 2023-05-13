"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)  #layer數
        self.sizes = sizes
        #sizes[1:] =>[3,1]  sizes[:-1] =>[2,3]
        #sizes=[2,3,1] => 生成一個list,內含兩個numpy array,第一個是3*1,第二個1*1 (input layer不算bias)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #sizes=[2,3,1] => 生成一個list,內含兩個numpy array,第一個是3*2,第二個3*1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        #創建一個新的列表 nabla_b，其中每個元素都是一個形狀和 self.biases列表中對應元素相同的numpy陣列，並且其所有元素初始化為零。
        #nabla就是梯度那個倒三角形符號
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #delta_nabla_b,delta_nabla_w長的size和nabla_b,nabla_w一樣
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #mini_batch的step 3的sum那邊
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #mini_batch的step 3的更新
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    
    #:x:training input(mnist:28*28=784 dimensional vector) y:desired output(mnist:10 dimensional vector)
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #創建一個新的列表 nabla_b，其中每個元素都是一個形狀和 self.biases列表中對應元素相同的numpy陣列，並且其所有元素初始化為零。
        #nabla_b:gradient of bias
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x  #a^1 (input layer 沒activation function)
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        #b,w:bias和weights中的第k個numpy array
        #sizes=[2,3,1]
        #self.biases=[arr(3*1),arr(1*1)]
        #self.weights[arr(3*2),arr(3*1)]
        #第一次跑的話就是 b=arr(3*1)和w=arr(3*2) 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #矩陣乘法
            zs.append(z)
            activation = sigmoid(z) #替換
            activations.append(activation)
        # backward pass
        #要對兩個 ndarray 進行 element-wise 乘法，可以使用 * 運算符
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #output error
        #step 5
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #check!
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): #一個layer一個layer的算回去
            #step 4
            z = zs[-l] #這是負欸樓,不是負一
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #step 5
            nabla_b[-l] = delta #這是負欸樓,不是負一
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #這是負欸樓,不是負一
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
