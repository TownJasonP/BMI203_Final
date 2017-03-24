'''
1) Implement a feed-forward, three-layer, neural network with standard sigmoidal
units. Your program should allow for variation in the size of input layer, hidden
layer, and output layer. You will need to write your code to support cross-validation.
We expect that you will be able to produce fast enough code to be of use in the
learning task at hand. You will want to make sure that your code can learn the
8x3x8 encoder problem prior to attempting the Rap1 learning task.
If your code is too slow or otherwise unworkable for the Rap1 learning task, you
will need to show results that indicate your code worked adequately to learn the
8x3x8 encoder problem. In this case, after showing adequacy for the 8x3x8 encoder,
you may obtain other neural network code, preferably in a language that runs
quickly (C, Matlab, etc.).

As an alternative, after showing that your NN code can run the 8x3x8 encoder,
you may choose to implement or obtain code for another machine learning method
(e.g. support vector machines or KNN).

NOTE: If you are unable to successfully implement the neural network code directly
yourself, you should not stop working. While it will be reflected in your project
grade, you should continue with the Rap1 learning task using code for the core
learning method obtained elsewhere. In this case, you should still document your
effort toward making a NN that can learn the 8x3x8 encoder.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def sigmoid(x):
    return(1/(1+np.exp(-x)))
def dsigmoid(x):
    return(sigmoid(x)*(1-sigmoid(x)))

plt.figure(figsize = (6,6), facecolor = 'white')
x = np.linspace(-10,10,1000)
plt.plot(x, sigmoid(x), lw = 2, alpha = 0.8, label = r'$y = \sigma$')
plt.plot(x, dsigmoid(x), lw = 2, alpha = 0.8, label = r'$y = d\sigma/dx$')
plt.axis([-5,5,-0.1,1.1])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 2)
plt.grid()
plt.savefig('../output/Activation_Function.png')

# create class representing a neural network
class NeuralNet():
    def __init__(self, input_data, hidden_size, target_output, alpha, lamda):
        # handle inputs
        self.input_data = input_data
        self.input_size = self.input_data.shape

        # handle hidden layer
        self.hidden_size = hidden_size

        # handle outputs
        self.target_output = target_output
        self.output_size = self.target_output.shape

        # learning parameters
        self.alpha = alpha
        self.lamda = lamda

        # initialize synaptic with small random numbers
        self.W1 = np.random.normal(0,0.01, size = (self.input_size[1], hidden_size))
        self.W2 = np.random.normal(0,0.01, size = (hidden_size, self.output_size[1]))
        # and bias weights
        self.b1 = np.random.normal(0, 0.01, size = (1,self.hidden_size))
        self.b2 = np.random.normal(0, 0.01, size = (1,self.output_size[1]))

    def feedforward(self):
        z2 = np.dot(self.input_data, self.W1) + self.b1
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.W2) + self.b2
        a3 = sigmoid(z3)
        return(a3)

    def cost_function(self):
        m = self.input_size[0]
        J1 = 1.0 / m * np.sum(0.5*(self.feedforward()-self.target_output)**2)
        J2 = self.lamda / 2.0 * (np.sum(np.sum(self.W1**2)) + np.sum(np.sum(self.W2**2)))
        J = J1 + J2
        return(J)

    def update_weights(self):
        # feedforward:
        z2 = np.dot(self.input_data, self.W1) + self.b1
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.W2) + self.b2
        a3 = sigmoid(z3)

        # backpropagate error
        error = a3 - self.target_output
        delta2 = np.multiply(error, dsigmoid(z3))
        dJdW2 = np.dot(a2.T, delta2)

        dist_error = np.dot(delta2, self.W2.T)
        delta1 = np.multiply(dist_error, dsigmoid(z2))
        dJdW1 = np.dot(self.input_data.T, delta1)

        # update weights
        self.W2 -= self.alpha * (dJdW2/self.input_size[0] + self.lamda * self.W2)
        self.b2 -= self.alpha * np.average(delta2, axis = 0)

        self.W1 -= self.alpha * (dJdW1/self.input_size[0] + self.lamda * self.W1)
        self.b1 -= self.alpha * np.average(delta1, axis = 0)

'''
Test over various parameters to look for optimal improvement over 100K iterations
'''

count = 1
plt.figure(figsize = (12,8))
inp = np.random.choice([0,1], size = (1000,8))
out = inp

for alpha in [1,0.1,0.01]:
    for lamda in [0,0.01,0.1]:

        plt.subplot(3,3,count)
        n = NeuralNet(inp, 3, out, alpha, lamda)
        cost = []
        for i in range(100000):
            n.update_weights()
            cost.append(n.cost_function())

        plt.plot(range(len(cost)), cost, color = 'blue')
        plt.xscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.ylim(0,1.1)
        plt.title('Cost Function over Iterations with\nalpha = ' + str(n.alpha) + ', lambda = ' + str(n.lamda))
        count += 1
plt.tight_layout()
plt.savefig('../output/parameter_performance.png')

'''
Generate images related to model performance and weights over
100K iterations (to make animation later using imageJ)
'''

inp = np.random.choice([0,1], size = (1000,8))
out = inp

n = NeuralNet(inp, 3, out, 0.1, 0)
cost = []
for i in range(100000):
    n.update_weights()
    cost.append(n.cost_function())

    if i%1000 == 0:
        x = np.ndarray.flatten(n.input_data)
        y = np.ndarray.flatten(n.feedforward())
        zero = [y[j] for j in range(len(x)) if x[j] == 0]
        ones = [y[j] for j in range(len(x)) if x[j] == 1]

        plt.figure(figsize = (8,6))
        plt.hist(zero, bins = np.linspace(0,1,51), alpha = 0.5, label = 'zero', color = 'blue')
        plt.axvline(np.average(zero), alpha = 0.5, ls = '--', color = 'blue')
        plt.hist(ones, bins = np.linspace(0,1,51), alpha = 0.5, label = 'one', color = 'green')
        plt.axvline(np.average(ones), alpha = 0.5, ls = '--', color = 'green')

        plt.title('Neural Net Performance at iteration ' + str(i))
        plt.legend(bbox_to_anchor = (1.5,1))
        plt.ylim(0,1000)
        plt.xlabel('Predicted probability that a given digit is 1')
        plt.ylabel('Frequency')
        plt.savefig('../output/performance_animation/' + str(i) + '.png', bbox_inches = 'tight')
        plt.close()

        plt.figure(figsize = (8,6))
        for wi in range(8):
            for wj in range(3):
                plt.plot([0, 1], [wi, wj+2.5], color = 'black', lw = np.abs(n.W1[wi][wj]), zorder = -1)

        for wi in range(3):
            for wj in range(8):
                plt.plot([1, 2], [wi+2.5, wj], color = 'black', lw = np.abs(n.W2[wi][wj]), zorder = -1)

        plt.scatter([0]*8, range(8), s = 200, c = 'white', marker = 'o')
        plt.scatter([1]*3, np.arange(2.5, 5.5), s = 200, c = 'white', marker = 'o')
        plt.scatter([2]*8, range(8), s = 200, c = 'white', marker = 'o')
        plt.xticks([])
        plt.yticks([])
        plt.title('Network Weights at iteration ' + str(i))
        plt.savefig('../output/weights_animation/' + str(i) + '.png')
        plt.close()
    if i%10000 == 0:
        print(i)
