'''
Design a training regime that will use both positive and negative training
data to induce your predictive model. Note that if you use a naive scheme here,
which overweights the negative data, your system will probably not converge (it
will just call everything a non-Rap1 site). Your system should be able to
quickly learn aspects of the Rap1 positives that allow elimination of much of
the negative training data. Therefore, you may want to be clever about caching
negative examples that form "interesting" cases and avoid running through all of
 the negative data all of the time.

MORE NOTES: You don't have to use the full 17 bases of each binding site. You
can use a subset if you think performance can be improved. For the negative
training data, you do not have to use the reverse complement of all the sequences,
 but it may improve performance.

	- How was your training regime designed so as to prevent the negative training
     data from overwhelming the positive training data?

    - What was your stop criterion for convergence in your learned parameters? How
    did you decide this?
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical

#import data generated in nn_Q2.py
x = np.load('../output/x_data.npy')
y = np.load('../output/y_data.npy')

# split into equally sized training and test sets with equally represented
# positive and negative examples (will test the effect of these numbers later)

pos_ix = [i for i in range(len(y)) if y[i] == 1]
neg_ix = [i for i in range(len(y)) if y[i] == 0]

n = int(len(pos_ix)/2)

train_set = pos_ix[0:n] + neg_ix[0:n]
test_set = pos_ix[n:2*n] + neg_ix[n:2*n]

x_train = x[train_set]
y_train = y[train_set]

x_test = x[test_set]
y_test = y[test_set]

'''
Since we have data with 1D structure, let's try implementing a 1D convolutional
neural network to analyze the binding sites using the Keras Package

In a way, this might have some physical justification - it seems vaguely
reminiscent of how transcription factors slide along dna, recognizing a few
nucleotides at a time
'''
# instantiate a sequential model using Keras
neural_net = Sequential()
# convolutional layer with 5 filters of length 5, taking the 17x 4 one-hot encoded
# sequences as input
neural_net.add(Conv1D(nb_filter = 5, filter_length = 5, input_shape = (17,4)))
# flatten everything for input into the next layers
neural_net.add(Flatten())
# fully connected layer with Relu activation
neural_net.add(Dense(10))
neural_net.add(Activation('relu'))
# fully connected layer connected to output with sigmoid activation
neural_net.add(Dense(1))
neural_net.add(Activation('sigmoid'))

# compile the model using mean squared error as the objective function and
# stochastic gradient descent as the optimization strategy
neural_net.compile(loss='mse', optimizer='sgd')

# train it for 250 epochs
neural_net.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 250)

# and feedforward the training and test sets to evaluate performance
train_performance = np.ndarray.flatten(neural_net.predict(x_train))
test_performance = np.ndarray.flatten(neural_net.predict(x_test))

# generate ROC curves for training and testing data
def roc(target, data):
    thresholds = np.linspace(0,1,1000)
    positives = [data[i] for i in range(len(target)) if target[i] == 1]
    negatives = [data[i] for i in range(len(target)) if target[i] == 0]
    tp = []
    fp = []
    for t in thresholds:
        tpr = len([i for i in positives if i > t])/len(positives)
        fpr = len([i for i in negatives if i > t])/len(negatives)
        tp.append(tpr)
        fp.append(fpr)
    return(fp,tp)

# calculate area under the roc curve
def auroc(x,y):
    delta_x = [x[i] - x[i+1] for i in range(len(x)-1)]
    delta_a = [delta_x[i] * y[i] for i in range(len(x)-1)]
    auc = np.sum(delta_a)
    return(auc)

plt.figure(figsize = (6,6), facecolor = 'white')

x1,y1 = roc(y_train, train_performance)
x2,y2 = roc(y_test, test_performance)

plt.plot(x1, y1, color = 'red', lw = 2, label = 'Train', alpha = 0.7)
plt.plot(x2,y2, color = 'blue', lw = 2, label = 'Test', alpha = 0.7)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for initial convolutional neural network')
plt.axis([-0.03, 1.03, -0.03, 1.03])
plt.legend(loc = 4)
plt.grid()
plt.savefig('../output/roc.png', bbox_inches = 'tight')

#calculate area under the roc curve for both sets
print('Area under the ROC curve for training data:')
print(auroc(x1,y1))

print('Area under the ROC curve for test data:')
print(auroc(x2,y2))
