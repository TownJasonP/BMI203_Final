'''
4) Given your learning system, perform cross validation experiments first to see
whether it is working, and then see how well your system performs.

	- Describe how you set up your experiment to measure your system's
	performance.

	- What set of learning parameters works the best? Please provide sample
	output from your system.

	- What are the effects of altering your system (e.g. number of hidden units
	or choice of kernel function)? Why do you think you observe these effects?

	- What other parameters, if any, affect performance?

'''
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical

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

# load in x and y data
x = np.load('../output/x_data.npy')
y = np.load('../output/y_data.npy')

pos_ix = [i for i in range(len(y)) if y[i] == 1]
neg_ix = [i for i in range(len(y)) if y[i] == 0]

# cross validate using parameters from nn_Q3.py by randomly shuffling training
# inputs and looking at the auroc distribution
auroc_train = []
auroc_test = []

for test in range(100):

	train_ix = np.random.choice(range(250), 125, replace = False)
	test_ix = [i for i in range(250) if i not in train_ix]

	train_set = [pos_ix[ix] for ix in train_ix] + [neg_ix[ix] for ix in train_ix]

	test_set = [pos_ix[ix] for ix in test_ix] + [neg_ix[ix] for ix in test_ix]

	x_train = x[train_set]
	y_train = y[train_set]

	x_test = x[test_set]
	y_test = y[test_set]

	neural_net = Sequential()
	neural_net.add(Conv1D(nb_filter = 5, filter_length = 5, input_shape = (17,4)))
	neural_net.add(Flatten())
	neural_net.add(Dense(10))
	neural_net.add(Activation('relu'))
	neural_net.add(Dense(1))
	neural_net.add(Activation('sigmoid'))
	neural_net.compile(loss='mse', optimizer='sgd')
	neural_net.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 250)

	train_performance = np.ndarray.flatten(neural_net.predict(x_train))
	test_performance = np.ndarray.flatten(neural_net.predict(x_test))

	x1,y1 = roc(y_train, train_performance)
	a1 = auroc(x1,y1)
	auroc_train.append(a1)

	x2,y2 = roc(y_test, test_performance)
	a2 = auroc(x2,y2)
	auroc_test.append(a2)

plt.figure(facecolor = 'white', figsize = (6,4))
bins = np.linspace(0.95,1,21)
plt.hist(auroc_test, bins = bins, alpha = 0.5, label = 'Test')
plt.hist(auroc_train, bins = bins, alpha = 0.5, label = 'Train')
plt.xlabel('AUROC')
plt.ylabel('Frequency')
plt.xlim(0.95,1.0)
plt.legend(loc = 0)
plt.title('Cross-Validation: AUROC for 100 Indpendent\nRandom Training/Testing Partitions')
plt.savefig('../output/cross_validation.png', bbox_inches = 'tight')
