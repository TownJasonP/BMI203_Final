import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras import backend as K

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

def train_and_test(num_filters, filter_size, deep_neurons):
	neural_net = Sequential()
	neural_net.add(Conv1D(nb_filter = num_filters, filter_length = filter_size, input_shape = (17,4)))
	neural_net.add(Flatten())
	neural_net.add(Dense(deep_neurons))
	neural_net.add(Activation('relu'))
	neural_net.add(Dense(1))
	neural_net.add(Activation('sigmoid'))
	neural_net.compile(loss='mse', optimizer='sgd')
	neural_net.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 100, verbose = 0)

	test_performance = np.ndarray.flatten(neural_net.predict(x_test))

	K.clear_session() # weird memory issue with keras - have to explicitly clear session using keras backend when training in loops

	x,y = roc(y_test, test_performance)
	a = auroc(x,y)
	return(a)

# scan across several parameters, iterate a few times and return average AUROC
# how do different parameters affect the output?
# reduce number of iterations to 100 make it easier to see any improvements

x = np.load('../output/x_data.npy')
y = np.load('../output/y_data.npy')

pos_ix = [i for i in range(len(y)) if y[i] == 1]
neg_ix = [i for i in range(len(y)) if y[i] == 0]

train_ix = np.random.choice(range(250), 125, replace = False)
test_ix = [i for i in range(250) if i not in train_ix]

train_set = [pos_ix[ix] for ix in train_ix] + [neg_ix[ix] for ix in train_ix]
test_set = [pos_ix[ix] for ix in test_ix] + [neg_ix[ix] for ix in test_ix]

x_train = x[train_set]
y_train = y[train_set]

x_test = x[test_set]
y_test = y[test_set]

#set up empty lists to store and save data
x1 = []
x2 = []
x3 = []
y0 = []
count = 0
for num_filters in range(1,10):
	for filter_size in range(1,10):
		for deep_neurons in range(2,20,2):
			for iteration in range(3): #repeat each one 3 times
				print('iteration: ' + str(count))
				count += 1
				x1.append(num_filters)
				x2.append(filter_size)
				x3.append(deep_neurons)
				start_time = time.time()
				y0.append(train_and_test(num_filters, filter_size, deep_neurons))
				print('Run time: ' + str(time.time() - start_time) + ' s')

optimization_data = pd.DataFrame()
optimization_data['Number_filters'] = x1
optimization_data['Filter_size'] = x2
optimization_data['Number_deepNeurons'] = x3
optimization_data['AUROC'] = y0
optimization_data.to_csv('../output/optimization_data.csv')
