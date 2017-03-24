'''
5) To provide an objective measure of your neural network's ability to recognize
binding sites, a test dataset has been provided (rap1-lieb-test.txt). There are
no class labels on these sequences.

Apply your training regime using the parameters and procedure that you optimized
for your cross-validation results to all the available training data. Run the
trained system on the test dataset. For each sequence, output the sequence and
its output value from the network, separated by a tab, as follows:

ACATCCGTGCACCATTT	0.927 AAAAAAACGCAACTAAT	0.123

If you do not use the full 17 bases, output the sequence subset as an additional
column (column 3), and in your writeup specify what portion of the sequence you
used. In the above example, the first sequence is a bona fide Rap1 site and the
second is not. They are correctly ranked relative to one another.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D

'''
Set up neural network using optimized values from nn_Q4.py
'''

output_files = ['../output/final_predictions_run1.txt', '../output/final_predictions_run2.txt']


# Use all positive data and an equal proportion of negative data this time
x = np.load('../output/x_data.npy')
y = np.load('../output/y_data.npy')

pos_ix = [i for i in range(len(y)) if y[i] == 1]
neg_ix = [i for i in range(len(y)) if y[i] == 0]

train_ix = range(250)

train_set = [pos_ix[ix] for ix in train_ix] + [neg_ix[ix] for ix in train_ix]

x_train = x[train_set]
y_train = y[train_set]

for i in range(2):

    neural_net = Sequential()
    neural_net.add(Conv1D(nb_filter = 10, filter_length = 5, input_shape = (17,4)))
    neural_net.add(Flatten())
    neural_net.add(Dense(10))
    neural_net.add(Activation('relu'))
    neural_net.add(Dense(1))
    neural_net.add(Activation('sigmoid'))
    neural_net.compile(loss='mse', optimizer='sgd')
    neural_net.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 250)

    # read in sequences from rap1-lieb-test.txt
    filepath = '../data/rap1-lieb-test.txt'
    file_text = open(filepath)
    sequences = []
    for line in file_text:
        sequence = line.strip()
        sequences.append(sequence)

    numerical_seq = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}

    # convert to numeric form
    def let_to_num(a_sequence_collection):
        n = len(a_sequence_collection)
        num_sequence = np.zeros((n, 17, 4))
        for i in range(n):
            for j in range(17):
                nuc = a_sequence_collection[i][j]
                num_sequence[i][j] = numerical_seq[nuc]
        return(num_sequence)

    num_sequences = let_to_num(sequences)

    # run cnn on data
    predictions = neural_net.predict(num_sequences)

    # output csv file
    output = open(output_files[i], 'w')
    for i in range(len(sequences)):
        output.write(str(sequences[i]) + '\t' + str(predictions[i][0]) + '\n')
    output.close()

# plot histogram of predicted probabilities
# Hoping to see a roughly bimodal distribution
plt.figure(figsize = (6,4), facecolor = 'white')
plt.hist(np.ndarray.flatten(predictions), bins = np.linspace(0,1,21), alpha = 0.5)
plt.ylim(0,500)
plt.title('Model Predictions on Unknown Data')
plt.xlabel('Probability of Being a Rap Binding Site')
plt.ylabel('Frequency')
plt.savefig('../output/probability_prediction_distribution.png', bbox_inches = 'tight')

run1 = pd.read_csv(output_files[0], sep = '\t', header = None)
run2 = pd.read_csv(output_files[1], sep = '\t', header = None)

plt.figure(figsize = (6,6), facecolor = 'white')
plt.scatter(run1[1], run2[1], marker = 'o', alpha = 0.2, color = 'blue')
plt.xlabel('Replicate 1 Predictions')
plt.ylabel('Replicate 2 Predictions')
plt.title('Convolutional Neural Network: Repeatability')
plt.axis([-0.05,1.05,-0.05,1.05])
plt.grid()
plt.savefig('../output/repeatability.png')

from scipy.stats import pearsonr
print('Pearson correlation: ' + str(pearsonr(run1[1], run2[1])[0]))

output = open('../output/tab_sep_predictions.txt', 'w')
for i in range(len(sequences)):
    output.write(str(sequences[i]) + '\t' + str(predictions[i][0]) + '\t')
output.close()
