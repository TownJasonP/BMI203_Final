'''
How might this system we've been working on actually be used? What happens if we
try to generate a probability distribution of binding over
 - a randomly generated sequence
 - a randomly generate sequence containing a rap1 binding site
 - some real genomic dna sequences associated with rap1 binding
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D

# Use all positive data and an equal proportion of negative data this time
x = np.load('../output/x_data.npy')
y = np.load('../output/y_data.npy')

pos_ix = [i for i in range(len(y)) if y[i] == 1]
neg_ix = [i for i in range(len(y)) if y[i] == 0]

train_ix = range(250)

train_set = [pos_ix[ix] for ix in train_ix] + [neg_ix[ix] for ix in train_ix]

x_train = x[train_set]
y_train = y[train_set]

neural_net = Sequential()
neural_net.add(Conv1D(nb_filter = 5, filter_length = 5, input_shape = (17,4)))
neural_net.add(Flatten())
neural_net.add(Dense(10))
neural_net.add(Activation('relu'))
neural_net.add(Dense(1))
neural_net.add(Activation('sigmoid'))
neural_net.compile(loss='mse', optimizer='sgd')
neural_net.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 250)


# convert to numeric form
numerical_seq = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}
def let_to_num(a_sequence_collection):
    n = len(a_sequence_collection)
    num_sequence = np.zeros((n, 17, 4))
    for i in range(n):
        for j in range(17):
            nuc = a_sequence_collection[i][j]
            num_sequence[i][j] = numerical_seq[nuc]
    return(num_sequence)

'''
Generate random sequence for convolving test
'''
plt.figure(figsize = (8,3), facecolor = 'white')
rand_seq = np.random.choice(a = ['A','T','C','G'], p = [0.3,0.3,0.2,0.2], size = 1000)
# run cnn on data
probs = []
for i in range(100):
    sub_string = [rand_seq[i:i+17]]
    num_sub_string = let_to_num(sub_string)
    prediction = neural_net.predict(num_sub_string)
    probs.append(prediction[0])
plt.plot(range(100), probs, color = 'blue', alpha = 0.9)
plt.xticks(range(100), rand_seq[0:100], size = 7)
plt.ylim(0,1)
plt.yticks([0,0.5,1])
plt.title('Predicted Binding Profile on Randomly Generated Sequence')
plt.ylabel('Predicted\nProbability', size = 8)
plt.xlabel('Sequence', size = 8)
plt.tight_layout()
plt.savefig('../output/random_binding.png')

'''
Generate a random sequence with an inserted binding site
'''
plt.figure(figsize = (8,3), facecolor = 'white')
positive_binding_site = [i for i in 'GCACCCATACATTACAT']
insert_seq = rand_seq
insert_seq[40:57] = positive_binding_site
probs = []
for i in range(100):
    sub_string = [insert_seq[i:i+17]]
    num_sub_string = let_to_num(sub_string)
    prediction = neural_net.predict(num_sub_string)
    probs.append(prediction[0])
plt.plot(range(100), probs, color = 'blue', alpha = 0.9)
plt.xticks(range(100), insert_seq[0:100], size = 7)
plt.ylim(0,1)
plt.yticks([0,0.5,1])
plt.fill_between(x = np.linspace(40,57,10), y1 = 0, y2 = 1, color = 'green', alpha = 0.2)
plt.title('Predicted Binding Profile on Randomly Generated Sequence')
plt.ylabel('Predicted\nProbability', size = 8)
plt.xlabel('Sequence', size = 8)
plt.tight_layout()
plt.savefig('../output/insertion_binding.png')
