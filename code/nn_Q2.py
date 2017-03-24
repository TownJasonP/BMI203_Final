'''
Set up the learning procedure to allow DNA sequences as input and
to produce an output of the likelihood that an input is a true Rap1
binding site. You will want the output value to be a real number no
matter what method you are using, since your performance on the blind
test set will be judged based on the area under an ROC curve. So,
please DO NOT threshold your output to produce a binary value.

	- As mentioned above, you may use your ANN implementation.
    Many very successful projects have done so in the past. However,
    you may also use KNN, SVM, random forests, or any other approach
    that you see fit.

	- Describe the machine learning approach in detail. This will include,
    for an ANN, a description of the network structure of your encodings
    of inputs and output. For an SVM, you will need to discuss input
    encoding as well as kernel function choices, etc... This will also
    include a description of the representation you have chosen for the
    input DNA sequences.
'''
 
import numpy as np
import matplotlib.pyplot as plt
import keras

# define reverse complements to expand positive binding site data set
def reverse_complement(sequence):
    comp_nucleotide = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    rev_seq = ''
    for nuc in sequence[::-1]:
        rev_seq += comp_nucleotide[nuc]
    return(rev_seq)

# import positive dna sequences and their reverse complements
filepath_rap_pos = '../data/rap1-lieb-positives.txt'
rap_pos = []
text_rap_pos = open(filepath_rap_pos)
for line in text_rap_pos:
    sequence = line.strip()
    rap_pos.append(sequence)
    rap_pos.append(reverse_complement(sequence))

# import negative binding sequences - we probably don't really need all of this
# data since we only have 137 positive examples, so just take the first 17
# nucleotides from each sequence
filepath_rap_neg = '../data/yeast-upstream-1k-negative.fa'
rap_neg = []
text_rap_neg = open(filepath_rap_neg)
for line in text_rap_neg:
    if line[0] != '>':
        sequence = line.strip()[0:17]
        if len(sequence) == 17:
            rap_neg.append(sequence)

# make sure there is no overlap (this should remove 2 sequences from rap_neg)
overlap = 0
for i in rap_neg:
    if i in rap_pos:
        rap_neg.remove(i)

# convert dna sequences to a numerically encoded form
numerical_seq = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}

def let_to_num(a_sequence_collection):
    n = len(a_sequence_collection)
    num_sequence = np.zeros((n, 17, 4))
    for i in range(n):
        for j in range(17):
            nuc = a_sequence_collection[i][j]
            num_sequence[i][j] = numerical_seq[nuc]
    return(num_sequence)

rap_pos_num = let_to_num(rap_pos)
rap_neg_num = let_to_num(rap_neg)

'''
Save outputs to easily access in next questions
'''

np.save('../output/rap_positives', rap_pos_num)
np.save('../output/rap_negatives', rap_neg_num)

x = np.append(rap_pos_num, rap_neg_num, axis = 0)
y = [1] * rap_pos_num.shape[0] + [0] * rap_neg_num.shape[0]

np.save('../output/x_data', x)
np.save('../output/y_data', y)

'''
Example figure of one-hot encoding that I'll be using to encode the nucleotide
information
'''

plt.figure(facecolor = 'white', figsize = (8,5))
plt.imshow(x[0].T, interpolation = 'nearest', cmap = 'Greys')
plt.yticks(range(4),['A','T','C','G'])
plt.xticks(range(17), range(1,18))
plt.ylabel('Nucleotide')
plt.xlabel('Position')
plt.title('One-Hot Encoding of Potential Binding Sites\n')
plt.savefig('../output/one_hot.png', bbox_inches = 'tight')

'''
Just out of curiosity:
How often do different nucleotides show up in different locations in the positive
and negative datasets?
'''

x1 = np.average(rap_pos_num, axis = 0)
x2 = np.average(rap_neg_num, axis = 0)

plt.figure(facecolor = 'white', figsize = (8,6))
plt.subplot(211)
plt.imshow(x1.T, interpolation = 'nearest', cmap = 'inferno', vmin = 0, vmax = 0.6)
plt.yticks(range(4),['A','T','C','G'])
plt.xticks(range(17), range(1,18))
plt.ylabel('Nucleotide')
plt.xlabel('Position')
plt.title('Nucleotide Frequencies in Rap-Binding Sites')
plt.colorbar(shrink = 0.5, ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6])

plt.subplot(212)
plt.imshow(x2.T, interpolation = 'nearest', cmap = 'inferno', vmin = 0, vmax = 0.6)
plt.yticks(range(4),['A','T','C','G'])
plt.xticks(range(17), range(1,18))
plt.ylabel('Nucleotide')
plt.xlabel('Position')
plt.title('Background Nucleotide Frequencies')
plt.colorbar(shrink = 0.5, ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6])

plt.tight_layout()
plt.savefig('../output/nuc_pos_frequencies.png', bbox_inches = 'tight')
