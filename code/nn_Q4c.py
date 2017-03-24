# visualize output from nn_Q4b.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('../output/optimization_data.csv')

num_neurons = data['Number_deepNeurons'].unique()

plt.figure(figsize = (8,8), facecolor = 'white')
count = 1
for i in enumerate(num_neurons):
    plt.subplot(3, 3, count)
    subset = data[data['Number_deepNeurons'] == i[1]]
    pivot = subset.pivot_table(values = 'AUROC', index = 'Number_filters', columns = 'Filter_size', aggfunc = 'mean')
    print(pivot)
    plt.imshow(pivot.values, interpolation = 'nearest', cmap = 'inferno', vmin = 0.7, vmax = 1)
    plt.title(str(i[1]) + ' Hidden Neurons', size = 8)
    plt.xticks(range(len(pivot.columns)), pivot.columns, size = 8)
    plt.yticks(range(len(pivot.index)), pivot.index, size = 8)
    plt.xlabel('Filter Size', size = 8)
    plt.ylabel('Number Filters', size = 8)
    #plt.colorbar(shrink = 0.5, ticks = [0.5, 1], orientation = 'horizontal')

    count += 1

plt.tight_layout()
plt.savefig('../output/param_optimization.png')
