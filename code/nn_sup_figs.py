'''
Make and save the colorbar for nn_Q4b/nn_Q4c - optimization of parameters
'''
import matplotlib.pyplot as plt
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig = plt.figure(figsize=(6, 0.5), facecolor = 'white')
ax = plt.subplot()

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.inferno
norm = mpl.colors.Normalize(vmin=0.7, vmax=1)
cb = mpl.colorbar.ColorbarBase(ax, cmap = cmap,norm = norm,orientation='horizontal')
cb.set_label('AUROC')

plt.savefig('../output/legend.png', bbox_inches = 'tight')
