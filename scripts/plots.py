# Data libraries
import pickle as pkl

# Plotting libraries
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def plotOnEmbedding(x_bins, y_bins):
    plt.plot(x_bins, y_bins, linestyle = '--', marker = 'o', color = 'black')
    plt.grid()
    plt.xlim([-0.6, 0.8])
    plt.ylim([-0.8, 1])
    plt.plot([0.0, 0.0], [-0.8, 1.0], color = 'black', linestyle = '--')
    plt.plot([-0.6, 0.8], [0.6, 0.6], color = 'black', linestyle = '--')
    plt.plot([0.5, 0.5], [-0.8, 1.0], color = 'black', linestyle = '--')
    plt.savefig('embedded.png')

def plot(embedding, labels, title):
    def adhocLines():
        plt.plot([0.4, 0.4], [-0.6, 0.6], color = 'black')
    
    # Color selection
    color_dict = {0 : 'red', 1 : 'blue', 'red' : 'Bad', 'blue' : 'Good'}
    colors = [color_dict[label] for label in labels]

    # Plotting
    dim = embedding.shape[1]
    fig = plt.figure() 
    points = [embedding[:, i] for i in range(dim)]
    if dim == 2:
        x, y = points
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color = colors) 
    else:
        x, y, z = points 
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(x, y, z, c = colors)
	plt.show()
        #with open('3dplot.pkl', 'wb') as pklfile:
        #    pkl.dump([x, y, z, colors], pklfile)
    #adhocLines()

    # Labeling the graph
    red_patch  = mpatches.Patch(color = 'red',  label = color_dict['red'])
    blue_patch = mpatches.Patch(color = 'blue', label = color_dict['blue'])
    plt.legend(handles=[red_patch, blue_patch], loc = 1)
    plt.title('%d-dimensional Spectral Embedding' % embedding.shape[1])
    plt.xlabel('%s images' % title)
    plt
    plt.savefig('embedded.png')


if __name__ == '__main__':
    print 'This is a helper script for plotting spectral embedding plots' 

