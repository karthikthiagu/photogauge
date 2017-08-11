# This is a code to perform spectral embedding on images.

# System libraries
import sys

# Data libraries
import h5py
import numpy as np
import cv2

# Embedding libraries
from sklearn.manifold import SpectralEmbedding, TSNE

# User defined libraries
from plots import plot

class Embed:
    
    # Initializes the data to perform clustering on - images, labels and the size
    def __init__(self, data):
        self.images = data['features'][:]
        self.labels = data['labels'][:]
        self.size = self.labels.shape[0] # Number of data points
        print 'Images are of shape = ', self.images.shape
        print 'Labels are of shape = ', self.labels.shape
        print 'Number of data points = %d' % self.size

    # Perform spectral embedding
    def spectralEmbedding(self, num_components):
        spectral = SpectralEmbedding(n_components = num_components, affinity = 'nearest_neighbors')
        embedding = spectral.fit_transform(self.images)
        return embedding
    def tsne(self, num_components):
        np.random.seed(0)
        tsne = TSNE(n_components = num_components, perplexity = 40.0, init = 'random', n_iter = 5000, learning_rate = 100, early_exaggeration = 2.0) 
        embedding = tsne.fit_transform(self.images)
        return embedding

def embedAndPlot(data, num_components, title):
    embed = Embed(data)
    #embedding = embed.spectralEmbedding(num_components)
    embedding = embed.tsne(num_components)
    plot(embedding, embed.labels, title)

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print 'Usage : python2.7 embedding.py datapath num_components title'
        exit()
    else:
        _, datapath, num_components, title = sys.argv
        data = h5py.File(datapath, 'r')
        num_components = int(num_components)

    embedAndPlot(data, num_components, title)

