import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
'''The make_blobs function from scikit-learn it is used to generate synthetic data for clustering mostly used for testing and demonstrating clustering 
algorithms like Kmeans which we will be using in this lab'''
np.random.seed(11)
x,y = make_blobs(n_samples=5000,centers=[[4,4],[-2,1],[2,-3],[1,1]],cluster_std=0.9)
'''n_samples is obviously number of samples which is the same as data points ?
centers [x,y] format means the space in which the samples are placed is 2d here we will have 4 clusters at the given centers and the points will spread
across them according to guassian normal distribution with a standard deviation of 0.9 which means that most of the points will be at a distance 0.9 units from
the center while some may be further but most of them will be within this distance'''
plt.scatter(x[:,0],x[:,1],marker='.')
'''0th column and 1st column of x are given to the scatter plot function'''
plt.savefig('./pngFiles/first_plot.png')
'''explanation of x and y, x and y will have values given to them by the make_blob function x is a 2d array having 5000 rows and 2 columns ? why because we have
5000 samples and we have 2d points so x,y makes our two columns 5000 x,y makes 5000 rows ? y is a 1d array of 4 labels in our case why 4 ? because we gave
the make blobs function 4 cluster centers'''
k_means= KMeans(init='k-means++',n_clusters=4,n_init=12)
'''init parameter determines how the initial cluster centroids are chosen because k means starts with random centroids and adjusts them k=means++ choses these 
initial centroids in a smart way n_init decides how many iterations will occur ? to find the best possible result k-means runs several times and picks the result
with the lowest inertia, inertia is the sum of squared distances between each point and the centroid of it's assigned cluster '''
k_means.fit(x)
