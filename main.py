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
across them according to guassian normal distribution with a standard deviation of 0.9'''