import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
'''The make_blobs function from scikit-learn it is used to generate synthetic data for clustering mostly used for testing and demonstrating clustering 
algorithms like Kmeans which we will be using in this lab'''
np.random.seed(11)