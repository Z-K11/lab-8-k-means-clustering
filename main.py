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
k_means_labels=k_means.labels_
print("labels =",np.unique(k_means_labels))
'''our labels are 0,1,2,3 because the number of clusters n_clusters was = 4 '''
k_means_cluster_center = k_means.cluster_centers_
print("cluster centers = ",k_means_cluster_center)
fig =plt.figure(figsize=(6,4))
'''matplotlib figure, the size is given in inches 6 length,height 4 inches'''
colors=plt.cm.Spectral(np.linspace(0,1,(len(set(k_means_labels)))))
print("Colors :\n",colors)
ax=fig.add_subplot(1,1,1)
for k,col in zip(range(len([[4,4],[-2,1],[2,-3],[1,1]])),colors):
    '''zip creates iteratable pair If colors = ['red', 'blue', 'green', 'yellow'], zip(...) would produce (0, 'red'), (1, 'blue'),etc
    length calculates the number of rows and range generates numbers from 0 for the rows i.e indices'''
    my_members = (k_means_labels==k)
    cluster_center=k_means_cluster_center[k]
    ax.plot(x[my_members,0],x[my_members,1],'w',markerfacecolor=col,marker='.')
    '''my_members includes the value of a lebel which can be from 0 - 3 in our case '''
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
    '''cluster_center[0] and cluster_center[1]: Coordinates of the cluster centroid (x and y).
    'o': Displays the centroid as a circle marker.
    markerfacecolor=col: Sets the centroid's fill color to match the cluster's color.
    markeredgecolor='k': Adds a black outline around the centroid marker for emphasis.
    markersize=6: Sets the size of the centroid marker.'''
ax.set_title('Kmeans')
ax.set_xticks(())
ax.set_yticks(())
plt.savefig('./pngFiles/kmeans_plot.png')
'''clustering into 3 clusters'''
k_means3= KMeans(init='k-means++',n_clusters=3,n_init=12)
k_means3.fit(x)
fig = plt.figure(figsize=(6,4))
k_means3_labels=k_means3.labels_
k_means3_centers=k_means3.cluster_centers_
colors =plt.cm.Spectral(np.linspace(0,1,(len(set(k_means3_labels)))))
ax=fig.add_subplot(1,1,1)
for k,col in zip(range(len(k_means3_centers)),colors):
    my_members=(k_means3_labels==k)
    cluster_centroids=k_means3_centers[k]
    ax.plot(x[my_members,0],x[my_members,1],'w',markerfacecolor=col,marker='.')
    ax.plot(cluster_centroids[0],cluster_centroids[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
plt.savefig('./pngFiles/kmeans_plot2.png')