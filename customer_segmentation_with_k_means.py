import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
dataframe = pd.read_csv('./csv/Cust_Segmentation.csv')
print(dataframe.head())
cust_df=dataframe.drop('Address',axis=1)
from sklearn.preprocessing import StandardScaler
x=cust_df.values[:,1:]
'''here 1: means select all columns from index 1 '''
x=np.nan_to_num(x)
df=StandardScaler().fit_transform(x)
print(df)
cluster_number =3
k_means= KMeans(init='k-means++',n_clusters=cluster_number,n_init=12)
k_means.fit(x)
label=k_means.labels_
print(label)
cust_df["cluster"]=label
print(cust_df.head())
print(cust_df.groupby('cluster').mean())
area=np.pi*(x[:,1])**2
plt.scatter(x[:,0],x[:,3],s=area,c=label.astype(float),alpha=0.5)
plt.xlabel('Age',fontsize=18)
plt.ylabel('Income',fontsize=16)
plt.savefig('./pngFiles/data_set_scatter.png')