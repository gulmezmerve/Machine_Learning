# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:44:17 2018

@author: mervet
"""
import numpy as np

train_all=np.loadtxt("opt_digits_train.txt",dtype="int",delimiter=",")
labels = train_all[:,-1]
train_all = np.delete(train_all,-1,1)  
clusters = np.zeros(len(train_all))
k = 20
n = train_all.shape[0]
c = train_all.shape[1]
mean = np.mean(train_all, axis = 0)
std = np.std(train_all, axis = 0)
centers = np.random.randn(k,c)+mean
centers_L2 = centers.copy()
centers_old = np.zeros(centers.shape) # to store old centers
centers_new = centers.copy() # Store new centers

clusters = np.zeros(n)
distances = np.zeros((n,k))
error = np.linalg.norm(centers_new - centers_old)

while error != 0:
    for i in range(k):
        distances[:,i] = np.linalg.norm(train_all - centers_new[i], axis=1,ord=1)

    clusters = np.argmin(distances, axis = 1)
    centers_old = centers_new.copy()

    for i in range(k):
        centers_new[i] = np.mean(train_all[clusters == i], axis=0)
        
    error = np.linalg.norm(centers_new - centers_old)
    
cluster_separete= []
y = []

for m in range(20):
    cluster_separete.append(labels[clusters == m])
for a in range(20):
    x = []
    for b in range(10):
        x.append((cluster_separete[a]==b).sum())   
    y.append(np.argmax(x))

total_error = 0
for z in range(20):
    total_true = (cluster_separete[z] == y[z]).sum()
    total_error = total_error+ cluster_separete[z].shape[0]-total_true
    
k_means_clustering_error = total_error/train_all.shape[0] 

####L2-NORM########
centers_old_L2 = np.zeros(centers_L2.shape) # to store old centers
centers_new_L2 = centers_L2.copy() # Store new centers
clusters_L2 = np.zeros(n)

distances_L2 = np.zeros((n,k))
error_L2 = np.linalg.norm(centers_new_L2 - centers_old_L2)
i = 0 
while error_L2 != 0:
    for i in range(k):
        distances_L2[:,i] = np.linalg.norm(train_all - centers_new_L2[i],axis=1)
    clusters_L2 = np.argmin(distances_L2, axis = 1)
    centers_old_L2 = centers_new_L2.copy()
    for i in range(k):
        centers_new_L2[i] = np.mean(train_all[clusters_L2 == i], axis=0)        
    error_L2 = np.linalg.norm(centers_new_L2 - centers_old_L2)



cluster_seperate_L2 = []
y_L2 = []

m = 0
for m in range(20):
    cluster_seperate_L2.append(labels[clusters_L2 == m])
for a in range(20):
    x = []
    for b in range(10):
        x.append((cluster_seperate_L2[a]==b).sum())   
    y_L2.append(np.argmax(x))
    
total_error_L2 = 0
for z in range(20):
    total_true_L2 = (cluster_seperate_L2[z] == y_L2[z]).sum()
    total_error_L2 = total_error_L2+ cluster_seperate_L2[z].shape[0]-total_true_L2

k_means_clustering_error_L2 = total_error_L2/train_all.shape[0] 

print('Q3-k_means_clustering_error L1 and L2')
print(k_means_clustering_error)
print(k_means_clustering_error_L2)