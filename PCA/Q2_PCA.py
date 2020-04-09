# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:59:00 2018

@author: mervet
"""

### PCA####

import numpy as np
import matplotlib.pyplot as plt

train_all=np.loadtxt("opt_digits_train.txt",dtype="int",delimiter=",")
train_no_deleted = train_all
labels = train_all[:,-1]
train_all = np.delete(train_all,-1,1) 
train_mean = train_all.mean(axis = 0)
train_zero = train_all-train_mean 
cov = ((train_all-train_all.mean(axis=0)).T).dot((train_all-train_all.mean(axis=0)))/train_all.shape[0]
values, vectors = np.linalg.eig(cov)

loss_of_information = 1-sum(values[0:2])/sum(values[:])
z1 = np.dot(train_zero, vectors[:,0])
z2 = np.dot(train_zero, vectors[:,1])

for i in range(10):
    z1_digits = z1[train_no_deleted[:,-1]==i]
    z2_digits = z2[train_no_deleted[:,-1]==i]
    plt.scatter(z1_digits,z2_digits,label =""+str(i))
plt.title('Reduced Data-Training Data- PCA ')
plt.xlabel('First Eigenvectör')
plt.ylabel('Second Eigenvectör')
plt.legend()

test_all = np.loadtxt("opt_digits_test.txt",dtype="int",delimiter=",")
test_labels = test_all[:,-1]
test_all = np.delete(test_all,-1,1)
test_mean = test_all.mean(axis = 0)
test_zero_mean = test_all-test_mean
test_z1 = np.dot(test_zero_mean, vectors[:,0])
test_z2 = np.dot(test_zero_mean, vectors[:,1])

success_cnt = 0 
for k in range(test_z1.shape[0]):
    distance =[]
    distance_1 = test_z1[k]-z1
    distance_2 = test_z2[k]-z2
    distance = distance_1*distance_1+distance_2*distance_2
    if(labels[np.argmin(distance)] == test_labels[k]):
        success_cnt = success_cnt+1 

test_error_PCA = 1-(success_cnt/test_all.shape[0])
print('Q2-PCA Test Error')
print(test_error_PCA)

