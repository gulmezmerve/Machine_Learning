# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:56:53 2018

@author: mervet
"""

import numpy as np
import matplotlib.pyplot as plt


class class_def:
    
    def __init__(self,class_def,number): # constructor
        self.class_def = class_def
        self.mean = class_def.mean(axis = 0)
        self.N = class_def.shape[0]
        self.total_digitnumber = number

    def std(self):
        return self.class_def.std(axis = 0)

    def scatter(self) :
        return np.cov(self.class_def.T)
    
    def cov(self) :
        return np.cov(self.class_def.T)
    
    def scatter_b(self,mean_vektör):
        return self.N*(((self.mean-mean_vektör).reshape(62,1)).dot((self.mean-mean_vektör).reshape(1,62)))
              
train_all=np.loadtxt("opt_digits_train.txt",dtype="int",delimiter=",")
labels = train_all[:,-1]
train_no_deleted = train_all
var = train_all.std(axis = 0 )
var_zero_index =[] 

#variance zero is deleted
for i in range(var.shape[0]):
    if var[i] == 0:
        var_zero_index.append(i)
     
train_all = np.delete(train_all,var_zero_index[0],1)  
train_all = np.delete(train_all,(var_zero_index[1]-1),1) 
     
digit0 = class_def(np.delete(train_all[train_all[:,-1]==0,:],-1,1),train_all.shape[0])
digit1 = class_def(np.delete(train_all[train_all[:,-1]==1,:],-1,1),train_all.shape[0])
digit2 = class_def(np.delete(train_all[train_all[:,-1]==2,:],-1,1),train_all.shape[0])
digit3 = class_def(np.delete(train_all[train_all[:,-1]==3,:],-1,1),train_all.shape[0])
digit4 = class_def(np.delete(train_all[train_all[:,-1]==4,:],-1,1),train_all.shape[0])
digit5 = class_def(np.delete(train_all[train_all[:,-1]==5,:],-1,1),train_all.shape[0])
digit6 = class_def(np.delete(train_all[train_all[:,-1]==6,:],-1,1),train_all.shape[0])
digit7 = class_def(np.delete(train_all[train_all[:,-1]==7,:],-1,1),train_all.shape[0])
digit8 = class_def(np.delete(train_all[train_all[:,-1]==8,:],-1,1),train_all.shape[0])
digit9 = class_def(np.delete(train_all[train_all[:,-1]==9,:],-1,1),train_all.shape[0])

#deleted label
train_all = np.delete(train_all,-1,1)  
mean_vectör =train_all.mean(axis=0)

sw = digit0.cov()+digit1.cov()+digit2.cov()+digit3.cov()+digit4.cov()+digit5.cov()+digit6.cov()+digit7.cov()+digit8.cov()+digit9.cov()
sb = digit0.scatter_b(mean_vectör)+digit1.scatter_b(mean_vectör)+digit2.scatter_b(mean_vectör)+digit3.scatter_b(mean_vectör)+digit4.scatter_b(mean_vectör)+digit5.scatter_b(mean_vectör)+digit6.scatter_b(mean_vectör)+digit7.scatter_b(mean_vectör)+digit8.scatter_b(mean_vectör)+digit9.scatter_b(mean_vectör)

ssb = np.linalg.inv(sw).dot(sb) 
eigval, vectors = np.linalg.eig(ssb)

z1 = np.dot(train_all, vectors[:,0])
z2 = np.dot(train_all, vectors[:,1])

for i in range(10):
    z1_digits = z1[train_no_deleted[:,-1]==i]
    z2_digits = z2[train_no_deleted[:,-1]==i]
    plt.scatter(z1_digits,z2_digits,label =""+str(i))

plt.title('Traning Data after LDA')
plt.legend()

test_all = np.loadtxt("opt_digits_test.txt",dtype="int",delimiter=",")
test_labels = test_all[:,-1]
test_all = np.delete(test_all,var_zero_index[0],1)  
test_all = np.delete(test_all,(var_zero_index[1]-1),1) 
test_all_no_deleted = test_all
test_all = np.delete(test_all,-1,1)  

z1_test = np.dot(test_all, vectors[:,0])
z2_test = np.dot(test_all, vectors[:,1])

plt.figure()
for i in range(10):
    z1_digits_test = z1_test[test_all_no_deleted[:,-1]==i]
    z2_digits_test = z2_test[test_all_no_deleted[:,-1]==i]
    plt.scatter(z1_digits_test,z2_digits_test,label =""+str(i))
plt.title('Test Data after LDA')
plt.legend()

success_cnt = 0
for k in range(test_labels.shape[0]):
    distance =[]
    distance_1 = z1_test[k]-z1
    distance_2 = z2_test[k]-z2
    distance = distance_1*distance_1+distance_2*distance_2
    if(labels[np.argmin(distance)] == test_labels[k]):
        success_cnt = success_cnt+1 
        
test_error_LDA = 1-(success_cnt/test_all.shape[0])

print('Q2-LDA Test Error')
print(test_error_LDA)