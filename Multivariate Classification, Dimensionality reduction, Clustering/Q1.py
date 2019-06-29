# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:32:22 2018

@author: mervet
"""

import numpy as np

class class_def:
    
    def __init__(self,class_def,number): # constructor
        self.class_def = class_def
        self.total_digitnumber = number
        
    def mean(self):
        return self.class_def.mean( axis=0 )
        
    def std(self):
        return self.class_def.std(axis = 0)

    def cov(self) :
        return np.cov(self.class_def.T)
    
    def prob(self):
        return self.class_def.shape[0]/self.total_digitnumber
    
    def inv(self):
        return np.linalg.inv(np.cov(self.class_def.T))
            
train_all=np.loadtxt("opt_digits_train.txt",dtype="int",delimiter=",")
train_no_deleted = train_all
var = train_all.std(axis = 0 )
var_zero_index =[] 

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

train_all = np.delete(train_all,-1,1)  

s = digit0.cov()*digit0.prob()+digit1.cov()*digit1.prob()+digit2.cov()*digit2.prob()+digit3.cov()*digit3.prob()+digit4.cov()*digit4.prob()+digit5.cov()*digit5.prob()+digit6.cov()*digit6.prob()+digit7.cov()*digit7.prob()+digit8.cov()*digit8.prob()+digit9.cov()*digit9.prob()

digits_mean = [digit0.mean(),digit1.mean(),digit2.mean(),digit3.mean(),digit4.mean(),digit5.mean(),digit6.mean(),digit7.mean(),digit8.mean(),digit9.mean()]
digits_prob = [digit0.prob(),digit1.prob(),digit2.prob(),digit3.prob(),digit4.prob(),digit5.prob(),digit6.prob(),digit7.prob(),digit8.prob(),digit9.prob()]
discrimant_func = []
accuracy_cnt = 0
confusian_matrix_train = np.zeros((10,10))
for i in range(train_all.shape[0]):
    discrimant_func = []
    for j in range(10):
       discrimant_func.append(-1/2*(train_all[i][:].reshape(1,62)-digits_mean[j].reshape(1,62)).dot(np.linalg.inv(s)).dot((train_all[i][:].reshape(62,1)-digits_mean[j].reshape(62,1)))+np.log(digits_prob[j]))
    if(train_no_deleted[i][-1] == np.argmax(discrimant_func,axis = 0)):
        accuracy_cnt = accuracy_cnt+1
        confusian_matrix_train[train_no_deleted[i][-1],np.argmax(discrimant_func,axis = 0)] += 1
    else:
        confusian_matrix_train[train_no_deleted[i][-1],np.argmax(discrimant_func,axis = 0)] += 1

train_error =1- accuracy_cnt/train_all.shape[0]

test_accuracy_cnt = 0 
test_all = np.loadtxt("opt_digits_test.txt",dtype="int",delimiter=",")
test_all_no_label = np.delete(test_all,-1,1)
test_all_no_label = np.delete(test_all_no_label,var_zero_index[0],1)  
test_all_no_label = np.delete(test_all_no_label,(var_zero_index[1]-1),1) 
test_labels = np.zeros((test_all_no_label.shape[0],1))
confusian_matrix_test = np.zeros((10,10))
for i in range(test_all_no_label.shape[0]):
    discrimant_func = []
    for j in range(10):
       discrimant_func.append(-1/2*(test_all_no_label[i][:].reshape(1,62)-digits_mean[j].reshape(1,62)).dot(np.linalg.inv(s)).dot((test_all_no_label[i][:].reshape(62,1)-digits_mean[j].reshape(62,1)))+np.log(digits_prob[j]))
    test_labels[i] = np.argmax(discrimant_func,axis = 0)
    if(test_all[i][-1] == np.argmax(discrimant_func,axis = 0)):
        confusian_matrix_test[test_all[i][-1],np.argmax(discrimant_func,axis = 0)] += 1
        test_accuracy_cnt = test_accuracy_cnt+1
    else:
        confusian_matrix_test[test_all[i][-1],np.argmax(discrimant_func,axis = 0)] += 1

        
test_error =1- test_accuracy_cnt/test_all_no_label.shape[0]

print('Q1-Multivariate Analisis Train and Test Error')
print(train_error)
print(test_error)

