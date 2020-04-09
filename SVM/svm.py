# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:57:51 2018

@author: mervet
"""

from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix

train_all=np.loadtxt("opt_digits_train.txt",dtype="int",delimiter=",")
train_labels = train_all[:,-1]
test_all = np.loadtxt("opt_digits_test.txt",dtype="int",delimiter=",")
test_labels = test_all[:,-1]
train_all = np.delete(train_all,-1,1) 
test_all = np.delete(test_all,-1,1) 


model = svm.SVC(kernel='linear') 
model.fit(train_all, train_labels)

predicted_train= model.predict(train_all)
predicted_test = model.predict(test_all)
con_mat_train = confusion_matrix(train_labels, predicted_train, [0, 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9])
con_mat_test = confusion_matrix(test_labels, predicted_test, [0, 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9])

total_accuracy_cnt_train = 0
total_accuracy_cnt_test = 0
equal_weight_accuracy_train = 0
equal_weight_accuracy_test = 0

for i in range(10):
    total_accuracy_cnt_train += (con_mat_train[i, i])
    
total_accuracy_train = total_accuracy_cnt_train / float(np.sum(con_mat_train))

equal_train = []
for i in range(10):
    equal_weight_accuracy_train += con_mat_train[i, i] / float(np.sum(con_mat_train[i, :]))
    equal_train.append(con_mat_train[i, i] / float(np.sum(con_mat_train[i, :])))
equal_weight_accuracy_train = equal_weight_accuracy_train/10
for i in range(10):
    total_accuracy_cnt_test += (con_mat_test[i, i])
    
total_accuracy_test = total_accuracy_cnt_test / float(np.sum(con_mat_test))

equal_test = []
for i in range(10):
    equal_weight_accuracy_test += con_mat_test[i, i] / float(np.sum(con_mat_test[i, :]))
    equal_test.append(con_mat_test[i, i] / float(np.sum(con_mat_test[i, :])))
equal_weight_accuracy_test = equal_weight_accuracy_test/10


