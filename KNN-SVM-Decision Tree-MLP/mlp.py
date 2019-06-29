# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 22:05:03 2018

@author: mervet
"""

import numpy as np

from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.core import Activation
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def decode(data):
    return np.argmax(data)

batch_size = 128
num_classes = 10
epochs = 20
train_all = np.loadtxt("opt_digits_train.txt", dtype="int", delimiter=",")
train_labels = train_all[:, -1]
test_all = np.loadtxt("opt_digits_test.txt", dtype="int", delimiter=",")
test_labels = test_all[:, -1]
train_all = np.delete(train_all, -1, 1)
test_all = np.delete(test_all, -1, 1)

train_all = train_all.astype('float32')
test_all = test_all.astype('float32')

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)
print(train_labels)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(64,)))
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_all.reshape((-1, 8 * 8)), train_labels_cat,
          batch_size=32, nb_epoch=10,
          verbose=1, callbacks=[],
          validation_data=None,
          class_weight=None,
          sample_weight=None)

decoded_data = []

predicted_train = model.predict(train_all)
predicted_train_decode = np.zeros((predicted_train.shape[0]))
for i in range(predicted_train.shape[0]):
    datum = predicted_train[i]
    predicted_train_decode[i] = decode(predicted_train[i])


predicted_test = model.predict(test_all)
predicted_test_decode = np.zeros((predicted_test.shape[0]))
for i in range(predicted_test.shape[0]):
    datum = predicted_test[i]
    predicted_test_decode[i] = decode(predicted_test[i])

con_mat_train = confusion_matrix(train_labels, predicted_train_decode, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
con_mat_test = confusion_matrix(test_labels, predicted_test_decode, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

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
equal_weight_accuracy_train = equal_weight_accuracy_train / 10
for i in range(10):
    total_accuracy_cnt_test += (con_mat_test[i, i])

total_accuracy_test = total_accuracy_cnt_test / float(np.sum(con_mat_test))
equal_test = []
for i in range(10):
    equal_weight_accuracy_test += con_mat_test[i, i] / float(np.sum(con_mat_test[i, :]))
    equal_test.append(con_mat_test[i, i] / float(np.sum(con_mat_test[i, :])))
equal_weight_accuracy_test = equal_weight_accuracy_test / 10


