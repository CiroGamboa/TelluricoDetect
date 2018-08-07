#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:38:18 2018

@author: CiroGamJr
"""

########################### DATA PRE-PROCESSING ###############################
#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('attributes_matrix_prot01.csv')
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

############################# MAKING THE ANN ##################################
#%% Importing the Keras libraries and packages
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#
##%% Initialising the ANN
#classifier = Sequential()
#
## Adding the input layer and the first hidden layer
##classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu', input_dim = 2))
#classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 2))
#
## Adding the second hidden layer
#classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
#
## Adding the output layer
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#
##%% Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
##%% Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


#%% Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 50)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 2, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


#%% Predicting the Test set results
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#
##%% Predict a single new observation
#"""
#DOP : 0.86746383
#RV2T : 0.7856372
#"""
#new_prediction = classifier.predict(sc.transform(np.array([[0.86746383,0.7856372]])))
#new_prediction = (new_prediction > 0.5)


