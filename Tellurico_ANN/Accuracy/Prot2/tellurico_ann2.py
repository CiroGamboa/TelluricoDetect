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
dataset = pd.read_csv('attributes_matrix_6.csv')
X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

############################# MAKING THE ANN ##################################

#%% Evaluating, Improving and Tunning the ANN

#%% Evaluating the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from keras.models import Sequential
#from keras.layers import Dense
#
#def build_classifier():
#    # Initialising the ANN
#    classifier = Sequential()
#    
#    # Adding the input layer and the first hidden layer
#    classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 14))
#    
#    # Adding the second hidden layer
#    classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))
#    
#    # Adding the output layer
#    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#    
#    # Compiling the ANN
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    
#    return classifier
#
#
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#mean = accuracies.mean()
#variance = accuracies.std()


#%% Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 14))
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [10, 25, 32],
              'epochs' : [100, 50, 500],
              'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_


















