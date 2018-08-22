#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:38:18 2018

@author: CiroGamJr
"""
#http://abshinn.github.io/python/sklearn/2014/06/08/grid-searching-in-all-the-right-places/
########################### DATA PRE-PROCESSING ###############################
#%% Importing the libraries
#import multiprocessing
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

if __name__ == "__main__":
#    import numpy as np
#    import matplotlib.pyplot as plt
    import pandas as pd
#    multiprocessing.set_start_method('forkserver', force=True)

    
    #%% Importing the dataset
    dataset = pd.read_csv('attributes_matrix_prot04_3stats_0.9.csv')
    X = dataset.iloc[:, 0:42].values
    y = dataset.iloc[:, 42].values
    
    #%% Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    #%% Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    ############################# MAKING THE ANN ##################################
    
    
    #%% Tuning the ANN
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    
    def build_classifier(optimizer):
        # Initialising the ANN
        classifier = Sequential()
        
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim = 21, init = 'uniform', activation = 'relu', input_dim = 42))
        
        # Adding the second hidden layer
        classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
        
        # Adding the output layer
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return classifier
    
    
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size' : [8, 16, 32, 64, 128],
                  'epochs' : [10, 50, 100, 250, 500],
                  'optimizer' : ['adam', 'rmsprop','adadelta']}
    
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1,
                               refit = True)
    
    # Fitting the ANN to the training set
    grid_search = grid_search.fit(X_train, y_train)
    
    
    #%% Getting the best parameters
    best_parameters = grid_search.best_params_ 
    best_accuracy = grid_search.best_score_
    best_classifier = grid_search.best_estimator_
    
    #%% Predicting the Test set results
    y_pred = best_classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    #%% Making the confusion matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
    test_acc = accuracy_score(y_true = y_test, y_pred = y_pred)
    
    #%% Predicting the Training set results
    y_pred_train = best_classifier.predict(X_train)
    y_pred_train = (y_pred_train > 0.5)
    
    #%% Making the confusion matrix for the test set
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm_train = confusion_matrix(y_true = y_train,y_pred = y_pred_train)
    train_acc = accuracy_score(y_true = y_train, y_pred = y_pred_train) 
    
    
    
#%% Export the ANN
def export_ann(classifier):
    # Serialize model to JSON
    json_ann = best_classifier.model.to_json()
    with open("tellurico_ann_3s_09.json",'w') as json_file:
        json_file.write(json_ann)
        
    # Serialize weights to HDF5
    classifier.model.save_weights("tellurico_ann_3s_09.h5")
    
    
#%% Import the ANN
def import_ann(X_test,y_test):
    from keras.models import model_from_json
    # Load JSON and create model
    json_file = open("tellurico_ann_3s_09.json",'r')
    loaded_json_ann = json_file.read()
    json_file.close()
    loaded_classifier = model_from_json(loaded_json_ann)
    
    # Load weights into new model
    loaded_classifier.load_weights("tellurico_ann_3s_09.h5")
    
    
    # Evaluate loaded model on test data
    loaded_classifier.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop',
                              metrics = ['accuracy'])
    
    # Predicting the Test set results
    y_pred = loaded_classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    # Making the confusion matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
    test_acc = accuracy_score(y_true = y_test, y_pred = y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Test set accuracy")
    print(test_acc)
    return loaded_classifier
    
    
    

    



















