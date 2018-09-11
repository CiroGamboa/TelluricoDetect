#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:59:24 2018

@author: tellurico
"""

from TelluricoANN import TelluricoANN
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0) 
#import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def import_ann(filename,loss_fn,optimizer,metric):
        
        from keras.models import model_from_json
        
        # Load JSON and create model
#        json_file = open("tellurico_ann3.json",'r')
        json_file = open(filename+".json",'r')

        loaded_json_ann = json_file.read()
        json_file.close()
        loaded_classifier = model_from_json(loaded_json_ann)
        
        # Load weights into new model
        loaded_classifier.load_weights(filename+".h5")
        
        # Evaluate loaded model on test data
        loaded_classifier.compile(loss = loss_fn, optimizer = optimizer,
                                  metrics = [metric])
        
        return loaded_classifier
    
dataset_folder = 'ultimate_datasets/'
computed_anns_folder = 'computed_anns/'



# 1 station
ann_1s = import_ann(filename = computed_anns_folder+'tellurico_ann_1s_05',
                    loss_fn = 'binary_crossentropy',
                    optimizer = 'adadelta',
                    metric = 'accuracy')

dataset_1s = pd.read_csv(dataset_folder+'attributes_matrix_prot04_1stats_0.5.csv')
X_1s = dataset_1s.iloc[:, 0:14].values
y_1s = dataset_1s.iloc[:, 14].values

X_train_1s, X_test_1s, y_train_1s, y_test_1s = train_test_split(X_1s, y_1s, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_1s = sc.fit_transform(X_train_1s)
X_test_1s = sc.transform(X_test_1s)

# 2 stations
ann_2s = import_ann(filename = computed_anns_folder+'tellurico_ann_2s_05',
                    loss_fn = 'binary_crossentropy',
                    optimizer = 'adadelta',
                    metric = 'accuracy')

dataset_2s = pd.read_csv(dataset_folder+'attributes_matrix_prot04_2stats_0.5.csv')
X_2s = dataset_2s.iloc[:, 0:28].values
y_2s = dataset_2s.iloc[:, 28].values

X_train_2s, X_test_2s, y_train_2s, y_test_2s = train_test_split(X_2s, y_2s, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_2s = sc.fit_transform(X_train_2s)
X_test_2s = sc.transform(X_test_2s)


# 3 stations
ann_3s = import_ann(filename = computed_anns_folder+'tellurico_ann_3s_05',
                    loss_fn = 'binary_crossentropy',
                    optimizer = 'adadelta',
                    metric = 'accuracy')

dataset_3s = pd.read_csv(dataset_folder+'attributes_matrix_prot04_3stats_0.5.csv')
X_3s = dataset_3s.iloc[:, 0:42].values
y_3s = dataset_3s.iloc[:, 42].values

X_train_3s, X_test_3s, y_train_3s, y_test_3s = train_test_split(X_3s, y_3s, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_3s = sc.fit_transform(X_train_3s)
X_test_3s = sc.transform(X_test_3s)


# 4 stations
ann_4s = import_ann(filename = computed_anns_folder+'tellurico_ann_4s_05',
                    loss_fn = 'binary_crossentropy',
                    optimizer = 'adadelta',
                    metric = 'accuracy')

dataset_4s = pd.read_csv(dataset_folder+'attributes_matrix_prot04_4stats_0.5.csv')
X_4s = dataset_4s.iloc[:, 0:55].values
y_4s = dataset_4s.iloc[:, 55].values

X_train_4s, X_test_4s, y_train_4s, y_test_4s = train_test_split(X_4s, y_4s, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_4s = sc.fit_transform(X_train_4s)
X_test_4s = sc.transform(X_test_4s)



import time
import numpy as np
import scipy.io as io
cantidad_observaciones = 1000
cantidad_estaciones = 4
iteraciones = 1000
acumul = 0.0
times = np.zeros((cantidad_observaciones,cantidad_estaciones))
for i in range(1, cantidad_observaciones+1):
    for stats_q in range(1, cantidad_estaciones+1):
        for ii in range(0, iteraciones):
            start = time.time()
            
            if stats_q == 1:
                y_pred = ann_1s.predict(X_test_1s[:i])
            elif stats_q == 2:
                y_pred = ann_2s.predict(X_test_2s[:i])
            elif stats_q == 3:
                y_pred = ann_3s.predict(X_test_3s[:i])
            elif stats_q == 4:
                y_pred = ann_4s.predict(X_test_4s[:i])
            
            
            end = time.time()
            acumul += (end - start)
        times[i-1][stats_q-1] = acumul/iteraciones
        acumul = 0.0

io.savemat('/home/tellurico/Tellurico/Variables/timesClassifier.mat',mdict={'times':times})







