#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:15:53 2018

En este código se extraen los sensores de cada estacion,
para posteriormente verificar si se tratan de triaxiles o monoaxilaes

@author: CiroGamJr
"""

# Import the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Estaciones_RSNC.csv',delimiter=";")
#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, 4].values
sensors_general = dataset["Tipo Sensor"].tolist()


# Create a list of unrepeated sensors
# Python code to remove duplicate elements
def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list
     
# Driver Code
sensors_filtered = Remove(sensors_general)
#print(sensors_filtered)

for el in sensors_filtered:
    print(el)

