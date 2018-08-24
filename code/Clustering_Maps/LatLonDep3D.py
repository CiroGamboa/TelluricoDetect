#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:06:28 2018

@author: CiroGamJr
"""

'''
INCOMPLETO: AGRUPAR SISMOS POR POSICION Y CAMBIAR EL TAMAÑO DEL MARCADOR HACIENDOLO MAS
GRANDE ENTRE MAS SISMOS TENGA
COLOREAR GRUPOS DE SISMOS DE ACUERDO A EPICENTROS SELECCIONADOS
'''

#def graph_LatLonDepth(events_file):
    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm


events = pd.read_csv('event_locations3D.csv')
event_list = events.values.tolist()
lats = []
lons = []
depths = []

for event in event_list:
    conversion = utm.from_latlon(event[0],event[1])
    lats.append(conversion[0])
    lons.append(conversion[1])
    depths.append(event[2])
     
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#        xs = randrange(n, 23, 32)
#        ys = randrange(n, 0, 100)
#        zs = randrange(n, zlow, zhigh)
#        ax.scatter(xs, ys, zs, c=c, marker=m)

#    xs = np.asarray(lats)
#    ys = np.asarray(lons)
#    zs = np.asarray(depths)

xs = lats
ys = lons
zs = depths

c = 'r'
m = '.'
ax.scatter(xs, ys, zs, c=c, marker=m, norm = True)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()