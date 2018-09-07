#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:17:59 2018

@author: CiroGamJr
"""

#def events_map():
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
#from obspy import read_inventory, read_events
import pandas as pd
#import Image
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch


llon = -75.343051
ulon = -71.360507
llat = 5.605886
ulat = 8.179697

# Read the latitude and longitude of the events
events = pd.read_csv('event_locations.csv')
event_list = events.values.tolist()
lats = []
lons = []

# Read the latitude and longitude of the stations
stations = pd.read_csv('Ubicacion_Estaciones.csv',sep=';')
station_list = stations.values.tolist()
stat_names = []
stat_lats = []
stat_lons = []

for station in station_list:
    lat = station[1]
    lon = station[2]
    if((lat<ulat and lat>llat) and (lon<ulon and lon>llon)):
        stat_names.append(station[0])
        stat_lats.append(lat)
        stat_lons.append(lon)


for event in event_list:
    lats.append(event[0])
    lons.append(event[1])

# Set up a custom basemap, example is taken from basemap users' manual
fig, ax = plt.subplots()
 
m = Basemap(llcrnrlon = llon, llcrnrlat = llat, urcrnrlon = ulon
,urcrnrlat = ulat,  resolution = 'f'  )

x,y = m(lons,lats)

x_stat,y_stat = m(stat_lons,stat_lats)

m.fillcontinents(color='wheat', lake_color='skyblue')
m.drawmapboundary(fill_color='skyblue')


# Plot stations
m.plot(x_stat,y_stat,'bo',marker='^',color='b', label = "Estación sismológica")

# Put station names

import matplotlib.patheffects as PathEffects

fig.bmap = m
for label,xpt,ypt in zip(stat_names, x_stat, y_stat):
    txt = plt.text(xpt, ypt, label, color = 'white')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
#plt.show()


m.readshapefile('gadm36_COL_shp/gadm36_COL_2', 'comarques', drawbounds = False)
ax.set_title("Sismos en Santander")


municipios_santander   = []
otros_municipios = []

for info, shape in zip(m.comarques_info, m.comarques):
#    if info['nombre'] == 'Selva':
    if info['NAME_1'] == 'Santander':
        municipios_santander.append( Polygon(np.array(shape), True) )
        lats = [dot[1] for dot in shape ]
        lons = [dot[0] for dot in shape ]
        max_lat = max(lats)
        min_lat = min(lats)
        
        max_lon = max(lons)
        min_lon = min(lons)
        
        for event in event_list:
            lat_actual = event[0]
            lon_actual = event[1]
            
            if((lat_actual<max_lat and lat_actual>min_lat) and (lon_actual<max_lon and lon_actual>min_lon)):
            
                lats.append(lat_actual)
                lons.append(lon_actual)
    else:
        otros_municipios.append( Polygon(np.array(shape), True) )
        
ax.add_collection(PatchCollection(municipios_santander, facecolor= 'palegreen', edgecolor='k', linewidths=1., zorder=2))
#ax.add_collection(PatchCollection(otros_municipios, facecolor= 'green', edgecolor='k', linewidths=1., zorder=2))
# Plot events
m.plot(x,y,'bo',marker=',',color='red')

plt.show()
