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

#4.0017634,-77.9249418,5.91
#-12.7577632,-61.9736614,5.99
llon = -84.699735
ulon = -61.020440
llat = -4.987862
ulat = 13.498384

#llon = -75.343051
#ulon = -71.360507
#llat = 5.605886
#ulat = 8.179697

#13.498384, -84.699735
#-4.987862, -61.020440


fig, ax = plt.subplots()
m = Basemap(llcrnrlon = llon, llcrnrlat = llat, urcrnrlon = ulon
,urcrnrlat = ulat,  resolution = 'f'  )
m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= True)

#m.drawcoastlines()
m.drawcountries()
#m.drawmapboundary(fill_color='skyblue')
#m.fillcontinents(color='wheat', lake_color='skyblue')

#m.drawmapboundary(fill_color='skyblue')
#m.shadedrelief()
#m.etopo()

# Add Santander
m.readshapefile('gadm36_COL_shp/gadm36_COL_1', 'comarques', drawbounds = False)
ax.set_title("Estaciones sismológicas en Colombia")

deps_mostrar = []
deps_ocultar = []
for info, shape in zip(m.comarques_info, m.comarques):
#    if info['nombre'] == 'Selva':
    if info['NAME_1'] == 'Santander':
        deps_mostrar.append( Polygon(np.array(shape), True) )
    else:
        deps_ocultar.append( Polygon(np.array(shape), True) )
        
ax.add_collection(PatchCollection(deps_mostrar, facecolor = 'palegreen', alpha = 0.5, edgecolor='k', linewidths=1., zorder=1))


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

x_stat,y_stat = m(stat_lons,stat_lats)
m.plot(x_stat,y_stat,'bo',marker='^',color='b')

plt.show()


