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

# setup albers equal area conic basemap
# lat_1 is first standard parallel.
# lat_2 is second standard parallel.
# lon_0, lat_0 is central point.
#    m = Basemap(width=8000000, height=7000000,
#                resolution='c', projection='aea',
#                lat_1=40., lat_2=60, lon_0=25, lat_0=40, ax=ax)

#m = Basemap(llcrnrlon=3.75,llcrnrlat=39.75,urcrnrlon=4.35,urcrnrlat=40.15, resolution = 'h', epsg=5520)
#m = Basemap(llcrnrlon= -75,llcrnrlat= -2,urcrnrlon=-72,urcrnrlat=-12, resolution = 'h', epsg=5520)
   
m = Basemap(llcrnrlon = llon, llcrnrlat = llat, urcrnrlon = ulon
,urcrnrlat = ulat,  resolution = 'f'  )

x,y = m(lons,lats)

x_stat,y_stat = m(stat_lons,stat_lats)

#m.drawcoastlines()
#m.drawcountries()
#m.drawstates()
m.fillcontinents(color='wheat', lake_color='skyblue')
#m.etopo()
# draw parallels and meridians.
#m.drawparallels(np.arange(-80., 81., 20.))
#m.drawmeridians(np.arange(-180., 181., 20.))
m.drawmapboundary(fill_color='skyblue')

# Plot events
m.plot(x,y,'bo',marker=',',color='red')

# Plot stations
m.plot(x_stat,y_stat,'bo',marker='^',color='b')

# Put station names
fig.bmap = m
for label,xpt,ypt in zip(stat_names, x_stat, y_stat):
    plt.text(xpt, ypt, label)
plt.show()


m.readshapefile('gadm36_COL_shp/gadm36_COL_2', 'comarques', drawbounds = False)
ax.set_title("Sismos en Santander")


municipios_santander   = []
otros_municipios = []

for info, shape in zip(m.comarques_info, m.comarques):
#    if info['nombre'] == 'Selva':
    if info['NAME_1'] == 'Santander':
        municipios_santander.append( Polygon(np.array(shape), True) )
    else:
        otros_municipios.append( Polygon(np.array(shape), True) )
        
ax.add_collection(PatchCollection(municipios_santander, facecolor= 'palegreen', edgecolor='k', linewidths=1., zorder=2))
ax.add_collection(PatchCollection(otros_municipios, facecolor= 'green', edgecolor='k', linewidths=1., zorder=2))


# we need to attach the basemap object to the figure, so that obspy knows about
# it and reuses it

#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
#import numpy as np
#
#map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
#    resolution = 'h', area_thresh = 0.1,
#    llcrnrlon=-136.25, llcrnrlat=56.0,
#    urcrnrlon=-134.25, urcrnrlat=57.75)
#
#map.drawcoastlines()
#map.drawcountries()
#map.fillcontinents(color = 'coral')
#map.drawmapboundary()
#
#lons = [-135.3318, -134.8331, -134.6572]
#lats = [57.0799, 57.0894, 56.2399]
#x,y = map(lons, lats)
#map.plot(x, y, 'bo', markersize=18)
#
#labels = ['Sitka', 'Baranof Warm Springs', 'Port Alexander']
#for label, xpt, ypt in zip(labels, x, y):
#    plt.text(xpt+10000, ypt+5000, label)
#
#plt.show()