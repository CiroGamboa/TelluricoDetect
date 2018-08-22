#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SI EL FACTOR DE AGRUPACION ES 5 O MENOS, PONER VALORES EN LA PUNTA D ELAS BARRAS
FILTRAR SFILES Y ELIMINAR FANTASMAS, SACAR LISTA DE NOMBRES DE WAVEFORMS


"""

#%%
'''
1. Amount of seisms vs Station
COMPLETO

https://matplotlib.org/users/colors.html
'''
def seisms_per_station(sfiles,group_factor=10,save_graphs=False):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import operator
        import os
#        import numpy as np
        
        mpl.style.use('seaborn')
        color = [0.26682728, 0.62113383, 0.52914209]

        events_per_station = {}
        amount_events = len(sfiles)
        
        for sfile in sfiles:
            last_station_name = " "
            stations = sfile.type_7
            for station in stations:
                try:
                    station_name = station['STAT']
                    
                    if station_name not in events_per_station:
                        events_per_station[station_name] = 1
                    else:
                        if station_name != last_station_name:
                            events_per_station[station_name] = events_per_station[station_name] + 1
                            last_station_name = station_name
                except:
                    pass
                        
        # This is a workaround, should be improved
        event_list = sorted(events_per_station.items(), key=operator.itemgetter(1))
        event_list.reverse()
        event_dict = {}
        
        segmented_graphs = []
        index = 0
        list_index = -1
        for event in event_list:
            
            if(index%group_factor == 0):
                segmented_graphs.append({})
                list_index += 1
                
            event_dict[event[0]] = event[1]
            segmented_graphs[list_index][event[0]] = event[1]
            index += 1
            
        #print(segmented_graphs)  
        
        plt.figure()
        plt.title("Sismos por estación 2010-2017, total eventos: "+str(amount_events), weight='bold')
        plt.bar(range(len(event_dict)), event_dict.values(), align='center',color=color)
        plt.xlabel("Indice de la estación",weight='bold')
        plt.ylabel("Cantidad de eventos",weight='bold')
        plt.grid(True)
        
        path = "IOfiles/Graphs/"
        if(save_graphs):   
            if not os.path.exists(path):
                os.makedirs(path)           
            plt.savefig(path+"SeismsXstationAll.png", dpi = 500)
        
        index = 0
        for graph in segmented_graphs:
            plt.figure()
            group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
            plt.title("Sismos por estación 2010-2017 ["+group_name+"], total eventos: "+str(amount_events),weight='bold')
            rects = plt.bar(range(len(graph)), graph.values(), align='center',color=color)
            plt.xticks(range(len(graph)), graph.keys())
            plt.xlabel("Indice de la estación",weight='bold')
            plt.ylabel("Cantidad de eventos",weight='bold')
            plt.grid(True)
            index += 1
            
            if(group_factor <= 15):
                for rect in rects:
                    height = rect.get_height()
                    percentage = (height*100)/amount_events
                    plt.text(rect.get_x() + rect.get_width()/2., height,
                            '%d' % int(percentage) + "%", ha='center', va='bottom')
                
            
            if(save_graphs):
                plt.savefig(path+"SeismsXstation"+group_name+".png", dpi = 500)
            
#%%
'''
2. Amount of seisms vs Station per Epicenter
INCOMPLETO: FALTA GUARDAR, USAR EL TRY-CATCH PARA DESCARTAR SFILES
- RANDOM SEED
'''
# Plot the events registered by stations in descendent order
def seisms_per_stationEpis(sfiles,group_factor=10,num_epis=10,save_graphs=False,exclude_main=False):
    import operator
    import numpy as np
    
    soft_colors = ['silver','tomato','purple','aqua','teal','khaki','orange','turquoise','yellowgreen','pink',]
    epis_in = get_epi_loc(sfiles)[0]
    used_epis = epis_in[:num_epis]
    used_epis = [x[0] for x in used_epis]

    if(exclude_main):
        used_epis.pop(0)
#    print(used_epis)
    events_per_station = {}
    events_quant = {}
    for sfile in sfiles:
        last_station_name = " "
        stations = sfile.type_7
        
        # There are Sfiles without this attribute, shall ignore them
        try:
            epistring = sfile.type_3['EPICENTER_LOCATION']
            ismain = (epistring == "LOS SANTOS - SANTANDER")
#            print(epistring)
#            if(ismain):
#                print("YA")
            
            if(not(exclude_main and ismain)):
                for station in stations:
                    station_name = station['STAT']
                    
                    if station_name not in events_per_station:
                        events_per_station[station_name] = {x:0 for x in used_epis}
                        events_quant[station_name] = 1
                        
                    else:
                        if station_name != last_station_name:
                            if epistring in used_epis:
    #                                if epistring not in events_per_station[station_name]:
    #                                    events_per_station[station_name][epistring] = 1
    #                                else:
                                events_per_station[station_name][epistring] += 1
                                events_quant[station_name] += 1
                                last_station_name = station_name                       
        except:
            pass
        
    #return events_quant
                        
        # This is a workaround, should be improved
    event_list = sorted(events_quant.items(), key=operator.itemgetter(1))
    event_list.reverse()
    event_dict = {}
    
    segmented_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            list_index += 1
            
        event_dict[event[0]] = event[1]
        segmented_graphs[list_index][event[0]] = event[1]
        index += 1
        
    #print(segmented_graphs)  
    
#    colors = [np.random.rand(3,) for i in range(0,num_epis)]
    colors = soft_colors[:num_epis]
    for segmented_graph in segmented_graphs:
        epis = {}
        for station in segmented_graph:
            if station not in epis:
                epis[station] = events_per_station[station]

        mul_bar_graph(epis, colors)
                
            
        
#        plt.figure()
#        plt.title("Seisms per station 2010-2017")
#        plt.bar(range(len(event_dict)), event_dict.values(), align='center')
#        plt.xlabel("Station index")
#        plt.ylabel("Amount of events")
#        
#        
# fig, ax1 = plt.subplots()
#    
#    x1 = [x - 0.4 for x in range(len(dis_dict))]
#    
#    color = 'tab:red'
#    ax1.set_xlabel('Station Index: Ordered by amount of seisms')
#    ax1.set_ylabel('Amount of Seisms', color=color)
#    ax1.bar(x1, amount_dict.values(), color=color, width = 0.4, align = 'edge')
#    ax1.tick_params(axis='y', labelcolor=color)
#    
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    
#    
#    color = 'tab:blue'
#    ax2.set_ylabel('Mean epicenter distance', color=color)  # we already handled the x-label with ax1
#    ax2.bar(range(len(dis_dict)),dis_dict.values(), color=color, width = 0.4, align = 'edge')
#    ax2.tick_params(axis='y', labelcolor=color)
#    
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
#    plt.title("Mean Dis VS Amount of Seisms per station 2010-2017")
        
        
#        
#        path = "IOfiles/Graphs/"
#        if(save_graphs):   
#            if not os.path.exists(path):
#                os.makedirs(path)           
#            plt.savefig(path+"SeismsXstationAll.png")
#        
#        index = 0
#        for graph in segmented_graphs:
#            plt.figure()
#            group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
#            plt.title("Seisms per station 2010-2017 ["+group_name+"]")
#            plt.bar(range(len(graph)), graph.values(), align='center')
#            plt.xticks(range(len(graph)), graph.keys())
#            plt.xlabel("Station name")
#            plt.ylabel("Amount of events")
#            index += 1
#            
#            if(save_graphs):
#                plt.savefig(path+"SeismsXstation"+group_name+".png")
            
      
    return events_per_station


def mul_bar_graph(epis = None,colors = None):
    import numpy as np
    import matplotlib.pyplot as plt
    
    
        #epis = {'EST1':{'A':3,'B':4,'C':2,'D':6},'EST2':{'A':1,'B':7,'C':8,'D':6},'EST3':{'A':8,'B':5,'C':1,'D':6}}
    if(epis == None):
        epis = {'EST1':{'A':3,'B':4,'C':1},'EST2':{'A':1,'B':7,'C':1},'EST3':{'A':8,'B':5,'C':1}}

    
    epi_groups = {}
    num_epis = 0
    for est in epis:
        actual_length = len(epis[est])
        if num_epis < actual_length:
            num_epis = actual_length
        for epi in epis[est]:
            if epi in epi_groups:
                epi_groups[epi].append(epis[est][epi])
            else:
                epi_groups[epi] = [epis[est][epi]]
        
    
    width = 0.9/num_epis
    ind = np.arange(len(epis))
    
    fig, ax = plt.subplots()
    
    rects = []
    delta_width = -(width*(num_epis)/2)
    
    
    if(colors == None):
        colors = [np.random.rand(3,) for i in range(0,len(epi_groups))]
    color_index = 0
    for key in epi_groups:
        rects.append(ax.bar(ind  + delta_width, epi_groups[key], width, color=colors[color_index], align='edge'))
        delta_width += width
        color_index += 1
    
    
    ax.grid(True)
    ax.set_ylabel('Amount of seisms per epicenter')
    ax.set_title('Amount of seisms per epicenter registered per station')
    ax.set_xticks((ind + (width*num_epis) / num_epis)-width)
    ax.set_xticklabels(est for est in epis)
    plt.xlabel("Station name")
    ax.legend([rect[0] for rect in rects],[epi for epi in epi_groups])

'''
ESTAS WEIRD FILES DEBEN SER UNA VALIDACION ADICIONAL EN SFILEANALYZER
'''
def get_epi_loc(sfiles):
    import operator
    epis = {}
    weird_sfiles = []
    for sfile in sfiles:
        try:
            epistring = sfile.type_3['EPICENTER_LOCATION']
            if(epistring not in epis):
                epis[epistring] = 1
            else:
                epis[epistring] += 1
        except:
            weird_sfiles.append(sfile.filename)
    
    epis_nums = sorted(epis.items(), key=operator.itemgetter(1))
    return [epis_nums[::-1],weird_sfiles]
   

#%%
'''
3. Amount of seisms vs Epicenter
INCOMPLETO: TAL VEZ ES BUENA IDEA ACORTAR LOS NOMBRES QUITANDO 'SANTANDER',
PERO HABRIA QUE TENER CUIDADO CON EPICENTROS FUERA DE SANTANDER 
'''
def events_per_epicenter(sfiles,group_factor=10,save_graphs=False):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    np.random.seed(1)
    color = np.random.rand(3,)
    
    event_list = get_epi_loc(sfiles)[0]
    print(event_list)
    event_dict = {}
    
    segmented_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            list_index += 1
            
        event_dict[event[0]] = event[1]
        segmented_graphs[list_index][event[0]] = event[1]
        index += 1
        
    #print(segmented_graphs)  
    
    plt.figure()
    plt.title("Seisms per Epicenter 2010-2017")
    plt.bar(range(len(event_dict)), event_dict.values(), align='center', color = color)
    plt.xlabel("Epicenter index")
    plt.ylabel("Amount of events")
    
    path = "IOfiles/Graphs/"
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"SeismsXstationAll.png")
    
    index = 0
    for graph in segmented_graphs:
        plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        plt.title("Seisms per station 2010-2017 ["+group_name+"]")
        plt.bar(range(len(graph)), graph.values(), align='center', color = color)
        keys = []
        for key in graph.keys():
            try:
                st = key.split('-')
                keys.append(st[0]+'\n'+st[1])
            except:
                keys.append(key)
        plt.xticks(range(len(graph)), keys)
        plt.xlabel("Epicenter name")
        plt.xticks(rotation=45)
        plt.ylabel("Amount of events")
        index += 1
        
        if(save_graphs):
            plt.savefig(path+"SeismsXstation"+group_name+".png")
        
        
    # Es buena idea verificar la fecha minima y la maxima para ponerlas como
    # parametro de entrada
    
    #print("Events per Station between 2010 and 2017") 
    #for event in event_dict:
     #   print("Station name: "+event+"\tEvents:\t"+str(event_dict[event]))
        
    return event_dict



#%%
'''
4. Amount of seisms vs Magnitude value
INCOMPLETO: LOS VALORES EN EL EJE X ESTAN MULTIPLICADOS POR 10 (6.0-->60),
DETALLES ESTETICOS
- NOMBRES
- DEGRADADO EN EL COLOR
- DIVIDIR EN 10
- PONER LINEA DE LA MEDIA
- ENVOLVENTE
- CAMBIAR EJE POR PORCENTAJE
- 
'''
def magnitude_values(sfiles):
#    https://matplotlib.org/gallery/statistics/hist.html
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter

    # Fixing random state for reproducibility
    np.random.seed(19680801)
    undefined_group = []
    magnitudes = []
    for sfile in sfiles:
            
        try:  
            mag = float(sfile.type_1['TYPE_OF_MAGNITUDE_1'])
            if(mag is not None):
                magnitudes.append(mag*10)
                
        except:
            undefined_group.append(sfile)

    fig, axs = plt.subplots(tight_layout=True)
    x_axis = np.arange(0,66,1)
    N, bins, patches = plt.hist(magnitudes,bins = x_axis)
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    axs.grid(True)
    axs.hist(magnitudes,bins = x_axis, density=True)
#    plt.plot(magnitudes)
#    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xlabel("Seismic magnitude [Richter Scale]")
    plt.ylabel("% of occurrency")
    plt.title("Distribution of Seismic magnitudes in Santander [2010-2017]")
#    plt.show()
#    return [magnitudes,hist]



#fig, axs = plt.subplots(1, 2, tight_layout=True)
#
## N is the count in each bin, bins is the lower-limit of the bin
#N, bins, patches = axs[0].hist(x, bins=n_bins)
#
## We'll color code by height, but you could use any scalar
#fracs = N / N.max()
#
## we need to normalize the data to 0..1 for the full range of the colormap
#norm = colors.Normalize(fracs.min(), fracs.max())
#
## Now, we'll loop through our objects and set the color of each accordingly
#for thisfrac, thispatch in zip(fracs, patches):
#    color = plt.cm.viridis(norm(thisfrac))
#    thispatch.set_facecolor(color)
#
## We can also normalize our inputs by the total number of counts
#axs[1].hist(x, bins=n_bins, density=True)
#
## Now we format the y-axis to display percentage
#axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

#%%
'''
4. Amount of seisms vs Magnitude value
INCOMPLETO: LOS VALORES EN EL EJE X ESTAN MULTIPLICADOS POR 10 (6.0-->60),
DETALLES ESTETICOS
- NOMBRES
- DEGRADADO EN EL COLOR
- DIVIDIR EN 10
- PONER LINEA DE LA MEDIA
- ENVOLVENTE
- CAMBIAR EJE POR PORCENTAJE
- 
'''
def depth_values(sfiles):
#    https://matplotlib.org/gallery/statistics/hist.html
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
#    from matplotlib.ticker import PercentFormatter

    # Fixing random state for reproducibility
    np.random.seed(19680801)
    depths = []
    for sfile in sfiles:
            
        try:  
            depth = float(sfile.type_1['DEPTH'])
            depths.append(depth)
                
        except:
            pass

    fig, axs = plt.subplots(tight_layout=True)
    x_axis = np.arange(0,200,0.1)
#    N, bins, patches = plt.hist(depths,bins = x_axis)
#    fracs = N / N.max()
#    norm = colors.Normalize(fracs.min(), fracs.max())
#    for thisfrac, thispatch in zip(fracs, patches):
#        color = plt.cm.viridis(norm(thisfrac))
#        thispatch.set_facecolor(color)
    
    axs.grid(True)
    axs.hist(depths,bins = x_axis, density=True)
#    plt.plot(magnitudes)
#    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xlabel("Seismic depths [Km]")
    plt.ylabel("% of occurrency")
    plt.title("Distribution of Seismic Depths in Santander [2010-2017]")

#%%
'''
4. Amount of seisms vs Magnitude value
INCOMPLETO: LOS VALORES EN EL EJE X ESTAN MULTIPLICADOS POR 10 (6.0-->60),
DETALLES ESTETICOS
- NOMBRES
- DEGRADADO EN EL COLOR
- DIVIDIR EN 10
- PONER LINEA DE LA MEDIA
- ENVOLVENTE
- CAMBIAR EJE POR PORCENTAJE
- 
'''
def cumulative_depth_values(sfiles):
#    https://matplotlib.org/gallery/statistics/hist.html
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
#    from matplotlib.ticker import PercentFormatter

    # Fixing random state for reproducibility
    np.random.seed(19680801)
    depths = []
    for sfile in sfiles:
            
        try:  
            depth = float(sfile.type_1['DEPTH'])
            depths.append(depth)
                
        except:
            pass

    fig, axs = plt.subplots(tight_layout=True)
    x_axis = np.arange(0,200,0.1)
#    N, bins, patches = plt.hist(depths,bins = x_axis)
#    fracs = N / N.max()
#    norm = colors.Normalize(fracs.min(), fracs.max())
#    for thisfrac, thispatch in zip(fracs, patches):
#        color = plt.cm.viridis(norm(thisfrac))
#        thispatch.set_facecolor(color)
    
    axs.grid(True)
    axs.hist(depths,bins = x_axis, density=True,cumulative=1)
#    plt.plot(magnitudes)
#    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xlabel("Seismic depths [Km]")
    plt.ylabel("% of occurrency")
    plt.title("Distribution of Seismic Depths in Santander [2010-2017]")






#%%
'''
5. Mean depth vs Epicenter
INCOMPLETO: DETALLES ESTETICOS MENORES, CAMBIAR NOMBRES A EJES, CORREGIR
LONGITUD DE NOMBRES EN EJE X
'''
def meanDepth_per_epicenter(sfiles,group_factor=5,save_graphs=False):
    import matplotlib.pyplot as plt
    import operator
    import os
    epis = {}
    weird_sfiles = []
    for sfile in sfiles:
        try:
            epistring = sfile.type_3['EPICENTER_LOCATION']
            depth = float(sfile.type_1['DEPTH'])
            
            if(epistring not in epis):
                epis[epistring] = depth
            else:
                epis[epistring] += depth
                epis[epistring] /= 2
        except:
            weird_sfiles.append(sfile.filename)
    
    event_list = sorted(epis.items(), key=operator.itemgetter(1))
    event_list.reverse()
    
    event_dict = {}
        
    segmented_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            list_index += 1
            
        event_dict[event[0]] = event[1]
        segmented_graphs[list_index][event[0]] = event[1]
        index += 1
        
    #print(segmented_graphs)  
    
    plt.figure()
    plt.title("Seisms per Epicenter 2010-2017")
    plt.bar(range(len(event_dict)), event_dict.values(), align='center')
    plt.xlabel("Epicenter index")
    plt.ylabel("Amount of events")
    
    path = "IOfiles/Graphs/"
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"SeismsXstationAll.png")
    
    index = 0
    for graph in segmented_graphs:
        plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        plt.title("Seisms per station 2010-2017 ["+group_name+"]")
        plt.bar(range(len(graph)), graph.values(), align='center')
        plt.xticks(range(len(graph)), graph.keys())
        plt.xlabel("Epicenter name")
        plt.xticks(rotation=45)
        plt.ylabel("Amount of events")
        index += 1
        
        if(save_graphs):
            plt.savefig(path+"SeismsXstation"+group_name+".png")
        
        
    # Es buena idea verificar la fecha minima y la maxima para ponerlas como
    # parametro de entrada
    
    #print("Events per Station between 2010 and 2017") 
    #for event in event_dict:
     #   print("Station name: "+event+"\tEvents:\t"+str(event_dict[event]))
        
    return event_dict


#%%
'''
6. Latitude vs Longitude vs Depth vs Epicenters
INCOMPLETO: FALTA TRANSFORMAR LAS COORDENADAS A CARTESIANAS Y COLOREAR 
POR GRUPOS (EPICENTROS)
'''
def graph_LatLonDepth(sfiles):
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    
    lats = []
    lons = []
    depths = []
    
    for sfile in sfiles:
        lat = sfile.type_1['LATITUDE']
        lon = sfile.type_1['LONGITUDE']
        depth = sfile.type_1['DEPTH']
        
        lats.append(float(lat))
        lons.append(float(lon))
        depths.append(float(depth))
         
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
    m = 'o'
    ax.scatter(xs, ys, zs, c=c, marker=m, norm = True)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()

#%%
'''
7. Latitude vs Longitude Clustering
'''

#%%
'''
8. Latitude vs Longitude in map
INCOMPLETO: SOLO SE LLEVA EL MAPA DE SANTANDER, NO SE HAN GRAFICADO LOS EVENTOS
ES POSIBLE QUE HAYA QUE AGRUPAR LOS EVENTOS POR UBICACION
'''
def events_map():
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    
    from obspy import read_inventory, read_events
    
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
   
    m = Basemap(llcrnrlon = -75.343051, llcrnrlat = 5.605886, urcrnrlon = -71.360507
,urcrnrlat = 8.179697,  resolution = 'h'  )

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.fillcontinents(color='wheat', lake_color='skyblue')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-80., 81., 20.))
    m.drawmeridians(np.arange(-180., 181., 20.))
    m.drawmapboundary(fill_color='skyblue')
    ax.set_title("Albers Equal Area Projection")
    
    # we need to attach the basemap object to the figure, so that obspy knows about
    # it and reuses it
    fig.bmap = m
    plt.show()


#%%
'''
9. Depth vs Epicenter (range, mean)
'''
def box_depth_perEpi(sfiles,group_factor=5,save_graphs=False):
    
    import matplotlib.pyplot as plt
#    import numpy as np
#    import operator
    import os
    epis = {}
    weird_sfiles = []
    for sfile in sfiles:
        try:
            epistring = sfile.type_3['EPICENTER_LOCATION']
            depth = float(sfile.type_1['DEPTH'])
#            mag = float(sfile.type_1['DEPTH'])
            
            if(epistring not in epis):
                epis[epistring] = [depth]
            else:
                epis[epistring].append(depth)
        except:
            weird_sfiles.append(sfile.filename)
    
    event_list = sorted(epis.items(), key=lambda e: len(e[1]))
    event_list.reverse()
    print(event_list[-1])
#    print(epis)
    
    event_dict = {}
        
    segmented_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            list_index += 1
            
        event_dict[event[0]] = event[1]
        segmented_graphs[list_index][event[0]] = event[1]
        index += 1
        
    #print(segmented_graphs)  
      
    path = "IOfiles/Graphs/"
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"SeismsXstationAll.png")
    
    index = 0
    for graph in segmented_graphs:
#        print(graph)
        plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        plt.title("Depth per epicenter 2010-2017 ["+group_name+"]")
#        plt.bar(range(len(graph)), graph.values(), align='center')
#        print(graph.values())
        plt.boxplot(graph.values())
        keys = []
        for key in graph.keys():
            try:
                st = key.split('-')
                keys.append(st[0]+'\n'+st[1])
            except:
                keys.append(key)
        plt.xticks([i+1 for i in range(0,len(graph))], keys,ha='center',rotation=45)
#        plt.xticks(range(len(graph)), graph.keys())
        plt.xlabel("Epicenter name")
#        plt.xticks(rotation=45)
        plt.ylabel("Depth")
#        plt.grid(True)
        index += 1
        
        if(save_graphs):
            plt.savefig(path+"SeismsXstation"+group_name+".png")
        
        
    # Es buena idea verificar la fecha minima y la maxima para ponerlas como
    # parametro de entrada
    
    #print("Events per Station between 2010 and 2017") 
    #for event in event_dict:
     #   print("Station name: "+event+"\tEvents:\t"+str(event_dict[event]))
        
    return event_list


#%%
'''
10. Magnitude vs Epicenter (range, mean)
'''
def box_mag_perEpi(sfiles,group_factor=5,save_graphs=False):
    
    import matplotlib.pyplot as plt
    import numpy as np
#    import operator
    import os
    epis = {}
    weird_sfiles = []
    for sfile in sfiles:
        try:
            epistring = sfile.type_3['EPICENTER_LOCATION']
#            depth = float(sfile.type_1['DEPTH'])
            mag = float(sfile.type_1['TYPE_OF_MAGNITUDE_1'])
            
            if(epistring not in epis):
                epis[epistring] = [mag]
            else:
                epis[epistring].append(mag)
        except:
            weird_sfiles.append(sfile.filename)
    
    event_list = sorted(epis.items(), key=lambda e: len(e[1]))
    event_list.reverse()
    print(event_list[-1])
#    print(epis)
    
    event_dict = {}
        
    segmented_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            list_index += 1
            
        event_dict[event[0]] = event[1]
        segmented_graphs[list_index][event[0]] = event[1]
        index += 1
        
    #print(segmented_graphs)  
      
    path = "IOfiles/Graphs/"
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"SeismsXstationAll.png")
    
    index = 0
    for graph in segmented_graphs:
#        print(graph)
        plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        plt.title("Seisms per station 2010-2017 ["+group_name+"]")
#        plt.bar(range(len(graph)), graph.values(), align='center')
#        print(graph.values())
        plt.boxplot(graph.values())
        keys = []
        for key in graph.keys():
            try:
                st = key.split('-')
                keys.append(st[0]+'\n'+st[1])
            except:
                keys.append(key)
        plt.xticks([i+1 for i in range(0,len(graph))], keys,ha='center',rotation=45)
#        plt.xticks(range(len(graph)), graph.keys())
        plt.xlabel("Epicenter name")
#        plt.xticks(rotation=45)
        plt.ylabel("Magnitude")
#        plt.grid(True)
        index += 1
        
        if(save_graphs):
            plt.savefig(path+"SeismsXstation"+group_name+".png")
        
        
    # Es buena idea verificar la fecha minima y la maxima para ponerlas como
    # parametro de entrada
    
    #print("Events per Station between 2010 and 2017") 
    #for event in event_dict:
     #   print("Station name: "+event+"\tEvents:\t"+str(event_dict[event]))
        
    return event_list

#%%
'''
11. Mean epicenter distance vs Station
INCOMPLETO: FALTA ORDENAR POR DISTANCIA, ESTA ORDENADO POR CANTIDAD DE SISMOS
'''
def epiDis_per_station(sfiles,group_factor=10,save_graphs=False,path = "IOfiles/Graphs/"):
    import matplotlib.pyplot as plt
    import os
    events_per_station = {}
    for sfile in sfiles:
        last_station_name = " "
        stations = sfile.type_7
        for station in stations:
            station_name = station['STAT']
#            if(station_name == 'TEIG'):
#                print(sfile.filename)
            
            try: # There are empty DIS fields
                station_dis = float(station['DIS'])
                if station_name not in events_per_station:
                    
                    #print(station['DIS'])
                    #print(sfile.filename)
                    #print(station_name)
                    
#                    if(station_dis > 250 and station_name == 'BAR2'):
#                        print("PELIGRO")
#                        print(sfile.filename)
#                        print(station_name)
#                        print(station_dis+'\n')
                    
                    events_per_station[station_name] = [1,station_dis]
                else:
                    if station_name != last_station_name:
                        events_per_station[station_name][0] += 1 
                        events_per_station[station_name][1] += station_dis
                        last_station_name = station_name
                        
#                        if(station_name == 'BAR2'):
#                            cont_bar += 1
            
            except:
                pass
                    
            
#    print(cont_bar)
    # This is a workaround, should be improved
    event_list = sorted(events_per_station.items(), key=lambda e: e[1][1])
    event_list.reverse()
    print(events_per_station)
    print(event_list)
    event_dict = {}
    
    segmented_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            list_index += 1
            
        event_dict[event[0]] = event[1][1]/event[1][0]
        segmented_graphs[list_index][event[0]] = event[1][1]/event[1][0]
        index += 1
        
        #print(event)
        
#        print("Data")
#        print(event[1][1])
#        print(event[1][0])
        
    #print(segmented_graphs)  
    
##########    
    plt.figure()
    plt.title("Mean epicenter distance per station 2010-2017")
    plt.bar(range(len(event_dict)), event_dict.values(), align='center')
    plt.xlabel("Station index")
    plt.ylabel("Mean epicenter distance")
    
    
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"MeanEpicDisXstationAll.png")
    
    index = 0
    for graph in segmented_graphs:
        plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        plt.title("Mean epicenter distance per station 2010-2017 ["+group_name+"]")
        plt.bar(range(len(graph)), graph.values(), align='center')
        print(graph.values())
        plt.xticks(range(len(graph)), graph.keys())
        plt.xlabel("Station name")
        plt.ylabel("Mean epicenter distance")
        index += 1
        
        if(save_graphs):
            plt.savefig(path+"MeanEpicDisXstation"+group_name+".png")
       
    return event_dict

#%%
'''
12. Mean epicenter distance vs Station vs Amount of Seisms
'''
def magDisEpi_per_station(sfiles,group_factor=10,save_graphs=False, order_by_mag = True, path = "IOfiles/Graphs/"):
    import matplotlib.pyplot as plt
    import operator
    import os 
    
    events_per_station = {}
    for sfile in sfiles:
        last_station_name = " "
        stations = sfile.type_7
        for station in stations:
            station_name = station['STAT']
#            if(station_name == 'TEIG'):
#                print(sfile.filename)
            
            try: # There are empty DIS fields
                station_dis = float(station['DIS'])
                if station_name not in events_per_station:
                    
                    
                    events_per_station[station_name] = [1,station_dis]
                else:
                    if station_name != last_station_name:
                        events_per_station[station_name][0] = events_per_station[station_name][0] + 1
                        events_per_station[station_name][1] = events_per_station[station_name][1] + station_dis
                        last_station_name = station_name
            
            except:
                pass
                    
            
#    print(cont_bar)
    # This is a workaround, should be improved
    event_list = sorted(events_per_station.items(), key=operator.itemgetter(1))
    event_list.reverse()
#    print(events_per_station)
    dis_dict = {}
    amount_dict = {}
    
    segmented_graphs = []
    amount_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            amount_graphs.append({})
            list_index += 1
            
        dis_dict[event[0]] = event[1][1]/event[1][0]
        amount_dict[event[0]] = event[1][0]
        
        segmented_graphs[list_index][event[0]] = event[1][1]/event[1][0]
        amount_graphs[list_index][event[0]] = event[1][0]
        index += 1
        
##########
    fig, ax1 = plt.subplots()
    
    x1 = [x - 0.4 for x in range(len(dis_dict))]
    
    color = 'tab:red'
    ax1.set_xlabel('Station Index: Ordered by amount of seisms')
    ax1.set_ylabel('Amount of Seisms', color=color)
    ax1.bar(x1, amount_dict.values(), color=color, width = 0.4, align = 'edge')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    
    color = 'tab:blue'
    ax2.set_ylabel('Mean epicenter distance', color=color)  # we already handled the x-label with ax1
    ax2.bar(range(len(dis_dict)),dis_dict.values(), color=color, width = 0.4, align = 'edge')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Mean Distance VS Amount of Seisms per station 2010-2017")
    plt.grid(True)
    #plt.show()
    
    
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"MeanDisVSamountXstationAll.png")
        
    index = 0
    for (graph_dis,graph_amount) in zip(segmented_graphs,amount_graphs):
        #plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        
        fig, ax1 = plt.subplots()
        plt.title("Mean Dis VS Amount of Seisms per station 2010-2017 ["+group_name+"]")
        
        x1 = [x - 0.4 for x in range(len(graph_dis))]
        color = 'tab:red'
        ax1.set_ylabel('Amount of Seisms', color=color)
        ax1.bar(x1, graph_amount.values(), color=color, width = 0.4, align = 'edge')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        
        color = 'tab:blue'
        ax2.set_ylabel('Mean epicenter distance', color=color)  # we already handled the x-label with ax1
        ax2.bar(range(len(graph_dis)),graph_dis.values(), color=color, width = 0.4, align = 'edge')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        
#        plt.bar(range(len(graph)), graph.values(), align='center')
        plt.xticks(range(len(graph_dis)), graph_dis.keys())
        plt.grid(True)
#        plt.xlabel("Station name")
#        plt.ylabel("Mean epicenter distance")
        index += 1
        

        if(save_graphs):
            plt.savefig(path+"MeanDisVSamountXstation"+group_name+".png")

##########        
        
    return dis_dict,amount_dict

#%%
'''
12. Mean epicenter distance vs Station vs Amount of Seisms
'''
def magDisHip_per_station(sfiles,group_factor=10,save_graphs=False, order_by_mag = True, path = "IOfiles/Graphs/"):
    import matplotlib.pyplot as plt
    import operator
    import os 
    
    events_per_station = {}
    for sfile in sfiles:
        last_station_name = " "
        stations = sfile.type_7
        for station in stations:
            station_name = station['STAT']
#            if(station_name == 'TEIG'):
#                print(sfile.filename)
            
            try: # There are empty DIS fields
                station_dis = float(station['DIS'])
                depth = float(sfile.type_1['DEPTH'])
                
                
                if station_name not in events_per_station:
                    
                    
                    events_per_station[station_name] = [1,station_dis]
                else:
                    if station_name != last_station_name:
                        events_per_station[station_name][0] = events_per_station[station_name][0] + 1
                        events_per_station[station_name][1] = events_per_station[station_name][1] + station_dis
                        last_station_name = station_name
            
            except:
                pass
                    
            
#    print(cont_bar)
    # This is a workaround, should be improved
    event_list = sorted(events_per_station.items(), key=operator.itemgetter(1))
    event_list.reverse()
#    print(events_per_station)
    dis_dict = {}
    amount_dict = {}
    
    segmented_graphs = []
    amount_graphs = []
    index = 0
    list_index = -1
    for event in event_list:
        
        if(index%group_factor == 0):
            segmented_graphs.append({})
            amount_graphs.append({})
            list_index += 1
            
        dis_dict[event[0]] = event[1][1]/event[1][0]
        amount_dict[event[0]] = event[1][0]
        
        segmented_graphs[list_index][event[0]] = event[1][1]/event[1][0]
        amount_graphs[list_index][event[0]] = event[1][0]
        index += 1
        
##########
    fig, ax1 = plt.subplots()
    
    x1 = [x - 0.4 for x in range(len(dis_dict))]
    
    color = 'tab:red'
    ax1.set_xlabel('Station Index: Ordered by amount of seisms')
    ax1.set_ylabel('Amount of Seisms', color=color)
    ax1.bar(x1, amount_dict.values(), color=color, width = 0.4, align = 'edge')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    
    color = 'tab:blue'
    ax2.set_ylabel('Mean epicenter distance', color=color)  # we already handled the x-label with ax1
    ax2.bar(range(len(dis_dict)),dis_dict.values(), color=color, width = 0.4, align = 'edge')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Mean Distance VS Amount of Seisms per station 2010-2017")
    plt.grid(True)
    #plt.show()
    
    
    if(save_graphs):   
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig(path+"MeanDisVSamountXstationAll.png")
        
    index = 0
    for (graph_dis,graph_amount) in zip(segmented_graphs,amount_graphs):
        #plt.figure()
        group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
        
        fig, ax1 = plt.subplots()
        plt.title("Mean Dis VS Amount of Seisms per station 2010-2017 ["+group_name+"]")
        
        x1 = [x - 0.4 for x in range(len(graph_dis))]
        color = 'tab:red'
        ax1.set_ylabel('Amount of Seisms', color=color)
        ax1.bar(x1, graph_amount.values(), color=color, width = 0.4, align = 'edge')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        
        color = 'tab:blue'
        ax2.set_ylabel('Mean epicenter distance', color=color)  # we already handled the x-label with ax1
        ax2.bar(range(len(graph_dis)),graph_dis.values(), color=color, width = 0.4, align = 'edge')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        
#        plt.bar(range(len(graph)), graph.values(), align='center')
        plt.xticks(range(len(graph_dis)), graph_dis.keys())
        plt.grid(True)
#        plt.xlabel("Station name")
#        plt.ylabel("Mean epicenter distance")
        index += 1
        

        if(save_graphs):
            plt.savefig(path+"MeanDisVSamountXstation"+group_name+".png")

##########        
        
    return dis_dict,amount_dict

#%%
'''
13. Station sampling rate vs Time
'''

#%%
'''
14. Quantity of components vs Time
'''

#%%
'''
15. Components vs Station vs Amount of seisms
INCOMPLETO: FALTA ARREGLAR EL TAMAÑO DE LAS GRAFICAS PARA QUE SE PUEDAN VER
TODOS LOS COMPONENTES BIEN SIN CORTARSE
'''

def components_per_station(sfiles,group_factor=10,save_graphs=False):
        import matplotlib.pyplot as plt
        import os
        import operator
        import TelluricoTools
        
        events_per_station = {}
        comps_per_station = {}
        for sfile in sfiles:
            last_station_name = " "
            stations = sfile.type_7
            for station in stations:
                station_name = station['STAT']
                station_comp = station['SP']
                
                if station_name not in events_per_station:
                    events_per_station[station_name] = 1
                    comps_per_station[station_name] = []
                else:
                    if station_name != last_station_name:
                        events_per_station[station_name] = events_per_station[station_name] + 1
                        last_station_name = station_name
                        
                comps_per_station[station_name].append(station_comp)
        #print(comps_per_station)
                        
        # This is a workaround, should be improved
        event_list = sorted(events_per_station.items(), key=operator.itemgetter(1))
        event_list.reverse()
        event_dict = {}
        
        segmented_graphs = []
        index = 0
        list_index = -1
        for (event,comp_dict_key) in zip(event_list,comps_per_station):
            
            #print(comp_dict_key)
            comps_per_station[comp_dict_key] = TelluricoTools.remove_duplicates(comps_per_station[comp_dict_key])
            
            if(index%group_factor == 0):
                segmented_graphs.append({})
                list_index += 1
                
            event_dict[event[0]] = event[1]
            segmented_graphs[list_index][event[0]] = event[1]
            index += 1
            
        #print(segmented_graphs)  
        #print(comps_per_station)
        plt.figure()
        plt.title("Seisms per station 2010-2017")
        plt.bar(range(len(event_dict)), event_dict.values(), align='center')
        plt.xlabel("Station index")
        plt.ylabel("Amount of events")
        
        path = "IOfiles/Graphs/"
        if(save_graphs):   
            if not os.path.exists(path):
                os.makedirs(path)           
            plt.savefig(path+"SeismsXstationAll.png")
        
        index = 0
        
        for graph in segmented_graphs:
            
            # Concat station components to station name
            new_keys = []
            string_name = ""
            for station_name in graph:
                string_name = station_name + '\n'
                comps = comps_per_station[station_name]
                for comp in comps:
                    string_name += comp + '\n'
                
                #print(string_name)
                #graph[station_name] = graph.pop(string_name)
                new_keys.append(string_name)
                
            group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
            #print(graph.keys())
            plt.figure()
            plt.title("Seisms per station 2010-2017 ["+group_name+"]")
            plt.bar(range(len(graph)), graph.values(), align='center')
            #plt.xticks(range(len(graph)), graph.keys())
            plt.xticks(range(len(graph)), new_keys)
            plt.xlabel("Station name")
            plt.ylabel("Amount of events")
            index += 1
            
            if(save_graphs):
                plt.savefig(path+"SeismsXstation"+group_name+".png")
            
            
        # Es buena idea verificar la fecha minima y la maxima para ponerlas como
        # parametro de entrada
        
        #print("Events per Station between 2010 and 2017") 
        #for event in event_dict:
         #   print("Station name: "+event+"\tEvents:\t"+str(event_dict[event]))
            
        return [event_dict,comps_per_station]
    

#%% Demos
'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import numpy as np
#
#
#def randrange(n, vmin, vmax):
#    '''
#    Helper function to make an array of random numbers having shape (n, )
#    with each number distributed Uniform(vmin, vmax).
#    '''
#    return (vmax - vmin)*np.random.rand(n) + vmin
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#n = 100
#
## For each set of style and range settings, plot n random points in the box
## defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#    xs = randrange(n, 23, 32)
#    ys = randrange(n, 0, 100)
#    zs = randrange(n, zlow, zhigh)
#    ax.scatter(xs, ys, zs, c=c, marker=m)
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()

'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import numpy as np
#
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
## Make data.
#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
#
## Plot the surface.
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
## Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#plt.show()

'''
Boxplot demo 1
'''
#import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.patches import Polygon
#
#
## Fixing random state for reproducibility
#np.random.seed(19680801)
#
## fake up some data
#spread = np.random.rand(50) * 100
#center = np.ones(25) * 50
#flier_high = np.random.rand(10) * 100 + 100
#flier_low = np.random.rand(10) * -100
#data = np.concatenate((spread, center, flier_high, flier_low), 0)
#
#fig, axs = plt.subplots(2, 3)
#
## basic plot
#axs[0, 0].boxplot(data)
#axs[0, 0].set_title('basic plot')
#
## notched plot
#axs[0, 1].boxplot(data, 1)
#axs[0, 1].set_title('notched plot')
#
## change outlier point symbols
#axs[0, 2].boxplot(data, 0, 'gD')
#axs[0, 2].set_title('change outlier\npoint symbols')
#
## don't show outlier points
#axs[1, 0].boxplot(data, 0, '')
#axs[1, 0].set_title("don't show\noutlier points")
#
## horizontal boxes
#axs[1, 1].boxplot(data, 0, 'rs', 0)
#axs[1, 1].set_title('horizontal boxes')
#
## change whisker length
#axs[1, 2].boxplot(data, 0, 'rs', 0, 0.75)
#axs[1, 2].set_title('change whisker length')
#
#fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
#                    hspace=0.4, wspace=0.3)
#
## fake up some more data
#spread = np.random.rand(50) * 100
#center = np.ones(25) * 40
#flier_high = np.random.rand(10) * 100 + 100
#flier_low = np.random.rand(10) * -100
#d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
#data.shape = (-1, 1)
#d2.shape = (-1, 1)
## Making a 2-D array only works if all the columns are the
## same length.  If they are not, then use a list instead.
## This is actually more efficient because boxplot converts
## a 2-D array into a list of vectors internally anyway.
#data = [data, d2, d2[::2, 0]]
#
## Multiple box plots on one Axes
#fig, ax = plt.subplots()
#ax.boxplot(data)
#
#plt.show()


'''
Boxplot demo 2
'''
#import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.patches import Polygon
#
#
## Fixing random state for reproducibility
#np.random.seed(19680801)
#
## fake up some data
#spread = np.random.rand(50) * 100
#center = np.ones(25) * 50
#flier_high = np.random.rand(10) * 100 + 100
#flier_low = np.random.rand(10) * -100
#data = np.concatenate((spread, center, flier_high, flier_low), 0)
#
#
#
## fake up some more data
#spread = np.random.rand(50) * 100
#center = np.ones(25) * 40
#flier_high = np.random.rand(10) * 100 + 100
#flier_low = np.random.rand(10) * -100
#d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
#data.shape = (-1, 1)
#d2.shape = (-1, 1)
## Making a 2-D array only works if all the columns are the
## same length.  If they are not, then use a list instead.
## This is actually more efficient because boxplot converts
## a 2-D array into a list of vectors internally anyway.
#data = [data, d2, d2[::2, 0]]
#
## Multiple box plots on one Axes
#fig, ax = plt.subplots()
#ax.boxplot(data,0,'')
#
#plt.show()

'''
Boxplot demo 3
'''

#import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.patches import Polygon
#
#
## Fixing random state for reproducibility
#np.random.seed(19680801)
#
#
#data = [[14,6,3,2,4,15,11,8,1,7,2,1,3,4,10,22,20],[14,6,3,2,4,15,11,8,1,7,2,1,3,4,10,22,20]]
#
## Multiple box plots on one Axes
#fig, ax = plt.subplots()
#ax.boxplot(data)
#
#plt.show()


'''
Barchart with percentages on top
'''
#import numpy as np
#import matplotlib.pyplot as plt
#
#n_groups = 5
#
#Zipf_Values = (100, 50, 33, 25, 20)
#Test_Values = (97, 56, 35, 22, 19)
#
#fig, ax = plt.subplots()
#
#index = np.arange(n_groups)
#bar_width = 0.35
#
#rects1 = plt.bar(index, Zipf_Values, bar_width, color='g', 
#    label='Zipf', alpha= 0.8)
#rects2 = plt.bar(index + bar_width, Test_Values, bar_width, color='y', 
#    label='Test Value', alpha= 0.8)
#
#plt.xlabel('Word')
#plt.ylabel('Frequency')
#plt.title('Zipf\'s Law: Les Miserables')
#plt.xticks(index + bar_width, ('The', 'Be', 'And', 'Of', 'A'))
#plt.legend()
#
#for rect in rects1:
#    height = rect.get_height()
#    ax.text(rect.get_x() + rect.get_width()/2., 0.99*height,
#            '%d' % int(height) + "%", ha='center', va='bottom')
#for rect in rects2:
#    height = rect.get_height()
#    ax.text(rect.get_x() + rect.get_width()/2., 0.99*height,
#            '%d' % int(height) + "%", ha='center', va='bottom')
#
#plt.tight_layout()
#plt.show()





