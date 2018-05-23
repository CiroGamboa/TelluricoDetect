#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:22:46 2018

@author: CiroGamJr


Fuente: http://stackabuse.com/python-list-files-in-a-directory/
"""

class SfileAnalyzer:
    
    def __init__(self,path):
        self.path = path
        self.sfiles = []
        
    def get_sfiles(self):
        import os, fnmatch
        from Sfile import Sfile
        
        list_of_files = os.listdir(path)  
        pattern = "*.S*"  
        for filename in list_of_files:  
            if fnmatch.fnmatch(filename, pattern):
                    
                sfile = Sfile(filename = filename, path = self.path)
                sfile.get_attributes()
                self.sfiles.append(sfile)
        
        print("Files found: ")
        print(len(self.sfiles))
    
#    def events_per_month(self):
        
        
    def events_per_year(self):
        
        import matplotlib.pyplot as plt
        #import plotly.plotly as py
        
        events = {}
        for sfile in self.sfiles:
            year = sfile.type_1['YEAR']
            
            if year not in events:
                events[year] = 1
            else:
                events[year] = events[year] + 1
                

        # Ascending order by year !!!!!
        #keys = sorted(events.keys())
        
        plt.figure()
        #D = {u'Label0':26, u'Label1': 17, u'Label2':30}
        plt.bar(range(len(events)), events.values(), align='center')
        plt.xticks(range(len(events)), events.keys())
        #plot_url = py.plot_mpl(dictionary, filename='mpl-dictionary')
        #print(sfile.type_1['YEAR'])
        
        print(events)
        
        
    def seisms_per_station(self):
        import matplotlib.pyplot as plt
        
        events_per_station = {}
        for sfile in self.sfiles:
            last_station_name = " "
            stations = sfile.type_7
            for station in stations:
                station_name = station['STAT']
                
                if station_name not in events_per_station:
                    events_per_station[station_name] = 1
                else:
                    if station_name != last_station_name:
                        events_per_station[station_name] = events_per_station[station_name] + 1
                        last_station_name = station_name
                
        plt.figure()
        plt.bar(range(len(events_per_station)), events_per_station.values(), align='center')
        plt.xticks(range(len(events_per_station)), events_per_station.keys())
        
        # Es buena idea verificar la fecha minima y la maxima para ponerlas como
        # parametro de entrada
        
        print("Events per Station between 2010 and 2017") 
        for event in events_per_station:
            print("Station name: "+event+"\tEvents:\t"+str(events_per_station[event]))
            
            
    
    def clustering(self):
        example_sfile = self.sfiles[0]
        
        for type_line in example_sfile.lines_info:
            for type_fields in type_line:
                print(type_fields)
                dic = type_line[type_fields]
                for param in dic:

                    if isinstance(param,dict):
                        for element in param:
                            print(element+"\t:\t"+param[element])
                        print("\n")
                           
                    else:
                        print(param+"\t:\t"+dic[param])
                print('\n')
        
            
        
#### Testing
path = "RSNC_Sfiles/"
analyzer = SfileAnalyzer(path)
analyzer.get_sfiles()





####### Bar chart
#import matplotlib.pyplot as plt
#import plotly.plotly as py
#
#dictionary = plt.figure()
#
#D = {u'Label0':26, u'Label1': 17, u'Label2':30}
#
#plt.bar(range(len(D)), D.values(), align='center')
#plt.xticks(range(len(D)), D.keys())
#
##plot_url = py.plot_mpl(dictionary, filename='mpl-dictionary')



##### Scatter plot
#import numpy as np
#import matplotlib.pyplot as plt
#
## Fixing random state for reproducibility
#np.random.seed(19680801)
#
#
#N = 50
#x = np.random.rand(N)
#y = np.random.rand(N)
#colors = np.random.rand(N)
#area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
#
#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#plt.show()



        
        
    