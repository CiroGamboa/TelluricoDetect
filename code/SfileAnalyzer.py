#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:22:46 2018

ESTA CLASE ESTA DEDICADA A ANALIZAR LOS SFILES DE FORMA GENERAL
CON EL FIN DE SACAR ESTADISTICAS 

@author: CiroGamJr


Fuente: http://stackabuse.com/python-list-files-in-a-directory/
"""
import os, fnmatch
import re
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
from tools import TelluricoTools
from Sfile import Sfile

class SfileAnalyzer:
    
    def __init__(self,path):
        self.path = path
        self.sfiles = []
        
    def get_sfiles(self):
        list_of_files = os.listdir(path)  
        pattern = "*.S*"  
        for filename in list_of_files:  
            if fnmatch.fnmatch(filename, pattern):
                    
                sfile = Sfile(filename = filename, path = self.path)
                sfile.get_params()
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
        
            
    def group_by_magnitude(self):
        # Solo por este ejemplo
        group1 = []
        group2 = []
        group3 = []
        undefined_group = []
        damaged_group = []
        
        key_value1 = 5.5
        key_value2 = 6.5
        
        for sfile in self.sfiles:
            mag_string = sfile.type_1['TYPE_OF_MAGNITUDE_1']
            try:  
                mag = float(mag_string)
                
                if(mag >= 0 and mag < key_value1):
                    group1.append(sfile.filename)        
                elif(mag >= key_value1 and mag < key_value2):
                    group2.append(sfile.filename)        
                elif(mag >= key_value2):
                    group3.append(sfile.filename)        
                else:
                    undefined_group.append(sfile.filename)
                    
            except:
                damaged_group.append(sfile.filename)
            

            
    
        print("GROUP 1: 0 >= mag < 5.5")
        print(len(group1))
        print("GROUP 2: 5.5 >= mag < 6.5")
        print(len(group2))
        print("GROUP 3: mag >= 6.5")
        print(len(group3))
        print("UNDEFINED GROUP: mag < 0")
        print(len(undefined_group))
        print("DAMAGED GROUP: CORRUPT FILES")
        print(len(damaged_group))
        
        
    def clear_corrupt(self):
        
        # List containing the names of corrupt files
        corrupt_files = []
        
        # List containing the names of repeated files
        repeated_files = []
        
        # Flag for avoid including the same file in many cases
        included = None
        for sfile in self.sfiles:
            
            included = False
            # Check if the magnitude can be parsed to float   
            try:  
                mag = float(sfile.type_1['TYPE_OF_MAGNITUDE_1'])
                
            except:
                corrupt_files.append(sfile.filename)
                included = True
                
            # Check if there are repeated files
            #pattern = re.compile("\.[1-9]+")
            #pattern.search("dog")
            if(re.search( r'[.][1-9]+', sfile.filename, re.M|re.I) is not None and not included):
                repeated_files.append(sfile.filename)
                included = True
                
        # Take out repeated Sfiles and Corrupt Sfiles in different folders
        if(len(corrupt_files) > 0):
            corrupt_path = "IOfiles/CorruptSfiles/"
            if not os.path.exists(corrupt_path):
                os.makedirs(corrupt_path)
                
            for file in corrupt_files:
                os.rename(self.path+file, corrupt_path+file)
                
        print("CORRUPT FILES FOUND:"+str(len(corrupt_files)))
            
        if(len(repeated_files) > 0):
            repeated_path = "IOfiles/RepeatedSfiles/"
            if not os.path.exists(repeated_path):
                os.makedirs(repeated_path)
                
            for file in repeated_files:
                os.rename(self.path+file, repeated_path+file)
                
        print("REPEATED FILES FOUND:"+str(len(repeated_files)))
        
        print(repeated_files)
            
        
#### Testing
path = "IOfiles/RSNC_Sfiles/"
analyzer = SfileAnalyzer(path)
analyzer.get_sfiles()
#analyzer.group_by_magnitude()


def plot_by_magnitude(sfiles):
    undefined_group = []
    magnitudes = []
    for sfile in sfiles:
            
        try:  
            mag = float(sfile.type_1['TYPE_OF_MAGNITUDE_1'])
            if(mag is not None):
                magnitudes.append(mag*10)
                
        except:
            undefined_group.append(sfile)
    print(magnitudes)
    #plt.hist(range(0,6.5), weights=magnitudes,bins = magnitudes)
    
    
    x_axis = np.arange(0,66,1)
    hist = plt.hist(magnitudes,bins = x_axis)
    plt.show()
    return [magnitudes,hist]
    #np.histogram(magnitudes)

# Group by the magnitude value
def group_by_magnitude(sfiles,ranges):
    
    groups = []
    for couple in ranges:
        groups.append(([],couple))
    
    undefined_group = []
    damaged_group = []
    
    for sfile in sfiles:
        mag_string = sfile.type_1['TYPE_OF_MAGNITUDE_1']
        grouped = False
        try:  
            mag = float(mag_string)
            
            for group in groups:
                if(mag >= group[1][0] and mag < group[1][1]):
                    group[0].append(mag)
                    grouped = True
            
            if(not grouped):
                undefined_group.append(mag)
            
        except:
            damaged_group.append(sfile.filename)
           
    return [groups,undefined_group,damaged_group]



# Group by the magnitude value and copy files
def arrange_by_magnitude(sfiles,ranges,waveform_paths):
    
    groups = []
    for couple in ranges:
        path = "IOfiles/Groups/"+str(couple[0])+"-"+str(couple[1])+"/"
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path+"/Sfiles/")
            os.makedirs(path+"/Waveforms/")
        groups.append(([],couple,path))
    
    undefined_group = []
    damaged_group = []
    
    for sfile in sfiles:
        mag_string = sfile.type_1['TYPE_OF_MAGNITUDE_1']
        grouped = False
        try:  
            mag = float(mag_string)
            
            for group in groups:
                if(mag >= group[1][0] and mag < group[1][1]):
                    
                    copyfile(src, dst)
                    group[0].append((sfile.filename,mag))
                    grouped = True
            
            if(not grouped):
                undefined_group.append(mag)
            
        except:
            damaged_group.append(sfile.filename)
           
    return [groups,undefined_group,damaged_group]
        

# Check in different directories for a file and copy to another location
'''
Example on input data:
    filename = '2015-05-05-2237-50M.COL___276'
    file_paths = ['/media/administrador/Tellurico_Dataset1/','/media/administrador/Tellurico_Dataset2/']
     
'''
def copy_file(filename,file_paths, destination_path):
    # Create the destination directory
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        
    found = False
    
    
    
    for file_path in file_paths:
        try:
            list_of_files = os.listdir(file_path)
            pattern = '*'+filename+'*' 
            #pattern = filename
            for file_iter in list_of_files:  
                print(file_iter)
                if fnmatch.fnmatch(file_iter, pattern):
                    print("Entro")
                    copyfile(file_path+file_iter, destination_path)
                    found = True
                    break
        except:
            print("Error")
            found = False
            
    if(found):
        print("FILE SUCCESFULLY COPIED")
    else:
        print("FILE NOT FOUND")
            
        
        
#list_of_files = os.listdir(path)  
#        pattern = "*.S*"  
#        for filename in list_of_files:  
#            if fnmatch.fnmatch(filename, pattern):
#                    
#                sfile = Sfile(filename = filename, path = self.path)
#                sfile.get_params()
#                self.sfiles.append(sfile)
#        
#        print("Files found: ")
#        print(len(self.sfiles))
            
    
        





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



        
        
    