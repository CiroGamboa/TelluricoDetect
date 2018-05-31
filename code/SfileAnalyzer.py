#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:22:46 2018
ESTA CLASE ESTA DEDICADA A ANALIZAR LOS SFILES DE FORMA GENERAL
CON EL FIN DE SACAR ESTADISTICAS 
@author: CiroGamJr
Fuente: http://stackabuse.com/python-list-files-in-a-directory/
"""

#%%
import os, fnmatch
import re
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tools import TelluricoTools
from Sfile import Sfile
#%%
class SfileAnalyzer:
    
    def __init__(self,path):
        self.path = path
        self.sfiles = []
        
    def get_sfiles(self):
        list_of_files = os.listdir(self.path)  
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
        
        
        def export_obj(path,analyzer):
            import pickle
            file_handler = open(path,'wr')
            pickle.dump(analyzer,file_handler)
            

#%%        
#### Testing
#path = "IOfiles/RSNC_Sfiles/"
#analyzer = SfileAnalyzer(path)
#analyzer.get_sfiles()
#analyzer.group_by_magnitude()
#%%

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
'''
    Input data example:
        sfiles = analyzer.sfiles
        ranges = [(0,4),(4,5.9),(5.9,10)]
        waveform_paths = []
        total_files = 100
        copy = False
'''

#%%
def arrange_by_magnitude(sfiles,ranges,waveform_paths,sfile_paths, total_files, copy_flag):
    
    groups = []
    filtered_groups = []
    for couple in ranges:
        path = "IOfiles/Groups/"+str(couple[0])+"-"+str(couple[1])+"/"
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path+"/Sfiles/")
            os.makedirs(path+"/Waveforms/")
            
        groups.append(([],couple,path))
        filtered_groups.append(([],couple,path))
    
    undefined_group = []
    damaged_group = []
    
    for sfile in sfiles:
        mag_string = sfile.type_1['TYPE_OF_MAGNITUDE_1']
        grouped = False
        try:  
            mag = float(mag_string)
            
            for group in groups:
                if(mag >= group[1][0] and mag < group[1][1]):
                    
                    #copyfile(src, dst)
                    group[0].append((sfile.filename,sfile.type_6['BINARY_FILENAME'],mag))
                    grouped = True
            
            if(not grouped):
                undefined_group.append(mag)
            
        except:
            damaged_group.append(sfile.filename)
           
    transfered_sfiles = 0
    transfered_waveforms = 0
    
    if(total_files != -1):
        random_indexes = []
        for group,filtered_group in zip(groups,filtered_groups):
            index = 0
            #print(len(group[0]))
            
            amount_records = len(group[0])
            files_per_group = int(total_files/len(groups))
            
            if(amount_records < files_per_group):
                files_per_group = amount_records
            
            random_indexes = np.random.choice(amount_records,files_per_group , replace=False)
            
            #print(len(random_indexes))
            #print(random_indexes)
                    
            for index in random_indexes:
                actual_record = group[0][index]
                filtered_group[0].append(actual_record)
                if(copy_flag):
                    sfile_copied = copy_file(actual_record[0],sfile_paths,filtered_group[2]+"Sfiles/")
                    waveform_copied = copy_file(actual_record[1],waveform_paths,filtered_group[2]+"Waveforms/")
                    if(sfile_copied):
                        transfered_sfiles += 1
                    if(waveform_copied):
                        transfered_waveforms += 1
                    #print(actual_record[0])
                    #print(filtered_group[2])
    
    if(copy_flag):
        print("FILES TRANSFERED:")
        print(str(transfered_sfiles)+"/"+str(total_files)+" sfiles")
        print(str(transfered_waveforms)+"/"+str(total_files)+" waveforms")
    
    return [groups,filtered_groups,undefined_group,damaged_group]
        
#%%
# Check in different directories for a file and copy to another location
'''
Example on input data:
    filename = '2015-05-05-2237-50M.COL___276'
    file_paths = ['/media/administrador/Tellurico_Dataset1/','/media/administrador/Tellurico_Dataset2/']
     
'''
def copy_file(filename,file_paths, destination_path):
    # Create the destination directory
    #print(file_paths)
    #print(destination_path)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        
    found = False
    
    for file_path in file_paths:
        try:
            list_of_files = os.listdir(file_path)
            pattern = '*'+filename+'*' 
            for file_iter in list_of_files:  
                #print(file_iter)
                if fnmatch.fnmatch(file_iter, pattern):
                    shutil.copy(file_path+file_iter, destination_path)
                    #print(file_path+file_iter)
                    found = True
                    break #Helps avoiding repeated files
                
        except:
           found = False
            
    if(not found):
        #print("FILE SUCCESFULLY COPIED")
        #print(".")
    #else:
        #print("FILE NOT FOUND")    
        #print(filename+'\n')
        print(filename+" "+destination_path)
        
    return found

#%%      PLOTS
# Plot the events registered by stations in descendent order
def seisms_per_station(sfiles,group_factor,save_graphs,include_comps):
        import matplotlib.pyplot as plt
        import operator
        
        
        events_per_station = {}
        for sfile in sfiles:
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
            plt.figure()
            group_name = str(group_factor*index+1)+"-"+str(group_factor*(index+1))
            plt.title("Seisms per station 2010-2017 ["+group_name+"]")
            plt.bar(range(len(graph)), graph.values(), align='center')
            plt.xticks(range(len(graph)), graph.keys())
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
            
        return event_dict

#%% Components in station
def components_per_station(sfiles,group_factor,save_graphs):
        import matplotlib.pyplot as plt

        import operator
        
        
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
    
    
    
    
    
    
    
    
#%%
  


      
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

#plt.figure()
#plt.bar(range(2), [3,5], align='center')
#plt.xticks(range(2), ['bra\nkj','vdd'])
#plt.xlabel("Station name")
#plt.ylabel("Amount of events")