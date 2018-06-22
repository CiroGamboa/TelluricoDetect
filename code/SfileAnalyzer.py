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
import numpy as np
import shutil
from tools import TelluricoTools
from Sfile import Sfile
#%%
class SfileAnalyzer:
    
    def __init__(self,path):
#        import os, fnmatch
#        import re
#        import numpy as np
#        import shutil
#        from tools import TelluricoTools
#        from Sfile import Sfile
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
'''
JULIAN VA A PASAR EL CODIGO PARA VALIDAR Wrong-P-Picking'
ES BUENA IDEA CHEQUEAR QUE LAS ESTACIONES REGISTRADAS ESTEN
DENTRO DE UNA LISTA DE ESTACIONES OFICIALES EN UNA BASE DE DATOS
TAMBIEN EL TIPO DE COMPONENTE U OTROS DATOS DE RELEVANCIA
'''
def separate_sfiles(sfiles,path,selected_stations=['BRR','RUS','PAM','PTB']):
#    cases = ['Repeated','Wrong-TYPE_OF_MAGNITUDE_1','Wrong-STAT','Wrong-EPICENTER_LOCATION',
#             'Wrong-DEPTH','Wrong-LATITUDE','Wrong-LONGITUDE','Wrong-DIS','Wrong-SP','Wrong-P_Picking']
#
#    
#    type_1 = ['LATITUDE','LONGITUDE','DEPTH','TYPE_OF_MAGNITUDE_1']
#    type_7 = ['STAT','SP','DIS'] # AQUI VA LO QUE ANALIZO JULIAN
#    type_3 = ['EPICENTER_LOCATION']
#    type_6 = ['BINARY_FILENAME']


    # List containing the names of corrupt files
#    corrupt_files = []
    
    corr_files = {}
    
    # List containing the names of repeated files
    repeated_files = []
    
    # Flag for avoid including the same file in many cases
    included = None
    for sfile in sfiles:
        
#        included = False # ESTO SE VA
        
        ################################################################
        # TYPE_1 PARAMETERS
        ################################################################
        # Check Latitude and Longitude
        # Use this: https://stackoverflow.com/questions/7861196/check-if-a-geopoint-with-latitude-and-longitude-is-within-a-shapefile
#        param = 'LATITUDE-LONGITUDE'
#        try:
##            print(':)')
#            a = 1
#        except:
#            if param not in corr_files:
#                corr_files[param] = [sfile.filename]
#            else:
#                corr_files[param].append(sfile.filename)
        
        # Check Depth
        # Add threshold above 10? 20? --> Ask Edward 
        param = 'DEPTH'
        try:
            depth = float(sfile.type_1[param])
        except:
            if param not in corr_files:
                corr_files[param] = [sfile.filename]
            else:
                corr_files[param].append(sfile.filename)
        
        param = 'TYPE_OF_MAGNITUDE_1'
        
        
        # Check if the magnitude can be parsed to float   
        try:  
            mag = float(sfile.type_1[param])
            
        except:
            if param not in corr_files:
                corr_files[param] = [sfile.filename]
            else:
                corr_files[param].append(sfile.filename)
          
        ################################################################
        # TYPE_7 PARAMETERS
        # PARA CHEQUEAR QUE EL NOMBRE DE LA ESTACION SEA CORRECTO
        # SERIA NECESARIO TENER UNA BASE DE DATOS CON LOS NOMBRES
        # DE LAS ESTACIONES, POR ENDE, POR AHORA, SE DA DE ENTRADA UN GRUPO
        # DE ESTACIONES DE INTERES, ASUMIENDO QUE DESDE QUE LOS DATOS
        # ASOCIADOS A ESAS ESTACIONES ESPECIFICAS NO ESTEN CORRUPTOS
        # NO SE DEBE DESCARTAR EL ARCHIVO
        ################################################################
        stations = sfile.type_7
        for station in stations:
            if selected_stations != 'all':
                param = 'STAT'
                station_name = station[param]
                if station_name in selected_stations:
                    # QUE PASO CON LOS COMPONENTES?????? -->SP
                    
                    param = 'DIS'
                    try:
                        station_dis = float(station[param])
                    except:
                        if param not in corr_files:
                            corr_files[param] = [sfile.filename]
                        else:
                            corr_files[param].append(sfile.filename)                        
             
            
        ################################################################
        # TYPE_3 PARAMETERS
        ################################################################
        param = 'EPICENTER_LOCATION'
        try:  
            epi = (len(sfile.type_3[param])>1)
        except:
            if param not in corr_files:
                corr_files[param] = [sfile.filename]
            else:
                corr_files[param].append(sfile.filename)
#            print(sfile.filename)
        
        ################################################################
        # TYPE_6 PARAMETERS
        ################################################################
        param = 'BINARY_FILENAME'
        try:  
            filename = (len(sfile.type_6[param])>20)
        except:
            if param not in corr_files:
                corr_files[param] = [sfile.filename]
            else:
                corr_files[param].append(sfile.filename)        
            
            
#            corrupt_files.append(sfile.filename)
#            included = True
            
        # Check if there are repeated files
        if(re.search( r'[.][1-9]+', sfile.filename, re.M|re.I) is not None and not included):
            repeated_files.append(sfile.filename)
#            included = True
            
    # Take out repeated Sfiles and Corrupt Sfiles in different folders
#    if(len(corrupt_files) > 0):
#        corrupt_path = "IOfiles/CorruptSfiles/"
#        if not os.path.exists(corrupt_path):
#            os.makedirs(corrupt_path)
#            
#        for file in corrupt_files:
#            os.rename(path+file, corrupt_path+file)
            
#    print("CORRUPT FILES FOUND:"+str(len(corrupt_files)))
#    print(corr_files)
        
    if(len(repeated_files) > 0):
        repeated_path = "IOfiles/RepeatedSfiles/"
        if not os.path.exists(repeated_path):
            os.makedirs(repeated_path)
            
        for file in repeated_files:
            os.rename(path+file, repeated_path+file)
            
    for error_type in corr_files:
        error_path = "IOfiles/"+error_type+"/"
        if not os.path.exists(error_path):
            os.makedirs(error_path)
            
        for file in corr_files[error_type]:
            os.rename(path+file, error_path+file)
#            print(file)
#        print(error_path)
            
    print("REPEATED FILES FOUND:"+str(len(repeated_files)))
    
    return corr_files
    
#    print(repeated_files)
    
#%%        
#### Testing
path = "IOfiles/Filtered_RSNC_Sfiles/"
analyzer = SfileAnalyzer(path)
analyzer.get_sfiles()
analyzer.group_by_magnitude()

#%%
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
     ESTE METODO DEBE IR A TELLURICOTOOLS
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
    
#%% Export in txt location of seismic events
def export_seismic_locations2D(sfiles):
    with open('event_locations2D.csv', 'a') as the_file:
        the_file.write('Latitude,Longitude\n')
        for sfile in sfiles:
            lat = sfile.type_1['LATITUDE']
            lon = sfile.type_1['LONGITUDE']
            the_file.write(lat+','+lon+'\n')
            

#%% Export in txt locations of seismic events, including depth
def export_seismic_locations3D(sfiles):
    with open('event_locations3D.csv', 'a') as the_file:
        the_file.write('Latitude,Longitude,Depth\n')
        for sfile in sfiles:
            lat = sfile.type_1['LATITUDE']
            lon = sfile.type_1['LONGITUDE']
            depth = sfile.type_1['DEPTH']
            the_file.write(lat+','+lon+','+depth+'\n')           
        

















































    
#%% Check if the record of P wave is attached only to the Z component
def check_Pcomponent2(sfiles):
    z = 0
    Z_st = {}
    z_files = []
    e = 0
    E_st = {}
    e_files = []
    n = 0
    N_st = {}
    n_files = []
    for sfile in sfiles:
        stations = sfile.type_7
        for station in stations:
            sp = station['SP']
            iphas = station['IPHAS']
            station_name = station['STAT']
            
            if(sp == 'E'):
                print(station_name)
                print(sfile.filename)
            
            if('P' in iphas and len(sp) > 1):
                #print(sp)
                if(sp[1] =='Z'):
                    z+=1
                    if(station_name in Z_st):
                        Z_st[station_name] += 1
                    else:
                        Z_st[station_name] = 1
                        
                    z_files.append(sfile.filename)
                elif(sp[1] =='N'):
                    n+=1
                    if(station_name in N_st):
                        N_st[station_name] += 1
                    else:
                        N_st[station_name] = 1
                        
                    n_files.append(sfile.filename)
                elif(sp[1] =='E'):
                    e+=1
                    if(station_name in E_st):
                        E_st[station_name] += 1
                    else:
                        E_st[station_name] = 1
                        
                    e_files.append(sfile.filename)
                    
    print('P in comp Z:'+str(z))
    print('P in comp E:'+str(e))
    print('P in comp N:'+str(n))
    
    z_files = TelluricoTools.remove_duplicates(z_files)
    e_files = TelluricoTools.remove_duplicates(e_files)
    n_files = TelluricoTools.remove_duplicates(n_files)
    
    return z_files,e_files,n_files

#%% Check if the record of P wave is attached only to the Z component
def check_Pcomponent(sfiles):
    z = {}
    n = {}
    e = {}
    for sfile in sfiles:
        stations = sfile.type_7
        for station in stations:
            sp = station['SP']
            iphas = station['IPHAS']
            station_name = station['STAT']
            mag = sfile.type_1['TYPE_OF_MAGNITUDE_1']
            
            if('P' in iphas and len(sp) > 1):
                if(sp[1] == 'Z'):
                    if(sfile.filename in z):
                        if(station_name not in z[sfile.filename][1]):
                            z[sfile.filename][1].append(station_name)
                    else:
                        z[sfile.filename] = (mag,[station_name])
                        
                if(sp[1] == 'N'):
                    if(sfile.filename in n):
                        if(station_name not in n[sfile.filename][1]):
                            n[sfile.filename][1].append(station_name)
                    else:
                        n[sfile.filename] = (mag,[station_name])
                if(sp[1] == 'E'):
                    if(sfile.filename in e):
                        if(station_name not in e[sfile.filename][1]):
                            e[sfile.filename][1].append(station_name)
                    else:
                        e[sfile.filename] = (mag,[station_name])
                    
#    print('P in comp Z:'+str(z))
#    print('P in comp E:'+str(e))
#    print('P in comp N:'+str(n))
    
#    z_files = TelluricoTools.remove_duplicates(z_files)
#    e_files = TelluricoTools.remove_duplicates(e_files)
#    n_files = TelluricoTools.remove_duplicates(n_files)
                        
    with open('z_Pwave.csv', 'a') as the_file:
        the_file.write('Filename,Magnitude,Stations\n')
        for zfile in z:
            stats = ""
            for stat in z[zfile][1]:
                stats+="/"+stat
            the_file.write(zfile+','+z[zfile][0]+','+stats+'\n')
            
            
    with open('n_Pwave.csv', 'a') as the_file:
        the_file.write('Filename,Magnitude,Stations\n')
        for nfile in n:
            stats = ""
            for stat in n[nfile][1]:
                stats+="/"+stat
            the_file.write(nfile+','+n[nfile][0]+','+stats+'\n')
            
            
    with open('e_Pwave.csv', 'a') as the_file:
        the_file.write('Filename,Magnitude,Stations\n')
        for efile in e:
            stats = ""
            for stat in e[efile][1]:
                stats+=" / "+stat
            the_file.write(efile+','+e[efile][0]+','+stats+'\n')

    
    return z,e,n


#%%
def concat_files(file1, file2):
    input_file = open(file2, 'r') 
    with open(file1, 'a') as the_file:
        for line in input_file:
            the_file.write(line)

#%%
def concat_mul_files(list_files):
    with open(list_files.pop(0),'a') as final_file:
        for file in list_files:
            input_file = open(file, 'r')
            for line in input_file:
                final_file.write(line)

#%%

def check_len_sfiles(sfiles, threeshold):
    corrupt = 0
    for sfile in sfiles:
        try:
            name = sfile.type_6['BINARY_FILENAME']
            l = len(name)
            if l <= threeshold:
                print(l+'-->'+name)
        except:
            print(sfile.filename)
            corrupt += 1
    
    print("Corrupt:"+str(corrupt))




    