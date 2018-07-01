#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:14:42 2018

@author: Julián Daío Miranda
"""

#%% Specific stations in sfiles

# Import the libraries
from obspy import read
from tools import TelluricoTools
import os
from SfileAnalyzer import SfileAnalyzer
from Waveform import Waveform
import gc
from multiprocessing import Process
import copy


#waveforms_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Filtered_RSNC_Waveforms/'
#sfiles_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Filtered_RSNC_Sfiles/'
#waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
#sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
waveforms_path = '/home/tellurico/Tellurico/Archivos/Waveforms/' # CCA
sfiles_path = '/home/tellurico/Tellurico/Archivos/Sfiles/' # CCA

stations = ['RUS','BRR','PAM','PTB']
stations_prov = copy.copy(stations)
waveforms_stations = []
delete_sfile = []
total_size = 0

sfileAnalyzer = SfileAnalyzer(sfiles_path)
sfileAnalyzer.get_sfiles()
sfiles = sfileAnalyzer.sfiles
waveforms = []
waveform_names = []

for sfile in sfiles:
    if(hasattr(sfile, 'type_6')):
        for station in sfile.type_7:
            if station['STAT'] in stations_prov:
                stations_prov.pop(stations_prov.index(station['STAT']))
            if len(stations_prov) == 0:
                if sfile.type_6['BINARY_FILENAME'] not in waveform_names:
                    waveform_names.append(sfile.type_6['BINARY_FILENAME'])
                    waveforms_stations.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
                    total_size += os.stat(sfiles_path + sfile.filename).st_size
                    break
    else:
        delete_sfile.append(sfile.filename)
    stations_prov = copy.copy(stations)
print('Total with req stations: ' + str(len(waveforms_stations)))

#%% Specific stations in sfiles and waveforms

# Import the libraries
from obspy import read
from tools import TelluricoTools
import os
from SfileAnalyzer import SfileAnalyzer
from Waveform import Waveform
import gc
from multiprocessing import Process
import copy
from pathlib import Path
import pickle

#waveforms_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Filtered_RSNC_Waveforms/'
#sfiles_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Filtered_RSNC_Sfiles/'
#waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
#sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA

#waveforms_path = '/home/tellurico/Tellurico/Archivos/Waveforms/' # CCA
#sfiles_path = '/home/tellurico/Tellurico/Archivos/Sfiles/' # CCA

waveforms_path = '/home/i201-20/Tellurico/Archivos/Waveforms/' # CCA
sfiles_path = '/home/i201-20/Tellurico/Archivos/Sfiles/' # CCA
waveforms_path_HD2 = '/media/i201-20/Tellurico_Dataset2/' # External HD2 - Local

stations = ['RUS','BRR','PAM']
stations_prov = copy.copy(stations)
waveforms_stations = []
delete_sfile = []
total_size = 0

sfileAnalyzer = SfileAnalyzer(sfiles_path)
sfileAnalyzer.get_sfiles()
sfiles = sfileAnalyzer.sfiles
waveforms = []
waveform_names = []

for sfile in sfiles:
    if(hasattr(sfile, 'type_6')):
        for station in sfile.type_7:
            if station['STAT'] in stations_prov:
                stations_prov.pop(stations_prov.index(station['STAT']))
            if len(stations_prov) == 0:
                if sfile.type_6['BINARY_FILENAME'] not in waveform_names:
                    waveform_path_and_name = waveforms_path_HD2 + 'download.php?file=' + sfile.type_6['BINARY_FILENAME'][0:4] + '%2F' + sfile.type_6['BINARY_FILENAME'][5:7]+ '%2F' + sfile.type_6['BINARY_FILENAME']
                    if Path(waveform_path_and_name).exists():
                        waveform_names.append(sfile.type_6['BINARY_FILENAME'])
                        waveforms_stations.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
                        total_size += os.stat(sfiles_path + sfile.filename).st_size
                        break
    else:
        delete_sfile.append(sfile.filename)
    stations_prov = copy.copy(stations)
print('Total files in HD2 with req stations: ' + str(len(waveforms_stations)))

file_var_name =  '/home/i201-20/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2_' + str(len(stations)) + 'stat.pckl'
f = open(file_var_name, 'wb')
pickle.dump(waveforms_stations, f)
f.close()

#%% All dataset -sfiles- read

import os
from SfileAnalyzer import SfileAnalyzer
from Waveform import Waveform
import gc
import copy
import pickle

''' DATASET READING AND PRE-PROCESSING '''
        
waveforms_path = '/media/administrador/Tellurico_Dataset1/' #Disco Duro 1
sfiles_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Filtered_RSNC_Sfiles/' #Local
sfileAnalyzer = SfileAnalyzer(sfiles_path)
sfileAnalyzer.get_sfiles()
sfiles = sfileAnalyzer.sfiles
waveforms = []
total_sfiles = 0

for sfile in sfiles:
    if(hasattr(sfile, 'type_6')):
        waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
        total_sfiles += 1

print('Total sfiles: ' + str(total_sfiles))

#file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms.pckl' ## variable name to be exported CCA

f = open(file_var_name, 'wb')
pickle.dump(waveforms, f)
f.close()

#%% #%% All dataset -waveforms- read

from obspy import read
import os
from SfileAnalyzer import SfileAnalyzer
from Waveform import Waveform
import gc
import copy
import pickle
from pathlib import Path

index_wrong = 0
index_valid = 0
waveforms_valid = {}
weights = []
error_flag = '0'

not_exists = []

#file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms.pckl' ## variable name to be exported CCA

f = open(file_var_name, 'rb')
waveforms = pickle.load(f)
f.close()

# Validate waveforms
for waveform in waveforms:
    try:
        waveform.waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
        waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
        if Path(waveform_path_and_name).exists():
            st = read(waveform_path_and_name)
            error_flag = '1'
            waveform.set_st(st)
            error_flag = '2'
            file_weigth = os.stat(waveform_path_and_name).st_size
            error_flag = '3'
            waveforms_valid[waveform] = file_weigth
            error_flag = '4'
            weights.append(file_weigth)
            error_flag = '5'
            index_valid += 1
            print('Waveform' + str(index_valid) + ' included')
        else:
            not_exists.append(waveform_path_and_name)
#        gc.collect()
    except:
        index_wrong += 1
        print('Error in waveform - ' + error_flag)
    error_flag = '0'
        
waveforms_valid = TelluricoTools.sort(waveforms_valid)
waveforms_valid.reverse()
print(str(len(waveforms_valid)) + ' valid files with stations')

#file_var_name =  '/home/administrador/Tellurico/Variables/Total_ProcessedWaveforms_HD1.pckl' ## TODO: variable name to be exported
file_var_name =  '/home/tellurico-admin/Variables/Total_ProcessedWaveforms_HD1.pckl' ## variable name to be exported CCA

toSave = [waveforms_valid, weights]
f = open(file_var_name, 'wb')
pickle.dump(toSave, f)
f.close()

#%% Validate same waveform-name in sfiles and save list again

file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms.pckl' ## variable name to be exported CCA
        
f = open(file_var_name, 'rb')
waveforms = pickle.load(f)
f.close()
total = 0

path_names = []
waveforms_valid = []

for i in range(0,len(waveforms)):
    waveforms[i].waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
    waveform_path_and_name = waveforms[i].waveform_path + 'download.php?file=' + waveforms[i].waveform_filename[0:4] + '%2F' + waveforms[i].waveform_filename[5:7]+ '%2F' + waveforms[i].waveform_filename
    if waveform_path_and_name not in path_names:
        path_names.append(waveform_path_and_name)
        waveforms_valid.append(waveforms[i])

print('Non repeated Total files: ' + str(len(path_names)))
print('Waveforms length: ' + str(len(waveforms_valid)))

f = open(file_var_name, 'wb')
pickle.dump(waveforms_valid, f)
f.close()

waveforms = copy.copy(waveforms_valid)
waveforms_valid = []

for i in range(0,len(waveforms)):
    waveforms[i].waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
    waveform_path_and_name = waveforms[i].waveform_path + 'download.php?file=' + waveforms[i].waveform_filename[0:4] + '%2F' + waveforms[i].waveform_filename[5:7]+ '%2F' + waveforms[i].waveform_filename
    if Path(waveform_path_and_name).exists():
        waveforms_valid.append(waveforms[i])
        
print('Total waveforms in HD2: ' + str(len(waveforms_valid)))

file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms_HD2.pckl'
f = open(file_var_name, 'wb')
pickle.dump(waveforms_valid, f)
f.close()
    
#%% Waveform stats analyzer

from obspy import read
import os
from SfileAnalyzer import SfileAnalyzer
from Waveform import Waveform
import gc
import copy
import pickle
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as ml

#file_var_name =  '/home/tellurico/Tellurico/Variables/HD2_Files/Total_ProcWaveforms_HD2.pckl' ## CCA
#        
#f = open(file_var_name, 'rb')
#waveform_stats = pickle.load(f)
#f.close()
#
#waveform_filenames = list(waveform_stats)
#waveform_filenames.sort()

statistics = {}
comps = np.zeros(len(waveform_filenames))
sr = np.zeros(len(waveform_filenames))
total_components = []
comps_station = {}

for waveform_filename in waveform_filenames:
    stats = waveform_stats[waveform_filename]
    for stat in stats:
        if(stat.station not in statistics):
            dictio = {'SP': copy.copy(comps), 'SR': copy.copy(sr)}
            comps_station[stat.station] = []
            statistics[stat.station] = dictio
        if(stat.channel not in total_components):
            total_components.append(stat.channel)

comps_station_copy = copy.copy(comps_station)
index = 0
w = []

for waveform_filename in waveform_filenames:
    stats = waveform_stats[waveform_filename]
    for stat in stats:
        ((statistics[stat.station])['SR'])[index] = stat.sampling_rate
        if(stat.channel[1] == 'H' and len(stat.channel) == 3 and stat.channel not in comps_station_copy[stat.station]):
            comps_station_copy[stat.station].append(stat.channel)
    for stat in comps_station_copy:
        ((statistics[stat])['SP'])[index] = len(comps_station_copy[stat])
    if index == 4600:
        print(comps_station_copy)
    comps_station_copy = copy.copy(comps_station)
    index += 1

stations_quant = len(statistics)
fig = ml.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for station in statistics:
    if stations_quant > 0:
        ax1.plot((statistics[station])['SP'])
        ax2.plot((statistics[station])['SR'])
        stations_quant -= 1

fig = ml.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Components')
ax1.plot((statistics['BRR'])['SP'])
ax1.plot((statistics['RUS'])['SP'])
ax1.plot((statistics['PTB'])['SP'])
ax1.plot((statistics['PAM'])['SP'])
ax2.set_title('Sampling Rate')
ax2.plot((statistics['BRR'])['SR'])
ax2.plot((statistics['RUS'])['SR'])
ax2.plot((statistics['PTB'])['SR'])
ax2.plot((statistics['PAM'])['SR'])

fig = ml.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Components - BRR')
ax1.plot((statistics['BRR'])['SP'])
ax2.set_title('Sampling Rate - BRR')
ax2.plot((statistics['BRR'])['SR'])

fig = ml.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Components - RUS')
ax1.plot((statistics['RUS'])['SP'])
ax2.set_title('Sampling Rate - RUS')
ax2.plot((statistics['RUS'])['SR'])

fig = ml.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Components - PAM')
ax1.plot((statistics['PAM'])['SP'])
ax2.set_title('Sampling Rate - PAM')
ax2.plot((statistics['PAM'])['SR'])

fig = ml.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Components - PTB')
ax1.plot((statistics['PTB'])['SP'])
ax2.set_title('Sampling Rate - PTB')
ax2.plot((statistics['PTB'])['SR'])

#%% Execute prototypes

from prototype_v0 import p_wave_extractor

p_wave_extractor(['RUS','BRR','PAM','PTB'],200,0.5)

#%% Concat all stations and P & Noise wave windows

from obspy import read
import os
from SfileAnalyzer import SfileAnalyzer
from Waveform import Waveform
import gc
import copy
import pickle
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as ml
import fnmatch

percents = [0.5, 0.9]
stats_quant = [3, 4]

total_waveforms = []
wave_stats = {} # Stations key dictionary
wave_percent = {} 

for stats in stats_quant:
    for percent in percents:
        wave_percent[percent] = []
    wave_stats[stats] = copy.copy(wave_percent)

p_wave_path = '/home/tellurico/Tellurico/Variables/HD2_Files/P_Waves_Noise/'

for stats in stats_quant:
    for percent in percents:
        path = p_wave_path + str(stats) + 'stat/' + str(percent) + '/'
        list_of_files = os.listdir(path)
        pattern = "Total_P-wave*"
        final_dict = {}
        for filename in list_of_files:  
            if fnmatch.fnmatch(filename, pattern):
                file_var_name =  path + filename
                f = open(file_var_name, 'rb')
                waveform_dict = pickle.load(f)
                for array in waveform_dict:
                    total_waveforms.append(array)
                f.close()
        (wave_stats[stats])[percent] = copy.copy(total_waveforms)
        total_waveforms = []

p_wave_path += 'Total_P-wave-noise_HD2.pckl'
f = open(p_wave_path, 'wb')
pickle.dump(wave_stats, f)
f.close()