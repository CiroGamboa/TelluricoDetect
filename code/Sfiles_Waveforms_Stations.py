#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:14:42 2018

@author: Julián Daío Miranda
"""

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
waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
stations = ['RUS','BRR','PAM','PTB']
stations_prov = copy.copy(stations)
waveforms_stations = []
delete_sfile = []
total_size = 0

sfileAnalyzer = SfileAnalyzer(sfiles_path)
sfileAnalyzer.get_sfiles()
sfiles = sfileAnalyzer.sfiles
waveforms = []

for sfile in sfiles:
    if(hasattr(sfile, 'type_6')):
        for station in sfile.type_7:
            if station['STAT'] in stations_prov:
                stations_prov.pop(stations_prov.index(station['STAT']))
            if len(stations_prov) == 0:
                waveforms_stations.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
                total_size += os.stat(sfiles_path + sfile.filename).st_size
                break
    else:
        delete_sfile.append(sfile.filename)
    stations_prov = copy.copy(stations)
print('Total with req stations: ' + str(len(waveforms_stations)))


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

file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
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

file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
f = open(file_var_name, 'rb')
waveforms = pickle.load(f)
f.close()

# Validate waveforms
for waveform in waveforms:
    try:
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

file_var_name =  '/home/administrador/Tellurico/Variables/Total_ProcessedWaveforms_HD1.pckl' ## TODO: variable name to be exported
toSave = [waveforms_valid, weights]
f = open(file_var_name, 'wb')
pickle.dump(toSave, f)
f.close()