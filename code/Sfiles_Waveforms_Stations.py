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