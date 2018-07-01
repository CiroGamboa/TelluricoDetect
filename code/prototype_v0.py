# -*- coding: utf-8 -*-
"""
Prototipo 0 de Tellurico
Es necesario establecer el patron de documentacion con sphinx
Plantillas y demas
Este prototipo tiene como objetivo arrancar el desarrollo
A partir del prototipo 1 y en adelante, vinculados con sprints
Se tendra todo documentado de forma estandar
Toda la documentacion debe ser en ingles
"""

# Import the libraries
from obspy import read
from tools import TelluricoTools
import numpy as np
#from tools import SeismicInfo 
#from tools import Transformations
from tools import TimeDomain_Attributes
#from tools import FreqDomain_Attributes
from tools import NonLinear_Attributes
#from tools import Correlation
#from obspy.signal.polarization import eigval
#import numpy as np
#import matplotlib.pyplot as ml
#from TraceComponent import TraceComponent
#from TraceGroup import TraceGroup
#from Event import Event
#import obspy.core.utcdatetime as dt
#import obspy.signal.filter as filt
#from obspy.signal.trigger import classic_sta_lta
#from obspy.signal.trigger import plot_trigger
#from obspy.imaging import spectrogram as spec
import os
import fnmatch
#import time
from SfileAnalyzer import SfileAnalyzer
#from Sfile import Sfile
from Waveform import Waveform
import gc
from multiprocessing import Process
import copy
import pickle
from pathlib import Path

class prototype_v0:
    
    def __init__(self):
        self.read_files()
    
    def read_files(self):

        ''' DATASET READING AND PRE-PROCESSING '''
        
        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
        # waveforms_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Waveforms/'
        sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
        # sfiles_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Sfiles/'
        sfileAnalyzer = SfileAnalyzer(sfiles_path)
        sfileAnalyzer.get_sfiles()
        sfiles = sfileAnalyzer.sfiles
        waveforms = []
        for sfile in sfiles:
            waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
        
        index = 0
        waveforms_valid = []
        for waveform in waveforms: #[32:64]
            try:
                read(waveform.waveform_path + waveform.waveform_filename)
                waveforms_valid.append(waveform)
            except:
                print(waveform.waveform_filename)
                index += 1
        print(index)
        
        ''' DATASET ATRIBUTES '''
        
        cores = os.cpu_count() - 1 # CPU cores
        step = int(len(waveforms_valid)/cores)
        stat = 'BRR'
        p = [None]*cores
        
        for i in range(1, (cores+1)):
            p[i-1] = Process(target=self.attributes, args=(('att_p' + str(i) + '.txt'),
                 waveforms_valid[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms_valid)-1)))],
                 stat,(i-1)*step,i))
            
        for i in range(0, cores):
            p[i].start()
        
        for i in range(0, cores):
            p[i].join()
        
#        self.concat_attrs_files('/home/administrador/Tellurico/TelluricoDetect/code/')
        self.concat_attrs_files('/home/tellurico-admin/TelluricoDetect/code/') #CCA
    
    # Feature extraction
    def attributes(self,filename, waveforms_valid, stat, begin, core):     
        index = begin
        observ_signal = {}
        observ_noise = {}
        with open(filename, 'a') as the_file:
            for waveform in waveforms_valid:
                [newEvent, stats_sort] = waveform.get_event()
                if(stat in newEvent.trace_groups):
                    [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
                    [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                    [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                    [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                    
                    observ_signal['DOP'] = TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)
                    observ_signal['RV2T'] = TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)
                    observ_signal['EntropyZ'] = NonLinear_Attributes.signal_entropy(p_signal_X)[1]
                    observ_signal['EntropyN'] = NonLinear_Attributes.signal_entropy(p_signal_Y)[1]
                    observ_signal['EntropyE'] = NonLinear_Attributes.signal_entropy(p_signal_Z)[1]
                    observ_signal['KurtosisZ'] = TimeDomain_Attributes.signal_kurtosis(p_signal_X)
                    observ_signal['KurtosisN'] = TimeDomain_Attributes.signal_kurtosis(p_signal_Y)
                    observ_signal['KurtosisE'] = TimeDomain_Attributes.signal_kurtosis(p_signal_Z)
                    observ_signal['SkewZ'] = TimeDomain_Attributes.signal_skew(p_signal_X)
                    observ_signal['SkewN'] = TimeDomain_Attributes.signal_skew(p_signal_Y)
                    observ_signal['SkewE'] = TimeDomain_Attributes.signal_skew(p_signal_Z)
#                    observ_signal['LyapunovZ'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_X)
#                    observ_signal['LyapunovN'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_Y)
#                    observ_signal['LyapunovE'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_Z)
                    observ_signal['CDZ'] = NonLinear_Attributes.corr_CD(p_signal_X, 1)
                    observ_signal['CDN'] = NonLinear_Attributes.corr_CD(p_signal_Y, 1)
                    observ_signal['CDE'] = NonLinear_Attributes.corr_CD(p_signal_Z, 1)
                    
                    observ_noise['DOP'] = TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)
                    observ_noise['RV2T'] = TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)
                    observ_noise['EntropyZ'] = NonLinear_Attributes.signal_entropy(noise_X)[1]
                    observ_noise['EntropyN'] = NonLinear_Attributes.signal_entropy(noise_Y)[1]
                    observ_noise['EntropyE'] = NonLinear_Attributes.signal_entropy(noise_Z)[1]
                    observ_noise['KurtosisZ'] = TimeDomain_Attributes.signal_kurtosis(noise_X)
                    observ_noise['KurtosisN'] = TimeDomain_Attributes.signal_kurtosis(noise_Y)
                    observ_noise['KurtosisE'] = TimeDomain_Attributes.signal_kurtosis(noise_Z)
                    observ_noise['SkewZ'] = TimeDomain_Attributes.signal_skew(noise_X)
                    observ_noise['SkewN'] = TimeDomain_Attributes.signal_skew(noise_Y)
                    observ_noise['SkewE'] = TimeDomain_Attributes.signal_skew(noise_Z)
#                    observ_noise['LyapunovZ'] = NonLinear_Attributes.lyapunov_exp_max(noise_X)
#                    observ_noise['LyapunovN'] = NonLinear_Attributes.lyapunov_exp_max(noise_Y)
#                    observ_noise['LyapunovE'] = NonLinear_Attributes.lyapunov_exp_max(noise_Z)
                    observ_noise['CDZ'] = NonLinear_Attributes.corr_CD(noise_X, 1)
                    observ_noise['CDN'] = NonLinear_Attributes.corr_CD(noise_Y, 1)
                    observ_noise['CDE'] = NonLinear_Attributes.corr_CD(noise_Z, 1)
                
                    str_row = ''
                    for feature in observ_signal:
                        str_row += str(observ_signal[feature])+','
                    str_row += str(1)
                    the_file.write(str_row+'\n')
                    
                    str_row = ''
                    for feature in observ_noise:
                        str_row += str(observ_noise[feature])+','
                    str_row += str(0)
                    the_file.write(str_row+'\n')
                    
                print('Waveform ' + str(index) + ' done - core ' + str(core))
                index += 1
                gc.collect()
        
    # Concat feature extraction files into one
    def concat_attrs_files(self, path):
        list_of_files = os.listdir(path)
        pattern = "att_p*"  
        with open('attributes_matrix_prot03.txt','a') as final_file:
            final_file.write('DOP; RV2T; EntropyZ; EntropyN; EntropyE; KurtosisZ; KurtosisN; KurtosisE; SkewZ; SkewN; SkewE; CDZ; CDN; CDE')
            for filename in list_of_files:  
                if fnmatch.fnmatch(filename, pattern):
                    input_file = open(filename, 'r')
                    for line in input_file:
                        final_file.write(line)
                
                
                
class prototype_v3:
    
    def __init__(self):
#        self.stations = ['RUS','BRR','PAM','PTB']
        self.stations = ['RUS','BRR','PAM', 'ALL']
        file_var_name = self.read_files()
        self.cores_distr(file_var_name)
    
    def read_files(self):

        ''' DATASET READING AND PRE-PROCESSING '''
        
#        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
#        sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
        sfiles_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Sfiles/' #Local
        sfileAnalyzer = SfileAnalyzer(sfiles_path)
        sfileAnalyzer.get_sfiles()
        sfiles = sfileAnalyzer.sfiles
        waveforms = []
        stations_prov = copy.copy(self.stations)
        
        if 'ALL' in stations_prov: # If all stations are analyzed
            for sfile in sfiles:
                waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
        else: # If certain stations are analyzed
            for sfile in sfiles:
                if(hasattr(sfile, 'type_6')):
                    for station in sfile.type_7:
                        if station['STAT'] in stations_prov and station['PHAS'].strip() == 'P':
                            stations_prov.pop(stations_prov.index(station['STAT']))
                        if len(stations_prov) == 0:
                            waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
                            break
                stations_prov = copy.copy(self.stations)
        
        index = 0
        waveforms_valid = {}
        weights = []
        
        # Validate waveforms
        for waveform in waveforms:
            try:
                st = read(waveform.waveform_path + waveform.waveform_filename)
                if 'ALL' in stations_prov: # If all stations are analyzed
                    file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
                    waveforms_valid[waveform] = file_weigth
                    weights.append(file_weigth)
                else:
                    for trace in st:
                        if(trace.stats.station in stations_prov):
                            stations_prov.pop(stations_prov.index(trace.stats.station))
                        if len(stations_prov) == 0:
                            file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
                            waveforms_valid[waveform] = file_weigth
                            weights.append(file_weigth)
                            break
                    stations_prov = copy.copy(self.stations)
            except:
                index += 1
        waveforms_valid = TelluricoTools.sort(waveforms_valid)
        waveforms_valid.reverse()
        print(str(len(waveforms_valid)) + ' valid files with stations')
        
        file_var_name =  'waveforms_valid_prot03.pckl' ## TODO: variable name to be exported
        toSave = [waveforms_valid, weights]
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()
        
        return file_var_name
        
    def cores_distr(self, file_var_name):
        
        f = open(file_var_name, 'rb')
        toRead = pickle.load(f)
        f.close()
        
        waveforms_valid = toRead[0]
        weights = toRead[1]
        
        ''' DATASET DISTRIBUTION IN CORES '''
        
        cores_quant = os.cpu_count() - 1 # CPU cores
        cores = []
        for i in range(0, cores_quant): cores.append([])
        
        w_flag = False
        max_weigth = 1.15*(np.sum(weights)/cores_quant)
        if(max_weigth < max(weights)): 
            max_weigth = max(weights)
            w_flag = True
        
        space = max_weigth
        for i in range(0, len(cores)):    
            for waveform in waveforms_valid:
                if(waveform[1] <= space):
                    cores[i].append(waveform[0])
                    space -= waveform[1]
                    waveforms_valid.pop(waveforms_valid.index(waveform))
                    weights.pop(weights.index(waveform[1]))
                    if(w_flag):
                        max_weigth = 1.15*(np.sum(weights)/(cores_quant-i))
                        if(max_weigth < max(weights)): 
                            max_weigth = max(weights)
                            w_flag = True
                        else:
                            w_flag = False
            space = max_weigth
        
        ''' DATASET FEATURES EXTRACTION '''
        
        p = []
        for i in range(0, cores_quant): p.append(None)
        
        for i in range(0, cores_quant):
            p[i] = Process(target=self.attributes, args=(('att_p' + str(i+1) + '.txt'), 
             cores[i], self.stations, len(cores[i]), (i+1)))
            
        for i in range(0, cores_quant):
            p[i].start()
        
        for i in range(0, cores_quant):
            p[i].join()
        
#        self.concat_attrs_files('/home/administrador/Tellurico/TelluricoDetect/code/')
        self.concat_attrs_files('/home/tellurico-admin/TelluricoDetect/code/') #CCA
    
    # Feature extraction
    def attributes(self,filename, waveforms_valid, stations, total, core):     
        index = 1
        observ_signal = ''
        observ_noise = ''
        with open(filename, 'a') as the_file:
            for waveform in waveforms_valid:
                [newEvent, stats_sort] = waveform.get_event()
                for stat in stations:
                    if(stat in newEvent.trace_groups):
                        [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
                        [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                        [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                        [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                        
                        observ_signal += str(TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)) + ','
                        observ_signal += str(TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)) + ','
                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_X)[1]) + ','
                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_Y)[1]) + ','
                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_Z)[1]) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_X)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_Y)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_Z)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_X)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_Y)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_Z)) + ','
                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_X, 1)) + ','
                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_Y, 1)) + ','
                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_Z, 1)) + ','
                        
                        observ_noise += str(TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)) + ','
                        observ_noise += str(TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)) + ','
                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_X)[1]) + ','
                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_Y)[1]) + ','
                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_Z)[1]) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_X)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_Y)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_Z)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_X)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_Y)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_Z)) + ','
                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_X, 1)) + ','
                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_Y, 1)) + ','
                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_Z, 1)) + ','

                observ_signal += str(1)
                the_file.write(waveform.waveform_filename + ',' + observ_signal+'\n')
                observ_noise += str(0)
                the_file.write(waveform.waveform_filename + ',' + observ_noise+'\n')
                observ_signal = ''
                observ_noise = ''
                        
                print('Waveform ' + str(index) + '/' + str(total) + ' done - core ' + str(core))
                index += 1
                gc.collect()
            print('Core ' + str(core) + ' DONE')
        
    # Concat feature extraction files into one
    def concat_attrs_files(self, path):
        list_of_files = os.listdir(path)
        pattern = "att_p*"  
        with open('attributes_matrix_prot03_3stats.txt','a') as final_file:
            final_file.write('Filename; DOP; RV2T; EntropyZ; EntropyN; EntropyE; KurtosisZ; KurtosisN; KurtosisE; SkewZ; SkewN; SkewE; CDZ; CDN; CDE')
            for filename in list_of_files:  
                if fnmatch.fnmatch(filename, pattern):
                    input_file = open(filename, 'r')
                    for line in input_file:
                        final_file.write(line)
                    os.remove(filename)
                    
                 
class prototype_v4:
    
    def __init__(self):
#        self.stations = ['RUS','BRR','PAM','PTB']
        self.stations = ['RUS','BRR','PAM', 'ALL']
        file_var_name = self.read_files()
        self.cores_distr(file_var_name)
    
    def read_files(self):

        ''' DATASET READING AND PRE-PROCESSING '''
        
#        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
#        sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
        sfiles_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Sfiles/' #Local
        sfileAnalyzer = SfileAnalyzer(sfiles_path)
        sfileAnalyzer.get_sfiles()
        sfiles = sfileAnalyzer.sfiles
        waveforms = []
        stations_prov = copy.copy(self.stations)
        
        if 'ALL' in stations_prov: # If all stations are analyzed
            for sfile in sfiles:
                waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
        else: # If certain stations are analyzed
            for sfile in sfiles:
                if(hasattr(sfile, 'type_6')):
                    for station in sfile.type_7:
                        if station['STAT'] in stations_prov and station['PHAS'].strip() == 'P':
                            stations_prov.pop(stations_prov.index(station['STAT']))
                        if len(stations_prov) == 0:
                            waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
                            break
                stations_prov = copy.copy(self.stations)
        
        index = 0
        waveforms_valid = {}
        weights = []
        
        # Validate waveforms
        for waveform in waveforms:
            try:
                st = read(waveform.waveform_path + waveform.waveform_filename)
                if 'ALL' in stations_prov: # If all stations are analyzed
                    waveform.set_st(st)
                    file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
                    waveforms_valid[waveform] = file_weigth
                    weights.append(file_weigth)
                else:
                    for trace in st:
                        if(trace.stats.station in stations_prov):
                            stations_prov.pop(stations_prov.index(trace.stats.station))
                        if len(stations_prov) == 0:
                            waveform.set_st(st)
                            file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
                            waveforms_valid[waveform] = file_weigth
                            weights.append(file_weigth)
                            break
                    stations_prov = copy.copy(self.stations)
            except:
                index += 1
        waveforms_valid = TelluricoTools.sort(waveforms_valid)
        waveforms_valid.reverse()
        print(str(len(waveforms_valid)) + ' valid files with stations')
        
        file_var_name =  'waveforms_valid_prot03.pckl' ## TODO: variable name to be exported
        toSave = [waveforms_valid, weights]
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()
        
        return file_var_name
        
    def cores_distr(self, file_var_name):
        
        f = open(file_var_name, 'rb')
        toRead = pickle.load(f)
        f.close()
        
        waveforms_valid = toRead[0]
        weights = toRead[1]
        
        ''' DATASET DISTRIBUTION IN CORES '''
        
        cores_quant = os.cpu_count() - 1 # CPU cores
        cores = []
        for i in range(0, cores_quant): cores.append([])
        
        w_flag = False
        max_weigth = 1.15*(np.sum(weights)/cores_quant)
        if(max_weigth < max(weights)): 
            max_weigth = max(weights)
            w_flag = True
        
        space = max_weigth
        for i in range(0, len(cores)):    
            for waveform in waveforms_valid:
                if(waveform[1] <= space):
                    cores[i].append(waveform[0])
                    space -= waveform[1]
                    waveforms_valid.pop(waveforms_valid.index(waveform))
                    weights.pop(weights.index(waveform[1]))
                    if(w_flag):
                        max_weigth = 1.15*(np.sum(weights)/(cores_quant-i))
                        if(max_weigth < max(weights)): 
                            max_weigth = max(weights)
                            w_flag = True
                        else:
                            w_flag = False
            space = max_weigth
        
        ''' DATASET FEATURES EXTRACTION '''
        
        p = []
        for i in range(0, cores_quant): p.append(None)
        
        for i in range(0, cores_quant):
            p[i] = Process(target=self.attributes, args=(('att_p' + str(i+1) + '.txt'), 
             cores[i], self.stations, len(cores[i]), (i+1)))
            
        for i in range(0, cores_quant):
            p[i].start()
        
        for i in range(0, cores_quant):
            p[i].join()
        
#        self.concat_attrs_files('/home/administrador/Tellurico/TelluricoDetect/code/')
        self.concat_attrs_files('/home/tellurico-admin/TelluricoDetect/code/') #CCA
    
    # Feature extraction
    def attributes(self,filename, waveforms_valid, stations, total, core):     
        index = 1
        observ_signal = ''
        observ_noise = ''
        with open(filename, 'a') as the_file:
            for waveform in waveforms_valid:
                [newEvent, stats_sort] = waveform.get_event_st()
                for stat in stations:
                    if(stat in newEvent.trace_groups):
                        [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
                        [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                        [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                        [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
                        
                        observ_signal += str(TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)) + ','
                        observ_signal += str(TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)) + ','
                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_X)[1]) + ','
                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_Y)[1]) + ','
                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_Z)[1]) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_X)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_Y)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_Z)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_X)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_Y)) + ','
                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_Z)) + ','
                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_X, 1)) + ','
                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_Y, 1)) + ','
                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_Z, 1)) + ','
                        
                        observ_noise += str(TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)) + ','
                        observ_noise += str(TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)) + ','
                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_X)[1]) + ','
                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_Y)[1]) + ','
                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_Z)[1]) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_X)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_Y)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_Z)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_X)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_Y)) + ','
                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_Z)) + ','
                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_X, 1)) + ','
                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_Y, 1)) + ','
                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_Z, 1)) + ','

                observ_signal += str(1)
                the_file.write(waveform.waveform_filename + ',' + observ_signal+'\n')
                observ_noise += str(0)
                the_file.write(waveform.waveform_filename + ',' + observ_noise+'\n')
                observ_signal = ''
                observ_noise = ''
                        
                print('Waveform ' + str(index) + '/' + str(total) + ' done - core ' + str(core))
                index += 1
                gc.collect()
            print('Core ' + str(core) + ' DONE')
        
    # Concat feature extraction files into one
    def concat_attrs_files(self, path):
        list_of_files = os.listdir(path)
        pattern = "att_p*"  
        with open('attributes_matrix_prot03_3stats.txt','a') as final_file:
            final_file.write('Filename; DOP; RV2T; EntropyZ; EntropyN; EntropyE; KurtosisZ; KurtosisN; KurtosisE; SkewZ; SkewN; SkewE; CDZ; CDN; CDE')
            for filename in list_of_files:  
                if fnmatch.fnmatch(filename, pattern):
                    input_file = open(filename, 'r')
                    for line in input_file:
                        final_file.write(line)
                    os.remove(filename)



class p_wave_extractor:
    
    def __init__(self,stations,window,percent):
#        self.stations = ['RUS','BRR','PAM','PTB']
#        self.stations = ['RUS','BRR','PAM']
        self.stations = stations
        self.window = window
        self.percent = percent
        self.read_files()
    
    def read_files(self):

        ''' DATASET READING AND PRE-PROCESSING '''
        
#        file_var_name_import = '/home/administrador/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' #TODO: Lab K
        file_var_name_import = '/home/i201-20/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2_' + str(len(self.stations)) + 'stat.pckl' # I201
#        file_var_name_import = '/home/tellurico/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' # CCA
        
        ''' WAVEFORMS READING '''
        
        f = open(file_var_name_import, 'rb')
        waveforms = pickle.load(f)
        f.close()

        ''' WAVEFORM IMPORT '''
        
        divisions = 1
        current_division = 1 ## Modify by block
        step = int(len(waveforms)/divisions)
        
        waveforms = waveforms[(current_division-1)*step:(((current_division!=divisions)*(current_division*step))+((current_division==divisions)*(len(waveforms)-1)))]
        cores = os.cpu_count() - 1 # CPU cores
        step = int(len(waveforms)/cores)
        p = [None]*cores
        
        for i in range(1, (cores+1)):
            p[i-1] = Process(target=self.p_wave_noise_extraction, args=(waveforms[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms)-1)))],self.stations,self.window,self.percent,i))
            
        for i in range(0, cores):
            p[i].start()
        
        for i in range(0, cores):
            p[i].join()
        
        self.p_wave_extractor(['RUS','BRR','PAM','PTB'],200,0.9)
    
    # Feature extraction
    def p_wave_noise_extraction(self, waveforms, stations, window, percent, core):
        waveforms_path_HD2 = '/media/i201-20/Tellurico_Dataset2/' # External HD2 - Local
        filename_variables_export = '/home/i201-20/Tellurico/Variables/HD2_Files/P_Waves_Noise/' + str(len(stations)) + 'stat/' + str(percent) + '/'
        total = len(waveforms)
        
        total_waveforms = []
        waveforms_extract = {} # Stations key dictionary
        wave_type = {'Px':[],'Py':[],'Pz':[],'Nx':[],'Ny':[],'Nz':[]} # Type of wave key dictionary
#        
        for stat in stations:
            waveforms_extract[stat] = copy.copy(wave_type)
            
        waveforms_extract_copy = copy.copy(waveforms_extract)
        
        index_total = 0
        index_part = 1
        for waveform in waveforms:
            try:
                waveform.waveform_path = waveforms_path_HD2 ## TODO: Only local
                waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
                st = read(waveform_path_and_name)
                waveform.set_stations(self.stations)
                waveform.set_st(st)
                [newEvent, stats_sort] = waveform.get_event_spec_stats()
                for stat in stations:
                    if(stat in newEvent.trace_groups):
                        [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
                        if(dataX != None and dataY != None and dataZ != None):
                            [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, window, newEvent.trace_groups[stat].P_Wave, percent)
                            [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, window, newEvent.trace_groups[stat].P_Wave, percent)
                            [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, window, newEvent.trace_groups[stat].P_Wave, percent)
                            waveforms_extract_copy[stat]['Px'] = p_signal_X
                            waveforms_extract_copy[stat]['Py'] = p_signal_Y
                            waveforms_extract_copy[stat]['Pz'] = p_signal_Z
                            waveforms_extract_copy[stat]['Nx'] = noise_X
                            waveforms_extract_copy[stat]['Ny'] = noise_Y
                            waveforms_extract_copy[stat]['Nz'] = noise_Z
                
                total_waveforms.append(waveforms_extract_copy)
                waveforms_extract_copy = copy.copy(waveforms_extract)
                index_total += 1
                print('Waveform ' + str(index_total) + '/' + str(total) + ' included - ' + str(core))
                waveform.set_st(None)
                
                if(index_total%10 == 0):
                    file_var_name =  filename_variables_export + 'Total_P-wave-noise_C' + str(core) + '_P' + str(index_part) + '_HD2.pckl' ## TODO: variable name to be exported CCA
                    toSave = total_waveforms
                    f = open(file_var_name, 'wb')
                    pickle.dump(toSave, f)
                    f.close()
                    index_part += 1
                    del(total_waveforms)
                    total_waveforms = []
                    gc.collect()
            except:
                pass
            gc.collect()
        
        file_var_name =  filename_variables_export + 'Total_P-wave-noise_C' + str(core) + '_P' + str(index_part) + '_HD2.pckl' ## TODO: variable name to be exported CCA
        toSave = total_waveforms
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()
                    
class waveforms_read:
    
    def __init__(self):
        self.read_files()
    
    def read_files(self):

        ''' WAVEFORMS READING '''

        #file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
#        file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms_HD2.pckl' ## variable name to be exported CCA
        file_var_name =  '/home/i201-20/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' ## variable name to be exported I201
        
        f = open(file_var_name, 'rb')
        waveforms = pickle.load(f)
        f.close()
#        total = 0
#        
#        path_names = []
        
#        for waveform in waveforms:
#            waveform.waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
#            waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
#            if not Path(waveform_path_and_name).exists():
#                waveforms.pop(waveforms.index(waveform))
#                total += 1
#            elif waveform_path_and_name not in path_names:
#                path_names.append(waveform_path_and_name)
#        
#        print('Total files in HD2: ' + str(len(waveforms)))
#        print('Total files rejected in HD2: ' + str(total))
#        print('Non repeated Total files in HD2: ' + str(len(path_names)))
        
#        ''' WAVEFORM IMPORT '''
#        
        divisions = 2
        current_division = 1 ## Modify by block
        step = int(len(waveforms)/divisions)
        
        waveforms = waveforms[(current_division-1)*step:(((current_division!=divisions)*(current_division*step))+((current_division==divisions)*(len(waveforms)-1)))]
        cores = os.cpu_count() - 1 # CPU cores
        step = int(len(waveforms)/cores)
        p = [None]*cores
        
        for i in range(1, (cores+1)):
            p[i-1] = Process(target=self.waveform_import, args=(waveforms[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms)-1)))],i,current_division))
            
        for i in range(0, cores):
            p[i].start()
        
        for i in range(0, cores):
            p[i].join()
    
    # 
    def waveform_import(self,waveforms,core,div):     
        
        total = len(waveforms)
        index_wrong = 0
        index_total = 0
        waveforms_valid = {}
        weights = []
        
        for waveform in waveforms:
            try:
                waveform.waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
                waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
                st = read(waveform_path_and_name)
                waveform.set_st(st)
                file_weigth = os.stat(waveform_path_and_name).st_size
                waveforms_valid[waveform] = file_weigth
                weights.append(file_weigth)
                gc.collect()
            except:
                index_wrong += 1
            index_total += 1
            print('Waveform ' + str(index_total) + '/' + str(total) + ' included - ' + str(core))
        
        waveforms_valid = TelluricoTools.sort(waveforms_valid)
        waveforms_valid.reverse()
        
        file_var_name =  '/home/tellurico-admin/Variables/Total_ProcWaveforms_C' + str(core) + '_D' + str(div) + '_HD2.pckl' ## variable name to be exported CCA
        toSave = [waveforms_valid, weights]
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()
        
class waveforms_read_v2:
    
    def __init__(self):
        self.read_files()
    
    def read_files(self):

        ''' WAVEFORMS READING '''

        #file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
        file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms_HD2.pckl' ## variable name to be exported CCA
        
        f = open(file_var_name, 'rb')
        waveforms = pickle.load(f)
        f.close()

        ''' WAVEFORM IMPORT '''
        
        divisions = 2
        current_division = 1 ## Modify by block
        step = int(len(waveforms)/divisions)
        
        waveforms = waveforms[(current_division-1)*step:(((current_division!=divisions)*(current_division*step))+((current_division==divisions)*(len(waveforms)-1)))]
        cores = os.cpu_count() - 1 # CPU cores
        step = int(len(waveforms)/cores)
        p = [None]*cores
        
        for i in range(1, (cores+1)):
            p[i-1] = Process(target=self.waveform_import, args=(waveforms[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms)-1)))],i,current_division))
            
        for i in range(0, cores):
            p[i].start()
        
        for i in range(0, cores):
            p[i].join()
    
    # 
    def waveform_import(self,waveforms,core,div):     
        
        total = len(waveforms)
        index_wrong = 0
        index_total = 0
        waveforms_valid = {}
        
        for waveform in waveforms:
            try:
                waveform.waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
                waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
                st = read(waveform_path_and_name)
                waveforms_valid[waveform.waveform_filename] = st
                gc.collect()
            except:
                index_wrong += 1
            index_total += 1
            print('Waveform ' + str(index_total) + '/' + str(total) + ' included - ' + str(core))
        
#        waveforms_valid = TelluricoTools.sort(waveforms_valid)
#        waveforms_valid.reverse()
        
        file_var_name =  '/home/tellurico-admin/Variables/Total_ProcWaveforms_C' + str(core) + '_D' + str(div) + '_HD2.pckl' ## variable name to be exported CCA
        toSave = waveforms_valid
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()

class waveforms_read_v3:
    
    def __init__(self):
        self.read_files()
    
    def read_files(self):
        
#        file_var_name_import = '/home/administrador/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' #TODO: Lab K
#        file_var_name_import = '/home/i201-20/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' # I201
        file_var_name_import = '/home/tellurico/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' # CCA
        
        ''' WAVEFORMS READING '''
        
        f = open(file_var_name_import, 'rb')
        waveforms = pickle.load(f)
        f.close()

        ''' WAVEFORM IMPORT '''
        
        divisions = 1
        current_division = 1 ## Modify by block
        step = int(len(waveforms)/divisions)
        
        waveforms = waveforms[(current_division-1)*step:(((current_division!=divisions)*(current_division*step))+((current_division==divisions)*(len(waveforms)-1)))]
        cores = os.cpu_count() - 1 # CPU cores
        step = int(len(waveforms)/cores)
        p = [None]*cores
        
        for i in range(1, (cores+1)):
            p[i-1] = Process(target=self.waveform_import, args=(waveforms[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms)-1)))],i,current_division))
            
        for i in range(0, cores):
            p[i].start()
        
        for i in range(0, cores):
            p[i].join()
    
    # 
    def waveform_import(self,waveforms,core,div):     
        
#        waveform_path_name = '/media/administrador/Tellurico_Dataset2/' # TODO: Lab K
        waveform_path_name = '/media/i201-20/Tellurico_Dataset2/' # I201
#        filename_variables_export = '/home/administrador/Tellurico/Variables/HD2_Files/' # Lab K
        filename_variables_export = '/home/i201-20/Tellurico/Variables/HD2_Files/' # I201
        
        total = len(waveforms)
        index_wrong = 0
        index_total = 0
        waveforms_valid = {}
        
        for waveform in waveforms:
            try:
#                waveform.waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
                waveform.waveform_path = waveform_path_name ## TODO: Only local
                waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
                st = read(waveform_path_and_name)
                stats = []
                for trace in st:
                    stats.append(trace.stats)
                waveforms_valid[waveform.waveform_filename] = stats
                gc.collect()
            except:
                index_wrong += 1
            index_total += 1
            print('Waveform ' + str(index_total) + '/' + str(total) + ' included - ' + str(core))
            
            if(index_total%20 == 0):
                file_var_name =  filename_variables_export + 'Total_ProcWaveforms_C' + str(core) + '_D' + str(div) + '_HD2.pckl' ## TODO: variable name to be exported CCA
                toSave = waveforms_valid
                f = open(file_var_name, 'wb')
                pickle.dump(toSave, f)
                f.close()
        
#        waveforms_valid = TelluricoTools.sort(waveforms_valid)
#        waveforms_valid.reverse()
        
        file_var_name =  filename_variables_export + 'Total_ProcWaveforms_C' + str(core) + '_D' + str(div) + '_HD2.pckl' ## TODO: variable name to be exported CCA
        toSave = waveforms_valid
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()
    
    # Concat all exported variables
    def concat_variables(self, path):
        list_of_files = os.listdir(path)
        pattern = "Total_ProcWaveforms*"
        final_dict = {}
        for filename in list_of_files:  
            if fnmatch.fnmatch(filename, pattern):
                f = open(path + filename, 'rb')
                waveforms_dict = pickle.load(f)
                for waveform_key in waveforms_dict:
                    final_dict[waveform_key] = waveforms_dict[waveform_key]
                f.close()
                gc.collect()
        
        f = open(path + 'Total_ProcWaveforms_HD2.pckl', 'wb')
        pickle.dump(final_dict, f)
        f.close()





class waveforms_read_v3_esp:
    
    def __init__(self):
        self.read_files()
    
    def read_files(self):
        
#        file_var_name_import = '/home/administrador/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' #TODO: Lab K
        file_var_name_import = '/home/i201-20/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' # I201
        
        ''' WAVEFORMS READING '''

        #file_var_name =  '/home/administrador/Tellurico/Variables/Total_InitialWaveforms.pckl' ## TODO: variable name to be exported
#        file_var_name =  '/home/tellurico-admin/Variables/Total_InitialWaveforms_HD2.pckl' ## variable name to be exported CCA
#        file_var_name =  '/home/administrador/Tellurico/Variables/CCA/Total_InitialWaveforms_HD2.pckl' ## variable name to be exported Locally
        
        f = open(file_var_name_import, 'rb')
        waveforms = pickle.load(f)
        f.close()

        ''' WAVEFORM IMPORT '''
        
        divisions = 1
        current_division = 1 ## Modify by block
        step = int(len(waveforms)/divisions)
        
        waveforms = waveforms[(current_division-1)*step:(((current_division!=divisions)*(current_division*step))+((current_division==divisions)*(len(waveforms)-1)))]
        cores = os.cpu_count() - 1 # CPU cores
        step = int(len(waveforms)/cores)
        p = [None]*cores
        
        i = 3
        waveforms_spet = []
        for waveform in waveforms[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms)-1)))]:
            waveforms_spet.append(waveform)
        
        i = 6
        for waveform in waveforms[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms)-1)))]:
            waveforms_spet.append(waveform)
        
        for i in range(1, (cores+1)):
            p[i-1] = Process(target=self.waveform_import, args=(waveforms_spet[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms_spet)-1)))],i,current_division))
            
        for i in range(0, cores):
            p[i].start()
        
        for i in range(0, cores):
            p[i].join()
    
    # 
    def waveform_import(self,waveforms,core,div):     
        
#        waveform_path_name = '/media/administrador/Tellurico_Dataset2/' # TODO: Lab K
        waveform_path_name = '/media/i201-20/Tellurico_Dataset2/' # I201
#        filename_variables_export = '/home/administrador/Tellurico/Variables/HD2_Files/' # Lab K
        filename_variables_export = '/home/i201-20/Tellurico/Variables/HD2_Files/' # I201
        
        total = len(waveforms)
        index_wrong = 0
        index_total = 0
        waveforms_valid = {}
        
        for waveform in waveforms:
            try:
#                waveform.waveform_path = '/run/user/1000/gvfs/sftp:host=10.154.12.15/media/administrador/Tellurico_Dataset2/' ## Only for CCA
                waveform.waveform_path = waveform_path_name ## TODO: Only local
                waveform_path_and_name = waveform.waveform_path + 'download.php?file=' + waveform.waveform_filename[0:4] + '%2F' + waveform.waveform_filename[5:7]+ '%2F' + waveform.waveform_filename
                st = read(waveform_path_and_name)
                stats = []
                for trace in st:
                    stats.append(trace.stats)
                waveforms_valid[waveform.waveform_filename] = stats
                gc.collect()
            except:
                index_wrong += 1
            index_total += 1
            print('Waveform ' + str(index_total) + '/' + str(total) + ' included - ' + str(core))
            
            if(index_total%20 == 0):
                file_var_name =  filename_variables_export + 'Total_ProcWaveforms_C' + str(core) + '_DS' + str(div) + '_HD2.pckl' ## TODO: variable name to be exported CCA
                toSave = waveforms_valid
                f = open(file_var_name, 'wb')
                pickle.dump(toSave, f)
                f.close()
        
#        waveforms_valid = TelluricoTools.sort(waveforms_valid)
#        waveforms_valid.reverse()
        
        file_var_name =  filename_variables_export + 'Total_ProcWaveforms_C' + str(core) + '_DS' + str(div) + '_HD2.pckl' ## TODO: variable name to be exported CCA
        toSave = waveforms_valid
        f = open(file_var_name, 'wb')
        pickle.dump(toSave, f)
        f.close()
        
                    
                    
#class prototype_v4:
#    
#    def __init__(self):
##        self.stations = ['RUS','BRR','PAM','PTB']
#        self.stations = ['RUS','BRR','PAM', 'ALL']
#        file_var_name = self.read_files()
#        self.cores_distr(file_var_name)
#    
#    def read_files(self):
#
#        ''' DATASET READING AND PRE-PROCESSING '''
#        
##        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
##        sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
#        waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
#        sfiles_path = '/home/administrador/Tellurico/Filtered_RSNC_Files/Sfiles/' #Local
#        sfileAnalyzer = SfileAnalyzer(sfiles_path)
#        sfileAnalyzer.get_sfiles()
#        sfiles = sfileAnalyzer.sfiles
#        waveforms = []
#        stations_prov = copy.copy(self.stations)
#        
#        if 'ALL' in stations_prov: # If all stations are analyzed
#            for sfile in sfiles:
#                waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
#        else: # If certain stations are analyzed
#            for sfile in sfiles:
#                if(hasattr(sfile, 'type_6')):
#                    for station in sfile.type_7:
#                        if station['STAT'] in stations_prov and station['PHAS'].strip() == 'P':
#                            stations_prov.pop(stations_prov.index(station['STAT']))
#                        if len(stations_prov) == 0:
#                            waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
#                            break
#                stations_prov = copy.copy(self.stations)
#        
#        index = 0
#        waveforms_name_st = {}
#        waveforms_name_w = {}
#        weights = []
#        
#        # Validate waveforms
#        for waveform in waveforms:
#            try:
#                st = read(waveform.waveform_path + waveform.waveform_filename)
#                if 'ALL' in stations_prov: # If all stations are analyzed
#                    waveform.set_st(st)
#                    file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
#                    waveforms_name_st[waveform.waveform_filename] = waveform
#                    waveforms_name_w[waveform.waveform_filename] = file_weigth
#                    weights.append(file_weigth)
#                else:
#                    waveform.set_st(st)
#                    for trace in st:
#                        if(trace.stats.station in stations_prov):
#                            stations_prov.pop(stations_prov.index(trace.stats.station))
#                        if len(stations_prov) == 0:
#                            file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
#                            waveforms_name_st[waveform.waveform_filename] = waveform
#                            waveforms_name_w[waveform.waveform_filename] = file_weigth
#                            weights.append(file_weigth)
#                            break
#                    stations_prov = copy.copy(self.stations)
#            except:
#                index += 1
#        waveforms_name_w = TelluricoTools.sort(waveforms_name_w)
#        waveforms_name_w.reverse()
#        print(str(len(waveforms_name_w)) + ' valid files with stations')
#        
#        file_var_name =  'waveforms_valid_prot03.pckl' ## TODO: variable name to be exported
#        toSave = [waveforms_name_w, waveforms_name_st, weights]
#        f = open(file_var_name, 'wb')
#        pickle.dump(toSave, f)
#        f.close()
#        
#        return file_var_name
#        
#    def cores_distr(self, file_var_name):
#        
#        f = open(file_var_name, 'rb')
#        toRead = pickle.load(f)
#        f.close()
#        
#        waveforms_name_w = toRead[0]
#        waveforms_name_st = toRead[1]
#        weights = toRead[2]
#        
#        ''' DATASET DISTRIBUTION IN CORES '''
#        
#        cores_quant = os.cpu_count() - 1 # CPU cores
#        cores = []
#        for i in range(0, cores_quant): cores.append([])
#        
#        w_flag = False
#        max_weigth = 1.15*(np.sum(weights)/cores_quant)
#        if(max_weigth < max(weights)): 
#            max_weigth = max(weights)
#            w_flag = True
#        
#        space = max_weigth
#        for i in range(0, len(cores)):    
#            for waveform in waveforms_name_w:
#                if(waveform[1] <= space):
#                    cores[i].append(waveforms_name_st[waveform])
#                    space -= waveform[1]
#                    waveforms_name_w.pop(waveforms_name_w.index(waveform))
#                    weights.pop(weights.index(waveforms_name_w[waveform]))
#                    if(w_flag):
#                        max_weigth = 1.15*(np.sum(weights)/(cores_quant-i))
#                        if(max_weigth < max(weights)): 
#                            max_weigth = max(weights)
#                            w_flag = True
#                        else:
#                            w_flag = False
#            space = max_weigth
#        
#        ''' DATASET FEATURES EXTRACTION '''
#        
#        p = []
#        for i in range(0, cores_quant): p.append(None)
#        
#        for i in range(0, cores_quant):
#            p[i] = Process(target=self.attributes, args=(('att_p' + str(i+1) + '.txt'), 
#             cores[i], self.stations, len(cores[i]), (i+1)))
#            
#        for i in range(0, cores_quant):
#            p[i].start()
#        
#        for i in range(0, cores_quant):
#            p[i].join()
#        
##        self.concat_attrs_files('/home/administrador/Tellurico/TelluricoDetect/code/')
#        self.concat_attrs_files('/home/tellurico-admin/TelluricoDetect/code/') #CCA
#    
#    # Feature extraction
#    def attributes(self,filename, waveforms_valid, stations, total, core):     
#        index = 1
#        observ_signal = ''
#        observ_noise = ''
#        with open(filename, 'a') as the_file:
#            for waveform in waveforms_valid:
#                waveform.set_st
#                [newEvent, stats_sort] = waveform.get_event()
#                for stat in stations:
#                    if(stat in newEvent.trace_groups):
#                        [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
#                        [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
#                        [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
#                        [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.9)
#                        
#                        observ_signal += str(TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)) + ','
#                        observ_signal += str(TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)) + ','
#                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_X)[1]) + ','
#                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_Y)[1]) + ','
#                        observ_signal += str(NonLinear_Attributes.signal_entropy(p_signal_Z)[1]) + ','
#                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_X)) + ','
#                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_Y)) + ','
#                        observ_signal += str(TimeDomain_Attributes.signal_kurtosis(p_signal_Z)) + ','
#                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_X)) + ','
#                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_Y)) + ','
#                        observ_signal += str(TimeDomain_Attributes.signal_skew(p_signal_Z)) + ','
#                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_X, 1)) + ','
#                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_Y, 1)) + ','
#                        observ_signal += str(NonLinear_Attributes.corr_CD(p_signal_Z, 1)) + ','
#                        
#                        observ_noise += str(TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)) + ','
#                        observ_noise += str(TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)) + ','
#                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_X)[1]) + ','
#                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_Y)[1]) + ','
#                        observ_noise += str(NonLinear_Attributes.signal_entropy(noise_Z)[1]) + ','
#                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_X)) + ','
#                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_Y)) + ','
#                        observ_noise += str(TimeDomain_Attributes.signal_kurtosis(noise_Z)) + ','
#                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_X)) + ','
#                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_Y)) + ','
#                        observ_noise += str(TimeDomain_Attributes.signal_skew(noise_Z)) + ','
#                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_X, 1)) + ','
#                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_Y, 1)) + ','
#                        observ_noise += str(NonLinear_Attributes.corr_CD(noise_Z, 1)) + ','
#
#                observ_signal += str(1)
#                the_file.write(waveform.waveform_filename + ',' + observ_signal+'\n')
#                observ_noise += str(0)
#                the_file.write(waveform.waveform_filename + ',' + observ_noise+'\n')
#                observ_signal = ''
#                observ_noise = ''
#                        
#                print('Waveform ' + str(index) + '/' + str(total) + ' done - core ' + str(core))
#                index += 1
#                gc.collect()
#            print('Core ' + str(core) + ' DONE')
#        
#    # Concat feature extraction files into one
#    def concat_attrs_files(self, path):
#        list_of_files = os.listdir(path)
#        pattern = "att_p*"  
#        with open('attributes_matrix_prot03_3stats.txt','a') as final_file:
#            final_file.write('Filename; DOP; RV2T; EntropyZ; EntropyN; EntropyE; KurtosisZ; KurtosisN; KurtosisE; SkewZ; SkewN; SkewE; CDZ; CDN; CDE')
#            for filename in list_of_files:  
#                if fnmatch.fnmatch(filename, pattern):
#                    input_file = open(filename, 'r')
#                    for line in input_file:
#                        final_file.write(line)
#                    os.remove(filename)
                            
                    




## FIXME Prueba de distribucion de cores
# 1. Corregir cuando se aproxima por debajo de 0.5 y cuando se aproxima por encima de 0.5
# 2. Algoritmo de distribucion de los cores por el peso del archivo
                        
#''' DATASET READING AND PRE-PROCESSING '''
#
#waveforms_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Waveforms/' #CCA
#sfiles_path = '/home/tellurico-admin/Tellurico_Archivos/Archivos_Prueba/PrototipoV0_1/Sfiles/' #CCA
#sfileAnalyzer = SfileAnalyzer(sfiles_path)
#sfileAnalyzer.get_sfiles()
#sfiles = sfileAnalyzer.sfiles
#waveforms = []
#for sfile in sfiles:
#    waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
#
#index = 0
#waveforms_valid = {}
#weights = []
#for waveform in waveforms: #[32:64]
#    try:
#        read(waveform.waveform_path + waveform.waveform_filename)
#        file_weigth = os.stat(waveform.waveform_path + waveform.waveform_filename).st_size
#        waveforms_valid[waveform] = file_weigth
#        weights.append(file_weigth)
#    except:
#        index += 1
#waveforms_valid = TelluricoTools.sort(waveforms_valid)
#waveforms_valid.reverse()
#
#''' DATASET DISTRIBUTION TO CORES '''
#
#cores_quant = os.cpu_count() - 1 # CPU cores
#cores = []
#for i in range(0, cores_quant): cores.append([])
#
#w_flag = False
#max_weigth = 1.15*(np.sum(weights)/cores_quant)
#if(max_weigth < max(weights)): 
#    max_weigth = max(weights)
#    w_flag = True
#
#space = max_weigth
#for i in range(0, len(cores)):    
#    for waveform in waveforms_valid:
#        if(waveform[1] <= space):
##            cores[i].append(waveform[1])
#            cores[i].append(waveform[0])
#            space -= waveform[1]
#            waveforms_valid.pop(waveforms_valid.index(waveform))
#            weights.pop(weights.index(waveform[1]))
#            if(w_flag):
#                max_weigth = 1.15*(np.sum(weights)/(cores_quant-i))
#                if(max_weigth < max(weights)): 
#                    max_weigth = max(weights)
#                    w_flag = True
#                else:
#                    w_flag = False
##            print(max_weigth)
#    space = max_weigth
#
#total = 0
#for core in cores:
##    print(np.sum(core))
#    total += len(core)
#print(total)
#
#''' DATASET ATRIBUTES '''
#
#step = int(len(waveforms_valid)/cores)
#stat = 'BRR'
#p = [None]*cores
#
#for i in range(1, (cores+1)):
#    p[i-1] = Process(target=self.attributes, args=(('att_p' + str(i) + '.txt'),
#         waveforms_valid[(i-1)*step:(((i!=cores)*(i*step))+((i==cores)*(len(waveforms_valid)-1)))],
#         stat,(i-1)*step,i))
########################################################################################
#
#
#
#
### Paradoja del cumpleaños para probabilidad de encontrar dos archivos con igual peso
#print ('Num. probabilidad')
#p = 1.0
#i=0
#for i in range(1, 500):
#    index = i*50
#    p = p * (100000001 - index) / 100000000
#    print ('%3d : %10.6f' % (index, 1-p))
                        
                        
                        


#from sys import getsizeof
    
    
    

#p[cores-1] = Process(target=self.attributes, args=(('att_p' + str(cores) + '.txt'),waveforms_valid[(cores-1)*step:len(waveforms_valid)-1],stat,(cores-1)*step,cores))

#def attributes(self,filename, waveforms_valid, stat, begin, core):     
#        index = begin
#        observ_signal = {}
#        observ_noise = {}
#        with open(filename, 'a') as the_file:
#            for waveform in waveforms_valid:
#                [newEvent, stats_sort] = waveform.get_event()
#                if(stat in newEvent.trace_groups):
#                    [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
#                    [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave)
#                    [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave)
#                    [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave)
#                    
#                    DOP_signal = TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)
#                    RV2T_signal = TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)
#                    DOP_noise = TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)
#                    RV2T_noise = TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)
#                    Kurtosis_signal_X = TimeDomain_Attributes.signal_kurtosis(p_signal_X)
#                    Kurtosis_signal_Y = TimeDomain_Attributes.signal_kurtosis(p_signal_Y)
#                    Kurtosis_signal_Z = TimeDomain_Attributes.signal_kurtosis(p_signal_Z)
#                    Kurtosis_noise_X = TimeDomain_Attributes.signal_kurtosis(noise_X)
#                    Kurtosis_noise_Y = TimeDomain_Attributes.signal_kurtosis(noise_Y)
#                    Kurtosis_noise_Z = TimeDomain_Attributes.signal_kurtosis(noise_Z)
#                
#                    the_file.write(str(DOP_signal)+','+str(RV2T_signal)+','+str(Kurtosis_signal_X)+','+
#                                   str(Kurtosis_signal_Y)+','+str(Kurtosis_signal_Z)+','+str(1)+'\n')
#                    the_file.write(str(DOP_noise)+','+str(RV2T_noise)+','+str(Kurtosis_noise_X)+','+
#                                   str(Kurtosis_noise_Y)+','+str(Kurtosis_noise_Z)+','+str(0)+'\n')
#                print('Waveform ' + str(index) + ' done - core ' + str(core))
#                index += 1
#                gc.collect()