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

class prototype_v0:
    
    def __init__(self):
        self.read_files()
    
    def read_files(self):

        ''' DATASET READING AND PRE-PROCESSING '''
        
        waveforms_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Waveforms/'
        sfiles_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Sfiles/'
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
        
        self.concat_attrs_files('/home/administrador/Tellurico/TelluricoDetect/code/')
    
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
        for filename in list_of_files:  
            if fnmatch.fnmatch(filename, pattern):
                input_file = open(filename, 'r')
                with open('attributes_matrix_6.txt','a') as final_file:
                    for line in input_file:
                        final_file.write(line)
                            






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