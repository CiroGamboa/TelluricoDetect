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
from tools import SeismicInfo 
from tools import Attributes
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml
from TraceComponent import TraceComponent
from TraceGroup import TraceGroup
from Event import Event
from Sfile import Sfile
from Waveform import Waveform
import obspy.core.utcdatetime as dt
import obspy.signal.filter as filt
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger
from obspy.imaging import spectrogram as spec
import time

''' DATASET READING AND PRE-PROCESSING '''

events = []

sfile = Sfile('18-0602-02L.S201306.txt', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
[newEvent, stats_sort] = Waveform('IOfiles/', '2013_06_2013-06-18-0559-59M.COL___261', sfile).get_event()
events.append(newEvent)
#sfile = Sfile('10-2055-44L.S201503', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2015-03-10-2049-48M.COL___284', sfile).get_event()
#events.append(newEvent)

''' DATASET ATRIBUTES '''

# Plot FFT for all stations of an specific event
for station in events[0].trace_groups:
        TelluricoTools.FFT(events[0].trace_groups[station].traces[0].waveform, 
                           events[0].trace_groups[station].traces[0].sampling_rate,
                           events[0].trace_groups[station].traces[0].station)

# Plot FFT for all P-wave traces of an specific Event
for stat in events[0].trace_groups:
        sub = 100
        init = events[0].trace_groups[stat].P_Wave
        inf_limit = init - sub
        sup_limit = init + sub
        TelluricoTools.FFT(TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), 
                           events[0].trace_groups[station].traces[0].sampling_rate,
                           events[0].trace_groups[station].traces[0].station)

# Plot FFT for all noise traces of an specific Event
for stat in events[0].trace_groups:
        sub = 50
        init = events[0].trace_groups[stat].P_Wave
        inf_limit = 0
        sup_limit = init - sub - 1
        TelluricoTools.FFT(TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), 
                           events[0].trace_groups[stat].traces[0].sampling_rate,
                           events[0].trace_groups[stat].traces[0].station)
        
# Plot FFT for all P-wave to S-wave traces of an specific Event
for stat in events[0].trace_groups:
    if(events[0].trace_groups[stat].S_Wave > 0):
        inf_limit = events[0].trace_groups[stat].P_Wave
        sup_limit = events[0].trace_groups[stat].S_Wave
        TelluricoTools.FFT(TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), 
                           events[0].trace_groups[stat].traces[0].sampling_rate,
                           events[0].trace_groups[stat].traces[0].station)

# Print components and sampling rates of stations
index = 0
for group in events[0].trace_groups:
    comp = len(events[0].trace_groups[group].traces)
    print(group + "\n")
    index += 1
    for i in range(0, comp):
        print("\t" + events[0].trace_groups[group].traces[i].channel + " - " +
              str(events[0].trace_groups[group].traces[i].original_sampling_rate) + " - " +
              str(events[0].trace_groups[group].traces[i].sampling_rate) + " - " +
              str(len(events[0].trace_groups[group].traces[i].filter_wave)))
print(index)

# Print accelerometers
#index = 0
#for group in events[0].trace_groups:
#    comp = len(events[0].trace_groups[group].traces)
#    for i in range(0, comp):
#        if(events[0].trace_groups[group].traces[i].channel[1] == 'N'):
#            print(group)
#            index += 1
#            break
#print(index)

# %matplotlib qt  
#ml.show()
# It is necessary to make a table describing the data convention
# PREGUNTA PARA EDWAR, CUALES SERIAN LOS MEJORES PARAMETROS PARA CALCULAR
# EL DOP DE LA SEÑAL.... VER REFERENCIA DE OBSPY: https://docs.obspy.org/packages/autogen/obspy.signal.polarization.eigval.html#obspy.signal.polarization.eigval

DOP = Attributes.DOP(events[0].trace_groups['YOP'].traces[0].waveform,
               events[0].trace_groups['YOP'].traces[1].waveform,
               events[0].trace_groups['YOP'].traces[2].waveform)

# Cumulative signal for all stations
for stat in events[0].trace_groups:
    signal = events[0].trace_groups[stat].traces[0].filter_wave
    cumulative_signal = events[0].trace_groups[stat].traces[0].cumulative()
    fig, ax = ml.subplots(2, 1)
    ax[0].plot(cumulative_signal)
    ax[1].plot(signal)
    



# DOP Calculation and Graph
stat = 'BRR'; 
SR = events[0].trace_groups[stat].traces[0].sampling_rate
sub = int(SR/2); # 1 second time window
#sub = int(SR/3)
init = int((2*60*SR)+(25.12*SR));
inf_limit = init - sub
sup_limit = init + sub
#inf_limit = 0
#sup_limit = 1000

dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].waveform,inf_limit,sup_limit)
dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].waveform,inf_limit,sup_limit)
dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].waveform,inf_limit,sup_limit)
#print(Attributes.DOP(dataX,dataY,dataZ))

Temps.graphComponents(events[0].trace_groups[stat].traces, inf_limit, sup_limit)
ml.figure(), ml.plot(TelluricoTools.FFT(dataX, events[0].trace_groups[stat].traces[0].sampling_rate, events[0].trace_groups[stat].traces[0].station))
noise = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].waveform,0,inf_limit)
ml.figure(), ml.plot(TelluricoTools.FFT(noise, events[0].trace_groups[stat].traces[0].sampling_rate, 'Noise'))
    


# DOP Calculation For all stations
for stat in events[0].trace_groups:
        SR = events[0].trace_groups[stat].traces[0].sampling_rate
        samples = range(10, 1001)
        DOPs = []
        for i in range(10, 1001):
            sub = i
            init = events[0].trace_groups[stat].P_Wave
            inf_limit = init - sub
            sup_limit = init + sub
            dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)
            dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].filter_wave,inf_limit,sup_limit)
            dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].filter_wave,inf_limit,sup_limit)
            DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
        ml.plot(samples,DOPs)
    
# DOP Calculation For one station
#stat = 'BRR'; 
#SR = events[0].trace_groups[stat].traces[0].sampling_rate
#samples = range(10, 5001)
#DOPs = []
#for i in range(10, 5001):
#    sub = i
#    init = int((2*60*SR)+(25.12*SR)) # BRR
##    init = int((2*60*SR)+(41.16*SR)) # CHI
##    init = int((3*60*SR)+(32.26*SR)) #BBAC
##    init = int((2*60*SR)+(53.30*SR)) #ARGC
##    init = int((2*60*SR)+(50.50*SR)) #COD
#    inf_limit = init - sub
#    sup_limit = init + sub
#    dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)
#    dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].filter_wave,inf_limit,sup_limit)
#    dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].filter_wave,inf_limit,sup_limit)
#    DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
#ml.plot(samples,DOPs)



# STA/LTA individual
stat = 'BRR'
df = events[0].trace_groups[stat].traces[0].sampling_rate
init = events[0].trace_groups[stat].P_Wave
sub = 1000
inf_limit = init - sub
sup_limit = init + sub
inf_limit = 0
sup_limit = events[0].trace_groups[stat].traces[0].npts
trace = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)
npts = len(events[0].trace_groups[stat].traces[0].filter_wave)
cft = classic_sta_lta(trace, int(5 * df), int(10 *df))
on_off = plot_trigger(trace, events[0].trace_groups['BRR'].traces[0].sampling_rate, npts, cft, 1.5, 0.5)

TelluricoTools.FFT(cft, df, stat)

# STA/LTA all stations
for stat in events[0].trace_groups:
    df = events[0].trace_groups[stat].traces[0].sampling_rate
    cft = classic_sta_lta(events[0].trace_groups[stat].traces[0].filter_wave, int(5 * df), int(10 *df))
    #plot_trigger(events[0].trace_groups['BRR'].traces[0], cft, 2.5, 0.5)
    fig, ax = ml.subplots(2, 1)
    ax[0].plot(events[0].trace_groups[stat].traces[0].filter_wave)
    ax[1].plot(cft)



#Attributes per window calculation for one station
slope = 20
stat = 'BRR'; 
window_size = 200 #window_size in samples
size = int(events[0].trace_groups[stat].traces[0].npts/slope)
DOPs = []
RV2Ts = []
DOPs_RV2T = []
for i in range(0, size-window_size):
    [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
    dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
    dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
    dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
    DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
    RV2Ts.append(Attributes.RV2T(dataX,dataY,dataZ))
    DOPs_RV2T.append(DOPs[i]*RV2Ts[i])
#    print(str(slope*i) + " to " + str((slope*i)+window_size-1))
fig = ml.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)
ax1.set_title('DOP per window - Station: ' + stat)
ax1.plot(DOPs)
ax2.set_title('RV2T per window - Station: ' + stat)
ax2.plot(RV2Ts)
ax3.set_title('RV2T*DOP per window - Station: ' + stat)
ax3.plot(DOPs_RV2T)

#Temporal complexity for attributes calculation per window for one station, increasing
# moving window and slope of one sample
stat = 'BRR'; 
DOPs_RV2T = []
window = []
time_list = []
init = events[0].trace_groups[stat].P_Wave
iterations = 50
max_window = 1000
acumul = 0.0
for i in range(5, int(max_window/2)):
    window_size = 2*i
    inf_limit = init - i
    sup_limit = init + i
    for ii in range(0, iterations):
        start = time.time()
        [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
        dataX = TelluricoTools.sub_trace(dataX.filter_wave,inf_limit,sup_limit)
        dataY = TelluricoTools.sub_trace(dataY.filter_wave,inf_limit,sup_limit)
        dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,inf_limit,sup_limit)
        result = Attributes.DOP(dataX,dataY,dataZ)*Attributes.RV2T(dataX,dataY,dataZ)
        end = time.time()
        acumul += (end - start)
    DOPs_RV2T.append(result)
    window.append(window_size)
    time_list.append(acumul/iterations)
ml.plot(window, time_list)
ml.title('Temporal complexity')
ml.xlabel('Window size (samples)')
ml.ylabel('Time (seconds)')
ml.grid(True)

#Attributes per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        DOPs = []
        RV2Ts = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
#        slope = int(window_size/2)
        slope = int(window_size/5)
        size = int(events[0].trace_groups[stat].traces[0].npts/slope)
        for i in range(0, size-window_size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
            RV2Ts.append(Attributes.RV2T(dataX,dataY,dataZ))
            DOPs_RV2T.append(DOPs[i]*RV2Ts[i])
#            print(str(slope*i) + " " + str((slope*i)+window_size-1))
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('DOP per window - Station: ' + stat)
        ax1.plot(DOPs)
        ax2.set_title('RV2T per window - Station: ' + stat)
        ax2.plot(RV2Ts)
        ax3.set_title('RV2T*DOP per window - Station: ' + stat)
        ax3.plot(DOPs_RV2T)

#Calculation of cross correlation between Attributes per window for all stations 
#in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        DOPs = []
        RV2Ts = []
        cross_corr = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
#        slope = int(window_size/2)
        slope = int(window_size/5)
        size = int(events[0].trace_groups[stat].traces[0].npts/slope)
        for i in range(0, size-window_size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
            RV2Ts.append(Attributes.RV2T(dataX,dataY,dataZ))
#            print(str(slope*i) + " " + str((slope*i)+window_size-1))
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('DOP per window - Station: ' + stat)
        ax1.plot(DOPs)
        ax2.set_title('RV2T per window - Station: ' + stat)
        ax2.plot(RV2Ts)
        print(stat + ' pearsons correlation: ' + str(Attributes.pearson_correlation(DOPs, RV2Ts)[0]))
    
    

# Envelope for one station
Attributes.envelope(events[0].trace_groups[stat].traces[0].filter_wave, 
                    events[0].trace_groups[stat].traces[0].sampling_rate)



# Entropy for one station, one trace
ml.plot(TelluricoTools.toIntArray(events[0].trace_groups['BRR'].traces[0].filter_wave))
ml.plot(events[0].trace_groups['BRR'].traces[0].filter_wave)
print(Attributes.entropy(TelluricoTools.toIntArray(events[0].trace_groups['BRR'].traces[0].filter_wave)))



# Spectrogram for one signal, one station
stat = 'BRR'
sub = 50
init = events[0].trace_groups[stat].P_Wave
inf_limit = init - sub
sup_limit = init + sub
spec.spectrogram(data = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), samp_rate = events[0].trace_groups[stat].traces[0].sampling_rate)

# Spectrogram for all stations, one component
for stat in events[0].trace_groups:
    sub = 50
    init = events[0].trace_groups[stat].P_Wave
    inf_limit = init - sub
    sup_limit = init + sub
    spec.spectrogram(data = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), samp_rate = events[0].trace_groups[stat].traces[0].sampling_rate)
