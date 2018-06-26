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
        
from obspy import read
from tools import TelluricoTools
from tools import SeismicInfo 
from tools import Transformations, TimeDomain_Attributes
#from tools import FreqDomain_Attributes
from tools import NonLinear_Attributes, Correlation, Filters, Graph
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
import os

''' CHANGE FILENAMES '''

#path = '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/PrototipoV0_1/Waveforms/'
#for filename in os.listdir(path):
#    if filename.startswith("download.php"):
#        old_file = os.path.join(path, filename)
#        new_file = os.path.join(path, filename[30:])
#        os.rename(old_file, new_file)

''' DATASET READING AND PRE-PROCESSING '''

events = []

#sfile = Sfile('18-0602-02L.S201306.txt', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2013_06_2013-06-18-0559-59M.COL___261', sfile).get_event()
#events.append(newEvent)
#sfile = Sfile('10-2055-44L.S201503', '/home/tellurico-admin/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2015-03-10-2049-48M.COL___284', sfile).get_event()
#events.append(newEvent)
#sfile = Sfile('04-2028-23L.S201709', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2017-09-04-2028-00M.COL___426', sfile).get_event()
#events.append(newEvent)
#sfile = Sfile('30-1858-02L.S201305', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2013-05-30-1848-50M.COL___218', sfile).get_event()
#events.append(newEvent)
#sfile = Sfile('16-2230-55L.S201104', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2011-04-16-2228-33S.COL___140', sfile).get_event()
#events.append(newEvent)
#sfile = Sfile('30-0429-45L.S201709', '/home/tellurico-admin/TelluricoDetect/code/IOfiles/')
#[newEvent, stats_sort] = Waveform('IOfiles/', '2017-09-30-0428-00M.COL___441', sfile).get_event()
#events.append(newEvent)
sfile = Sfile('18-0602-02L.S201306.txt', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
[newEvent, stats_sort] = Waveform('IOfiles/', '2013_06_2013-06-18-0559-59M.COL___261', sfile).get_event()
events.append(newEvent)

''' DATASET ATRIBUTES '''

# Graph P-waves of all stations and components
window_size = 200
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        Graph.graph_P_waves(events[0].trace_groups[stat], events[0].trace_groups[stat].P_Wave, window_size, stat, 0.9)

# Plot FFT for all stations of an specific event
for station in events[0].trace_groups:
        Transformations.FFT(events[0].trace_groups[station].traces[0].waveform, 
                           events[0].trace_groups[station].traces[0].sampling_rate,
                           events[0].trace_groups[station].traces[0].station)

# Plot FFT for all P-wave traces of an specific Event
for stat in events[0].trace_groups:
        sub = 10000
        init = events[0].trace_groups[stat].P_Wave
        inf_limit = init - sub
        sup_limit = init + sub
        Transformations.FFT(TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), 
                           events[0].trace_groups[station].traces[0].sampling_rate,
                           events[0].trace_groups[station].traces[0].station)

# Plot FFT for all noise traces of an specific Event
for stat in events[0].trace_groups:
        sub = 50
        init = events[0].trace_groups[stat].P_Wave
        inf_limit = 0
        sup_limit = init - sub - 1
        Transformations.FFT(TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].waveform,inf_limit,sup_limit), 
                           events[0].trace_groups[stat].traces[0].sampling_rate,
                           events[0].trace_groups[stat].traces[0].station)
        
# Plot FFT for all P-wave to S-wave traces of an specific Event
for stat in events[0].trace_groups:
    if(events[0].trace_groups[stat].S_Wave > 0):
        inf_limit = events[0].trace_groups[stat].P_Wave
        sup_limit = events[0].trace_groups[stat].S_Wave
        Transformations.FFT(TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit), 
                           events[0].trace_groups[stat].traces[0].sampling_rate,
                           events[0].trace_groups[stat].traces[0].station)

# Plot filteres signal all stations of an specific event
low = 8
high = 10
for station in events[0].trace_groups:
    dataX = events[0].trace_groups[station].traces[0]
    print(dataX)
    dataY = events[0].trace_groups[station].traces[1]
    dataZ = events[0].trace_groups[station].traces[2]
    dataX_F = Filters.bandpass_filter(dataX.waveform, dataX.sampling_rate, low, high)
    dataY_F = Filters.bandpass_filter(dataX.waveform, dataX.sampling_rate, low, high)
    dataZ_F = Filters.bandpass_filter(dataX.waveform, dataX.sampling_rate, low, high)
    Graph.graphComponents([dataX_F, dataY_F, dataZ_F])

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

# Print differences in starttime between components of all stations
for stat in events[0].trace_groups:
    start_0 = events[0].trace_groups[stat].traces[0].starttime
    start_1 = events[0].trace_groups[stat].traces[1].starttime
    start_2 = events[0].trace_groups[stat].traces[2].starttime
    if(start_0 != start_1 or start_0 != start_2 or start_1 != start_2):
       print(stat);
       print(start_0);  
       print(start_1);
       print(start_2); 
       print('\n\n')

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

DOP = TimeDomain_Attributes.DOP(events[0].trace_groups['YOP'].traces[0].waveform,
               events[0].trace_groups['YOP'].traces[1].waveform,
               events[0].trace_groups['YOP'].traces[2].waveform)

# Cumulative signal for all stations
for stat in events[0].trace_groups:
    signal = events[0].trace_groups[stat].traces[0].filter_wave
    cumulative_signal = events[0].trace_groups[stat].traces[0].cumulative()
    fig, ax = ml.subplots(2, 1)
    ax[0].plot(cumulative_signal)
    ax[1].plot(signal)
    


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
            DOPs.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
        ml.plot(samples,DOPs)



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

Transformations.FFT(cft, df, stat)

# STA/LTA all stations
for stat in events[0].trace_groups:
    df = events[0].trace_groups[stat].traces[0].sampling_rate
    cft = classic_sta_lta(events[0].trace_groups[stat].traces[0].filter_wave, int(5 * df), int(10 *df))
    #plot_trigger(events[0].trace_groups['BRR'].traces[0], cft, 2.5, 0.5)
    fig, ax = ml.subplots(2, 1)
    ax[0].plot(events[0].trace_groups[stat].traces[0].filter_wave)
    ax[1].plot(cft)



#Attributes per window calculation for one station and slope of 1 sample
slope = 1
stat = 'SDV'; 
window_size = 200 #window_size in samples
size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
DOPs = []
RV2Ts = []
DOPs_RV2T = []
for i in range(0, size-window_size):
    [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
    dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
    dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
    dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
    DOPs.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
    RV2Ts.append(TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
    DOPs_RV2T.append(DOPs[i]*RV2Ts[i])
#    print(str(slope*i) + " to " + str((slope*i)+window_size-1))
fig = ml.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)
ax1.plot(events[0].trace_groups[stat].traces[0].filter_wave)
ax2.set_title('DOP per window - Station: ' + stat)
ax2.plot(DOPs)
ax3.set_title('RV2T per window - Station: ' + stat)
ax3.plot(RV2Ts)

#Attributes per window calculation for one station
slope = 40
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
    DOPs.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
    RV2Ts.append(TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
    DOPs_RV2T.append(DOPs[i]*RV2Ts[i])
#    print(str(slope*i) + " to " + str((slope*i)+window_size-1))
vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
fig = ml.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)
ax1.set_title('DOP per window - Station: ' + stat)
ax1.plot(DOPs)
i, j = ax1.get_ylim()
ax1.vlines(vline, i, j, color='r', lw=1)
ax2.set_title('RV2T per window - Station: ' + stat)
ax2.plot(RV2Ts)
i, j = ax2.get_ylim()
ax2.vlines(vline, i, j, color='r', lw=1)
ax3.set_title('RV2T*DOP per window - Station: ' + stat)
ax3.plot(DOPs_RV2T)
i, j = ax3.get_ylim()
ax3.vlines(vline, i, j, color='r', lw=1)


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
        result = TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ)
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

## TODO: DOP and RV2T Calculation
#DOP, RV2T per window calculation for all stations in order of epicentral distance
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
#        slope = 1
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)

        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            DOPs.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
            RV2Ts.append(TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
            DOPs_RV2T.append(DOPs[i]*RV2Ts[i])
#            print(str(slope*i) + " " + str((slope*i)+window_size-1))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('DOP per window - Station: ' + stat)
        ax1.plot(DOPs)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('RV2T per window - Station: ' + stat)
        ax2.plot(RV2Ts)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('RV2T*DOP per window - Station: ' + stat)
        ax3.plot(DOPs_RV2T)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        
## TODO: Entropy Calculation
#Entropy per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        Entropy = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            Entropy[0].append(NonLinear_Attributes.signal_entropy(dataX)[1])
            Entropy[1].append(NonLinear_Attributes.signal_entropy(dataY)[1])
            Entropy[2].append(NonLinear_Attributes.signal_entropy(dataZ)[1])
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        ax1.set_title('Entropy in X per window - Station: ' + stat)
        ax1.plot(Entropy[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Entropy in Y per window - Station: ' + stat)
        ax2.plot(Entropy[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Entropy in Z per window - Station: ' + stat)
        ax3.plot(Entropy[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)

#Entropy in magnitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        Entropy = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            Entropy.append(NonLinear_Attributes.signal_entropy(TelluricoTools.getResultantTrace(dataX,dataY,dataZ))[1])
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('Entropy per window - Station: ' + stat)
        ax1.plot(Entropy)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.plot(DOPs_RV2T)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)

## TODO: Kurtosis and STD Calculation
#Kurtosis and std per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        kurt = [[],[],[]]
        stdev = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            kurt[0].append(TimeDomain_Attributes.signal_kurtosis(dataX))
            kurt[1].append(TimeDomain_Attributes.signal_kurtosis(dataY))
            kurt[2].append(TimeDomain_Attributes.signal_kurtosis(dataZ))
            stdev[0].append(TimeDomain_Attributes.signal_std(dataX))
            stdev[1].append(TimeDomain_Attributes.signal_std(dataY))
            stdev[2].append(TimeDomain_Attributes.signal_std(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(421)
        ax2 = fig.add_subplot(423)
        ax3 = fig.add_subplot(425)
        ax4 = fig.add_subplot(427)
        ax5 = fig.add_subplot(422)
        ax6 = fig.add_subplot(424)
        ax7 = fig.add_subplot(426)
        ax8 = fig.add_subplot(428)
        ax1.set_title('Kurtosis in X per window - Station: ' + stat)
        ax1.plot(kurt[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Kurtosis in Y per window - Station: ' + stat)
        ax2.plot(kurt[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Kurtosis in Z per window - Station: ' + stat)
        ax3.plot(kurt[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        ax5.set_title('Standard Dev in X per window - Station: ' + stat)
        ax5.plot(stdev[0])
        i, j = ax5.get_ylim()
        ax5.vlines(vline, i, j, color='r', lw=1)
        ax6.set_title('Standard Dev in Y per window - Station: ' + stat)
        ax6.plot(stdev[1])
        i, j = ax6.get_ylim()
        ax6.vlines(vline, i, j, color='r', lw=1)
        ax7.set_title('Standard Dev in Z per window - Station: ' + stat)
        ax7.plot(stdev[2])
        i, j = ax7.get_ylim()
        ax7.vlines(vline, i, j, color='r', lw=1)
        ax8.plot(DOPs_RV2T)
        i, j = ax8.get_ylim()
        ax8.vlines(vline, i, j, color='r', lw=1)

#Kurtosis and std magnitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        kurt = []
        stdev = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            kurt.append(TimeDomain_Attributes.signal_kurtosis(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            stdev.append(TimeDomain_Attributes.signal_std(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('Kurtosis per window - Station: ' + stat)
        ax1.plot(kurt)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Standard Dev per window - Station: ' + stat)
        ax2.plot(stdev)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.plot(DOPs_RV2T)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        
## TODO: Skewness Calculation
#Skewness per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        skewness = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            skewness[0].append(TimeDomain_Attributes.signal_skew(dataX))
            skewness[1].append(TimeDomain_Attributes.signal_skew(dataY))
            skewness[2].append(TimeDomain_Attributes.signal_skew(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        ax1.set_title('Skewness X - ' + stat)
        ax1.plot(skewness[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Skewness Y - ' + stat)
        ax2.plot(skewness[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Skewness Z - ' + stat)
        ax3.plot(skewness[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)

#Skewness in magitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        skewness = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            skewness.append(TimeDomain_Attributes.signal_skew(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('Skewness X - ' + stat)
        ax1.plot(skewness)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.plot(DOPs_RV2T)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)

#Skewness in Z per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        skewness = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            skewness.append(TimeDomain_Attributes.signal_skew(dataX))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('Skewness Z - ' + stat)
        ax1.plot(skewness)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.plot(DOPs_RV2T)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)

## TODO: Geometric and Harmonic Mean Calculation
#Geometric and Harmonic Mean per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        gmean_array = [[],[],[]]
        hmean_array = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            gmean_array[0].append(TimeDomain_Attributes.signal_gmean(dataX))
            gmean_array[1].append(TimeDomain_Attributes.signal_gmean(dataY))
            gmean_array[2].append(TimeDomain_Attributes.signal_gmean(dataZ))
            hmean_array[0].append(TimeDomain_Attributes.signal_hmean(dataX))
            hmean_array[1].append(TimeDomain_Attributes.signal_hmean(dataY))
            hmean_array[2].append(TimeDomain_Attributes.signal_hmean(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(421)
        ax2 = fig.add_subplot(423)
        ax3 = fig.add_subplot(425)
        ax4 = fig.add_subplot(427)
        ax5 = fig.add_subplot(422)
        ax6 = fig.add_subplot(424)
        ax7 = fig.add_subplot(426)
        ax8 = fig.add_subplot(428)
        ax1.set_title('Geometric Mean X - : ' + stat)
        ax1.plot(gmean_array[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Geometric Mean Y - ' + stat)
        ax2.plot(gmean_array[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Geometric Mean Z - ' + stat)
        ax3.plot(gmean_array[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        ax5.set_title('Harmonic Mean X - ' + stat)
        ax5.plot(hmean_array[0])
        i, j = ax5.get_ylim()
        ax5.vlines(vline, i, j, color='r', lw=1)
        ax6.set_title('Harmonic Mean Y - ' + stat)
        ax6.plot(hmean_array[1])
        i, j = ax6.get_ylim()
        ax6.vlines(vline, i, j, color='r', lw=1)
        ax7.set_title('Harmonic Mean Z - ' + stat)
        ax7.plot(hmean_array[2])
        i, j = ax7.get_ylim()
        ax7.vlines(vline, i, j, color='r', lw=1)
        ax8.plot(DOPs_RV2T)
        i, j = ax8.get_ylim()
        ax8.vlines(vline, i, j, color='r', lw=1)
        
#Geometric and Harmonic Mean in magnitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        gmean_array = []
        hmean_array = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            gmean_array.append(TimeDomain_Attributes.signal_gmean(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            hmean_array.append(TimeDomain_Attributes.signal_hmean(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('Geometric Mean - : ' + stat)
        ax1.plot(gmean_array)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Harmonic Mean - ' + stat)
        ax2.plot(hmean_array)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.plot(DOPs_RV2T)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        
## TODO: Mean and Median Calculation
#Mean and Median per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        mean = [[],[],[]]
        median = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            mean[0].append(np.mean(dataX))
            mean[1].append(np.mean(dataY))
            mean[2].append(np.mean(dataZ))
            median[0].append(np.median(dataX))
            median[1].append(np.median(dataY))
            median[2].append(np.median(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(421)
        ax2 = fig.add_subplot(423)
        ax3 = fig.add_subplot(425)
        ax4 = fig.add_subplot(427)
        ax5 = fig.add_subplot(422)
        ax6 = fig.add_subplot(424)
        ax7 = fig.add_subplot(426)
        ax8 = fig.add_subplot(428)
        ax1.set_title('Mean in X per window - ' + stat)
        ax1.plot(mean[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Mean in Y per window - ' + stat)
        ax2.plot(mean[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Mean in Z per window - ' + stat)
        ax3.plot(mean[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        ax5.set_title('Median in X per window - ' + stat)
        ax5.plot(median[0])
        i, j = ax5.get_ylim()
        ax5.vlines(vline, i, j, color='r', lw=1)
        ax6.set_title('Median in Y per window - ' + stat)
        ax6.plot(median[1])
        i, j = ax6.get_ylim()
        ax6.vlines(vline, i, j, color='r', lw=1)
        ax7.set_title('Median in Z per window - ' + stat)
        ax7.plot(median[2])
        i, j = ax7.get_ylim()
        ax7.vlines(vline, i, j, color='r', lw=1)
        ax8.plot(DOPs_RV2T)
        i, j = ax8.get_ylim()
        ax8.vlines(vline, i, j, color='r', lw=1)

#Mean and Median in magnitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        mean = []
        median = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            mean.append(np.mean(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            median.append(np.median(TelluricoTools.getResultantTrace(dataX,dataY,dataZ)))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('Mean per window - ' + stat)
        ax1.plot(mean)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Median per window - ' + stat)
        ax2.plot(median)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.plot(DOPs_RV2T)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)

## TODO: Actovity Calculation
#Activity from Hjorth Parameters per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        activity = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            [actX, c, m] = NonLinear_Attributes.hjorth_params(dataX)
            [actY, c, m] = NonLinear_Attributes.hjorth_params(dataY)
            [actZ, c, m] = NonLinear_Attributes.hjorth_params(dataZ)
            activity[0].append(actX)
            activity[1].append(actY)
            activity[2].append(actZ)
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Activity X - ' + stat)
        ax1.plot(activity[0])
        ax2.set_title('Activity Y - ' + stat)
        ax2.plot(activity[1])
        ax3.set_title('Activity Z - ' + stat)
        ax3.plot(activity[2])
        ax4.plot(DOPs_RV2T)
        
#Hjorth Parameters in magnitude: activity, complexity and morbidity per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        activity = []
        complexity = []
        morbidity = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            [act, com, mor] = NonLinear_Attributes.hjorth_params(TelluricoTools.getResultantTrace(dataX,dataY,dataZ))
            activity.append(act)
            complexity.append(com)
            morbidity.append(mor)
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Activity - ' + stat)
        ax1.plot(activity)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Complexity - ' + stat)
        ax2.plot(complexity)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Morbidity - ' + stat)
        ax3.plot(morbidity)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)

## TODO: Pearsons Correlation Calculation
#Pearson Correlation per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        pearson = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        for i in range(0, size-window_size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            pearson[0].append(Correlation.pearson_correlation(dataX, dataY)[0])
            pearson[1].append(Correlation.pearson_correlation(dataY, dataZ)[0])
            pearson[2].append(Correlation.pearson_correlation(dataX, dataZ)[0])
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Pearsons XY - ' + stat)
        ax1.plot(pearson[0])
        ax2.set_title('Pearsons YZ - ' + stat)
        ax2.plot(pearson[1])
        ax3.set_title('Pearsons XZ - ' + stat)
        ax3.plot(pearson[2])
        ax4.plot(DOPs_RV2T)
        
## TODO: Petrosian Fractal Dimension Calculation
#Petrosian Fractal Dimension per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        petrosian = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        for i in range(0, size-window_size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            petrosian[0].append(Transformations.petrosian_fd(dataX))
            petrosian[1].append(Transformations.petrosian_fd(dataY))
            petrosian[2].append(Transformations.petrosian_fd(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Petrosian X - ' + stat)
        ax1.plot(petrosian[0])
        ax2.set_title('Petrosian Y - ' + stat)
        ax2.plot(petrosian[1])
        ax3.set_title('Petrosian Z - ' + stat)
        ax3.plot(petrosian[2])
        ax4.plot(DOPs_RV2T)

## TODO: Hjorth Parameters: activity, complexity and morbidity Calculation
#Hjorth Parameters: activity, complexity and morbidity per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        activity = [[],[],[]]
        complexity = [[],[],[]]
        morbidity = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        size = int((events[0].trace_groups[stat].S_Wave-window_size)/slope)
#        print(events[0].trace_groups[stat].S_Wave)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            [actX, comX, morX] = NonLinear_Attributes.hjorth_params(dataX)
            [actY, comY, morY] = NonLinear_Attributes.hjorth_params(dataY)
            [actZ, comZ, morZ] = NonLinear_Attributes.hjorth_params(dataZ)
            activity[0].append(actX); activity[1].append(actY); activity[2].append(actZ)
            complexity[0].append(comX); complexity[1].append(comY); complexity[2].append(comY)
            morbidity[0].append(morX); morbidity[1].append(morY); morbidity[2].append(morZ)
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(431)
        ax2 = fig.add_subplot(434)
        ax3 = fig.add_subplot(437)
        ax5 = fig.add_subplot(432)
        ax6 = fig.add_subplot(435)
        ax7 = fig.add_subplot(438)        
        ax9 = fig.add_subplot(433)
        ax10 = fig.add_subplot(436)
        ax11 = fig.add_subplot(439)
        
        ax1.set_title('Activity X - ' + stat)
        ax1.plot(activity[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Activity Y - ' + stat)
        ax2.plot(activity[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Activity Z - ' + stat)
        ax3.plot(activity[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax5.set_title('Complexity X - ' + stat)
        ax5.plot(complexity[0])
        i, j = ax5.get_ylim()
        ax5.vlines(vline, i, j, color='r', lw=1)
        ax6.set_title('Complexity Y - ' + stat)
        ax6.plot(complexity[1])
        i, j = ax6.get_ylim()
        ax6.vlines(vline, i, j, color='r', lw=1)
        ax7.set_title('Complexity Z - ' + stat)
        ax7.plot(complexity[2])
        i, j = ax7.get_ylim()
        ax7.vlines(vline, i, j, color='r', lw=1)
        ax9.set_title('Morbidity X - ' + stat)
        ax9.plot(morbidity[0])
        i, j = ax9.get_ylim()
        ax9.vlines(vline, i, j, color='r', lw=1)
        ax10.set_title('Morbidity Y - ' + stat)
        ax10.plot(morbidity[1])
        i, j = ax10.get_ylim()
        ax10.vlines(vline, i, j, color='r', lw=1)
        ax11.set_title('Morbidity Z - ' + stat)
        ax11.plot(morbidity[2])
        i, j = ax11.get_ylim()
        ax11.vlines(vline, i, j, color='r', lw=1)
        

#Calculation of cross correlation between Attributes DOP and RV2T per window for all stations 
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
            DOPs.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
            RV2Ts.append(TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
#            print(str(slope*i) + " " + str((slope*i)+window_size-1))
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('DOP per window - Station: ' + stat)
        ax1.plot(DOPs)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('RV2T per window - Station: ' + stat)
        ax2.plot(RV2Ts)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        print(stat + ' pearsons correlation: ' + str(Correlation.pearson_correlation(DOPs, RV2Ts)[0]))
    
    
    

# Envelope for one station
TimeDomain_Attributes.envelope(events[0].trace_groups[stat].traces[0].filter_wave, 
                    events[0].trace_groups[stat].traces[0].sampling_rate)



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



# Logarithmic Welch Periodogram for one filtered signal and random-positioned noise, one station
stat = 'BRR'
window_size = 1000
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].filter_wave, window_size, init, 0.5)
Transformations.welch_periodogram_log(signal, events[0].trace_groups[stat].traces[0].sampling_rate)
Transformations.welch_periodogram_log(noise, events[0].trace_groups[stat].traces[0].sampling_rate)

# Logarithmic Welch Periodogram for one original signal and random-positioned noise, one station
stat = 'BRR'
window_size = 1000
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].waveform, window_size, init, 0.5)
Transformations.welch_periodogram_log(signal, events[0].trace_groups[stat].traces[0].sampling_rate)
Transformations.welch_periodogram_log(noise, events[0].trace_groups[stat].traces[0].sampling_rate)

# Linear Welch Periodogram for one filtered signal and random-positioned noise, one station
stat = 'BRR'
window_size = 1000
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].filter_wave, window_size, init, 0.5)
Transformations.welch_periodogram_linear(signal, events[0].trace_groups[stat].traces[0].sampling_rate)
Transformations.welch_periodogram_linear(noise, events[0].trace_groups[stat].traces[0].sampling_rate)

# Linear Welch Periodogram for one original signal and random-positioned noise, one station
stat = 'BRR'
window_size = 1000
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].waveform, window_size, init, 0.5)
Transformations.welch_periodogram_linear(signal, events[0].trace_groups[stat].traces[0].sampling_rate)
Transformations.welch_periodogram_linear(noise, events[0].trace_groups[stat].traces[0].sampling_rate)

# Linear Welch Periodogram for one original window P_Wave to S_Wave signal and random-positioned noise, one station
stat = 'BRR'
window_size = 1000
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p2s_noise_extraction(events[0].trace_groups[stat].traces[0].waveform, init, events[0].trace_groups[stat].S_Wave)
Transformations.welch_periodogram_linear(signal, events[0].trace_groups[stat].traces[0].sampling_rate)
Transformations.welch_periodogram_linear(noise, events[0].trace_groups[stat].traces[0].sampling_rate)

# Difference in Linear Welch Periodogram for one filtered signal and random-positioned noise, one station
stat = 'BRR'
window_size = 200
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].filter_wave, window_size, init, 0.5)
Transformations.welch_periodogram_linear_diff(signal, noise, events[0].trace_groups[stat].traces[0].sampling_rate)

# Difference in Linear Welch Periodogram for one original signal and random-positioned noise, one station
stat = 'BRR'
window_size = 200
init = events[0].trace_groups[stat].P_Wave
[signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].waveform, window_size, init, 0.5)
Transformations.welch_periodogram_linear_diff(signal, noise, events[0].trace_groups[stat].traces[0].sampling_rate)

#Progressive Periodogram per window for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        perio_max = [[],[],[]]
        DOPs_RV2T = []
        window_size = 200
        slope = int(window_size/5)
        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        for i in range(0, size-window_size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            perio_max[0].append(max(Transformations.periodogram_linear(dataX, events[0].trace_groups[stat].traces[0].sampling_rate)))
            perio_max[1].append(max(Transformations.periodogram_linear(dataY, events[0].trace_groups[stat].traces[0].sampling_rate)))
            perio_max[2].append(max(Transformations.periodogram_linear(dataZ, events[0].trace_groups[stat].traces[0].sampling_rate)))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Max Power X - ' + stat)
        ax1.plot(perio_max[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Max Power Y - ' + stat)
        ax2.plot(perio_max[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Max Power Z - ' + stat)
        ax3.plot(perio_max[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
    
    
    
# P-wave and noise extraction for all stations and one event
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    [p_signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].filter_wave, 200, 
                    events[0].trace_groups[stat].P_Wave, 0.5)
    fig = ml.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title('P_Wave - ' + stat)
    ax1.plot(p_signal)
    ax2.set_title('Noise - ' + stat)
    ax2.plot(noise)

# P-wave to S-wave and noise extraction for all stations and one event
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(events[0].trace_groups[stat].S_Wave != 0):
        [p_signal, noise] = TelluricoTools.p2s_noise_extraction(events[0].trace_groups[stat].traces[0].filter_wave, 
                        events[0].trace_groups[stat].P_Wave, events[0].trace_groups[stat].S_Wave)
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('P_Wave - ' + stat)
        ax1.plot(p_signal)
        ax2.set_title('Noise - ' + stat)
        ax2.plot(noise)



# Autoregressive model
stat = 'BRR'
[p_signal, noise] = TelluricoTools.p_noise_extraction(events[0].trace_groups[stat].traces[0].filter_wave, 200, 
                    events[0].trace_groups[stat].P_Wave, 0.5)
AR_signal = NonLinear_Attributes.AR_coeff(p_signal)
ml.plot(AR_signal)
AR_noise = NonLinear_Attributes.AR_coeff(noise)
ml.plot(AR_noise)



## TODO: Máx Lyapunov Exponent Calculation
#Máx Lyapunov Exponent per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        lyapv = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            lyapv[0].append(NonLinear_Attributes.lyapunov_exp_max(dataX))
            lyapv[1].append(NonLinear_Attributes.lyapunov_exp_max(dataY))
            lyapv[2].append(NonLinear_Attributes.lyapunov_exp_max(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Lyapunov Exponent X - ' + stat)
        ax1.plot(lyapv[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Lyapunov Exponent Y - ' + stat)
        ax2.plot(lyapv[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Lyapunov Exponent Z - ' + stat)
        ax3.plot(lyapv[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        
#Máx Lyapunov and Hurst Exponent in magnitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        lyapv = []
        hurst = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            lyapv.append(NonLinear_Attributes.lyapunov_exp_max(TelluricoTools.getResultantTraceNorm(dataX,dataY,dataZ)))
            hurst.append(NonLinear_Attributes.hurst_exp(TelluricoTools.getResultantTraceNorm(dataX,dataY,dataZ)))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        
        ax1.set_title('Lyapunov Exponent - ' + stat)
        ax1.plot(lyapv)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Hurst Exponent - ' + stat)
        ax2.plot(hurst)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.plot(DOPs_RV2T)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)



## TODO: Hurst Exponent Calculation
#Hurst Exponent per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        hurst = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            hurst[0].append(NonLinear_Attributes.hurst_exp(dataX))
            hurst[1].append(NonLinear_Attributes.hurst_exp(dataY))
            hurst[2].append(NonLinear_Attributes.hurst_exp(dataZ))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Hurst Exponent X - ' + stat)
        ax1.plot(hurst[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Hurst Exponent Y - ' + stat)
        ax2.plot(hurst[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Hurst Exponent Z - ' + stat)
        ax3.plot(hurst[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)



## TODO: Correlation Dimension Calculation
#Correlation Dimension per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        corrDim = [[],[],[]]
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            corrDim[0].append(NonLinear_Attributes.corr_CD(dataX, 1))
            corrDim[1].append(NonLinear_Attributes.corr_CD(dataY, 1))
            corrDim[2].append(NonLinear_Attributes.corr_CD(dataZ, 1))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.set_title('Correlation Dimension X - ' + stat)
        ax1.plot(corrDim[0])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('Correlation Dimension Y - ' + stat)
        ax2.plot(corrDim[1])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('Correlation Dimension Z - ' + stat)
        ax3.plot(corrDim[2])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax4.plot(DOPs_RV2T)
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        
#Correlation Dimension in magnitude per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        corrDim = []
        DOPs_RV2T = []
        window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
        window_size = 200
        slope = int(window_size/5)
#        size = int(len(events[0].trace_groups[stat].traces[0].filter_wave)/slope)
        size = int((len(events[0].trace_groups[stat].traces[0].filter_wave)-window_size)/slope)
        for i in range(0, size):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,(slope*i),((slope*i)+window_size-1))
            corrDim.append(NonLinear_Attributes.corr_CD(TelluricoTools.getResultantTrace(dataX,dataY,dataZ), 1))
            DOPs_RV2T.append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ)*TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
        
        vline = round((events[0].trace_groups[stat].P_Wave-window_size)/slope)
        fig = ml.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        ax1.set_title('Correlation Dimension X - ' + stat)
        ax1.plot(corrDim)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.plot(DOPs_RV2T)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)

## TODO: ALL FEATURES FOR ALL STATIONS IN ORDER OF EPICENTRAL DISTANCE
# DOP, RV2T, Entropy, Kurtosis, Skewness, Max Lyapunov Exp, Correlation Dimension
for tuple_sort in stats_sort[:5]:
    stat = (tuple_sort[0])
    if(stat in events[0].trace_groups):
        observ_signal = {'DOP':[],'RV2T':[],'Entropy':[],'Kurtosis':[],'Skew':[],'Lyapunov':[],'CD':[]}
        window_size = 50
        p_mark = events[0].trace_groups[stat].P_Wave
        windows = 2
        init = p_mark - (windows*window_size) - int(window_size/2)
        fin = p_mark + (windows*window_size) + int(window_size/2)
        for i in range(init, fin+1):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,i,i+window_size)
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,i,i+window_size)
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,i,i+window_size)
            resultant = TelluricoTools.getResultantTrace(dataX,dataY,dataZ)
            
            observ_signal['DOP'].append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
            observ_signal['RV2T'].append(TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
            observ_signal['Entropy'].append(NonLinear_Attributes.signal_entropy(resultant)[1])
            observ_signal['Kurtosis'].append(TimeDomain_Attributes.signal_kurtosis(resultant))
            observ_signal['Skew'].append(TimeDomain_Attributes.signal_skew(resultant))
            observ_signal['Lyapunov'].append(NonLinear_Attributes.lyapunov_exp_max(dataX))
            observ_signal['CD'].append(NonLinear_Attributes.corr_CD(resultant, 1))
        
        vline = (windows*window_size)
        fig = ml.figure()
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(334, sharex=ax1)
        ax3 = fig.add_subplot(337, sharex=ax1)
        ax4 = fig.add_subplot(332, sharex=ax1)
        ax5 = fig.add_subplot(335, sharex=ax1)
        ax6 = fig.add_subplot(333, sharex=ax1)
        ax7 = fig.add_subplot(336, sharex=ax1)
        
        ax1.set_title('DOP - ' + stat)
        ax1.plot(observ_signal['DOP'])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax1.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax2.set_title('RV2T - ' + stat)
        ax2.plot(observ_signal['RV2T'])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax2.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax3.set_title('Entropy - ' + stat)
        ax3.plot(observ_signal['Entropy'])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax3.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax4.set_title('Kurtosis - ' + stat)
        ax4.plot(observ_signal['Kurtosis'])
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        ax4.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax5.set_title('Skewness - ' + stat)
        ax5.plot(observ_signal['Skew'])
        i, j = ax5.get_ylim()
        ax5.vlines(vline, i, j, color='r', lw=1)
        ax5.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax6.set_title('Lyapunov Exp - ' + stat)
        ax6.plot(observ_signal['Lyapunov'])
        i, j = ax6.get_ylim()
        ax6.vlines(vline, i, j, color='r', lw=1)
        ax6.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax7.set_title('Correlation Dimension - ' + stat)
        ax7.plot(observ_signal['CD'])
        i, j = ax7.get_ylim()
        ax7.vlines(vline, i, j, color='r', lw=1)
        ax7.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)

## TODO: ALL FEATURES FOR CERTAIN STATIONS IN ORDER OF EPICENTRAL DISTANCE
# DOP, RV2T, Entropy, Kurtosis, Skewness, Max Lyapunov Exp, Correlation Dimension
stations = ['RUS','BRR','PAM']
for stat in stations:
    if(stat in events[0].trace_groups):
        observ_signal = {'DOP':[],'RV2T':[],'Entropy':[],'Kurtosis':[],'Skew':[],'Lyapunov':[],'CD':[]}
        window_size = 50
        p_mark = events[0].trace_groups[stat].P_Wave
        windows = 2
        init = p_mark - (windows*window_size) - int(window_size/2)
        fin = p_mark + (windows*window_size) + int(window_size/2)
        for i in range(init, fin+1):
            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(events[0].trace_groups[stat])
            dataX = TelluricoTools.sub_trace(dataX.filter_wave,i,i+window_size)
            dataY = TelluricoTools.sub_trace(dataY.filter_wave,i,i+window_size)
            dataZ = TelluricoTools.sub_trace(dataZ.filter_wave,i,i+window_size)
            resultant = TelluricoTools.getResultantTrace(dataX,dataY,dataZ)
            
            observ_signal['DOP'].append(TimeDomain_Attributes.DOP(dataX,dataY,dataZ))
            observ_signal['RV2T'].append(TimeDomain_Attributes.RV2T(dataX,dataY,dataZ))
            observ_signal['Entropy'].append(NonLinear_Attributes.signal_entropy(resultant)[1])
            observ_signal['Kurtosis'].append(TimeDomain_Attributes.signal_kurtosis(resultant))
            observ_signal['Skew'].append(TimeDomain_Attributes.signal_skew(resultant))
            observ_signal['CD'].append(NonLinear_Attributes.corr_CD(resultant, 1))
        
        vline = (windows*window_size)
        fig = ml.figure()
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(334, sharex=ax1)
        ax3 = fig.add_subplot(337, sharex=ax1)
        ax4 = fig.add_subplot(332, sharex=ax1)
        ax5 = fig.add_subplot(335, sharex=ax1)
        ax6 = fig.add_subplot(333, sharex=ax1)
        
        ax1.set_title('DOP - ' + stat)
        ax1.plot(observ_signal['DOP'])
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax1.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax2.set_title('RV2T - ' + stat)
        ax2.plot(observ_signal['RV2T'])
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax2.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax3.set_title('Entropy - ' + stat)
        ax3.plot(observ_signal['Entropy'])
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)
        ax3.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax4.set_title('Kurtosis - ' + stat)
        ax4.plot(observ_signal['Kurtosis'])
        i, j = ax4.get_ylim()
        ax4.vlines(vline, i, j, color='r', lw=1)
        ax4.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax5.set_title('Skewness - ' + stat)
        ax5.plot(observ_signal['Skew'])
        i, j = ax5.get_ylim()
        ax5.vlines(vline, i, j, color='r', lw=1)
        ax5.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)
        ax6.set_title('Correlation Dimension - ' + stat)
        ax6.plot(observ_signal['CD'])
        i, j = ax6.get_ylim()
        ax6.vlines(vline, i, j, color='r', lw=1)
        ax6.vlines(vline-(0.9*(window_size/2)), i, j, color='k', lw=1)


from SfileAnalyzer import SfileAnalyzer
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

for i in range(0,1000):
    stat = 'BRR'
    observ_signal = {}
    observ_noise = {}
#    start = time.time()
    [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
    [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.5)
    [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.5)
    [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.5)
    
    
    observ_signal['DOP'] = TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)
    #print('DOP Signal')
    observ_signal['RV2T'] = TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)
    #print('RV2T Signal')
    observ_signal['EntropyZ'] = NonLinear_Attributes.signal_entropy(p_signal_X)[1]
    observ_signal['EntropyN'] = NonLinear_Attributes.signal_entropy(p_signal_Y)[1]
    observ_signal['EntropyE'] = NonLinear_Attributes.signal_entropy(p_signal_Z)[1]
    #print('Entropy Signal')
    observ_signal['KurtosisZ'] = TimeDomain_Attributes.signal_kurtosis(p_signal_X)
    observ_signal['KurtosisN'] = TimeDomain_Attributes.signal_kurtosis(p_signal_Y)
    observ_signal['KurtosisE'] = TimeDomain_Attributes.signal_kurtosis(p_signal_Z)
    #print('Kurtosis Signal')
    observ_signal['SkewZ'] = TimeDomain_Attributes.signal_skew(p_signal_X)
    observ_signal['SkewN'] = TimeDomain_Attributes.signal_skew(p_signal_Y)
    observ_signal['SkewE'] = TimeDomain_Attributes.signal_skew(p_signal_Z)
    #print('Skew Signal')
    observ_signal['LyapunovZ'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_X)
    observ_signal['LyapunovN'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_Y)
    observ_signal['LyapunovE'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_Z)
    #print('Lyapunov Signal')
    observ_signal['CDZ'] = NonLinear_Attributes.corr_CD(p_signal_X, 1)
    observ_signal['CDN'] = NonLinear_Attributes.corr_CD(p_signal_Y, 1)
    observ_signal['CDE'] = NonLinear_Attributes.corr_CD(p_signal_Z, 1)
    #print('CD Signal')
    
    observ_noise['DOP'] = TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)
    #print('DOP Noise')
    observ_noise['RV2T'] = TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)
    #print('RV2T Noise')
    observ_noise['EntropyZ'] = NonLinear_Attributes.signal_entropy(noise_X)[1]
    observ_noise['EntropyN'] = NonLinear_Attributes.signal_entropy(noise_Y)[1]
    observ_noise['EntropyE'] = NonLinear_Attributes.signal_entropy(noise_Z)[1]
    #print('Entropy Noise')
    observ_noise['KurtosisZ'] = TimeDomain_Attributes.signal_kurtosis(noise_X)
    observ_noise['KurtosisN'] = TimeDomain_Attributes.signal_kurtosis(noise_Y)
    observ_noise['KurtosisE'] = TimeDomain_Attributes.signal_kurtosis(noise_Z)
    #print('Kurtosis Noise')
    observ_noise['SkewZ'] = TimeDomain_Attributes.signal_skew(noise_X)
    observ_noise['SkewN'] = TimeDomain_Attributes.signal_skew(noise_Y)
    observ_noise['SkewE'] = TimeDomain_Attributes.signal_skew(noise_Z)
    #print('Skew Noise')
    observ_noise['LyapunovZ'] = NonLinear_Attributes.lyapunov_exp_max(noise_X)
    observ_noise['LyapunovN'] = NonLinear_Attributes.lyapunov_exp_max(noise_Y)
    observ_noise['LyapunovE'] = NonLinear_Attributes.lyapunov_exp_max(noise_Z)
    #print('Lyapunov Noise')
    observ_noise['CDZ'] = NonLinear_Attributes.corr_CD(noise_X, 1)
    observ_noise['CDN'] = NonLinear_Attributes.corr_CD(noise_Y, 1)
    observ_noise['CDE'] = NonLinear_Attributes.corr_CD(noise_Z, 1)
    #print('CD Noise')
#    end = time.time()
    #print(str(end-start) + ' seconds')


stat = 'BRR'
observ_signal = {}
observ_noise = {}

[dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
[p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.5)
[p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.5)
[p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave, 0.5)


observ_signal['DOP'] = TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)
print('DOP Signal')
observ_signal['RV2T'] = TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)
print('RV2T Signal')
observ_signal['EntropyZ'] = NonLinear_Attributes.signal_entropy(p_signal_X)[1]
observ_signal['EntropyN'] = NonLinear_Attributes.signal_entropy(p_signal_Y)[1]
observ_signal['EntropyE'] = NonLinear_Attributes.signal_entropy(p_signal_Z)[1]
print('Entropy Signal')
observ_signal['KurtosisZ'] = TimeDomain_Attributes.signal_kurtosis(p_signal_X)
observ_signal['KurtosisN'] = TimeDomain_Attributes.signal_kurtosis(p_signal_Y)
observ_signal['KurtosisE'] = TimeDomain_Attributes.signal_kurtosis(p_signal_Z)
print('Kurtosis Signal')
observ_signal['SkewZ'] = TimeDomain_Attributes.signal_skew(p_signal_X)
observ_signal['SkewN'] = TimeDomain_Attributes.signal_skew(p_signal_Y)
observ_signal['SkewE'] = TimeDomain_Attributes.signal_skew(p_signal_Z)
print('Skew Signal')
observ_signal['LyapunovZ'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_X)
observ_signal['LyapunovN'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_Y)
observ_signal['LyapunovE'] = NonLinear_Attributes.lyapunov_exp_max(p_signal_Z)
print('Lyapunov Signal')
observ_signal['CDZ'] = NonLinear_Attributes.corr_CD(p_signal_X, 1)
observ_signal['CDN'] = NonLinear_Attributes.corr_CD(p_signal_Y, 1)
observ_signal['CDE'] = NonLinear_Attributes.corr_CD(p_signal_Z, 1)
print('CD Signal')

observ_noise['DOP'] = TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)
print('DOP Noise')
observ_noise['RV2T'] = TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)
print('RV2T Noise')
observ_noise['EntropyZ'] = NonLinear_Attributes.signal_entropy(noise_X)[1]
observ_noise['EntropyN'] = NonLinear_Attributes.signal_entropy(noise_Y)[1]
observ_noise['EntropyE'] = NonLinear_Attributes.signal_entropy(noise_Z)[1]
print('Entropy Noise')
observ_noise['KurtosisZ'] = TimeDomain_Attributes.signal_kurtosis(noise_X)
observ_noise['KurtosisN'] = TimeDomain_Attributes.signal_kurtosis(noise_Y)
observ_noise['KurtosisE'] = TimeDomain_Attributes.signal_kurtosis(noise_Z)
print('Kurtosis Noise')
observ_noise['SkewZ'] = TimeDomain_Attributes.signal_skew(noise_X)
observ_noise['SkewN'] = TimeDomain_Attributes.signal_skew(noise_Y)
observ_noise['SkewE'] = TimeDomain_Attributes.signal_skew(noise_Z)
print('Skew Noise')
observ_noise['LyapunovZ'] = NonLinear_Attributes.lyapunov_exp_max(noise_X)
observ_noise['LyapunovN'] = NonLinear_Attributes.lyapunov_exp_max(noise_Y)
observ_noise['LyapunovE'] = NonLinear_Attributes.lyapunov_exp_max(noise_Z)
print('Lyapunov Noise')
observ_noise['CDZ'] = NonLinear_Attributes.corr_CD(noise_X, 1)
observ_noise['CDN'] = NonLinear_Attributes.corr_CD(noise_Y, 1)
observ_noise['CDE'] = NonLinear_Attributes.corr_CD(noise_Z, 1)

str_row = ''
for feature in observ_signal:
    str_row += str(observ_signal[feature])+','
str_row += str(1)
print(str_row)
#the_file.write(str_row+'\n')

str_row = ''
for feature in observ_noise:
    str_row += str(observ_noise[feature])+','
str_row += str(0)
print(str_row)
#the_file.write(str_row+'\n')



## -*- coding: utf-8 -*-
#"""
#Prototipo 0 de Tellurico
#Es necesario establecer el patron de documentacion con sphinx
#Plantillas y demas
#Este prototipo tiene como objetivo arrancar el desarrollo
#A partir del prototipo 1 y en adelante, vinculados con sprints
#Se tendra todo documentado de forma estandar
#Toda la documentacion debe ser en ingles
#"""
#
## Import the libraries
#from obspy import read
#from tools import TelluricoTools
#from tools import SeismicInfo 
#from tools import Transformations, TimeDomain_Attributes, FreqDomain_Attributes
#from tools import NonLinear_Attributes, Correlation
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
#import time
#from SfileAnalyzer import SfileAnalyzer
#from Sfile import Sfile
#from Waveform import Waveform
#import gc
#
#''' DATASET READING AND PRE-PROCESSING '''
#
#waveforms_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Waveforms/'
#sfiles_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Sfiles/'
#sfileAnalyzer = SfileAnalyzer(sfiles_path)
#sfileAnalyzer.get_sfiles()
#sfiles = sfileAnalyzer.sfiles
#waveforms = []
#for sfile in sfiles:
#    waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
#
#index = 0
#waveforms_valid = []
#for waveform in waveforms:
#    try:
#        st = read(waveform.waveform_path + waveform.waveform_filename)
#        waveforms_valid.append(waveform)
#    except:
#        print(waveform.waveform_filename)
#        index += 1
#print(index)
#
#''' DATASET ATRIBUTES '''
#
##events = []
##
##index = 30
##for waveform in waveforms_valid[30:50]:
##    [newEvent, stats_sort] = waveform.get_event()
##    events.append(newEvent)
##    print('Waveform ' + str(index) + ' done')
##    index += 1
##
##
### DOP, RV2T, es_sismo
##stat = 'BRR'
##with open('attributes_matrix.txt', 'a') as the_file:
##    for event in events:
##        if(stat in event.trace_groups):
##            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(event.trace_groups[stat])
##            [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, event.trace_groups[stat].P_Wave)
##            [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, event.trace_groups[stat].P_Wave)
##            [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, event.trace_groups[stat].P_Wave)
##            
##            DOP_signal = TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)
##            RV2T_signal = TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)
##            DOP_noise = TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)
##            RV2T_noise = TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)
##        
##            the_file.write(str(DOP_signal)+','+str(RV2T_signal)+','+str(1)+'\n')
##            the_file.write(str(DOP_noise)+','+str(RV2T_noise)+','+str(0)+'\n')
#    
#stat = 'BRR'        
#index = 1
#with open('attributes_matrix_prot02.txt', 'a') as the_file:
#    for waveform in waveforms_valid:
#        [newEvent, stats_sort] = waveform.get_event()
#        if(stat in newEvent.trace_groups):
#            [dataX, dataY, dataZ] = TelluricoTools.xyz_array(newEvent.trace_groups[stat])
#            [p_signal_X, noise_X] = TelluricoTools.p_noise_extraction(dataX.filter_wave, 200, newEvent.trace_groups[stat].P_Wave)
#            [p_signal_Y, noise_Y] = TelluricoTools.p_noise_extraction(dataY.filter_wave, 200, newEvent.trace_groups[stat].P_Wave)
#            [p_signal_Z, noise_Z] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, 200, newEvent.trace_groups[stat].P_Wave)
#            
#            DOP_signal = TimeDomain_Attributes.DOP(p_signal_X,p_signal_Y,p_signal_Z)
#            RV2T_signal = TimeDomain_Attributes.RV2T(p_signal_X,p_signal_Y,p_signal_Z)
#            DOP_noise = TimeDomain_Attributes.DOP(noise_X,noise_Y,noise_Z)
#            RV2T_noise = TimeDomain_Attributes.RV2T(noise_X,noise_Y,noise_Z)
#            Entropy_signal_X = NonLinear_Attributes.signal_entropy(p_signal_X)[1]
#            Entropy_signal_Y = NonLinear_Attributes.signal_entropy(p_signal_Y)[1]
#            Entropy_signal_Z = NonLinear_Attributes.signal_entropy(p_signal_Z)[1]
#            Entropy_noise_X = NonLinear_Attributes.signal_entropy(noise_X)[1]
#            Entropy_noise_Y = NonLinear_Attributes.signal_entropy(noise_Y)[1]
#            Entropy_noise_Z = NonLinear_Attributes.signal_entropy(noise_Z)[1]
#        
#            the_file.write(str(DOP_signal)+','+str(RV2T_signal)+','+str(Entropy_signal_X)+','+
#                           str(Entropy_signal_Y)+','+str(Entropy_signal_Z)+','+str(1)+'\n')
#            the_file.write(str(DOP_noise)+','+str(RV2T_noise)+','+str(Entropy_noise_X)+','+
#                           str(Entropy_noise_Y)+','+str(Entropy_noise_Z)+','+str(0)+'\n')
#        print('Waveform ' + str(index) + ' done')
#        index += 1
#        gc.collect()
#
#''' END OF ATRIBUTES '''





## Import the libraries
#from obspy import read
#from tools import TelluricoTools
#from TraceComponent import TraceComponent
#from TraceGroup import TraceGroup
#from Event import Event
#import obspy.core.utcdatetime as dt
#
#class Waveform:
#
#    def __init__(self, waveform_path, waveform_filename, sfile):
#        
#        # COMPOROBAR SI LA RUTA Y EL ARCHIVO EXISTEN
#        self.waveform_path = waveform_path
#        self.waveform_filename = waveform_filename
#        self.sfile = sfile
#        
#    def get_event(self):
#        st = read(self.waveform_path + self.waveform_filename)
#        traces = []
#        for trace in st:
#            if(TelluricoTools.check_trace(trace) and trace.stats.channel[1] != 'N'):
#                traces.append(trace)
#        
##        station_name = traces[0].stats.station
##        trace_group = TraceGroup(station_name)
##        newEvent = Event(None)
##        for trace in traces:
##            if(trace.stats.station == station_name):
##                trace_group.addTrace(TraceComponent(trace))
##            else:
##                newEvent.addTraceGroup(trace_group, station_name)
##                station_name = trace.stats.station
##                trace_group = TraceGroup(station_name)
##                trace_group.addTrace(TraceComponent(trace))
##        newEvent.addTraceGroup(trace_group, station_name)
#        
#        newEvent = Event(None)
#        for trace in traces:
#            if(trace.stats.station not in newEvent.trace_groups):
#                trace_group = TraceGroup(trace.stats.station)
#                trace_group.addTrace(TraceComponent(trace))
#                newEvent.addTraceGroup(trace_group, trace.stats.station)
#            else:
#                newEvent.trace_groups[trace.stats.station].addTrace(TraceComponent(trace))
#                
#        for station in self.sfile.type_7:
#                if(station['STAT'] in newEvent.trace_groups):
#                    newEvent.trace_groups[station['STAT']].epicentral_dist = station['DIS']
#                    if(station['PHAS'] == 'P'):
#                        year = newEvent.trace_groups[station['STAT']].traces[0].starttime.year
#                        month = newEvent.trace_groups[station['STAT']].traces[0].starttime.month
#                        day = newEvent.trace_groups[station['STAT']].traces[0].starttime.day
#                        if int(station['HR']) < int(newEvent.trace_groups[station['STAT']].traces[0].starttime.hour):
#                            day += 1
#                        newEvent.trace_groups[station['STAT']].P_Wave = int((dt.UTCDateTime(year,month,day,
#                              int(station['HR']),int(station['MM']),float(station['SECON'])) - 
#                              newEvent.trace_groups[station['STAT']].traces[0].starttime)*
#                              newEvent.trace_groups[station['STAT']].traces[0].original_sampling_rate)
##                        print("P-Wave: " + station['STAT'] + ": " + str(newEvent.trace_groups[station['STAT']].P_Wave))
#                        new_df = newEvent.trace_groups[station['STAT']].traces[0].sampling_rate
#                        original_df = newEvent.trace_groups[station['STAT']].traces[0].original_sampling_rate
#                        if(original_df != new_df):
#                            newEvent.trace_groups[station['STAT']].P_Wave = round((new_df/original_df)*newEvent.trace_groups[station['STAT']].P_Wave)
#        #                print("P-Wave: " + station['STAT'] + ": " + str(newEvent.trace_groups[station['STAT']].P_Wave))
#            #            print(dt.UTCDateTime(year,month,day,int(station['HR']),int(station['MM']),float(station['SECON'])))
#                    if(station['PHAS'] == 'S'):
#                        year = newEvent.trace_groups[station['STAT']].traces[0].starttime.year
#                        month = newEvent.trace_groups[station['STAT']].traces[0].starttime.month
#                        day = newEvent.trace_groups[station['STAT']].traces[0].starttime.day
#                        if int(station['HR']) < int(newEvent.trace_groups[station['STAT']].traces[0].starttime.hour):
#                            day += 1
#                        newEvent.trace_groups[station['STAT']].S_Wave = int((dt.UTCDateTime(year,month,day,
#                              int(station['HR']),int(station['MM']),float(station['SECON'])) - 
#                              newEvent.trace_groups[station['STAT']].traces[0].starttime)*
#                              newEvent.trace_groups[station['STAT']].traces[0].original_sampling_rate)
##                        print("S-Wave: " + station['STAT'] + ": " + str(newEvent.trace_groups[station['STAT']].S_Wave))            
#                        new_df = newEvent.trace_groups[station['STAT']].traces[0].sampling_rate
#                        original_df = newEvent.trace_groups[station['STAT']].traces[0].original_sampling_rate
#                        if(original_df != new_df):
#                            newEvent.trace_groups[station['STAT']].S_Wave = round((new_df/original_df)*newEvent.trace_groups[station['STAT']].S_Wave)
#        #                print("S-Wave: " + station['STAT'] + ": " + str(newEvent.trace_groups[station['STAT']].S_Wave))
##                else:
##                    print(station['STAT'])
#        
#        stats_delete = []
#        stats_sort = {}
#        for station_wave in newEvent.trace_groups:
#            if(newEvent.trace_groups[station_wave].P_Wave == 0):
#                stats_delete.append(station_wave)   
#            elif(newEvent.trace_groups[station_wave].S_Wave > 0):
#                newEvent.trace_groups[station_wave].alert_time =  (newEvent.trace_groups[station_wave].S_Wave -
#                    newEvent.trace_groups[station_wave].P_Wave)/newEvent.trace_groups[station_wave].traces[0].sampling_rate
#                if(len(newEvent.trace_groups[station_wave].traces) == 3):
#                    if(newEvent.trace_groups[station_wave].epicentral_dist.strip() != ''):
#                        stats_sort[station_wave] = float(newEvent.trace_groups[station_wave].epicentral_dist)
#                    else:
#                        stats_delete.append(station_wave)
#            if(len(newEvent.trace_groups[station_wave].traces) != 3 and station_wave not in stats_delete):
#                stats_delete.append(station_wave)
#        #    else:
#        #        ml.plot(newEvent.trace_groups[station_wave].traces[0].filter_wave)
#                
#        #        print("Alert time " + station_wave + ": " + str(newEvent.trace_groups[station_wave].alert_time))                                         
#        for stat in stats_delete:
#                newEvent.trace_groups.pop(stat)
#        
#        stats_sort = TelluricoTools.sort(stats_sort)
#        
#        return newEvent, stats_sort    
    
    
    

    
#from obspy import read
#from tools import TelluricoTools
#from tools import SeismicInfo 
#from tools import Transformations, TimeDomain_Attributes
##from tools import FreqDomain_Attributes
#from tools import NonLinear_Attributes, Correlation
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
#import time
#from SfileAnalyzer import SfileAnalyzer
#from Sfile import Sfile
#from Waveform import Waveform
#import gc
#from multiprocessing import Process
#
#class prototype_v0:
#    
#    def __init__(self):
#        self.read_files()
#    
#    def read_files(self):
#
#        ''' DATASET READING AND PRE-PROCESSING '''
#        
#        waveforms_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Waveforms/'
#        sfiles_path = '/home/administrador/Tellurico/Archivos_Prueba/PrototipoV0_1/Sfiles/'
#        sfileAnalyzer = SfileAnalyzer(sfiles_path)
#        sfileAnalyzer.get_sfiles()
#        sfiles = sfileAnalyzer.sfiles
#        waveforms = []
#        for sfile in sfiles:
#            waveforms.append(Waveform(waveforms_path, sfile.type_6['BINARY_FILENAME'], sfile))
#        
#        index = 0
#        waveforms_valid = []
#        for waveform in waveforms:
#            try:
#                read(waveform.waveform_path + waveform.waveform_filename)
#                waveforms_valid.append(waveform)
#            except:
#                print(waveform.waveform_filename)
#                index += 1
#        print(index)
#        
#        ''' DATASET ATRIBUTES '''
#        
#        cores = 3
#        step = int(len(waveforms_valid)/cores)
#        stat = 'BRR'
#        
#        p1 = Process(target=self.attributes, args=('att_p1.txt',waveforms_valid[0:step],stat,0))
#        p2 = Process(target=self.attributes, args=('att_p2.txt',waveforms_valid[step:2*step],stat,step))
#        p3 = Process(target=self.attributes, args=('att_p3.txt',waveforms_valid[2*step:3*step],stat,2*step))
#        p4 = Process(target=self.attributes, args=('att_p4.txt',waveforms_valid[3*step:len(waveforms_valid)],stat,3*step))
#        
#        p1.start()
#        p2.start()
#        p3.start()
#        p4.start()
#        
#        p1.join()
#        p2.join()
#        p3.join()
#        p4.join()
#    
#    def attributes(self,filename, waveforms_valid, stat, begin):     
#        index = begin
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
#                print('Waveform ' + str(index) + ' done')
#                index += 1
#                gc.collect()





#    def p_wave_mark(self):
#        [channel_Z, channel_N, channel_E] = self.xyz_array()
#        channels = {'Z':channel_Z, 'N':channel_N, 'E': channel_E}
#        if(channel_Z != None):
#            year = channel_Z.starttime.year
#            month = channel_Z.starttime.month
#            day = channel_Z.starttime.day
#            if int(self.P_Wave_original['HR']) < int(channel_Z.starttime.hour):
#                day += 1
#            self.P_Wave = int((dt.UTCDateTime(year,month,day,
#                  int(self.P_Wave_original['HR']),int(self.P_Wave_original['MM']),float(self.P_Wave_original['SECON'])) - 
#                  channel_Z.starttime)*channel_Z.original_sampling_rate)
#            new_df = channel_Z.sampling_rate
#            original_df = channel_Z.original_sampling_rate
#            if(original_df != new_df):
#                self.P_Wave = round((new_df/original_df)*self.P_Wave)
#            
#            dif_ZN = int((channel_Z.starttime - channel_N.starttime)*channel_Z.sampling_rate)
#            dif_ZE = int((channel_Z.starttime - channel_E.starttime)*channel_Z.sampling_rate)
#            
#            if(dif_ZN != 0):
#                if(dif_ZN > 0):
#                    channel_N.filter_wave = channel_N.filter_wave[dif_ZN:]
#                else:
#                    channel_N.filter_wave = [*np.zeros(abs(dif_ZN)), *channel_N.filter_wave]
#            
#            if(dif_ZE != 0):
#                if(dif_ZE > 0):
#                    channel_E.filter_wave = channel_E.filter_wave[dif_ZE:]
#                else:
#                    channel_E.filter_wave = [*np.zeros(abs(dif_ZE)), *channel_E.filter_wave]
#            
#            npts_min = min([len(channel_Z.filter_wave), len(channel_N.filter_wave), len(channel_E.filter_wave)])
##            print('Z: ' + str(len(channel_Z.filter_wave)) + ', N: ' + str(len(channel_N.filter_wave)) + 
##                  ', E: ' + str(len(channel_E.filter_wave)) + ' - Min: ' + str(npts_min))
#            channel_Z.filter_wave = channel_Z.filter_wave[0:npts_min]
#            channel_N.filter_wave = channel_N.filter_wave[0:npts_min]
#            channel_E.filter_wave = channel_E.filter_wave[0:npts_min]
#            return True
#        else:
#            return False