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

''' DATASET READING AND PRE-PROCESSING '''

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
import math
import obspy.core.utcdatetime as dt
import obspy.signal.filter as filt
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger
from obspy.imaging import spectrogram as spec

events = []

# Read seismograms
st = read('IOfiles/2013_06_2013-06-18-0559-59M.COL___261')
#st = read('IOfiles/2015-03-10-2049-48M.COL___284')
sfile = Sfile('18-0602-02L.S201306.txt', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#sfile.print_params()

#SeismicInfo.printMedata(st)

# Get a non-zero trace
test_trace = None
traces = []
for trace in st:
    if(TelluricoTools.check_trace(trace) and trace.stats.channel[1] != 'N'):
        traces.append(trace)

station_name = traces[0].stats.station
trace_group = TraceGroup(station_name)
newEvent = Event(None)
for trace in traces:
    if(trace.stats.station == station_name):
        trace_group.addTrace(TraceComponent(trace))
    else:
        newEvent.addTraceGroup(trace_group, station_name)
        station_name = trace.stats.station
        trace_group = TraceGroup(station_name)
        trace_group.addTrace(TraceComponent(trace))
newEvent.addTraceGroup(trace_group, station_name)
events.append(newEvent)
        
for station in sfile.type_7:
    if(station['STAT'] in events[0].trace_groups):
        events[0].trace_groups[station['STAT']].epicentral_dist = station['DIS']
        if(station['PHAS'] == 'P'):
            year = events[0].trace_groups[station['STAT']].traces[0].starttime.year
            month = events[0].trace_groups[station['STAT']].traces[0].starttime.month
            day = events[0].trace_groups[station['STAT']].traces[0].starttime.day
            if int(station['HR']) < int(events[0].trace_groups[station['STAT']].traces[0].starttime.hour):
                day += 1
            events[0].trace_groups[station['STAT']].P_Wave = int((dt.UTCDateTime(year,month,day,
                  int(station['HR']),int(station['MM']),float(station['SECON'])) - 
                  events[0].trace_groups[station['STAT']].traces[0].starttime)*
                  events[0].trace_groups[station['STAT']].traces[0].sampling_rate)
#            print("P-Wave: " + station['STAT'] + ": " + str(events[0].trace_groups[station['STAT']].P_Wave))
        if(station['PHAS'] == 'S'):
            year = events[0].trace_groups[station['STAT']].traces[0].starttime.year
            month = events[0].trace_groups[station['STAT']].traces[0].starttime.month
            day = events[0].trace_groups[station['STAT']].traces[0].starttime.day
            if int(station['HR']) < int(events[0].trace_groups[station['STAT']].traces[0].starttime.hour):
                day += 1
            events[0].trace_groups[station['STAT']].S_Wave = int((dt.UTCDateTime(year,month,day,
                  int(station['HR']),int(station['MM']),float(station['SECON'])) - 
                  events[0].trace_groups[station['STAT']].traces[0].starttime)*
                  events[0].trace_groups[station['STAT']].traces[0].sampling_rate)
#            print("S-Wave: " + station['STAT'] + ": " + str(events[0].trace_groups[station['STAT']].S_Wave))
    else:
        print(station['STAT'])

stats_delete = []
stats_sort = {}
for station_wave in events[0].trace_groups:
    if(events[0].trace_groups[station_wave].P_Wave == 0):
        stats_delete.append(station_wave)   
    elif(events[0].trace_groups[station_wave].S_Wave > 0):
        stats_sort[station_wave] = float(events[0].trace_groups[station_wave].epicentral_dist)
        events[0].trace_groups[station_wave].alert_time =  (events[0].trace_groups[station_wave].S_Wave -
            events[0].trace_groups[station_wave].P_Wave)/events[0].trace_groups[station_wave].traces[0].sampling_rate 
#    else:
#        ml.plot(events[0].trace_groups[station_wave].traces[0].filter_wave)
        
#        print("Alert time " + station_wave + ": " + str(events[0].trace_groups[station_wave].alert_time))                                         
for stat in stats_delete:
        events[0].trace_groups.pop(stat)

# Station classification by ammount of components
compClassif = []
comp1 = []; comp2 = []; comp3 = []; comp4 = []
compClassif.append(comp1); compClassif.append(comp2); compClassif.append(comp3); compClassif.append(comp4)
for group in events[0].trace_groups:
    compClassif[len(events[0].trace_groups[group].traces) - 1].append(group)
total = len(compClassif[0]) + len(compClassif[1]) + len(compClassif[2])

stats_sort = TelluricoTools.sort(stats_sort)

#print("Total: " + str(total))

#Butterworth-Bandpass Filter
#for station_wave in events[0].trace_groups:        
#    filter_signal = filt.bandpass(events[0].trace_groups[station_wave].traces[0].waveform, 
#                  1, 8, events[0].trace_groups[stat].traces[0].sampling_rate, corners=4, zerophase=False)
#    fig, ax = ml.subplots(2, 1)
#    ax[0].plot(events[0].trace_groups[station_wave].traces[0].waveform)
#    ax[1].plot(filter_signal)

#Plot filter 
#for station_wave in events[0].trace_groups:        
#    fig, ax = ml.subplots(2, 1)
#    ax[0].plot(events[0].trace_groups[station_wave].traces[0].waveform)
#    ax[1].plot(events[0].trace_groups[station_wave].traces[0].filter_wave)

''' DATASET ATRIBUTES '''

# Plot traces for an specific Event
for station in events[0].trace_groups:
#        ml.figure(0)
#        ml.plot(events[0].trace_groups[station].traces[0].waveform)
#        ml.figure(1)
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
for group in events[0].trace_groups:
    comp = len(events[0].trace_groups[group].traces)
    print(group + "\n")
    for i in range(0, comp):
        print("\t" + events[0].trace_groups[group].traces[i].channel + " - " + 
              str(events[0].trace_groups[group].traces[i].sampling_rate))
        
# Print accelerometers
index = 0
for group in events[0].trace_groups:
    comp = len(events[0].trace_groups[group].traces)
    for i in range(0, comp):
        if(events[0].trace_groups[group].traces[i].channel[1] == 'N'):
            print(group)
            index += 1
            break
print(index)

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
    if(stat in compClassif[2]):
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
stat = 'BRR'; 
SR = events[0].trace_groups[stat].traces[0].sampling_rate
samples = range(10, 5001)
DOPs = []
for i in range(10, 5001):
    sub = i
    init = int((2*60*SR)+(25.12*SR)) # BRR
#    init = int((2*60*SR)+(41.16*SR)) # CHI
#    init = int((3*60*SR)+(32.26*SR)) #BBAC
#    init = int((2*60*SR)+(53.30*SR)) #ARGC
#    init = int((2*60*SR)+(50.50*SR)) #COD
    inf_limit = init - sub
    sup_limit = init + sub
    dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)
    dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].filter_wave,inf_limit,sup_limit)
    dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].filter_wave,inf_limit,sup_limit)
    DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
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
slope = 50
stat = 'BRR'; 
window_size = 100 #window_size in samples
size = int(events[0].trace_groups[stat].traces[0].npts/slope)
DOPs = []
RV2Ts = []
for i in range(0, size-1):
    dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,(slope*i),((slope*i)+window_size-1))
    dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].filter_wave,(slope*i),((slope*i)+window_size-1))
    dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].filter_wave,(slope*i),((slope*i)+window_size-1))
    DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
    RV2Ts.append(Attributes.RV2T(dataX,dataY,dataZ))
#    print(str(slope*i) + " to " + str((slope*i)+window_size-1))
fig, ax = ml.subplots(2, 1)
ax[0].plot(DOPs)
ax[1].plot(RV2Ts)

#Attributes per window calculation for all stations
for stat in events[0].trace_groups:
    if(len(events[0].trace_groups[stat].traces) == 3):
        DOPs = []
        RV2Ts = []
        for i in range(0, size-1):
            window_size = 2*int(events[0].trace_groups[stat].traces[0].sampling_rate)
            slope = int(window_size/2)
            size = int(events[0].trace_groups[stat].traces[0].npts/slope)
            dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].filter_wave,(slope*i),((slope*i)+window_size-1))
            DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
            RV2Ts.append(Attributes.RV2T(dataX,dataY,dataZ))
            print(str(slope*i) + " " + str((slope*i)+window_size-1))
        fig, ax = ml.subplots(3, 1)
        ax[0].set_title('P_wave window - Station: ' + stat)
        ax[0].plot(DOPs)
        ax[0].set_title('DOP per window - Station: ' + stat)
        ax[0].plot(DOPs)
        ax[1].set_title('RV2T per window - Station: ' + stat)
        ax[1].plot(RV2Ts)

#Attributes per window calculation for all stations in order of epicentral distance
for tuple_sort in stats_sort:
    stat = (tuple_sort[0])
    if(len(events[0].trace_groups[stat].traces) == 3):
        DOPs = []
        RV2Ts = []
        for i in range(0, size-1):
            window_size = int(events[0].trace_groups[stat].traces[0].sampling_rate)
            slope = int(window_size/2)
            size = int(events[0].trace_groups[stat].traces[0].npts/slope)
            dataX = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,(slope*i),((slope*i)+window_size-1))
            dataY = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[1].filter_wave,(slope*i),((slope*i)+window_size-1))
            dataZ = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[2].filter_wave,(slope*i),((slope*i)+window_size-1))
            DOPs.append(Attributes.DOP(dataX,dataY,dataZ))
            RV2Ts.append(Attributes.RV2T(dataX,dataY,dataZ))
            print(str(slope*i) + " " + str((slope*i)+window_size-1))
        fig, ax = ml.subplots(2, 1)
        ax[0].set_title('DOP per window - Station: ' + stat)
        ax[0].plot(DOPs)
        ax[1].set_title('RV2T per window - Station: ' + stat)
        ax[1].plot(RV2Ts)
    
    

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
