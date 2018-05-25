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
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml
from TraceComponent import TraceComponent
from TraceGroup import TraceGroup
from Event import Event

events = []

# Read seismograms
st = read('IOfiles/2013_06_2013-06-18-0559-59M.COL___261')

#SeismicInfo.printMedata(st)

# Get a non-zero trace
test_trace = None
traces = []
for trace in st:
    if(TelluricoTools.check_trace(trace)):
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
        
# Get al traces un array
#waveforms = []
for station in newEvent.trace_groups:
        ml.figure(0)
        ml.plot(newEvent.trace_groups[station].traces[0].waveform)
#        ml.figure(1)
#        ml.plot(TelluricoTools.FFT(TelluricoTools.toArray(TelluricoTools.normalice(trace))))

# Print sampling rate of traces
for trace in st:
    if(TelluricoTools.check_trace(trace)):
        print(trace.stats.sampling_rate)

# %matplotlib qt  
#ml.show()
# It is necessary to make a table describing the data convention
# PREGUNTA PARA EDWAR, CUALES SERIAN LOS MEJORES PARAMETROS PARA CALCULAR
# EL DOP DE LA SEÑAL.... VER REFERENCIA DE OBSPY: https://docs.obspy.org/packages/autogen/obspy.signal.polarization.eigval.html#obspy.signal.polarization.eigval

DOP = eigval(datax = test_trace[0],
             datay = test_trace[1],
             dataz = test_trace[2],
             fk = [1, 1, 1, 1, 1],
             normf = 1.0)

signal = TelluricoTools.toArray(test_trace[2])
cumulative_signal = TelluricoTools.cumulative(test_trace[2])
ml.plot(cumulative_signal)
ml.show()
ml.plot(signal)
ml.show()

for i in range(0, 3):
    cumulative_signal = TelluricoTools.cumulative(test_trace[i])
    ml.plot(cumulative_signal)
    
#st.spectrogram(wlen=60, log=True, title='BW.RJOB ' + str(st[0].stats.starttime))

TelluricoTools.FFT(TelluricoTools.toArray([(test_trace[1])[x] for x in range(0, 16000)]))