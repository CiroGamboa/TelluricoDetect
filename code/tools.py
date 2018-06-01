# -*- coding: utf-8 -*-
"""
This is gonna be a class with handy methods for general use
"""

'''
Class dedicated to provide helpful methos for overall use in Tellurico
'''

from obspy import read
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml
from TraceComponent import TraceComponent
import math
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger
from scipy.signal import hilbert
import pandas as pd
import scipy as sc
import operator

class TelluricoTools:
    
    # Remove duplicated objects in a list
    def remove_duplicates(values):
        output = []
        seen = set()
        for value in values:
            # If value has not been encountered yet,
            # ... add it to both list and set.
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output
    
    # Check if the trace contains information different from zero
    def check_trace(trace):
        for sample in trace:
            if sample != 0:
                return True
        return False

    # Trace to array
    def toIntArray(trace):
        output = []
        for sample in trace:
            output.append(int(sample))
        return output
    
    def sub_trace(trace, inf_limit, sup_limit):
        output = [trace[x] for x in range(inf_limit, sup_limit)]
        return output
    
    # Fast Fourier Transform
    def FFT(trace, SR, station):
        Fs = SR;  # sampling rate
        Ts = 1.0/Fs; # sampling interval
        t = np.arange(0,(len(trace)/Fs),Ts) # time vector
        n = len(trace) # length of the signal
        k = np.arange(n)
        T = n/Fs
        frq = k/T # two sides frequency range
        frq = frq[range(int(n/2))] # one side frequency range
        
        Y = np.fft.fft(trace)/n # fft computing and normalization
        Y = Y[range(int(n/2))]
        
        #%matplotlib qt
        fig, ax = ml.subplots(2, 1)
        ax[0].plot(t,trace)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ml.suptitle('Station: ' + station + ', SR: ' + str(SR), fontsize=20)
        ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('|Y(freq)|')
    
    # Dictionary sorting by value / Timsort algorithm
    def sort(dictionary):
        sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
        return sorted_dict # (sorted_dict[0])[1] how to access a value in a tuple
    
    def xyz_array(traceGroup):
        for trace in traceGroup.traces:
            if(trace.channel[2] == 'Z'):
                dataX = trace
            elif(trace.channel[2] == 'N'):
                dataY = trace
            elif(trace.channel[2] == 'E'):
                dataZ = trace
        return [dataX, dataY, dataZ]

class Attributes:
    
    # DOP Calculation
    def DOP(datax, datay, dataz):
        eigvalues = eigval(datax = datax, datay = datay, dataz = dataz,
             fk = [1, 1, 1, 1, 1], normf = 1.0)
        l1 = (eigvalues[0])[0]
        l2 = (eigvalues[1])[0]
        l3 = (eigvalues[2])[0]
        return (math.pow((l1-l2),2) + math.pow((l2-l3),2) + math.pow((l3-l1),2))/(2*math.pow((l1+l2+l3),2))
    
    # STA/LTA trigger function
    def STA_LTA(trace, SR):
        cft = classic_sta_lta(trace, int(5 * SR), int(10 * SR))
        plot_trigger(trace, cft, 1.5, 0.5)
    
    # RV2T Calculus
    def RV2T(dataX, dataY, dataZ):
        size = len(dataX)
        num = 0; den = 0
        for i in range(0, size):
            num += math.pow(dataX[i], 2)
            den += math.pow(dataX[i], 2) + math.pow(dataY[i], 2) + math.pow(dataZ[i], 2)
        return num/den
    
    # Signal envelope calculation
    def envelope(trace, fs):
        samples = len(trace)
        t = np.arange(samples) / fs
        
        analytic_signal = hilbert(trace)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*fs)
        
        fig = ml.figure()
        ax0 = fig.add_subplot(211)
        ax0.plot(t, trace, label='signal')
        ax0.plot(t, amplitude_envelope, label='envelope')
        ax0.set_xlabel("time in seconds")
        ax0.legend()
        ax1 = fig.add_subplot(212)
        ax1.plot(t[1:], instantaneous_frequency)
        ax1.set_xlabel("time in seconds")
        ax1.set_ylim(0.0, 120.0)
        
        fig = ml.figure()
        ml.plot(t, amplitude_envelope)
    
    # Signal entropy calculation
    def entropy(trace):
        y = np.bincount(trace)/len(trace)
        ii = np.nonzero(y)[0]
        out = np.vstack((ii, y[ii])).T
        entropy=sc.stats.entropy(out)  # input probabilities to get the entropy
        return entropy

class SeismicInfo:
    
    # Print metadata all traces
    def printMedata(st):
        for trace in st:
            if(TelluricoTools.check_trace(trace)):
                print(trace.stats)

##### MAKE DE DESIGN OF THE TOOLS CLASS, CONTAINING USEFUL INFORMATION
    ### ABOUT HOW TO USE OBSPY AND OTHER TOOLS
'''
# Import the libraries
from obspy import read

# Read seismograms
st = read('IOfiles/2013_06_2013-06-18-0559-59M.COL___261')

# Get one trace
tr = st[20]

# Access Meta Data
print(tr.stats)

# Access Waveform Data
print(tr.data)

# Plot Complete Data
st.plot( )

# Plot single trace
tr.plot()

## Compare station info between Sfile and Waveform
# Retrieve stations in Waveform

# In this example there are 39 different stations in the Sfile
waveform_stations = []
for trace in st:
    waveform_stations.append(trace.stats.station)
    print(trace.stats.station)
                
'''

class Temps:
#    def graphComponents(traces, inf_limit, sup_limit):
#        components = len(traces)
#        if(inf_limit == None):
#            inf_limit = 0
#        if(sup_limit == None):
#            sup_limit = len(traces[0].waveform) - 1
#        fig, ax = ml.subplots(components, 1)
#        for i in range(0, components):
#            ax[i].set_title('Channel: ' + traces[i].channel, fontsize=16)
#            ax[i].plot(TelluricoTools.sub_trace(traces[i].waveform,inf_limit,sup_limit))
    def graphComponents(traces):
        comps = len(traces)
        components = str(comps) + '11'
        fig = ml.figure()
        ax1 = fig.add_subplot(int(components))
        ax1.set_title('Channel: ' + traces[0].channel, fontsize=16)
        ax1.plot(traces[0].waveform)
        for i in range(1, comps):
            ax = fig.add_subplot(int(str(comps) + '1' + str(i + 1)), sharex=ax1)
            ax.set_title('Channel: ' + traces[i].channel, fontsize=16)
            ax.plot(traces[i].waveform)


