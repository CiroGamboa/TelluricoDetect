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
from TraceClass import TraceClass

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

    # 
    def cumulative(trace):
        output = []
        acumula = 0.0
        for sample in trace:
            acumula += (sample*sample)
            output.append(acumula)
        return output
    
    def toArray(trace):
        output = []
        for sample in trace:
            output.append(sample)
        return output
    
    def normalice(trace):
        output = []
        mean= np.mean(trace)
        for sample in trace:
            output.append(sample - mean)
        return output
    
    def FFT(trace):
        Fs = trace.stats.sampling_rate;  # sampling rate
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
        ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('|Y(freq)|')

class SeismicInfo:
    
    # Print metadata al traces
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




