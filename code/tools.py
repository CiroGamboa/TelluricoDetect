# -*- coding: utf-8 -*-
"""
This is gonna be a class with handy methods for general use
"""

'''
Class dedicated to provide helpful methos for overall use in Tellurico
'''
#%matplotlib qt
#from obspy import read
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml
#from TraceComponent import TraceComponent
import math
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger
#from scipy.signal import lfilter 
from scipy.signal import resample
from scipy.signal import hilbert
from scipy.signal import periodogram
#import pandas as pd
import scipy as sc
import operator
from scipy.stats import pearsonr
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import hmean
from scipy.stats import gmean
import random
import obspy.signal.filter as filt
#from scipy.special import entr
import univariate as univariate
from sklearn.metrics.cluster import homogeneity_score
#from pylab import *
#from spectrum import *
from nitime.algorithms.autoregressive import AR_est_LD
import nolds

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
    
    #Extract a window from trace
    def sub_trace(trace, inf_limit, sup_limit):
        output = [trace[x] for x in range(inf_limit, sup_limit)]
        return output
    
    #Resample the trace to specific samples
    def resample(trace, actual_df, new_df):
        total_samples = int(len(trace)*(new_df/actual_df))
        return resample(trace, total_samples, t=None, axis=0, window=None)
    
    # Dictionary sorting by value / Timsort algorithm
    def sort(dictionary):
        sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
        return sorted_dict # (sorted_dict[0])[1] how to access a value in a tuple
    
    #Extract Z, N and E components from a group of traces
    def xyz_array(traceGroup):
        for trace in traceGroup.traces:
            if(trace.channel[2] == 'Z'):
                dataX = trace
            elif(trace.channel[2] == 'N'):
                dataY = trace
            elif(trace.channel[2] == 'E'):
                dataZ = trace
        return [dataX, dataY, dataZ]
    
     # Get the resultant trace from all components
    def getResultantTrace(dataX, dataY, dataZ):
        result = (((np.asarray(dataX)**2)+(np.asarray(dataY)**2)+(np.asarray(dataZ)**2))**(1/2) + 1)
        return result
    
    def getResultantTraceNorm(dataX, dataY, dataZ):
        result = (((np.asarray(dataX)**2)+(np.asarray(dataY)**2)+(np.asarray(dataZ)**2))**(1/2) + 1)
        return result/max(result)
    
    #Extract a P-wave window and a noisy window // Per is the Left side of P Wave
    def p_noise_extraction(trace, window_size, P_Wave, per):
        inf_limit = random.randint(int(2*window_size), P_Wave-int(2*window_size))
        init = int(per*window_size)
        return TelluricoTools.sub_trace(trace,P_Wave-init,P_Wave+(window_size-init)), TelluricoTools.sub_trace(trace,inf_limit,inf_limit+window_size)
    
    #Extract a P-wave to S-wave window and a noisy window
    def p2s_noise_extraction(trace, P_Wave, S_Wave):
        signal = TelluricoTools.sub_trace(trace,P_Wave,S_Wave)
        inf_limit = random.randint(len(signal), P_Wave-len(signal))
        noise = TelluricoTools.sub_trace(trace,inf_limit,inf_limit+len(signal))
        return signal, noise

class Transformations:
    
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
    
    #Logarithmic Spectrum - Welch Periodogram
    def welch_periodogram_log(trace, fs):
        f, Pxx_den = periodogram(trace, fs)
        ml.semilogy(f, Pxx_den)
#        ml.ylim([1e-7, 1e7])
        ml.xlabel('frequency [Hz]')
        ml.ylabel('PSD [dB]')
        ml.show()
        ml.grid()
    
    #Linear Spectrum - Welch Periodogram
    def welch_periodogram_linear(trace, fs):
        f, Pxx_den = periodogram(trace, fs)
        ml.semilogy(f, np.sqrt(Pxx_den))
        ml.xlabel('frequency [Hz]')
        ml.ylabel('Linear spectrum [RMS]')
        ml.show()
        ml.grid()
    
    #Difference between to signals linear power spectrum - Welch Periodogram
    def welch_periodogram_linear_diff(t1, t2, fs):
        f, Pxx_den_1 = periodogram(t1, fs)
        f, Pxx_den_2 = periodogram(t2, fs)
        ml.semilogy(f, np.sqrt(Pxx_den_1-Pxx_den_2))
        ml.xlabel('frequency [Hz]')
        ml.ylabel('Linear spectrum [RMS]')
        ml.show()
        ml.grid()
        
    #Linear Spectrum - Periodogram
    def periodogram_linear(trace, fs):
        f, Pxx_den = periodogram(trace, fs)
        return np.sqrt(Pxx_den)
        
    # Computes the Petrosian Fractal Dimension
    def petrosian_fd(trace):
        return univariate.pfd(trace)

class TimeDomain_Attributes:
    
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
        
    # Signal Geometric Mean
    def signal_gmean(trace):
        trace += 2*abs(np.min(trace))
        trace = [int(i) for i in trace]
        return gmean(trace)
    
    # Signal Harmonic Mean
    def signal_hmean(trace):
        trace += 2*abs(np.min(trace))
        trace = [int(i) for i in trace]
        return hmean(trace)
        
    # Signal Kurtosis
    def signal_kurtosis(trace):
        return kurtosis(trace)
    
    # Signal Skewness
    def signal_skew(trace):
        return skew(trace, axis=0, bias=True)
    
    # Signal Standard Deviation
    def signal_std(trace):
        return np.std(trace)
        
#class FreqDomain_Attributes:
    
     

class NonLinear_Attributes:
    
    # Signal entropy calculation (Shannon Entropy)
    def signal_entropy(trace):
        trace += 2*abs(np.min(trace))
        trace = [int(i) for i in trace]
        y = np.bincount(trace)/len(trace) # Probabilities
        ii = np.nonzero(y)[0]
        out = np.vstack((ii, y[ii])).T
        entropy=sc.stats.entropy(out)
        return entropy
    
     # Computes the Hjorth parameters
    def hjorth_params(trace):
        return univariate.hjorth(trace)
    
    # Autoregresive coefficients
    def AR_coeff(trace):
        if(type(trace) == list):
            trace = np.asarray(trace)
        return AR_est_LD(trace, order=100, rxx=None)[0]
    
    # Largest Lyapunov Exponent according to Eckmann et al.
    # https://pypi.org/project/nolds/
    def lyapunov_exp_max(trace):
        return nolds.lyap_r(trace)
    
    # Lyapunov Exponent according to Rosenstein et al.
    # https://pypi.org/project/nolds/
    def lyapunov_exp(trace):
        return nolds.lyap_e(trace)
    
    # Hurst Exponent
    # https://pypi.org/project/nolds/
    def hurst_exp(trace):
        return nolds.hurst_rs(trace)
    
    # Correlation Dimension according to Grassberger-Procaccia algorithm 
    # https://pypi.org/project/nolds/
    def corr_CD(trace, dim):
        return nolds.corr_dim(trace, dim)
        

class Correlation:
    
    #Correlation between signals
    def cross_correlation(t1, t2):
        if(len(t1) == len(t2)):
            return sc.correlate(t1, t2, mode='same') / len(t1)
        else:
            return 0
    
    # Pearsons correlation coefficient
    def pearson_correlation(t1, t2):
        if(len(t1) == len(t2)):
            return pearsonr(t1, t2)
        else:
            return 2.0

class Filters:
    
    # Bandpass filter
    def bandpass_filter(trace, SR, low_freq, high_freq):
        return filt.bandpass(trace, low_freq, high_freq, SR, corners=4, zerophase=False)

class Metrics:
    
    def homogeneity(t1, t2):
        return homogeneity_score(t1, t2)

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

class Graph:
    
    # Graphing components
    def graphComponents(traces):
        comps = len(traces)
        components = str(comps) + '11'
        fig = ml.figure()
        ax1 = fig.add_subplot(int(components))
        ax1.plot(traces[0])
        for i in range(1, comps):
            ax = fig.add_subplot(int(str(comps) + '1' + str(i + 1)), sharex=ax1)
            ax.plot(traces[i])
    
    def graph_P_waves(traces, P_Wave, window_size, stat, per):
        [dataX, dataY, dataZ] = TelluricoTools.xyz_array(traces)
        [signalX, noiseX] = TelluricoTools.p_noise_extraction(dataX.filter_wave, window_size, P_Wave, per)
        [signalY, noiseY] = TelluricoTools.p_noise_extraction(dataY.filter_wave, window_size, P_Wave, per)
        [signalZ, noiseZ] = TelluricoTools.p_noise_extraction(dataZ.filter_wave, window_size, P_Wave, per)
        
        vline = int((1+per)*(window_size/2))
        fig = ml.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('Z Component - Station: ' + stat)
        ax1.plot(signalX)
        i, j = ax1.get_ylim()
        ax1.vlines(vline, i, j, color='r', lw=1)
        ax2.set_title('N Component - Station: ' + stat)
        ax2.plot(signalY)
        i, j = ax2.get_ylim()
        ax2.vlines(vline, i, j, color='r', lw=1)
        ax3.set_title('E Component  - Station: ' + stat)
        ax3.plot(signalZ)
        i, j = ax3.get_ylim()
        ax3.vlines(vline, i, j, color='r', lw=1)


