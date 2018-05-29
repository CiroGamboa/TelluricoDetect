#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:25:33 2018

@author: administrador
"""
#%% Ejemplo de FFT con señal seno
from obspy import read
from tools import TelluricoTools
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml

## Ejemplo de FFT

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))]

%matplotlib qt
fig, ax = ml.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

#%% Ejemplo de DateTime
from obspy import read
from tools import TelluricoTools
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml
import datetime
import obspy.core.utcdatetime as dt
#print(datetime.datetime.isoformat(2008, 10, 1, 12, 30, 35, 45020))
date = dt.UTCDateTime("2008-10-01T12:30:35.045020Z")
print(date.microsecond)
#2008, 10, 1, 12, 30, 35, 45020
#print(dt.UTCDateTime("2008-10-02T00:01:55") - dt.UTCDateTime("2008-10-01T23:58:00"))
print(dt.UTCDateTime(2008,10,2,0,1,55.034) - dt.UTCDateTime("2008-10-01T23:58:00"))
#2011-01-21 02:37:21
#d=datetime.datetime.strptime("2013-06-18T06:00:00.000000", "YYYY-MM-DDTHH:MM:SS.mmmmmm") #Get your naive datetime object


#d=datetime.datetime.strptime("2013-06-18T06:00:00.000000", "%Y-%m-%dT%H:%M:%S") #Get your naive datetime object
#d=d.replace(tzinfo=datetime.timezone.utc) #Convert it to an aware datetime object in UTC time.
#d=d.astimezone() #Convert it to your local timezone (still aware)
#print(d.strftime("%d %b %Y (%I:%M:%S:%f %p) %Z")) #Print it with a directive of choice

#%% STA/LTA Algorithm
from obspy.signal.trigger import classic_sta_lta
from obspy.core import read
from obspy.signal.trigger import plot_trigger
trace = read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate
cft = classic_sta_lta(trace.data, int(5 * df), int(10 * df))
plot_trigger(trace, cft, 1.5, 0.5)
#%% Signal envelope
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*fs)

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)