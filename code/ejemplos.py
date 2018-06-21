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
plot_trigger(trace, df, len(trace), cft, 1.5, 0.5)
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
#%%Entropy Calculation
import numpy as np
#import pandas as pd
#import perfplot
from scipy.stats import itemfreq
x = [3.6, 2, 1, 1, 1, 1, 1, 1, 3, 3]
y = np.bincount(x)/len(x)
ii = np.nonzero(y)[0]
out = np.vstack((ii, y[ii])).T
entropy=sc.stats.entropy(out)  # input probabilities to get the entropy
print(entropy) 
#%% Sorting a dictionary by value
import operator
x = {'e': 2.23, 'as': 4.53, 'we': 3.87, 'fd': 1.23, 'brr': 0.71, 'BAR2': 3.26}
sorted_x = sorted(x.items(), key=operator.itemgetter(1))
print(sorted_x)
#%% Time elapsed
import time

start = time.time()
print("hello")
end = time.time()
print(str(end - start) + " seconds")
print("%.6f  seconds" % (end - start))
#%% Ejemplo de plot
fig = ml.figure()
ax1 = fig.add_subplot(211)
t = np.arange(0, 100)
y = np.arange(100, 200)
z = np.arange(300, 400)
ax1.plot(t, y, 'k')
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, z, 'k')
#%% Resample of signals
from scipy.signal import resample
from obspy import read
import matplotlib.pyplot as ml
st = read('IOfiles/2015-03-10-2049-48M.COL___284')
ml.plot(st[0].data)
actual_df = st[0].stats.sampling_rate
new_df = 100.0
total_samples = int(len(trace)*(new_df/actual_df))
sampled = resample(st[0].data, total_samples, t=None, axis=0, window=None)
ml.plot(sampled)
#%% Ejemplo de mapa
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

from obspy import read_inventory, read_events

# Set up a custom basemap, example is taken from basemap users' manual
fig, ax = plt.subplots()

# setup albers equal area conic basemap
# lat_1 is first standard parallel.
# lat_2 is second standard parallel.
# lon_0, lat_0 is central point.
m = Basemap(llcrnrlon = 3.75, llcrnrlat = 39.75, urcrnrlon = 4.35 , urcrnrlat = 40.15, resolution = 'f', epsg = 5520)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='wheat', lake_color='skyblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-80., 81., 20.))
m.drawmeridians(np.arange(-180., 181., 20.))
m.drawmapboundary(fill_color='skyblue')
ax.set_title("Albers Equal Area Projection")

# we need to attach the basemap object to the figure, so that obspy knows about
# it and reuses it
fig.bmap = m

# now let's plot some data on the custom basemap:
#inv = read_inventory()
#inv.plot(fig=fig, show=False)
#cat = read_events()
#cat.plot(fig=fig, show=False, title="", colorbar=False)

plt.show()
#%% Cross Correlation between signals
from scipy import signal
import matplotlib.pyplot as ml

sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = sig + np.random.randn(len(sig))
corr = signal.correlate(sig_noise, np.ones(128), mode='same')
#ml.plot(corr)
sig1 = np.zeros(1000); sig1[400:601] = 1
sig2 = np.zeros(1000); sig2[400:601] = np.arange(0, (1+1/200), 1/200)
corr = signal.correlate(sig1, sig1, mode='same')/len(sig1)
ml.plot(corr); ml.plot(sig1); ml.plot(sig2)

#%% Pearsons correlation between signals
from scipy.stats import pearsonr

sig1 = np.zeros(1000); sig1[400:601] = 1
sig2 = np.zeros(1000); sig2[400:601] = np.arange(0, (1+1/200), 1/200)
print(pearsonr(sig1, sig2)[0])
#%%Entropy calculation
import numpy as np
from scipy.special import entr

#x = np.arange(1000,2000,1)
x = np.ones(100)
p = x/np.sum(x)
print(np.sum(entr(p))/np.log(2))
#%%Pyrem library
import univariate as univariate
import numpy as np

noise = np.random.normal(size=int(1e4))
activity, complexity, morbidity = univariate.hjorth(noise)

#%% Welch Periodogram
from scipy.signal import periodogram

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
f, Pxx_den = signal.periodogram(x, fs)
ml.semilogy(f, Pxx_den)
ml.ylim([1e-7, 1e2])
ml.xlabel('frequency [Hz]')
ml.ylabel('PSD [V**2/Hz]')
ml.show()

#%% Homogenity (statistics)
from sklearn.metrics.cluster import homogeneity_score

homogeneity_score([0, 0, 2, 2], [1, 1, 0, 0])

#%%Multiprocessing tasks
from multiprocessing import Process

n = 100000

p1 = Process(target=factorial, args=(n,1))
p2 = Process(target=factorial, args=(n,2))
p3 = Process(target=factorial, args=(n,3))
p4 = Process(target=factorial, args=(n,4))

start = time.time()
p1.start()
p2.start()
p3.start()
p4.start()

p1.join()
p2.join()
p3.join()
p4.join()
end = time.time()

print('Time: ' + str(end-start) + ' seconds')

def factorial(n, p):
    result = 2
    for i in range(3, n + 1):
        result *= i
    print('Process :' + str(p) + ' done')

#%%Multiprocessing tasks
from multiprocessing import Process

n = 100000
cores = 4
p = [0, 0, 0, 0]

for i in range(0, cores):
    p[i] = Process(target=factorial, args=(n,i+1))
    p[i].start()

for i in range(0, cores):
    p[i].join()

def factorial(n, p):
    result = 2
    for i in range(3, n + 1):
        result *= i
    print('Process :' + str(p) + ' done')
    
#%%Yule Walker
from pylab import *
from spectrum import *

a = [1, -2.2137, 2.9403, -2.1697, 0.9606]
y = lfilter([1], a, randn(1, 1024))
ar, variance, coeff_reflection = aryule(y[0], 20)

ml.plot(coeff_reflection)


from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
plot_acf(series, lags=31)
pyplot.show()

#%% AR Coeff
#from pandas import Series
#from matplotlib import pyplot
#from pandas.plotting import autocorrelation_plot
#
#t = np.arange(1, 1000, 1)
#series = rand(1, 1000)
#autocorrelation_plot(t, series)
#pyplot.show()

from nitime.algorithms.autoregressive import AR_est_LD

Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
signal = np.sin(2 * np.pi * f * x / Fs)
signal = events[0].trace_groups['BRR'].traces[0].filter_wave

AR = AR_est_LD(signal, order=100, rxx=None)[0]
print(AR)
ml.plot(AR)
#%%3D Plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = 9*rand(1, n)+23
    ys = 100*rand(1, n)
    zs = (zhigh-zlow)*rand(1, 100)+zlow
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#%% Linearity test
from scipy import stats
import numpy as np

x = np.random.random(10)
y = np.random.random(10)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("r-squared: " + str(r_value**2))

#%% Core division
cores = 9
waveform_valid_l = 98
step = int(waveform_valid_l/cores)

for i in range(1, (cores+1)):
    print(str((i-1)*step) + " to " + str((i!=cores)*(i*step) + (i==cores)*(waveform_valid_l-1)))

#%% Drawing vlines to plot

a = rand(100)*100
line_in = 20

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(a)

i, j = ax1.get_ylim()
ax1.vlines(line_in, i, j, color='r', lw=1)

#%% Lyapunov Exponent
from math import log
import numpy as np

stat = 'BRR'
sub = 200
init = events[0].trace_groups[stat].P_Wave
inf_limit = init - sub
sup_limit = init + sub
data = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)

N = len(data)
eps = 0.001
lyapunovs = [[] for i in range(N)]

for i in range(N):
    for j in range(i + 1, N):
        if np.abs(data[i] - data[j]) < eps:
            for k in range(min(N - i, N - j)):
                lyapunovs[k].append(log(np.abs(data[i+k] - data[j+k])))
                
for i in range(len(lyapunovs)):
    if len(lyapunovs[i]):
        string = str((i, sum(lyapunovs[i]) / len(lyapunovs[i])))
        print(string)

#%% Largest Lyapunov Exponent according to Eckmann et al.
import nolds
import numpy as np

stat = 'BRR'
sub = 100
init = events[0].trace_groups[stat].P_Wave
inf_limit = init - sub
sup_limit = init + sub
data = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)

lexp = nolds.lyap_r(data)
print(lexp)

#%% Hurst Exponent
import nolds
import numpy as np

stat = 'BRR'
sub = 100
init = events[0].trace_groups[stat].P_Wave
inf_limit = init - sub
sup_limit = init + sub
data = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)

lexp = nolds.hurst_rs(data)
print(lexp)

#%% Correlation Dimension according to Grassberger-Procaccia algorithm
import nolds
import numpy as np

stat = 'BRR'
sub = 100
init = events[0].trace_groups[stat].P_Wave
inf_limit = init - sub
sup_limit = init + sub
data = TelluricoTools.sub_trace(events[0].trace_groups[stat].traces[0].filter_wave,inf_limit,sup_limit)

lexp = nolds.corr_dim(data, 1)
print(lexp)