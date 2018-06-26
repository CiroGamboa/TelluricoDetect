#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:01:39 2018

@author: administrador
"""

from obspy import read
from obspy.signal.polarization import eigval
import numpy as np
import matplotlib.pyplot as ml
import math
import obspy.core.utcdatetime as dt
import obspy.signal.filter as filt
from scipy.signal import resample

class TraceComponent:
    
    # Filter, normalice and resample signals
    def __init__(self, trace):
        self.waveform = self.normalice(trace)
        self.network = trace.stats.network
        self.station = trace.stats.station
        self.location = trace.stats.location
        self.channel = trace.stats.channel + trace.stats.location
        self.starttime = dt.UTCDateTime(trace.stats.starttime)
        self.endtime = dt.UTCDateTime(trace.stats.endtime)
#        self.sampling_rate = trace.stats.sampling_rate
        self.sampling_rate = 100.0
        self.original_sampling_rate = trace.stats.sampling_rate
        self.delta = trace.stats.delta
        self.npts = trace.stats.npts
        self.calib = trace.stats.calib
        self.formatseed = trace.stats._format
        try:
            self.mseed = trace.stats.mseed
        except:
            pass
        self.filter_wave = self.resample_trace(self.bandpass_filter(self.waveform, 
                    self.sampling_rate), self.original_sampling_rate, 100.0)
    
    # Normalice the trace, deleting d.c. level
    def normalice(self, trace):
        output = []
        mean= np.mean(trace)
        for sample in trace:
            output.append(sample - mean)
        return output
    
    # Bandpass Butterworth filter
    def bandpass_filter(self, waveform, SR):
        return filt.bandpass(waveform, 1, 8, SR, corners=4, zerophase=False)
    
    # Resample the trace
    def resample_trace(self, trace, actual_df, new_df):
        if(actual_df != new_df):
            total_samples = int(len(trace)*(new_df/actual_df))
            return resample(trace, total_samples, t=None, axis=0, window=None)
        else:
            return trace
    
    # Check if the trace contains information different from zero
    def check_trace(self, trace):
        for sample in trace:
            if sample != 0:
                return True
        return False
    
    # Cumulative distribution of traces
    def cumulative(self):
        output = []
        acumula = 0.0
        for sample in self.filter_wave:
            acumula += math.pow(sample, 2)
            output.append(acumula)
        return output
    
    def sub_trace(self, inf_limit, sup_limit):
        output = [self.waveform[x] for x in range(inf_limit, sup_limit)]
        return output