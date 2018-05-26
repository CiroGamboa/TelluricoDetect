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

class TraceComponent:
        
    def __init__(self, trace):
        self.waveform = self.normalice(trace)
        self.network = trace.stats.network
        self.station = trace.stats.station
        self.location = trace.stats.location
        self.channel = trace.stats.channel
        self.starttime = trace.stats.starttime
        self.endtime = trace.stats.endtime
        self.sampling_rate = trace.stats.sampling_rate
        self.delta = trace.stats.delta
        self.npts = trace.stats.npts
        self.calib = trace.stats.calib
        self.formatseed = trace.stats._format
        self.mseed = trace.stats.mseed
    
    # Normalice the trace, deleting d.c. level
    def normalice(self, trace):
        output = []
        mean= np.mean(trace)
        for sample in trace:
            output.append(sample - mean)
        return output
    
    # Check if the trace contains information different from zero
    def check_trace(self, trace):
        for sample in trace:
            if sample != 0:
                return True
        return False
    
    # Cumulative distribution of traces
    def cumulative(self, trace):
        output = []
        acumula = 0.0
        for sample in trace:
            acumula += math.pow(sample, 2)
            output.append(acumula)
        return output
    
    def sub_trace(self, inf_limit, sup_limit):
        output = [self.waveform[x] for x in range(inf_limit, sup_limit)]
        return output