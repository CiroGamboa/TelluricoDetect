#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:30:41 2018

@author: administrador
"""
import math
import matplotlib.pyplot as ml

class TraceGroup:
    
    def __init__(self, codeStation):
        self.codeStation = codeStation
        self.P_Wave = 0 # in samples
        self.S_Wave = 0 # in samples
        self.alert_time = 0.0 # in seconds
        self.epicentral_dist = -1.0
        self.traces = []
    
    # Add a new trace to the trace group
    def addTrace(self, traceComponent):
        self.traces.append(traceComponent)
    
    # get the resultant trace from all components
    def getResultantTrace(self):
        resultantTrace = []
        components = len(self.traces)
        samples = self.traces[0].npts
        result = 0.0
        for i in range(0, samples):
            for ii in range(0, components):
                result += math.pow((self.traces[ii].waveform[i]),2)
            result = math.sqrt(result)
            resultantTrace.append(result)
        return resultantTrace
    
#    def graphComponents(self):
#        components = len(self.traces)
#        fig, ax = ml.subplots(components, 1)
#        for i in range(0, components):
#            ax[i].set_title('Channel: ' + self.traces[i].channel, fontsize=16)
#            ax[i].plot(self.traces[i].waveform)
        
    def graphComponents(self):
        comps = len(self.traces)
        components = str(comps) + '11'
        fig = ml.figure()
        ax1 = fig.add_subplot(int(components))
        ax1.set_title('Channel: ' + self.traces[0].channel, fontsize=16)
        ax1.plot(self.traces[0].waveform)
        for i in range(1, comps):
            ax = fig.add_subplot(int(str(comps) + '1' + str(i + 1)), sharex=ax1)
            ax.set_title('Channel: ' + self.traces[i].channel, fontsize=16)
            ax.plot(self.traces[i].waveform)
        