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
    
    def graphComponents(self):
        components = len(self.traces)
        fig, ax = ml.subplots(components, 1)
        for i in range(0, components):
            ax[i].plot(self.traces[i].waveform)
        