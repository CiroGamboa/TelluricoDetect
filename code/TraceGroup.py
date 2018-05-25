#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:30:41 2018

@author: administrador
"""

class TraceGroup:
    
    def __init__(self, codeStation):
        self.codeStation = codeStation
        self.traces = []
    
    def addTrace(self, traceComponent):
        self.traces.append(traceComponent)