#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:49:11 2018

@author: administrador
"""

class Event:
    
    def __init__(self, sfile):
        self.sfile = sfile
        self.trace_groups = {}
    
    def addTraceGroup(self, traceGroup, station_name):
        self.trace_groups[station_name] = traceGroup
        
    def print_stations(self):
        for station in self.trace_groups:
            print(self.trace_groups[station].codeStation)