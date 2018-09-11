#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:30:41 2018

@author: administrador
"""
import math
import matplotlib.pyplot as ml
import obspy.core.utcdatetime as dt
import numpy as np

class TraceGroup:
    
    def __init__(self, codeStation):
        self.codeStation = codeStation
        self.P_Wave_original = {}
        self.P_Wave = 0 # in samples
        self.S_Wave = 0 # in samples
        self.alert_time = 0.0 # in seconds
        self.epicentral_dist = -1.0
        self.traces = []
    
    # Add a new trace to the trace group
    def addTrace(self, traceComponent):
        self.traces.append(traceComponent)
        
    # Stablish the P reference of the P wave in the corresponding channel
    def p_wave_mark(self):
        [channel_Z, channel_N, channel_E] = self.xyz_array()
        channels = {'Z':channel_Z, 'N':channel_N, 'E': channel_E}
        channel_ref = channels.pop(self.P_Wave_original['P'])
        if(channel_ref != None):
            year = channel_ref.starttime.year
            month = channel_ref.starttime.month
            day = channel_ref.starttime.day
            if int(self.P_Wave_original['HR']) < int(channel_ref.starttime.hour):
                day += 1
            self.P_Wave = int((dt.UTCDateTime(year,month,day,
                  int(self.P_Wave_original['HR']),int(self.P_Wave_original['MM']),float(self.P_Wave_original['SECON'])) - 
                  channel_ref.starttime)*channel_ref.original_sampling_rate)
            new_df = channel_ref.sampling_rate
            original_df = channel_ref.original_sampling_rate
            if(original_df != new_df):
                self.P_Wave = round((new_df/original_df)*self.P_Wave)
            
            self.p_wave_reference(channel_ref, channels)
            
            return True
        else:
            return False
    
    # Mark the P wave according to the registered channel and synchronize all the channels to the P-wave mark
    def p_wave_reference(self, channel_ref, dictionary):
        channel_1 = dictionary[list(dictionary)[0]]
        channel_2 = dictionary[list(dictionary)[1]]
        dif_Refvs1 = int((channel_ref.starttime - channel_1.starttime)*channel_ref.sampling_rate)
        dif_Refvs2 = int((channel_ref.starttime - channel_2.starttime)*channel_ref.sampling_rate)                       
        
        if(dif_Refvs1 != 0):
            if(dif_Refvs1 > 0):
                channel_1.filter_wave = channel_1.filter_wave[dif_Refvs1:]
            else:
                channel_1.filter_wave = [*np.zeros(abs(dif_Refvs1)), *channel_1.filter_wave]
        
        if(dif_Refvs2 != 0):
            if(dif_Refvs2 > 0):
                channel_2.filter_wave = channel_2.filter_wave[dif_Refvs2:]
            else:
                channel_2.filter_wave = [*np.zeros(abs(dif_Refvs2)), *channel_2.filter_wave]
        
        npts_min = min([len(channel_ref.filter_wave), len(channel_1.filter_wave), len(channel_2.filter_wave)])
        
        channel_ref.filter_wave = channel_ref.filter_wave[0:npts_min]
        channel_1.filter_wave = channel_1.filter_wave[0:npts_min]
        channel_2.filter_wave = channel_2.filter_wave[0:npts_min]
        
        return channel_ref, channel_1, channel_2
        
    # Separate Z, N and E components
    def xyz_array(self):
        dataX = None; dataY = None; dataZ = None
        for trace in self.traces:
            if(trace.channel[2] == 'Z'):
                dataX = trace
            elif(trace.channel[2] == 'N' or trace.channel[2] == '1'):
                dataY = trace
            elif(trace.channel[2] == 'E' or trace.channel[2] == '2'):
                dataZ = trace
        if(dataX == None or dataY == None or dataZ == None):
            return [None, None, None]
        else:
            return [dataX, dataY, dataZ]
    
    # Get the resultant trace from all components
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
    
    # Graph the original components of each waveform
    def graphOriginalComponents(self, stat):
        comps = len(self.traces)
        components = str(comps) + '11'
        fig = ml.figure()
        ax1 = fig.add_subplot(int(components))
        ax1.set_title('Channel: ' + self.traces[0].channel + ' - stat: ' + stat, fontsize=16)
        ax1.plot(self.traces[0].waveform)
        for i in range(1, comps):
            ax = fig.add_subplot(int(str(comps) + '1' + str(i + 1)), sharex=ax1)
            ax.set_title('Channel: ' + self.traces[i].channel, fontsize=16)
            ax.plot(self.traces[i].waveform)
    
    # Graph the filtered and normaliced waveforms
    def graphComponents(self):
        comps = len(self.traces)
        components = str(comps) + '11'
        fig = ml.figure()
        ax1 = fig.add_subplot(int(components))
        ax1.set_title('Channel: ' + self.traces[0].channel, fontsize=16)
        ax1.plot(self.traces[0].filter_wave)
        for i in range(1, comps):
            ax = fig.add_subplot(int(str(comps) + '1' + str(i + 1)), sharex=ax1)
            ax.set_title('Channel: ' + self.traces[i].channel, fontsize=16)
            ax.plot(self.traces[i].filter_wave)
        