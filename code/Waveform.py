#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:39:32 2018

@author: administrador
"""

# Import the libraries
from obspy import read
from tools import TelluricoTools
from TraceComponent import TraceComponent
from TraceGroup import TraceGroup
from Event import Event
import obspy.core.utcdatetime as dt

class Waveform:

    def __init__(self, waveform_path, waveform_filename, sfile):
        
        # COMPOROBAR SI LA RUTA Y EL ARCHIVO EXISTEN
        self.waveform_path = waveform_path
        self.waveform_filename = waveform_filename
        self.sfile = sfile
        
    def get_event(self):
        st = read(self.waveform_path + self.waveform_filename)
        traces = []
        for trace in st:
            if(TelluricoTools.check_trace(trace) and trace.stats.channel[1] != 'N'):
                traces.append(trace)
        
        newEvent = Event(None)
        for trace in traces:
            if(trace.stats.station not in newEvent.trace_groups):
                trace_group = TraceGroup(trace.stats.station)
                trace_group.addTrace(TraceComponent(trace))
                newEvent.addTraceGroup(trace_group, trace.stats.station)
            else:
                newEvent.trace_groups[trace.stats.station].addTrace(TraceComponent(trace))
                
        for station in self.sfile.type_7:
                if(station['STAT'] in newEvent.trace_groups):
                    newEvent.trace_groups[station['STAT']].epicentral_dist = station['DIS']
                    if(station['PHAS'] == 'P'):
                        newEvent.trace_groups[station['STAT']].P_Wave_original['P'] = ((station['SP'])[1])
                        newEvent.trace_groups[station['STAT']].P_Wave_original['HR'] = int(station['HR'])
                        newEvent.trace_groups[station['STAT']].P_Wave_original['MM'] = int(station['MM'])
                        newEvent.trace_groups[station['STAT']].P_Wave_original['SECON'] = float(station['SECON'])
                    if(station['PHAS'] == 'S'):
                        year = newEvent.trace_groups[station['STAT']].traces[0].starttime.year
                        month = newEvent.trace_groups[station['STAT']].traces[0].starttime.month
                        day = newEvent.trace_groups[station['STAT']].traces[0].starttime.day
                        if int(station['HR']) < int(newEvent.trace_groups[station['STAT']].traces[0].starttime.hour):
                            day += 1
                        newEvent.trace_groups[station['STAT']].S_Wave = int((dt.UTCDateTime(year,month,day,
                              int(station['HR']),int(station['MM']),float(station['SECON'])) - 
                              newEvent.trace_groups[station['STAT']].traces[0].starttime)*
                              newEvent.trace_groups[station['STAT']].traces[0].original_sampling_rate)          
                        new_df = newEvent.trace_groups[station['STAT']].traces[0].sampling_rate
                        original_df = newEvent.trace_groups[station['STAT']].traces[0].original_sampling_rate
                        if(original_df != new_df):
                            newEvent.trace_groups[station['STAT']].S_Wave = round((new_df/original_df)*newEvent.trace_groups[station['STAT']].S_Wave)
        
        stats_delete = []
        stats_sort = {}
        
        for station_wave in newEvent.trace_groups:
            if(len(newEvent.trace_groups[station_wave].traces) == 3 and bool(newEvent.trace_groups[station_wave].P_Wave_original)):
#                print(station_wave + '\n' + newEvent.trace_groups[station_wave].traces[0].channel + '\n' +
#                      newEvent.trace_groups[station_wave].traces[1].channel + '\n' +
#                      newEvent.trace_groups[station_wave].traces[2].channel + '\n')
                result = newEvent.trace_groups[station_wave].p_wave_mark()
                if(not result):
                    stats_delete.append(station_wave)
#                print(station_wave)
                if(newEvent.trace_groups[station_wave].epicentral_dist.strip() != ''):
                        stats_sort[station_wave] = float(newEvent.trace_groups[station_wave].epicentral_dist)
                if(newEvent.trace_groups[station_wave].S_Wave > 0):
                    newEvent.trace_groups[station_wave].alert_time =  (newEvent.trace_groups[station_wave].S_Wave -
                                newEvent.trace_groups[station_wave].P_Wave)/newEvent.trace_groups[station_wave].traces[0].sampling_rate
            else:
                stats_delete.append(station_wave)
                                              
        for stat in stats_delete:
                newEvent.trace_groups.pop(stat)
        
        stats_sort = TelluricoTools.sort(stats_sort)
        
        return newEvent, stats_sort