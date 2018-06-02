#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:22:41 2018

@author: administrador
"""

# Read seismograms
#st = read('IOfiles/2013_06_2013-06-18-0559-59M.COL___261')
#sfile = Sfile('18-0602-02L.S201306.txt', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#st = read('IOfiles/2015-03-10-2049-48M.COL___284')
#sfile = Sfile('10-2055-44L.S201503', '/home/administrador/Tellurico/TelluricoDetect/code/IOfiles/')
#sfile.print_params()

#SeismicInfo.printMedata(st)

# Get a non-zero trace
#traces = []
#for trace in st:
#    if(TelluricoTools.check_trace(trace) and trace.stats.channel[1] != 'N'):
#        traces.append(trace)
#
#station_name = traces[0].stats.station
#trace_group = TraceGroup(station_name)
#newEvent = Event(None)
#for trace in traces:
#    if(trace.stats.station == station_name):
#        trace_group.addTrace(TraceComponent(trace))
#    else:
#        newEvent.addTraceGroup(trace_group, station_name)
#        station_name = trace.stats.station
#        trace_group = TraceGroup(station_name)
#        trace_group.addTrace(TraceComponent(trace))
#newEvent.addTraceGroup(trace_group, station_name)
#events.append(newEvent)
#        
#for station in sfile.type_7:
#        if(station['STAT'] in events[0].trace_groups):
#            events[0].trace_groups[station['STAT']].epicentral_dist = station['DIS']
#            if(station['PHAS'] == 'P'):
#                year = events[0].trace_groups[station['STAT']].traces[0].starttime.year
#                month = events[0].trace_groups[station['STAT']].traces[0].starttime.month
#                day = events[0].trace_groups[station['STAT']].traces[0].starttime.day
#                if int(station['HR']) < int(events[0].trace_groups[station['STAT']].traces[0].starttime.hour):
#                    day += 1
#                events[0].trace_groups[station['STAT']].P_Wave = int((dt.UTCDateTime(year,month,day,
#                      int(station['HR']),int(station['MM']),float(station['SECON'])) - 
#                      events[0].trace_groups[station['STAT']].traces[0].starttime)*
#                      events[0].trace_groups[station['STAT']].traces[0].original_sampling_rate)
#                print("P-Wave: " + station['STAT'] + ": " + str(events[0].trace_groups[station['STAT']].P_Wave))
#                new_df = events[0].trace_groups[station['STAT']].traces[0].sampling_rate
#                original_df = events[0].trace_groups[station['STAT']].traces[0].original_sampling_rate
#                if(original_df != new_df):
#                    events[0].trace_groups[station['STAT']].P_Wave = round((new_df/original_df)*events[0].trace_groups[station['STAT']].P_Wave)
##                print("P-Wave: " + station['STAT'] + ": " + str(events[0].trace_groups[station['STAT']].P_Wave))
#    #            print(dt.UTCDateTime(year,month,day,int(station['HR']),int(station['MM']),float(station['SECON'])))
#            if(station['PHAS'] == 'S'):
#                year = events[0].trace_groups[station['STAT']].traces[0].starttime.year
#                month = events[0].trace_groups[station['STAT']].traces[0].starttime.month
#                day = events[0].trace_groups[station['STAT']].traces[0].starttime.day
#                if int(station['HR']) < int(events[0].trace_groups[station['STAT']].traces[0].starttime.hour):
#                    day += 1
#                events[0].trace_groups[station['STAT']].S_Wave = int((dt.UTCDateTime(year,month,day,
#                      int(station['HR']),int(station['MM']),float(station['SECON'])) - 
#                      events[0].trace_groups[station['STAT']].traces[0].starttime)*
#                      events[0].trace_groups[station['STAT']].traces[0].original_sampling_rate)
#                print("S-Wave: " + station['STAT'] + ": " + str(events[0].trace_groups[station['STAT']].S_Wave))            
#                new_df = events[0].trace_groups[station['STAT']].traces[0].sampling_rate
#                original_df = events[0].trace_groups[station['STAT']].traces[0].original_sampling_rate
#                if(original_df != new_df):
#                    events[0].trace_groups[station['STAT']].S_Wave = round((new_df/original_df)*events[0].trace_groups[station['STAT']].S_Wave)
##                print("S-Wave: " + station['STAT'] + ": " + str(events[0].trace_groups[station['STAT']].S_Wave))
#        else:
#            print(station['STAT'])
#
#stats_delete = []
#stats_sort = {}
#for station_wave in events[0].trace_groups:
#    if(events[0].trace_groups[station_wave].P_Wave == 0):
#        stats_delete.append(station_wave)   
#    elif(events[0].trace_groups[station_wave].S_Wave > 0):
#        stats_sort[station_wave] = float(events[0].trace_groups[station_wave].epicentral_dist)
#        events[0].trace_groups[station_wave].alert_time =  (events[0].trace_groups[station_wave].S_Wave -
#            events[0].trace_groups[station_wave].P_Wave)/events[0].trace_groups[station_wave].traces[0].sampling_rate 
#    if(len(events[0].trace_groups[station_wave].traces) != 3 and station_wave not in stats_delete):
#        stats_delete.append(station_wave)
##    else:
##        ml.plot(events[0].trace_groups[station_wave].traces[0].filter_wave)
#        
##        print("Alert time " + station_wave + ": " + str(events[0].trace_groups[station_wave].alert_time))                                         
#for stat in stats_delete:
#        events[0].trace_groups.pop(stat)
#
# Station classification by ammount of components
#compClassif = []
#comp1 = []; comp2 = []; comp3 = []; comp4 = []
#compClassif.append(comp1); compClassif.append(comp2); compClassif.append(comp3); compClassif.append(comp4)
#for group in events[0].trace_groups:
#    compClassif[len(events[0].trace_groups[group].traces) - 1].append(group)
#total = len(compClassif[0]) + len(compClassif[1]) + len(compClassif[2])
#
#stats_sort = TelluricoTools.sort(stats_sort)
#
##print("Total: " + str(total))
#
##Butterworth-Bandpass Filter
##for station_wave in events[0].trace_groups:        
##    filter_signal = filt.bandpass(events[0].trace_groups[station_wave].traces[0].waveform, 
##                  1, 8, events[0].trace_groups[stat].traces[0].sampling_rate, corners=4, zerophase=False)
##    fig, ax = ml.subplots(2, 1)
##    ax[0].plot(events[0].trace_groups[station_wave].traces[0].waveform)
##    ax[1].plot(filter_signal)
#
##Plot filter 
##for station_wave in events[0].trace_groups:        
##    fig, ax = ml.subplots(2, 1)
##    ax[0].plot(events[0].trace_groups[station_wave].traces[0].waveform)
##    ax[1].plot(events[0].trace_groups[station_wave].traces[0].filter_wave)