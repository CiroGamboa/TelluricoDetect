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


#%% MD2 - Lab 3
import numpy as np
import matplotlib.pyplot as ml
import random, math, pickle

# Easy
path = '/home/tellurico/Descargas/'
path_in = path + 'lab3_easy_in.txt'
path_out = path + 'lab3_easy_out.txt'

cases = 50
total = 0.0
str_in = ""
str_out = ""

str_in += str(cases) + "\n\n"
for i in range(0, cases):
    totalVertices = int(random.randint(3, 10))
    p1x = 0
    p1y = 0
    str_in += str(p1x) + " " + str(p1y) + "\n"
    for ii in range(0, totalVertices):
        p2x = int(random.randint(-100, 100))*100
        p2y = int(random.randint(-100, 100))*100
        while(p2x == 0 and p2y == 0):
            p2x = int(random.randint(-100, 100))*100
            p2y = int(random.randint(-100, 100))*100
        str_in += str(p1x) + " " + str(p1y) + " " + str(p2x) + " " + str(p2y) + "\n"
        total += math.sqrt(math.pow(p2x-p1x, 2) + math.pow(p2y-p1y, 2))
        p1x = p2x
        p1y = p2y
        
        with open(path_in, 'a') as the_file:
            the_file.write(str_in) 
        str_in = ""
        
    str_in += "\n"
    total *= 2
    total /= 20000
    str_out = str(int(total)) + ":" + str(round(60*(total-int(total)))) + "\n"
    total = 0

    with open(path_out, 'a') as the_file:
        the_file.write(str_out)
    
    str_out = ""

# Hard
path = '/home/tellurico/Descargas/'
path_in = path + 'lab3_hard_in.txt'
path_out = path + 'lab3_hard_out.txt'

cases = 5000
total = 0.0
str_in = ""
str_out = ""

str_in += str(cases) + "\n\n"
for i in range(0, cases):
    totalVertices = int(random.randint(10, 50))
    p1x = 0
    p1y = 0
    str_in += str(p1x) + " " + str(p1y) + "\n"
    for ii in range(0, totalVertices):
        p2x = int(random.randint(-100, 100))*100
        p2y = int(random.randint(-100, 100))*100
        while(p2x == 0 and p2y == 0):
            p2x = int(random.randint(-100, 100))*100
            p2y = int(random.randint(-100, 100))*100
        str_in += str(p1x) + " " + str(p1y) + " " + str(p2x) + " " + str(p2y) + "\n"
        total += math.sqrt(math.pow(p2x-p1x, 2) + math.pow(p2y-p1y, 2))
        p1x = p2x
        p1y = p2y
        
        with open(path_in, 'a') as the_file:
            the_file.write(str_in) 
        str_in = ""
        
    str_in += "\n"
    total *= 2
    total /= 20000
    str_out = str(int(total)) + ":" + str(round(60*(total-int(total)))) + "\n"
    total = 0

    with open(path_out, 'a') as the_file:
        the_file.write(str_out)
    
    str_out = ""

#%% MD1 - Lab 1
import numpy as np
import matplotlib.pyplot as ml
import random, math, pickle

# Easy
path = '/home/tellurico/Descargas/MD1/'
path_in = path + 'lab1_easy_in.txt'
path_out = path + 'lab1_easy_out.txt'

total = 0.0
str_in = ""
str_out = ""
fibo_values = []
up_to = 500

for n in range(0, up_to):
    fibo_values.append(int(((1+math.sqrt(5))**n-(1-math.sqrt(5))**n)/(2**n*math.sqrt(5))))

cont = 3
ulam_i = [1,2,3]
ulam_j = [1,2,3]
cand = 4
while(cont <= up_to):
    res = []
    for i in ulam_i:
        for j in ulam_j:
            if i == j or j > i: pass
            else:
                res.append(i+j)
    if res.count(cand) == 1:
        ulam_i.append(cand)
        ulam_j.append(cand)
        cont += 1
    cand += 1

# Easy
limit = 100
cases = 25
numbers = []
for i in range(0, cases):
    number = random.randint(1, limit)
    while(number in numbers):
        number = int(random.randint(1, limit))
    numbers.append(number)
    with open(path_in, 'a') as the_file_in:
        the_file_in.write(str(number) + "\n") 
    with open(path_out, 'a') as the_file_out:
        the_file_out.write(str(fibo_values[number] + ulam_i[number-1]) + "\n")
with open(path_in, 'a') as the_file_in:
    the_file_in.write(str(-1) + "\n")
        
path_in = path + 'lab1_hard_in.txt'
path_out = path + 'lab1_hard_out.txt'
        
# Hard
limit = 500
cases = 250
numbers = []
for i in range(0, cases):
    number = random.randint(1, limit)
    while(number in numbers):
        number = int(random.randint(1, limit))
    numbers.append(number)
    with open(path_in, 'a') as the_file_in:
        the_file_in.write(str(number) + "\n") 
    with open(path_out, 'a') as the_file_out:
        the_file_out.write(str(fibo_values[number] + ulam_i[number-1]) + "\n")
with open(path_in, 'a') as the_file_in:
    the_file_in.write(str(-1) + "\n")
    
#%% MD1 - Lab 2
import numpy as np
import matplotlib.pyplot as ml
import random, math, pickle
import sys, copy

# Easy
path = '/home/tellurico/Descargas/MD1/'
path_in = path + 'lab2_easy_in.txt'
path_out = path + 'lab2_easy_out.txt'
    
number_limit = 100
N_limit = 100
K_limit = 20
cases = 25
numbers = []
str_numbers = ""

with open(path_in, 'a') as the_file_in:
    the_file_in.write(str(cases) + "\n")

    for i in range(0, cases):
        N = random.randint(1, N_limit)
        K = random.randint(2, K_limit)
        str_NyK = str(N) + " " + str(K)
        the_file_in.write(str_NyK + "\n")
        
        for ii in range(0, N):
            number = random.randint(-number_limit, number_limit)
            str_numbers += str(number) + " "
            numbers.append(number)
        
        str_numbers = str_numbers[0:(len(str_numbers) - 1)]
        the_file_in.write(str_numbers + "\n")
        
        # Operation ------------------------------------
        n = len(numbers)

        rem = [[False]*K for _ in range(n)]
        rem[0][numbers[0]%K] = True
    
        for i in range(1, n):
            for j in range(K):
                if rem[i-1][j]:
                    rem[i][(j+numbers[i])%K] = True
                    rem[i][(j+K-numbers[i])%K] = True
        # End Operation ------------------------------------
        
        if rem[n-1][0]:
            with open(path_out, 'a') as the_file_out:
                the_file_out.write("Divisible\n")
        else:
            with open(path_out, 'a') as the_file_out:
                the_file_out.write("Not divisible\n")
        
        numbers = []
        str_numbers = ""

#Hard
        
path = '/home/tellurico/Descargas/MD1/'
path_in = path + 'lab2_hard_in.txt'
path_out = path + 'lab2_hard_out.txt'
        
number_limit = 10000
N_limit = 10000
K_limit = 100
cases = 200
numbers = []
str_numbers = ""

with open(path_in, 'a') as the_file_in:
    the_file_in.write(str(cases) + "\n")

    for i in range(0, cases):
        N = random.randint(1, N_limit)
        K = random.randint(2, K_limit)
        str_NyK = str(N) + " " + str(K)
        the_file_in.write(str_NyK + "\n")
        
        for ii in range(0, N):
            number = random.randint(-number_limit, number_limit)
            str_numbers += str(number) + " "
            numbers.append(number)
        
        str_numbers = str_numbers[0:(len(str_numbers) - 1)]
        the_file_in.write(str_numbers + "\n")
        
        # Operation ------------------------------------
        n = len(numbers)

        rem = [[False]*K for _ in range(n)]
        rem[0][numbers[0]%K] = True
    
        for i in range(1, n):
            for j in range(K):
                if rem[i-1][j]:
                    rem[i][(j+numbers[i])%K] = True
                    rem[i][(j+K-numbers[i])%K] = True
        # End Operation ------------------------------------
        
        if rem[n-1][0]:
            with open(path_out, 'a') as the_file_out:
                the_file_out.write("Divisible\n")
        else:
            with open(path_out, 'a') as the_file_out:
                the_file_out.write("Not divisible\n")
        
        numbers = []
        str_numbers = ""