#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name:
    RSNC_data_adquisition_analysis.py
    
Description:
    Code for automatically download and analize 
    seisms information from the RSNC webpage
    
Dependencies:
    Obspy
    
Related tutorials:
    http://stackabuse.com/read-a-file-line-by-line-in-python/
    
Author:
    Tellurico team
    
Date:
    February, 2018

"""

# Import the libraries
import re

# Import the file
filepath = ""
filename = "10-2055-44L.S201503.txt"

# Read the file
lines = []
with open(filepath+filename) as sfile:
    for cnt, line in enumerate(sfile):          
        # Split by line break and space
        lines.append(re.split(' +',line.strip()))
        
        
        
        
        
        
    





