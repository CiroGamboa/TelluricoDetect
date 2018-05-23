#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:19:41 2018

@author: CiroGamJr
"""
###############################################################################
################################## Testing ####################################
###############################################################################

from Sfile import Sfile
## Read one file and print its attributes
#filename = "10-2055-44L.S201503.txt"
filename = "05-1631-49L.S201109.txt"
path = ""

sfile = Sfile(filename = filename, path = path)
sfile.get_attributes()
sfile.print_attributes()


###############################################################################
## Read many files
import os

sfiles = []
path = "/RSNC_Sfiles"
for root, dirs, files in os.walk("."):
    for filename in files:
        if '.S' in filename:
            sfile = Sfile(filename = filename, path = path)
            sfiles.append(sfile)
        #print(filename)
        #else:
            #print(filename)

print("Files found: ")
print(len(sfiles))
        
        
###############################################################################
# Read file and split elements by space
# Import the libraries
import re

# Import the file
filepath = ""
filename = "10-2055-44L.S201503.txt"
lines = []
with open(filepath+filename) as sfile:
    for cnt, line in enumerate(sfile):          
        # Split by line break and space
        lines.append(re.split(' +',line.strip()))
        


        