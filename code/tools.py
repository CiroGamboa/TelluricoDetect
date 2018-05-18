# -*- coding: utf-8 -*-
"""
This is gonna be a class with handy methods for general use
"""

'''
Class dedicated to provide helpful methos for overall use in Tellurico
'''
class TelluricoTools:
    
    # Remove duplicated objects in a list
    def remove_duplicates(values):
        output = []
        seen = set()
        for value in values:
            # If value has not been encountered yet,
            # ... add it to both list and set.
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output
    
    # Check if the trace contains information different from zero
    def check_trace(trace):
        for sample in trace:
            if sample != 0:
                return True
        return False


##### MAKE DE DESIGN OF THE TOOLS CLASS, CONTAINING USEFUL INFORMATION
    ### ABOUT HOW TO USE OBSPY AND OTHER TOOLS
'''
# Import the libraries
from obspy import read

# Read seismograms
st = read('IOfiles/2013_06_2013-06-18-0559-59M.COL___261')

# Get one trace
tr = st[20]

# Access Meta Data
print(tr.stats)

# Access Waveform Data
print(tr.data)

# Plot Complete Data
st.plot( )

# Plot single trace
tr.plot()

## Compare station info between Sfile and Waveform
# Retrieve stations in Waveform

# In this example there are 39 different stations in the Sfile
waveform_stations = []
for trace in st:
    waveform_stations.append(trace.stats.station)
    print(trace.stats.station)
    


            
'''




