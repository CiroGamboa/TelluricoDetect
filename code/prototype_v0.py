# -*- coding: utf-8 -*-
"""
Prototipo 0 de Tellurico
Es necesario establecer el patron de documentacion con sphinx
Plantillas y demas
Este prototipo tiene como objetivo arrancar el desarrollo
A partir del prototipo 1 y en adelante, vinculados con sprints
Se tendra todo documentado de forma estandar
Toda la documentacion debe ser en ingles
"""

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
    
    
# Remove duplicates in list
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

# Remove duplicates from this list.
waveform_stattions_filtered = remove_duplicates(waveform_stations)
print(waveform_stattions_filtered)





