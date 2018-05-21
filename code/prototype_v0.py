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
from tools import TelluricoTools
from obspy.signal.polarization import eigval

# Read seismograms
st = read('IOfiles/2013_06_2013-06-18-0559-59M.COL___261')


# Get a non-zero trace
test_trace = None
index = 0
for trace in st:
    if(TelluricoTools.check_trace(trace)):
        test_trace = [trace,st[index+1], st[index+2]]
        break
    index = index + 1
    
    

# It is necessary to make a table describing the data convention
# PREGUNTA PARA EDWAR, CUALES SERIAN LOS MEJORES PARAMETROS PARA CALCULAR
# EL DOP DE LA SEÑAL.... VER REFERENCIA DE OBSPY: https://docs.obspy.org/packages/autogen/obspy.signal.polarization.eigval.html#obspy.signal.polarization.eigval

DOP = eigval(datax = test_trace[0],
             datay = test_trace[1],
             dataz = test_trace[2],
             fk = [1, 1, 1, 1, 1],
             normf = 1.0)
    





