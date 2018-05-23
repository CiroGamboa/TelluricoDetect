#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:57:45 2018

CREAR PLANTILLAS PARA CODIGO!

@author: CiroGamJr
"""

# Importar las librerias
from obspy import read
#import matplotlib.pyplot as plt


# Leer los sismogramas
st = read('2015_03_2015-03-10-2049-48M.COL___284')

# Cada archivo esta compuesto por muchos sismogramas correspondientes a
# cada eje de cada estacion. Todos los archivos de onda, vienen con metadata

# Adquirir un registro
registro = st[1]

# Mostrar metadata del registro
print(registro.stats)