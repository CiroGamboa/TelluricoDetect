#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:19:41 2018

EN ESTA CLASE SE DETERMINAN LOS PARAMETROS DE UN SISMO
ESTABLECIDOS EN SU ARCHIVO SFILE CORRESPONDIENTE

@author: CiroGamJr
"""
# MIRAR QUE HACER CUANDO HAY VARIAS LINEAS DEL MISMO TIPO

class Sfile:
    
    # Instance the object with the name and path of the sfile
    
    # SERIA BUENA IDEA QUE FUESE UN DICT, CON LA DESCRIBION DE CADA PARAM
    params = ['STAT','SP','IPHASW','D','HRMM','SECON','CODA',
              'AMPLIT','PERI','AZIMU','VELO','AIN','AR','TRES',
              'W','DIS','CAS', 'YEAR','MONTH','DAY','HOUR',
              'MINUTES','SECONDS','LOCATION','DISTANCE','EVENT_ID',
              'LATITUDE','LONGITUDE','DEPTH','DEPTH_INDICATOR',
              'LOCATING_INDICATOR','HYPOCENTER_REPORTING_AGENCY',
              'NUMBER_OF_STATIONS_USED','RMS_OF_TIME_RESIDUALS',
              'MAGNITUDE_NUMBER_1','TYPE_OF_MAGNITUDE_1',
              'MAGNITUDE_REPORTING_AGENCY_1','MAGNITUDE_NUMBER_2',
              'TYPE_OF_MAGNITUDE_2','MAGNITUDE_REPORTING_AGENCY_2',
              'MAGNITUDE_NUMBER_3','TYPE_OF_MAGNITUDE_3',
              'MAGNITUDE_REPORTING_AGENCY_3','LINE_TYPE','GAP',
              'ORIGIN_TIME_ERROR','LATITUDE_ERROR','LONGITUDE_ERROR',
              'DEPTH_ERROR','COVARIANCE_XY','COVARIANCE_XZ',
              'COVARIANCE_YZ','EPICENTER_LOCATION','BINARY_FILENAME',
              'LAST_ACTION_DONE','DATETIME_LAST_ACTION']
    
    def __init__(self, filename, path):
        
        # COMPOROBAR SI LA RUTA Y EL ARCHIVO EXISTEN
        self.path = path
        self.filename = filename
        self.lines_info = [] # List that contains the information, classified by linetype
        
                   
    # Get each attribute from the Sfile 
    def get_params(self):        
        lines = []
        with open(self.path+self.filename) as sfile:
            station_flag = False
            for cnt, line in enumerate(sfile):
                
                if station_flag and line[1] is not ' ':
                    station_info = {
                                    'STAT'          : line[1:6].strip(),
                                    'SP'            : line[6:8].strip(),
                                    'IPHAS'        : line[9:14].strip(),
                                    'W1'             : line[15],
                                    'D'             : line[16],
                                    'HRMM'          : line[18:22].strip(),
                                    'SECON'         : line[23:28].strip(),
                                    'CODA'          : line[29:33].strip(),
                                    'AMPLIT'        : line[34:40].strip(),
                                    'PERI'          : line[41:45].strip(),
                                    'AZIMU'         : line[46:51].strip(),
                                    'VELO'          : line[52:56].strip(),
                                    'AIN'           : line[57:60].strip(),
                                    'AR'            : line[61:63].strip(),
                                    'TRES'          : line[64:68].strip(),
                                    'W2'             : line[69],
                                    'DIS'           : line[70:75].strip(),
                                    'CAS'           : line[76:79].strip()
                                }
                    
                    # Revisar si se actualiza en el dic
                    self.type_7.append(station_info)
                    #self.lines_info.append({'TYPE_7':self.type_7})
                else:
                    # Type 7: Station data
                    if line[79] == '7':
                        station_flag = True
                        self.type_7 = []
                        self.lines_info.append({'TYPE_7':self.type_7})
                                 
                    # Type 1: Hypocenter line
                    if line[79] == '1':
                        
                        # FILL ALL THE FIELDS
                        self.type_1 = {
                                       'YEAR'                           :   line[1:5].strip(),
                                       'MONTH'                          :   line[6:8].strip(),
                                       'DAY'                            :   line[8:10].strip(),
                                       'HOUR'                           :   line[11:13].strip(),
                                       'MINUTES'                        :   line[13:15].strip(),
                                       'SECONDS'                        :   line[16:20].strip(),
                                       'LOCATION'                       :   ' ',
                                       'DISTANCE'                       :   line[21].strip(),
                                       'EVENT_ID'                       :   line[23:30].strip(),
                                       'LATITUDE'                       :   line[23:30].strip(),
                                       'LONGITUDE'                      :   line[30:38].strip(),
                                       'DEPTH'                          :   line[38:43].strip(),
                                       'DEPTH_INDICATOR'                :   ' ',
                                       'LOCATING_INDICATOR'             :   ' ',
                                       'HYPOCENTER_REPORTING_AGENCY'    :   line[45:48].strip(),
                                       'NUMBER_OF_STATIONS_USED'        :   line[48:51].strip(),
                                       'RMS_OF_TIME_RESIDUALS'          :   ' ',
                                       'MAGNITUDE_NUMBER_1'             :   ' ',
                                       'TYPE_OF_MAGNITUDE_1'            :   line[55:59].strip(),
                                       'MAGNITUDE_REPORTING_AGENCY_1'   :   ' ',
                                       'MAGNITUDE_NUMBER_2'             :   ' ',
                                       'TYPE_OF_MAGNITUDE_2'            :   ' ',
                                       'MAGNITUDE_REPORTING_AGENCY_2'   :   ' ',
                                       'MAGNITUDE_NUMBER_3'             :   ' ',
                                       'TYPE_OF_MAGNITUDE_3'            :   ' ',
                                       'MAGNITUDE_REPORTING_AGENCY_3'   :   ' '
                                }
                        
                        # Avoid saving unused information
                        lines.append(line)
                        self.lines_info.append({'TYPE_1':self.type_1})
                       
                    # Type E: Hypocenter error
                    elif line[79] == 'E':
                        
                        # THERE MAY BE MORE FIELDS, FIELD THEM IN THE DIC
                        self.type_E = {'LINE_TYPE'              :   'E',
                                       'GAP'                    :   line[5:8].strip(),
                                       'ORIGIN_TIME_ERROR'      :   line[14:20].strip(),
                                       'LATITUDE_ERROR'         :   line[24:30].strip(),
                                       'LONGITUDE_ERROR'        :   line[32:38].strip(),
                                       'DEPTH_ERROR'            :   line[38:43].strip(),
                                       'COVARIANCE_XY'          :   line[43:55].strip(),
                                       'COVARIANCE_XZ'          :   line[55:67].strip(),
                                       'COVARIANCE_YZ'          :   line[67:79].strip()       
                                }

                        # Avoid saving unused information
                        lines.append(line)
                        self.lines_info.append({'TYPE_E':self.type_E})
                    
                    # Type 3: Epicenter location
                    elif line[79] == '3':
                        # THERE MAY BE MORE FIELDS, FIELD THEM IN THE DIC
                        if('Epicentro' in line):
                            self.type_3 = {
                                           'EPICENTER_LOCATION'     :   line[12:79].strip()
                                    }
                            
                            # Avoid saving unused information
                            lines.append(line)
                            self.lines_info.append({'TYPE_3':self.type_3})
                        
                    # Type 6: File name
                    elif line[79] == '6':
                        # THERE MAY BE MORE FIELDS, FIELD THEM IN THE DIC
                        self.type_6 = {
                                       'BINARY_FILENAME'        :   line[0:79].strip()
                                }
                        
                        lines.append(line)
                        self.lines_info.append({'TYPE_6':self.type_6})
                        
                    # Type I: ID Line
                    elif line[79] == 'I':  
                        # THERE MAY BE MORE FIELDS, FIELD THEM IN THE DIC
                        self.type_I = {
                                       'LAST_ACTION_DONE'       :   line[8:11].strip(),
                                       'DATETIME_LAST_ACTION'   :   line[12:26].strip()
                                }
                        
                        lines.append(line)
                        self.lines_info.append({'TYPE_I':self.type_I})
        
    def print_params(self):
        '''
        The variable lines_info is a list that contains dicts corresponding to the 
        types of the lines, Example: [{'TYPE_1': info},{'TYPE_E' : info}, ...]
        
        The dicts inside lines_info ({'TYPE_1': info}.{'TYPE_E' : info},...) contains
        a dic or a list of dics with all the parameters name and info for each line type
        Example {'TYPE_1' : {'YEAR': '1996','MONTH': '06',...}}
        
        '''
        for type_line in self.lines_info:
            for type_fields in type_line:
                print(type_fields)
                dic = type_line[type_fields]
                for param in dic:

                    if isinstance(param,dict):
                        for element in param:
                            print(element+"\t:\t"+param[element])
                        print("\n")
                           
                    else:
                        print(param+"\t:\t"+dic[param])
                print('\n')
