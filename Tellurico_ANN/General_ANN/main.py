#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:55:22 2018

@author: CiroGamJr


FALTA ENTRENAR LAS ANN CON LA NUEVA CLASE Y EXPORTAR SUS FICHAS TECNICAS
PERMITIENDO IMPORTARLAS TAMBIEN, ES BUENA IDEA PENSAR EN UN ESTANDAR, INDEPENDIENTE
SI SE ENTRENA DE FORMA SIMPLE, KFLOD O GRID


NO SIRVIO IMPORTAR LA RED Y LUEGO USAR EL MISMO CLASIFICADOR PARA HACER MAS ENTRENAMIENTOS
TOCA ADQUIRIR LOS PARAMETROS DE LA RED Y HACERLA DESDE CERO SIN IMPORTAR. PARA ELLO ES BUENA IDEA
REALIZAR LO INDICADO EN EL APARTADO ANTERIOR


"""

from TelluricoANN import TelluricoANN

if __name__ == "__main__":
    
    dataset_folder = 'datasets/'
    computed_anns_folder = 'computed_anns/'
    
    
    #%% 3 stations - 0.5 of phase
    ann_3s_05 = {   'init_info' : 
                    {
                        'dataset_filename' : 'attributes_matrix_prot04_3stats_0.5.csv',
                        'X_rows'           : (0,42),
                        'y_rows'           : (42,43)  
                    }            
                }
                    
    classifier_3s_05 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
                                    X_rows = ann_3s_05['init_info']['X_rows'], 
                                    y_rows = ann_3s_05['init_info']['y_rows'])
    
    classifier_3s_05.import_ann(filename = computed_anns_folder + 'tellurico_ann_3s_05',
                                loss_fn = 'binary_crossentropy',
                                optimizer = 'rmsprop',
                                metric = 'accuracy')
    
    ann_3s_05['metrics'] = classifier_3s_05.re_train(classifier = classifier_3s_05.get_imported_ann(), 
             test_size = 0.2, 
             scoring = 'accuracy',
             iters = 2, 
             cv = 2, 
             n_jobs = 1)
    
    ann_3s_05['TelluricoANN'] = classifier_3s_05
    

    
#    #%% 3 stations - 0.9 of phase
#    ann_3s_09 = {   'init_info' : 
#                    {
#                        'dataset_filename' : 'attributes_matrix_prot04_3stats_0.5.csv',
#                        'X_rows'           : (0,42),
#                        'y_rows'           : (42,43)  
#                    } 
#                }
#
#    classifier_3s_09 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
#                                    X_rows = ann_3s_05['init_info']['X_rows'], 
#                                    y_rows = ann_3s_05['init_info']['y_rows'])
#    
#    classifier_3s_09.import_ann(filename = 'tellurico_ann_3s_05',
#                                loss_fn = 'binary_crossentropy',
#                                optimizer = 'adam',
#                                metric = 'accuracy')
#    
#    
#    #%% 4 stations - 0.5 of phase
#    ann_4s_05 = {   'init_info' : 
#                    {
#                        'dataset_filename' : 'attributes_matrix_prot04_3stats_0.5.csv',
#                        'X_rows'           : (0,55),
#                        'y_rows'           : (55,56)  
#                    }
#                }
#                    
#    classifier_4s_05 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
#                                    X_rows = ann_3s_05['init_info']['X_rows'], 
#                                    y_rows = ann_3s_05['init_info']['y_rows'])
#    
#    classifier_4s_05.import_ann(filename = 'tellurico_ann_3s_05',
#                                loss_fn = 'binary_crossentropy',
#                                optimizer = 'rmsprop',
#                                metric = 'accuracy')
#                        
#    #%% 4 stations - 0.9 of phase
#    ann_4s_09 = {   'init_info' : 
#                    {
#                        'dataset_filename' : 'attributes_matrix_prot04_3stats_0.5.csv',
#                        'X_rows'           : (0,55),
#                        'y_rows'           : (55,56)  
#                    }
#                }
#                    
#    classifier_4s_09 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
#                                    X_rows = ann_3s_05['init_info']['X_rows'], 
#                                    y_rows = ann_3s_05['init_info']['y_rows'])
#    
#    classifier_4s_09.import_ann(filename = 'tellurico_ann_3s_05',
#                                loss_fn = 'binary_crossentropy',
#                                optimizer = 'rmsprop',
#                                metric = 'accuracy')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    #%% Example inputs for training
#    grid_dict = {
#                    'proc_params' : {
#                        'test_size'       : 0.2,
#                        'validation_size' : 0.2,
#                        'metric'          : 'accuracy',
#                        'arch'            : (55, 28, 14, 1),
#                        'hidden_act_fn'   : 'relu',
#                        'output_act_fn'   : 'sigmoid',
#                        'loss_fn'         : 'binary_crossentropy',
#                        'n_jobs'          : -2,
#                        'cv'              : 10,
#                        're_train'        : 10
#                    },
#                    
#                    'param_grid' : {
#                        'batch_size'      : [10, 25, 32],
#                        'epochs'          : [100, 50, 500],
#                        'optimizer'       : ['adam', 'rmsprop']
#                    }
#            }
                    
                    