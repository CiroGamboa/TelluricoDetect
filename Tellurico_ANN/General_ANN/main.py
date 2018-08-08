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
from numpy.random import seed
seed(1)
#import gc

if __name__ == "__main__":
    
    dataset_folder = 'datasets/'
    computed_anns_folder = 'computed_anns/'
    retrain = 10
    
    #%% 3 stations - 0.5 of phase
    ann_3s_05 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_3stats_0.5.csv',
                        'X_rows'           : (0,42),
                        'y_rows'           : (42,43)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (42, 21, 10, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 25,
                        'epochs'          : 50,
                        'optimizer'       : 'rmsprop'
                    }
                }
                    
#    classifier_3s_05 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
#                                    X_rows = ann_3s_05['init_info']['X_rows'], 
#                                    y_rows = ann_3s_05['init_info']['y_rows'])
    

    
    metrics_3s_05 = []
    for train in range(0,retrain):
        classifier_3s_05 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
                                X_rows = ann_3s_05['init_info']['X_rows'], 
                                y_rows = ann_3s_05['init_info']['y_rows'])
        kfold_output = classifier_3s_05.kfold_train(ann_3s_05,random_state = train)
        metrics_3s_05.append(kfold_output[1]['metrics'])
    ann_3s_05['TelluricoANN'] = classifier_3s_05
    ann_3s_05['metrics'] = metrics_3s_05
    

#    classifier_3s_05.import_ann(filename = computed_anns_folder + 'tellurico_ann_3s_05',
#                                loss_fn = 'binary_crossentropy',
#                                optimizer = 'rmsprop',
#                                metric = 'accuracy')
    
#    ann_3s_05['metrics'] = classifier_3s_05.re_train(classifier = classifier_3s_05.get_imported_ann(), 
#             test_size = 0.2, 
#             scoring = 'accuracy',
#             iters = 2, 
#             cv = 2, 
#             n_jobs = 1)
    

    

    
    #%% 3 stations - 0.9 of phase
    ann_3s_09 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_3stats_0.9.csv',
                        'X_rows'           : (0,42),
                        'y_rows'           : (42,43)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (42, 21, 10, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 10,
                        'epochs'          : 50,
                        'optimizer'       : 'adam'
                    }
                }
                    
    
    metrics_3s_09 = []
    for train in range(0,retrain):
        classifier_3s_09 = TelluricoANN(filename = dataset_folder+ann_3s_09['init_info']['dataset_filename'],
                                X_rows = ann_3s_09['init_info']['X_rows'], 
                                y_rows = ann_3s_09['init_info']['y_rows'])
        kfold_output = classifier_3s_09.kfold_train(ann_3s_09,random_state = train)
        metrics_3s_09.append(kfold_output[1]['metrics'])
    ann_3s_09['TelluricoANN'] = classifier_3s_09
    ann_3s_09['metrics'] = metrics_3s_09
    
    
    
    
    #%% 4 stations - 0.5 of phase
    ann_4s_05 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_4stats_0.5.csv',
                        'X_rows'           : (0,55),
                        'y_rows'           : (55,56)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (55, 28, 14, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 10,
                        'epochs'          : 50,
                        'optimizer'       : 'rmsprop'
                    }
                }
                    
    
    metrics_4s_05 = []
    for train in range(0,retrain):
        classifier_4s_05 = TelluricoANN(filename = dataset_folder+ann_4s_05['init_info']['dataset_filename'],
                                X_rows = ann_4s_05['init_info']['X_rows'], 
                                y_rows = ann_4s_05['init_info']['y_rows'])
        kfold_output = classifier_4s_05.kfold_train(ann_4s_05,random_state = train)
        metrics_4s_05.append(kfold_output[1]['metrics'])
    ann_4s_05['TelluricoANN'] = classifier_4s_05
    ann_4s_05['metrics'] = metrics_4s_05    





    #%% 4 stations - 0.9 of phase
    ann_4s_09 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_4stats_0.9.csv',
                        'X_rows'           : (0,55),
                        'y_rows'           : (55,56)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (55, 28, 14, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 10,
                        'epochs'          : 50,
                        'optimizer'       : 'rmsprop'
                    }
                }


    metrics_4s_09 = []
    for train in range(0,retrain):
        classifier_4s_09 = TelluricoANN(filename = dataset_folder+ann_4s_09['init_info']['dataset_filename'],
                                X_rows = ann_4s_09['init_info']['X_rows'], 
                                y_rows = ann_4s_09['init_info']['y_rows'])
        kfold_output = classifier_4s_09.kfold_train(ann_4s_09,random_state = train)
        metrics_4s_09.append(kfold_output[1]['metrics'])
    ann_4s_09['TelluricoANN'] = classifier_4s_09
    ann_4s_09['metrics'] = metrics_4s_09
    

    
#%% Export metric variables
import pickle
filename = 'metrics_3s_05'
outfile = open(filename,'wb')
pickle.dump(metrics_3s_05,outfile)
outfile.close()

filename = 'metrics_3s_09'
outfile = open(filename,'wb')
pickle.dump(metrics_3s_09,outfile)
outfile.close()

filename = 'metrics_4s_05'
outfile = open(filename,'wb')
pickle.dump(metrics_4s_05,outfile)
outfile.close()

filename = 'metrics_4s_09'
outfile = open(filename,'wb')
pickle.dump(metrics_4s_09,outfile)
outfile.close()
    
    
    
    
    
    
    
    
    
    
    
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
                    
                    