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
seed(0)
from tensorflow import set_random_seed
set_random_seed(0) 

import pickle

if __name__ == "__main__":
    
    dataset_folder = 'ultimate_datasets/'
    computed_anns_folder = 'computed_anns/'
    retrain = 13

    #%% 1 station - 0.5 of phase
    ann_1s_05 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_1stats_0.5.csv',
                        'X_rows'           : (0,14),
                        'y_rows'           : (14,15)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (14, 7, 4, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 32,
                        'epochs'          : 500,
                        'optimizer'       : 'adadelta'
                    }
                }
                    
    
    metrics_1s_05 = []
    for train in range(0,retrain):
        classifier_1s_05 = TelluricoANN(filename = dataset_folder+ann_1s_05['init_info']['dataset_filename'],
                                X_rows = ann_1s_05['init_info']['X_rows'], 
                                y_rows = ann_1s_05['init_info']['y_rows'])
        kfold_output = classifier_1s_05.kfold_train(ann_1s_05,random_state = train)
        metrics_1s_05.append(kfold_output[1]['metrics'])
    ann_1s_05['TelluricoANN'] = classifier_1s_05
    ann_1s_05['metrics'] = metrics_1s_05
    
    filename = 'metrics_1s_05'
    outfile = open(filename,'wb')
    pickle.dump(metrics_1s_05,outfile)
    outfile.close()


    #%% 1 station - 0.9 of phase

    ann_1s_09 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_1stats_0.9.csv',
                        'X_rows'           : (0,14),
                        'y_rows'           : (14,15)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (14, 7, 4, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 8,
                        'epochs'          : 100,
                        'optimizer'       : 'adadelta'
                    }
                }



                    
    
    metrics_1s_09 = []
    for train in range(0,retrain):
        classifier_1s_09 = TelluricoANN(filename = dataset_folder+ann_1s_09['init_info']['dataset_filename'],
                                X_rows = ann_1s_09['init_info']['X_rows'], 
                                y_rows = ann_1s_09['init_info']['y_rows'])
        kfold_output = classifier_1s_09.kfold_train(ann_1s_09,random_state = train)
        metrics_1s_09.append(kfold_output[1]['metrics'])
    ann_1s_09['TelluricoANN'] = classifier_1s_09
    ann_1s_09['metrics'] = metrics_1s_09

    filename = 'metrics_1s_09'
    outfile = open(filename,'wb')
    pickle.dump(metrics_1s_09,outfile)
    outfile.close()


    #%% 2 stations - 0.5 of phase
    ann_2s_05 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_2stats_0.5.csv',
                        'X_rows'           : (0,28),
                        'y_rows'           : (28,29)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (28, 14, 7, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 8,
                        'epochs'          : 50,
                        'optimizer'       : 'adadelta'
                    }
                }
                    
    
    metrics_2s_05 = []
    for train in range(0,retrain):
        classifier_2s_05 = TelluricoANN(filename = dataset_folder+ann_2s_05['init_info']['dataset_filename'],
                                X_rows = ann_2s_05['init_info']['X_rows'], 
                                y_rows = ann_2s_05['init_info']['y_rows'])
        kfold_output = classifier_2s_05.kfold_train(ann_2s_05,random_state = train)
        metrics_2s_05.append(kfold_output[1]['metrics'])
    ann_2s_05['TelluricoANN'] = classifier_2s_05
    ann_2s_05['metrics'] = metrics_2s_05
    
    filename = 'metrics_2s_05'
    outfile = open(filename,'wb')
    pickle.dump(metrics_2s_05,outfile)
    outfile.close()
    
 
    %% 2 stations - 0.9 of phase
    ann_2s_09 = {   'init_info' : {
                        'dataset_filename' : 'attributes_matrix_prot04_2stats_0.9.csv',
                        'X_rows'           : (0,28),
                        'y_rows'           : (28,29)  
                    },
                    'proc_params' : {
                                'test_size'       : 0.2,
                                'validation_size' : 0.2,
                                'metric'          : 'accuracy',
                                'arch'            : (28, 14, 7, 1),
                                'hidden_act_fn'    : 'relu',
                                'output_act_fn'    : 'sigmoid',
                                'loss_fn'         : 'binary_crossentropy',
                                'cv'              : 10,
                    },
                    
                    'best_params' : {
                        'batch_size'      : 16,
                        'epochs'          : 10,
                        'optimizer'       : 'adadelta'
                    }
                }
                    
    
    metrics_2s_09 = []
    for train in range(0,retrain):
        classifier_2s_09 = TelluricoANN(filename = dataset_folder+ann_2s_09['init_info']['dataset_filename'],
                                X_rows = ann_2s_09['init_info']['X_rows'], 
                                y_rows = ann_2s_09['init_info']['y_rows'])
        kfold_output = classifier_2s_09.kfold_train(ann_2s_09,random_state = train)
        metrics_2s_09.append(kfold_output[1]['metrics'])
    ann_2s_09['TelluricoANN'] = classifier_2s_09
    ann_2s_09['metrics'] = metrics_2s_09
    
    filename = 'metrics_2s_09'
    outfile = open(filename,'wb')
    pickle.dump(metrics_2s_09,outfile)
    outfile.close()

     %% 3 stations - 0.5 of phase
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
                        'batch_size'      : 128,
                        'epochs'          : 50,
                        'optimizer'       : 'adadelta'
                    }
                }
                        
    metrics_3s_05 = []
    for train in range(0,retrain):
        classifier_3s_05 = TelluricoANN(filename = dataset_folder+ann_3s_05['init_info']['dataset_filename'],
                                X_rows = ann_3s_05['init_info']['X_rows'], 
                                y_rows = ann_3s_05['init_info']['y_rows'])
        kfold_output = classifier_3s_05.kfold_train(ann_3s_05,random_state = train)
        metrics_3s_05.append(kfold_output[1]['metrics'])
    ann_3s_05['TelluricoANN'] = classifier_3s_05
    ann_3s_05['metrics'] = metrics_3s_05
        
    filename = 'metrics_3s_05'
    outfile = open(filename,'wb')
    pickle.dump(metrics_3s_05,outfile)
    outfile.close()


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
                        'batch_size'      : 8,
                        'epochs'          : 10,
                        'optimizer'       : 'adadelta'
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
    
    filename = 'metrics_3s_09'
    outfile = open(filename,'wb')
    pickle.dump(metrics_3s_09)
    outfile.close()
    

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
                        'batch_size'      : 64,
                        'epochs'          : 10,
                        'optimizer'       : 'adadelta'
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


    filename = 'metrics_4s_05'
    outfile = open(filename,'wb')
    pickle.dump(metrics_4s_05,outfile)
    outfile.close()


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
                        'batch_size'      : 8,
                        'epochs'          : 10,
                        'optimizer'       : 'adadelta'
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
    
    filename = 'metrics_4s_09'
    outfile = open(filename,'wb')
    pickle.dump(metrics_4s_09,outfile)
    outfile.close()

#%% Export metric variables
#import pickle
#filename = 'metrics_3s_05'
#outfile = open(filename,'wb')
#pickle.dump(metrics_3s_05,outfile)
#outfile.close()
#
#filename = 'metrics_3s_09'
#outfile = open(filename,'wb')
#pickle.dump(metrics_3s_09,outfile)
#outfile.close()
#
#filename = 'metrics_4s_05'
#outfile = open(filename,'wb')
#pickle.dump(metrics_4s_05,outfile)
#outfile.close()
#
#filename = 'metrics_4s_09'
#outfile = open(filename,'wb')
#pickle.dump(metrics_4s_09,outfile)
#outfile.close()
    
    
    
    
    
    
    
    
    
    
    
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

                   