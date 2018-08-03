#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:45:18 2018

@author: CiroGamJr
"""

class TelluricoANN():
    
    def __init__(self,filename,X_rows, y_rows):
        self.dataset_file = filename
        self.simple_trained_ann = None
        self.kfold_trained_ann = None
        self.grid_trained_ann = None
        self.imported_ann = None
        
        # Importing the dataset
        import pandas as pd
        dataset = pd.read_csv(self.dataset_file)
        self.X = dataset.iloc[:, X_rows[0]:X_rows[1]].values
        self.y = dataset.iloc[:, y_rows[0]:y_rows[1]].values
    
    
    def get_imported_ann(self):
        return self.imported_ann
    
    
    
    def import_dataset(self, test_size, random_state):    
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        
        return X_train,y_train,X_test,y_test
          
    def build_classifier(self, proc_params, optimizer):
        '''
        The layers are built by default with the rectifier activation function
        for the input and hidden layers and the sigmoid function is used for 
        the output layer. The loss function is by default binary_crossentropy.
        In future versions those parameters could be modified.
        
        '''
        from keras.models import Sequential
        from keras.layers import Dense
        
        # Initialising the ANN
        classifier = Sequential()
        arch = proc_params['arch']
        hidden_act_fn = proc_params['hidden_act_fn']
        output_act_fn = proc_params['output_act_fn']
        metric = proc_params['metric']
        loss_fn = proc_params['loss_fn']
        
        index = 0
        last_index = len(arch) - 1
        for layer in arch[1:]:
        
            # Adding the input layer and the first hidden layer
            if index == 0:
                classifier.add(Dense(output_dim = layer, init = 'uniform', activation = hidden_act_fn, input_dim = arch[0]))
            
            elif index == last_index:
                # Adding the output layer
                classifier.add(Dense(output_dim = layer, init = 'uniform', activation = output_act_fn))
                
            else:
                # Adding the second hidden layer
                classifier.add(Dense(output_dim = layer, init = 'uniform', activation = hidden_act_fn))
                
            index += 1
        
        # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = loss_fn, metrics = [metric])
        return classifier
    
    def simple_train(self):
        pass
    
    def kfold_train(self, kfold_dict): # FALTAAAA
        '''
        Example dict:
            {
                    'proc_params' : {
                        'test_size'       : 0.2,
                        'validation_size' : 0.2,
                        'metric'          : 'accuracy',
                        'arch'            : (55, 28, 14, 1),
                        'hidden_act_fn    : 'relu',
                        'output_act_fn    : 'sigmoid',
                        'loss_fn'         : 'binary_crossentropy',
                        'n_jobs'          : -2,
                        'cv'              : 10,
                        're_train'        : 10
                    },
                    
                    'best_params' : {
                        'batch_size'      : 10,
                        'epochs'          : 100,
                        'optimizer'       : 'adam'
                    }
            }
        '''
        
        
        from keras.wrappers.scikit_learn import KerasClassifier

        classifier = KerasClassifier(build_fn = self.build_classifier, proc_params = kfold_dict['proc_params'])
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
        mean = accuracies.mean()
        #variance = accuracies.std()
    
#    def grid_search_train(self,metric,architecture, parameters, cv):
    def grid_search_train(self,grid_dict):
        
        '''
        Example dict:
            {
                    'proc_params' : {
                        'test_size'       : 0.2,
                        'validation_size' : 0.2,
                        'metric'          : 'accuracy',
                        'arch'            : (55, 28, 14, 1),
                        'hidden_act_fn    : 'relu',
                        'output_act_fn    : 'sigmoid',
                        'loss_fn'         : 'binary_crossentropy',
                        'n_jobs'          : -2,
                        'cv'              : 10,
                        're_train'        : 10
                    },
                    
                    'param_grid' : {
                        'batch_size'      : [10, 25, 32],
                        'epochs'          : [100, 50, 500],
                        'optimizer'       : ['adam', 'rmsprop']
                    }
            }
        '''

        
        
        # Defining the parameters for running the Grid Search
        
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import GridSearchCV
        
        classifier = KerasClassifier(build_fn = self.build_classifier, proc_params = grid_dict['proc_params'])

        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = grid_dict['param_grid'],
                                   scoring = grid_dict['proc_params']['metric'],
                                   cv = grid_dict['proc_params']['cv'],
                                   n_jobs = grid_dict['proc_params']['n_jobs'],
                                   refit = True)
        
        
                
        X_train,y_train,X_test,y_test = self.import_dataset(grid_dict['proc_params']['test_size'], random_state = 0)
        
        
        # Fitting the ANN to the training set
        grid_search = grid_search.fit(X_train, y_train)
        
        # Getting the best parameters
        best_parameters = grid_search.best_params_ 
        best_score = grid_search.best_score_
        best_classifier = grid_search.best_estimator_
        base_classifier = best_classifier.copy()
           
        # Predicting the Test set results
        y_pred = best_classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        
        # Making the confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
#        test_score = accuracy_score(y_true = y_test, y_pred = y_pred)
        
        
        # Put all the ANN results together
        metrics = {}
        metrics['validation_score'] = best_score
        metrics['confusion_matrix'] = cm
        
        grid_dict['best_parameters'] = best_parameters
        grid_dict['metrics'] = {**metrics, **self.compute_metrics(cm)}
        self.grid_trained_ann = (base_classifier, grid_dict)
        
        
    def retrained_grid_train(self,grid_dict):
        
        '''
        Example dict:
            {
                    'proc_params' : {
                        'test_size'       : 0.2,
                        'validation_size' : 0.2,
                        'metric'          : 'accuracy',
                        'arch'            : (55, 28, 14, 1),
                        'hidden_act_fn    : 'relu',
                        'output_act_fn    : 'sigmoid',
                        'loss_fn'         : 'binary_crossentropy',
                        'n_jobs'          : -2,
                        'cv'              : 10,
                        're_train'        : 10
                    },
                    
                    'param_grid' : {
                        'batch_size'      : [10, 25, 32],
                        'epochs'          : [100, 50, 500],
                        'optimizer'       : ['adam', 'rmsprop']
                    }
            }
        '''

        
        
        # Defining the parameters for running the Grid Search
        
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import GridSearchCV
        
        classifier = KerasClassifier(build_fn = self.build_classifier, proc_params = grid_dict['proc_params'])

        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = grid_dict['param_grid'],
                                   scoring = grid_dict['proc_params']['metric'],
                                   cv = grid_dict['proc_params']['cv'],
                                   n_jobs = grid_dict['proc_params']['n_jobs'],
                                   refit = True)
        
        
                
        X_train,y_train,X_test,y_test = self.import_dataset(grid_dict['proc_params']['test_size'], random_state = 0)
        
        
        # Fitting the ANN to the training set
        grid_search = grid_search.fit(X_train, y_train)
        
        # Getting the best parameters
        from copy import copy
        best_parameters = grid_search.best_params_ 
#        best_score = grid_search.best_score_ # VALIDACIOOOOONNN
        best_classifier = grid_search.best_estimator_
        base_classifier = copy(best_classifier)
           
        
        grid_dict['best_parameters'] = best_parameters
        
        metrics = self.re_train(classifier = base_classifier,
                                test_size = grid_dict['proc_params']['test_size'],
                                iters = grid_dict['proc_params']['re_train'],
                                cv = grid_dict['proc_params']['cv'], 
                                n_jobs = grid_dict['proc_params']['n_jobs'])
        
        grid_dict['metrics'] = metrics
        self.grid_trained_ann = (base_classifier, grid_dict)
#        metrics = {}
#        metrics['validation_score'] = []
#        metrics['confusion_matrix'] = []
#        metrics['test_fp_rate'] = []
#        metrics['test_tp_rate'] = []
#        metrics['test_precision'] = []
#        metrics['test_accuracy'] = []
#        metrics['test_recall'] = []
#        metrics['test_F1_score'] = []
#        
#        for index in range(0,re_train):
#            if index != 0:
#                X_train,y_train,X_test,y_test = self.import_dataset(grid_dict['proc_params']['test_size'], random_state = index)
#                scores = cross_val_score(estimator = best_classifier, X = X_train, y = y_train, cv = grid_dict['proc_params']['cv'], n_jobs = grid_dict['proc_params']['n_jobs'])
#                best_score = scores.mean()
#        
#        
#            # Predicting the Test set results
#            y_pred = best_classifier.predict(X_test)
#            y_pred = (y_pred > 0.5)
#            
#            # Making the confusion matrix
#            cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
#        
#        
#            # Put all the ANN results together
#            metrics['validation_score'] = metrics['validation_score'].append(best_score)
#            metrics['confusion_matrix'] = metrics['confusion_matrix'].append(cm)
#            
#            computed_metrics  = self.compute_metrics(cm)
#            metrics['test_fp_rate'] = metrics['test_fp_rate'].append(computed_metrics['test_fp_rate'])
#            metrics['test_tp_rate'] = metrics['test_tp_rate'].append(computed_metrics['test_tp_rate'])
#            metrics['test_precision'] = metrics['test_precision'].append(computed_metrics['test_precision'])
#            metrics['test_accuracy'] = metrics['test_accuracy'].append(computed_metrics['test_accuracy'])
#            metrics['test_recall'] = metrics['test_recall'].append(computed_metrics['test_recall'])
#            metrics['test_F1_score'] = metrics['test_F1_score'].append(computed_metrics['test_F1_score'])
#        


        
        
        
    def re_train(self, classifier, test_size, scoring, iters, cv, n_jobs):
        # TODOS ESTOS PARAMETROS DE ENTRADA DEBERIAN VENIR EN FORMA DE DICT
        
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import cross_val_score
        
        metrics = {}
        metrics['validation_score'] = []
        metrics['confusion_matrix'] = []
        metrics['test_fp_rate'] = []
        metrics['test_tp_rate'] = []
        metrics['test_precision'] = []
        metrics['test_accuracy'] = []
        metrics['test_recall'] = []
        metrics['test_F1_score'] = []
        
        
        # Temporal
        best_score = 0.95
        
        
        for index in range(0,iters):
            X_train,y_train,X_test,y_test = self.import_dataset(test_size, random_state = index)
            
            if index != 0:
                scores = cross_val_score(estimator = classifier, X = X_train, scoring = scoring, y = y_train, cv = cv, n_jobs = n_jobs)
                best_score = scores.mean()
        
        
            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5)
            
            # Making the confusion matrix
            cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
        
        
            ''' VALIDACIONNNNNNNNN
            '''
        
        
        
            # Put all the ANN results together
            metrics['validation_score'] = metrics['validation_score'].append(best_score)
            metrics['confusion_matrix'] = metrics['confusion_matrix'].append(cm)
            
            computed_metrics  = self.compute_metrics(cm)
            metrics['test_fp_rate'] = metrics['test_fp_rate'].append(computed_metrics['test_fp_rate'])
            metrics['test_tp_rate'] = metrics['test_tp_rate'].append(computed_metrics['test_tp_rate'])
            metrics['test_precision'] = metrics['test_precision'].append(computed_metrics['test_precision'])
            metrics['test_accuracy'] = metrics['test_accuracy'].append(computed_metrics['test_accuracy'])
            metrics['test_recall'] = metrics['test_recall'].append(computed_metrics['test_recall'])
            metrics['test_F1_score'] = metrics['test_F1_score'].append(computed_metrics['test_F1_score'])
            
            
        return metrics
        
        
    def compute_metrics(self, cm):
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]
        
        P = TP + FP
        N = TN + FN
        
        fp_rate = FP/N
        tp_rate = TP/P
        precision = TP/(TP + FP)
        accuracy = (TP + TN)/(P + N)
        recall = TP/P
        F1_score = 2/((1/precision)+(1/recall))
        
        return {'test_fp_rate':fp_rate,
                'test_tp_rate':tp_rate,
                'test_precision':precision,
                'test_accuracy':accuracy,
                'test_recall':recall,
                'test_F1_score':F1_score}
        
  
    # Export the ANN
    def export_ann(self,classifier,filename):
        # Serialize model to JSON
        json_ann = classifier.model.to_json()
        with open(filename+".json",'w') as json_file:
            json_file.write(json_ann)
            
        # Serialize weights to HDF5
        classifier.model.save_weights(filename+".h5")
        
        
    # Import the ANN
    def import_ann(self,filename,loss_fn,optimizer,metric):
        
        from keras.models import model_from_json
        
        # Load JSON and create model
#        json_file = open("tellurico_ann3.json",'r')
        json_file = open(filename+".json",'r')

        loaded_json_ann = json_file.read()
        json_file.close()
        loaded_classifier = model_from_json(loaded_json_ann)
        
        # Load weights into new model
        loaded_classifier.load_weights(filename+".h5")
        
        
        # Evaluate loaded model on test data
        loaded_classifier.compile(loss = loss_fn, optimizer = optimizer,
                                  metrics = [metric])
        
        # Predicting the Test set results
#        y_pred = loaded_classifier.predict(X_test)
#        y_pred = (y_pred > 0.5)
        
        # Making the confusion matrix
#        from sklearn.metrics import confusion_matrix, accuracy_score
#        cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
#        test_acc = accuracy_score(y_true = y_test, y_pred = y_pred)
#        print("Confusion Matrix")
#        print(cm)
#        print("Test set accuracy")
#        print(test_acc)
        
        self.imported_ann = loaded_classifier
#        return loaded_classifier
    


    
    
    
    
    
    
    
    
    
    