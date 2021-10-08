#!/usr/bin/env python

"""
(Multi-Resolution) Online Deterministic Annealing (ODA) for Classification and Clustering
Christos N. Mavridis & John S. Baras,
Department of Electrical and Computer Engineering, University of Maryland
email: <mavridis@umd.edu>
"""

#%% Import Modules

import pickle
import numpy as np

# from oda_python import ODA 
from oda_jit import ODA 

#%% Problem Parameters

def run(# Dataset
        data_file = '../tests/demo/data/data',
        load_file = '',
        results_file = '../tests/demo/demo-1',
        # Resolutions
        res=[1],
        # Temperature
        Tmax = [100.0, 10.0], 
        Tmin = [0.01, 0.0001], 
        gamma_schedule = [[0.1,0.1,0.1],[0.1,0.1]],  
        gamma_steady = [0.8, 0.8], 
        # Regularization
        perturb_param = [0.1, 0.01],
        effective_neighborhood = [0.1, 0.01], 
        py_cut = [0.001],
        # Termination
        Kmax = [100, 100], 
        timeline_limit = 1000, 
        error_threshold = [0.01, 0.01],
        error_threshold_count = 5,
        # Convergence
        em_convergence = [0.0001, 0.0001], 
        convergence_counter_threshold = [5, 5], 
        stop_separation = [1000-1, 1000-1], 
        convergence_loops = [0, 0], 
        bb_init = [0.9, 0.9], 
        bb_step = [0.9, 0.9], 
        # Bregman Divergence
        Bregman_phi = ['phi_Eucl', 'phi_Eucl'], 
        # Verbose
        plot_curves = True,
        show_domain = False,
        keepscore = True
        ): 
    
    # Data: list of (n) lists of (r) m-vectors, where r=dim is the number of resolutions 
    # Labels: list of (n) labels
    with open(data_file+'.pkl', mode='rb') as file:
        train_data,train_labels,test_data,test_labels = pickle.load(file)
    
    # Resolutions
    train_data = [[td[r] for r in res] for td in train_data]
    test_data = [[td[r] for r in res] for td in test_data]
    depth = len(train_data[0])
    
    print('*** ODA ***')
    
    if len(load_file)>0:
        
        with open(load_file+'.pkl', mode='rb') as file:
            clf = pickle.load(file)
        
        clf.load(train_data=train_data,train_labels=train_labels,
                  # classes=classes, node_id=[1], parent = [],
                  depth=depth,keepscore=keepscore,
                  timeline_limit=timeline_limit,
                  Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,error_threshold=error_threshold,
                  error_threshold_count=error_threshold_count,
                  gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
                  Bregman_phi=Bregman_phi,
                  em_convergence=em_convergence,
                  py_cut=py_cut,
                  convergence_counter_threshold=convergence_counter_threshold,
                  perturb_param=perturb_param,
                  effective_neighborhood=effective_neighborhood,
                  # y_init=y_init,dim=2,dom_min=train_min,dom_max=train_max,
                  convergence_loops=convergence_loops,stop_separation=stop_separation,
                  bb_init=bb_init,bb_step=bb_step)
        
    else:
            
        clf = ODA(train_data=train_data,train_labels=train_labels,
                  # classes=classes, node_id=[1], parent = [],
                  depth=depth,keepscore=keepscore,
                  timeline_limit=timeline_limit,
                  Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,error_threshold=error_threshold,
                  error_threshold_count=error_threshold_count,
                  gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
                  Bregman_phi=Bregman_phi,
                  em_convergence=em_convergence,
                  py_cut=py_cut,
                  convergence_counter_threshold=convergence_counter_threshold,
                  perturb_param=perturb_param,
                  effective_neighborhood=effective_neighborhood,
                  # y_init=y_init,dim=2,dom_min=train_min,dom_max=train_max,
                  convergence_loops=convergence_loops,stop_separation=stop_separation,
                  bb_init=bb_init,bb_step=bb_step)

    clf.fit(test_data=test_data,test_labels=test_labels)
    
    print('*** ODA ***')
    accuTrain = 1-clf.score(train_data, train_labels)
    accuTest = 1-clf.score(test_data, test_labels)
    print(f'Train Accuracy: {accuTrain}') 
    print(f'Test Accuracy: {accuTest}')    
    print(f'Running time: {np.sum(clf.myTime):.1f}s')

    if results_file != '':
        with open(results_file+'.pkl', mode='wb') as file:
            pickle.dump(clf, file) 
            
    if keepscore & plot_curves:
        clf.plot_curve(results_file)

    return clf

#%%

if __name__ == '__main__':
    run()
