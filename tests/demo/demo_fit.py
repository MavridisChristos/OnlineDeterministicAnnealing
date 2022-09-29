#!/usr/bin/env python

"""
2D Binary Classification Demo with Gaussian Mixtures
(Multi-Resolution) Online Deterministic Annealing (ODA) for Classification and Clustering

Christos N. Mavridis
Department of Electrical and Computer Engineering, University of Maryland
email: <mavridis@umd.edu>
"""

#%% Import Modules

import pickle
import numpy as np

import os
import sys
sys.path.append('../../oda/')

from oda import ODA 
import demo_domain

#%% Problem Parameters

def run(# Dataset
        data_file = './data/data',
        load_file = '',
        results_file = 'demo',
        # Resolutions
        res=[1,1,1], # 0: lowest
        # Temperature
        Tmax = [0.9,1e-1,1e-2], 
        Tmin = [5*1e-2,1e-2,5*1e-4], 
        gamma_schedule = [[0.1,0.5],[],[]],  
        gamma_steady = [0.8], 
        # Regularization
        lvq=[1],
        regression=False,
        perturb_param = [1e-1],
        effective_neighborhood = [1e-0], 
        py_cut = [1e-6],
        # Termination
        Kmax = [5,5,10], 
        timeline_limit = 1e3, 
        error_threshold = [0.0],
        error_threshold_count = [2],
        # Convergence
        em_convergence = [1e-1], 
        convergence_counter_threshold = [5], 
        stop_separation = [1e6-1], 
        convergence_loops = [0], 
        bb_init = [0.9], 
        bb_step = [0.9], 
        # Bregman Divergence
        Bregman_phi = ['phi_Eucl'], 
        # Verbose
        plot_curves = True,
        show_domain = True,
        keepscore = 3,
        jit = True
        ): 
    
    # Data: list of (n) lists of (r) m-vectors, where r=dim is the number of resolutions 
    # Labels: list of (n) labels
    with open(data_file+'.pkl', mode='rb') as file:
        train_data,train_labels,test_data,test_labels = pickle.load(file)
    
    # Resolutions
    train_data = [[td[r] for r in res] for td in train_data]
    test_data = [[td[r] for r in res] for td in test_data]
    
    # For clustering
    # train_labels = [0 for i in range(len(train_labels))] 
    # test_labels = [0 for i in range(len(test_labels))] 
    
    print('*** ODA ***')
    
    if len(load_file)>0:
        
        with open(load_file+'.pkl', mode='rb') as file:
            clf = pickle.load(file)
        
        clf.load()
        
    else:
            
        clf = ODA(
        # Data
        train_data=train_data,
        train_labels=train_labels, 
        # Bregman divergence
        Bregman_phi=Bregman_phi,
        # Termination
        Kmax=Kmax,
        timeline_limit = timeline_limit,
        error_threshold=error_threshold,
        error_threshold_count=error_threshold_count,
        # Temperature
        Tmax=Tmax,
        Tmin=Tmin,
        gamma_schedule=gamma_schedule,
        gamma_steady=gamma_steady,
        # Tree Structure
        node_id=[0],
        parent=None,
        # Regularization
        lvq=lvq, # 0:ODA, 1:soft clustering with no perturbation/merging, 2: LVQ update with no perturbation/merging
        regression=regression,
        py_cut=py_cut,
        perturb_param=perturb_param, 
        effective_neighborhood=effective_neighborhood, 
        # Convergence
        em_convergence=em_convergence, 
        convergence_counter_threshold=convergence_counter_threshold,
        convergence_loops=convergence_loops,
        stop_separation=stop_separation,
        bb_init=bb_init,
        bb_step=bb_step,
        # Verbose
        keepscore=keepscore,
        jit=jit
        )

    # Fit Model
    clf.fit(test_data=test_data,test_labels=test_labels)
    
    # Save Results
    print('*** ODA ***')
    print(f'*** {results_file} ***')
    accuTrain = clf.score(train_data, train_labels)
    accuTest = clf.score(test_data, test_labels)
    print(f'Train Error: {accuTrain}') 
    print(f'Test Error: {accuTest}')    
    print(f'Running time: {np.sum(clf.myTime):.1f}s')

    if results_file != '':
        os.makedirs('./'+results_file+'/', exist_ok=True)
        with open('./'+results_file+'/'+results_file+'.pkl', mode='wb') as file:
            pickle.dump(clf, file) 
            
    if keepscore>2 and plot_curves:
        os.makedirs('./'+results_file+'/', exist_ok=True)
        print('*** Plotting Performance Curve ***')
        clf.plot_curve('./'+results_file+'/'+results_file,show=False,save = True)

    if show_domain:
        print('*** Plotting Domain ***')
        if len(res)<2:
            demo_domain.show(clf=clf, res=res, plot_folder='./'+results_file+'/domain/')
        else:
            demo_domain.show_resolutions(clf=clf, res=res, plot_folder='./'+results_file+'/domain/')

    print('*** All Done ***')
    
    return clf

#%%

if __name__ == '__main__':
    clf = run()

#%%

"""
2D Binary Classification Demo with Gaussian Mixtures
(Multi-Resolution) Online Deterministic Annealing (ODA) for Classification and Clustering

Christos N. Mavridis
Department of Electrical and Computer Engineering, University of Maryland
email: <mavridis@umd.edu>
"""