#!/usr/bin/env python

"""
Online Deterministic Annealing (ODA) for Classification and Clustering
Christos N. Mavridis & John S. Baras,
Department of Electrical and Computer Engineering, University of Maryland
email: <mavridis@umd.edu>
"""

#%% Import Modules

import time
import pickle
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append('./')
from oda_class import ODA 

# plt.ioff() 
plt.close('all')

np.random.seed(13)

#%% Problem Parameters

# Dataset
data_folder = '../' 
data_file = 'data/data_gaussian.pkl'
model_file = 'results/results_gaussian.pkl' # for loading existing model
results_file = 'results/results_gaussian.pkl'

# Method
supervised = True
supervised = False

#%% ODA PARAMETERS

# Bregman Divergence
Bregman_phi = 'phi_Eucl' # 'phi_Eucl', 'phi_KL', 'phi_IS'

# Temperature
Kmax = 100 * 2 
Tmax = 100.0 if Bregman_phi=='phi_Eucl' else 100
Tmin = 0.003 if Bregman_phi=='phi_Eucl' else 0.0005
T_split = -0.2 if Bregman_phi=='phi_Eucl' else 0.01
gamma_steady = 0.8 if Bregman_phi=='phi_Eucl' else 0.8 # T'=gamma*T
gamma_schedule = [0.01,0.1,0.1]  if Bregman_phi=='phi_Eucl' else [0.1,0.1,0.1] 

# EM Convergence
em_convergence = 0.0001 if Bregman_phi=='phi_Eucl' else 0.0001
perturb_param = 0.01 if Bregman_phi=='phi_Eucl' else 0.01
effective_neighborhood = 0.0005 if Bregman_phi=='phi_Eucl' else 0.0005
convergence_counter_threshold = 5 # how many convergence decisions mean actual convergence
stop_separation = 150 - 1 # after which cycle to stop treating distributions as independent
convergence_loops = 0 # if >0 forces how many loops to be done

# SA stepsizes
bb_init = 0.9 # initial stepsize of stochastic approximation: 1/(bb+1)
bb_step = 0.9 # bb+=bb_step

# Intial conditions
load_initial = True
load_initial = False
li = -1 # -1 for last
bad_initial = False

#%% DataSet

# Data: list of n vectors # NOT [n x m]: n data x m features
# Labels: list of n labels

# Read Dataset
with open(data_folder+data_file, mode='rb') as file:
    train_data,train_labels,test_data,test_labels = pickle.load(file)
    
# Domain Info
train_min = np.min(np.min(train_data,0)) #-0.1
train_max = np.max(np.max(train_data,0)) #+0.1
train_domain = train_max-train_min 

train_samples = len(train_data)
test_samples = len(test_data) 

if not supervised:
    train_labels = list(np.zeros(len(train_data)))
    test_labels = list(np.zeros(len(test_data)))

#%% SVM Classification

if supervised:    
    clf = svm.SVC(kernel='linear') # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    clf.fit(train_data, train_labels)
    
    svmTrain = clf.predict(train_data)
    if test_samples > 0:
        svmTest = clf.predict(test_data)
    
    train_correct = [1 for i in range(len(svmTrain)) if svmTrain[i]==train_labels[i]]
    if test_samples > 0:
        test_correct = [1 for i in range(len(svmTest)) if svmTest[i]==test_labels[i]]
    
    print(f'******************** SVM ********************')
    print(f'Train Accuracy: {(len(train_correct))/train_samples}') 
    if test_samples > 0:
        print(f'Test Accuracy: {(len(test_correct))/test_samples}')    

#%% Neural Network Classification

if supervised:
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(50), random_state=1) # 'sgd', 'adam', 'lbfgs'
    
    clf.fit(train_data, train_labels)
    
    nnTrain = clf.predict(train_data)
    if test_samples > 0:
        nnTest = clf.predict(test_data)
    
    train_correct = [1 for i in range(len(nnTrain)) if nnTrain[i]==train_labels[i]]
    if test_samples > 0:
        test_correct = [1 for i in range(len(nnTest)) if nnTest[i]==test_labels[i]]
    
    print(f'******************** NN ********************')
    print(f'Train Accuracy: {(len(train_correct))/train_samples}') 
    if test_samples > 0:
        print(f'Test Accuracy: {(len(test_correct))/test_samples}')    

#%% Initalize ODA 

# Labels
if supervised:
    classes = list(np.unique(train_labels)) 
else:
    classes = [0]

# Initial Conditions
y_init = train_min + 0.5*train_domain*np.ones_like(train_data[0])

# Scale Hyper Parameters
dim = len(train_data[0])
scale_p = train_domain**2 if Bregman_phi == 'phi_Eucl' else train_domain

Tmax = Tmax*dim*scale_p
Tmin = Tmin*dim*scale_p
T_split = T_split*dim*scale_p

em_convergence = em_convergence*dim*scale_p
perturb_param = perturb_param*dim*scale_p 
effective_neighborhood = effective_neighborhood*dim*scale_p

# If predefined number of loops
convergence_loops = np.ceil(convergence_loops * train_samples)


#%% Define Class

myK = []
myT = []
myY = []
myYlabels = []
myTrainError = []
myTestError = []
myLoops = []    
myTime = []

# myoda.find_default_parameters()
# myoda.find_default_parameters(data=train_data, datalabels=train_labels)

if load_initial:
    with open(data_folder+model_file, mode='rb') as file:
        myK,myT,myY,myYlabels,myTrainError,myTestError,myLoops,myTime,myOda = pickle.load(file)
    # nY = myY[li].copy()
    # nPy = myPy[li].copy()
    # nYlabels = myYlabels[li].copy()
    # nT = myT[li]
    # myoda.em_steps = myLoops[li]
    # myoda.overwrite_codevectors(new_y=nY,
    #                             new_ylabels=nYlabels,
    #                             new_py=nPy)
    # myoda.T = nT
    # myoda.true_convergence_counter = li if li>0 else len(myT)-1-li
    myoda = myOda
    # myoda.T = myoda.lastT
    myoda.Tmin = Tmin
    myoda.stopped = False
    # myoda.true_convergence_counter -= 1
    # add the rest parameters
    
else:

    myoda = ODA(classes=classes,Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
                     gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
                     em_convergence=em_convergence,
                     convergence_counter_threshold=convergence_counter_threshold,
                     perturb_param=perturb_param,
                     effective_neighborhood=effective_neighborhood,
                     Bregman_phi=Bregman_phi, y_init=y_init,
                     convergence_loops=convergence_loops,
                     bb_init=bb_init,bb_step=bb_step)

if bad_initial:
    bad_init = []
    bad_init_labels = []
    for c in classes:
        bad_init.append(myoda.y[c]-train_domain*np.random.choice([1,-1]))
        bad_init_labels.append(myoda.ylabels[c]) 
    myoda.overwrite_codevectors(new_y=bad_init,
                                new_ylabels=bad_init_labels)
    
#%% Initial Performance
    
print(f'******************** ODA ********************')
print(f'0(+0): T = {myoda.lastT:.4f}, K = {myoda.lastK} [0s]' )

if supervised:
    d_train = myoda.leaf_distortion(data=train_data, labels=train_labels)
    print(f'Train Accuracy: {1-d_train:.4f}')
    if test_samples > 0:
        d_test = myoda.leaf_distortion(data=test_data, labels=test_labels)
        print(f'Test Accuracy: {1-d_test:.4f}')
else:
    d_train = myoda.leaf_distortion(data=train_data, labels=train_labels)
    print(f'Training_D = {d_train:.4f}')
    if test_samples > 0:
        d_test = myoda.leaf_distortion(data=test_data, labels=test_labels)
        print(f'Testing_D = {d_test:.4f}')
    
# Append Results
if not load_initial:
    myK.append(len(myoda.leaf_Y().copy()))
    myT.append(myoda.lowest_T())
    myY.append(myoda.leaf_Y().copy())
    myYlabels.append(myoda.leaf_Ylabels().copy())
    myTrainError.append(d_train)
    if test_samples > 0:
        myTestError.append(d_test)
    myLoops.append(myoda.em_steps)
    myTime.append(0)

#%% ODA

last_T = myoda.lowest_T() 
last_loops = 0 if not load_initial else myLoops[-1]
current_loops = 0 if not load_initial else myLoops[-1]

t0 = time.time()
for k in range(1000): # upper bound on passes of the entire dataset
        
    for i in range(len(train_data)):
    
        datum = train_data[i][:]
        datum_label = train_labels[i]
        current_loops += 1
        
        if not myoda.stopped:
            
            myoda.train_step(datum, datum_label)
        
            if myoda.lowest_T() < last_T:
                
                t1 = time.time()
                nY = myoda.leaf_Y().copy()
                nK = len(nY)
                nYlabels = myoda.leaf_Ylabels().copy()
                nT = myoda.lowest_T()
                
                print(f'{current_loops}(+{current_loops-last_loops}): ' +
                      f'T = {nT:.4f}, K = {nK} [{t1-t0:.0f}s] ')
                if supervised:
                    d_train = myoda.leaf_distortion(data=train_data, labels=train_labels)
                    print(f'Train Accuracy: {1-d_train:.4f}')
                    if test_samples > 0:
                        d_test = myoda.leaf_distortion(data=test_data, labels=test_labels)
                        print(f'Test Accuracy: {1-d_test:.4f}')
                else:
                    d_train = myoda.leaf_distortion(data=train_data, labels=train_labels)
                    print(f'Testing_D = {d_train:.4f}')
                    if test_samples > 0:
                        d_test = myoda.leaf_distortion(data=test_data, labels=test_labels)
                        print(f'Testing_D = {d_test:.4f}')
                    
                # Append Results
                myK.append(nK)
                myT.append(nT)
                myY.append(nY)
                myYlabels.append(nYlabels)
                myTrainError.append(d_train)
                if test_samples > 0:
                    myTestError.append(d_test)
                # myLoops.append(myoda.em_steps)
                myLoops.append(k*len(train_data)+i+1)
                myTime.append(t1-t0)
                
                # stop separation of distributions
                if myoda.true_convergence_counter>stop_separation:
                    myoda.separate=False
                
                # Split
                if nT < T_split and T_split < last_T:
                    
                    myoda.split()
                    # myoda.perturb_all()
                    
                    print(f'******************** Split *******************')
                    print(f'{current_loops}(+{current_loops-last_loops}): ' +
                          f'T = {nT:.4f}, K = {nK}')
                    if supervised:
                        d_train = myoda.leaf_distortion(data=train_data, labels=train_labels)
                        print(f'Train Accuracy: {1-d_train:.4f}')
                        if test_samples > 0:
                            d_test = myoda.leaf_distortion(data=test_data, labels=test_labels)
                            print(f'Test Accuracy: {1-d_test:.4f}')
                    else:
                        d_train = myoda.leaf_distortion(data=train_data, labels=train_labels)
                        print(f'Training_D = {d_train:.4f}')
                        if test_samples > 0:
                            d_test = myoda.leaf_distortion(data=test_data, labels=test_labels)
                            print(f'Testing_D = {d_test:.4f}')
                
                
                last_T = myoda.lowest_T()
                last_loops = current_loops
        

#%% Best Results
                
# idx = np.argmin(myTrainError)
idx = np.argsort(myTestError)[:5] if test_samples > 0 else np.argsort(myTrainError)[:5]
print(f'************ ODA **************')
for i in idx:
    test_string = f'Test Accuracy: {1-myTestError[i]:.3f}, ' if test_samples > 0 else f''
    print(test_string + f'Train Accuracy: {1-myTrainError[i]:.3f}, ' + f'K = {myK[i]}, ' +
           f'Observations: {myLoops[i]}, ' + f'T = {myT[i]:.3f} ' + f'[{myTime[i]:.0f}].')

#%% Save results to file 
    
my_results = [myK,myT,myY,myYlabels,myTrainError,myTestError,myLoops,myTime,myoda]
            
if results_file != '':
    with open(data_folder+results_file, mode='wb') as file:
        pickle.dump(my_results, file) 
 
