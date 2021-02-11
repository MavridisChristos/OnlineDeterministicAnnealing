#!/usr/bin/env python

"""
Online Deterministic Annealing (ODA) for Classification and Clustering
Christos N. Mavridis & John S. Baras,
Department of Electrical and Computer Engineering, University of Maryland
email: <mavridis@umd.edu>
"""

#%% Import Modules

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import sys
sys.path.append('./')
from oda_class import ODA 

plt.close('all')
np.random.seed(13)

#%% Problem Parameters

# Dataset
data_folder = '../' 
data_file = 'data/data_gaussians-1.pkl'
results_file = 'results/results_gaussians.pkl'

#%% ODA PARAMETERS

# Temperature
Kmax = 100*2 
Tmax = 100.0 
Tmin = 0.001 
gamma_schedule = [0.1,0.1,0.1]  
gamma_steady = 0.8 # T'=gamma*T

# Bregman Divergence
Bregman_phi = 'phi_Eucl' # 'phi_Eucl', 'phi_KL'

# EM Convergence
em_convergence = 0.0001 
perturb_param = 0.01 
effective_neighborhood = 0.0005 

#%% DataSet

# Data: list of n m-vectors 
# Labels: list of n labels

# Read Dataset
with open(data_folder+data_file, mode='rb') as file:
    train_data,train_labels,test_data,test_labels = pickle.load(file)
    
#%% ODA for Classification: End-to-End
        
clf = ODA(train_data=train_data,train_labels=train_labels,
         Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
         gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
         Bregman_phi=Bregman_phi,
         em_convergence=em_convergence,
         perturb_param=perturb_param,
         effective_neighborhood=effective_neighborhood)

print(f'*** ODA End-to-End ***')
clf.fit(train_data, train_labels)

print(f'*** ODA End-to-End ***')
accuTrain = 1-clf.score(train_data, train_labels)
accuTest = 1-clf.score(test_data, test_labels)
print(f'Train Accuracy: {accuTrain}') 
print(f'Test Accuracy: {accuTest}')    

#%% Look up History and Plot Training Curve

accuTrain = []
accuTest = []
print(f'*** Training History ***')
for i in range(len(clf.myY)):
    y = clf.myY[i]
    ylabels = clf.myYlabels[i]
    clfTrain = clf.predictX(y,ylabels,train_data)
    clfTest = clf.predictX(y,ylabels,test_data)
    train_correct = [1 for j in range(len(clfTrain)) if clfTrain[j]==train_labels[j]]
    test_correct = [1 for j in range(len(clfTest)) if clfTest[j]==test_labels[j]]
    accuTrain.append(len(train_correct)/len(train_data))
    accuTest.append(len(test_correct)/len(test_data))
    print(f'Samples: {clf.myLoops[i]}, ' +
                          f'T = {clf.myT[i]:.4f}, K = {clf.myK[i]}')
    print(f'Train Accuracy: {accuTrain[-1]}') 
    print(f'Test Accuracy: {accuTest[-1]}')    

# Plot Training Curve
txt_x = 0.01
txt_y = 0.03
fig=plt.figure(figsize=(10,7))
x = clf.myLoops
y = accuTrain
plt.plot(x,y,label='Training', 
                 color='b', marker='s',linestyle='solid', 
                 linewidth=8, markersize=12)
y = accuTest
plt.plot(x,y,label='Testing', 
                 color='r', marker='D',linestyle='solid', 
                 linewidth=8, markersize=12)
for i, j in zip(np.arange(len(x)),np.arange(len(y))):
            pl.text(x[i]+txt_x, y[j]+txt_y, str(clf.myK[i]), color='r', fontsize=22,
                    fontweight='bold')    
plt.title('Training History', fontsize = 22)
plt.ylabel('Accuracy %', fontsize = 22)
plt.xlabel('# Samples Observed', fontsize = 22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.grid(color='gray', linestyle='-', linewidth=2, alpha = 0.3)
plt.legend(prop={'size': 32})
fig.savefig(data_folder+'results/'+'training_curve.png',format='png')

# Plot in Data space
for i in range(len(clf.myK)):
    clf.plot_dspace_X(y=clf.myY[i], ylabels = clf.myYlabels[i], data=train_data, labels=train_labels,  
                   title=f'#{i}: Accuracy: {accuTrain[i]:.3f}, ' +
                   f'{clf.myLoops[i]} samples, T = {clf.myT[i]:.3f}, K = {clf.myK[i]}', 
                   plot_folder=data_folder+'results/',
                   plot_counter = i)
    

#%% ODA for Unsupervised Learning
        
no_labels = [0 for i in range(len(train_labels))] 

clf = ODA(train_data=train_data,train_labels=no_labels,
         Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
         gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
         Bregman_phi=Bregman_phi,
         em_convergence=em_convergence,
         perturb_param=perturb_param,
         effective_neighborhood=effective_neighborhood)

print(f'*** Unsupervised ODA ***')
clf.fit(train_data, no_labels, keepscore=True)

# Plot Training Curve
txt_x = 2.0
txt_y = 2.0
plt.figure(figsize=(10,7))
x = clf.myLoops[1:]
y = clf.myTrainError[1:]
plt.plot(x,y,label='Training', 
                 color='k', marker='D',linestyle='solid', 
                 linewidth=8, markersize=12)
for i, j in zip(np.arange(len(x)),np.arange(len(y))):
            pl.text(x[i]+txt_x, y[j]+txt_y, str(clf.myK[i]), color='k', fontsize=22,
                    fontweight='bold')    
plt.title('Training History', fontsize = 22)
plt.ylabel('Distortion', fontsize = 22)
plt.xlabel('# Samples Observed', fontsize = 22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.grid(color='gray', linestyle='-', linewidth=2, alpha = 0.3)
plt.legend(prop={'size': 32})
plt.show()


#%% ODA for Classification: Trade-off 
        
clf = ODA(train_data=train_data,train_labels=train_labels,
         Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
         gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
         Bregman_phi=Bregman_phi,
         em_convergence=em_convergence,
         perturb_param=perturb_param,
         effective_neighborhood=effective_neighborhood)

print(f'*** ODA Trade-off ***')
clf.fit(train_data, train_labels, test_data, test_labels, keepscore=True)

print(f'*** ODA Trade-off: Smallest K for >97% ***')
idx = [i for i in range(len(clf.myTestError)) if clf.myTestError[i]<0.03]
errsrt = np.argsort([clf.myK[i] for i in idx])
j = idx[errsrt[0]]
print(f'Samples: {clf.myLoops[j]}, K: {clf.myK[j]}, T: {clf.myT[j]:.4f}')
print(f'Train Accuracy: {1-clf.myTrainError[j]:.3f}') 
print(f'Test Accuracy: {1-clf.myTestError[j]:.3f}')    

#%% ODA Progressively Trained

clf = ODA(train_data=train_data,train_labels=train_labels,
         Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
         gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
         Bregman_phi=Bregman_phi,
         em_convergence=em_convergence,
         perturb_param=perturb_param,
         effective_neighborhood=effective_neighborhood)

print(f'*** ODA Progressively Trained ***')

accuTrain = [0]
accuTest = [0]
while not clf.stopped and accuTest[-1]<0.98:
    
    clf.fit(train_data, train_labels, alltheway=False)
    
    clfTrain = clf.predict(train_data)
    clfTest = clf.predict(test_data)
    train_correct = [1 for j in range(len(clfTrain)) if clfTrain[j]==train_labels[j]]
    test_correct = [1 for j in range(len(clfTest)) if clfTest[j]==test_labels[j]]
    accuTrain.append(len(train_correct)/len(train_data))
    accuTest.append(len(test_correct)/len(test_data))
    print(f'Train Accuracy: {accuTrain[-1]}') 
    print(f'Test Accuracy: {accuTest[-1]}')    
    
#%% ODA with Unknown Online Observations

# Domain Estimation; here known
train_min = np.min(np.min(train_data,0)) #-0.1
train_max = np.max(np.max(train_data,0)) #+0.1
train_domain = train_max-train_min 
classes = list(np.unique(train_labels))

print(f'*** ODA Online Unknown Observations ***')

clf = ODA(classes=classes,Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
         gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
         Bregman_phi=Bregman_phi,
         em_convergence=em_convergence,
         perturb_param=perturb_param,
         effective_neighborhood=effective_neighborhood,
         y_init=[],dim=2,dom_min=train_min,dom_max=train_max)

last_T = clf.lowest_T() 
                
while not clf.stopped:
    
    # Observe a sample; here pick next from dataset
    idx = clf.current_sample%len(train_data)
    datum = train_data[idx][:]
    datum_label = train_labels[idx]
    clf.current_sample += 1
    
    # Train ODA with one sample
    clf.train_step(datum, datum_label)
    
    if clf.lowest_T() < last_T: # if converged to lower temperature
        
        nT = clf.lowest_T()
        nY = clf.leaf_Y().copy()
        nYlabels = clf.leaf_Ylabels().copy()
        nK = len(nY)        
        print(f'Samples: {clf.current_sample}(+{clf.current_sample-clf.last_sample}): ' +
                  f'T = {nT:.4f}, K = {nK}')        
        clf.myT.append(nT)
        clf.myK.append(nK)
        clf.myY.append(nY)
        clf.myYlabels.append(nYlabels)
        clf.myLoops.append(clf.current_sample)
        clf.last_sample = clf.current_sample
        
        last_T = clf.lowest_T()

#%% ODA All parameters

# Domain Estimation; here known
train_min = np.min(np.min(train_data,0))
train_max = np.max(np.max(train_data,0)) 
train_domain = train_max-train_min 
classes = list(np.unique(train_labels))
y_init = train_min + 0.5*(train_max-train_min)*np.ones(2)

Tsplit = -1 # >0 for tree-structured ODA 
convergence_counter_threshold = 5 # how many convergence decisions mean actual convergence
stop_separation = 1000-1 # after which cycle to stop treating distributions as independent
convergence_loops = 0 # if >0 forces how many loops to be done
bb_init = 0.9 # initial stepsize of stochastic approximation: 1/(bb+1)
bb_step = 0.9 # bb+=bb_step

print(f'*** ODA All Parameters ***')

clf = ODA(classes=classes,Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,Tsplit=-Tsplit,
         gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
         Bregman_phi=Bregman_phi,
         em_convergence=em_convergence,
         convergence_counter_threshold=convergence_counter_threshold,
         perturb_param=perturb_param,
         effective_neighborhood=effective_neighborhood,
         y_init=y_init,dim=2,dom_min=train_min,dom_max=train_max,
         convergence_loops=convergence_loops,stop_separation=stop_separation,
         bb_init=bb_init,bb_step=bb_step)

last_T = clf.lowest_T() 
while not clf.stopped:
    
    # Observe a sample; here pick next from dataset
    idx = clf.current_sample%len(train_data)
    datum = train_data[idx][:]
    datum_label = train_labels[idx]
    clf.current_sample += 1
    
    # Train ODA with one sample
    clf.train_step(datum, datum_label)
    
    if clf.lowest_T() < last_T: # if converged to lower temperature
        
        nT = clf.lowest_T()
        nY = clf.leaf_Y().copy()
        nYlabels = clf.leaf_Ylabels().copy()
        nK = len(nY)        
        print(f'Samples: {clf.current_sample}(+{clf.current_sample-clf.last_sample}): ' +
                  f'T = {nT:.4f}, K = {nK}')        
        clf.myT.append(nT)
        clf.myK.append(nK)
        clf.myY.append(nY)
        clf.myYlabels.append(nYlabels)
        clf.myLoops.append(clf.current_sample)
        clf.last_sample = clf.current_sample
        
        last_T = clf.lowest_T()
        

 
