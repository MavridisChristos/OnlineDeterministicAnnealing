#!/usr/bin/env python
"""
Demo,
Learning algorithms to compare
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""

#%% Import Modules

import time
import pickle
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from oda_class import ODA 

# from sklearn_lvq import GlvqModel
from sklearn.cluster import KMeans

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

np.random.seed(13)

#%% Problem Parameters

# Dataset

data_folder = '../' 
data_file = 'data/data_blobs'
results_file = 'results/results_blobs'
ext = '.pkl'

# Method
supervised = True
kfolds = 5
# supervised = False

#%% DataSet

# Read Dataset
if supervised:
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(kfolds):
        with open(data_folder+data_file+f'-{i+1}'+ext, mode='rb') as file:
            trd,trl,ted,tel = pickle.load(file)
            train_data.append(trd)
            train_labels.append(trl)
            test_data.append(ted)
            test_labels.append(tel)
else:
    with open(data_folder+data_file+'-0'+ext, mode='rb') as file:
        train_data,train_labels,test_data,test_labels = pickle.load(file)
    
# Domain Info
if supervised:
    train_min = []
    train_max = []
    train_domain = []
    train_samples = []
    test_samples = []
    for k in range(kfolds):
        trmin = np.min(np.min(train_data[k],0)) #-0.1
        trmax = np.max(np.max(train_data[k],0)) #+0.1
        train_min.append(trmin)
        train_max.append(trmax)
        train_domain.append(trmax-trmin) 
        train_samples.append(len(train_data[k]))
        test_samples.append(len(test_data[k])) 
else:
    train_labels = [0 for i in range(len(train_data))]
    test_labels = [0 for i in range(len(test_data))]
    train_min = np.min(np.min(train_data,0))
    train_max = np.max(np.max(train_data,0))
    train_domain = train_max-train_min 
    train_samples = len(train_data)
    test_samples = len(test_data) 

#%% SVM Classification

SVM = []

if supervised:    
    for k in range(kfolds):
        CLFk = []
        clf = svm.SVC(kernel='linear') # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        clf.fit(train_data[k], train_labels[k])
        
        clfTrain = clf.predict(train_data[k])
        if test_samples[k] > 0:
            clfTest = clf.predict(test_data[k])
        
        train_correct = [1 for i in range(len(clfTrain)) if clfTrain[i]==train_labels[k][i]]
        if test_samples[k] > 0:
            test_correct = [1 for i in range(len(clfTest)) if clfTest[i]==test_labels[k][i]]
        
        print(f'******************** SVM ********************')
        print(f'Train Accuracy: {(len(train_correct))/train_samples[k]}') 
        if test_samples[k] > 0:
            print(f'Test Accuracy: {(len(test_correct))/test_samples[k]}')    
            
        CLFk.append(len(train_correct)/train_samples[k])
        CLFk.append(f1_score(train_labels[k], clfTrain, average='macro'))
        if test_samples[k] > 0:
            CLFk.append(len(test_correct)/test_samples[k])
            CLFk.append(f1_score(test_labels[k], clfTest, average='macro'))
        SVM.append(CLFk)
    
#%% Neural Network Classification

NN = []

if supervised:    
    for k in range(kfolds):
        CLFk = []
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(50), random_state=1) # 'sgd', 'adam', 'lbfgs'
        clf.fit(train_data[k], train_labels[k])
        
        clfTrain = clf.predict(train_data[k])
        if test_samples[k] > 0:
            clfTest = clf.predict(test_data[k])
        
        train_correct = [1 for i in range(len(clfTrain)) if clfTrain[i]==train_labels[k][i]]
        if test_samples[k] > 0:
            test_correct = [1 for i in range(len(clfTest)) if clfTest[i]==test_labels[k][i]]
        
        print(f'******************** NN ********************')
        print(f'Train Accuracy: {(len(train_correct))/train_samples[k]}') 
        if test_samples[k] > 0:
            print(f'Test Accuracy: {(len(test_correct))/test_samples[k]}')    
            
        CLFk.append(len(train_correct)/train_samples[k])
        CLFk.append(f1_score(train_labels[k], clfTrain, average='macro'))
        if test_samples[k] > 0:
            CLFk.append(len(test_correct)/test_samples[k])
            CLFk.append(f1_score(test_labels[k], clfTest, average='macro'))
        NN.append(CLFk)
    
#%% Random Forests Classification

RF = []

if supervised:    
    for k in range(kfolds):
        CLFk = []
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(train_data[k], train_labels[k])
        
        clfTrain = clf.predict(train_data[k])
        if test_samples[k] > 0:
            clfTest = clf.predict(test_data[k])
        
        train_correct = [1 for i in range(len(clfTrain)) if clfTrain[i]==train_labels[k][i]]
        if test_samples[k] > 0:
            test_correct = [1 for i in range(len(clfTest)) if clfTest[i]==test_labels[k][i]]
        
        print(f'******************** RF ********************')
        print(f'Train Accuracy: {(len(train_correct))/train_samples[k]}') 
        if test_samples[k] > 0:
            print(f'Test Accuracy: {(len(test_correct))/test_samples[k]}')    
            
        CLFk.append(len(train_correct)/train_samples[k])
        CLFk.append(f1_score(train_labels[k], clfTrain, average='macro'))
        if test_samples[k] > 0:
            CLFk.append(len(test_correct)/test_samples[k])
            CLFk.append(f1_score(test_labels[k], clfTest, average='macro'))
        RF.append(CLFk)

#%% ODA

YODA = []

if supervised:    
    for k in range(kfolds):
        CLFk = []
        clf = ODA(train_data=train_data[k],train_labels=train_labels[k],
                 classes=[0],Kmax=200,Tmax=100,Tmin=0.0005,Tsplit=-1,
                 gamma_schedule=[0.01,0.1,0.1],gamma_steady=0.8,
                 em_convergence=0.0001,convergence_counter_threshold=5,
                 perturb_param=0.01,effective_neighborhood=0.0005,
                 Bregman_phi='phi_Eucl', y_init=[],dim=1,dom_min=0,dom_max=1,
                 convergence_loops=0,stop_separation=1000,
                 bb_init=0.9,bb_step=0.9)
        
        clf.fit(train_data[k], train_labels[k],keepscore=True,alltheway=False)
        
        clfTrain = clf.predict(train_data[k])
        if test_samples[k] > 0:
            clfTest = clf.predict(test_data[k])
        
        train_correct = [1 for i in range(len(clfTrain)) if clfTrain[i]==train_labels[k][i]]
        if test_samples[k] > 0:
            test_correct = [1 for i in range(len(clfTest)) if clfTest[i]==test_labels[k][i]]
        
        print(f'******************** ODA ********************')
        print(f'Train Accuracy: {(len(train_correct))/train_samples[k]}') 
        if test_samples[k] > 0:
            print(f'Test Accuracy: {(len(test_correct))/test_samples[k]}')    
            
        CLFk.append(len(train_correct)/train_samples[k])
        CLFk.append(f1_score(train_labels[k], clfTrain, average='macro'))
        if test_samples[k] > 0:
            CLFk.append(len(test_correct)/test_samples[k])
            CLFk.append(f1_score(test_labels[k], clfTest, average='macro'))
        YODA.append(CLFk)
    
        
        
#%% ODA
    
MYODA = []

if supervised:
    for k in range(kfolds):
        ODAk = []
        odaAtrain = []
        odaF1train = []
        odaAtest = []
        odaF1test = []
        odaObs = []
        odaK = []
        
        print(f'******************** ODA ********************')
        myoda = ODA(train_data=train_data[k],train_labels=train_labels[k],
                     classes=[0],Kmax=100,Tmax=100,Tmin=0.0005,
                     gamma_schedule=[0.1,0.1,0.001],gamma_steady=0.8, #0.1,0.1,0.1
                     em_convergence=0.00001,convergence_counter_threshold=5,
                     perturb_param=0.01,effective_neighborhood=0.005,
                     Bregman_phi='phi_Eucl', y_init=[0.5],
                     convergence_loops=0,
                     bb_init=0.9,bb_step=0.9)        
        # myoda.separate=False
        
        clfTrain = myoda.predict(train_data[k])
        if test_samples[k] > 0:
            clfTest = myoda.predict(test_data[k])
        train_correct = [1 for i in range(len(clfTrain)) if clfTrain[i]==train_labels[k][i]]
        if test_samples[k] > 0:
            test_correct = [1 for i in range(len(clfTest)) if clfTest[i]==test_labels[k][i]]
        odaAtrain.append(len(train_correct)/train_samples[k])
        odaF1train.append(f1_score(train_labels[k], clfTrain, average='macro'))
        if test_samples[k] > 0:
            odaAtest.append(len(test_correct)/test_samples[k])
            odaF1test.append(f1_score(test_labels[k], clfTest, average='macro'))
        odaObs.append(0)
        odaK.append(myoda.K)
        
        last_T = myoda.lowest_T()
        last_loops = 0 
        current_loops = 0 
        t0 = time.time()
        for kk in range(1000): 
            for i in range(len(train_data[k])):
                datum = train_data[k][i][:]
                datum_label = train_labels[k][i]
                current_loops += 1
                if not myoda.stopped:
                    myoda.train_step(datum, datum_label)
                    # print(f'K={myoda.K}, T = {myoda.lowest_T()}, lastT = {last_T}, A = {myoda.lowest_T()==myoda.T}')
                    if myoda.lowest_T() < last_T:
                        t1 = time.time()
                        nY = myoda.leaf_Y().copy()
                        nK = len(nY)
                        nYlabels = myoda.leaf_Ylabels().copy()
                        nT = myoda.lowest_T()
                        
                        clfTrain = myoda.predict(train_data[k])
                        if test_samples[k] > 0:
                            clfTest = myoda.predict(test_data[k])
                        
                        train_correct = [1 for i in range(len(clfTrain)) if clfTrain[i]==train_labels[k][i]]
                        if test_samples[k] > 0:
                            test_correct = [1 for i in range(len(clfTest)) if clfTest[i]==test_labels[k][i]]
                        
                        print(f'{current_loops}(+{current_loops-last_loops}): ' +
                      f'T = {nT:.4f}, K = {nK} [{t1-t0:.0f}s] ')
                        print(f'Train Accuracy: {(len(train_correct))/train_samples[k]}') 
                        if test_samples[k] > 0:
                            print(f'Test Accuracy: {(len(test_correct))/test_samples[k]}')    
            
                        odaAtrain.append(len(train_correct)/train_samples[k])
                        odaF1train.append(f1_score(train_labels[k], clfTrain, average='weighted'))
                        if test_samples[k] > 0:
                            odaAtest.append(len(test_correct)/test_samples[k])
                            odaF1test.append(f1_score(test_labels[k], clfTest, average='weighted'))
                        odaObs.append(current_loops)
                        odaK.append(nK)
                        
                        last_T = myoda.lowest_T()
                        last_loops = current_loops
                        
        ODAk.append(odaAtrain)
        ODAk.append(odaF1train)
        if test_samples[k] > 0:
            ODAk.append(odaAtest)
            ODAk.append(odaF1test)    
        ODAk.append(odaObs)
        ODAk.append(odaK)
        MYODA.append(ODAk)



#%%
#%%
#%% Unsupervised
        
        
        
        
#%% Distortion
        
def distortion(centers,train_data):
    dd = 0.0
    for d in train_data:
        dists = [np.dot(d-c,d-c) for c in centers]
        dd+= np.min(dists)
    return dd

#%% DA
    
DA = []

if not supervised:
    
    daD = []
    daObs = []
    daK = []
    
    print(f'********************DA ********************')
    clr = ODA(train_data=train_data,train_labels=train_labels,
                     classes=[0],Kmax=100,Tmax=1,Tmin=0.0004,
                     gamma_schedule=[0.01,0.1],gamma_steady=0.8, #0.1,0.1,0.1
                     em_convergence=0.00001,convergence_counter_threshold=5,
                     perturb_param=0.01,effective_neighborhood=0.005,
                     Bregman_phi='phi_Eucl', y_init=[0.5],
                     convergence_loops=0,
                     bb_init=0.9,bb_step=0.9)        
    
    daD.append(distortion(clr.y.copy(),train_data))
    daObs.append(0)
    daK.append(clr.K)            
    
    cnt = 0
    for i in range(1000): 
        if not clr.stopped:   
            clr.train_batch(train_data,[0 for i in range(len(train_labels))])
            if clr.converged:
                
                print(f'#{clr.em_steps}(+{clr.sa_steps}): ' +
                      f'T = {clr.T/clr.gamma:.4f}, K = {clr.K}')
                # d_train = clr.leaf_distortion(data=train_data, labels=[0 for i in range(len(train_labels))])
                # print(f'Training_D = {d_train:.4f}')
            
                obs = clr.em_steps
                centers = clr.y.copy()
                daD.append(distortion(centers,train_data))
                daObs.append(obs)
                daK.append(clr.K)

    daD.pop()
    daObs.pop()
    daK.pop()
    clustersK = list(np.unique(daK))
    
    DA.append(daD)
    DA.append(daObs)
    DA.append(daK)

#%% k-means
        
KMEANS = []

if not supervised:
    kmeansD = []
    kmeansObs = []
    kmeansK = []
    print(f'******************** K-means ********************')
    kmeansD.append(distortion(train_data[np.random.choice(range(len(train_data)))],train_data))
    kmeansObs.append(0)
    kmeansK.append(1)
    for k in clustersK:
        clr = KMeans(n_clusters=k, init='random', 
                     n_init=1, max_iter=3000, algorithm='full')
        clr.fit(train_data)
        
        obs = train_samples * clr.n_iter_
        centers = clr.cluster_centers_
        
        kmeansD.append(distortion(centers,train_data))
        kmeansObs.append(obs)
        kmeansK.append(k)
    
    KMEANS.append(kmeansD)
    KMEANS.append(list(np.cumsum(kmeansObs)))
    KMEANS.append(kmeansK)

#%% sVQ

VQ = []

if not supervised:
    vqD = []
    vqObs = []
    vqK = []
    
    print(f'******************** VQ ********************')
    
    vqD.append(distortion(train_data[np.random.choice(range(len(train_data)))],train_data))
    vqObs.append(0)
    vqK.append(1)
        
    for k in clustersK:
        
        y_init = train_data[np.random.choice(range(len(train_data)))]
        # y_init = train_min + 0.5*train_domain*np.ones_like(train_data[0]) 
        clr = ODA(classes=[0],Kmax=100,Tmax=100,Tmin=0.005,
                 gamma_schedule=[0.01,0.1,0.1],gamma_steady=0.8,
                 em_convergence=0.00001,convergence_counter_threshold=5,
                 perturb_param=0.00001,effective_neighborhood=0.0005,
                 Bregman_phi='phi_Eucl', y_init=y_init,
                 convergence_loops=0,
                 bb_init=0.9,bb_step=0.9)  
        new_y = []
        new_ylabels = []
        for kk in range(k):
            new_y.append(train_data[np.random.choice(range(len(train_data)))])
            # y_init * (1 + 0.1*train_domain*2*(np.random.rand(len(train_data[0]))-0.5)))
            # np.random.uniform(low=train_min, high=train_max, size=len(train_data[0]))
            new_ylabels.append(0) 
        clr.overwrite_codevectors(new_y=new_y,
                                new_ylabels=new_ylabels)
    
        cnt = 0
        for j in range(1000): 
            for i in range(len(train_data)):
                datum = train_data[i][:]
                datum_label = train_labels[i]
                if not clr.converged:
                    clr.train_lvq(datum, datum_label)     
                    cnt+=1
                else:
                    break
        
        obs = cnt
        centers = clr.y
        
        vqD.append(distortion(centers,train_data))
        vqObs.append(obs)
        vqK.append(k)
        
    VQ.append(vqD)
    VQ.append(list(np.cumsum(vqObs)))
    VQ.append(vqK)                

#%% ODA
    
MYODAu = []

if not supervised:
    
    odaD = []
    odaObs = []
    odaK = []
    
    print(f'******************** ODA ********************')
    myoda = ODA(train_data=train_data,train_labels=train_labels,
                 classes=[0],Kmax=100,Tmax=100,Tmin=0.0004,
                 gamma_schedule=[0.001,0.1,0.1],gamma_steady=0.8,
                 em_convergence=0.0001,convergence_counter_threshold=5,
                 perturb_param=0.01,effective_neighborhood=0.0005,
                 Bregman_phi='phi_Eucl', y_init=[0.5],
                 convergence_loops=0,
                 bb_init=0.9,bb_step=0.9)       
    
    odaD.append(distortion(myoda.y.copy(),train_data))
    odaObs.append(0)
    odaK.append(myoda.K)
    
    last_T = myoda.lowest_T() 
    last_loops = 0 
    current_loops = 0 
    t0 = time.time()
    for kk in range(1000): 
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
                    nT = myoda.lowest_T()
                    print(f'{current_loops}(+{current_loops-last_loops}): ' +
                  f'T = {nT:.4f}, K = {nK} [{t1-t0:.0f}s] ')
                    obs = current_loops
                    centers = nY
                    odaD.append(distortion(centers,train_data))
                    odaObs.append(obs)
                    odaK.append(nK)
                    
                    last_T = myoda.lowest_T()
                    last_loops = current_loops
                    
    MYODAu.append(odaD)
    MYODAu.append(odaObs)
    MYODAu.append(odaK)

        
#%% Save results to file 
    
if supervised:
    my_results = [SVM,NN,RF,MYODA]       
    if results_file != '':
        with open(data_folder+results_file+ext, mode='wb') as file:
            pickle.dump(my_results, file) 
else:
    my_results = [DA,KMEANS,VQ,MYODAu]       
    if results_file != '':
        with open(data_folder+results_file+'-u'+ext, mode='wb') as file:
            pickle.dump(my_results, file) 
    

#%% GLVQ

# GlvqD = []        
# GlvqObs = []

# if supervised:
    
#     for k in [2,10,16]:
        
#         clr = GlvqModel(prototypes_per_class=k)
#         clr.fit(train_data,train_labels)
        
#         obs = train_samples * clr.n_iter_
#         # centers = clr.cluster_centers_
        
#         glvqTrain = clf.predict(train_data)
#         if test_samples > 0:
#             glvqTest = clf.predict(test_data)
        
#         train_correct = [1 for i in range(len(glvqTrain)) if glvqTrain[i]==train_labels[i]]
#         if test_samples > 0:
#             test_correct = [1 for i in range(len(glvqTest)) if glvqTest[i]==test_labels[i]]
        
#         print(f'******************** GLVQ ********************')
#         print(f'Train Accuracy: {(len(train_correct))/train_samples}') 
#         if test_samples > 0:
#             print(f'Test Accuracy: {(len(test_correct))/test_samples}') 
        
#         KMeansD.append((len(train_correct))/train_samples)
#         KMeansObs.append(obs)