#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Import Modules

#%% Import Modules

import pickle
# import math
import numpy as np
import matplotlib.pyplot as plt
# import pylab as pl

import os
import sys
sys.path.append('../../oda/')
from oda_jit import ODA 

# plt.ion()
plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Read results

# Results file
res = [1,1]
results_file = 'demo-11'

# Plotting Options
plot_folder = './domain-11'
plot_fig = False
save_fig = True

if save_fig:
    os.makedirs(plot_folder, exist_ok=True)
    
#%% Load Data

# # Read Dataset
# with open(data_file+'.pkl', mode='rb') as file:
#     # train_data,train_labels,test_data,test_labels,gauss_centers,gauss_sigmas = pickle.load(file)
#     train_data,train_labels,test_data,test_labels = pickle.load(file)

# # Resolutions
# train_data = [[td[r] for r in res] for td in train_data]
# test_data = [[td[r] for r in res] for td in test_data]
# depth = len(train_data[0])
    
with open(results_file+'.pkl', mode='rb') as file:
    clf = pickle.load(file)     

#%% Algorithm Plot

# Colors 
a = 1 # centroids
aa = 0.08 # data
aaa = 0.008 # regions

def proj(x): 
    u = np.array([1,1])
    u = u/np.linalg.norm(u)
    x = np.array(x)
    return np.array(np.dot(u,x))

def proj_back(x):
    u = np.array([1,1])
    u = u/np.linalg.norm(u)
    x = np.array(x)
    return np.array(u*x)

def plt_oda(clf, plot_counter=1, bias=0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')#,
                     #xlim=(-2, 3), ylim=(-5, 5)) #aspect='equal',
    # ax.grid(True)
    plt.xticks([],'')
    plt.yticks([],'')
    
    root = clf
    while root.parent:
        root = root.parent
    
    # Plot Data   
    
    # if len(clf.id)<2:
    #     data = [proj_back(dt[0]) for dt in clf.train_data_arxiv]
    # else:
    #     data = [td[1] for td in clf.train_data_arxiv]
    
    if not clf.parent:
        data = [td[-1] for td in clf.train_data_arxiv]
        labels = clf.train_labels_arxiv
    else:
        data = [td[-1] for td in clf.parent.train_data_arxiv]
        labels = clf.parent.train_labels_arxiv
    
    x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
    y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    ax.plot([x_plot[i][0] for i in range(len(x_plot))],
            [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
    ax.plot([y_plot[i][0] for i in range(len(y_plot))],
            [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    if len(res)>1 and res[0]<res[1] and len(clf.id)<2:
        data = [proj_back(dt[0]) for dt in clf.train_data_arxiv]
    
        labels = clf.train_labels_arxiv
        
        x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
        y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
        
        ax.plot([x_plot[i][0] for i in range(len(x_plot))],
                [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
        ax.plot([y_plot[i][0] for i in range(len(y_plot))],
                [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    # Mesh
    delta = 0.01
    xm = np.arange(0.0, 1.0, delta)
    ym = np.arange(0.0, 1.0, delta)
    
    # Call lower Resolution
    if clf.parent:
        
        clf.plt_counter += 1
        
        # Plot partion at R1 
        centroids = clf.parent.myY[clf.parent.plt_counter-1]
        clabels = clf.parent.myYlabels[clf.parent.plt_counter-1]
        black_data = []
        red_data = []
        tree_data = [[] for i in clf.parent.children]
        for xi in xm:
            for yi in ym:
                di = proj([xi,yi]) if len(centroids[0])<2 else np.array([xi,yi])
                dists = [clf.BregmanD(di,yj)[0] for yj in centroids]
                j = np.argmin(dists)
                tree_data[j].append(np.array([xi,yi]))
                if clabels[j] == 0:
                    black_data.append(np.array([xi,yi]))
                else:
                    red_data.append(np.array([xi,yi]))
                   
        ax.plot([b[0] for b in black_data],
                [b[1] for b in black_data],'ks',alpha=aaa)
        ax.plot([r[0] for r in red_data],
                [r[1] for r in red_data],'rs',alpha=aaa)
        
        # Plot partion at R2 
        for c in range(len(clf.parent.children)):
            if clf.parent.children[c].plt_counter>0:
                centroids = clf.parent.children[c].myY[clf.parent.children[c].plt_counter]
                clabels = clf.parent.children[c].myYlabels[clf.parent.children[c].plt_counter]
                black_data = []
                red_data = []
                for datum in tree_data[c]:
                    di = proj(datum) if len(centroids[0])<2 else datum
                    dists = [clf.BregmanD(di,yj)[0] for yj in centroids]
                    j = np.argmin(dists)
                    if clabels[j] == 0:
                        black_data.append(datum)
                    else:
                        red_data.append(datum)
                           
                ax.plot([b[0] for b in black_data],
                        [b[1] for b in black_data],'ks',alpha=0.5*aaa)
                ax.plot([r[0] for r in red_data],
                        [r[1] for r in red_data],'rs',alpha=1.5*aaa)
    
                # Plot Centroids
                if len(centroids[0])<2:
                    centroids = [proj_back(cd) for cd in centroids]
                for i in range(len(centroids)):
                    # centroids
                    if clabels[i]==0:
                        class_color = 'k'
                    else:
                        class_color = 'r'
                    ax.plot(centroids[i][0],centroids[i][1],color = class_color, marker = 'D',alpha=a)
    else:
        
        # Plot partion at R1 
        centroids = clf.myY[clf.plt_counter]
        clabels = clf.myYlabels[clf.plt_counter]
        black_data = []
        red_data = []
        for xi in xm:
            for yi in ym:
                di = proj([xi,yi]) if len(centroids[0])<2 else np.array([xi,yi])
                dists = [clf.BregmanD(di,yj)[0] for yj in centroids]
                j = np.argmin(dists)
                if clabels[j] == 0:
                    black_data.append(np.array([xi,yi]))
                else:
                    red_data.append(np.array([xi,yi]))
                   
        ax.plot([b[0] for b in black_data],
                [b[1] for b in black_data],'ks',alpha=aaa)
        ax.plot([r[0] for r in red_data],
                [r[1] for r in red_data],'rs',alpha=aaa)
    
        # Plot Centroids
        if len(centroids[0])<2:
            centroids = [proj_back(cd) for cd in centroids]
        for i in range(len(centroids)):
            # centroids
            if clabels[i]==0:
                class_color = 'k'
            else:
                class_color = 'r'
            ax.plot(centroids[i][0],centroids[i][1],color = class_color, marker = 'D',alpha=a)
    
    title=f'Acc.: {1-clf.myTrainError[clf.plt_counter]:.3f}, ' \
          f'Obs.: {clf.myLoops[clf.plt_counter]}, T = {clf.myT[clf.plt_counter]:.3f}, ' \
          f'K = {root.myTreeK[plot_counter]:03d}'
          # f'K = {clf.myK[clf.plt_counter]:03d}, ' 
    # plt.title(f'#{plot_counter+bias:03d}: '+title)
    plt.text(0.5, 0.01, title, horizontalalignment='center',verticalalignment='center', fontsize=10,fontweight='bold')    
    
    if not clf.parent:
        clf.plt_counter += 1
        
    # save figure
    if save_fig:
        fig.savefig(f'./{plot_folder}/{plot_counter+bias:03d}.png', format = 'png')
    
    # show or hide figure
    if plot_fig:
        plt.show()
    
    if not plot_fig:
        plt.close('all')


def plt_data(clf, plot_counter=0, res=0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')
    
    # Plot Data   
    data = [td[-1] for td in clf.train_data_arxiv]
        
    labels = clf.train_labels_arxiv
    
    x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
    y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    ax.plot([x_plot[i][0] for i in range(len(x_plot))],
            [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
    ax.plot([y_plot[i][0] for i in range(len(y_plot))],
            [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    if res==0:
        data = [proj_back(dt[0]) for dt in clf.train_data_arxiv]
    
        x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
        y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
        
        ax.plot([x_plot[i][0] for i in range(len(x_plot))],
                [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
        ax.plot([y_plot[i][0] for i in range(len(y_plot))],
                [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    space = 'Low Resolution' if res==0 else 'High Resolution'
    title=f'#{plot_counter:03d}: Data Space - ' + space 
    plt.title(title)
    
    # save figure
    if save_fig:
        fig.savefig(f'./{plot_folder}/{plot_counter:03d}.png', format = 'png')
    
    # show or hide figure
    if plot_fig:
        plt.show()
    
    if not plot_fig:
        plt.close('all')
        
#%% Plot

if plot_fig or save_fig:        
    
    i=0
    for r in reversed(list(set(res))):
        plt_data(clf=clf, plot_counter=i, res=r)
        i+=1
    
    for i in range(len(clf.timeline)):
        nid = clf.timeline[i]
        node = clf
        for child_id in nid[1:]:
            node=node.children[child_id-1]    
        plt_oda(clf=node, plot_counter=i, bias=len(list(set(res))))

