"""
(Tree-Structured) Online Deterministic Annealing for Classification and Clustering
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""

# Parameter values reflect 1D features in [0,1].

#%% Import Modules

import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pylab as pl
# import concurrent.futures
plt.ioff()
    
np.random.seed(13)

#%% The Class
    
class ODA:
    
    # init
    ###########################################################################
    def __init__(self,
                 # Data
                 train_data=[],
                 train_labels=[], 
                 classes=[0], 
                 dim=[1],
                 dom_min=[0],
                 dom_max=[1],
                 y_init=[], 
                 # Tree Level - Resolution
                 node_id=[1],
                 depth=1,
                 parent=None,
                 # Temperature
                 Tmax=[100.0],
                 Tmin=[0.001],
                 gamma_schedule=[[0.1,0.1,0.1]],
                 gamma_steady=[0.8],
                 # Regularization
                 perturb_param=[0.1], 
                 effective_neighborhood=[0.1], 
                 py_cut=[0.001],
                 # Termination
                 Kmax=[100],
                 timeline_limit = 100000,
                 error_threshold=[0.01],
                 error_threshold_count=5,
                 # Convergence
                 em_convergence=[0.001], 
                 convergence_counter_threshold=[5],
                 convergence_loops=[0],
                 stop_separation=[1000000000],
                 bb_init=[0.9],
                 bb_step=[0.9],
                 # Bregman divergence
                 Bregman_phi=['phi_KL'],
                 # Verbose
                 keepscore=True
                 ):
        
        # Archive
        self.train_data_arxiv = train_data.copy()
        self.train_labels_arxiv = train_labels.copy()
        self.Kmax_arxiv = Kmax.copy()
        self.Tmax_arxiv = Tmax.copy()
        self.Tmin_arxiv = Tmin.copy()
        self.error_threshold_arxiv = error_threshold.copy()
        self.gamma_schedule_arxiv = gamma_schedule.copy()
        self.gamma_steady_arxiv = gamma_steady.copy()
        self.Bregman_phi_arxiv = Bregman_phi.copy()
        self.em_convergence_arxiv = em_convergence.copy()
        self.perturb_param_arxiv = perturb_param.copy()
        self.effective_neighborhood_arxiv = effective_neighborhood.copy()
        self.py_cut_arxiv = py_cut.copy()
        self.convergence_loops_arxiv = convergence_loops.copy()
        self.convergence_counter_threshold_arxiv = convergence_counter_threshold.copy()
        self.bb_init_arxiv = bb_init.copy()
        self.bb_step_arxiv = bb_step.copy()
        self.stop_separation_arxiv = stop_separation.copy()
        self.dim_arxiv = dim.copy()
        self.dom_min_arxiv = dom_min.copy()
        self.dom_max_arxiv = dom_max.copy()
        
        # Tree & Resolution parameters
        self.id = node_id.copy()
        self.resolution = len(node_id)-1
        self.depth = depth # or any parameter list
        self.keepscore = keepscore
                
        # self.test_data = test_data
        # self.test_labels = test_labels
        
        # Bregman Divergence parameters
        self.Bregman_phi = Bregman_phi[self.resolution] # 'phi_Eucl', 'phi_KL', 'phi_IS'
        # self.phi = eval('self.'+ self.Bregman_phi)
        
        # Initial Values from dataset if available 
        if len(train_data)>0:
            train_samples = len(train_data)    
            classes = list(np.unique(train_labels))
            # Domain Info
            self.dim_arxiv = [len(td) for td in train_data[0]]
            self.dom_min_arxiv = [np.min(np.min([td[r] for td in train_data],0)) for r in range(len(train_data[0]))]
            self.dom_max_arxiv = [np.max(np.max([td[r] for td in train_data],0)) for r in range(len(train_data[0]))]
            # Initial Conditions
            # y_init = train_min + 0.5*train_domain*np.ones_like(train_data[0][0])
            if not len(y_init)> 0:
                y_init = train_data[np.random.choice(range(len(train_data)))][self.resolution]
            # If predefined number of loops
            self.convergence_loops = np.ceil(convergence_loops[self.resolution] * train_samples)
        else:
            self.convergence_loops = convergence_loops[self.resolution]
        
        # Domain info
        self.dim = self.dim_arxiv[self.resolution]
        self.dom_min = self.dom_min_arxiv[self.resolution]
        self.dom_max = self.dom_max_arxiv[self.resolution]   
        
        # Scale Parameters
        if self.Bregman_phi == 'phi_Eucl':
            self.dom = (self.dom_max-self.dom_min)**2   
        elif self.Bregman_phi == 'phi_KL':
            self.dom = (self.dom_max-self.dom_min)
                    
        # Set limit parameters
        self.Kmax = 2*Kmax[self.resolution] # because of the perturbation
        self.Tmax = Tmax[self.resolution]*self.dim*self.dom
        self.Tmin = Tmin[self.resolution]*self.dim*self.dom
        self.gamma_schedule = gamma_schedule[self.resolution]
        self.gamma_steady = gamma_steady[self.resolution]
        if len(self.gamma_schedule)>0:
            self.gamma = self.gamma_schedule[0] 
        else:
            self.gamma = self.gamma_steady# T'=gamma*T
        self.error_threshold = error_threshold[self.resolution]
        
        # EM Convergence parameters
        self.em_convergence = em_convergence[self.resolution]*self.dim*self.dom 
        self.perturb_param = perturb_param[self.resolution]*self.dim*self.dom 
        self.effective_neighborhood = effective_neighborhood[self.resolution]*self.dim*self.dom
        self.py_cut = py_cut[self.resolution]
        # self.practical_zero = 1e-9
        
        # Training parameters
        
        self.convergence_counter_threshold = convergence_counter_threshold[self.resolution]
        # self.run_batch = run_batch # if True, run originial DA algorithm
        # self.batch_size = batch_size # Stochastic/mini-batch Version  
        self.bb_init= bb_init[self.resolution] # initial stepsize of stochastic approximation: 1/(bb+1)
        self.bb_step = bb_step[self.resolution] # 0.5 # bb+=bb_step
        self.separate = True
        self.stop_separation = stop_separation[self.resolution]
    
        # Codevectors
        self.y = []
        self.ylabels = []
        self.py = []
        self.sxpy= []
        self.old_y = []
        
        # Init y
        if not len(y_init)> 0:
            y_init = self.dom_min + 0.5*(self.dom_max-self.dom_min)*np.ones(self.dim)
        for c in classes:
            # self.y.append(y_init)
            self.y.append(y_init* (1 + 0.01*(np.random.rand(len(y_init))-0.5)))
            self.ylabels.append(c) 
            self.py.append(1.0/len(classes))
            self.sxpy.append(self.py[-1]*self.y[-1])
        self.classes = classes
                
        # State Parameters
        self.T = self.Tmax
        self.K = len(self.y)
        self.perturbed = False
        self.converged = False
        self.convergence_counter = 1
        self.true_convergence_counter = -1
        self.trained = False
        self.error_threshold_reached = 0
        self.error_threshold_count = error_threshold_count
        self.timeline_limit = timeline_limit
        
        # Convergence parameters
        self.bb = self.bb_init
        self.sa_steps = 0
        self.em_steps = 0 # obsolete
        self.low_p_warnings = 0

        # Children
        self.children = []
        self.parent = parent
        self.timeline = [self.id]
        
        # Copies for parallel computation, i.e. tree-structure
        # self.lastK = self.K
        # self.lastT = self.T + 1e-6
        # self.lastY = self.y.copy()
        # self.lastPy = self.py.copy()
        # self.lastYlabels = self.ylabels.copy()
    
        # Keep record for each temperature
        self.myK = [self.K]
        self.myT = [self.T]
        self.myY = [self.y.copy()]
        self.myYlabels = [self.ylabels.copy()]
        self.myTrainError = [1]
        self.myTestError = [1]
        self.myLoops = [0]    
        self.myTime = [0]
        self.myTreeK = [self.K]
        
        self.last_sample = 0 # sample at which last convergence
        self.current_sample = 0
        self.plt_counter = 0
        
        self.tik = time.perf_counter()
        self.tok = time.perf_counter()
    
    
    # Fit
    ###########################################################################
    def fit(self,train_data=[],train_labels=[],test_data=[],test_labels=[]):
        
        
        if len(train_data)==0 and len(self.train_data_arxiv)>0:
            train_data = self.train_data_arxiv.copy()
            train_labels = self.train_labels_arxiv.copy()
        fit_sample = 0
        datalen = len(train_data)
        
        self.tik = time.perf_counter()
        # Whlie the entire tree is not trained
        while not self.trained: 
            
            idx = fit_sample % datalen
            datum = train_data[idx].copy()
            datum_label = train_labels[idx]
            
            # Train this or children. train_step() is recursive.
            self.train_step(datum, datum_label, 
                            train_data=train_data, train_labels=train_labels, 
                            test_data=test_data, test_labels=test_labels)
    
            fit_sample += 1
    
    def train_step(self, datum, datum_label=0, 
                   train_data=[],train_labels=[], test_data=[],test_labels=[]):
        
        if not self.trained:
            
            if len(self.children)>0:
                # check if all children trained
                self.check_trained()
                if not self.trained:
                    
                    # find the winner child
                    # y = self.y.copy()
                    # dists = [self.BregmanD(datum[self.resolution],yj)[0] for yj in y] # in my resolution
                    # j = np.argmin(dists)
                    j,_ = _winner(np.array(self.y), np.array(datum[self.resolution]), self.Bregman_phi)
                    
                    # Recursively call train_step(). Each child will use its own resolution of the data.
                    self.children[j].train_step(datum,datum_label,
                                                train_data,train_labels, 
                                                test_data,test_labels)
                else:
                    # if just trained, report time
                    self.tok = time.perf_counter()
                    self.myTime.append(self.tok-self.tik)
            else:
                # Check criteria to stop training self and split, if possible
                stop_K = self.K>self.Kmax
                stop_T = self.T<=self.Tmin
                self.error_threshold_reached = self.check_error_threshold()
                stop_error = self.error_threshold_reached>self.error_threshold_count
                stop_timeline,len_timeline = self.check_timeline_limit()
                if stop_K or stop_T or stop_error or stop_timeline:
                    
                    if stop_K:
                        print('*** Maximum number of codevectors reached. ***')
                    if stop_T:
                        print('*** Minimum temperature reached. ***')
                    if stop_error:
                        print('*** Minimum error reached. ***')
                    if stop_timeline:
                        print('*** Maximum number of converged codevectors reached. ***')
                    
                    if self.resolution+1<self.depth and not stop_K and not stop_error and not stop_timeline:
                        # if reached minimum temperature split
                        self.split(datum)
                        print(f'ID: {self.id}: Trained. Splitting..')
                    else:
                        self.trained = True
                        print(f'ID: {self.id}: Trained')
                        self.tok = time.perf_counter()
                        self.myTime.append(self.tok-self.tik)
                        # Prune siblings that are not updated at all
                        self.prune_siblings()
                        
                # Otherwise, train this cell
                else: 
                    self.current_sample += 1
                    # insert perturbations and initialize SA stepsizes
                    if not self.perturbed:
                        self.perturb()
                        self.converged = False     
                        self.bb = self.bb_init
                        self.sa_steps = 0
                    
                    # Stochastic Approximation Step
                    self.sa_step(datum, datum_label)
                    
                    # Check Convergence
                    self.check_convergence()
                    
                    if self.converged:
                        self.tok = time.perf_counter()
                        
                        # Find effective clusters
                        self.find_effective_clusters() 
                        self.pop_idle_clusters() 
                        
                        self.myT.append(self.T)
                        self.myK.append(self.K) # TO CREATE update_K()
                        self.myY.append(self.y.copy())
                        self.myYlabels.append(self.ylabels.copy())
                        self.myLoops.append(self.current_sample)
                        self.myTime.append(self.tok-self.tik)
                        self.put_in_timeline(self.id.copy())
                        tK = self.treeK()
                        
                        print(f'{len_timeline} -- ID: {self.id}: '+ 
                              f'Samples: {self.current_sample}(+{self.current_sample-self.last_sample}): ' +
                              f'T = {self.T:.4f}, K = {self.K}, treeK = {tK}, [+{self.tok-self.tik:.1f}s]')
                        
                        if self.keepscore:
                            d_train = self.score(train_data,train_labels)
                            self.myTrainError.append(d_train)
                            print(f'Train Error: {d_train:.4f}')
                            if len(test_data)>0:
                                d_test = self.score(test_data,test_labels)
                                self.myTestError.append(d_test)
                                print(f'Test Error: {d_test:.4f}')
                                
                            if d_train<self.error_threshold:
                                print('*** Training Error threshold reached. ***')
                                self.confirm_error_threshold()
                        
                        self.last_sample = self.current_sample
        
                        self.update_T()
                        self.perturbed=False
                        
                        self.tik = time.perf_counter()
    
    def sa_step(self, datum, datum_label):
        
        self.old_y = self.y.copy()
        self.low_p_warnings = 0
        datum = datum[self.resolution]
        
        for i in range(self.K): # for all centroids
            
            self.py[i], self.sxpy[i], self.y[i]= _sa_update(i, 
                np.array(self.y), np.array(self.ylabels), 
                np.array(self.py), np.array(self.sxpy), 
                np.array(datum), np.array(datum_label), self.separate, self.T, 
                self.bb, phi=self.Bregman_phi)
            
            # self.py[i], self.sxpy[i], self.y[i] = self._wtf_sa_step2(i,
            #     self.y.copy(), self.ylabels.copy(), 
            #     self.py.copy(), self.sxpy.copy(), 
            #     datum.copy(), datum_label.copy(), self.separate, self.T, 
            #     self.bb, self.Bregman_phi)
            # d = [self.BregmanD(datum,self.y[k])[0] for k in range(len(self.y))]
            # # d = [_BregmanD(np.array(datum),np.array(self.y[k]), self.Bregman_phi) 
            # #           for k in range(len(self.y))]
            # py = self.py.copy()
            # # For doing separate ODA for each class
            # if self.separate:
            #     py = [self.py[i] if datum_label==self.ylabels[i] else 0 for i in range(len(self.py))]
            # pyx_sum = np.dot(py,[np.exp(-dj/self.T) for dj in d])
            # if pyx_sum == 0: # e.g. if no codevectors of the same class as observation
            #     pyx = 0 # or break
            # else:    
            #     pyx = py[i]*np.exp(-d[i]/self.T)/pyx_sum
            # # SA update
            # self.py[i] = self.py[i] + 1/(self.bb+1)*(pyx - self.py[i])
            # self.sxpy[i] = self.sxpy[i] + 1/(self.bb+1)*(pyx*datum - self.sxpy[i])
            # self.y[i] = self.sxpy[i]/self.py[i]  
            
            # Warning sign
            # if pyx!=pyx: # or pyx<1e-32:
            #     self.low_p_warnings+=1
        
        self.bb += self.bb_step 
        self.sa_steps += 1
        self.em_steps += 1
    
        # Warning
        # if self.low_p_warnings == len(self.y):
        #     # print(f'Number of Idle codevectors: {self.low_p_warnings}/{self.K}')
        #     self.low_p_warnings = 0
        #     print(f'WARNING: Conditional Probabilities too small.')
        #     # +
        #     #               f'You may need to consider scaling the input space,'+
        #     #               f'or using different initial values.')
            
            
    # ODA Functions
    ###########################################################################
    
    def check_convergence(self):
        
        if self.convergence_loops>0: # if predefined number of loops
            if self.sa_steps>=self.convergence_loops:
                if self.convergence_counter > self.convergence_counter_threshold:
                    self.converged = True
                    self.convergence_counter = 1
                    self.true_convergence_counter +=1
                    if self.true_convergence_counter>self.stop_separation:
                        self.separate=False
                else:
                    self.convergence_counter += 1
        else:
            # self.BregmanD(self.old_y[i],self.y[i])[0]
            if np.all([_BregmanD(np.array(self.old_y[i]),np.array(self.y[i]),self.Bregman_phi) < \
                       self.em_convergence * (1+self.bb_init)/(self.bb+1)
                                                        for i in range(self.K)]):   
                if self.convergence_counter > self.convergence_counter_threshold:
                    self.converged = True
                    self.convergence_counter = 1
                    self.true_convergence_counter +=1
                    if self.true_convergence_counter>self.stop_separation:
                        self.separate=False
                else:
                    self.convergence_counter += 1
                    
    def perturb(self):
        # insert perturbations of all effective yi
        for i in reversed(range(self.K)):
            new_yi = self.y[i] + self.perturb_param*2*(np.random.rand(len(self.y[i]))-0.5)
            self.py[i] = self.py[i]/2.0
            self.sxpy[i] = self.py[i]*self.y[i]
            self.y.append(new_yi)
            self.ylabels.append(self.ylabels[i]) 
            self.py.append(self.py[i])
            self.sxpy.append(self.py[i]*new_yi)
        self.K = len(self.y)
        self.perturbed = True
    
    def find_effective_clusters(self):
        i=0
        while i<self.K:
            for j in reversed(np.arange(i+1,self.K)):
                # self.BregmanD(self.y[i],self.y[j])[0]
                if _BregmanD(np.array(self.y[i]),np.array(self.y[j]),self.Bregman_phi)< \
                    self.effective_neighborhood and self.ylabels[i]==self.ylabels[j]:
                    self.py[i] = self.py[i]+self.py[j]
                    self.sxpy[i] = self.y[i]*self.py[i]
                    self.y.pop(j)
                    self.ylabels.pop(j)
                    self.py.pop(j)
                    self.sxpy.pop(j)
                    self.K-=1
            i+=1
    
    def pop_idle_clusters(self):
        i = 0
        while i < len(self.y):
            if self.py[i]<self.py_cut:
                # print('*** Idle Codevector Pruned ***')
                self.pop_codevector(i)
            else:
                i+=1
    
    def prune_siblings(self):
        
        # Solution 1: Prune them
        # i = 0
        # while i < len(self.parent.children):
        #     if self.parent.children[i].current_sample == 0:
        #         print(f'***Idle Sibling Pruned -- ID: {self.parent.children[i].id}***')
                
        #         self.parent.children.pop(i)
        #         self.parent.pop_codevector(i)
        #         # update K,Y,Ylabels
        #         self.parent.myK[-1] = self.parent.K 
        #         self.parent.myY[-1] = self.parent.y
        #         self.parent.myYlabels[-1] = self.parent.ylabels
                                
        #     else:
        #         i+=1
                
        # Solution 2: # Just declare them trained and prune their children
        if self.parent:
            for sibling in self.parent.children:
                if sibling.current_sample == 0:
                    sibling.trained = True
                    # sibling.children=[]
                    print(f'*** Sibling Died: len(sibling.myY) = {len(sibling.myY)} ***')
    
    def split(self,datum):
        
        for i in range(len(self.y)):
            self.children.append( 
                ODA(train_data=[],#self.train_data_arxiv,
                    train_labels=[],#self.train_labels_arxiv,
                    classes=self.classes, 
                    keepscore=self.keepscore,
                    # Child id
                    node_id=self.id+[i+1],depth=self.depth, 
                    parent = self,
                    Kmax=self.Kmax_arxiv,
                    Tmax=self.Tmax_arxiv,Tmin=self.Tmin_arxiv,
                    error_threshold = self.error_threshold_arxiv,
                    error_threshold_count = self.error_threshold_count,
                    gamma_schedule=self.gamma_schedule_arxiv,
                    gamma_steady=self.gamma_steady_arxiv,
                    Bregman_phi=self.Bregman_phi_arxiv,
                    em_convergence=self.em_convergence_arxiv,
                    convergence_counter_threshold=self.convergence_counter_threshold_arxiv,
                    perturb_param=self.perturb_param_arxiv,
                    effective_neighborhood=self.effective_neighborhood_arxiv,
                    py_cut=self.py_cut_arxiv,
                    # Child initial condition
                    y_init=datum[self.resolution+1], #y_init=self.y[i],
                    dim=self.dim_arxiv, dom_min=self.dom_min_arxiv, dom_max=self.dom_max_arxiv,
                    convergence_loops=self.convergence_loops_arxiv,
                    stop_separation=self.stop_separation_arxiv,
                    bb_init=self.bb_init_arxiv,bb_step=self.bb_step_arxiv)
                                 )
    
    # def split_or_stop(self, datum):
    #     if self.resolution+1<self.depth:
    #         self.split(datum)
    #         print(f'ID: {self.id}: Trained. Splitting..')
    #     else:
    #         self.trained = True
    #         print(f'ID: {self.id}: Trained')
    #         self.tok = time.perf_counter()
    #         self.myTime.append(self.tok-self.tik)
    #         # Prune siblings that are not updated at all
    #         self.prune_siblings()    
    
    def overwrite_codevectors(self,new_y,new_ylabels,new_py=[]): # new_y must be a list
        self.y = new_y
        # self.lastY = new_y
        self.ylabels = new_ylabels
        # self.lastYlabels = new_ylabels
        if new_py==[]:
            self.py = [1.0 / len(self.y) for i in range(len(self.y))] 
        else:
            self.py = new_py
        self.sxpy= [self.y[i]*self.py[i] for i in range(len(self.y))]    
        self.K = len(self.y)
        
    # To be checked
    def pop_codevector(self,idx): # bad_y must be a codevector
        # idx = self.y.index(bad_y)
        self.y.pop(idx)
        self.ylabels.pop(idx)
        self.py.pop(idx)
        # may need to comment out next two lines
        self.py = [float(p)/sum(self.py) for p in self.py]
        self.sxpy= [self.y[i]*self.py[i] for i in range(len(self.y))]    
        self.K = len(self.y)
                
    def check_trained(self):
        if len(self.children)>0:
            Ts = [child.check_trained() for child in self.children]
            if np.all(Ts):
                self.trained=True
        return self.trained
    
    def put_in_timeline(self, my_id):
        if self.parent:
            self.parent.timeline.append(my_id)
            self.parent.put_in_timeline(my_id)
        else:
            if not len(self.children)>0:
                self.timeline.append(my_id)
    
    def check_timeline_limit(self):
        if self.parent:
            return self.parent.check_timeline_limit()
        else:
            return self.timeline_limit < len(self.timeline), len(self.timeline)
    
    def treeK(self,sub=False):
        
        if sub:
            if len(self.children)>0:
                return np.sum([child.treeK(sub=True) for child in self.children])
            else:
                return self.myK[-1]
        else:
            node = self
            while node.parent:
               node = node.parent
            tK = node.treeK(sub=True)
            node.myTreeK.append(tK)
            return tK
    
    def confirm_error_threshold(self):
        if self.parent:
            self.parent.confirm_error_threshold()
        else:
            self.error_threshold_reached += 1
    
    def check_error_threshold(self):
        if self.parent:
            return self.parent.check_error_threshold()
        else:
            return self.error_threshold_reached
        
    def update_T(self):
        
        if self.true_convergence_counter < len(self.gamma_schedule):
            self.gamma = self.gamma_schedule[self.true_convergence_counter]
        else:
            self.gamma = self.gamma_steady
            self.perturb_param = self.gamma * self.perturb_param
            self.effective_neighborhood = self.gamma * self.effective_neighborhood
            if self.py_cut > 0.000001: #0.000001
                self.py_cut = self.gamma * self.py_cut
            if self.em_convergence > 0.0001: #0.000001
                self.em_convergence = self.gamma * self.em_convergence
        
        self.T = self.gamma * self.T
        
    
    
    # Score
    ###########################################################################
    
    def score(self, data, labels):
        if self.parent:
            return self.parent.score(data, labels)
        else:
            classification = True if len(self.classes)>1 else False
            d = 0.0
            for i in range(len(data)):
                if classification:
                    d += self.datum_classification_error(data[i], labels[i])
                else:
                    d += self.datum_dissimilarity(data[i], labels[i])         
            return d/len(data) if classification else d      
    
    def datum_dissimilarity(self, datum, label):
    
        # ignore labels
        # y = self.y.copy()
        # dists = [_BregmanD(np.array(datum[self.resolution]),np.array(yj),self.Bregman_phi) for yj in y]
        # j = np.argmin(dists)
        # d = dists[j]
        
        # y = self.y
        y = self.myY[-1] 
        j,d = _winner(np.array(y), np.array(datum[self.resolution]), self.Bregman_phi)
        
        if len(self.children) > 0:
            return self.children[j].datum_dissimilarity(datum,label)
        else:
            return d   
    
    def datum_classification_error(self, datum, label):
        
        # y = self.y.copy()
        # dists = [_BregmanD(np.array(datum[self.resolution]),np.array(yj),self.Bregman_phi) for yj in y]
        # j = np.argmin(dists)
        
        # y = self.y
        y = self.myY[-1]
        j,_ = _winner(np.array(y), np.array(datum[self.resolution]), self.Bregman_phi)
        
        # if len(self.children)>0:
        # if I have children and the winner child has converged at least once
        if len(self.children)>0 and len(self.children[j].myY)>1: 
            return self.children[j].datum_classification_error(datum,label)
        else:
            decision_label = self.ylabels[j]
            d = 0 if decision_label == label else 1
            return d      
    
    # convert number of convergent configurations to Tmin given self.Tmax
    def n2Tmin(self, n):
        T = self.Tmax_arxiv[self.resolution]
        for i in range(n):
            if i < len(self.gamma_schedule):
                gamma = self.gamma_schedule[i]
            else:
                gamma = self.gamma_steady
            T = gamma * T
        return T
    
    # Bregman Divergences
    ###########################################################################
        
    def BregmanD(self,x, y):
        pzero = 1e-9
        d = 0
        if self.Bregman_phi == 'phi_Eucl':
            d = np.dot(x-y,x-y)
        elif self.Bregman_phi == 'phi_KL':
            x[x<pzero] =pzero 
            y[y<pzero] =pzero    
            d = np.dot(x,np.log(x)-np.log(y)) - np.sum(x-y)
        return (d, 0, 0)
    
    # def BregmanDs(self,x, y):
        
    #     phi = self.phi
    #     d = phi(x)[0] - phi(y)[0] - np.dot(phi(y)[1], x-y)
        
    #     return (d,
    #             - np.dot(phi(y)[2], x-y)
    #             )
    
    # def phi_Eucl(self,x):
    #     lenx = len(x) if hasattr(x, "__len__") else 1
    #     return (np.dot(x,x), 
    #             2*x, 
    #             np.diag(2*np.ones(lenx)) 
    #             )
    
    # def phi_KL(self,x):
    #     lenx = True if hasattr(x, "__len__") else False
    #     xx = x.copy()
        
    #     if not lenx:
    #         if xx < self.practical_zero:
    #             xx = self.practical_zero
    #     else:
    #         for i in range(len(x)):
    #             if xx[i] < self.practical_zero:
    #                 xx[i] = self.practical_zero
    
    #     return (np.dot(xx,np.log(xx)), 
    #             np.ones(len(x)) + np.log(xx),
    #             np.diag(np.ones(len(x))*1/xx)
    #             )
    
    # def phi_IS(self,x):
    #     lenx = len(x) if hasattr(x, "__len__") else 1
    #     return (-np.dot(np.log(x), np.ones(lenx)),
    #             -np.ones(lenx)/x,
    #             np.diag(np.ones(lenx)/x**2)
    #             )
    
    
    # Leaf nodes 
    
    # def leaf_Y(self):
    #     if len(self.children) > 0:
    #         leafY = []
    #         for child in self.children:
    #             leafY = leafY + child.leaf_Y() 
    #         return leafY
    #     else:
    #         return self.y
    
    # def leaf_Ylabels(self):
    #     if len(self.children) > 0:
    #         leafYlabels = []
    #         for child in self.children:
    #             leafYlabels = leafYlabels + child.leaf_Ylabels() 
    #         return leafYlabels
    #     else:
    #         return self.ylabels
    
    # to include training a subset of the leaf nodes
    def load(self,
                 train_data=[],
                 train_labels=[], 
                 keepscore=False,
                 timeline_limit=100000,
                 # classes=[0], #same for children
                 # node_id=[1],
                 depth=1,
                 # parent = None,
                 Kmax=[100],
                 Tmax=[100.0],
                 Tmin=[0.001],
                 error_threshold=[0.04],
                 error_threshold_count = 5,
                 gamma_schedule=[[0.1,0.1,0.1]],
                 gamma_steady=[0.8],
                 Bregman_phi=['phi_Eucl'],
                 em_convergence = [0.0001], 
                 convergence_counter_threshold = [5],
                 perturb_param = [0.01], 
                 effective_neighborhood = [0.0005], 
                 py_cut = [0.0001],
                 convergence_loops=[0],
                 stop_separation=[1000],
                 # y_init=[], #not used in children
                 dim=[1],
                 dom_min=[0],
                 dom_max=[1],
                 bb_init=[0.9],
                 bb_step=[0.9]):
        
        
        # Archive
        self.train_data_arxiv = train_data.copy()
        self.train_labels_arxiv = train_labels.copy()
        self.Kmax_arxiv = Kmax.copy()
        self.Tmax_arxiv = Tmax.copy()
        self.Tmin_arxiv = Tmin.copy()
        self.error_threshold_arxiv = error_threshold.copy()
        self.gamma_schedule_arxiv = gamma_schedule.copy()
        self.gamma_steady_arxiv = gamma_steady.copy()
        self.Bregman_phi_arxiv = Bregman_phi.copy()
        self.em_convergence_arxiv = em_convergence.copy()
        self.perturb_param_arxiv = perturb_param.copy()
        self.effective_neighborhood_arxiv = effective_neighborhood.copy()
        self.py_cut_arxiv = py_cut.copy()
        self.convergence_loops_arxiv = convergence_loops.copy()
        self.convergence_counter_threshold_arxiv = convergence_counter_threshold.copy()
        self.bb_init_arxiv = bb_init.copy()
        self.bb_step_arxiv = bb_step.copy()
        self.stop_separation_arxiv = stop_separation.copy()
        
        # Tree & Resolution parameters
        # self.id = node_id.copy()
        # self.resolution = len(node_id)-1
        self.depth = depth 
        self.keepscore = keepscore
                
        # self.test_data = test_data
        # self.test_labels = test_labels
        
        # Bregman Divergence parameters
        self.Bregman_phi = Bregman_phi[self.resolution] # 'phi_Eucl', 'phi_KL', 'phi_IS'
        # self.phi = eval('self.'+ self.Bregman_phi)
        
        # Initial Values from dataset if available 
        if len(train_data)>0:
            train_samples = len(train_data)    
            # classes = list(np.unique(train_labels))
            # Domain Info
            self.dim_arxiv = [len(td) for td in train_data[0]]
            self.dom_min_arxiv = [np.min(np.min([td[r] for td in train_data],0)) for r in range(len(train_data[0]))]
            self.dom_max_arxiv = [np.max(np.max([td[r] for td in train_data],0)) for r in range(len(train_data[0]))]
            # Initial Conditions
            # y_init = train_min + 0.5*train_domain*np.ones_like(train_data[0][0])
            # if not len(y_init)> 0:
            #     y_init = train_data[np.random.choice(range(len(train_data)))][self.resolution]
            # If predefined number of loops
            self.convergence_loops = np.ceil(convergence_loops[self.resolution] * train_samples)
        else:
            self.convergence_loops = convergence_loops[self.resolution]
            self.dim_arxiv = dim.copy()
            self.dom_min_arxiv = dom_min.copy()
            self.dom_max_arxiv = dom_max.copy()
        
        
        # Domain info
        self.dim = self.dim_arxiv[self.resolution]
        self.dom_min = self.dom_min_arxiv[self.resolution]
        self.dom_max = self.dom_max_arxiv[self.resolution]   
        
        # # Scale Parameters
        if self.Bregman_phi == 'phi_Eucl':
            self.dom = (self.dom_max-self.dom_min)**2   
        elif self.Bregman_phi == 'phi_KL':
            self.dom = (self.dom_max-self.dom_min)
                    
        # Set limit parameters
        self.Kmax = 2*Kmax[self.resolution]
        self.Tmax = Tmax[self.resolution]*self.dim*self.dom
        self.Tmin = Tmin[self.resolution]*self.dim*self.dom
        self.gamma_schedule = gamma_schedule[self.resolution]
        self.gamma_steady = gamma_steady[self.resolution]
        if len(self.gamma_schedule)>0:
            self.gamma = self.gamma_schedule[0] 
        else:
            self.gamma = self.gamma_steady# T'=gamma*T
        self.error_threshold = error_threshold[self.resolution]
        
        # EM Convergence parameters
        # self.em_convergence = em_convergence[self.resolution]*self.dim*self.dom 
        # self.perturb_param = perturb_param[self.resolution]*self.dim*self.dom 
        # self.effective_neighborhood = effective_neighborhood[self.resolution]*self.dim*self.dom
        # self.py_cut = py_cut[self.resolution]
        # self.practical_zero = 1e-9
        
        # Training parameters
        
        self.convergence_counter_threshold = convergence_counter_threshold[self.resolution]
        # self.run_batch = run_batch # if True, run originial DA algorithm
        # self.batch_size = batch_size # Stochastic/mini-batch Version  
        self.bb_init= bb_init[self.resolution] # initial stepsize of stochastic approximation: 1/(bb+1)
        self.bb_step = bb_step[self.resolution] # 0.5 # bb+=bb_step
        self.separate = True
        self.stop_separation = stop_separation[self.resolution]
    
        # Codevectors
        # self.y = []
        # self.ylabels = []
        # self.py = []
        # self.sxpy= []
        # self.old_y = []
        
        # Init y
        # if not len(y_init)> 0:
        #     y_init = self.dom_min + 0.5*(self.dom_max-self.dom_min)*np.ones(self.dim)
        # for c in classes:
        #     # self.y.append(y_init)
        #     self.y.append(y_init* (1 + 0.02*(np.random.rand(len(y_init))-0.5)))
        #     self.ylabels.append(c) 
        #     self.py.append(1.0/len(classes))
        #     self.sxpy.append(self.py[-1]*self.y[-1])
        # self.classes = classes
                
        # State Parameters
        # self.T = self.Tmax
        # self.K = len(self.y)
        self.perturbed = False
        self.converged = False
        # self.convergence_counter = 1
        # self.true_convergence_counter = -1
        self.trained = False
        self.error_threshold_reached = 0
        self.error_threshold_count = error_threshold_count
        self.timeline_limit = timeline_limit
        
        # Convergence parameters
        self.bb = self.bb_init
        # self.sa_steps = 0
        # self.em_steps = 0 # obsolete
        self.low_p_warnings = 0

        # Children
        # self.children = []
        # self.parent = parent
        # self.timeline = []
        
        # Copies for parallel computation, i.e. tree-structure
        # self.lastK = self.K
        # self.lastT = self.T + 1e-6
        # self.lastY = self.y.copy()
        # self.lastPy = self.py.copy()
        # self.lastYlabels = self.ylabels.copy()
    
        # Keep record for each temperature
        # self.myK = [self.K]
        # self.myT = [self.T]
        # self.myY = [self.y.copy()]
        # self.myYlabels = [self.ylabels.copy()]
        # self.myTrainError = [1]
        # self.myTestError = [1]
        # self.myLoops = [0]    
        # self.myTime = [0]
        
        # self.last_sample = 0 # sample at which last convergence
        # self.current_sample = 0
        # self.plt_counter = 0
        
        self.tik = time.perf_counter()
        self.tok = time.perf_counter()            
        
        if len(self.children)>0:
            for child in self.children:
                child.load([],[], 
                 self.keepscore,self.timeline_limit,self.depth,
                 self.Kmax_arxiv,self.Tmax_arxiv,self.Tmin_arxiv,self.error_threshold_arxiv,self.error_threshold_count,
                 self.gamma_schedule_arxiv,self.gamma_steady_arxiv,self.Bregman_phi_arxiv,self.em_convergence_arxiv, 
                 self.convergence_counter_threshold_arxiv,self.perturb_param_arxiv, 
                 self.effective_neighborhood_arxiv,self.py_cut_arxiv,self.convergence_loops_arxiv,
                 self.stop_separation_arxiv,self.dim_arxiv,self.dom_min_arxiv,
                 self.dom_max_arxiv,self.bb_init_arxiv,self.bb_step_arxiv)
        
    ## Algorithm Plot
    ###########################################################################
    
    def plot_dspace_X(self, y, ylabels, data, labels, title='', plot_folder='./',
                      plot_counter = 0):
        
        a = 1
        aa = 0.2
    
        if len(data[0])<2:
            y = [[cd,0] for cd in y]
            data = [[dt,0] for dt in data]
        
        if len(data[0])>2:
            print('Error: Dimnesions more than 2.')
            return
        
        # Create new Figure
        fig = plt.figure()
        ax = fig.add_subplot(111)#, autoscale_on=True, aspect='equal')#,
                             #xlim=(-2, 3), ylim=(-5, 5)) #aspect='equal',
        # ax.grid(True)
        plt.xticks([],'')
        plt.yticks([],'')
        
        classes = np.unique(labels)
        colors = ['k','r','b','m','y']
     
        dlabels = _predictX(y,ylabels,data)
        
        for c in range(len(classes)):
            
            # data points
            ax.scatter([data[i][0] for i in range(len(data)) if dlabels[i]==classes[c] ],
                    [data[i][1] for i in range(len(data)) if dlabels[i]==classes[c] ],
                    color=colors[c],marker='.',alpha=aa)
            # y
            ax.scatter([y[i][0] for i in range(len(y)) if ylabels[i]==classes[c] ],
                    [y[i][1] for i in range(len(y)) if ylabels[i]==classes[c] ],
                    color=colors[c],marker='D',alpha=a)
                
        plt.title(title)
        
        # save figure
        fig.savefig(plot_folder+f'{plot_counter:03d}.png',
                        format = 'png')
        
        plt.close('all')
        
        
        
        
    def plot_curve(self,figname,
                    # Parameters
                    fig_size=(12, 8),
                    font_size = 38,
                    label_size = 38,
                    legend_size = 32,
                    line_width = 12,
                    marker_size = 8,
                    fill_size=10,
                    line_alpha = 0.6,
                    txt_size = 32,
                    txt_x = 1.0,
                    txt_y = 0.03,
                    font_weight = 'bold', 
                    ylim = 0.5
                   ):
        
        # Variables        
        
        idx = []
        myK = []
        tK=[]
        myT = []
        myY = []
        myYlabels = []
        myTrainError = []
        myTestError = []
        myLoops = []    
        myTime = []
        
        # Read results
        for i in range(len(self.timeline[1:])):
            nid = self.timeline[i+1]
            node = self
            for child_id in nid[1:]:
                node=node.children[child_id-1]   
            node.plt_counter = 1
            idx.append(i)
            
        for i in range(len(self.timeline[1:])):
            nid = self.timeline[i+1]
            node = self
            for child_id in nid[1:]:
                node=node.children[child_id-1]    
            
            myK.append(node.myK[node.plt_counter])
            tK.append(self.myTreeK[i+1])
            myT.append(node.myT[node.plt_counter])
            myY.append(node.myY[node.plt_counter])
            myYlabels.append(node.myYlabels[node.plt_counter])
            myTrainError.append(node.myTrainError[node.plt_counter])
            myTestError.append(node.myTestError[node.plt_counter])
            myLoops.append(node.myLoops[node.plt_counter])   
            myTime.append(node.myTime[node.plt_counter])
            
            node.plt_counter += 1
            
        # Figure
        
        fig,ax = plt.subplots(figsize=fig_size,tight_layout = {'pad': 1})
        
        # Label axes
        ax.set_ylim(-0.05,ylim+0.01)
        ax.set_xlabel('T-epochs '+f'({np.sum(self.myTime):.1f}s)', fontsize = font_size)
        ax.set_ylabel('Class. Error', fontsize = font_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    
        x=idx
        y=myTrainError
        clr='k'
        ax.plot(x, y, label='Train', 
          color=clr, marker='o',linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        # yy = fill_size*np.ones_like(y)
        # ax.fill_between(x, y+yy, y-yy, facecolor=clr, alpha=line_alpha)
        # for i, j in zip(np.arange(len(x),step=5),np.arange(len(y),step=5)):
        #     if y[j]<ylim:
        #         pl.text(x[i]+txt_x, y[j]+txt_y, str(myK[i]), color=clr, fontsize=txt_size,
        #                 fontweight=font_weight)    
                
        x=idx
        y=myTestError
        clr='r'
        ax.plot(x, y, label='Test', 
          color=clr, marker='o',linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        # yy = fill_size*np.ones_like(y)
        # ax.fill_between(x, y+yy, y-yy, facecolor=clr, alpha=line_alpha)
        text_r = True
        for i, j in zip(np.arange(len(x),step=1),np.arange(len(y),step=1)):
            # if y[j]<ylim:
            #     pl.text(x[i]+txt_x, y[j]+txt_y, str(myK[i]), color=clr, fontsize=txt_size,
            #             fontweight=font_weight)    
            if i==0:
                pl.text(x[i]-txt_x, ylim-0.05, 'R'+str(len(self.timeline[i])), color='k', fontsize=txt_size)  
            if i>0 and len(self.timeline[i-1])<len(self.timeline[i]) and text_r:
                pl.text(x[i]+2*txt_x, ylim-0.05, 'R'+str(len(self.timeline[i])), color='k', fontsize=txt_size)  
                if len(self.timeline[i])>2:
                    text_r = False

        for i in x:
            if len(self.timeline[i-1])<len(self.timeline[i]):
                plt.axvline(x=i,c='k')
            if len(self.timeline[i])>2:
                break

        plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
        plt.legend(prop={'size': legend_size})
    
        
        #plt.show()
        fig.savefig(figname+'.png',
                    format = 'png')
        
#%% Numba Functions

@njit(cache=True,nogil=True)
def _winner(y, datum, phi='phi_KL'):
    dists = np.zeros(len(y))
    for i in range(len(y)):
        dists[i]=_BregmanD(datum,y[i],phi)
    j = np.argmin(dists)
    return j, dists[j]

@njit(cache=True,nogil=True)
def _BregmanD(x, y, phi='phi_KL'):
    if phi == 'phi_Eucl':
        d = _dot(x-y,x-y)
    elif phi == 'phi_KL':
        pzero = 1e-9
        logx = np.zeros_like(x)
        logy = np.zeros_like(y)
        sxy=0
        for i in range(len(x)):
            if x[i]<pzero:
                x[i]=pzero
            if y[i]<pzero:
                y[i]=pzero    
            logx[i] = np.log(x[i])
            logy[i] = np.log(y[i])
            sxy += x[i]-y[i]
        d = _dot(x,logx-logy) - sxy
    return d

@njit(cache=True,nogil=True)
def _dot(x, y):
    s = 0.0
    for i in range(len(x)):
        s += x[i]*y[i]
    return s
        
@njit(cache=True)
def _sa_update(idx, y, ylabels, py, sxpy, datum, datum_label, sep, T, bb, phi='phi_KL'):
    
    pzero = 1e-9
    
    selfpy = py[idx]
    selfsxpy = sxpy[idx]
    
    if sep:
        for i in range(len(py)):
            if datum_label != ylabels[i]:
                py[i]=0
    
    dists = np.zeros(len(y))
    gibbs = np.zeros(len(y))
    for i in range(len(y)):
        dists[i]=_BregmanD(datum,y[i],phi)
        gibbs[i] = np.exp(-dists[i]/T)
    
    pyx_sum = _dot(py,gibbs)
    
    pyx = 0.0
    if pyx_sum == 0: # e.g. if no codevectors of the same class as observation
        pyx = 0 # set it zero (or break)
    else:    
        pyx = py[idx]*gibbs[idx]/pyx_sum
    
    # SA update
    pypy = selfpy + 1/(bb+1)*(pyx - selfpy)
    sxpysxpy = selfsxpy + 1/(bb+1)*(pyx*datum - selfsxpy)
    yy = sxpysxpy/pypy
    
    if phi == 'phi_KL':
        for i in range(len(yy)):
            if yy[i]<pzero:
                yy[i] = pzero
    
    return pypy,sxpysxpy,yy

# def _predict_threads(y, ylabels, data):
#     out = np.zeros(len(data))
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         args = ((y,ylabels,datum) for datum in data) 
#         res = executor.map(_datum_predict_threads,args)
#     for i,r in enumerate(res):
#         out[i]=r
#     return out
    
# @njit(cache=True, nogil=True)
# def _datum_predict_threads(args):
#     y, ylabels, datum = args
#     dists = np.zeros(len(y))
#     for i in range(len(y)):
#         dists[i]=_BregmanD(datum,y[i])
#     j = np.argmin(dists)
#     return ylabels[j]

def _predictX(y,ylabels,data):
 out = []
 for i in range(len(data)):
     # dists = [self.BregmanD(data[i][self.resolution],yj)[0] for yj in y]
     # j = np.argmin(dists)
     j,_ = _winner(np.array(y), np.array(data[i]), 'phi_Eucl')
     out.append(ylabels[j])        
 return out

        
