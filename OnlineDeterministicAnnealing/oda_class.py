"""
(Tree-Structured) Online Deterministic Annealing for Classification and Clustering
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""

# Parameter values reflect 1D features in [0,1].

#%% Import Modules

import numpy as np
import matplotlib.pyplot as plt
    
#%% The Class
    
class ODA:
    
    # init
    ###########################################################################
    def __init__(self,train_data=[],train_labels=[],
                 classes=[0],Kmax=200,Tmax=100,Tmin=0.001,Tsplit=-1,
                 gamma_schedule=[0.01,0.1,0.1],gamma_steady=0.8,
                 em_convergence=0.0001,convergence_counter_threshold=5,
                 perturb_param=0.01,effective_neighborhood=0.0005,
                 Bregman_phi='phi_Eucl', y_init=[],dim=1,dom_min=0,dom_max=1,
                 convergence_loops=0,stop_separation=1000,
                 bb_init=0.9,bb_step=0.9):
        
        # Domain info
        self.dim = dim
        self.dom_min = dom_min
        self.dom_max = dom_max
        
        # Bregman Divergence parameters
        self.Bregman_phi = Bregman_phi # 'phi_Eucl', 'phi_KL', 'phi_IS'
        self.phi = eval('self.'+ self.Bregman_phi)
        
        # Initial Values from dataset if one
        if len(train_data)>0:
            # Domain Info
            self.dim = len(train_data[0])
            self.dom_min = np.min(np.min(train_data,0)) #-0.1
            self.dom_max = np.max(np.max(train_data,0)) #+0.1
            train_samples = len(train_data)    
            classes = list(np.unique(train_labels))
            # Initial Conditions
            # y_init = train_min + 0.5*train_domain*np.ones_like(train_data[0])
            y_init = train_data[np.random.choice(range(len(train_data)))]
            # If predefined number of loops
            convergence_loops = np.ceil(convergence_loops * train_samples)
        
        # Scale Parameters
        if self.Bregman_phi == 'phi_Eucl':
            self.dom = (self.dom_max-self.dom_min)**2   
        elif self.Bregman_phi == 'phi_KL':
            self.dom = (self.dom_max-self.dom_min)
        Tmax = Tmax*self.dim*self.dom
        Tmin = Tmin*self.dim*self.dom
        Tsplit = Tsplit*self.dim*self.dom
        em_convergence = em_convergence*self.dim*self.dom
        perturb_param = perturb_param*self.dim*self.dom 
        effective_neighborhood = effective_neighborhood*self.dim*self.dom
            
        # Set limit parameters
        self.Kmax = Kmax
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.Tsplit = Tsplit
        self.gamma_schedule = gamma_schedule
        self.gamma_steady = gamma_steady
        if len(self.gamma_schedule)>0:
            self.gamma = self.gamma_schedule[0] 
        else:
            self.gamma = self.gamma_steady# T'=gamma*T
        
        # EM Convergence parameters
        self.em_convergence = em_convergence 
        self.perturb_param = perturb_param
        self.effective_neighborhood = effective_neighborhood
        self.practical_zero = 1e-9
        
        # Training parameters
        self.convergence_loops = convergence_loops
        self.convergence_counter_threshold = convergence_counter_threshold
        # self.run_batch = run_batch # if True, run originial DA algorithm
        # self.batch_size = batch_size # Stochastic/mini-batch Version  
        self.bb_init= bb_init # initial stepsize of stochastic approximation: 1/(bb+1)
        self.bb_step = bb_step # 0.5 # bb+=bb_step
        self.separate = True
        self.stop_separation = stop_separation
    
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
            self.y.append(y_init* (1 + 0.02*(np.random.rand(len(y_init))-0.5)))
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
        self.stopped = False
        self.bb = self.bb_init
        self.sa_steps = 0
        self.em_steps = 0 # obsolete
        self.low_p_warnings = 0

        # Children
        self.children = []
        # self.leaf_convergence = False

        # Copies for parallel computation, i.e. tree-structure
        self.lastK = self.K
        self.lastT = self.T + 1e-6
        self.lastY = self.y.copy()
        self.lastPy = self.py.copy()
        self.lastYlabels = self.ylabels.copy()
    
        # Keep record for each temperature
        self.myK = [self.lastK]
        self.myT = [self.lastT]
        self.myY = [self.lastY]
        self.myYlabels = [self.lastYlabels]
        self.myTrainError = [1]
        self.myTestError = [1]
        self.myLoops = [0]    
        
        self.last_sample = 0 
        self.current_sample = 0
    
    
    # Fit
    ###########################################################################
    def fit(self,train_data,train_labels,test_data=[],test_labels=[],
                keepscore=False,alltheway=True):
        
        last_T = self.lowest_T() 
        # break_now = False
        
        # for kk in range(1000): 
        #     for i in range(len(train_data)):
                
        while not self.stopped:
            
            idx = self.current_sample%len(train_data)
            datum = train_data[idx][:]
            datum_label = train_labels[idx]
            self.current_sample += 1
            self.train_step(datum, datum_label)
            
            if self.lowest_T() < last_T:
                nT = self.lowest_T()
                nY = self.leaf_Y().copy()
                nYlabels = self.leaf_Ylabels().copy()
                nK = len(nY)
                self.myT.append(nT)
                self.myK.append(nK)
                self.myY.append(nY)
                self.myYlabels.append(nYlabels)
                self.myLoops.append(self.current_sample)
                
                print(f'Samples: {self.current_sample}(+{self.current_sample-self.last_sample}): ' +
                          f'T = {nT:.4f}, K = {nK}')
                if keepscore:
                    d_train = self.score(train_data,train_labels)
                    self.myTrainError.append(d_train)
                    print(f'Train Error: {d_train:.4f}')
                    if len(test_data)>0:
                        d_test = self.score(test_data,test_labels)
                        self.myTestError.append(d_test)
                        print(f'Test Error: {d_test:.4f}')
                
                # Split
                if nT < self.Tsplit and self.Tsplit < last_T:
                    
                    self.split()
                    # self.perturb_all()
                
                    # to be written
                
                self.last_sample = self.current_sample
        
                if not alltheway:
                    break
                else:
                    last_T = self.lowest_T()
                
            #     else:
            #         break_now = True
                
            #     if break_now:
            #         break
            # if break_now:
            #     break
    
    
    def train_step(self, datum, datum_label=0):
        
        if len(self.children) > 0:
            y = self.y.copy()
            dists = [self.BregmanD(datum,yj)[0] for yj in y]
            j = np.argmin(dists)
            self.children[j].train_step(datum,datum_label)
        else:
            # Termination Criteria
            if self.K>self.Kmax or self.T<=self.Tmin:
                self.stopped = True
                # if self.K>self.Kmax:
                #     print('Maximum number of codevectors reached.')
                # if self.T<=self.Tmin:
                #     print('Minimum Temperature reached.')
            else:
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
                    # Find effective clusters
                    self.find_effective_clusters() 
                    self.pop_idle_clusters() 
                    
                    # Save parameters of last convergence
                    self.lastK = self.K
                    self.lastT = self.T
                    self.lastY = self.y.copy()
                    self.lastPy = self.py.copy()
                    self.lastYlabels = self.ylabels.copy()
    
                    self.update_T()
                    self.perturbed=False
    
    def sa_step(self, datum, datum_label):
        
        self.old_y = self.y.copy()
        self.low_p_warnings = 0
        
        for i in range(self.K): # for all centroids
                        
            d = [self.BregmanD(datum,self.y[k])[0] 
                     for k in range(len(self.y))]
            
            py = self.py.copy()
            # For doing separate ODA for each class
            if self.separate:
                py = [self.py[i] if datum_label==self.ylabels[i] else 0 for i in range(len(self.py))]
                
            pyx_sum = np.dot(py,np.exp([-dj/self.T for dj in d]))
            pyx = self.py[i]*np.exp(-d[i]/self.T)/pyx_sum
            
            # Warning sign
            if pyx!=pyx or pyx<1e-12:
                self.low_p_warnings+=1
            
            # SA update
            sign = 1 if datum_label == self.ylabels[i] else 0
            self.py[i] = self.py[i] + 1/(self.bb+1)*(sign*pyx - self.py[i])
            self.sxpy[i] = self.sxpy[i] + 1/(self.bb+1)*(sign*pyx*datum - self.sxpy[i])
            self.y[i] = self.sxpy[i]/self.py[i]  
            
        self.bb += self.bb_step 
        self.sa_steps += 1
        self.em_steps += 1
    
        # Warning
        if self.low_p_warnings == len(self.y):
            # print(f'Number of Idle codevectors: {self.low_p_warnings}/{self.K}')
            self.low_p_warnings = 0
            print(f'WARNING: Conditional Probabilities too small.'+
                          f'You may need to consider scaling the input space,'+
                          f'or using different initial values.')
            
    
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
            if np.all([self.BregmanD(self.old_y[i],self.y[i])[0]<self.em_convergence * (1+self.bb_init)/(self.bb+1)
                                                        for i in range(self.K)]):   
                if self.convergence_counter > self.convergence_counter_threshold:
                    self.converged = True
                    self.convergence_counter = 1
                    self.true_convergence_counter +=1
                    if self.true_convergence_counter>self.stop_separation:
                        self.separate=False
                else:
                    self.convergence_counter += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Predict
    ###########################################################################
    def predict(self, data):
        out = []
        for i in range(len(data)):
            datum = data[i]
            decision_label = self.datum_predict(datum)
            out.append(decision_label)        
        return out
    
    def datum_predict(self,datum):
        y = self.lastY.copy()
        dists = [self.BregmanD(datum,yj)[0] for yj in y]
        j = np.argmin(dists)
        
        if len(self.children) > 0:
            return self.children[j].datum_predict(datum)
        else:
            return self.lastYlabels[j]
            
    def predictX(self,y,ylabels,data):
        out = []
        for i in range(len(data)):
            dists = [self.BregmanD(data[i],yj)[0] for yj in y]
            j = np.argmin(dists)
            out.append(ylabels[j])        
        return out
        
        
        
        
        
        
        
        
        
        
    # Score
    ###########################################################################
    def score(self, data, labels):
        
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
        y = self.lastY.copy()
        dists = [self.BregmanD(datum,yj)[0] for yj in y]
        j = np.argmin(dists)
        
        if len(self.children) > 0:
            return self.children[j].datum_dissimilarity(datum,label)
        else:
            return dists[j]    
    
    def datum_classification_error(self, datum, label):
        
        y = self.lastY.copy()
        dists = [self.BregmanD(datum,yj)[0] for yj in y]
        j = np.argmin(dists)
        
        if len(self.children) > 0:
            return self.children[j].datum_classification_error(datum,label)
        else:
            decision_label = self.lastYlabels[j]
            d = 0 if decision_label == label else 1
            return d
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ODA Functions
    ###########################################################################
    
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
                if self.BregmanD(self.y[i],self.y[j])[0] < \
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
            if self.py[i]<1e-7:
                # print(f'***Idle Codevector Pruned: p = {self.py[i]}***')
                self.pop_codevector(i)
            else:
                i+=1
    
    def overwrite_codevectors(self,new_y,new_ylabels,new_py=[]): # new_y must be a list
        self.y = new_y
        self.lastY = new_y
        self.ylabels = new_ylabels
        self.lastYlabels = new_ylabels
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
                
    def lowest_T(self):
        
        if len(self.children) > 0:
            Ts = [child.lastT for child in self.children]
            return np.max(Ts)
        else:
            return self.lastT
    
    def leaf_Y(self):
        
        if len(self.children) > 0:
            leafY = []
            for i in range(len(self.children)):
                leafY = leafY + self.children[i].lastY
            return leafY
        else:
            return self.lastY
        
    def leaf_Ylabels(self):
        
        if len(self.children) > 0:
            leafYlabels = []
            for i in range(len(self.children)):
                leafYlabels = leafYlabels + self.children[i].lastYlabels
            return leafYlabels
        else:
            return self.lastYlabels
                
    def update_T(self):
        
        if self.true_convergence_counter < len(self.gamma_schedule):
            self.gamma = self.gamma_schedule[self.true_convergence_counter]
        else:
            self.gamma = self.gamma_steady
        
        self.T = self.gamma * self.T
        
    
    
    
    
    
    
    
    
    
    # Bregman Divergences
    ###########################################################################
    def BregmanD(self,x, y):
        
        d = 0
        if self.Bregman_phi == 'phi_Eucl':
            d = np.dot(x-y,x-y)
        elif self.Bregman_phi == 'phi_KL':
            x[x<self.practical_zero] =self.practical_zero 
            y[y<self.practical_zero] =self.practical_zero    
            d = np.dot(x,np.log(x)-np.log(y)) - np.sum(x-y)
        return (d, 0, 0)
    
    def BregmanDs(self,x, y):
        
        phi = self.phi
        d = phi(x)[0] - phi(y)[0] - np.dot(phi(y)[1], x-y)
        
        return (d,
                - np.dot(phi(y)[2], x-y)
                )
    
    def phi_Eucl(self,x):
        lenx = len(x) if hasattr(x, "__len__") else 1
        return (np.dot(x,x), 
                2*x, 
                np.diag(2*np.ones(lenx)) 
                )
    
    def phi_KL(self,x):
        lenx = True if hasattr(x, "__len__") else False
        xx = x.copy()
        
        if not lenx:
            if xx < self.practical_zero:
                xx = self.practical_zero
        else:
            for i in range(len(x)):
                if xx[i] < self.practical_zero:
                    xx[i] = self.practical_zero
    
        return (np.dot(xx,np.log(xx)), 
                np.ones(len(x)) + np.log(xx),
                np.diag(np.ones(len(x))*1/xx)
                )
    
    def phi_IS(self,x):
        lenx = len(x) if hasattr(x, "__len__") else 1
        return (-np.dot(np.log(x), np.ones(lenx)),
                -np.ones(lenx)/x,
                np.diag(np.ones(lenx)/x**2)
                )
    
    
    
    
    
    
    
    
    
    # Split
    ###########################################################################
            
    def split(self):
        
        for i in range(len(self.lastY)):
            self.children.append( ODA(classes=self.classes,
                 Kmax=self.Kmax,Tmax=self.lastT,Tmin=self.Tmin,
                 gamma_schedule=[],gamma_steady=self.gamma_steady,
                 em_convergence=self.em_convergence,
                 convergence_counter_threshold=self.convergence_counter_threshold,
                 perturb_param=self.perturb_param,
                 effective_neighborhood=self.effective_neighborhood,
                 Bregman_phi=self.Bregman_phi,y_init=self.lastY[i],
                 convergence_loops=self.convergence_loops,
                 bb_init=self.bb_init,bb_step=self.bb_step)
                                 )
        
        
    def perturb_all(self):
     # insert perturbations of all effective yi
        for i in reversed(range(self.K)):
            new_yi = self.y[i] + self.perturb_param*2*(np.random.rand(len(self.y[i]))-0.5)
            new_pyi = self.py[i]/len(self.classes)
            self.py[i] = new_pyi
            self.sxpy[i] = self.py[i]*self.y[i]
            for c in self.classes:
                new_yi = self.y[i] + self.perturb_param*2*(np.random.rand(len(self.y[i]))-0.5)
                self.y.append(new_yi)
                self.ylabels.append(c) 
                self.py.append(new_pyi)
                self.sxpy.append(new_pyi*new_yi)
        self.K = len(self.y)
     # self.perturbed = True 
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # Offline Deterministic Annealing
    ###########################################################################
            
    def train_batch(self, data, data_labels=0):
        
        # Termination Criteria
        if self.K>self.Kmax or self.T<=self.Tmin:
            self.stopped = True
            if self.K>self.Kmax:
                print('Maximum number of codevectors reached.')
            if self.T<=self.Tmin:
                print('Minimum Temperature reached.')
        else:
            # check temperature
            # if self.T<self.Tmin:
            #     self.T=self.Tmin
            
            # insert perturbations and initialize SA stepsizes
            if not self.perturbed:
                self.perturb()
                self.converged = False     
                self.bb = self.bb_init
                self.sa_steps = 0
            
            # DA step
            self.batch_step(data, data_labels)
            
            # Check Convergence
            self.check_convergence()
            
            if self.converged:
                # Find effective clusters
                    self.find_effective_clusters() 
                    self.pop_idle_clusters() 
                    
                    # Parameters in last convergence
                    self.lastK = self.K
                    self.lastT = self.T
                    self.lastY = self.y.copy()
                    self.lastPy = self.py.copy()
                    self.lastYlabels = self.ylabels.copy()
    
                    self.update_T()
                    self.perturbed=False
                
    
    def batch_step(self, data, data_labels):
        
        batch_size = len(data)
        self.old_y = self.y.copy()
                    
        for i in range(self.K): # for all centroids
            sum_yi = np.zeros(self.y[i].shape)
            sum_pyi = 0.0
            n = 0
            for j in range(len(data)):  
                datum = data[j]
                datum_label = data_labels[j]
                d = [self.BregmanD(datum,self.y[k])[0] 
                         for k in range(len(self.y))]
                py = self.py.copy()
                # For doing separate ODA for each class
                if self.separate:
                    py = [self.py[i] if datum_label==self.ylabels[i] else 0 for i in range(len(self.py))]
                
                pyx_sum = np.dot(py,np.exp([-dj/self.T for dj in d]))
                pyx = self.py[i]*np.exp(-d[i]/self.T)/pyx_sum
                
                if datum_label == self.ylabels[i]:
                    n += 1 # maybe increase n even for the other samples
                    sum_pyi = sum_pyi + 1/n * (pyx - sum_pyi) # calculate the sum
                    sum_yi = sum_yi + 1/n * (pyx*datum - sum_yi) # calculate the sum
                
            self.py[i] = sum_pyi
            self.sxpy[i] = sum_yi
            self.y[i] = self.sxpy[i]/self.py[i]  
            
        self.sa_steps += batch_size
        self.em_steps += batch_size















    # LVQ
    ###########################################################################
                    
    def train_lvq(self, datum, datum_label=0):
        
        # Termination Criteria
        if self.converged:
            print('(L)VQ Converged.')
            self.lastK = self.K
            self.lastT = self.T
            self.lastY = self.y.copy()
            self.lastPy = self.py.copy()
            self.lastYlabels = self.ylabels.copy()
        else:
            
            # SA step
            self.lvq_step(datum, datum_label)
            
            # Check Convergence
            self.check_convergence()
            
            # if self.converged:
                # Find effective clusters
                # self.find_effective_clusters() 
                
    def train_lloyds(self, data, data_labels=0):
        
        # Termination Criteria
        if self.converged:
            print('Converged.')
        else:
            
            # Lloyds Algorithm
            self.lloyds_step(data, data_labels)
            
            # Check Convergence
            self.check_convergence()
            
            # if self.converged:
                # Find effective clusters
                # self.find_effective_clusters() 
    
    def lvq_step(self, datum, datum_label):
        self.old_y = self.y.copy()
        
        d = [self.BregmanDs(datum,self.y[k])[0] 
                     for k in range(len(self.y))]
        
        winner = np.argmin(d) # w for winner
        delta_d = self.BregmanDs(datum,self.y[winner])[1]
        
        # LVQ
        sign = 1 if datum_label == self.ylabels[winner] else -1
        
        # Asynchronous SA
        self.y[winner] = self.y[winner] - 1/(self.bb+1)*sign*delta_d
        
        self.bb += self.bb_step 
        self.sa_steps += 1
        self.em_steps += 1
    
    def lloyds_step(self, data, data_labels):
        
        batch_size = len(data)
        self.old_y = self.y.copy()
        
        sum_y = np.zeros_like(self.y)
        n = np.zeros(self.K)
            
        for j in range(len(data)):
            datum = data[j]
            datum_label = data_labels[j]
            d = [self.BregmanDs(datum,self.y[k])[0] 
                     for k in range(len(self.y))]
            winner = np.argmin(d) 
            n[winner] += 1
            sign = 1 if datum_label == self.ylabels[winner] else 0
            sum_y[winner] = sum_y[winner] + 1/n[winner]*sign*(datum - sum_y[winner])
           
        self.y = sum_y
            
        self.em_steps += batch_size
        
        
        
    
        
#%% Algorithm Plot

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
     
        dlabels = self.predictX(y,ylabels,data)
        
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
        
        