#!/usr/bin/env python

"""
(Multi-Resolution) Online Deterministic Annealing (ODA) for Classification and Clustering
Christos N. Mavridis & John S. Baras,
Department of Electrical and Computer Engineering, University of Maryland
email: <mavridis@umd.edu>
"""

#%% Modules

import sys
sys.path.append('../../oda/')
import train_oda

#%% Parameters

data_file = './data/data'
load_file = ''
# load_file = 'demo-1'
results_file = 'demo-1'

# Specify data resolution (lowest=0) for each tree layer 
res = [1] 

# Temperature 
Tmax = [10.0] # Scaled wrt domain size
Tmin = [0.0001] # Scaled wrt domain size
gamma_schedule = [[0.1,0.1]] # gamma values until gamma=gamma_steady --
gamma_steady = [0.8] # -- T' = gamma * T

# Regularization
perturb_param = [0.1] # Perturb codevectors. Scaled wrt domain size and T.
effective_neighborhood = [0.1] # Threshold under which codevectors are merged. Scaled wrt domain size and T.
py_cut = [0.001] # Probability under which a codevector is pruned. Scaled wrt T.

# Termination
Kmax = [500] # Maximum number of codevectors allowed for each cell.
timeline_limit = 500 - 1 # Stop after timeline_limit T-epochs.
error_threshold = [0.001] # Stop after reaching error_threshold --
error_threshold_count = 3 # -- for error_threshold_count times.

# Convergence
em_convergence = [0.01] # T-epoch is finished when d(y',y)<em_convergence --
convergence_counter_threshold = [5] # -- for convergence_counter_threshold times
stop_separation = [100000-1] # After stop-separation T-epochs stop treating distributions as independent
convergence_loops = [0] # if>0 forces convergence_loops observations until T-epoch is finished
bb_init = [0.9] # initial bb value for stochastic approximation stepsize: 1/(bb+1) --
bb_step = [0.9] # -- bb+=bb_step

# Bregman Divergence
Bregman_phi = ['phi_Eucl'] # Possible values: 'phi_Eucl', 'phi_KL'

# Verbose 
plot_curves = True # Save figure with training & testing error curve.
show_domain = False # Create folder with figures depicting the data space. Not supported yet.
keepscore = True # Compute error after each T-epoch

#%% Run Experiments

clf = train_oda.run(data_file=data_file,results_file=results_file,load_file=load_file,
                    res=res,plot_curves=plot_curves,show_domain=show_domain,
                    keepscore=keepscore,
                    timeline_limit=timeline_limit,
                    Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
                    error_threshold=error_threshold,
                    error_threshold_count=error_threshold_count,
                    gamma_schedule=gamma_schedule,gamma_steady=gamma_steady,
                    Bregman_phi=Bregman_phi,
                    em_convergence=em_convergence,
                    py_cut = py_cut,
                    convergence_counter_threshold=convergence_counter_threshold,
                    perturb_param=perturb_param,
                    effective_neighborhood=effective_neighborhood,
                    convergence_loops=convergence_loops,stop_separation=stop_separation,
                    bb_init=bb_init,bb_step=bb_step
                    )
                
                    
                


