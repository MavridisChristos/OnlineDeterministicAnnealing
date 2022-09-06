# Online Deterministic Annealing (ODA)

>A progressively-growing competitive-learning 'neural network' architecture
>with inherent interpretability, robustness, and regularization properties. 
>ODA is based on the principles of vector quantization and annealing optimization, 
>and is trained with a gradient-free stochastic approximation algorithm.
>Applications include online unsupervised and supervised learning [1],
>reinforcement learning [2], 
>adaptive graph partitioning [3], and swarm leader detection.

## Contact 

Christos N. Mavridis, Ph.D., \
Department of Electrical and Computer Engineering, \
University of Maryland \
https://mavridischristos.github.io/ \
```mavridis (at) umd.edu``` 

## Description
 
Inherent in virtually every
iterative machine learning algorithm is the problem 
of hyper-parameter tuning, which includes three major design parameters: 
(a) the complexity of the model, e.g., the number of neurons in a neural network, 
(b) the initial conditions, which heavily affect the 
behavior of the algorithm, and
(c) the dissimilarity measure used to quantify its performance.

Online Deterministic Annealing (ODA) is an online prototype-based learning algorithm 
for classification and clustering, 
that progressively increases the number of prototypes as needed.
That is, **the complexity of the model adapts to the online data observations**.
The prototypes can be viewed as neurons living in the data space itself, 
making ODA a progressively growing competitive-learning neural network architecture 
with inherent interpretation properties regarding the weights of the neurons.

The learning rule of ODA 
is formulated as an online **gradient-free stochastic approximation** algorithm
that solves a sequence of appropriately defined optimization problems,
simulating an **annealing process**.
The annealing nature of the algorithm contributes to
avoiding poor local minima, 
offers robustness with respect to the initial conditions,
and provides a means 
to progressively increase the complexity of the learning model
through an intuitive bifurcation phenomenon.


ODA is interpretable, requires minimal 
hyper-parameter tuning, and 
allows online control over the performance-complexity trade-off.
Finally, **Bregman divergences** (which include the widely used Kullback-Leibler divergence)
appear naturally as a family of dissimilarity measures 
that enhance both 
the performance and the computational complexity
of the learning algorithm.

	
## Usage

The ODA architecture is coded in the ODA class inside ```OnlineDeterministicAnnealing/oda_jit.py```:
	
	from oda_jit import ODA

Regarding the data format, they need to be a list of *(n)* lists of *(m=1)* *d*-vectors (np.arrays):

	train_data = [[np.array], [np.array], [np.array], ...]

The labels need to be a list of *(n)* labels, preferably integer numbers (for numba.jit)

	train_labels = [ int, int , int, ...]

The simplest way to train ODA on a dataset is:

	
	clf = ODA(train_data=train_data,train_labels=train_labels)
	clf.fit(test_data=test_data,test_labels=test_labels)

You can also specify the parameters 

- ```Kmax```: the maximum number of codevectors you allow; 
- ```Tmax``` and ```Tmin```: the maximum and minimum temperatures of the annealing optimization. These are scaled with respect to the domain size and the Bregman divergence used. Typical values are between 100 and 0.0001;
- ```Bregman_phi```: the Bregman divergence used. Right now the squared Euclidean distance 'phi_Eucl' and the KL divergence 'phi_KL' are supported.

The code becomes:	

	clf = ODA(train_data=train_data,train_labels=train_labels,
   	          Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
	          Bregman_phi=Bregman_phi)

    clf.fit(train_data, train_labels, test_data, test_labels, keepscore=True)


## Unsupervised Learning

For unsupervised learning replace:

    train_labels = [0 for i in range(len(train_labels))] 


## Demo

All parameters are treated as lists of *m* parameters, one for each resolution. \
Here *m=1*. See below for multiple resolutions. 

Demo file in one resolution:

	tests/demo/demo-1.py 

The file  ```OnlineDeterministicAnnealing/train_oda.py``` is used to train the ODA algorithm with the following
list of the parameters:


Data

- ```data_file='./data/data'```: .pkl file with the appropriate data format.
- ```load_file='demo-1'```: if not empty string, load existing model.
- ```results_file='demo-1'```: name of the .pkl file to store clf.

Resolutions
- ```res=[1]```: specify data resolution (lowest=0) for each tree layer (here one layer). 


Temperature
- ```Tmax=[10]``` and ```Tmin=[0.001]```: maximum and minimum temperatures of the annealing optimization. Scaled wrt the domain size and the Bregman divergence used. 
- ```gamma_schedule=[[0.1,0.1]]```: gamma values until gamma=gamma_steady --
- ```gamma_steady=[0.8]```: -- T' = gamma * T


Regularization
- ```perturb_param=[0,1]```: Perturbation level for codevectors. Scaled wrt domain size and T.
- ```effective_neighborhood=[0.1]```: Threshold under which codevectors are merged. Scaled wrt domain size and T.
- ```py_cut=[0.001]```: Probability under which a codevector is pruned. Scaled wrt T.


Termination
- ```Kmax=[500]```: maximum number of codevectors allowed for each cell. 
- ```timeline_limit=500-1```: Stop after timeline_limit T-epochs.
- ```error_threshold=[0.001]```: Stop after reaching error_threshold --
- ```error_threshold_count=3```: -- for error_threshold_count times.


Convergence
- ```em_convergence=[0.01]```: T-epoch is finished when d(y',y)<em_convergence --
- ```convergence_counter_threshold=[5]```: -- for convergence_counter_threshold times.
- ```stop_separation=[100000-1]```: After stop-separation T-epochs stop treating distributions as independent
- ```convergence_loops=[0]```: if>0 forces convergence_loops observations until T-epoch is finished.
- ```bb_init=[0.9]```: initial bb value for stochastic approximation stepsize: 1/(bb+1) --
- ```bb_step=[0.9]```: -- bb+=bb_step.


Bregman divergence
- ```Bregman_phi=['phi_Eucl']```: the Bregman divergence used. Right now the squared Euclidean distance 'phi_Eucl' and the KL divergence 'phi_KL' are supported. 

Verbose
- ```keepscore=True```: Compute error after each T-epoch
- ```plot_curves=True```: Save figure with training & testing error curve.
- ```show_domain=False```: Create folder with figures depicting the data space. Not supported yet.


The code becomes:

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

## Results History

The model parameters after training include:

    clf.myY, clf.myYlabels, clf.myK, clf.myTreeK, clf.myT, clf.myLoops, clf.myTime, clf.myTrainError, clf.myTestError


## Tree Structure and Multiple Resolutions

Demo file for Tree-Structured ODA using one resolution:

	tests/demo/demo-11.py 

For multiple resolutions every parameter becomes a list of *m* parameters.
Example for *m=2*:

	Tmax = [100.0, 1.0]
	Tmin = [0.01, 0.0001]

The training data should look like this:

	train_data = [[np.array, np.array, ...], [np.array, np.array, ...], [np.array, np.array, ...], ...]


Demo file for Tree-Structured ODA using hierarchically increasing resolutions:

	tests/demo/demo-01.py 

## Citing
If you use this work in an academic context, please cite the following:

[1] Christos N. Mavridis and John S. Baras, 
"**[Online Deterministic Annealing for Classification and Clustering](https://arxiv.org/pdf/2102.05836.pdf)**",
IEEE TCNS, 2022.

    @article{mavridis2022online,
          title={Online Deterministic Annealing for Classification and Clustering}, 
          author={Mavridis, Christos N. and Baras, John S.},
          journal={IEEE Transactions on Neural Networks and Learning Systems},
	  year={2022},
          volume={},  
	  number={},  
	  pages={1-10},  
	  doi={10.1109/TNNLS.2021.3138676}
          }
	  
Other references:

[2] Christos N. Mavridis and John S. Baras, 
"**[Maximum-Entropy Input Estimation for Gaussian Processes in Reinforcement Learning](https://mavridischristos.github.io/publications.html)**",
CDC, 2021.

[3] Christos N. Mavridis and John S. Baras, 
"**[Progressive Graph Partitioning Based on Information Diffusion](https://mavridischristos.github.io/publications.html)**",
CDC, 2021.
	





