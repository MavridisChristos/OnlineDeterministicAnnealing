# Online Deterministic Annealing (ODA)
 
This repository contains the oda_class.py module and demo files to run ODA for classification and clustering.
Will be registered in pypi soon.

## Citing
If you use this work in an academic context, please cite the following publication:

Christos N. Mavridis, John S. Baras, 
"**Online Deterministic Annealing for Classification and Clustering**,"
ArXiv. 2021, [PDF](link_to_pdf)

    @ARTICLE{mavridis2021online,
	  author={Mavridis, Christos N. and Baras, John S.},
	  journal={ArXiv}, 
	  title={Online Deterministic Annealing for Classification and Clustering}, 
	  year={2021}
	  }  
	  
## Installation

Download and import oda_class.py.
Will be available through 

    pip install online-deterministic-annealing
	
## Usage

For a complete guide see oda_demo.py.
We assume data are a list of n m-vectors, and labels a list of n labels
To train ODA with a dataset, we recommend a version of:

    clf = ODA(train_data=train_data,train_labels=train_labels,
                     Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
                     Bregman_phi=Bregman_phi)

    clf.fit(train_data, train_labels, test_data, test_labels, keepscore=True)
    
    print(f'*** ODA Trade-off: Smallest K for >97% ***')
    idx = [i for i in range(len(clf.myTestError)) if clf.myTestError[i]<0.03]
    errsrt = np.argsort([clf.myK[i] for i in idx])
    j = idx[errsrt[0]]
    print(f'Samples: {clf.myLoops[j]}, K: {clf.myK[j]}, T: {clf.myT[j]:.4f}')
    print(f'Train Accuracy: {1-clf.myTrainError[j]:.3f}') 
    print(f'Test Accuracy: {1-clf.myTestError[j]:.3f}')    

You can access the intermediate model parameters in:

    ODA.myY, ODA.myYlabels, ODA.myK, ODA.myT, ODA.myLoops, ODA.myTrainError, ODA.myTestError

For unsupervised learning replace:

    train_labels = [0 for i in range(len(train_labels))] 

`ODA.score` will return the average distortion. 
