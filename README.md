# Online Deterministic Annealing (ODA)
 
This repository contains the oda_class.py module and demo files to run ODA for classification and clustering.
Will be registered in pypi soon.

## Citing
If you use this work in an academic context, please cite the following publication:

Christos N. Mavridis, John S. Baras, 
"**Online Deterministic Annealing for Classification and Clustering**,"
ArXiv. 2021, [PDF](link_to_pdf)

    @misc{mavridis2021online,
          title={Online Deterministic Annealing for Classification and Clustering}, 
          author={Christos Mavridis and John Baras},
          year={2021},
          eprint={2102.05836},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
          }
	  
## Installation

Download and import oda_class.py.
Will be available through 

    pip install online-deterministic-annealing
	
## Usage

For a complete guide see oda_demo.py.
We assume data are a list of n m-vectors, and labels a list of n labels
The simplest way to train ODA on a dataset is:

    from oda_class import ODA 
    
    clf = ODA(train_data=train_data,train_labels=train_labels,
                     Kmax=Kmax,Tmax=Tmax,Tmin=Tmin,
                     Bregman_phi=Bregman_phi)

    clf.fit(train_data, train_labels, test_data, test_labels, keepscore=True)

You can access the intermediate model parameters in:

    ODA.myY, ODA.myYlabels, ODA.myK, ODA.myT, ODA.myLoops, ODA.myTrainError, ODA.myTestError

For unsupervised learning replace:

    train_labels = [0 for i in range(len(train_labels))] 

`ODA.score` will return the average distortion instead of the misclassification error. 
