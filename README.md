# MultiView ICA

## Install

Clone the repository

`git clone https://github.com/hugorichard/multiviewica.git`

Install MultiView ICA

`pip install -e .`


## Requirements

* numpy >= 1.16
* scipy >= 1.12
* python-picard >= 0.4 (``pip install python-picard``)
## Experiments

### Synthetic experiment

Run the experiment on synthetic data

`python examples/synthetic_experiment.py`

![synthetic_experiment](./figures/synthetic_experiment.png)

In order to reproduce the figure in the paper, use (might take a long time):
```
# sigmas: data noise
# m: number of subjects
# k: number of components
# n: number of samples
sigmas = np.logspace(-2, 1, 21)
n_seeds = 100
m, k, n = 10, 15, 1000
```

### Experiments on fMRI data

#### Download and mask Sherlock data

Move into the data directory

``cd multiviewica/data``

Launch the download script (Runtime ``34m6.751s``)

`` bash download_data.sh ``

Mask the data (Runtime ``15m27.104s``)

``python mask_data.py``

#### Reconstructing BOLD signal of missing subjects

Move into the examples directory

``cd multiviewica/examples``

Run the experiment on masked data (Runtime ``15m6.653s``)

``python reconstruction_experiment.py``

![reconstruction_experiment](./examples/reconstruction.png)

This runs the experiment with ``n_components = 5`` and benchmark ``PCA + GroupICA`` as well as ``PermICA`` and ``MultiView ICA`` with subject specific PCA for dimension reduction.

#### Timesegment matching

Move into the examples directory

``cd multiviewica/examples``

Run the experiment on masked data (Runtime ``4m55.119s``)

``python timesegment_matching.py``

![timesegment_matching](./examples/timesegment_matching.png)

This runs the experiment with ``n_components = 5`` and benchmark ``PCA + GroupICA`` as well as ``PermICA`` and ``MultiView ICA`` using subject specific PCA for dimension reduction.
