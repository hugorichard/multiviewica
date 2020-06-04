# MultiView ICA

## Install

Clone the repository

`git clone https://github.com/hugorichard/multiviewica.git`

Create a virual environment

`virtualenv -p python3 venv`


Activate the virtual environment

`source venv/bin/activate`

Move into the MultiView ICA directory

``cd multiviewica``

Install MultiView ICA

`pip install -e .`

## Experiments

### Synthetic experiment

Move into the examples directory

``cd multiviewica/examples``

Run the experiment on synthetic data (Runtime ``4min 28s``)

`python synthetic_experiment.py`

![synthetic_experiment](./examples/synthetic_experiment.png)

By default the experiment is run with
```
# sigmas: data noise
# m: number of subjects
# k: number of components
# n: number of samples
sigmas = np.logspace(-2, 1, 6)
n_seeds = 10
m, k, n = 10, 3, 1000
```

The figure in the paper is obtained with
```
# sigmas: data noise
# m: number of subjects
# k: number of components
# n: number of samples
sigmas = np.logspace(-2, 1, 6)
n_seeds = 100
m, k, n = 10, 15, 1000
```
These parameters are defined in `synthetic_experiment.py`.

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

![reconstruction_experiment](./examples/reconstruction_experiment.png)

This runs the experiment with ``n_components = 5`` and benchmark ``PCA + GroupICA`` as well as ``PermICA`` and ``MultiView ICA`` with subject specific PCA for dimension reduction.

Run the plotting script

``python reconstruction_experiment_plot.py``

#### Timesegment matching

Move into the examples directory

``cd multiviewica/examples``

Run the experiment on masked data (Runtime ``4m55.119s``)

``python timesegment_matching.py``

![timesegment_matching](./examples/timesegment_matching.png)

This runs the experiment with ``n_components = 5`` and benchmark ``PCA + GroupICA`` as well as ``PermICA`` and ``MultiView ICA`` using subject specific PCA for dimension reduction.
