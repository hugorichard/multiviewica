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

Run the experiment on synthetic data

`python synthetic_experiment.py`

It runs in `4min 28s` and creates a figure `synthetic_experiment.png`:

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

Launch the download script (runtime ``34m6.751s``)

`` bash download_data.sh ``

Mask the data (runtime ``46m55.789s``)

``python mask_data.py``

#### Reconstruction experiment

Move into the examples directory

``cd multiviewica/examples``

Run the experiment on masked data (runtime ``76m23.689s``)

``python reconstruction_experiment.py``

This runs the experiment with ``n_components = 5`` and benchmark ``PCA + GroupICA`` as well as ``PermICA`` and ``MultiView ICA`` with subject specific PCA for dimension reduction.

Run the plotting script

``python reconstruction_experiment_plot.py``
