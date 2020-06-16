"""
==============================
fMRI utils
==============================


"""


# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np


def load_and_concat(paths):
    """
    Load data and concatenate temporally
    Parameters
    ----------
    paths: np array of shape (n_subjects, n_runs)
        paths to data arrays
    Returns
    -------
    concat: np array of shape (n_subjects, n_voxels, n_timeframes)
    """
    n_subjects, n_runs = paths.shape
    concat = []
    for i in range(n_subjects):
        X_i = np.concatenate(
            [np.load(paths[i, j]) for j in range(n_runs)], axis=1
        )
        concat.append(X_i)
    return np.array(concat)
