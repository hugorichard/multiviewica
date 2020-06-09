import numpy as np
from sklearn.utils.extmath import randomized_svd
from fastsrm.identifiable_srm import IdentifiableFastSRM


def online_dot(paths, A):
    """
    Load and concatenate data of each subjects in paths
    and returns its dot product with A
    Parameters
    ----------
    paths: np array of shape (n_subjects, n_runs)
        paths to data arrays
    A: np array of shape n_timeframes, n_components
    Returns
    -------
    res: np array of shape (n_subjects, n_voxels, n_components)
    """
    n_subjects, n_runs = paths.shape
    res = []
    for i in range(n_subjects):
        X_i = np.concatenate(
            [np.load(paths[i, j]) for j in range(n_runs)], axis=1
        )
        res.append(X_i.dot(A))
    return np.array(res)


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


def reduce_data(paths, n_components):
    """
    Reduce and concatenate temporally an array of path
    using subject specific PCA

    Parameters
    ----------
    paths: np array of shape (n_subjects, n_runs)
        paths to data arrays
    n_components: int
        Number of components to keep in PCA

    Returns
    -------
    reduced: np array of shape (n_subjects, n_components, n_timeframes)
        Reduced data
    basis: np array of shape (n_subjects, n_voxels, n_components)
        Dimension reduction matrices
    """
    n_subjects, n_runs = paths.shape
    basis = []
    reduced = []
    for i in range(n_subjects):
        X_i = np.concatenate(
            [np.load(paths[i, j]) for j in range(n_runs)], axis=1
        )
        U_i, S_i, V_i = randomized_svd(X_i, n_components=n_components)
        reduced.append(np.diag(S_i).dot(V_i))
        basis.append(U_i)
    return np.array(basis), np.array(reduced)


def srm(paths, n_components):
    """
    Reduce data using FastSRM
    """
    srm = IdentifiableFastSRM(
        n_components=n_components,
        tol=1e-10,
        verbose=True,
        aggregate=None,
        identifiability="decorr",
    )
    S = srm.fit_transform(paths)
    S = np.array([np.concatenate(s, axis=1) for s in S])
    W = np.array([w.T for w in srm.basis_list])
    return W, S
