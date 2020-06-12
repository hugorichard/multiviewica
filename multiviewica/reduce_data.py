# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np
from sklearn.utils.extmath import randomized_svd
from fastsrm.identifiable_srm import IdentifiableFastSRM


def reduce_data(X, n_components, dimension_reduction):
    """
    Reduce the number of features in X to n_components

    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    dimension_reduction: str, optional
        if srm: use srm to reduce the data
        if pca: use group specific pca to reduce the data

    Returns
    -------
    projection: np array of shape (n_groups, n_components, n_features)
        the projection matrix that projects data in reduced space
    reduced: np array of shape (n_groups, n_components, n_samples)
        Reduced data
    """
    if n_components is None:
        return None, X
    else:
        if dimension_reduction == "pca":
            return pca_reduce_data(X, n_components)
        elif dimension_reduction == "srm":
            return srm_reduce_data(X, n_components)
        else:
            ValueError(
                "Dimension reduction %s is not implemented" % (dimension_reduction)
            )


def pca_reduce_data(X, n_components):
    """
    Reduce the number of features in X via group specific PCA
    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    Returns
    -------
    projection: np array of shape (n_groups, n_components, n_features)
        the projection matrix that projects data in reduced space
    reduced: np array of shape (n_groups, n_components, n_samples)
        Reduced data
    """
    n_groups, n_features, n_samples = X.shape
    reduced = []
    basis = []
    for i in range(n_groups):
        U_i, S_i, V_i = randomized_svd(X[i], n_components=n_components)
        reduced.append(S_i.reshape(-1, 1) * V_i)
        basis.append(U_i.T)
    return np.array(basis), np.array(reduced)


def srm_reduce_data(X, n_components):
    """
    Reduce the number of features in X via FastSRM
    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    Returns
    -------
    projection: np array of shape (n_groups, n_components, n_features)
        The projection matrix that projects data in reduced space
    reduced: np array of shape (n_groups, n_components, n_samples)
        Reduced data
    """
    srm = IdentifiableFastSRM(
        n_components=n_components,
        tol=1e-10,
        aggregate=None,
        identifiability="decorr",
    )
    S = np.array(srm.fit_transform([x for x in X]))
    W = np.array(srm.basis_list)
    return W, S
