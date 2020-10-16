# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np
import scipy
from picard import picard
from multiviewica.reduce_data import reduce_data


def permica(
    X,
    n_components=None,
    dimension_reduction="pca",
    max_iter=1000,
    random_state=None,
    tol=1e-7,
):
    """
    Performs one ICA per group (ex: subject) and align sources
    using the hungarian algorithm.

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
    max_iter : int, optional
        Maximum number of iterations to perform
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.

    Returns
    -------
    P : np array of shape (n_groups, n_components, n_features)
        K is the projection matrix that projects data in reduced space
    W : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    S : np array of shape (n_components, n_samples)
        Estimated source

    See also
    --------
    groupica
    multiviewica
    """
    P, X = reduce_data(
        X, n_components=n_components, dimension_reduction=dimension_reduction
    )
    n_pb, p, n = X.shape
    W = np.zeros((n_pb, p, p))
    S = np.zeros((n_pb, p, n))
    for i, x in enumerate(X):
        Ki, Wi, Si = picard(
            x,
            ortho=False,
            extended=False,
            centering=False,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        scale = np.linalg.norm(Si, axis=1)
        S[i] = Si / scale[:, None]
        W[i] = np.dot(Wi, Ki) / scale[:, None]
    orders, signs, S = _find_ordering(S)
    for i, (order, sign) in enumerate(zip(orders, signs)):
        W[i] = sign[:, None] * W[i][order, :]
    return P, W, S


def _hungarian(M):
    u, order = scipy.optimize.linear_sum_assignment(-abs(M))
    vals = M[u, order]
    return order, np.sign(vals)


def _find_ordering(S_list, n_iter=10):
    n_pb, p, _ = S_list.shape
    for i in range(len(S_list)):
        S_list[i] /= np.linalg.norm(S_list[i], axis=1, keepdims=1)
    S = S_list[0].copy()
    order = np.arange(p)[None, :] * np.ones(n_pb, dtype=int)[:, None]
    signs = np.ones_like(order)
    for _ in range(n_iter):
        for i, s in enumerate(S_list[1:]):
            M = np.dot(S, s.T)
            order[i + 1], signs[i + 1] = _hungarian(M)
        S = np.zeros_like(S)
        for i, s in enumerate(S_list):
            S += signs[i][:, None] * s[order[i]]
        S /= n_pb
    return order, signs, S
