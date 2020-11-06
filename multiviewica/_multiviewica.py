# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np
import warnings
from scipy.linalg import expm
from .reduce_data import reduce_data
from ._permica import permica
from ._groupica import groupica
from time import time


def multiviewica(
    X,
    n_components=None,
    dimension_reduction="pca",
    noise=1.0,
    max_iter=1000,
    init="permica",
    random_state=None,
    tol=1e-3,
    verbose=False,
):
    """
    Performs MultiViewICA.
    It optimizes:
    :math:`l(W) = mean_t [sum_k log(cosh(Y_{avg}(t)[k])) + sum_i l_i(X_i(t))]`
    where
    :math:`l_i(X_i(t)) = - log(|W_i|) + 1/(2 noise) ||W_iX_i(t) - Y_{avg}(t)||^2`
    :math:`X_i` is the data of group i (ex: subject i)
    :math:`W_i` is the mixing matrix of subject i
    and
    :math:`Y_avg = mean_i W_i X_i`

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
    noise : float, optional
        Gaussian noise level
    max_iter : int, optional
        Maximum number of iterations to perform
    init : str or np array of shape (n_groups, n_components, n_components)
        If permica: initialize with perm ICA, if groupica, initialize with
        group ica. Else, use the provided array to initialize.
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.
    verbose : bool, optional
        Print information

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
    permica
    """
    P, X = reduce_data(
        X, n_components=n_components, dimension_reduction=dimension_reduction
    )
    # Initialization
    if type(init) is str:
        if init not in ["permica", "groupica"]:
            raise ValueError("init should either be permica or groupica")
        if init == "permica":
            _, W, S = permica(
                X, max_iter=max_iter, random_state=random_state, tol=tol
            )
        else:
            _, W, S = groupica(
                X, max_iter=max_iter, random_state=random_state, tol=tol
            )
    else:
        if type(init) is not np.ndarray:
            raise TypeError("init should be a numpy array")
        W = init
    # Performs multiview ica
    W, S = _multiview_ica_main(
        X, noise=noise, n_iter=max_iter, tol=tol, init=W, verbose=verbose,
    )
    return P, W, S


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))


def _multiview_ica_main(
    X_list,
    noise=1.0,
    n_iter=1000,
    tol=1e-6,
    verbose=False,
    init=None,
    ortho=False,
    return_gradients=False,
    timing=False,
):
    tol_init = None
    if tol > 0 and tol_init is None:
        tol_init = tol

    if tol == 0 and tol_init is None:
        tol_init = 1e-6

    # Turn list into an array to make it compatible with the rest of the code
    if type(X_list) == list:
        X_list = np.array(X_list)

    # Init
    n_pb, p, n = X_list.shape
    basis_list = init.copy()
    Y_avg = np.mean([np.dot(W, X) for W, X in zip(basis_list, X_list)], axis=0)
    # Start scaling
    g_norms = 0
    g_list = []
    for i in range(n_iter):
        g_norms = 0
        # Start inner loop: decrease the loss w.r.t to each W_j
        convergence = False
        for j in range(n_pb):
            X = X_list[j]
            W_old = basis_list[j].copy()
            # Y_denoise is the estimate of the sources without Y_j
            Y_denoise = Y_avg - W_old.dot(X) / n_pb
            # Perform one ICA quasi-Newton step
            converged, basis_list[j], g_norm = _noisy_ica_step(
                W_old, X, Y_denoise, noise, n_pb, ortho, scale=True
            )
            convergence = convergence or converged
            # Update the average vector (estimate of the sources)
            Y_avg += np.dot(basis_list[j] - W_old, X) / n_pb
            g_norms = max(g_norm, g_norms)

        # If line search does not converge for any subject we ll stop there
        if convergence is False:
            break

        if verbose:
            print(
                "it %d, loss = %.4e, g=%.4e"
                % (
                    i + 1,
                    _loss_total(basis_list, X_list, Y_avg, noise),
                    g_norms,
                )
            )
        if g_norms < tol_init:
            break
    # Start outer loop
    if timing:
        t0 = time()
        timings = []
    g_norms = 0
    for i in range(n_iter):
        g_norms = 0
        convergence = False
        # Start inner loop: decrease the loss w.r.t to each W_j
        for j in range(n_pb):
            X = X_list[j]
            W_old = basis_list[j].copy()
            # Y_denoise is the estimate of the sources without Y_j
            Y_denoise = Y_avg - W_old.dot(X) / n_pb
            # Perform one ICA quasi-Newton step
            converged, basis_list[j], g_norm = _noisy_ica_step(
                W_old, X, Y_denoise, noise, n_pb, ortho
            )
            # Update the average vector (estimate of the sources)
            Y_avg += np.dot(basis_list[j] - W_old, X) / n_pb
            g_norms = max(g_norm, g_norms)
            convergence = converged or convergence
        if convergence is False:
            break

        g_list.append(g_norms)
        if timing:
            timings.append(
                (
                    i,
                    time() - t0,
                    _loss_total(basis_list, X_list, Y_avg, noise),
                    g_norms,
                )
            )

        if verbose:
            print(
                "it %d, loss = %.4e, g=%.4e"
                % (
                    i + 1,
                    _loss_total(basis_list, X_list, Y_avg, noise),
                    g_norms,
                )
            )
        if g_norms < tol:
            break

    else:
        warnings.warn(
            "Multiview ICA has not converged - gradient norm: %e " % g_norms
        )
    if return_gradients:
        return basis_list, Y_avg, g_list

    if timing:
        return basis_list, Y_avg, timings
    return basis_list, Y_avg


def _loss_total(basis_list, X_list, Y_avg, noise):
    n_pb, p, _ = basis_list.shape
    loss = np.mean(_logcosh(Y_avg)) * p
    for i, (W, X) in enumerate(zip(basis_list, X_list)):
        Y = W.dot(X)
        loss -= np.linalg.slogdet(W)[1]
        loss += 1 / (2 * noise) * np.mean((Y - Y_avg) ** 2) * p
    return loss


def _loss_partial(W, X, Y_denoise, noise, n_pb):
    p, _ = W.shape
    Y = np.dot(W, X)
    loss = -np.linalg.slogdet(W)[1]
    loss += np.mean(_logcosh(Y / n_pb + Y_denoise)) * p
    fact = (1 - 1 / n_pb) / (2 * noise)
    loss += fact * np.mean((Y - n_pb * Y_denoise / (n_pb - 1)) ** 2) * p
    return loss


def _noisy_ica_step(
    W,
    X,
    Y_denoise,
    noise,
    n_pb,
    ortho,
    lambda_min=0.001,
    n_ls_tries=50,
    scale=False,
):
    """
    ICA minimization using quasi Newton method. Used in the inner loop.
    Returns
    -------
    converged: bool
        True if line search has converged
    new_W: np array of shape (m, p, p)
        New values for the basis
    g_norm: float
    """
    p, n = X.shape
    loss0 = _loss_partial(W, X, Y_denoise, noise, n_pb)
    Y = W.dot(X)
    Y_avg = Y / n_pb + Y_denoise

    # Compute relative gradient and Hessian
    thM = np.tanh(Y_avg)
    G = np.dot(thM, Y.T) / n / n_pb
    # print(G)
    const = 1 - 1 / n_pb
    res = Y - Y_denoise / const
    G += np.dot(res, Y.T) * const / noise / n
    G -= np.eye(p)
    if scale:
        G = np.diag(np.diag(G))
    # print(G)
    if ortho:
        G = 0.5 * (G - G.T)
    g_norm = np.max(np.abs(G))

    # These are the terms H_{ijij} of the approximated hessian
    # (approximation H2 in Pierre's thesis)
    h = np.dot((1 - thM ** 2) / n_pb ** 2 + const / noise, (Y ** 2).T,) / n

    # Regularize
    discr = np.sqrt((h - h.T) ** 2 + 4.0)
    eigenvalues = 0.5 * (h + h.T - discr)
    problematic_locs = eigenvalues < lambda_min
    np.fill_diagonal(problematic_locs, False)
    i_pb, j_pb = np.where(problematic_locs)
    h[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    # Compute Newton's direction
    det = h * h.T - 1
    direction = (h.T * G - G.T) / det
    if ortho:
        direction = 0.5 * (direction - direction.T)
    # print(direction)
    # Line search
    step = 1
    for j in range(n_ls_tries):
        if ortho:
            new_W = expm(-step * direction).dot(W)
        else:
            new_W = W - step * direction.dot(W)
        new_loss = _loss_partial(new_W, X, Y_denoise, noise, n_pb)
        if new_loss < loss0:
            return True, new_W, g_norm
        else:
            step /= 2.0
    return False, W, g_norm
