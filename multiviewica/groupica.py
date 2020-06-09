import numpy as np
from picard import picard


def groupica(X, max_iter=1000, random_state=None, tol=1e-7):
    """
    Performs PCA on concatenated data across groups (ex: subjects)
    and apply ICA on reduced data.

    Parameters
    ----------
    X : np array of shape (n_groups, n_components, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
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
    W : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    S : np array of shape (n_components, n_samples)
        Estimated source
    """
    n_pb, p, n = X.shape
    X_concat = np.vstack(X)
    U, S, V = np.linalg.svd(X_concat, full_matrices=False)
    U = U[:, :p]
    S = S[:p]
    V = V[:p]
    X_reduced = np.diag(S).dot(V)
    U = np.split(U, n_pb, axis=0)
    K, W, S = picard(
        X_reduced,
        ortho=False,
        extended=False,
        centering=False,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    scale = np.linalg.norm(S, axis=1)
    S = S / scale[:, None]
    W = np.dot(W, K) / scale[:, None]
    return np.array([W.dot(np.linalg.inv(u)) for u in U]), S
