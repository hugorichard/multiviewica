import pytest
import numpy as np
from multiviewica.permica import _hungarian, permica
from multiviewica.groupica import groupica
from multiviewica.multiviewica import multiviewica


def normalize(A):
    A_ = A - np.mean(A, axis=1, keepdims=True)
    A_ = A_ / np.linalg.norm(A_, axis=1, keepdims=True)
    return A_


def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


def error(M):
    order, _ = _hungarian(M)
    return 1 - M[np.arange(M.shape[0]), order]


@pytest.mark.parametrize("algo", [permica, groupica, multiviewica])
def test_multiview(algo):
    sigma = 1e-4
    # Test that multiview is better than perm_ica
    n, p, t = 3, 5, 1000
    # Generate signals
    rng = np.random.RandomState(None)
    S_true = rng.laplace(size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, p, p)
    noises = rng.randn(n, p, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [sigma * N for A, N in zip(A_list, noises)]
    # Run ICA
    W, S = algo(X, tol=1e-5)
    dist = np.mean([amari_d(W[i], A_list[i]) for i in range(n)])
    S = normalize(S)
    err = np.mean(error(np.abs(S.dot(S_true.T))))
    assert dist < 0.01
    assert err < 0.01
