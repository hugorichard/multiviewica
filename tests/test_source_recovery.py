# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import pytest
import numpy as np
from multiviewica import _hungarian, permica, groupica, multiviewica


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


@pytest.mark.parametrize(
    ("algo, init"),
    [
        (permica, None),
        (groupica, None),
        (multiviewica, "permica"),
        (multiviewica, "groupica"),
    ],
)
@pytest.mark.parametrize("dimension_reduction", ["pca", "srm"])
def test_ica(algo, dimension_reduction, init):
    # Test that all algo can recover the sources
    sigma = 1e-4
    n, v, p, t = 3, 10, 5, 1000
    # Generate signals
    rng = np.random.RandomState(0)
    S_true = rng.laplace(size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, v, p)
    noises = rng.randn(n, v, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [sigma * N for A, N in zip(A_list, noises)]
    # Run ICA
    if init is None:
        K, W, S = algo(
            X,
            n_components=5,
            dimension_reduction=dimension_reduction,
            tol=1e-5,
        )
    else:
        K, W, S = algo(
            X,
            n_components=5,
            dimension_reduction=dimension_reduction,
            tol=1e-5,
            init=init,
        )
    dist = np.mean([amari_d(W[i].dot(K[i]), A_list[i]) for i in range(n)])
    S = normalize(S)
    err = np.mean(error(np.abs(S.dot(S_true.T))))
    assert dist < 0.01
    assert err < 0.01


def test_supergaussian():
    # Test with super Gaussian data
    # should only work when density in the model is super-Gaussian
    rng = np.random.RandomState(0)
    sigma = 1e-4
    n, p, t = 8, 2, 1000
    S_true = rng.laplace(size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, p, p)
    noises = rng.randn(n, p, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [sigma * N for A, N in zip(A_list, noises)]

    for fun in ["quadratic", "logcosh", "abs"]:
        K, W, S = multiviewica(X, fun=fun)
        dist = np.mean([amari_d(W[i], A_list[i]) for i in range(n)])
        S = normalize(S)
        err = np.mean(error(np.abs(S.dot(S_true.T))))
        print(dist, err, fun)
        if fun == "quadratic":
            assert dist > 0.1
            assert err > 0.1
        else:
            assert dist < 0.01
            assert err < 0.01


def test_subgaussian():
    # Test with super Gaussian data
    # should only work when density in the model is super-Gaussian
    rng = np.random.RandomState(0)
    sigma = 1e-5
    n, p, t = 8, 2, 1000
    S_true = rng.uniform(-1, 1, size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, p, p)
    noises = rng.randn(n, p, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [sigma * N for A, N in zip(A_list, noises)]

    for fun in ["quadratic", "logcosh", "abs"]:
        K, W, S = multiviewica(X, fun=fun)
        dist = np.mean([amari_d(W[i], A_list[i]) for i in range(n)])
        S = normalize(S)
        err = np.mean(error(np.abs(S.dot(S_true.T))))
        if fun == "quadratic":
            assert dist < 0.01
            assert err < 0.01
        else:
            assert dist > 0.1
            assert err > 0.1
