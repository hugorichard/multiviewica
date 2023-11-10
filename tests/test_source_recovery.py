# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import pytest
import numpy as np
from multiviewica import _hungarian, permica, groupica, multiviewica


def normalize(A):
    A_ = A - np.mean(A, axis=1, keepdims=True)
    A_ = A_ / np.std(A_, axis=1, keepdims=True)
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
    rng = np.random.RandomState()
    sigmas = rng.randn(3) 
    n, p, t = 5, 3, 1000
    S_true = rng.laplace(size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, p, p)
    noises = rng.randn(n, p, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [A.dot(sigmas.reshape(-1, 1) * N) for A, N in zip(A_list, noises)]
    W_init = rng.randn(n, p, p)

    for fun in ["quartic", "logcosh", "abs"]:
        K, W, S = multiviewica(X, init=W_init, fun=fun)
        dist = np.mean([amari_d(W[i], A_list[i]) for i in range(n)])
        print(fun, dist)

def test_subgaussian():
    # Test with sub Gaussian data
    # should only work when density in the model is sub-Gaussian
    rng = np.random.RandomState(0)
    n, p, t = 4, 4, 1000
    cov = rng.randn(p).reshape(-1, 1)
    S_true = rng.randn(p, t)
    S_true = np.sign(S_true) * np.abs(S_true) ** 0.7
    S_true = normalize(S_true)
    A_list = rng.randn(n, p, p)
    noises = rng.randn(n, p, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [A.dot(cov.reshape(-1, 1) * N) for A, N in zip(A_list, noises)]
    W_init = rng.randn(n, p, p)
    K, W, S = multiviewica(X, init=W_init)
    for fun in ["quartic", "logcosh", "abs"]:
        K, W, S = multiviewica(X, init=W_init, fun=fun)
        dist = np.mean([amari_d(W[i], A_list[i]) for i in range(n)])
        print(fun, dist)

def dist(W, A):
    return np.mean([amari_d(w, a) for w, a in zip(W, A)])

def test_gaussianf():
    n_samples = 1000
    n_components = 4
    n_subjects = 4
    # Test with super Gaussian data:
    # should only work when density in the model is super-Gaussian
    rng = np.random.RandomState(0)
    cov = rng.randn(n_components).reshape(-1, 1)
    S = rng.randn(n_components, n_samples)
    S = normalize(S)
    A = rng.randn(n_subjects, n_components, n_components)
    N = np.array(
        [cov * rng.randn(n_components, n_samples) for _ in range(n_subjects)]
    )
    X = np.array([a.dot(S + n) for a, n in zip(A, N)])
    W_init = rng.randn(n_subjects, n_components, n_components)
    W_true = np.array([np.linalg.pinv(a) for a in A])
    _, W_gica2, S = groupica(X)
    _, W_mv, S_mv = multiviewica(
        X, init=W_init, verbose=True, max_iter=5000, tol=1e-5
    )
    # W_ga, _, _, S_ga = gavica_em(X, W_init, n_samples, verbose=False, tol=1e-7)
    res = np.array([dist(W, A) for W in [W_gica2, W_mv]])
    print(res)

