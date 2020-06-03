import numpy as np
from multiviewica.permica import permica
from multiviewica.groupica import groupica
from multiviewica.multiviewica import multiviewica
import matplotlib.pyplot as plt


def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


algos = [
    ("MultiViewICA", "cornflowerblue", multiviewica),
    ("PermICA", "green", permica),
    ("GroupICA", "coral", groupica),
]
plots = []
for name, color, algo in algos:
    means = []
    lows = []
    highs = []
    sigmas = np.logspace(-2, 1, 6)
    for sigma in sigmas:
        dists = []
        for seed in range(10):
            n, p, t = 10, 3, 1000
            rng = np.random.RandomState(None)
            S_true = rng.laplace(size=(p, t))
            A_list = rng.randn(n, p, p)
            noises = rng.randn(n, p, t)
            X = np.array([A.dot(S_true) for A in A_list])
            X += [sigma * A.dot(N) for A, N in zip(A_list, noises)]
            W, S = algo(X, tol=1e-4, max_iter=10000)
            dist = np.mean([amari_d(W[i], A_list[i]) for i in range(n)])
            dists.append(dist)
        dists = np.array(dists)
        mean = np.mean(dists)
        low = np.quantile(dists, 0.1)
        high = np.quantile(dists, 0.9)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    lows = np.array(lows)
    highs = np.array(highs)
    means = np.array(means)
    plots.append((highs, lows, means))

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.latex.preview": True,
}
plt.rcParams.update(rc)
plt.figure(figsize=(4, 2))
for i, (name, color, algo) in enumerate(algos):
    highs, lows, means = plots[i]
    plt.fill_between(
        sigmas, lows, highs, color=color, alpha=0.3,
    )
    plt.loglog(
        sigmas, means, label=name, color=color,
    )
plt.legend()
x_ = plt.xlabel(r"Data noise")
y_ = plt.ylabel(r"Amari distance")
plt.savefig(
    "synthetic_experiment.png",
    bbox_extra_artists=[x_, y_],
    bbox_inches="tight",
)
