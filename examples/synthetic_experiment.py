import numpy as np
from multiviewica.permica import permica
from multiviewica.groupica import groupica
from multiviewica.multiviewica import multiviewica
import matplotlib.pyplot as plt
from tqdm import tqdm

# Make experiment
def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


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
algos = [
    ("MultiViewICA", "cornflowerblue", multiviewica),
    ("PermICA", "green", permica),
    ("GroupICA", "coral", groupica),
]
for name, color, algo in algos:
    means = []
    lows = []
    highs = []
    sigmas = np.logspace(-3, 1, 4)
    for sigma in tqdm(sigmas):
        dists = []
        for seed in range(5):
            # Test that multiview is better than perm_ica
            n, p, t = 10, 3, 1000
            # Generate signals
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
    plt.fill_between(
        sigmas, np.array(lows), np.array(highs), color=color, alpha=0.3,
    )
    plt.loglog(
        sigmas, means, label=name, color=color,
    )

plt.legend()
x_ = plt.xlabel(r"Data noise")
y_ = plt.ylabel(r"Amari distance")
plt.savefig(
    "synthetic_experiment.pdf",
    bbox_extra_artists=[x_, y_],
    bbox_inches="tight",
)
