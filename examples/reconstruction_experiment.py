"""
==============================
Reconstruction experiment
==============================


"""


# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np

import os

from multiviewica import permica, groupica, multiviewica
from fmri_utils import load_and_concat
import matplotlib.pyplot as plt
from plot_utils import confidence_interval

n_components = 5
n_seeds = 10

n_subjects = 17
n_runs = 5
paths = np.array(
    [
        [
            os.path.join(
                "..", "data", "masked_movie_files", "sub%i_run%i.npy" % (i, j)
            )
            for j in range(n_runs)
        ]
        for i in range(n_subjects)
    ]
)

algos = [
    ("srm", "MultiViewICA", multiviewica),
    ("pca", "PCA+GroupICA", groupica),
    ("srm", "PermICA", permica),
]

res = []
for seed in range(n_seeds):
    rng = np.random.RandomState(seed)
    shuffled_subs = np.arange(n_subjects)
    rng.shuffle(shuffled_subs)
    train_subs = shuffled_subs[: int(0.8 * n_subjects)]
    test_subs = shuffled_subs[int(0.8 * n_subjects) :]
    shuffled_runs = np.arange(n_runs)
    rng.shuffle(shuffled_runs)
    train_runs = shuffled_runs[: int(0.8 * n_runs)]
    test_runs = shuffled_runs[int(0.8 * n_runs) :]
    train_paths = paths[np.arange(n_subjects), :][:, train_runs]
    test_paths = paths[train_subs, :][:, test_runs]
    validation_paths = paths[test_subs, :][:, test_runs]
    data_train = load_and_concat(train_paths)
    data_test = load_and_concat(test_paths)
    data_val = load_and_concat(validation_paths)
    res_ = []
    for dimension_reduction, name, algo in algos:
        K, W, S = algo(
            data_train,
            n_components=n_components,
            dimension_reduction=dimension_reduction,
            tol=1e-5,
            max_iter=10000,
        )
        forward = [W[i].dot(K[i]) for i in range(n_subjects)]
        backward = [np.linalg.pinv(forward[i]) for i in range(n_subjects)]
        # Let us use dual regression for PCA+GroupICA to increase perf
        if name == "PCA+GroupICA":
            backward = [x.dot(np.linalg.pinv(S)) for x in data_train]
            forward = np.linalg.pinv(backward)
        # if name == "PermICA" or name == "MultiViewICA":
        #     # With PermICA and MultiViewICA we use SRM as preprocessing
        #     A, X_train = srm(train_paths, n_components=n_components)
        #     W, S = algo(X_train, tol=1e-5, max_iter=10000)
        #     forward = [W[i].dot(A[i].T) for i in range(n_subjects)]
        #     backward = [
        #         A[i].dot(np.linalg.inv(W[i])) for i in range(n_subjects)
        #     ]
        # elif name == "PCA+GroupICA":
        #     # With PCA+GroupICA we use subject specific PCA
        #     A, X_train = reduce_data(train_paths, n_components=n_components)
        #     W, S = algo(X_train, tol=1e-5, max_iter=10000)
        #     # We use double regression to compute forward operator
        #     backward = online_dot(train_paths, np.linalg.pinv(S))
        #     forward = [np.linalg.pinv(b) for b in backward]

        shared_test = np.mean(
            [forward[i].dot(data_test[k]) for k, i in enumerate(train_subs)],
            axis=0,
        )

        pred = [backward[i].dot(shared_test) for i in test_subs]
        var_e = np.mean(
            [
                1 - (data_val[k] - pred[k]).var(axis=1)
                for k in range(len(test_subs))
            ],
            axis=0,
        )
        mean_r2 = np.mean(var_e)
        res_.append(mean_r2)
    res.append(res_)

# Plotting
cm = plt.cm.tab20

algos = [
    ("MultiViewICA", cm(0)),
    ("PCA+GroupICA", cm(7)),
    ("PermICA", cm(2)),
]

res = np.array(res)
fig, ax = plt.subplots()
for i, (algo, color) in enumerate(algos):
    res_algo = res[:, i].flatten()
    av = np.mean(res_algo)
    low, high = confidence_interval(res_algo)
    low = av - low
    high = high - av
    ax.bar(
        i,
        height=[av],
        width=0.8,
        label=algo,
        color=color,
        yerr=np.array([[low], [high]]),
    )
plt.ylabel(r"Mean R2 score")
plt.xticks([0, 1, 2], ["MultiViewICA", "PCA+GroupICA", "PermICA"])
fig.legend(
    ncol=3, loc="upper center",
)
plt.savefig(
    "../figures/reconstruction.png", bbox_inches="tight",
)
