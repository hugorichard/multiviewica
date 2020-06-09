import numpy as np

import os

from multiviewica.permica import permica
from multiviewica.groupica import groupica
from multiviewica import multiviewica
from multiviewica.reduce_data import reduce_data, load_and_concat, online_dot
from reconstruction_utils import get_sherlock_roi
import matplotlib.pyplot as plt
from plot_utils import confidence_interval

n_components = 5
n_seeds = 25

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
    ("MultiViewICA", multiviewica),
    ("PCA+GroupICA", groupica),
    ("PermICA", permica),
]

sherlock_roi = get_sherlock_roi()
res = []
for seed in range(n_seeds):
    rng = np.random.RandomState(seed)
    shuffled_subs = np.arange(n_subjects)
    rng.shuffle(shuffled_subs)
    train_subs = shuffled_subs[: int(0.8 * n_subjects)]
    test_subs = shuffled_subs[int(0.8 * n_subjects):]
    shuffled_runs = np.arange(n_runs)
    rng.shuffle(shuffled_runs)
    train_runs = shuffled_runs[: int(0.8 * n_runs)]
    test_runs = shuffled_runs[int(0.8 * n_runs):]

    train_paths = paths[np.arange(n_subjects), :][:, train_runs]
    test_paths = paths[train_subs, :][:, test_runs]
    validation_paths = paths[test_subs, :][:, test_runs]

    A, X_train = reduce_data(train_paths, n_components=n_components)
    data_test = load_and_concat(test_paths)
    data_val = load_and_concat(validation_paths)

    res_ = []
    for name, algo in algos:
        W, S = algo(X_train, tol=1e-4, max_iter=10000)

        # With PCA+GroupICA we use double regression to compute forward
        if name == "PCA+GroupICA":
            backward = online_dot(train_paths, np.linalg.pinv(S))
            forward = [np.linalg.pinv(b) for b in backward]
        else:
            forward = [W[i].dot(A[i].T) for i in range(n_subjects)]
            backward = [
                A[i].dot(np.linalg.inv(W[i])) for i in range(n_subjects)
            ]

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
        mean_r2 = np.mean(var_e[sherlock_roi])
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
    "reconstruction.png", bbox_inches="tight",
)
