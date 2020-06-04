import numpy as np
from sklearn.model_selection import KFold

from multiviewica.permica import permica
from multiviewica.groupica import groupica
from multiviewica.multiviewica import multiviewica
from multiviewica.reduce_data import reduce_data, load_and_concat, online_dot
from timesegment_matching_utils import time_segment_matching
import os
import joblib
import matplotlib.pyplot as plt
from plot_utils import confidence_interval


n_subjects = 17
n_runs = 5
n_components = 5

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
cv = KFold(n_splits=5, shuffle=False)
res = []
for i, (train_runs, test_runs) in enumerate(cv.split(np.arange(n_runs))):
    train_paths = paths[:, train_runs]
    test_paths = paths[:, test_runs]
    A, X_train = reduce_data(train_paths, n_components=n_components)
    data_test = load_and_concat(test_paths)
    res_ = []
    for name, algo in algos:
        W, S = algo(X_train, tol=1e-4, max_iter=10000)
        # With PCA+GroupICA we use double regression to compute forward
        if name == "PCA+GroupICA":
            backward = online_dot(train_paths, np.linalg.pinv(S))
            forward = [np.linalg.pinv(b) for b in backward]
        else:
            forward = [W[i].dot(A[i].T) for i in range(n_subjects)]
        shared = np.array(
            [forward[i].dot(data_test[i]) for i in range(n_subjects)]
        )
        cv_scores = time_segment_matching(shared, win_size=9)
        res_.append(cv_scores)
    res.append(res_)

# Plotting
cm = plt.cm.tab20

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.latex.preview": True,
}
plt.rcParams.update(rc)

algos = [
    ("MultiViewICA", cm(0)),
    ("PCA+GroupICA", cm(7)),
    ("PermICA", cm(2)),
]

res = np.array(res)

fig, ax = plt.subplots()
for i, (algo, color) in enumerate(algos):
    res_algo = res[:, i, :].flatten()
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
plt.ylabel(r"Accuracy")
plt.xticks([0, 1, 2], ["MultiViewICA", "PCA+GroupICA", "PermICA"])
fig.legend(
    ncol=3, loc="upper center",
)
plt.savefig(
    "timesegment_matching.png", bbox_inches="tight",
)
