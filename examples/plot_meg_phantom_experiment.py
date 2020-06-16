"""
================================
Experitments on MEG Phantom data
================================


"""


# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import find_events
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

from multiviewica import multiviewica, permica, groupica


data_path = bst_phantom_elekta.data_path()

raw_fname = op.join(data_path, "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif")
raw = read_raw_fif(raw_fname)

# Delete bad sensor

events = find_events(raw, "STI201")
raw.info["bads"] = ["MEG2421"]

# Filter

raw.fix_mag_coil_types()
raw = mne.preprocessing.maxwell_filter(raw, origin=(0.0, 0.0, 0.0))
raw.filter(1.0, 40.0)


# Select magnometers
picks = mne.pick_types(raw.info, meg="mag")

# Epoch signals
tmin, tmax = -0.2, 0.5
epochs = mne.Epochs(
    raw,
    events,
    None,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    preload=True,
)
# Get data from two consecutive epochs
n_epochs = 1
n_dipoles = 3
X = np.array(
    [
        np.hstack(epochs[str(i)].get_data()[1: 1 + n_epochs])
        for i in range(1, 1 + n_dipoles)
    ]
)

# Normalize
X /= np.std(X)

# Apply individual PCA
dim = 20

# Get sources with multiviewICA
K_m, W_m, S_multiview = multiviewica(
    X,
    verbose=True,
    n_components=dim,
    dimension_reduction="pca",
    random_state=0,
)
K_g, W_g, S_group = groupica(
    X, n_components=dim, dimension_reduction="pca", random_state=0
)
K_p, W_p, S_perm = permica(
    X, n_components=dim, dimension_reduction="pca", random_state=0
)

S_m = [np.dot(W, K.dot(x)) for K, W, x in zip(K_m, W_m, X)]
S_g = [np.dot(W, K.dot(x)) for K, W, x in zip(K_g, W_g, X)]
S_p = [np.dot(W, K.dot(x)) for K, W, x in zip(K_p, W_p, X)]
# Compute true source
true_source = np.zeros(701)
true_source[200:320] = np.sin(np.linspace(0, 5 * np.pi, 120))
true_source /= np.std(true_source)
true_source = np.tile(true_source, n_epochs)


def find_best_source(S):
    S /= np.std(S, axis=1)[:, None]
    corr = np.dot(S, true_source)
    source_idx = np.argmax(np.abs(corr))
    return S[source_idx] * np.sign(corr[source_idx])


time = np.linspace(-0.2, 0.7 * n_epochs, len(true_source))

for i in range(n_dipoles):
    plt.figure()
    for S, name in zip(
        [S_g, S_p, S_m], ["Group ICA", "Perm ICA", "Multiview ICA"]
    ):
        plt.plot(time, find_best_source(S[i]), label=name)
    plt.plot(time, true_source, label="True source", color="k")
    plt.title("Dipole %d" % (i + 1))
    plt.legend()
plt.show()
