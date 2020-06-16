"""
==============================
Time segment matching utils
==============================


"""

# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import numpy as np
import scipy.stats as stats


def time_segment_matching(
    data, win_size=10,
):
    """
    Performs time segment matching experiment
    (code inspired from brainiak tutorials at
    https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/10-func-align.html)

    Parameters
    ----------
    data: array of shape (n_subjects, n_components, n_timeframes)
        Input shared responses
    Returns
    -------
    cv_score: np array of shape (n_subjects)
        Per-subject accuracy
    """
    # Pull out shape information
    n_subjs = len(data)
    (n_features, n_TR) = data[0].shape  # Voxel/feature by timepoint

    # How many segments are there (account for edges)
    n_seg = n_TR - win_size

    # mysseg prediction prediction
    train_data = np.zeros((n_features * win_size, n_seg))

    # Concatenate the data across participants
    for ppt_counter in range(n_subjs):
        for window_counter in range(win_size):
            train_data[
                window_counter
                * n_features : (window_counter + 1)
                * n_features,
                :,
            ] += data[ppt_counter][:, window_counter : window_counter + n_seg]

    # Iterate through the participants, leaving one out
    accuracy = np.zeros(shape=n_subjs)
    for ppt_counter in range(n_subjs):

        # Preset
        test_data = np.zeros((n_features * win_size, n_seg))

        for window_counter in range(win_size):
            test_data[
                window_counter
                * n_features : (window_counter + 1)
                * n_features,
                :,
            ] = data[ppt_counter][:, window_counter : (window_counter + n_seg)]

        # Take this participant data away
        train_ppts = stats.zscore((train_data - test_data), axis=0, ddof=1)
        test_ppts = stats.zscore(test_data, axis=0, ddof=1)

        # Correlate the two data sets
        corr_mtx = test_ppts.T.dot(train_ppts)

        # If any segments have a correlation difference less than the window size and they aren't the same segments then set the value to negative infinity
        for seg_1 in range(n_seg):
            for seg_2 in range(n_seg):
                if abs(seg_1 - seg_2) < win_size and seg_1 != seg_2:
                    corr_mtx[seg_1, seg_2] = -np.inf

        # Find the segement with the max value
        rank = np.argmax(corr_mtx, axis=1)

        # Find the number of segments that were matched for this participant
        accuracy[ppt_counter] = sum(rank == range(n_seg)) / float(n_seg)

    return accuracy
