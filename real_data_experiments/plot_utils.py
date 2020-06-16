"""
==============================
Plot utils
==============================


"""

# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause


import scipy.stats as st
import numpy as np


def confidence_interval(a, conf=0.95):
    """
    Return a confidence interval assuming normal distribution
    Parameters
    ----------
    a: np array
        input data
    conf: float
        level of confidence
    Returns
    -------
        (low, high): low and high limit of the interval
    """
    return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
