"""
==============================
Reconstruction utils
==============================


"""

# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

import nibabel as nib
import os
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiMasker
import numpy as np


def get_sherlock_roi():
    """
    Return sherlock roi as a numpy array
    """
    example_img = nib.load(
        os.path.join("..", "data", "movie_files", "sherlock_movie_s1.nii")
    )
    mask_img = nib.load(os.path.join("..", "data", "mask_img.nii.gz"))
    mask_img = resample_to_img(mask_img, example_img, interpolation="nearest")
    masker = NiftiMasker(mask_img=mask_img,).fit()
    sherlock_img = nib.load(os.path.join("..", "data", "sherlock_roi.nii.gz"))
    sherlock_img = resample_to_img(
        sherlock_img, example_img, interpolation="nearest"
    )
    sherlock_roi = masker.transform(sherlock_img).astype(int)[0]
    return sherlock_roi
