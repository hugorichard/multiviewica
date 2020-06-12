import nibabel as nib
from glob import glob
import numpy as np
import os
from nilearn.image import new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img

example_img = nib.load(os.path.join("movie_files", "sherlock_movie_s1.nii"))
mask_img = nib.load("sherlock_roi.nii.gz")
mask_img = resample_to_img(mask_img, example_img, interpolation="nearest")
masker = NiftiMasker(
    mask_img=mask_img,
    standardize=True,
    smoothing_fwhm=None,
    detrend=True,
    high_pass=1.0 / 140,
    t_r=1.5,
).fit()

# Let us first split sherlock data into 5 runs
for i, f in enumerate(glob(os.path.join("movie_files", "*"))):
    niimg = nib.load(f)
    X = niimg.get_data()
    X = np.array_split(X, 5, axis=-1)
    for j, x in enumerate(X):
        os.makedirs("masked_movie_files", exist_ok=True)
        np.save(
            os.path.join("masked_movie_files", "sub%i_run%i" % (i, j)),
            masker.transform(new_img_like(niimg, x)).T,
        )
