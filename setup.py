from setuptools import setup, find_packages

setup(
    name="multiviewica",
    description="Multi-view ICA",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['numpy>=1.12', 'scipy>=0.18.0',
                      'matplotlib>=2.0.0',
                      'scikit-learn>=0.23', 'mne>=0.20', 'python-picard',
                      'nibabel', 'fastsrm', 'nilearn']
)
