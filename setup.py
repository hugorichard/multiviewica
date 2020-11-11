from setuptools import setup, find_packages

setup(
    name="multiviewica",
    description="Multi-view ICA",
    version="0.0.1",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['numpy>=1.12', 'scikit-learn>=0.23', 'python-picard',
                      'fastsrm', 'scipy>=0.18.0', 'matplotlib>=2.0.0']
)
