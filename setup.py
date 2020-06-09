from setuptools import setup, find_packages

setup(
    name="multiviewica",
    install_requires=["scikit-learn"],
    author="Hugo RICHARD, Luigi GRESELE, Pierre ABLIN",
    author_email="hugo.richard@inria.fr, luigi.gresele@inria.fr, pierre.ablin@gmail.com",
    description="Multi-view ICA",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
)
