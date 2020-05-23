from setuptools import setup, find_packages

setup(
    name="multiviewica",
    install_requires=["scikit-learn"],
    author="Hugo RICHARD, Luigi GRESELE",
    author_email="hugo.richard@inria.fr, luigi.gresele@inria.fr",
    description="Non linear SRM",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
)
