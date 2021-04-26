# utf-8
# Python 3.9
# 2021-04-13


import setuptools
import hyper_tuner


setuptools.setup(
    name="hyper_tuner",
    version=hyper_tuner.__version__,
    description="Hyperparameter's tuner for ML models",
    long_description=open("README.md").read(),
    author="Ivan Strazov",
    author_email="ivanstrazov@gmail.com",
    url="https://github.com/IvanStrazov/hyper_tuner/",
    # license=open("LICENSE").read(),
    keywords="hyperparameters tune ml",

    packages=setuptools.find_packages(exclude=("config"))
)
