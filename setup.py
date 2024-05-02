#!/usr/bin/env python
from setuptools import setup, find_packages
from distutils.core import setup

setup(
    name="zero_order_gpmpc",
    version="0.0.1",
    author="Amon Lahr, Joshua Näf",
    author_email="amlahr@ethz.ch",
    description="A tailored SQP algorithm for learning-based model predictive control with ellipsoidal uncertainties",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "linear_operator @ git+https://github.com/naefjo/linear_operator@feature/exo-gp",
        "gpytorch @ git+https://github.com/naefjo/gpytorch@feature/exo-gp",
    ],
)
