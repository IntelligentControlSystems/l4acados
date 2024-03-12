#!/usr/bin/env python
from setuptools import setup, find_packages
from os.path import dirname, abspath, basename
from distutils.core import setup

setup(
    name='zero_order_gpmpc',
    version='0.0.1',
    author='Amon Lahr',
    author_email='amlahr@ethz.ch',
    description='A tailored SQP algorithm for learning-based model predictive control with ellipsoidal uncertainties',
    packages = find_packages(),
    install_requires=[
        'gpytorch==1.11',
    ],
)
