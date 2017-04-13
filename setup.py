"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='motmetrics',
    version=open('motmetrics/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Metrics for multiple object tracker benchmarking.',    
    author='Christoph Heindl',
    url='https://github.com/cheind/py-motmetrics',
    license='MIT',
    install_requires=required,
    packages=['motmetrics', 'motmetrics.tests', 'motmetrics.apps'],
    keywords='tracker MOT evaluation metrics compare'
)