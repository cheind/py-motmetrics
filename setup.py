"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from distutils.core import setup

setup(
    name='motmetrics',
    version=open('motmetrics/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Metrics for multiple object tracker benchmarking.',
    author='Christoph Heindl',
    url='https://github.com/cheind/py-motmetrics',
    packages=['motmetrics', 'motmetrics.tests', 'motmetrics.apps'],
)