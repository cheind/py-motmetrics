
from distutils.core import setup

setup(
    name='motmetrics',
    version=open('motmetrics/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Metrics for multiple object tracker evaluation.',
    author='Christoph Heindl',
    url='https://github.com/cheind/py-motmetrics',
    packages=['motmetrics', 'motmetrics.test', 'motmetrics.apps'],
)