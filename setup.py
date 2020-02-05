"""py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import io

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with io.open('requirements.txt') as f:
    required = f.read().splitlines()

with io.open('Readme.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='motmetrics',
    version=io.open('motmetrics/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Metrics for multiple object tracker benchmarking.',
    author='Christoph Heindl, Jack Valmadre',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cheind/py-motmetrics',
    license='MIT',
    install_requires=required,
    packages=['motmetrics', 'motmetrics.tests', 'motmetrics.apps'],
    include_package_data=True,
    keywords='tracker MOT evaluation metrics compare'
)
