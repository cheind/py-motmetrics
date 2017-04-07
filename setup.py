
from distutils.core import setup

setup(
    name='clearmot',
    version=open('clearmot/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='CLEAR MOT metrics for multiple object tracker evaluation',
    author='Christoph Heindl',
    url='https://github.com/cheind/py-clearmot',
    packages=['clearmot', 'clearmot.test', 'clearmot.apps'],
)