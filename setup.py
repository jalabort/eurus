import os
from setuptools import setup, find_packages


setup(
    name='eurus',
    version='0.0.0',
    packages=find_packages(),
    scripts=[[os.path.join(root, f) for f in files]
             for root, _, files in
             os.walk(os.path.join('eurus', 'cli', 'bin'))][0],
)
