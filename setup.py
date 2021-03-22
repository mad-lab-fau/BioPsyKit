"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='biopsykit',
    version='0.2.0',
    description='Library for analyzing ECG data.',

    packages=find_packages(exclude='examples'),

    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pytz',
        'seaborn',
        'neurokit2',
        'tqdm'
    ],
)
