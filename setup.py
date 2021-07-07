#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='neuralcvd',
    version='0.0.1',
    description='Neural network-based integration of polygenic and clinical information: Development and validation of a prediction model for 10 year risk of major adverse cardiac events in the UK Biobank cohort',
    author='TeamCardioRS',
    author_email='thore.buergel@charite.de',
    url='https://github.com/thbuerg/NeuralCVD.git',
    install_requires=[
        'pytorch-lightning',
                      ],
    packages=find_packages(),
)