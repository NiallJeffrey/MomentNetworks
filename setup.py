#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

# tensorflow also required, but can break setup if requires
setup(name='momentnetworks',
      version='0.1',
      description='Estimation_posterior_moments',
      author='Niall Jeffrey',
      url='https://github.com/NiallJeffrey/MomentNetworks',
      packages=find_packages(),
      install_requires=[
          "emcee",
          "getdist",
	  "chainconsumer",
          "scipy",
	  "matplotlib",
      ])
