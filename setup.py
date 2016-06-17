#!/usr/bin/env python

from __future__ import absolute_import
import io
import re
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def find_version():
    version_file = io.open(os.path.join(os.path.dirname(__file__), 'getdist/__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='GetDist',
      version=find_version(),
      description='GetDist Monte Carlo sample analysis, plotting and GUI',
      author='Antony Lewis',
      url="https://github.com/cmbant/getdist",
      packages=['getdist', 'getdist.gui', 'paramgrid', 'getdist_tests'],
      scripts=['GetDist.py', 'GetDistGUI.py'],
      test_suite='getdist_tests',
      package_data={'getdist': ['analysis_defaults.ini', 'distparam_template.ini']},
      requires=[
          'numpy',
          'matplotlib',
          'six',
          "scipy (>=0.11.0)",
          'PySide'],
      #  optional (for faster file read)
      # 'pandas (>=0.14.0)'
      classifiers=[
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
      ],
      keywords=['MCMC', 'KDE', 'sample', 'density estimation', 'plot']
      )
