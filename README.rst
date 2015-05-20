===================
GetDist
===================
:GetDist: MCMC sample analysis, plotting and GUI
:Version: 0.2.0
:Author: Antony Lewis
:Homepage: http://cosmologist.info/
.. image:: https://secure.travis-ci.org/cmbant/getdist.png?branch=master
  :target: https://secure.travis-ci.org/cmbant/getdist

Description:
============

GetDist is a package for analysing Monte Carlo samples, including correlated samples 
from Markov Chain Monte Carlo (MCMC). 

* **Point and click GUI** - select chain files, view plots, marginalized constraints, latex tables and more
* **Plotting library** - make custom publication-ready 1D, 2D, 3D-scatter, triangle and other plots
* **Named parameters** - simple handling of many parameters using parameter names 
* **Optimized Kernel Density Estimation** - automated optimal bandwidth choice for 1D and 2D densities (Botev et al. Improved Sheather-Jones method)
* **Convergence diagonistics** - including correlation length and diagonalized Gelman-Rubin statistics

Getting Started:
================
Install getdist from download using::

    $ sudo python setup.py install

You can test if things are working using the unit test by running

    $ python setup.py test


Dependencies:
=============
* Python 2.7+ or 3.4+
* PySide (optional, only needed for GUI)
* matplotlib
* scipy
* Working latex installation (for some plotting/table functions)

Using with CosmoMC
=============

This GetDist package is general, but is mainly developed for analysing cosmology data
using chains from the CosmoMC program. No need to install this package separately if you
have a full CosmoMC installation. Detailed help is available for plotting Planck chains
and using CosmoMC parameter grids

 http://cosmologist.info/cosmomc/readme_python.html


