===================
GetDist
===================
:GetDist: MCMC sample analysis, plotting and GUI
:Version: 0.2.0
:Author: Antony Lewis
:Homepage: https://github.com/cmbant/getdist

.. image:: https://secure.travis-ci.org/cmbant/getdist.png?branch=master
  :target: https://secure.travis-ci.org/cmbant/getdist
.. image:: http://img.shields.io/pypi/v/GetDist.svg?style=flat
        :target: https://pypi.python.org/pypi/GetDist/

Description
============

GetDist is a package for analysing Monte Carlo samples, including correlated samples
from Markov Chain Monte Carlo (MCMC).

* **Point and click GUI** - select chain files, view plots, marginalized constraints, latex tables and more
* **Plotting library** - make custom publication-ready 1D, 2D, 3D-scatter, triangle and other plots
* **Named parameters** - simple handling of many parameters using parameter names, including LaTex labels and prior bounds
* **Optimized Kernel Density Estimation** - automated optimal bandwidth choice for 1D and 2D densities (Botev et al. Improved Sheather-Jones method), with boundary and bias correction
* **Convergence diagnostics** - including correlation length and diagonalized Gelman-Rubin statistics
* **Latex tables** for marginalized 1D constraints

See the `Plot Gallery <http://htmlpreview.github.io/?https://github.com/cmbant/getdist/blob/master/docs/plot_gallery.html>`_.


Getting Started
================

Install getdist using pip::

    $ sudo pip install getdist

or from source files using::

    $ sudo python setup.py install

You can test if things are working using the unit test by running::

    $ python setup.py test

Check the dependencies listed in the next section are installed. You can then use the getdist module from your scripts, or
use the GUI program GetDistGUI.py.


Dependencies
=============
* Python 2.7+ or 3.4+
* matplotlib
* scipy
* PySide (optional, only needed for GUI)
* Working latex installation (for some plotting/table functions)

Python distributions like Anaconda have most of what you need (except for latex). To install binary backages on Linux-like systems
install pacakages *py-matplotlib, py-scipy, py-pyside, texlive-latex-extra, texlive-fonts-recommended, dvipng*. 
For example on a Mac using Python 2.7 from `MacPorts <https://www.macports.org/install.php>`_::

   sudo port install python27
   sudo port select --set python python27
   sudo port install py-matplotlib
   sudo port install py-scipy
   sudo port install py-pyside
   sudo port install texlive-latex-extra
   sudo port install texlive-fonts-recommended
   sudo port install dvipng

Algorithm details
==================

Details of kernel density estimation (KDE) algorithms and references are give in the
`GetDist Notes <http://cosmologist.info/notes/GetDist.pdf>`_.

Samples file format
===================

The GetDist GUI (and getdist.loadMCSamples function) read parameter sample/chain files in plain text format.
In general there are a set of plain text files of the form::
  
  xxx_1.txt
  xxx_2.txt
  ...
  xxx.paramnames
  xxx.ranges

where "xxx" is some root file name.

The .txt files are separate chain files (there can also be just one xxx.txt file). Each row of each sample .txt files is in the format

    *weight like param1 param2 param3* ...

The *weight* gives the number of samples (or importance weight) with these parameters. *like* gives -log(likelihood), and *param1, param2...* are the values of the parameters at the sample point. The first two columns can be 1 and 0 if they are not known or used.

The .paramnames file lists the names of the parameters, one per line, optionally followed by a latex label. Names cannot include spaces, and if they end in "*" they are interpreted as derived (rather than MCMC) parameters, e.g.::

 x1   x_1
 y1   y_1
 x2   x_2
 xy*  x_1+y_1

The .ranges file gives hard bounds for the parameters, e.g.::

 x1  -5 5
 x2   0 N

Note that not all parameters need to be specified, and "N" can be used to denote that a particular upper or lower limit is unbounded. The ranges are used to determine densities and plot bounds if there are samples near the boundary; if there are no samples anywhere near the boundary the ranges have no affect on plot bounds, which are chosen appropriately for the range of the samples.

Loading samples
===================

To load an MCSamples object from text files do::

	 from getdist import loadMCSamples
	 samples = loadMCSamples('/path/to/xxx', dist_settings={'ignore_rows':0.3})

Here *dist_settings* gives optional parameter settings for the analysis. *ignore_rows* is useful for MCMC chains where you want to
discard some fraction from the start of each chain as burn in (use a number >0 to discard a fixed number of sample lines rather than a fraction).
The MCSamples object can be passed to plot functions, or used to get many results. For example to plot marginalized parameter densities 
for parameter names *x1* and *x2*::

    from getdist import plots
    g = plots.getSinglePlotter()
    g.plot_2d(samples, ['x1', 'x2'])

For plotting, when you have many different chain files in the same directory, 
you can work directly with the root names. For example to compare *x* and *y* constraints
from two chains with root names *xxx* and *yyy*::

	from getdist import plots
	
	g = plots.getSinglePlotter(chain_dir='/path/to/', analysis_settings={'ignore_rows':0.3})

	g.plot_2d(['xxx','yyy], ['x', 'y'])


MCSamples objects can also be constructed directly from numpy arrays in memory, see the example in the `Plot Gallery <http://htmlpreview.github.io/?https://github.com/cmbant/getdist/blob/master/docs/plot_gallery.html>`_.

Using with CosmoMC
===================

This GetDist package is general, but is mainly developed for analysing chains from the CosmoMC sampling program.
No need to install this package separately if you have a full CosmoMC installation.
Detailed help is available for plotting Planck chains
and using CosmoMC parameter grids in the `Readme <http://cosmologist.info/cosmomc/readme_python.html>`_.
