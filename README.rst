===================
GetDist
===================
:GetDist: MCMC sample analysis, plotting and GUI
:Author: Antony Lewis
:Homepage: https://getdist.readthedocs.io
:Source: https://github.com/cmbant/getdist
:Reference: https://arxiv.org/abs/1910.13970

.. image:: https://github.com/cmbant/getdist/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/cmbant/getdist/actions/workflows/tests.yml
.. image:: https://img.shields.io/pypi/v/GetDist.svg?style=flat
   :target: https://pypi.python.org/pypi/GetDist/
.. image:: https://readthedocs.org/projects/getdist/badge/?version=latest
   :target: https://getdist.readthedocs.io/en/latest
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/cmbant/getdist/master?filepath=docs%2Fplot_gallery.ipynb
.. image:: https://img.shields.io/badge/arXiv-1910.13970-b31b1b.svg?color=0B6523
   :target: https://arxiv.org/abs/1910.13970

Description
============

GetDist is a Python package for analysing Monte Carlo samples, including correlated samples
from Markov Chain Monte Carlo (MCMC).

* **Point and click GUI** - select chain files, view plots, marginalized constraints, LaTeX tables and more (Qt-based desktop app and Streamlit web interface)
* **Plotting library** - make custom publication-ready 1D, 2D, 3D-scatter, triangle and other plots
* **Named parameters** - simple handling of many parameters using parameter names, including LaTeX labels and prior bounds
* **Optimized Kernel Density Estimation** - automated optimal bandwidth choice for 1D and 2D densities (Botev et al. Improved Sheather-Jones method), with boundary and bias correction
* **Convergence diagnostics** - including correlation length and diagonalized Gelman-Rubin statistics
* **LaTeX tables** for marginalized 1D constraints

See the `Plot Gallery and tutorial <https://getdist.readthedocs.io/en/latest/plot_gallery.html>`_
(`run online <https://mybinder.org/v2/gh/cmbant/getdist/master?filepath=docs%2Fplot_gallery.ipynb>`_)
and `GetDist Documentation <https://getdist.readthedocs.io/en/latest/index.html>`_.


Getting Started
================

Install getdist using pip::

    $ pip install getdist

or from source files using::

    $ pip install -e /path/to/source/

You can test if things are working using the unit test by running::

    $ python -m unittest getdist.tests.getdist_test

Check the dependencies listed in the next section are installed. You can then use the getdist module from your scripts, or
use the GetDist GUI (*getdist-gui* command).

Once installed, the best way to get up to speed is probably to read through
the `Plot Gallery and tutorial <https://getdist.readthedocs.io/en/latest/plot_gallery.html>`_.

Dependencies
=============
* Python 3.8+
* matplotlib
* scipy
* PySide6 - optional, only needed for Qt-based GUI
* Streamlit - optional, only needed for web-based GUI
* Working LaTeX installation (not essential, only for some plotting/table functions)

Python distributions like Anaconda have most of what you need (except for LaTeX).

To use the Qt-based `GUI <https://getdist.readthedocs.io/en/latest/gui.html>`_ you need PySide6.
To use the Streamlit web interface, you need Streamlit.
See the `GUI docs <https://getdist.readthedocs.io/en/latest/gui.html#installation>`_ for suggestions on how to install both.

Algorithm details
==================

Details of kernel density estimation (KDE) algorithms and references are give in the GetDist notes
`arXiv:1910.13970 <https://arxiv.org/pdf/1910.13970>`_.

Samples file format
===================

GetDist can be used in scripts and interactively with standard numpy arrays
(as in the `examples <https://getdist.readthedocs.io/en/latest/plot_gallery.html>`_).
Scripts and the `GetDist GUI <https://getdist.readthedocs.io/en/latest/gui.html>`_ can also read parameter sample/chain files in plain text format
(or in the format output by the `Cobaya <https://cobaya.readthedocs.io>`__ sampling program).
In general plain text files of the form::

  xxx_1.txt
  xxx_2.txt
  ...
  xxx.paramnames
  xxx.ranges

where "xxx" is some root file name.

The .txt files are separate chain files (there can also be just one xxx.txt file). Each row of each sample .txt file is in the format

  *weight like param1 param2 param3* ...

The *weight* gives the number of samples (or importance weight) with these parameters. *like* gives -log(likelihood), and *param1, param2...* are the values of the parameters at the sample point. The first two columns can be 1 and 0 if they are not known or used.

The .paramnames file lists the names of the parameters, one per line, optionally followed by a LaTeX label. Names cannot include spaces, and if they end in "*" they are interpreted as derived (rather than MCMC) parameters, e.g.::

 x1   x_1
 y1   y_1
 x2   x_2
 xy*  x_1+y_1

The .ranges file gives hard bounds for the parameters, e.g.::

 x1  -5 5
 x2   0 N

Note that not all parameters need to be specified, and "N" can be used to denote that a particular upper or lower limit is unbounded. The ranges are used to determine densities and plot bounds if there are samples near the boundary; if there are no samples anywhere near the boundary the ranges have no affect on plot bounds, which are chosen appropriately for the range of the samples.

There can also optionally be a .properties.ini file, which can specify *burn_removed=T* to ensure no burn in is removed, or *ignore_rows=x* to ignore the first
fraction *x* of the file rows (or if *x > 1*, the specified number of rows).

Loading samples
===================

To load an MCSamples object from text files do::

     from getdist import loadMCSamples
     samples = loadMCSamples('/path/to/xxx', settings={'ignore_rows':0.3})

Here *settings* gives optional parameter settings for the analysis. *ignore_rows* is useful for MCMC chains where you want to
discard some fraction from the start of each chain as burn in (use a number >1 to discard a fixed number of sample lines rather than a fraction).
The MCSamples object can be passed to plot functions, or used to get many results. For example, to plot marginalized parameter densities
for parameter names *x1* and *x2*::

    from getdist import plots
    g = plots.get_single_plotter()
    g.plot_2d(samples, ['x1', 'x2'])

When you have many different chain files in the same directory,
plotting can work directly with the root file names. For example to compare *x* and *y* constraints
from two chains with root names *xxx* and *yyy*::

    from getdist import plots
    g = plots.get_single_plotter(chain_dir='/path/to/', analysis_settings={'ignore_rows':0.3})
    g.plot_2d(['xxx','yyy'], ['x', 'y'])


MCSamples objects can also be constructed directly from numpy arrays in memory, see the example
in the `Plot Gallery <https://getdist.readthedocs.io/en/latest/plot_gallery.html>`_.

GetDist script
===================

If you have chain files on on disk, you can also quickly calculate convergence and marginalized statistics using the *getdist* script:

    usage: getdist [-h] [--ignore_rows IGNORE_ROWS] [-V] [ini_file] [chain_root]

    GetDist sample analyser

    positional arguments:
      *ini_file*              .ini file with analysis settings (optional, if omitted uses defaults

      *chain_root*            Root name of chain to analyse (e.g. chains/test), required unless file_root specified in ini_file

    optional arguments:
      -h, --help            show this help message and exit
      --ignore_rows IGNORE_ROWS
                            set initial fraction of chains to cut as burn in
                            (fraction of total rows, or >1 number of rows);
                            overrides any value in ini_file if set
      --make_param_file MAKE_PARAM_FILE
                        Produce a sample distparams.ini file that you can edit
                        and use when running GetDist
      -V, --version         show program's version number and exit

where *ini_file* is optionally a .ini file listing *key=value* parameter option values, and chain_root is the root file name of the chains.
For example::

   getdist distparams.ini chains/test_chain

This produces a set of files containing parameter means and limits (.margestats), N-D likelihood contour boundaries and best-fit sample (.likestats),
convergence diagnostics (.converge), parameter covariance and correlation (.covmat and .corr), and optionally various simple plotting scripts.
If no *ini_file* is given, default settings are used. The *ignore_rows* option allows some of the start of each chain file to be removed as burn in.

To customize settings you can run::

   getdist --make_param_file distparams.ini

to produce the setting file distparams.ini, edit it, then run with your custom settings.

GetDist GUI
===================

GetDist provides two graphical user interfaces:

1. **Qt-based Desktop App**: Run *getdist-gui* to use the traditional desktop interface. This requires PySide6 to be installed.

2. **Streamlit Web Interface**: Run *getdist-streamlit* to use the browser-based interface. This requires Streamlit to be installed.

Both interfaces allow you to open a folder of chain files, then easily select, open, plot and compare, as well as viewing standard GetDist outputs and tables.

You can also try the Streamlit interface online at `<https://getdist-gui-test.streamlit.app/>`_ (with fixed example chains).

See the `GUI Documentation <https://getdist.readthedocs.io/en/latest/gui.html>`_ for more details on both interfaces.


Using with CosmoMC and Cobaya
=============================

This GetDist package is general, but is mainly developed for analysing chains from the `CosmoMC <https://cosmologist.info/cosmomc>`_
and `Cobaya <https://cobaya.readthedocs.io/>`_ sampling programs.
No need to install this package separately if you have a full CosmoMC installation; the Cobaya installation will also install GetDist as a dependency.
Detailed help is available for plotting Planck chains
and using CosmoMC parameter grids in the `Readme <https://cosmologist.info/cosmomc/readme_python.html>`_.

Citation
===================
You can refer to the notes::

     @article{Lewis:2019xzd,
      author         = "Lewis, Antony",
      title          = "{GetDist: a Python package for analysing Monte Carlo
                        samples}",
      year           = "2019",
      eprint         = "1910.13970",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.IM",
      SLACcitation   = "%%CITATION = ARXIV:1910.13970;%%",
      url            = "https://getdist.readthedocs.io"
     }


and references therein as appropriate.

LLM Integration
===================
For AI assistants and LLM agents working with GetDist, a single-file context document is available at `GetDist LLM Context <https://help.cosmologist.info/api/context/getdist>`_. This document provides a comprehensive overview of GetDist's functionality, common usage patterns, and best practices in a format optimized for LLM context windows.

===================

.. image:: https://raw.githubusercontent.com/CobayaSampler/cobaya/master/img/Sussex_white.svg
   :alt: University of Sussex
   :target: https://www.sussex.ac.uk/astronomy/
   :height: 200px
   :width: 200px

.. image:: https://raw.githubusercontent.com/CobayaSampler/cobaya/master/img/ERC_white.svg
   :alt: European Research Council
   :target: https://erc.europa.eu/
   :height: 200px
   :width: 200px

.. image:: https://cdn.cosmologist.info/antony/STFC_white.svg
   :alt: STFC
   :target: https://stfc.ukri.org/
   :height: 200px
   :width: 200px
