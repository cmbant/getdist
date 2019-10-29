getdist.plots
==================================

.. currentmodule:: getdist.plots

This module is used for making plots from samples. The :func:`~.plots.get_single_plotter` and :func:`~get_subplot_plotter` functions are used to make a plotter instance,
which is then used to make and export plots.

Many plotter functions take a **roots** argument, which is either a root name for
some chain files, or an in-memory :class:`~.mcsamples.MCSamples` instance. You can also make comparison plots by giving a list of either of these.

Parameter are referenced simply by name (as specified in the .paramnames file when loading from file, or set in the :class:`~.mcsamples.MCSamples` instance).
For functions that takes lists of parameters, these can be just lists of names.
You can also use glob patterns to match specific subsets of parameters (e.g. *x\** to match all parameters with names starting with *x*).

.. autosummary::
   :toctree: _summaries
   :nosignatures:

   get_single_plotter
   get_subplot_plotter
   GetDistPlotter
   GetDistPlotSettings

.. automodule:: getdist.plots
   :members:
   :exclude-members: makeList, getSinglePlotter, getSubplotPlotter, getPlotter, RootInfo



