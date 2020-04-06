GetDist GUI
===================

Run the *getdist-gui* script to run the graphical user interface. This requires `PySide2 <https://wiki.qt.io/Qt_for_Python>`_ to be installed, but will run on Windows, Linux and Mac.

It allows you to open a folder of chain files, then easily select, open, plot and compare, as well as viewing standard GetDist outputs and tables.

.. image:: https://cdn.cosmologist.info/antony/getdist/gui_planck2018.png

It can open chain files under a selected directry structure (and also `paramgrid <https://cosmologist.info/cosmomc/readme_grids.html>`_ directories as show above).
See the `intro <https://getdist.readthedocs.io/en/latest/intro.html>`_ for a description of chain file formats.  A grid of sample chains files can be
downloaded `here <http://pla.esac.esa.int/pla/#cosmology>`_, after downloading a file just unzip and open the main directory in the GUI.

After opening a directory, you can select each chain root name you want to plot. It is then added to the list box below.
The selected chains can be dragged and dropped to change the order if needed.  Then select the parameter names to plot in the checkboxes below that,
and correspond to the names available in the first selected chain.

The Gui supports 1D, 2D (line and filled), 3D (select two parameters and "color by"), and triangle and rectangle plots.

Script preview
###############

Use the option on the File menu to export a plot as-is to PDF or other image file. For better quality (i.e. not formatted for the current window shape)
and fine control (e.g. add custom legend text, etc), export the script, edit and then run it separately.
The Script Preview tab also gives a convenient way to view the script for the current plot,
and preview exactly what it will produce when run:

.. image:: https://cdn.cosmologist.info/antony/getdist/gui_script2018.png

You can also edit and customize the script, or open and play with existing plot scripts.

Statistics and tables
######################

The Data menu has an option to let you view the parameter statistics (.margestats) and latex tables, convergence stats, and view PCA constraints for
selected parameters. Note that you need a working latex installation to view rendered parameter tables.


Settings
###########

The Options menu allows you to change a settings defining how limits, lines and contours are calculated, and customize plot options.
The "Plot module config" option lets you use a different module to define the plotting functions (the default is getdist.plots).

Installation
##############

To run the GUI you need PySide2. This is not included in default dependencies
because it is only needed for the GUI and on some systems installation from pip may not work easily.

Install PySide2 from pip using::

 pip install PySide2

You can also use `Anaconda <https://www.anaconda.com/distribution/>`_,
making a consistent new environment from conda-forge (which includes PySide2) using::

  conda create -n py37forge -c conda-forge python=3.7 scipy pandas matplotlib PySide2

(note that PySide2 is currently not included in the default Anaconda packages).

Once PySide2 is set up, (re)install getdist and you should then be able to use the getdist-gui script on your path.
On a Mac the installation will also make a GetDist GUI Mac app, which you can find using Spotlight.

If you don't want to install dependencies locally, you can also use a pre-configured `virtual environment <https://cosmologist.info/CosmoBox/>`_.
