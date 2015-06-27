GetDist GUI
===================

Run the GetDistGUI.py script to run the graphical user interface. This requires PySide to be installed, but will run on Windows, Linux and Mac.

It allows you to open a folder of chain files, then easily select, open, plot and compare, as well as viewing standard GetDist outputs and tables.

.. image:: http://cosmologist.info/cosmomc/pics/planck2015/gui_planck.png


It can open chain files in a selected directry (and also `paramgrid <http://cosmologist.info/cosmomc/readme_grids.html>`_ directories as show above).
See the `intro` for a description of chain file formats.

After opening a directory, you can select each chain root name you want to plot. It is then added to the list box below. 
The selected chains can be dragged and dropped to change the order if needed.  Then select the parameter names to plot in the checkboxes below that.

The Gui supports 1D, 2D (line and filled), 3D (select two parameters and "color by"), and triangle and rectangle plots.

Script preview
###############

Use the option on the File menu to export a plot as-is to PDF or other image file. For better quality (i.e. not formatted for the current window shape) 
and fine control (e.g. add custom legend text, etc), export the script, edit and then run it separately. 
The Script Preview tab also gives a convenient way to view the script for the current plot,
and preview exactly what it will produce when run:

.. image:: http://cosmologist.info/cosmomc/pics/planck2015/gui_script.png

You can also edit and customize the script, or open and play with existing plot scripts.

Statistics and tables
######################

The Data menu has an option to let you view the parameter statistics (.margestats) and latex tables, convergence stats, and view PCA constraints for 
selected parameters. Note that you need a working latex installation to view rendered parameter tables.


Settings
###########

The Options menu allows you to change a settings defining how limits, lines and contours are calculated, and customize plot options. 
The "Plot module config" option lets you use a different module to define the plotting functions (the default is getdist.plots).
