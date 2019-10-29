from __future__ import absolute_import
from __future__ import print_function
import os
import copy
import matplotlib
import sys
import six
import warnings
import logging

matplotlib.use('Agg', warn=False)
import matplotlib.patches
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.axis
import matplotlib.lines
from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.font_manager import font_scalings
import numpy as np
from paramgrid import gridconfig, batchjob
import getdist
from getdist import MCSamples, loadMCSamples, ParamNames, ParamInfo, IniFile
from getdist.chains import chainFiles
from getdist.paramnames import escapeLatex, makeList, mergeRenames
from getdist.densities import Density2D
from getdist.gaussian_mixtures import MixtureND
from getdist.matplotlib_ext import BoundedMaxNLocator, SciFuncFormatter
from getdist._base import _BaseObject

"""Plotting scripts for GetDist outputs"""


class GetDistPlotError(Exception):
    """
    An exception that is raised when there is an error plotting
    """
    pass


class GetDistPlotSettings(_BaseObject):
    """
    Settings class (colors, sizes, font, styles etc.)

    :ivar alpha_factor_contour_lines: alpha factor for adding contour lines between filled contours
    :ivar alpha_filled_add: alpha for adding filled contours to a plot
    :ivar axes_fontsize: Size for axis font at reference axis size
    :ivar axes_labelsize: Size for axis label font at reference axis size
    :ivar axis_marker_color: The color for a marker
    :ivar axis_marker_ls: The line style for a marker
    :ivar axis_marker_lw: The line width for a marker
    :ivar axis_tick_powerlimits: exponents at which to use scientific notation for axis tick labels
    :ivar axis_tick_max_labels: maximum number of tick labels per axis
    :ivar axis_tick_step_groups: steps to try for axis ticks, in grouped in order of preference
    :ivar axis_tick_x_rotation: The rotation for the x tick label in degrees
    :ivar axis_tick_y_rotation: The rotation for the y tick label in degrees
    :ivar colorbar_axes_fontsize: size for tick labels on colorbar (None for default to match axes font size)
    :ivar colorbar_label_pad: padding for the colorbar label
    :ivar colorbar_label_rotation: angle to rotate colorbar label (set to zero if -90 default gives layout problem)
    :ivar colorbar_tick_rotation: angle to rotate colorbar tick labels
    :ivar colormap: a `Matplotlib color map <https://www.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_ for shading
    :ivar colormap_scatter: a Matplotlib `color map <https://www.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_
                            for 3D scatter plots
    :ivar constrained_layout: use matplotlib's constrained-layout to fit plots within the figure and avoid overlaps
    :ivar fig_width_inch: The width of the figure in inches
    :ivar figure_legend_frame: draw box around figure legend
    :ivar figure_legend_loc: The location for the figure legend
    :ivar figure_legend_ncol: number of columns for figure legend (set to zero to use defaults)
    :ivar fontsize: font size for text (and ultimate fallback when others not set)
    :ivar legend_colored_text: use colored text for legend labels rather than separate color blocks
    :ivar legend_fontsize: The font size for the legend (defaults to fontsize)
    :ivar legend_frac_subplot_margin: fraction of subplot size to use for spacing figure legend above plots
    :ivar legend_frame: draw box around legend
    :ivar legend_loc: The location for the legend
    :ivar legend_rect_border: whether to have black border around solid color boxes in legends
    :ivar line_dash_styles: dict mapping line styles to detailed dash styles,
                            default:  {'--': (3, 2), '-.': (4, 1, 1, 1)}
    :ivar line_labels: True if you want to automatically add legends when adding more than one line to subplots
    :ivar line_styles: list of default line styles/colors (['-k', '-r', '--C0', ...]) or name of a standard colormap
                       (e.g. tab10), or a list of tuples of line styles and colors for each line
    :ivar linewidth: relative linewidth (at reference size)
    :ivar linewidth_contour: linewidth for lines in filled contours
    :ivar linewidth_meanlikes: linewidth for mean likelihood lines
    :ivar no_triangle_axis_labels: whether subplots in triangle plots should show axis labels if not at the edge
    :ivar norm_1d_density: whether to normolize 1D densities (otherwise normalized to unit peak value)
    :ivar norm_prob_label: label for the y axis in normalized 1D density plots
    :ivar num_plot_contours: number of contours to plot in 2D plots (up to number of contours in analysis settings)
    :ivar num_shades: number of distinct colors to use for shading shaded 2D plots
    :ivar param_names_for_labels: file name of .paramnames file to use for overriding parameter labels for plotting
    :ivar plot_args: dict, or list of dicts, giving settings like color, ls, alpha, etc. to apply for a plot or each
                     line added
    :ivar plot_meanlikes: include mean likelihood lines in 1D plots
    :ivar prob_label: label for the y axis in unnormalized 1D density plots
    :ivar prob_y_ticks: show ticks on y axis for 1D density plots
    :ivar progress: write out some status
    :ivar scaling: True to scale down fonts and lines for smaller subplots; False to use fixed sizes.
    :ivar scaling_max_axis_size: font sizes will only be scaled for subplot widths (in inches) smaller than this.
    :ivar scaling_factor: factor by which to multiply the different of the axis size to the reference size when
                          scaling font sizes
    :ivar scaling_reference_size: axis width (in inches) at which font sizes are specified.
    :ivar scatter_size: size of points in "3D" scatter plots
    :ivar shade_level_scale: shading contour colors are put at [0:1:spacing]**shade_level_scale
    :ivar shade_meanlikes: 2D shading uses mean likelihoods rather than marginalized density
    :ivar solid_colors: List of default colors for filled 2D plots or the name of a colormap (e.g. tab10).  If a list,
                        each element is either a color, or a tuple of values for different contour levels.
    :ivar solid_contour_palefactor: factor by which to make 2D outer filled contours paler when only specifying
                                    one contour color
    :ivar subplot_size_ratio: ratio of width and height of subplots
    :ivar tight_layout: use tight_layout to layout, avoid overlaps and remove white space; if it doesn't work
                        try constrained_layout. If true it is applied when calling :func:`~GetDistPlotter.finish_plot`
                        (which is called automatically by plots_xd(), triangle_plot and rectangle_plot).
    :ivar title_limit: show parameter limits over 1D plots, 1 for first limit (68% default), 2 second, etc.
    :ivar title_limit_labels: whether or not to include parameter label when adding limits above 1D plots
    :ivar title_limit_fontsize: font size to use for limits in plot titles (defaults to axes_labelsize)
    """

    _deprecated = {'lab_fontsize': 'axes_labelsize',
                   'colorbar_rotation': 'colorbar_tick_rotation',
                   'font_size ': 'fontsize',
                   'legend_frac_subplot_line': None,
                   'legend_position_config': None,
                   'lineM': 'line_styles',
                   'lw1': 'linewidth',
                   'lw_contour': 'linewidth_contour',
                   'lw_likes': 'linewidth_meanlikes',
                   'thin_long_subplot_ticks': None,
                   'tick_prune': None,
                   'tight_gap_fraction': None,
                   'x_label_rotation': 'axis_tick_x_rotation'
                   }

    def __init__(self, subplot_size_inch=2, fig_width_inch=None):
        """
        If fig_width_inch set, fixed setting for fixed total figure size in inches.
        Otherwise use subplot_size_inch to determine default font sizes etc.,
        and figure will then be as wide as necessary to show all subplots at specified size.

        :param subplot_size_inch: Determines the size of subplots, and hence default font sizes
        :param fig_width_inch: The width of the figure in inches, If set, forces fixed total size.
        """
        self.scaling = True
        self.scaling_reference_size = 3.5  # reference subplot size for font sizes etc.
        self.scaling_max_axis_size = self.scaling_reference_size
        self.scaling_factor = 2

        self.plot_meanlikes = False
        self.prob_label = None
        # self.prob_label = 'Probability'
        self.norm_prob_label = 'P'
        self.prob_y_ticks = False
        self.norm_1d_density = False
        # : line styles/colors
        self.line_styles = ['-k', '-r', '-b', '-g', '-m', '-c', '-y', '--k', '--r', '--b', '--g', '--m']

        self.plot_args = None
        self.line_dash_styles = {'--': (3, 2), '-.': (4, 1, 1, 1)}
        self.line_labels = True
        self.num_shades = 80
        self.shade_level_scale = 1.8  # contour levels at [0:1:spacing]**shade_level_scale

        self.progress = False

        self.fig_width_inch = fig_width_inch  # if you want to force specific fixed width
        self.tight_layout = True
        self.constrained_layout = False
        self.no_triangle_axis_labels = True

        # see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
        self.colormap = "Blues"
        self.colormap_scatter = "jet"
        self.colorbar_tick_rotation = None
        self.colorbar_label_pad = 0
        self.colorbar_label_rotation = -90
        self.colorbar_axes_fontsize = 11

        self.subplot_size_inch = subplot_size_inch
        self.subplot_size_ratio = None

        self.param_names_for_labels = None

        self.legend_colored_text = False
        self.legend_loc = 'best'
        self.legend_frac_subplot_margin = 0.05
        self.legend_fontsize = 12
        self.legend_frame = True
        self.legend_rect_border = False

        self.figure_legend_loc = 'upper center'
        self.figure_legend_frame = True
        self.figure_legend_ncol = 0

        self.linewidth = 1
        self.linewidth_contour = 0.6
        self.linewidth_meanlikes = 0.5

        self.num_plot_contours = 2
        self.solid_contour_palefactor = 0.6
        self.solid_colors = ['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm', 'r']
        self.alpha_filled_add = 0.85
        self.alpha_factor_contour_lines = 0.5
        self.shade_meanlikes = False

        self.axes_fontsize = 11
        self.axes_labelsize = 14

        self.axis_marker_color = 'gray'
        self.axis_marker_ls = '--'
        self.axis_marker_lw = 0.5

        self.axis_tick_powerlimits = (-4, 5)
        self.axis_tick_max_labels = 7
        self.axis_tick_step_groups = [[1, 2, 5, 10], [2.5, 3, 4, 6, 8], [1.5, 7, 9]]
        self.axis_tick_x_rotation = 0
        self.axis_tick_y_rotation = 0

        self.scatter_size = 3

        self.fontsize = 12

        self.title_limit = 0
        self.title_limit_labels = True
        self.title_limit_fontsize = None
        self._fail_on_not_exist = True

    def _numerical_fontsize(self, size):
        size = size or self.fontsize or 11
        if isinstance(size, six.string_types):
            scale = font_scalings.get(size)
            return self.fontsize * (scale or 1)
        return size or self.fontsize

    def scaled_fontsize(self, ax_size, var, default=None):
        var = self._numerical_fontsize(var or default)
        if not self.scaling or self.scaling_max_axis_size is not None and not self.scaling_max_axis_size:
            return var
        if self.scaling_max_axis_size is None or ax_size < (self.scaling_max_axis_size or self.scaling_reference_size):
            return max(5, var + self.scaling_factor * (ax_size - self.scaling_reference_size))
        else:
            return var + 2 * (self.scaling_max_axis_size - self.scaling_reference_size)

    def scaled_linewidth(self, ax_size, linewidth):
        linewidth = linewidth or self.linewidth
        if not self.scaling:
            return linewidth
        return max(0.6, linewidth * ax_size / self.scaling_reference_size)

    def set_with_subplot_size(self, size_inch=3.5, size_mm=None, size_ratio=None):
        """
        Sets the subplot's size, either in inches or in millimeters.
        If both are set, uses millimeters.

        :param size_inch: The size to set in inches; is ignored if size_mm is set.
        :param size_mm: None if not used, otherwise the size in millimeters we want to set for the subplot.
        :param size_ratio: ratio of height to width of subplots
        """
        if size_mm:
            size_inch = size_mm * 0.0393700787
        self.subplot_size_inch = size_inch
        self.subplot_size_ratio = size_ratio

    def rc_sizes(self, axes_fontsize=None, lab_fontsize=None, legend_fontsize=None):
        """
        Sets the font sizes by default from matplotlib.rcParams defaults

        :param axes_fontsize: The font size for the plot axes tick labels (default: xtick.labelsize).
        :param lab_fontsize: The font size for the plot's axis labels (default: axes.labelsize)
        :param legend_fontsize: The font size for the plot's legend (default: legend.fontsize)
        """
        self.fontsize = self._numerical_fontsize(rcParams['font.size'])
        self.legend_fontsize = legend_fontsize or self._numerical_fontsize(rcParams['legend.fontsize'])
        self.axes_labelsize = lab_fontsize or self._numerical_fontsize(rcParams['axes.labelsize'])
        self.axes_fontsize = axes_fontsize or self._numerical_fontsize(rcParams['xtick.labelsize'])

    def __str__(self):
        sets = self.__dict__.copy()
        for key, value in list(sets.items()):
            if key.startswith('_'):
                sets.pop(key)
        return str(sets)


default_settings = GetDistPlotSettings()
defaultSettings = default_settings


def get_plotter(style=None, **kwargs):
    """
    Creates a new plotter and returns it

    :param style: name of a plotter style (associated with custom plotter class/settings), otherwise uses active
    :param kwargs: arguments for the style's :class:`~getdist.plots.GetDistPlotter`
    :return: The :class:`GetDistPlotter` instance
    """
    return _style_manager.active_class(style)(**kwargs)


def get_single_plotter(ratio=None, width_inch=None, scaling=None, rc_sizes=False, style=None, **kwargs):
    """
    Get a :class:`~.plots.GetDistPlotter` for making a single plot of fixed width.

    For a half-column plot for a paper use width_inch=3.464.

    Use this or :func:`~get_subplot_plotter` to make a :class:`~.plots.GetDistPlotter` instance for making plots.
    This function will use the active style by default, which will determine defaults for the various optional
    parameters (see :func:`~set_active_style`).

    :param ratio: The ratio between height and width.
    :param width_inch:  The width of the plot in inches
    :param scaling: whether to scale down fonts and line widths for small subplot axis sizes
                    (relative to reference sizes, 3.5 inch)
    :param rc_sizes: set default font sizes from matplotlib's current rcParams if no explicit settings passed in kwargs
    :param style: name of a plotter style (associated with custom plotter class/settings), otherwise uses active
    :param kwargs: arguments for :class:`GetDistPlotter`
    :return: The :class:`~.plots.GetDistPlotter` instance
    """
    return _style_manager.active_class(style).get_single_plotter(ratio=ratio, width_inch=width_inch, scaling=scaling,
                                                                 rc_sizes=rc_sizes, **kwargs)


def get_subplot_plotter(subplot_size=None, width_inch=None, scaling=None, rc_sizes=False,
                        subplot_size_ratio=None, style=None, **kwargs):
    """
    Get a :class:`~.plots.GetDistPlotter` for making an array of subplots.

    If width_inch is None, just makes plot as big as needed for given subplot_size, otherwise fixes total width
    and sets default font sizes etc. from matplotlib's default rcParams.

    Use this or :func:`~get_single_plotter` to make a :class:`~.plots.GetDistPlotter` instance for making plots.
    This function will use the active style by default, which will determine defaults for the various optional
    parameters (see :func:`~set_active_style`).


    :param subplot_size: The size of each subplot in inches
    :param width_inch: Optional total width in inches
    :param scaling: whether to scale down fonts and line widths for small sizes (relative to reference sizes, 3.5 inch)
    :param rc_sizes: set default font sizes from matplotlib's current rcParams if no explicit settings passed in kwargs
    :param subplot_size_ratio: ratio of height to width for subplots
    :param style: name of a plotter style (associated with custom plotter class/settings), otherwise uses active
    :param kwargs: arguments for :class:`GetDistPlotter`
    :return: The :class:`GetDistPlotter` instance
    """
    return _style_manager.active_class(style).get_subplot_plotter(subplot_size=subplot_size, width_inch=width_inch,
                                                                  scaling=scaling, rc_sizes=rc_sizes,
                                                                  subplot_size_ratio=subplot_size_ratio, **kwargs)


# Aliases for backwards compatibility
getPlotter = get_plotter
getSubplotPlotter = get_subplot_plotter
getSinglePlotter = get_single_plotter


class RootInfo(object):
    """
    Class to hold information about a set of samples loaded from file
    """
    __slots__ = ['root', 'batch', 'path']

    def __init__(self, root, path, batch=None):
        """
        :param root: The root file to use.
        :param path: The path the root file is in.
        :param batch: optional batch object if loaded from a grid of results
        """
        self.root = root
        self.batch = batch
        self.path = path


class MCSampleAnalysis(_BaseObject):
    """
    A class that loads and analyses samples, mapping root names to :class:`~.mcsamples.MCSamples` objects with caching.
    Typically accessed as the instance stored in plotter.sample_analyser, for example to
    get an :class:`~.mcsamples.MCSamples` instance from a root name being used by a plotter,
    use plotter.sample_analyser.samples_for_root(name).
    """

    def __init__(self, chain_locations, settings=None):
        """
        :param chain_locations: either a directory or the path of a grid of runs;
               it can also be a list of such, which is searched in order
        :param settings: Either an :class:`~.inifile.IniFile` instance,
               the name of an .ini file, or a dict holding sample analysis settings.
        """
        self.chain_dirs = []
        self.chain_locations = []
        self.ini = None
        self.chain_settings_have_priority = True
        if chain_locations is not None:
            if isinstance(chain_locations, six.string_types):
                chain_locations = [chain_locations]
            for chain_dir in chain_locations:
                self.add_chain_dir(chain_dir)
        self.reset(settings)

    def add_chain_dir(self, chain_dir):
        """
        Adds a new chain directory or grid path for searching for samples

        :param chain_dir: The directory to add
        """
        if chain_dir in self.chain_locations:
            return
        self.chain_locations.append(chain_dir)
        is_batch = isinstance(chain_dir, batchjob.batchJob)
        if is_batch or gridconfig.pathIsGrid(chain_dir):
            if is_batch:
                batch = chain_dir
            else:
                batch = batchjob.readobject(chain_dir)
            self.chain_dirs.append(batch)
            # this gets things like specific parameter limits etc. specific to the grid
            # yuk, this should only be for old Planck grids. New ones don't need getdist_common
            # should instead set custom settings in the grid setting file
            if os.path.exists(batch.commonPath + 'getdist_common.ini'):
                batchini = IniFile(batch.commonPath + 'getdist_common.ini')
                if self.ini:
                    self.ini.params.update(batchini.params)
                else:
                    self.ini = batchini
        else:
            self.chain_dirs.append(chain_dir)

    def reset(self, settings=None, chain_settings_have_priority=True):
        """
        Resets the caches, starting afresh optionally with new analysis settings

        :param settings: Either an :class:`~.inifile.IniFile` instance,
               the name of an .ini file, or a dict holding sample analysis settings.
        :param chain_settings_have_priority: whether to prioritize settings saved with the chain
        """
        self.analysis_settings = {}
        if isinstance(settings, IniFile):
            ini = settings
        elif isinstance(settings, dict):
            ini = IniFile(getdist.default_getdist_settings)
            ini.params.update(settings)
        else:
            ini = IniFile(settings or getdist.default_getdist_settings)
        if self.ini:
            self.ini.params.update(ini.params)
        else:
            self.ini = ini
        self.mcsamples = {}
        # Dicts. 1st key is root; 2nd key is param
        self.densities_1D = dict()
        self.densities_2D = dict()
        self.single_samples = dict()
        self.chain_settings_have_priority = chain_settings_have_priority

    def samples_for_root(self, root, file_root=None, cache=True, settings=None):
        """
        Gets :class:`~.mcsamples.MCSamples` from root name
        (or just return root if it is already an MCSamples instance).

        :param root: The root name (without path, e.g. my_chains)
        :param file_root: optional full root path, by default searches in self.chain_dirs
        :param cache: if True, return cached object if already loaded
        :param settings: optional dictionary of settings to use
        :return: :class:`~.mcsamples.MCSamples` for the given root name
        """
        if isinstance(root, MCSamples):
            return root
        if isinstance(root, MixtureND):
            raise GetDistPlotError('MixtureND is a distribution not a set of samples')
        elif not isinstance(root, six.string_types):
            raise GetDistPlotError('Root names must be strings (or MCSamples instances)')
        if os.path.isabs(root):
            # deal with just-folder prefix
            if root.endswith((os.sep, "/")):
                root = os.path.basename(root[:-1]) + os.sep
            else:
                root = os.path.basename(root)
        if root in self.mcsamples and cache:
            return self.mcsamples[root]
        job_item = None
        if self.chain_settings_have_priority:
            dist_settings = settings or {}
        else:
            dist_settings = {}
        if not file_root:
            from getdist.cobaya_interface import _separator_files
            for chain_dir in self.chain_dirs:
                if hasattr(chain_dir, "resolveRoot"):
                    job_item = chain_dir.resolveRoot(root)
                    if job_item:
                        file_root = job_item.chainRoot
                        if hasattr(chain_dir, 'getdist_options'):
                            dist_settings.update(chain_dir.getdist_options)
                        dist_settings.update(job_item.dist_settings)
                        break
                else:
                    name = os.path.join(chain_dir, root)
                    if any([chainFiles(name, separator=sep)
                            for sep in ['_', _separator_files]]):
                        file_root = name
                        break
        if not file_root:
            raise GetDistPlotError('chain not found: ' + root)
        if not self.chain_settings_have_priority:
            dist_settings.update(self.ini.params)
            if settings:
                dist_settings.update(settings)
        self.mcsamples[root] = loadMCSamples(file_root, self.ini, job_item, settings=dist_settings)
        return self.mcsamples[root]

    def add_roots(self, roots):
        """
        A wrapper for add_root that adds multiple file roots

        :param roots: An iterable containing filenames or :class:`RootInfo` objects to add
        """
        for root in roots:
            self.add_root(root)

    def add_root(self, file_root):
        """
        Add a root file for some new samples

        :param file_root: Either a file root name including path or a :class:`RootInfo` instance
        :return: :class:`~.mcsamples.MCSamples` instance for given root file.
        """
        if isinstance(file_root, RootInfo):
            if file_root.batch:
                return self.samples_for_root(file_root.root)
            else:
                return self.samples_for_root(file_root.root, os.path.join(file_root.path, file_root.root))
        else:
            return self.samples_for_root(os.path.basename(file_root), file_root)

    def remove_root(self, file_root):
        """
        Remove a given root file (does not delete it)

        :param file_root: The file root to remove
        """
        root = os.path.basename(file_root)
        self.mcsamples.pop(root, None)
        self.single_samples.pop(root, None)
        self.densities_1D.pop(root, None)
        self.densities_2D.pop(root, None)

    def get_density(self, root, param, likes=False):
        """
        Get :class:`~.densities.Density1D` for given root name and parameter

        :param root:  The root name of the samples to use
        :param param: name of the parameter
        :param likes: whether to include mean likelihood in density.likes
        :return:  :class:`~.densities.Density1D` instance with 1D marginalized density
        """
        rootdata = self.densities_1D.get(root)
        if rootdata is None:
            rootdata = {}
            self.densities_1D[root] = rootdata
        if isinstance(param, ParamInfo):
            name = param.name
        else:  #
            name = param
        samples = self.samples_for_root(root)
        key = (name, likes)
        rootdata.pop((name, not likes), None)
        density = rootdata.get(key)
        if density is None:
            density = samples.get1DDensityGridData(name, meanlikes=likes)
            if density is None:
                return None
            rootdata[key] = density
        return density

    def get_density_grid(self, root, param1, param2, conts=2, likes=False):
        """
        Get 2D marginalized density for given root name and parameters

        :param root: The root name for samples to use.
        :param param1: x parameter
        :param param2: y parameter
        :param conts: number of contour levels (up to maximum calculated using contours in analysis settings)
        :param likes: whether to include mean likelihoods
        :return: :class:`~.densities.Density2D` instance with marginalized density
        """
        rootdata = self.densities_2D.get(root)
        if not rootdata:
            rootdata = {}
            self.densities_2D[root] = rootdata
        key = (param1.name, param2.name, likes, conts)
        density = rootdata.get(key)
        if not density:
            samples = self.samples_for_root(root)
            density = samples.get2DDensityGridData(param1.name, param2.name, num_plot_contours=conts, meanlikes=likes)
            if density is None:
                return None
            rootdata[key] = density
        return density

    def load_single_samples(self, root):
        """
        Gets a set of unit weight samples for given root name, e.g. for making sample scatter plot

        :param root: The root name to use.
        :return: array of unit weight samples
        """
        if root not in self.single_samples:
            self.single_samples[root] = self.samples_for_root(root).makeSingleSamples()
        return self.single_samples[root]

    def params_for_root(self, root, label_params=None):
        """
        Returns a :class:`~.paramnames.ParamNames` with names and labels for parameters used by samples with a
        given root name.

        :param root: The root name of the samples to use.
        :param label_params: optional name of .paramnames file containing labels to use for plots, overriding default
        :return: :class:`~.paramnames.ParamNames` instance
        """
        if hasattr(root, 'paramNames'):
            names = root.paramNames
        else:
            samples = self.samples_for_root(root)
            names = samples.getParamNames()
        if label_params is not None:
            names.setLabelsAndDerivedFromParamNames(os.path.join(batchjob.getCodeRootPath(), label_params))
        return names

    def bounds_for_root(self, root):
        """
        Returns an object with get_upper/getUpper and get_lower/getLower to get hard prior bounds for given root name

        :param root: The root name to use.
        :return: object with get_upper() or getUpper() and get_lower() or getLower() functions
        """
        if hasattr(root, 'get_upper') or hasattr(root, 'getUpper'):
            return root
        else:
            return self.samples_for_root(root)  # defines getUpper and getLower, all that's needed


class GetDistPlotter(_BaseObject):
    """
    Main class for making plots from one or more sets of samples.

    :ivar settings: a :class:`GetDistPlotSettings` instance with settings
    :ivar subplots: a 2D array of :class:`~matplotlib:matplotlib.axes.Axes` for subplots
    :ivar sample_analyser: a :class:`MCSampleAnalysis` instance for getting :class:`~.mcsamples.MCSamples`
         and derived data from a given root name tag (e.g. sample_analyser.samples_for_root('rootname'))
    """

    def __init__(self, chain_dir=None, settings=None, analysis_settings=None, auto_close=False):
        """

        :param chain_dir: Set this to a directory or grid root to search for chains
                          (can also be a list of such, searched in order)
        :param analysis_settings: The settings to be used by :class:`MCSampleAnalysis` when analysing samples
        :param auto_close: whether to automatically close the figure whenever a new plot made or this instance released
        """

        self.chain_dir = chain_dir
        if settings is None:
            self.set_default_settings()
        else:
            self.settings = settings
        self.sample_analyser = MCSampleAnalysis(chain_dir or getdist.default_grid_root, analysis_settings)
        self.auto_close = auto_close
        self.fig = None
        self.new_plot()

    def set_default_settings(self):
        self.settings = copy.deepcopy(default_settings)

    _style_rc = {}

    @classmethod
    def get_single_plotter(cls, scaling=None, rc_sizes=False, **kwargs):
        ratio = kwargs.pop("ratio", None) or 3 / 4.
        width_inch = kwargs.pop("width_inch", None) or 6
        plotter = cls(**kwargs)
        plotter.settings.set_with_subplot_size(width_inch, size_ratio=ratio)
        if scaling is not None:
            plotter.settings.scaling = scaling
        plotter.settings.fig_width_inch = width_inch
        if not kwargs.get('settings') and rc_sizes:
            plotter.settings.rc_sizes()
        plotter.make_figure(1)
        return plotter

    @classmethod
    def get_subplot_plotter(cls, subplot_size=None, width_inch=None, scaling=True, rc_sizes=False,
                            subplot_size_ratio=None, **kwargs):
        plotter = cls(**kwargs)
        plotter.settings.set_with_subplot_size(subplot_size or 2, size_ratio=subplot_size_ratio)
        if scaling is not None:
            plotter.settings.scaling = scaling
        if width_inch:
            plotter.settings.fig_width_inch = width_inch
            if not kwargs.get('settings') and rc_sizes:
                plotter.settings.rc_sizes()
        return plotter

    def __del__(self):
        if self.auto_close and self.fig:
            plt.close(self.fig)

    def new_plot(self, close_existing=None):
        """
        Resets the given plotter to make a new empty plot.

        :param close_existing: True to close any current figure
        """
        if close_existing is None:
            close_existing = self.auto_close
        self.extra_artists = []
        self.contours_added = []
        self.lines_added = dict()
        self.param_name_sets = dict()
        self.param_bounds_sets = dict()
        if close_existing and self.fig:
            plt.close(self.fig)
        self.fig = None
        self.subplots = None
        self.plot_col = 0
        self._last_ax = None

    def show_all_settings(self):
        """
        Prints settings and library versions
        """
        print('Python version:', sys.version)
        print('\nMatplotlib version:', matplotlib.__version__)
        print('\nGetDist Plot Settings:')
        print('GetDist version:', getdist.__version__)
        sets = self.settings.__dict__
        for key, value in list(sets.items()):
            print(key, ':', value)
        print('\nRC params:')
        for key, value in list(matplotlib.rcParams.items()):
            print(key, ':', value)

    def _get_plot_args(self, plotno, **kwargs):
        """
        Get plot arguments for the given plot line number

        :param plotno: The index of the line added to a plot
        :param kwargs: optional settings to override in the current ones
        :return: The updated dict of arguments.
        """
        if isinstance(self.settings.plot_args, dict):
            args = self.settings.plot_args
        elif isinstance(self.settings.plot_args, (list, tuple)):
            if len(self.settings.plot_args) > plotno:
                args = self.settings.plot_args[plotno]
                if args is None:
                    args = dict()
            else:
                args = {}
        elif not self.settings.plot_args:
            args = dict()
        else:
            raise GetDistPlotError(
                'plot_args must be list of dictionaries or dictionary: %s' % self.settings.plot_args)
        args.update(kwargs)
        return args

    def _get_dashes_for_ls(self, ls):
        """
        Gets the dash style for the given line style.

        :param ls: The line style
        :return: The dash style.
        """
        return self.settings.line_dash_styles.get(ls)

    def _get_default_ls(self, plotno=0):
        """
        Get default line style, taken from settings.line_styles

        :param plotno: The number of the line added to the plot to get the style of.
        :return: Tuple of line style and color for default line style (e.g. ('-', 'r')).
        """
        try:
            res = self._get_color_at_index(self.settings.line_styles, plotno)
            if matplotlib.colors.is_color_like(res):
                return '-', res
            if isinstance(res, six.string_types):
                i = 0
                while i < len(res) and res[i] in ['-', '.', ':']:
                    i += 1
                return res[:i], res[i:]
            else:
                # assume tuple of line style and color
                return res[0], res[1]
        except IndexError:
            print('Error adding line ' + str(plotno) + ': Add more default line stype entries to settings.line_styles')
            raise

    def _get_line_styles(self, plotno, **kwargs):
        """
        Gets the styles of the line for the given line added to a plot

        :param plotno: The number of the line added to the plot.
        :param kwargs: Params for :func:`~GetDistPlotter._get_plot_args`.
        :return: dict with ls, dashes, lw and color set appropriately
        """
        args = self._get_plot_args(plotno, **kwargs)
        if 'ls' not in args:
            args['ls'] = self._get_default_ls(plotno)[0]
        if 'dashes' not in args:
            dashes = self._get_dashes_for_ls(args['ls'])
            if dashes is not None:
                args['dashes'] = dashes
        if 'color' not in args:
            args['color'] = self._get_default_ls(plotno)[1]
        if 'lw' not in args:
            args['lw'] = self._scaled_linewidth(self.settings.linewidth)
        return args

    def _get_color(self, plotno, **kwargs):
        """
        Get the color for the given line number

        :param plotno: line number added to plot
        :param kwargs: arguments for :func:`~GetDistPlotter._get_line_styles`
        :return: The color.
        """
        return self._get_line_styles(plotno, **kwargs)['color']

    def _get_color_at_index(self, colors, i=None):
        """
         Get color at index

        :param colors: colormap name, a colormap or array of colors
        :param i: index, or None to return the color array
        :return: color or array of colors
        """
        if isinstance(colors, six.string_types):
            colormap = getattr(cm, colors, None)
            if colormap is None:
                raise GetDistPlotError('Unknown matplotlib colormap %s' % colors)
        else:
            colormap = colors
        colors = getattr(colormap, 'colors', None) or colormap
        if i is None:
            return colors
        if i >= len(colors):
            raise IndexError('Color index out of range %s' % i)
        return colors[i]

    def _get_linestyle(self, plotno, **kwargs):
        """
        Get line style for given plot line number.

        :param plotno: line number added to plot
        :param kwargs: arguments for :func:`~GetDistPlotter._get_line_styles`
        :return: The line style for the given plot line.
        """
        return self._get_line_styles(plotno, **kwargs)['ls']

    def _get_alpha_2d(self, plotno, **kwargs):
        """
        Get the alpha for the given 2D contour added to plot

        :param plotno:  The index of contours added to the plot
        :param kwargs: arguments for :func:`~GetDistPlotter._get_line_styles`,
            These may also include: filled
        :return: The alpha for the given plot contours
        """
        args = self._get_plot_args(plotno, **kwargs)
        if kwargs.get('filled') and plotno > 0:
            default = self.settings.alpha_filled_add
        else:
            default = 1
        return args.get('alpha', default)

    def param_names_for_root(self, root):
        """
        Get the parameter names and labels :class:`~.paramnames.ParamNames` instance for the given root name

        :param root: The root name of the samples.
        :return: :class:`~.paramnames.ParamNames` instance
        """
        if root not in self.param_name_sets:
            self.param_name_sets[root] = \
                self.sample_analyser.params_for_root(root, label_params=self.settings.param_names_for_labels)
        return self.param_name_sets[root]

    def param_bounds_for_root(self, root):
        """
        Get any hard prior bounds for the parameters with root file name

        :param root: The root name to be used
        :return: object with get_upper() or getUpper() and get_lower() or getLower() bounds functions
        """
        if root not in self.param_bounds_sets:
            self.param_bounds_sets[root] = self.sample_analyser.bounds_for_root(root)
        return self.param_bounds_sets[root]

    def _check_param_ranges(self, root, name, xmin, xmax):
        """
        Checks The upper and lower bounds are not outside hard priors

        :param root: The root file to use.
        :param name: The param name to check.
        :param xmin: The lower bound
        :param xmax: The upper bound
        :return: The bounds (highest lower limit, and lowest upper limit)
        """
        d = self.param_bounds_for_root(root)
        low = d.getLower(name)
        if low is not None:
            xmin = max(xmin, low) if xmin is not None else low
        up = d.getUpper(name)
        if up is not None:
            xmax = min(xmax, up) if xmax is not None else up
        return xmin, xmax

    def _get_param_bounds(self, roots, name):
        xmin, xmax = None, None
        for root in roots:
            xmin, xmax = self._check_param_ranges(root, name, xmin, xmax)
        return xmin, xmax

    def add_1d(self, root, param, plotno=0, normalized=None, ax=None, title_limit=None, **kwargs):
        """
        Low-level function to add a 1D marginalized density line to a plot

        :param root: The root name of the samples
        :param param: The parameter name
        :param plotno: The index of the line being added to the plot
        :param normalized: True if areas under lines should match, False if normalized to unit maximum.
                           Default from settings.norm_1d_density.
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param title_limit: if not None, a maginalized limit (1,2..) to print as the title of the plot
        :param kwargs: arguments for :func:`~matplotlib:matplotlib.pyplot.plot`
        :return: min, max for the plotted density
        """
        param = self._check_param(root, param)
        ax = self.get_axes(ax, pars=(param,))
        normalized = normalized if normalized is not None else self.settings.norm_1d_density
        if isinstance(root, MixtureND):
            density = root.density1D(param.name)
            if not normalized:
                density.normalize(by='max')
        else:
            density = self.sample_analyser.get_density(root, param, likes=self.settings.plot_meanlikes)
            if density is None:
                return None

        title_limit = title_limit if title_limit is not None else self.settings.title_limit
        if normalized:
            density.normalize()

        kwargs = self._get_line_styles(plotno, **kwargs)
        self.lines_added[plotno] = kwargs
        l, = ax.plot(density.x, density.P, **kwargs)
        if kwargs.get('dashes'):
            l.set_dashes(kwargs['dashes'])
        if self.settings.plot_meanlikes:
            kwargs['lw'] = self._scaled_linewidth(self.settings.linewidth_likes)
            ax.plot(density.x, density.likes, **kwargs)
        if title_limit:
            if isinstance(root, MixtureND):
                raise ValueError('title_limit not currently supported for MixtureND')
            samples = self.sample_analyser.samples_for_root(root)
            if self.settings.title_limit_labels:
                caption = samples.getInlineLatex(param, limit=title_limit)
            else:
                _, texs = samples.getLatex([param], title_limit)
                caption = texs[0]
            if '---' not in caption:
                ax.set_title('$' + caption + '$', fontsize=self._scaled_fontsize(self.settings.title_limit_fontsize,
                                                                                 self.settings.axes_fontsize))

        return density.bounds()

    def _get_paler_colors(self, color_rgb, n_levels, pale_factor=None):
        # convert a color into an array of colors for used in contours
        color = matplotlib.colors.colorConverter.to_rgb(color_rgb)
        pale_factor = pale_factor or self.settings.solid_contour_palefactor
        cols = [color]
        for _ in range(1, n_levels):
            cols = [[c * (1 - pale_factor) + pale_factor for c in cols[0]]] + cols
        return cols

    def add_2d_density_contours(self, density, **kwargs):
        """
        Low-level function to add 2D contours to a plot using provided density

        :param density: a :class:`.densities.Density2D` instance
        :param kwargs: arguments for :func:`~GetDistPlotter.add_2d_contours`
        :return: bounds (from :func:`~.densities.GridDensity.bounds`) of density
        """
        return self.add_2d_contours(None, density=density, **kwargs)

    def add_2d_contours(self, root, param1=None, param2=None, plotno=0, of=None, cols=None, contour_levels=None,
                        add_legend_proxy=True, param_pair=None, density=None, alpha=None, ax=None, **kwargs):
        """
        Low-level function to add 2D contours to plot for samples with given root name and parameters

        :param root: The root name of samples to use or a MixtureND gaussian mixture
        :param param1: x parameter
        :param param2: y parameter
        :param plotno: The index of the contour lines being added
        :param of: the total number of contours being added (this is line plotno of of)
        :param cols: optional list of colors to use for contours, by default uses default for this plotno
        :param contour_levels: levels at which to plot the contours, by default given by contours array in
                               the analysis settings
        :param add_legend_proxy: True if should add a proxy to the legend of this plot.
        :param param_pair: an [x,y] parameter name pair if you prefer to provide this rather than param1 and param2
        :param density: optional :class:`~.densities.Density2D` to plot rather than that computed automatically
                        from the samples
        :param alpha: alpha for the contours added
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: optional keyword arguments:

               - **filled**: True to make filled contours
               - **color**: top color to automatically make paling contour colours for a filled plot
               - kwargs for :func:`~matplotlib:matplotlib.pyplot.contour` and :func:`~matplotlib:matplotlib.pyplot.contourf`
        :return: bounds (from :meth:`~.densities.GridDensity.bounds`) for the 2D density plotted
        """

        ax = self.get_axes(ax)
        if density is None:
            param1, param2 = self.get_param_array(root, param_pair or [param1, param2])
            ax.getdist_params = (param1, param2)
            if isinstance(root, MixtureND):
                density = root.marginalizedMixture(params=[param1, param2]).density2D()
            else:
                density = self.sample_analyser.get_density_grid(root, param1, param2,
                                                                conts=self.settings.num_plot_contours,
                                                                likes=self.settings.shade_meanlikes)
            if density is None:
                if add_legend_proxy:
                    self.contours_added.append(None)
                return None
        if alpha is None:
            alpha = self._get_alpha_2d(plotno, **kwargs)
        if contour_levels is None:
            if not hasattr(density, 'contours'):
                contours = self.sample_analyser.ini.ndarray('contours')
                if contours is not None:
                    contours = contours[:self.settings.num_plot_contours]
                density.contours = density.getContourLevels(contours)
            contour_levels = density.contours

        if add_legend_proxy:
            proxy_ix = len(self.contours_added)
            self.contours_added.append(None)
        elif None in self.contours_added and self.contours_added.index(None) == plotno:
            proxy_ix = plotno
        else:
            proxy_ix = -1

        def clean_args(_args):  # prevent unused argument warnings
            _args = dict(_args)
            _args.pop('color', None)
            _args.pop('ls', None)
            _args.pop('lw', None)
            return _args

        if kwargs.get('filled'):
            if cols is None:
                color = kwargs.get('color')
                if color is None:
                    color = self._get_color_at_index(self.settings.solid_colors,
                                                     (of - plotno - 1) if of is not None else plotno)
                if isinstance(color, six.string_types) or matplotlib.colors.is_color_like(color):
                    cols = self._get_paler_colors(color, len(contour_levels))
                else:
                    cols = color
            levels = sorted(np.append([density.P.max() + 1], contour_levels))
            cs = ax.contourf(density.x, density.y, density.P, levels, colors=cols, alpha=alpha, **clean_args(kwargs))
            if proxy_ix >= 0:
                self.contours_added[proxy_ix] = (
                    matplotlib.patches.Rectangle((0, 0), 1, 1, fc=matplotlib.colors.to_rgb(cs.tcolors[-1][0])))
            ax.contour(density.x, density.y, density.P, levels[:1], colors=cs.tcolors[-1],
                       linewidths=self._scaled_linewidth(self.settings.linewidth_contour),
                       alpha=alpha * self.settings.alpha_factor_contour_lines, **clean_args(kwargs))
        else:
            args = self._get_line_styles(plotno, **kwargs)
            linestyles = [args['ls']]
            cols = [args['color']]
            lws = args['lw']  # not linewidth_contour is only used for filled contours
            kwargs = self._get_plot_args(plotno, **kwargs)
            kwargs['alpha'] = alpha
            cs = ax.contour(density.x, density.y, density.P, sorted(contour_levels), colors=cols, linestyles=linestyles,
                            linewidths=lws, **clean_args(kwargs))
            dashes = args.get('dashes')
            if dashes:
                for c in cs.collections:
                    c.set_dashes([(0, dashes)])
            if proxy_ix >= 0:
                line = matplotlib.lines.Line2D([0, 1], [0, 1], ls=linestyles[0], lw=lws, color=cols[0],
                                               alpha=args.get('alpha'))
                if dashes:
                    line.set_dashes(dashes)
                self.contours_added[proxy_ix] = line

        return density.bounds()

    def add_2d_shading(self, root, param1, param2, colormap=None, density=None, ax=None, **kwargs):
        """
        Low-level function to add 2D density shading to the given plot.

        :param root: The root name of samples to use
        :param param1: x parameter
        :param param2: y parameter
        :param colormap: color map, default to settings.colormap (see :class:`GetDistPlotSettings`)
        :param density: optional user-provided :class:`~.densities.Density2D` to plot rather than
                        the auto-generated density from the samples
         :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: keyword arguments for :func:`~matplotlib:matplotlib.pyplot.contourf`
        """
        param1, param2 = self.get_param_array(root, [param1, param2])
        ax = self.get_axes(ax, pars=(param1, param2))
        density = density or self.sample_analyser.get_density_grid(root, param1, param2,
                                                                   conts=self.settings.num_plot_contours,
                                                                   likes=self.settings.shade_meanlikes)
        if density is None:
            return
        if colormap is None:
            colormap = self.settings.colormap
        scalar_map = cm.ScalarMappable(cmap=colormap)
        cols = scalar_map.to_rgba(np.linspace(0, 1, self.settings.num_shades))
        # make sure outside area white and nice fade
        n = min(self.settings.num_shades // 3, 20)
        white = np.array([1, 1, 1, 1])
        # would be better to fade in alpha, but then the extra contourf fix doesn't work well
        for i in range(n):
            cols[i + 1] = (white * (n - i) + np.array(cols[i + 1]) * i) / float(n)
        cols[0][3] = 0  # keep edges clear
        # pcolor(density.x, density.y, density.P, cmap=self.settings.colormap, vmin=1. / self.settings.num_shades)
        levels = np.linspace(0, 1, self.settings.num_shades) ** self.settings.shade_level_scale
        if self.settings.shade_meanlikes:
            points = density.likes
        else:
            points = density.P
        ax.contourf(density.x, density.y, points, self.settings.num_shades, colors=cols, levels=levels, **kwargs)
        # doing contourf gets rid of annoying white lines in pdfs
        ax.contour(density.x, density.y, points, self.settings.num_shades, colors=cols, levels=levels, **kwargs)

    def add_2d_covariance(self, means, cov, xvals=None, yvals=None, def_width=4.0, samples_per_std=50., **kwargs):
        """
        Plot 2D Gaussian ellipse. By default plots contours for 1 and 2 sigma.
        Specify contour_levels argument to plot other contours (for density normalized to peak at unity).

        :param means: array of y
        :param cov: the 2x2 covariance
        :param xvals: optional array of x values to evaluate at
        :param yvals: optional array of y values to evaluate at
        :param def_width: if evaluation array not specified, width to use in units of standard deviation
        :param samples_per_std: if evaluation array not specified, number of grid points per standard deviation
        :param kwargs: keyword arguments for :func:`~GetDistPlotter.add_2D_contours`
        """

        cov = np.asarray(cov)
        assert (cov.shape[0] == 2 and cov.shape[1] == 2)
        if xvals is None:
            err = np.sqrt(cov[0, 0])
            xvals = np.arange(means[0] - def_width * err, means[0] + def_width * err, err / samples_per_std)
        if yvals is None:
            err = np.sqrt(cov[1, 1])
            yvals = np.arange(means[1] - def_width * err, means[1] + def_width * err, err / samples_per_std)
        x, y = np.meshgrid(xvals - means[0], yvals - means[1])
        inv_cov = np.linalg.inv(cov)
        like = x ** 2 * inv_cov[0, 0] + 2 * x * y * inv_cov[0, 1] + y ** 2 * inv_cov[1, 1]
        density = Density2D(xvals, yvals, np.exp(-like / 2))
        density.contours = [0.32, 0.05]
        return self.add_2d_density_contours(density, **kwargs)

    def add_2d_mixture_projection(self, mixture, param1, param2, **kwargs):
        density = mixture.marginalizedMixture(params=[param1, param2]).density2D()
        return self.add_2d_density_contours(density, **kwargs)

    def add_x_marker(self, marker, color=None, ls=None, lw=None, ax=None, **kwargs):
        """
        Adds a vertical line marking some x value. Optional arguments can override default settings.

        :param marker: The x coordinate of the location the marker line
        :param color: optional color of the marker
        :param ls: optional line style of the marker
        :param lw: optional line width
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: additional arguments to pass to :func:`~matplotlib:matplotlib.pyplot.axvline`
        """
        if color is None:
            color = self.settings.axis_marker_color
        if ls is None:
            ls = self.settings.axis_marker_ls
        if lw is None:
            lw = self.settings.axis_marker_lw
        self.get_axes(ax).axvline(marker, ls=ls, color=color, lw=lw, **kwargs)

    def add_y_marker(self, marker, color=None, ls=None, lw=None, ax=None, **kwargs):
        """
        Adds a horizontal line marking some y value. Optional arguments can override default settings.

        :param marker: The y coordinate of the location the marker line
        :param color: optional color of the marker
        :param ls: optional line style of the marker
        :param lw: optional line width.
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: additional arguments to pass to :func:`~matplotlib:matplotlib.pyplot.axhline`
        """
        if color is None:
            color = self.settings.axis_marker_color
        if ls is None:
            ls = self.settings.axis_marker_ls
        if lw is None:
            lw = self.settings.axis_marker_lw
        self.get_axes(ax).axhline(marker, ls=ls, color=color, lw=lw, **kwargs)

    def add_x_bands(self, x, sigma, color='gray', ax=None, alpha1=0.15, alpha2=0.1, **kwargs):
        """
        Adds vertical shaded bands showing one and two sigma ranges.

        :param x: central x value for bands
        :param sigma: 1 sigma error on x
        :param color: The base color to use
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param alpha1: alpha for the 1 sigma band; note this is drawn on top of the 2 sigma band. Set to zero if you
                       only want 2 sigma band
        :param alpha2: alpha for the 2 sigma band. Set to zero if you only want 1 sigma band
        :param kwargs: optional keyword arguments for :func:`~matplotlib:matplotlib.pyplot.axvspan`

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
            g = plots.get_single_plotter(width_inch=4)
            g.plot_2d([samples1, samples2], ['x0','x1'], filled=False);
            g.add_x_bands(0, 1)
        """
        ax = self.get_axes(ax)
        c = color
        if alpha2 > 0:
            ax.axvspan((x - sigma * 2), (x + sigma * 2), color=c, alpha=alpha2, **kwargs)
        if alpha1 > 0:
            ax.axvspan((x - sigma), (x + sigma), color=c, alpha=alpha1, **kwargs)

    def add_y_bands(self, y, sigma, color='gray', ax=None, alpha1=0.15, alpha2=0.1, **kwargs):
        """
        Adds horizontal shaded bands showing one and two sigma ranges.

        :param y: central y value for bands
        :param sigma: 1 sigma error on y
        :param color: The base color to use
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param alpha1: alpha for the 1 sigma band; note this is drawn on top of the 2 sigma band. Set to zero if
                       you only want 2 sigma band
        :param alpha2: alpha for the 2 sigma band. Set to zero if you only want 1 sigma band
        :param kwargs: optional keyword arguments for :func:`~matplotlib:matplotlib.pyplot.axhspan`

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples= gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=1)
            g = plots.get_single_plotter(width_inch=4)
            g.plot_2d(samples, ['x0','x1'], filled=True);
            g.add_y_bands(0, 1)
        """
        ax = self.get_axes(ax)
        c = color
        if alpha2 > 0:
            ax.axhspan((y - sigma * 2), (y + sigma * 2), color=c, alpha=alpha2, **kwargs)
        if alpha1 > 0:
            ax.axhspan((y - sigma), (y + sigma), color=c, alpha=alpha1, **kwargs)

    def add_bands(self, x, y, errors, color='gray', nbands=2, alphas=(0.25, 0.15, 0.1), lw=0.2,
                  lw_center=None, linecolor='k', ax=None):
        """
        Add a constraint band as a function of x showing e.g. a 1 and 2 sigma range.

        :param x: array of x values
        :param y: array of central values for the band as function of x
        :param errors: array of errors as a function of x
        :param color: a fill color
        :param nbands: number of bands to plot. If errors are 1 sigma, using nbands=2 will plot 1 and 2 sigma.
        :param alphas: tuple of alpha factors to use for each error band
        :param lw: linewidth for the edges of the bands
        :param lw_center: linewidth for the central mean line (zero or None not to have one, the default)
        :param linecolor: a line color for central line
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        """
        ax = self.get_axes(ax)
        if np.isscalar(y):
            y = np.ones(len(x)) * y
        for i in reversed(range(nbands)):
            ax.fill_between(x, y - (i + 1) * errors, y + (i + 1) * errors, color=color, alpha=alphas[i], lw=lw)
        if lw_center:
            ax.plot(x, y, color=linecolor or color, lw=lw_center)

    def _update_limit(self, bounds, curbounds):
        """
        Calculates the merge of two upper and lower limits, so result encloses both ranges

        :param bounds:  bounds to update
        :param curbounds:  bounds to add
        :return: The new limits
        """
        if not bounds:
            return curbounds
        if curbounds is None or curbounds[0] is None:
            return bounds
        return min(curbounds[0], bounds[0]), max(curbounds[1], bounds[1])

    def _update_limits(self, res, xlims, ylims, do_resize=True):
        """
        update 2D limits with new x and y limits (expanded unless doResize is False)

        :param res: The current limits
        :param xlims: The new lims for x
        :param ylims: The new lims for y.
        :param do_resize: True if should resize, False otherwise.
        :return: The newly calculated limits.
        """
        if res is None:
            return xlims, ylims
        if xlims is None and ylims is None:
            return res
        if not do_resize:
            return xlims, ylims
        else:
            return self._update_limit(res[0], xlims), self._update_limit(res[1], ylims)

    def _make_line_args(self, nroots, **kwargs):
        line_args = kwargs.get('line_args')
        if line_args is None:
            line_args = kwargs.get('contour_args')
        if line_args is None:
            line_args = [{}] * nroots
        elif isinstance(line_args, dict):
            line_args = [line_args] * nroots
        if len(line_args) < nroots:
            line_args += [{}] * (nroots - len(line_args))
        colors = self._get_color_at_index(kwargs.get('colors'))

        def _get_list(tag):
            ret = kwargs.get(tag)
            if ret is None:
                return None
            if not isinstance(ret, (list, tuple)):
                return [ret] * nroots
            return ret

        lws = _get_list('lws')
        alphas = _get_list('alphas')
        ls = _get_list('ls')
        for i, args in enumerate(line_args):
            c = args.copy()  # careful to copy before modifying any
            line_args[i] = c
            if colors and i < len(colors) and colors[i]:
                c['color'] = colors[i]
            if ls and i < len(ls) and ls[i]:
                c['ls'] = ls[i]
            if alphas and i < len(alphas) and alphas[i] is not None:
                c['alpha'] = alphas[i]
            if lws and i < len(lws) and lws[i]:
                c['lw'] = lws[i]
        return line_args

    def _make_contour_args(self, nroots, **kwargs):
        contour_args = self._make_line_args(nroots, **kwargs)
        filled = kwargs.get('filled')
        if filled and not isinstance(filled, bool):
            for cont, fill in zip(contour_args, filled):
                cont['filled'] = fill
        for cont in contour_args:
            if cont.get('filled') is None:
                cont['filled'] = filled or False
        return contour_args

    def _set_axis_formatter(self, axis, x):
        power_limits = self.settings.axis_tick_powerlimits
        if not x:
            # Avoid offset text on y axis where won't work on subplots
            ymin, ymax = axis.get_view_interval()
            if max(abs(ymax), abs(ymin)) <= 10 ** (power_limits[0] + 1) \
                    or max(abs(ymin), abs(ymax)) >= 10 ** power_limits[1]:
                axis.set_major_formatter(SciFuncFormatter())
                return

        formatter = ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_powerlimits(power_limits)
        axis.set_major_formatter(formatter)

    def _set_axis_properties(self, axis, rotation=0, tick_label_size=None):
        tick_label_size = self._scaled_fontsize(tick_label_size, self.settings.axes_fontsize)
        axis.set_tick_params(which='major', labelrotation=rotation, labelsize=tick_label_size)
        axis.get_offset_text().set_fontsize(tick_label_size * 3 / 4 if tick_label_size > 7 else tick_label_size)
        if isinstance(axis, matplotlib.axis.YAxis):
            self._auto_ticks(axis, prune=self._share_kwargs.get('hspace') is not None)
            if abs(rotation - 90) < 45:
                for ticklabel in axis.get_ticklabels():
                    ticklabel.set_verticalalignment("center")
        else:
            self._auto_ticks(axis, prune=self._share_kwargs.get('wspace') is not None)

    def _set_main_axis_properties(self, axis, x):
        """
        Sets axis properties.

        :param axis: The axis to set properties to.
        :param x: True if x axis, False for y axis
        """
        self._set_axis_formatter(axis, x)
        self._set_axis_properties(axis, self.settings.axis_tick_x_rotation if x else self.settings.axis_tick_y_rotation)

    @staticmethod
    def _no_x_ticklabels(ax):
        ax.tick_params(labelbottom=False)
        ax.xaxis.offsetText.set_visible(False)

    @staticmethod
    def _no_y_ticklabels(ax):
        ax.tick_params(labelleft=False)
        ax.yaxis.offsetText.set_visible(False)

    def set_axes(self, params=(), lims=None, do_xlabel=True, do_ylabel=True, no_label_no_numbers=False, pos=None,
                 color_label_in_axes=False, ax=None, **other_args):
        """
        Set the axis labels and ticks, and various styles. Do not usually need to call this directly.

        :param params: [x,y] list of the :class:`~.paramnames.ParamInfo` for the x and y parameters to use for labels
        :param lims: optional [xmin, xmax, ymin, ymax] to fix specific limits for the axes
        :param do_xlabel: True if should include label for x axis.
        :param do_ylabel: True if should include label for y axis.
        :param no_label_no_numbers: True to hide tick labels
        :param pos: optional position of the axes ['left' | 'bottom' | 'width' | 'height']
        :param color_label_in_axes: If True, and params has at last three entries, puts text in the axis to label
                                    the third parameter
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param other_args: Not used, just quietly ignore so that set_axes can be passed general kwargs
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance
        """
        ax = self.get_axes(ax)
        if lims is not None:
            ax.axis(lims)
        if do_xlabel or not no_label_no_numbers:
            self._set_main_axis_properties(ax.xaxis, True)
        if pos is not None:
            ax.set_position(pos)
        if do_xlabel and len(params) > 0:
            self.set_xlabel(params[0], ax)
        elif no_label_no_numbers:
            self._no_x_ticklabels(ax)
        if do_ylabel or not no_label_no_numbers:
            self._set_main_axis_properties(ax.yaxis, False)
        if len(params) > 1:
            if do_ylabel:
                self.set_ylabel(params[1], ax)
            elif no_label_no_numbers:
                self._no_y_ticklabels(ax)
        if color_label_in_axes and len(params) > 2:
            self.add_text(params[2].latexLabel(), ax=ax)
        return ax

    def set_xlabel(self, param, ax=None):
        """
        Sets the label for the x axis.

        :param param: the :class:`~.paramnames.ParamInfo` for the x axis parameter
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        """
        ax = self.get_axes(ax)
        lab_fontsize = self._scaled_fontsize(self.settings.axes_labelsize)
        ax.set_xlabel(param.latexLabel(), fontsize=lab_fontsize, verticalalignment='baseline',
                      labelpad=4 + lab_fontsize)

    def set_ylabel(self, param, ax=None, **kwargs):
        """
        Sets the label for the y axis.

        :param param: the :class:`~.paramnames.ParamInfo` for the y axis parameter
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: opional extra arguments for Axes set_ylabel
        """
        ax = self.get_axes(ax)
        ax.set_ylabel(param.latexLabel(), fontsize=self._scaled_fontsize(self.settings.axes_labelsize), **kwargs)

    def plot_1d(self, roots, param, marker=None, marker_color=None, label_right=False, title_limit=None,
                no_ylabel=False, no_ytick=False, no_zero=False, normalized=False, param_renames={}, ax=None, **kwargs):
        """
        Make a single 1D plot with marginalized density lines.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param param: the parameter name to plot
        :param marker: If set, places a marker at given coordinate.
        :param marker_color: If set, sets the marker color.
        :param label_right: If True, label the y axis on the right rather than the left
        :param title_limit: If not None, a maginalized limit (1,2..) of the first root to print as the title of the plot
        :param no_ylabel: If True excludes the label on the y axis
        :param no_ytick: If True show no y ticks
        :param no_zero: If true does not show tick label at zero on y axis
        :param normalized: plot normalized densities (if False, densities normalized to peak at 1)
        :param param_renames: optional dictionary mapping input parameter names to equivalent names used by the samples
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: additional optional keyword arguments:

                * **lims**: optional limits for x range of the plot [xmin, xmax]
                * **ls** : list of line styles for the different lines plotted
                * **colors**: list of colors for the different lines plotted
                * **lws**: list of line widths for the different lines plotted
                * **alphas**: list of alphas for the different lines plotted
                * **line_args**: a list of dictionaries with settings for each set of lines
                * arguments for :func:`~GetDistPlotter.set_axes`

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
            g = plots.get_single_plotter(width_inch=4)
            g.plot_1d([samples1, samples2], 'x0', marker=0)

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
            g = plots.get_single_plotter(width_inch=3)
            g.plot_1d([samples1, samples2], 'x0', normalized=True, colors=['green','black'])

        """
        roots = makeList(roots)
        ax = self.get_axes(ax, pars=(param,))
        plotparam = None
        plotroot = None
        _ret_range = kwargs.pop('_ret_range', None)
        _no_finish = kwargs.pop('_no_finish', False)
        line_args = self._make_line_args(len(roots), **kwargs)
        xmin, xmax = None, None
        for i, root in enumerate(roots):
            root_param = self._check_param(root, param, param_renames)
            if not root_param:
                continue
            bounds = self.add_1d(root, root_param, i, normalized=normalized, title_limit=title_limit if not i else 0,
                                 ax=ax, **line_args[i])
            xmin, xmax = self._update_limit(bounds, (xmin, xmax))
            if bounds is not None and not plotparam:
                plotparam = root_param
                plotroot = root
        if plotparam is None:
            raise GetDistPlotError('No roots have parameter: ' + str(param))
        if marker is not None:
            self.add_x_marker(marker, marker_color, ax=ax)
        if 'lims' in kwargs and kwargs['lims'] is not None:
            xmin, xmax = kwargs['lims']
        else:
            xmin, xmax = self._check_param_ranges(plotroot, plotparam.name, xmin, xmax)
        if normalized:
            mx = ax.yaxis.get_view_interval()[-1]
        else:
            mx = 1.099
        kwargs['lims'] = [xmin, xmax, 0, mx]
        self.set_axes([plotparam], ax=ax, **kwargs)

        if normalized:
            lab = self.settings.norm_prob_label
        else:
            lab = self.settings.prob_label
        if lab and not no_ylabel:
            if label_right:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            ax.set_ylabel(lab, fontsize=self._scaled_fontsize(self.settings.axes_labelsize))
        if no_ytick or not self.settings.prob_y_ticks:
            ax.tick_params(left=False, labelleft=False)
        elif no_ylabel:
            self._no_y_ticklabels(ax)
        elif no_zero and not normalized:
            ticks = ax.get_yticks()
            if ticks[-1] > 1:
                ticks = ticks[:-1]
            ax.set_yticks(ticks[1:])
        if _ret_range:
            return xmin, xmax
        elif not _no_finish and len(self.fig.axes) == 1:
            self.finish_plot()

    def plot_2d(self, roots, param1=None, param2=None, param_pair=None, shaded=False,
                add_legend_proxy=True, line_offset=0, proxy_root_exclude=(), ax=None, **kwargs):
        """
        Create a single 2D line, contour or filled plot.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param param1: x parameter name
        :param param2:  y parameter name
        :param param_pair: An [x,y] pair of params; can be set instead of param1 and param2
        :param shaded: True if plot should be a shaded density plot (for the first samples plotted)
        :param add_legend_proxy: True if should add to the legend proxy
        :param line_offset: line_offset if not adding first contours to plot
        :param proxy_root_exclude: any root names not to include when adding to the legend proxy
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: additional optional arguments:

                * **filled**: True for filled contours
                * **lims**: list of limits for the plot [xmin, xmax, ymin, ymax]
                * **ls** : list of line styles for the different sample contours plotted
                * **colors**: list of colors for the different sample contours plotted
                * **lws**: list of line widths for the different sample contours plotted
                * **alphas**: list of alphas for the different sample contours plotted
                * **line_args**: a list of dictionaries with settings for each set of contours
                * arguments for :func:`~GetDistPlotter.set_axes`
        :return: The xbounds, ybounds of the plot.

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.get_single_plotter(width_inch = 4)
            g.plot_2d([samples1,samples2], 'x1', 'x2', filled=True);

        """
        roots = makeList(roots)
        if isinstance(param1, (list, tuple)):
            param_pair = param1
            param1 = None
        _no_finish = kwargs.pop('_no_finish', False)
        param_pair = self.get_param_array(roots[0], param_pair or [param1, param2])
        ax = self.get_axes(ax, pars=param_pair)
        if self.settings.progress:
            print('plotting: ', [param.name for param in param_pair])
        if shaded and not kwargs.get('filled'):
            self.add_2d_shading(roots[0], param_pair[0], param_pair[1], ax=ax)
        xbounds, ybounds = None, None
        contour_args = self._make_contour_args(len(roots), **kwargs)
        for i, root in enumerate(roots):
            res = self.add_2d_contours(root, param_pair[0], param_pair[1], line_offset + i, of=len(roots), ax=ax,
                                       add_legend_proxy=add_legend_proxy and root not in proxy_root_exclude,
                                       **contour_args[i])
            xbounds, ybounds = self._update_limits(res, xbounds, ybounds)
        if xbounds is None:
            return
        if 'lims' not in kwargs:
            lim1 = self._check_param_ranges(roots[0], param_pair[0].name, xbounds[0], xbounds[1])
            lim2 = self._check_param_ranges(roots[0], param_pair[1].name, ybounds[0], ybounds[1])
            kwargs['lims'] = [lim1[0], lim1[1], lim2[0], lim2[1]]

        self.set_axes(param_pair, ax=ax, **kwargs)
        if not _no_finish and len(self.fig.axes) == 1:
            self.finish_plot()
        return xbounds, ybounds

    def default_col_row(self, nplot=1, nx=None, ny=None):
        """
        Get default subplot columns and rows depending on number of subplots.

        :param nplot: total number of subplots
        :param nx: optional specified number of columns
        :param ny: optional specified number of rows
        :return: n_cols, n_rows
        """
        plot_col = nx or int(round(np.sqrt(nplot / 1.4)))
        plot_row = ny or (nplot + plot_col - 1) // plot_col
        return plot_col, plot_row

    def make_figure(self, nplot=1, nx=None, ny=None, xstretch=1.0, ystretch=1.0, sharex=False, sharey=False):
        """
        Makes a new figure with one or more subplots.

        :param nplot: number of subplots
        :param nx: number of subplots in each row
        :param ny: number of subplots in each column
        :param xstretch: The parameter of how much to stretch the width, 1 is default
        :param ystretch: The parameter of how much to stretch the height, 1 is default. Note this multiplies
                         settings.subplot_size_ratio before determining actual stretch.
        :param sharex: no vertical space between subplots
        :param sharey: no horizontal space between subplots
        :return: The plot_col, plot_row numbers of subplots for the figure
        """
        self.new_plot()
        self.plot_col, self.plot_row = self.default_col_row(nplot, nx=nx, ny=ny)

        if self.settings.subplot_size_ratio:
            ystretch = ystretch * self.settings.subplot_size_ratio
        if self.settings.fig_width_inch is not None:
            figsize = (self.settings.fig_width_inch,
                       (self.settings.fig_width_inch * self.plot_row * ystretch) / (self.plot_col * xstretch))
            self._ax_width = self.settings.fig_width_inch / self.plot_col
        else:
            self._ax_width = self.settings.subplot_size_inch * xstretch
            figsize = (self.settings.subplot_size_inch * self.plot_col * xstretch,
                       self.settings.subplot_size_inch * self.plot_row * ystretch)
        if self.settings.constrained_layout:
            self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        else:
            self.fig = plt.figure(figsize=figsize)
        self.gridspec = matplotlib.gridspec.GridSpec(nrows=self.plot_row, ncols=self.plot_col, figure=self.fig)

        if sharey:
            self._share_kwargs = {'w_pad': 0, 'wspace': 0}
        else:
            self._share_kwargs = {}
        if sharex:
            self._share_kwargs.update({'h_pad': 0, 'hspace': 0})

        if self.settings.constrained_layout and self._share_kwargs:
            self.fig.set_constrained_layout_pads(**self._share_kwargs)

        self.subplots = np.ndarray((self.plot_row, self.plot_col), dtype=object)
        self.subplots[:, :] = None
        return self.plot_col, self.plot_row

    def get_param_array(self, root, params=None, renames={}):
        """
        Gets an array of :class:`~.paramnames.ParamInfo` for named params
        in the given `root`.

        If a parameter is not found in `root`, returns the original ParamInfo if ParamInfo
        was passed, or fails otherwise.

        :param root: The root name of the samples to use
        :param params: the parameter names (if not specified, get all)
        :param renames: optional dictionary mapping input names and equivalent names
                        used by the samples
        :return: list of :class:`~.paramnames.ParamInfo` instances for the parameters
        """
        if hasattr(root, 'param_names'):
            names = root.param_names
        elif hasattr(root, 'paramNames'):
            names = root.paramNames
        elif hasattr(root, 'names'):
            names = ParamNames(names=root.names, default=getattr(root, 'dim', 0))
        else:
            names = self.param_names_for_root(root)

        if params is None or len(params) == 0:
            return names.names
        # Fail only for parameters for which a string was passed
        if isinstance(params, six.string_types):
            error = True
        else:
            is_param_info = [isinstance(param, ParamInfo) for param in params]
            error = [not a for a in is_param_info]
            # Add renames of given ParamInfo's to the renames dict
            renames_from_param_info = {param.name: getattr(param, "renames", [])
                                       for i, param in enumerate(params) if is_param_info[i]}
            renames = mergeRenames(renames, renames_from_param_info)
            params = [getattr(param, "name", param) for param in params]
        old = [(old if isinstance(old, ParamInfo) else ParamInfo(old)) for old in params]
        return [new or old for new, old in zip(
            names.parsWithNames(params, error=error, renames=renames),
            old)]

    def _check_param(self, root, param, renames={}):
        """
        Get :class:`~.paramnames.ParamInfo` for given name for samples with specified root

        If a parameter is not found in `root`, returns the original ParamInfo if ParamInfo
        was passed, or fails otherwise.

        :param root: The root name of the samples
        :param param: The parameter name (or :class:`~.paramnames.ParamInfo`)
        :param renames: optional dictionary mapping input names and equivalent names
                        used by the samples
        :return: a :class:`~.paramnames.ParamInfo` instance, or None if name not found
        """
        if isinstance(param, ParamInfo):
            name = param.name
            if hasattr(param, 'renames'):
                renames = {name: makeList(renames.get(name, [])) + list(param.renames)}
        else:
            name = param
        # NB: If a parameter is not found, errors only if param is a ParamInfo instance
        return self.param_names_for_root(root).parWithName(name, error=(name == param), renames=renames)

    def param_latex_label(self, root, name, label_params=None):
        """
        Returns the latex label for given parameter.

        :param root: root name of the samples having the parameter (or :class:`~.mcsamples.MCSamples` instance)
        :param name:  The param name
        :param label_params: optional name of .paramnames file to override parameter name labels
        :return: The latex label
        """
        if label_params is not None:
            p = self.sample_analyser.params_for_root(root, label_params=label_params).parWithName(name)
        else:
            p = self._check_param(root, name)
        if not p:
            raise GetDistPlotError('Parameter not found: ' + name)
        return p.latexLabel()

    def add_legend(self, legend_labels, legend_loc=None, line_offset=0, legend_ncol=None, colored_text=None,
                   figure=False, ax=None, label_order=None, align_right=False, fontsize=None,
                   figure_legend_outside=True, **kwargs):
        """
        Add a legend to the axes or figure.

        :param legend_labels: The labels
        :param legend_loc: The legend location, default from settings
        :param line_offset: The offset of plotted lines to label (e.g. 1 to not label first line)
        :param legend_ncol: The number of columns in the legend, defaults to 1
        :param colored_text:
                             - True: legend labels are colored to match the lines/contours
                             - False: colored lines/boxes are drawn before black labels
        :param figure: True if legend is for the figure rather than the selected axes
        :param ax: if figure == False, the :class:`~matplotlib:matplotlib.axes.Axes` instance to use; defaults to
                    current axes.
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving
                            specific order of line indices
        :param align_right: True to align legend text at the right
        :param fontsize: The size of the font, default from settings
        :param figure_legend_outside: whether figure legend is outside or inside the subplots box
        :param kwargs: optional extra arguments for legend function
        :return: a :class:`matplotlib:matplotlib.legend.Legend` instance
        """
        if legend_loc is None:
            if figure:
                legend_loc = self.settings.figure_legend_loc
            else:
                legend_loc = self.settings.legend_loc
        legend_ncol = legend_ncol or self.settings.figure_legend_ncol or 1
        if colored_text is None:
            colored_text = self.settings.legend_colored_text
        lines = []
        if len(self.contours_added) == 0:
            for i in range(len(legend_labels)):
                args = self.lines_added.get(i)
                if not args:
                    if not figure:
                        ax_lines = self.get_axes(ax).lines
                        if len(ax_lines) > i:
                            lines.append(ax_lines[i])
                            continue
                    args = self._get_line_styles(i + line_offset)
                args.pop('filled', None)
                lines.append(matplotlib.lines.Line2D([0, 1], [0, 1], **args))
        else:
            lines = self.contours_added
        args = kwargs.copy()
        args['ncol'] = legend_ncol
        args['prop'] = {'size': self._scaled_fontsize(fontsize or self.settings.legend_fontsize
                                                      or self.settings.axes_labelsize)}
        if colored_text:
            args['handlelength'] = 0
            args['handletextpad'] = 0
        if label_order is not None:
            if str(label_order) == '-1':
                label_order = list(reversed(range(len(lines))))
            lines = [lines[i] for i in label_order]
            legend_labels = [legend_labels[i] for i in label_order]
        if figure:
            if figure_legend_outside and args.get('bbox_to_anchor') is None:
                # this should put directly on top/below of figure
                if legend_loc in ['best', 'center']:
                    legend_loc = 'upper center'
                loc1, loc2 = legend_loc.split(' ')
                if loc1 == 'center':
                    raise ValueError('Cannot use centre location for figure legend outside')
                subloc = ('upper', 'center', 'lower')[['lower', 'center', 'upper'].index(loc1)]
                new_legend_loc = subloc + ' ' + loc2
                frac = self.settings.legend_frac_subplot_margin
                if loc1 == 'upper':
                    args['bbox_to_anchor'] = (0 if loc2 == 'left' else
                                              (self.plot_col if loc2 == 'right' else self.plot_col / 2),
                                              1 + frac)
                    args['bbox_transform'] = self.subplots[0, 0].transAxes
                else:
                    args['bbox_to_anchor'] = (0 if loc2 == 'left' else (1 if loc2 == 'right' else 0.5),
                                              -frac / self.plot_row)
                    args['bbox_transform'] = self.fig.transFigure
                args['borderaxespad'] = 0
                legend_loc = new_legend_loc
                self.legend = self.fig.legend(lines, legend_labels, loc=legend_loc, **args)
            else:
                self.legend = self.fig.legend(lines, legend_labels, loc=legend_loc, **args)

            if not self.settings.figure_legend_frame:
                self.legend.get_frame().set_edgecolor('none')
        else:
            args['frameon'] = self.settings.legend_frame and not colored_text
            self.legend = self.get_axes(ax).legend(lines, legend_labels, loc=legend_loc, **args)
        if align_right:
            vp = self.legend._legend_box._children[-1]._children[0]
            for c in vp._children:
                c._children.reverse()
            vp.align = "right"
        if not self.settings.legend_rect_border:
            for rect in self.legend.get_patches():
                rect.set_edgecolor(rect.get_facecolor())
        if colored_text:
            for h, text in zip(self.legend.legendHandles, self.legend.get_texts()):
                h.set_visible(False)
                if isinstance(h, matplotlib.lines.Line2D):
                    c = h.get_color()
                elif isinstance(h, matplotlib.patches.Patch):
                    c = h.get_facecolor()
                else:
                    continue
                text.set_color(c)
        return self.legend

    def _scaled_fontsize(self, var, default=None):
        return self.settings.scaled_fontsize(self._ax_width, var, default)

    def _scaled_linewidth(self, linewidth):
        return self.settings.scaled_linewidth(self._ax_width, linewidth)

    def _subplots_adjust(self):
        if not self.settings.constrained_layout and self._share_kwargs:
            self.fig.subplots_adjust(wspace=self._share_kwargs.get('wspace'), hspace=self._share_kwargs.get('hspace'))

    def _tight_layout(self, rect=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gridspec.tight_layout(self.fig, h_pad=self._share_kwargs.get('h_pad'),
                                       w_pad=self._share_kwargs.get('w_pad'), rect=rect)

    def finish_plot(self, legend_labels=None, legend_loc=None, line_offset=0, legend_ncol=None, label_order=None,
                    no_extra_legend_space=False, no_tight=False, **legend_args):
        """
        Finish the current plot, adjusting subplot spacing and adding legend if required.

        :param legend_labels: The labels for a figure legend
        :param legend_loc: The legend location, default from settings (figure_legend_loc)
        :param line_offset: The offset of plotted lines to label (e.g. 1 to not label first line)
        :param legend_ncol: The number of columns in the legend, defaults to 1
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving
                            specific order of line indices
        :param no_extra_legend_space: True to put figure legend inside the figure box
        :param no_tight: don't use :func:`~matplotlib:matplotlib.pyplot.tight_layout` to adjust subplot positions
        :param legend_args: optional parameters for the legend
        """
        has_legend = self.settings.line_labels and legend_labels is not None and len(legend_labels) > 0

        if self.settings.tight_layout and not self.settings.constrained_layout and not no_tight:
            self._tight_layout()

        if has_legend:
            self.extra_artists = [self.add_legend(legend_labels,
                                                  legend_loc or self.settings.figure_legend_loc, line_offset,
                                                  legend_ncol, label_order=label_order, figure=True,
                                                  figure_legend_outside=not no_extra_legend_space, **legend_args)]
        self._subplots_adjust()

    def _root_display_name(self, root, i):
        if hasattr(root, 'get_label'):
            root = root.get_label()
        elif hasattr(root, 'getLabel'):
            root = root.getLabel()
        elif hasattr(root, 'label'):
            root = root.label
        elif hasattr(root, 'get_name'):
            root = escapeLatex(root.get_name())
        elif hasattr(root, 'getName'):
            root = escapeLatex(root.getName())
        elif isinstance(root, six.string_types):
            return self._root_display_name(self.sample_analyser.samples_for_root(root), i)
        if not root:
            root = 'samples' + str(i)
        return root

    def _default_legend_labels(self, legend_labels, roots):
        """
        Returns default legend labels, based on name tags of samples

        :param legend_labels: The current legend labels.
        :param roots: The root names of the samples
        :return: A list of labels
        """
        if legend_labels is None:
            if len(roots) < 2:
                return []
            return [self._root_display_name(root, i) for i, root in enumerate(roots) if root is not None]
        else:
            return legend_labels

    def plots_1d(self, roots, params=None, legend_labels=None, legend_ncol=None, label_order=None, nx=None,
                 param_list=None, roots_per_param=False, share_y=None, markers=None, title_limit=None,
                 xlims=None, param_renames={}, **kwargs):
        """
        Make an array of 1D marginalized density subplots

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param params: list of names of parameters to plot
        :param legend_labels: list of legend labels
        :param legend_ncol: Number of columns for the legend.
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving
                            specific order of line indices
        :param nx: number of subplots per row
        :param param_list: name of .paramnames file listing specific subset of parameters to plot
        :param roots_per_param: True to use a different set of samples for each parameter:
                      plots param[i] using roots[i] (where roots[i] is the list of sample root names to use for
                      plotting parameter i).  This is useful for example for  plotting one-parameter extensions of a
                      baseline model, each with various data combinations.
        :param share_y: True for subplots to share a common y axis with no horizontal space between subplots
        :param markers: optional dict giving vertical marker values indexed by parameter, or a list of marker values
                        for each parameter plotted
        :param title_limit: if not None, a maginalized limit (1,2..) of the first root to print as the title
                            of each of the plots
        :param xlims: list of [min,max] limits for the range of each parameter plot
        :param param_renames: optional dictionary holding mapping between input names and equivalent names used in
                              the samples.
        :param kwargs: optional keyword arguments for :func:`~GetDistPlotter.plot_1d`
        :return: The plot_col, plot_row subplot dimensions of the new figure

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.get_subplot_plotter()
            g.plots_1d([samples1, samples2], ['x0', 'x1', 'x2'], nx=3, share_y=True, legend_ncol =2,
                         markers={'x1':0}, colors=['red', 'green'], ls=['--', '-.'])

        """
        roots = makeList(roots)
        if roots_per_param:
            params = [self._check_param(root[0], param, param_renames) for root, param in zip(roots, params)]
        else:
            params = self.get_param_array(roots[0], params, param_renames)
        if param_list is None:
            param_list = kwargs.pop('paramList', None)
        if param_list is not None:
            wanted_params = ParamNames(param_list).list()
            params = [param for param in params if
                      param.name in wanted_params or param_renames.get(param.name, '') in wanted_params]
        nparam = len(params)
        if share_y is None:
            share_y = self.settings.prob_label is not None and nparam > 1
        elif nx is None and len(params) < 6:
            nx = len(params)
        plot_col, plot_row = self.make_figure(nparam, nx=nx, sharey=share_y)
        plot_roots = roots
        for i, param in enumerate(params):
            ax = self._subplot_number(i, pars=(param,),
                                      sharey=None if (i == 0 or not share_y or self.settings.norm_1d_density) else
                                      self.subplots[0, 0])
            if roots_per_param:
                plot_roots = roots[i]
            marker = self._get_marker(markers, i, param.name)
            no_ticks = share_y and i % self.plot_col > 0
            self.plot_1d(plot_roots, param, no_ytick=no_ticks, no_ylabel=no_ticks, marker=marker,
                         param_renames=param_renames, title_limit=title_limit, ax=ax, _no_finish=True, **kwargs)
            if xlims is not None:
                ax.set_xlim(xlims[i][0], xlims[i][1])

        self.finish_plot(self._default_legend_labels(legend_labels, roots), legend_ncol=legend_ncol,
                         label_order=label_order)

        return plot_col, plot_row

    def plots_2d(self, roots, param1=None, params2=None, param_pairs=None, nx=None, legend_labels=None,
                 legend_ncol=None, label_order=None, filled=False, shaded=False, **kwargs):
        """
        Make an array of 2D line, filled or contour plots.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of either of these) for the
                      samples to plot
        :param param1: x parameter to plot
        :param params2: list of y parameters to plot against x
        :param param_pairs: list of [x,y] parameter pairs to plot; either specify param1, param2, or param_pairs
        :param nx: number of subplots per row
        :param legend_labels: The labels used for the legend.
        :param legend_ncol: The amount of columns in the legend.
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving
                            specific order of line indices
        :param filled: True to plot filled contours
        :param shaded: True to shade by the density for the first root plotted
        :param kwargs: optional keyword arguments for :func:`~GetDistPlotter.plot_2d`
        :return: The plot_col, plot_row subplot dimensions of the new figure

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.get_subplot_plotter(subplot_size=4)
            g.settings.legend_frac_subplot_margin = 0.05
            g.plots_2d([samples1, samples2], param_pairs=[['x0', 'x1'], ['x1', 'x2']],
                                    nx=2, legend_ncol=2, colors=['blue', 'red'])
        """
        pairs = []
        roots = makeList(roots)
        if isinstance(param1, (list, tuple)) and len(param1) == 2:
            params2 = [param1[1]]
            param1 = param1[0]
        if param_pairs is None:
            if param1 is not None:
                param1 = self._check_param(roots[0], param1)
                params2 = self.get_param_array(roots[0], params2)
                for param in params2:
                    if param.name != param1.name:
                        pairs.append((param1, param))
            else:
                raise GetDistPlotError('No parameter or parameter pairs for 2D plot')
        else:
            for pair in param_pairs:
                pairs.append((self._check_param(roots[0], pair[0]), self._check_param(roots[0], pair[1])))
        if filled and shaded:
            raise GetDistPlotError("Plots cannot be both filled and shaded")
        plot_col, plot_row = self.make_figure(len(pairs), nx=nx)

        for i, pair in enumerate(pairs):
            ax = self._subplot_number(i, pars=pair)
            self.plot_2d(roots, param_pair=pair, filled=filled, shaded=not filled and shaded,
                         add_legend_proxy=i == 0, ax=ax, _no_finish=True, **kwargs)

        self.finish_plot(self._default_legend_labels(legend_labels, roots), legend_ncol=legend_ncol,
                         label_order=label_order)
        return plot_col, plot_row

    def plots_2d_triplets(self, root_params_triplets, nx=None, filled=False, x_lim=None):
        """
        Creates an array of 2D plots, where each plot uses different samples, x and y parameters

        :param root_params_triplets: a list of (root, x, y) giving sample root names, and x and y parameter names to
                                     plot in each subplot
        :param nx: number of subplots per row
        :param filled:  True for filled contours
        :param x_lim: limits for all the x axes.
        :return: The plot_col, plot_row subplot dimensions of the new figure
        """
        plot_col, plot_row = self.make_figure(len(root_params_triplets), nx=nx)
        for i, (root, param1, param2) in enumerate(root_params_triplets):
            ax = self._subplot_number(i, pars=(param1, param2))
            self.plot_2d(root, param_pair=[param1, param2], filled=filled, add_legend_proxy=i == 0,
                         ax=ax, _no_finish=True)
            if x_lim is not None:
                ax.set_xlim(x_lim)
        self.finish_plot()
        return plot_col, plot_row

    def get_axes(self, ax=None, pars=None):
        """
        Get the axes instance corresponding to the given subplot (y,x) coordinates, parameter list, or otherwise
        if ax is None get the last subplot axes used, or generate the first (possibly only) subplot if none.

        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes`, (y,x) subplot coordinate,
                   tuple of parameter names, or None to get last axes used or otherwise default to first subplot
        :param pars: optional list of parameters to associate with the axes
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance, or None if the specified axes don't exist
        """
        if isinstance(ax, int):
            ax = self._subplot_number(ax)
        elif isinstance(ax, (list, tuple)):
            if isinstance(ax[0], six.string_types) or isinstance(ax[0], ParamInfo):
                ax = self.get_axes_for_params(*ax)
            else:
                ax = self._subplot(ax[1], ax[0])
        else:
            ax = ax or self._last_ax
            if not ax:
                if self.fig and len(self.fig.axes):
                    # Allow attaching to axes created externally via pyplot commands
                    ax = self.fig.axes[0]
                    if self.subplots[0, 0] is None:
                        self._last_ax = ax
                        self.subplots[0, 0] = ax
                else:
                    ax = self._subplot_number(0)
        if pars is not None and ax is not None:
            ax.getdist_pars = pars
        return ax

    def _subplot(self, x, y, pars=None, **kwargs):
        """
        Create a subplot with given parameters.

        :param x: x location in the subplot grid
        :param y: y location in the subplot grid
        :param kwargs: arguments for :func:`~matplotlib:matplotlib.pyplot.subplot`
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance for the subplot axes
        """
        ax = self.subplots[y, x]
        if not ax:
            self.subplots[y, x] = ax = self.fig.add_subplot(self.gridspec[y, x], **kwargs)
        if pars is not None:
            ax.getdist_params = pars
        self._last_ax = ax
        return ax

    def _subplot_number(self, i, pars=None, **kwargs):
        """
        Create a subplot with given index.

        :param i: index of the subplot
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance for the subplot axes
        """
        if self.fig is None and i == 0:
            self.make_figure()
        return self._subplot(i % self.plot_col, i // self.plot_col, pars=pars, **kwargs)

    def _auto_ticks(self, axis, max_ticks=None, prune=True):
        axis.set_major_locator(
            BoundedMaxNLocator(nbins=max_ticks or self.settings.axis_tick_max_labels, prune=prune,
                               step_groups=self.settings.axis_tick_step_groups))

    @staticmethod
    def _inner_ticks(ax, top_and_left=True):
        for axis in [ax.get_xaxis(), ax.get_yaxis()]:
            axis.set_tick_params(which='both', direction='in', right=top_and_left, top=top_and_left)

    @staticmethod
    def _get_marker(markers, index, name):
        if markers is not None:
            if isinstance(markers, dict):
                return markers.get(name)
            elif index < len(markers):
                return markers[index]
        return None

    @staticmethod
    def _make_param_object(names, samples, obj=None):
        class SampleNames(object):
            pass

        obj = obj or SampleNames()
        for i, par in enumerate(names.names):
            setattr(obj, par.name, samples[:, i])
        return obj

    def triangle_plot(self, roots, params=None, legend_labels=None, plot_3d_with_param=None, filled=False, shaded=False,
                      contour_args=None, contour_colors=None, contour_ls=None, contour_lws=None, line_args=None,
                      label_order=None, legend_ncol=None, legend_loc=None, title_limit=None, upper_roots=None,
                      upper_kwargs={}, upper_label_right=False, diag1d_kwargs={}, markers=None, marker_args={},
                      param_limits={}, **kwargs):
        """
        Make a trianglular array of 1D and 2D plots.

        A triangle plot is an array of subplots with 1D plots along the diagonal, and 2D plots in the lower corner.
        The upper triangle can also be used by setting upper_roots.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param params: list of parameters to plot (default: all, can also use glob patterns to match groups of
                       parameters)
        :param legend_labels: list of legend labels
        :param plot_3d_with_param: for the 2D plots, make sample scatter plot, with samples colored by this parameter
                                   name (to make a '3D' plot)
        :param filled: True for filled contours
        :param shaded: plot shaded density for first root (cannot be used with filled)
        :param contour_args: optional dict (or list of dict) with arguments for each 2D plot
                            (e.g. specifying color, alpha, etc)
        :param contour_colors: list of colors for plotting contours (for each root)
        :param contour_ls: list of Line styles for contours (for each root)
        :param contour_lws: list of Line widths for contours (for each root)
        :param line_args: dict (or list of dict) with arguments for each 2D plot (e.g. specifying ls, lw, color, etc)
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving
                            specific order of line indices
        :param legend_ncol: The number of columns for the legend
        :param legend_loc: The location for the legend
        :param title_limit: if not None, a maginalized limit (1,2..) to print as the title of the first root on the
                            diagonal 1D plots
        :param upper_roots: set to fill the upper triangle with subplots using this list of sample root names
        :param upper_kwargs: dict for same-named arguments for use when making upper-triangle 2D plots
                             (contour_colors, etc). Set show_1d=False to not add to the diagonal.
        :param upper_label_right: when using upper_roots whether to label the y axis on the top-right axes
                                  (splits labels between left and right, but avoids labelling 1D y axes top left)
        :param diag1d_kwargs: list of dict for arguments when making 1D plots on grid diagonal
        :param markers: optional dict giving marker values indexed by parameter, or a list of marker values for
                        each parameter plotted
        :param marker_args: dictionary of optional arguments for adding markers (passed to axvline and/or axhline)
        :param param_limits: a dictionary holding a mapping from parameter names to axis limits for that parameter
        :param kwargs: optional keyword arguments for :func:`~GetDistPlotter.plot_2d`
                       or :func:`~GetDistPlotter.plot_3d` (lower triangle only)

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.get_subplot_plotter()
            g.triangle_plot([samples1, samples2], filled=True, legend_labels = ['Contour 1', 'Contour 2'])

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.get_subplot_plotter()
            g.triangle_plot([samples1, samples2], ['x0','x1','x2'], plot_3d_with_param='x3')

        """
        roots = makeList(roots)
        params = self.get_param_array(roots[0], params)
        plot_col = len(params)
        if plot_3d_with_param is not None:
            col_param = self._check_param(roots[0], plot_3d_with_param)
        self.make_figure(nx=plot_col, ny=plot_col, sharex=self.settings.no_triangle_axis_labels,
                         sharey=self.settings.no_triangle_axis_labels)
        lims = dict()
        if kwargs.pop('filled_compare', False):
            filled = True

        def _axis_y_limit_changed(_ax):
            _lims = _ax.get_ylim()
            other = _ax._shared_x_axis
            if _lims != other.get_xlim():
                other.set_xlim(_lims)

        def _axis_x_limit_changed(_ax):
            _lims = _ax.get_xlim()
            other = _ax._shared_y_axis
            if _lims != other.get_ylim():
                other.set_ylim(_lims)

        def def_line_args(cont_args, cont_colors):
            cols = []
            for plotno, _arg in enumerate(cont_args):
                if not _arg.get('filled'):
                    if cont_colors is not None and len(cont_colors) > plotno:
                        cols.append(cont_colors[plotno])
                    else:
                        cols.append(None)
                else:
                    cols.append(_arg.get('color') or self._get_color_at_index(self.settings.solid_colors,
                                                                              len(cont_args) - plotno - 1))
            _line_args = []
            for col in cols:
                if col is None:
                    _line_args.append({})
                else:
                    if isinstance(col, (tuple, list)) and not matplotlib.colors.is_color_like(col):
                        col = col[-1]
                    _line_args += [{'color': col}]
            return _line_args

        if upper_roots is not None:
            if plot_3d_with_param is not None:
                logging.warning("triangle_plot upper_roots currently doesn't work with plot_3d_with_param")
            upper_contour_args = self._make_contour_args(len(upper_roots), filled=upper_kwargs.get('filled', filled),
                                                         contour_args=upper_kwargs.get('contour_args', contour_args),
                                                         colors=upper_kwargs.get('contour_colors', contour_colors),
                                                         ls=upper_kwargs.get('contour_ls', contour_ls),
                                                         lws=upper_kwargs.get('contour_lws', contour_lws))
            upper_line_args = upper_kwargs.get('line_args') or def_line_args(upper_contour_args,
                                                                             upper_kwargs.get('contour_colors',
                                                                                              contour_colors))
            upargs = self._make_line_args(len(upper_roots), line_args=upper_line_args,
                                          ls=upper_kwargs.get('contour_ls', contour_ls),
                                          lws=upper_kwargs.get('contour_lws', contour_lws))

        contour_args = self._make_contour_args(len(roots), filled=filled, contour_args=contour_args,
                                               colors=contour_colors, ls=contour_ls, lws=contour_lws)
        if line_args is None:
            line_args = def_line_args(contour_args, contour_colors)
        line_args = self._make_line_args(len(roots), line_args=line_args, ls=contour_ls, lws=contour_lws)
        roots1d = copy.copy(roots)
        if upper_roots is not None:
            show_1d = upper_kwargs.get('show_1d', True)
            if isinstance(show_1d, bool):
                show_1d = [show_1d] * len(upargs)
            for root, arg, show in zip(upper_roots, upargs, show_1d):
                if show and root not in roots1d:
                    roots1d.append(root)
                    line_args.append(arg)

        bottom = len(params) - 1
        for i, param in enumerate(params):
            for i2 in range(bottom, i, -1):
                self._subplot(i, i2, pars=(param, params[i2]),
                              sharex=self.subplots[bottom, i] if i2 != bottom else None,
                              sharey=self.subplots[i2, 0] if i > 0 else None)

            ax = self._subplot(i, i, pars=(param,), sharex=self.subplots[bottom, i] if i != bottom else None)
            marker = self._get_marker(markers, i, param.name)
            self._inner_ticks(ax, False)
            xlim = self.plot_1d(roots1d, param, marker=marker, do_xlabel=i == plot_col - 1,
                                no_label_no_numbers=self.settings.no_triangle_axis_labels, title_limit=title_limit,
                                label_right=True, no_zero=True, no_ylabel=True, no_ytick=True, line_args=line_args,
                                lims=param_limits.get(param.name), ax=ax, _ret_range=True, **diag1d_kwargs)
            lims[i] = xlim
            if i > 0:
                ax._shared_y_axis = self.subplots[i, 0]
                ax.callbacks.connect('xlim_changed', _axis_x_limit_changed)

        if upper_roots is not None:
            if not upper_label_right:
                # make label on first 1D plot appropriate for 2D plots in rest of row
                label_ax = self.subplots[0, 0].twinx()
                self._inner_ticks(label_ax)
                label_ax.yaxis.tick_left()
                label_ax.yaxis.set_label_position('left')
                label_ax.yaxis.set_offset_position('left')
                label_ax.set_ylim(lims[0])
                self.set_ylabel(params[0], ax=label_ax)
                self._set_main_axis_properties(label_ax.yaxis, False)
                self.subplots[0, 0].yaxis.set_visible(False)
            else:
                label_ax = self.subplots[0, bottom]

            for y, param in enumerate(params[:-1]):
                for x in range(bottom, y, -1):
                    if y > 0:
                        share = self.subplots[y, 0]
                    else:
                        share = label_ax if (y < bottom or not upper_label_right) else None
                    self._subplot(x, y, pars=(params[x], param), sharex=self.subplots[bottom, x], sharey=share)

        for i, param in enumerate(params):
            marker = self._get_marker(markers, i, param.name)
            for i2 in range(i + 1, len(params)):
                param2 = params[i2]
                pair = [param, param2]
                marker2 = self._get_marker(markers, i2, param2.name)
                ax = self.subplots[i2, i]
                if plot_3d_with_param is not None:
                    self.plot_3d(roots, pair + [col_param], color_bar=False, line_offset=1, add_legend_proxy=False,
                                 do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0, contour_args=contour_args,
                                 no_label_no_numbers=self.settings.no_triangle_axis_labels, ax=ax, **kwargs)
                else:
                    self.plot_2d(roots, param_pair=pair, do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0,
                                 no_label_no_numbers=self.settings.no_triangle_axis_labels, shaded=shaded,
                                 add_legend_proxy=i == 0 and i2 == 1, contour_args=contour_args, ax=ax, **kwargs)
                if marker is not None:
                    self.add_x_marker(marker, ax=ax, **marker_args)
                if marker2 is not None:
                    self.add_y_marker(marker2, ax=ax, **marker_args)
                self._inner_ticks(ax)
                if i == 0:
                    ax.set_ylim(lims[i2])

                ax._shared_x_axis = self.subplots[bottom, i2]
                ax.callbacks.connect('ylim_changed', _axis_y_limit_changed)

                if i2 == bottom:
                    ax.set_xlim(lims[i])
                if i > 0:
                    ax._shared_y_axis = self.subplots[i, 0]
                    ax.callbacks.connect('xlim_changed', _axis_x_limit_changed)

                if upper_roots is not None:
                    if i == 0:
                        ax._shared_y_axis = label_ax
                        ax.callbacks.connect('xlim_changed', _axis_x_limit_changed)

                    ax = self.subplots[i, i2]
                    pair.reverse()
                    if plot_3d_with_param is not None:
                        self.plot_3d(upper_roots, pair + [col_param], color_bar=False, line_offset=1,
                                     add_legend_proxy=False, ax=ax, do_xlabel=False,
                                     do_ylabel=upper_label_right and i2 == bottom, contour_args=upper_contour_args,
                                     no_label_no_numbers=self.settings.no_triangle_axis_labels)
                    else:
                        self.plot_2d(upper_roots, param_pair=pair, do_xlabel=False,
                                     do_ylabel=upper_label_right and i2 == bottom,
                                     no_label_no_numbers=self.settings.no_triangle_axis_labels, shaded=shaded,
                                     add_legend_proxy=i == 0 and i2 == 1, ax=ax,
                                     proxy_root_exclude=[root for root in upper_roots if root in roots],
                                     contour_args=upper_contour_args)
                    if marker is not None:
                        self.add_y_marker(marker, ax=ax, **marker_args)
                    if marker2 is not None:
                        self.add_x_marker(marker2, ax=ax, **marker_args)
                    if upper_label_right and i2 == bottom:
                        ax.yaxis.set_label_position('right')
                        ax.yaxis.set_offset_position('right')
                        ax.yaxis.set_tick_params(which='both', labelright=True, labelleft=False)
                        self.set_ylabel(params[i], ax=ax, rotation=-90, va='bottom')

                    ax.set_xlim(lims[i2])
                    ax.set_ylim(lims[i])
                    ax._shared_x_axis = self.subplots[bottom, i]
                    ax.callbacks.connect('ylim_changed', _axis_y_limit_changed)
                    self._inner_ticks(ax)

        self._subplots_adjust()

        if plot_3d_with_param is not None:
            bottom = 0.5
            if len(params) == 2:
                bottom += 0.1
            cb = self.fig.colorbar(self.last_scatter, cax=self.fig.add_axes([0.9, bottom, 0.03, 0.35]))
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            self.rotate_yticklabels(cb.ax, rotation=self.settings.colorbar_tick_rotation or 0,
                                    labelsize=self.settings.colorbar_axes_fontsize)
            self.add_colorbar_label(cb, col_param, label_rotation=-self.settings.colorbar_label_rotation)

        labels = self._default_legend_labels(legend_labels, roots1d)

        if not legend_loc and self.settings.figure_legend_loc == 'upper center' and \
                len(params) < 4 and upper_roots is None:
            legend_loc = 'upper right'
        else:
            legend_loc = legend_loc or self.settings.figure_legend_loc
        args = {}
        if 'upper' in legend_loc and upper_roots is None:
            args['bbox_to_anchor'] = (self.plot_col / (2 if 'center' in legend_loc else 1), 1)
            args['bbox_transform'] = self.subplots[0, 0].transAxes
            args['borderaxespad'] = 0

        self.finish_plot(labels, label_order=label_order,
                         legend_ncol=legend_ncol or self.settings.figure_legend_ncol or (
                             None if upper_roots is None else len(labels)), legend_loc=legend_loc,
                         no_extra_legend_space=upper_roots is None, no_tight=title_limit or self.settings.title_limit,
                         **args)

    def rectangle_plot(self, xparams, yparams, yroots=None, roots=None, plot_roots=None, plot_texts=None,
                       xmarkers=None, ymarkers=None, marker_args={}, param_limits={},
                       legend_labels=None, legend_ncol=None, label_order=None, **kwargs):
        """
        Make a grid of 2D plots.

        A rectangle plot shows all x parameters plotted againts all y parameters in a grid of subplots with no spacing.

        Set roots to use the same set of roots for every plot in the rectangle, or set
        yroots (list of list of roots) to use different set of roots for each row of the plot; alternatively
        plot_roots allows you to specify explicitly (via list of list of list of roots) the set of roots for each
        individual subplot.

        :param xparams: list of parameters for the x axes
        :param yparams: list of parameters for the y axes
        :param yroots: (list of list of roots) allows use of different set of root names for each row of the plot;
                       set either roots or yroots
        :param roots: list of root names or :class:`~.mcsamples.MCSamples` instances.
                Uses the same set of roots for every plot in the rectangle; set either roots or yroots.
        :param plot_roots: Allows you to specify (via list of list of list of roots) the set of roots
                           for each individual subplot
        :param plot_texts: a 2D array (or list of lists) of a text label to put in each subplot
                           (use a None entry to skip one)
        :param xmarkers: optional dict giving vertical marker values indexed by parameter, or a list of marker values
                         for each x parameter plotted
        :param ymarkers: optional dict giving horizontal marker values indexed by parameter, or a list of marker values
                         for each y parameter plotted
        :param marker_args: arguments for :func:`~GetDistPlotter.add_x_marker` and :func:`~GetDistPlotter.add_y_marker`
        :param param_limits: a dictionary holding a mapping from parameter names to axis limits for that parameter
        :param legend_labels: list of labels for the legend
        :param legend_ncol: The number of columns for the legend
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific
                            order of line indices
        :param kwargs: arguments for :func:`~GetDistPlotter.plot_2d`.
        :return: the 2D list of :class:`~matplotlib:matplotlib.axes.Axes` created

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.get_subplot_plotter()
            g.rectangle_plot(['x0','x1'], ['x2','x3'], roots = [samples1, samples2], filled=True)
        """
        xparams = makeList(xparams)
        yparams = makeList(yparams)
        self.make_figure(nx=len(xparams), ny=len(yparams), sharex=len(yparams), sharey=len(xparams))
        sharey = None
        yshares = []
        xshares = []
        ax_arr = []
        if plot_roots and yroots or roots and yroots or plot_roots and roots:
            raise GetDistPlotError('rectangle plot: must have one of roots, yroots, plot_roots')
        if roots:
            roots = makeList(roots)
        limits = dict()
        for x, xparam in enumerate(xparams):
            sharex = None
            if plot_roots:
                yroots = plot_roots[x]
            elif roots:
                yroots = [roots for _ in yparams]
            axarray = []
            xmarker = self._get_marker(xmarkers, x, xparam)

            for y, (yparam, subplot_roots) in enumerate(zip(yparams, yroots)):
                if x > 0:
                    sharey = yshares[y]
                ax = self._subplot(x, y, pars=(xparam, yparam), sharex=sharex, sharey=sharey)
                if y == 0:
                    sharex = ax
                    xshares.append(ax)
                ymarker = self._get_marker(ymarkers, y, yparam)

                res = self.plot_2d(subplot_roots, param_pair=[xparam, yparam], do_xlabel=y == len(yparams) - 1,
                                   do_ylabel=x == 0, add_legend_proxy=x == 0 and y == 0, ax=ax, **kwargs)
                if xmarker is not None:
                    self.add_x_marker(xmarker, ax=ax, **marker_args)
                if ymarker is not None:
                    self.add_y_marker(ymarker, ax=ax, **marker_args)
                limits[xparam], limits[yparam] = self._update_limits(res, limits.get(xparam), limits.get(yparam))
                if y != len(yparams) - 1:
                    self._no_x_ticklabels(ax)
                if x != 0:
                    self._no_y_ticklabels(ax)
                if x == 0:
                    yshares.append(ax)
                if plot_texts and plot_texts[x][y]:
                    self.add_text_left(plot_texts[x][y], y=0.9, ax=ax)
                self._inner_ticks(ax)
                axarray.append(ax)
            ax_arr.append(axarray)
        for xparam, ax in zip(xparams, xshares):
            ax.set_xlim(param_limits.get(xparam, limits[xparam]))
        for yparam, ax in zip(yparams, yshares):
            ax.set_ylim(param_limits.get(yparam, limits[yparam]))
        self._subplots_adjust()
        if roots:
            legend_labels = self._default_legend_labels(legend_labels, roots)
        self.finish_plot(legend_labels=legend_labels, label_order=label_order,
                         legend_ncol=legend_ncol or self.settings.figure_legend_ncol or len(legend_labels))
        return ax_arr

    def rotate_xticklabels(self, ax=None, rotation=90, labelsize=None):
        """
        Rotates the x-tick labels by given rotation (degrees)

        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param rotation: How much to rotate in degrees.
        :param labelsize: size for tick labels (default from settings.axes_fontsize)
        """
        self._set_axis_properties(self.get_axes(ax).xaxis, rotation, labelsize)

    def rotate_yticklabels(self, ax=None, rotation=90, labelsize=None):
        """
        Rotates the y-tick labels by given rotation (degrees)

        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param rotation: How much to rotate in degrees.
        :param labelsize: size for tick labels (default from settings.axes_fontsize)
        """
        self._set_axis_properties(self.get_axes(ax).yaxis, rotation, labelsize)

    def add_colorbar(self, param, orientation='vertical', mappable=None, ax=None, **ax_args):
        """
        Adds a color bar to the given plot.

        :param param: a :class:`~.paramnames.ParamInfo` with label for the parameter the color bar is describing
        :param orientation: The orientation of the color bar (default: 'vertical')
        :param mappable: the thing to color, defaults to current scatter
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param ax_args: extra arguments -

               **color_label_in_axes** - if True, label is not added (insert as text label in plot instead)
        :return: The new :class:`~matplotlib:matplotlib.colorbar.Colorbar` instance
        """
        cb = self.fig.colorbar(mappable, orientation=orientation, ax=self.get_axes(ax))
        cb.set_alpha(1)
        if not ax_args.get('color_label_in_axes'):
            self.add_colorbar_label(cb, param)
        self._set_axis_properties(cb.ax.yaxis if orientation == 'vertical' else cb.ax.xaxis,
                                  self.settings.colorbar_tick_rotation or 0,
                                  self.settings.colorbar_axes_fontsize)
        return cb

    def add_line(self, xdata, ydata, zorder=0, color=None, ls=None, ax=None, **kwargs):
        """
        Adds a line to the given axes, using :class:`~matplotlib:matplotlib.lines.Line2D`

        :param xdata: pair of x coordinates
        :param ydata: pair of y coordinates
        :param zorder: Z-order for Line2D
        :param color: The color of the line, uses settings.axis_marker_color by default
        :param ls: The line style to be used, uses settings.axis_marker_ls by default
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs:  Additional arguments for :class:`~matplotlib:matplotlib.lines.Line2D`
        """
        if color is None:
            color = self.settings.axis_marker_color
        if ls is None:
            ls = self.settings.axis_marker_ls
        self.get_axes(ax).add_line(matplotlib.lines.Line2D(xdata, ydata, color=color, ls=ls, zorder=zorder, **kwargs))

    def add_colorbar_label(self, cb, param, label_rotation=None):
        """
        Adds a color bar label.

        :param cb: a :class:`~matplotlib:matplotlib.colorbar.Colorbar` instance
        :param param: a :class:`~.paramnames.ParamInfo` with label for the plotted parameter
        :param label_rotation: If set rotates the label (degrees)
        """

        label_rotation = label_rotation or self.settings.colorbar_label_rotation
        kwargs = {}
        if label_rotation and (10 < -label_rotation < 170):
            kwargs['va'] = 'bottom'
        cb.set_label(param.latexLabel(), fontsize=self._scaled_fontsize(self.settings.axes_labelsize),
                     rotation=label_rotation, labelpad=self.settings.colorbar_label_pad, **kwargs)

    def add_2d_scatter(self, root, x, y, color='k', alpha=1, extra_thin=1, scatter_size=None, ax=None):
        """
        Low-level function to adds a 2D sample scatter plot to the current axes (or ax if specified).

        :param root: The root name of the samples to use
        :param x: name of x parameter
        :param y: name of y parameter
        :param color: color to plot the samples
        :param alpha: The alpha to use.
        :param extra_thin: thin the weight one samples by this additional factor before plotting
        :param scatter_size: point size (default: settings.scatter_size)
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :return: (xmin, xmax), (ymin, ymax) bounds for the axes.
        """

        kwargs = {'fixed_color': color}
        return self.add_3d_scatter(root, [x, y], False, alpha, extra_thin, scatter_size, ax, **kwargs)

    def add_3d_scatter(self, root, params, color_bar=True, alpha=1, extra_thin=1, scatter_size=None,
                       ax=None, alpha_samples=False, **kwargs):
        """
        Low-level function to add a 3D scatter plot to the current axes (or ax if specified).

        :param root: The root name of the samples to use
        :param params:  list of parameters to plot
        :param color_bar: True to add a colorbar for the plotted scatter color
        :param alpha: The alpha to use.
        :param extra_thin: thin the weight one samples by this additional factor before plotting
        :param scatter_size: point size (default: settings.scatter_size)
        :param alpha_samples: use all samples, giving each point alpha corresponding to relative weight
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: arguments for :func:`~GetDistPlotter.add_colorbar`
        :return: (xmin, xmax), (ymin, ymax) bounds for the axes.
        """
        ax = self.get_axes(ax)
        params = self.get_param_array(root, params)
        if alpha_samples:
            mcsamples = self.sample_analyser.samples_for_root(root)
            weights, pts = mcsamples.weights, mcsamples.samples
        else:
            pts = self.sample_analyser.load_single_samples(root)
            weights = 1
        names = self.param_names_for_root(root)
        fixed_color = kwargs.get('fixed_color')  # if actually just a plain scatter plot
        samples = []
        for param in params:
            if hasattr(param, 'getDerived'):
                samples.append(param.getDerived(self._make_param_object(names, pts)))
            else:
                samples.append(pts[:, names.numberOfName(param.name)])
        if alpha_samples:
            # use most samples, but alpha with weight
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize, to_rgb
            max_weight = weights.max()
            dup_fac = 4
            filt = weights > max_weight / (100 * dup_fac)
            x = samples[0][filt]
            y = samples[1][filt]
            z = samples[2][filt]
            # split up high-weighted samples into multiple copies
            weights = weights[filt] / max_weight * dup_fac
            intweights = np.ceil(weights)
            thin_ix = mcsamples.thin_indices(1, intweights)
            x = x[thin_ix]
            y = y[thin_ix]
            z = z[thin_ix]
            weights /= intweights
            weights = weights[thin_ix]
            mappable = ScalarMappable(Normalize(z.min(), z.max()), self.settings.colormap_scatter)
            mappable.set_array(z)
            cols = mappable.to_rgba(z)
            if fixed_color:
                cols[:, :3] = to_rgb(fixed_color)
            cols[:, 3] = weights / dup_fac * alpha
            alpha = None
            self.last_scatter = mappable
            ax.scatter(x, y, edgecolors='none', s=scatter_size or self.settings.scatter_size,
                       c=cols, alpha=alpha)
        else:
            if extra_thin > 1:
                samples = [pts[::extra_thin] for pts in samples]
            self.last_scatter = ax.scatter(samples[0], samples[1], edgecolors='none',
                                           s=scatter_size or self.settings.scatter_size,
                                           c=fixed_color or samples[2],
                                           cmap=self.settings.colormap_scatter, alpha=alpha)

        if color_bar and not fixed_color:
            self.last_colorbar = self.add_colorbar(params[2], mappable=self.last_scatter, ax=ax, **kwargs)
        xbounds = [min(samples[0]), max(samples[0])]
        r = xbounds[1] - xbounds[0]
        xbounds[0] -= r / 20
        xbounds[1] += r / 20
        ybounds = [min(samples[1]), max(samples[1])]
        r = ybounds[1] - ybounds[0]
        ybounds[0] -= r / 20
        ybounds[1] += r / 20
        return [xbounds, ybounds]

    def plot_2d_scatter(self, roots, param1, param2, color='k', line_offset=0, add_legend_proxy=True, **kwargs):
        """
        Make a 2D sample scatter plot.

        If roots is a list of more than one, additional densities are plotted as contour lines.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param param1: name of x parameter
        :param param2: name of y parameter
        :param color: color to plot the samples
        :param line_offset: The line index offset for added contours
        :param add_legend_proxy: True if should add a legend proxy
        :param kwargs: additional optional arguments:

                * **filled**: True for filled contours for second and later items in roots
                * **lims**: limits for the plot [xmin, xmax, ymin, ymax]
                * **ls** : list of line styles for the different sample contours plotted
                * **colors**: list of colors for the different sample contours plotted
                * **lws**: list of linewidths for the different sample contours plotted
                * **alphas**: list of alphas for the different sample contours plotted
                * **line_args**: a list of dict with settings for contours from each root
        """
        kwargs = kwargs.copy()
        kwargs['fixed_color'] = color
        self.plot_3d(roots, [param1, param2], color_bar=False, line_offset=line_offset,
                     add_legend_proxy=add_legend_proxy, **kwargs)

    def plot_3d(self, roots, params=None, params_for_plots=None, color_bar=True, line_offset=0,
                add_legend_proxy=True, alpha_samples=False, ax=None, **kwargs):
        """
        Make a 2D scatter plot colored by the value of a third parameter (a 3D plot).

        If roots is a list of more than one, additional densities are plotted as contour lines.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param params: list with the three parameter names to plot (x, y, color)
        :param params_for_plots: list of parameter triplets to plot for each root plotted; more general
                                 alternative to params
        :param color_bar: True if should include a color bar
        :param line_offset: The line index offset for added contours
        :param add_legend_proxy: True if should add a legend proxy
        :param alpha_samples: if True, use alternative scatter style where all samples are plotted alphaed by
                              their weights
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: additional optional arguments:

                * **filled**: True for filled contours for second and later items in roots
                * **lims**: limits for the plot [xmin, xmax, ymin, ymax]
                * **ls** : list of line styles for the different sample contours plotted
                * **colors**: list of colors for the different sample contours plotted
                * **lws**: list of linewidths for the different sample contours plotted
                * **alphas**: list of alphas for the different sample contours plotted
                * **line_args**: a list of dict with settings for contours from each root
                * arguments for :func:`~GetDistPlotter.add_colorbar`

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=3, nMCSamples=2)
            g = plots.get_single_plotter(width_inch=4)
            g.plot_3d([samples1, samples2], ['x0','x1','x2']);
        """
        roots = makeList(roots)
        _no_finish = kwargs.pop('_no_finish', False)
        if params_for_plots:
            if params is not None:
                raise GetDistPlotError('plot_3d uses either params OR params_for_plots')
            params_for_plots = [self.get_param_array(root, p) for p, root in zip(params_for_plots, roots)]
        else:
            if not params:
                raise GetDistPlotError('No parameters for plot_3d!')
            params = self.get_param_array(roots[0], params)
            params_for_plots = [params for _ in roots]  # all the same
        ax = self.get_axes(ax, pars=params_for_plots[0])
        contour_args = self._make_contour_args(len(roots) - 1, **kwargs)
        xlims, ylims = self.add_3d_scatter(roots[0], params_for_plots[0], color_bar=color_bar,
                                           alpha_samples=alpha_samples, ax=ax, **kwargs)
        for i, root in enumerate(roots[1:]):
            params = params_for_plots[i + 1]
            res = self.add_2d_contours(root, params[0], params[1], i + line_offset, add_legend_proxy=add_legend_proxy,
                                       zorder=i + 1, ax=ax, **contour_args[i])
            xlims, ylims = self._update_limits(res, xlims, ylims)
        if 'lims' not in kwargs:
            params = params_for_plots[0]
            lim1 = self._check_param_ranges(roots[0], params[0].name, xlims[0], xlims[1])
            lim2 = self._check_param_ranges(roots[0], params[1].name, ylims[0], ylims[1])
            kwargs['lims'] = [lim1[0], lim1[1], lim2[0], lim2[1]]
        self.set_axes(params, ax=ax, **kwargs)
        if not _no_finish and self.plot_row == 1 and self.plot_col == 1:
            self.finish_plot()

    def plots_3d(self, roots, param_sets, nx=None, legend_labels=None, **kwargs):
        """
        Create multiple 3D subplots

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                      the samples to plot
        :param param_sets: A list of triplets of parameter names to plot [(x,y, color), (x2,y2,color2)..]
        :param nx: number of subplots per row
        :param legend_labels: list of legend labels
        :param kwargs: keyword arguments for  :func:`~GetDistPlotter.plot_3d`
        :return: The plot_col, plot_row subplot dimensions of the new figure

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=5, nMCSamples=2)
            g = plots.get_subplot_plotter(subplot_size=4)
            g.plots_3d([samples1, samples2], [['x0', 'x1', 'x2'], ['x3', 'x4', 'x2']], nx=2);
        """
        roots = makeList(roots)
        sets = [[self._check_param(roots[0], param) for param in param_group] for param_group in param_sets]
        plot_col, plot_row = self.make_figure(len(sets), nx=nx, ystretch=1 / 1.3)

        for i, triplet in enumerate(sets):
            ax = self._subplot_number(i, pars=triplet)
            self.plot_3d(roots, triplet, ax=ax, _no_finish=True, **kwargs)
        self.finish_plot(self._default_legend_labels(legend_labels, roots[1:]))
        return plot_col, plot_row

    def plots_3d_z(self, roots, param_x, param_y, param_z=None, max_z=None, **kwargs):
        """
        Make set of sample scatter subplots of param_x against param_y, each coloured by values of parameters
        in param_z (all if None). Any second or more samples in root are shown as contours.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for
                       the samples to plot
        :param param_x: x parameter name
        :param param_y: y parameter name
        :param param_z: list of parameter to names to color samples in each subplot (default: all)
        :param max_z: The maximum number of z parameters we should use.
        :param kwargs: keyword arguments for :func:`~GetDistPlotter.plot_3d`
        :return: The plot_col, plot_row subplot dimensions of the new figure
        """
        roots = makeList(roots)
        param_z = self.get_param_array(roots[0], param_z)
        if max_z is not None and len(param_z) > max_z:
            param_z = param_z[:max_z]
        param_x, param_y = self.get_param_array(roots[0], [param_x, param_y])
        sets = [[param_x, param_y, z] for z in param_z if z != param_x and z != param_y]
        return self.plots_3d(roots, sets, **kwargs)

    def add_text(self, text_label, x=0.95, y=0.06, ax=None, **kwargs):
        """
        Add text to given axis.

        :param text_label: The label to add.
        :param x: The x coordinate of where to add the label
        :param y: The y coordinate of where to add the label.
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: keyword arguments for :func:`~matplotlib:matplotlib.pyplot.text`
        """
        args = {'horizontalalignment': 'right' if x > 0.5 else 'left', 'verticalalignment': 'center',
                'fontsize': self._scaled_fontsize(self.settings.fontsize)}
        args.update(kwargs)
        ax = self.get_axes(ax)
        ax.text(x, y, text_label, transform=ax.transAxes, **args)

    def add_text_left(self, text_label, x=0.05, y=0.06, ax=None, **kwargs):
        """
        Add text to the left, Wraps add_text.

        :param text_label: The label to add.
        :param x: The x coordinate of where to add the label
        :param y: The y coordinate of where to add the label.
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance (or y,x subplot coordinate)
                   to add to (defaults to current plot or the first/main plot if none)
        :param kwargs: keyword arguments for :func:`~matplotlib:matplotlib.pyplot.text`
        """
        args = {'horizontalalignment': 'left'}
        args.update(kwargs)
        self.add_text(text_label, x, y, ax, **args)

    def export(self, fname=None, adir=None, watermark=None, tag=None):
        """
        Exports given figure to a file. If the filename is not specified, saves to a file with the same
        name as the calling script (useful for plot scripts where the script name matches the output figure).

        :param fname: The filename to export to. The extension (.pdf, .png, etc) determines the file type
        :param adir: The directory to save to
        :param watermark: a watermark text, e.g. to make the plot with some pre-final version number
        :param tag: A suffix to add to the filename.
        """
        if fname is None:
            fname = os.path.basename(sys.argv[0]).replace('.py', '')
        if tag:
            fname += '_' + tag
        if '.' not in fname:
            fname += '.' + getdist.default_plot_output
        if adir is not None and os.sep not in fname and '/' not in fname:
            fname = os.path.join(adir, fname)
        adir = os.path.dirname(fname)
        if adir and not os.path.exists(adir):
            os.makedirs(adir)
        if watermark:
            self.fig.text(0.45, 0.5, escapeLatex(watermark), fontsize=30, color='gray',
                          ha='center', va='center', alpha=0.2)

        self.fig.savefig(fname, bbox_extra_artists=self.extra_artists, bbox_inches='tight')

    def get_axes_for_params(self, *pars, **kwargs):
        """
        Get axes corresponding to given parameters

        :param pars: x or x,y or x,y,color parameters
        :param kwargs: set ordered=False to match y,x as well as x,y
        :return: axes instance or None if not found
        """
        ordered = kwargs.get('ordered', True)
        par_list = [p.name if isinstance(p, ParamInfo) else p for p in pars]
        if not ordered:
            par_list = set(par_list)
            func = set
        else:
            func = list
        for ax in self.subplots.reshape(-1):
            if ax:
                params = getattr(ax, 'getdist_params', None)
                if params is not None and \
                        func([p.name if isinstance(p, ParamInfo) else p for p in params]) == par_list:
                    self._last_ax = ax
                    return ax
        return None

    def samples_for_root(self, root, file_root=None, cache=True, settings=None):
        """
        Gets :class:`~.mcsamples.MCSamples` from root name
        (or just return root if it is already an MCSamples instance).

        :param root: The root name (without path, e.g. my_chains)
        :param file_root: optional full root path, by default searches in self.chain_dirs
        :param cache: if True, return cached object if already loaded
        :param settings: optional dictionary of settings to use
        :return: :class:`~.mcsamples.MCSamples` for the given root name
        """
        return self.sample_analyser.samples_for_root(root, file_root, cache, settings)


style_name = 'default'


class StyleManager(object):
    def __init__(self):
        self._plot_styles = {style_name: GetDistPlotter}
        self.active_style = style_name
        self._orig_rc = None

    def active_class(self, style=None):
        if style:
            self.set_active_style(style)
        return self._plot_styles[self.active_style]

    def set_active_style(self, name=None):
        name = name or style_name
        old_style = self.active_style
        if name != self.active_style:
            if name not in self._plot_styles:
                raise ValueError("Unknown style %s. Make sure you have imported the relevant style module." % name)
            if self._orig_rc is None:
                self._orig_rc = rcParams.copy()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rcParams.clear()
                    rcParams.update(self._orig_rc)

            self.active_style = name
            rcParams.update(self._plot_styles[name]._style_rc)
            if name == style_name:
                self._orig_rc = None
        return old_style

    def add_plotter_style(self, name, cls, activate=False):
        self._plot_styles[name] = cls
        if activate:
            self.set_active_style(name)


_style_manager = StyleManager()


def set_active_style(name=None):
    """
    Set an active style name. Each style name is associated with a :class:`~getdist.plots.GetDistPlotter` class
    used to generate plots, with optional custom plot settings and rcParams.
    The corresponding style module must have been loaded before using this.

    Note that because style modules can change rcParams, which is a global parameter,
    in general style settings are changed globally until changed back. But if your style does not change rcParams
    then you can also just pass a style name parameter when you make a plot instance.

    The supplied example styles are 'default', 'tab10' (default matplotlib color scheme) and 'planck' (more
    compilcated example using latex and various customized settings). Use :func:`add_plotter_style` to add
    your own style class.

    :param name: name of the style, or none to revert to default
    :return:  the previously active style name
    """
    return _style_manager.set_active_style(name)


def add_plotter_style(name, cls, activate=False):
    """
    Add a plotting style, consistenting of style name and a class type to use when making plotter instances.

    :param name: name for the style
    :param cls: a class inherited from :class:`~getdist.plots.GetDistPlotter`
    :param activate: whether to make it the active style
    """
    _style_manager.add_plotter_style(name, cls, activate)
