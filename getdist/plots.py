from __future__ import absolute_import
from __future__ import print_function
import os
import copy
import matplotlib
import sys
import six
import warnings

matplotlib.use('Agg', warn=False)
from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
from paramgrid import gridconfig, batchjob
import getdist
from getdist import MCSamples, loadMCSamples, ParamNames, ParamInfo, IniFile
from getdist.parampriors import ParamBounds
from getdist.densities import Density1D, Density2D
import logging

"""Plotting scripts for GetDist outputs"""


def makeList(roots):
    """
    Checks if the given parameter is a list, If not, Creates a list with the parameter as an item in it.

    :param roots: The parameter to check
    :return: A list containing the parameter.
    """
    if isinstance(roots, (list, tuple)):
        return roots
    else:
        return [roots]


class GetDistPlotError(Exception):
    """
    An exception that is raised when there is an error plotting
    """
    pass


class GetDistPlotSettings(object):
    """
    Settings class (colors, sizes, font, styles etc.)

    :ivar alpha_factor_contour_lines: alpha factor for adding contour lines between filled contours
    :ivar alpha_filled_add: alpha for adding filled contours to a plot
    :ivar axis_marker_color: The color for a marker
    :ivar axis_marker_ls: The line style for a marker
    :ivar axis_marker_lw: The line width for a marker
    :ivar colorbar_label_pad: padding for the colorbar labels
    :ivar colorbar_label_rotation: angle to rotate colorbar label (set to zero if -90 default gives layout problem)
    :ivar colorbar_rotation: angle to rotate colorbar tick labels
    :ivar colormap: a `Matplotlib color map <http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_ for shading
    :ivar colormap_scatter: a `Matplotlib color map <http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_ for 3D plots
    :ivar default_dash_styles: dict mapping line styles to detailed dash styles, default:  {'--': (3, 2), '-.': (4, 1, 1, 1)}
    :ivar fig_width_inch: The width of the figure in inches
    :ivar figure_legend_frame: draw box around figure legend
    :ivar figure_legend_loc: The location for the figure legend
    :ivar figure_legend_ncol: number of columns for figure legend
    :ivar legend_fontsize: The font size for the legend
    :ivar legend_frac_subplot_line: fraction of _subplot size to use per line for spacing figure legend
    :ivar legend_frac_subplot_margin: fraction of _subplot size to use for spacing figure legend above plots
    :ivar legend_frame: draw box around legend
    :ivar legend_loc: The location for the legend
    :ivar legend_position_config: recipe for positioning figure border (default 1)
    :ivar legend_rect_border: whether to have black border around solid color boxes in legends
    :ivar line_labels: True if you want to automatically add legends when adding more than one line to subplots
    :ivar lineM: list of default line styles/colors (['-k','-r'...])
    :ivar no_triangle_axis_labels: whether subplots in triangle plots should show axis labels if not at the edge
    :ivar norm_prob_label: label for the y axis in normalized 1D density plots
    :ivar num_plot_contours: number of contours to plot in 2D plots (up to number of contours in analysis settings)
    :ivar num_shades: number of distinct colors to use for shading shaded 2D plots
    :ivar param_names_for_labels: file name of .paramnames file to use for overriding parameter labels for plotting
    :ivar plot_args: dict, or list of dicts, giving settings like color, ls, alpha, etc. to apply for a plot or each line added
    :ivar plot_meanlikes: include mean likelihood lines in 1D plots
    :ivar prob_label: label for the y axis in unnormalized 1D density plots
    :ivar prob_y_ticks: show ticks on y axis for 1D density plots
    :ivar progress: write out some status
    :ivar shade_level_scale: shading contour colors are put at [0:1:spacing]**shade_level_scale
    :ivar shade_meanlikes: 2D shading uses mean likelihoods rather than marginalized density
    :ivar solid_colors: List of default colors for filled 2D plots. Each element is either a color, or a tuple of values for different contour levels.
    :ivar solid_contour_palefactor: factor by which to make 2D outer filled contours paler when only specifying one contour colour
    :ivar tick_prune: None, 'upper' or 'lower' to prune ticks
    :ivar tight_gap_fraction: fraction of plot width for closest tick to the edge
    :ivar tight_layout: use tight_layout to lay out and remove white space
    :ivar x_label_rotation: The rotation for the x label in degrees.
    """

    def __init__(self, subplot_size_inch=2, fig_width_inch=None):
        """
        If fig_width_inch set, fixed setting for fixed total figure size in inches.
        Otherwise use subplot_size_inch to determine default font sizes etc.,
        and figure will then be as wide as necessary to show all subplots at specified size.

        :param subplot_size_inch: Determines the size of subplots, and hence default font sizes
        :param fig_width_inch: The width of the figure in inches, If set, forces fixed total size.
        """
        self.plot_meanlikes = False
        self.shade_meanlikes = False
        self.prob_label = None
        # self.prob_label = 'Probability'
        self.norm_prob_label = 'P'
        self.prob_y_ticks = False
        self.lineM = ['-k', '-r', '-b', '-g', '-m', '-c', '-y', '--k', '--r', '--b', '--g',
                      '--m']  # : line styles/colors
        self.plot_args = None
        self.solid_colors = ['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm',
                             'r']
        self.default_dash_styles = {'--': (3, 2), '-.': (4, 1, 1, 1)}
        self.line_labels = True
        self.x_label_rotation = 0
        self.num_shades = 80
        self.shade_level_scale = 1.8  # contour levels at [0:1:spacing]**shade_level_scale
        self.fig_width_inch = fig_width_inch  # if you want to force specific fixed width
        self.progress = False
        self.tight_layout = True
        self.no_triangle_axis_labels = True
        # see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
        self.colormap = "Blues"
        self.colormap_scatter = "jet"
        self.colorbar_rotation = None  # e.g. -90
        self.colorbar_label_pad = 0
        self.colorbar_label_rotation = -90  # seems to cause problems with some versions, can set to zero

        self.setWithSubplotSize(subplot_size_inch)

        self.param_names_for_labels = None
        self.tick_prune = None  # 'lower' or 'upper'
        self.tight_gap_fraction = 0.13  # space between ticks and the edge

        self.legend_loc = 'best'
        self.figure_legend_loc = 'upper center'
        self.legend_frame = True
        self.figure_legend_frame = True
        self.figure_legend_ncol = 1

        self.legend_rect_border = False
        self.legend_position_config = 1

        self.legend_frac_subplot_margin = 0.2
        self.legend_frac_subplot_line = 0.1
        self.legend_fontsize = None

        self.num_plot_contours = 2
        self.solid_contour_palefactor = 0.6
        self.alpha_filled_add = 0.85
        self.alpha_factor_contour_lines = 0.5

        self.axis_marker_color = 'gray'
        self.axis_marker_ls = '--'
        self.axis_marker_lw = 0.5

    def setWithSubplotSize(self, size_inch=3.5, size_mm=None):
        """
        Sets the subplot's size, either in inches or in millimeters.
        If both are set, uses millimeters.

        :param size_inch: The size to set in inches; is ignored if size_mm is set.
        :param size_mm: None if not used, otherwise the size in millimeters we want to set for the subplot.
        """
        if size_mm: size_inch = size_mm * 0.0393700787
        self.subplot_size_inch = size_inch
        self.lab_fontsize = 7 + 2 * self.subplot_size_inch
        self.axes_fontsize = 4 + 2 * self.subplot_size_inch
        self.legend_fontsize = self.axes_fontsize
        self.font_size = self.lab_fontsize
        self.lw1 = self.subplot_size_inch / 3.0
        self.lw_contour = self.lw1 * 0.6
        self.lw_likes = self.subplot_size_inch / 6.0
        self.scatter_size = 3
        if size_inch > 4: self.scatter_size = size_inch * 2
        self.colorbar_axes_fontsize = self.axes_fontsize
        if self.colorbar_label_rotation:  self.colorbar_label_pad = size_inch * 3

    def rcSizes(self, axes_fontsize=None, lab_fontsize=None, legend_fontsize=None):
        """
        Sets the font sizes by default from matplotlib.rcParams defaults
        
        :param axes_fontsize: The font size for the plot axes tick labels (default: xtick.labelsize).
        :param lab_fontsize: The font size for the plot's axis labels (default: axes.labelsize)
        :param legend_fontsize: The font size for the plot's legend (default: legend.fontsize)
        """
        self.font_size = rcParams['font.size']
        self.legend_fontsize = legend_fontsize or rcParams['legend.fontsize']
        self.lab_fontsize = lab_fontsize or rcParams['axes.labelsize']
        self.axes_fontsize = axes_fontsize or rcParams['xtick.labelsize']
        if isinstance(self.axes_fontsize, six.integer_types):
            self.colorbar_axes_fontsize = self.axes_fontsize - 1
        else:
            self.colorbar_axes_fontsize = 'smaller'


defaultSettings = GetDistPlotSettings()


def getPlotter(**kwargs):
    """
    Creates a new plotter and returns it

    :param kwargs: arguments for :class:`~getdist.plots.GetDistPlotter`
    :return: The :class:`GetDistPlotter` instance
    """
    return GetDistPlotter(**kwargs)


def getSinglePlotter(ratio=3 / 4., width_inch=6, **kwargs):
    """
    Get a :class:`~.plots.GetDistPlotter` for making a single plot of fixed width. 
    
    For a half-column plot for a paper use width_inch=3.464.
    
    Use this or :func:`~getSubplotPlotter` to make a :class:`~.plots.GetDistPlotter` instance for making plots.
    If you want customized sizes or styles for all plots, you can make a new module
    defining these functions, and then use it exactly as a replacement for getdist.plots.

    :param ratio: The ratio between height and width.
    :param width_inch:  The width of the plot in inches
    :param kwargs: arguments for :class:`GetDistPlotter`
    :return: The :class:`~.plots.GetDistPlotter` instance
    """
    plotter = getPlotter(**kwargs)
    plotter.settings.setWithSubplotSize(width_inch)
    plotter.settings.fig_width_inch = width_inch
    plotter.make_figure(1, xstretch=1. / ratio)
    return plotter


def getSubplotPlotter(subplot_size=2, width_inch=None, **kwargs):
    """
    Get a :class:`~.plots.GetDistPlotter` for making an array of subplots. 
    
    If width_inch is None, just makes plot as big as needed for given subplot_size, otherwise fixes total width 
    and sets default font sizes etc. from matplotlib's default rcParams.

    Use this or :func:`~getSinglePlotter` to make a :class:`~.plots.GetDistPlotter` instance for making plots.
    If you want customized sizes or styles for all plots, you can make a new module
    defining these functions, and then use it exactly as a replacement for getdist.plots.

    :param subplot_size: The size of each subplot in inches
    :param width_inch: Optional total width in inches
    :param kwargs: arguments for :class:`GetDistPlotter`
    :return: The :class:`GetDistPlotter` instance
    """
    plotter = getPlotter(**kwargs)
    plotter.settings.setWithSubplotSize(subplot_size)
    if width_inch:
        plotter.settings.fig_width_inch = width_inch
        if not kwargs.get('settings'): plotter.settings.rcSizes()
    if subplot_size < 3 and kwargs.get('settings') is None and not width_inch:
        plotter.settings.axes_fontsize += 2
        plotter.settings.colorbar_axes_fontsize += 2
        plotter.settings.legend_fontsize = plotter.settings.lab_fontsize + 1
    return plotter


class SampleAnalysisGetDist(object):
    # Old class to support pre-computed GetDist plot_data output
    def __init__(self, plot_data):
        self.plot_data = plot_data
        self.newPlot()
        self.paths = dict()

    def newPlot(self):
        self.single_samples = dict()

    def get_density_grid(self, root, param1, param2, conts=2, likes=False):
        res = self.load_2d(root, param1, param2)
        if likes: res.likes = self.load_2d(root, param1, param2, '_likes', no_axes=True)
        if res is None: return None
        if conts > 0: res.contours = self.load_2d(root, param1, param2, '_cont', no_axes=True)[0:conts]
        return res

    def get_density(self, root, param, likes=False):
        pts = self.load_1d(root, param)
        if pts is None: return None
        result = Density1D(pts[:, 0], pts[:, 1])
        if likes: result.likes = self.load_1d(root, param, '.likes')[:, 1]
        return result

    def load_single_samples(self, root):
        if not root in self.single_samples: self.single_samples[root] = np.loadtxt(
            self.plot_data_file(root) + '_single.txt')[:, 2:]
        return self.single_samples[root]

    def paramsForRoot(self, root, labelParams=None):
        names = ParamNames(self.plot_data_file(root) + '.paramnames')
        if labelParams is not None: names.setLabelsAndDerivedFromParamNames(labelParams)
        return names

    def boundsForRoot(self, root):
        return ParamBounds(self.plot_data_file(root) + '.bounds')

    def plot_data_file(self, root):
        # find first match to roots that exist in list of plot_data paths
        if os.sep in root: return root
        path = self.paths.get(root, None)
        if path is not None: return path
        for datadir in self.plot_data:
            path = datadir + os.sep + root
            if os.path.exists(path + '.paramnames'):
                self.paths[root] = path
                return path
        return self.plot_data[0] + os.sep + root

    def plot_data_file_1D(self, root, name):
        return self.plot_data_file(root) + '_p_' + name

    def plot_data_file_2D(self, root, name1, name2):
        fname = self.plot_data_file(root) + '_2D_' + name2 + '_' + name1
        if not os.path.exists(fname):
            return self.plot_data_file(root) + '_2D_' + name1 + '_' + name2, True
        else:
            return fname, False

    def load_1d(self, root, param, ext='.dat'):
        fname = self.plot_data_file_1D(root, param.name) + ext
        if not hasattr(param, 'plot_data'): param.plot_data = dict()
        if not fname in param.plot_data:
            if not os.path.exists(fname):
                param.plot_data[fname] = None
            else:
                param.plot_data[fname] = np.loadtxt(fname)
        return param.plot_data[fname]

    def load_2d(self, root, param1, param2, ext='', no_axes=False):
        fname, transpose = self.plot_data_file_2D(root, param1.name, param2.name)
        if not os.path.exists(fname + ext): return None
        pts = np.loadtxt(fname + ext)
        if transpose: pts = pts.transpose()
        if no_axes: return pts
        x = np.loadtxt(fname + '_x')
        y = np.loadtxt(fname + '_y')
        if transpose:
            return Density2D(y, x, pts)
        else:
            return Density2D(x, y, pts)


class RootInfo(object):
    """
    Class to hold information about a set of samples loaded from file
    """

    def __init__(self, root, path, batch=None):
        """
        :param root: The root file to use.
        :param path: The path the root file is in.
        :param batch: optional batch object if loaded from a grid of results
        """
        self.root = root
        self.batch = batch
        self.path = path


class MCSampleAnalysis(object):
    """
    A class that loads and analyses samples, mapping root names to :class:`~.mcsamples.MCSamples` objects with caching.
    Typically accessed as the instance stored in plotter.sampleAnalyser, for example to 
    get an :class:`~.mcsamples.MCSamples` instance from a root name being used by a plotter, use plotter.sampleAnalyser.samplesForRoot(name).
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
        if chain_locations is not None:
            if isinstance(chain_locations, six.string_types):
                chain_locations = [chain_locations]
            for chain_dir in chain_locations:
                self.addChainDir(chain_dir)
        self.reset(settings)

    def addChainDir(self, chain_dir):
        """
        Adds a new chain directory or grid path for searching for samples

        :param chain_dir: The directory to add
        """
        if chain_dir in self.chain_locations: return
        self.chain_locations.append(chain_dir)
        isBatch = isinstance(chain_dir, batchjob.batchJob)
        if isBatch or gridconfig.pathIsGrid(chain_dir):
            if isBatch:
                batch = chain_dir
            else:
                batch = batchjob.readobject(chain_dir)
            self.chain_dirs.append(batch)
            # this gets things like specific parameter limits etc. specific to the grid
            # TODO: yuk, should get rid of this next refactor when grids should store this information
            if os.path.exists(batch.commonPath + 'getdist_common.ini'):
                batchini = IniFile(batch.commonPath + 'getdist_common.ini')
                if self.ini:
                    self.ini.params.update(batchini.params)
                else:
                    self.ini = batchini
        else:
            self.chain_dirs.append(chain_dir)

    def reset(self, settings=None):
        """
        Resets the caches, starting afresh optionally with new analysis settings

        :param settings: Either an :class:`~.inifile.IniFile` instance,
               the name of an .ini file, or a dict holding sample analysis settings.
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

    def samplesForRoot(self, root, file_root=None, cache=True):
        """
        Gets :class:`~.mcsamples.MCSamples` from root name (or just return root if it is already an MCSamples instance).

        :param root: The root name (without path, e.g. my_chains)
        :param file_root: optional full root path, by default searches in self.chain_dirs
        :param cache: if True, return cached object if already loaded
        :return: :class:`~.mcsamples.MCSamples` for the given root name
        """
        if isinstance(root, MCSamples): return root
        if os.path.isabs(root):
            root = os.path.basename(root)
        if root in self.mcsamples and cache: return self.mcsamples[root]
        jobItem = None
        dist_settings = {}
        if not file_root:
            for chain_dir in self.chain_dirs:
                if hasattr(chain_dir, "resolveRoot"):
                    jobItem = chain_dir.resolveRoot(root)
                    if jobItem:
                        file_root = jobItem.chainRoot
                        dist_settings = jobItem.dist_settings
                        break
                else:
                    name = os.path.join(chain_dir, root)
                    if os.path.exists(name + '_1.txt') or os.path.exists(name + '.txt'):
                        file_root = name
                        break
        if not file_root:
            raise GetDistPlotError('chain not found: ' + root)

        self.mcsamples[root] = loadMCSamples(file_root, self.ini, jobItem, settings=dist_settings)
        return self.mcsamples[root]

    def addRoots(self, roots):
        """
        A wrapper for addRoot that adds multiple file roots

        :param roots: An iterable containing filenames or :class:`RootInfo` objects to add
        """
        for root in roots:
            self.addRoot(root)

    def addRoot(self, file_root):
        """
        Add a root file for some new samples

        :param file_root: Either a file root name including path or a :class:`RootInfo` instance
        :return: :class:`~.mcsamples.MCSamples` instance for given root file.
        """
        if isinstance(file_root, RootInfo):
            if file_root.batch:
                return self.samplesForRoot(file_root.root)
            else:
                return self.samplesForRoot(file_root.root, os.path.join(file_root.path, file_root.root))
        else:
            return self.samplesForRoot(os.path.basename(file_root), file_root)

    def removeRoot(self, file_root):
        """
        Remove a given root file (does not delete it)

        :param file_root: The file root to remove
        """
        root = os.path.basename(file_root)
        self.mcsamples.pop(root, None)
        self.single_samples.pop(root, None)
        self.densities_1D.pop(root, None)
        self.densities_2D.pop(root, None)

    def newPlot(self):
        pass

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
        samples = self.samplesForRoot(root)
        key = (name, likes)
        rootdata.pop((name, not likes), None)
        density = rootdata.get(key)
        if density is None:
            density = samples.get1DDensityGridData(name, meanlikes=likes)
            if density is None: return None
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
            samples = self.samplesForRoot(root)
            density = samples.get2DDensityGridData(param1.name, param2.name, num_plot_contours=conts, meanlikes=likes)
            if density is None: return None
            rootdata[key] = density
        return density

    def load_single_samples(self, root):
        """
        Gets a set of unit weight samples for given root name, e.g. for making sample scatter plot

        :param root: The root name to use.
        :return: array of unit weight samples
        """
        if not root in self.single_samples:
            self.single_samples[root] = self.samplesForRoot(root).makeSingleSamples()
        return self.single_samples[root]

    def paramsForRoot(self, root, labelParams=None):
        """
        Returns a :class:`~.paramnames.ParamNames` with names and labels for parameters used by samples with a given root name.

        :param root: The root name of the samples to use.
        :param labelParams: optional name of .paramnames file containing labels to use for plots, overriding default
        :return: :class:`~.paramnames.ParamNames` instance
        """
        samples = self.samplesForRoot(root)
        names = samples.getParamNames()
        if labelParams is not None:
            names.setLabelsAndDerivedFromParamNames(os.path.join(batchjob.getCodeRootPath(), labelParams))
        return names

    def boundsForRoot(self, root):
        """
        Returns an object with getUpper and getLower to get hard prior bounds for given root name

        :param root: The root name to use.
        :return: object with getUpper() and getLower() functions
        """
        return self.samplesForRoot(root)  # #defines getUpper and getLower, all that's needed


class GetDistPlotter(object):
    """
    Main class for making plots from one or more sets of samples.

    :ivar settings: a :class:`GetDistPlotSettings` instance with settings
    :ivar subplots: a 2D array of :class:`~matplotlib:matplotlib.axes.Axes` for subplots
    :ivar sampleAnalyser: a :class:`MCSampleAnalysis` instance for getting :class:`~.mcsamples.MCSamples` 
         and derived data from a given root name tag (e.g. sampleAnalyser.samplesForRoot('rootname'))
    """

    def __init__(self, plot_data=None, chain_dir=None, settings=None, analysis_settings=None, mcsamples=True):
        """
        
        :param plot_data: (deprecated) directory name if you have pre-computed plot_data/ directory from GetDist; None by default
        :param chain_dir: Set this to a directory or grid root to search for chains (can also be a list of such, searched in order)
        :param analysis_settings: The settings to be used by :class:`MCSampleAnalysis` when analysing samples
        :param mcsamples: if True defaults to current method of using :class:`MCSampleAnalysis` instance to analyse chains on demand
        """
        self.chain_dir = chain_dir
        if settings is None:
            self.settings = copy.deepcopy(defaultSettings)
        else:
            self.settings = settings
        if chain_dir is None and plot_data is None: chain_dir = getdist.default_grid_root
        if isinstance(plot_data, six.string_types):
            self.plot_data = [plot_data]
        else:
            self.plot_data = plot_data
        if chain_dir is not None or mcsamples and plot_data is None:
            self.sampleAnalyser = MCSampleAnalysis(chain_dir, analysis_settings)
        else:
            self.sampleAnalyser = SampleAnalysisGetDist(self.plot_data)
        self.newPlot()

    def newPlot(self):
        """
        Resets the given plotter to make a new empty plot.
        """
        self.extra_artists = []
        self.contours_added = []
        self.lines_added = dict()
        self.param_name_sets = dict()
        self.param_bounds_sets = dict()
        self.sampleAnalyser.newPlot()
        self.fig = None
        self.subplots = None
        self.plot_col = 0

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
        elif isinstance(self.settings.plot_args, list):
            if len(self.settings.plot_args) > plotno:
                args = self.settings.plot_args[plotno]
                if args is None: args = dict()
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
        return self.settings.default_dash_styles.get(ls, None)

    def _get_default_ls(self, plotno=0):
        """
        Get default line style.

        :param plotno: The number of the line added to the plot to get the style of.
        :return: The default line style.
        """
        try:
            return self.settings.lineM[plotno]
        except IndexError:
            print('Error adding line ' + str(plotno) + ': Add more default line stype entries to settings.lineM')
            raise

    def _get_line_styles(self, plotno, **kwargs):
        """
        Gets the styles of the line for the given line added to a plot

        :param plotno: The number of the line added to the plot.
        :param kwargs: Params for :func:`~GetDistPlotter._get_plot_args`.
        :return: dict with ls, dashes, lw and color set appropriately 
        """
        args = self._get_plot_args(plotno, **kwargs)
        if not 'ls' in args: args['ls'] = self._get_default_ls(plotno)[:-1]
        if not 'dashes' in args:
            dashes = self._get_dashes_for_ls(args['ls'])
            if dashes is not None: args['dashes'] = dashes
        if not 'color' in args:
            args['color'] = self._get_default_ls(plotno)[-1]
        if not 'lw' in args: args['lw'] = self.settings.lw1
        return args

    def _get_color(self, plotno, **kwargs):
        """
        Get the color for the given line number

        :param plotno: line number added to plot
        :param kwargs: arguments for :func:`~GetDistPlotter._get_line_styles`
        :return: The color.
        """
        return self._get_line_styles(plotno, **kwargs)['color']

    def _get_linestyle(self, plotno, **kwargs):
        """
        Get line style for given plot line number.

        :param plotno: line number added to plot
        :param kwargs: arguments for :func:`~GetDistPlotter._get_line_styles`
        :return: The line style for the given plot line.
        """
        return self._get_line_styles(plotno, **kwargs)['ls']

    def _get_alpha2D(self, plotno, **kwargs):
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

    def paramNamesForRoot(self, root):
        """
        Get the parameter names and labels :class:`~.paramnames.ParamNames` instance for the given root name

        :param root: The root name of the samples.
        :return: :class:`~.paramnames.ParamNames` instance
        """
        if not root in self.param_name_sets: self.param_name_sets[root] = self.sampleAnalyser.paramsForRoot(root,
                                                                                                            labelParams=self.settings.param_names_for_labels)
        return self.param_name_sets[root]

    def paramBoundsForRoot(self, root):
        """
        Get any hard prior bounds for the parameters with root file name

        :param root: The root name to be used
        :return: object with getUpper() and getLower() bounds functions
        """
        if not root in self.param_bounds_sets: self.param_bounds_sets[root] = self.sampleAnalyser.boundsForRoot(root)
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
        d = self.paramBoundsForRoot(root)
        low = d.getLower(name)
        if low is not None: xmin = max(xmin, low)
        up = d.getUpper(name)
        if up is not None: xmax = min(xmax, up)
        return xmin, xmax

    def add_1d(self, root, param, plotno=0, normalized=False, ax=None, **kwargs):
        """
        Low-level function to add a 1D marginalized density line to a plot

        :param root: The root name of the samples
        :param param: The parameter name
        :param plotno: The index of the line being added to the plot
        :param normalized: True if areas under lines should match, False if normalized to unit maximum
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param kwargs: arguments for :func:`~matplotlib:matplotlib.pyplot.plot`
        :return: min, max for the plotted density
        """
        ax = ax or plt.gca()
        param = self._check_param(root, param)
        density = self.sampleAnalyser.get_density(root, param, likes=self.settings.plot_meanlikes)
        if density is None: return None;
        if normalized: density.normalize()

        kwargs = self._get_line_styles(plotno, **kwargs)
        self.lines_added[plotno] = kwargs
        l, = ax.plot(density.x, density.P, **kwargs)
        if kwargs.get('dashes'):
            l.set_dashes(kwargs['dashes'])
        if self.settings.plot_meanlikes:
            kwargs['lw'] = self.settings.lw_likes
            ax.plot(density.x, density.likes, **kwargs)

        return density.bounds()

    def add_2d_density_contours(self, density, **kwargs):
        """
        Low-level function to add 2D contours to a plot using provided density

        :param density: a :class:`.densities.Density2D` instance
        :param kwargs: arguments for :func:`~GetDistPlotter.add_2d_contours`
        :return: bounds (from :func:`.~densities.GridDensity.bounds`) of density
        """
        return self.add_2d_contours(None, density=density, **kwargs)

    def add_2d_contours(self, root, param1=None, param2=None, plotno=0, of=None, cols=None, contour_levels=None,
                        add_legend_proxy=True, param_pair=None, density=None, alpha=None, ax=None, **kwargs):
        """
        Low-level function to add 2D contours to plot for samples with given root name and parameters

        :param root: The root name of samples to use
        :param param1: x parameter
        :param param2: y parameter
        :param plotno: The index of the contour lines being added
        :param of: the total number of contours being added (this is line plotno of of)
        :param cols: optional list of colors to use for contours, by default uses default for this plotno
        :param contour_levels: levels at which to plot the contours, by default given by contours array in the analysis settings
        :param add_legend_proxy: True if should add a proxy to the legend of this plot.
        :param param_pair: an [x,y] parameter name pair if you prefer to provide this rather than param1 and param2
        :param density: optional :class:`~.densities.Density2D` to plot rather than that computed automatically from the samples
        :param alpha: alpha for the contours added
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param kwargs: optional keyword arguments:
        
               - **filled**: True to make filled contours
               - **color**: top color to automatically make paling contour colours for a filled plot
               - kwargs for :func:`~matplotlib:matplotlib.pyplot.contour` and :func:`~matplotlib:matplotlib.pyplot.contourf`
        :return: bounds (from :meth:`~.densities.GridDensity.bounds`) for the 2D density plotted
        """
        ax = ax or plt.gca()
        if density is None:
            param1, param2 = self.get_param_array(root, param_pair or [param1, param2])

            density = self.sampleAnalyser.get_density_grid(root, param1, param2,
                                                           conts=self.settings.num_plot_contours,
                                                           likes=self.settings.shade_meanlikes)
            if density is None:
                if add_legend_proxy: self.contours_added.append(None)
                return None
        if alpha is None: alpha = self._get_alpha2D(plotno, **kwargs)
        if contour_levels is None: contour_levels = density.contours

        if add_legend_proxy:
            proxyIx = len(self.contours_added)
            self.contours_added.append(None)
        elif None in self.contours_added and self.contours_added.index(None) == plotno:
            proxyIx = plotno
        else:
            proxyIx = -1

        if kwargs.get('filled'):
            if cols is None:
                color = kwargs.get('color')
                if color is None:
                    if of is not None:
                        color = self.settings.solid_colors[of - plotno - 1]
                    else:
                        color = self.settings.solid_colors[plotno]
                if isinstance(color, six.string_types):
                    cols = [matplotlib.colors.colorConverter.to_rgb(color)]
                    for _ in range(1, len(contour_levels)):
                        cols = [[c * (1 - self.settings.solid_contour_palefactor) +
                                 self.settings.solid_contour_palefactor for c in cols[0]]] + cols
                else:
                    cols = color
            levels = sorted(np.append([density.P.max() + 1], contour_levels))
            CS = ax.contourf(density.x, density.y, density.P, levels, colors=cols, alpha=alpha, **kwargs)
            if proxyIx >= 0: self.contours_added[proxyIx] = (plt.Rectangle((0, 0), 1, 1, fc=CS.tcolors[-1][0]))
            ax.contour(density.x, density.y, density.P, levels[:1], colors=CS.tcolors[-1],
                       linewidths=self.settings.lw_contour, alpha=alpha * self.settings.alpha_factor_contour_lines,
                       **kwargs)
        else:
            args = self._get_line_styles(plotno, **kwargs)
            # if color is None: color = self._get_color(plotno, **kwargs)
            # cols = [color]
            # if ls is None: ls = self._get_linestyle(plotno, **kwargs)
            linestyles = [args['ls']]
            cols = [args['color']]
            kwargs = self._get_plot_args(plotno, **kwargs)
            kwargs['alpha'] = alpha
            CS = ax.contour(density.x, density.y, density.P, contour_levels, colors=cols, linestyles=linestyles,
                            linewidths=self.settings.lw_contour, **kwargs)
            dashes = args.get('dashes')
            if dashes:
                for c in CS.collections:
                    c.set_dashes([(0, dashes)])
            if proxyIx >= 0:
                line = plt.Line2D([0, 1], [0, 1], ls=linestyles[0], lw=self.settings.lw_contour, color=cols[0],
                                  alpha=args.get('alpha'))
                if dashes: line.set_dashes(dashes)
                self.contours_added[proxyIx] = line

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
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param kwargs: keyword arguments for :func:`~matplotlib:matplotlib.pyplot.contourf`
        """
        ax = ax or plt.gca()
        param1, param2 = self.get_param_array(root, [param1, param2])
        density = density or self.sampleAnalyser.get_density_grid(root, param1, param2,
                                                                  conts=self.settings.num_plot_contours,
                                                                  likes=self.settings.shade_meanlikes)
        if density is None: return
        if colormap is None: colormap = self.settings.colormap
        scalarMap = cm.ScalarMappable(cmap=colormap)
        cols = scalarMap.to_rgba(np.linspace(0, 1, self.settings.num_shades))
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

    def _updateLimit(self, bounds, curbounds):
        """
        Calculates the merge of two upper and lower limits, so result encloses both ranges

        :param bounds:  bounds to update
        :param curbounds:  bounds to add
        :return: The new limits
        """
        if not bounds: return curbounds
        if curbounds is None or curbounds[0] is None: return bounds
        return min(curbounds[0], bounds[0]), max(curbounds[1], bounds[1])

    def _updateLimits(self, res, xlims, ylims, doResize=True):
        """
        update 2D limits with new x and y limits (expanded unless doResize is False)

        :param res: The current limits
        :param xlims: The new lims for x
        :param ylims: The new lims for y.
        :param doResize: True if should resize, False otherwise.
        :return: The newly calculated limits.
        """
        if res is None: return xlims, ylims
        if xlims is None and ylims is None: return res
        if not doResize:
            return xlims, ylims
        else:
            return self._updateLimit(res[0], xlims), self._updateLimit(res[1], ylims)

    def _make_line_args(self, nroots, **kwargs):
        line_args = kwargs.get('line_args')
        if line_args is None: line_args = kwargs.get('contour_args')
        if line_args is None:
            line_args = [{}] * nroots
        elif isinstance(line_args, dict):
            line_args = [line_args] * nroots
        if len(line_args) < nroots: line_args += [{}] * (nroots - len(line_args))
        colors = kwargs.get('colors')
        lws = kwargs.get('lws')
        alphas = kwargs.get('alphas')
        ls = kwargs.get('ls')
        for i, args in enumerate(line_args):
            c = args.copy()  # careful to copy before modifying any
            line_args[i] = c
            if colors and i < len(colors) and colors[i]:
                c['color'] = colors[i]
            if ls and i < len(ls) and ls[i]: c['ls'] = ls[i]
            if alphas and i < len(alphas) and alphas[i]: c['alpha'] = alphas[i]
            if lws and i < len(lws) and lws[i]: c['lw'] = lws[i]
        return line_args

    def _make_contour_args(self, nroots, **kwargs):
        contour_args = self._make_line_args(nroots, **kwargs)
        filled = kwargs.get('filled')
        if filled and not isinstance(filled, bool):
            for cont, fill in zip(contour_args, filled):
                cont['filled'] = fill
        for cont in contour_args:
            if cont.get('filled') is None: cont['filled'] = filled or False
        return contour_args

    def plot_2d(self, roots, param1=None, param2=None, param_pair=None, shaded=False,
                add_legend_proxy=True, line_offset=0, proxy_root_exclude=[], **kwargs):
        """
        Create a single 2D line, contour or filled plot.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param param1: x parameter name
        :param param2:  y parameter name
        :param param_pair: An [x,y] pair of params; can be set instead of param1 and param2
        :param shaded: True if plot should be a shaded density plot (for the first samples plotted)
        :param add_legend_proxy: True if should add to the legend proxy
        :param line_offset: line_offset if not adding first contours to plot
        :param proxy_root_exclude: any root names not to include when adding to the legend proxy
        :param kwargs: additional optional arguments:
        
                * **filled**: True for filled contours
                * **lims**: list of limits for the plot [xmin, xmax, ymin, ymax]
                * **ls** : list of line styles for the different sample contours plotted 
                * **colors**: list of colors for the different sample contours plotted 
                * **lws**: list of line widths for the different sample contours plotted
                * **alphas**: list of alphas for the different sample contours plotted 
                * **line_args**: a list of dictionaries with settings for each set of contours
                * arguments for :func:`~GetDistPlotter.setAxes`
        :return: The xbounds, ybounds of the plot.
        
        .. plot::
           :include-source:
           
            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.getSinglePlotter(width_inch = 4)
            g.plot_2d([samples1,samples2], 'x1', 'x2', filled=True);

        """
        if self.fig is None: self.make_figure()
        roots = makeList(roots)
        if isinstance(param1, (list, tuple)):
            param_pair = param1
            param1 = None
        param_pair = self.get_param_array(roots[0], param_pair or [param1, param2])
        if self.settings.progress: print('plotting: ', [param.name for param in param_pair])
        if shaded and not kwargs.get('filled'): self.add_2d_shading(roots[0], param_pair[0], param_pair[1])
        xbounds, ybounds = None, None
        contour_args = self._make_contour_args(len(roots), **kwargs)
        for i, root in enumerate(roots):
            res = self.add_2d_contours(root, param_pair[0], param_pair[1], line_offset + i, of=len(roots),
                                       add_legend_proxy=add_legend_proxy and not root in proxy_root_exclude,
                                       **contour_args[i])
            xbounds, ybounds = self._updateLimits(res, xbounds, ybounds)
        if xbounds is None: return
        if not 'lims' in kwargs:
            lim1 = self._check_param_ranges(roots[0], param_pair[0].name, xbounds[0], xbounds[1])
            lim2 = self._check_param_ranges(roots[0], param_pair[1].name, ybounds[0], ybounds[1])
            kwargs['lims'] = [lim1[0], lim1[1], lim2[0], lim2[1]]

        self.setAxes(param_pair, **kwargs)
        return xbounds, ybounds

    def add_x_marker(self, marker, color=None, ls=None, lw=None, ax=None, **kwargs):
        """
        Adds a vertical line marking some x value. Optional arguments can override default settings.

        :param marker: The x coordinate of the location the marker line
        :param color: optional color of the marker
        :param ls: optional line style of the marker
        :param lw: optional line width
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param kwargs: additional arguments to pass to :func:`~matplotlib:matplotlib.pyplot.axvline`
        """
        if color is None: color = self.settings.axis_marker_color
        if ls is None: ls = self.settings.axis_marker_ls
        if lw is None: lw = self.settings.axis_marker_lw
        (ax or plt.gca()).axvline(marker, ls=ls, color=color, lw=lw, **kwargs)

    def add_y_marker(self, marker, color=None, ls=None, lw=None, ax=None, **kwargs):
        """
        Adds a horizontal line marking some y value. Optional arguments can override default settings.

        :param marker: The y coordinate of the location the marker line
        :param color: optional color of the marker
        :param ls: optional line style of the marker
        :param lw: optional line width.
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param kwargs: additional arguments to pass to :func:`~matplotlib:matplotlib.pyplot.axhline`
        """
        if color is None: color = self.settings.axis_marker_color
        if ls is None: ls = self.settings.axis_marker_ls
        if lw is None: lw = self.settings.axis_marker_lw
        (ax or plt.gca()).axhline(marker, ls=ls, color=color, lw=lw, **kwargs)

    def add_x_bands(self, x, sigma, color='gray', ax=None, alpha1=0.15, alpha2=0.1, **kwargs):
        """
        Adds vertical shaded bands showing one and two sigma ranges.

        :param x: central x value for bands
        :param sigma: 1 sigma error on x
        :param color: The base color to use
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add the bands to (defaults to current plot)
        :param alpha1: alpha for the 1 sigma band; note this is drawn on top of the 2 sigma band. Set to zero if you only want 2 sigma band
        :param alpha2: alpha for the 2 sigma band. Set to zero if you only want 1 sigma band
        :param kwargs: optional keyword arguments for :func:`~matplotlib:matplotlib.pyplot.axvspan`

        .. plot::
           :include-source:
           
            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
            g = plots.getSinglePlotter(width_inch=4)
            g.plot_2d([samples1, samples2], ['x0','x1'], filled=False);
            g.add_x_bands(0, 1)
        """
        ax = ax or plt.gca()
        c = color
        if alpha2 > 0: ax.axvspan((x - sigma * 2), (x + sigma * 2), color=c, alpha=alpha2, **kwargs)
        if alpha1 > 0: ax.axvspan((x - sigma), (x + sigma), color=c, alpha=alpha1, **kwargs)

    def add_y_bands(self, y, sigma, color='gray', ax=None, alpha1=0.15, alpha2=0.1, **kwargs):
        """
        Adds horizontal shaded bands showing one and two sigma ranges.

        :param y: central y value for bands
        :param sigma: 1 sigma error on y
        :param color: The base color to use
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add the bands to (defaults to current plot)
        :param alpha1: alpha for the 1 sigma band; note this is drawn on top of the 2 sigma band. Set to zero if you only want 2 sigma band
        :param alpha2: alpha for the 2 sigma band. Set to zero if you only want 1 sigma band
        :param kwargs: optional keyword arguments for :func:`~matplotlib:matplotlib.pyplot.axhspan`

        .. plot::
           :include-source:
           
            from getdist import plots, gaussian_mixtures
            samples= gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=1)
            g = plots.getSinglePlotter(width_inch=4)
            g.plot_2d(samples, ['x0','x1'], filled=True);
            g.add_y_bands(0, 1)
        """
        ax = ax or plt.gca()
        c = color
        if alpha2 > 0: ax.axhspan((y - sigma * 2), (y + sigma * 2), color=c, alpha=alpha2, **kwargs)
        if alpha1 > 0: ax.axhspan((y - sigma), (y + sigma), color=c, alpha=alpha1, **kwargs)

    def _set_locator(self, axis, x=False, prune=None):
        """
        Set the locator for ticks

        :param axis: The axis instance
        :param x: True if x axis, False for y axis
        :param prune: Parameter for MaxNLocator constructor,  ['lower' | 'upper' | 'both' | None]
        """
        if x: xmin, xmax = axis.get_view_interval()
        if x and (abs(xmax - xmin) < 0.01 or max(abs(xmin), abs(xmax)) >= 1000):
            axis.set_major_locator(plt.MaxNLocator(self.settings.subplot_size_inch / 2 + 3, prune=prune))
        else:
            axis.set_major_locator(plt.MaxNLocator(self.settings.subplot_size_inch / 2 + 4, prune=prune))

    def _setAxisProperties(self, axis, x, prune=None):
        """
        Sets axis properties.

        :param axis: The axis to set properties to.
        :param x: True if x axis, False for y axis
        :param prune: Parameter for MaxNLocator constructor, ,  ['lower' | 'upper' | 'both' | None]
        """
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        axis.set_major_formatter(formatter)
        plt.tick_params(axis='both', which='major', labelsize=self.settings.axes_fontsize)
        if x and self.settings.x_label_rotation != 0: plt.setp(plt.xticks()[1], rotation=self.settings.x_label_rotation)
        self._set_locator(axis, x, prune=prune)

    def setAxes(self, params=[], lims=None, do_xlabel=True, do_ylabel=True, no_label_no_numbers=False, pos=None,
                prune=None, color_label_in_axes=False, ax=None, **other_args):
        """
        Set the axis labels and ticks, and various styles. Do not usually need to call this directly.

        :param params: [x,y] list of the :class:`~.paramnames.ParamInfo` for the x and y parameters to use for labels
        :param lims: optional [xmin, xmax, ymin, ymax] to fix specific limits for the axes
        :param do_xlabel: True if should include label for x axis.
        :param do_ylabel: True if should include label for y axis.
        :param no_label_no_numbers: True to hide tick labels
        :param pos: optional position of the axes ['left' | 'bottom' | 'width' | 'height']
        :param prune: Parameter for MaxNLocator constructor,  ['lower' | 'upper' | 'both' | None]
        :param color_label_in_axes: If True, and params has at last three entries, puts text in the axis to label the third parameter
        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, defaults to current axes.
        :param other_args: Not used, just quietly ignore so that setAxes can be passed general kwargs
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance
        """
        ax = ax or plt.gca()
        if lims is not None: ax.axis(lims)
        if prune is None: prune = self.settings.tick_prune
        self._setAxisProperties(ax.xaxis, True, prune)
        if pos is not None: ax.set_position(pos)
        if do_xlabel and len(params) > 0:
            self.set_xlabel(params[0])
        elif no_label_no_numbers:
            ax.set_xticklabels([])
        if len(params) > 1:
            self._setAxisProperties(ax.yaxis, False, prune)
            if do_ylabel:
                self.set_ylabel(params[1])
            elif no_label_no_numbers:
                ax.set_yticklabels([])
        if color_label_in_axes and len(params) > 2: self.add_text(params[2].latexLabel())
        return ax

    def set_xlabel(self, param, ax=None):
        """
        Sets the label for the x axis.

        :param param: the :class:`~.paramnames.ParamInfo` for the x axis parameter
        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, defaults to current axes.
        """
        ax = ax or plt.gca()
        ax.set_xlabel(param.latexLabel(), fontsize=self.settings.lab_fontsize,
                      verticalalignment='baseline',
                      labelpad=4 + self.settings.font_size)  # test_size because need a number not e.g. 'medium'

    def set_ylabel(self, param, ax=None):
        """
        Sets the label for the y axis.

        :param param: the :class:`~.paramnames.ParamInfo` for the y axis parameter
        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, defaults to current axes.
        """
        ax = ax or plt.gca()
        ax.set_ylabel(param.latexLabel(), fontsize=self.settings.lab_fontsize)

    def plot_1d(self, roots, param, marker=None, marker_color=None, label_right=False,
                no_ylabel=False, no_ytick=False, no_zero=False, normalized=False, param_renames={}, **kwargs):
        """
        Make a single 1D plot with marginalized density lines.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param param: the parameter name to plot
        :param marker: If set, places a marker at given coordinate.
        :param marker_color: If set, sets the marker color.
        :param label_right: If True, label the y axis on the right rather than the left
        :param no_ylabel: If True excludes the label on the y axis
        :param no_ytick: If True show no y ticks
        :param no_zero: If true does not show tick label at zero on y axis
        :param normalized: plot normalized densities (if False, densities normalized to peak at 1)
        :param param_renames: optional dictionary mapping input parameter names to equivalent names used by the samples
        :param kwargs: additional optional keyword arguments:

                * **lims**: optional limits for x range of the plot [xmin, xmax]
                * **ls** : list of line styles for the different lines plotted 
                * **colors**: list of colors for the different lines plotted 
                * **lws**: list of line widths for the different lines plotted
                * **alphas**: list of alphas for the different lines plotted 
                * **line_args**: a list of dictionaries with settings for each set of lines
                * arguments for :func:`~GetDistPlotter.setAxes`

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
            g = plots.getSinglePlotter(width_inch=4)
            g.plot_1d([samples1, samples2], 'x0', marker=0)

        .. plot::
           :include-source:

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
            g = plots.getSinglePlotter(width_inch=3)
            g.plot_1d([samples1, samples2], 'x0', normalized=True, colors=['green','black'])
        
        """
        roots = makeList(roots)
        if self.fig is None: self.make_figure()
        plotparam = None
        plotroot = None
        line_args = self._make_line_args(len(roots), **kwargs)
        xmin, xmax = None, None
        for i, root in enumerate(roots):
            root_param = self._check_param(root, param, param_renames)
            if not root_param: continue
            bounds = self.add_1d(root, root_param, i, normalized=normalized, **line_args[i])
            xmin, xmax = self._updateLimit(bounds, (xmin, xmax))
            if bounds is not None and not plotparam:
                plotparam = root_param
                plotroot = root
        if plotparam is None: raise GetDistPlotError('No roots have parameter: ' + str(param))
        if marker is not None: self.add_x_marker(marker, marker_color)
        if not 'lims' in kwargs:
            xmin, xmax = self._check_param_ranges(plotroot, plotparam.name, xmin, xmax)
            if normalized:
                mx = plt.gca().yaxis.get_view_interval()[-1]
            else:
                mx = 1.099
            kwargs['lims'] = [xmin, xmax, 0, mx]
        ax = self.setAxes([plotparam], **kwargs)

        if normalized:
            lab = self.settings.norm_prob_label
        else:
            lab = self.settings.prob_label
        if lab and not no_ylabel:
            if label_right:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_ylabel(lab)
            else:
                ax.set_ylabel(lab)
        if no_ytick or not self.settings.prob_y_ticks:
            ax.set_yticks([])
        elif no_ylabel:
            ax.set_yticklabels([])
        elif no_zero and not normalized:
            ticks = ax.get_yticks()
            if ticks[-1] > 1: ticks = ticks[:-1]
            ax.set_yticks(ticks[1:])

    def make_figure(self, nplot=1, nx=None, ny=None, xstretch=1.0, ystretch=1.0):
        """
        Makes a new figure.

        :param nplot: number of subplots
        :param nx: number of subplots in each row
        :param ny: number of subplots in each column
        :param xstretch: The parameter of how much to stretch the width, 1 is default
        :param ystretch: The parameter of how much to stretch the height, 1 is default
        :return: The plot_col, plot_row numbers of subplots for the figure
        """
        self.newPlot()
        if nx is None:
            self.plot_col = int(round(np.sqrt(nplot / 1.4)))
        else:
            self.plot_col = nx
        if ny is None:
            self.plot_row = (nplot + self.plot_col - 1) // self.plot_col
        else:
            self.plot_row = ny
        if self.settings.fig_width_inch is not None:
            self.fig = plt.figure(figsize=(self.settings.fig_width_inch,
                                           (self.settings.fig_width_inch * self.plot_row * ystretch) / (
                                               self.plot_col * xstretch)))
        else:
            self.fig = plt.figure(figsize=(self.settings.subplot_size_inch * self.plot_col * xstretch,
                                           self.settings.subplot_size_inch * self.plot_row * ystretch))
        self.subplots = np.ndarray((self.plot_row, self.plot_col), dtype=object)
        self.subplots[:, :] = None
        return self.plot_col, self.plot_row

    def get_param_array(self, root, params=None, renames={}):
        """
        Gets an array of :class:`~.paramnames.ParamInfo` for named params

        :param root: The root name of the samples to use
        :param params: the parameter names (if not specified, get all)
        :param renames: optional dictionary mapping input names and equivalent names used by the samples
        :return: list of :class:`~.paramnames.ParamInfo` instances for the parameters
        """
        if params is None or len(params) == 0:
            return self.paramNamesForRoot(root).names
        else:
            if isinstance(params, six.string_types) or \
                    not all([isinstance(param, ParamInfo) for param in params]):
                return self.paramNamesForRoot(root).parsWithNames(params, error=True, renames=renames)
        return params

    def _check_param(self, root, param, renames={}):
        """
        Get :class:`~.paramnames.ParamInfo` for given name for samples with specified root

        :param root: The root name of the samples
        :param param: The parameter name (or :class:`~.paramnames.ParamInfo`)
        :param renames: optional dictionary mapping input names and equivalent names used by the samples
        :return: a :class:`~.paramnames.ParamInfo` instance, or None if name not found
        """
        if not isinstance(param, ParamInfo):
            return self.paramNamesForRoot(root).parWithName(param, error=True, renames=renames)
        elif renames:
            return self.paramNamesForRoot(root).parWithName(param.name, error=False, renames=renames)
        return param

    def param_latex_label(self, root, name, labelParams=None):
        """
        Returns the latex label for given parameter.

        :param root: root name of the samples having the parameter (or :class:`~.mcsamples.MCSamples` instance)
        :param name:  The param name
        :param labelParams: optional name of .paramnames file to override parameter name labels
        :return: The latex label
        """
        if labelParams is not None:
            p = self.sampleAnalyser.paramsForRoot(root, labelParams=labelParams).parWithName(name)
        else:
            p = self._check_param(root, name)
        if not p: raise GetDistPlotError('Parameter not found: ' + name)
        return p.latexLabel()

    def add_legend(self, legend_labels, legend_loc=None, line_offset=0, legend_ncol=None, colored_text=False,
                   figure=False, ax=None, label_order=None, align_right=False, fontsize=None):
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
        :param ax: if figure == False, the :class:`~matplotlib:matplotlib.axes.Axes` instance to use; defaults to current axes. 
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific order of line indices 
        :param align_right: True to align legend text at the right
        :param fontsize: The size of the font, default from settings
        :return: a :class:`matplotlib:matplotlib.legend.Legend` instance
        """
        if legend_loc is None:
            if figure:
                legend_loc = self.settings.figure_legend_loc
            else:
                legend_loc = self.settings.legend_loc
        if legend_ncol is None: legend_ncol = self.settings.figure_legend_ncol
        lines = []
        if len(self.contours_added) == 0:
            for i in enumerate(legend_labels):
                args = self.lines_added.get(i[0]) or self._get_line_styles(i[0] + line_offset)
                args.pop('filled', None)
                lines.append(plt.Line2D([0, 1], [0, 1], **args))
        else:
            lines = self.contours_added
        args = {'ncol': legend_ncol}
        if fontsize or self.settings.legend_fontsize: args['prop'] = {'size': fontsize or self.settings.legend_fontsize}
        if colored_text:
            args['handlelength'] = 0
            args['handletextpad'] = 0
        if label_order is not None:
            if str(label_order) == '-1': label_order = list(range(len(lines))).reverse()
            lines = [lines[i] for i in label_order]
            legend_labels = [legend_labels[i] for i in label_order]
        if figure:
            # args['frameon'] = self.settings.figure_legend_frame
            self.legend = self.fig.legend(lines, legend_labels, loc=legend_loc, **args)
            if not self.settings.figure_legend_frame:
                # this works with tight_layout
                self.legend.get_frame().set_edgecolor('none')
        else:
            args['frameon'] = self.settings.legend_frame and not colored_text
            self.legend = (ax or plt.gca()).legend(lines, legend_labels, loc=legend_loc, **args)
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
                if isinstance(h, plt.Line2D):
                    c = h._get_color()
                elif isinstance(h, matplotlib.patches.Patch):
                    c = h.get_facecolor()
                else:
                    continue
                text.set_color(c)
        return self.legend

    def finish_plot(self, legend_labels=None, legend_loc=None, line_offset=0, legend_ncol=None, label_order=None,
                    no_gap=False, no_extra_legend_space=False, no_tight=False):
        """
        Finish the current plot, adjusting subplot spacing and adding legend

        :param legend_labels: The labels
        :param legend_loc: The legend location, default from settings
        :param line_offset: The offset of plotted lines to label (e.g. 1 to not label first line)
        :param legend_ncol: The number of columns in the legend, defaults to 1 
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific order of line indices 
        :param no_gap: True if should leave no subplot padding in tight_layout
        :param no_extra_legend_space: True to prevent making additional space above subplots for the legend
        :param no_tight: don't use :func:`~matplotlib:matplotlib.pyplot.tight_layout` to adjust subplot positions
        """
        has_legend = self.settings.line_labels and legend_labels and len(legend_labels) > 1
        if self.settings.tight_layout and not no_tight:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if no_gap:
                    plt.tight_layout(h_pad=0, w_pad=0)
                else:
                    plt.tight_layout()

        if has_legend:
            if legend_ncol is None: legend_ncol = self.settings.figure_legend_ncol
            if legend_loc is None: legend_loc = self.settings.figure_legend_loc
            self.extra_artists = [
                self.add_legend(legend_labels, legend_loc, line_offset, legend_ncol, label_order=label_order,
                                figure=True)]
            if self.settings.tight_layout and not no_extra_legend_space:
                nrows = len(legend_labels) // legend_ncol
                if self.settings.legend_position_config == 1:
                    frac = self.settings.legend_frac_subplot_margin + nrows * self.settings.legend_frac_subplot_line
                else:
                    frac = self.settings.legend_frac_subplot_margin + (
                                                                          nrows * self.settings.legend_fontsize * 0.015) / self.settings.subplot_size_inch
                if self.plot_row == 1: frac = min(frac, 0.5)
                if 'upper' in legend_loc:
                    plt.subplots_adjust(top=1 - frac / self.plot_row)
                elif 'lower' in legend_loc:
                    plt.subplots_adjust(bottom=frac / self.plot_row)

    def _escapeLatex(self, text):
        if matplotlib.rcParams['text.usetex']:
            return text.replace('_', '{\\textunderscore}')
        else:
            return text

    def _rootDisplayName(self, root, i):
        if isinstance(root, MCSamples):
            root = root.getName()
        if not root: root = 'samples' + str(i)
        return self._escapeLatex(root)

    def _default_legend_labels(self, legend_labels, roots):
        """
        Returns default legend labels, based on name tags of samples

        :param legend_labels: The current legend labels.
        :param roots: The root names of the samples
        :return: A list of labels
        """
        if legend_labels is None:
            return [self._rootDisplayName(root, i) for i, root in enumerate(roots) if root is not None]
        else:
            return legend_labels

    def plots_1d(self, roots, params=None, legend_labels=None, legend_ncol=None, label_order=None, nx=None,
                 paramList=None, roots_per_param=False, share_y=None, markers=None, xlims=None, param_renames={},
                 **kwargs):
        """
        Make an array of 1D marginalized density subplots 

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param params: list of names of parameters to plot
        :param legend_labels: list of legend labels
        :param legend_ncol: Number of columns for the legend.
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific order of line indices 
        :param nx: number of subplots per row 
        :param paramList: name of .paramnames file listing specific subset of parameters to plot
        :param roots_per_param: True to use a different set of samples for each parameter: 
                      plots param[i] using roots[i] (where roots[i] is the list of sample root names to use for plotting parameter i). 
                      This is useful for example for  plotting one-parameter extensions of a baseline model, each with various data combinations.
        :param share_y: True for subplots to share a common y axis with no horizontal space between subplots
        :param markers: optional dict giving vertical markers index by parameter, or a list of marker values for each parameter plotted
        :param xlims: list of [min,max] limits for the range of each parameter plot
        :param param_renames: optional dictionary holding mapping between input names and equivalent names used in the samples.
        :param kwargs: optional keyword arguments for :func:`~GetDistPlotter.plot_1d`
        :return: The plot_col, plot_row subplot dimensions of the new figure

        .. plot::
           :include-source: 

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.getSubplotPlotter()
            g.plots_1d([samples1, samples2], ['x0', 'x1', 'x2'], nx=3, share_y=True, legend_ncol =2,
                         markers={'x1':0}, colors=['red', 'green'], ls=['--', '-.'])

        """
        roots = makeList(roots)
        if roots_per_param:
            params = [self._check_param(root[0], param, param_renames) for root, param in zip(roots, params)]
        else:
            params = self.get_param_array(roots[0], params, param_renames)
        if paramList is not None:
            wantedParams = self._paramNameListFromFile(paramList)
            params = [param for param in params if
                      param.name in wantedParams or param_renames.get(param.name, '') in wantedParams]
        nparam = len(params)
        if share_y is None: share_y = self.settings.prob_label is not None and nparam > 1
        plot_col, plot_row = self.make_figure(nparam, nx=nx)
        plot_roots = roots
        for i, param in enumerate(params):
            ax = self._subplot_number(i)
            if roots_per_param: plot_roots = roots[i]
            marker = None
            if markers is not None:
                if isinstance(markers, dict):
                    marker = markers.get(param.name, None)
                elif i < len(markers):
                    marker = markers[i]
            self.plot_1d(plot_roots, param, no_ylabel=share_y and i % self.plot_col > 0, marker=marker,
                         param_renames=param_renames, **kwargs)
            if xlims is not None: ax.set_xlim(xlims[i][0], xlims[i][1])
            if share_y: self._spaceTicks(ax.xaxis, expand=True)
        self.finish_plot(self._default_legend_labels(legend_labels, roots), legend_ncol=legend_ncol,
                         label_order=label_order)
        if share_y: plt.subplots_adjust(wspace=0)
        return plot_col, plot_row

    def plots_2d(self, roots, param1=None, params2=None, param_pairs=None, nx=None, legend_labels=None,
                 legend_ncol=None, label_order=None, filled=False, shaded=False, **kwargs):
        """
        Make an array of 2D line, filled or contour plots.
        
        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of either of these) for the samples to plot
        :param param1: x parameter to plot
        :param params2: list of y parameters to plot against x
        :param param_pairs: list of [x,y] parameter pairs to plot; either specify param1, param2, or param_pairs
        :param nx: number of subplots per row
        :param legend_labels: The labels used for the legend.
        :param legend_ncol: The amount of columns in the legend.
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific order of line indices 
        :param filled: True to plot filled contours
        :param shaded: True to shade by the density for the first root plotted
        :param kwargs: optional keyword arguments for :func:`~GetDistPlotter.plot_2d`
        :return: The plot_col, plot_row subplot dimensions of the new figure
        
        .. plot::
           :include-source: 

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.getSubplotPlotter(subplot_size=4)
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
                    if param.name != param1.name: pairs.append((param1, param))
            else:
                raise GetDistPlotError('No parameter or parameter pairs for 2D plot')
        else:
            for pair in param_pairs:
                pairs.append((self._check_param(roots[0], pair[0]), self._check_param(roots[0], pair[1])))
        if filled and shaded:
            raise GetDistPlotError("Plots cannot be both filled and shaded")
        plot_col, plot_row = self.make_figure(len(pairs), nx=nx)

        for i, pair in enumerate(pairs):
            self._subplot_number(i)
            self.plot_2d(roots, param_pair=pair, filled=filled, shaded=not filled and shaded,
                         add_legend_proxy=i == 0, **kwargs)

        self.finish_plot(self._default_legend_labels(legend_labels, roots), legend_ncol=legend_ncol,
                         label_order=label_order)
        return plot_col, plot_row

    def _subplot(self, x, y, **kwargs):
        """
        Create a subplot with given parameters.

        :param x: x location in the subplot grid
        :param y: y location in the subplot grid
        :param kwargs: arguments for :func:`~matplotlib:matplotlib.pyplot.subplot`
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance for the subplot axes
        """
        self.subplots[y, x] = ax = plt.subplot(self.plot_row, self.plot_col, y * self.plot_col + x + 1, **kwargs)
        return ax

    def _subplot_number(self, i):
        """
        Create a subplot with given index.

        :param i: index of the subplot
        :return: an :class:`~matplotlib:matplotlib.axes.Axes` instance for the subplot axes
        """
        self.subplots[i // self.plot_col, i % self.plot_col] = ax = plt.subplot(self.plot_row, self.plot_col, i + 1)
        return ax

    def plots_2d_triplets(self, root_params_triplets, nx=None, filled=False, x_lim=None):
        """
        Creates an array of 2D plots, where each plot uses different samples, x and y parameters

        :param root_params_triplets: a list of (root, x, y) giving sample root names, and x and y parameter names to plot in each subplot
        :param nx: number of subplots per row
        :param filled:  True for filled contours
        :param x_lim: limits for all the x axes.
        :return: The plot_col, plot_row subplot dimensions of the new figure
        """
        plot_col, plot_row = self.make_figure(len(root_params_triplets), nx=nx)
        for i, (root, param1, param2) in enumerate(root_params_triplets):
            ax = self._subplot_number(i)
            self.plot_2d(root, param_pair=[param1, param2], filled=filled, add_legend_proxy=i == 0)
            if x_lim is not None: ax.set_xlim(x_lim)
        self.finish_plot()
        return plot_col, plot_row

    def _spaceTicks(self, axis, expand=True):
        """
        Space the axis ticks so there are none near the edges (which are likely to overlap on packed subplots)

        :param axis: axis instance
        :param expand: if True, increase axis range so existing ticks are safely not near edgel
                        otherwise remove end ticks
        :return: list of tick values
        """
        lims = axis.get_view_interval()
        tick = [x for x in axis.get_ticklocs() if lims[0] < x < lims[1]]
        gap_wanted = (lims[1] - lims[0]) * self.settings.tight_gap_fraction
        if expand:
            lims = [min(tick[0] - gap_wanted, lims[0]), max(tick[-1] + gap_wanted, lims[1])]
            axis.set_view_interval(lims[0], lims[1])
        else:
            if tick[0] - lims[0] < gap_wanted: tick = tick[1:]
            if lims[1] - tick[-1] < gap_wanted: tick = tick[:-1]
        axis.set_ticks(tick)
        return tick

    def triangle_plot(self, roots, params=None, legend_labels=None, plot_3d_with_param=None, filled=False, shaded=False,
                      contour_args=None, contour_colors=None, contour_ls=None, contour_lws=None, line_args=None,
                      label_order=None, legend_ncol=None, legend_loc=None, upper_roots=None, upper_kwargs={}, **kwargs):
        """
        Make a trianglular array of 1D and 2D plots. 
        
        A triangle plot is an array of subplots with 1D plots along the diagonal, and 2D plots in the lower corner.
        The upper triangle can also be used by setting upper_roots.

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param params: list of parameters to plot (default: all, can also use glob patterns to match groups of parameters)
        :param legend_labels: list of legend labels
        :param plot_3d_with_param: for the 2D plots, make sample scatter plot, with samples colored by this parameter name (to make a '3D' plot)
        :param filled: True for filled contours
        :param shaded: plot shaded density for first root (cannot be used with filled)
        :param contour_args: optional dict (or list of dict) with arguments for each 2D plot (e.g. specifying color, alpha,etc)
        :param contour_colors: list of colors for plotting contours (for each root)
        :param contour_ls: list of Line styles for contours (for each root)
        :param contour_lws: list of Line widths for contours (for each root)
        :param line_args: dict (or list of dict) with arguments for each 2D plot (e.g. specifying ls, lw, color, etc)
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific order of line indices 
        :param legend_ncol: The number of columns for the legend
        :param legend_loc: The location for the legend
        :param upper_roots: set to fill the upper triangle with subplots using this list of sample root names 
                             (TODO: this needs some work to easily work without a lot of tweaking)
        :param upper_kwargs: list of dict for arguments when making upper-triangle 2D plots
        :param kwargs: optional keyword arguments for :func:`~GetDistPlotter.plot_2d` or :func:`~GetDistPlotter.plot_3d` (lower triangle only)
        
        .. plot::
           :include-source: 

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.getSubplotPlotter()
            g.triangle_plot([samples1, samples2], filled=True, legend_labels = ['Contour 1', 'Contour 2'])

        .. plot::
           :include-source: 

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.getSubplotPlotter()
            g.triangle_plot([samples1, samples2], ['x0','x1','x2'], plot_3d_with_param='x3')

        """
        roots = makeList(roots)
        params = self.get_param_array(roots[0], params)
        plot_col = len(params)
        if plot_3d_with_param is not None: col_param = self._check_param(roots[0], plot_3d_with_param)
        self.make_figure(nx=plot_col, ny=plot_col)
        lims = dict()
        ticks = dict()
        filled = kwargs.get('filled_compare', filled)

        def defLineArgs(cont_args):
            cols = []
            for plotno, _arg in enumerate(cont_args):
                if not _arg.get('filled'):
                    if contour_colors is not None and len(contour_colors) > plotno:
                        cols.append(contour_colors[plotno])
                    else:
                        cols.append(None)
                else:
                    cols.append(_arg.get('color', None) or self.settings.solid_colors[len(cont_args) - plotno - 1])
            _line_args = []
            for col in cols:
                if col is None:
                    _line_args.append({})
                else:
                    if isinstance(col, (tuple, list)): col = col[-1]
                    _line_args += [{'color': col}]
            return _line_args

        contour_args = self._make_contour_args(len(roots), filled=filled, contour_args=contour_args,
                                               colors=contour_colors, ls=contour_ls, lws=contour_lws)
        if line_args is None:
            line_args = defLineArgs(contour_args)
        line_args = self._make_line_args(len(roots), line_args=line_args, ls=contour_ls, lws=contour_lws)
        roots1d = copy.copy(roots)
        if upper_roots is not None:
            if plot_3d_with_param is not None:
                logging.warning("triangle_plot currently doesn't fully work with plot_3d_with_param")
            upper_contour_args = self._make_contour_args(len(upper_roots), **upper_kwargs)
            args = upper_kwargs.copy()
            args['line_args'] = args.get('line_args') or defLineArgs(upper_contour_args)
            upargs = self._make_line_args(len(upper_roots), **args)
            for root, arg in zip(upper_roots, upargs):
                if not root in roots1d:
                    roots1d.append(root)
                    line_args.append(arg)

        for i, param in enumerate(params):
            ax = self._subplot(i, i)
            self.plot_1d(roots1d, param, do_xlabel=i == plot_col - 1,
                         no_label_no_numbers=self.settings.no_triangle_axis_labels,
                         label_right=True, no_zero=True, no_ylabel=True, no_ytick=True, line_args=line_args)
            # set no_ylabel=True for now, can't see how to not screw up spacing with right-sided y label
            if self.settings.no_triangle_axis_labels: self._spaceTicks(ax.xaxis)
            lims[i] = ax.get_xlim()
            ticks[i] = ax.get_xticks()
        for i, param in enumerate(params):
            for i2 in range(i + 1, len(params)):
                param2 = params[i2]
                pair = [param, param2]
                ax = self._subplot(i, i2)
                if plot_3d_with_param is not None:
                    self.plot_3d(roots, pair + [col_param], color_bar=False, line_offset=1, add_legend_proxy=False,
                                 do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0, contour_args=contour_args,
                                 no_label_no_numbers=self.settings.no_triangle_axis_labels, **kwargs)
                else:
                    self.plot_2d(roots, param_pair=pair, do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0,
                                 no_label_no_numbers=self.settings.no_triangle_axis_labels, shaded=shaded,
                                 add_legend_proxy=i == 0 and i2 == 1, contour_args=contour_args, **kwargs)
                ax.set_xticks(ticks[i])
                ax.set_yticks(ticks[i2])
                ax.set_xlim(lims[i])
                ax.set_ylim(lims[i2])

                if upper_roots is not None:
                    ax = self._subplot(i2, i)
                    pair.reverse()
                    if plot_3d_with_param is not None:
                        self.plot_3d(upper_roots, pair + [col_param], color_bar=False, line_offset=1,
                                     add_legend_proxy=False,
                                     do_xlabel=False, do_ylabel=False, contour_args=upper_contour_args,
                                     no_label_no_numbers=self.settings.no_triangle_axis_labels)
                    else:
                        self.plot_2d(upper_roots, param_pair=pair, do_xlabel=False, do_ylabel=False,
                                     no_label_no_numbers=self.settings.no_triangle_axis_labels, shaded=shaded,
                                     add_legend_proxy=i == 0 and i2 == 1,
                                     proxy_root_exclude=[root for root in upper_roots if root in roots],
                                     contour_args=upper_contour_args)
                    ax.set_xticks(ticks[i2])
                    ax.set_yticks(ticks[i])
                    ax.set_xlim(lims[i2])
                    ax.set_ylim(lims[i])

        if upper_roots is not None:
            # make label on first 1D plot appropriate for 2D plots in rest of row
            label_ax = self.subplots[0, 0].twinx()
            label_ax.yaxis.tick_left()
            label_ax.yaxis.set_label_position('left')
            label_ax.yaxis.set_offset_position('left')
            label_ax.set_ylim(lims[0])
            label_ax.set_yticks(ticks[0])
            self.set_ylabel(params[0], ax=label_ax)
            self._setAxisProperties(label_ax.yaxis, False)

        if self.settings.no_triangle_axis_labels: plt.subplots_adjust(wspace=0, hspace=0)
        if plot_3d_with_param is not None:
            bottom = 0.5
            if len(params) == 2: bottom += 0.1;
            cb = self.fig.colorbar(self.last_scatter, cax=self.fig.add_axes([0.9, bottom, 0.03, 0.35]))
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            self.add_colorbar_label(cb, col_param, label_rotation=-self.settings.colorbar_label_rotation)

        labels = self._default_legend_labels(legend_labels, roots1d)
        if not legend_loc and len(params) < 4 and upper_roots is None:
            legend_loc = 'upper right'
        self.finish_plot(labels, label_order=label_order,
                         legend_ncol=legend_ncol or (None if upper_roots is None else len(labels)),
                         legend_loc=legend_loc, no_gap=self.settings.no_triangle_axis_labels,
                         no_extra_legend_space=upper_roots is None)

    def rectangle_plot(self, xparams, yparams, yroots=None, roots=None, plot_roots=None, plot_texts=None,
                       xmarkers=None, ymarkers=None, marker_args={}, param_limits={},
                       legend_labels=None, legend_ncol=None, label_order=None, **kwargs):
        """
        Make a grid of 2D plots.
        
        A rectangle plot shows all x parameters plotted againts all y parameters in a grid of subplots with no spacing.
        
        Set roots to use the same set of roots for every plot in the rectangle, or set
        yroots (list of list of roots) to use different set of roots for each row of the plot; alternatively
        plot_roots allows you to specify explicitly (via list of list of list of roots) the set of roots for each individual subplot

        :param xparams: list of parameters for the x axes
        :param yparams: list of parameters for the y axes
        :param yroots: (list of list of roots) allows use of different set of root names for each row of the plot;
                       set either roots or yroots
        :param roots: list of root names or :class:`~.mcsamples.MCSamples` instances. 
                Uses the same set of roots for every plot in the rectangle; set either roots or yroots.
        :param plot_roots: Allows you to specify (via list of list of list of roots) the set of roots for each individual subplot
        :param plot_texts: a 2D array (or list of lists) of a text label to put in each subplot (use a None entry to skip one)
        :param xmarkers: list of markers for the x axis
        :param ymarkers: list of markers for the y axis
        :param marker_args: arguments for :func:`~GetDistPlotter.add_x_marker` and :func:`~GetDistPlotter.add_y_marker`
        :param param_limits: a dictionary holding a mapping from parameter names to axis limits for that parameter
        :param legend_labels: list of labels for the legend
        :param legend_ncol: The number of columns for the legend
        :param label_order: minus one to show legends in reverse order that lines were added, or a list giving specific order of line indices 
        :param kwargs: arguments for :func:`~GetDistPlotter.plot_2d`.
        :return: the 2D list of :class:`~matplotlib:matplotlib.axes.Axes` created

        .. plot::
           :include-source: 

            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=4, nMCSamples=2)
            g = plots.getSubplotPlotter()
            g.rectangle_plot(['x0','x1'], ['x2','x3'], roots = [samples1, samples2], filled=True)
        """
        self.make_figure(nx=len(xparams), ny=len(yparams))
        # f, plots = subplots(len(yparams), len(xparams), sharex='col', sharey='row')
        sharey = None
        yshares = []
        xshares = []
        ax_arr = []
        if plot_roots and yroots or roots and yroots or plot_roots and roots:
            raise GetDistPlotError('rectangle plot: must have one of roots, yroots, plot_roots')
        if roots: roots = makeList(roots)
        limits = dict()
        for x, xparam in enumerate(xparams):
            sharex = None
            if plot_roots:
                yroots = plot_roots[x]
            elif roots:
                yroots = [roots for _ in yparams]
            axarray = []
            for y, (yparam, subplot_roots) in enumerate(zip(yparams, yroots)):
                if x > 0: sharey = yshares[y]
                ax = self._subplot(x, y, sharex=sharex, sharey=sharey)
                if y == 0:
                    sharex = ax
                    xshares.append(ax)
                res = self.plot_2d(subplot_roots, param_pair=[xparam, yparam], do_xlabel=y == len(yparams) - 1,
                                   do_ylabel=x == 0, add_legend_proxy=x == 0 and y == 0, **kwargs)
                if ymarkers is not None and ymarkers[y] is not None: self.add_y_marker(ymarkers[y], **marker_args)
                if xmarkers is not None and xmarkers[x] is not None: self.add_x_marker(xmarkers[x], **marker_args)
                limits[xparam], limits[yparam] = self._updateLimits(res, limits.get(xparam), limits.get(yparam))
                if y != len(yparams) - 1: plt.setp(ax.get_xticklabels(), visible=False)
                if x != 0: plt.setp(ax.get_yticklabels(), visible=False)
                if x == 0: yshares.append(ax)
                if plot_texts and plot_texts[x][y]:
                    self.add_text_left(plot_texts[x][y], y=0.9, ax=ax)
                axarray.append(ax)
            ax_arr.append(axarray)
        for xparam, ax in zip(xparams, xshares):
            ax.set_xlim(param_limits.get(xparam, limits[xparam]))
            self._spaceTicks(ax.xaxis)
            ax.set_xlim(ax.xaxis.get_view_interval())
        for yparam, ax in zip(yparams, yshares):
            ax.set_ylim(param_limits.get(yparam, limits[yparam]))
            self._spaceTicks(ax.yaxis)
            ax.set_ylim(ax.yaxis.get_view_interval())
        plt.subplots_adjust(wspace=0, hspace=0)
        if roots: legend_labels = self._default_legend_labels(legend_labels, roots)
        self.finish_plot(no_gap=True, legend_labels=legend_labels, label_order=label_order,
                         legend_ncol=legend_ncol or len(legend_labels))
        return ax_arr

    def rotate_yticklabels(self, ax=None, rotation=90):
        """
        Rotates the y-tick labels by given rotation (degrees)

        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, defaults to current axes.
        :param rotation: How much to rotate in degrees.
        """
        if ax is None: ax = plt.gca()
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_rotation(rotation)

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
        cb = plt.colorbar(mappable, orientation=orientation, ax=ax)
        cb.set_alpha(1)
        cb.draw_all()
        if not ax_args.get('color_label_in_axes'):
            self.add_colorbar_label(cb, param)
            if self.settings.colorbar_rotation is not None:
                self.rotate_yticklabels(cb.ax, self.settings.colorbar_rotation)
                labels = [label.get_text() for label in cb.ax.yaxis.get_ticklabels()[::2]]
                cb.ax.yaxis.set_ticks(cb.ax.yaxis.get_ticklocs()[::2])
                cb.ax.yaxis.set_ticklabels(labels)
        return cb

    def add_line(self, xdata, ydata, zorder=0, color=None, ls=None, ax=None, **kwargs):
        """
        Adds a line to the given axes, using :class:`~matplotlib:matplotlib.lines.Line2D`

        :param xdata: pair of x coordinates
        :param ydata: pair of y coordinates
        :param zorder: Z-order for Line2D
        :param color: The color of the line, uses settings.axis_marker_color by default
        :param ls: The line style to be used, uses settings.axis_marker_ls by default
        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, defaults to current axes
        :param kwargs:  Additional arguments for :class:`~matplotlib:matplotlib.lines.Line2D`
        """
        if color is None: color = self.settings.axis_marker_color
        if ls is None: ls = self.settings.axis_marker_ls
        (ax or plt.gca()).add_line(plt.Line2D(xdata, ydata, color=color, ls=ls, zorder=zorder, **kwargs))

    def add_colorbar_label(self, cb, param, label_rotation=None):
        """
        Adds a color bar label.

        :param cb: a :class:`~matplotlib:matplotlib.colorbar.Colorbar` instance
        :param param: a :class:`~.paramnames.ParamInfo` with label for the plotted parameter
        :param label_rotation: If set rotates the label (degrees)
        """
        if label_rotation is None:
            label_rotation = self.settings.colorbar_label_rotation
        cb.set_label(param.latexLabel(), fontsize=self.settings.lab_fontsize,
                     rotation=label_rotation, labelpad=self.settings.colorbar_label_pad)
        plt.setp(plt.getp(cb.ax, 'ymajorticklabels'), fontsize=self.settings.colorbar_axes_fontsize)

    def _makeParamObject(self, names, samples):
        class sampleNames(object): pass

        p = sampleNames()
        for i, par in enumerate(names.names):
            setattr(p, par.name, samples[:, i])
        return p

    def add_2d_scatter(self, root, x, y, color='k', alpha=1, extra_thin=1, scatter_size=None, ax=None):
        """
        Low-level function to adds a 2D sample scatter plot to the current axes (or ax if specified).

        :param root: The root name of the samples to use
        :param param1: name of x parameter
        :param param2: name of y parameter
        :param color: color to plot the samples
        :param alpha: The alpha to use.
        :param extra_thin: thin the weight one samples by this additional factor before plotting
        :param scatter_size: point size (default: settings.scatter_size)
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :return: (xmin, xmax), (ymin, ymax) bounds for the axes.
        """

        kwargs = {'fixed_color':color}
        return self.add_3d_scatter(root, [x, y], False, alpha, extra_thin, scatter_size, ax, **kwargs)

    def add_3d_scatter(self, root, params, color_bar=True, alpha=1, extra_thin=1, scatter_size=None, ax=None, **kwargs):
        """
        Low-level function to add a 3D scatter plot to the current axes (or ax if specified).

        :param root: The root name of the samples to use
        :param params:  list of parameters to plot
        :param color_bar: True to add a colorbar for the plotted scatter color
        :param alpha: The alpha to use.
        :param extra_thin: thin the weight one samples by this additional factor before plotting
        :param scatter_size: point size (default: settings.scatter_size)
        :param ax: optional :class:`~matplotlib:matplotlib.axes.Axes` instance to add to (defaults to current plot)
        :param kwargs: arguments for :func:`~GetDistPlotter.add_colorbar`
        :return: (xmin, xmax), (ymin, ymax) bounds for the axes.
        """
        params = self.get_param_array(root, params)
        pts = self.sampleAnalyser.load_single_samples(root)
        names = self.paramNamesForRoot(root)
        fixed_color = kwargs.get('fixed_color')  # if actually just a plain scatter plot
        samples = []
        for param in params:
            if hasattr(param, 'getDerived'):
                samples.append(param.getDerived(self._makeParamObject(names, pts)))
            else:
                samples.append(pts[:, names.numberOfName(param.name)])
        if extra_thin > 1:
            samples = [pts[::extra_thin] for pts in samples]
        self.last_scatter = (ax or plt.gca()).scatter(samples[0], samples[1], edgecolors='none',
                                        s=scatter_size or self.settings.scatter_size, c=fixed_color or samples[2],
                                        cmap=self.settings.colormap_scatter, alpha=alpha)
        if not ax: plt.sci(self.last_scatter)
        if color_bar and not fixed_color: self.last_colorbar = self.add_colorbar(params[2], mappable=self.last_scatter, ax=ax, **kwargs)
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

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
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
        self.plot_3d(roots, [param1, param2], False, line_offset, add_legend_proxy, **kwargs)

    def plot_3d(self, roots, params=None, params_for_plots=None, color_bar=True, line_offset=0,
                add_legend_proxy=True, **kwargs):
        """
        Make a 2D scatter plot colored by the value of a third parameter (a 3D plot).
        
        If roots is a list of more than one, additional densities are plotted as contour lines. 

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param params: list with the three parameter names to plot (x, y, color)
        :param params_for_plots: list of parameter triplets to plot for each root plotted; more general alternative to params
        :param color_bar: True if should include a color bar
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
                * arguments for :func:`~GetDistPlotter.add_colorbar`

        .. plot::
           :include-source:
           
            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=3, nMCSamples=2)
            g = plots.getSinglePlotter(width_inch=4)
            g.plot_3d([samples1, samples2], ['x0','x1','x2']);
        """
        roots = makeList(roots)
        if params_for_plots:
            if params is not None: raise GetDistPlotError('plot_3d uses either params OR params_for_plots')
            params_for_plots = [self.get_param_array(root, p) for p, root in zip(params_for_plots, roots)]
        else:
            if not params: raise GetDistPlotError('No parameters for plot_3d!')
            params = self.get_param_array(roots[0], params)
            params_for_plots = [params for _ in roots]  # all the same
        if self.fig is None: self.make_figure()
        if kwargs.get('filled_compare') is not None:
            kwargs = kwargs.copy()
            kwargs['filled'] = kwargs['filled_compare']
        contour_args = self._make_contour_args(len(roots) - 1, **kwargs)
        xlims, ylims = self.add_3d_scatter(roots[0], params_for_plots[0], color_bar=color_bar, **kwargs)
        for i, root in enumerate(roots[1:]):
            params = params_for_plots[i + 1]
            res = self.add_2d_contours(root, params[0], params[1], i + line_offset, add_legend_proxy=add_legend_proxy,
                                       zorder=i + 1, **contour_args[i])
            xlims, ylims = self._updateLimits(res, xlims, ylims)
        if not 'lims' in kwargs:
            params = params_for_plots[0]
            lim1 = self._check_param_ranges(roots[0], params[0].name, xlims[0], xlims[1])
            lim2 = self._check_param_ranges(roots[0], params[1].name, ylims[0], ylims[1])
            kwargs['lims'] = [lim1[0], lim1[1], lim2[0], lim2[1]]
        self.setAxes(params, **kwargs)

    def plots_3d(self, roots, param_sets, nx=None, legend_labels=None, **kwargs):
        """
        Create multiple 3D subplots 

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param param_sets: A list of triplets of parameter names to plot [(x,y, color), (x2,y2,color2)..]
        :param nx: number of subplots per row
        :param legend_labels: list of legend labels
        :param kwargs: keyword arguments for  :func:`~GetDistPlotter.plot_3d`
        :return: The plot_col, plot_row subplot dimensions of the new figure
        
        .. plot::
           :include-source:
           
            from getdist import plots, gaussian_mixtures
            samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=5, nMCSamples=2)
            g = plots.getSubplotPlotter(subplot_size=4)
            g.plots_3d([samples1, samples2], [['x0', 'x1', 'x2'], ['x3', 'x4', 'x2']], nx=2);
        """
        roots = makeList(roots)
        sets = [[self._check_param(roots[0], param) for param in param_group] for param_group in param_sets]
        plot_col, plot_row = self.make_figure(len(sets), nx=nx, xstretch=1.3)

        for i, triplet in enumerate(sets):
            self._subplot_number(i)
            self.plot_3d(roots, triplet, **kwargs)
        self.finish_plot(self._default_legend_labels(legend_labels, roots[1:]), no_tight=True)
        return plot_col, plot_row

    def plots_3d_z(self, roots, param_x, param_y, param_z=None, max_z=None, **kwargs):
        """
        Make set of sample scatter subplots of param_x against param_y, each coloured by values of parameters in param_z (all if None).
        Any second or more samples in root are shown as contours

        :param roots: root name or :class:`~.mcsamples.MCSamples` instance (or list of any of either of these) for the samples to plot
        :param param_x: x parameter name
        :param param_y: y parameter name
        :param param_z: list of parameter to names to color samples in each subplot (default: all)
        :param max_z: The maximum number of z parameters we should use.
        :param kwargs: keyword arguments for :func:`~GetDistPlotter.plot_3d`
        :return: The plot_col, plot_row subplot dimensions of the new figure
        """
        roots = makeList(roots)
        param_z = self.get_param_array(roots[0], param_z)
        if max_z is not None and len(param_z) > max_z: param_z = param_z[:max_z]
        param_x, param_y = self.get_param_array(roots[0], [param_x, param_y])
        sets = [[param_x, param_y, z] for z in param_z if z != param_x and z != param_y]
        return self.plots_3d(roots, sets, **kwargs)

    def add_text(self, text_label, x=0.95, y=0.06, ax=None, **kwargs):
        """
        Add text to given axis.

        :param text_label: The label to add.
        :param x: The x coordinate of where to add the label
        :param y: The y coordinate of where to add the label.
        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, 
                   index or [x,y] coordinate of subplot to use, or default to current axes.
        :param kwargs: keyword arguments for :func:`~matplotlib:matplotlib.pyplot.text` 
        """
        args = {'horizontalalignment': 'right', 'verticalalignment': 'center'}
        args.update(kwargs)
        if isinstance(ax, int):
            ax = self.fig.axes[ax]
        if isinstance(ax, (list, tuple)):
            ax = self.subplots[ax[0], ax[1]]
        else:
            ax = ax or plt.gca()
        ax.text(x, y, text_label, transform=ax.transAxes, **args)

    def add_text_left(self, text_label, x=0.05, y=0.06, ax=None, **kwargs):
        """
        Add text to the left, Wraps add_text.

        :param text_label: The label to add.
        :param x: The x coordinate of where to add the label
        :param y: The y coordinate of where to add the label.
        :param ax: the :class:`~matplotlib:matplotlib.axes.Axes` instance to use, defaults to current axes.
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
        if fname is None: fname = os.path.basename(sys.argv[0]).replace('.py', '')
        if tag: fname += '_' + tag
        if not '.' in fname: fname += '.' + getdist.default_plot_output
        if adir is not None and not os.sep in fname: fname = os.path.join(adir, fname)
        adir = os.path.dirname(fname)
        if adir and not os.path.exists(adir): os.makedirs(adir)
        if watermark:
            self.fig.text(0.45, 0.5, self._escapeLatex(watermark), fontsize=30, color='gray', ha='center', va='center',
                          alpha=0.2)

        self.fig.savefig(fname, bbox_extra_artists=self.extra_artists, bbox_inches='tight')

    def _paramNameListFromFile(self, fname):
        """
        Reads param names for a file.

        :param fname: The file to read
        :return: A list of param names
        """
        p = ParamNames(fname)
        return [name.name for name in p.names]
