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
    if isinstance(roots, (list, tuple)):
        return roots
    else:
        return [roots]


class GetDistPlotError(Exception):
    pass


class GetDistPlotSettings(object):
    """Default sizes, font, styles etc settings for use by plots"""

    def __init__(self, subplot_size_inch=2, fig_width_inch=None):
        # if fig_width_inch set, forces fixed size, subplot_size_inch then just determines font sizes etc
        # otherwise width as wide as necessary to show all subplots at specified size

        self.plot_meanlikes = False
        self.shade_meanlikes = False
        self.prob_label = None
        self.norm_prob_label = 'P'
        self.prob_y_ticks = False
        # self.prob_label = 'Probability'
        self.lineM = ['-k', '-r', '-b', '-g', '-m', '-c', '-y', '--k', '--r', '--b', '--g', '--m']
        self.plot_args = None
        self.solid_colors = ['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm',
                             'r']  # '#66CC99'
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
    return GetDistPlotter(**kwargs)


def getSinglePlotter(ratio=3 / 4., width_inch=6, **kwargs):
    """ 
    Wrapper functions to get plotter to make single plot of fixed width (default: one page column width)
    """
    plotter = getPlotter(**kwargs)
    plotter.settings.setWithSubplotSize(width_inch)
    plotter.settings.fig_width_inch = width_inch
    # if settings is None: plotter.settings.rcSizes()
    plotter.make_figure(1, xstretch=1. / ratio)
    return plotter


def getSubplotPlotter(subplot_size=2, width_inch=None, **kwargs):
    """ 
    Wrapper functions to get plotter to make array of subplots 
    if width_inch is None, just makes plot as big as needed
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
    def __init__(self, root, path, batch=None):
        self.root = root
        self.batch = batch
        self.path = path


class MCSampleAnalysis(object):
    def __init__(self, chain_locations, settings=None):
        """chain_locations is either a directory or the path of a grid of runs;
           it can also be a list of such, which is searched in order"""
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
            # yuk, should get rid of this next refactor when grids should store this information
            if os.path.exists(batch.commonPath + 'getdist_common.ini'):
                batchini = IniFile(batch.commonPath + 'getdist_common.ini')
                if self.ini:
                    self.ini.params.update(batchini.params)
                else:
                    self.ini = batchini
        else:
            self.chain_dirs.append(chain_dir)

    def reset(self, settings=None):
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

        self.mcsamples[root] = loadMCSamples(file_root, self.ini, jobItem, dist_settings=dist_settings)
        return self.mcsamples[root]

    def addRootGrid(self, root):
        return self.samplesForRoot(root)

    def addRoots(self, roots):
        for root in roots:
            self.addRoot(root)

    def addRoot(self, file_root):
        if isinstance(file_root, RootInfo):
            if file_root.batch:
                return self.samplesForRoot(file_root.root)
            else:
                return self.samplesForRoot(file_root.root, os.path.join(file_root.path, file_root.root))
        else:
            return self.samplesForRoot(os.path.basename(file_root), file_root)

    def removeRoot(self, file_root):
        root = os.path.basename(file_root)
        self.mcsamples.pop(root, None)
        self.single_samples.pop(root, None)
        self.densities_1D.pop(root, None)
        self.densities_2D.pop(root, None)

    def newPlot(self):
        pass

    def get_density(self, root, param, likes=False):
        rootdata = self.densities_1D.get(root)
        if rootdata is None:
            rootdata = {}
            self.densities_1D[root] = rootdata

        name = param.name
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
        if not root in self.single_samples:
            self.single_samples[root] = self.samplesForRoot(root).makeSingleSamples()
        return self.single_samples[root]

    def paramsForRoot(self, root, labelParams=None):
        samples = self.samplesForRoot(root)
        names = samples.paramNames
        if labelParams is not None:
            names.setLabelsAndDerivedFromParamNames(os.path.join(batchjob.getCodeRootPath(), labelParams))
        return names

    def boundsForRoot(self, root):
        return self.samplesForRoot(root)  # #defines getUpper and getLower, all that's needed


class GetDistPlotter(object):
    def __init__(self, plot_data=None, chain_dir=None, settings=None, analysis_settings=None, mcsamples=True):
        """
        Set plot_data to directory name if you have pre-computed plot_data/ directory from GetDist
        Set chain_dir to directly to use chains in the given directory (can also be a list of directories to search)
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
        print('Python version:', sys.version)
        print('\nMatplotlib version:', matplotlib.__version__)
        print('\nGetDist Plot Settings:')
        sets = self.settings.__dict__
        for key, value in list(sets.items()):
            print(key, ':', value)
        print('\nRC params:')
        for key, value in list(matplotlib.rcParams.items()):
            print(key, ':', value)

    def get_plot_args(self, plotno, **kwargs):
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

    def get_dashes_for_ls(self, ls):
        return self.settings.default_dash_styles.get(ls, None)

    def get_default_ls(self, plotno=0):
        try:
            return self.settings.lineM[plotno]
        except IndexError:
            print('Error adding line ' + str(plotno) + ': Add more default line stype entries to settings.lineM')
            raise

    def get_line_styles(self, plotno, **kwargs):
        args = self.get_plot_args(plotno, **kwargs)
        if not 'ls' in args: args['ls'] = self.get_default_ls(plotno)[:-1]
        if not 'dashes' in args:
            dashes = self.get_dashes_for_ls(args['ls'])
            if dashes is not None: args['dashes'] = dashes
        if not 'color' in args:
            args['color'] = self.get_default_ls(plotno)[-1]
        if not 'lw' in args: args['lw'] = self.settings.lw1
        return args

    def get_color(self, plotno, **kwargs):
        return self.get_line_styles(plotno, **kwargs)['color']

    def get_linestyle(self, plotno, **kwargs):
        return self.get_line_styles(plotno, **kwargs)['ls']

    def get_alpha2D(self, plotno, **kwargs):
        args = self.get_plot_args(plotno, **kwargs)
        if kwargs.get('filled') and plotno > 0:
            default = self.settings.alpha_filled_add
        else:
            default = 1
        return args.get('alpha', default)

    def paramNamesForRoot(self, root):
        if not root in self.param_name_sets: self.param_name_sets[root] = self.sampleAnalyser.paramsForRoot(root,
                                                                                                            labelParams=self.settings.param_names_for_labels)
        return self.param_name_sets[root]

    def paramBoundsForRoot(self, root):
        if not root in self.param_bounds_sets: self.param_bounds_sets[root] = self.sampleAnalyser.boundsForRoot(root)
        return self.param_bounds_sets[root]

    def checkBounds(self, root, name, xmin, xmax):
        d = self.paramBoundsForRoot(root)
        low = d.getLower(name)
        if low is not None: xmin = max(xmin, low)
        up = d.getUpper(name)
        if up is not None: xmax = min(xmax, up)
        return xmin, xmax

    def add_1d(self, root, param, plotno=0, normalized=False, **kwargs):
        param = self.check_param(root, param)
        density = self.sampleAnalyser.get_density(root, param, likes=self.settings.plot_meanlikes)
        if density is None: return None;
        if normalized: density.normalize()

        kwargs = self.get_line_styles(plotno, **kwargs)
        self.lines_added[plotno] = kwargs
        l, = plt.plot(density.x, density.P, **kwargs)
        if kwargs.get('dashes'):
            l.set_dashes(kwargs['dashes'])
        if self.settings.plot_meanlikes:
            kwargs['lw'] = self.settings.lw_likes
            plt.plot(density.x, density.likes, **kwargs)

        return density.bounds()

    def add_2d_density_contours(self, density, **kwargs):
        return self.add_2d_contours(None, density=density, **kwargs)

    def add_2d_contours(self, root, param1=None, param2=None, plotno=0, of=None, cols=None, contour_levels=None,
                        add_legend_proxy=True, param_pair=None, density=None, alpha=None, **kwargs):
        if density is None:
            param1, param2 = self.get_param_array(root, param_pair or [param1, param2])

            density = self.sampleAnalyser.get_density_grid(root, param1, param2,
                                                           conts=self.settings.num_plot_contours,
                                                           likes=self.settings.shade_meanlikes)
            if density is None:
                if add_legend_proxy: self.contours_added.append(None)
                return None
        if alpha is None: alpha = self.get_alpha2D(plotno, **kwargs)
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
                        cols = [[c * (
                            1 - self.settings.solid_contour_palefactor) + self.settings.solid_contour_palefactor for c
                                 in
                                 cols[0]]] + cols
                else:
                    cols = color
            levels = sorted(np.append([density.P.max() + 1], contour_levels))
            CS = plt.contourf(density.x, density.y, density.P, levels, colors=cols, alpha=alpha, **kwargs)
            if proxyIx >= 0: self.contours_added[proxyIx] = (plt.Rectangle((0, 0), 1, 1, fc=CS.tcolors[-1][0]))
            plt.contour(density.x, density.y, density.P, levels[:1], colors=CS.tcolors[-1],
                        linewidths=self.settings.lw_contour, alpha=alpha * self.settings.alpha_factor_contour_lines,
                        **kwargs)
        else:
            args = self.get_line_styles(plotno, **kwargs)
            # if color is None: color = self.get_color(plotno, **kwargs)
            # cols = [color]
            #            if ls is None: ls = self.get_linestyle(plotno, **kwargs)
            linestyles = [args['ls']]
            cols = [args['color']]
            kwargs = self.get_plot_args(plotno, **kwargs)
            kwargs['alpha'] = alpha
            CS = plt.contour(density.x, density.y, density.P, contour_levels, colors=cols, linestyles=linestyles,
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

    def add_2d_shading(self, root, param1, param2, colormap=None, density=None):
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
        plt.contourf(density.x, density.y, points, self.settings.num_shades, colors=cols, levels=levels)
        # doing contourf gets rid of annoying white lines in pdfs
        plt.contour(density.x, density.y, points, self.settings.num_shades, colors=cols, levels=levels)

    def updateLimit(self, bounds, curbounds):
        if not bounds: return curbounds
        if curbounds is None or curbounds[0] is None: return bounds
        return min(curbounds[0], bounds[0]), max(curbounds[1], bounds[1])

    def updateLimits(self, res, xlims, ylims, doResize=True):
        if res is None: return xlims, ylims
        if xlims is None and ylims is None: return res
        if not doResize:
            return xlims, ylims
        else:
            return self.updateLimit(res[0], xlims), self.updateLimit(res[1], ylims)

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
            xbounds, ybounds = self.updateLimits(res, xbounds, ybounds)
        if xbounds is None: return
        if not 'lims' in kwargs:
            lim1 = self.checkBounds(roots[0], param_pair[0].name, xbounds[0], xbounds[1])
            lim2 = self.checkBounds(roots[0], param_pair[1].name, ybounds[0], ybounds[1])
            kwargs['lims'] = [lim1[0], lim1[1], lim2[0], lim2[1]]

        self.setAxes(param_pair, **kwargs)
        return xbounds, ybounds

    def add_1d_marker(self, marker, color=None, ls=None):
        self.add_x_marker(marker, color, ls)

    def add_x_marker(self, marker, color=None, ls=None, lw=None):
        if color is None: color = self.settings.axis_marker_color
        if ls is None: ls = self.settings.axis_marker_ls
        if lw is None: lw = self.settings.axis_marker_lw
        plt.axvline(marker, ls=ls, color=color, lw=lw)

    def add_y_marker(self, marker, color=None, ls=None, lw=None):
        if color is None: color = self.settings.axis_marker_color
        if ls is None: ls = self.settings.axis_marker_ls
        if lw is None: lw = self.settings.axis_marker_lw
        plt.axhline(marker, ls=ls, color=color, lw=lw)

    def add_y_bands(self, y, sigma, xlim=None, color='gray', ax=None, alpha1=0.15, alpha2=0.1):
        ax = ax or plt.gca()
        if xlim is None: xlim = ax.xaxis.get_view_interval()
        one = np.array([1, 1])
        c = color
        if alpha2 > 0: ax.fill_between(xlim, one * (y - sigma * 2), one * (y + sigma * 2), facecolor=c, alpha=alpha2,
                                       edgecolor=c, lw=0)
        if alpha1 > 0: ax.fill_between(xlim, one * (y - sigma), one * (y + sigma), facecolor=c, alpha=alpha1,
                                       edgecolor=c, lw=0)

    def set_locator(self, axis, x=False, prune=None):
        if x: xmin, xmax = axis.get_view_interval()
        if x and (abs(xmax - xmin) < 0.01 or max(abs(xmin), abs(xmax)) >= 1000):
            axis.set_major_locator(plt.MaxNLocator(self.settings.subplot_size_inch / 2 + 3, prune=prune))
        else:
            axis.set_major_locator(plt.MaxNLocator(self.settings.subplot_size_inch / 2 + 4, prune=prune))

    def setAxisProperties(self, axis, x, prune=None):
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        axis.set_major_formatter(formatter)
        plt.tick_params(axis='both', which='major', labelsize=self.settings.axes_fontsize)
        if x and self.settings.x_label_rotation != 0: plt.setp(plt.xticks()[1], rotation=self.settings.x_label_rotation)
        self.set_locator(axis, x, prune=prune)

    def setAxes(self, params=[], lims=None, do_xlabel=True, do_ylabel=True, no_label_no_numbers=False, pos=None,
                prune=None,
                color_label_in_axes=False, ax=None, **other_args):
        ax = ax or plt.gca()
        if lims is not None: ax.axis(lims)
        if prune is None: prune = self.settings.tick_prune
        self.setAxisProperties(ax.xaxis, True, prune)
        if pos is not None: ax.set_position(pos)  # # set [left, bottom, width, height] for the figure
        if do_xlabel and len(params) > 0:
            self.set_xlabel(params[0])
        elif no_label_no_numbers:
            ax.set_xticklabels([])
        if len(params) > 1:
            self.setAxisProperties(ax.yaxis, False, prune)
            if do_ylabel:
                self.set_ylabel(params[1])
            elif no_label_no_numbers:
                ax.set_yticklabels([])
        if color_label_in_axes and len(params) > 2: self.add_text(params[2].latexLabel())
        return ax

    def set_xlabel(self, param, ax=None):
        ax = ax or plt.gca()
        ax.set_xlabel(param.latexLabel(), fontsize=self.settings.lab_fontsize,
                      verticalalignment='baseline',
                      labelpad=4 + self.settings.font_size)  # test_size because need a number not e.g. 'medium'

    def set_ylabel(self, param, ax=None):
        ax = ax or plt.gca()
        ax.set_ylabel(param.latexLabel(), fontsize=self.settings.lab_fontsize)

    def plot_1d(self, roots, param, marker=None, marker_color=None, label_right=False,
                no_ylabel=False, no_ytick=False, no_zero=False, normalized=False, param_renames={}, **kwargs):
        roots = makeList(roots)
        if self.fig is None: self.make_figure()
        plotparam = None
        plotroot = None
        line_args = self._make_line_args(len(roots), **kwargs)
        xmin, xmax = None, None
        for i, root in enumerate(roots):
            root_param = self.check_param(root, param, param_renames)
            if not root_param: continue
            bounds = self.add_1d(root, root_param, i, normalized=normalized, **line_args[i])
            xmin, xmax = self.updateLimit(bounds, (xmin, xmax))
            if bounds is not None and not plotparam:
                plotparam = root_param
                plotroot = root
        if plotparam is None: raise GetDistPlotError('No roots have parameter: ' + str(param))
        if marker is not None: self.add_x_marker(marker, marker_color)
        if not 'lims' in kwargs:
            xmin, xmax = self.checkBounds(plotroot, plotparam.name, xmin, xmax)
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

    def get_param_array(self, root, in_params=None, renames={}):
        if in_params is None or len(in_params) == 0:
            return self.paramNamesForRoot(root).names
        else:
            if not all([isinstance(param, ParamInfo) for param in in_params]):
                return self.paramNamesForRoot(root).parsWithNames(in_params, error=True, renames=renames)
        return in_params

    def check_param(self, root, param, renames={}):
        if not isinstance(param, ParamInfo):
            return self.paramNamesForRoot(root).parWithName(param, error=True, renames=renames)
        elif renames:
            return self.paramNamesForRoot(root).parWithName(param.name, error=False, renames=renames)
        return param

    def param_latex_label(self, root, param, labelParams=None):
        if labelParams is not None:
            p = self.sampleAnalyser.paramsForRoot(root, labelParams=labelParams).parWithName(param)
        else:
            p = self.check_param(root, param)
        if not p: raise GetDistPlotError('Parameter not found: ' + param)
        return p.latexLabel()

    def add_legend(self, legend_labels, legend_loc=None, line_offset=0, legend_ncol=None, colored_text=False,
                   figure=False, ax=None, label_order=None, align_right=False, fontsize=None):
        if legend_loc is None:
            if figure:
                legend_loc = self.settings.figure_legend_loc
            else:
                legend_loc = self.settings.legend_loc
        if legend_ncol is None: legend_ncol = self.settings.figure_legend_ncol
        lines = []
        if len(self.contours_added) == 0:
            for i in enumerate(legend_labels):
                args = self.lines_added.get(i[0]) or self.get_line_styles(i[0] + line_offset)
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
            if label_order == '-1': label_order = list(range(len(lines))).reverse()
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
                    c = h.get_color()
                elif isinstance(h, matplotlib.patches.Patch):
                    c = h.get_facecolor()
                else:
                    continue
                text.set_color(c)
        return self.legend

    def finish_plot(self, legend_labels=None, legend_loc=None, line_offset=0, legend_ncol=None, label_order=None,
                    no_gap=False, no_extra_legend_space=False, no_tight=False):
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

    def default_legend_labels(self, legend_labels, roots):
        if legend_labels is None:
            return [self._rootDisplayName(root, i) for i, root in enumerate(roots) if root is not None]
        else:
            return legend_labels

    def plots_1d(self, roots, params=None, legend_labels=None, legend_ncol=None, label_order=None, nx=None,
                 paramList=None, roots_per_param=False, share_y=None, markers=None, xlims=None, param_renames={}):
        roots = makeList(roots)
        if roots_per_param:
            params = [self.check_param(root[0], param, param_renames) for root, param in zip(roots, params)]
        else:
            params = self.get_param_array(roots[0], params, param_renames)
        if paramList is not None:
            wantedParams = self.paramNameListFromFile(paramList)
            params = [param for param in params if
                      param.name in wantedParams or param_renames.get(param.name, '') in wantedParams]
        nparam = len(params)
        if share_y is None: share_y = self.settings.prob_label is not None and nparam > 1
        plot_col, plot_row = self.make_figure(nparam, nx=nx)
        plot_roots = roots
        for i, param in enumerate(params):
            ax = self.subplot_number(i)
            if roots_per_param: plot_roots = roots[i]
            marker = None
            if markers is not None:
                if isinstance(markers, dict):
                    marker = markers.get(param.name, None)
                elif i < len(markers):
                    marker = markers[i]
            self.plot_1d(plot_roots, param, no_ylabel=share_y and i % self.plot_col > 0, marker=marker,
                         param_renames=param_renames)
            if xlims is not None: ax.set_xlim(xlims[i][0], xlims[i][1])
            if share_y: self.spaceTicks(ax.xaxis, expand=True)
        self.finish_plot(self.default_legend_labels(legend_labels, roots), legend_ncol=legend_ncol,
                         label_order=label_order)
        if share_y: plt.subplots_adjust(wspace=0)
        return plot_col, plot_row

    def plots_2d(self, roots, param1=None, params2=None, param_pairs=None, nx=None, legend_labels=None,
                 legend_ncol=None, label_order=None, filled=False, shaded=False):
        pairs = []
        roots = makeList(roots)
        if isinstance(param1, (list, tuple)) and len(param1) == 2:
            params2 = [param1[1]]
            param1 = param1[0]
        if param_pairs is None:
            if param1 is not None:
                param1 = self.check_param(roots[0], param1)
                params2 = self.get_param_array(roots[0], params2)
                for param in params2:
                    if param.name != param1.name: pairs.append((param1, param))
            else:
                raise GetDistPlotError('No parameter or parameter pairs for 2D plot')
        else:
            for pair in param_pairs:
                pairs.append((self.check_param(roots[0], pair[0]), self.check_param(roots[0], pair[1])))
        if filled and shaded:
            raise GetDistPlotError("Plots cannot be both filled and shaded")
        plot_col, plot_row = self.make_figure(len(pairs), nx=nx)

        for i, pair in enumerate(pairs):
            self.subplot_number(i)
            self.plot_2d(roots, param_pair=pair, filled=filled, shaded=not filled and shaded,
                         add_legend_proxy=i == 0)

        self.finish_plot(self.default_legend_labels(legend_labels, roots), legend_ncol=legend_ncol,
                         label_order=label_order)
        return plot_col, plot_row

    def subplot(self, x, y, **kwargs):
        self.subplots[y, x] = ax = plt.subplot(self.plot_row, self.plot_col, y * self.plot_col + x + 1, **kwargs)
        return ax

    def subplot_number(self, i):
        self.subplots[i // self.plot_col, i % self.plot_col] = ax = plt.subplot(self.plot_row, self.plot_col, i + 1)
        return ax

    def plots_2d_triplets(self, root_params_triplets, nx=None, filled=False, x_lim=None):
        plot_col, plot_row = self.make_figure(len(root_params_triplets), nx=nx)
        for i, (root, param1, param2) in enumerate(root_params_triplets):
            ax = self.subplot_number(i)
            self.plot_2d(root, param_pair=[param1, param2], filled=filled, add_legend_proxy=i == 0)
            if x_lim is not None: ax.set_xlim(x_lim)
        self.finish_plot()
        return plot_col, plot_row

    def spaceTicks(self, axis, expand=True):
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

    def triangle_plot(self, roots, in_params=None, legend_labels=None, plot_3d_with_param=None, filled=False,
                      filled_compare=False, shaded=False,
                      contour_args=None, contour_colors=None, contour_ls=None, contour_lws=None, line_args=None,
                      label_order=None, legend_ncol=None, legend_loc=None, upper_roots=None, upper_kwargs={}):
        roots = makeList(roots)
        params = self.get_param_array(roots[0], in_params)
        plot_col = len(params)
        if plot_3d_with_param is not None: col_param = self.check_param(roots[0], plot_3d_with_param)
        self.make_figure(nx=plot_col, ny=plot_col)
        lims = dict()
        ticks = dict()
        if filled_compare: filled = filled_compare

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
            ax = self.subplot(i, i)
            self.plot_1d(roots1d, param, do_xlabel=i == plot_col - 1,
                         no_label_no_numbers=self.settings.no_triangle_axis_labels,
                         label_right=True, no_zero=True, no_ylabel=True, no_ytick=True, line_args=line_args)
            # set no_ylabel=True for now, can't see how to not screw up spacing with right-sided y label
            if self.settings.no_triangle_axis_labels: self.spaceTicks(ax.xaxis)
            lims[i] = ax.get_xlim()
            ticks[i] = ax.get_xticks()
        for i, param in enumerate(params):
            for i2 in range(i + 1, len(params)):
                param2 = params[i2]
                pair = [param, param2]
                ax = self.subplot(i, i2)
                if plot_3d_with_param is not None:
                    self.plot_3d(roots, pair + [col_param], color_bar=False, line_offset=1, add_legend_proxy=False,
                                 do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0, contour_args=contour_args,
                                 no_label_no_numbers=self.settings.no_triangle_axis_labels)
                else:
                    self.plot_2d(roots, param_pair=pair, do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0,
                                 no_label_no_numbers=self.settings.no_triangle_axis_labels, shaded=shaded,
                                 add_legend_proxy=i == 0 and i2 == 1, contour_args=contour_args)
                ax.set_xticks(ticks[i])
                ax.set_yticks(ticks[i2])
                ax.set_xlim(lims[i])
                ax.set_ylim(lims[i2])

                if upper_roots is not None:
                    ax = self.subplot(i2, i)
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
            self.setAxisProperties(label_ax.yaxis, False)

        if self.settings.no_triangle_axis_labels: plt.subplots_adjust(wspace=0, hspace=0)
        if plot_3d_with_param is not None:
            bottom = 0.5
            if len(params) == 2: bottom += 0.1;
            cb = self.fig.colorbar(self.last_scatter, cax=self.fig.add_axes([0.9, bottom, 0.03, 0.35]))
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            self.add_colorbar_label(cb, col_param, label_rotation=-self.settings.colorbar_label_rotation)

        labels = self.default_legend_labels(legend_labels, roots1d)
        if not legend_loc and len(params) < 4 and upper_roots is None:
            legend_loc = 'upper right'
        self.finish_plot(labels, label_order=label_order,
                         legend_ncol=legend_ncol or (None if upper_roots is None else len(labels)),
                         legend_loc=legend_loc, no_gap=self.settings.no_triangle_axis_labels,
                         no_extra_legend_space=upper_roots is None)

    def rectangle_plot(self, xparams, yparams, yroots=None, roots=None, plot_roots=None, plot_texts=None,
                       ymarkers=None, xmarkers=None, param_limits={}, legend_labels=None, legend_ncol=None,
                       label_order=None, marker_args={}, **kwargs):
        """
            roots uses the same set of roots for every plot in the rectangle
            yroots (list of list of roots) allows use of different set of roots for each row of the plot
            plot_roots allows you to specify (via list of list of list of roots) the set of roots for each individual subplot
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
                ax = self.subplot(x, y, sharex=sharex, sharey=sharey)
                if y == 0:
                    sharex = ax
                    xshares.append(ax)
                res = self.plot_2d(subplot_roots, param_pair=[xparam, yparam], do_xlabel=y == len(yparams) - 1,
                                   do_ylabel=x == 0, add_legend_proxy=x == 0 and y == 0, **kwargs)
                if ymarkers is not None and ymarkers[y] is not None: self.add_y_marker(ymarkers[y], **marker_args)
                if xmarkers is not None and xmarkers[x] is not None: self.add_x_marker(xmarkers[x], **marker_args)
                limits[xparam], limits[yparam] = self.updateLimits(res, limits.get(xparam), limits.get(yparam))
                if y != len(yparams) - 1: plt.setp(ax.get_xticklabels(), visible=False)
                if x != 0: plt.setp(ax.get_yticklabels(), visible=False)
                if x == 0: yshares.append(ax)
                if plot_texts and plot_texts[x][y]:
                    self.add_text_left(plot_texts[x][y], y=0.9, ax=ax)
                axarray.append(ax)
            ax_arr.append(axarray)
        for xparam, ax in zip(xparams, xshares):
            ax.set_xlim(param_limits.get(xparam, limits[xparam]))
            self.spaceTicks(ax.xaxis)
            ax.set_xlim(ax.xaxis.get_view_interval())
        for yparam, ax in zip(yparams, yshares):
            ax.set_ylim(param_limits.get(yparam, limits[yparam]))
            self.spaceTicks(ax.yaxis)
            ax.set_ylim(ax.yaxis.get_view_interval())
        plt.subplots_adjust(wspace=0, hspace=0)
        if roots: legend_labels = self.default_legend_labels(legend_labels, roots)
        self.finish_plot(no_gap=True, legend_labels=legend_labels, label_order=label_order,
                         legend_ncol=legend_ncol or len(legend_labels))
        return ax_arr

    def rotate_yticklabels(self, ax=None, rotation=90):
        if ax is None: ax = plt.gca()
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_rotation(rotation)

    def add_colorbar(self, param, orientation='vertical', **ax_args):
        cb = plt.colorbar(orientation=orientation)
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

    def add_line(self, P1, P2, zorder=0, color=None, ls=None, ax=None, **kwargs):
        if color is None: color = self.settings.axis_marker_color
        if ls is None: ls = self.settings.axis_marker_ls
        (ax or plt.gca()).add_line(plt.Line2D(P1, P2, color=color, ls=ls, zorder=zorder, **kwargs))

    def add_colorbar_label(self, cb, param, label_rotation=None):
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

    def add_3d_scatter(self, root, in_params, color_bar=True, alpha=1, extra_thin=1, **ax_args):
        params = self.get_param_array(root, in_params)
        pts = self.sampleAnalyser.load_single_samples(root)
        names = self.paramNamesForRoot(root)
        samples = []
        for param in params:
            if hasattr(param, 'getDerived'):
                samples.append(param.getDerived(self._makeParamObject(names, pts)))
            else:
                samples.append(pts[:, names.numberOfName(param.name)])
        if extra_thin > 1:
            samples = [pts[::extra_thin] for pts in samples]
        self.last_scatter = plt.scatter(samples[0], samples[1], edgecolors='none',
                                        s=self.settings.scatter_size, c=samples[2], cmap=self.settings.colormap_scatter,
                                        alpha=alpha)
        if color_bar: self.last_colorbar = self.add_colorbar(params[2], **ax_args)
        xbounds = [min(samples[0]), max(samples[0])]
        r = xbounds[1] - xbounds[0]
        xbounds[0] -= r / 20
        xbounds[1] += r / 20
        ybounds = [min(samples[1]), max(samples[1])]
        r = ybounds[1] - ybounds[0]
        ybounds[0] -= r / 20
        ybounds[1] += r / 20
        return [xbounds, ybounds]

    def plot_3d(self, roots, in_params=None, params_for_plots=None, color_bar=True, line_offset=0,
                add_legend_proxy=True, **kwargs):
        roots = makeList(roots)
        if params_for_plots:
            params_for_plots = [self.get_param_array(root, p) for p, root in zip(params_for_plots, roots)]
        else:
            if not in_params: raise GetDistPlotError('No parameters for plot_3d!')
            params = self.get_param_array(roots[0], in_params)
            params_for_plots = [params for _ in roots]  # all the same
        if self.fig is None: self.make_figure()
        contour_args = self._make_contour_args(len(roots) - 1, **kwargs)
        xlims, ylims = self.add_3d_scatter(roots[0], params_for_plots[0], color_bar=color_bar, **kwargs)
        for i, root in enumerate(roots[1:]):
            params = params_for_plots[i + 1]
            res = self.add_2d_contours(root, params[0], params[1], i + line_offset, add_legend_proxy=add_legend_proxy,
                                       zorder=i + 1, **contour_args[i])
            xlims, ylims = self.updateLimits(res, xlims, ylims)
        if not 'lims' in kwargs:
            params = params_for_plots[0]
            lim1 = self.checkBounds(roots[0], params[0].name, xlims[0], xlims[1])
            lim2 = self.checkBounds(roots[0], params[1].name, ylims[0], ylims[1])
            kwargs['lims'] = [lim1[0], lim1[1], lim2[0], lim2[1]]
        self.setAxes(params, **kwargs)

    def plots_3d(self, roots, param_sets, nx=None, filled_compare=False, legend_labels=None, **kwargs):
        roots = makeList(roots)
        sets = [[self.check_param(roots[0], param) for param in param_group] for param_group in param_sets]
        plot_col, plot_row = self.make_figure(len(sets), nx=nx, xstretch=1.3)

        for i, triplet in enumerate(sets):
            self.subplot_number(i)
            self.plot_3d(roots, triplet, filled=filled_compare, **kwargs)
        self.finish_plot(self.default_legend_labels(legend_labels, roots[1:]), no_tight=True)
        return plot_col, plot_row

    def plots_3d_z(self, roots, param_x, param_y, param_z=None, max_z=None, **kwargs):
        """Make set of plots of param_x against param_y, each coloured by values of parameters in param_z (all if None)"""
        roots = makeList(roots)
        param_z = self.get_param_array(roots[0], param_z)
        if max_z is not None and len(param_z) > max_z: param_z = param_z[:max_z]
        param_x, param_y = self.get_param_array(roots[0], [param_x, param_y])
        sets = [[param_x, param_y, z] for z in param_z if z != param_x and z != param_y]
        return self.plots_3d(roots, sets, **kwargs)

    def add_text(self, text_label, x=0.95, y=0.06, ax=None, **kwargs):
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
        args = {'horizontalalignment': 'left'}
        args.update(kwargs)
        self.add_text(text_label, x, y, ax, **args)

    def export(self, fname=None, adir=None, watermark=None, tag=None):
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

    def paramNameListFromFile(self, fname):
        p = ParamNames(fname)
        return [name.name for name in p.names]
