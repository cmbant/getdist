import os
import glob
import logging
import copy
import pickle
import math
import time
from typing import Mapping

import numpy as np
from scipy.stats import norm
import getdist
from getdist import chains, types, covmat, ParamInfo, IniFile, ParamNames, cobaya_interface
from getdist.densities import Density1D, Density2D
from getdist.chains import Chains, chainFiles, last_modified, WeightedSampleError, ParamError
from getdist.convolve import convolve1D, convolve2D
from getdist.cobaya_interface import MCSamplesFromCobaya
import getdist.kde_bandwidth as kde
from getdist.parampriors import ParamBounds

pickle_version = 22


class MCSamplesError(WeightedSampleError):
    """
    An Exception that is raised when there is an error inside the MCSamples class.
    """
    pass


class SettingError(MCSamplesError):
    """
    An Exception that indicates bad settings.
    """
    pass


class BandwidthError(MCSamplesError):
    """
    An Exception that indicate KDE bandwidth failure.
    """
    pass


def loadMCSamples(file_root, ini=None, jobItem=None, no_cache=False, settings=None):
    """
    Loads a set of samples from a file or files.

    Sample files are plain text (*file_root.txt*) or a set of files (*file_root_1.txt*, *file_root_2.txt*, etc.).

    Auxiliary files **file_root.paramnames** gives the parameter names
    and (optionally) **file_root.ranges** gives hard prior parameter ranges.

    For a description of the various analysis settings and default values see
    `analysis_defaults.ini <https://getdist.readthedocs.org/en/latest/analysis_settings.html>`_.

    :param file_root: The root name of the files to read (no extension)
    :param ini: The name of a .ini file with analysis settings to use
    :param jobItem: an optional grid jobItem instance for a CosmoMC grid output
    :param no_cache: Indicates whether or not we should cache loaded samples in a pickle
    :param settings: dictionary of analysis settings to override defaults
    :return: The :class:`MCSamples` instance
    """
    files = chainFiles(file_root)
    if not files:  # try new Cobaya format
        files = chainFiles(file_root, separator='.')
    path, name = os.path.split(file_root)
    cache_dir = getdist.make_cache_dir()
    if cache_dir:
        import hashlib
        cache_name = name + '_' + hashlib.md5(os.path.abspath(path).encode('utf-8')).hexdigest()[:10]
        path = cache_dir
    else:
        cache_name = name
    if not os.path.exists(path):
        os.mkdir(path)
    cachefile = os.path.join(path, cache_name) + '.py_mcsamples'
    samples = MCSamples(file_root, jobItem=jobItem, ini=ini, settings=settings)
    if os.path.isfile(file_root + '.paramnames'):
        allfiles = files + [file_root + '.ranges', file_root + '.paramnames', file_root + '.properties.ini']
    else:  # Cobaya
        folder = os.path.dirname(file_root)
        prefix = os.path.basename(file_root)
        allfiles = files + [
            os.path.join(folder, f) for f in os.listdir(folder) if (
                    f.startswith(prefix) and
                    any(f.lower().endswith(end) for end in ['updated.yaml', 'full.yaml']))]
    if not no_cache and os.path.exists(cachefile) and last_modified(allfiles) < os.path.getmtime(cachefile):
        try:
            with open(cachefile, 'rb') as inp:
                cache = pickle.load(inp)
            if cache.version == pickle_version and samples.ignore_rows == cache.ignore_rows \
                    and samples.min_weight_ratio == cache.min_weight_ratio:
                changed = len(samples.contours) != len(cache.contours) or \
                          np.any(np.array(samples.contours) != np.array(cache.contours))
                cache.updateSettings(ini=ini, settings=settings, doUpdate=changed)
                return cache
        except Exception:
            pass
    if not len(files):
        raise IOError('No chains found: ' + file_root)
    samples.readChains(files)
    if no_cache:
        if os.path.exists(cachefile):
            os.remove(cachefile)
    else:
        samples.savePickle(cachefile)
    return samples


class Kernel1D:
    def __init__(self, winw, h):
        self.winw = winw
        self.h = h
        self.x = np.arange(-winw, winw + 1)
        Win = np.exp(-(self.x / h) ** 2 / 2.)
        self.Win = Win / np.sum(Win)


# =============================================================================

class MCSamples(Chains):
    """
    The main high-level class for a collection of parameter samples.

    Derives from :class:`.chains.Chains`, adding high-level functions including
    Kernel Density estimates, parameter ranges and custom settings.
    """

    def __init__(self, root=None, jobItem=None, ini=None, settings=None, ranges=None,
                 samples=None, weights=None, loglikes=None, **kwargs):
        """
        For a description of the various analysis settings and default values see
        `analysis_defaults.ini <https://getdist.readthedocs.org/en/latest/analysis_settings.html>`_.


        :param root: A root file name when loading from file
        :param jobItem: optional jobItem for parameter grid item. Should have jobItem.chainRoot and jobItem.batchPath
        :param ini: a .ini file to use for custom analysis settings
        :param settings: a dictionary of custom analysis settings
        :param ranges: a dictionary giving any additional hard prior bounds for parameters,
                       eg. {'x':[0, 1], 'y':[None,2]}
        :param samples: if not loading from file, array of parameter values for each sample, passed
                        to :meth:`setSamples`, or list of arrays if more than one chain
        :param weights: array of weights for samples, or list of arrays if more than one chain
        :param loglikes: array of -log(Likelihood) for samples, or list of arrays if more than one chain
        :param kwargs: keyword arguments passed to inherited classes, e.g. to manually make a samples object from
                       sample arrays in memory:

               - **paramNamesFile**: optional name of .paramnames file with parameter names
               - **names**: list of names for the parameters, or list of arrays if more than one chain
               - **labels**: list of latex labels for the parameters
               - **renames**: dictionary of parameter aliases
               - **ignore_rows**:

                     - if int >=1: The number of rows to skip at the file in the beginning of the file
                     - if float <1: The fraction of rows to skip at the beginning of the file
               - **label**: a latex label for the samples
               - **name_tag**: a name tag for this instance
               -  **sampler**: string describing the type of samples; if "nested" or "uncorrelated"
                  the effective number of samples is calculated using uncorrelated approximation. If not specified
                  will be read from the root.properties.ini file if it exists and otherwise default to "mcmc".

        """
        Chains.__init__(self, root, jobItem=jobItem, **kwargs)

        self.version = pickle_version

        self.markers = {}

        self.ini = ini
        if self.jobItem:
            self.batch_path = self.jobItem.batchPath
        else:
            self.batch_path = ''

        self._readRanges()
        if ranges:
            self.setRanges(ranges)

        # Other variables
        self.range_ND_contour = 1
        self.range_confidence = 0.001
        self.num_bins = 128
        self.fine_bins = 1024
        self.num_bins_2D = 40
        self.fine_bins_2D = 256
        self.smooth_scale_1D = -1.
        self.smooth_scale_2D = -1.
        self.num_bins_ND = 12
        self.boundary_correction_order = 1
        self.mult_bias_correction_order = 1
        self.max_corr_2D = 0.95
        self.use_effective_samples_2D = False
        self.contours = np.array([0.68, 0.95])
        self.max_scatter_points = 2000
        self.credible_interval_threshold = 0.05

        self.shade_likes_is_mean_loglikes = False

        self.likeStats = None
        self.max_mult = 0
        self.mean_mult = 0
        self.plot_data_dir = ""
        if root:
            self.rootname = os.path.basename(root)
        else:
            self.rootname = ""

        self.rootdirname = ""
        self.indep_thin = 0
        if 'ignore_rows' in kwargs:
            if settings is None:
                settings = {}
            settings['ignore_rows'] = kwargs['ignore_rows']
        self.ignore_rows = float(kwargs.get('ignore_rows', 0))
        # Do not remove burn-in for nested sampler samples
        if self.sampler == "nested" and not np.isclose(self.ignore_rows, 0):
            raise ValueError("Should not remove burn-in from Nested Sampler samples.")
        self.subplot_size_inch = 4.0
        self.subplot_size_inch2 = self.subplot_size_inch
        self.subplot_size_inch3 = 6.0
        self.plot_output = getdist.default_plot_output
        self.out_dir = ""
        self.no_warning_params = []
        self.no_warning_chi2_params = True

        self.max_split_tests = 4
        self.force_twotail = False

        self.corr_length_thin = 0
        self.corr_length_steps = 15
        self.converge_test_limit = 0.95

        self.done_1Dbins = False
        self.density1D = dict()

        self.updateSettings(ini=ini, settings=settings)

        if root and os.path.exists(root + '.properties.ini'):
            # any settings in properties.ini override settings for this specific chain
            self.properties = IniFile(root + '.properties.ini')
            self._setBurnOptions(self.properties)
            if self.properties.bool('burn_removed', False):
                self.ignore_frac = 0.
                self.ignore_lines = 0
            self.label = self.label or self.properties.params.get('label', None)
            if 'sampler' not in kwargs:
                self.setSampler(self.properties.string('sampler', self.sampler))
        else:
            self.properties = IniFile()
            if root and self.paramNames and self.paramNames.info_dict:
                if cobaya_interface.get_burn_removed(self.paramNames.info_dict):
                    self.properties.params['burn_removed'] = True
                    self.ignore_frac = 0.
                    self.ignore_lines = 0
                if not self.label:
                    self.label = cobaya_interface.get_sample_label(self.paramNames.info_dict)
                    if self.label:
                        self.properties.params['label'] = self.label
                if 'sampler' not in kwargs:
                    self.setSampler(cobaya_interface.get_sampler_type(self.paramNames.info_dict))
                self.properties.params['sampler'] = self.sampler
        if self.ignore_frac or self.ignore_rows:
            self.properties.params['burn_removed'] = True

        if samples is not None:
            self.readChains(samples, weights, loglikes)

    def copy(self, label=None, settings=None):
        """
        Create a copy of this sample object

        :param label: optional lable for the new copy
        :param settings: optional modified settings for the new copy
        :return: copyied :class:`MCSamples` instance
        """
        new = copy.deepcopy(self)
        if label:
            new.label = label
        if settings is not None:
            new.needs_update = True
            new.updateSettings(settings)
        return new

    def setRanges(self, ranges):
        """
        Sets the ranges parameters, e.g. hard priors on positivity etc.
        If a min or max value is None, then it is assumed to be unbounded.

        :param ranges: A list or a tuple of [min,max] values for each parameter,
                       or a dictionary giving [min,max] values for specific parameter names
        """
        if isinstance(ranges, (list, tuple)):
            for i, minmax in enumerate(ranges):
                self.ranges.setRange(self.parName(i), minmax)
        elif isinstance(ranges, Mapping):
            for key, value in ranges.items():
                self.ranges.setRange(key, value)
        elif isinstance(ranges, ParamBounds):
            self.ranges = copy.deepcopy(ranges)
        else:
            raise ValueError('MCSamples ranges parameter must be list or dict')
        self.needs_update = True

    def parName(self, i, starDerived=False):
        """
        Gets the name of i'th parameter

        :param i: The index of the parameter
        :param starDerived: add a star at the end of the name if the parameter is derived
        :return: The name of the parameter (string)
        """
        return self.paramNames.name(i, starDerived)

    def parLabel(self, i):
        """
        Gets the latex label of the parameter

        :param i: The index or name of a parameter.
        :return: The parameter's label.
        """
        if isinstance(i, str):
            return self.paramNames.parWithName(i).label
        else:
            return self.paramNames.names[i].label

    def _setBurnOptions(self, ini):
        """
        Sets the ignore_rows value from configuration.

        :param ini: The :class:`.inifile.IniFile` to be used
        """
        ini.setAttr('ignore_rows', self)
        self.ignore_lines = int(self.ignore_rows)
        if not self.ignore_lines:
            self.ignore_frac = self.ignore_rows
        else:
            self.ignore_frac = 0
        ini.setAttr('min_weight_ratio', self)

    def initParameters(self, ini):
        """
        Initializes settings.
        Gets parameters from :class:`~.inifile.IniFile`.

        :param ini:  The :class:`~.inifile.IniFile` to be used
        """
        self._setBurnOptions(ini)

        ini.setAttr('range_ND_contour', self)
        ini.setAttr('range_confidence', self)
        ini.setAttr('num_bins', self)
        ini.setAttr('fine_bins', self)

        ini.setAttr('num_bins_2D', self)
        ini.setAttr('fine_bins_2D', self)

        ini.setAttr('smooth_scale_1D', self)
        ini.setAttr('smooth_scale_2D', self)

        ini.setAttr('boundary_correction_order', self, 1)
        ini.setAttr('mult_bias_correction_order', self, 1)

        ini.setAttr('num_bins_ND', self)

        ini.setAttr('max_scatter_points', self)
        ini.setAttr('credible_interval_threshold', self)

        ini.setAttr('subplot_size_inch', self)
        ini.setAttr('subplot_size_inch2', self)
        ini.setAttr('subplot_size_inch3', self)
        ini.setAttr('plot_output', self)

        ini.setAttr('force_twotail', self)
        if self.force_twotail:
            logging.warning('Computing two tail limits')
        ini.setAttr('max_corr_2D', self)

        if ini.hasKey('contours'):
            ini.setAttr('contours', self)
        elif ini.hasKey('num_contours'):
            num_contours = ini.int('num_contours', 2)
            self.contours = np.array([ini.float('contour' + str(i + 1)) for i in range(num_contours)])
        # how small the end bin must be relative to max to use two tail
        self.max_frac_twotail = []
        for i, contour in enumerate(self.contours):
            max_frac = np.exp(-1.0 * math.pow(norm.ppf((1 - contour) / 2), 2) / 2)
            if ini:
                max_frac = ini.float('max_frac_twotail' + str(i + 1), max_frac)
            self.max_frac_twotail.append(max_frac)

        ini.setAttr('converge_test_limit', self, self.contours[-1])
        ini.setAttr('corr_length_thin', self)
        ini.setAttr('corr_length_steps', self)
        ini.setAttr('no_warning_params', self, [])
        ini.setAttr('no_warning_chi2_params', self, True)
        self.batch_path = ini.string('batch_path', self.batch_path, allowEmpty=False)

    def _initLimits(self, ini=None):
        bin_limits = ""
        if ini:
            bin_limits = ini.string('all_limits', '')

        self.markers = {}

        for par in self.paramNames.names:
            if bin_limits:
                line = bin_limits
            else:
                line = ''
                if ini and 'limits[%s]' % par.name in ini.params:
                    line = ini.string('limits[%s]' % par.name)
            if line:
                limits = line.split()
                if len(limits) == 2:
                    self.ranges.setRange(par.name, limits)

            par.limmin = self.ranges.getLower(par.name)
            par.limmax = self.ranges.getUpper(par.name)
            par.has_limits_bot = par.limmin is not None
            par.has_limits_top = par.limmax is not None

            if ini and 'marker[%s]' % par.name in ini.params:
                line = ini.string('marker[%s]' % par.name)
                if line:
                    self.markers[par.name] = float(line)

    def updateSettings(self, settings=None, ini=None, doUpdate=True):
        """
        Updates settings from a .ini file or dictionary

        :param settings: The a dict containing settings to set, taking preference over any values in ini
        :param ini: The name of .ini file to get settings from, or an :class:`~.inifile.IniFile` instance; by default
                    uses current settings
        :param doUpdate: True if should update internal computed values, False otherwise (e.g. if want to make
                         other changes first)
        """
        assert (settings is None or isinstance(settings, Mapping))
        if not ini:
            ini = self.ini
        elif isinstance(ini, str):
            ini = IniFile(ini)
        else:
            ini = copy.deepcopy(ini)
        if not ini:
            ini = IniFile(getdist.default_getdist_settings)
        if settings:
            ini.params.update(settings)
        self.ini = ini
        if ini:
            self.initParameters(ini)
        if doUpdate and self.samples is not None:
            self.updateBaseStatistics()

    def readChains(self, files_or_samples, weights=None, loglikes=None):
        """
        Loads samples from a list of files or array(s), removing burn in,
        deleting fixed parameters, and combining into one self.samples array

        :param files_or_samples: The list of file names to read, samples or list of samples
        :param weights: array of weights if setting from arrays
        :param loglikes: array of -2 log(likelihood) if setting from arrays
        :return: self.
        """
        self.loadChains(self.root, files_or_samples, weights=weights, loglikes=loglikes)

        if self.ignore_frac and (
                not self.jobItem or (not self.jobItem.isImportanceJob and not self.jobItem.isBurnRemoved())):
            self.removeBurnFraction(self.ignore_frac)
            if chains.print_load_details:
                print('Removed %s as burn in' % self.ignore_frac)
        elif not int(self.ignore_rows):
            if chains.print_load_details:
                print('Removed no burn in')

        self.deleteFixedParams()

        # Make a single array for chains
        if self.chains is not None:
            self.makeSingle()

        self.updateBaseStatistics()

        return self

    def updateBaseStatistics(self):
        """
        Updates basic computed statistics (y, covariance etc), e.g. after a change in samples or weights

        :return: self
        """
        super().updateBaseStatistics()
        mult_max = (self.mean_mult * self.numrows) / min(self.numrows // 2, 500)
        outliers = np.sum(self.weights > mult_max)
        if outliers != 0:
            logging.warning('outlier fraction %s ', float(outliers) / self.numrows)

        self.indep_thin = 0
        self._setCov()
        self.done_1Dbins = False
        self.density1D = dict()

        self._initLimits(self.ini)

        for par in self.paramNames.names:
            par.N_eff_kde = None

        # Get ND confidence region
        self._setLikeStats()
        return self

    def makeSingleSamples(self, filename="", single_thin=None):
        """
        Make file of unit weight samples by choosing samples
        with probability proportional to their weight.

        :param filename: The filename to write to, leave empty if no output file is needed
        :param single_thin: factor to thin by; if not set generates as many samples as it can
                            up to self.max_scatter_points
        :return: numpy array of selected weight-1 samples if no filename
        """
        if single_thin is None:
            single_thin = max(1, self.norm / self.max_mult / self.max_scatter_points)
        rand = np.random.random_sample(self.numrows)

        if filename:
            with open(filename, 'w') as f:
                for i, r in enumerate(rand):
                    if r <= self.weights[i] / self.max_mult / single_thin:
                        f.write("%16.7E" % 1.0)
                        f.write("%16.7E" % (self.loglikes[i]))
                        for j in range(self.n):
                            f.write("%16.7E" % (self.samples[i][j]))
                        f.write("\n")
        else:
            return self.samples[rand <= self.weights / (self.max_mult * single_thin)]

    def writeThinData(self, fname, thin_ix, cool=1):
        """
        Writes samples at thin_ix to file

        :param fname: The filename to write to.
        :param thin_ix: Indices of the samples to write
        :param cool: if not 1, cools the samples by this factor
        """
        nparams = self.samples.shape[1]
        if cool != 1:
            logging.info('Cooled thinned output with temp: %s', cool)
        MaxL = np.max(self.loglikes)
        with open(fname, 'w') as f:
            i = 0
            for thin in thin_ix:
                if cool != 1:
                    newL = self.loglikes[thin] * cool
                    f.write("%16.7E" % (
                        np.exp(-(newL - self.loglikes[thin]) - MaxL * (1 - cool))))
                    f.write("%16.7E" % newL)
                    for j in range(nparams):
                        f.write("%16.7E" % (self.samples[i][j]))
                else:
                    f.write("%f" % 1.)
                    f.write("%f" % (self.loglikes[thin]))
                    for j in range(nparams):
                        f.write("%16.7E" % (self.samples[i][j]))
                i += 1
        print('Wrote ', len(thin_ix), ' thinned samples')

    def getCovMat(self):
        """
        Gets the CovMat instance containing covariance matrix for all the non-derived parameters
        (for example useful for subsequent MCMC runs to orthogonalize the parameters)

        :return: A :class:`~.covmat.CovMat` object holding the covariance
        """
        nparamNonDerived = self.paramNames.numNonDerived()
        return covmat.CovMat(matrix=self.fullcov[:nparamNonDerived, :nparamNonDerived],
                             paramNames=self.paramNames.list()[:nparamNonDerived])

    def writeCovMatrix(self, filename=None):
        """
        Writes the covrariance matrix of non-derived parameters to a file.

        :param filename: The filename to write to; default is file_root.covmat
        """
        filename = filename or self.rootdirname + ".covmat"
        self.getCovMat().saveToFile(filename)

    def writeCorrelationMatrix(self, filename=None):
        """
        Write the correlation matrix to a file

        :param filename: The file to write to, If none writes to file_root.corr
        """
        filename = filename or self.rootdirname + ".corr"
        np.savetxt(filename, self.getCorrelationMatrix(), fmt="%15.7E")

    def getFractionIndices(self, weights, n):
        """
        Calculates the indices of weights that split the weights into sets of equal 1/n fraction of the total weight

        :param weights: array of weights
        :param n: number of groups to split in to
        :return: array of indices of the boundary rows in the weights array
        """
        cumsum = np.cumsum(weights)
        fraction_indices = np.append(np.searchsorted(cumsum, np.linspace(0, 1, n, endpoint=False) * self.norm),
                                     self.weights.shape[0])
        return fraction_indices

    def PCA(self, params, param_map=None, normparam=None, writeDataToFile=False, filename=None,
            conditional_params=(), n_best_only=None):
        """
        Perform principle component analysis (PCA). In other words,
        get eigenvectors and eigenvalues for normalized variables
        with optional (log modulus) mapping to find power law fits.

        :param params: List of names of the parameters to use
        :param param_map: A transformation to apply to parameter values;  A list or string containing
                          either N (no transformation) or L (for log transform) for each parameter.
                          By default uses log if no parameter values cross zero

        :param normparam: optional name of parameter to normalize result (i.e. this parameter will have unit power)
        :param writeDataToFile: True if should write the output to file.
        :param filename: The filename to write, by default root_name.PCA.
        :param conditional_params: optional list of parameters to treat as fixed,
               i.e. for PCA conditional on fixed values of these parameters
        :param n_best_only: return just the short summary constraint for the tightest n_best_only constraints
        :return: a string description of the output of the PCA
        """
        logging.info('Doing PCA for %s parameters', len(params))
        if len(conditional_params):
            logging.info('conditional %u fixed parameters', len(conditional_params))

        PCAtext = 'PCA for parameters:\n'

        params = [name for name in params if self.paramNames.parWithName(name)]
        nparams = len(params)
        indices = [self.index[param] for param in params]
        conditional_params = [self.index[param] for param in conditional_params]
        indices += conditional_params

        if normparam:
            if normparam in params:
                normparam = params.index(normparam)
            else:
                normparam = -1
        else:
            normparam = -1

        n = len(indices)
        PCdata = self.samples[:, indices].copy()
        PClabs = []

        PCmean = np.zeros(n)
        sd = np.zeros(n)
        newmean = np.zeros(n)
        newsd = np.zeros(n)
        if param_map is None:
            param_map = ''
            for par in self.paramNames.parsWithNames(params):
                self._initParamRanges(par.name)
                if par.param_max < 0 or par.param_min < (par.param_max - par.param_min) / 10:
                    param_map += 'N'
                else:
                    param_map += 'L'

        doexp = False
        for i, parix in enumerate(indices):
            if i < nparams:
                label = self.parLabel(parix)
                if param_map[i] == 'L':
                    doexp = True
                    PCdata[:, i] = np.log(PCdata[:, i])
                    PClabs.append("ln(" + label + ")")
                elif param_map[i] == 'M':
                    doexp = True
                    PCdata[:, i] = np.log(-1.0 * PCdata[:, i])
                    PClabs.append("ln(-" + label + ")")
                else:
                    PClabs.append(label)
                PCAtext += "%10s :%s\n" % (str(parix + 1), str(PClabs[i]))

            PCmean[i] = np.dot(self.weights, PCdata[:, i]) / self.norm
            PCdata[:, i] -= PCmean[i]
            sd[i] = np.sqrt(np.dot(self.weights, PCdata[:, i] ** 2) / self.norm)
            if sd[i] != 0:
                PCdata[:, i] /= sd[i]

        PCAtext += "\n"
        PCAtext += 'Correlation matrix for reduced parameters\n'
        correlationMatrix = np.ones((n, n))
        for i in range(n):
            for j in range(i):
                correlationMatrix[j][i] = np.dot(self.weights, PCdata[:, i] * PCdata[:, j]) / self.norm
                correlationMatrix[i][j] = correlationMatrix[j][i]
        for i in range(nparams):
            PCAtext += '%12s :' % params[i]
            for j in range(n):
                PCAtext += '%8.4f' % correlationMatrix[j][i]
            PCAtext += '\n'

        if len(conditional_params):
            u = np.linalg.inv(correlationMatrix)
            u = u[np.ix_(list(range(len(params))), list(range(len(params))))]
            u = np.linalg.inv(u)
            n = nparams
            PCdata = PCdata[:, :nparams]
        else:
            u = correlationMatrix
        evals, evects = np.linalg.eig(u)
        isorted = evals.argsort()
        u = np.transpose(evects[:, isorted])  # redefining u

        PCAtext += '\n'
        PCAtext += 'e-values of correlation matrix\n'
        for i in range(n):
            isort = isorted[i]
            PCAtext += 'PC%2i: %8.4f\n' % (i + 1, evals[isort])

        PCAtext += '\n'
        PCAtext += 'e-vectors\n'
        for j in range(n):
            PCAtext += '%3i:' % (indices[j] + 1)
            for i in range(n):
                isort = isorted[i]
                PCAtext += '%8.4f' % (evects[j][isort])
            PCAtext += '\n'

        if normparam != -1:
            # Set so parameter normparam has exponent 1
            for i in range(n):
                u[i, :] = u[i, :] / u[i, normparam] * sd[normparam]
        else:
            # Normalize so main component has exponent 1
            for i in range(n):
                maxi = np.abs(u[i, :]).argmax()
                u[i, :] = u[i, :] / u[i, maxi] * sd[maxi]

        nrows = PCdata.shape[0]
        for i in range(nrows):
            PCdata[i, :] = np.dot(u, PCdata[i, :])
            if doexp:
                PCdata[i, :] = np.exp(PCdata[i, :])

        PCAtext += '\n'
        PCAtext += 'Principle components\n'
        PCAmodeTexts = []
        for i in range(n):
            isort = isorted[i]
            summary = 'PC%i (e-value: %f)\n' % (i + 1, evals[isort])
            for j in range(n):
                label = self.parLabel(indices[j])
                if param_map[j] in ['L', 'M']:
                    expo = "%f" % (1.0 / sd[j] * u[i][j])
                    if param_map[j] == "M":
                        div = "%f" % (-np.exp(PCmean[j]))
                    else:
                        div = "%f" % (np.exp(PCmean[j]))
                    summary += '[%f]  (%s/%s)^{%s}\n' % (u[i][j], label, div, expo)
                else:
                    expo = "%f" % (sd[j] / u[i][j])
                    if doexp:
                        summary += '[%f]   exp((%s-%f)/%s)\n' % (u[i][j], label, PCmean[j], expo)
                    else:
                        summary += '[%f]   (%s-%f)/%s)\n' % (u[i][j], label, PCmean[j], expo)
            newmean[i] = self.mean(PCdata[:, i])
            newsd[i] = np.sqrt(self.mean((PCdata[:, i] - newmean[i]) ** 2))
            summary += '          = %f +- %f\n' % (newmean[i], newsd[i])
            summary += '\n'
            PCAmodeTexts += [summary]
            PCAtext += summary

        # Find out how correlated these components are with other parameters
        PCAtext += 'Correlations of principle components\n'
        comps = ["%8i" % i for i in range(1, n + 1)]
        PCAtext += '%s\n' % ("".join(comps))

        for i in range(n):
            PCdata[:, i] = (PCdata[:, i] - newmean[i]) / newsd[i]

        for j in range(n):
            PCAtext += 'PC%2i' % (j + 1)
            for i in range(n):
                PCAtext += '%8.3f' % (self.mean(PCdata[:, i] * PCdata[:, j]))
            PCAtext += '\n'

        for j in range(self.n):
            PCAtext += '%4i' % (j + 1)
            for i in range(n):
                PCAtext += '%8.3f' % (
                        np.sum(self.weights * PCdata[:, i]
                               * (self.samples[:, j] - self.means[j]) / self.sddev[j]) / self.norm)

            PCAtext += '   (%s)\n' % (self.parLabel(j))

        if writeDataToFile:
            with open(filename or self.rootdirname + ".PCA", "w", encoding='utf-8') as f:
                f.write(PCAtext)
        if n_best_only:
            if n_best_only == 1:
                return PCAmodeTexts[0]
            return PCAmodeTexts[:n_best_only]
        else:
            return PCAtext

    def getNumSampleSummaryText(self):
        """
        Returns a summary text describing numbers of parameters and samples,
        and various measures of the effective numbers of samples.

        :return: The summary text as a string.
        """
        lines = 'using %s rows, %s parameters; mean weight %s, tot weight %s\n' % (
            self.numrows, self.paramNames.numParams(), self.mean_mult, self.norm)
        if self.indep_thin != 0:
            lines += 'Approx indep samples (N/corr length): %s\n' % (round(self.norm / self.indep_thin))
        lines += 'Equiv number of single samples (sum w)/max(w): %s\n' % (round(self.norm / self.max_mult))
        lines += 'Effective number of weighted samples (sum w)^2/sum(w^2): %s\n' % (
            int(self.norm ** 2 / np.dot(self.weights, self.weights)))
        return lines

    # noinspection PyUnboundLocalVariable
    def getConvergeTests(self, test_confidence=0.95, writeDataToFile=False,
                         what=('MeanVar', 'GelmanRubin', 'SplitTest', 'RafteryLewis', 'CorrLengths'),
                         filename=None, feedback=False):
        """
        Do convergence tests.

        :param test_confidence: confidence limit to test for convergence (two-tail, only applies to some tests)
        :param writeDataToFile: True if should write output to a file
        :param what: The tests to run. Should be a list of any of the following:

            - 'MeanVar': Gelman-Rubin sqrt(var(chain mean)/mean(chain var)) test in individual parameters (multiple chains only)
            - 'GelmanRubin':  Gelman-Rubin test for the worst orthogonalized parameter (multiple chains only)
            - 'SplitTest': Crude test for variation in confidence limits when samples are split up into subsets
            - 'RafteryLewis': `Raftery-Lewis test <http://www.stat.washington.edu/tech.reports/raftery-lewis2.ps>`_ (integer weight samples only)
            - 'CorrLengths': Sample correlation lengths
        :param filename: The filename to write to, default is file_root.converge
        :param feedback: If set to True, Prints the output as well as returning it.
        :return: text giving the output of the tests
        """
        lines = ''
        nparam = self.n

        chainlist = self.getSeparateChains()
        num_chains_used = len(chainlist)
        if num_chains_used > 1 and feedback:
            print('Number of chains used = ', num_chains_used)
        for chain in chainlist:
            chain.setDiffs()
        parForm = self.paramNames.parFormat()
        parNames = [parForm % self.parName(j) for j in range(nparam)]
        limits = np.array([1 - (1 - test_confidence) / 2, (1 - test_confidence) / 2])

        if 'CorrLengths' in what:
            lines += "Parameter autocorrelation lengths " \
                     "(effective number of samples N_eff = tot weight/weight length)\n"
            lines += "\n"
            lines += parForm % "" + '%15s %15s %15s\n' % ('Weight Length', 'Sample length', 'N_eff')
            maxoff = np.min([chain.weights.size // 10 for chain in chainlist])
            maxN = 0
            for j in range(nparam):
                corr = np.zeros(maxoff + 1)
                for chain in chainlist:
                    corr += chain.getAutocorrelation(j, maxoff, normalized=False) * chain.norm
                corr /= self.norm * self.vars[j]
                ix = np.argmin(corr > 0.05 * corr[0])
                N = corr[0] + 2 * np.sum(corr[1:ix])
                maxN = max(N, maxN)
                form = '%15.2E'
                if self.mean_mult > 1:
                    form = '%15.2f'
                lines += parNames[j] + form % N + ' %15.2f %15i\n' % (N / self.mean_mult, self.norm / N)
            self.indep_thin = maxN
            lines += "\n"

        if num_chains_used > 1 and 'MeanVar' in what:
            lines += "\n"
            lines += "mean convergence stats using remaining chains\n"
            lines += "param sqrt(var(chain mean)/mean(chain var))\n"
            lines += "\n"

            between_chain_var = np.zeros(nparam)
            in_chain_var = np.zeros(nparam)
            for chain in chainlist:
                between_chain_var += (chain.means - self.means) ** 2
            between_chain_var /= (num_chains_used - 1)

            for j in range(nparam):
                # Get stats for individual chains - the variance of the y over the mean of the variances
                for chain in chainlist:
                    in_chain_var[j] += np.dot(chain.weights, chain.diffs[j] ** 2)

                in_chain_var[j] /= self.norm
                lines += parNames[j] + "%10.4f  %s\n" % (
                    math.sqrt(between_chain_var[j] / in_chain_var[j]), self.parLabel(j))
            lines += "\n"

        nparamMC = self.paramNames.numNonDerived()
        if num_chains_used > 1 and nparamMC > 0 and 'GelmanRubin' in what:

            D = self.getGelmanRubinEigenvalues(chainlist=chainlist)
            if D is not None:
                self.GelmanRubin = np.max(D)
                lines += "var(mean)/mean(var) for eigenvalues of covariance of y of orthonormalized parameters\n"
                for jj, Di in enumerate(D):
                    lines += "%3i%13.5f\n" % (jj + 1, Di)
                # noinspection PyStringFormat
                GRSummary = " var(mean)/mean(var), remaining chains, worst e-value: R-1 = %13.5F" % self.GelmanRubin
            else:
                self.GelmanRubin = None
                GRSummary = 'Gelman-Rubin covariance not invertible (parameter not moved?)'
                logging.warning(GRSummary)
            if feedback:
                print(GRSummary)
            lines += "\n"

        if 'SplitTest' in what:
            # Do tests for robustness under using splits of the samples
            # Return the rms ([change in upper/lower quantile]/[standard deviation])
            # when data split into 2, 3,.. sets
            lines += "Split tests: rms_n([delta(upper/lower quantile)]/sd) n={2,3,4}, limit=%.0f%%:\n" % (
                    100 * self.converge_test_limit)
            lines += "i.e. mean sample splitting change in the quantiles in units of the st. dev.\n"
            lines += "\n"

            frac_indices = []
            for i in range(self.max_split_tests - 1):
                frac_indices.append(self.getFractionIndices(self.weights, i + 2))
            for j in range(nparam):
                split_tests = np.zeros((self.max_split_tests - 1, 2))
                confids = self.confidence(self.samples[:, j], limits)
                for ix, frac in enumerate(frac_indices):
                    split_n = 2 + ix
                    for f1, f2 in zip(frac[:-1], frac[1:]):
                        split_tests[ix, :] += (self.confidence(self.samples[:, j], limits, start=f1,
                                                               end=f2) - confids) ** 2

                    split_tests[ix, :] = np.sqrt(split_tests[ix, :] / split_n) / self.sddev[j]
                for endb, typestr in enumerate(['upper', 'lower']):
                    lines += parNames[j]
                    for ix in range(self.max_split_tests - 1):
                        lines += "%9.4f" % (split_tests[ix, endb])
                    lines += " %s\n" % typestr
            lines += "\n"

        class LoopException(Exception):
            pass

        if np.all(np.abs(self.weights - self.weights.astype(int)) < 1e-4 / self.max_mult):
            if 'RafteryLewis' in what:
                # Raftery and Lewis method
                # See http://www.stat.washington.edu/tech.reports/raftery-lewis2.ps
                # Raw non-importance sampled chains only
                thin_fac = np.empty(num_chains_used, dtype=int)
                epsilon = 0.001

                nburn = np.zeros(num_chains_used, dtype=int)
                markov_thin = np.zeros(num_chains_used, dtype=int)
                hardest = -1
                hardestend = 0
                for ix, chain in enumerate(chainlist):
                    thin_fac[ix] = int(round(np.max(chain.weights)))
                    try:
                        for j in range(nparamMC):
                            # Get binary chain depending on whether above or below confidence value
                            confids = self.confidence(chain.samples[:, j], limits, weights=chain.weights)
                            for endb in [0, 1]:
                                u = confids[endb]
                                while True:
                                    thin_ix = self.thin_indices(thin_fac[ix], chain.weights)
                                    thin_rows = len(thin_ix)
                                    if thin_rows < 2:
                                        break
                                    binchain = np.ones(thin_rows, dtype=int)
                                    binchain[chain.samples[thin_ix, j] >= u] = 0
                                    indexes = binchain[:-2] * 4 + binchain[1:-1] * 2 + binchain[2:]
                                    # Estimate transitions probabilities for 2nd order process
                                    tran = np.bincount(indexes, minlength=8).reshape((2, 2, 2))
                                    # tran[:, :, :] = 0
                                    # for i in range(2, thin_rows):
                                    #     tran[binchain[i - 2]][binchain[i - 1]][binchain[i]] += 1

                                    # Test whether 2nd order is better than Markov using BIC statistic
                                    g2 = 0
                                    for i1 in [0, 1]:
                                        for i2 in [0, 1]:
                                            for i3 in [0, 1]:
                                                if tran[i1][i2][i3] != 0:
                                                    fitted = float(
                                                        (tran[i1][i2][0] + tran[i1][i2][1]) *
                                                        (tran[0][i2][i3] + tran[1][i2][i3])) \
                                                             / float(tran[0][i2][0] + tran[0][i2][1] +
                                                                     tran[1][i2][0] + tran[1][i2][1])
                                                    focus = float(tran[i1][i2][i3])
                                                    g2 += math.log(focus / fitted) * focus
                                    g2 *= 2

                                    if g2 - math.log(float(thin_rows - 2)) * 2 < 0:
                                        break
                                    thin_fac[ix] += 1

                                # Get Markov transition probabilities for binary processes
                                if np.sum(tran[:, 0, 1]) == 0 or np.sum(tran[:, 1, 0]) == 0:
                                    thin_fac[ix] = 0
                                    raise LoopException()

                                alpha = np.sum(tran[:, 0, 1]) / float(np.sum(tran[:, 0, 0]) + np.sum(tran[:, 0, 1]))
                                beta = np.sum(tran[:, 1, 0]) / float(np.sum(tran[:, 1, 0]) + np.sum(tran[:, 1, 1]))
                                probsum = alpha + beta
                                tmp1 = math.log(probsum * epsilon / max(alpha, beta)) / math.log(abs(1.0 - probsum))
                                if int(tmp1 + 1) * thin_fac[ix] > nburn[ix]:
                                    nburn[ix] = int(tmp1 + 1) * thin_fac[ix]
                                    hardest = j
                                    hardestend = endb

                        markov_thin[ix] = thin_fac[ix]

                        # Get thin factor to have independent samples rather than Markov
                        hardest = max(hardest, 0)
                        u = self.confidence(self.samples[:, hardest], (1 - test_confidence) / 2, hardestend == 0)

                        while True:
                            thin_ix = self.thin_indices(thin_fac[ix], chain.weights)
                            thin_rows = len(thin_ix)
                            if thin_rows < 2:
                                break
                            binchain = np.ones(thin_rows, dtype=int)
                            binchain[chain.samples[thin_ix, hardest] >= u] = 0
                            indexes = binchain[:-1] * 2 + binchain[1:]
                            # Estimate transitions probabilities for 2nd order process
                            tran2 = np.bincount(indexes, minlength=4).reshape(2, 2)
                            # tran2[:, :] = 0
                            # for i in range(1, thin_rows):
                            # tran2[binchain[i - 1]][binchain[i]] += 1

                            # Test whether independence is better than Markov using BIC statistic
                            g2 = 0
                            for i1 in [0, 1]:
                                for i2 in [0, 1]:
                                    if tran2[i1][i2] != 0:
                                        fitted = float(
                                            (tran2[i1][0] + tran2[i1][1]) *
                                            (tran2[0][i2] + tran2[1][i2])) / float(thin_rows - 1)
                                        focus = float(tran2[i1][i2])
                                        if fitted <= 0 or focus <= 0:
                                            print('Raftery and Lewis estimator had problems')
                                            return
                                        g2 += np.log(focus / fitted) * focus
                            g2 *= 2

                            if g2 - np.log(float(thin_rows - 1)) < 0:
                                break

                            thin_fac[ix] += 1
                    except LoopException:
                        pass
                    except:
                        thin_fac[ix] = 0
                    if thin_fac[ix] and thin_rows < 2:
                        thin_fac[ix] = 0

                lines += "Raftery&Lewis statistics\n"
                lines += "\n"
                lines += "chain  markov_thin  indep_thin    nburn\n"

                for ix in range(num_chains_used):
                    if thin_fac[ix] == 0:
                        lines += "%4i      Failed/not enough samples\n" % ix
                    else:
                        lines += "%4i%12i%12i%12i\n" % (
                            ix, markov_thin[ix], thin_fac[ix], nburn[ix])

                self.RL_indep_thin = np.max(thin_fac)

                if feedback:
                    if not np.all(thin_fac != 0):
                        print('RL: Not enough samples to estimate convergence stats')
                    else:
                        print('RL: Thin for Markov: ', np.max(markov_thin))
                        print('RL: Thin for indep samples:  ', str(self.RL_indep_thin))
                        print('RL: Estimated burn in steps: ', np.max(nburn), ' (',
                              int(round(np.max(nburn) / self.mean_mult)), ' rows)')
                lines += "\n"

            if 'CorrSteps' in what:
                # Get correlation lengths.
                # We ignore the fact that there are jumps between chains, so slight underestimate
                lines += "Parameter auto-correlations as function of step separation\n"
                lines += "\n"
                if self.corr_length_thin != 0:
                    autocorr_thin = self.corr_length_thin
                else:
                    if self.indep_thin == 0:
                        autocorr_thin = 20
                    elif self.indep_thin <= 30:
                        autocorr_thin = 5
                    else:
                        autocorr_thin = int(5 * (self.indep_thin / 30))

                thin_ix = self.thin_indices(autocorr_thin)
                thin_rows = len(thin_ix)
                maxoff = int(min(self.corr_length_steps, thin_rows // (2 * num_chains_used)))

                if maxoff > 0:
                    corrs = np.zeros([maxoff, nparam])
                    for chain in chainlist:
                        thin_ix = chain.thin_indices(autocorr_thin)
                        thin_rows = len(thin_ix)
                        maxoff = min(maxoff, thin_rows // autocorr_thin)
                        for j in range(nparam):
                            diff = chain.diffs[j][thin_ix]
                            for off in range(1, maxoff + 1):
                                corrs[off - 1][j] += np.dot(diff[off:], diff[:-off]) / (thin_rows - off) / \
                                                     self.vars[j]
                    corrs /= len(chainlist)

                    lines += parForm % ""
                    for i in range(maxoff):
                        lines += "%8i" % ((i + 1) * autocorr_thin)
                    lines += "\n"
                    for j in range(nparam):
                        label = self.parLabel(j)
                        lines += parNames[j]
                        for i in range(maxoff):
                            lines += "%8.3f" % corrs[i][j]
                        lines += " %s\n" % label

        if writeDataToFile:
            with open(filename or (self.rootdirname + '.converge'), 'w', encoding='utf-8') as f:
                f.write(lines)
        return lines

    def _get1DNeff(self, par, param):
        N_eff = getattr(par, 'N_eff_kde', None)
        if N_eff is None:
            par.N_eff_kde = self.getEffectiveSamplesGaussianKDE(param, scale=par.sigma_range)
            N_eff = par.N_eff_kde
        return N_eff

    def getAutoBandwidth1D(self, bins, par, param, mult_bias_correction_order=None, kernel_order=1, N_eff=None):
        """
        Get optimized kernel density bandwidth (in units of the range of the bins)
        Based on optimal Improved Sheather-Jones bandwidth for basic Parzen kernel, then scaled if higher-order method
        being used. For details see the notes at `arXiv:1910.13970 <https://arxiv.org/abs/1910.13970>`_.

        :param bins: numpy array of binned weights for the samples
        :param par: A :class:`~.paramnames.ParamInfo` instance for the parameter to analyse
        :param param: index of the parameter to use
        :param mult_bias_correction_order: order of multiplicative bias correction (0 is basic Parzen kernel);
               by default taken from instance settings.
        :param kernel_order: order of the kernel
               (0 is Parzen, 1 does linear boundary correction, 2 is a higher-order kernel)
        :param N_eff: effective number of samples. If not specified estimated using weights, autocorrelations,
                      and fiducial bandwidth
        :return: kernel density bandwidth (in units the range of the bins)
        """
        if N_eff is None:
            N_eff = self._get1DNeff(par, param)
        h = kde.gaussian_kde_bandwidth_binned(bins, Neff=N_eff)
        bin_range = max(par.param_max, par.range_max) - min(par.param_min, par.range_min)
        if h is None or h < 0.01 * N_eff ** (-1. / 5) * (par.range_max - par.range_min) / bin_range:
            hnew = 1.06 * par.sigma_range * N_eff ** (-1. / 5) / bin_range
            if par.name not in self.no_warning_params \
                    and (not self.no_warning_chi2_params or 'chi2_' not in par.name and 'minuslog' not in par.name):
                msg = 'auto bandwidth for %s very small or failed (h=%s,N_eff=%s). Using fallback (h=%s)' % (
                    par.name, h, N_eff, hnew)
                if getattr(self, 'raise_on_bandwidth_errors', False):
                    raise BandwidthError(msg)
                else:
                    logging.warning(msg)
            h = hnew

        par.kde_h = h
        m = self.mult_bias_correction_order if mult_bias_correction_order is None else mult_bias_correction_order
        if kernel_order > 1:
            m = max(m, 1)
        if m:
            # higher order method
            # e.g.  http://biomet.oxfordjournals.org/content/82/2/327.full.pdf+html
            # some prefactors given in  http://eprints.whiterose.ac.uk/42950/6/taylorcc2%5D.pdf
            # Here we just take unit prefactor relative to Gaussian
            # and rescale the optimal h for standard KDE to accounts for higher order scaling
            # Should be about 1.3 x larger for Gaussian, but smaller in some other cases
            return h * N_eff ** (1. / 5 - 1. / (4 * m + 5))
        else:
            return h

    def getAutoBandwidth2D(self, bins, parx, pary, paramx, paramy, corr, rangex, rangey, base_fine_bins_2D,
                           mult_bias_correction_order=None, min_corr=0.2, N_eff=None, use_2D_Neff=False):
        """
        Get optimized kernel density bandwidth matrix in parameter units, using Improved Sheather Jones method in
        sheared parameters. The shearing is determined using the covariance, so you know the distribution is
        multi-modal, potentially giving 'fake' correlation, turn off shearing by setting min_corr=1.
        For details see the notes `arXiv:1910.13970 <https://arxiv.org/abs/1910.13970>`_.

        :param bins: 2D numpy array of binned weights
        :param parx: A :class:`~.paramnames.ParamInfo` instance for the x parameter
        :param pary: A :class:`~.paramnames.ParamInfo` instance for the y parameter
        :param paramx: index of the x parameter
        :param paramy: index of the y parameter
        :param corr: correlation of the samples
        :param rangex: scale in the x parameter
        :param rangey: scale in the y parameter
        :param base_fine_bins_2D: number of bins to use for re-binning in rotated parameter space
        :param mult_bias_correction_order: multiplicative bias correction order (0 is Parzen kernel); by default taken
                                           from instance settings
        :param min_corr: minimum correlation value at which to bother de-correlating the parameters
        :param N_eff: effective number of samples. If not specified, uses rough estimate that accounts for
                      weights and strongly-correlated nearby samples (see notes)
        :param use_2D_Neff: if N_eff not specified, whether to use 2D estimate of effective number, or approximate from
                            the 1D results (default from use_effective_samples_2D setting)
        :return: kernel density bandwidth matrix in parameter units
        """
        if N_eff is None:
            if (use_2D_Neff if use_2D_Neff is not None else self.use_effective_samples_2D) and abs(corr) < 0.999:
                # For multi-modal could overestimate width, and hence underestimate number of samples
                N_eff = self.getEffectiveSamplesGaussianKDE_2d(paramx, paramy)
            else:
                N_eff = min(self._get1DNeff(parx, paramx), self._get1DNeff(pary, paramy))

        logging.debug('%s %s AutoBandwidth2D: N_eff=%s, corr=%s', parx.name, pary.name, N_eff, corr)
        has_limits = parx.has_limits or pary.has_limits
        do_correlated = not parx.has_limits or not pary.has_limits

        def fallback_widths(ex):
            msg = '2D kernel density bandwidth optimizer failed for %s, %s. Using fallback width: %s' % (
                parx.name, pary.name, ex)
            if getattr(self, 'raise_on_bandwidth_errors', False):
                raise BandwidthError(msg)
            logging.warning(msg)
            _hx = parx.sigma_range / N_eff ** (1. / 6)
            _hy = pary.sigma_range / N_eff ** (1. / 6)
            return _hx, _hy, max(min(corr, self.max_corr_2D), -self.max_corr_2D)

        if min_corr < abs(corr) <= self.max_corr_2D and do_correlated:
            # 'shear' the data so fairly uncorrelated, making sure shear keeps any bounds on one parameter unchanged
            # the binning step will rescale to make roughly isotropic as assumed
            # by the 2D kernel optimizer psi_{ab} derivatives
            i, j = paramx, paramy
            imax, imin = None, None
            if parx.has_limits_bot:
                imin = parx.range_min
            if parx.has_limits_top:
                imax = parx.range_max
            if pary.has_limits:
                i, j = j, i
                if pary.has_limits_bot:
                    imin = pary.range_min
                if pary.has_limits_top:
                    imax = pary.range_max

            cov = self.getCov(pars=[i, j])
            S = np.linalg.cholesky(cov)
            ichol = np.linalg.inv(S)
            S *= ichol[0, 0]
            r = ichol[1, :] / ichol[0, 0]
            p1 = self.samples[:, i]
            p2 = r[0] * self.samples[:, i] + r[1] * self.samples[:, j]

            bin1, r1 = kde.bin_samples(p1, nbins=base_fine_bins_2D, range_min=imin, range_max=imax)
            bin2, r2 = kde.bin_samples(p2, nbins=base_fine_bins_2D)
            rotbins, _ = self._make2Dhist(bin1, bin2, base_fine_bins_2D, base_fine_bins_2D)
            try:
                opt = kde.KernelOptimizer2D(rotbins, N_eff, 0, do_correlation=not has_limits)
                hx, hy, c = opt.get_h()
                hx *= r1
                hy *= r2
                kernelC = S.dot(np.array([[hx ** 2, hx * hy * c], [hx * hy * c, hy ** 2]])).dot(S.T)
                hx, hy, c = np.sqrt(kernelC[0, 0]), np.sqrt(kernelC[1, 1]), kernelC[0, 1] / np.sqrt(
                    kernelC[0, 0] * kernelC[1, 1])
                if pary.has_limits:
                    hx, hy = hy, hx
                    # print 'derotated pars', hx, hy, c
            except ValueError as e:
                hx, hy, c = fallback_widths(e)
        elif abs(corr) > self.max_corr_2D or not do_correlated and corr > 0.8:
            c = max(min(corr, self.max_corr_2D), -self.max_corr_2D)
            hx = parx.sigma_range / N_eff ** (1. / 6)
            hy = pary.sigma_range / N_eff ** (1. / 6)
        else:
            try:
                opt = kde.KernelOptimizer2D(bins, N_eff, corr, do_correlation=not has_limits,
                                            fallback_t=(min(pary.sigma_range / rangey,
                                                            parx.sigma_range / rangex) / N_eff ** (1. / 6)) ** 2)
                hx, hy, c = opt.get_h()
                hx *= rangex
                hy *= rangey
            except ValueError as e:
                hx, hy, c = fallback_widths(e)

        if mult_bias_correction_order is None:
            mult_bias_correction_order = self.mult_bias_correction_order
        logging.debug('hx/sig, hy/sig, corr =%s, %s, %s', hx / parx.err, hy / pary.err, c)
        if mult_bias_correction_order:
            scale = 1.1 * N_eff ** (1. / 6 - 1. / (2 + 4 * (1 + mult_bias_correction_order)))
            hx *= scale
            hy *= scale
            logging.debug('hx/sig, hy/sig, corr, scale =%s, %s, %s, %s', hx / parx.err, hy / pary.err, c, scale)
        return hx, hy, c

    def _initParamRanges(self, j, paramConfid=None):
        if isinstance(j, str):
            j = self.index[j]
        paramVec = self.samples[:, j]
        return self._initParam(self.paramNames.names[j], paramVec, self.means[j], self.sddev[j], paramConfid)

    def _initParam(self, par, paramVec, mean=None, sddev=None, paramConfid=None):
        if mean is None:
            mean = paramVec.mean()
        if sddev is None:
            sddev = paramVec.std()
        par.err = sddev
        par.mean = mean
        par.param_min = np.min(paramVec)
        par.param_max = np.max(paramVec)
        paramConfid = paramConfid or self.initParamConfidenceData(paramVec)
        # sigma_range is estimate related to shape of structure in the distribution = std dev for Gaussian
        # search for peaks using quantiles,
        # e.g. like simplified version of Janssen 95 (http://dx.doi.org/10.1080/10485259508832654)
        confid_points = np.linspace(0.1, 0.9, 9)
        confids = self.confidence(paramConfid,
                                  np.array([self.range_confidence, 1 - self.range_confidence] + list(confid_points)))
        par.range_min, par.range_max = confids[0:2]
        confids[1:-1] = confids[2:]
        confids[0] = par.param_min
        confids[-1] = par.param_max
        diffs = confids[4:] - confids[:-4]
        scale = np.min(diffs) / 1.049
        if np.all(diffs > par.err * 1.049) and np.all(diffs < scale * 1.5):
            # very flat, can use bigger
            par.sigma_range = scale
        else:
            par.sigma_range = min(par.err, scale)
        if self.range_ND_contour >= 0 and self.likeStats:
            if self.range_ND_contour >= par.ND_limit_bot.size:
                raise SettingError("range_ND_contour should be -1 (off), or 0, 1 for first or second contour level")
            par.range_min = min(max(par.range_min - par.err, par.ND_limit_bot[self.range_ND_contour]), par.range_min)
            par.range_max = max(max(par.range_max + par.err, par.ND_limit_top[self.range_ND_contour]), par.range_max)

        smooth_1D = par.sigma_range * 0.4

        if par.has_limits_bot:
            if par.range_min - par.limmin > 2 * smooth_1D and par.param_min - par.limmin > smooth_1D:
                # long way from limit
                par.has_limits_bot = False
            else:
                par.range_min = par.limmin

        if par.has_limits_top:
            if par.limmax - par.range_max > 2 * smooth_1D and par.limmax - par.param_max > smooth_1D:
                par.has_limits_top = False
            else:
                par.range_max = par.limmax

        if not par.has_limits_bot:
            par.range_min -= smooth_1D * 2

        if not par.has_limits_top:
            par.range_max += smooth_1D * 2

        par.has_limits = par.has_limits_top or par.has_limits_bot

        return par

    def _binSamples(self, paramVec, par, num_fine_bins, borderfrac=0.1):

        # High resolution density (sampled many times per smoothing scale). First and last bins are half width

        border = (par.range_max - par.range_min) * borderfrac
        binmin = min(par.param_min, par.range_min)
        if not par.has_limits_bot:
            binmin -= border
        binmax = max(par.param_max, par.range_max)
        if not par.has_limits_top:
            binmax += border
        fine_width = (binmax - binmin) / (num_fine_bins - 1)
        ix = ((paramVec - binmin) / fine_width + 0.5).astype(int)
        return ix, fine_width, binmin, binmax

    def get1DDensity(self, name, **kwargs):
        """
        Returns a :class:`~.densities.Density1D` instance for parameter with given name. Result is cached.

        :param name: name of the parameter
        :param kwargs: arguments for :func:`~MCSamples.get1DDensityGridData`
        :return: A :class:`~.densities.Density1D` instance for parameter with given name
        """
        if self.needs_update:
            self.updateBaseStatistics()
        if not kwargs:
            density = self.density1D.get(name, None)
            if density is not None:
                return density
        return self.get1DDensityGridData(name, **kwargs)

    # noinspection PyUnboundLocalVariable
    def get1DDensityGridData(self, j, paramConfid=None, meanlikes=False, **kwargs):
        """
        Low-level function to get a :class:`~.densities.Density1D` instance for the marginalized 1D density
        of a parameter. Result is not cached.

        :param j: a name or index of the parameter
        :param paramConfid: optional cached :class:`~.chains.ParamConfidenceData` instance
        :param meanlikes: include mean likelihoods
        :param kwargs: optional settings to override instance settings of the same name (see `analysis_settings`):

               - **smooth_scale_1D**
               - **boundary_correction_order**
               - **mult_bias_correction_order**
               - **fine_bins**
               - **num_bins**
        :return: A :class:`~.densities.Density1D` instance
        """

        if self.needs_update:
            self.updateBaseStatistics()
        j = self._parAndNumber(j)[0]
        if j is None:
            return None

        par = self._initParamRanges(j, paramConfid)
        num_bins = kwargs.get('num_bins', self.num_bins)
        smooth_scale_1D = kwargs.get('smooth_scale_1D', self.smooth_scale_1D)
        boundary_correction_order = kwargs.get('boundary_correction_order', self.boundary_correction_order)
        mult_bias_correction_order = kwargs.get('mult_bias_correction_order', self.mult_bias_correction_order)
        fine_bins = kwargs.get('fine_bins', self.fine_bins)

        paramrange = par.range_max - par.range_min
        if paramrange <= 0:
            raise MCSamplesError('Parameter range is <= 0: ' + par.name)
        width = paramrange / (num_bins - 1)

        bin_indices, fine_width, binmin, binmax = self._binSamples(self.samples[:, j], par, fine_bins)
        bins = np.bincount(bin_indices, weights=self.weights, minlength=fine_bins)

        if meanlikes:
            if self.shade_likes_is_mean_loglikes:
                w = self.weights * self.loglikes
            else:
                w = self.weights * np.exp((self.mean_loglike - self.loglikes))
            finebinlikes = np.bincount(bin_indices, weights=w, minlength=fine_bins)

        if smooth_scale_1D <= 0:
            # Set automatically.
            bandwidth = self.getAutoBandwidth1D(bins, par, j, mult_bias_correction_order,
                                                boundary_correction_order) * (binmax - binmin)
            # for low sample numbers with big tails (e.g. from nested), prevent making too wide
            bandwidth = min(bandwidth, paramrange / 4)
            smooth_1D = bandwidth * abs(smooth_scale_1D) / fine_width

        elif smooth_scale_1D < 1.0:
            smooth_1D = smooth_scale_1D * par.err / fine_width
        else:
            smooth_1D = smooth_scale_1D * width / fine_width

        if smooth_1D < 2:
            logging.warning('fine_bins not large enough to well sample smoothing scale - ' + par.name)

        smooth_1D = min(max(1., smooth_1D), fine_bins // 2)

        logging.debug("%s 1D sigma_range, std: %s, %s; smooth_1D_bins: %s ", par.name, par.sigma_range, par.err,
                      smooth_1D)

        winw = min(int(round(2.5 * smooth_1D)), fine_bins // 2 - 2)
        kernel = Kernel1D(winw, smooth_1D)

        cache = {}
        conv = convolve1D(bins, kernel.Win, 'same', cache=cache)
        fine_x = np.linspace(binmin, binmax, fine_bins)
        density1D = Density1D(fine_x, P=conv, view_ranges=[par.range_min, par.range_max])

        if meanlikes:
            rawbins = conv.copy()

        if par.has_limits and boundary_correction_order >= 0:
            # correct for cuts allowing for normalization over window
            prior_mask = np.ones(fine_bins + 2 * winw)
            if par.has_limits_bot:
                prior_mask[winw] = 0.5
                prior_mask[: winw] = 0
            if par.has_limits_top:
                prior_mask[-(winw + 1)] = 0.5
                prior_mask[-winw:] = 0
            a0 = convolve1D(prior_mask, kernel.Win, 'valid', cache=cache)
            ix = np.nonzero(a0 * density1D.P)
            a0 = a0[ix]
            normed = density1D.P[ix] / a0
            if boundary_correction_order == 0:
                density1D.P[ix] = normed
            elif boundary_correction_order <= 2:
                # linear boundary kernel, e.g. Jones 1993, Jones and Foster 1996
                # www3.stat.sinica.edu.tw/statistica/oldpdf/A6n414.pdf after Eq 1b, expressed for general prior mask
                # cf arXiv:1411.5528
                xWin = kernel.Win * kernel.x
                a1 = convolve1D(prior_mask, xWin, 'valid', cache=cache)[ix]
                a2 = convolve1D(prior_mask, xWin * kernel.x, 'valid', cache=cache, cache_args=[1])[ix]
                xP = convolve1D(bins, xWin, 'same', cache=cache)[ix]
                if boundary_correction_order == 1:
                    corrected = (density1D.P[ix] * a2 - xP * a1) / (a0 * a2 - a1 ** 2)
                else:
                    # quadratic correction
                    a3 = convolve1D(prior_mask, xWin * kernel.x ** 2, 'valid', cache=cache, cache_args=[1])[ix]
                    a4 = convolve1D(prior_mask, xWin * kernel.x ** 3, 'valid', cache=cache, cache_args=[1])[ix]
                    x2P = convolve1D(bins, xWin * kernel.x, 'same', cache=cache, cache_args=[1])[ix]
                    denom = a4 * a2 * a0 - a4 * a1 ** 2 - a2 ** 3 - a3 ** 2 * a0 + 2 * a1 * a2 * a3
                    A = a4 * a2 - a3 ** 2
                    B = a2 * a3 - a4 * a1
                    C = a3 * a1 - a2 ** 2
                    corrected = (density1D.P[ix] * A + xP * B + x2P * C) / denom
                density1D.P[ix] = normed * np.exp(np.minimum(corrected / normed, 4) - 1)
            else:
                raise SettingError('Unknown boundary_correction_order (expected 0, 1, 2)')
        elif boundary_correction_order == 2:
            # higher order kernel
            # eg. see http://www.jstor.org/stable/2965571
            xWin2 = kernel.Win * kernel.x ** 2
            x2P = convolve1D(bins, xWin2, 'same', cache=cache)
            a2 = np.sum(xWin2)
            a4 = np.dot(xWin2, kernel.x ** 2)
            corrected = (density1D.P * a4 - a2 * x2P) / (a4 - a2 ** 2)
            ix = density1D.P > 0
            density1D.P[ix] *= np.exp(np.minimum(corrected[ix] / density1D.P[ix], 2) - 1)

        if mult_bias_correction_order:
            prior_mask = np.ones(fine_bins)
            if par.has_limits_bot:
                prior_mask[0] *= 0.5
            if par.has_limits_top:
                prior_mask[-1] *= 0.5
            a0 = convolve1D(prior_mask, kernel.Win, 'same', cache=cache, cache_args=[2])
            for _ in range(mult_bias_correction_order):
                # estimate using flattened samples to remove second order biases
                # mostly good performance, see http://www.jstor.org/stable/2965571 method 3,1 for first order
                prob1 = density1D.P.copy()
                prob1[prob1 == 0] = 1
                fine = bins / prob1
                conv = convolve1D(fine, kernel.Win, 'same', cache=cache, cache_args=[2])
                density1D.setP(density1D.P * conv)
                density1D.P /= a0

        density1D.normalize('max', in_place=True)
        if not kwargs:
            self.density1D[par.name] = density1D

        if meanlikes:
            ix = density1D.P > 0
            finebinlikes[ix] /= density1D.P[ix]
            binlikes = convolve1D(finebinlikes, kernel.Win, 'same', cache=cache, cache_args=[2])
            binlikes[ix] *= density1D.P[ix] / rawbins[ix]
            if self.shade_likes_is_mean_loglikes:
                maxbin = np.min(binlikes)
                binlikes = np.where((binlikes - maxbin) < 30, np.exp(-(binlikes - maxbin)), 0)
                binlikes[rawbins == 0] = 0
            binlikes /= np.max(binlikes)
            density1D.likes = binlikes
        else:
            density1D.likes = None

        return density1D

    def _setEdgeMask2D(self, parx, pary, prior_mask, winw, alledge=False):
        if parx.has_limits_bot:
            prior_mask[:, winw] /= 2
            prior_mask[:, :winw] = 0
        if parx.has_limits_top:
            prior_mask[:, -(winw + 1)] /= 2
            prior_mask[:, -winw:] = 0
        if pary.has_limits_bot:
            prior_mask[winw, :] /= 2
            prior_mask[:winw:] = 0
        if pary.has_limits_top:
            prior_mask[-(winw + 1), :] /= 2
            prior_mask[-winw:, :] = 0
        if alledge:
            prior_mask[:, :winw] = 0
            prior_mask[:, -winw:] = 0
            prior_mask[:winw:] = 0
            prior_mask[-winw:, :] = 0

    def _getScaleForParam(self, par):
        # Also ensures that the 1D limits are initialized
        density = self.get1DDensity(par)
        mn, mx, lim_bot, lim_top = density.getLimits(0.5, accuracy_factor=1)
        if lim_bot or lim_top:
            scale = (mx - mn) / 0.675
        else:
            scale = (mx - mn) / (2 * 0.675)
        return scale

    def _make2Dhist(self, ixs, iys, xsize, ysize):
        flatix = ixs + iys * xsize
        # note arrays are indexed y,x

        return np.bincount(flatix, weights=self.weights,
                           minlength=xsize * ysize).reshape((ysize, xsize)), flatix

    def get2DDensity(self, x, y, normalized=False, **kwargs):
        """
        Returns a :class:`~.densities.Density2D` instance with marginalized 2D density.

        :param x: index or name of x parameter
        :param y: index or name of y parameter
        :param normalized: if False, is normalized so the maximum is 1, if True, density is normalized
        :param kwargs: keyword arguments for the :func:`get2DDensityGridData` function
        :return: :class:`~.densities.Density2D` instance
        """
        if self.needs_update:
            self.updateBaseStatistics()
        density = self.get2DDensityGridData(x, y, get_density=True, **kwargs)
        if normalized:
            density.normalize(in_place=True)
        return density

    # noinspection PyUnboundLocalVariable
    def get2DDensityGridData(self, j, j2, num_plot_contours=None, get_density=False, meanlikes=False, **kwargs):
        """
        Low-level function to get 2D plot marginalized density and optional additional plot data.

        :param j: name or index of the x parameter
        :param j2: name or index of the y parameter.
        :param num_plot_contours: number of contours to calculate and return in density.contours
        :param get_density: only get the 2D marginalized density, don't calculate confidence level members
        :param meanlikes: calculate mean likelihoods as well as marginalized density
                          (returned as array in density.likes)
        :param kwargs: optional settings to override instance settings of the same name (see `analysis_settings`):

            - **fine_bins_2D**
            - **boundary_correction_order**
            - **mult_bias_correction_order**
            - **smooth_scale_2D**
        :return: a :class:`~.densities.Density2D` instance
        """
        if self.needs_update:
            self.updateBaseStatistics()
        start = time.time()
        j, parx = self._parAndNumber(j)
        j2, pary = self._parAndNumber(j2)
        if j is None or j2 is None:
            return None

        self._initParamRanges(j)
        self._initParamRanges(j2)

        base_fine_bins_2D = kwargs.get('fine_bins_2D', self.fine_bins_2D)
        boundary_correction_order = kwargs.get('boundary_correction_order', self.boundary_correction_order)
        mult_bias_correction_order = kwargs.get('mult_bias_correction_order', self.mult_bias_correction_order)
        smooth_scale_2D = float(kwargs.get('smooth_scale_2D', self.smooth_scale_2D))

        has_prior = parx.has_limits or pary.has_limits

        corr = self.getCorrelationMatrix()[j2][j]
        if corr == 1:
            logging.warning('Parameters are 100%% correlated: %s, %s', parx.name, pary.name)

        logging.debug('Doing 2D: %s - %s', parx.name, pary.name)
        logging.debug('sample x_err, y_err, correlation: %s, %s, %s', parx.err, pary.err, corr)

        # keep things simple unless obvious degeneracy
        if abs(self.max_corr_2D) > 1:
            raise SettingError('max_corr_2D cannot be >=1')
        if abs(corr) < 0.1:
            corr = 0.

        # for tight degeneracies increase bin density
        angle_scale = max(0.2, np.sqrt(1 - min(self.max_corr_2D, abs(corr)) ** 2))

        nbin2D = int(round(self.num_bins_2D / angle_scale))
        fine_bins_2D = base_fine_bins_2D
        if corr:
            scaled = 192 * int(3 / angle_scale) // 3
            if base_fine_bins_2D < scaled and int(1 / angle_scale) > 1:
                fine_bins_2D = scaled

        ixs, finewidthx, xbinmin, xbinmax = self._binSamples(self.samples[:, j], parx, fine_bins_2D)
        iys, finewidthy, ybinmin, ybinmax = self._binSamples(self.samples[:, j2], pary, fine_bins_2D)

        xsize = fine_bins_2D
        ysize = fine_bins_2D

        histbins, flatix = self._make2Dhist(ixs, iys, xsize, ysize)

        if meanlikes:
            likeweights = self.weights * np.exp(self.mean_loglike - self.loglikes)
            finebinlikes = np.bincount(flatix, weights=likeweights,
                                       minlength=xsize * ysize).reshape((ysize, xsize))

        # smooth_x and smooth_y should be in rotated bin units
        if smooth_scale_2D < 0:

            rx, ry, corr = self.getAutoBandwidth2D(histbins, parx, pary, j, j2, corr, xbinmax - xbinmin,
                                                   ybinmax - ybinmin,
                                                   base_fine_bins_2D,
                                                   mult_bias_correction_order=mult_bias_correction_order)

            rx = rx * abs(smooth_scale_2D) / finewidthx
            ry = ry * abs(smooth_scale_2D) / finewidthy
        elif smooth_scale_2D < 1.0:
            rx = smooth_scale_2D * parx.err / finewidthx
            ry = smooth_scale_2D * pary.err / finewidthy
        else:
            rx = smooth_scale_2D * fine_bins_2D / nbin2D
            ry = smooth_scale_2D * fine_bins_2D / nbin2D

        smooth_scale = float(max(rx, ry))
        logging.debug('corr, rx, ry: %s, %s, %s', corr, rx, ry)

        if smooth_scale < 2:
            logging.warning('fine_bins_2D not large enough for optimal density')

        winw = int(round(2.5 * smooth_scale))

        Cinv = np.linalg.inv(np.array([[ry ** 2, rx * ry * corr], [rx * ry * corr, rx ** 2]]))
        ix1, ix2 = np.mgrid[-winw:winw + 1, -winw:winw + 1]
        Win = np.exp(-(ix1 ** 2 * Cinv[0, 0] + ix2 ** 2 * Cinv[1, 1] + 2 * Cinv[1, 0] * ix1 * ix2) / 2)
        Win /= np.sum(Win)

        logging.debug('time 2D binning and bandwidth: %s ; bins: %s', time.time() - start, fine_bins_2D)
        start = time.time()
        cache = {}
        convolvesize = xsize + 2 * winw + Win.shape[0]
        bins2D = convolve2D(histbins, Win, 'same', largest_size=convolvesize, cache=cache)

        if meanlikes:
            bin2Dlikes = convolve2D(finebinlikes, Win, 'same', largest_size=convolvesize, cache=cache, cache_args=[2])
            if mult_bias_correction_order:
                ix = bin2Dlikes > 0
                finebinlikes[ix] /= bin2Dlikes[ix]
                likes2 = convolve2D(finebinlikes, Win, 'same', largest_size=convolvesize, cache=cache, cache_args=[2])
                likes2[ix] *= bin2Dlikes[ix]
                bin2Dlikes = likes2
            del finebinlikes
            mx = 1e-4 * np.max(bins2D)
            bin2Dlikes[bins2D > mx] /= bins2D[bins2D > mx]
            bin2Dlikes[bins2D <= mx] = 0
        else:
            bin2Dlikes = None

        if has_prior and boundary_correction_order >= 0:
            # Correct for edge effects
            prior_mask = np.ones((ysize + 2 * winw, xsize + 2 * winw))
            self._setEdgeMask2D(parx, pary, prior_mask, winw)
            a00 = convolve2D(prior_mask, Win, 'valid', largest_size=convolvesize, cache=cache)
            ix = a00 * bins2D > np.max(bins2D) * 1e-8
            a00 = a00[ix]
            normed = bins2D[ix] / a00
            if boundary_correction_order == 1:
                # linear boundary correction
                indexes = np.arange(-winw, winw + 1)
                y = np.empty(Win.shape)
                for i in range(Win.shape[0]):
                    y[:, i] = indexes
                winx = Win * indexes
                winy = Win * y
                a10 = convolve2D(prior_mask, winx, 'valid', largest_size=convolvesize, cache=cache)[ix]
                a01 = convolve2D(prior_mask, winy, 'valid', largest_size=convolvesize, cache=cache)[ix]
                a20 = convolve2D(prior_mask, winx * indexes, 'valid', largest_size=convolvesize, cache=cache,
                                 cache_args=[1])[ix]
                a02 = convolve2D(prior_mask, winy * y, 'valid', largest_size=convolvesize, cache=cache,
                                 cache_args=[1])[ix]
                a11 = convolve2D(prior_mask, winy * indexes, 'valid', largest_size=convolvesize, cache=cache,
                                 cache_args=[1])[ix]
                xP = convolve2D(histbins, winx, 'same', largest_size=convolvesize, cache=cache)[ix]
                yP = convolve2D(histbins, winy, 'same', largest_size=convolvesize, cache=cache)[ix]
                denom = (a20 * a01 ** 2 + a10 ** 2 * a02 - a00 * a02 * a20 + a11 ** 2 * a00 - 2 * a01 * a10 * a11)
                A = a11 ** 2 - a02 * a20
                Ax = a10 * a02 - a01 * a11
                Ay = a01 * a20 - a10 * a11
                corrected = (bins2D[ix] * A + xP * Ax + yP * Ay) / denom
                bins2D[ix] = normed * np.exp(np.minimum(corrected / normed, 4) - 1)
            elif boundary_correction_order == 0:
                # simple boundary correction by normalization
                bins2D[ix] = normed
            else:
                raise SettingError('unknown boundary_correction_order (expected 0 or 1)')

        if mult_bias_correction_order:
            prior_mask = np.ones((ysize + 2 * winw, xsize + 2 * winw))
            self._setEdgeMask2D(parx, pary, prior_mask, winw, alledge=True)
            a00 = convolve2D(prior_mask, Win, 'valid', largest_size=convolvesize, cache=cache, cache_args=[2])
            for _ in range(mult_bias_correction_order):
                box = histbins.copy()
                ix2 = bins2D > np.max(bins2D) * 1e-8
                box[ix2] /= bins2D[ix2]
                bins2D *= convolve2D(box, Win, 'same', largest_size=convolvesize, cache=cache, cache_args=[2])
                bins2D /= a00

        x = np.linspace(xbinmin, xbinmax, xsize)
        y = np.linspace(ybinmin, ybinmax, ysize)
        density = Density2D(x, y, bins2D,
                            view_ranges=[(parx.range_min, parx.range_max), (pary.range_min, pary.range_max)])
        density.normalize('max', in_place=True)
        if get_density:
            return density

        ncontours = len(self.contours)
        if num_plot_contours:
            ncontours = min(num_plot_contours, ncontours)
        contours = self.contours[:ncontours]

        logging.debug('time 2D convolutions: %s', time.time() - start)

        # Get contour containing contours(:) of the probability
        density.contours = density.getContourLevels(contours)

        if meanlikes:
            bin2Dlikes /= np.max(bin2Dlikes)
            density.likes = bin2Dlikes
        else:
            density.likes = None

        return density

    # This ND code was contributed but not updated, and currently seems not to work; welcome pull request to restore
    # def _setRawEdgeMaskND(self, parv, prior_mask):
    #     ndim = len(parv)
    #     vrap = parv[::-1]
    #     mskShape = prior_mask.shape
    #
    #     if len(mskShape) != ndim:
    #         raise ValueError("parv and prior_mask or different sizes!")
    #
    #     # create a slice object iterating over everything
    #     mskSlices = [slice(None) for _ in range(ndim)]
    #
    #     for i in range(ndim):
    #         if vrap[i].has_limits_bot:
    #             mskSlices[i] = 0
    #             prior_mask[mskSlices] /= 2
    #             mskSlices[i] = slice(None)
    #
    #         if vrap[i].has_limits_top:
    #             mskSlices[i] = mskShape[i] - 1
    #             prior_mask[mskSlices] /= 2
    #             mskSlices[i] = slice(None)
    #
    # def _flattenValues(self, ixs, xsizes):
    #     ndim = len(ixs)
    #
    #     q = ixs[0]
    #     for i in range(1, ndim):
    #         q = q + np.prod(xsizes[0:i]) * ixs[i]
    #     return q
    #
    # def _unflattenValues(self, q, xsizes):
    #     ndim = len(xsizes)
    #
    #     ixs = list([np.array(q) for _ in range(ndim)])
    #
    #     if ndim == 1:
    #         ixs[0] = q
    #         return ixs
    #
    #     ixs[ndim - 1] = q / np.prod(xsizes[0:ndim - 1])
    #
    #     acc = 0
    #     for k in range(ndim - 2, -1, -1):
    #         acc = acc + ixs[k + 1] * np.prod(xsizes[0:k + 1])
    #         if k > 0:
    #             ixs[k] = (q - acc) / np.prod(xsizes[0:k])
    #         else:
    #             ixs[k] = q - acc
    #
    #     return ixs
    #
    # def _makeNDhist(self, ixs, xsizes):
    #
    #     if len(ixs) != len(xsizes):
    #         raise ValueError('index and size arrays are of unequal length')
    #
    #     flatixv = self._flattenValues(ixs, xsizes)
    #
    #     # to be removed debugging only
    #     if np.count_nonzero(np.asarray(ixs) - self._unflattenValues(flatixv, xsizes)) != 0:
    #         raise ValueError('ARG!!! flatten/unflatten screwed')
    #
    #     # note arrays are indexed y,x
    #     return np.bincount(flatixv, weights=self.weights,
    #                        minlength=np.prod(xsizes)).reshape(xsizes[::-1], order='C'), flatixv
    #
    # def getRawNDDensity(self, xs, normalized=False, **kwargs):
    #     """
    #     Returns a :class:`~.densities.DensityND` instance with marginalized ND density.
    #
    #     :param xs: indices or names of x_i parameters
    #     :param normalized: if False, is normalized so the maximum is 1, if True, density is normalized
    #     :param kwargs: keyword arguments for the :meth:`~.mcsamples.MCSamples.getRawNDDensityGridData` function
    #     :return: :class:`~.densities.DensityND` instance
    #     """
    #     if self.needs_update:
    #         self.updateBaseStatistics()
    #     density = self.getRawNDDensityGridData(xs, get_density=True, **kwargs)
    #     if normalized:
    #         density.normalize(in_place=True)
    #     return density
    #
    # # noinspection PyUnresolvedReferences
    # def getRawNDDensityGridData(self, js, num_plot_contours=None, get_density=False,
    #                             meanlikes=False, maxlikes=False, **kwargs):
    #     """
    #     Low-level function to get unsmooth ND plot marginalized
    #     density and optional additional plot data (no KDE).
    #
    #     :param js: vector of names or indices of the x_i parameters
    #     :param num_plot_contours: number of contours to calculate and return in density.contours
    #     :param get_density: only get the ND marginalized density, no additional plot data, no contours.
    #     :param meanlikes: calculate mean likelihoods as well as marginalized density
    #                      (returned as array in density.likes)
    #     :param maxlikes: calculate the profile likelihoods in addition to the others
    #                      (returned as array in density.maxlikes)
    #     :param kwargs: optional settings to override instance settings of the same name (see `analysis_settings`):
    #
    #     :return: a :class:`~.densities.DensityND` instance
    #     """
    #
    #     if self.needs_update:
    #         self.updateBaseStatistics()
    #
    #     ndim = len(js)
    #
    #     jv, parv = zip(*[self._parAndNumber(j) for j in js])
    #
    #     if None in jv:
    #         return None
    #
    #     [self._initParamRanges(j) for j in jv]
    #
    #     boundary_correction_order = kwargs.get('boundary_correction_order', self.boundary_correction_order)
    #     has_prior = any(parv[i].has_limits for i in range(ndim))
    #
    #     nbinsND = kwargs.get('num_bins_ND', self.num_bins_ND)
    #     ixv, widthv, xminv, xmaxv = zip(*[self._binSamples(self.samples[:, jv[i]],
    #                                                        parv[i], nbinsND) for i in range(ndim)])
    #
    #     # could also be non-equals over the dimensions
    #     xsizev = nbinsND * np.ones(ndim, dtype=np.int)
    #
    #     binsND, flatixv = self._makeNDhist(ixv, xsizev)
    #
    #     if has_prior and boundary_correction_order >= 0:
    #         # Correct for edge effects
    #         prior_mask = np.ones(xsizev[::-1])
    #         self._setRawEdgeMaskND(parv, prior_mask)
    #         binsND /= prior_mask
    #
    #     if meanlikes:
    #         likeweights = self.weights * np.exp(self.mean_loglike - self.loglikes)
    #         binNDlikes = np.bincount(flatixv, weights=likeweights,
    #                                  minlength=np.prod(xsizev)).reshape(xsizev[::-1], order='C')
    #     else:
    #         binNDlikes = None
    #
    #     if maxlikes:
    #         binNDmaxlikes = np.zeros(binsND.shape)
    #         ndindex = zip(*[ixv[i] for i in range(ndim)[::-1]])
    #         bestfit = np.max(-self.loglikes)
    #
    #         for irec in range(len(self.loglikes)):
    #             binNDmaxlikes[ndindex[irec]] = max(binNDmaxlikes[ndindex[irec]],
    #                                                np.exp(-bestfit - self.loglikes[irec]))
    #     else:
    #         binNDmaxlikes = None
    #
    #     xv = [np.linspace(xminv[i], xmaxv[i], xsizev[i]) for i in range(ndim)]
    #     views = [(parv[i].range_min, parv[i].range_max) for i in range(ndim)]
    #
    #     density = DensityND(xv, binsND, view_ranges=views)
    #
    #     # density.normalize('integral', in_place=True)
    #     density.normalize('max', in_place=True)
    #     if get_density:
    #         return density
    #
    #     ncontours = len(self.contours)
    #     if num_plot_contours:
    #         ncontours = min(num_plot_contours, ncontours)
    #     contours = self.contours[:ncontours]
    #
    #     # Get contour containing contours(:) of the probability
    #     density.contours = density.getContourLevels(contours)
    #
    #     if meanlikes:
    #         binNDlikes /= np.max(binNDlikes)
    #         density.likes = binNDlikes
    #     else:
    #         density.likes = None
    #
    #     if maxlikes:
    #         density.maxlikes = binNDmaxlikes
    #         density.maxcontours = getOtherContourLevels(binNDmaxlikes, contours, half_edge=False)
    #     else:
    #         density.maxlikes = None
    #
    #     return density

    def _setLikeStats(self):
        """
        Get and store LikeStats (see :func:`MCSamples.getLikeStats`)
        """
        if self.loglikes is None:
            self.likeStats = None
            return None
        m = types.LikeStats()
        bestfit_ix = np.argmin(self.loglikes)
        maxlike = self.loglikes[bestfit_ix]
        m.logLike_sample = maxlike
        if np.max(self.loglikes) - maxlike < 30:
            m.logMeanInvLike = np.log(self.mean(np.exp(self.loglikes - maxlike))) + maxlike
        else:
            m.logMeanInvLike = None

        m.meanLogLike = self.mean_loglike
        m.logMeanLike = -np.log(self.mean(np.exp(-(self.loglikes - maxlike)))) + maxlike
        # assuming maxlike is well determined
        m.complexity = 2 * (self.mean_loglike - maxlike)

        m.names = self.paramNames.names

        # get N-dimensional confidence region
        indexes = self.loglikes.argsort()
        cumsum = np.cumsum(self.weights[indexes])
        m.ND_cont1, m.ND_cont2 = np.searchsorted(cumsum, self.norm * self.contours[0:2])
        for j, par in enumerate(self.paramNames.names):
            region1 = self.samples[indexes[:m.ND_cont1], j]
            region2 = self.samples[indexes[:m.ND_cont2], j]
            par.ND_limit_bot = np.array([np.min(region1), np.min(region2)])
            par.ND_limit_top = np.array([np.max(region1), np.max(region2)])
            par.bestfit_sample = self.samples[bestfit_ix][j]

        self.likeStats = m
        return m

    def _readRanges(self):
        if self.root:
            ranges_file = self.root + '.ranges'
            if os.path.isfile(ranges_file):
                self.ranges = ParamBounds(ranges_file)
                return
            ranges_file = cobaya_interface.cobaya_params_file(self.root)
            if ranges_file:
                self.ranges = ParamBounds(ranges_file)
                return

        self.ranges = ParamBounds()

    def getBounds(self):
        """
        Returns the bounds in the form of a :class:`~.parampriors.ParamBounds` instance, for example
        for determining plot ranges

        Bounds are not  the same as self.ranges, as if samples are not near the range boundary, the bound is set to None

        :return: a :class:`~.parampriors.ParamBounds` instance
        """
        bounds = ParamBounds()
        bounds.names = self.paramNames.list()
        for par in self.paramNames.names:
            if par.has_limits_bot:
                bounds.lower[par.name] = par.limmin
            if par.has_limits_top:
                bounds.upper[par.name] = par.limmax
        return bounds

    def getUpper(self, name):
        """
        Return the upper limit of the parameter with the given name.

        :param name: parameter name
        :return: The upper limit if name exists, None otherwise.
        """
        par = self.paramNames.parWithName(name)
        if par:
            return getattr(par, 'limmax', None)
        return None

    def getLower(self, name):
        """
        Return the lower limit of the parameter with the given name.

        :param name: parameter name
        :return: The lower limit if name exists, None otherwise.
        """
        par = self.paramNames.parWithName(name)
        if par:
            return getattr(par, 'limmin', None)
        return None

    def getBestFit(self, max_posterior=True):
        """
        Returns a :class:`~.types.BestFit` object with best-fit point stored in .minimum or .bestfit file

       :param max_posterior: whether to get maximum posterior (from .minimum file)
                             or maximum likelihood (from .bestfit file)
       :return:
        """
        ext = '.minimum' if max_posterior else '.bestfit'
        bf_file = self.root + ext
        if os.path.exists(bf_file):
            return types.BestFit(bf_file, max_posterior=max_posterior)
        else:
            raise MCSamplesError('Best fit can only be included if loaded from file and file_root%s exists '
                                 '(cannot be calculated from samples)' % ext)

    def getMargeStats(self, include_bestfit=False):
        """
        Returns a :class:`~.types.MargeStats` object with marginalized 1D parameter constraints

        :param include_bestfit: if True, set best fit values by loading from root_name.minimum file (assuming it exists)
        :return: A :class:`~.types.MargeStats` instance
        """
        self._setDensitiesandMarge1D()
        m = types.MargeStats()
        m.hasBestFit = False
        m.limits = self.contours
        m.names = self.paramNames.names
        if include_bestfit:
            m.addBestFit(self.getBestFit())
        return m

    def getLikeStats(self):
        """
        Get best fit sample and n-D confidence limits, and various likelihood based statistics

        :return: a :class:`~.types.LikeStats` instance storing N-D limits for parameter i in
                 result.names[i].ND_limit_top, result.names[i].ND_limit_bot, and best-fit sample value
                 in result.names[i].bestfit_sample
        """
        return self.likeStats or self._setLikeStats()

    def getTable(self, columns=1, include_bestfit=False, **kwargs):
        """
        Creates and returns a :class:`~.types.ResultTable` instance. See also :func:`~MCSamples.getInlineLatex`.

        :param columns: number of columns in the table
        :param include_bestfit: True if should include the bestfit parameter values (assuming set)
        :param kwargs: arguments for :class:`~.types.ResultTable` constructor.
        :return: A :class:`~.types.ResultTable` instance
        """
        return types.ResultTable(columns, [self.getMargeStats(include_bestfit)], **kwargs)

    def getLatex(self, params=None, limit=1, err_sig_figs=None):
        """
        Get tex snippet for constraints on a list of parameters

        :param params: list of parameter names, or a single parameter name
        :param limit: which limit to get, 1 is the first (default 68%), 2 is the second
                     (limits array specified by self.contours)
        :param err_sig_figs: significant figures in the error
        :return: labels, texs: a list of parameter labels, and a list of tex snippets,
                               or for a single parameter, the latex snippet.
        """
        if isinstance(params, str):
            return self.getInlineLatex(params, limit, err_sig_figs)

        marge = self.getMargeStats()
        if params is None:
            params = marge.list()

        formatter = types.NoLineTableFormatter()
        if err_sig_figs:
            formatter.numberFormatter.err_sf = err_sig_figs
        texs = []
        labels = []
        for par in params:
            tex = marge.texValues(formatter, par, limit=limit)
            if tex is not None:
                texs.append(tex[0])
                labels.append((par if isinstance(par, ParamInfo) else marge.parWithName(par)).getLabel())
            else:
                texs.append(None)
                labels.append(None)

        return labels, texs

    def getInlineLatex(self, param, limit=1, err_sig_figs=None):
        r"""
        Get snippet like: A=x\\pm y. Will adjust appropriately for one and two tail limits.

        :param param: The name of the parameter
        :param limit: which limit to get, 1 is the first (default 68%), 2 is the second
                     (limits array specified by self.contours)
        :param err_sig_figs: significant figures in the error
        :return: The tex snippet.
        """
        labels, texs = self.getLatex([param], limit, err_sig_figs)
        if texs[0] is None:
            raise ValueError('parameter %s not found' % param)
        if not texs[0][0] in ['<', '>']:
            return labels[0] + ' = ' + texs[0]
        else:
            return labels[0] + ' ' + texs[0]

    def _setDensitiesandMarge1D(self, max_frac_twotail=None, meanlikes=False):
        """
        Get all the 1D densities; result is cached.

        :param max_frac_twotail: optional override for self.max_frac_twotail
        :param meanlikes: include mean likelihoods
        """
        if self.done_1Dbins:
            return

        for j in range(self.n):
            paramConfid = self.initParamConfidenceData(self.samples[:, j])
            self.get1DDensityGridData(j, paramConfid=paramConfid, meanlikes=meanlikes)
            self._setMargeLimits(self.paramNames.names[j], paramConfid, max_frac_twotail)

        self.done_1Dbins = True

    # noinspection PyUnboundLocalVariable
    def _setMargeLimits(self, par, paramConfid, max_frac_twotail=None, density1D=None):
        """
        Get limits, one or two tail depending on whether posterior
        goes to zero at the limits or not

        :param par:  The :class:`~.paramnames.ParamInfo` to set limits for
        :param paramConfid: :class:`~.chains.ParamConfidenceData` instance
        :param max_frac_twotail: optional override for self.max_frac_twotail
        :param density1D: any existing density 1D instance to use
        """
        if max_frac_twotail is None:
            max_frac_twotail = self.max_frac_twotail
        par.limits = []
        density1D = density1D or self.get1DDensity(par.name)
        interpGrid = None
        for ix1, contour in enumerate(self.contours):

            marge_limits_bot = par.has_limits_bot and not self.force_twotail and density1D.P[0] > max_frac_twotail[ix1]
            marge_limits_top = par.has_limits_top and not self.force_twotail and density1D.P[-1] > max_frac_twotail[ix1]

            if not marge_limits_bot or not marge_limits_top:
                # give limit
                if not interpGrid:
                    interpGrid = density1D.initLimitGrids()
                tail_limit_bot, tail_limit_top, marge_limits_bot, marge_limits_top = density1D.getLimits(contour,
                                                                                                         interpGrid)
                limfrac = 1 - contour

                if marge_limits_bot:
                    # fix to end of prior range
                    tail_limit_bot = par.range_min
                elif marge_limits_top:
                    # 1 tail limit
                    tail_limit_bot = self.confidence(paramConfid, limfrac, upper=False)
                else:
                    # 2 tail limit
                    tail_confid_bot = self.confidence(paramConfid, limfrac / 2, upper=False)

                if marge_limits_top:
                    tail_limit_top = par.range_max
                elif marge_limits_bot:
                    tail_limit_top = self.confidence(paramConfid, limfrac, upper=True)
                else:
                    tail_confid_top = self.confidence(paramConfid, limfrac / 2, upper=True)

                if not marge_limits_bot and not marge_limits_top:
                    # Two tail, check if limits are at very different density
                    if (math.fabs(density1D.Prob(tail_confid_top) -
                                  density1D.Prob(tail_confid_bot)) < self.credible_interval_threshold):
                        tail_limit_top = tail_confid_top
                        tail_limit_bot = tail_confid_bot

                lim = [tail_limit_bot, tail_limit_top]
            else:
                # no limit
                lim = [par.range_min, par.range_max]

            if marge_limits_bot and marge_limits_top:
                tag = 'none'
            elif marge_limits_bot:
                tag = '>'
            elif marge_limits_top:
                tag = '<'
            else:
                tag = 'two'
            par.limits.append(types.ParamLimit(lim, tag))

    def getCorrelatedVariable2DPlots(self, num_plots=12, nparam=None):
        """
        Gets a list of most correlated variable pair names.

        :param num_plots: The number of plots
        :param nparam: maximum number of pairs to get
        :return: list of [x,y] pair names
        """
        nparam = nparam or self.paramNames.numNonDerived()
        try_t = 1e5
        x, y = 0, 0
        cust2DPlots = []
        correlationMatrix = self.correlationMatrix
        for _ in range(num_plots):
            try_b = -1e5
            for ix1 in range(nparam):
                for ix2 in range(ix1 + 1, nparam):
                    if try_b < abs(correlationMatrix[ix1][ix2]) < try_t:
                        try_b = abs(correlationMatrix[ix1][ix2])
                        x, y = ix1, ix2
            if try_b == -1e5:
                break
            try_t = try_b
            cust2DPlots.append([self.parName(x), self.parName(y)])

        return cust2DPlots

    def addDerived(self, paramVec, name, label='', comment='', range=None):
        """
        Adds a new derived parameter

        :param paramVec: The vector of parameter values to add. For example a combination of
                         parameter arrays from MCSamples.getParams()
        :param name: The name for the new parameter
        :param label: optional latex label for the parameter
        :param comment: optional comment describing the parameter
        :param range: if specified, a tuple of min, max values for the new parameter hard prior bounds
                      (either can be None for one-side bound)
        :return: The added parameter's :class:`~.paramnames.ParamInfo` object
        """

        if range is not None:
            self.ranges.setRange(name, range)
        return super().addDerived(paramVec, name, label=label, comment=comment)

    def getParamBestFitDict(self, best_sample=False, want_derived=True, want_fixed=True, max_posterior=True):
        """
        Gets an ordered dictionary of parameter values for the best fit point,
        assuming calculated results from mimimization runs in .minimum (max posterior) .bestfit (max likelihood)
        files exists.

        Can also get the best-fit (max posterior) sample, which typically has a likelihood that differs significantly
        from the true best fit in high dimensions.

        :param best_sample: load from global minimum files (False, default) or using maximum posterior sample (True)
        :param want_derived: include derived parameters
        :param want_fixed: also include values of any fixed parameters
        :param max_posterior: whether to get maximum posterior (from .minimum file) or maximum likelihood
                             (from .bestfit file)
        :return: ordered dictionary of parameter values
        """
        if best_sample:
            if not max_posterior:
                raise ValueError('best_fit_sample is only maximum posterior')
            return self.getParamSampleDict(np.argmin(self.loglikes))
        else:
            res = self.getBestFit(max_posterior=max_posterior).getParamDict(include_derived=want_derived)
        if want_fixed:
            res.update(self.ranges.fixedValueDict())
        return res

    def getParamSampleDict(self, ix, want_derived=True, want_fixed=True):
        """
        Gets a dictionary of parameter values for sample number ix

        :param ix: index of the sample to return (zero based)
        :param want_derived: include derived parameters
        :param want_fixed: also include values of any fixed parameters
        :return: ordered dictionary of parameter values
        """
        res = super().getParamSampleDict(ix, want_derived=want_derived)
        if want_fixed:
            res.update(self.ranges.fixedValueDict())
        return res

    def getCombinedSamplesWithSamples(self, samps2, sample_weights=(1, 1)):
        """
        Make a new  :class:`MCSamples` instance by appending samples from samps2 for parameters which are in common.
        By default they are weighted so that the probability mass of each set of samples is the same,
        independent of tha actual sample sizes. The Weights parameter can be adjusted to change the
        relative weighting.
        :param samps2:  :class:`MCSamples` instance to merge
        :param sample_weights: relative weights for combining the samples. Set to None to just directly append samples.
        :return: a new  :class:`MCSamples` instance with the combined samples
        """

        params = ParamNames()
        params.names = [ParamInfo(name=p.name, label=p.label, derived=p.isDerived) for p in samps2.paramNames.names if
                        p.name in self.paramNames.list()]
        if self.loglikes is not None and samps2.loglikes is not None:
            loglikes = np.concatenate([self.loglikes, samps2.loglikes])
        else:
            loglikes = None
        if sample_weights is None:
            fac = 1
            sample_weights = (1, 1)
        else:
            fac = np.sum(self.weights) / np.sum(samps2.weights)
        weights = np.concatenate([self.weights * sample_weights[0], samps2.weights * sample_weights[1] * fac])
        p1 = self.getParams()
        p2 = samps2.getParams()
        samples = np.array([np.concatenate([getattr(p1, name), getattr(p2, name)]) for name in params.list()]).T
        samps = MCSamples(samples=samples, weights=weights, loglikes=loglikes, paramNamesFile=params, ignore_rows=0,
                          ranges=self.ranges, settings=copy.deepcopy(self.ini.params))
        return samps

    def saveTextMetadata(self, root, properties=None):
        """
        Saves metadata about the sames to text files with given file root

        :param root: root file name
        :param properties: optional dictiory of values to save in root.properties.ini
        """
        super().saveTextMetadata(root)
        self.ranges.saveToFile(root + '.ranges')
        ini_name = root + '.properties.ini'
        if properties or self.properties and self.properties.params or self.label:
            if os.path.exists(ini_name):
                ini = IniFile(ini_name)
            else:
                ini = IniFile()
            if self.properties:
                ini.params.update(self.properties.params)
            if self.label:
                ini.params.update({'label': self.label})
            ini.params.update(properties or {})
            ini.saveFile(ini_name)
        elif os.path.exists(ini_name):
            os.remove(ini_name)

    def saveChainsAsText(self, root, make_dirs=False, properties=None):
        if self.chains is None:
            chain_list = self.getSeparateChains()
        else:
            chain_list = self.chains
        for i, chain in enumerate(chain_list):
            chain.saveAsText(root, i, make_dirs)
        self.saveTextMetadata(root, properties)

    # Write functions for console script
    def _writeScriptPlots1D(self, filename, plotparams=None, ext=None):
        """
        Write a script that generates a 1D plot. Only intended for use by getdist script.

        :param filename: The filename to write to.
        :param plotparams: The list of parameters to plot (default: all)
        :param ext: The extension for the filename, Default if None
        """
        text = 'markers = ' + (str(self.markers) if self.markers else 'None') + '\n'
        if plotparams:
            text += 'g.plots_1d(roots,[' + ",".join(['\'' + par + '\'' for par in plotparams]) + '], markers=markers)'
        else:
            text += 'g.plots_1d(roots, markers=markers)'
        self._WritePlotFile(filename, self.subplot_size_inch, text, '', ext)

    def _writeScriptPlots2D(self, filename, plot_2D_param=None, cust2DPlots=(), ext=None):
        """
        Write script that generates a 2 dimensional plot. Only intended for use by getdist script.

        :param filename: The filename to write to.
        :param plot_2D_param: parameter to plot other variables against
        :param cust2DPlots: list of parts of parameter names to plot
        :param ext: The extension for the filename, Default if None
        :return: A dictionary indexed by pairs of parameters where 2D densities have been calculated
        """
        done2D = {}
        text = 'pairs=[]\n'
        plot_num = 0
        if len(cust2DPlots):
            cuts = [par1 + '__' + par2 for par1, par2 in cust2DPlots]
        for j, par1 in enumerate(self.paramNames.list()):
            if plot_2D_param or cust2DPlots:
                if par1 == plot_2D_param:
                    continue
                j2min = 0
            else:
                j2min = j + 1

            for j2 in range(j2min, self.n):
                par2 = self.parName(j2)
                if plot_2D_param and par2 != plot_2D_param:
                    continue
                # noinspection PyUnboundLocalVariable
                if len(cust2DPlots) and (par1 + '__' + par2) not in cuts:
                    continue
                if (par1, par2) not in done2D:
                    plot_num += 1
                    done2D[(par1, par2)] = True
                    text += "pairs.append(['%s','%s'])\n" % (par1, par2)
        text += 'g.plots_2d(roots,param_pairs=pairs,filled=True)'
        self._WritePlotFile(filename, self.subplot_size_inch2, text, '_2D', ext)
        return done2D

    def _writeScriptPlotsTri(self, filename, triangle_params, ext=None):
        """
        Write a script that generates a triangle plot. Only intended for use by getdist script.

        :param filename: The filename to write to.
        :param triangle_params: list of parameter names to plot
        :param ext: The extension for the filename, Default if None
        """
        text = 'g.triangle_plot(roots, %s)' % triangle_params
        self._WritePlotFile(filename, self.subplot_size_inch, text, '_tri', ext)

    def _writeScriptPlots3D(self, filename, plot_3D, ext=None):
        """
        Writes a script that generates a 3D (coloured-scatter) plot. Only intended for use by getdist script.

        :param filename: The filename to write to
        :param plot_3D: list of [x,y,z] parameters for the 3 Dimensional plots
        :param ext: The extension for the filename, Default if None
        """
        text = 'sets=[]\n'
        for pars in plot_3D:
            text += "sets.append(['%s','%s','%s'])\n" % tuple(pars)
        text += 'g.plots_3d(roots,sets)'
        self._WritePlotFile(filename, self.subplot_size_inch3, text, '_3D', ext)

    def _WritePlotFile(self, filename, subplot_size, text, tag, ext=None):
        """
        Write plot file.
        Used by other functions

        :param filename: The filename to write to
        :param subplot_size: The size of the subplot.
        :param text: The text to write after the headers.
        :param tag: Tag used for the filename the created file will export to.
        :param ext: The extension for the filename, Default if None
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("import getdist.plots as plots, os\n")
            f.write("g=plots.GetDistPlotter(chain_dir=r'%s')\n" % (self.batch_path or os.path.dirname(self.root)))

            f.write("g.settings.set_with_subplot_size(%s)\n" % subplot_size)
            f.write("roots = ['%s']\n" % self.rootname)
            f.write(text + '\n')
            ext = ext or self.plot_output
            fname = self.rootname + tag + '.' + ext
            f.write("g.export(os.path.join(r'%s',r'%s'))\n" % (self.out_dir, fname))


# Useful functions


def getRootFileName(rootdir):
    """
    Gets the root name of chains in given directory (assuming only one set of chain files).

    :param rootdir: The directory to check
    :return: The root file name.
    """
    rootFileName = ""
    pattern = os.path.join(rootdir, '*_*.txt')
    chain_files = glob.glob(pattern)
    chain_files.sort()
    if chain_files:
        chain_file0 = chain_files[0]
        rindex = chain_file0.rindex('_')
        rootFileName = chain_file0[:rindex]
    return rootFileName


def _dummy_usage():
    assert MCSamplesFromCobaya and ParamError
