from __future__ import absolute_import
from __future__ import print_function
import os
import glob
import logging
import copy
import pickle
import math
import time
import numpy as np
from scipy.stats import norm
import getdist
from getdist import chains, types, covmat, ParamInfo, IniFile
from getdist.densities import Density1D, Density2D
from getdist.chains import Chains, chainFiles, lastModified
from getdist.convolve import convolve1D, convolve2D
import getdist.kde_bandwidth as kde
from getdist.parampriors import ParamBounds
import six

pickle_version = 21


class MCSamplesError(Exception):
    pass


class SettingError(MCSamplesError):
    pass


class ParamError(MCSamplesError):
    pass


def loadMCSamples(file_root, ini=None, jobItem=None, no_cache=False, dist_settings={}):
    files = chainFiles(file_root)
    path, name = os.path.split(file_root)
    path = getdist.cache_dir or path
    if not os.path.exists(path): os.mkdir(path)
    cachefile = os.path.join(path, name) + '.py_mcsamples'
    samples = MCSamples(file_root, jobItem=jobItem, ini=ini, settings=dist_settings)
    allfiles = files + [file_root + '.ranges', file_root + '.paramnames']
    if not no_cache and os.path.exists(cachefile) and lastModified(allfiles) < os.path.getmtime(cachefile):
        try:
            with open(cachefile, 'rb') as inp:
                cache = pickle.load(inp)
            if cache.version == pickle_version and samples.ignore_rows == cache.ignore_rows:
                changed = len(samples.contours) != len(cache.contours) or \
                          np.any(np.array(samples.contours) != np.array(cache.contours))
                cache.updateSettings(ini=ini, settings=dist_settings, doUpdate=changed)
                return cache
        except:
            pass
    if not len(files):
        raise IOError('No chains found: ' + file_root)
    samples.readChains(files)
    samples.savePickle(cachefile)
    return samples


class Kernel1D(object):
    def __init__(self, winw, h):
        self.winw = winw
        self.h = h
        self.x = np.arange(-winw, winw + 1)
        Win = np.exp(-(self.x / h) ** 2 / 2.)
        self.Win = Win / np.sum(Win)


# =============================================================================

class MCSamples(Chains):
    def __init__(self, root=None, jobItem=None, ini=None, settings=None, ranges=None, **kwargs):
        Chains.__init__(self, root, jobItem=jobItem, **kwargs)

        self.version = pickle_version

        self.markers = {}

        self.ini = ini

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
        self.boundary_correction_order = 1
        self.mult_bias_correction_order = 1
        self.max_corr_2D = 0.95
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
        self.ignore_rows = float(kwargs.get('ignore_rows', 0))
        self.subplot_size_inch = 4.0
        self.subplot_size_inch2 = self.subplot_size_inch
        self.subplot_size_inch3 = 6.0
        self.plot_output = getdist.default_plot_output
        self.out_dir = ""

        self.max_split_tests = 4
        self.force_twotail = False

        self.corr_length_thin = 0
        self.corr_length_steps = 15
        self.converge_test_limit = 0.95

        self.done_1Dbins = False
        self.density1D = dict()

        self.updateSettings(ini=ini, settings=settings)

    def setRanges(self, ranges):
        if isinstance(ranges, (list, tuple)):
            for i, minmax in enumerate(ranges):
                self.ranges.setRange(self.parName(i), minmax)
        elif isinstance(ranges, dict):
            for key, value in six.iteritems(ranges):
                self.ranges.setRange(key, value)
        else:
            raise ValueError('MCSamples ranges parameter must be list or dict')
        self.needs_update = True

    def parName(self, i, starDerived=False):
        return self.paramNames.name(i, starDerived)

    def parLabel(self, i):
        return self.paramNames.names[i].label

    def initParameters(self, ini):

        ini.setAttr('ignore_rows', self)
        self.ignore_lines = int(self.ignore_rows)
        if not self.ignore_lines:
            self.ignore_frac = self.ignore_rows
        else:
            self.ignore_frac = 0

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

        ini.setAttr('max_scatter_points', self)
        ini.setAttr('credible_interval_threshold', self)

        ini.setAttr('subplot_size_inch', self)
        ini.setAttr('subplot_size_inch2', self)
        ini.setAttr('subplot_size_inch3', self)
        ini.setAttr('plot_output', self)

        ini.setAttr('force_twotail', self)
        if self.force_twotail: logging.warning('Computing two tail limits')
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

    def _initLimits(self, ini=None):

        bin_limits = ""
        if ini: bin_limits = ini.string('all_limits', '')

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
        assert (settings is None or isinstance(settings, dict))
        if not ini:
            ini = self.ini
        elif isinstance(ini, six.string_types):
            ini = IniFile(ini)
        else:
            ini = copy.deepcopy(ini)
        if not ini: ini = IniFile(getdist.default_getdist_settings)
        if settings:
            ini.params.update(settings)
        self.ini = ini
        if ini: self.initParameters(ini)
        if doUpdate and self.samples is not None: self.updateBaseStatistics()

    def readChains(self, chain_files):
        # Used for by plotting scripts and gui

        self.loadChains(self.root, chain_files)

        if self.ignore_frac and (not self.jobItem or
                                     (not self.jobItem.isImportanceJob and not self.jobItem.isBurnRemoved())):
            self.removeBurnFraction(self.ignore_frac)
            if chains.print_load_details: print('Removed %s as burn in' % self.ignore_frac)
        else:
            if chains.print_load_details: print('Removed no burn in')

        self.deleteFixedParams()

        # Make a single array for chains
        self.makeSingle()

        self.updateBaseStatistics()

        return self

    def updateBaseStatistics(self):
        super(MCSamples, self).updateBaseStatistics()
        mult_max = (self.mean_mult * self.numrows) / min(self.numrows // 2, 500)
        outliers = np.sum(self.weights > mult_max)
        if outliers != 0:
            logging.warning('outlier fraction %s ', float(outliers) / self.numrows)

        self.indep_thin = 0
        self.setCov()
        self.done_1Dbins = False
        self.density1D = dict()

        self._initLimits(self.ini)

        # Get ND confidence region
        self.setLikeStats()
        return self

    def makeSingleSamples(self, filename="", single_thin=None):
        """
        Make file of weight-1 samples by choosing samples
        with probability given by their weight.
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
            # return data
            return self.samples[rand <= self.weights / (self.max_mult * single_thin)]

    def WriteThinData(self, fname, thin_ix, cool):
        nparams = self.samples.shape[1]
        if cool != 1: logging.info('Cooled thinned output with temp: %s', cool)
        MaxL = np.max(self.loglikes)
        with open(fname, 'w') as f:
            i = 0
            for thin in thin_ix:
                if cool != 1:
                    newL = self.loglikes[thin] * cool
                    f.write("%16.7E" % (
                        np.exp(-(newL - self.loglikes[thin]) - MaxL * (1 - cool))))
                    f.write("%16.7E" % newL)
                    for j in nparams:
                        f.write("%16.7E" % (self.samples[i][j]))
                else:
                    f.write("%f" % 1.)
                    f.write("%f" % (self.loglikes[thin]))
                    for j in nparams:
                        f.write("%16.7E" % (self.samples[i][j]))
                i += 1
        print('Wrote ', len(thin_ix), ' thinned samples')

    def getCovMat(self):
        nparamNonDerived = self.paramNames.numNonDerived()
        return covmat.CovMat(matrix=self.fullcov[:nparamNonDerived, :nparamNonDerived],
                             paramNames=self.paramNames.list()[:nparamNonDerived])

    def writeCovMatrix(self, filename=None):
        filename = filename or self.rootdirname + ".covmat"
        self.getCovMat().saveToFile(filename)

    def writeCorrelationMatrix(self, filename=None):
        filename = filename or self.rootdirname + ".corr"
        np.savetxt(filename, self.getCorrelationMatrix(), fmt="%15.7E")

    def getFractionIndices(self, weights, n):
        cumsum = np.cumsum(weights)
        fraction_indices = np.append(np.searchsorted(cumsum, np.linspace(0, 1, n, endpoint=False) * self.norm),
                                     self.weights.shape[0])
        return fraction_indices

    def PCA(self, params, param_map=None, normparam=None, writeDataToFile=False, filename=None, conditional_params=[]):
        """
        Perform principle component analysis. In other words,
        get eigenvectors and eigenvalues for normalized variables
        with optional (log) mapping.
        """

        logging.info('Doing PCA for %s parameters', len(params))
        if len(conditional_params): logging.info('conditional %u fixed parameters', len(conditional_params))

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
                if par.param_max < 0 or par.param_min < par.param_max - par.param_min:
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
            if sd[i] != 0: PCdata[:, i] /= sd[i]

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
            if doexp: PCdata[i, :] = np.exp(PCdata[i, :])

        PCAtext += '\n'
        PCAtext += 'Principle components\n'

        for i in range(n):
            isort = isorted[i]
            PCAtext += 'PC%i (e-value: %f)\n' % (i + 1, evals[isort])
            for j in range(n):
                label = self.parLabel(indices[j])
                if param_map[j] in ['L', 'M']:
                    expo = "%f" % (1.0 / sd[j] * u[i][j])
                    if param_map[j] == "M":
                        div = "%f" % (-np.exp(PCmean[j]))
                    else:
                        div = "%f" % (np.exp(PCmean[j]))
                    PCAtext += '[%f]  (%s/%s)^{%s}\n' % (u[i][j], label, div, expo)
                else:
                    expo = "%f" % (sd[j] / u[i][j])
                    if doexp:
                        PCAtext += '[%f]   exp((%s-%f)/%s)\n' % (u[i][j], label, PCmean[j], expo)
                    else:
                        PCAtext += '[%f]   (%s-%f)/%s)\n' % (u[i][j], label, PCmean[j], expo)

            newmean[i] = self.mean(PCdata[:, i])
            newsd[i] = np.sqrt(self.mean((PCdata[:, i] - newmean[i]) ** 2))
            PCAtext += '          = %f +- %f\n' % (newmean[i], newsd[i])
            PCAtext += '\n'

        # Find out how correlated these components are with other parameters
        PCAtext += 'Correlations of principle components\n'
        l = ["%8i" % i for i in range(1, n + 1)]
        PCAtext += '%s\n' % ("".join(l))

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
            with open(filename or self.rootdirname + ".PCA", "w") as f:
                f.write(PCAtext)
        return PCAtext

    def getNumSampleSummaryText(self):
        lines = 'using %s rows, %s parameters; mean weight %s, tot weight %s\n' % (
            self.numrows, self.paramNames.numParams(), self.mean_mult, self.norm)
        if self.indep_thin != 0:
            lines += 'Approx indep samples (N/corr length): %s\n' % (round(self.norm / self.indep_thin))
        lines += 'Equiv number of single samples (sum w)/max(w): %s\n' % (round(self.norm / self.max_mult))
        lines += 'Effective number of weighted samples (sum w)^2/sum(w^2): %s\n' % (
            int(self.norm ** 2 / np.dot(self.weights, self.weights)))
        return lines

    def getConvergeTests(self, limfrac, writeDataToFile=False,
                         what=['MeanVar', 'GelmanRubin', 'SplitTest', 'RafteryLewis', 'CorrLengths'],
                         filename=None, feedback=False):
        """
        Do convergence tests. 
        """
        lines = ''
        nparam = self.n

        chainlist = self.getSeparateChains()
        num_chains_used = len(chainlist)
        if num_chains_used > 1 and feedback:
            print('Number of chains used = ', num_chains_used)
        for chain in chainlist: chain.setDiffs()
        parForm = self.paramNames.parFormat()
        parNames = [parForm % self.parName(j) for j in range(nparam)]
        limits = np.array([1 - (1 - limfrac) / 2, (1 - limfrac) / 2])

        if 'CorrLengths' in what:
            lines += "Parameter autocorrelation lengths (effective number of samples N_eff = tot weight/weight length)\n"
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
                if self.mean_mult > 1: form = '%15.2f'
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
                # Get stats for individual chains - the variance of the means over the mean of the variances
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
                lines += "var(mean)/mean(var) for eigenvalues of covariance of means of orthonormalized parameters\n"
                for jj, Di in enumerate(D):
                    lines += "%3i%13.5f\n" % (jj + 1, Di)
                GRSummary = " var(mean)/mean(var), remaining chains, worst e-value: R-1 = %13.5F" % self.GelmanRubin
            else:
                self.GelmanRubin = None
                GRSummary = logging.warning('Gelman-Rubin covariance not invertible (parameter not moved?)')
            if feedback: print(GRSummary)
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

        if np.all(np.abs(self.weights - self.weights.astype(np.int)) < 1e-4 / self.max_mult):
            if 'RafteryLewis' in what:
                # Raftery and Lewis method
                # See http://www.stat.washington.edu/tech.reports/raftery-lewis2.ps
                # Raw non-importance sampled chains only
                thin_fac = np.empty(num_chains_used, dtype=np.int)
                epsilon = 0.001

                nburn = np.zeros(num_chains_used, dtype=np.int)
                markov_thin = np.zeros(num_chains_used, dtype=np.int)
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
                                    if thin_rows < 2: break
                                    binchain = np.ones(thin_rows, dtype=np.int)
                                    binchain[chain.samples[thin_ix, j] >= u] = 0
                                    indexes = binchain[:-2] * 4 + binchain[1:-1] * 2 + binchain[2:]
                                    # Estimate transitions probabilities for 2nd order process
                                    tran = np.bincount(indexes, minlength=8).reshape((2, 2, 2))
                                    #                                    tran[:, :, :] = 0
                                    #                                    for i in range(2, thin_rows):
                                    #                                        tran[binchain[i - 2]][binchain[i - 1]][binchain[i]] += 1

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

                                    if g2 - math.log(float(thin_rows - 2)) * 2 < 0: break
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
                        u = self.confidence(self.samples[:, hardest], (1 - limfrac) / 2, hardestend == 0)

                        while True:
                            thin_ix = self.thin_indices(thin_fac[ix], chain.weights)
                            thin_rows = len(thin_ix)
                            if thin_rows < 2: break
                            binchain = np.ones(thin_rows, dtype=np.int)
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

                            if g2 - np.log(float(thin_rows - 1)) < 0: break

                            thin_fac[ix] += 1
                    except LoopException:
                        pass
                    except:
                        thin_fac[ix] = 0
                    if thin_fac[ix] and thin_rows < 2: thin_fac[ix] = 0

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
                # Get correlation lengths. We ignore the fact that there are jumps between chains, so slight underestimate
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
                    if False:
                        # ignore ends of chains
                        corrs = np.zeros([maxoff, nparam])
                        for j in range(nparam):
                            diff = self.samples[thin_ix, j] - self.means[j]
                            for off in range(1, maxoff + 1):
                                corrs[off - 1][j] = np.dot(diff[off:], diff[:-off]) / (thin_rows - off) / self.vars[j]
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
                    else:
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
            with open(filename or (self.rootdirname + '.converge'), 'w') as f:
                f.write(lines)
        return lines

    def _get1DNeff(self, par, param):
        N_eff = getattr(par, 'N_eff_kde', None)
        if N_eff is None:
            par.N_eff_kde = self.getEffectiveSamplesGaussianKDE(param, scale=par.sigma_range)
            N_eff = par.N_eff_kde
        return N_eff

    def getAutoBandwidth1D(self, bins, par, param, mult_bias_correction_order=None, kernel_order=1):
        """
        Get default kernel density bandwidth (in units the range of the bins)
        Based on optimal result for basic Parzen kernel, then scaled if higher-order method being used
        """
        N_eff = self._get1DNeff(par, param)
        h = kde.gaussian_kde_bandwidth_binned(bins, Neff=N_eff)
        par.kde_h = h
        m = mult_bias_correction_order
        if m is None: m = self.mult_bias_correction_order
        if kernel_order > 1: m = max(m, 1)
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
                           mult_bias_correction_order=None, min_corr=0.2):
        """
        get kernel density bandwidth matrix in parameter units
        """
        N_eff = max(self._get1DNeff(parx, paramx), self._get1DNeff(pary, paramy))
        logging.debug('%s %s AutoBandwidth2D: N_eff=%s, corr=%s', parx.name, pary.name, N_eff, corr)
        has_limits = parx.has_limits or pary.has_limits
        do_correlated = not parx.has_limits or not pary.has_limits

        if min_corr < abs(corr) <= self.max_corr_2D and do_correlated:
            # 'shear' the data so fairly uncorrelated, making sure shear keeps any bounds on one parameter unchanged
            # the binning step will rescale to make roughly isotropic as assumed by the 2D kernel optimizer psi_{ab} derivatives
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

            bin1, R1 = kde.bin_samples(p1, nbins=base_fine_bins_2D, range_min=imin, range_max=imax)
            bin2, R2 = kde.bin_samples(p2, nbins=base_fine_bins_2D)
            rotbins, _ = self._make2Dhist(bin1, bin2, base_fine_bins_2D, base_fine_bins_2D)
            opt = kde.KernelOptimizer2D(rotbins, N_eff, 0, do_correlation=not has_limits)
            hx, hy, c = opt.get_h()
            hx *= R1
            hy *= R2
            kernelC = S.dot(np.array([[hx ** 2, hx * hy * c], [hx * hy * c, hy ** 2]])).dot(S.T)
            hx, hy, c = np.sqrt(kernelC[0, 0]), np.sqrt(kernelC[1, 1]), kernelC[0, 1] / np.sqrt(
                kernelC[0, 0] * kernelC[1, 1])
            if pary.has_limits:
                hx, hy = hy, hx
                #            print 'derotated pars', hx, hy, c
        elif abs(corr) > self.max_corr_2D or not do_correlated and corr > 0.8:
            c = max(min(corr, self.max_corr_2D), -self.max_corr_2D)
            hx = parx.sigma_range / N_eff ** (1. / 6)
            hy = pary.sigma_range / N_eff ** (1. / 6)
        else:
            opt = kde.KernelOptimizer2D(bins, N_eff, corr, do_correlation=not has_limits)
            hx, hy, c = opt.get_h()
            hx *= rangex
            hy *= rangey

        if mult_bias_correction_order is None: mult_bias_correction_order = self.mult_bias_correction_order
        logging.debug('hx/sig, hy/sig, corr =%s, %s, %s', hx / parx.err, hy / pary.err, c)
        if mult_bias_correction_order:
            scale = 1.1 * N_eff ** (1. / 6 - 1. / (2 + 4 * (1 + mult_bias_correction_order)))
            hx *= scale
            hy *= scale
            logging.debug('hx/sig, hy/sig, corr, scale =%s, %s, %s, %s', hx / parx.err, hy / pary.err, c, scale)
        return hx, hy, c

    def _initParamRanges(self, j, paramConfid=None):
        if isinstance(j, six.string_types): j = self.index[j]
        paramVec = self.samples[:, j]
        return self._initParam(self.paramNames.names[j], paramVec, self.means[j], self.sddev[j], paramConfid)

    def _initParam(self, par, paramVec, mean=None, sddev=None, paramConfid=None):
        if mean is None: mean = paramVec.mean()
        if sddev is None: sddev = paramVec.std()
        par.err = sddev
        par.mean = mean
        par.param_min = np.min(paramVec)
        par.param_max = np.max(paramVec)
        paramConfid = paramConfid or self.initParamConfidenceData(paramVec)
        # sigma_range is estimate related to shape of structure in the distribution = std dev for Gaussian
        # search for peaks using quantiles, e.g. like simplified version of Janssen 95 (http://dx.doi.org/10.1080/10485259508832654)
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
        ix = ((paramVec - binmin) / fine_width + 0.5).astype(np.int)
        return ix, fine_width, binmin, binmax

    def get1DDensity(self, name, **kwargs):
        """
        Returns a Density1D instance for parameter with given name
        """
        if self.needs_update: self.updateBaseStatistics()
        if not kwargs:
            density = self.density1D.get(name, None)
            if density is not None: return density
        return self.get1DDensityGridData(name, get_density=True, **kwargs)

    def get1DDensityGridData(self, j, writeDataToFile=False, get_density=False, paramConfid=None, meanlikes=False,
                             **kwargs):

        j = self._parAndNumber(j)[0]
        if j is None: return None

        par = self._initParamRanges(j, paramConfid)
        num_bins = kwargs.get('num_bins', self.num_bins)
        smooth_scale_1D = kwargs.get('smooth_scale_1D', self.smooth_scale_1D)
        boundary_correction_order = kwargs.get('boundary_correction_order', self.boundary_correction_order)
        mult_bias_correction_order = kwargs.get('mult_bias_correction_order', self.mult_bias_correction_order)
        fine_bins = kwargs.get('fine_bins', self.fine_bins)

        paramrange = par.range_max - par.range_min
        if paramrange == 0: raise MCSamplesError('Parameter range is zero: ' + par.name)
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
            smooth_1D = self.getAutoBandwidth1D(bins, par, j, mult_bias_correction_order, boundary_correction_order) \
                        * (binmax - binmin) * abs(smooth_scale_1D) / fine_width
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
        Kernel = Kernel1D(winw, smooth_1D)

        cache = {}
        conv = convolve1D(bins, Kernel.Win, 'same', cache=cache)
        fine_x = np.linspace(binmin, binmax, fine_bins)
        density1D = Density1D(fine_x, P=conv, view_ranges=[par.range_min, par.range_max])

        if meanlikes: rawbins = conv.copy()

        if par.has_limits and boundary_correction_order >= 0:
            # correct for cuts allowing for normalization over window
            prior_mask = np.ones(fine_bins + 2 * winw)
            if par.has_limits_bot:
                prior_mask[winw] = 0.5
                prior_mask[: winw] = 0
            if par.has_limits_top:
                prior_mask[-winw] = 0.5
                prior_mask[-winw:] = 0
            a0 = convolve1D(prior_mask, Kernel.Win, 'valid', cache=cache)
            ix = np.nonzero(a0 * density1D.P)
            a0 = a0[ix]
            normed = density1D.P[ix] / a0
            if boundary_correction_order == 0:
                density1D.P[ix] = normed
            elif boundary_correction_order <= 2:
                # linear boundary kernel, e.g. Jones 1993, Jones and Foster 1996
                #  www3.stat.sinica.edu.tw/statistica/oldpdf/A6n414.pdf after Eq 1b, expressed for general prior mask
                # cf arXiv:1411.5528
                xWin = Kernel.Win * Kernel.x
                a1 = convolve1D(prior_mask, xWin, 'valid', cache=cache)[ix]
                a2 = convolve1D(prior_mask, xWin * Kernel.x, 'valid', cache=cache)[ix]
                xP = convolve1D(bins, xWin, 'same', cache=cache)[ix]
                if boundary_correction_order == 1:
                    corrected = (density1D.P[ix] * a2 - xP * a1) / (a0 * a2 - a1 ** 2)
                else:
                    # quadratic correction
                    a3 = convolve1D(prior_mask, xWin * Kernel.x ** 2, 'valid', cache=cache)[ix]
                    a4 = convolve1D(prior_mask, xWin * Kernel.x ** 3, 'valid', cache=cache)[ix]
                    x2P = convolve1D(bins, xWin * Kernel.x, 'same', cache=cache)[ix]
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
            xWin2 = Kernel.Win * Kernel.x ** 2
            x2P = convolve1D(bins, xWin2, 'same', cache=cache)
            a2 = np.sum(xWin2)
            a4 = np.dot(xWin2, Kernel.x ** 2)
            corrected = (density1D.P * a4 - a2 * x2P) / (a4 - a2 ** 2)
            ix = density1D.P > 0
            density1D.P[ix] = density1D.P[ix] * np.exp(np.minimum(corrected[ix] / density1D.P[ix], 2) - 1)

        if mult_bias_correction_order:
            prior_mask = np.ones(fine_bins)
            if par.has_limits_bot:
                prior_mask[0] *= 0.5
            if par.has_limits_top:
                prior_mask[-1] *= 0.5
            a0 = convolve1D(prior_mask, Kernel.Win, 'same', cache=cache)
            for _ in range(mult_bias_correction_order):
                # estimate using flattened samples to remove second order biases
                # mostly good performance, see http://www.jstor.org/stable/2965571 method 3,1 for first order
                prob1 = density1D.P.copy()
                prob1[prob1 == 0] = 1
                fine = bins / prob1
                conv = convolve1D(fine, Kernel.Win, 'same', cache=cache)
                density1D.setP(density1D.P * conv)
                density1D.P /= a0

        density1D.normalize('max', in_place=True)
        if not kwargs: self.density1D[par.name] = density1D

        if get_density: return density1D

        if meanlikes:
            ix = density1D.P > 0
            finebinlikes[ix] /= density1D.P[ix]
            binlikes = convolve1D(finebinlikes, Kernel.Win, 'same', cache=cache)
            binlikes[ix] *= density1D.P[ix] / rawbins[ix]
            if self.shade_likes_is_mean_loglikes:
                maxbin = np.min(binlikes)
                binlikes = np.where((binlikes - maxbin) < 30, np.exp(-(binlikes - maxbin)), 0)
                binlikes[rawbins == 0] = 0
            binlikes /= np.max(binlikes)
            density1D.likes = binlikes
        else:
            density1D.likes = None

        if writeDataToFile:
            # get thinner grid over restricted range for plotting
            x = par.range_min + np.arange(num_bins) * width
            bincounts = density1D.Prob(x)

            if meanlikes:
                likeDensity = Density1D(fine_x, P=binlikes)
                likes = likeDensity.Prob(x)
            else:
                likes = None

            fname = self.rootname + "_p_" + par.name
            filename = os.path.join(self.plot_data_dir, fname + ".dat")
            with open(filename, 'w') as f:
                for xval, binval in zip(x, bincounts):
                    f.write("%16.7E%16.7E\n" % (xval, binval))

            if meanlikes:
                filename_like = os.path.join(self.plot_data_dir, fname + ".likes")
                with open(filename_like, 'w') as f:
                    for xval, binval in zip(x, likes):
                        f.write("%16.7E%16.7E\n" % (xval, binval))

            density = Density1D(x, bincounts)
            density.likes = likes
            return density
        else:
            return density1D

    def _setEdgeMask2D(self, parx, pary, prior_mask, winw, alledge=False):
        if parx.has_limits_bot:
            prior_mask[:, winw] /= 2
            prior_mask[:, :winw] = 0
        if parx.has_limits_top:
            prior_mask[:, -winw] /= 2
            prior_mask[:, -winw:] = 0
        if pary.has_limits_bot:
            prior_mask[winw, :] /= 2
            prior_mask[:winw:] = 0
        if pary.has_limits_top:
            prior_mask[-winw, :] /= 2
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

    def _parAndNumber(self, name):
        if isinstance(name, ParamInfo): name = name.name
        if isinstance(name, six.string_types):
            name = self.index.get(name, None)
            if name is None: return None, None
        if isinstance(name, six.integer_types):
            return name, self.paramNames.names[name]
        raise ParamError("Unknown parameter type %s" % name)

    def _make2Dhist(self, ixs, iys, xsize, ysize):
        flatix = ixs + iys * xsize
        # note arrays are indexed y,x
        return np.bincount(flatix, weights=self.weights,
                           minlength=xsize * ysize).reshape((ysize, xsize)), flatix

    def get2DDensity(self, x, y, **kwargs):
        """
        Returns a Density2D instance for parameters with given names
        """
        if self.needs_update: self.updateBaseStatistics()
        return self.get2DDensityGridData(x, y, get_density=True, **kwargs)

    def get2DDensityGridData(self, j, j2, writeDataToFile=False,
                             num_plot_contours=None, get_density=False, meanlikes=False, **kwargs):
        """
        Get 2D plot data.
        """
        if self.needs_update: self.updateBaseStatistics()
        start = time.time()
        j, parx = self._parAndNumber(j)
        j2, pary = self._parAndNumber(j2)
        if j is None or j2 is None: return None

        self._initParamRanges(j)
        self._initParamRanges(j2)

        base_fine_bins_2D = kwargs.get('fine_bins_2D', self.fine_bins_2D)
        boundary_correction_order = kwargs.get('boundary_correction_order', self.boundary_correction_order)
        mult_bias_correction_order = kwargs.get('mult_bias_correction_order', self.mult_bias_correction_order)
        smooth_scale_2D = float(kwargs.get('smooth_scale_2D', self.smooth_scale_2D))

        has_prior = parx.has_limits or pary.has_limits

        corr = self.getCorrelationMatrix()[j2][j]
        if corr == 1: logging.warning('Parameters are 100%% correlated: %s, %s', parx.name, pary.name)

        logging.debug('Doing 2D: %s - %s', parx.name, pary.name)
        logging.debug('sample x_err, y_err, correlation: %s, %s, %s', parx.err, pary.err, corr)

        # keep things simple unless obvious degeneracy
        if abs(self.max_corr_2D) > 1: raise SettingError('max_corr_2D cannot be >=1')
        if abs(corr) < 0.1: corr = 0.

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
            bin2Dlikes = convolve2D(finebinlikes, Win, 'same', largest_size=convolvesize, cache=cache)
            if mult_bias_correction_order:
                ix = bin2Dlikes > 0
                finebinlikes[ix] /= bin2Dlikes[ix]
                likes2 = convolve2D(finebinlikes, Win, 'same', largest_size=convolvesize, cache=cache)
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
                a20 = convolve2D(prior_mask, winx * indexes, 'valid', largest_size=convolvesize, cache=cache)[ix]
                a02 = convolve2D(prior_mask, winy * y, 'valid', largest_size=convolvesize, cache=cache)[ix]
                a11 = convolve2D(prior_mask, winy * indexes, 'valid', largest_size=convolvesize, cache=cache)[ix]
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
            a00 = convolve2D(prior_mask, Win, 'valid', largest_size=convolvesize, cache=cache)
            for _ in range(mult_bias_correction_order):
                box = histbins.copy()  # careful with cache in convolve2D.
                ix2 = bins2D > np.max(bins2D) * 1e-8
                box[ix2] /= bins2D[ix2]
                bins2D *= convolve2D(box, Win, 'same', largest_size=convolvesize, cache=cache)
                bins2D /= a00

        x = np.linspace(xbinmin, xbinmax, xsize)
        y = np.linspace(ybinmin, ybinmax, ysize)

        density = Density2D(x, y, bins2D,
                            view_ranges=[(parx.range_min, parx.range_max), (pary.range_min, pary.range_max)])
        density.normalize('max', in_place=True)
        if get_density: return density

        ncontours = len(self.contours)
        if num_plot_contours: ncontours = min(num_plot_contours, ncontours)
        contours = self.contours[:ncontours]

        logging.debug('time 2D convolutions: %s', time.time() - start)

        # Get contour containing contours(:) of the probability
        density.contours = density.getContourLevels(contours)

        # now make smaller num_bins grid between ranges for plotting
        # x = parx.range_min + np.arange(nbin2D + 1) * widthx
        # y = pary.range_min + np.arange(nbin2D + 1) * widthy
        # bins2D = density.Prob(x, y)
        # bins2D[bins2D < 1e-30] = 0

        if meanlikes:
            bin2Dlikes /= np.max(bin2Dlikes)
            density.likes = bin2Dlikes
        else:
            density.likes = None

        if writeDataToFile:
            # note store things in confusing transpose form
            #            if meanlikes:
            #                filedensity = Density2D(x, y, bin2Dlikes)
            #                bin2Dlikes = filedensity.Prob(x, y)

            plotfile = self.rootname + "_2D_%s_%s" % (parx.name, pary.name)
            filename = os.path.join(self.plot_data_dir, plotfile)
            np.savetxt(filename, bins2D.T, "%16.7E")
            np.savetxt(filename + "_y", x, "%16.7E")
            np.savetxt(filename + "_x", y, "%16.7E")
            np.savetxt(filename + "_cont", np.atleast_2d(density.contours), "%16.7E")
            if meanlikes:
                np.savetxt(filename + "_likes", bin2Dlikes.T, "%16.7E")
                #       res = Density2D(x, y, bins2D)
                #       res.contours = density.contours
                #       res.likes = bin2Dlikes
        return density

    def setLikeStats(self):
        # Find best fit sample and mean likelihood
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
        self.ranges = ParamBounds()

    def getBounds(self):
        # note not the same as self.ranges, as updated for actual plot ranges depending on posterior
        bounds = ParamBounds()
        bounds.names = self.paramNames.list()
        for par in self.paramNames.names:
            if par.has_limits_bot:
                bounds.lower[par.name] = par.limmin
            if par.has_limits_top:
                bounds.upper[par.name] = par.limmax
        return bounds

    def getUpper(self, name):
        par = self.paramNames.parWithName(name)
        if par:
            return par.limmax
        return None

    def getLower(self, name):
        par = self.paramNames.parWithName(name)
        if par:
            return par.limmin
        return None

    def getMargeStats(self, include_bestfit=False):
        self.setDensitiesandMarge1D()
        m = types.MargeStats()
        m.hasBestFit = False
        m.limits = self.contours
        m.names = self.paramNames.names
        if include_bestfit:
            m.addBestFit(self.getLikeStats)
        return m

    def getLikeStats(self):
        return self.likeStats or self.setLikeStats()

    def getTable(self, columns=1, include_bestfit=False, **kwargs):
        return types.ResultTable(columns, [self.getMargeStats(include_bestfit)], **kwargs)

    def getLatex(self, params=None, limit=1):
        """ Get tex snippet for constraint on parameters in params """
        marge = self.getMargeStats()
        if params is None: params = marge.list()

        formatter = types.NoLineTableFormatter()
        texs = []
        labels = []
        for par in params:
            tex = marge.texValues(formatter, par, limit=limit)
            if tex is not None:
                texs.append(tex[0])
                labels.append(marge.parWithName(par).getLabel())
            else:
                texs.append(None)
                labels.append(None)

        return labels, texs

    def getInlineLatex(self, param, limit=1):
        """Get snippet like: A=x\pm y"""
        labels, texs = self.getLatex([param], limit)
        if not texs[0][0] in ['<', '>']:
            return labels[0] + ' = ' + texs[0]
        else:
            return labels[0] + ' ' + texs[0]

    def setDensitiesandMarge1D(self, max_frac_twotail=None, writeDataToFile=False, meanlikes=False):
        if self.done_1Dbins: return

        for j in range(self.n):
            paramConfid = self.initParamConfidenceData(self.samples[:, j])
            self.get1DDensityGridData(j, writeDataToFile, get_density=not writeDataToFile, paramConfid=paramConfid,
                                      meanlikes=meanlikes)
            self.setMargeLimits(self.paramNames.names[j], paramConfid, max_frac_twotail)

        self.done_1Dbins = True

    def setMargeLimits(self, par, paramConfid, max_frac_twotail=None, density1D=None):
        # Get limits, one or two tail depending on whether posterior
        # goes to zero at the limits or not
        if max_frac_twotail is None:
            max_frac_twotail = self.max_frac_twotail
        par.limits = []
        density1D = density1D or self.get1DDensity(par.name)
        interpGrid = None
        for ix1, contour in enumerate(self.contours):

            marge_limits_bot = par.has_limits_bot and \
                               not self.force_twotail and density1D.P[0] > max_frac_twotail[ix1]
            marge_limits_top = par.has_limits_top and \
                               not self.force_twotail and density1D.P[-1] > max_frac_twotail[ix1]

            if not marge_limits_bot or not marge_limits_top:
                # give limit
                if not interpGrid: interpGrid = density1D.initLimitGrids()
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
        # gets most correlated variable pair names
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

    def saveAsText(self, root, chain_index=None, make_dirs=False):
        super(MCSamples, self).saveAsText(root, chain_index, make_dirs)
        if not chain_index:
            self.ranges.saveToFile(root + '.ranges')

    # Write functions for GetDist.py

    def WriteScriptPlots1D(self, filename, plotparams=None, ext=None):
        text = 'markers=' + str(self.markers) + '\n'
        if plotparams:
            text += 'g.plots_1d(roots,[' + ",".join(['\'' + par + '\'' for par in plotparams]) + '], markers=markers)'
        else:
            text += 'g.plots_1d(roots, markers=markers)'
        self.WritePlotFile(filename, self.subplot_size_inch, text, '', ext)

    def WriteScriptPlots2D(self, filename, plot_2D_param, cust2DPlots, writeDataToFile=False, ext=None,
                           shade_meanlikes=False):
        done2D = {}
        text = 'pairs=[]\n'
        plot_num = 0
        if cust2DPlots:
            cuts = [par1 + '__' + par2 for par1, par2 in cust2DPlots]
        for j, par1 in enumerate(self.paramNames.list()):
            if plot_2D_param or cust2DPlots:
                if par1 == plot_2D_param: continue
                j2min = 0
            else:
                j2min = j + 1

            for j2 in range(j2min, self.n):
                par2 = self.parName(j2)
                if plot_2D_param and par2 != plot_2D_param: continue
                if cust2DPlots and (par1 + '__' + par2) not in cuts: continue
                plot_num += 1
                done2D[(par1, par2)] = True
                if writeDataToFile:
                    self.get2DDensityGridData(j, j2, writeDataToFile=True, meanlikes=shade_meanlikes)
                text += "pairs.append(['%s','%s'])\n" % (par1, par2)
        text += 'g.plots_2d(roots,param_pairs=pairs)'
        self.WritePlotFile(filename, self.subplot_size_inch2, text, '_2D', ext)
        return done2D

    def WriteScriptPlotsTri(self, filename, triangle_params, ext=None):
        text = 'g.triangle_plot(roots, %s)' % triangle_params
        self.WritePlotFile(filename, self.subplot_size_inch, text, '_tri', ext)

    def WriteScriptPlots3D(self, filename, plot_3D, ext=None):
        text = 'sets=[]\n'
        for pars in plot_3D:
            text += "sets.append(['%s','%s','%s'])\n" % tuple(pars)
        text += 'g.plots_3d(roots,sets)'
        self.WritePlotFile(filename, self.subplot_size_inch3, text, '_3D', ext)

    def WritePlotFile(self, filename, subplot_size, text, tag, ext=None):
        with open(filename, 'w') as f:
            f.write("import getdist.plots as plots, os\n")
            if self.plot_data_dir:
                f.write("g=plots.GetDistPlotter(plot_data=r'%s')\n" % self.plot_data_dir)
            else:
                f.write("g=plots.GetDistPlotter(chain_dir=r'%s')\n" % os.path.dirname(self.root))

            f.write("g.settings.setWithSubplotSize(%s)\n" % subplot_size)
            f.write("roots = ['%s']\n" % self.rootname)
            f.write(text + '\n')
            ext = ext or self.plot_output
            fname = self.rootname + tag + '.' + ext
            f.write("g.export(os.path.join(r'%s',r'%s'))\n" % (self.out_dir, fname))


# ==============================================================================

# Useful functions

def GetChainRootFiles(rootdir):
    pattern = os.path.join(rootdir, '*.paramnames')
    files = [os.path.splitext(f)[0] for f in glob.glob(pattern)]
    files.sort()
    return files


def GetRootFileName(rootdir):
    rootFileName = ""
    pattern = os.path.join(rootdir, '*_*.txt')
    chain_files = glob.glob(pattern)
    chain_files.sort()
    if chain_files:
        chain_file0 = chain_files[0]
        rindex = chain_file0.rindex('_')
        rootFileName = chain_file0[:rindex]
    return rootFileName

# ==============================================================================
