from __future__ import print_function
import os
import random
import numpy as np
from collections import OrderedDict
from getdist.paramnames import ParamNames, ParamInfo, escapeLatex
from getdist.convolve import autoConvolve
import getdist.cobaya_interface as cobaya
import pickle
import six
import logging
import copy

# whether to write to terminal chain names and burn in details when loaded from file
print_load_details = True

_int_types = six.integer_types + (np.integer,)

try:
    import pandas

    use_pandas = True
except ImportError:
    use_pandas = False


class WeightedSampleError(Exception):
    """
    An exception that is raised when a WeightedSamples error occurs
    """
    pass


class ParamError(WeightedSampleError):
    """
    An Exception that indicates a bad parameter.
    """
    pass


def lastModified(files):
    """
    Returns the the latest "last modified" time for the given list of files. Ignores files that do not exist.

    :param files: An iterable of file names.
    :return: The latest "last modified" time
    """
    return max([os.path.getmtime(fname) for fname in files if os.path.exists(fname)])


def slice_or_none(x, start=None, end=None):
    return getattr(x, "__getitem__", lambda _: None)(slice(start, end))


def chainFiles(root, chain_indices=None, ext='.txt', separator="_",
               first_chain=0, last_chain=-1, chain_exclude=None):
    """
    Creates a list of file names for samples given a root name and optional filters

    :param root: Root name for files (no extension)
    :param chain_indices: If True, only indexes inside the list included, If False, includes all indexes.
    :param ext: extension for files
    :param separator: separator character used to indicate chain number (usually _ or .)
    :param first_chain: The first index to include.
    :param last_chain: The last index to include.
    :param chain_exclude: A list of indexes to exclude, None to include all
    :return: The list of file names
    """
    index = -1
    files = []
    while True:
        index += 1
        fname = root
        if index > 0:
            # deal with just-folder prefix
            if not root.endswith((os.sep, "/")):
                fname += separator
            fname += str(index)
        if not fname.endswith(ext):
            fname += ext
        if index > first_chain and not os.path.exists(fname) or 0 < last_chain < index:
            break
        if (chain_indices is None or index in chain_indices) \
                and (chain_exclude is None or index not in chain_exclude) \
                and index >= first_chain and os.path.exists(fname):
            files.append(fname)
    return files


_pandas_suggestion = True


def loadNumpyTxt(fname, skiprows=None):
    """
    Utility routine to loads numpy array from file.
    Uses faster pandas read routine if pandas is installed, or falls back to numpy's loadtxt otherwise

    :param fname: The file to load
    :param skiprows: The number of rows to skip at the begging of the file
    :return: numpy array of the data values
    """
    try:
        if use_pandas:
            return pandas.read_csv(fname, delim_whitespace=True, header=None, dtype=np.float64,
                                   skiprows=skiprows, comment='#').values
        else:
            global _pandas_suggestion
            if _pandas_suggestion:
                _pandas_suggestion = False
                logging.warning('Install pandas for faster reading from text files')
            return np.loadtxt(fname, skiprows=skiprows or 0)
    except ValueError:
        print('Error reading %s' % fname)
        raise


def getSignalToNoise(C, noise=None, R=None, eigs_only=False):
    """
    Returns w, M, where w is the eigenvalues of the signal to noise (small y better constrained)

    :param C: covariance matrix
    :param noise: noise matrix
    :param R: rotation matrix, defaults to inverse of Cholesky root of the noise matrix
    :param eigs_only: only return eigenvalues
    :return: eigenvalues and matrix
    """
    if R is None:
        if noise is None:
            raise WeightedSampleError('Must give noise or rotation R')
        R = np.linalg.inv(np.linalg.cholesky(noise))

    M = np.dot(R, C).dot(R.T)
    if eigs_only:
        return np.linalg.eigvalsh(M)
    else:
        w, U = np.linalg.eigh(M)
        U = np.dot(U.T, R)
        return w, U


def covToCorr(cov, copy=True):
    """
    Convert covariance matrix to correlation matrix

    :param cov: The covariance matrix to work on
    :param copy: True if we shouldn't modify the input matrix, False otherwise.
    :return: correlation matrix
    """
    if copy:
        cov = cov.copy()
    for i, di in enumerate(np.sqrt(cov.diagonal())):
        if di:
            cov[i, :] /= di
            cov[:, i] /= di
    return cov


class ParamConfidenceData(object):
    """
    a cache object for confidence interval data
    """
    pass


class ParSamples(object):
    """
    An object used as a container for named parameter sample arrays
    """
    pass


class WeightedSamples(object):
    """
    WeightedSamples is the base class for a set of weighted parameter samples

    :ivar weights:  array of weights for each sample (default: array of 1)
    :ivar loglikes: array of -log(Likelihoods) for each sample (default: array of 0)
    :ivar samples: n_samples x n_parameters numpy array of parameter values
    :ivar n: number of parameters
    :ivar numrows: number of samples positions (rows in the samples array)
    :ivar name_tag: name tag for the samples
    """

    def __init__(self, filename=None, ignore_rows=0, samples=None, weights=None, loglikes=None, name_tag=None,
                 label=None, files_are_chains=True, min_weight_ratio=1e-30):
        """
        :param filename: A filename of a plain text file to load from
        :param ignore_rows:
            - if int >=1: The number of rows to skip at the file in the beginning of the file
            - if float <1: The fraction of rows to skip at the beginning of the file
        :param samples: array of parameter values for each sample, passed to :func:`setSamples`
        :param weights: array of weights
        :param loglikes: array of -log(Likelihood)
        :param name_tag: The name of this instance.
        :param label: latex label for these samples
        :param files_are_chains: use False if the samples file (filename) does not start with two columns giving
                                 weights and -log(Likelihoods)
        :param min_weight_ratio: remove samples with weight less than min_weight_ratio times the maximum weight
        """

        self.precision = '%.8e'
        self.min_weight_ratio = min_weight_ratio
        if filename:
            cols = loadNumpyTxt(filename, skiprows=ignore_rows)
            if not len(cols):
                raise WeightedSampleError('Empty chain: %s' % filename)
            self.setColData(cols, are_chains=files_are_chains)
            self.name_tag = name_tag or os.path.basename(filename)
        else:
            self.setSamples(slice_or_none(samples, ignore_rows),
                            slice_or_none(weights, ignore_rows),
                            slice_or_none(loglikes, ignore_rows))
            self.name_tag = name_tag
            if samples is not None and int(ignore_rows):
                if print_load_details:
                    print('Removed %s lines as burn in' % ignore_rows)
        self.label = label
        self.needs_update = True

    def setColData(self, coldata, are_chains=True):
        """
        Set the samples given an array loaded from file

        :param coldata: The array with columns of [weights, -log(Likelihoods)] and sample parameter values
        :param are_chains: True if coldata starts with two columns giving weight and -log(Likelihood)
        """
        if are_chains:
            self.setSamples(coldata[:, 2:], coldata[:, 0], coldata[:, 1])
        else:
            self.setSamples(coldata)

    def getLabel(self):
        """
        Return the latex label for the samples

        :return: the label
        """
        return self.label or escapeLatex(self.getName())

    def getName(self):
        """
        Returns the name tag of these samples.

        :return: The name tag
        """
        return self.name_tag

    def setSamples(self, samples, weights=None, loglikes=None, min_weight_ratio=None):
        """
        Sets the samples from numpy arrays

        :param samples: The samples values, n_samples x n_parameters numpy array, or can be a list of parameter vectors
        :param weights: Array of weights for each sample. Defaults to 1 for all samples if unspecified.
        :param loglikes: Array of -log(Likelihood) values for each sample
        :param min_weight_ratio: remove samples with weight less than min_weight_ratio of the maximum
        """
        self.weights = weights
        self.loglikes = loglikes
        self.samples = samples
        if samples is not None:
            if isinstance(samples, (list, tuple)):
                samples = np.hstack([x.reshape(-1, 1) for x in samples])
            elif len(samples.shape) == 1:
                samples = np.atleast_2d(samples).transpose()
            self.samples = samples
            self.n = self.samples.shape[1]
            self.numrows = self.samples.shape[0]
            if min_weight_ratio is None: min_weight_ratio = self.min_weight_ratio
            if min_weight_ratio is not None and min_weight_ratio >= 0:
                self.setMinWeightRatio(min_weight_ratio)
        self._weightsChanged()

    def changeSamples(self, samples):
        """
        Sets the samples without changing weights and loglikes.

        :param samples: The samples to set
        """
        self.setSamples(samples, self.weights, self.loglikes)

    def _weightsChanged(self):
        if self.weights is not None:
            self.norm = np.sum(self.weights)
        elif self.samples is not None:
            self.weights = np.ones(self.numrows)
            self.norm = np.float64(self.numrows)
        self.means = None
        self.mean_loglike = None
        self.diffs = None
        self.fullcov = None
        self.correlationMatrix = None
        self.vars = None
        self.sddev = None
        self.needs_update = True

    def _makeParamvec(self, par):
        if isinstance(par, _int_types):
            if 0 <= par < self.n:
                return self.samples[:, par]
            elif par == -1:
                if self.loglikes is None:
                    raise WeightedSampleError('Samples do not have logLikes (par=-1)' % par)
                return self.loglikes
            elif par == -2:
                return self.weights
            else:
                raise WeightedSampleError('Parameter %i does not exist' % par)
        return par

    def getCov(self, nparam=None, pars=None):
        """
        Get covariance matrix of the parameters. By default uses all parameters, or can limit to max number or list.

        :param nparam: if specified, only use the first nparam parameters
        :param pars: if specified, a list of parameter indices (0,1,2..) to include
        :return: covariance matrix.
        """
        if self.fullcov is None:
            self._setCov()
        if pars is not None:
            return self.fullcov[np.ix_(pars, pars)]
        else:
            return self.fullcov[:nparam, :nparam]

    def _setCov(self):
        """
        Calculate and save the full covariance.

        :return: The full covariance matrix
        """
        self.fullcov = self.cov()
        return self.fullcov

    def getCorrelationMatrix(self):
        """
        Get the correlation matrix of all parameters

        :return: The correlation matrix
        """
        if self.correlationMatrix is None:
            self.correlationMatrix = covToCorr(self.getCov())
        return self.correlationMatrix

    def setMeans(self):
        """
        Calculates and saves the means of the samples

        :return: numpy array of parameter means
        """
        self.means = self.weights.dot(self.samples) / self.norm
        if self.loglikes is not None:
            self.mean_loglike = self.weights.dot(self.loglikes) / self.norm
        else:
            self.mean_loglike = None
        return self.means

    def getMeans(self, pars=None):
        """
        Gets the parameter means, from saved array if previously calculated.

        :param pars: optional list of parameter indices to return means for
        :return: numpy array of parameter means
        """
        if self.means is None:
            self.setMeans()
        if pars is None:
            return self.means
        else:
            return np.array([self.means[i] for i in pars])

    def getVars(self):
        """
        Get the parameter variances

        :return: A numpy array of variances.
        """
        if self.means is None: self.setMeans()
        self.vars = np.empty(self.n)
        for i in range(self.n):
            self.vars[i] = self.weights.dot((self.samples[:, i] - self.means[i]) ** 2) / self.norm
        self.sddev = np.sqrt(self.vars)
        return self.vars

    def setDiffs(self):
        """
        saves self.diffs array of parameter differences from the y, e.g. to later calculate variances etc.

        :return: array of differences
        """
        self.diffs = self.mean_diffs()
        return self.diffs

    def getAutocorrelation(self, paramVec, maxOff=None, weight_units=True, normalized=True):
        """
        Gets auto-correlation of an array of parameter values (e.g. for correlated samples from MCMC)

        By default uses weight units (i.e. standard units for separate samples from original chain).
        If samples are made from multiple chains, neglects edge effects.

        :param paramVec: an array of parameter values, or the int index of the parameter in stored samples to use
        :param maxOff: maximum autocorrelation distance to return
        :param weight_units: False to get result in sample point (row) units; weight_units=False gives standard definition for raw chains
        :param normalized: Set to False to get covariance (note even if normalized, corr[0]<>1 in general unless weights are unity).
        :return: zero-based array giving auto-correlations
        """
        if maxOff is None:
            maxOff = self.n - 1
        d = self.mean_diff(paramVec) * self.weights
        corr = autoConvolve(d, n=maxOff + 1, normalize=True)
        if normalized: corr /= self.var(paramVec)
        if weight_units:
            return corr * d.size / self.get_norm()
        else:
            return corr

    def getCorrelationLength(self, j, weight_units=True, min_corr=0.05, corr=None):
        """
        Gets the auto-correlation length for parameter j

        :param j: The index of the parameter to use
        :param weight_units: False to get result in sample point (row) units; weight_units=False gives standard
                             definition for raw chains
        :param min_corr: specifies a minimum value of the autocorrelation to use, e.g. where sampling noise is
                         typically as large as the calculation
        :param corr: The auto-correlation array to use, calculated internally by default using :func:`getAutocorrelation`
        :return: the auto-correlation length
        """
        if corr is None:
            corr = self.getAutocorrelation(j, self.numrows // 10, weight_units=weight_units)
        ix = np.argmin(corr > min_corr * corr[0])
        N = corr[0] + 2 * np.sum(corr[1:ix])
        return N

    def getEffectiveSamples(self, j=0, min_corr=0.05):
        """
        Gets effective number of samples N_eff so that the error on mean of parameter j is sigma_j/N_eff

        :param j: The index of the param to use.
        :param min_corr: the minimum value of the auto-correlation to use when estimating the correlation length
        """
        return self.get_norm() / self.getCorrelationLength(j, min_corr=min_corr)

    def getEffectiveSamplesGaussianKDE(self, paramVec, h=0.2, scale=None, maxoff=None, min_corr=0.05):
        """
        Roughly estimate an effective sample number for use in the leading term for the MISE
        (mean integrated squared error) of a Gaussian-kernel KDE (Kernel Density Estimate). This is used for
        optimizing the kernel bandwidth, and though approximate should be better than entirely ignoring sample
        correlations, or only counting distinct samples.

        Uses fiducial assumed kernel scale h; result does depend on this (typically by factors O(2))

        For bias-corrected KDE only need very rough estimate to use in rule of thumb for bandwidth.

        In the limit h-> 0 (but still >0) answer should be correct (then just includes MCMC rejection duplicates).
        In reality correct result for practical h should depends on shape of the correlation function.

        If self.sampler is 'nested' or 'uncorrelated' return result for uncorrelated samples.

        :param paramVec: parameter array, or int index of parameter to use
        :param h: fiducial assumed kernel scale.
        :param scale: a scale parameter to determine fiducial kernel width, by default the parameter standard deviation
        :param maxoff: maximum value of auto-correlation length to use
        :param min_corr: ignore correlations smaller than this auto-correlation
        :return: A very rough effective sample number for leading term for the MISE of a Gaussian KDE.
        """
        if getattr(self, "sampler", "") in ["nested", "uncorrelated"]:
            return self.get_norm() ** 2 / np.dot(self.weights, self.weights)
        d = self._makeParamvec(paramVec)
        # Result does depend on kernel width, but hopefully not strongly around typical values ~ sigma/4
        kernel_std = (scale or self.std(d)) * h
        # Dependence is from very correlated points due to MCMC rejections;
        # Shouldn't need more than about correlation length
        if maxoff is None:
            maxoff = int(self.getCorrelationLength(d, weight_units=False) * 1.5) + 4
        maxoff = min(maxoff, self.numrows // 10)  # can get problems otherwise if weights are all very large
        uncorr_len = self.numrows // 2
        uncorr_term = 0
        nav = 0
        # first get expected value of each term for uncorrelated samples
        for k in range(uncorr_len, uncorr_len + 5):
            nav += self.numrows - k
            diff2 = (d[:-k] - d[k:]) ** 2 / kernel_std ** 2
            uncorr_term += np.dot(np.exp(-diff2 / 4) * self.weights[:-k], self.weights[k:])
        uncorr_term /= nav

        corr = np.zeros(maxoff + 1)
        corr[0] = np.dot(self.weights, self.weights)
        n = float(self.numrows)
        for k in range(1, maxoff + 1):
            diff2 = (d[:-k] - d[k:]) ** 2 / kernel_std ** 2
            corr[k] = np.dot(np.exp(-diff2 / 4) * self.weights[:-k], self.weights[k:]) - (n - k) * uncorr_term
            if corr[k] < min_corr * corr[0]:
                corr[k] = 0
                break
        N = corr[0] + 2 * np.sum(corr[1:])
        return self.get_norm() ** 2 / N

    def getEffectiveSamplesGaussianKDE_2d(self, i, j, h=0.3, maxoff=None, min_corr=0.05):
        """
        Roughly estimate an effective sample number for use in the leading term for the 2D MISE.
        If self.sampler is 'nested' or 'uncorrelated' return result for uncorrelated samples.

        :param i: parameter array, or int index of first parameter to use
        :param j: parameter array, or int index of second parameter to use
        :param h: fiducial assumed kernel scale.
        :param maxoff: maximum value of auto-correlation length to use
        :param min_corr: ignore correlations smaller than this auto-correlation
        :return: A very rough effective sample number for leading term for the MISE of a Gaussian KDE.
        """
        if getattr(self, "sampler", "") in ["nested", "uncorrelated"]:
            return self.get_norm() ** 2 / np.dot(self.weights, self.weights)
        d1 = self._makeParamvec(i)
        d2 = self._makeParamvec(j)
        cov = self.cov([d1, d2])
        if abs(cov[0, 1]) > np.sqrt(cov[0, 0] * cov[1, 1]) * 0.999:
            # totally correlated, fall back to 1D
            return self.getEffectiveSamplesGaussianKDE(i, h=h, min_corr=min_corr)
        # result does depend on kernel width, use fiducial h
        kernel_inv = np.linalg.inv(cov) / h ** 2

        # Dependence is from very correlated points due to MCMC rejections;
        # Shouldn't need more than about correlation length
        if maxoff is None:
            maxoff = int(max(self.getCorrelationLength(d1, weight_units=False),
                             self.getCorrelationLength(d2, weight_units=False)) * 1.5) + 4
        maxoff = min(maxoff, self.numrows // 10)  # can get problems otherwise if weights are all very large
        uncorr_len = self.numrows // 2
        uncorr_term = 0
        nav = 0
        # first get expected value of each term for uncorrelated samples
        for k in range(uncorr_len, uncorr_len + 5):
            nav += self.numrows - k
            delta = np.vstack((d1[:-k] - d1[k:], d2[:-k] - d2[k:]))
            diff2 = np.sum(delta * kernel_inv.dot(delta), 0)
            uncorr_term += np.dot(np.exp(-diff2 / 4) * self.weights[:-k], self.weights[k:])
        uncorr_term /= nav

        corr = np.zeros(maxoff + 1)
        corr[0] = np.dot(self.weights, self.weights)
        n = float(self.numrows)
        for k in range(1, maxoff + 1):
            delta = np.vstack((d1[:-k] - d1[k:], d2[:-k] - d2[k:]))
            diff2 = np.sum(delta * kernel_inv.dot(delta), 0)
            corr[k] = np.dot(np.exp(-diff2 / 4) * self.weights[:-k], self.weights[k:]) - (n - k) * uncorr_term
            if corr[k] < min_corr * corr[0]:
                corr[k] = 0
                break
        N = corr[0] + 2 * np.sum(corr[1:])
        return self.get_norm() ** 2 / N

    def weighted_sum(self, paramVec, where=None):
        """
        Calculates the weighted sum of a parameter vector, sum_i w_i p_i

        :param paramVec: array of parameter values or int index of parameter to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: weighted sum
        """
        paramVec = self._makeParamvec(paramVec)
        if where is None:
            return self.weights.dot(paramVec)
        return np.dot(paramVec[where], self.weights[where])

    def get_norm(self, where=None):
        """
        gets the normalization, the sum of the sample weights: sum_i w_i

        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: normalization
        """
        if where is None:
            if self.norm is None:
                self.norm = np.sum(self.weights)
            return self.norm
        else:
            return np.sum(self.weights[where])

    def mean(self, paramVec, where=None):
        """
        Get the mean of the given parameter vector.

        :param paramVec: array of parameter values or int index of parameter to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: parameter mean
        """
        if isinstance(paramVec, (list, tuple)):
            return np.array([self.weighted_sum(p, where) for p in paramVec]) / self.get_norm(where)
        else:
            return self.weighted_sum(paramVec, where) / self.get_norm(where)

    def var(self, paramVec, where=None):
        """
        Get the variance of the given parameter vector.

        :param paramVec: array of parameter values or int index of parameter to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: parameter variance
        """
        if isinstance(paramVec, (list, tuple)):
            return np.array([self.var(p) for p in paramVec])
        if where is not None:
            return np.dot(self.mean_diff(paramVec, where) ** 2, self.weights[where]) / self.get_norm(where)
        else:
            return np.dot(self.mean_diff(paramVec) ** 2, self.weights) / self.get_norm()

    def std(self, paramVec, where=None):
        """
        Get the standard deviation of the given parameter vector.

        :param paramVec: array of parameter values or int index of parameter to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: parameter standard deviation.
        """
        return np.sqrt(self.var(paramVec, where))

    def cov(self, pars=None, where=None):
        """
        Get parameter covariance

        :param pars: if specified, a list of parameter vectors or int indices to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: The covariance matrix
        """
        diffs = self.mean_diffs(pars, where)
        if pars is None:
            pars = list(range(self.n))
        n = len(pars)
        cov = np.empty((n, n))
        if where is not None:
            weights = self.weights[where]
        else:
            weights = self.weights
        for i, diff in enumerate(diffs):
            weightdiff = diff * weights
            for j in range(i, n):
                cov[i, j] = weightdiff.dot(diffs[j])
                cov[j, i] = cov[i, j]
        cov /= self.get_norm(where)
        return cov

    def corr(self, pars=None):
        """
        Get the correlation matrix

        :param pars: If specified, list of parameter vectors or int indices to use
        :return: The correlation matrix.
        """
        return covToCorr(self.cov(pars))

    def mean_diff(self, paramVec, where=None):
        """
        Calculates an array of differences between a parameter vector and the mean parameter value

        :param paramVec: array of parameter values or int index of parameter to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: array of p_i - mean(p_i)
        """
        if isinstance(paramVec, _int_types) and paramVec >= 0 and where is None:
            if self.diffs is not None:
                return self.diffs[paramVec]
            return self.samples[:, paramVec] - self.getMeans()[paramVec]
        paramVec = self._makeParamvec(paramVec)
        if where is None:
            return paramVec - self.mean(paramVec)
        else:
            return paramVec[where] - self.mean(paramVec, where)

    def mean_diffs(self, pars=None, where=None):
        """
        Calculates a list of parameter vectors giving distances from parameter means

        :param pars: if specified, list of parameter vectors or int parameter indices to use
        :param where: if specified, a filter for the samples to use (where x>=5 would mean only process samples with x>=5).
        :return: list of arrays p_i-mean(p-i) for each parameter
        """
        if pars is None:
            pars = self.n
        if isinstance(pars, _int_types) and pars >= 0 and where is None:
            means = self.getMeans()
            return [self.samples[:, i] - means[i] for i in range(pars)]
        return [self.mean_diff(i, where) for i in pars]

    def twoTailLimits(self, paramVec, confidence):
        """
        Calculates two-tail equal-area confidence limit by counting samples in the tails

        :param paramVec: array of parameter values or int index of parameter to use
        :param confidence: confidence limit to calculate, e.g. 0.95 for 95% confidence
        :return: min, max values for the confidence interval
        """
        limits = np.array([(1 - confidence) / 2, 1 - (1 - confidence) / 2])
        return self.confidence(paramVec, limits)

    def initParamConfidenceData(self, paramVec, start=0, end=None, weights=None):
        """
        Initialize cache of data for calculating confidence intervals

        :param paramVec: array of parameter values or int index of parameter to use
        :param start: The sample start index to use
        :param end: The sample end index to use, use None to go all the way to the end of the vector
        :param weights: A numpy array of weights for each sample, defaults to self.weights
        :return: :class:`~.chains.ParamConfidenceData` instance
        """
        if weights is None:
            weights = self.weights
        d = ParamConfidenceData()
        d.paramVec = self._makeParamvec(paramVec)[start:end]
        d.norm = np.sum(weights[start:end])
        d.indexes = d.paramVec.argsort()
        weightsort = weights[start + d.indexes]
        d.cumsum = np.cumsum(weightsort)
        return d

    def confidence(self, paramVec, limfrac, upper=False, start=0, end=None, weights=None):
        """
        Calculate sample confidence limits, not using kernel densities just counting samples in the tails

        :param paramVec: array of parameter values or int index of parameter to use
        :param limfrac: fraction of samples in the tail, e.g. 0.05 for a 95% one-tail limit, or 0.025 for a 95% two-tail limit
        :param upper: True to get upper limit, False for lower limit
        :param start: Start index for the vector to use
        :param end: The end index, use None to go all the way to the end of the vector.
        :param weights:  numpy array of weights for each sample, by default self.weights
        :return: confidence limit (parameter value when limfac of samples are further in the tail)
        """
        if isinstance(paramVec, ParamConfidenceData):
            d = paramVec
        else:
            d = self.initParamConfidenceData(paramVec, start, end, weights)

        if not upper:
            target = d.norm * limfrac
        else:
            target = d.norm * (1 - limfrac)
        ix = np.searchsorted(d.cumsum, target)
        return d.paramVec[d.indexes[np.minimum(ix, d.indexes.shape[0] - 1)]]

    def getSignalToNoise(self, params, noise=None, R=None, eigs_only=False):
        """
        Returns w, M, where w is the eigenvalues of the signal to noise (small y better constrained)

        :param params: list of parameters indices to use
        :param noise: noise matrix
        :param R: rotation matrix, defaults to inverse of Cholesky root of the noise matrix
        :param eigs_only: only return eigenvalues
        :return: w, M, where w is the eigenvalues of the signal to noise (small y better constrained)
        """
        C = self.cov(params)
        return getSignalToNoise(C, noise, R, eigs_only)

    def thin_indices(self, factor, weights=None):
        """
        Indices to make single weight 1 samples. Assumes integer weights.

        :param factor: The factor to thin by, should be int.
        :param weights: The weights to thin, None if this should use the weights stored in the object.
        :return: array of indices of samples to keep
        """
        if weights is None:  weights = self.weights
        numrows = len(weights)
        norm1 = np.sum(weights)
        weights = weights.astype(np.int)
        norm = np.sum(weights)

        if abs(norm - norm1) > 1e-4:
            raise WeightedSampleError('Can only thin with integer weights')
        if factor != int(factor):
            raise WeightedSampleError('Thin factor must be integer')
        factor = int(factor)
        if factor >= np.max(weights):
            cumsum = np.cumsum(weights) // factor
            # noinspection PyTupleAssignmentBalance
            _, thin_ix = np.unique(cumsum, return_index=True)
        else:
            tot = 0
            i = 0
            thin_ix = np.empty(norm // factor, dtype=np.int)
            ix = 0
            mult = weights[i]
            while i < numrows:
                if mult + tot < factor:
                    tot += mult
                    i += 1
                    if i < numrows: mult = weights[i]
                else:
                    thin_ix[ix] = i
                    ix += 1
                    if mult == factor - tot:
                        i += 1
                        if i < numrows: mult = weights[i]
                    else:
                        mult -= (factor - tot)
                    tot = 0

        return thin_ix

    def random_single_samples_indices(self):
        """
        Returns an array of sample indices that give a list of weight-one samples, by randomly
        selecting samples depending on the sample weights

        :return: array of sample indices
        """
        max_weight = np.max(self.weights)
        thin_ix = []
        for i in range(self.numrows):
            P = self.weights[i] / max_weight
            if random.random() < P:
                thin_ix.append(i)
        return np.array(thin_ix, dtype=np.int)

    def thin(self, factor):
        """
        Thin the samples by the given factor, giving set of samples with unit weight

        :param factor: The factor to thin by
        """
        thin_ix = self.thin_indices(factor)
        self.setSamples(self.samples[thin_ix, :], loglikes=None if self.loglikes is None else self.loglikes[thin_ix],
                        min_weight_ratio=-1)

    def filter(self, where):
        """
        Filter the stored samples to keep only samples matching filter

        :param where: list of sample indices to keep, or boolean array filter (e.g. x>5 to keep only samples where x>5)
        """
        self.setSamples(self.samples[where, :], self.weights[where],
                        None if self.loglikes is None else self.loglikes[where], min_weight_ratio=-1)

    def reweightAddingLogLikes(self, logLikes):
        """
        Importance sample the samples, by adding logLike (array of -log(likelihood values) to the currently stored likelihoods,
        and re-weighting accordingly, e.g. for adding a new data constraint

        :param logLikes: array of -log(likelihood) for each sample to adjust
        """
        scale = np.min(logLikes)
        if self.loglikes is not None:
            self.loglikes += logLikes
        self.weights *= np.exp(-(logLikes - scale))
        self._weightsChanged()

    def cool(self, cool):
        """
        Cools the samples, i.e. multiples log likelihoods by cool factor and re-weights accordingly

        :param cool: cool factor
        """
        MaxL = np.max(self.loglikes)
        newL = self.loglikes * cool
        self.weights = self.weights * np.exp(-(newL - self.loglikes) - (MaxL * (1 - cool)))
        self.loglikes = newL
        self._weightsChanged()

    def deleteZeros(self):
        """
        Removes samples with zero weight

        """
        self.filter(self.weights > 0)

    def setMinWeightRatio(self, min_weight_ratio=1e-30):
        """
        Removes samples with weight less than min_weight_ratio times the maximum weight

        :param min_weight_ratio: minimum ratio to max to exclude
        """
        if self.weights is not None and min_weight_ratio >= 0:
            max_weight = np.max(self.weights)
            min_weight = np.min(self.weights)
            if min_weight < max_weight * min_weight_ratio:
                self.filter(self.weights > max_weight * min_weight_ratio)

    def deleteFixedParams(self):
        """
        Removes parameters that do not vary (are the same in all samples)

        :return: list of fixed parameter indices that were removed
        """
        fixed = []
        for i in range(self.samples.shape[1]):
            if np.all(self.samples[:, i] == self.samples[0, i]):
                fixed.append(i)
        self.changeSamples(np.delete(self.samples, fixed, 1))
        return fixed

    def removeBurn(self, remove=0.3):
        """
        removes burn in from the start of the samples

        :param remove: fraction of samples to remove, or if int >1, the number of sample rows to remove
        """
        if remove >= 1:
            ix = int(remove)
        else:
            ix = int(round(self.numrows * remove))
        if self.weights is not None:
            self.weights = self.weights[ix:]
        if self.loglikes is not None:
            self.loglikes = self.loglikes[ix:]
        self.changeSamples(self.samples[ix:, :])

    def saveAsText(self, root, chain_index=None, make_dirs=False):
        """
        Saves the samples as text files

        :param root: The root name to use
        :param chain_index: Optional index to be used for the samples' filename, zero based, e.g. for saving one of multiple chains
        :param make_dirs: True if this should create the directories if necessary.
        """
        if self.loglikes is not None:
            loglikes = self.loglikes
        else:
            loglikes = np.zeros(self.numrows)
        if make_dirs and not os.path.exists(os.path.dirname(root)):
            os.makedirs(os.path.dirname(root))
        if root.endswith('.txt'):
            root = root[:-3]
        np.savetxt(root + ('' if chain_index is None else '_' + str(chain_index + 1)) + '.txt',
                   np.hstack((self.weights.reshape(-1, 1), loglikes.reshape(-1, 1), self.samples)),
                   fmt=self.precision)


class Chains(WeightedSamples):
    """
    Holds one or more sets of weighted samples, for example a set of MCMC chains.
    Inherits from :class:`~.chains.WeightedSamples`, also adding parameter names and labels

    :ivar paramNames: a :class:`~.paramnames.ParamNames` instance holding the parameter names and labels
    """

    def __init__(self, root=None, jobItem=None, paramNamesFile=None, names=None, labels=None, renames=None,
                 sampler=None, **kwargs):
        """

        :param root: optional root name for files
        :param jobItem: optional jobItem for parameter grid item
        :param paramNamesFile: optional filename of a .paramnames files that holds parameter names
        :param names: optional list of names for the parameters
        :param labels: optional list of latex labels for the parameters
        :param renames: optional dictionary of parameter aliases
        :param sampler: string describing the type of samples (default :mcmc); if "nested" or "uncorrelated"
              the effective number of samples is calculated using uncorrelated approximation
        :param kwargs: extra options for :class:`~.chains.WeightedSamples`'s constructor

        """

        self.chains = None
        WeightedSamples.__init__(self, **kwargs)
        self.jobItem = jobItem
        self.ignore_lines = float(kwargs.get('ignore_rows', 0))
        self.root = root
        if not paramNamesFile and root:
            mid = not root.endswith((os.sep, "/"))
            endings = ['.paramnames', ('__' if mid else '') + 'full.yaml',
                       (cobaya._separator_files if mid else '') + 'updated.yaml']
            try:
                paramNamesFile = next(
                    root + ending for ending in endings if os.path.exists(root + ending))
            except StopIteration:
                paramNamesFile = None
        self.setParamNames(paramNamesFile or names)
        if labels is not None:
            self.paramNames.setLabels(labels)
        if renames is not None:
            self.updateRenames(renames)
        # If Cobaya sample and label not set manually via keyword, try to load from yaml
        if not self.label and isinstance(paramNamesFile, six.string_types) and paramNamesFile.endswith(".yaml"):
            self.label = cobaya.get_sample_label(paramNamesFile)
        # Sampler that generated the chain -- assume "mcmc"
        if isinstance(sampler, six.string_types):
            if sampler.lower() not in ["mcmc", "nested", "uncorrelated"]:
                raise ValueError("Unknown sampler type %s" % sampler)
            self.sampler = sampler.lower()
        elif isinstance(paramNamesFile, six.string_types) and paramNamesFile.endswith("yaml"):
            self.sampler = cobaya.get_sampler_type(paramNamesFile)
        else:
            self.sampler = "mcmc"

    def setParamNames(self, names=None):
        """
        Sets the names of the params.

        :param names: Either a :class:`~.paramnames.ParamNames` object, the name of a .paramnames file to load, a list
                      of name strings, otherwise use default names (param1, param2...).
        """
        self.paramNames = None
        if isinstance(names, ParamNames):
            self.paramNames = copy.deepcopy(names)
        elif isinstance(names, six.string_types):
            self.paramNames = ParamNames(names)
        elif names is not None:
            self.paramNames = ParamNames(names=names)
        elif self.samples is not None:
            self.paramNames = ParamNames(default=self.n)
        if self.paramNames:
            self._getParamIndices()
        self.needs_update = True

    def filter(self, where):
        """
        Filter the stored samples to keep only samples matching filter

        :param where: list of sample indices to keep, or boolean array filter (e.g. x>5 to keep only samples where x>5)
        """

        if self.chains is None:
            if hasattr(self, 'chain_offsets'):
                # must update chain_offsets to be able to correctly split back into separate filtered chains if needed
                lens = [0]
                for off1, off2 in zip(self.chain_offsets[:-1], self.chain_offsets[1:]):
                    lens.append(np.count_nonzero(where[off1:off2]))
                self.chain_offsets = np.cumsum(np.array(lens))
            super(Chains, self).filter(where)
        else:
            raise ValueError('chains are separated, makeSingle first or call filter on individual chains')

    def getParamNames(self):
        """
        Get :class:`~.paramnames.ParamNames` object with names for the parameters

        :return: :class:`~.paramnames.ParamNames` object giving parameter names and labels
        """
        return self.paramNames

    def _getParamIndices(self):
        """
        Gets the indices of the params.

        :return: A dict mapping the param name to the parameter index.
        """
        if self.samples is not None and len(self.paramNames.names) != self.n:
            raise WeightedSampleError("paramNames size does not match number of parameters in samples")
        index = dict()
        for i, name in enumerate(self.paramNames.names):
            index[name.name] = i
        self.index = index
        return self.index

    def _parAndNumber(self, name):
        """
        Get index and ParamInfo for a name or index

        :param name: name or parameter index
        :return: index, ParamInfo instance
        """
        if isinstance(name, ParamInfo):
            name = name.name
        if isinstance(name, six.string_types):
            name = self.index.get(name, None)
            if name is None:
                return None, None
        if isinstance(name, _int_types):
            return name, self.paramNames.names[name]
        raise ParamError("Unknown parameter type %s" % name)

    def getRenames(self):
        """
        Gets dictionary of renames known to each parameter.
        """
        return self.paramNames.getRenames()

    def updateRenames(self, renames):
        """
        Updates the renames known to each parameter with the given dictionary of renames.
        """
        self.paramNames.updateRenames(renames)

    def setParams(self, obj):
        """
        Adds array variables obj.name1, obj.name2 etc, where
        obj.name1 is the vector of samples with name 'name1'

        if a parameter name is of the form aa.bb.cc, it makes subobjects so you can reference obj.aa.bb.cc

        :param obj: The object instance to add the parameter vectors variables
        :return: The obj after alterations.
        """
        for i, name in enumerate(self.paramNames.names):
            path = name.name.split('.')
            ob = obj
            for p in path[:-1]:
                if not hasattr(ob, p):
                    setattr(ob, p, ParSamples())
                ob = getattr(ob, p)
            setattr(ob, path[-1], self.samples[:, i])
        return obj

    def getParams(self):
        """
        Creates a :class:`~.chains.ParSamples` object, with variables giving vectors for all the parameters,
        for example samples.getParams().name1 would be the vector of samples with name 'name1'

        :return: A :class:`~.chains.ParSamples` object containing all the parameter vectors, with attributes given by the parameter names
        """
        pars = ParSamples()
        self.setParams(pars)
        return pars

    def getParamSampleDict(self, ix, want_derived=True):
        """
        Returns a dictionary of parameter values for sample number ix

        :param ix: sample index
        :param want_derived: include derived parameters
        :return: ordered dictionary of parameter values
        """
        res = OrderedDict()
        for i, name in enumerate(self.paramNames.names):
            if want_derived or not name.isDerived:
                res[name.name] = self.samples[ix, i]
        res['weight'] = self.weights[ix]
        res['loglike'] = self.loglikes[ix]
        return res

    def _makeParamvec(self, par):
        if self.needs_update:
            self.updateBaseStatistics()
        if isinstance(par, ParamInfo):
            par = par.name
        if isinstance(par, six.string_types):
            return self.samples[:, self.index[par]]
        return WeightedSamples._makeParamvec(self, par)

    def updateChainBaseStatistics(self):
        # old name, use updateBaseStatistics
        return self.updateBaseStatistics()

    def updateBaseStatistics(self):
        """
        Updates basic computed statistics for this chain, e.g. after any changes to the samples or weights

        :return: self after updating statistics.
        """
        self.getVars()
        self.mean_mult = self.norm / self.numrows
        self.max_mult = np.max(self.weights)
        self._getParamIndices()
        self.needs_update = False
        return self

    def addDerived(self, paramVec, name, **kwargs):
        """
        Adds a new parameter

        :param paramVec: The vector of parameter values to add.
        :param name: The name for the new parameter
        :param kwargs: arguments for paramnames' :func:`.paramnames.ParamList.addDerived`
        :return: The added parameter's :class:`~.paramnames.ParamInfo` object
        """
        if self.paramNames.parWithName(name):
            raise ValueError('Parameter with name %s already exists' % name)
        self.changeSamples(np.c_[self.samples, paramVec])
        return self.paramNames.addDerived(name, **kwargs)

    def loadChains(self, root, files_or_samples, weights=None, loglikes=None,
                   ignore_lines=None):
        """
        Loads chains from files.

        :param root: Root name
        :param files_or_samples: list of file names or list of arrays of samples, or single array of samples
        :param weights: if loading from arrays of samples, corresponding list of arrays of weights
        :param loglikes: if loading from arrays of samples, corresponding list of arrays of -2 log(likelihood)
        :param ignore_lines: Amount of lines at the start of the file to ignore, None if should not ignore
        :return: True if loaded successfully, False if none loaded
        """
        self.chains = []
        self.samples = None
        self.weights = None
        self.loglikes = None
        if ignore_lines is None:
            ignore_lines = self.ignore_lines
        WSkwargs = {"ignore_rows": ignore_lines,
                    "min_weight_ratio": self.min_weight_ratio}
        if isinstance(files_or_samples, six.string_types) or isinstance(files_or_samples[0], six.string_types):
            # From files
            if weights is not None or loglikes is not None:
                raise ValueError('weights and loglikes not needed reading from file')
            if isinstance(files_or_samples, six.string_types):
                files_or_samples = [files_or_samples]
            self.name_tag = self.name_tag or os.path.basename(root)
            for fname in files_or_samples:
                if print_load_details:
                    print(fname)
                try:
                    self.chains.append(WeightedSamples(fname, **WSkwargs))
                except WeightedSampleError:
                    if print_load_details:
                        print('Ignored file %s (likely empty)' % fname)
            nchains = len(self.chains)
            if not nchains:
                raise WeightedSampleError('loadChains - no chains found for ' + root)
        else:
            # From arrays
            def array_dimension(a):
                # Dimension for numpy or list/tuple arrays, not very safe (does not work if string elements)
                d = 0
                while True:
                    try:
                        a = a[0]
                        d += 1
                    except:
                        return d

            dim = array_dimension(files_or_samples)
            if dim in [1, 2]:
                self.chains = None
                self.setSamples(slice_or_none(files_or_samples, ignore_lines),
                                slice_or_none(weights, ignore_lines),
                                slice_or_none(loglikes, ignore_lines), self.min_weight_ratio)
                if self.paramNames is None:
                    self.paramNames = ParamNames(default=self.n)
                nchains = 1
            elif dim == 3:
                for i, samples_i in enumerate(files_or_samples):
                    self.chains.append(WeightedSamples(
                        samples=samples_i, loglikes=None if loglikes is None else loglikes[i],
                        weights=None if weights is None else weights[i], **WSkwargs))
                if self.paramNames is None:
                    self.paramNames = ParamNames(default=self.chains[0].n)
                nchains = len(self.chains)
            else:
                raise ValueError('samples or files must be array of samples, or a list of arrays or files')
        self._weightsChanged()
        return nchains > 0

    def getGelmanRubinEigenvalues(self, nparam=None, chainlist=None):
        """
        Assess convergence using var(mean)/mean(var) in the orthogonalized parameters
        c.f. Brooks and Gelman 1997.

        :param nparam: The number of parameters (starting at first), by default uses all of them
        :param chainlist: list of :class:`~.chains.WeightedSamples`, the samples to use. Defaults to all the separate chains in this instance.
        :return: array of  var(mean)/mean(var) for orthogonalized parameters
        """
        if chainlist is None:
            chainlist = self.getSeparateChains()
        nparam = nparam or self.paramNames.numNonDerived()
        meanscov = np.zeros((nparam, nparam))
        means = self.getMeans()[:nparam]
        meancov = np.zeros(meanscov.shape)
        for chain in chainlist:
            diff = chain.getMeans()[:nparam] - means
            meanscov += np.outer(diff, diff)
            meancov += chain.getCov(nparam)
        meanscov /= (len(chainlist) - 1)
        meancov /= len(chainlist)
        w, U = np.linalg.eigh(meancov)
        if np.min(w) > 0:
            U /= np.sqrt(w)
            D = np.linalg.eigvalsh(np.dot(U.T, meanscov).dot(U))
            return D
        else:
            return None

    def getGelmanRubin(self, nparam=None, chainlist=None):
        """
        Assess the convergence using the maximum var(mean)/mean(var) of orthogonalized parameters
        c.f. Brooks and Gelman 1997.

        :param nparam: The number of parameters, by default uses all
        :param chainlist: list of :class:`~.chains.WeightedSamples`, the samples to use. Defaults to all the separate chains in this instance.
        :return: The worst var(mean)/mean(var) for orthogonalized parameters. Should be <<1 for good convergence.
        """
        return np.max(self.getGelmanRubinEigenvalues(nparam, chainlist))

    def makeSingle(self):
        """
        Combines separate chains into one samples array, so self.samples has all the samples
        and this instance can then be used as a general :class:`~.chains.WeightedSamples` instance.

        :return: self
        """
        self.chain_offsets = np.cumsum(np.array([0] + [chain.samples.shape[0] for chain in self.chains]))
        weights = None if self.chains[0].weights is None else np.hstack([chain.weights for chain in self.chains])
        loglikes = None if self.chains[0].loglikes is None else np.hstack([chain.loglikes for chain in self.chains])
        self.setSamples(np.vstack([chain.samples for chain in self.chains]), weights, loglikes, min_weight_ratio=-1)
        self.chains = None
        self.needs_update = True
        return self

    def getSeparateChains(self):
        """
        Gets a list of samples for separate chains.
        If the chains have already been combined, uses the stored sample offsets to reconstruct the array (generally no array copying)

        :return: The list of :class:`~.chains.WeightedSamples` for each chain.
        """
        if self.chains is not None:
            return self.chains
        chainlist = []
        for off1, off2 in zip(self.chain_offsets[:-1], self.chain_offsets[1:]):
            chainlist.append(WeightedSamples(samples=self.samples[off1:off2], weights=self.weights[off1:off2],
                                             loglikes=self.loglikes[off1:off2]))
        return chainlist

    def removeBurnFraction(self, ignore_frac):
        """
        Remove a fraction of the samples as burn in

        :param ignore_frac: fraction of sample points to remove from the start of the samples, or each chain if not combined
        """
        if self.samples is not None:
            self.removeBurn(ignore_frac)
            self.chains = None
            self.needs_update = True
        else:
            for chain in self.chains:
                chain.removeBurn(ignore_frac)

    def deleteFixedParams(self):
        """
        Delete parameters that are fixed (the same value in all samples)
        """
        if self.samples is not None:
            fixed = WeightedSamples.deleteFixedParams(self)
            self.chains = None
        else:
            fixed = []
            chain = self.chains[0]
            for i in range(chain.n):
                if np.all(chain.samples[:, i] == chain.samples[0, i]): fixed.append(i)
            for chain in self.chains:
                chain.changeSamples(np.delete(chain.samples, fixed, 1))
        self.paramNames.deleteIndices(fixed)
        self._getParamIndices()

    def saveAsText(self, root, chain_index=None, make_dirs=False):
        """
        Saves the samples as text files, including parameter names as .paramnames file.

        :param root: The root name to use
        :param chain_index: Optional index to be used for the filename, zero based, e.g. for saving one of multiple chains
        :param make_dirs: True if this should (recursively) create the directory if it doesn't exist
        """
        super(Chains, self).saveAsText(root, chain_index, make_dirs)
        if not chain_index:
            self.saveTextMetadata(root)

    def saveTextMetadata(self, root):
        """
        Saves metadata about the sames to text files with given file root

        :param root: root file name
        """
        self.paramNames.saveAsText(root + '.paramnames')

    def savePickle(self, filename):
        """
        Save the current object to a file in pickle format

        :param filename: The file to write to
        """

        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
