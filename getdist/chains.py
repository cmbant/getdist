from __future__ import print_function
import os
import random
import numpy as np
from getdist.paramnames import ParamNames, ParamInfo
from getdist.convolve import autoConvolve
import six

# whether to write to terminal chain names and burn in details when loaded from file
print_load_details = True

try:
    import pandas
    from distutils.version import LooseVersion

    use_pandas = LooseVersion(pandas.version.version) > LooseVersion("0.14.0")
except:
    use_pandas = False


class WeightedSampleError(Exception):
    pass


def lastModified(files):
    return max([os.path.getmtime(fname) for fname in files if os.path.exists(fname)])


def chainFiles(root, chain_indices=None, ext='.txt', first_chain=0, last_chain=-1, chain_exclude=None):
    index = -1
    files = []
    while True:
        index += 1
        fname = root + ('', '_' + str(index))[index > 0]
        if not ext in fname: fname += ext
        if index > 0 and not os.path.exists(fname) or 0 < last_chain <= index: break
        if (chain_indices is None or index in chain_indices) \
                and (chain_exclude is None or not index in chain_exclude) \
                and index >= first_chain and os.path.exists(fname):
            files.append(fname)
    return files


def loadNumpyTxt(fname, skiprows=None):
    if use_pandas:
        return pandas.read_csv(fname, delim_whitespace=True, header=None, dtype=np.float64, skiprows=skiprows).values
    else:
        return np.loadtxt(fname, skiprows=skiprows)


def getSignalToNoise(C, noise=None, R=None, eigs_only=False):
    if R is None:
        if noise is None: raise WeightedSampleError('Must give noise or rotation R')
        R = np.linalg.inv(np.linalg.cholesky(noise))

    M = np.dot(R, C).dot(R.T)
    if eigs_only:
        return np.linalg.eigvalsh(M)
    else:
        w, U = np.linalg.eigh(M)
        U = np.dot(U.T, R)
        return w, U


def covToCorr(cov, copy=True):
    if copy: cov = cov.copy()
    for i, di in enumerate(np.sqrt(cov.diagonal())):
        if di:
            cov[i, :] /= di
            cov[:, i] /= di
    return cov


class ParamConfidenceData(object): pass


class ParSamples(object): pass


class WeightedSamples(object):
    def __init__(self, filename=None, ignore_rows=0, samples=None, weights=None, loglikes=None, name_tag=None,
                 files_are_chains=True):
        if filename:
            cols = loadNumpyTxt(filename, skiprows=ignore_rows)
            self.setColData(cols, are_chains=files_are_chains)
            self.name_tag = name_tag or os.path.basename(filename)
        else:
            self.setSamples(samples, weights, loglikes)
            self.name_tag = name_tag
        self.needs_update = True

    def setColData(self, coldata, are_chains=True):
        if are_chains:
            self.setSamples(coldata[:, 2:], coldata[:, 0], coldata[:, 1])
        else:
            self.setSamples(coldata)

    def getName(self):
        return self.name_tag

    def setSamples(self, samples, weights=None, loglikes=None):
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
        self._weightsChanged()

    def changeSamples(self, samples):
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
        if isinstance(par, six.integer_types):
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
        if self.fullcov is None:
            self.setCov()
        if pars is not None:
            return self.fullcov[np.ix_(pars, pars)]
        else:
            return self.fullcov[:nparam, :nparam]

    def setCov(self):
        self.fullcov = self.cov()
        return self.fullcov

    def getCorrelationMatrix(self):
        if self.correlationMatrix is None:
            self.correlationMatrix = covToCorr(self.getCov())
        return self.correlationMatrix

    def setMeans(self):
        self.means = self.weights.dot(self.samples) / self.norm
        if self.loglikes is not None:
            self.mean_loglike = self.weights.dot(self.loglikes) / self.norm
        else:
            self.mean_loglike = None
        return self.means

    def getMeans(self):
        if self.means is None:
            return self.setMeans()
        return self.means

    def getVars(self):
        if self.means is None: self.setMeans()
        self.vars = np.empty(self.n)
        for i in range(self.n):
            self.vars[i] = self.weights.dot((self.samples[:, i] - self.means[i]) ** 2) / self.norm
        self.sddev = np.sqrt(self.vars)
        return self.vars

    def setDiffs(self):
        self.diffs = self.mean_diffs()
        return self.diffs

    def getAutocorrelation(self, paramVec, maxOff=None, weight_units=True, normalized=True):
        """ get auto covariance in weight units (i.e. standard units for separate samples from original chain); 
            divide by var to normalize
            weight_units=False to get result in sample point (row) units; weight_units=False gives standard definition for raw chains
            set normalized=False to get covariance (note even if normalized, corr[0]<>1 in general unless weights are unity)
            If samples are made from multiple chains, neglects edge effects
       """
        if maxOff is None: maxOff = self.n - 1
        d = self.mean_diff(paramVec) * self.weights
        corr = autoConvolve(d, n=maxOff + 1, normalize=True)
        if normalized: corr /= self.var(paramVec)
        if weight_units:
            return corr * d.size / self.get_norm()
        else:
            return corr

    def getCorrelationLength(self, j, weight_units=True, min_corr=0.05, corr=None):
        if corr is None:
            corr = self.getAutocorrelation(j, self.numrows // 10, weight_units=weight_units)
        ix = np.argmin(corr > min_corr * corr[0])
        N = corr[0] + 2 * np.sum(corr[1:ix])
        return N

    def getEffectiveSamples(self, j=0, min_corr=0.05):
        """
        Gets effective number of samples N_eff so that the error on mean of parameter j is sigma_j/N_eff
        """
        return self.get_norm() / self.getCorrelationLength(j, min_corr=min_corr)

    def getEffectiveSamplesGaussianKDE(self, paramVec, h=0.2, scale=None, maxoff=None, min_corr=0.05):
        """
         Estimate very roughly effective sample number for leading term for variance of Gaussian KDE MISE
         Uses fiducial assumed kernel scale h; result does depend on this (typically by factors O(2))
         For bias-corrected KDE only need very rough estimate to use in rule of thumb for bandwidth
         In the limit h-> 0 (but still >0) answer should be correct (then just includes MCMC rejection duplicates), 
         In reality correct result for practical h should depends on shape of correlation function
        """
        d = self._makeParamvec(paramVec)
        # Result does depend on kernel width, but hopefully not strongly around typical values ~ sigma/4
        kernel_std = (scale or self.std(d)) * h
        # Dependence is from very correlated points due to MCMC rejections; shouldn't need more than about correlation length
        if maxoff is None: maxoff = int(self.getCorrelationLength(d, weight_units=False) * 1.5) + 4
        uncorr_len = self.numrows // 2
        UncorrTerm = 0
        nav = 0
        # first get expected value of each term for uncorrelated samples
        for k in range(uncorr_len, uncorr_len + 5):
            nav += self.numrows - k
            diff2 = (d[:-k] - d[k:]) ** 2 / kernel_std ** 2
            UncorrTerm += np.dot(np.exp(-diff2 / 4) * self.weights[:-k], self.weights[k:])
        UncorrTerm /= nav

        corr = np.zeros(maxoff + 1)
        corr[0] = np.dot(self.weights, self.weights)
        n = float(self.numrows)
        for k in range(1, maxoff + 1):
            diff2 = (d[:-k] - d[k:]) ** 2 / kernel_std ** 2
            corr[k] = np.dot(np.exp(-diff2 / 4) * self.weights[:-k], self.weights[k:]) - (n - k) * UncorrTerm
            if corr[k] < min_corr * corr[0]:
                corr[k] = 0
                break
        N = corr[0] + 2 * np.sum(corr[1:])
        return self.get_norm() ** 2 / N

    def weighted_sum(self, paramVec, where=None):
        paramVec = self._makeParamvec(paramVec)
        if where is None: return self.weights.dot(paramVec)
        return np.dot(paramVec[where], self.weights[where])

    def get_norm(self, where=None):
        if where is None:
            if self.norm is None: self.norm = np.sum(self.weights)
            return self.norm
        else:
            return np.sum(self.weights[where])

    def mean(self, paramVec, where=None):
        return self.weighted_sum(paramVec, where) / self.get_norm(where)

    def var(self, paramVec, where=None):
        if where is not None:
            return np.dot(self.mean_diff(paramVec, where) ** 2, self.weights[where]) / self.get_norm(where)
        else:
            return np.dot(self.mean_diff(paramVec) ** 2, self.weights) / self.get_norm()

    def std(self, paramVec, where=None):
        return np.sqrt(self.var(paramVec, where))

    def cov(self, pars=None, where=None):
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
        return covToCorr(self.cov(pars))

    def mean_diff(self, paramVec, where=None):
        if isinstance(paramVec, six.integer_types) and paramVec >= 0 and where is None:
            if self.diffs is not None:
                return self.diffs[paramVec]
            return self.samples[:, paramVec] - self.getMeans()[paramVec]
        paramVec = self._makeParamvec(paramVec)
        if where is None:
            return paramVec - self.mean(paramVec)
        else:
            return paramVec[where] - self.mean(paramVec, where)

    def mean_diffs(self, pars=None, where=None):
        if pars is None: pars = self.n
        if isinstance(pars, six.integer_types) and pars >= 0 and where is None:
            means = self.getMeans()
            return [self.samples[:, i] - means[i] for i in range(pars)]
        return [self.mean_diff(i, where) for i in pars]

    def twoTailLimits(self, paramVec, confidence):
        limits = np.array([(1 - confidence) / 2, 1 - (1 - confidence) / 2])
        return self.confidence(paramVec, limits)

    def initParamConfidenceData(self, paramVec, start=0, end=None, weights=None):
        if weights is None: weights = self.weights
        d = ParamConfidenceData()
        d.paramVec = self._makeParamvec(paramVec)[start:end]
        d.norm = np.sum(weights[start:end])
        d.indexes = d.paramVec.argsort()
        weightsort = weights[start + d.indexes]
        d.cumsum = np.cumsum(weightsort)
        return d

    def confidence(self, paramVec, limfrac, upper=False, start=0, end=None, weights=None):
        """ 
        Raw sample confidence limits, not using kernel densities
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
        Returns w, M, where w is the eigenvalues of the signal to noise (small means better constrained)
        """
        C = self.cov(params)
        return getSignalToNoise(C, noise, R, eigs_only)

    def thin_indices(self, factor, weights=None):
        """
        Indices to make single weight 1 samples. Assumes intefer weights
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

    def randomSingleSamples_indices(self):
        max_weight = np.max(self.weights)
        thin_ix = []
        for i in range(self.numrows):
            P = self.weights[i] / max_weight
            if random.random() < P:
                thin_ix.append(i)
        return np.array(thin_ix, dtype=np.int)

    def thin(self, factor):
        thin_ix = self.thin_indices(factor)
        self.setSamples(self.samples[thin_ix, :], loglikes=self.loglikes[thin_ix])

    def filter(self, where):
        self.setSamples(self.samples[where, :], self.weights[where], self.loglikes[where])

    def reweightAddingLogLikes(self, logLikes):
        scale = np.min(logLikes)
        self.loglikes += logLikes
        self.weights *= np.exp(-(logLikes - scale))
        self._weightsChanged()

    def cool(self, cool):
        MaxL = np.max(self.loglikes)
        newL = self.loglikes * cool
        self.weights = self.weights * np.exp(-(newL - self.loglikes) - (MaxL * (1 - cool)))
        self.loglikes = newL
        self._weightsChanged()

    def deleteZeros(self):
        self.filter(self.weights == 0)

    def deleteFixedParams(self):
        fixed = []
        for i in range(self.samples.shape[1]):
            if np.all(self.samples[:, i] == self.samples[0, i]):
                fixed.append(i)
        self.changeSamples(np.delete(self.samples, fixed, 1))
        return fixed

    def removeBurn(self, remove=0.3):
        if remove >= 1:
            ix = int(remove)
        else:
            ix = int(round(self.numrows * remove))
        if self.weights is not None:
            self.weights = self.weights[ix:]
        if self.loglikes is not None:
            self.loglikes = self.loglikes[ix:]
        self.changeSamples(self.samples[ix:, :])


class Chains(WeightedSamples):
    def __init__(self, root=None, jobItem=None, paramNamesFile=None, names=None, labels=None, **kwargs):
        WeightedSamples.__init__(self, **kwargs)
        self.jobItem = jobItem
        self.precision = '%.8e'
        self.ignore_lines = float(kwargs.get('ignore_rows', 0))
        self.root = root
        if not paramNamesFile and root and os.path.exists(root + '.paramnames'):
            paramNamesFile = root + '.paramnames'
        self.needs_update = True
        self.chains = None
        self.setParamNames(paramNamesFile or names)
        if labels is not None:
            self.paramNames.setLabels(labels)

    def setParamNames(self, names=None):
        self.paramNames = None
        if isinstance(names, ParamNames):
            self.paramNames = names
        elif isinstance(names, six.string_types):
            self.paramNames = ParamNames(names)
        elif names is not None:
            self.paramNames = ParamNames(names=names)
        elif self.samples is not None:
            self.paramNames = ParamNames(default=self.n)
        if self.paramNames:
            self.getParamIndices()

    def getParamNames(self):
        return self.paramNames

    def getParamIndices(self):
        if self.samples is not None and len(self.paramNames.names) != self.n:
            raise WeightedSampleError("paramNames size does not match number of parameters in samples")
        index = dict()
        for i, name in enumerate(self.paramNames.names):
            index[name.name] = i
        self.index = index
        return self.index

    def setParams(self, obj):
        # makes obj.xx, obj.yy etc as given by the parameter names, where
        # obj.xx is the vector of samples with name 'xx'
        # if xx is of the form aa.bb.cc, it makes subobjects so you can reference obj.aa.bb.cc
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
        pars = ParSamples()
        self.setParams(pars)
        return pars

    def _makeParamvec(self, par):
        if self.needs_update: self.updateBaseStatistics()
        if isinstance(par, ParamInfo): par = par.name
        if isinstance(par, six.string_types):
            return self.samples[:, self.index[par]]
        return WeightedSamples._makeParamvec(self, par)

    def updateChainBaseStatistics(self):
        # old name, use updateBaseStatistics
        return self.updateBaseStatistics()

    def updateBaseStatistics(self):
        self.getVars()
        self.mean_mult = self.norm / self.numrows
        self.max_mult = np.max(self.weights)
        self.getParamIndices()
        self.needs_update = False
        return self

    def addDerived(self, paramVec, name, **kwargs):
        if self.paramNames.parWithName(name):
            raise ValueError('Parameter with name %s already exists' % name)
        self.changeSamples(np.c_[self.samples, paramVec])
        return self.paramNames.addDerived(name, **kwargs)

    def loadChains(self, root, files, ignore_lines=None):
        self.chains = []
        self.samples = None
        self.weights = None
        self.loglikes = None
        self.name_tag = self.name_tag or os.path.basename(root)
        for fname in files:
            if print_load_details: print(fname)
            self.chains.append(WeightedSamples(fname, ignore_lines or self.ignore_lines))
        if len(self.chains) == 0:
            raise WeightedSampleError('loadChains - no chains found for ' + root)
        if self.paramNames is None:
            self.paramNames = ParamNames(default=self.chains[0].n)
        self._weightsChanged()
        return len(self.chains) > 0

    def getGelmanRubinEigenvalues(self, nparam=None, chainlist=None):
        # Assess convergence in the var(mean)/mean(var) in the worst eigenvalue
        # c.f. Brooks and Gelman 1997
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
        return np.max(self.getGelmanRubinEigenvalues(nparam, chainlist))

    def makeSingle(self):
        self.chain_offsets = np.cumsum(np.array([0] + [chain.samples.shape[0] for chain in self.chains]))
        weights = np.hstack((chain.weights for chain in self.chains))
        loglikes = np.hstack((chain.loglikes for chain in self.chains))
        self.setSamples(np.vstack((chain.samples for chain in self.chains)), weights, loglikes)
        self.chains = None
        self.needs_update = True
        return self

    def getSeparateChains(self):
        if self.chains is not None:
            return self.chains
        chainlist = []
        for off1, off2 in zip(self.chain_offsets[:-1], self.chain_offsets[1:]):
            chainlist.append(WeightedSamples(samples=self.samples[off1:off2], weights=self.weights[off1:off2],
                                             loglikes=self.loglikes[off1:off2]))
        return chainlist

    def removeBurnFraction(self, ignore_frac):
        if self.samples is not None:
            self.removeBurn(ignore_frac)
            self.chains = None
            self.needs_update = True
        else:
            for chain in self.chains:
                chain.removeBurn(ignore_frac)

    def deleteFixedParams(self):
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
        self.getParamIndices()

    def saveAsText(self, root, chain_index=None, make_dirs=False):
        if self.loglikes is not None:
            loglikes = self.loglikes
        else:
            loglikes = np.zeros(self.numrows)
        if make_dirs and not os.path.exists(os.path.dirname(root)):
            os.makedirs(os.path.dirname(root))
        np.savetxt(root + ('' if chain_index is None else '_' + str(chain_index + 1)) + '.txt',
                   np.hstack((self.weights.reshape(-1, 1), loglikes.reshape(-1, 1), self.samples)),
                   fmt=self.precision)
        if not chain_index: self.paramNames.saveAsText(root + '.paramnames')

    def savePickle(self, filename):
        import pickle

        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
