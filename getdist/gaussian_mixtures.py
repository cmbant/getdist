import numpy as np
from getdist import mcsamples, densities
import six


def make_2D_Cov(sigmax, sigmay, corr):
    return np.array([[sigmax ** 2, sigmax * sigmay * corr], [sigmax * sigmay * corr, sigmay ** 2]])


class MixtureND(object):
    """
    Gaussian mixture model with optional boundary ranges
    """

    def __init__(self, means, covs, weights=None, lims=None, names=None, label=''):
        self.means = np.asarray(means)
        self.dim = self.means.shape[1]
        self.covs = [np.array(cov) for cov in covs]
        self.invcovs = [np.linalg.inv(cov) for cov in self.covs]
        if weights is None: weights = [1. / len(means)] * len(means)
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= np.sum(weights)
        self.norms = (2 * np.pi) ** (0.5 * self.dim) * np.array([np.sqrt(np.linalg.det(cov)) for cov in self.covs])
        self.lims = lims
        self.names = names
        self.label = label
        self.total_mean = np.atleast_1d(np.dot(self.weights, self.means))
        self.total_cov = np.zeros((self.dim, self.dim))
        for mean, cov, weight, totmean in zip(self.means, self.covs, self.weights, self.total_mean):
            self.total_cov += weight * (cov + np.outer(mean - totmean, mean - totmean))

    def sim(self, size):
        tot = 0
        res = []
        block = None
        while True:
            for num, mean, cov in zip(np.random.multinomial(block or size, self.weights), self.means, self.covs):
                if num > 0:
                    v = np.random.multivariate_normal(mean, cov, size=num)
                    if self.lims is not None:
                        for i, (mn, mx) in enumerate(self.lims):
                            if mn is not None: v = v[v[:, i] >= mn]
                            if mx is not None: v = v[v[:, i] <= mx]
                    tot += v.shape[0]
                    res.append(v)
            if tot >= size:
                break
            if block is None:
                block = min(max(size, 100000), int(1.1 * (size * (size - tot))) // max(tot, 1) + 1)
        samples = np.vstack(res)
        if len(res) > 1: samples = np.random.permutation(samples)
        if tot != size:
            samples = samples[:-(tot - size), :]
        return samples

    def MCSamples(self, size, names=None, logLikes=False, **kwargs):
        if names is None: names = self.names
        samples = self.sim(size)
        if logLikes:
            loglikes = -np.log(self.pdf(samples))
        else:
            loglikes = None
        return mcsamples.MCSamples(samples=samples, loglikes=loglikes, names=names, ranges=self.lims, **kwargs)

    def autoRanges(self, sigma_max=4, lims=None):
        res = []
        if lims is None: lims = self.lims
        for i, (mn, mx) in enumerate(lims):
            covmin = None
            covmax = None
            if mn is None or mx is None:
                for mean, cov in zip(self.means, self.covs):
                    sigma = np.sqrt(cov[i, i])
                    xmin, xmax = mean[i] - sigma_max * sigma, mean[i] + sigma_max * sigma
                    if mn is not None: xmax = max(xmax, mn + sigma_max * sigma)
                    if mx is not None: xmin = min(xmin, mx - sigma_max * sigma)
                    covmin = min(xmin, covmin) if covmin is not None else xmin
                    covmax = max(xmax, covmax)
            res.append((covmin if mn is None else mn, covmax if mx is None else mx))
        return res

    def pdf(self, x):
        """
        Calculate the PDF. Note this assumes x and y are within the boundaries (does not return zero outside)
        Result is also only normalized if no boundaries
        """
        tot = None
        for i, (mean, icov, weight, norm) in enumerate(zip(self.means, self.invcovs, self.weights, self.norms)):
            dx = x - mean
            res = np.exp(-np.einsum('ik,km,im->i', dx, icov, dx) / 2) / norm
            if not i:
                tot = res * weight
            else:
                tot += res * weight
        return tot

    def pdf_marged(self, index, x, no_limit_marge=False):
        if isinstance(index, six.string_types): index = self.names.index(index)
        if not no_limit_marge: self.checkNoLimits([index])
        tot = None
        for i, (mean, cov, weight) in enumerate(zip(self.means, self.covs, self.weights)):
            dx = x - mean[index]
            var = cov[index, index]
            res = np.exp(-dx ** 2 / var / 2) / np.sqrt(2 * np.pi * var)
            if not i:
                tot = res * weight
            else:
                tot += res * weight
        return tot

    def density1D(self, index=0, num_points=1024, sigma_max=4, no_limit_marge=False):
        if isinstance(index, six.string_types): index = self.names.index(index)
        if not no_limit_marge: self.checkNoLimits([index])
        mn, mx = self.autoRanges(sigma_max)[index]
        x = np.linspace(mn, mx, num_points)
        like = self.pdf_marged(index, x)
        return densities.Density1D(x, like)

    def marginalizedMixture(self, params, label=None, no_limit_marge=False):
        indices = []
        for p in params:
            if isinstance(p, six.string_types):
                indices.append(self.names.index(p))
            else:
                indices.append(p)
        if not no_limit_marge: self.checkNoLimits(indices)
        indices = np.array(indices)
        if self.names is not None:
            names = self.names[indices]
        else:
            names = None
        if self.lims is not None:
            lims = self.lims[indices]
        else:
            lims = None
        if label is None: label = self.label
        covs = [cov[np.ix_(indices, indices)] for cov in self.covs]
        return MixtureND(self.means[indices], covs, self.weights, lims=lims,
                         names=names, label=label)

    def checkNoLimits(self, keep_params):
        if self.lims is None: return
        for i, lim in enumerate(self.lims):
            if not i in keep_params and (lim[0] is not None or lim[1] is not None):
                raise Exception(
                    'In general can only marginalize analytically if no hard boundary limits: ' + self.label)


class Mixture2D(MixtureND):
    """
    Simulate from Gaussian mixture model in 2D with optional boundaries for fixed x and y ranges
    """

    def __init__(self, means, covs, weights=None, lims=None, names=['x', 'y'],
                 xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
        if lims is not None:
            limits = self._updateLimits(lims, xmin, xmax, ymin, ymax)
        else:
            limits = [(xmin, xmax), (ymin, ymax)]
        mats = []
        for cov in covs:
            if isinstance(cov, (list, tuple)) and len(cov) == 3:
                mats.append(make_2D_Cov(*cov))
            else:
                mats.append(cov)
        MixtureND.__init__(self, means, mats, weights, limits, names=names, **kwargs)

    def _updateLimits(self, lims, xmin=None, xmax=None, ymin=None, ymax=None):
        xmin = xmin if xmin is not None else lims[0][0]
        xmax = xmax if xmax is not None else lims[0][1]
        ymin = ymin if ymin is not None else lims[1][0]
        ymax = ymax if ymax is not None else lims[1][1]
        return [(xmin, xmax), (ymin, ymax)]

    def density2D(self, num_points=1024, xmin=None, xmax=None, ymin=None, ymax=None, sigma_max=5):
        lims = self._updateLimits(self.lims, xmin, xmax, ymin, ymax)
        (xmin, xmax), (ymin, ymax) = self.autoRanges(sigma_max, lims=lims)
        x = np.linspace(xmin, xmax, num_points)
        y = np.linspace(ymin, ymax, num_points)
        xx, yy = np.meshgrid(x, y)
        like = self.pdf(xx, yy)
        return densities.Density2D(x, y, like)

    def pdf(self, x, y=None):
        """
        Calculate the PDF. Note this assumes x and y are within the boundaries (does not return zero outside)
        Result is also only normalized if no boundaries
        """
        if y is None: return super(Mixture2D, self).pdf(x)
        tot = None
        for i, (mean, icov, weight, norm) in enumerate(zip(self.means, self.invcovs, self.weights, self.norms)):
            dx = x - mean[0]
            dy = y - mean[1]
            res = np.exp(-(dx ** 2 * icov[0, 0] + 2 * dx * dy * icov[0, 1] + dy ** 2 * icov[1, 1]) / 2) / norm
            if not i:
                tot = res * weight
            else:
                tot += res * weight
        return tot


class Gaussian2D(Mixture2D):
    def __init__(self, mean, cov, **kwargs):
        super(Gaussian2D, self).__init__([mean], [cov], **kwargs)


class Mixture1D(MixtureND):
    """
    Simulate from Gaussian mixture model in 1D with optional boundaries for fixed ranges
    """

    def __init__(self, means, sigmas, weights=None, lims=None, name='x',
                 xmin=None, xmax=None, **kwargs):
        if lims is not None:
            limits = [(xmin if xmin is not None else lims[0], xmax if xmax is not None else lims[1])]
        else:
            limits = [(xmin, xmax)]
        covs = [np.atleast_2d(sigma ** 2) for sigma in sigmas]
        means = [[mean] for mean in means]
        MixtureND.__init__(self, means, covs, weights, limits, names=[name], **kwargs)

    def pdf(self, x):
        return self.pdf_marged(0, x)


class Gaussian1D(Mixture1D):
    def __init__(self, mean, sigma, **kwargs):
        super(Gaussian1D, self).__init__([mean], [sigma], **kwargs)


class RandomTestMixtureND(MixtureND):
    """
    class for randomly generating an N-D gaussian for testing
    """

    def __init__(self, ndim=4, ncomponent=1, names=None, weights=None, seed=0, label='RandomMixture'):
        if seed: np.random.seed(seed)
        covs = []
        for _ in range(ncomponent):
            A = np.random.rand(ndim, ndim)
            covs.append(np.dot(A, A.T))
        super(RandomTestMixtureND, self).__init__(np.random.rand(ncomponent, ndim), covs, weights=weights,
                                                  lims=None, names=names, label=label)


def randomTestMCSamples(ndim=4, ncomponent=1, nsamp=10009, nMCSamples=1, seed=10, names=None, labels=None):
    """
    get a list of MCSamples instances with random samples from random covariances and means
    """
    if seed: np.random.seed(seed)
    if names is None: names = ["x%s" % i for i in range(ndim)]
    if labels is None: labels = ["x_{%s}" % i for i in range(ndim)]
    return [RandomTestMixtureND(ndim, ncomponent, names).MCSamples(nsamp, labels=labels,
                                                                   name_tag='Sim %s' % (i + 1)) for i in
            range(nMCSamples)]
