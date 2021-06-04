import numpy as np
from getdist.densities import Density1D, Density2D
from getdist.paramnames import ParamNames
from getdist.mcsamples import MCSamples
import copy


def make_2D_Cov(sigmax, sigmay, corr):
    return np.array([[sigmax ** 2, sigmax * sigmay * corr], [sigmax * sigmay * corr, sigmay ** 2]])


class MixtureND:
    """
    Gaussian mixture model with optional boundary ranges. Includes functions for generating samples and projecting.
    """

    def __init__(self, means, covs, weights=None, lims=None, names=None, label='', labels=None):
        """
        :param means: list of y for each Gaussian in the mixture
        :param covs: list of covariances for the Gaussians in the mixture
        :param weights: optional weight for each component (defaults to equal weight)
        :param lims: optional list of hard limits for each parameter, [[x1min,x1max], [x2min,x2max]];
                     use None for no limit
        :param names: list of names (strings) for each parameter. If not set, set to "param1", "param2"...
        :param label: name for labelling this mixture
        :param labels: list of latex labels for each parameter. If not set, defaults to p_{1}, p_{2}...
        """

        self.means = np.asarray(means)
        self.dim = self.means.shape[1]
        self.covs = [np.array(cov) for cov in covs]
        self.invcovs = [np.linalg.inv(cov) for cov in self.covs]
        if weights is None:
            weights = [1. / len(means)] * len(means)
        self.weights = np.array(weights, dtype=np.float64)
        if np.sum(self.weights) <= 0:
            raise ValueError('Weight <= 0 in MixtureND')
        self.weights /= np.sum(weights)
        self.norms = (2 * np.pi) ** (0.5 * self.dim) * np.array([np.sqrt(np.linalg.det(cov)) for cov in self.covs])
        self.lims = lims
        self.paramNames = ParamNames(names=names, default=self.dim, labels=labels)
        self.names = self.paramNames.list()
        self.label = label
        self.total_mean = np.atleast_1d(np.dot(self.weights, self.means))
        self.total_cov = np.zeros((self.dim, self.dim))
        for mean, cov, weight, totmean in zip(self.means, self.covs, self.weights, self.total_mean):
            self.total_cov += weight * (cov + np.outer(mean - totmean, mean - totmean))

    def sim(self, size, random_state=None):
        """
        Generate an array of independent samples

        :param size: number of samples
        :param random_state: random number Generator or seed
        :return: 2D array of sample values
        """
        tot = 0
        res = []
        block = None
        random_state = np.random.default_rng(random_state)
        while True:
            for num, mean, cov in zip(random_state.multinomial(block or size, self.weights), self.means, self.covs):
                if num > 0:
                    v = random_state.multivariate_normal(mean, cov, size=num)
                    if self.lims is not None:
                        for i, (mn, mx) in enumerate(self.lims):
                            if mn is not None:
                                v = v[v[:, i] >= mn]
                            if mx is not None:
                                v = v[v[:, i] <= mx]
                    tot += v.shape[0]
                    res.append(v)
            if tot >= size:
                break
            if block is None:
                block = min(max(size, 100000), int(1.1 * (size * (size - tot))) // max(tot, 1) + 1)
        samples = np.vstack(res)
        if len(res) > 1:
            samples = random_state.permutation(samples)
        if tot != size:
            samples = samples[:-(tot - size), :]
        return samples

    def MCSamples(self, size, names=None, logLikes=False, random_state=None, **kwargs):
        """
        Gets a set of independent samples from the mixture as a  :class:`.mcsamples.MCSamples` object
        ready for plotting etc.

        :param size: number of samples
        :param names: set to override existing names
        :param logLikes: if True set the sample likelihood values from the pdf, if false, don't store log likelihoods
        :param random_state: random seed or Generator
        :return: a new :class:`.mcsamples.MCSamples` instance
        """
        samples = self.sim(size, random_state=random_state)
        if logLikes:
            loglikes = -np.log(self.pdf(samples))
        else:
            loglikes = None
        return MCSamples(samples=samples, loglikes=loglikes, paramNamesFile=copy.deepcopy(self.paramNames),
                         names=names, ranges=self.lims, **kwargs)

    def autoRanges(self, sigma_max=4, lims=None):
        res = []
        if lims is None:
            lims = self.lims
        if lims is None:
            lims = [(None, None) for _ in range(self.dim)]
        for i, (mn, mx) in enumerate(lims):
            covmin = None
            covmax = None
            if mn is None or mx is None:
                for mean, cov in zip(self.means, self.covs):
                    sigma = np.sqrt(cov[i, i])
                    xmin, xmax = mean[i] - sigma_max * sigma, mean[i] + sigma_max * sigma
                    if mn is not None:
                        xmax = max(xmax, mn + sigma_max * sigma)
                    if mx is not None:
                        xmin = min(xmin, mx - sigma_max * sigma)
                    covmin = min(xmin, covmin) if covmin is not None else xmin
                    covmax = max(xmax, covmax) if covmax is not None else xmax
            res.append((covmin if mn is None else mn, covmax if mx is None else mx))
        return res

    def pdf(self, x):
        """
        Calculate the PDF. Note this assumes x is within the boundaries (does not return zero outside)
        Result is also only normalized if no boundaries.

        :param x: array of parameter values to evaluate at
        :return: pdf at x
        """
        tot = None
        x = np.asarray(x)
        for i, (mean, icov, weight, norm) in enumerate(zip(self.means, self.invcovs, self.weights, self.norms)):
            dx = x - mean
            if len(x.shape) == 1:
                res = np.exp(-icov.dot(dx).dot(dx) / 2) / norm
            else:
                res = np.exp(-np.einsum('ik,km,im->i', dx, icov, dx) / 2) / norm
            if not i:
                tot = res * weight
            else:
                tot += res * weight
        return tot

    def pdf_marged(self, index, x, no_limit_marge=False):
        """
        Calculate the 1D marginalized PDF. Only works if no other parameter limits are marginalized

        :param index: index or name of parameter
        :param x: value to evaluate PDF at
        :param no_limit_marge: if true don't raise an error if mixture has limits
        :return: marginalized 1D pdf at x
        """
        if isinstance(index, str):
            index = self.names.index(index)
        if not no_limit_marge:
            self.checkNoLimits([index])
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
        """
        Get 1D marginalized density. Only works if no hard limits in other parameters.

        :param index: parameter name or index
        :param num_points: number of grid points to evaluate PDF
        :param sigma_max: maximum number of standard deviations away from y to include in computed range
        :param no_limit_marge: if true don't raise error if limits on other parameters
        :return: :class:`~.densities.Density1D` instance
        """
        if isinstance(index, str):
            index = self.names.index(index)
        if not no_limit_marge:
            self.checkNoLimits([index])
        mn, mx = self.autoRanges(sigma_max)[index]
        x = np.linspace(mn, mx, num_points)
        like = self.pdf_marged(index, x)
        return Density1D(x, like)

    def density2D(self, params=None, num_points=1024, xmin=None, xmax=None, ymin=None, ymax=None, sigma_max=5):
        """
        Get 2D marginalized density for a pair of parameters.

        :param params: list of two parameter names or indices to use. If already 2D, can be None.
        :param num_points: number of grid points for evaluation
        :param xmin: optional lower value for first parameter
        :param xmax: optional upper value for first parameter
        :param ymin: optional lower value for second parameter
        :param ymax: optional upper value for second parameter
        :param sigma_max: maximum number of standard deviations away from mean to include in calculated range
        :return: :class:`~.densities.Density2D` instance
        """
        if self.dim > 2 or params is not None or not isinstance(self, Mixture2D):
            mixture = self.marginalizedMixture(params=params)
        elif self.dim != 2:
            raise Exception('density2D requires at least two dimensions')
        else:
            mixture = self

        # noinspection PyProtectedMember
        return mixture._density2D(num_points=num_points, xmin=xmin, xmax=xmax, ymin=ymin,
                                  ymax=ymax, sigma_max=sigma_max)

    def _params_to_indices(self, params):
        indices = []
        if params is None:
            params = self.names
        for p in params:
            if isinstance(p, str):
                indices.append(self.names.index(p))
            elif hasattr(p, 'name'):
                indices.append(self.names.index(p.name))
            else:
                indices.append(p)
        return indices

    def marginalizedMixture(self, params, label=None, no_limit_marge=False) -> 'MixtureND':
        """
        Calculates a reduced mixture model by marginalization over unwanted parameters

        :param params: array of parameter names or indices to retain.
                       If none, will simply return a copy of this mixture.
        :param label: optional label for the marginalized mixture
        :param no_limit_marge: if true don't raise an error if mixture has limits.
        :return: a new marginalized  :class:`MixtureND` instance
        """

        indices = self._params_to_indices(params)
        if not no_limit_marge:
            self.checkNoLimits(indices)
        indices = np.array(indices)
        if self.names is not None:
            names = [self.names[i] for i in indices]
        else:
            names = None
        if self.lims is not None:
            lims = [self.lims[i] for i in indices]
        else:
            lims = None
        if label is None:
            label = self.label
        covs = [cov[np.ix_(indices, indices)] for cov in self.covs]
        means = [mean[indices] for mean in self.means]
        if len(indices) == 2:
            tp = Mixture2D
        else:
            tp = MixtureND
        mixture = tp(means, covs, self.weights, lims=lims,
                     names=names, label=label)
        mixture.paramNames.setLabelsAndDerivedFromParamNames(self.paramNames)
        return mixture

    def conditionalMixture(self, fixed_params, fixed_param_values, label=None):
        """
        Returns a reduced conditional mixture model for the distribution when certainly parameters are fixed.

        :param fixed_params: list of names or numbers of parameters to fix
        :param fixed_param_values:  list of values for the fixed parameters
        :param label: optional label for the new mixture
        :return: A new :class:`MixtureND` instance with cov_i = Projection(Cov_i^{-1})^{-1} and shifted conditional y
        """

        fixed_params = self._params_to_indices(fixed_params)
        self.checkNoLimits(fixed_params)
        keep_params = [i for i in range(self.dim) if i not in fixed_params]
        if not len(keep_params):
            raise ValueError('conditionalMixture must leave at least one non-fixed parameter')
        new_means = []
        new_covs = []
        new_weights = []
        for mean, cov, invcov, weight in zip(self.means, self.covs, self.invcovs, self.weights):
            deltas = np.asarray(fixed_param_values) - mean[fixed_params]
            new_cov = np.linalg.inv(invcov[np.ix_(keep_params, keep_params)])
            new_mean = mean[keep_params] - new_cov.dot(invcov[np.ix_(keep_params, fixed_params)].dot(deltas))
            if len(self.weights) == 1 and False:
                logw = 0
            else:
                logw = invcov[np.ix_(fixed_params, fixed_params)].dot(deltas).dot(deltas) \
                       + np.log(np.linalg.det(cov[np.ix_(fixed_params, fixed_params)]
                                              - cov[np.ix_(fixed_params, keep_params)]
                                              .dot(np.linalg.inv(cov[np.ix_(keep_params, keep_params)])
                                                   .dot(cov[np.ix_(keep_params, fixed_params)]))))
            new_weights.append(logw)
            new_means.append(new_mean)
            new_covs.append(new_cov)

        new_weights = np.exp(-(np.asarray(new_weights) - min(new_weights)) / 2)
        if self.names is not None:
            names = [self.names[i] for i in keep_params]
        else:
            names = None
        mixture = MixtureND(new_means, new_covs, new_weights, names=names, label=label)
        mixture.paramNames.setLabelsAndDerivedFromParamNames(self.paramNames)
        return mixture

    def checkNoLimits(self, keep_params):
        if self.lims is None:
            return
        for i, lim in enumerate(self.lims):
            if i not in keep_params and (lim[0] is not None or lim[1] is not None):
                raise Exception(
                    'In general can only marginalize analytically if no hard boundary limits: ' + self.label)

    def getUpper(self, name):
        if self.lims is None:
            return None
        return self.lims[self.names.index(name)][1]

    def getLower(self, name):
        if self.lims is None:
            return None
        return self.lims[self.names.index(name)][1]


class Mixture2D(MixtureND):
    """
    Gaussian mixture model in 2D with optional boundaries for fixed x and y ranges
    """

    def __init__(self, means, covs, weights=None, lims=None, names=('x', 'y'),
                 xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
        """
        :param means: list of y for each Gaussian in the mixture
        :param covs: list of covariances for the Gaussians in the mixture. Instead of 2x2 arrays,
                     each cov can also be a list of [sigma_x, sigma_y, correlation] parameters
        :param weights: optional weight for each component (defaults to equal weight)
        :param lims: optional list of hard limits for each parameter, [[x1min,x1max], [x2min,x2max]];
                     use None for no limit
        :param names: list of names (strings) for each parameter. If not set, set to x, y
        :param xmin: optional lower hard bound for x
        :param xmax: optional upper hard bound for x
        :param ymin: optional lower hard bound for y
        :param ymax: optional upper hard bound for y
        :param kwargs: arguments passed to :class:`MixtureND`
        """

        if lims is not None:
            limits = self._updateLimits(lims, xmin, xmax, ymin, ymax)
        else:
            limits = [(xmin, xmax), (ymin, ymax)]
        mats = []
        for cov in covs:
            if isinstance(cov, (list, tuple)) and len(cov) == 3 and not isinstance(cov[0], (list, tuple)):
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

    def _density2D(self, num_points=1024, xmin=None, xmax=None, ymin=None, ymax=None, sigma_max=5):
        lims = self._updateLimits(self.lims, xmin, xmax, ymin, ymax)
        (xmin, xmax), (ymin, ymax) = self.autoRanges(sigma_max, lims=lims)
        x = np.linspace(xmin, xmax, num_points)
        y = np.linspace(ymin, ymax, num_points)
        xx, yy = np.meshgrid(x, y)
        like = self.pdf(xx, yy)
        return Density2D(x, y, like)

    def pdf(self, x, y=None):
        """
        Calculate the PDF. Note this assumes x and y are within the boundaries (does not return zero outside)
        Result is also only normalized if no boundaries

        :param x: value of x to evaluate pdf
        :param y: optional value of y to evaluate pdf. If not specified, returns 1D marginalized value for x.
        :return: value of pdf at x or x,y
        """
        if y is None:
            return super().pdf(x)
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
    """
    Simple special case of a 2D Gaussian mixture model with only one Gaussian component
    """

    def __init__(self, mean, cov, **kwargs):
        """
        :param mean: 2 element array with mean
        :param cov: 2x2 array of covariance, or list of [sigma_x, sigma_y, correlation] values
        :param kwargs: arguments passed to :class:`Mixture2D`
        """
        super().__init__([mean], [cov], **kwargs)


class GaussianND(MixtureND):
    """
        Simple special case of a Gaussian mixture model with only one Gaussian component
    """

    def __init__(self, mean, cov, is_inv_cov=False, **kwargs):
        """
        :param mean: array specifying y of parameters
        :param cov: covariance matrix (or filename of text file with covariance matrix)
        :param is_inv_cov: set True if cov is actually an inverse covariance
        :param kwargs: arguments passed to :class:`MixtureND`
        """
        if isinstance(mean, str):
            mean = np.loadtxt(mean)
        if isinstance(cov, str):
            cov = np.loadtxt(cov)
        if is_inv_cov:
            cov = np.linalg.inv(cov)
        super().__init__([mean], [cov], **kwargs)


class Mixture1D(MixtureND):
    """
    Gaussian mixture model in 1D with optional boundaries for fixed ranges
    """

    def __init__(self, means, sigmas, weights=None, lims=None, name='x',
                 xmin=None, xmax=None, **kwargs):
        """
        :param means: array of y for each component
        :param sigmas: array of standard deviations for each component
        :param weights: weights for each component (defaults to equal weight)
        :param lims: optional array limits for each component
        :param name: parameter name (default 'x')
        :param xmin: optional lower limit
        :param xmax:  optional upper limit
        :param kwargs: arguments passed to :class:`MixtureND`
        """
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
    """
    Simple 1D Gaussian
    """

    def __init__(self, mean, sigma, **kwargs):
        """
        :param mean: mean
        :param sigma:  standard deviation
        :param kwargs:  arguments passed to :class:`Mixture1D`
        """
        super().__init__([mean], [sigma], **kwargs)


class RandomTestMixtureND(MixtureND):
    """
    class for randomly generating an N-D gaussian mixture for testing (a mixture with random parameters, not random
    samples from the mixture).
    """

    def __init__(self, ndim=4, ncomponent=1, names=None, weights=None, seed=None, label='RandomMixture'):
        """
        :param ndim: number of dimensions
        :param ncomponent: number of components
        :param names: names for the parameters
        :param weights: weights for each component
        :param seed:  random seed or Generator
        :param label: label for the generated mixture
        """
        random_state = np.random.default_rng(seed)
        covs = []
        for _ in range(ncomponent):
            A = random_state.random((ndim, ndim))
            covs.append(np.dot(A, A.T))
        super().__init__(random_state.random((ncomponent, ndim)), covs, weights=weights,
                         lims=None, names=names, label=label)


def randomTestMCSamples(ndim=4, ncomponent=1, nsamp=10009, nMCSamples=1, seed=10, names=None, labels=None):
    """
    get a MCSamples instance, or list of MCSamples instances with random samples from random covariances and y
    """
    if names is None:
        names = ["x%s" % i for i in range(ndim)]
    if labels is None:
        labels = ["x_{%s}" % i for i in range(ndim)]
    seed = np.random.default_rng(seed)
    result = [RandomTestMixtureND(ndim, ncomponent, names, seed=seed).MCSamples(
        nsamp, labels=labels, name_tag='Sim %s' % (i + 1), random_state=seed) for i in range(nMCSamples)]
    if nMCSamples > 1:
        return result
    else:
        return result[0]
