import numpy as np
from scipy.interpolate import splrep, splev, RectBivariateSpline, LinearNDInterpolator
from typing import Sequence


class DensitiesError(Exception):
    pass


defaultContours = (0.68, 0.95)


class InterpGridCache:
    __slots__ = "factor", "grid", "sortgrid", "bign", "norm", "softgrid", "cumsum"


# noinspection PyTypeChecker
def getContourLevels(inbins, contours=defaultContours, missing_norm=0, half_edge=True):
    """
     Get contour levels enclosing "contours" fraction of the probability, for any dimension bins array

     :param inbins: binned density.
     :param contours: list or tuple of confidence contours to calculate, default [0.68, 0.95]
     :param missing_norm: accounts of any points not included in inbins
                         (e.g. points in far tails that are not in inbins)
     :param half_edge: If True, edge bins are only half integrated over in each direction.
     :return: list of density levels

    """
    contour_levels = np.zeros(len(contours))
    if half_edge:
        abins = inbins.copy()
        lastindices = [-1] + [slice(None, None, None) for _ in abins.shape[1:]]
        firstindices = [0] + [slice(None, None, None) for _ in abins.shape[1:]]
        for _ in abins.shape:
            abins[tuple(lastindices)] /= 2
            abins[tuple(firstindices)] /= 2
            lastindices = np.roll(lastindices, 1)
            firstindices = np.roll(firstindices, 1)
    else:
        abins = inbins
    norm = np.sum(abins)
    targets = (1 - np.array(contours)) * norm - missing_norm
    bins = abins.reshape(-1)
    indexes = inbins.reshape(-1).argsort()
    sortgrid = bins[indexes]
    cumsum = np.cumsum(sortgrid)
    ixs = np.searchsorted(cumsum, targets)
    for i, ix in enumerate(ixs):
        if ix == 0:
            raise DensitiesError("Contour level outside plotted ranges")
        h = cumsum[ix] - cumsum[ix - 1]
        d = (cumsum[ix] - targets[i]) / h
        contour_levels[i] = sortgrid[ix] * (1 - d) + d * sortgrid[ix - 1]
    return contour_levels


class GridDensity:
    """
    Base class for probability density grids (normalized or not)

    :ivar P: array of density values
    """
    norm: float
    axes: Sequence[np.ndarray]
    view_ranges: Sequence[Sequence[float]]
    norm_integral: callable

    def normalize(self, by='integral', in_place=False):
        """
        Normalize the density grid

        :param by: 'integral' for standard normalization, or 'max', to normalize so the maximum value is unity
        :param in_place: if True, normalize in place, otherwise make copy (in case self.P is used elsewhere)
        """

        if by == 'integral':
            norm = self.norm_integral()
        elif by == 'max':
            norm = np.max(self.P)
            if norm == 0:
                raise DensitiesError('no samples in bin')
        else:
            raise DensitiesError("Density: unknown normalization")
        if in_place:
            self.P /= norm
        else:
            self.setP(self.P / norm)
        self.spl = None
        return self

    def setP(self, P=None):
        """
        Set the density grid values

        :param P: numpy array of density values
        """
        if P is not None:
            for size, ax in zip(P.shape, self.axes):
                if size != ax.size:
                    raise DensitiesError("Array size mismatch in Density arrays: P %s, axis %s" % (size, ax.size))
            self.P = P
        else:
            self.P = np.zeros([ax.size for ax in self.axes])
        self.spl = None

    def bounds(self):
        """
         Get bounds in order x, y, z..

         :return: list of (min,max) values
        """
        if self.view_ranges is not None:
            return self.view_ranges
        else:
            b = [(ax[0], ax[-1]) for ax in self.axes]
            b.reverse()
        return b

    def getContourLevels(self, contours=defaultContours):
        """
        Get contour levels

        :param contours: list of confidence limits to get (default [0.68, 0.95])
        :return: list of contour levels
        """
        return getContourLevels(self.P, contours)


class Density1D(GridDensity):
    """
    Class for 1D marginalized densities, inheriting from :class:`GridDensity`.
    You can call it like a InterpolatedUnivariateSpline obect to get interpolated values, or call Prob.

    """

    def __init__(self, x, P=None, view_ranges=None):
        """
        :param x: array of x values
        :param P: array of densities at x values
        :param view_ranges: optional range for viewing density
        """
        self.n = x.size
        self.axes = [x]
        self.x = x
        self.view_ranges = view_ranges
        self.spacing = x[1] - x[0]
        self.setP(P)

    def bounds(self):
        """
        Get min, max bounds (from view_ranges if set)
        """
        if self.view_ranges is not None:
            return self.view_ranges
        return self.x[0], self.x[-1]

    def _initSpline(self):
        self.spl = splrep(self.x, self.P, s=0)

    def Prob(self, x, derivative=0):
        """
        Calculate density at position x by interpolation in the density grid

        :param x: x value
        :param derivative: optional order of derivative to calculate (default: no derivative)
        :return: P(x) density value
        """
        if self.spl is None:
            self._initSpline()
        if isinstance(x, (np.ndarray, list, tuple)):
            return splev(x, self.spl, derivative, ext=1)
        else:
            return splev([x], self.spl, derivative, ext=1)

    __call__ = Prob

    def integrate(self, P):
        return ((P[0] + P[-1]) / 2 + np.sum(P[1:-1])) * self.spacing

    def norm_integral(self):
        return self.integrate(self.P)

    def initLimitGrids(self, factor=None):

        if self.spl is None:
            self._initSpline()

        g = InterpGridCache()
        if factor is None:
            g.factor = max(2, 20000 // self.n)
        else:
            g.factor = factor
        g.bign = (self.n - 1) * g.factor + 1
        vecx = self.x[0] + np.arange(g.bign) * self.spacing / g.factor
        g.grid = splev(vecx, self.spl)

        norm = np.sum(g.grid)
        g.norm = norm - (0.5 * self.P[-1]) - (0.5 * self.P[0])

        g.sortgrid = np.sort(g.grid)
        g.cumsum = np.cumsum(g.sortgrid)
        return g

    def getLimits(self, p, interpGrid=None, accuracy_factor=None):
        """
        Get parameter equal-density confidence limits (a credible interval).
        If the density is bounded, may only have a one-tail limit.

        :param p: list of limits to calculate, e.g. [0.68, 0.95]
        :param interpGrid: optional pre-computed cache
        :param accuracy_factor: parameter to boost default accuracy for fine sampling
        :return: list of (min, max, has_min, has_top) values
                where has_min and has_top are True or False depending on whether lower and upper limit exists
        """
        g = interpGrid or self.initLimitGrids(accuracy_factor)
        parr = np.atleast_1d(p)
        targets = (1 - parr) * g.norm
        ixs = np.searchsorted(g.cumsum, targets)
        results = []
        for ix, target in zip(ixs, targets):
            trial = g.sortgrid[ix]
            if ix > 0:
                d = g.cumsum[ix] - g.cumsum[ix - 1]
                frac = (g.cumsum[ix] - target) / d
                trial = (1 - frac) * trial + frac * g.sortgrid[ix + 1]

            finespace = self.spacing / g.factor
            lim_bot = (g.grid[0] >= trial)
            if lim_bot:
                mn = self.x[0]
            else:
                i = np.argmax(g.grid > trial)
                d = (g.grid[i] - trial) / (g.grid[i] - g.grid[i - 1])
                mn = self.x[0] + (i - d) * finespace

            lim_top = (g.grid[-1] >= trial)
            if lim_top:
                mx = self.x[-1]
            else:
                i = g.bign - np.argmax(g.grid[::-1] > trial) - 1
                d = (g.grid[i] - trial) / (g.grid[i] - g.grid[i + 1])
                mx = self.x[0] + (i + d) * finespace
            if parr is not p:
                return mn, mx, lim_bot, lim_top
            results.append((mn, mx, lim_bot, lim_top))
        return results


class Density2D(GridDensity):
    """
    Class for 2D marginalized densities, inheriting from :class:`GridDensity`.
    You can call it like a :class:`~scipy:scipy.interpolate.RectBivariateSpline` object to get interpolated values.
    """

    def __init__(self, x, y, P=None, view_ranges=None):
        """
        :param x: array of x values
        :param y: array of y values
        :param P: 2D array of density values at x, y
        :param view_ranges: optional ranges for viewing density
        """
        self.x = x
        self.y = y
        self.axes = [y, x]
        self.view_ranges = view_ranges
        self.spacing = (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
        self.setP(P)

    def integrate(self, P):
        norm = np.sum(P[1:-1, 1:-1]) + (P[0, 0] + P[0, -1] + P[-1, 0] + P[-1, -1]) / 4.0 \
               + (np.sum(P[1:-1, 0]) + np.sum(P[0, 1:-1]) + np.sum(P[1:-1, -1]) + np.sum(P[-1, 1:-1])) / 2.0
        norm *= self.spacing
        return norm

    def norm_integral(self):
        return self.integrate(self.P)

    def _initSpline(self):
        self.spl = RectBivariateSpline(self.x, self.y, self.P.T, s=0)

    def Prob(self, x, y, grid=False):
        """
        Evaluate density at x,y using interpolation

        :param x: x value or array
        :param y: y value or array
        :param grid: whether to make a grid, see :class:`~scipy:scipy.interpolate.RectBivariateSpline`. Default False.
        """
        self.__call__(x, y, grid=grid)

    def __call__(self, *args, **kwargs):
        if self.spl is None:
            self._initSpline()
        return self.spl(*args, **kwargs)


class DensityND(GridDensity):
    """
    Class for ND marginalized densities, inheriting from :class:`GridDensity`
    and :class:`~scipy:scipy.interpolate.LinearNDInterpolator`.

    This is not well tested recently.
    """

    def __init__(self, xs, P=None, view_ranges=None):
        """
        :param xs: list of arrays of x values
        :param P: ND array of density values at xs
        :param view_ranges: optional ranges for viewing density
        """
        self.dim = len(xs)

        # for compatability and tests
        self.x = xs[0]
        if self.dim >= 2:
            self.y = xs[1]
        if self.dim >= 3:
            self.z = xs[2]

        self.xs = xs
        self.axes = xs[::-1]
        self.view_ranges = view_ranges

        self.spacing = 1.
        for i in range(len(xs)):
            self.spacing = self.spacing * (self.xs[i][1] - self.xs[i][0])

        self.setP(P)

    def integrate(self, P):

        ndim = len(P)

        multinorm = np.zeros(ndim + 1)

        # count inner volume and all boundaries separately
        for ind in np.ndindex(P.shape):
            dni = list(np.asarray(ind) - np.asarray(P.shape) + 1)
            # how many indices are on the minimal value
            niszero = ind.count(0)
            # how many indices are on the maximal value
            nismax = dni.count(0)

            # number of indices matching boundaries
            nboundaries = niszero + nismax

            # accumulate P over all hypersurfaces
            multinorm[nboundaries] += P[ind]

        norm = 0.
        for i in range(len(multinorm)):
            # i=0, we are inside
            # i=1, one coordinate on the boundary -> hypercube face of dimension D-1 -> 1 co-dimension
            # i=2, two coordinates on the boundary -> hypercube line of dimension D-2 -> 2 co-dimensions
            # etc...

            norm += multinorm[i] / 2 ** i

        return norm

    def norm_integral(self):
        return self.integrate(self.P)

    def _initSpline(self):
        self.spl = LinearNDInterpolator(self.xs, self.P.T, rescale=True)

    def Prob(self, xs):
        """
        Evaluate density at x,y,z using interpolation
        """
        if self.spl is None:
            self._initSpline()
        return self.spl.__call__(xs)

    __call__ = Prob
