import numpy as np
from scipy import fftpack
from scipy.optimize import fsolve, brentq, minimize
from getdist.convolve import dct2d
import logging
import warnings

"""
Code to find optimal bandwidths for basic kernel density estimators in 1 and 2D
Adapted from Matlab code by Zdravko Botev
Extended to include correlation estimation and numerical AMISE minimization in 2D case
Antony Lewis, 2015-03

Original code notice:

Copyright (c) 2007, Zdravko Botev
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
    * Neither the name of The University of Queensland nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

rootpi = np.sqrt(np.pi)
pisquared = np.pi ** 2

_kde_lmax_bandwidth = 7
_kde_consts_1d = np.array([(1 + 0.5 ** (j + 0.5)) / 3 * np.prod(np.arange(1, 2 * j, 2)) / (rootpi / np.sqrt(2.))
                           for j in range(_kde_lmax_bandwidth - 1, 1, -1)])


def _bandwidth_fixed_point(h, N, I, logI, a2):
    if h <= 0:
        return h - 1

    f = 2 * np.pi ** (2 * _kde_lmax_bandwidth) * np.dot(a2,
                                                        np.exp(_kde_lmax_bandwidth * logI - I * (pisquared * h ** 2)))
    for j, const in zip(range(_kde_lmax_bandwidth - 1, 1, -1), _kde_consts_1d):
        try:
            t_j = (const / N / f) ** (2 / (3. + 2 * j))
        except:
            print(f, h, N, j)
            raise
        f = 2 * np.pi ** (2 * j) * np.dot(a2, np.exp(j * logI - I * (pisquared * t_j)))
        if not f:
            raise Exception('zero f in _bandwidth_fixed_point (non-convergence)')
    return h - (2 * N * rootpi * f) ** (-1. / 5)


def bin_samples(samples, range_min=None, range_max=None, nbins=2046, edge_fac=0.1):
    mx = np.max(samples)
    mn = np.min(samples)
    delta = mx - mn
    if range_min is None:
        range_min = mn - delta * edge_fac
    if range_max is None:
        range_max = mx + delta * edge_fac
    R = range_max - range_min
    dx = R / (nbins - 1)
    bins = (samples - range_min) / dx
    return bins.astype(int), R


def gaussian_kde_bandwidth(samples, Neff=None, range_min=None, range_max=None, nbins=2046):
    if Neff is None:
        Neff = np.count_nonzero(np.diff(samples)) + 1
    bins, R = bin_samples(samples, range_min, range_max, nbins)
    data = np.bincount(bins, minlength=nbins)
    h = gaussian_kde_bandwidth_binned(data, Neff)
    if h is None:
        return None
    else:
        return h * R


def gaussian_kde_bandwidth_binned(data, Neff, a=None):
    """
     Return optimal standard kernel bandwidth assuming data is binned from Neff independent samples
     Return value is the bandwidth in units of the range of data (i.e. multiply by max(data)-min(data)),
     or None if failed

     Uses Improved Sheather-Jones algorithm from
     Kernel density estimation via diffusion: Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
     Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
     http://arxiv.org/abs/1011.2602
    """
    I = np.arange(1, data.size) ** 2
    logI = np.log(I)
    if a is None:
        a = fftpack.dct(data / np.sum(data))
    a2 = (a[1:] / 2) ** 2
    try:
        n_scaling = Neff ** (-1. / 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hfrac = 0.53 * n_scaling  #
            hfrac = fsolve(_bandwidth_fixed_point, hfrac, (Neff, I, logI, a2), xtol=hfrac / 20, factor=1)[0]
        if hfrac < 0.019 * n_scaling:
            # may be finding second solution, check with brent
            try:
                hfrac = brentq(_bandwidth_fixed_point, 0.019 * n_scaling, 0.5, (Neff, I, logI, a2), xtol=hfrac / 20)
            except Exception:
                # Could get sign error for the bounds in brentq, in which case small answer may be correct
                # or the method has failed (e.g. flat distribution between two bounds)
                pass
        return hfrac
    except Exception as e:
        logging.warning('1D auto bandwidth failed. Using fallback: %s' % e)
        return None


# 2D functions

K = np.array(
    [1 / np.sqrt(2 * np.pi)] + [(-1) ** j * np.prod(np.arange(1, 2 * j, 2)) / np.sqrt(2 * np.pi) for j in range(1, 5)])
Kodd = np.array([1] + [np.prod(np.arange(1, 2 * j, 2)) / 2. ** (j + 1) / np.sqrt(np.pi) for j in range(1, 9)])


class KernelOptimizer2D:
    def __init__(self, data, Neff, correlation, do_correlation=True, fallback_t=None):
        size = data.shape[0]
        if size != data.shape[1]:
            raise ValueError('KernelOptimizer2D only handles square arrays currently')
        self.a2 = dct2d(data / np.sum(data))[1:, 1:] ** 2
        self.I = np.arange(1, size, dtype=np.float64) ** 2
        self.logI = np.log(self.I)
        self.do_correlation = do_correlation
        if do_correlation:
            self.aFFT = np.fft.fft2(data[:, :] / np.sum(data))
            self.aFFT *= np.conj(self.aFFT)
        self.N = Neff
        self.corr = correlation
        try:
            # t is the bandwidth squared (used for estimating moments), calculated using fixed point
            self.t_star = brentq(self._bandwidth_fixed_point_2D, 0, 0.1, xtol=0.001 ** 2)
            # noinspection PyTypeChecker
            if fallback_t and self.t_star > 0.01 and self.t_star > 2 * fallback_t:
                # For 2D distributions with boundaries, fixed point can overestimate significantly
                logging.debug('KernelOptimizer2D Using fallback (t* > 2*t_gallback)')
                self.t_star = fallback_t
        except Exception:
            if fallback_t is not None:
                # Note the fallback result actually appears to be better in some cases,
                # e.g. Gaussian with four cuts
                logging.debug('2D kernel density optimizer using fallback plugin width %s' % (np.sqrt(fallback_t)))
                self.t_star = fallback_t
            else:
                raise

    def _bandwidth_fixed_point_2D(self, t):
        sum_func = self.func2d([0, 2], t) + self.func2d([2, 0], t) + 2 * self.func2d([1, 1], t)
        time = (2 * np.pi * self.N * sum_func) ** (-1. / 3)
        return (t - time) / time

    def psi(self, s, time):
        w = -self.I * (pisquared * time)
        wx = np.exp(w + self.logI * s[0])
        wy = np.exp(w + self.logI * s[1])
        return (-1) ** np.sum(s) * wy.dot(self.a2).dot(wx.T) * np.pi ** (2 * np.sum(s)) / 4

    def func2d(self, s, t):
        sums = np.sum(s)
        if sums <= 4:
            sum_func = self.func2d([s[0] + 1, s[1]], t) + self.func2d([s[0], s[1] + 1], t)
            const = (1 + 0.5 ** (sums + 1)) / 3
            time = (-2 * const * K[s[0]] * K[s[1]] / self.N / sum_func) ** (1. / (2 + sums))
            return self.psi(s, time)
        else:
            return self.psi(s, t)

    def func2d_odd(self, s, t):
        sums = np.sum(s)
        if sums <= 8:
            sum_func = self.func2d_odd([s[0] + 2, s[1]], t) + self.func2d_odd([s[0], s[1] + 2], t)
            const = 8 * (1 - 2. ** (-sums - 1)) / 3.
            # recall time is h^2
            time = (const * self.p00 * Kodd[s[0]] * Kodd[s[1]] / self.N ** 2 / sum_func ** 2) ** (1. / (3 + sums))
            return self.psi_odd(s, time)
        else:
            return self.psi_odd(s, t)

    def psi_odd(self, s, time):
        f = np.fft.fftfreq(self.aFFT.shape[0], d=1. / self.aFFT.shape[0])
        w = np.exp(-f ** 2 * (4 * pisquared * time))
        wx = w * f ** s[0]
        wy = w * f ** s[1]
        return wy.dot(self.aFFT).real.dot(wx.T) * (2 * np.pi) ** (np.sum(s))

    def AMISE(self, cov, corr=None):
        hx = cov[0]
        hy = cov[1]
        if corr is not None:
            c = corr
        else:
            c = cov[2]
        var = 1. / (4 * np.pi * hx * hy * np.sqrt(1 - c ** 2) * self.N)
        bias = 0.25 * (
                hx ** 4 * self.p[4, 0] + hy ** 4 * self.p[0, 4]
                + 2 * hx ** 2 * hy ** 2 * self.p[2, 2] * (2 * c ** 2 + 1)
                + 4 * c * hx * hy * (hx ** 2 * self.p[3, 1] + hy ** 2 * self.p[1, 3]))
        if bias < 0:
            raise Exception("bias not positive definite")
        return var + bias

    def get_h(self, do_correlation=None):
        if do_correlation is None:
            do_correlation = self.do_correlation
        p = np.zeros((5, 5))
        fixed = False
        tpsi = self.t_star
        if fixed:
            p_02 = self.psi([0, 2], tpsi)
            p_20 = self.psi([2, 0], tpsi)
            p_11 = self.psi([1, 1], tpsi)
        else:
            p_02 = self.func2d([0, 2], tpsi)
            p_20 = self.func2d([2, 0], tpsi)
            p_11 = self.func2d([1, 1], tpsi)
        h_x = (p_02 ** (3. / 4) / (4 * np.pi * self.N * p_20 ** (3. / 4) * (p_11 + np.sqrt(p_20 * p_02)))) ** (1. / 6)
        h_y = (p_20 ** (3. / 4) / (4 * np.pi * self.N * p_02 ** (3. / 4) * (p_11 + np.sqrt(p_20 * p_02)))) ** (1. / 6)
        corr = 0
        if not do_correlation:
            return h_x, h_y, corr

        p[0, 4] = p_02
        p[4, 0] = p_20
        p[2, 2] = p_11

        # p[0, 0] = self.psi([0, 0], tpsi)
        # self.p00 = p[0, 0]
        # p[1, 3] = self.psi_odd([1, 3], tpsi)
        # p[3, 1] = self.psi_odd([3, 1], tpsi)

        p[0, 0] = self.func2d([0, 0], tpsi)
        self.p00 = p[0, 0]
        p[1, 3] = self.func2d_odd([1, 3], tpsi)
        p[3, 1] = self.func2d_odd([3, 1], tpsi)

        self.p = p
        AMISE = self.AMISE(np.array([h_x, h_y, 0]))
        if self.corr:
            try:
                res = minimize(self.AMISE, np.array([h_x, h_y]) / np.sqrt(1 - abs(self.corr)), (self.corr,),
                               method='TNC', bounds=[(0.001, 0.3), (0.001, 0.3)])
                if res.success:
                    AMISEcorr = self.AMISE(res.x, self.corr)
                    if AMISEcorr < AMISE:
                        h_x, h_y = res.x
                        corr = self.corr
                        AMISE = AMISEcorr

            except:
                logging.debug('AMISE fixed correlation optimization failed')
                pass
        try:
            res = minimize(self.AMISE, np.array([h_x, h_y, self.corr]), (None,), method='TNC',
                           bounds=[(0.001, 0.3), (0.001, 0.3), (-0.99, 0.99)])
            if res.success:
                AMISEopt = self.AMISE(res.x)
                if AMISEopt < AMISE * 0.9:
                    h_x, h_y, corr = res.x
        except:
            logging.debug('AMISE optimization failed')
            pass
        return h_x, h_y, corr

    def get_hdiag(self):
        return self.get_h(do_correlation=False)
