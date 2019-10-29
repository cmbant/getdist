from __future__ import absolute_import
from __future__ import print_function

try:
    from getdist.plots import get_subplot_plotter
except ImportError:
    import sys, os

    sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from getdist.plots import get_subplot_plotter
import matplotlib.pyplot as plt
import numpy as np
from getdist.gaussian_mixtures import Mixture2D, Mixture1D, Gaussian1D, Gaussian2D, make_2D_Cov
from getdist import chains
from getdist.mcsamples import BandwidthError
import logging

default_nsamp = 10000


def simFiles(prob, file_root, sample_lengths=[1000, 2000, 5000, 10000, 20000, 50000, 100000], text=True):
    for nn in sample_lengths:
        samples = prob.MCSamples(nn, logLikes=True)
        if text:
            samples.saveAsText(file_root + '_' + str(nn))
        else:
            samples.savePickle(file_root + '.py_mcsamples')


def compareSimPlot2D(g, samples, density, pars=['x', 'y']):
    g.plot_2d(samples, pars)
    density.normalize('max')
    levels = density.getContourLevels(contours=[0.68, 0.95])
    g.add_2d_density_contours(density, filled=False, color='C0', contour_levels=levels)
    levels = density.getContourLevels(contours=[0.2, 0.4, 0.6, 0.8])
    g.add_2d_density_contours(density, filled=False, color='C3', alpha=0.5, contour_levels=levels)


def compareSimPlot(g, samples, density, par='x', normalized=True):
    g.plot_1d(samples, par, colors=['C0'], normalized=normalized)
    density.normalize('integral' if normalized else 'max')
    plt.plot(density.x, density.P, color='C3')
    if normalized: plt.ylim(0, plt.gca().get_ylim()[1] * 1.1)


def plot1DSim(g, prob, nsamp=default_nsamp, settings={}):
    samps = prob.MCSamples(nsamp, settings=settings)
    compareSimPlot(g, samps, prob.density1D())


def plot2DSim(g, prob, nsamp=default_nsamp, settings={}):
    samps = prob.MCSamples(nsamp, settings=settings)
    compareSimPlot2D(g, samps, prob.density2D())


def compare1D(g, probs, nsamp=default_nsamp, settings={}):
    samples = []
    for i, prob in enumerate(probs):
        samps = prob.MCSamples(nsamp, settings=settings)
        samples.append(samps)

    g.make_figure(len(probs))
    for i, (samps, prob) in enumerate(zip(samples, probs)):
        g._subplot_number(i)
        compareSimPlot(g, samps, prob.density1D())
        g.add_text_left(prob.label, y=0.98, fontsize=8, verticalalignment='top')
    plt.subplots_adjust()


def compare2D(g, probs, nsamp=default_nsamp, settings={}):
    samples = []
    for i, prob in enumerate(probs):
        samps = prob.MCSamples(nsamp, settings=settings)
        samples.append(samps)

    g.make_figure(len(probs))
    for i, (samps, prob) in enumerate(zip(samples, probs)):
        g._subplot_number(i)
        compareSimPlot2D(g, samps, prob.density2D())
        g.add_text_left(prob.label, y=0.96, fontsize=8, verticalalignment='top')
    plt.subplots_adjust()


def get2DMises(prob, nsamp=default_nsamp, nsim=20, scales=np.arange(0.6, 1.5, 0.1), settings={}):
    # Get 2D MISE (mean integrated square error) as function of bandwidth scaling
    Mises = np.zeros(np.asarray(scales).size)
    for _ in range(nsim):
        samps = prob.MCSamples(nsamp, settings=settings)
        for i, scale in enumerate(scales):
            density = samps.get2DDensity('x', 'y', smooth_scale_2D=-scale)
            density.normalize()
            if i == 0:
                xgrid, ygrid = np.meshgrid(density.x, density.y)
                mean = prob.pdf(xgrid, ygrid)
                mean /= density.integrate(mean)
            Mises[i] += np.sum((mean - density.P) ** 2) / np.sum(mean ** 2)
    Mises /= (nsim - 1)
    return scales, Mises


def get1DMises(prob, nsamp=default_nsamp, nsim=50, scales=[0.6, 1.5, 0.1], settings={}):
    # Get 1D MISE (mean integrated square error) as function of bandwidth scaling
    Mises = np.zeros(np.asarray(scales).size)
    failures = 0
    for _ in range(nsim):
        samps = prob.MCSamples(nsamp, settings=settings)
        samps.raise_on_bandwidth_errors = False
        try:
            for i, scale in enumerate(scales):
                density = samps.get1DDensity('x', smooth_scale_1D=-scale)
                density.normalize()
                if i == 0:
                    mean = prob.pdf(density.x)
                    if prob.lims is not None:
                        mean /= density.integrate(mean)
                Mises[i] += np.sum((mean - density.P) ** 2) / np.sum(mean ** 2)
        except BandwidthError:
            failures += 1
    Mises /= (nsim - failures - 1)
    if failures:
        logging.warning('%s failures for 1D bandwidth with %s' % (failures, prob.label))
    return scales, Mises


class Test1DDistributions(object):
    def __init__(self):
        self.gauss = Gaussian1D(0, 0.5, label='Gaussian')
        self.skew = Mixture1D([0, 1], [1, 0.4], [0.6, 0.4], label='skew')
        self.tailed = Mixture1D([0, 0], [1, 3], [0.8, 0.2], label='tailed')
        self.flat = Gaussian1D(0, 3, xmin=-1, xmax=2, label='flat')

        self.broad = Mixture1D([0, 0.3], [1, 2], [0.6, 0.4], label='broad')
        self.flat_top = Mixture1D([0, 1.5, 3], [1, 1, 1], [0.4, 0.2, 0.4], label='flat top')
        self.bimodal = []
        self.bimodal.append(Mixture1D([0, 2], [0.5, 0.5], [0.6, 0.4], label='bimodal 1'))
        self.bimodal.append(Mixture1D([0, 2], [0.2, 0.5], [0.5, 0.5], label='bimodal 2'))
        self.trimodal = []
        self.trimodal.append(Mixture1D([0, 2, 5], [0.2, 0.7, 0.4], label='trimodal'))
        self.cut_gaussians = self.cutGaussians()
        self.shape_set = [self.gauss, self.skew, self.tailed, self.broad, self.flat, self.flat_top]
        self.all = self.shape_set + self.bimodal + self.trimodal + self.cut_gaussians

    def cutGaussians(self, sigma=1, cut_x=[-1.5, -1, -0.5, 0, 1, 1.5]):
        return [Gaussian1D(0, sigma, xmin=cut, label=r'Gaussian [$x>%s$]' % cut) for cut in cut_x]

    def distributions(self):
        return self.all


class Test2DDistributions(object):
    def __init__(self):
        self.gauss = Gaussian2D([0, 0], (0.7, 1, 0.3), label='Gaussian')

        self.bending = Mixture2D([[0, 0], [2, 1.8]], [(np.sqrt(0.5), 1, 0.9), (1, 1, 0.8)], [0.6, 0.4], xmin=-1,
                                 label='bending')

        self.hammer = Mixture2D([[0, 0], [1, 1.8]], [(np.sqrt(0.5), 1, 0.9), (0.3, 1, -0.7)], [0.5, 0.5],
                                label='hammer')

        cov = make_2D_Cov(np.sqrt(0.5), 1, 0.1)
        self.skew = Mixture2D([[0, 0], [0, 1.2]], [cov, cov / 4], [0.5, 0.5], label='skew')

        cov = make_2D_Cov(np.sqrt(0.5), 1, 0.1)
        self.broadtail = Mixture2D([[0, 0], [0, 0.2]], [cov, cov * 8], [0.9, 0.1], label='broad tail')

        self.rotating = Mixture2D([[0, 0], [0, 0.2]], [(1, 1, 0.5), (2, 2, -0.5)], [0.6, 0.4], label='rotating')

        self.tight = Mixture2D([[0, 0], [2.5, 3.5]], [(1, 1, 0.99), (1, 1.5, 0.98)], [0.6, 0.4], label='tight')

        self.cut_correlated = Gaussian2D([0, 0], (0.7, 1, 0.95), ymin=0.3, xmax=1.2, label='cut correlated')

        self.flat = Gaussian2D([0, 0], (1, 2, 0), ymin=-1, ymax=2.1, xmin=-1, xmax=0.2, label='flat')

        self.shape_set = [self.gauss, self.bending, self.hammer, self.skew, self.broadtail, self.rotating, self.tight,
                          self.cut_correlated, self.flat]

        self.cut_gaussians = self.cutGaussians((0.7, 1, 0.3))

        # these examples are from Wand and Jones 93
        self.bimodal = []
        self.bimodal.append(
            Mixture2D([[-1, 0], [1, 0]], [(2. / 3, 2. / 3, 0), (2. / 3, 2. / 3, 0)], label='bimodal WJ1'))
        self.bimodal.append(
            Mixture2D([[-3. / 2, 0], [3. / 2, 0]], [(1. / 4, 1, 0), (1. / 4, 1, 0)], label='bimodal WJ2'))
        self.bimodal.append(
            Mixture2D([[-1, 1], [1, -1]], [(2. / 3, 2. / 3, 3. / 5), (2. / 3, 2. / 3, 3. / 5)], label='bimodal WJ3'))
        self.bimodal.append(
            Mixture2D([[1, -1], [-1, 1]], [(2. / 3, 2. / 3, 7. / 10), (2. / 3, 2. / 3, 0)], label='bimodal WJ4'))

        self.trimodal = []
        self.trimodal.append(Mixture2D([[-6. / 5, 6. / 5], [6. / 5, -6. / 5], [0, 0]],
                                       [(3. / 5, 3. / 5, 3. / 10), (3. / 5, 3. / 5, -3. / 5), (0.25, 0.25, 0.2)],
                                       weights=[9, 9, 2], label='trimodal WJ1'))
        self.trimodal.append(Mixture2D([[-6. / 5, 0], [6. / 5, 0], [0, 0]],
                                       [(3. / 5, 3. / 5, 0.7), (3. / 5, 3. / 5, 0.7), (0.25, 0.25, -0.7)],
                                       label='trimodal WJ2'))
        self.trimodal.append(Mixture2D([[-1, 0], [1, 2 * np.sqrt(3) / 3], [1, -2 * np.sqrt(3) / 3]],
                                       [(0.6, 0.7, 0.6), (0.6, 0.7, 0), (0.4, 0.7, 0)], weights=[3, 3, 1],
                                       label='trimodal WJ3'))

        self.quadrimodal = []
        self.quadrimodal.append(Mixture2D([[-1, 1], [-1, -1], [1, -1], [1, 1]],
                                          [(2. / 3, 2. / 3, 2. / 5), (2. / 3, 2. / 3, 3. / 5), (2. / 3, 2. / 3, -0.7),
                                           (2. / 3, 2. / 3, -0.5)],
                                          weights=[1, 3, 1, 3], label='quadrimodal'))

        self.all = self.shape_set + self.bimodal + self.trimodal + self.quadrimodal + self.cut_gaussians

    def cutGaussians(self, cov, cut_x=[-2, -1, -0.5, 0, 1, 1.5, 2]):
        return [Gaussian2D([0, 0], cov, xmin=cut, label=r'Gaussian [$x>%s$]' % cut) for cut in cut_x]

    def distributions(self):
        return self.all


def plot_compare_method(ax, prob, colors=['k'], sims=100, nsamp=default_nsamp,
                        scalings=[0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5, 1.7], test_settings=[None], linestyles=['-']):
    # compare Parzen estimator with higher order
    print(prob.label, ', size = ', nsamp)
    if len(colors) == 1: colors = colors * len(scalings)
    if len(linestyles) == 1: linestyles = linestyles * len(scalings)
    miselist = np.empty((len(scalings), len(test_settings)))
    for i, (settings, ls, color) in enumerate(zip(test_settings, linestyles, colors)):
        if prob.dim == 1:
            scales, MISEs = get1DMises(prob, nsamp=nsamp, scales=scalings, nsim=sims, settings=settings)
        else:
            scales, MISEs = get2DMises(prob, nsamp=nsamp, scales=scalings, nsim=sims, settings=settings)
        ax.plot(scales, MISEs, ls=ls, color=color)
        miselist[:, i] = MISEs
    for i, scale in enumerate(scalings):
        print(scale, miselist[i, :])
    ax.set_yscale('log')
    ax.set_xlim([scalings[0], scalings[-1]])


def plot_compare_probs_methods(ax, probs, colors=plt.rcParams["axes.prop_cycle"].by_key()["color"], **kwargs):
    for prob, col in zip(probs, colors):
        plot_compare_method(ax, prob, col, **kwargs)


def compare_method_nsims(g, probs, sizes=[1000, 10000], **kwargs):
    g.make_figure(len(sizes))
    for i, size in enumerate(sizes):
        ax = g._subplot_number(i)
        plot_compare_probs_methods(ax, probs, nsmap=size, **kwargs)


def compare_method(probs, nx=2, fname='', **kwargs):
    ny = (len(probs) - 1) // nx + 1
    fig, axs = plt.subplots(ny, nx, sharex=True, sharey=True, squeeze=False, figsize=(nx * 3, ny * 3))
    for i, prob in enumerate(probs):
        ax = axs.reshape(-1)[i]
        plot_compare_method(ax, prob, **kwargs)
        ax.text(0.05, 0.06, prob.label, transform=ax.transAxes,
                horizontalalignment='left', bbox=dict(facecolor='floralwhite', alpha=0.9))
        ax.axvline(1, color='gray', ls='--', alpha=0.5)
        if prob.dim == 2:
            if kwargs.get('nsamp') > 15000:
                ax.set_ylim(6e-6, 8e-3)
            elif kwargs.get('nsamp') > 5000:
                ax.set_ylim(2e-4, 5e-2)
        else:
            if kwargs.get('nsamp') > 15000:
                ax.set_ylim(6e-6, 8e-4)
            elif kwargs.get('nsamp') > 5000:
                ax.set_ylim(4e-5, 6e-3)
    plt.subplots_adjust(wspace=0, hspace=0)
    if fname:
        fig.savefig(fname, bbox_inches='tight')


def join_subplots(ax_array):
    for ax in ax_array.reshape(-1):
        if ax is not None:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)


def run_test_program(plots=['dists_2D', 'dists_1D'], sims=100, nsamp=default_nsamp, mbc=1, bco=1):
    import time

    chains.print_load_details = False
    plt.rc("ytick", direction="in")
    plt.rc("xtick", direction="in")

    test1D = Test1DDistributions()
    test2D = Test2DDistributions()
    test_settings = {'mult_bias_correction_order': mbc, 'boundary_correction_order': bco,
                     'smooth_scale_1D': -1, 'smooth_scale_2D': -1}
    g = get_subplot_plotter(subplot_size=2)

    colors = ['k', 'C0', 'C1', 'C2', 'C3', 'C4']

    if 'ISE_1D' in plots:
        compare_method(test1D.distributions(), nx=3,
                       test_settings=[{'mult_bias_correction_order': 1, 'boundary_correction_order': 1},
                                      {'mult_bias_correction_order': 2, 'boundary_correction_order': 1},
                                      {'mult_bias_correction_order': 0, 'boundary_correction_order': 0},
                                      {'mult_bias_correction_order': 0, 'boundary_correction_order': 1},
                                      {'mult_bias_correction_order': 0, 'boundary_correction_order': 2},
                                      ], colors=colors, linestyles=['-', '-', ':', '-.', '--'],
                       fname='compare_method_1d_N%s.pdf' % nsamp,
                       sims=sims, nsamp=nsamp
                       )

    if 'ISE_2D' in plots:
        compare_method(test2D.distributions(), nx=4,
                       test_settings=[{'mult_bias_correction_order': 1, 'boundary_correction_order': 1},
                                      {'mult_bias_correction_order': 2, 'boundary_correction_order': 1},
                                      {'mult_bias_correction_order': 0, 'boundary_correction_order': 0},
                                      {'mult_bias_correction_order': 0, 'boundary_correction_order': 1},
                                      ], colors=colors, linestyles=['-', '-', ':', '-.', '--'],
                       fname='compare_method_2d_N%s.pdf' % nsamp,
                       sims=sims, nsamp=nsamp
                       )

    if plots is None or 'dists_1D' in plots:
        g.new_plot()
        start = time.time()
        compare1D(g, test1D.distributions(), nsamp=nsamp, settings=test_settings)
        print('1D timing:', time.time() - start)
        join_subplots(g.subplots)
        plt.savefig('test_dists_1D_mbc%s_bco%s_N%s.pdf' % (mbc, bco, nsamp), bbox_inches='tight')

    if plots is None or 'dists_2D' in plots:
        g.new_plot()
        start = time.time()
        compare2D(g, test2D.distributions(), nsamp=nsamp, settings=test_settings)
        print('2D timing:', time.time() - start)
        join_subplots(g.subplots)
        plt.savefig('test_dists_2D_mbc%s_bco%s_N%s.pdf' % (mbc, bco, nsamp), bbox_inches='tight')

    if False:
        print('testing 1D gaussian MISE...')
        scales, MISEs = get1DMises(test1D.gauss)
        for scale, MISE in zip(scales, MISEs):
            print(scale, MISE, np.sqrt(MISE))
        print('testing 2D gaussian MISE...')
        scales, MISEs = get2DMises(test2D.gauss)
        for scale, MISE in zip(scales, MISEs):
            print(scale, MISE, np.sqrt(MISE))


if __name__ == "__main__":
    # program to make the 1D and 2D plots in the getdist notes comparison method accuracies
    # To get all four plots use python test_distributions.py --plots ISE_1D ISE_2D dists_1D dists_2D

    # run_test_program(sims=10, nsamp=default_nsamp)
    import argparse

    parser = argparse.ArgumentParser(description='make getdist test plots from test Gaussian mixture distributions')
    parser.add_argument('--sims', type=int, default=100, help='Number of simulations per case')
    parser.add_argument('--nsamp', type=int, default=10000, help='Number of (independent) samples per simulation')
    parser.add_argument('--plots', nargs='*', default=['dists_1D', 'dists_2D', 'ISE_1D', 'ISE_2D'],
                        help='names of plots to make')
    parser.add_argument('--mbc', type=int, default=1, help='baseline mult_bias_correction_order')
    parser.add_argument('--bco', type=int, default=1, help='baseline boundary_correction_order')
    args = parser.parse_args()
    run_test_program(**vars(args))
