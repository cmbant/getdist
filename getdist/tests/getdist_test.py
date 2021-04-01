import tempfile
import os
import numpy as np
import unittest
import subprocess
import shutil
from getdist import loadMCSamples, plots, IniFile
from getdist.tests.test_distributions import Test2DDistributions, Gaussian1D, Gaussian2D
from getdist.mcsamples import MCSamples
from getdist.styles.tab10 import style_name as tab10
from getdist.styles.planck import style_name as planck
from getdist.parampriors import ParamBounds
from matplotlib import rcParams
import matplotlib.pyplot as plt


class GetDistFileTest(unittest.TestCase):
    """test reading files, convergence routines and getdist script"""

    def setUp(self):
        np.random.seed(10)

        # Simulate some chain files
        prob = Test2DDistributions().bimodal[0]
        self.tempdir = os.path.join(tempfile.gettempdir(), 'gettdist_tests')
        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
        self.root = os.path.join(self.tempdir, 'testchain')
        for n in range(3):
            mcsamples = prob.MCSamples(4000, logLikes=True)
            mcsamples.saveAsText(self.root, chain_index=n)

    def tearDown(self):
        os.chdir(tempfile.gettempdir())
        shutil.rmtree(self.tempdir)

    def testFileLoadPlot(self):
        samples = loadMCSamples(self.root, settings={'ignore_rows': 0.1})
        g = plots.get_single_plotter(chain_dir=self.tempdir, analysis_settings={'ignore_rows': 0.1})
        self.assertEqual(g.samples_for_root('testchain').numrows, samples.numrows,
                         "Inconsistent chain loading")
        self.assertEqual(g.samples_for_root('testchain').getTable().tableTex(),
                         samples.getTable().tableTex(), 'Inconsistent load result')
        samples.getConvergeTests(0.95)
        self.assertAlmostEqual(0.0009368, samples.GelmanRubin, 5, 'Gelman Rubin error, got ' + str(samples.GelmanRubin))

        g = plots.get_single_plotter()
        g.plot_3d(samples, ['x', 'y', 'x'])
        g.export(self.root + '_plot.pdf')

        g = plots.get_single_plotter(chain_dir=self.tempdir,
                                     analysis_settings={'ignore_rows': 0.1, 'contours': [0.68, 0.95, 0.99]})
        g.settings.num_plot_contours = 3
        g.plot_2d('testchain', ['x', 'y'])

    def testGetDist(self):
        from getdist.command_line import getdist_command

        os.chdir(self.tempdir)
        res = getdist_command([self.root])
        # Note this can fail if your local analysis defaults changes the default ignore_rows
        self.assertTrue('-Ln(mean like)  = 2.30' in res, res)
        fname = 'testchain_pars.ini'
        getdist_command(['--make_param_file', fname])
        ini = IniFile(fname)
        ini.params['no_plots'] = False
        ini.params['plot_2D_num'] = 1
        ini.params['plot1'] = 'x y'
        ini.params['num_3D_plots'] = 1
        ini.params['3D_plot1'] = 'x y x'
        ini.params['triangle_params'] = '*[xy]*'

        ini.saveFile(fname)
        res = getdist_command([fname, self.root])
        self.assertTrue('-Ln(mean like)  = 2.30' in res)

        def check_run():
            for f in ['.py', '_2D.py', '_3D.py', '_tri.py']:
                pyname = self.root + f
                self.assertTrue(os.path.isfile(pyname))
                subprocess.check_output(['python', pyname])
                pdf = self.root + f.replace('py', 'pdf')
                self.assertTrue(os.path.isfile(pdf))
                os.remove(pdf)
                os.remove(pyname)

        check_run()


class GetDistTest(unittest.TestCase):
    """test some getdist routines and plotting"""

    def setUp(self):
        np.random.seed(10)
        self.testdists = Test2DDistributions()

    def testBestFit(self):
        samples = self.testdists.bimodal[0].MCSamples(12000, logLikes=True)
        bestSample = samples.getParamBestFitDict(best_sample=True)
        self.assertAlmostEqual(bestSample['loglike'], 1.708, 2)

    def testTables(self):
        self.samples = self.testdists.bimodal[0].MCSamples(12000, logLikes=True)
        self.assertEqual(str(self.samples.getLatex(limit=2)),
                         "(['x', 'y'], ['0.0^{+2.1}_{-2.1}', '0.0^{+1.3}_{-1.3}'])", "MCSamples.getLatex error")
        table = self.samples.getTable(columns=1, limit=1, paramList=['x'])
        self.assertTrue(r'0.0\pm 1.2' in table.tableTex(), "Table tex error")

    def testPCA(self):
        samples = self.testdists.bending.MCSamples(12000, logLikes=True)
        self.assertTrue('e-value: 0.097' in samples.PCA(['x', 'y']))

    def testLimits(self):
        samples = self.testdists.cut_correlated.MCSamples(12000, logLikes=False)
        stats = samples.getMargeStats()
        lims = stats.parWithName('x').limits
        self.assertAlmostEqual(lims[0].lower, 0.2205, 3)
        self.assertAlmostEqual(lims[1].lower, 0.0491, 3)
        self.assertTrue(lims[2].onetail_lower)

        # check some analytics (note not very accurate actually)
        samples = Gaussian1D(0, 1, xmax=1).MCSamples(1500000, logLikes=False)
        stats = samples.getMargeStats()
        lims = stats.parWithName('x').limits
        self.assertAlmostEqual(lims[0].lower, -0.792815, 2)
        self.assertAlmostEqual(lims[0].upper, 0.792815, 2)
        self.assertAlmostEqual(lims[1].lower, -1.72718, 2)

    def testDensitySymmetries(self):
        # check flipping samples gives flipped density
        samps = Gaussian1D(0, 1, xmin=-1, xmax=4).MCSamples(12000)
        d = samps.get1DDensity('x')
        samps.samples[:, 0] *= -1
        samps = MCSamples(samples=samps.samples, names=['x'], ranges={'x': [-4, 1]})
        d2 = samps.get1DDensity('x')
        self.assertTrue(np.allclose(d.P, d2.P[::-1]))

        samps = Gaussian2D([0, 0], np.diagflat([1, 2]), xmin=-1, xmax=2, ymin=0, ymax=3).MCSamples(12000)
        d = samps.get2DDensity('x', 'y')
        samps.samples[:, 0] *= -1
        samps = MCSamples(samples=samps.samples, names=['x', 'y'], ranges={'x': [-2, 1], 'y': [0, 3]})
        d2 = samps.get2DDensity('x', 'y')
        self.assertTrue(np.allclose(d.P, d2.P[:, ::-1]))
        samps.samples[:, 0] *= -1
        samps.samples[:, 1] *= -1
        samps = MCSamples(samples=samps.samples, names=['x', 'y'], ranges={'x': [-1, 2], 'y': [-3, 0]})
        d2 = samps.get2DDensity('x', 'y')
        self.assertTrue(np.allclose(d.P, d2.P[::-1, ::], atol=1e-5))

    def testLoads(self):
        # test initiating from multiple chain arrays
        samps = []
        for i in range(3):
            samps.append(Gaussian2D([1.5, -2], np.diagflat([1, 2])).MCSamples(1001 + i * 10, names=['x', 'y']))
        fromChains = MCSamples(samples=[s.samples for s in samps], names=['x', 'y'])
        mean = np.sum([s.norm * s.mean('x') for s in samps]) / np.sum([s.norm for s in samps])
        meanChains = fromChains.mean('x')
        self.assertAlmostEqual(mean, meanChains)

    def testMixtures(self):
        from getdist.gaussian_mixtures import Mixture2D, GaussianND

        cov1 = [[0.001 ** 2, 0.0006 * 0.05], [0.0006 * 0.05, 0.05 ** 2]]
        cov2 = [[0.01 ** 2, -0.005 * 0.03], [-0.005 * 0.03, 0.03 ** 2]]
        mean1 = [0.02, 0.2]
        mean2 = [0.023, 0.09]
        mixture = Mixture2D([mean1, mean2], [cov1, cov2], names=['zobs', 't'], labels=[r'z_{\rm obs}', 't'],
                            label='Model')
        tester = 0.03
        cond = mixture.conditionalMixture(['zobs'], [tester])
        marge = mixture.marginalizedMixture(['zobs'])
        # test P(x,y) = P(y)P(x|y)
        self.assertAlmostEqual(mixture.pdf([tester, 0.15]), marge.pdf([tester]) * cond.pdf([0.15]))

        samples = mixture.MCSamples(3000, label='Samples')
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples, mixture], filled=False)
        g.new_plot()
        g.plot_1d(cond, 't')

        s1 = 0.0003
        covariance = [[s1 ** 2, 0.6 * s1 * 0.05, 0], [0.6 * s1 * 0.05, 0.05 ** 2, 0.2 ** 2], [0, 0.2 ** 2, 2 ** 2]]
        mean = [0.017, 1, -2]
        gauss = GaussianND(mean, covariance)
        g = plots.get_subplot_plotter()
        g.triangle_plot(gauss, filled=True)

    def testPlots(self):
        self.samples = self.testdists.bimodal[0].MCSamples(12000, logLikes=True)
        g = plots.get_single_plotter(auto_close=True)
        samples = self.samples
        p = samples.getParams()
        samples.addDerived(p.x + (5 + p.y) ** 2, name='z')
        samples.addDerived(p.x, name='x.yx', label='forPattern')
        samples.addDerived(p.y, name='x.2', label='x_2')
        samples.updateBaseStatistics()

        g.plot_1d(samples, 'x')
        g.new_plot()
        g.plot_1d(samples, 'y', normalized=True, marker=0.1, marker_color='b')
        g.new_plot()
        g.plot_2d(samples, 'x', 'y')
        g.new_plot()
        g.plot_2d(samples, 'x', 'y', filled=True)
        g.new_plot()
        g.plot_2d(samples, 'x', 'y', shaded=True)
        g.new_plot()
        g.plot_2d_scatter(samples, 'x', 'y', color='red', colors=['blue'])
        g.new_plot()
        g.plot_3d(samples, ['x', 'y', 'z'])

        g = plots.get_subplot_plotter(width_inch=8.5, auto_close=True)
        g.plots_1d(samples, ['x', 'y'], share_y=True)
        g.new_plot()
        g.triangle_plot(samples, ['x', 'y', 'z'])
        self.assertTrue(g.get_axes_for_params('x', 'z') == g.subplots[2, 0])
        self.assertTrue(g.get_axes_for_params('z', 'x', ordered=False) == g.subplots[2, 0])
        self.assertTrue(g.get_axes_for_params('x') == g.subplots[0, 0])
        self.assertTrue(g.get_axes_for_params('x', 'p', 'q') is None)
        self.assertTrue(g.get_axes(ax=('x', 'z')) == g.subplots[2, 0])
        self.assertTrue(g.get_axes(ax=(2, 0)) == g.subplots[2, 0])

        g.new_plot()
        g.triangle_plot(samples, ['x', 'y'], plot_3d_with_param='z')
        g.new_plot()
        g.rectangle_plot(['x', 'y'], ['z'], roots=samples, filled=True)
        prob2 = self.testdists.bimodal[1]
        samples2 = prob2.MCSamples(12000)
        g.new_plot()
        g.triangle_plot([samples, samples2], ['x', 'y'])
        g.new_plot()
        g.plots_2d([samples, samples2], param_pairs=[['x', 'y'], ['x', 'z']])
        g.new_plot()
        g.plots_2d([samples, samples2], 'x', ['z', 'y'])
        g.new_plot()
        self.assertEqual([name.name for name in samples.paramNames.parsWithNames('x.*')], ['x.yx', 'x.2'])
        g.triangle_plot(samples, 'x.*')
        samples.updateSettings({'contours': '0.68 0.95 0.99'})
        g.settings.num_plot_contours = 3
        g.plot_2d(samples, 'x', 'y', filled=True)
        g.add_y_bands(0.2, 1.5)
        g.add_x_bands(-0.1, 1.2, color='red')
        g.new_plot()
        omm = np.arange(0.1, 0.7, 0.01)
        g.add_bands(omm, 0.589 * omm ** (-0.25), 0.03 * omm ** (-0.25), nbands=3)

        g = plots.get_subplot_plotter()
        import copy
        for upper in [False, True]:
            g.triangle_plot([samples, samples2], ['x', 'y', 'z'], filled=True,
                            upper_roots=[copy.deepcopy(samples)], upper_kwargs={'contour_colors': ['green']},
                            legend_labels=['1', '2', '3'], upper_label_right=upper)
            for i in range(3):
                for j in range(i):
                    self.assertTrue(g.subplots[i, j].get_xlim() == g.subplots[j, i].get_ylim())
                    self.assertTrue(g.subplots[i, j].get_ylim() == g.subplots[j, i].get_xlim())
                    self.assertTrue(g.subplots[i, j].get_xlim() == g.subplots[j, j].get_xlim())

    def test_styles(self):
        tmp = rcParams.copy()
        plots.set_active_style(tab10)
        g = plots.get_single_plotter()
        self.assertEqual(g.settings.line_styles.name, 'tab10')
        plots.set_active_style(planck)
        g = plots.get_single_plotter()
        self.assertTrue(g.settings.prob_y_ticks)
        plots.set_active_style(tab10)
        g = plots.get_single_plotter()
        self.assertEqual(g.settings.line_styles.name, 'tab10')
        plots.set_active_style()
        g = plots.get_single_plotter()
        self.assertFalse(g.settings.prob_y_ticks)
        g = plots.get_single_plotter(style='tab10')
        self.assertEqual(g.settings.line_styles.name, 'tab10')
        plots.set_active_style('planck')
        plots.set_active_style()
        self.assertDictEqual(tmp, rcParams)


class UtilTest(unittest.TestCase):
    """test bounded and unbounded tick assignment"""

    def _plot_with_params(self, scale, x, off, prune, default=False):
        from getdist.matplotlib_ext import BoundedMaxNLocator

        fig, axs = plt.subplots(1, 1, figsize=(x, 1))
        axs.plot([off - scale, off + scale], [0, 1])
        axs.set_yticks([])
        if not default:
            axs.xaxis.set_major_locator(BoundedMaxNLocator(prune=prune))
        axs.xaxis.get_major_formatter().useOffset = False
        fig.suptitle("%s: scale %g, size %g, offset %g" % ('Default' if default else 'Bounded', scale, x, off),
                     fontsize=6)
        return fig, axs

    def test_one_locator(self):
        self._plot_with_params(0.01, 1, 0.05, True)
        plt.draw()

    def test_y(self):
        from getdist.matplotlib_ext import BoundedMaxNLocator
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot([0, 1], [0, 1])
        ax.yaxis.set_major_locator(BoundedMaxNLocator(prune=True))

        def check_ticks(bounds, expected):
            ax.set_ylim(bounds)
            ticks = ax.get_yticks()
            if len(ticks) != len(expected) or not np.allclose(ticks, expected):
                raise self.failureException("Wrong ticks %s for bounds %s" % (ticks, bounds))

        check_ticks([0.0253, 0.02915], [0.026, 0.027, 0.028])

    def test_specifics(self):
        testdists = Test2DDistributions()
        samples = testdists.bimodal[0].MCSamples(1000, logLikes=True)
        g = plots.get_subplot_plotter(auto_close=True)
        g.settings.prob_label = r'$P$'
        g.settings.prob_y_ticks = True
        g.plot_1d(samples, 'x', _no_finish=True)
        ax = g.get_axes()
        self.assertTrue(np.allclose(ax.get_yticks(), [0, 0.5, 1]), "Wrong probability ticks")

        def check_ticks(bounds, expected):
            ax.set_xlim(bounds)
            ticks = ax.get_xticks()
            if len(ticks) != len(expected) or not np.allclose(ticks, expected):
                raise self.failureException("Wrong ticks %s for bounds %s" % (ticks, bounds))

        check_ticks([-5.2, 5.2], [-4, -2, 0, 2, 4])
        check_ticks([0, 8.2], [0, 2, 4, 6, 8])
        check_ticks([0.0219, 0.02232], [0.022, 0.0222])
        check_ticks([-0.009, 0.009], [-0.008, 0., 0.008])
        g.make_figure(nx=2, ny=1, sharey=True)
        ax = g.get_axes()
        g._set_main_axis_properties(ax.xaxis, True)
        ax.set_yticks([])
        check_ticks([-0.009, 0.009], [-0.006, 0., 0.006])
        check_ticks([1, 1.0004], [1.0001, 1.0003])

    def test_locator(self):
        import matplotlib.backends.backend_pdf
        # Set TMPSMALL env variable to save the output PDF for inspection
        local = os.environ.get('TMPSMALL')
        temp = os.path.join(local or tempfile.gettempdir(), 'output.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(temp)
        fails = []
        for x in np.arange(1, 5, 0.5):
            for scale in [1e-4, 0.9e-2, 1e-1, 1, 14, 3000]:
                for off in [scale / 3, 1, 7.4 * scale]:
                    for prune in [True, False]:
                        fig, ax = self._plot_with_params(scale, x, off, prune)
                        pdf.savefig(fig, bbox_inches='tight')
                        if not len(ax.get_xticks()) or x >= 2 > len(ax.get_xticks()) and scale > 1e-4:
                            fails.append([scale, x, off, prune])
                        plt.close(fig)
                    if local:
                        fig, ax = self._plot_with_params(scale, x, off, True, True)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
        pdf.close()
        if not local:
            os.remove(temp)

        self.assertFalse(len(fails), "Too few ticks for %s" % fails)


class CobayaTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = os.path.join(tempfile.gettempdir(), 'gettdist_tests')
        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
        os.chdir(self.tempdir)
        self.path = os.getenv('TRAVIS_BUILD_DIR', os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        self.path = os.path.normpath(os.path.join(self.path, 'getdist_testchains', 'cobaya'))

    def tearDown(self):
        os.chdir(tempfile.gettempdir())
        shutil.rmtree(self.tempdir)

    def test_chains(self):
        if os.path.exists(self.path):
            root = os.path.join(self.path, 'DES_shear')
            samples = loadMCSamples(root, settings={'ignore_rows': 0.3}, no_cache=True)
            self.assertAlmostEqual(samples.mean('ombh2'), 0.02764592190482377, 6)
            pars = samples.getParamSampleDict(10)
            self.assertAlmostEqual(0.06, pars['mnu'], 6)
            self.assertAlmostEqual(samples.getUpper('ns'), 1.07, 6)
            self.assertAlmostEqual(samples.getLower('ns'), 0.87, 6)
            self.assertEqual(samples.getLower('DES_DzS2'), None)
            self.assertAlmostEqual(0, pars['omk'])
            from getdist.command_line import getdist_command
            res = getdist_command([root])
            self.assertTrue('-log(Like) = 95.49' in res, res)

    def test_planck_chains(self):
        if os.path.exists(self.path):
            root = os.path.join(self.path, 'compare_devel_drag')
            samples = loadMCSamples(root, settings={'ignore_rows': 0.3}, no_cache=True)
            self.assertAlmostEqual(samples.mean('ombh2'), 0.0223749, 6)
            self.assertAlmostEqual(samples.getUpper('H0'), 100, 6)
            self.assertEqual(samples.getLower('sigma8'), None)
            samples.saveAsText(r'planck_test')
            ranges = ParamBounds('planck_test.ranges')
            for par in samples.paramNames.names:
                self.assertEqual(samples.getUpper(par.name), ranges.getUpper(par.name))
