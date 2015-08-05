from __future__ import absolute_import
from __future__ import print_function
import tempfile
import os
import re
import numpy as np
import unittest
import subprocess
import shutil
from getdist import loadMCSamples, plots, IniFile
from getdist_tests.test_distributions import Test2DDistributions, Gaussian1D, Gaussian2D
from getdist.mcsamples import MCSamples


class GetDistFileTest(unittest.TestCase):
    """test reading files, convergence routines and GetDist.py script"""

    def setUp(self):
        np.random.seed(10)

        # Simulate some chain files
        prob = Test2DDistributions().bimodal[0]
        self.tempdir = tempfile.gettempdir()
        self.root = os.path.join(self.tempdir, 'testchain')
        for n in range(3):
            mcsamples = prob.MCSamples(4000, logLikes=True)
            mcsamples.saveAsText(self.root, chain_index=n)

    def tearDown(self):
        for f in os.listdir(self.tempdir):
            if re.search('testchain*', f):
                os.remove(os.path.join(self.tempdir, f))

    def testFileLoadPlot(self):
        samples = loadMCSamples(self.root, settings={'ignore_rows': 0.1})
        g = plots.getSinglePlotter(chain_dir=self.tempdir, analysis_settings={'ignore_rows': 0.1})
        self.assertEqual(g.sampleAnalyser.samplesForRoot('testchain').numrows, samples.numrows,
                         "Inconsistent chain loading")
        self.assertEqual(g.sampleAnalyser.samplesForRoot('testchain').getTable().tableTex(),
            samples.getTable().tableTex(), 'Inconsistent load result')
        samples.getConvergeTests(0.95)
        self.assertAlmostEqual(0.0009368, samples.GelmanRubin, 5, 'Gelman Rubin error, got ' + str(samples.GelmanRubin))

        g = plots.getSinglePlotter()
        g.plot_3d(samples, ['x', 'y', 'x'])
        g.export(self.root + '_plot.pdf')

        g = plots.getSinglePlotter(chain_dir=self.tempdir,
                                    analysis_settings={'ignore_rows': 0.1, 'contours':[0.68, 0.95, 0.99]})
        g.settings.num_plot_contours = 3
        g.plot_2d('testchain', ['x', 'y'])

    def testGetDist(self):

        def callGetDist(args):
            if os.getenv('TRAVIS', None):
                return str(subprocess.check_output(['GetDist.py'] + args))
            else:
                return str(subprocess.check_output(['python', os.path.join(os.path.dirname(__file__), '..' + os.sep, 'GetDist.py')] + args))

        os.chdir(self.tempdir)
        res = callGetDist([self.root])
        self.assertTrue('-Ln(mean like)  = 2.30' in res)
        fname = 'testchain_pars.ini'
        callGetDist(['--make_param_file', fname])
        ini = IniFile(fname)
        ini.params['no_plots'] = False
        ini.params['plot_2D_num'] = 1
        ini.params['plot1'] = 'x y'
        ini.params['num_3D_plots'] = 1
        ini.params['3D_plot1'] = 'x y x'
        ini.params['plot_data_dir'] = ''
        ini.params['triangle_params'] = '*[xy]*'

        ini.saveFile(fname)
        res = callGetDist([fname, self.root])
        self.assertTrue('-Ln(mean like)  = 2.30' in res)
        self.assertFalse(os.path.isfile(os.path.join(self.tempdir, 'plot_data', 'testchain_2D_x_y')))
        def checkRun():
            for f in ['.py', '_2D.py', '_3D.py', '_tri.py']:
                pyname = self.root + f
                self.assertTrue(os.path.isfile(pyname))
                subprocess.check_output(['python', pyname])
                pdf = self.root + f.replace('py', 'pdf')
                self.assertTrue(os.path.isfile(pdf))
                os.remove(pdf)
                os.remove(pyname)

        checkRun()

        ini.params['plot_data_dir'] = 'plot_data/'
        ini.saveFile(fname)
        res = callGetDist([fname, self.root])
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'plot_data', 'testchain_2D_x_y')))
        checkRun()
        shutil.rmtree(os.path.join(self.tempdir, 'plot_data'))


class GetDistTest(unittest.TestCase):
    """test some getdist routines and plotting"""

    def setUp(self):
        np.random.seed(10)
        self.testdists = Test2DDistributions()

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
        samps = MCSamples(samples=samps.samples, names=['x'], ranges={'x':[-4, 1]})
        d2 = samps.get1DDensity('x')
        self.assertTrue(np.allclose(d.P, d2.P[::-1]))

        samps = Gaussian2D([0, 0], np.diagflat([1, 2]), xmin=-1, xmax=2, ymin=0, ymax=3).MCSamples(12000)
        d = samps.get2DDensity('x', 'y')
        samps.samples[:, 0] *= -1
        samps = MCSamples(samples=samps.samples, names=['x', 'y'], ranges={'x':[-2, 1], 'y':[0, 3]})
        d2 = samps.get2DDensity('x', 'y')
        self.assertTrue(np.allclose(d.P, d2.P[:, ::-1]))
        samps.samples[:, 0] *= -1
        samps.samples[:, 1] *= -1
        samps = MCSamples(samples=samps.samples, names=['x', 'y'], ranges={'x':[-1, 2], 'y':[-3, 0]})
        d2 = samps.get2DDensity('x', 'y')
        self.assertTrue(np.allclose(d.P, d2.P[::-1, ::], atol=1e-5))


    def testPlots(self):
        self.samples = self.testdists.bimodal[0].MCSamples(12000, logLikes=True)
        g = plots.getSinglePlotter()
        samples = self.samples
        p = samples.getParams()
        samples.addDerived(p.x + (5 + p.y) ** 2, name='z')
        samples.addDerived(p.x, name='x.yx', label='forPattern')
        samples.addDerived(p.y, name='x.2', label='x_2')
        samples.updateBaseStatistics()

        g.plot_1d(samples, 'x')
        g.newPlot()
        g.plot_1d(samples, 'y', normalized=True, marker=0.1, marker_color='b')
        g.newPlot()
        g.plot_2d(samples, 'x', 'y')
        g.newPlot()
        g.plot_2d(samples, 'x', 'y', filled=True)
        g.newPlot()
        g.plot_2d(samples, 'x', 'y', shaded=True)
        g.newPlot()
        g.plot_2d_scatter(samples, 'x', 'y', color='red', colors=['blue'])
        g.newPlot()
        g.plot_3d(samples, ['x', 'y', 'z'])

        g = plots.getSubplotPlotter(width_inch=8.5)
        g.plots_1d(samples, ['x', 'y'], share_y=True)
        g.newPlot()
        g.triangle_plot(samples, ['x', 'y', 'z'])
        g.newPlot()
        g.triangle_plot(samples, ['x', 'y'], plot_3d_with_param='z')
        g.newPlot()
        g.rectangle_plot(['x', 'y'], ['z'], roots=samples, filled=True)
        prob2 = self.testdists.bimodal[1]
        samples2 = prob2.MCSamples(12000)
        g.newPlot()
        g.triangle_plot([samples, samples2], ['x', 'y'])
        g.newPlot()
        g.plots_2d([samples, samples2], param_pairs=[['x', 'y'], ['x', 'z']])
        g.newPlot()
        g.plots_2d([samples, samples2], 'x', ['z', 'y'])
        g.newPlot()
        self.assertEquals([name.name for name in samples.paramNames.parsWithNames('x.*')], ['x.yx', 'x.2'])
        g.triangle_plot(samples, 'x.*')
        samples.updateSettings({'contours': '0.68 0.95 0.99'})
        g.settings.num_contours = 3
        g.plot_2d(samples, 'x', 'y', filled=True)
        g.add_y_bands(0.2, 1.5)
        g.add_x_bands(-0.1, 1.2, color='red')

