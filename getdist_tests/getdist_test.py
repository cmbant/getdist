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
from getdist_tests.test_distributions import Test2DDistributions


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

    def testLoad(self):
        samples = loadMCSamples(self.root, dist_settings={'ignore_rows': 0.1})
        g = plots.getSinglePlotter(chain_dir=self.tempdir, analysis_settings={'ignore_rows': 0.1})
        self.assertEqual(g.sampleAnalyser.samplesForRoot('testchain').numrows, samples.numrows,
                         "Inconsistent chain loading")
        self.assertEqual(g.sampleAnalyser.samplesForRoot('testchain').getTable().tableTex(),
            samples.getTable().tableTex(), 'Inconsistent load result')
        samples.getConvergeTests(0.95)
        self.assertAlmostEqual(0.0009368, samples.GelmanRubin, 5, 'Gelman Rubin error, got ' + str(samples.GelmanRubin))

    def testPlotFile(self):
        samples = loadMCSamples(self.root, dist_settings={'ignore_rows': 0.1})
        g = plots.getSinglePlotter()
        g.plot_3d(samples, ['x', 'y', 'x'])
        g.export(self.root + '_plot.pdf')

    def testLoadName(self):
        g = plots.getSinglePlotter(chain_dir=self.tempdir,
                                    analysis_settings={'ignore_rows': 0.3, 'contours':[0.68, 0.95, 0.99]})

        g.settings.num_plot_contours = 3
        g.plot_2d('testchain', ['x', 'y'])

    def testGetDist(self):

        def callGetDist(args):
            if os.getenv('TRAVIS', None):
                return str(subprocess.check_output(['GetDist.py'] + args))
            else:
                return str(subprocess.check_output(['python', os.path.join(os.path.dirname(__file__), '../', 'GetDist.py')] + args))

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
        self.samples = self.testdists.bimodal[0].MCSamples(12000, logLikes=True)

    def testTables(self):
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
        self.assertAlmostEqual(lims[0].lower, 0.2175, 3)
        self.assertAlmostEqual(lims[1].lower, 0.0548, 3)
        self.assertTrue(lims[2].onetail_lower)

    def testPlots(self):
        g = plots.getSinglePlotter()
        samples = self.samples
        p = samples.getParams()
        samples.addDerived(p.x + (5 + p.y) ** 2, name='z')
        samples.updateBaseStatistics()

        g.plot_2d(samples, 'x', 'y')
        g.newPlot()
        g.plot_2d(samples, 'x', 'y', filled=True)
        g.newPlot()
        g.plot_2d(samples, 'x', 'y', shaded=True)
        g.newPlot()
        g.plots_1d(samples, ['x', 'y'])
        g.newPlot()
        g.plot_3d(samples, ['x', 'y', 'z'])
        g = plots.getSubplotPlotter(width_inch=8.5)
        g.triangle_plot(samples, ['x', 'y', 'z'])
        g.newPlot()
        g.triangle_plot(samples, ['x', 'y'], plot_3d_with_param='z')
        g.newPlot()
        g.rectangle_plot(['x', 'y'], ['z'], roots=samples, filled=True)
        prob2 = self.testdists.bimodal[1]
        samples2 = prob2.MCSamples(12000)
        g.newPlot()
        g.triangle_plot([samples, samples2], ['x', 'y'])

        samples.updateSettings({'contours': '0.68 0.95 0.99'})
        g.settings.num_contours = 3
        g.plot_2d(samples, 'x', 'y', filled=True)

