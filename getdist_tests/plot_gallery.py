from getdist import plots
from getdist_tests.test_distributions import Test2DDistributions
import copy


def get_test_samples(nsamp=10000):
    # start with some 2D distributions
    testdists = Test2DDistributions()
    s = testdists.gauss.MCSamples(nsamp, logLikes=True, names=['x', 'y'], labels=['x_1', 'y_1'])
    s2 = testdists.bimodal[0].MCSamples(nsamp, logLikes=True, names=['x', 'y'], labels=['x_1', 'y_1'])
    gauss2 = testdists.gauss.MCSamples(nsamp, logLikes=True, names=['x', 'y'], labels=['x_1', 'y_1'])


    # then add (correlated) samples to the first as new parameters
    p = s.getParams()
    p2 = gauss2.getParams()

    s.addDerived(p.x + p2.x * 0.1 + 0.5, name='x2', label='x_2')
    s.addDerived(p.y + p2.y * 0.3, name='y2', label='y_2')

    # then add (uncorrelated) samples to the first as new parameters
    p = s2.getParams()
    s.addDerived(p.x * 1.1 + 0.5, name='x3', label='x_3')
    s.addDerived(p.y * 0.9, name='y3', label='y_3')

    s.updateChainBaseStatistics()
    return s, s2


def make_plots(samples, s2, ext='.png'):
    g = plots.getSinglePlotter(width_inch=4)
    g.plot_1d(s2, 'x')
    g.export('plot_1d' + ext)

    g = plots.getSinglePlotter(width_inch=4)
    g.plot_1d([samples, s2], 'x')
    g.export('plot_1d_compare' + ext)

    g = plots.getSinglePlotter(width_inch=4)
    g.plot_1d([samples, s2], 'x', normalized=True)
    g.export('plot_1d_normalized' + ext)

    g = plots.getSinglePlotter(width_inch=4)
    g.plot_2d([samples, s2], 'x', 'y')
    g.add_x_marker(0)
    g.add_y_bands(1, 0.4)
    g.export('plot_2d' + ext)

    g = plots.getSinglePlotter(width_inch=4, ratio=1)
    g.plot_2d([samples, s2], 'x', 'y', filled=True)
    g.add_legend(['sim 1', 'sim 2'], colored_text=True)
    g.export('plot_2d_compare_filled' + ext)

    g = plots.getSinglePlotter(width_inch=4)
    g.plot_2d([samples, s2], 'x', 'y', shaded=True)
    g.export('plot_2d_compare_shaded' + ext)

    g = plots.getSinglePlotter(width_inch=4, ratio=3 / 5.)
    g.settings.legend_fontsize = 10
    g.plot_2d([samples, s2], 'x', 'y', filled=True, colors=['green', ('#F7BAA6', '#E03424')], lims=[-4, 7, -3, 3])
    g.add_legend(['Gaussian', '2D bimodal'], legend_loc='upper right');
    g.export('plot_2d_customized' + ext)


    g = plots.getSinglePlotter(width_inch=4)
    g.plot_3d(samples, ['x', 'y', 'x2'])
    g.export('plot_3d' + ext)

    g = plots.getSubplotPlotter(width_inch=7)
    g.settings.axes_fontsize = 9
    g.settings.legend_fontsize = 10
    g.plots_1d(samples, ['x', 'y', 'x2', 'y2'], nx=2)
    g.export('plots_1d' + ext)

    g = plots.getSubplotPlotter(subplot_size=2.5)
    g.plots_2d(samples, param_pairs=[['x', 'y'], ['x2', 'y2']], nx=2, filled=True)
    g.export('plots_2d' + ext)

    # make some offset samples
    samples2 = copy.deepcopy(samples)
    samples2.samples[:, :] += 0.4
    samples2.updateChainBaseStatistics()

    g = plots.getSubplotPlotter(width_inch=7)
    legends = ['Simulation', 'Offset simulation']
    g.settings.axes_fontsize = 9
    g.settings.legend_fontsize = 10
    g.triangle_plot([samples, samples2], ['x', 'y', 'x2', 'y2'], filled_compare=True,
                    legend_labels=legends, legend_loc='upper right')
    g.export('triangle_plot' + ext)

    g = plots.getSubplotPlotter(width_inch=5)
    g.settings.axes_fontsize = 9
    g.settings.legend_fontsize = 11
    g.triangle_plot([samples, samples2], ['x', 'y', 'x2'], plot_3d_with_param='y2', legend_labels=legends)
    g.export('triangle_plot_scatter' + ext)

    g = plots.getSubplotPlotter(width_inch=5)
    g.settings.figure_legend_frame = False
    g.rectangle_plot(['x', 'y'], ['x2', 'y2'], roots=[samples, samples2], filled=True,
                     plot_texts=[['Case 1', None], ['Case 2', None]])
    g.export('rectangle_plot' + ext)


if __name__ == '__main__':
    s1, s2 = get_test_samples()
    make_plots(s1, s2)
