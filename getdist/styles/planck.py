import os
from getdist import plots


# Style that roughly follows the Planck parameter papers; uses latex formatting and sans-serif font.

class PlanckPlotter(plots.GetDistPlotter):
    # common setup for matplotlib
    _style_rc = {'axes.labelsize': 9,
                 'font.size': 8,
                 'legend.fontsize': 8,
                 'xtick.labelsize': 8,
                 'ytick.labelsize': 8,
                 'ytick.major.pad': 4,
                 'xtick.major.pad': 4,
                 'text.usetex': True,
                 'text.latex.preamble': r'\usepackage{%s}' % (
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sfmath').replace(os.sep, '/')),
                 'font.family': 'sans-serif',
                 'font.sans-serif': ['FreeSans', 'Tahoma', 'DejaVu Sans', 'Verdana']}

    def set_default_settings(self):
        s = plots.GetDistPlotSettings()
        s.rc_sizes()
        s.legend_frame = False
        s.figure_legend_frame = False
        s.prob_label = r'$P/P_{\rm max}$'
        s.norm_prob_label = 'Probability density'
        s.prob_y_ticks = True
        s.alpha_filled_add = 0.85
        s.solid_contour_palefactor = 0.6

        s.solid_colors = [('#8CD3F5', '#006FED'), ('#F7BAA6', '#E03424'), ('#D1D1D1', '#A1A1A1'), 'g', 'cadetblue',
                          'olive', 'darkcyan']
        s.axis_marker_lw = 0.6
        s.linewidth_contour = 1
        s.colorbar_axes_fontsize = 8
        # This is how to override default labels for parameters with specified names
        s.param_names_for_labels = os.path.normpath(os.path.join(os.path.dirname(__file__), 'planck.paramnames'))
        self.settings = s

    @classmethod
    def get_single_plotter(cls, **kwargs):
        scaling = kwargs.pop('scaling', None)
        kwargs.pop('rc_sizes', None)
        width_inch = kwargs.pop("width_inch", None) or 3.464
        return super().get_single_plotter(scaling=scaling if scaling is not None else False,
                                          rc_sizes=True, width_inch=width_inch, **kwargs)

    @classmethod
    def get_subplot_plotter(cls, **kwargs):
        scaling = kwargs.pop('scaling', None)
        kwargs.pop('rc_sizes', None)
        return super().get_subplot_plotter(scaling=scaling if scaling is not None else False,
                                           rc_sizes=True, **kwargs)


style_name = 'planck'
plots.add_plotter_style(style_name, PlanckPlotter)
