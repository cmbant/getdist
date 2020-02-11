from getdist import plots
from matplotlib import cm


# Simple style that uses matplotlib's default color table for contours and lines

class DefaultColorsPlotter(plots.GetDistPlotter):
    # noinspection PyUnresolvedReferences
    def set_default_settings(self):
        s = plots.GetDistPlotSettings()
        s.solid_colors = cm.tab10
        s.line_styles = cm.tab10
        s.colormap_scatter = 'viridis'
        self.settings = s


style_name = 'tab10'
plots.add_plotter_style(style_name, DefaultColorsPlotter)
