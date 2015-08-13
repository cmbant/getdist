import getdist.plots as gp, os
import getdist.densities as gd
import numpy as np
import matplotlib.pyplot as plt

outdir=''

g=gp.getPlotter(chain_dir='chains/', analysis_settings={'ignore_rows': 0.1})
g.settings.setWithSubplotSize(4.0000)
g.settings.shade_meanlikes=True

roots = ['sr2ndlog_base']
params = ['sr2','sr1']
g.plot_2d(roots, params, filled=False)

samples = g.sampleAnalyser.samplesForRoot('sr2ndlog_base')

densityGrid = samples.get2DDensityGridData(params[0],params[1],
                                           num_plot_contours=2)

rawDensityGrid2D = samples.getRawNDDensityGridData(params,
                                                   num_plot_contours=2,
                                                   num_bins_ND=60,
                                                   boundary_correction_order=0,
                                                   meanlikes=True,
                                                   maxlikes=True)


##add shading on maxlikes by overwriting meanlikes
#rawDensityGrid2D.likes = rawDensityGrid2D.maxlikes
#g.add_2d_shading('sr2ndlog_base',params[0],params[1],density=rawDensityGrid2D)

##add contours over maxlikes
#ax = plt.gca()
#CS = ax.contour(rawDensityGrid2D.x, rawDensityGrid2D.y, rawDensityGrid2D.maxlikes,rawDensityGrid2D.maxcontours)


#probability contours
g.add_2d_density_contours(density=rawDensityGrid2D)


g.export(os.path.join(outdir,'testndim.png'))
