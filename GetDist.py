#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import os
import subprocess
import getdist
import io
from getdist import MCSamples, chains, IniFile


def runScript(fname):
    subprocess.Popen(['python', fname])


def doError(msg):
    if __name__ == '__main__':
        import sys

        print(msg)
        sys.exit()
    raise ValueError(msg)


def main(args):
    no_plots = False
    chain_root = args.chain_root
    if args.ini_file is None and chain_root is None:
        doError('Must give either a .ini file of parameters or a chain file root name. Run "GetDist.py -h" for help.')
    if not '.ini' in args.ini_file and chain_root is None:
        # use default settings acting on chain_root, no plots
        chain_root = args.ini_file
        args.ini_file = getdist.default_getdist_settings
        no_plots = True
    if not os.path.isfile(args.ini_file):
        doError('Parameter file does not exist: ' + args.ini_file)
    if chain_root and chain_root.endswith('.txt'):
        chain_root = chain_root[:-4]

    # Input parameters
    ini = IniFile(args.ini_file)

    # File root
    if chain_root is not None:
        in_root = chain_root
    else:
        in_root = ini.params['file_root']
    if not in_root:
        doError('Chain Root file name not given ')
    rootname = os.path.basename(in_root)

    if args.ignore_rows is not None:
        ignorerows = args.ignore_rows
    else:
        ignorerows = ini.float('ignore_rows', 0.0)

    samples_are_chains = ini.bool('samples_are_chains', True)
    
    paramnames = ini.string('parameter_names', '')

    # Create instance of MCSamples
    mc = MCSamples(in_root, files_are_chains=samples_are_chains, paramNamesFile=paramnames)

    mc.initParameters(ini)

    if ini.bool('adjust_priors', False) or ini.bool('map_params', False):
        doError(
            'To adjust priors or define new parameters, use a separate python script; see the python getdist docs for examples')

    plot_ext = ini.string('plot_ext', 'py')
    finish_run_command = ini.string('finish_run_command', '')

    no_plots = ini.bool('no_plots', no_plots)
    plots_only = ini.bool('plots_only', False)
    no_tests = plots_only or ini.bool('no_tests', False)

    thin_factor = ini.int('thin_factor', 0)
    thin_cool = ini.float('thin_cool', 1.0)

    make_single_samples = ini.bool('make_single_samples', False)
    single_thin = ini.int('single_thin', 1)
    cool = ini.float('cool', 1.0)

    chain_exclude = ini.int_list('exclude_chain')

    shade_meanlikes = ini.bool('shade_meanlikes', False)
    plot_meanlikes = ini.bool('plot_meanlikes', False)

    dumpNDbins = ini.bool('dump_ND_bins', False)

    out_dir = ini.string('out_dir', './')
    if out_dir:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        print('producing files in directory ', out_dir)
    mc.out_dir = out_dir

    out_root = ini.string('out_root', '')
    if out_root:
        rootname = out_root
        print('producing files with with root ', out_root)
    mc.rootname = rootname

    rootdirname = os.path.join(out_dir, rootname)
    mc.rootdirname = rootdirname

    if 'do_minimal_1d_intervals' in ini.params:
        doError('do_minimal_1d_intervals no longer used; set credible_interval_threshold instead')

    line = ini.string('PCA_params', '')
    if line.lower() == 'all':
        PCA_params = mc.paramNames.list()
    else:
        PCA_params = line.split()
    PCA_num = ini.int('PCA_num', len(PCA_params))
    if PCA_num != 0:
        if PCA_num < 2:
            doError('Can only do PCA for 2 or more parameters')
        PCA_func = ini.string('PCA_func', '')
        # Characters representing functional mapping
        if PCA_func == '':
            PCA_func = ['N'] * PCA_num  # No mapping
        PCA_NormParam = ini.string('PCA_normparam', '') or None

    make_scatter_samples = ini.bool('make_scatter_samples', False)

    # ==============================================================================

    first_chain = ini.int('first_chain', 0)
    last_chain = ini.int('chain_num', -1)
    # -1 means keep reading until one not found

    # Chain files
    chain_files = chains.chainFiles(in_root, first_chain=first_chain, last_chain=last_chain,
                                    chain_exclude=chain_exclude)

    mc.loadChains(in_root, chain_files)

    mc.removeBurnFraction(ignorerows)
    mc.deleteFixedParams()
    mc.makeSingle()

    def filterParList(namestring, num=None):
        if not namestring.strip():
            pars = mc.paramNames.list()
        else:
            pars = []
            for name in namestring.split():
                if '?' in name or '*' in name:
                    pars += mc.paramNames.getMatches(name, strings=True)
                elif mc.paramNames.parWithName(name):
                    pars.append(name)
        if num is not None and len(pars) != num:
            print('%iD plot has missing parameter or wrong number of parameters: %s' % (num, pars))
            pars = None
        return pars


    if cool != 1:
        print('Cooling chains by ', cool)
        mc.cool(cool)

    mc.updateBaseStatistics()

    if not no_tests:
        mc.getConvergeTests(mc.converge_test_limit, writeDataToFile=True, feedback=True)

    mc.writeCovMatrix()
    mc.writeCorrelationMatrix()

    # Output thinned data if requested
    # Must do this with unsorted output
    if thin_factor != 0:
        thin_ix = mc.thin_indices(thin_factor)
        filename = rootdirname + '_thin.txt'
        mc.writeThinData(filename, thin_ix, thin_cool)

    print(mc.getNumSampleSummaryText().strip())
    if mc.likeStats: print(mc.likeStats.likeSummary().strip())

    if PCA_num > 0 and not plots_only:
        mc.PCA(PCA_params, PCA_func, PCA_NormParam, writeDataToFile=True)

    if not no_plots or dumpNDbins:
        # set plot_data_dir before we generate the 1D densities below
        plot_data_dir = ini.string('plot_data_dir', default='', allowEmpty=True)
        if plot_data_dir and not os.path.isdir(plot_data_dir):
            os.mkdir(plot_data_dir)
    else:
        plot_data_dir = None
    mc.plot_data_dir = plot_data_dir

    # Do 1D bins
    mc._setDensitiesandMarge1D(writeDataToFile=not no_plots and plot_data_dir, meanlikes=plot_meanlikes)

    if not no_plots:
        # Output files for 1D plots
        print('Calculating plot data...')

        plotparams = []
        line = ini.string('plot_params', '')
        if line not in ['', '0']:
            plotparams = filterParList(line)

        line = ini.string('plot_2D_param', '').strip()
        plot_2D_param = None
        if line and line != '0':
            plot_2D_param = line

        cust2DPlots = []
        if not plot_2D_param:
            # Use custom array of specific plots
            num_cust2D_plots = ini.int('plot_2D_num', 0)
            for i in range(1, num_cust2D_plots + 1):
                line = ini.string('plot' + str(i))
                pars = filterParList(line, 2)
                if pars is not None:
                    cust2DPlots.append(pars)
                else:
                    num_cust2D_plots -= 1

                
        triangle_params = []
        triangle_plot = ini.bool('triangle_plot', False)
        if triangle_plot:
            line = ini.string('triangle_params', '')
            triangle_params = filterParList(line)
            triangle_num = len(triangle_params)
            triangle_plot = triangle_num > 1

        num_3D_plots = ini.int('num_3D_plots', 0)
        plot_3D = []
        for ix in range(1, num_3D_plots + 1):
            line = ini.string('3D_plot' + str(ix))
            pars = filterParList(line, 3)
            if pars is not None:
                plot_3D.append(pars)
            else:
                num_3D_plots -= 1
            
      
        # Produce file of weight-1 samples if requested
        if (num_3D_plots and not make_single_samples or make_scatter_samples) and not no_plots:
            make_single_samples = True
            single_thin = max(1, int(round(mc.norm / mc.max_mult)) // mc.max_scatter_points)

        if plot_data_dir:
            if make_single_samples:
                filename = os.path.join(plot_data_dir, rootname.strip() + '_single.txt')
                mc.makeSingleSamples(filename, single_thin)

            # Write paramNames file
            mc.getParamNames().saveAsText(os.path.join(plot_data_dir, rootname + '.paramnames'))
            mc.getBounds().saveToFile(os.path.join(plot_data_dir, rootname + '.bounds'))

        make_plots = ini.bool('make_plots', False)

        done2D = {}

        filename = rootdirname + '.' + plot_ext
        mc.writeScriptPlots1D(filename, plotparams)
        if make_plots: runScript(filename)

        # Do 2D bins
        if plot_2D_param == 'corr':
            # In this case output the most correlated variable combinations
            print('...doing 2D plots for most correlated variables')
            cust2DPlots = mc.getCorrelatedVariable2DPlots()
            plot_2D_param = None
        elif plot_2D_param:
            mc.paramNames.parWithName(plot_2D_param, error=True)  # just check

        if cust2DPlots or plot_2D_param:
            print('...producing 2D plots')
            filename = rootdirname + '_2D.' + plot_ext
            done2D = mc.writeScriptPlots2D(filename, plot_2D_param, cust2DPlots,
                                           writeDataToFile=plot_data_dir, shade_meanlikes=shade_meanlikes)
            if make_plots: runScript(filename)

        if triangle_plot:
            # Add the off-diagonal 2D plots
            print('...producing triangle plot')
            filename = rootdirname + '_tri.' + plot_ext
            mc.writeScriptPlotsTri(filename, triangle_params)
            for i, p2 in enumerate(triangle_params):
                for p1 in triangle_params[i + 1:]:
                    if not done2D.get((p1, p2)) and plot_data_dir:
                        mc.get2DDensityGridData(p1, p2, writeDataToFile=True, meanlikes=shade_meanlikes)
            if make_plots: runScript(filename)

        # Do 3D plots (i.e. 2D scatter plots with coloured points)
        if num_3D_plots:
            print('...producing ', num_3D_plots, '2D colored scatter plots')
            filename = rootdirname + '_3D.' + plot_ext
            mc.writeScriptPlots3D(filename, plot_3D)
            if make_plots: runScript(filename)

    if not plots_only:
        # Write out stats marginalized
        mc.getMargeStats().saveAsText(rootdirname + '.margestats')

        # Limits from global likelihood
        if mc.loglikes is not None: mc.getLikeStats().saveAsText(rootdirname + '.likestats')


    if dumpNDbins:
        num_bins_ND = ini.int('num_bins_ND', 10)
        line = ini.string('ND_params','')
        
        if line not in ["",'0']:
            ND_params = filterParList(line)
            print(ND_params)

            ND_dim=len(ND_params)
            print(ND_dim)
           
            mc.getRawNDDensityGridData(ND_params, writeDataToFile=True,
                                       meanlikes=shade_meanlikes)
    



    # System command
    if finish_run_command:
        finish_run_command = finish_run_command.replace('%ROOTNAME%', rootname)
        finish_run_command = finish_run_command.replace('%PLOTDIR%', plot_data_dir)
        finish_run_command = finish_run_command.replace('%PLOTROOT%', os.path.join(plot_data_dir, rootname))
        os.system(finish_run_command)


if __name__ == '__main__':
    try:
        import argparse
    except ImportError:
        print('Make sure you are using python 2.7+')
        raise

    parser = argparse.ArgumentParser(description='GetDist sample analyser')
    parser.add_argument('ini_file', nargs='?',
                        help='.ini file with analysis settings (optional, if omitted uses defaults)')
    parser.add_argument('chain_root', nargs='?',
                        help='Root name of chain to analyse (e.g. chains/test), required unless file_root specified in ini_file')
    parser.add_argument('--ignore_rows',
                        help='set initial fraction of chains to cut as burn in (fraction of total rows, or >1 number of rows); overrides any value in ini_file if set')
    parser.add_argument('--make_param_file',
                        help='Produce a sample distparams.ini file that you can edit and use when running GetDist')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + getdist.__version__)
    args = parser.parse_args()
    if args.make_param_file:
        content = io.open(getdist.distparam_template).read()
        analysis = io.open(getdist.default_getdist_settings).read()
        content = content.replace('%%%ANALYSIS_DEFAULTS%%%', analysis)
        with io.open(args.make_param_file, 'w') as f:
            f.write(content)
        print('Template .ini file written to ' + args.make_param_file)
    else:
        main(args)
