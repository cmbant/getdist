import os
import subprocess
import getdist
import sys
import logging
from getdist import MCSamples, chains, IniFile


def runScript(fname):
    subprocess.Popen(['python', fname])


# noinspection PyUnboundLocalVariable
def getdist_script(args, exit_on_error=True):
    def do_error(msg):
        if exit_on_error:
            print(msg)
            sys.exit()
        raise ValueError(msg)

    result = []

    def doprint(*s):
        result.append(" ".join([str(x) for x in s]))
        print(*s)

    no_plots = False
    chain_root = args.chain_root
    if args.ini_file is None and chain_root is None:
        do_error('Must give either a .ini file of parameters or a chain file root name. Run "getdist -h" for help.')
    if '.ini' not in args.ini_file and chain_root is None:
        # use default settings acting on chain_root, no plots
        chain_root = args.ini_file
        args.ini_file = getdist.default_getdist_settings
        no_plots = True
    if not os.path.isfile(args.ini_file):
        do_error('Parameter file does not exist: ' + args.ini_file)
    if chain_root and chain_root.endswith('.txt'):
        chain_root = chain_root[:-4]

    if chain_root is not None and ('*' in chain_root or '?' in chain_root):
        import glob
        import copy
        for ending in ['.paramnames', 'updated.yaml']:
            for f in glob.glob(chain_root + ending):
                fileargs = copy.copy(args)
                fileargs.chain_root = f.replace(ending, '')
                getdist_script(fileargs)
        return

    # Input parameters
    ini = IniFile(args.ini_file)

    for item in set(ini.params.keys()).intersection(
            {'make_single_samples', 'single_thin', 'dump_ND_bins', 'plot_meanlikes', 'shade_meanlikes',
             'plot_data_dir', 'force_twotail'}):
        if ini.string(item) not in [0, 'F']:
            logging.warning('%s is no longer supported by getdist, value ignored' % item)

    # File root
    if chain_root is not None:
        in_root = chain_root
    else:
        in_root = ini.params['file_root']
    if not in_root:
        do_error('Chain Root file name not given ')
    rootname = os.path.basename(in_root)

    if args.ignore_rows is not None:
        ignorerows = args.ignore_rows
    else:
        ignorerows = ini.float('ignore_rows', 0.0)

    samples_are_chains = ini.bool('samples_are_chains', True)

    paramnames = ini.string('parameter_names', '')

    # Create instance of MCSamples
    mc = MCSamples(in_root, ini=ini, files_are_chains=samples_are_chains, paramNamesFile=paramnames)

    if ini.bool('adjust_priors', False) or ini.bool('map_params', False):
        do_error('To adjust priors or define new parameters, use a separate python script; '
                 'see the python getdist docs for examples')

    plot_ext = ini.string('plot_ext', 'py')
    finish_run_command = ini.string('finish_run_command', '')

    no_plots = ini.bool('no_plots', no_plots)
    plots_only = ini.bool('plots_only', False)
    no_tests = plots_only or ini.bool('no_tests', False)

    thin_factor = ini.int('thin_factor', 0)
    thin_cool = ini.float('thin_cool', 1.0)

    cool = ini.float('cool', 1.0)

    chain_exclude = ini.int_list('exclude_chain')

    out_dir = ini.string('out_dir', './')
    if out_dir:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        doprint('producing files in directory ', out_dir)
    mc.out_dir = out_dir

    out_root = ini.string('out_root', '')
    if out_root:
        rootname = out_root
        doprint('producing files with with root ', out_root)
    mc.rootname = rootname

    rootdirname = os.path.join(out_dir, rootname)
    mc.rootdirname = rootdirname

    if 'do_minimal_1d_intervals' in ini.params:
        do_error('do_minimal_1d_intervals no longer used; set credible_interval_threshold instead')

    line = ini.string('PCA_params', '')
    if line.lower() == 'all':
        PCA_params = mc.paramNames.list()
    else:
        PCA_params = line.split()
    PCA_num = ini.int('PCA_num', len(PCA_params))
    if PCA_num != 0:
        if PCA_num < 2:
            do_error('Can only do PCA for 2 or more parameters')
        PCA_func = ini.string('PCA_func', '')
        # Characters representing functional mapping
        if PCA_func == '':
            PCA_func = ['N'] * PCA_num  # No mapping
        PCA_NormParam = ini.string('PCA_normparam', '') or None

    # ==============================================================================

    first_chain = ini.int('first_chain', 0)
    last_chain = ini.int('chain_num', -1)
    # -1 y keep reading until one not found

    # Chain files
    for separator in ['_', '.']:
        chain_files = chains.chainFiles(in_root, first_chain=first_chain, last_chain=last_chain,
                                        chain_exclude=chain_exclude, separator=separator)
        if chain_files:
            break

    mc.loadChains(in_root, chain_files)

    mc.removeBurnFraction(ignorerows)
    if chains.print_load_details:
        if ignorerows:
            doprint('Removed %s as burn in' % ignorerows)
        else:
            doprint('Removed no burn in')

    mc.deleteFixedParams()
    mc.makeSingle()

    def filterParList(namestring, num=None):
        if not namestring.strip():
            _pars = mc.paramNames.list()
        else:
            _pars = []
            for name in namestring.split():
                if '?' in name or '*' in name:
                    _pars += mc.paramNames.getMatches(name, strings=True)
                elif mc.paramNames.parWithName(name):
                    _pars.append(name)
        if num is not None and len(_pars) != num:
            doprint('%iD plot has missing parameter or wrong number of parameters: %s' % (num, _pars))
            _pars = None
        return _pars

    if cool != 1:
        doprint('Cooling chains by ', cool)
        mc.cool(cool)

    mc.updateBaseStatistics()

    if not no_tests:
        mc.getConvergeTests(mc.converge_test_limit, writeDataToFile=True, feedback=True)

    mc.writeCovMatrix()
    mc.writeCorrelationMatrix()

    # Output thinned data if requested
    # Must do this with unsorted output
    if thin_factor > 1:
        thin_ix = mc.thin_indices(thin_factor)
        filename = rootdirname + '_thin.txt'
        mc.writeThinData(filename, thin_ix, thin_cool)

    doprint(mc.getNumSampleSummaryText().strip())
    if mc.likeStats:
        doprint(mc.likeStats.likeSummary().strip())

    if PCA_num > 0 and not plots_only:
        mc.PCA(PCA_params, PCA_func, PCA_NormParam, writeDataToFile=True)

    # Do 1D bins
    mc._setDensitiesandMarge1D()

    if not no_plots:
        # Output files for 1D plots

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

        make_plots = ini.bool('make_plots', False) or args.make_plots

        filename = rootdirname + '.' + plot_ext
        mc._writeScriptPlots1D(filename, plotparams)
        if make_plots:
            runScript(filename)

        # Do 2D bins
        if plot_2D_param == 'corr':
            # In this case output the most correlated variable combinations
            doprint('...doing 2D plots for most correlated variables')
            cust2DPlots = mc.getCorrelatedVariable2DPlots()
            plot_2D_param = None
        elif plot_2D_param:
            mc.paramNames.parWithName(plot_2D_param, error=True)  # just check

        if cust2DPlots or plot_2D_param:
            doprint('...producing 2D plots')
            filename = rootdirname + '_2D.' + plot_ext
            mc._writeScriptPlots2D(filename, plot_2D_param, cust2DPlots)
            if make_plots:
                runScript(filename)

        if triangle_plot:
            # Add the off-diagonal 2D plots
            doprint('...producing triangle plot')
            filename = rootdirname + '_tri.' + plot_ext
            mc._writeScriptPlotsTri(filename, triangle_params)
            if make_plots:
                runScript(filename)

        # Do 3D plots (i.e. 2D scatter plots with coloured points)
        if num_3D_plots:
            doprint('...producing ', num_3D_plots, '2D colored scatter plots')
            filename = rootdirname + '_3D.' + plot_ext
            mc._writeScriptPlots3D(filename, plot_3D)
            if make_plots:
                runScript(filename)

    if not plots_only:
        # Write out stats marginalized
        mc.getMargeStats().saveAsText(rootdirname + '.margestats')

        # Limits from global likelihood
        if mc.loglikes is not None:
            mc.getLikeStats().saveAsText(rootdirname + '.likestats')

    # System command
    if finish_run_command:
        finish_run_command = finish_run_command.replace('%ROOTNAME%', rootname)
        os.system(finish_run_command)

    return "\n".join(result)


def make_param_file(file_name, feedback=True):
    with open(getdist.distparam_template, encoding="utf-8-sig") as f:
        content = f.read()
    with open(getdist.default_getdist_settings, encoding="utf-8-sig") as f:
        analysis = f.read()
    content = content.replace('%%%ANALYSIS_DEFAULTS%%%', analysis)
    with open(file_name, 'w', encoding="utf-8") as f:
        f.write(content)
    if feedback:
        print('Template .ini file written to ' + file_name)


def getdist_command(args=None):
    import argparse
    import getdist

    parser = argparse.ArgumentParser(description='GetDist sample analyser')
    parser.add_argument('ini_file', nargs='?',
                        help='.ini file with analysis settings (optional, if omitted uses defaults)')
    parser.add_argument('chain_root', nargs='?',
                        help='Root name of chain to analyse (e.g. chains/test), required unless file'
                             '_root specified in ini_file')
    parser.add_argument('--ignore_rows', type=float,
                        help='set initial fraction of chains to cut as burn in (fraction of total rows'
                             ', or >1 number of rows); overrides any value in ini_file if set')
    parser.add_argument('--make_param_file',
                        help='Produce a sample distparams.ini file that you can edit and use when running GetDist')
    parser.add_argument('--make_plots', action='store_true', help='Make PDFs from any requested plot script files')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + getdist.__version__)
    args = parser.parse_args(args)
    if args.make_param_file:
        make_param_file(args.make_param_file)
    else:
        return getdist_script(args)


def getdist_gui():
    from getdist.gui.mainwindow import run_gui

    if sys.platform == "darwin":
        # On Mac need to run .app with plist to get menu name right (and avoid menu bugs)
        import subprocess
        import os

        path = os.path.join(os.path.dirname(getdist.gui.__file__), 'GetDist GUI.app')
        if os.path.exists(path):
            if subprocess.call(["/usr/bin/open", "-a", path], env=os.environ):
                print("Error running 'GetDist GUI.app'. This may be a Catalina issue, any ideas?\n"
                      "Attempting to run script directly, using non-unified menus.")
                run_gui()
        else:
            print('GetDist GUI.app not found; not running getdist-gui, getdist package not installed '
                  'or no valid PySide2 found when setup was run. Running script...')
            run_gui()
    else:
        run_gui()
