from __future__ import absolute_import
from __future__ import print_function
import os
import copy
import sys
import six
from getdist import IniFile
from paramgrid import batchjob, batchjob_args

# defaults here are for Planck. Unltimately should be refactored out of general code but harmless.

default_params = dict()
default_params['mnu'] = '0.02 0 5 0.1 0.03'
default_params['omegak'] = '-0.0008 -0.3 0.3 0.001 0.001'  # starting exactly on flat seems to confuse minimizer
default_params['w'] = '-0.995 -3 -0.3 0.02 0.02'
default_params['nnu'] = '3.046 0.05 10 0.05 0.05'
default_params['nrun'] = '0 -1 1 0.005 0.001'
default_params['nrunrun'] = '0 -1 1 0.005 0.001'
default_params['r'] = '0 0 3 0.03 0.03'
default_params['Alens'] = '1 0 10 0.05 0.05'
default_params['yhe'] = '0.245 0.1 0.5 0.006 0.006'
default_params['alpha1'] = '0 -1 1 0.0003 0.0003'
default_params['deltazrei'] = '0.5 0.1 3 0.3 0.3'
default_params['wa'] = '0 -2 2 0.3 0.3'
default_params['meffsterile'] = '0.1 0 3 0.1 0.03'
default_params['Aphiphi'] = '1 0 10 0.02 0.02'
default_params['Alensf'] = '1 0 10 0.03 0.03'
default_params['nt'] = '0 -3 3 0.2 0.02'

default_param_extra_opts = {
    'mnu': {'num_massive_neutrinos': 3, 'neutrino_hierarchy': 'degenerate'},
    'meffsterile': {'param[mnu]': '0.06', 'param[nnu]': '3.1 3.046 10 0.05 0.05', 'num_massive_neutrinos': 1,
                    'accuracy_level': 1.2},
    'yhe': {'bbn_consistency': False},
    'r': {'compute_tensors': True},
    'nt': {'inflation_consistency': False, 'lmax_tensor': 1000}
}


def getArgs(vals=None):
    parser = batchjob_args.argParser('Initialize grid using settings file')
    parser.add_argument('batchPath',
                        help='root directory containing/to contain the grid (e.g. ./PLA where directories base, base_xx etc are under ./PLA)')
    parser.add_argument('settingName', nargs='?',
                        help='python setting file (without .py) for making or updating a grid, usually found as python/settingName.py')
    parser.add_argument('--readOnly', action='store_true', help='option to configure an already-run existing grid')
    return parser.parse_args(vals)


def setMinimize(jobItem, ini):
    ini.params['action'] = 2
    ini.params['lmin_store_all_cmb'] = 2500
    if 'omegak' in jobItem.param_set: ini.params['accuracy_level'] = 1.2
    if 'meffsterile' in jobItem.param_set: ini.params['sterile_mphys_max'] = 10000


def updateIniParams(ini, params, path):
    for iniitem in params:
        if isinstance(iniitem, dict):
            ini.params.update(iniitem)
        elif isinstance(iniitem, six.string_types):
            ini.defaults.append(path + iniitem)
        elif isinstance(iniitem, (list, tuple)):
            updateIniParams(ini, iniitem, path)
        else:
            raise Exception('Unknown item in setting .ini/param list')


def pathIsGrid(batchPath):
    return os.path.exists(batchjob.grid_cache_file(batchPath)) or os.path.exists(
        os.path.join(batchPath, 'config', 'config.ini'))


def makeGrid(batchPath, settingName=None, settings=None, readOnly=False, interactive=False):
    batchPath = os.path.abspath(batchPath) + os.sep

    # 0: chains, 1: importance sampling, 2: best-fit, 3: best-fit and Hessian
    cosmomcAction = 0

    if not settings:
        if not settingName:
            if not pathIsGrid(batchPath):
                raise Exception('Need to give name of setting file if batchPath/config does not exist')
            readOnly = True
            sys.path.insert(0, batchPath + 'config')
            sys.modules['batchJob'] = batchjob  # old name
            settings = __import__(IniFile(batchPath + 'config/config.ini').params['setting_file'].replace('.py', ''))
        else:
            settings = __import__(settingName, fromlist=['dummy'])

    batch = batchjob.batchJob(batchPath, settings.ini_dir)

    if hasattr(settings, 'skip'): batch.skip = settings.skip
    batch.makeItems(settings, messages=not readOnly)
    if readOnly:
        for jobItem in [b for b in batch.jobItems]:
            if not jobItem.chainExists():
                batch.jobItems.remove(jobItem)
        batch.save()
        print('OK, configured grid with %u existing chains' % (len(batch.jobItems)))
        return batch
    else:
        batch.makeDirectories(settings.__file__)
        batch.save()

    # priors and widths for parameters which are varied
    start_at_bestfit = getattr(settings, 'start_at_bestfit', False)
    params = getattr(settings, 'params', default_params)
    param_extra = getattr(settings, 'param_extra_opts', default_param_extra_opts)

    for jobItem in batch.items(wantSubItems=False):

        jobItem.makeChainPath()
        ini = IniFile()

        job_param_extra = getattr(jobItem, 'param_extra_opts', None)
        for param in jobItem.param_set:
            ini.params['param[' + param + ']'] = params[param]
            if param_extra is not None and param in param_extra:
                ini.params.update(param_extra[param])
            if job_param_extra is not None and param in job_param_extra:
                ini.params.update(job_param_extra[param])

        if hasattr(settings, 'extra_opts'):
            ini.params.update(settings.extra_opts)
        if hasattr(jobItem, 'extra_opts'):
            ini.params.update(jobItem.extra_opts)

        ini.params['file_root'] = jobItem.chainRoot

        cov_dir_name = getattr(settings, 'cov_dir', 'planck_covmats')
        covdir = os.path.join(batch.basePath, cov_dir_name)
        covmat = os.path.join(covdir, jobItem.name + '.covmat')
        if not os.path.exists(covmat):
            covNameMappings = getattr(settings, 'covNameMappings', None)
            mapped_name_norm = jobItem.makeNormedName(covNameMappings)[0]
            covmat_normed = os.path.join(covdir, mapped_name_norm + '.covmat')
            covmat = covmat_normed
            if not os.path.exists(covmat) and covNameMappings is not None:
                covmat = os.path.join(covdir, "_".join(
                    [covNameMappings.get(x, x) for x in jobItem.name.split('_')]) + '.covmat')
            if not os.path.exists(covmat) and hasattr(jobItem.data_set, 'covmat'):
                covmat = batch.basePath + jobItem.data_set.covmat
                if not os.path.exists(covmat):
                    covmat = os.path.join(covdir, jobItem.data_set.covmat)
            if not os.path.exists(covmat) and hasattr(settings, 'covmat'):
                covmat = batch.basePath + settings.covmat
                if not os.path.exists(covmat):
                    covmat = os.path.join(covdir, settings.covmat)
        else:
            covNameMappings = None
        if os.path.exists(covmat):
            ini.params['propose_matrix'] = covmat
            if getattr(settings, 'newCovmats', True): ini.params['MPI_Max_R_ProposeUpdate'] = 20
        else:
            hasCov = False
            ini.params['MPI_Max_R_ProposeUpdate'] = 20
            covmat_try = []
            if 'covRenamer' in dir(settings):
                covmat_try += settings.covRenamer(jobItem.name)
                covmat_try += settings.covRenamer(mapped_name_norm)
            if hasattr(settings, 'covrenames'):
                for aname in [jobItem.name, mapped_name_norm]:
                    covmat_try += [aname.replace(old, new, 1) for old, new in settings.covrenames if old in aname]
                    for new1, old1 in settings.covrenames:
                        if old1 in aname:
                            name = aname.replace(old1, new1, 1)
                            covmat_try += [name.replace(old, new, 1) for old, new in settings.covrenames if old in name]
            if 'covWithoutNameOrder' in dir(settings):
                if covNameMappings:
                    removes = copy.deepcopy(covNameMappings)
                else:
                    removes = dict()
                for name in settings.covWithoutNameOrder:
                    if name in jobItem.data_set.names:
                        removes[name] = ''
                        covmat_try += [jobItem.makeNormedName({name: ''})[0]]
                        covmat_try += [jobItem.makeNormedName(removes)[0]]
            covdir2 = os.path.join(batch.basePath, getattr(settings, 'cov_dir_fallback', cov_dir_name))
            for name in covmat_try:
                covmat = os.path.join(batch.basePath, covdir2, name + '.covmat')
                if os.path.exists(covmat):
                    ini.params['propose_matrix'] = covmat
                    print('covmat ' + jobItem.name + ' -> ' + name)
                    hasCov = True
                    break
            if not hasCov: print('WARNING: no matching specific covmat for ' + jobItem.name)

        ini.params['start_at_bestfit'] = start_at_bestfit
        updateIniParams(ini, jobItem.data_set.params, batch.commonPath)
        for deffile in settings.defaults:
            ini.defaults.append(batch.commonPath + deffile)
        if hasattr(settings, 'override_defaults'):
            ini.defaults = [batch.commonPath + deffile for deffile in settings.override_defaults] + ini.defaults

        ini.params['action'] = cosmomcAction
        ini.saveFile(jobItem.iniFile())
        if not start_at_bestfit:
            setMinimize(jobItem, ini)
            variant = '_minimize'
            ini.saveFile(jobItem.iniFile(variant))

            # add ini files for importance sampling runs
        for imp in jobItem.importanceJobs():
            if getattr(imp, 'importanceFilter', None): continue
            if batch.hasName(imp.name.replace('_post', '')):
                raise Exception('importance sampling something you already have?')
            for minimize in (False, True):
                if minimize and not getattr(imp, 'want_minimize', True): continue
                ini = IniFile()
                updateIniParams(ini, imp.importanceSettings, batch.commonPath)
                if cosmomcAction == 0 and not minimize:
                    for deffile in settings.importanceDefaults:
                        ini.defaults.append(batch.commonPath + deffile)
                    ini.params['redo_outroot'] = imp.chainRoot
                    ini.params['action'] = 1
                else:
                    ini.params['file_root'] = imp.chainRoot
                if minimize:
                    setMinimize(jobItem, ini)
                    variant = '_minimize'
                else:
                    variant = ''
                ini.defaults.append(jobItem.iniFile())
                ini.saveFile(imp.iniFile(variant))
                if cosmomcAction != 0: break

    if not interactive: return batch
    print('Done... to run do: python python/runbatch.py ' + batchPath)
    if not start_at_bestfit:
        print('....... for best fits: python python/runbatch.py ' + batchPath + ' --minimize')
    print('')
    print('for importance sampled: python python/runbatch.py ' + batchPath + ' --importance')
    print('for best-fit for importance sampled: python python/runbatch.py ' + batchPath + ' --importance_minimize')
