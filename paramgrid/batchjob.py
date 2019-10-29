from __future__ import absolute_import
from __future__ import print_function
import os
import shutil
import pickle
import copy
import sys
import time
import six
import getdist
from getdist import types, IniFile
from getdist.mcsamples import loadMCSamples


def grid_cache_file(directory):
    directory = os.path.abspath(directory)
    if getdist.cache_dir:
        import hashlib
        return os.path.join(getdist.cache_dir, '_batch_'
                            + hashlib.md5(directory.encode('utf-8')).hexdigest()[:10]) + '.pyobj'
    return os.path.join(directory, 'batch.pyobj')


def resetGrid(directory):
    fname = grid_cache_file(directory)
    if os.path.exists(fname):
        os.remove(fname)


def readobject(directory=None):
    # load this here to prevent circular
    from paramgrid import gridconfig

    if directory is None:
        directory = sys.argv[1]
    fname = grid_cache_file(directory)
    if not os.path.exists(fname):
        if gridconfig.pathIsGrid(directory):
            return gridconfig.makeGrid(directory, readOnly=True, interactive=False)
        return None
    try:
        config_dir = os.path.abspath(directory) + os.sep + 'config'
        if os.path.exists(config_dir):
            # set path in case using functions defined and hence imported from in settings file
            sys.path.insert(0, config_dir)
        with open(fname, 'rb') as inp:
            grid = pickle.load(inp)
        if not os.path.exists(grid.basePath):
            raise FileNotFoundError('Directory not found %s' % grid.basePath)
        return grid
    except Exception as e:
        print('Error loading cached batch object: ', e)
        resetGrid(directory)
        if gridconfig.pathIsGrid(directory):
            return gridconfig.makeGrid(directory, readOnly=True, interactive=False)
        raise


def saveobject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def makePath(s):
    if not os.path.exists(s): os.makedirs(s)


def nonEmptyFile(fname):
    return os.path.exists(fname) and os.path.getsize(fname) > 0


def getCodeRootPath():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')) + os.sep


class propertiesItem(object):
    def propertiesIni(self):
        if os.path.exists(self.propertiesIniFile()):
            return IniFile(self.propertiesIniFile())
        else:
            ini = IniFile()
            ini.original_filename = self.propertiesIniFile()
            return ini


class dataSet(object):
    def __init__(self, names, params=None, covmat=None, dist_settings=None):
        if not dist_settings:
            dist_settings = {}
        if isinstance(names, six.string_types): names = [names]
        if params is None:
            params = [(name + '.ini') for name in names]
        else:
            params = self.standardizeParams(params)
        if covmat is not None: self.covmat = covmat
        self.names = names
        self.params = params  # can be an array of items, either ini file name or dictionaries of parameters
        self.tag = "_".join(self.names)
        self.dist_settings = dist_settings

    def add(self, name, params=None, overrideExisting=True, dist_settings=None):
        if params is None: params = [name]
        params = self.standardizeParams(params)
        if dist_settings: self.dist_settings.update(dist_settings)
        if overrideExisting:
            self.params = params + self.params  # can be an array of items, either ini file name or dictionaries of parameters
        else:
            self.params += params
        if name is not None:
            self.names += [name]
            self.tag = "_".join(self.names)
        return self

    def addEnd(self, name, params, dist_settings=None):
        if not dist_settings:
            dist_settings = {}
        return self.add(name, params, overrideExisting=False, dist_settings=dist_settings)

    def copy(self):
        return copy.deepcopy(self)

    def extendForImportance(self, names, params):
        data = copy.deepcopy(self)
        if not '_post_' in data.tag:
            data.tag += '_post_' + "_".join(names)
        else:
            data.tag += '_' + "_".join(names)
        data.importanceNames = names
        data.importanceParams = data.standardizeParams(params)
        data.names += data.importanceNames
        data.params += data.importanceParams
        return data

    def standardizeParams(self, params):
        if isinstance(params, dict) or isinstance(params, six.string_types): params = [params]
        for i in range(len(params)):
            if isinstance(params[i], six.string_types) and not '.ini' in params[i]: params[i] += '.ini'
        return params

    def hasName(self, name):
        if isinstance(name, six.string_types):
            return name in self.names
        else:
            return any([True for i in name if i in self.names])

    def hasAll(self, name):
        if isinstance(name, six.string_types):
            return name in self.names
        else:
            return all([(i in self.names) for i in name])

    def tagReplacing(self, x, y):
        items = []
        for name in self.names:
            if name == x:
                if y != '': items.append(y)
            else:
                items.append(name)
        return "_".join(items)

    def namesReplacing(self, dic):
        if dic is None: return self.names
        items = []
        for name in self.names:
            if name in dic:
                val = dic[name]
                if val: items.append(val)
            else:
                items.append(name)
        return items

    def makeNormedDatatag(self, dic):
        return "_".join(sorted(self.namesReplacing(dic)))


class jobGroup(object):
    def __init__(self, name, params=None, importanceRuns=None, datasets=None):
        if importanceRuns is None:
            importanceRuns = []
        if params is None:
            params = [[]]
        if datasets is None:
            datasets = []
            self.params = params
            self.groupName = name
            self.importanceRuns = importanceRuns
            self.datasets = datasets
        self.extra_options = {}


class importanceSetting(object):
    def __init__(self, names, inis=None, dist_settings=None, minimize=True):
        if not inis:
            inis = []
        self.names = names
        self.inis = inis
        self.dist_settings = dist_settings or {}
        self.want_minimize = minimize

    def wantImportance(self, jobItem):
        return True


class importanceFilter(importanceSetting):
    # class for trivial importance sampling filters that can be done in python,
    # e.g. restricting a parameter to a new range

    def __init__(self, names, dist_settings=None, minimize=False):
        self.names = names
        self.inis = [self]
        self.dist_settings = dist_settings or {}
        self.want_minimize = minimize


class jobItem(propertiesItem):
    def __init__(self, path, param_set, data_set, base='base', minimize=True):
        self.param_set = param_set
        if not isinstance(data_set, dataSet): data_set = dataSet(data_set[0], data_set[1])
        self.data_set = data_set
        self.base = base
        self.paramtag = "_".join([base] + param_set)
        self.datatag = data_set.tag
        self.name = self.paramtag + '_' + self.datatag
        self.batchPath = path
        self.relativePath = self.paramtag + os.sep + self.datatag + os.sep
        self.chainPath = path + self.relativePath
        self.chainRoot = self.chainPath + self.name
        self.distPath = self.chainPath + 'dist' + os.sep
        self.distRoot = self.distPath + self.name
        self.isImportanceJob = False
        self.importanceItems = []
        self.want_minimize = minimize
        self.result_converge = None
        self.group = None
        self.dist_settings = copy.copy(data_set.dist_settings)
        self.makeIDs()

    def iniFile(self, variant=''):
        if not self.isImportanceJob:
            return self.batchPath + 'iniFiles' + os.sep + self.name + variant + '.ini'
        else:
            return self.batchPath + 'postIniFiles' + os.sep + self.name + variant + '.ini'

    def propertiesIniFile(self):
        return self.chainRoot + '.properties.ini'

    def isBurnRemoved(self):
        return self.propertiesIni().bool('burn_removed')

    def makeImportance(self, importanceRuns):
        for impRun in importanceRuns:
            if isinstance(impRun, importanceSetting):
                if not impRun.wantImportance(self): continue
            else:
                if len(impRun) > 2 and not impRun[2].wantImportance(self): continue
                impRun = importanceSetting(impRun[0], impRun[1])
            if len(set(impRun.names).intersection(self.data_set.names)) > 0:
                print('importance job duplicating parent data set: %s with %s' % (self.name, impRun.names))
                continue
            data = self.data_set.extendForImportance(impRun.names, impRun.inis)
            job = jobItem(self.batchPath, self.param_set, data, minimize=impRun.want_minimize)
            job.importanceTag = "_".join(impRun.names)
            job.importanceSettings = impRun.inis
            if not '_post_' in self.name:
                tag = '_post_' + job.importanceTag
            else:
                tag = '_' + job.importanceTag
            job.name = self.name + tag
            job.chainRoot = self.chainRoot + tag
            job.distPath = self.distPath
            job.chainPath = self.chainPath
            job.relativePath = self.relativePath
            job.distRoot = self.distRoot + tag
            job.datatag = self.datatag + tag
            job.isImportanceJob = True
            job.parent = self
            job.group = self.group
            job.dist_settings.update(impRun.dist_settings)
            if isinstance(impRun, importanceFilter):
                job.importanceFilter = impRun
            job.makeIDs()
            self.importanceItems.append(job)

    def makeNormedName(self, dataSubs=None):
        normed_params = "_".join(sorted(self.param_set))
        normed_data = self.data_set.makeNormedDatatag(dataSubs)
        normed_name = self.base
        if len(normed_params) > 0: normed_name += '_' + normed_params
        normed_name += '_' + normed_data
        return normed_name, normed_params, normed_data

    def makeIDs(self):
        self.normed_name, self.normed_params, self.normed_data = self.makeNormedName()

    def matchesDatatag(self, tagList):
        if self.datatag in tagList or self.normed_data in tagList: return True
        return self.datatag.replace('_post', '') in [tag.replace('_post', '') for tag in tagList]

    def hasParam(self, name):
        if isinstance(name, six.string_types):
            return name in self.param_set
        else:
            return any([True for i in name if i in self.param_set])

    def importanceJobs(self):
        return self.importanceItems

    def importanceJobsRecursive(self):
        res = copy.copy(self.importanceItems)
        for r in self.importanceItems:
            res += r.importanceJobsRecursive()
        return res

    def removeImportance(self, job):
        if job in self.importanceItems:
            self.importanceItems.remove(job)
        else:
            for j in self.importanceItems:
                j.removeImportance(job)

    def makeChainPath(self):
        makePath(self.chainPath)
        return self.chainPath

    def writeIniLines(self, f):
        outfile = open(self.iniFile(), 'w')
        outfile.write("\n".join(f))
        outfile.close()

    def chainName(self, chain=1):
        return self.chainRoot + '_' + str(chain) + '.txt'

    def chainExists(self, chain=1):
        fname = self.chainName(chain)
        return nonEmptyFile(fname)

    def chainNames(self, num_chains=None):
        if num_chains:
            return [self.chainName(i) for i in range(num_chains)]
        else:
            i = 1
            chains = []
            while self.chainExists(i):
                chains.append(self.chainName(i))
                i += 1
            return chains

    def allChainExists(self, num_chains):
        return all([self.chainExists(i + 1) for i in range(num_chains)])

    def chainFileDate(self, chain=1):
        return os.path.getmtime(self.chainName(chain))

    def chainsDodgy(self, interval=600):
        dates = []
        i = 1
        while os.path.exists(self.chainName(i)):
            dates.append(os.path.getmtime(self.chainName(i)))
            i += 1
        return os.path.exists(self.chainName(i + 1)) or max(dates) - min(dates) > interval

    def notRunning(self):
        if not self.chainExists(): return False  # might be in queue
        lastWrite = self.chainFileDate()
        return lastWrite < time.time() - 5 * 60

    def chainMinimumExists(self):
        fname = self.chainRoot + '.minimum'
        return nonEmptyFile(fname)

    def chainBestfit(self, paramNameFile=None):
        bf_file = self.chainRoot + '.minimum'
        if nonEmptyFile(bf_file):
            return types.BestFit(bf_file, paramNameFile)
        return None

    def chainMinimumConverged(self):
        bf = self.chainBestfit()
        if bf is None: return False
        return bf.logLike < 1e29

    def convergeStat(self):
        fname = self.chainRoot + '.converge_stat'
        if not nonEmptyFile(fname): return None, None
        textFileHandle = open(fname)
        textFileLines = textFileHandle.readlines()
        textFileHandle.close()
        return float(textFileLines[0].strip()), len(textFileLines) > 1 and textFileLines[1].strip() == 'Done'

    def chainFinished(self):
        if self.isImportanceJob:
            done = self.parent.convergeStat()[1]
            if done is None or self.parentChanged() or not self.notRunning(): return False
        else:
            done = self.convergeStat()[1]
        if done is None: return False
        return done

    def wantCheckpointContinue(self, minR=0):
        R, done = self.convergeStat()
        if R is None: return False
        if not os.path.exists(self.chainRoot + '_1.chk'): return False
        return not done and R > minR

    def getDistExists(self):
        return os.path.exists(self.distRoot + '.margestats')

    def getDistNeedsUpdate(self):
        return self.chainExists() and (
                not self.getDistExists() or self.chainFileDate() > os.path.getmtime(self.distRoot + '.margestats'))

    def parentChanged(self):
        return not self.chainExists() or self.chainFileDate() < self.parent.chainFileDate()

    def R(self):
        if self.result_converge is None:
            fname = self.distRoot + '.converge'
            if not nonEmptyFile(fname): return None
            self.result_converge = types.ConvergeStats(fname)
        return float(self.result_converge.worstR())

    def hasConvergeBetterThan(self, R, returnNotExist=False):
        try:
            chainR = self.R()
            if chainR is None: return returnNotExist
            return chainR <= R
        except:
            print('WARNING: Bad .converge for ' + self.name)
            return returnNotExist

    def loadJobItemResults(self, paramNameFile=None, bestfit=True, bestfitonly=False, noconverge=False, silent=False):
        self.result_converge = None
        self.result_marge = None
        self.result_likemarge = None
        self.result_bestfit = self.chainBestfit(paramNameFile)
        if not bestfitonly:
            marge_root = self.distRoot
            if self.getDistExists():
                if not noconverge: self.result_converge = types.ConvergeStats(marge_root + '.converge')
                self.result_marge = types.MargeStats(marge_root + '.margestats', paramNameFile)
                self.result_likemarge = types.LikeStats(marge_root + '.likestats')
                if self.result_bestfit is not None and bestfit: self.result_marge.addBestFit(self.result_bestfit)
            elif not silent:
                print('missing: ' + marge_root)

    def getMCSamples(self, ini=None, settings={}):
        return loadMCSamples(self.chainRoot, jobItem=self, ini=ini, settings=settings)


class batchJob(propertiesItem):
    def __init__(self, path, iniDir, cosmomcPath=None):
        self.batchPath = path
        self.skip = []
        self.basePath = cosmomcPath or getCodeRootPath()
        self.commonPath = self.basePath + iniDir
        self.subBatches = []
        self.jobItems = None
        self.getdist_options = {}

    def propertiesIniFile(self):
        return os.path.join(self.batchPath, 'config', 'config.ini')

    def makeItems(self, settings, messages=True):
        self.jobItems = []
        self.getdist_options = getattr(settings, 'getdist_options', self.getdist_options)
        allImportance = getattr(settings, 'importanceRuns', [])
        for group in settings.groups:
            for data_set in group.datasets:
                for param_set in group.params:
                    item = jobItem(self.batchPath, param_set, data_set)
                    if hasattr(group, 'groupName'): item.group = group.groupName
                    if hasattr(group, 'extra_opts'): item.extra_opts = group.extra_opts
                    if hasattr(group, 'param_extra_opts'): item.param_extra_opts = group.param_extra_opts
                    if not item.name in self.skip:
                        item.makeImportance(group.importanceRuns)
                        item.makeImportance(allImportance)
                        self.jobItems.append(item)
        for item in getattr(settings, 'jobItems', []):
            self.jobItems.append(item)
            item.makeImportance(allImportance)

        if hasattr(settings, 'importance_filters'):
            for job in self.jobItems:
                for item in job.importanceJobs():
                    item.makeImportance(settings.importance_filters)
                job.makeImportance(settings.importance_filters)

        for item in list(self.items()):
            for x in [imp for imp in item.importanceJobsRecursive()]:
                if self.has_normed_name(x.normed_name):
                    if messages: print('replacing importance sampling run with full run: ' + x.name)
                    item.removeImportance(x)
        for item in list(self.items()):
            for x in [imp for imp in item.importanceJobsRecursive()]:
                if self.has_normed_name(x.normed_name, wantImportance=True, exclude=x):
                    if messages: print('removing duplicate importance sampling run: ' + x.name)
                    item.removeImportance(x)

    def items(self, wantSubItems=True, wantImportance=False):
        for item in self.jobItems:
            yield (item)
            if wantImportance:
                for imp in item.importanceJobsRecursive():
                    if not imp.name in self.skip: yield (imp)

        if wantSubItems:
            for subBatch in self.subBatches:
                for item in subBatch.items(wantSubItems, wantImportance): yield (item)

    def hasName(self, name, wantSubItems=True):
        for jobItem in self.items(wantSubItems):
            if jobItem.name == name: return True
        return False

    def has_normed_name(self, name, wantSubItems=True, wantImportance=False, exclude=None):
        return self.normed_name_item(name, wantSubItems, wantImportance, exclude) is not None

    def normed_name_item(self, name, wantSubItems=True, wantImportance=False, exclude=None):
        for jobItem in self.items(wantSubItems, wantImportance):
            if jobItem.normed_name == name and not jobItem is exclude: return jobItem
        return None

    def normalizeDataTag(self, tag):
        return "_".join(sorted(tag.replace('_post', '').split('_')))

    def resolveName(self, paramtag, datatag, wantSubItems=True, wantImportance=True, raiseError=True, base='base',
                    returnJobItem=False):
        if paramtag:
            if isinstance(paramtag, six.string_types): paramtag = paramtag.split('_')
            paramtags = [base] + sorted(paramtag)
        else:
            paramtag = [base]
            paramtags = [base]
        name = "_".join(paramtags) + '_' + self.normalizeDataTag(datatag)
        jobItem = self.normed_name_item(name, wantSubItems, wantImportance)
        if jobItem is not None: return (jobItem.name, jobItem)[returnJobItem]
        if raiseError:
            raise Exception('No match for paramtag, datatag... ' + "_".join(paramtag) + ', ' + datatag)
        else:
            return None

    def resolveRoot(self, root):
        for jobItem in self.items(True, True):
            if jobItem.name == root: return jobItem
        return self.normed_name_item(root, True, True)

    def save(self, filename=''):
        saveobject(self, (grid_cache_file(self.batchPath), filename)[filename != ''])

    def makeDirectories(self, setting_file=None):
        makePath(self.batchPath)
        if setting_file:
            makePath(self.batchPath + 'config')
            setting_file = setting_file.replace('.pyc', '.py')
            shutil.copy(setting_file, self.batchPath + 'config')
            props = self.propertiesIni()
            props.params['setting_file'] = os.path.split(setting_file)[-1]
            props.saveFile()
        makePath(self.batchPath + 'iniFiles')
        makePath(self.batchPath + 'postIniFiles')
