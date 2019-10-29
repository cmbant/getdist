from __future__ import absolute_import
from __future__ import print_function
import fnmatch
import sys
import six

try:
    import argparse
except:
    print('use "module load" to load python 2.7+')
    sys.exit()
from paramgrid import batchjob


def argParser(desc=''):
    return argparse.ArgumentParser(description=desc)


class batchArgs(object):
    def __init__(self, desc='', importance=True, noBatchPath=False, notExist=False, notall=False, converge=False,
                 plots=False, batchPathOptional=False):
        self.parser = argparse.ArgumentParser(description=desc)
        if not noBatchPath:
            if batchPathOptional:
                self.parser.add_argument('batchPath', nargs='?', help='directory containing the grid')
            else:
                self.parser.add_argument('batchPath', help='directory containing the grid')
        if converge: self.parser.add_argument('--converge', type=float, default=0, help='minimum R-1 convergence')
        self.importanceParameter = importance
        self.notExist = notExist
        self.notall = notall
        self.doplots = plots

    def parseForBatch(self, vals=None):
        if self.importanceParameter:
            self.parser.add_argument('--noimportance', action='store_true',
                                     help='original chains only, no importance sampled')
            self.parser.add_argument('--importance', nargs='*', default=None,
                                     help='data names for importance sampling runs to include')
            self.parser.add_argument('--importancetag', nargs='*', default=None,
                                     help='importance tags for importance sampling runs to include')

        self.parser.add_argument('--name', default=None, nargs='+',
                                 help='specific chain full name only (base_paramx_data1_data2)')
        self.parser.add_argument('--param', default=None, nargs='+',
                                 help='runs including specific parameter only (paramx)')
        self.parser.add_argument('--paramtag', default=None, nargs='+',
                                 help='runs with specific parameter tag only (base_paramx)')
        self.parser.add_argument('--data', nargs='+', default=None, help='runs including specific data only (data1)')
        self.parser.add_argument('--datatag', nargs='+', default=None,
                                 help='runs with specific data tag only (data1_data2)')
        self.parser.add_argument('--musthave_data', nargs='+', default=None,
                                 help='include only runs that include specific data (data1)')
        self.parser.add_argument('--skip_data', nargs='+', default=None,
                                 help='skip runs containing specific data (data1)')
        self.parser.add_argument('--skip_param', nargs='+', default=None,
                                 help='skip runs containing specific parameter (paramx)')
        self.parser.add_argument('--group', default=None, nargs='+', help='include only runs with given group names')
        self.parser.add_argument('--skip_group', default=None, nargs='+', help='exclude runs with given group names')

        if self.notExist:
            self.parser.add_argument('--notexist', action='store_true',
                                     help='only include chains that don\'t already exist on disk')
        if self.notall:
            self.parser.add_argument('--notall', type=int, default=None,
                                     help='only include chains where all N chains don\'t already exist on disk')
        if self.doplots:
            self.parser.add_argument('--plot_data', nargs='*', default=None,
                                     help='directory/ies containing getdist output plot_data')
            self.parser.add_argument('--paramNameFile', default='clik_latex.paramnames',
                                     help=".paramnames file for custom labels for parameters")
            self.parser.add_argument('--param_list', default=None,
                                     help=".paramnames file listing specific parameters to include (only)")
            self.parser.add_argument('--size_inch', type=float, default=None, help='output subplot size in inches')
            self.parser.add_argument('--nx', default=None, help='number of plots per row')
            self.parser.add_argument('--outputs', nargs='+', default=['pdf'], help='output file type (default: pdf)')

        args = self.parser.parse_args(vals)
        self.args = args
        if args.batchPath:
            self.batch = batchjob.readobject(args.batchPath)
            if self.batch is None:
                raise Exception('batchPath %s does not exist or is not initialized with makeGrid.py' % args.batchPath)
            if self.doplots:
                import getdist.plots as plots
                from getdist import paramnames

                if args.paramList is not None: args.paramList = paramnames.ParamNames(args.paramList)
                g = plots.GetDistPlotter(chain_dir=self.batch.batchPath)
                if args.size_inch is not None: g.settings.setWithSubplotSize(args.size_inch)
                return self.batch, self.args, g
            else:
                return self.batch, self.args
        else:
            return None, self.args

    def wantImportance(self, jobItem):
        return (self.args.importancetag is None or len(self.args.importancetag) == 0 or
                jobItem.importanceTag in self.args.importancetag) and \
               (self.args.importance is None or len(self.args.importance) == 0 or
                len([True for x in self.args.importance if x in jobItem.data_set.importanceNames]))

    def jobItemWanted(self, jobItem):
        return not jobItem.isImportanceJob and (
                self.args.importance is None) or jobItem.isImportanceJob and self.wantImportance(jobItem)

    def nameMatches(self, jobItem):
        if self.args.name is None: return True
        for pat in self.args.name:
            if fnmatch.fnmatch(jobItem.name, pat): return True
        return False

    def groupMatches(self, jobItem):
        return (self.args.group is None or jobItem.group in self.args.group) and (
                self.args.skip_group is None or not jobItem.group in self.args.skip_group)

    def dataMatches(self, jobItem):
        if self.args.musthave_data is not None and not jobItem.data_set.hasAll(self.args.musthave_data): return False
        if self.args.datatag is None:
            if self.args.skip_data is not None and jobItem.data_set.hasName(self.args.skip_data): return False
            return self.args.data is None or jobItem.data_set.hasName(self.args.data)
        else:
            return jobItem.datatag in self.args.datatag

    def paramsMatch(self, jobItem):
        if self.args.paramtag is None:
            if self.args.param is None:
                return not self.args.skip_param or not jobItem.hasParam(self.args.skip_param)
            for pat in self.args.param:
                if pat in jobItem.param_set: return not self.args.skip_param or not jobItem.hasParam(
                    self.args.skip_param)
            return False
        else:
            return jobItem.paramtag in self.args.paramtag

    def filteredBatchItems(self, wantSubItems=True, chainExist=False):
        for jobItem in self.batch.items(wantImportance=not self.args.noimportance, wantSubItems=wantSubItems):
            if (not chainExist or jobItem.chainExists()) and (
                    self.jobItemWanted(jobItem) and self.nameMatches(jobItem) and self.paramsMatch(
                jobItem) and self.dataMatches(jobItem)
                    and self.groupMatches(jobItem)): yield (jobItem)

    def sortedParamtagDict(self, chainExist=True):
        items = dict()
        for jobItem in self.filteredBatchItems():
            if not chainExist or jobItem.chainExists():
                if not jobItem.paramtag in items: items[jobItem.paramtag] = []
                items[jobItem.paramtag].append(jobItem)
        return sorted(six.iteritems(items))

    def filterForDataCompare(self, batch, datatags, getDistExists=False):
        items = []
        for tag, data in zip([self.batch.normalizeDataTag(data) for data in datatags], datatags):
            items += [jobItem for jobItem in batch if (jobItem.datatag == data or jobItem.normed_data == tag)
                      and (not getDistExists or jobItem.getDistExists())]
        return items
