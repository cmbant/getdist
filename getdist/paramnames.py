import os
import fnmatch
from itertools import chain


def makeList(roots):
    """
    Checks if the given parameter is a list.
    If not, Creates a list with the parameter as an item in it.

    :param roots: The parameter to check
    :return: A list containing the parameter.
    """
    if isinstance(roots, (list, tuple)):
        return roots
    else:
        return [roots]


def escapeLatex(text):
    if text:
        import matplotlib
        if matplotlib.rcParams['text.usetex']:
            return text.replace('_', '{\\textunderscore}')
    return text


def mergeRenames(*dicts, **kwargs):
    """
    Joins several dicts of renames.

    If `keep_names_1st=True` (default: `False`), keeps empty entries when possible
    in order to preserve the parameter names of the first input dictionary.

    Returns a merged dictionary of renames,
    whose keys are chosen from the left-most input.
    """
    keep_names_1st = kwargs.pop("keep_names_1st", False)
    if kwargs:
        raise ValueError("kwargs not recognized: %r" % kwargs)
    sets = list(chain(*[[set([k] + (makeList(v or [])))
                         for k, v in dic.items()] for dic in dicts]))
    # If two sets have elements in common, join them.
    something_changed = True
    out = []
    while something_changed:
        something_changed = False
        for i in range(1, len(sets)):
            if sets[0].intersection(sets[i]):
                sets[0] = sets[0].union(sets.pop(i))
                something_changed = True
                break
        if not something_changed and sets:
            out += [sets.pop(0)]
            if len(sets):
                something_changed = True
    merged = {}
    for params in out:
        for dic in dicts:
            p = set(dic).intersection(params)
            if p and (params != p or keep_names_1st):
                key = p.pop()
                params.remove(key)
                merged[key] = list(params)
                break
    return merged


class ParamInfo:
    """
    Parameter information object.

    :ivar name: the parameter name tag (no spacing or punctuation)
    :ivar label: latex label (without enclosing $)
    :ivar comment: any descriptive comment describing the parameter
    :ivar isDerived: True if a derived parameter, False otherwise (e.g. for MCMC parameters)
    """

    def __init__(self, line=None, name='', label='', comment='', derived=False,
                 renames=None, number=None):
        self.setName(name)
        self.isDerived = derived
        self.label = label or name
        self.comment = comment
        self.filenameLoadedFrom = ''
        self.number = number
        self.renames = makeList(renames or [])
        if line is not None:
            self.setFromString(line)

    def nameEquals(self, name):
        if isinstance(name, ParamInfo):
            return name.name == name
        else:
            return name == name

    def setFromString(self, line):
        items = line.split(None, 1)
        name = items[0]
        if name.endswith('*'):
            name = name.strip('*')
            self.isDerived = True
        self.setName(name)
        if len(items) > 1:
            tmp = items[1].split('#', 1)
            self.label = tmp[0].strip().replace('!', '\\')
            if len(tmp) > 1:
                self.comment = tmp[1].strip()
            else:
                self.comment = ''
        return self

    def setName(self, name):
        if not isinstance(name, str):
            raise ValueError('"name" must be a parameter name string not %s: %s' % (type(name), name))
        if '*' in name or '?' in name or ' ' in name or '\t' in name:
            raise ValueError('Parameter names must not contain spaces, * or ?')
        self.name = name

    def getLabel(self):
        if self.label:
            return self.label
        else:
            return self.name

    def latexLabel(self):
        if self.label:
            return '$' + self.label + '$'
        else:
            return self.name

    def setFromStringWithComment(self, items):
        self.setFromString(items[0])
        if items[1] != 'NULL':
            self.comment = items[1]

    def string(self, wantComments=True):
        res = self.name
        if self.isDerived:
            res += '*'
        res = res + '\t' + self.label
        if wantComments and self.comment != '':
            res = res + '\t#' + self.comment
        return res

    def __str__(self):
        return self.string()


class ParamList:
    """
    Holds an orders list of :class:`ParamInfo` objects describing a set of parameters.

    :ivar names: list of :class:`ParamInfo` objects
    """
    loadFromFile: callable

    def __init__(self, fileName=None, setParamNameFile=None, default=0, names=None, labels=None):
        """
        :param fileName: name of .paramnames file to load from
        :param setParamNameFile: override specific parameter names' labels using another file
        :param default: set to int>0 to automatically generate that number of default names and labels
                       (param1, p_{1}, etc.)
        :param names: a list of name strings to use
        """
        self.names = []
        self.info_dict = None  # if read from yaml file, saved here
        if default:
            self.setDefault(default)
        if names is not None:
            self.setWithNames(names)
        if fileName is not None:
            self.loadFromFile(fileName)
        if setParamNameFile is not None:
            self.setLabelsFromParamNames(setParamNameFile)
        if labels is not None:
            self.setLabels(labels)

    def setDefault(self, n):
        self.names = [ParamInfo(name='param' + str(i + 1), label='p_{' + str(i + 1) + '}') for i in range(n)]
        return self

    def setWithNames(self, names):
        self.names = [ParamInfo(name) for name in names]
        return self

    def setLabels(self, labels):
        for name, label in zip(self.names, labels):
            name.label = label

    def numDerived(self):
        return len([1 for info in self.names if info.isDerived])

    def list(self):
        """
        Gets a list of parameter name strings
        """
        return [name.name for name in self.names]

    def labels(self):
        """
        Gets a list of parameter labels
        """
        return [name.label for name in self.names]

    def listString(self):
        return " ".join(self.list())

    def numParams(self):
        return len(self.names)

    def numNonDerived(self):
        return len([1 for info in self.names if not info.isDerived])

    def parWithNumber(self, num):
        for par in self.names:
            if par.number == num:
                return par
        return None

    def _check_name_str(self, name):
        if not isinstance(name, str):
            raise ValueError('"name" must be a parameter name string not %s: %s' % (type(name), name))

    def parWithName(self, name, error=False, renames=None):
        """
        Gets the :class:`ParamInfo` object for the parameter with the given name

        :param name: name of the parameter
        :param error: if True raise an error if parameter not found, otherwise return None
        :param renames: a dictionary that is used to provide optional name mappings
                        to the stored names
        """
        self._check_name_str(name)
        given_names = {name}
        if renames:
            given_names.update(makeList(renames.get(name, [])))
        for par in self.names:
            known_names = set([par.name] + makeList(getattr(par, 'renames', [])) +
                              (makeList(renames.get(par.name, [])) if renames else []))
            if known_names.intersection(given_names):
                return par
        if error:
            raise Exception("parameter name not found: %s" % name)
        return None

    def numberOfName(self, name):
        """
        Gets the parameter number of the given parameter name

        :param name: parameter name tag
        :return: index of the parameter, or -1 if not found
        """
        self._check_name_str(name)
        for i, par in enumerate(self.names):
            if par.name == name:
                return i
        return -1

    def hasParam(self, name):
        return self.numberOfName(name) != -1

    def parsWithNames(self, names, error=False, renames=None):
        """
        gets the list of :class:`ParamInfo` instances for given list of name strings.
        Also expands any names that are globs into list with matching parameter names

        :param names: list of name strings
        :param error: if True, raise an error if any name not found,
                      otherwise returns None items. Can be a list of length `len(names)`
        :param renames: optional dictionary giving mappings of parameter names
        """
        res = []
        if isinstance(names, str):
            names = [names]
        errors = makeList(error)
        if len(errors) < len(names):
            errors = len(names) * errors
        for name, error in zip(names, errors):
            if isinstance(name, ParamInfo):
                res.append(name)
            else:
                if '?' in name or '*' in name:
                    res += self.getMatches(name)
                else:
                    res.append(self.parWithName(name, error, renames))
        return res

    def getMatches(self, pattern, strings=False):
        pars = []
        for par in self.names:
            if fnmatch.fnmatchcase(par.name, pattern):
                if strings:
                    pars.append(par.name)
                else:
                    pars.append(par)
        return pars

    def setLabelsFromParamNames(self, fname):
        self.setLabelsAndDerivedFromParamNames(fname, False)

    def setLabelsAndDerivedFromParamNames(self, fname, set_derived=True):
        if isinstance(fname, ParamNames):
            p = fname
        else:
            p = ParamNames(fname)
        for par in p.names:
            param = self.parWithName(par.name)
            if param is not None:
                param.label = par.label
                if set_derived:
                    param.isDerived = par.isDerived

    def getRenames(self, keep_empty=False):
        """
        Gets dictionary of renames known to each parameter.
        """
        return {param.name: getattr(param, "renames", [])
                for param in self.names
                if (getattr(param, "renames", False) or keep_empty)}

    def updateRenames(self, renames):
        """
        Updates the renames known to each parameter with the given dictionary of renames.
        """
        merged_renames = mergeRenames(
            self.getRenames(keep_empty=True), renames, keep_names_1st=True)
        known_names = self.list()
        for name, rename in merged_renames.items():
            if name in known_names:
                self.parWithName(name).renames = rename

    def fileList(self, fname):
        with open(fname, encoding='utf-8-sig') as f:
            textFileLines = f.readlines()
        return textFileLines

    def deleteIndices(self, indices):
        self.names = [name for i, name in enumerate(self.names) if i not in indices]

    def filteredCopy(self, params):
        usedNames = self.__class__()
        for name in self.names:
            if isinstance(params, list):
                p = name.name in params
            else:
                p = params.parWithName(name.name)
            if p:
                usedNames.names.append(name)
        return usedNames

    def addDerived(self, name, **kwargs):
        """
        adds a new parameter

        :param name: name tag for the new parameter
        :param kwargs: other arguments for constructing the new :class:`ParamInfo`
        """
        if kwargs.get('derived') is None:
            kwargs['derived'] = True
        self._check_name_str(name)
        kwargs['name'] = name
        self.names.append(ParamInfo(**kwargs))
        return self.names[-1]

    def maxNameLen(self):
        return max([len(name.name) for name in self.names])

    def parFormat(self):
        maxLen = max(9, self.maxNameLen()) + 1
        return "%-" + str(maxLen) + "s"

    def name(self, ix, tag_derived=False):
        par = self.names[ix]
        if tag_derived and par.isDerived:
            return par.name + '*'
        else:
            return par.name

    def __str__(self):
        text = ''
        for par in self.names:
            text += par.string() + '\n'
        return text

    def saveAsText(self, filename):
        """
        Saves to a plain text .paramnames file

        :param filename: filename to save to
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(self))

    def getDerivedNames(self):
        """
        Get the names of all derived parameters
        """
        return [name.name for name in self.names if name.isDerived]

    def getRunningNames(self):
        """
        Get the names of all running (non-derived) parameters
        """
        return [name.name for name in self.names if not name.isDerived]


class ParamNames(ParamList):
    """
    Holds an orders list of :class:`ParamInfo` objects describing a set of parameters,
    inheriting from :class:`ParamList`.

    Can be constructed programmatically, and also loaded and saved to a .paramnames files, which is a plain text file
    giving the names and optional label and comment for each parameter, in order.

    :ivar names: list of :class:`ParamInfo` objects describing each parameter
    :ivar filenameLoadedFrom: if loaded from file, the file name
    """

    def loadFromFile(self, fileName):
        """
        loads from fileName, a plain text .paramnames file or a "full" yaml file
        """

        self.filenameLoadedFrom = os.path.split(fileName)[1]
        extension = os.path.splitext(fileName)[-1]
        if extension == '.paramnames':
            with open(fileName, encoding='utf-8-sig') as f:
                self.names = [ParamInfo(line) for line in [s.strip() for s in f] if line != '']
        elif extension.lower() in ('.yaml', '.yml'):
            from getdist import yaml_tools
            from getdist.cobaya_interface import get_info_params, is_sampled_param
            from getdist.cobaya_interface import is_derived_param, _p_label, _p_renames
            self.info_dict = yaml_tools.yaml_load_file(fileName)
            info_params = get_info_params(self.info_dict)
            # first sampled, then derived
            self.names = [ParamInfo(name=param, label=(info or {}).get(_p_label, param),
                                    renames=(info or {}).get(_p_renames))
                          for param, info in info_params.items() if is_sampled_param(info)]
            self.names += [ParamInfo(name=param, label=(info or {}).get(_p_label, param),
                                     renames=(info or {}).get(_p_renames), derived=True)
                           for param, info in info_params.items() if is_derived_param(info)]
        else:
            raise ValueError('ParanNames must be loaded from .paramnames or .yaml/.yml file, '
                             'found %s' % fileName)

    def loadFromKeyWords(self, keywordProvider):
        num_params_used = keywordProvider.keyWord_int('num_params_used')
        num_derived_params = keywordProvider.keyWord_int('num_derived_params')
        nparam = num_params_used + num_derived_params
        for i in range(nparam):
            info = ParamInfo()
            info.setFromStringWithComment(keywordProvider.keyWordAndComment('param_' + str(i + 1)))
            self.names.append(info)
        return nparam

    def saveKeyWords(self, keywordProvider):
        keywordProvider.setKeyWord_int('num_params_used', len(self.names) - self.numDerived())
        keywordProvider.setKeyWord_int('num_derived_params', self.numDerived())
        for i, name in enumerate(self.names):
            keywordProvider.setKeyWord('param_' + str(i + 1), name.string(False).replace('\\', '!'),
                                       name.comment)
