# AL 2011-2015
import os
import fnmatch
import six


class ParamInfo(object):
    """
    Parameter information object.
    
    :ivar name: the parameter name tag (no spacing or punctuation)
    :ivar label: latex label (without enclosing $)
    :ivar comment: any descriptive comment describing the parameter
    :ivar isDerived: True if a derived parameter, False otherwise (e.g. for MCMC parameters)
    """

    def __init__(self, line=None, name='', label='', comment='', derived=False, number=None):
        self.setName(name)
        self.isDerived = derived
        self.label = label or name
        self.comment = comment
        self.filenameLoadedFrom = ''
        self.number = number
        if line is not None:
            self.setFromString(line)

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
        if items[1] != 'NULL': self.comment = items[1]

    def string(self, wantComments=True):
        res = self.name
        if self.isDerived: res += '*'
        res = res + '\t' + self.label
        if wantComments and self.comment != '':
            res = res + '\t#' + self.comment
        return res

    def __str__(self):
        return self.string()


class ParamList(object):
    """
    Holds an orders list of :class:`ParamInfo` objects describing a set of parameters.
        
    :ivar names: list of :class:`ParamInfo` objects
    """

    def __init__(self, fileName=None, setParamNameFile=None, default=0, names=None):
        """
        :param fileName: name of .paramnames file to load from
        :param setParamNameFile: override specific parameter names' labels using another file
        :param default: set to int>0 to automatically generate that number of default names and labels (param1, p_{1}, etc.)
        :param names: a list of name strings to use
        """
        self.names = []
        if default: self.setDefault(default)
        if names is not None: self.setWithNames(names)
        if fileName is not None: self.loadFromFile(fileName)
        if setParamNameFile is not None: self.setLabelsAndDerivedFromParamNames(setParamNameFile)

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

    def parWithName(self, name, error=False, renames={}):
        """
        Gets the :class:`ParamInfo` object for the parameter with the given name
        
        :param name: name of the parameter
        :param error: if True raise an error if parameter not found, otherwise return None
        :param renames: a dictionary that is used to provide optional name mappings to the stored names
        """
        for par in self.names:
            if par.name == name or renames.get(par.name, '') == name:
                return par
        if error: raise Exception("parameter name not found: " + name)
        return None

    def numberOfName(self, name):
        """
        Gets the parameter number of the given parameter name
        
        :param name: parameter name tag
        :return: index of the parameter, or -1 if not found
        """
        for i, par in enumerate(self.names):
            if par.name == name: return i
        return -1

    def parsWithNames(self, names, error=False, renames={}):
        """
        gets the list of :class:`ParamInfo` instances for given list of name strings.
        Also expands any names that are globs into list with matching parameter names

        :param names: list of name strings
        :param error: if True, raise an error if any name not found, otherwise returns None items
        :param renames: optional dictionary giving mappings of parameter names
        """
        res = []
        if isinstance(names, six.string_types):
            names = [names]
        for name in names:
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

    def setLabelsAndDerivedFromParamNames(self, fname):
        p = ParamNames(fname)
        for par in p.names:
            param = self.parWithName(par.name)
            if not param is None:
                param.label = par.label
                param.isDerived = par.isDerived

    def fileList(self, fname):
        with open(fname) as f:
            textFileLines = f.readlines()
        return textFileLines

    def deleteIndices(self, indices):
        self.names = [name for i, name in enumerate(self.names) if not i in indices]

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
        if kwargs.get('derived') is None: kwargs['derived'] = True
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
        with open(filename, 'w') as f:
            f.write(str(self))


class ParamNames(ParamList):
    """
    Holds an orders list of :class:`ParamInfo` objects describing a set of parameters, inheriting from :class:`ParamList`.
    
    Can be constructed programmatically, and also loaded and saved to a .paramnames files, which is a plain text file
    giving the names and optional label and comment for each parameter, in order.
    
    :ivar names: list of :class:`ParamInfo` objects describing each parameter
    :ivar filenameLoadedFrom: if loaded from file, the file name
    """

    def loadFromFile(self, fileName):
        """
        loads from fileName, a plain text .paramnames file
        """

        self.filenameLoadedFrom = os.path.split(fileName)[1]
        with open(fileName) as f:
            self.names = [ParamInfo(line) for line in [s.strip() for s in f] if line != '']

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
