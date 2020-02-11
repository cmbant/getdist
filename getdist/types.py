import decimal
import os
from io import BytesIO
import numpy as np
from getdist.paramnames import ParamInfo, ParamList
import tempfile

_sci_tolerance = 4


class TextFile:
    def __init__(self, lines=None):
        if isinstance(lines, str):
            lines = [lines]
        self.lines = lines or []

    def write(self, outfile):
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.lines))


def texEscapeText(string):
    return string.replace('_', '{\\textunderscore}')


def times_ten_power(exponent):
    return r'\cdot 10^{%d}' % exponent


def float_to_decimal(f):
    # http://docs.python.org/library/decimal.html#decimal-faq
    """Convert a floating point number to a Decimal with no loss of information"""
    n, d = f.as_integer_ratio()
    numerator, denominator = decimal.Decimal(n), decimal.Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)
    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)
    return result


# noinspection PyUnboundLocalVariable
def numberFigs(number, sigfig, sci=False):
    # http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python/2663623#2663623
    assert (sigfig > 0)
    try:
        d = decimal.Decimal(number)
    except TypeError:
        d = float_to_decimal(float(number))
    if sci:
        exponent = d.adjusted()
        if abs(exponent) > _sci_tolerance:
            d = decimal.getcontext().multiply(d, float_to_decimal(10. ** -exponent))
        else:
            exponent = 0
    sign, digits = d.as_tuple()[0:2]
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))
    shift = d.adjusted()
    result = int(''.join(map(str, digits[:sigfig])))
    # Round the result
    if len(digits) > sigfig and digits[sigfig] >= 5:
        result += 1
    result = list(str(result))
    # Rounding can change the length of result
    # If so, adjust shift
    shift += len(result) - sigfig
    # reset len of result to sigfig
    result = result[:sigfig]
    if shift >= sigfig - 1:
        # Tack more zeros on the end
        result += ['0'] * (shift - sigfig + 1)
    elif 0 <= shift:
        # Place the decimal point in between digits
        result.insert(shift + 1, '.')
    else:
        # Tack zeros on the front
        assert (shift < 0)
        result = ['0.'] + ['0'] * (-shift - 1) + result
    if sign:
        result.insert(0, '-')
    if sci:
        return ''.join(result), exponent
    return ''.join(result)


class NumberFormatter:
    def __init__(self, sig_figs=4, separate_limit_tol=0.1, err_sf=2):
        self.sig_figs = sig_figs
        self.separate_limit_tol = separate_limit_tol
        self.err_sf = err_sf

    # noinspection PyUnboundLocalVariable
    def namesigFigs(self, value, limplus, limminus, wantSign=True, sci=False):
        frac = limplus / (abs(value) + limplus)
        sf = self.sig_figs
        if frac > 0.1 and 100 > value >= 20:
            sf = 2
        elif frac > 0.01 and value < 1000:
            sf = 3
        err_sf = self.err_sf
        if value >= 20 and frac > 0.1 and limplus >= 2:
            err_sf = 1
        if sci:
            # First, call without knowing sig figs, to get the exponent
            exponent = self.formatNumber(max(abs(value - limminus), abs(value + limplus)), sci=True)[1]
            if exponent:
                value, limplus, limminus = [
                    (lambda x: decimal.getcontext().multiply(
                        float_to_decimal(x), float_to_decimal(10. ** -exponent)))(lim)
                    for lim in [value, limplus, limminus]]
        plus_str = self.formatNumber(limplus, err_sf, wantSign)
        minus_str = self.formatNumber(limminus, err_sf, wantSign)
        res = self.formatNumber(value, sf)
        maxdp = max(self.decimal_places(plus_str), self.decimal_places(minus_str))
        while maxdp < self.decimal_places(res):
            sf -= 1
            if sf == 0:
                res = ('%.' + str(maxdp) + 'f') % value
                if float(res) == 0.0:
                    res = ('%.' + str(maxdp) + 'f') % 0
                break
            else:
                res = self.formatNumber(value, sf)

        while self.decimal_places(plus_str) > self.decimal_places(res):
            sf += 1
            res = self.formatNumber(value, sf)
        if sci:
            return res, plus_str, minus_str, exponent
        else:
            return res, plus_str, minus_str

    # noinspection PyUnboundLocalVariable
    def formatNumber(self, value, sig_figs=None, wantSign=False, sci=False):
        if sig_figs is None:
            sf = self.sig_figs
        else:
            sf = sig_figs
        s = numberFigs(value, sf, sci=sci)
        if sci:
            s, exponent = s
        if wantSign:
            if s[0] != '-' and float(s) < 0:
                s = '-' + s
            if float(s) > 0:
                s = '+' + s
        if sci:
            return s, exponent
        else:
            return s

    def decimal_places(self, s):
        i = s.find('.')
        if i > 0:
            return len(s) - i - 1
        return 0

    def plusMinusLimit(self, limit, upper, lower):
        return limit != 1 or abs(abs(upper / lower) - 1) > self.separate_limit_tol


class TableFormatter:
    def __init__(self):
        self.border = '|'
        self.endofrow = '\\\\'
        self.hline = '\\hline'
        self.paramText = 'Parameter'
        self.aboveTitles = self.hline
        self.majorDividor = '|'
        self.minorDividor = '|'
        self.colDividor = '||'
        self.belowTitles = ''
        self.headerWrapper = " %s"
        self.noConstraint = '---'
        self.spacer = ' '  # just to make output more readable
        self.colSeparator = self.spacer + '&' + self.spacer
        self.numberFormatter = NumberFormatter()

    def getLine(self, position=None):
        if position is not None and hasattr(self, position):
            return getattr(self, position)
        return self.hline

    def belowTitleLine(self, colsPerParam, numResults=None):
        return self.getLine("belowTitles")

    def startTable(self, ncol, colsPerResult, numResults):
        part = self.majorDividor + (" c" + self.minorDividor) * (colsPerResult - 1) + ' c'
        return '\\begin{tabular} {' + self.border + " l " + part * numResults + (
                self.colDividor + " l " + part * numResults) * (
                       ncol - 1) + self.border + '}'

    def endTable(self):
        return '\\end{tabular}'

    def titleSubColumn(self, colsPerResult, title):
        return ' \\multicolumn{' + str(
            colsPerResult) + '}{' + self.majorDividor + 'c' + self.majorDividor + '}{' + self.formatTitle(title) + '}'

    def formatTitle(self, title):
        return '\\bf ' + texEscapeText(title)

    def texEquation(self, txt):
        if txt and txt[0] != '$':
            return '$' + txt + '$'
        else:
            return txt

    def textAsColumn(self, txt, latex=False, separator=False, bold=False):
        wid = len(txt)
        if latex:
            wid += 2
            if bold:
                wid += 11
        res = txt + self.spacer * max(0, 28 - wid)
        if latex:
            res = self.texEquation(res)
            if bold:
                res = '{\\boldmath' + res + '}'
        if separator:
            res += self.colSeparator
        return res


class OpenTableFormatter(TableFormatter):
    def __init__(self):
        TableFormatter.__init__(self)
        self.border = ''
        self.aboveTitles = r'\noalign{\vskip 3pt}' + self.hline + r'\noalign{\vskip 1.5pt}' \
                           + self.hline + r'\noalign{\vskip 5pt}'
        self.belowTitles = r'\noalign{\vskip 3pt}' + self.hline
        self.aboveHeader = ''
        self.belowHeader = self.hline
        self.minorDividor = ''
        self.belowFinalRow = ''

    def titleSubColumn(self, colsPerResult, title):
        return ' \\multicolumn{' + str(colsPerResult) + '}{' + 'c' + '}{' + self.formatTitle(title) + '}'


class NoLineTableFormatter(OpenTableFormatter):
    def __init__(self):
        OpenTableFormatter.__init__(self)
        self.aboveHeader = ''
        # self.belowHeader = r'\noalign{\vskip 5pt}'
        self.minorDividor = ''
        self.majorDividor = ''
        self.belowFinalRow = self.hline
        self.belowBlockRow = self.hline
        self.colDividor = '|'
        self.hline = ''

    def belowTitleLine(self, colsPerParam, numResults=None):
        return r'\noalign{\vskip 3pt}\cline{2-' + str(colsPerParam * numResults + 1) + r'}\noalign{\vskip 3pt}'


class ResultTable:
    """
    Class for holding a latex table of parameter statistics
    """

    def __init__(self, ncol, results, limit=2, tableParamNames=None, titles=None, formatter=None,
                 numFormatter=None, blockEndParams=None, paramList=None, refResults=None, shiftSigma_indep=False,
                 shiftSigma_subset=False):
        """
        :param ncol: number of columns
        :param results: a :class:`MargeStats` or :class:`BestFit` instance, or a list of them for
                        comparing different results
        :param limit: which limit to include (1 is first limit calculated, usually 68%, 2 the second, usually 95%)
        :param tableParamNames: optional :class:`~.paramnames.ParamNames` instance listing particular
                                parameters to include
        :param titles: optional titles describing different results
        :param formatter: a table formatting class
        :param numFormatter: a number formatting class
        :param blockEndParams: mark parameters in blocks, ending on this list of parameter names
        :param paramList: a list of parameter names (strings) to include
        :param refResults: for showing parameter shifts, a reference :class:`MargeStats` instance to show differences to
        :param shiftSigma_indep: show parameter shifts in sigma assuming data are independent
        :param shiftSigma_subset: show parameter shifts in sigma assuming data are a subset of each other
        """
        # results is a margeStats or bestFit table
        self.lines = []
        if formatter is None:
            self.format = NoLineTableFormatter()
        else:
            self.format = formatter
        self.ncol = ncol
        if tableParamNames is None:
            self.tableParamNames = results[0]
        else:
            self.tableParamNames = tableParamNames
        if paramList is not None:
            self.tableParamNames = self.tableParamNames.filteredCopy(paramList)
        if numFormatter is not None:
            self.format.numFormatter = numFormatter

        self.results = results
        self.boldBaseParameters = True
        self.colsPerResult = len(results[0].getColumnLabels(limit))
        self.colsPerParam = len(results) * self.colsPerResult
        self.limit = limit
        self.refResults = refResults
        self.shiftSigma_indep = shiftSigma_indep
        self.shiftSigma_subset = shiftSigma_subset

        nparams = self.tableParamNames.numParams()
        numrow = nparams // ncol
        if nparams % ncol != 0:
            numrow += 1
        rows = []
        for par in self.tableParamNames.names[0:numrow]:
            rows.append([par])
        for col in range(1, ncol):
            for i in range(numrow * col, min(numrow * (col + 1), nparams)):
                rows[i - numrow * col].append(self.tableParamNames.names[i])

        self.lines.append(self.format.startTable(ncol, self.colsPerResult, len(results)))
        if titles is not None:
            self.addTitlesRow(titles)
        self.addHeaderRow()
        for row in rows[:-1]:
            self.addFullTableRow(row)
            if ncol == 1 and blockEndParams is not None and row[0].name in blockEndParams:
                self.addLine("belowBlockRow")
            else:
                self.addLine("belowRow")
        self.addFullTableRow(rows[-1])
        self.addLine("belowFinalRow")
        self.endTable()

    def addFullTableRow(self, row):
        txt = self.format.colSeparator.join(self.paramLabelColumn(param) + self.paramResultsTex(param) for param in row)
        if not self.ncol == len(row):
            txt += self.format.colSeparator * ((1 + self.colsPerParam) * (self.ncol - len(row)))
        self.lines.append(txt + self.format.endofrow)

    def addLine(self, position):
        if self.format.getLine(position) is None:  # no line is appended if the attribute is None
            return self.lines
        else:
            return self.lines.append(self.format.getLine(position))

    def addTitlesRow(self, titles):
        self.addLine("aboveTitles")
        cols = [self.format.titleSubColumn(1, '')]
        cols += [self.format.titleSubColumn(self.colsPerResult, title) for title in titles]
        self.lines.append(self.format.colSeparator.join(cols * self.ncol) + self.format.endofrow)

        belowTitleLine = self.format.belowTitleLine(self.colsPerResult, self.colsPerParam // self.colsPerResult)
        if belowTitleLine:
            self.lines.append(belowTitleLine)

    def addHeaderRow(self):
        self.addLine("aboveHeader")
        cols = [self.format.headerWrapper % self.format.paramText]
        for result in self.results:
            cols += [self.format.headerWrapper % s for s in result.getColumnLabels(self.limit)]
        self.lines.append(self.format.colSeparator.join(cols * self.ncol) + self.format.endofrow)

        self.addLine("belowHeader")

    def paramResultsTex(self, param):
        return self.format.colSeparator.join(self.paramResultTex(result, param) for result in self.results)

    def paramResultTex(self, result, p):
        values = result.texValues(self.format, p, self.limit, self.refResults,
                                  shiftSigma_subset=self.shiftSigma_subset, shiftSigma_indep=self.shiftSigma_indep)
        if values is not None:
            if len(values) > 1:
                txt = self.format.textAsColumn(values[1], True, separator=True)
            else:
                txt = ''
            txt += self.format.textAsColumn(values[0], values[0] != self.format.noConstraint)
            return txt
        else:
            return self.format.textAsColumn('') * len(result.getColumnLabels(self.limit))

    def paramLabelColumn(self, param):
        return self.format.textAsColumn(param.getLabel(), True, separator=True, bold=not param.isDerived)

    def endTable(self):
        self.lines.append(self.format.endTable())

    def tableTex(self, document=False, latex_preamble=None, packages=('amsmath', 'amssymb', 'bm')):
        """
        Get the latex string for the table

        :param document: if True, make a full latex file, if False just the snippet for including in another file
        :param latex_preamble: any preamble to include in the latex file
        :param packages: list of packages to load
        """

        if document:
            lines = [r'\documentclass{article}', r'\pagestyle{empty}']
            for package in packages:
                lines.append(r'\usepackage{%s}' % package)
            lines.append('\\renewcommand{\\arraystretch}{1.5}')
            if latex_preamble:
                lines.append(latex_preamble)
            lines.append('\\begin{document}')
            lines += self.lines
            lines.append('\\end{document}')
        else:
            lines = self.lines
        return "\n".join(lines)

    def write(self, fname, **kwargs):
        """
        Write the latex for the table to a file

        :param fname: filename to write
        :param kwargs: arguments for :func:`~ResultTable.tableTex`
        """
        TextFile(self.tableTex(**kwargs)).write(fname)

    def tablePNG(self, dpi=None, latex_preamble=None, filename=None, bytesIO=False):
        """
        Get a .png file image of the table. You must have latex installed to use this.

        :param dpi: dpi settings for the png
        :param latex_preamble: any latex preamble
        :param filename: filename to save to (defaults to file in the temp directory)
        :param bytesIO: if True, return a BytesIO instance holding the .png data
        :return: if bytesIO, the BytesIO instance, otherwise name of the output file
        """
        texfile = tempfile.mktemp(suffix='.tex')
        self.write(texfile, document=True, latex_preamble=latex_preamble)
        basefile = os.path.splitext(texfile)[0]
        outfile = filename or basefile + '.png'
        old_pwd = os.getcwd()

        def runCommand(command):
            command += ' 2>%s 1>&2' % os.devnull
            os.system(command)

        try:
            os.chdir(os.path.dirname(texfile))
            runCommand('latex %s' % texfile)
            cmd = 'dvipng'
            if dpi:
                cmd += ' -D %s' % dpi
            cmd += ' -T tight -x 1000 -z 9 --truecolor -o "%s" "%s" ' \
                   % (outfile, basefile + '.dvi')
            runCommand(cmd)
        finally:
            for f in [basefile + ext for ext in ('.tex', '.dvi', '.aux', '.log')]:
                if os.path.isfile(f):
                    os.remove(f)
            os.chdir(old_pwd)
        if bytesIO:
            with open(outfile, 'rb') as f:
                result = BytesIO(f.read())
            os.remove(outfile)
            result.seek(0)
            return result
        else:
            return outfile


class ParamResults(ParamList):
    """
    Base class for a set of parameter results, inheriting from :class:`~.paramnames.ParamList`,
    so that self.names is a list of :class:`~.paramnames.ParamInfo` instances for each parameter, which
    have attribute holding results for the different parameters.
    """
    pass


class LikelihoodChi2:
    name: str
    tag: str
    chisq: float


class BestFit(ParamResults):
    """
    Class holding the result of a likelihood minimization, inheriting from :class:`ParamResults`.
    The data is read from a specific formatted text file (.minimum or .bestfit) as output by CosmoMC or Cobaya.
    """

    def __init__(self, fileName=None, setParamNameFile=None, want_fixed=False, max_posterior=True):
        """
        :param fileName: text file to load from, assumed to be in CosmoMC's .minimum format
        :param setParamNameFile: optional name of .paramnames file listing preferred parameter labels for the parameters
        :param want_fixed:  whether to include values of parameters that are not allowed to vary
        :param max_posterior: whether the file is a maximum posterior (default) or maximum likelihood
        """

        ParamResults.__init__(self)
        self.max_posterior = max_posterior
        if fileName is not None:
            self.loadFromFile(fileName, want_fixed=want_fixed)
        if setParamNameFile is not None:
            self.setLabelsFromParamNames(setParamNameFile)

    def getColumnLabels(self, **kwargs):
        return ['Best fit']

    def loadFromFile(self, filename, want_fixed=False):
        textFileLines = self.fileList(filename)
        first = textFileLines[0].strip().split('=')
        if first[0].strip() == 'weight':
            self.weight = float(first[1].strip())
            del (textFileLines[0])
            first = textFileLines[0].strip().split('=')
        if first[0].strip() != '-log(Like)':
            raise Exception('Error in format of parameter (best fit) file')
        self.logLike = float(first[1].strip())
        isFixed = False
        isDerived = False
        self.chiSquareds = []
        chunks = 0
        if len(textFileLines[1].strip()) > 0:
            del (textFileLines[1])  # if it has chi2 line as well
        for ix in range(2, len(textFileLines)):
            line = textFileLines[ix]
            if len(line.strip()) == 0:
                chunks += 1
                isFixed = not isFixed
                isDerived = True
                if chunks == 3:
                    if ix + 2 >= len(textFileLines):
                        break
                    for likePart in textFileLines[ix + 2:]:
                        if len(likePart.strip()) != 0:
                            (chisq, name) = [s.strip() for s in likePart.split(None, 2)][1:]
                            name = [s.strip() for s in name.split(':', 1)]
                            if len(name) > 1:
                                (kind, name) = name
                            else:
                                kind = ''
                            chi2 = LikelihoodChi2()
                            if '=' in name:
                                chi2.tag, chi2.name = [s.strip() for s in name.split('=')]
                            else:
                                chi2.tag, chi2.name = None, name
                            chi2.chisq = float(chisq)
                            self.chiSquareds.append((kind, chi2))
                    break
                continue
            if not isFixed or want_fixed:
                param = ParamInfo()
                param.isFixed = isFixed
                param.isDerived = isDerived
                (param.number, param.best_fit, param.name, param.label) = [s.strip() for s in line.split(None, 3)]
                param.number = int(param.number)
                param.best_fit = float(param.best_fit)
                self.names.append(param)

    def sortedChiSquareds(self):
        likes = dict()
        for kind, val in self.chiSquareds:
            if kind not in likes:
                likes[kind] = []
            likes[kind].append(val)
        return sorted(iter(likes.items()))

    def chiSquareForKindName(self, kind, name):
        for akind, val in self.chiSquareds:
            if akind == kind and val.name == name:
                return val.chisq
        return None

    def texValues(self, formatter, p, **kwargs):
        param = self.parWithName(p.name)
        if param is not None:
            return [formatter.numberFormatter.formatNumber(param.best_fit)]
        else:
            return None

    def getParamDict(self, include_derived=True):
        res = dict()
        for i, name in enumerate(self.names):
            if include_derived or not name.isDerived:
                res[name.name] = name.best_fit
        res['weight'] = 1
        res['loglike'] = self.logLike
        return res


class ParamLimit:
    """
    Class containing information about a marginalized parameter limit.

    :ivar lower: lower limit
    :ivar upper: upper limit
    :ivar twotail: True if a two-tail limit, False if one-tail
    :ivar onetail_upper: True if one-tail upper limit
    :ivar onetail_lower: True if one-tail lower limit
    """

    def __init__(self, minmax, tag='two'):
        """
        :param minmax: a [min,max] tuple with lower and upper limits. Entries be None if no limit.
        :param tag: a text tag descibing the limit, one of ['two' | '>' | '<' | 'none']
        """

        self.lower = minmax[0]
        self.upper = minmax[1]
        self.twotail = tag == 'two'
        self.onetail_upper = tag == '>'
        self.onetail_lower = tag == '<'

    def limitTag(self):
        """
        :return: Short text tag describing the type of limit (one-tail or two tail):

                - *two*: two-tail limit
                - *>*: a one-tail upper limit
                - *<*: a one-tail lower limit
                - *none*: no limits (both boundaries have high probability)
        """
        if self.twotail:
            return 'two'
        elif self.onetail_upper:
            return '>'
        elif self.onetail_lower:
            return '<'
        else:
            return 'none'

    def limitType(self):
        """
        :return: a text description of the type of limit. One of:

            - *two tail*
            - *one tail upper limit*
            - *one tail lower limit*
            - *none*
        """
        if self.twotail:
            return 'two tail'
        elif self.onetail_upper:
            return 'one tail upper limit'
        elif self.onetail_lower:
            return 'one tail lower limit'
        else:
            return 'none'

    def __str__(self):
        """
        :return: string representation of lower and upper bounds, with text description of the limit type
        """
        return "%g %g %s" % (self.lower, self.upper, self.limitTag())


class MargeStats(ParamResults):
    """
    Stores marginalized 1D parameter statistics, including mean, variance and confidence limits,
    inheriting from :class:`ParamResults`.

    Values are stored as attributes of the :class:`~.paramnames.ParamInfo` objects stored in self.names.
    Use *par= margeStats.parWithName('xxx')* to get the :class:`~.paramnames.ParamInfo` for parameter *xxx*;
    Values stored are:

    - *par.mean*: parameter mean
    - *par.err*: standard deviation
    - *limits*: list of :class:`~.types.ParamLimit` objects for the stored number of marginalized limits

    For example to get the first and second lower limits (default 68% and 95%) for parameter *xxx*::

         print(margeStats.names.parWithName('xxx').limits[0].lower)
         print(margeStats.names.parWithName('xxx').limits[1].lower)

    See  :class:`~.types.ParamLimit` for details of limits.
    """

    def loadFromFile(self, filename):
        """
        Load from a plain text file

        :param filename: file to load from
        """
        textFileLines = self.fileList(filename)
        lims = textFileLines[0].split(':')[1]
        self.limits = [float(s.strip()) for s in lims.split(';')]
        self.hasBestFit = False
        for line in textFileLines[3:]:
            if len(line.strip()) == 0:
                break
            param = ParamInfo()
            items = [s.strip() for s in line.split(None, len(self.limits) * 3 + 3)]
            param.name = items[0]
            if param.name[-1] == '*':
                param.isDerived = True
                param.name = param.name[:-1]
            param.mean = float(items[1])
            param.err = float(items[2])
            param.label = items[-1]
            param.limits = []
            for i in range(len(self.limits)):
                param.limits.append(ParamLimit([float(s) for s in items[3 + i * 3:5 + i * 3]], items[5 + i * 3]))
            self.names.append(param)

    def headerLine(self, inc_limits=False):
        parForm = self.parFormat()
        text = ""
        text += parForm % "parameter" + "  "
        text += "%-15s" % "mean"
        text += "%-15s" % "sddev"
        for j, limit in enumerate(self.limits):
            if inc_limits:
                tag = "_%.0f%%" % (limit * 100)
                limtxt = 'type'
            else:
                tag = str(j + 1)
                limtxt = "limit" + tag
            text += "%-15s" % ("lower" + tag)
            text += "%-15s" % ("upper" + tag)
            text += "%-7s" % limtxt
        return text, parForm

    def __str__(self):
        contours_str = '; '.join([str(c) for c in self.limits])
        header, parForm = self.headerLine()
        text = ""
        text += "Marginalized limits: %s\n\n" % contours_str
        text += header
        text += "\n"

        for j, par in enumerate(self.names):
            text += parForm % (self.name(j, True))
            text += "%15.7E%15.7E" % (par.mean, par.err)
            for lim in par.limits:
                text += "%15.7E%15.7E  %-5s" % (lim.lower, lim.upper, lim.limitTag())
            text += "   %s\n" % par.label
        return text

    def addBestFit(self, bf):
        self.hasBestFit = True
        self.logLike = bf.logLike
        # the next line deletes parameters not in best-fit;
        # this is good e.g. to get rid of yhe from importance sampled result
        self.names = [x for x in self.names if bf.parWithName(x.name) is not None]
        for par in self.names:
            param = bf.parWithName(par.name)
            par.best_fit = param.best_fit
            par.isDerived = param.isDerived

    def limitText(self, limit):
        txt = str(round(self.limits[limit - 1] * 100.))
        if txt.endswith(".0"):  # e.g. 95.0 -> 95
            txt = txt.split(".")[0]
        return txt

    def getColumnLabels(self, limit=2):
        if self.hasBestFit:
            res = ['Best fit']
        else:
            res = []
        return res + [self.limitText(limit) + '\\% limits']

    def texValues(self, formatter, p, limit=2, refResults=None, shiftSigma_indep=False, shiftSigma_subset=False):
        if not isinstance(p, ParamInfo):
            param = self.parWithName(p)
        else:
            param = self.parWithName(p.name)
        if param is not None:
            lim = param.limits[limit - 1]
            sf = 3
            if param.name.startswith('chi2'):
                # Chi2 for low dof are very skewed, always want mean and sigma or limit
                res, sigma, _ = formatter.numberFormatter.namesigFigs(param.mean, param.err, param.err, wantSign=False,
                                                                      sci=False)
                if limit == 1:
                    res += r'\pm ' + sigma
                else:
                    # in this case give mean and effective dof
                    res += r'\,({\nu\rm{:}\,%.1f})' % (param.err ** 2 / 2)
                    # res, plus_str, minus_str =
                    # formatter.numberFormatter.namesigFigs(param.mean, lim.upper - param.mean, lim.lower, sci=False)
                    # res += '^{' + plus_str + '}_{>' + minus_str + '}'
            elif lim.twotail:
                if not formatter.numberFormatter.plusMinusLimit(limit, lim.upper - param.mean, lim.lower - param.mean):
                    res, plus_str, _, exponent = formatter.numberFormatter.namesigFigs(param.mean, param.err, param.err,
                                                                                       wantSign=False, sci=True)
                    res += r'\pm ' + plus_str
                else:
                    res, plus_str, minus_str, exponent = formatter.numberFormatter.namesigFigs(param.mean,
                                                                                               lim.upper - param.mean,
                                                                                               lim.lower - param.mean,
                                                                                               sci=True)
                    res += '^{' + plus_str + '}_{' + minus_str + '}'
                if exponent:
                    res = r'\left(\,%s\,\right)' % res + times_ten_power(exponent)
            elif lim.onetail_upper:
                res, exponent = formatter.numberFormatter.formatNumber(lim.upper, sf, sci=True)
                res = '< ' + res
                if exponent:
                    res += times_ten_power(exponent)
            elif lim.onetail_lower:
                res, exponent = formatter.numberFormatter.formatNumber(lim.lower, sf, sci=True)
                res = '> ' + res
                if exponent:
                    res += times_ten_power(exponent)
            else:
                res = formatter.noConstraint
            if refResults is not None and res != formatter.noConstraint:
                refVal = refResults.parWithName(param.name)
                if refVal is not None:
                    delta = param.mean - refVal.mean
                    if shiftSigma_indep or shiftSigma_subset:
                        res += r'\quad('
                        if shiftSigma_subset:
                            # give mean shift in sigma units for subset data (regularized to max sigma/20)
                            subset_sigma = np.sqrt(abs(param.err ** 2 - refVal.err ** 2))
                            res += '%+.1f \\sigma_s' % (delta / max(subset_sigma, refVal.err / 20))
                        if shiftSigma_indep:
                            # give mean shift in sigma units for independent data
                            indep_sigma = np.sqrt(param.err ** 2 + refVal.err ** 2)
                            res += ', %+.1f \\sigma_i' % (delta / indep_sigma)
                        res += ')'
                    else:
                        res += r'\quad(%+.1f \\sigma)' % (delta / refVal.err)
            if self.hasBestFit:  # add best fit too
                rangew = (lim.upper - lim.lower) / 10
                bestfit, _, _, exponent = formatter.numberFormatter.namesigFigs(param.best_fit, rangew, -rangew,
                                                                                sci=True)
                if exponent:
                    bestfit += times_ten_power(exponent)
                return [res, bestfit]
            return [res]
        else:
            return None


class LikeStats(ParamResults):
    """
    Stores likelihood-related statistics, including best-fit sample and extremal values of the N-D confidence region,
    inheriting from :class:`ParamResults`.
    TODO: currently only saves to text, does not load full data from file
    """

    def loadFromFile(self, filename):
        textFileLines = self.fileList(filename)
        results = dict()
        for line in textFileLines:
            if len(line.strip()) == 0:
                break
            name, value = [x.strip() for x in line.split('=')]
            results[name] = float(value)
        self.logLike_sample = results.get('Best fit sample -log(Like)', None)
        self.logMeanInvLike = results.get('Ln(mean 1/like)', None)
        self.meanLogLike = results.get('mean(-Ln(like))', None)
        self.logMeanLike = results.get('-Ln(mean like)', None)
        self.complexity = results.get('complexity', None)

        # TODO: load N-D limits

    def likeSummary(self):
        text = "Best fit sample -log(Like) = %f\n" % self.logLike_sample
        if self.logMeanInvLike:
            text += "Ln(mean 1/like) = %f\n" % self.logMeanInvLike
        text += "mean(-Ln(like)) = %f\n" % self.meanLogLike
        text += "-Ln(mean like)  = %f\n" % self.logMeanLike
        # text += "complexity = %f\n" % self.complexity

        return text

    def headerLine(self):
        return self.parFormat() % "parameter" + '  bestfit        lower1         upper1         lower2         upper2\n'

    def __str__(self):
        text = self.likeSummary()
        parForm = self.parFormat()
        if len(self.names):
            text += "\n"
            text += self.headerLine()
            for j, par in enumerate(self.names):
                text += parForm % (self.name(j, True))
                text += '%15.7E%15.7E%15.7E%15.7E%15.7E   %s\n' % (par.bestfit_sample,
                                                                   par.ND_limit_bot[0], par.ND_limit_top[0],
                                                                   par.ND_limit_bot[1], par.ND_limit_top[1], par.label)
        return text


class ConvergeStats(ParamResults):
    def loadFromFile(self, filename):
        try:
            textFileLines = self.fileList(filename)
            self.R_eigs = []
            for i in range(len(textFileLines)):
                if textFileLines[i].find('var(mean)') >= 0:
                    for line in textFileLines[i + 1:]:
                        if len(line.strip()) == 0:
                            break
                        try:
                            self.R_eigs.append(line.split()[1])
                        except Exception:
                            self.R_eigs.append('1e30')
                elif 'Parameter auto-correlations' in textFileLines[i]:
                    self.auto_correlation_steps = [int(s) for s in textFileLines[i + 2].split()]
                    self.auto_correlations = []
                    self.auto_correlation_pars = []
                    for line in textFileLines[i + 3:]:
                        if len(line.strip()) == 0:
                            break
                        items = line.split(None, len(self.auto_correlation_steps) + 1)
                        self.auto_correlation_pars.append(items[0])
                        self.auto_correlations.append([float(s) for s in items[1:-1]])
        except:
            print('Error reading: ' + filename)
            raise

    def worstR(self):
        return self.R_eigs[len(self.R_eigs) - 1]
