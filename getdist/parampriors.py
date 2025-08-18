import os

import numpy as np


class ParamBounds:
    """
    Class for holding list of parameter bounds (e.g. for plotting, or hard priors).
    A limit is None if not specified, denoted by 'N' if read from a string or file

    :ivar names: list of parameter names
    :ivar lower: dict of lower limits, indexed by parameter name
    :ivar upper: dict of upper limits, indexed by parameter name
    """

    def __init__(self, fileName=None):
        """
        :param fileName: optional file name to read from
        """
        self.names = []
        self.lower = {}
        self.upper = {}
        self.periodic = set()
        if fileName is not None:
            self.loadFromFile(fileName)

    def loadFromFile(self, fileName):
        self.filenameLoadedFrom = os.path.split(fileName)[1]
        extension = os.path.splitext(fileName)[-1]
        if extension in (".ranges", ".bounds"):
            with open(fileName, encoding="utf-8-sig") as f:
                for line in f:
                    strings = [text.strip() for text in line.split()]
                    if len(strings) in [3, 4]:
                        self.setRange(strings[0], strings[1:])
        elif extension in (".yaml", ".yml"):
            from getdist.cobaya_interface import get_info_params, get_range

            info_params = get_info_params(fileName)
            for p, info in info_params.items():
                self.setRange(p, get_range(info))
        else:
            raise ValueError("ParamBounds must be loaded from .bounds, .ranges or .yaml/.yml file, not %s" % fileName)

    def __str__(self):
        s = ""
        for name in self.names:
            valMin = self.getLower(name)
            if valMin is not None:
                lim1 = "%15.7E" % valMin
            else:
                lim1 = "    N"
            valMax = self.getUpper(name)
            if valMax is not None:
                lim2 = "%15.7E" % valMax
            else:
                lim2 = "    N"
            if name in self.periodic:
                s += "%22s%17s%17s%10s\n" % (name, lim1, lim2, "periodic")
            else:
                s += "%22s%17s%17s\n" % (name, lim1, lim2)

        return s

    def saveToFile(self, fileName):
        """
        Save to a plain text file

        :param fileName: file name to save to
        """
        with open(fileName, "w", encoding="utf-8") as f:
            f.write(str(self))

    def _check_name(self, name):
        if not isinstance(name, str):
            raise ValueError(f'"name" must be a parameter name string not {type(name)}: {name}')

    def setFixed(self, name, value):
        self.setRange(name, (value, value))

    def setRange(self, name, strings):
        if strings[0] is None and strings[1] is None:
            return
        self._check_name(name)
        if strings[0] != "N" and strings[0] is not None and strings[0] != -np.inf:
            self.lower[name] = float(strings[0])
        if strings[1] != "N" and strings[1] is not None and strings[1] != np.inf:
            self.upper[name] = float(strings[1])
        if len(strings) > 2:
            periodic = strings[2]
            if periodic is True or isinstance(periodic, str) and periodic.upper() in ["T", "TRUE", "PERIODIC"]:
                if name not in self.upper or name not in self.lower:
                    raise ValueError(f"Periodic parameter must have lower and upper bound: {name}")
                self.periodic.add(name)
            elif periodic is not False and (not isinstance(periodic, str) or periodic.upper() not in ["F", "FALSE"]):
                raise ValueError(f"Unknown value for periodic range settings for param {name}: {periodic}")

        if name not in self.names:
            self.names.append(name)

    def getUpper(self, name):
        """
        :param name: parameter name
        :return: upper limit, or None if not specified
        """
        self._check_name(name)
        return self.upper.get(name, None)

    def getLower(self, name):
        """
        :param name: parameter name
        :return: lower limit, or None if not specified
        """
        self._check_name(name)
        return self.lower.get(name, None)

    def fixedValue(self, name):
        """
        :param name: parameter name
        :return: if range has zero width return fixed value else return None
        """
        lower = self.lower.get(name, None)
        if lower is not None:
            higher = self.upper.get(name, None)
            if higher is not None:
                if higher == lower:
                    return lower
        return None

    def fixedValueDict(self):
        """
        :return: dictionary of fixed parameter values
        """
        res = {}
        for name in self.names:
            value = self.fixedValue(name)
            if value is not None:
                res[name] = value
        return res
