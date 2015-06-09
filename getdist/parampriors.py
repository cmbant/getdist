import os


class ParamBounds(object):
    def __init__(self, fileName=None):
        self.names = []
        self.lower = {}
        self.upper = {}
        if fileName is not None: self.loadFromFile(fileName)

    def loadFromFile(self, fileName):
        self.filenameLoadedFrom = os.path.split(fileName)[1]
        with open(fileName) as f:
            for line in f:
                strings = [text.strip() for text in line.split()]
                if len(strings) == 3:
                    self.setRange(strings[0], strings[1:])

    def __str__(self):
        s = ''
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
            s += "%22s%17s%17s\n" % (name, lim1, lim2)
        return s

    def saveToFile(self, fileName):
        with open(fileName, 'w') as f:
            f.write(str(self))

    def setRange(self, name, strings):
        if strings[0] != 'N' and strings[0] is not None: self.lower[name] = float(strings[0])
        if strings[1] != 'N' and strings[1] is not None: self.upper[name] = float(strings[1])
        if not name in self.names: self.names.append(name)

    def getUpper(self, name):
        return self.upper.get(name, None)

    def getLower(self, name):
        return self.lower.get(name, None)
