from __future__ import print_function
import numpy as np
import io


class CovMat(object):
    """
    Class holding a covariance matrix for some named parameters

    :ivar matrix: the covariance matrix  (square numpy array)
    :ivar paramNames: list of parameter name strings
    """

    def __init__(self, filename='', matrix=None, paramNames=None):
        """
        :param filename: optionally, a file name to load from

        """
        if not paramNames:
            paramNames = []
        self.matrix = matrix
        self.paramNames = paramNames
        self.size = 0
        if matrix is not None: self.size = matrix.shape[0]
        if filename != '':
            self.loadFromFile(filename)

    def paramNameString(self):
        return " ".join(self.paramNames)

    def loadFromFile(self, filename):
        with open(filename) as f:
            first = f.readline().strip()
            if first.startswith('#'):
                self.paramNames = first[1:].split()
                self.size = len(self.paramNames)
            else:
                raise Exception('.covmat must now have parameter names header')
            self.matrix = np.loadtxt(f)

    def saveToFile(self, filename):
        """
        Save the covariance matrix to a text file, with comment header listing the parameter names

        :param filename: name of file to save to (.covmat)
        """
        with io.open(filename, 'wb') as fout:
            fout.write(('# ' + self.paramNameString() + '\n').encode('UTF-8'))
            np.savetxt(fout, self.matrix, '%15.7E')

    def rescaleParameter(self, name, scale):
        """
        Used to rescale a covariance if a parameter is renormalized

        :param name: parameter name to rescale
        :param scale: value to rescale by
        """
        if name in self.paramNames:
            i = self.paramNames.index(name)
            self.matrix[:, i] = self.matrix[:, i] * scale
            self.matrix[i, :] = self.matrix[i, :] * scale
        else:
            print('Not in covmat: ' + name)

    def mergeCovmatWhereNew(self, cov2):
        params1 = self.paramNames
        params2 = cov2.paramNames

        C = CovMat()
        C.paramNames.extend(params1)

        for param in cov2.paramNames:
            if param not in C.paramNames:
                C.paramNames.append(param)
        l1 = len(params1)
        l2 = len(params2)
        l = len(C.paramNames)

        map1 = dict(list(zip(params1, list(range(0, l1)))))
        map2 = dict(list(zip(params2, list(range(0, l2)))))
        covmap = dict(list(zip(list(range(0, l)), C.paramNames)))

        C.matrix = np.zeros((l, l))
        for i in range(0, l):
            for j in range(0, l):
                if C.paramNames[i] in params1 and C.paramNames[j] in params1:
                    C.matrix[i, j] = self.matrix[map1[covmap[i]], map1[covmap[j]]]
                elif C.paramNames[i] in params2 and C.paramNames[j] in params2:
                    C.matrix[i, j] = cov2.matrix[map2[covmap[i]], map2[covmap[j]]]

        return C

    def correlation(self):
        """
        Get the correlation matrix

        :return: numpy array giving the correlation matrix
        """
        m = self.matrix.copy()
        for i in range(self.size):
            s = np.sqrt(self.matrix[i, i])
            m[i, :] /= s
            m[:, i] /= s
        return m

    def plot(self):
        """
        Plot the correlation matrix as grid of colored squares
        """
        import matplotlib.pyplot as plt

        plt.pcolor(self.correlation())
        plt.colorbar()
        sz = self.size
        plt.yticks(np.arange(0.5, sz + .5), list(range(1, sz + 1)))
        plt.gca().set_yticklabels(self.paramNames)
        plt.xticks(np.arange(0.5, sz + .5), list(range(1, sz + 1)))
        plt.xlim([0, sz])
        plt.ylim([0, sz])
