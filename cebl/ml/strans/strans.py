import matplotlib.pyplot as plt
import numpy as np

from cebl import util


class STrans:
    """Base class for linear, spatial signal transforms.
    """
    def __init__(self, s, lags=0, demean=True):
        s = util.colmat(s)
        self.dtype = s.dtype

        self.nDim = s.shape[1]
        self.lags = lags
        self.demean = demean
        self.means = self.lagize(s).mean(axis=0)
        self.nComp = len(self.means)

        self.w = np.eye(self.nComp, dtype=self.dtype)
        self.wInv = np.eye(self.nComp, dtype=self.dtype)

    def lagize(self, s):
        return util.timeEmbed(s, lags=self.lags)

    def prep(self, s):
        s = self.lagize(util.colmat(s))

        if self.demean:
            return s - self.means
        else:
            return s

    def getNComp(self):
        return self.nComp

    def transform(self, s, comp=None, remove=False):
        s = self.prep(s)

        y = s.dot(self.w)

        if comp is None or not comp:
            return y
        else:
            compInd = np.array([remove,]*s.shape[1])
            compInd[np.array(comp)] = not remove

            return y[:,compInd]

    def filter(self, s, comp, remove=False):
        s = self.prep(s)

        y = s.dot(self.w)

        if comp is None or not comp:
            compInd = np.ones(s.shape[1], dtype=self.dtype)
        else:
            compInd = np.empty(s.shape[1], dtype=self.dtype)
            compInd[...] = remove
            compInd[np.array(comp)] = not remove

        compMat = np.diag(compInd)

        filt = y.dot(compMat).dot(self.wInv) + self.means
        return filt[:,:self.nDim]

    def plotTransform(self, s, comp=None, remove=False, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        y = self.transform(s, comp=comp, remove=remove)
        lines = ax.plot(y+util.colsep(y), **kwargs)

        return {'ax': ax, 'lines': lines}

    def plotFilter(self, s, comp, remove=False, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        filt = self.filter(s, comp=comp, remove=remove)
        lines = ax.plot(filt+util.colsep(filt), **kwargs)

        return {'ax': ax, 'lines': lines}
