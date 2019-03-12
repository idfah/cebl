import numpy as np

from cebl import util
from cebl.util.errm import *


class Regression:
    def __init__(self, nIn, nOut):
        self.nIn = nIn
        self.nOut = nOut

    def train(self, x, g):
        raise NotImplementedError("train not implemented.")

    def eval(self, x):
        raise NotImplementedError("eval not implemented.")

    def evals(self, xs, *args, **kwargs):
        xs = np.asarray(xs)

        if xs.ndim == 3:
            x = xs.reshape((xs.shape[0]*xs.shape[1], -1))
        else:
            x = xs.ravel()

        y = self.eval(x, *args, **kwargs)

        if y.ndim == 2:
            return y.reshape((xs.shape[0], xs.shape[1], -1))
        else:
            return y.reshape((xs.shape[0], xs.shape[1]))

    def resid(self, x, g, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return g - y

    def abe(self, x, g, axis=None, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return abe(y, g, axis=axis)

    def sse(self, x, g, axis=None, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return sse(y, g, axis=axis)

    def mse(self, x, g, axis=None, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return mse(y, g, axis=axis)

    def rmse(self, x, g, axis=None, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return rmse(y, g, axis=axis)

    def nrmse(self, x, g, axis=None, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return nrmse(y, g, axis=axis)

    def gini(self, x, g, normalize=True, *args, **kwargs):
        y = self.eval(x, *args, **kwargs)
        return gini(y, g, normalize=normalize)
