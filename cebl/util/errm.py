"""Error metrics.
"""

import numpy as np


def lpnorm(x, y=None, p=2, axis=None):
    """LP-Norm along a given axis.
    """
    if y is None:
        v = x
    else:
        v = y-x

    if p == np.inf:
        return np.max(np.abs(v), axis=axis)
    else:
        return np.sum(np.abs(v)**p, axis=axis)**(1.0/p)

def sae(y, g=0.0, axis=None):
    return np.sum(np.abs(y-g), axis=axis)

def sse(y, g=0.0, axis=None):
    return np.sum((y-g)**2, axis=axis)
    
def mae(y, g=0.0, axis=None):
    return np.mean(np.abs(y-g), axis=axis)

def mse(y, g=0.0, axis=None):
    return np.mean((y-g)**2, axis=axis)

def rmse(y, g=0.0, axis=None):
    return np.sqrt(np.mean((y-g)**2, axis=axis))

def nrmse(y, g=0.0, mn=None, mx=None, axis=None):
    r = y-g
    if mn is None:
        mn = np.min(r, axis=axis)
    if mx is None:
        mx = np.max(r, axis=axis)

    return rmse(r, axis=axis)/(mx-mn)

def gini(y, g, normalize=True):
    if y.ndim > 1 or g.ndim > 1:
        raise RuntimeError('Gini does not currently support more than one axis.')

    if normalize:
        return gini(y, g, normalize=False) / gini(g, g, normalize=False)

    gSort = g[np.argsort(-y)]

    gTotal = np.sum(gSort)
    gCum = np.cumsum(gSort)
    ng = len(gSort)
    ginis = (gCum / float(gTotal)) - ((np.arange(ng)+1.0) / float(ng))

    return np.sum(ginis)
