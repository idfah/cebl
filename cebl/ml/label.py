import numpy as np

from cebl import util


def vectorFromList(classData, combined=False):
    x = np.vstack([util.colmat(cls) for cls in classData])
    vector = np.repeat(np.arange(len(classData), dtype=x.dtype),
        [np.asarray(cls).shape[0] for cls in classData])

    if combined:
        return np.hstack((x,vector[:,None]))
    else:
        return x, vector

def vectorFromIndicators(indicators):
    return np.array([np.argmax(row) for row in indicators])

def listFromVector(x, vector, nCls=None):
    x = np.asarray(x)
    if nCls is None:
        nCls = np.max(vector)
    labels = np.arange(nCls+1, dtype=x.dtype)
    return [x[np.where(vector == l)] for l in labels]

def listFromIndicators(x, indicators):
    vector = vectorFromIndicators(indicators)
    return listFromVector(x, vector)

def indicatorsFromVector(vector, nCls=None, conf=1.0):
    dtype = np.result_type(vector.dtype, np.float32)

    if nCls is None:
        nCls = np.max(vector)+1

    labels = np.arange(nCls, dtype=dtype)
    indicators = np.ones((len(vector), len(labels)), dtype=dtype)
    indicators = ((indicators*vector[:,None]) == (indicators*labels))

    offset = (1.0 - conf) / (nCls-1)
    indicators = indicators * (conf-offset) + offset

    return indicators.astype(dtype, copy=False)

def indicatorsFromList(classData, conf=1.0, combined=False):
    x = np.vstack([util.colmat(cls) for cls in classData])
    vector = np.repeat(np.arange(len(classData), dtype=x.dtype),
        [np.asarray(cls).shape[0] for cls in classData])

    nCls = len(classData)
    labels = np.arange(nCls, dtype=x.dtype)
    indicators = np.ones((len(vector), nCls), dtype=x.dtype)
    indicators = ((indicators*vector[:,None]) == (indicators*labels))

    offset = (1.0 - conf) / (nCls-1)
    indicators = indicators * (conf-offset) + offset
    indicators = indicators.astype(x.dtype, copy=False)

    if combined:
        return np.hstack((x, indicators))
    else:
        return x, indicators

def makeVector(col, returnKeys=False):
    keys = np.unique(col)
    table = {k:i for i,k in enumerate(keys)}
    vector = np.array([table[c] for c in col])

    if returnKeys:
        return keys, vector
    else:
        return vector

def makeIndicators(col, returnKeys=False, *args, **kwargs):
    keys, vector = makeVector(col, returnKeys=True)
    indicators = indicatorsFromVector(vector, *args, **kwargs)

    if returnKeys:
        return keys, indicators
    else:
        return indicators
