import numpy as np
import scipy.special as sps

from cebl import util


def _twist(x, f, prime, twist=1.0e-3, **kwargs):
    if prime == 0:
        return f(x, prime=0, **kwargs) + twist * x

    elif prime == 1:
        return f(x, prime=1, **kwargs) + twist

    else:
        return f(x, prime=prime, **kwargs)

def linear(x, prime=0):
    if prime == 0:
        return x

    if prime == 1:
        return np.ones_like(x)

    else:
        return np.zeros_like(x)

def tanh(x, prime=0):
    if prime == 0:
        return util.tanh(x)

    elif prime == 1:
        return 1.0 - util.tanh(x)**2

    else:
        raise NotImplementedError("%d order derivative not implemented." % int(prime))

def tanhTwist(x, prime=0, **kwargs):
    return _twist(x, tanh, prime, **kwargs)

def lecun(x, prime=0):
    if prime == 0:
        return 1.7159 * util.tanh((2.0/3.0) * x)

    elif prime == 1:
        return 1.7159 * (2.0/3.0) * (1.0 - util.tanh((2.0/3.0) * x)**2)

    else:
        raise NotImplementedError("%d order derivative not implemented." % int(prime))

def lecunTwist(x, prime=0, **kwargs):
    return _twist(x, lecun, prime, **kwargs)

def logistic(x, prime=0):
    if prime == 0:
        ##v = np.empty_like(x)
        ##mask = x < 0.0

        ##zl = np.exp(x[mask])
        ##zl = 1.0 / (1.0 + zl)
        ##v[mask] = zl

        ##zh = np.exp(-x[~mask])
        ##zh = zh / (1.0 + zh)
        ##v[~mask] = zh

        v = sps.expit(x)

        return v

    elif prime == 1:
        return logistic(x) * (1.0 - logistic(x))

    else:
        raise NotImplementedError("%d order derivative not implemented." % int(prime))

def logisticTwist(x, prime=0, **kwargs):
    return _twist(x, logistic, prime, **kwargs)

def gaussian(x, prime=0):
    if prime == 0:
        return np.exp(-x**2)

    elif prime == 1:
        return -2.0*x*np.exp(-x**2)

    else:
        raise NotImplementedError("%d order derivative not implemented." % int(prime))

def gaussianTwist(x, prime=0, **kwargs):
    return _twist(x, logistic, prime, **kwargs)

def rectifier(x, prime=0):
    #v = np.zeros_like(x)
    #mask = x >= 0.0

    #if prime == 0:
    #    v[mask] = x[mask]

    #elif prime == 1:
    #    v[mask] = 1.0

    mask = x < 0.0

    if prime == 0:
        v = x.copy()
        v[mask] = 0.0

    elif prime == 1:
        v = np.ones_like(x)
        v[mask] = 0.0

    return v

def rectifierTwist(x, prime=0, **kwargs):
    return _twist(x, rectifier, prime, **kwargs)

def softplus(x, prime=0):
    if prime == 0:
        return np.log(1.0 + np.exp(x))

    elif prime == 1:
        return logistic(x)

    else:
        raise NotImplementedError("%d order derivative not implemented." % int(prime))

def softplusTwist(x, prime=0, **kwargs):
    return _twist(x, softplus, prime, **kwargs)

def exprect(x, prime=0):
    #v = np.empty_like(x)
    #mask = x >= 0.0
    #nmask = ~mask

    #if prime == 0:
    #    v[mask] = x[mask]
    #    v[nmask] = np.exp(x[nmask]) - 1.0

    #elif prime == 1:
    #    v[mask] = 1.0
    #    v[nmask] = np.exp(x[nmask])

    mask = x < 0.0

    if prime == 0:
        v = x.copy()
        v[mask] = np.exp(v[mask]) - 1.0

    elif prime == 1:
        v = np.ones_like(x)
        v[mask] = np.exp(v[mask])

    return v

def exprectTwist(x, prime=0, **kwargs):
    return _twist(x, exprect, prime, **kwargs)
