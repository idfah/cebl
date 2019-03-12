"""Parameter and weight initializers.
"""
import numpy as np


def lecun(size):
    """Method recommended by LeCun in efficient backprop paper.
    """
    if isinstance(size, (int,)):
        size = (size,)

    return np.random.uniform(-np.sqrt(3.0 / size[0]),
        np.sqrt(3.0 / size[0]), size=size)

def nguyen(size, scale=(-1.0, 1.0), overlap=0.3):
    """
    fanIn = size[0]
    numNeurons = size[1]

    sep = 1.0 - overlap

    weights = np.random.uniform(scale[0], scale[1], (fanIn, numNeurons))

    beta = sep * numNeurons**(1.0/fanIn)
    l2 = np.sqrt(np.sum(weights**2))

    weights[:-1,:] *= beta/l2

    return weights
    # XXX - idfah
    """
    raise NotImplementedError("nguyen not yet implemented")

def runif(size, low=-0.01, high=0.01):
    """Random uniform.
    """
    return np.random.uniform(low, high, size)

def rnorm(size, scale=0.01):
    """Random normal.
    """
    return np.random.normal(0.0, scale, size)

def esp(size, specRadius=0.7):
    """Scale by the spectral radius (largest eigenvalue) to achieve
    the Echo State Property (ESP).
    """
    a = np.random.uniform(-1.0, 1.0, size=size)

    n = a.shape[1]
    b = a[-n:,:]

    # this is a very slow way to compute the spectral radius - XXX idfah
    d = np.linalg.eigvals(b)

    a *= specRadius / np.max(np.abs(d))
    return a
