import hashlib
import numpy as np


def accum(x, n, accumf=np.sum, truncate=True, axis=None):
    """Accumulate a number adjecent values in strides along a given axis.

    Args:
        x:          Matrix-like data to accumulate.

        n:          Number of adjacent values to accumulate,
                    i.e., stride width.

        accumf:     Function to use for accumulation.
                    numpy.sum is used by default.
                    Should accept axis argument.

        truncate:   If the number of values along the given
                    axis is not a multiple of n and truncate
                    is true (default), then truncate the
                    remaining values.  If truncate is false,
                    the matrix will be padded with zeros.

        axis:       The axis to accumulate along.  If
                    None (default) then the matrix will
                    be flattened and will accumulate along
                    remaining dimension.

    Returns:
        The accumulated matrix.

    Examples:
        >>> x = np.reshape((1,)*(4*3*3), (4, 3, 3))
        >>> x
        array([[[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]],

               [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]],

               [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]],

               [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]])
        >>> accum(x, 2, axis=None)
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        >>> accum(x, 2, axis=0)
        array([[[2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]],

               [[2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]]])
        >>> accum(x, 3, axis=1)
        array([[[3, 3, 3]],

               [[3, 3, 3]],

               [[3, 3, 3]],

               [[3, 3, 3]]])
    """
    if n < 1:
        raise ValueError('n must be >= 1')

    # make sure we have a numpy array
    x = np.asarray(x)

    if axis is None:
        # flatten if multi-dimensional
        if np.ndim(x) > 1:
            x = x.reshape((-1,))

        # axis is now zero
        axis = 0

    if n == 1:
        return x

    # length along axis to accumulate
    nObs = x.shape[axis]

    # if axis is not a multiple of n
    if (nObs%n) != 0:
        # truncate remaining elements
        if truncate:
            s = [slice(None),]*len(x.shape)
            s[axis] = slice(0, nObs-(nObs%n))
            x = x[s]

        # pad with zeros in case of one-dimensional array
        elif len(x.shape) == 1:
            x = np.append(x, np.zeros(n-(nObs%n), dtype=x.dtype))

        # pad with zeros in case of multi-dimensional array
        else:
            padShape = list(x.shape)
            padShape[axis] = n-(nObs%n)
            pad = np.zeros(padShape, dtype=x.dtype)
            x = np.concatenate((x, pad), axis=axis)

    # separate axis to accumulate
    sepShape = list(x.shape)
    sepShape.insert(axis+1, n)
    sepShape[axis] = x.shape[axis]//n
    #sepShape = x.shape[:axis] + (x.shape[axis]//n,) + \
    #    (n,) + x.shape[(axis+1):]
    sep = x.reshape(sepShape)

    # accumf into new shape
    return accumf(sep, axis=axis+1)
    #return np.apply_along_axis(func1d=accumf, axis=axis+1, arr=sep)

def bias(x, value=1, axis=-1):
    """Add a bias value to the given axis.

    Args:
        x:      Matrix-like data.

        value:  Bias value to append.
                Defaults to 1.

        axis:   Axis to which value is appended.
                Defaults to last axis.

    Returns:
        A new matrix with value appended.

    Examples:
        >>> import numpy as np
        >>> from cebl import util

        >>> a = np.arange(10)
        >>> util.bias(a, axis=0)
        array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
        >>> util.bias(a, axis=1)
        array([[ 0.,  1.],
               [ 1.,  1.],
               [ 2.,  1.],
               [ 3.,  1.],
               [ 4.,  1.],
               [ 5.,  1.],
               [ 6.,  1.],
               [ 7.,  1.],
               [ 8.,  1.],
               [ 9.,  1.]])
        >>> util.bias(a, axis=None)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1])

        >>> a = np.random.random((3, 2))
        >>> util.bias(a, axis=0)
        array([[ 0.5734496 ,  0.41789283],
               [ 0.15415034,  0.99381062],
               [ 0.80518692,  0.86804327],
               [ 1.        ,  1.        ]])
        >>> util.bias(a, axis=1)
        array([[ 0.5734496 ,  0.41789283,  1.        ],
               [ 0.15415034,  0.99381062,  1.        ],
               [ 0.80518692,  0.86804327,  1.        ]])
        >>> util.bias(a, value=3)
        array([[ 0.5734496 ,  0.41789283,  3.        ],
               [ 0.15415034,  0.99381062,  3.        ],
               [ 0.80518692,  0.86804327,  3.        ]])

    """
    x = np.asarray(x)
    dtype = np.result_type(x, value)

    if axis is None:
        return np.append(x, value)

    if x.ndim == 1:
        if axis == -1:
            axis = 1
        if axis == 1:
            x = x[:,None]
        elif axis == 0:
            x = x[None,:]

    if axis == -1:
        axis = x.ndim-1

    xbShape = list(x.shape)
    xbShape[axis] += 1

    xb = np.empty(xbShape, dtype=dtype)

    xSlices = [slice(None) if i != axis else slice(None, -1) for i in range(xb.ndim)]
    bSlices = [slice(-1, None) if i == axis else slice(None) for i in range(xb.ndim)]

    xb[xSlices] = x
    xb[bSlices] = value

    return xb

def capInf(x, copy=False):
    x = np.array(x, copy=copy)

    mn = np.finfo(x.dtype).min
    mx = np.finfo(x.dtype).max

    if x.ndim == 0:
        if x < mn:
            x[...] = mn
        if x > mx:
            x[...] = mx
    else:
        x[x < mn] = mn
        x[x > mx] = mx

    return x

def capZero(x, copy=False):
    """
    Notes:  If copy is False and x is a numpy array,
            then x is modified in place.
    """
    x = np.array(x, copy=copy)

    tiny = np.finfo(x.dtype).tiny

    if x.ndim == 0:
        if x < tiny:
            x[...] = tiny
    else:
        x[x < tiny] = tiny

    return x

def colmat(x, dtype=None, copy=False):
    x = np.array(x, copy=copy)

    if dtype is not None:
        x = x.astype(dtype, copy=False)

    if x.ndim == 1 and x.size > 0:
        x = x.reshape((x.shape[0], -1))

    return x

def colsep(x, scale=None, returnScale=False):
    x = colmat(x)

    if scale is None or np.isclose(scale, 0.0):
        if np.isclose(np.min(x), np.max(x)):
            scale = 1.0
        else:
            scale = np.max(np.abs(x))

    sep = -np.arange(x.shape[1], dtype=x.dtype)*2.0*scale

    if returnScale:
        return sep, scale
    else:
        return sep

def hashArray(x):
    return hashlib.sha1(np.ascontiguousarray(x).view(np.uint8)).hexdigest()

def nearZero(x):
    x = np.asarray(x)
    tiny = np.finfo(x.dtype).tiny
    return np.abs(tiny) <= tiny

def nextpow2(x):
    """Find the first integer 2**n greater than x.
    """
    return int(2**np.ceil(np.log2(x)))

def punion(probs, axis=None):
    """Find the unions of given list of probabilities assuming indepdendence.

    Args:
        probs:  Matrix-like probabilities to union.

        axis:   Axis along which union will be performed.

    Returns:
        Matrix of probability unions.
    """
    def punion1d(probs):
        """Union for 1d array.
        """
        finalp = 0.0
        for p in probs:
            finalp += p*(1.0-finalp)
        return finalp

    probs = np.asarray(probs)

    if axis is None:
        return punion1d(probs.reshape((-1,)))
    else:
        return np.apply_along_axis(func1d=punion1d, axis=axis, arr=probs)

def segmat(xs, dtype=None, copy=False):
    xs = np.array(xs, copy=copy)

    if dtype is not None:
        xs = xs.astype(dtype, copy=False)

    if xs.ndim == 1:
        return xs.reshape((1, xs.shape[0], 1))

    if xs.ndim == 2:
        return xs.reshape((xs.shape[0], xs.shape[1], -1))

    return xs

def segdot(x1, x2):
    assert x1.ndim == 3
    assert x2.ndim == 2

    return x1.reshape((-1, x1.shape[-1])).dot(x2).reshape((x1.shape[0], -1, x2.shape[-1]))

def softmaxM1(x):
    mx = np.max((np.max(x), 0.0))
    emx = capZero(np.exp(-mx))
    terms = capZero(np.exp(x-mx))
    denom = (emx + np.sum(terms, axis=1)).reshape((-1, 1))
    return np.hstack((terms/denom, emx/denom))
    
def logSoftmaxM1(x):
    mx = np.max((np.max(x), 0.0))
    xmx = x - mx
    emx = capZero(np.exp(-mx))

    terms = capZero(np.exp(xmx))
    denom = (emx + np.sum(terms, axis=1)).reshape((-1, 1))

    logDenom = np.log(capZero(denom))

    return np.hstack((xmx-logDenom, -mx-logDenom))

def softmax(x):
    mx = np.max((np.max(x), 0.0))
    terms = capZero(np.exp(x-mx))
    denom = np.sum(terms, axis=1).reshape((-1, 1))
    return terms/denom

def logSoftmax(x):
    mx = np.max((np.max(x), 0.0))
    xmx = x - mx

    terms = capZero(np.exp(xmx))
    denom = np.sum(terms, axis=1).reshape((-1, 1))

    logDenom = np.log(capZero(denom))

    return xmx - logDenom
