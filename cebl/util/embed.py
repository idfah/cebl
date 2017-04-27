import numpy as np
import numpy.lib.stride_tricks as npst

from arr import colmat


def slidingWindow(s, span, stride=None, axis=0):
    """Sliding window.
    """
    #s = np.ascontiguousarray(s)
    s = np.require(s, requirements=['C', 'O'])

    if stride is None:
        stride = span

    # catch some bad values since this is a common place for
    # bugs to crop up in other routines
    if span > s.shape[axis]:
        raise ValueError('Span of %d exceeds input length of %d.' % (span, s.shape[axis]))

    if span < 0:
        raise ValueError('Negative span of %d is invalid.' % span)

    if stride < 1:
        raise ValueError('Stride of %d is not positive.' % stride)

    nWin = int(np.ceil((s.shape[axis]-span+1) / float(stride)))

    shape = list(s.shape)
    shape[axis] = span
    shape.insert(axis, nWin)

    strides = list(s.strides)
    strides.insert(axis, stride*strides[axis])

    return npst.as_strided(s, shape=shape, strides=strides)

def timeEmbed(s, lags=1, stride=1, axis=0):
    """Time-delay embedding.

    Notes:
        Only copies s if necessary.
    """
    #s = np.ascontiguousarray(s)
    s = np.require(s, requirements=['C', 'O'])

    if lags == 0:
        return s

    if s.ndim == 1:
        if axis == -1:
            axis = 1
        if axis == 1:
            s = s.reshape((1,-1))
        elif axis == 0:
            s = s.reshape((-1,1))

    emb = slidingWindow(s, span=lags+1, stride=stride, axis=axis)

    # wrap negative axes
    if axis < 0 and axis >= -s.ndim:
        axis = axis % s.ndim

    # if last axis then wrap back
    # double check that this works for all axes XXX - idfah
    if ((axis + 1) % s.ndim) == 0:
        emb = np.rollaxis(emb, axis+1, axis-1)
        shape = list(emb.shape)
        m = shape.pop(axis)
        shape[axis-1] *= m

        return emb.reshape(shape)

    else:
        shape = list(emb.shape)
        m = shape.pop(axis+1)
        shape[axis+1] *= m

        return emb.reshape(shape)
