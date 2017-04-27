import numpy as np


def blockShuffle(x, n, axis=None):
    """Shuffle along a given axis but leave
    blocks of size n next to each other.

    Args:
        x:      Matrix-like data to shuffle.

        n:      Size of blocks to keep adjacent.

        axis:   Axis along which shuffle will be done.

    Returns:
        Matrix of data, shuffled in place.

    Notes:
        This probably needs checked XXX - idfah.
    """
    x = np.asarray(x)

    if axis is None:
        if (x.size % n) != 0:
            raise Exception('x.size = %d is not a multiple of n = %d.' % (x.size, n))

        shp = x.shape

        indx = np.reshape(range(x.size), (x.size//n,n))
        np.random.shuffle(indx)
        indx = indx.reshape((-1,))

        return np.reshape(x.reshape((-1,))[indx], shp)

    else:
        if (x.shape[axis] % n) != 0:
            raise Exception('x.shape[%d] = %d is not a multiple of n = %d.' % \
                (axis, x.shape[axis], n))

        lax = x.shape[axis]
        indx = np.reshape(range(lax), (lax//n,n))
        np.random.shuffle(indx)
        indx = indx.reshape((-1,))

        s = [slice(None),]*len(x.shape)
        s[axis] = indx

        return x[s]

def cycle(x, n):
    """Cycle through the values of a list until it has length n.

    Args:
        x:      List to cycle through.

        n:      New length of list.

    Returns:
        New list with length n.

    Examples:
        >>> import util

        >>> x = range(4)

        >>> util.cycle(x, 3)
        [0, 1, 2]

        >>> util.cycle(x, 6)
        [0, 1, 2, 3, 0, 1]
    """
    l = int(np.ceil(n/float(len(x))))
    return (x*l)[0:n]

