"""Utilities for packing and unpacking matrices of
weights into flat vectors.
"""

import numpy as np


def pack(paramList):
    """Pack the values in a list of numpy arrays
    into a single 1d numpy array.

    Args:
        paramList:  A list of numpy arrays to pack.

    Returns:
        A 1d numpy array containing all the values
        of the arrays in paramList.

    Notes:
        See packedViews for a potentially more
        efficient strategy.
    """
    return np.concatenate([p.ravel() for p in paramList])

def unpack(paramArray, paramList):
    """Unpack a 1d numpy array into a list of numpy arrays.

    Args:
        paramArray: A 1d numpy array containing the values
                    to pack.  The size of this array must
                    be equal to the total combined number
                    of elements in the arrays in paramList.

        paramList:  A list of numpy arrays to receive the
                    values in paramArray.  The arrays in
                    this list are modified in place.

    Returns:
        A copy of paramList.  The arrays in paramList are
        modified in place to contain the values in paramArray.

    Notes:
        See packedViews for a potentially more
        efficient strategy.
    """
    start = 0
    for param in paramList:
        n = param.size
        end = start + n
        param.ravel()[:] = paramArray[start:end]
        start = end
    return paramList

def packedViews(shapes, dtype=np.float):
    """Return a list of numpy array views of the contiguous
    values in an empty 1d numpy array.

    Args:
        shapes: A list of tuples each of which specifies
                the shape of a desired numpy array view
                into a packed, 1d numpy array.

        dtype:  The data type of the array and views.
                Defaults to np.float.

    Returns:
        A list of numpy array views.  The first element of this
        list is an empty 1d numpy array with size equal to all
        the elements of the array sizes specified by the shapes
        argument.  The remaining elements are numpy array views
        into the contiguous values of the first argument with
        shapes specified by the shapes argument.

    Notes:
        This function allows one to maintain a packed version
        of multiple numpy arrays that is modified whenever
        the individual arrays are modified without the need for
        packing and unpacking the arrays.  Beware, however,
        that the arrays must be modified in place in order for
        this approach to work.

    Examples:
        >>> import numpy as np
        >>> from cebl import util

        >>> packed, view1, view2 = util.packedViews(((2,2), (2,3,4)))

        >>> view1[...] = np.arange(view1.size).reshape(view1.shape)
        >>> view1
        array([[ 0.,  1.],
               [ 2.,  3.]])

        >>> view2[...] = 7.0
        >>> view2
        array([[[ 7.,  7.,  7.,  7.],
                [ 7.,  7.,  7.,  7.],
                [ 7.,  7.,  7.,  7.]],

               [[ 7.,  7.,  7.,  7.],
                [ 7.,  7.,  7.,  7.],
                [ 7.,  7.,  7.,  7.]]])

        >>> packed
        array([ 0.,  1.,  2.,  3.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,
                7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,
                7.,  7.])
    """
    sizes = [np.product(sh) for sh in shapes]
    totalSize = np.sum(sizes)

    ends = np.cumsum(sizes)
    starts = np.roll(ends, 1)
    starts[0] = 0

    packed = np.empty(totalSize, dtype=dtype)
    views = [packed[st:en].reshape(sh)
             for st, en, sh in zip(starts, ends, shapes)]

    return [packed] + views
