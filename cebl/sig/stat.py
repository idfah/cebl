"""Signal statistical measures.
"""
import numpy as np


def autoCorrelation(s):
    """Compute the autocorrelations of s.

    Args:
        s:  A numpy array or list of floating point
            values representing the signal.

    Returns:
        A numpy array where the i'th values contains
        the autocorrelation of the signal at i time lags.
    """
    s = np.array(s, copy=False)

    def ac1d(x):
        var = x.var()
        x = x - x.mean()
        r = np.correlate(x[-x.size:], x, mode="full")
        return r[r.size//2:] / (var * np.arange(x.shape[0], 0, -1))

    return np.apply_along_axis(ac1d, 0, s)
