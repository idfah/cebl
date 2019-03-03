"""Statistics utilities.
"""

import numpy as np
import scipy.stats as spstats


def tconf(x, width=0.95, axis=None):
    """Compute confidence intervals of the student's t distribution.

    Args:
        x:      A list or numpy array of floats of
                observed values.

        width:  The width of the confidence interval.

        axis:   The axis along which to compute the
                confidence intervales.

    Returns:
        If axis is None, a tuple containing the (lower, upper)
        confidence intervals.  If axis is not None, then a matrix
        containing the pairs of lower and upper confidence intervals
        is returned.

    Examples:
        > x = np.random.normal(size=1000, loc=2.2, scale=2.0)

        > util.conf(x)
        (2.1420429487795936, 2.3901103741951317)

        > util.conf(x, width=0.9, axis=1)
        array([[ 1.7757,  2.8146],
               [ 1.3863,  2.4277],
               [ 1.7898,  2.8078],
               [ 1.7131,  2.768 ],
               [ 1.5982,  2.6495],
               [ 1.6322,  2.6527],
               [ 1.0066,  2.0738],
               [ 1.6856,  2.726 ],
               [ 1.3983,  2.4251],
               [ 1.5353,  2.5691]])

    """
    x = np.array(x, copy=False)

    def conf1(v):
        return spstats.t.interval(
            width, len(v)-1, loc=np.mean(v), scale=spstats.sem(v))

    return conf1(x.ravel()) if axis is None else np.apply_along_axis(conf1, axis, x)
