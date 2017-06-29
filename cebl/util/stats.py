import numpy as np
import scipy.stats as spstats


def conf(x, width=0.95, axis=None):
    def conf1(v):
        return spstats.t.interval(width, len(v)-1,
                    loc=np.mean(v), scale=spstats.sem(v))

    return conf1(x.ravel()) if axis is None else np.apply_along_axis(conf1, axis, x)
