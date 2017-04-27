import numpy as np

def autoCorrelation(s):
    def ac1d(x):
        var = x.var()
        x = x - x.mean()
        r = np.correlate(x[-x.size:], x, mode='full')
        return r[r.size//2:] / (var * np.arange(x.shape[0], 0, -1))

    return np.apply_along_axis(ac1d, 0, s)
