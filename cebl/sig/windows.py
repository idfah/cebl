import numpy as np

# include all window functions in scipy.signal.windows
# pylint: disable=unused-wildcard-import
from scipy.signal.windows import *


def kroneckerDelta(n):
    taps = np.zeros(n)
    taps[n//2] = 1.0
    return taps

def lanczos(n, radius=3):
    taps = np.linspace(-radius, radius, n)
    return np.sinc(taps/radius)

def ramp(n):
    if n < 3:
        return np.zeros(n)

    window = np.arange(n)*1.0

    corner1 = int(np.floor(n/3.0))
    corner2 = int(np.ceil(n*2/3.0))

    up = window[:corner1]
    top = window[corner1:corner2]
    down = window[corner2:]

    up[:] = up[:] / len(up[:])
    top[:] = 1.0
    down[:] = -(down[:] - down[-1]) / len(down[:])

    return window

def sinc(n, radius=3, freq=1.0):
    taps = np.linspace(-radius, radius, n)
    return freq * np.sinc(freq*taps)
