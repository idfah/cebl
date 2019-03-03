"""Window functions.

Note:
    Includes all window functions from scipy.signal.windows
"""
import numpy as np

# include all window functions in scipy.signal.windows
# pylint: disable=unused-wildcard-import,wildcard-import
from scipy.signal.windows import *


def kroneckerDelta(n):
    """Kronecker delta window.

    Args:
        n:  The width of the window.

    Returns:
        A numpy array of width n containing
        the values of the window.
    """
    taps = np.zeros(n)
    taps[n//2] = 1.0
    return taps

def lanczos(n, radius=3):
    """Lanczos window.

    Args:
        n:          The width of the window.

        radius:     Radius of the Lanczos function,
                    i.e., the width of the central
                    lobe of the sinc function.

    Returns:
        A numpy array of width n containing
        the values of the window.
    """
    taps = np.linspace(-radius, radius, n)
    return np.sinc(taps/radius)

def ramp(n, corner1=None, corner2=None):
    """Ramp window.  Linearly increases until
    corner1, then flat until corner2, then linearly
    decreases until the end of the window.

    Args:
        n:          The width of the window.

        corner1:    Number of time steps after which
                    the window levels off.  If None
                    (default) then will be floor(n/3).

        corner2:    Number of time steps after which
                    the window begins to decrease.
                    If None (default) then will be
                    ceil(n*2/3).

    Note:
        Raises a RuntimeError if corner1 > corner2.

    Returns:
        A numpy array of width n containing
        the values of the window.
    """
    if n < 3:
        return np.zeros(n)

    window = np.arange(n)*1.0

    if corner1 is None:
        corner1 = int(np.floor(n/3.0))

    if corner2 is None:
        corner2 = int(np.ceil(n*2/3.0))

    if corner1 > corner2:
        raise RuntimeError("Invalid ramp window corners.")

    up = window[:corner1]
    top = window[corner1:corner2]
    down = window[corner2:]

    up[:] = up[:] / len(up[:])
    top[:] = 1.0
    down[:] = -(down[:] - down[-1]) / len(down[:])

    return window

def sinc(n, radius=3, freq=1.0):
    """Sinc window.

    Args:
        n:  The width of the window.

    Returns:
        A numpy array of width n containing
        the values of the window.
    """
    taps = np.linspace(-radius, radius, n)
    return freq * np.sinc(freq*taps)
