"""Miscellaneous yet useful functions.
"""

import numpy as np


def gaussian(x, radius=2):
    """Simple gaussian function.

    Args:
        x:      An array of input values.
        radius: Width of the gaussian.

    Returns:
        exp(-0.5 * (x / radius)**2)
    """
    return np.exp(-0.5*(x/radius)**2)

def lanczos(x, freq, radius=3):
    """Lanczos function.
    """
    l = 0.5*freq * np.sinc(0.5*freq*x) * np.sinc(x/radius)
    l[(x < -radius) | (x > radius)] = 0.0
    return l
