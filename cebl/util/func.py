import numpy as np


def gaussian(x, radius=2):
    return np.exp(-0.5*(x/radius)**2)

def lanczos(x, freq, radius=3):
    l = 0.5*freq * np.sinc(0.5*freq*x) * np.sinc(x/radius)
    l[(x < -radius) | (x > radius)] = 0.0
    return l

