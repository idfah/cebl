import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

from cebl import util

import windows


def movingAverage(s, width=2, kernelFunc=windows.boxcar, *args, **kwargs):
    s = util.colmat(s)

    kernel = kernelFunc(width, *args, **kwargs)
    kernel /= kernel.sum()
    kernel = kernel.astype(s.dtype, copy=False)

    return np.apply_along_axis(
        np.convolve, axis=0, arr=s,
        v=kernel, mode='same')

def savitzkyGolay(s, *args, **kwargs):
    return spsig.savgol_filter(s, *args, axis=0, **kwargs)

def wiener(s, size=None, noise=None):
    # have to add astype because spsig.wiener does not preserve dtype of argument, bug that should be reported XXX - idfah
    return np.apply_along_axis(
        spsig.wiener, axis=0, arr=s,
        mysize=size, noise=noise).astype(s.dtype, copy=False)


def demoSmooth():
    n = 1024
    x = np.linspace(-10.0*np.pi, 10.0*np.pi, n)

    y1 = np.cos(x)
    y2 = np.cos(x) + np.random.normal(scale=0.2, size=x.shape)
    y3 = np.cumsum(np.random.normal(scale=0.1, size=x.shape))

    y = np.vstack((y1,y2,y3)).T

    maWidth = 9
    yMA = movingAverage(y, maWidth)

    gaWidth = 9
    yGA = movingAverage(y, gaWidth, kernelFunc=windows.gaussian, std=1.5)

    sgWidth = 21 
    sgOrder = 3
    ySG = savitzkyGolay(y, sgWidth, sgOrder)

    wnSize = 9
    wnNoise = 0.5
    yWN = wiener(y, wnSize, wnNoise)

    sep = -np.arange(0,3)[None,:]*2.0

    fig = plt.figure()

    axMA = fig.add_subplot(2,2, 1)
    axMA.plot(x, y+sep, color='grey', linewidth=3)
    axMA.plot(x, yMA+sep)
    axMA.set_title('Moving Average %d' % maWidth)
    axMA.autoscale(tight=True)

    axGA = fig.add_subplot(2,2, 2)
    axGA.plot(x, y+sep, color='grey', linewidth=3)
    axGA.plot(x, yGA+sep)
    axGA.set_title('Gaussian Moving Average %d' % gaWidth)
    axGA.autoscale(tight=True)

    axSG = fig.add_subplot(2,2, 3)
    axSG.plot(x, y+sep, color='grey', linewidth=3)
    axSG.plot(x, ySG+sep)
    axSG.set_title('Savitzky-Golay %d %d' % (sgWidth, sgOrder))
    axSG.autoscale(tight=True)

    axWN = fig.add_subplot(2,2, 4)
    axWN.plot(x, y+sep, color='grey', linewidth=3)
    axWN.plot(x, yWN+sep)
    axWN.set_title('Wiener %d %3.2f' % (wnSize, wnNoise))
    #axWN.set_title('Wiener')
    axWN.autoscale(tight=True)

    fig.tight_layout()


if __name__ == '__main__':
    demoSmooth()
    plt.show()
