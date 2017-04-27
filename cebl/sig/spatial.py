import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

from cebl import util

import windows


def commonAverageReference(s, m1=False):
    s = util.colmat(s)

    if m1:
        nDim = s.shape[1]
        colSum = s.sum(axis=1)
        s -= np.hstack([(colSum-s[:,i])[:,None] for i in xrange(nDim)])/(nDim-1)
        return s
    else:
        s -= s.mean(axis=1)[:,None]
        return s

def car(s, *args, **kwargs):
    """Alias for commonAverageReference
    """
    return commonAverageReference(s, *args, **kwargs)

def meanSeparate(s, recover=False):
    s = util.colmat(s)

    if recover:
        mean = s[:,-1].copy()
        s[:,-1] = -s[:,:-1].sum(axis=1)

        return s + mean[:,None]

    else:
        mean = s.mean(axis=1)
        s -= mean[:,None]
        s[:,-1] = mean
        return s

def sharpenOld(s, kernelFunc, dist=None, scale=None,
            normalize=False, m1=False, *args, **kwargs):
    s = util.colmat(s)

    if dist is None:
        dist = np.arange(s.shape[1])+1.0
        dist = np.abs(dist[None,:]-dist[:,None])

        #dist = np.insert(spsig.triang(s.shape[1]-1, sym=False), 0, 0.0)
        #dist = np.vstack([np.roll(dist, i) for i in xrange(dist.size)])

    if scale is None:
        # minimum off-diagonal distance
        scale = np.min(dist[np.asarray(1.0-np.eye(dist.shape[0]), dtype=np.bool)])

    kernel = kernelFunc(dist.T/scale, *args, **kwargs)

    if m1:
        np.fill_diagonal(kernel, 0.0)

    if normalize:
        kernel = kernel/np.abs(kernel.sum(axis=0))

    return s - s.dot(kernel)

def sharpen(s, radius=0.3, mix=1.0, dist=None):
    s = util.colmat(s)

    if dist is None:
        dist = np.arange(s.shape[1])+1.0
        dist = np.abs(dist[None,:]-dist[:,None])
        #dist = np.insert(spsig.triang(s.shape[1]-1, sym=False), 0, 0.0)
        #dist = np.vstack([np.roll(dist, i) for i in xrange(dist.size)])

    kernel = util.gaussian(dist.T, radius=radius)
    kernel /= kernel.sum(axis=0)

    return (1.0-mix)*s + mix*(s - s.dot(kernel))

def demoSharpen():
    t = np.linspace(-10*np.pi, 10*np.pi, 200)
    s1 = np.sin(t)
    s2 = np.sin(t+np.pi)
    s3 = np.sin(2*t)
    ss = np.sin(0.2*t)
    #s = (np.vstack((s3,s1+ss,s1+ss,s1+ss,s2+ss,s2+ss,s2,s2,s2,s3,s3,s3))).T
    s = (np.vstack((s1+ss,s1+ss,s1+ss,s2+ss,s2+ss,s2,s2,s2,s3,s3,s3))).T
    colors = ['blue',]*3 + ['orange',]*2 + ['green',]*3 + ['red',]*3

    sepDist = 3.0
    sep = -np.arange(s.shape[1])*sepDist

    fig = plt.figure(figsize=(18,10))

    xlim = (-10*np.pi, 10*np.pi)
    ylim = (-s.shape[1]*sepDist, sepDist)

    axSig = fig.add_subplot(1,3, 1)
    axSig.set_color_cycle(colors)
    axSig.plot(t, s+sep)
    axSig.set_xlim(xlim)
    axSig.set_ylim(ylim)
    axSig.set_title('Orignial')

    axCar = fig.add_subplot(1,3, 2)
    axCar.set_color_cycle(colors)
    axCar.plot(t, commonAverageReference(s)+sep)
    axCar.set_xlim(xlim)
    axCar.set_ylim(ylim)
    axCar.set_title('Common Average Reference')

    axGaus = fig.add_subplot(1,3, 3)
    axGaus.set_color_cycle(colors)
    #axGaus.plot(t, sharpen(s, kernelFunc=windows.unitGaussian, radius=0.5, mix=0.5)+sep)
    axGaus.plot(t, sharpen(s, radius=1.0)+sep)
    axGaus.set_xlim(xlim)
    axGaus.set_ylim(ylim)
    axGaus.set_title('Gaussian Sharpen')

    #axLanc = fig.add_subplot(1,3, 3)
    #axLanc.plot(t, sharpen(s, kernelFunc=util.lanczos, freq=0.99, radius=3.0, normalize=False)+sep)
    #axLanc.set_xlim(xlim)
    #axLanc.set_ylim(ylim)
    #axLanc.set_title('Lanczos Sharpen')

    fig.tight_layout()


if __name__ == '__main__':
    demoSharpen()
    plt.show()
