"""Principal Components Analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

from cebl import util

from .strans import STrans


class PrincipalComponentsAnalysis(STrans):
    def __init__(self, s, *args, **kwargs):
        STrans.__init__(self, s, *args, **kwargs)
        self.train(s)

    def train(self, s):
        s = self.prep(s)

        if s.shape[0] >= s.shape[1]:
            u, d, v = np.linalg.svd(s, full_matrices=False)

        else: # if dims outnumber obs then we have exactly zero components
            u, dSub, v = np.linalg.svd(s, full_matrices=True)
            d = np.zeros(s.shape[1], dtype=s.dtype)
            d[:s.shape[0]] = dSub

        self.mags = d**2 / (s.shape[0]-1)
        #self.mags /= np.sum(self.mags)

        self.w[...] = v.T * (1.0/d[:,None])
        self.wInv[...] = v * d

    def getMags(self):
        return self.mags

    def plotMags(self, standardize=True, ax=None, **kwargs):
        result = {}
        if ax is None:
            fig = plt.figure()
            result['fig'] = fig
            ax = fig.add_subplot(1, 1, 1)
        result['ax'] = ax

        mags = self.mags / np.sum(self.mags) if standardize else self.mags

        sep = np.arange(len(mags))
        bars = plt.bar(sep, mags, **kwargs)
        result['bars'] = bars

        return result

class PCA(PrincipalComponentsAnalysis):
    pass

def demoPCA():
    n = 1000
    t = np.linspace(0.0, 30*np.pi, n)

    s1 = spsig.sawtooth(t)
    s2 = np.cos(0.5*t)
    #s3 = np.random.normal(scale=1.2, size=t.size)
    s3 = np.random.uniform(-2.0, 2.0, size=t.size)
    s = np.vstack((s1, s2, s3)).T

    theta1 = np.pi/6.0
    rot1 = np.array([[np.cos(theta1), -np.sin(theta1), 0.0],
                     [np.sin(theta1),  np.cos(theta1), 0.0],
                     [0.0,            0.0,           1.0]])

    theta2 = np.pi/4.0
    rot2 = np.array([[ np.cos(theta2), 0.0, np.sin(theta2)],
                     [ 0.0,            1.0, 0.0],
                     [-np.sin(theta2), 0.0, np.cos(theta2)]])

    theta3 = np.pi/5.0
    rot3 = np.array([[1.0, 0.0,             0.0],
                     [0.0, np.cos(theta3), -np.sin(theta3)],
                     [0.0, np.sin(theta3),  np.cos(theta3)]])

    sMixed = s.dot(rot1).dot(rot2).dot(rot3)

    lags = 0
    pcaFilt = PCA(sMixed, lags=lags)

    ##pcaFilt.plotMags()

    fig = plt.figure()

    axOrig = fig.add_subplot(4, 1, 1)
    axOrig.plot(s+util.colsep(s))
    axOrig.set_title('Unmixed Signal')
    axOrig.autoscale(tight=True)

    axMixed = fig.add_subplot(4, 1, 2)
    axMixed.plot(sMixed+util.colsep(sMixed))
    axMixed.set_title('Mixed Signal (3d rotation)')
    axMixed.autoscale(tight=True)

    axUnmixed = fig.add_subplot(4, 1, 3)
    pcaFilt.plotTransform(sMixed, ax=axUnmixed)
    axUnmixed.set_title('PCA Components')
    axUnmixed.autoscale(tight=True)

    axCleaned = fig.add_subplot(4, 1, 4)
    pcaFilt.plotFilter(sMixed, comp=(1, 2,), ax=axCleaned)
    axCleaned.set_title('Cleaned Signal (First Component Removed)')
    axCleaned.autoscale(tight=True)

    fig.tight_layout()

def demoPCA2d():
    n = 1000
    theta = np.pi/6.0
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])

    s1 = np.vstack((np.random.normal(scale=3, size=n),
                    np.random.normal(scale=0.3, size=n))).T
    s2 = s1.dot(rot)
    s = np.vstack((s1, s2))

    pca = PCA(s)
    y = pca.transform(s)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(s[:,0], s[:,1])
    ax.arrow(0.0, 0.0, pca.wInv[0,0]/pca.mags[0], pca.wInv[0,1]/pca.mags[0],
             head_width=0.05, head_length=0.1, color='red')
    ax.arrow(0.0, 0.0, pca.wInv[1,0]/pca.mags[1], pca.wInv[1,1]/pca.mags[1],
             head_width=0.05, head_length=0.1, color='red')
    ax.grid()

if __name__ == '__main__':
    demoPCA()
    #demoPCA2d()
    plt.show()
