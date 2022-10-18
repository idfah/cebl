"""Independent components analysis.
"""

import matplotlib.pyplot as plt
import numpy as np

import scipy.signal as spsig
import scipy.stats as spstat

from cebl import util

from .strans import STrans


class IndependentComponentsAnalysis(STrans):
    """

    References:
        @article{bell1995information,
          title={An information-maximization approach to blind separation and blind deconvolution},
          author={Bell, Anthony J and Sejnowski, Terrence J},
          journal={Neural computation},
          volume={7},
          number={6},
          pages={1129--1159},
          year={1995},
          publisher={MIT Press}
        }

        @inproceedings{girolami1997generalised,
          title={Generalised independent component analysis through unsupervised learning with emergent bussgang properties},
          author={Girolami, Mark and Fyfe, C},
          booktitle={Neural Networks, 1997., International Conference on},
          volume={3},
          pages={1788--1791},
          year={1997},
          organization={IEEE}
        }

        @article{lee1999independent,
          title={Independent component analysis using an extended infomax algorithm for mixed subgaussian and supergaussian sources},
          author={Lee, Te-Won and Girolami, Mark and Sejnowski, Terrence J},
          journal={Neural computation},
          volume={11},
          number={2},
          pages={417--441},
          year={1999},
          publisher={MIT Press}
        }
    """
    def __init__(self, s, lags=0, kurtosis='adapt',
                 learningRate=1.5, tolerance=1.0e-6, maxIter=10000,
                 callback=None, verbose=False, *args, **kwargs):
        STrans.__init__(self, s, lags=lags, *args, **kwargs)

        self.train(s, kurtosis=kurtosis,
                   learningRate=learningRate,
                   tolerance=tolerance, maxIter=maxIter,
                   callback=callback, verbose=verbose)

    def train(self, s, kurtosis, learningRate, tolerance, maxIter, callback, verbose):
        s = self.prep(s)

        wPrev = np.empty(self.w.shape)
        grad = np.empty((self.nComp, self.nComp))

        I = np.eye(self.nComp, dtype=self.dtype)
        n = 1.0/s.shape[0]

        iteration = 0
        while True:
            y = s.dot(self.w)

            if kurtosis == 'sub':
                k = -1
            elif kurtosis == 'super':
                k = 1
            elif kurtosis == 'adapt':
                #k = np.sign(np.mean(1.0-util.tanh(y)**2, axis=0) *
                #            np.mean(y**2, axis=0) -
                #            np.mean(y*util.tanh(y), axis=0))

                k = np.sign(spstat.kurtosis(y, axis=0))
                k[np.isclose(k,0.0)] = -1.0

            grad[...] = (I - k*util.tanh(y).T.dot(y) - y.T.dot(y)).T.dot(self.w) * n

            wPrev[...] = self.w
            self.w += learningRate * grad

            wtol = np.max(np.abs(wPrev-self.w))

            if verbose:
                print('%d %6f' % (iteration, wtol))

            if callback is not None:
                callback(iteration, wtol)

            if wtol < tolerance:
                self.reason = 'tolerance'
                break

            elif np.max(np.abs(self.w)) > 1.0e100:
                self.reason = 'diverge'
                break

            if iteration >= maxIter:
                self.reason = 'maxiter'
                break

            iteration += 1

        if verbose:
            print('Reason: ' + self.reason)

        self.w /= np.sqrt(np.sum(self.w**2, axis=0))
        self.wInv[...] = np.linalg.pinv(self.w)

class ICA(IndependentComponentsAnalysis):
    pass

def demoICA():
    t = np.linspace(0.0, 30*np.pi, 1000)

    s1 = spsig.sawtooth(t)
    s2 = np.cos(5.0*t)
    s3 = np.random.uniform(-1.0, 1.0, size=t.size)
    s = np.vstack((s1,s2,s3)).T

    m = np.random.random((3,3))
    m /= m.sum(axis=0)

    sMixed = s.dot(m)

    icaFilt = ICA(sMixed, kurtosis='sub', verbose=True)

    fig = plt.figure()

    axOrig = fig.add_subplot(4,1, 1)
    axOrig.plot(s+util.colsep(s))
    axOrig.set_title('Unmixed Signal')
    axOrig.autoscale(tight=True)

    axMixed = fig.add_subplot(4,1, 2)
    axMixed.plot(sMixed+util.colsep(sMixed))
    axMixed.set_title('Mixed Signal (random transform)')
    axMixed.autoscale(tight=True)

    axUnmixed = fig.add_subplot(4,1, 3)
    icaFilt.plotTransform(sMixed, ax=axUnmixed)
    axUnmixed.set_title('ICA Components')
    axUnmixed.autoscale(tight=True)

    axCleaned = fig.add_subplot(4,1, 4)
    icaFilt.plotFilter(sMixed, comp=(0,1,), ax=axCleaned)
    axCleaned.set_title('Cleaned Signal (First two components kept)')
    axCleaned.autoscale(tight=True)

    fig.tight_layout()


if __name__ == '__main__':
    demoICA()
    plt.show()
