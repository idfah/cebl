"""Maximum Signal Fraction Analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

from cebl import util

from .strans import STrans


class MaximumSignalFraction(STrans):
    def __init__(self, s, *args, **kwargs):
        """
        Refs:
            @phdthesis{knight2003signal,
              title={Signal fraction analysis and artifact removal in EEG},
              author={Knight, James N},
              year={2003},
              school={Colorado State University}
            }

            @article{anderson2006geometric,
              title={Geometric subspace methods and time-delay embedding for EEG artifact removal and classification},
              author={Anderson, Charles W and Knight, James N and O'Connor, Tim and Kirby, Michael J and Sokolov, Artem},
              journal={Neural Systems and Rehabilitation Engineering, IEEE Transactions on},
              volume={14},
              number={2},
              pages={142--146},
              year={2006},
              publisher={IEEE}
            }

            @inproceedings{hundley2001solution,
              title={A solution procedure for blind signal separation using the maximum noise fraction approach: algorithms and examples},
              author={Hundley, D and Kirby, M and Anderle, Markus},
              booktitle={Proceedings of the Conference on Independent Component Analysis, San Diego, CA},
              pages={337--342},
              year={2001}
            }
        """
        STrans.__init__(self, s, *args, **kwargs)
        self.train(s)

    def train(self, s):
        s = self.prep(s)

        u, d, v = np.linalg.svd(s, full_matrices=False)
        #dInv = np.diag(1.0 / d)
        dInv = 1.0 / d[:,None]

        # estimated covariance of noise
        s1 = s[1:]
        s2 = s[:-1]
        z = 0.5 * (s1 - s2).T.dot(s1 - s2)

        #zHat = dInv.dot(v.dot(z.dot(v.T.dot(dInv))))
        zHat = dInv * (v.dot(z.dot(v.T * dInv.T)))
        e, wHat = np.linalg.eig(zHat)

        # sort eigenvalues
        idx = e.argsort()
        e = e[idx]
        wHat = wHat[:,idx]

        #self.w[...] = v.T.dot(dInv.dot(wHat))
        self.w = v.T.dot(dInv * wHat)
        self.wInv[...] = np.linalg.pinv(self.w)

class MSF(MaximumSignalFraction):
    pass


def demoMSF():
    t = np.linspace(0.0, 30*np.pi, 1000)

    s1 = spsig.sawtooth(t) #+ 3.0
    s2 = np.cos(5.0*t)
    s3 = np.random.uniform(-1.0, 1.0, size=t.size)
    s = np.vstack((s1,s2,s3)).T

    #m = np.array([ [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5] ])
    m = np.random.random((3,3))
    m /= m.sum(axis=0)

    sMixed = s.dot(m)

    msfFilt = MSF(sMixed, lags=0)
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
    msfFilt.plotTransform(sMixed, ax=axUnmixed)
    axUnmixed.set_title('MSF Components')
    axUnmixed.autoscale(tight=True)

    axCleaned = fig.add_subplot(4,1, 4)
    msfFilt.plotFilter(sMixed, comp=(2,), remove=True, ax=axCleaned)
    axCleaned.set_title('Cleaned Signal (Last Component Removed)')
    axCleaned.autoscale(tight=True)

    fig.tight_layout()


if __name__ == '__main__':
    demoMSF()
    plt.show()
