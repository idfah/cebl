"""Common Spatial Patterns.
"""

import matplotlib.pyplot as plt
import numpy as np

from cebl import util

from .strans import STrans


class CommonSpatialPatterns(STrans):
    def __init__(self, s1, s2, *args, **kwargs):
        s1 = util.colmat(s1)
        s2 = util.colmat(s2)

        STrans.__init__(self, np.vstack((s1,s2)), *args, **kwargs)

        self.train(s1, s2)

    def train(self, s1, s2):
        s1 = self.prep(s1)
        s2 = self.prep(s2)

        if True: # maybe faster and more stable?
            u, d, v = np.linalg.svd(np.vstack((s1,s2)), full_matrices=False)
            dInv = np.sqrt(0.5*(s1.shape[0]+s2.shape[0]-2)) / d[:,None]

            c1 = s1.T.dot(s1)
            c2 = s2.T.dot(s2)
            z = (c1 - c2)

            zHat = dInv * (v.dot(z.dot(v.T * dInv.T)))
            e, wHat = np.linalg.eig(zHat)

            # sort eigenvalues
            idx = e.argsort()[::-1]
            e = e[idx]
            wHat = wHat[:,idx]

            self.w = v.T.dot(dInv * wHat)
            self.wInv[...] = (d * wHat.T).dot(v)

        else: # commonly accepted approach in the literature
            c1 = s1.T.dot(s1)
            c2 = s2.T.dot(s2)
            cc = c1 + c2

            l, u = np.linalg.eig(cc)
            idx = l.argsort()[::-1] # sort
            l = l[idx]
            u = u[:,idx]

            p = np.diag(1.0/np.sqrt(l)).dot(u.T)

            ra = p.dot(c1).dot(p.T)
            #rb = p.dot(c2).dot(p.T)

            f, q = np.linalg.eig(ra) # q should be same for ra and rb
            idx = f.argsort()[::-1] # sort
            f = f[idx]
            q = q[:,idx]

            self.w = p.T.dot(q)
            self.wInv = np.linalg.pinv(self.w)

class CSP(CommonSpatialPatterns):
    pass

def demoCSP():
    n1 = 1000
    t1 = np.linspace(0.0, 30*np.pi, n1)

    s1 = np.random.random((n1, 5))
    s1[:,0] += 1.00 * np.sin(t1)
    s1[:,1] += 0.75 * np.cos(t1)
    s1[:,2] += 0.50 * np.sin(t1)
    s1[:,3] += 0.25 * np.cos(t1)
    s1[:,4] += 0.10 * np.sin(t1)

    #n2 = 500
    n2 = 1000
    t2 = np.linspace(0.0, 30*np.pi, n2)

    s2 = np.random.random((n2, 5))
    s2[:,4] += 0.25 * np.cos(t2)
    s2[:,3] += 0.75 * np.sin(t2)
    s2[:,2] += 1.00 * np.cos(t2)
    s2[:,1] += 0.75 * np.sin(t2)
    s2[:,0] += 0.25 * np.cos(t2)

    fig = plt.figure()

    axS1 = fig.add_subplot(4,1, 1)
    axS1.plot(s1 + util.colsep(s1))
    axS1.set_title('Class 1 Signal')

    axS2 = fig.add_subplot(4,1, 2)
    axS2.plot(s2 + util.colsep(s2))
    axS2.set_title('Class 2 Signal')

    cspFilt = CSP(s1, s2, lags=0)

    csp1 = cspFilt.transform(s1)#, comp=(2,), remove=True)
    csp2 = cspFilt.transform(s2)#, comp=(2,), remove=True)

    print(csp1.var(axis=0))
    print(csp2.var(axis=0))

    axCSP1 = fig.add_subplot(4,1, 3)
    axCSP1.plot(csp1 + util.colsep(csp1))
    axCSP1.set_title('Class 1 Signal CSP')

    axCSP2 = fig.add_subplot(4,1, 4)
    axCSP2.plot(csp2 + util.colsep(csp2))
    axCSP2.set_title('Class 2 Signal CSP')

if __name__ == '__main__':
    demoCSP()
    plt.show()
