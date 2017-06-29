import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from cebl import util

from .. import optim
from .. import paraminit as pinit
from ..regression import Regression

from . import transfer


class ElmanRecurrentNetwork(Regression, optim.Optable):
    def __init__(self, x, g, nHidden=10, transient=0, phi=transfer.tanh,
                 #iwInitFunc=pinit.lecun, rwInitFunc=pinit.lecun,
                 hwInitFunc=pinit.esp, vwInitFunc=pinit.lecun, optimFunc=optim.scg,
                 **kwargs):
        x = util.segmat(x)
        g = util.segmat(g) # flattenOut? XXX - idfah

        self.dtype = np.result_type(x.dtype, g.dtype)

        Regression.__init__(self, x.shape[2], g.shape[2])
        optim.Optable.__init__(self)

        self.nHidden   = nHidden
        self.transient = transient
        self.phi       = phi

        self.pw, self.hw, self.vw = \
            util.packedViews(((self.nIn+self.nHidden+1, self.nHidden),
                              (self.nHidden+1, self.nOut)),
                             dtype=self.dtype)

        self.iw = self.hw[:(self.nIn+1)]
        self.rw = self.hw[(self.nIn+1):]

        # initialize weights
        #self.iw[...] = iwInitFunc(self.iw.shape).astype(self.dtype, copy=False)
        #self.rw[...] = rwInitFunc(self.rw.shape).astype(self.dtype, copy=False)
        self.hw[...] = hwInitFunc(self.hw.shape).astype(self.dtype, copy=False)
        self.vw[...] = vwInitFunc(self.vw.shape).astype(self.dtype, copy=False)

        # train the network
        if optimFunc is not None:
            self.train(x, g, optimFunc, **kwargs)

    def train(self, x, g, optimFunc, **kwargs):
        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

    def parameters(self):
        return self.pw

    def evalRecs(self, x, context=None, returnContext=False):
        x = util.segmat(x)

        x1 = util.bias(x)

        nSeg = x1.shape[0]
        nObs = x1.shape[1]
        nIn1 = x1.shape[2]

        r = np.empty((nSeg, nObs, self.nHidden), dtype=self.dtype)

        if context is None:
            context = np.zeros((nSeg, self.nHidden), dtype=self.dtype)

        x1c = np.empty((nSeg, nIn1+self.nHidden), dtype=self.dtype)

        for t in xrange(nObs):
            x1c[:,:nIn1] = x1[:,t]
            x1c[:,nIn1:] = context

            r[:,t] = self.phi(x1c.dot(self.hw))
            context[...] = r[:,t]

        if returnContext:
            return r, context
        else:
            return r

    def eval(self, x, returnTransient=False):
        x = util.segmat(x)

        r = self.evalRecs(x)
        y = r.dot(self.vw[:-1]) + self.vw[-1]

        if not returnTransient:
            y = y[:,self.transient:]

        return y

    def error(self, x, g, *args, **kwargs):
        x = util.segmat(x)
        g = util.segmat(g)[:,self.transient:]

        # evaluate network
        y = self.eval(x, returnTransient=False)

        # figure mse
        return np.mean((y-g)**2)

    def gradient(self, x, g, unrollSteps=10, returnError=True):
        x = util.segmat(x)
        g = util.segmat(g)

        # packed views of the hidden and visible gradient matrices
        pg, hg, vg = util.packedViews((self.hw.shape, self.vw.shape),
                                      dtype=self.dtype)

        x1 = util.bias(x)

        nSeg = x1.shape[0]
        nObs = x1.shape[1]
        nIn1 = x1.shape[2]

        h = np.empty((nSeg, nObs, self.nHidden), dtype=self.dtype)
        r = np.empty((nSeg, nObs, self.nHidden), dtype=self.dtype)
        x1c = np.empty((nSeg, nObs, nIn1+self.nHidden), dtype=self.dtype)
        context = np.zeros((nSeg, self.nHidden), dtype=self.dtype)

        for t in xrange(nObs):
            x1c[:,t,:nIn1] = x1[:,t]
            x1c[:,t,nIn1:] = context

            h[:,t] = x1c[:,t].dot(self.hw)
            r[:,t] = self.phi(h[:,t])
            context[...] = r[:,t]

        r1 = util.bias(r)
        y = r1.dot(self.vw)
        rPrime = self.phi(h, 1)

        # error components, ditch transient
        e = (y - g)[:,self.transient:]
        delta = np.zeros(g.shape, dtype=self.dtype)
        delta[:,self.transient:] = 2.0 * e / e.size

        # visible layer gradient
        r1f = r1.reshape((-1, r1.shape[-1]))
        deltaf = delta.reshape((-1, delta.shape[-1]))
        vg[...] = r1f.T.dot(deltaf)

        vwDelta = delta.dot(self.vw[:-1].T)

        gamma = np.zeros((nSeg, unrollSteps, self.nHidden), dtype=self.dtype)
        #delta = np.zeros((nSeg, nObs-self.transient, self.nHidden), dtype=self.dtype)
        delta = np.zeros((nSeg, nObs, self.nHidden), dtype=self.dtype)

        ##hg[...] = 0.0

        # backward pass for hidden layer, unrolled through time
        #for t in xrange(nObs-self.transient-1, 0, -1):
        for t in xrange(nObs-1, 0, -1):
            rPrimet = rPrime[:,t][:,None,:]
            #x1ct = x1c[:,t][:,None,:]
            ##x1ct = x1c[:,t]

            beta = gamma[:,:-1]
            beta = beta.dot(self.rw.T)

            gamma[:,0] = vwDelta[:,t]
            gamma[:,1:] = beta
            gamma *= rPrimet

            ##x1ctf = np.tile(x1ct, unrollSteps).reshape((-1, x1ct.shape[-1]))
            ##gammaf = gamma.reshape((-1, gamma.shape[-1]))
            delta[:,t] = gamma.sum(axis=1)

            #hg += x1ctf.T.dot(gammaf)
            ##hg += x1ct.T.dot(gamma.sum(axis=1))

            ##hg += x1ct.T.dot(gamma.swapaxes(0,1)).sum(axis=1)

        x1cf = x1c.reshape((-1, x1c.shape[-1]))
        deltaf = delta.reshape((-1, delta.shape[-1]))
        #hg[...] = x1c.reshape((-1, x1c.shape[-1])).T.dot(delta.reshape((-1, d.shape[-1])))
        hg[...] = x1cf.T.dot(deltaf)

        if returnError:
            return np.mean(e**2), pg
        else:
            return pg

class ERN(ElmanRecurrentNetwork):
    pass

def demoERNTXOR():
    def xor(a, b):
        if (a == True) and (b == True):
            return False

        if a or b:
            return True

        return False

    n = 500
    horizon = 5
    transient = horizon+1
    unrollSteps = horizon+2
    #unrollSteps = 4

    x = np.random.randint(0,2, size=n).astype(np.float32)
    g = np.array([int(xor(x[i-horizon], x[i-horizon-1])) if i > horizon
            else 0 for i in xrange(len(x))], dtype=np.float32)

    #options = {'maxfev': 1000}
    #net = ERN(x[None,...], g[None,...], nHidden=10,
    net = ERN((x[0:250],x[250:500]), (g[0:250],g[250:500]), nHidden=20,
               unrollSteps=unrollSteps, transient=transient,
               #optimFunc=optim.sciopt, method='Powell',
               #optimFunc=optim.scg,
               #optimFunc=optim.pso, nParticles=20, vInit=0.01, momentum=0.85, pAttract=0.2, gAttract=0.6,
               #optimFunc=optim.alopex, stepInit=0.0001, tempIter=20,
               maxIter=10000, accuracy=0.001, precision=0.0, verbose=True,
               phi=transfer.lecun)

    # redo for test data
    x = np.random.randint(0,2, size=n).astype(np.float32)
    g = np.array([int(xor(x[i-horizon], x[i-horizon-1])) if i > horizon
            else 0 for i in xrange(len(x))], dtype=np.float32)

    out = net.eval(x[None,...], returnTransient=True)[0]

    fig = plt.figure()
    axTarg = fig.add_subplot(2,1,1)
    axTarg.bar(range(len(g)), g)
    axTarg.set_xlim((0, len(g)))
    axTarg.set_ylim((0.0,1.0))

    axOut = fig.add_subplot(2,1,2)
    axOut.bar(range(len(g)), out)
    axOut.set_xlim((0, len(g)))
    axOut.set_ylim((0.0,1.0))

def demoERNTXORPSO():
    def xor(a, b):
        if (a == True) and (b == True):
            return False

        if a or b:
            return True

        return False

    n = 500
    horizon = 5
    transient = horizon+1
    unrollSteps = horizon+2

    xTrain = np.random.randint(0,2, size=n).astype(np.float32)
    gTrain = np.array([int(xor(xTrain[i-horizon], xTrain[i-horizon-1])) if i > horizon
            else 0 for i in xrange(len(xTrain))], dtype=np.float32)

    #momentums = np.array((0.3, 0.6, 0.8, 0.9))#np.arange(0.2, 1.0, 0.2)
    momentums = np.array((0.7,))#np.arange(0.2, 1.0, 0.2)
    pAttracts = np.arange(0.1, 1.1, 0.1)
    gAttracts = np.arange(0.1, 1.1, 0.1)

    its = np.zeros((momentums.size, pAttracts.size, gAttracts.size))
    ers = np.zeros((momentums.size, pAttracts.size, gAttracts.size))

    for i,momentum in enumerate(momentums):
        for j,pAttract in enumerate(pAttracts):
            for k,gAttract in enumerate(gAttracts):
                print 'momentum: ', momentum
                print 'pAttract: ', pAttract
                print 'gAttract: ', gAttract

                net = ERN(xTrain[None,...], gTrain[None,...], nHidden=32,
                           unrollSteps=unrollSteps, transient=transient,
                           optimFunc=optim.pso, nParticles=20, vInit=0.01,
                           momentum=momentum, pAttract=pAttract, gAttract=gAttract,
                           maxIter=500, accuracy=0.005, precision=0.0, verbose=False)

                its[i,j,k] = net.trainResult['iteration']
                ers[i,j,k] = net.trainResult['error']
                print 'Error: ', ers[i,j,k]
                print '======='

    bi, bj, bk = np.unravel_index(np.argmin(ers), ers.shape)
    print '======='
    print 'Best: ', ers[bi,bj,bk], momentums[bi], pAttracts[bj], gAttracts[bk]
    print '======='

    for i,momentum in enumerate(momentums):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(ers[i], origin='lowerleft')#, interpolation='none')
        fig.colorbar(im)
        ax.set_title('ers momentum: %.2f' % momentum)
        ax.set_xlabel('pAttract')
        ax.set_ylabel('gAttract')

    for i,momentum in enumerate(momentums):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(its[i], origin='lowerleft')#, interpolation='none')
        fig.colorbar(im)
        ax.set_title('its momentum: %.2f' % momentum)
        ax.set_xlabel('pAttract')
        ax.set_ylabel('gAttract')

if __name__ == '__main__':
    demoERNTXOR()
    plt.show()
