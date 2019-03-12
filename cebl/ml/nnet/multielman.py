import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from cebl import util

from .. import optim
from .. import paraminit as pinit
from .. import stand
from ..regression import Regression

from . import transfer


class MultilayerElmanRecurrentNetwork(Regression, optim.Optable):
    def __init__(self, x, g, recs=(8,4,2), transient=0, phi=transfer.tanh,
                 #iwInitFunc=pinit.lecun, rwInitFunc=pinit.lecun,
                 hwInitFunc=pinit.esp, vwInitFunc=pinit.lecun, optimFunc=optim.scg,
                 **kwargs):
        x = util.segmat(x)
        g = util.segmat(g)

        self.dtype = np.result_type(x.dtype, g.dtype)

        Regression.__init__(self, x.shape[2], g.shape[2])
        optim.Optable.__init__(self)

        self.transient = transient
        self.phi       = phi

        self.nRecHiddens = list(recs)
        self.nRecLayers = len(self.nRecHiddens)

        self.layerDims = [(self.nIn+self.nRecHiddens[0]+1, self.nRecHiddens[0])]
        for l in range(1, self.nRecLayers):
            self.layerDims.append((self.nRecHiddens[l-1]+self.nRecHiddens[l]+1, self.nRecHiddens[l]))
        self.layerDims.append((self.nRecHiddens[-1]+1, self.nOut))

        views = util.packedViews(self.layerDims, dtype=self.dtype)
        self.pw  = views[0]
        self.hws = views[1:-1]
        self.vw  = views[-1]

        self.iws = []
        self.rws = []
        nIn = self.nIn
        for l in range(self.nRecLayers):
            iw = self.hws[l][:(nIn+1)]
            rw = self.hws[l][(nIn+1):]
            self.iws.append(iw)
            self.rws.append(rw)

            #self.iws[l][...] = iwInitFunc(iw.shape).astype(self.dtype, copy=False)
            #self.rws[l][...] = rwInitFunc(rw.shape).astype(self.dtype, copy=False)

            nIn = self.nRecHiddens[l]

            self.hws[l][...] = hwInitFunc(self.hws[l].shape).astype(self.dtype, copy=False)

        self.vw[...] = vwInitFunc(self.vw.shape).astype(self.dtype, copy=False)

        # train the network
        if optimFunc is not None:
            self.train(x, g, optimFunc, **kwargs)

    def train(self, x, g, optimFunc, **kwargs):
        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

    def parameters(self):
        return self.pw

    def evalRecs(self, x, contexts=None, returnContexts=False):
        x = util.segmat(x)

        x1 = util.bias(x)

        nSeg = x1.shape[0]
        nObs = x1.shape[1]

        r1Prev = x1
        rs = []

        if contexts is None:
            contexts = [np.zeros((nSeg, self.nRecHiddens[l]), dtype=self.dtype)
                        for l in range(self.nRecLayers)]

        for l in range(self.nRecLayers):
            nIn1 = r1Prev.shape[2]

            r = np.empty((nSeg, nObs, self.nRecHiddens[l]), dtype=self.dtype)
            r1c = np.empty((nSeg, nIn1+self.nRecHiddens[l]), dtype=self.dtype)
            context = contexts[l]

            for t in range(nObs):
                r1c[:,:nIn1] = r1Prev[:,t]
                r1c[:,nIn1:] = context

                r[:,t] = self.phi(r1c.dot(self.hws[l]))
                context[...] = r[:,t]

            r1Prev = util.bias(r)
            rs.append(r)

        if returnContexts:
            return rs, contexts
        else:
            return rs

    def eval(self, x, returnTransient=False):
        x = util.segmat(x)

        r = self.evalRecs(x)[-1]
        if not returnTransient:
            r = r[:,self.transient:]

        y = r.dot(self.vw[:-1]) + self.vw[-1]

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

        if isinstance(unrollSteps, (int,)):
            unrollSteps = [unrollSteps,]*self.nRecLayers

        views = util.packedViews(self.layerDims, dtype=self.dtype)
        pg  = views[0]
        hgs = views[1:-1]
        vg  = views[-1]

        x1 = util.bias(x)

        nSeg = x1.shape[0]
        nObs = x1.shape[1]

        r1Prev = x1
        r1cs = []
        rPrimes = []

        for l in range(self.nRecLayers):
            nIn1 = r1Prev.shape[2]

            r = np.empty((nSeg, nObs, self.nRecHiddens[l]), dtype=self.dtype)
            h = np.empty((nSeg, nObs, self.nRecHiddens[l]), dtype=self.dtype)
            r1c = np.empty((nSeg, nObs, nIn1+self.nRecHiddens[l]), dtype=self.dtype)
            context = np.zeros((nSeg, self.nRecHiddens[l]), dtype=self.dtype)

            for t in range(nObs):
                r1c[:,t,:nIn1] = r1Prev[:,t]
                r1c[:,t,nIn1:] = context

                h[:,t] = r1c[:,t].dot(self.hws[l])
                r[:,t] = self.phi(h[:,t])
                context[...] = r[:,t]

            r1Prev = util.bias(r)
            r1cs.append(r1c)

            rPrime = self.phi(h, 1)
            rPrimes.append(rPrime)

        # evaluate visible layer
        r1 = r1Prev
        y = r1.dot(self.vw)

        # error components, ditch transient
        e = (y - g)[:,self.transient:]
        delta = np.zeros(g.shape, dtype=self.dtype)
        delta[:,self.transient:] = 2.0 * e / e.size

        # visible layer gradient
        r1f = r1.reshape((-1, r1.shape[-1]))
        deltaf = delta.reshape((-1, delta.shape[-1]))
        vg[...] = r1f.T.dot(deltaf)

        # backward pass through each layer
        w = self.vw
        for l in range(self.nRecLayers-1, -1, -1):
            r1c = r1cs[l]
            rwsTrans = self.rws[l].T
            rPrime = rPrimes[l]

            deltaPrev = delta.dot(w[:-1].T)

            gamma = np.zeros((nSeg, unrollSteps[l], self.nRecHiddens[l]), dtype=self.dtype)
            #delta = np.zeros((nSeg, nObs-self.transient, self.nRecHiddens[l]), dtype=self.dtype)
            delta = np.zeros((nSeg, nObs, self.nRecHiddens[l]), dtype=self.dtype)

            # unrolled through time
            #for t in range(nObs-self.transient-1, 0, -1):
            for t in range(nObs-1, 0, -1):
                rPrimet = rPrime[:,t][:,None,:]

                beta = gamma[:,:-1]
                beta = beta.dot(rwsTrans)

                gamma[:,0] = deltaPrev[:,t]
                gamma[:,1:] = beta
                gamma *= rPrimet

                delta[:,t] = gamma.sum(axis=1)

            r1cf = r1c.reshape((-1, r1c.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            hgs[l][...] = r1cf.T.dot(deltaf)

            #print("hg %d: %f" % (l, np.sqrt(np.mean(hgs[l]**2))))

            w = self.iws[l]

        if returnError:
            return np.mean(e**2), pg
        else:
            return pg

class MERN(MultilayerElmanRecurrentNetwork):
    pass

def demoMERNTXOR():
    def xor(a, b):
        if (a == True) and (b == True):
            return False

        if a or b:
            return True

        return False

    n = 500
    horizon = 10
    transient = horizon+1
    unrollSteps = horizon+2
    #unrollSteps = 3

    x = np.random.randint(0,2, size=n).astype(np.float32)
    g = np.array([int(xor(x[i-horizon], x[i-horizon-1])) if i > horizon
            else 0 for i in range(len(x))], dtype=np.float32)

    std = stand.Standardizer(x)
    x = std.apply(x)
    g = std.apply(g)

    #options = {"maxfev": 1000}
    #net = MERN(x[None,...], g[None,...], recs=(10,10,),
    net = MERN((x[0:250],x[250:500]), (g[0:250],g[250:500]), recs=(20,10),
               #iwInitFunc=lambda size: np.random.uniform(-0.2, 0.2, size=size),
               #rwInitFunc=lambda size: np.random.uniform(-0.2, 0.2, size=size),
               #vwInitFunc=lambda size: np.random.uniform(-0.2, 0.2, size=size),
               phi=transfer.tanhTwist,
               unrollSteps=unrollSteps, transient=transient,
               #optimFunc=optim.sciopt, method="Powell",
               optimFunc=optim.scg,
               #optimFunc=optim.rprop, stepUp=1.02, stepDown=0.4,
               #optimFunc=optim.pso, nParticles=20, vInit=0.01, momentum=0.85, pAttract=0.2, gAttract=0.6,
               #optimFunc=optim.alopex, stepInit=0.0005, tempIter=20,
               maxIter=1000, accuracy=0.001, precision=0.0,
               eTrace=True, pTrace=True, verbose=True)

    # redo for test data
    x = np.random.randint(0,2, size=n).astype(np.float32)
    g = np.array([int(xor(x[i-horizon], x[i-horizon-1])) if i > horizon
            else 0 for i in range(len(x))], dtype=np.float32)

    x = std.apply(x)
    g = std.apply(g)

    out = net.eval(x[None,...], returnTransient=True)[0]
    hout = net.evalRecs(x[None,...])
    hout = [ho[0] for ho in hout]

    fig = plt.figure(figsize=(20,12))

    # targets
    axTarg = fig.add_subplot(4,2,1)
    axTarg.bar(range(len(g)), g)
    axTarg.set_xlim((0, len(g)))
    axTarg.set_ylim((-1.5,1.5))
    axTarg.set_title("Test Targets")

    # network output
    axOut = fig.add_subplot(4,2,3)
    axOut.bar(range(len(g)), out)
    axOut.set_xlim((0, len(g)))
    axOut.set_ylim((-1.5,1.5))
    axOut.set_title("Test Outputs")

    # first layer output
    axH1 = fig.add_subplot(4,2,5)
    #axH1.bar(range(len(hout[0])), hout[0])
    axH1.plot(hout[1] + util.colsep(hout[1]))
    #axH1.set_xlim((0, len(g)))
    #axH1.set_ylim((0.0,1.0))
    axH1.set_title("H2")

    # second layer output
    axH2 = fig.add_subplot(4,2,7)
    #axH2.bar(range(len(hout[1])), hout[1])
    axH2.plot(hout[0] + util.colsep(hout[0]))
    #axH2.set_xlim((0, len(g)))
    #axH2.set_ylim((0.0,1.0))
    axH2.set_title("H1")

    axETrace = fig.add_subplot(4,2,2)
    axETrace.plot(np.array(net.trainResult["eTrace"]))

    axPTrace = fig.add_subplot(4,2,4)
    axPTrace.plot(np.array(net.trainResult["pTrace"]))

    axH1Dens = fig.add_subplot(4,2,6)
    t = np.arange(0.0,4.0,0.01)
    nb, bins, patches = axH1Dens.hist(hout[0].ravel(), normed=True,
            orientation="horizontal", label="Activations")
    axH1Dens.plot(np.linspace(0.0,np.max(nb),t.size), np.tanh(t-2.0),
                  linewidth=2, label=r"$\phi$") # label=r"$\phi="+self.phi.__name__)
    axH1Dens.legend(loc="lower right")

    axH2Dens = fig.add_subplot(4,2,8)
    t = np.arange(0.0,4.0,0.01)
    nb, bins, patches = axH2Dens.hist(hout[0].ravel(), normed=True,
            orientation="horizontal", label="Activations")
    axH2Dens.plot(np.linspace(0.0,np.max(nb),t.size), np.tanh(t-2.0),
                  linewidth=2, label=r"$\phi$") # label=r"$\phi="+self.phi.__name__)
    axH2Dens.legend(loc="lower right")

if __name__ == "__main__":
    demoMERNTXOR()
    plt.show()
