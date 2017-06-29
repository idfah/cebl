import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.lib.stride_tricks as npst
import scipy as sp
import scipy.signal as spsig

from cebl import util

from .. import label
from .. import optim
from .. import paraminit as pinit
from .. import stand
from ..regression import Regression

from . import transfer
from .ddembed import *


class ConvolutionalNetworkRegression(Regression, optim.Optable):
    def __init__(self, x, g, convs=((8,16),(16,8)), nHidden=None,
                 transFunc=transfer.lecun, weightInitFunc=pinit.lecun,
                 penalty=None, elastic=1.0, optimFunc=optim.scg, **kwargs):
        x = util.segmat(x)
        g = util.segmat(g)
        self.dtype = np.result_type(x.dtype, g.dtype)

        Regression.__init__(self, x.shape[2], g.shape[2])
        optim.Optable.__init__(self)

        self.nConvHiddens, self.convWidths = zip(*convs)
        self.nConvLayers = len(convs)
        self.nHidden = nHidden

        self.layerDims = [(self.nIn*self.convWidths[0]+1, self.nConvHiddens[0]),]
        for l in xrange(1, self.nConvLayers):
            ni = self.nConvHiddens[l-1] * self.convWidths[l] + 1
            no = self.nConvHiddens[l]
            self.layerDims.append((ni, no))

        if self.nHidden is None:
            self.layerDims.append((self.nConvHiddens[-1]+1, self.nOut))
        else:
            self.layerDims.append((self.nConvHiddens[-1]+1, self.nHidden))
            self.layerDims.append((self.nHidden+1, self.nOut))

        self.transFunc = transFunc if util.isiterable(transFunc) \
                else (transFunc,) * (len(self.layerDims)-1)
        assert len(self.transFunc) == (len(self.layerDims)-1)

        views = util.packedViews(self.layerDims, dtype=self.dtype)
        self.pw  = views[0]

        if self.nHidden is None:
            self.cws = views[1:-1]
            self.hw = None
            self.vw = views[-1]
        else:
            self.cws = views[1:-2]
            self.hw  = views[-2]
            self.vw  = views[-1]

        if not util.isiterable(weightInitFunc):
            weightInitFunc = (weightInitFunc,) * (self.nConvLayers+2)
        assert len(weightInitFunc) == (len(self.cws) + 2)

        self.penalty = penalty
        if self.penalty is not None:
            if not util.isiterable(self.penalty):
                self.penalty = (self.penalty,) * (self.nConvLayers+2)
        assert (self.penalty is None) or (len(self.penalty) == (len(self.cws) + 2))

        self.elastic = elastic if util.isiterable(elastic) \
                else (elastic,) * (self.nConvLayers+2)
        assert (len(self.elastic) == (len(self.cws) + 2))

        # initialize weights
        for cw, wif in zip(self.cws, weightInitFunc):
            cw[...] = wif(cw.shape).astype(self.dtype, copy=False)

        if self.nHidden is not None:
            self.hw[...] = weightInitFunc[-2](self.hw.shape).astype(self.dtype, copy=False)

        self.vw[...] = weightInitFunc[-1](self.vw.shape).astype(self.dtype, copy=False)

        # train the network
        if optimFunc is not None:
            self.train(x, g, optimFunc, **kwargs)

    def train(self, x, g, optimFunc, **kwargs):
        x = util.segmat(x)
        x = np.require(x, requirements=['O', 'C'])

        g = util.segmat(g)
        g = np.require(g, requirements=['O', 'C'])

        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

    def parameters(self):
        return self.pw

    def evalConvs(self, x):
        x = util.segmat(x)

        c = x
        cs = []
        for l, cw in enumerate(self.cws):
            width = self.convWidths[l]
            phi = self.transFunc[l]

            c = util.timeEmbed(c, lags=width-1, axis=1)
            c = phi(util.segdot(c, cw[:-1]) + cw[-1])
                
            cs.append(c)

        return cs

    def eval(self, x):
        x = util.segmat(x)

        # evaluate convolutional layers
        c = self.evalConvs(x)[-1]

        # evaluate hidden layer
        z = self.transFunc[-1](util.segdot(c, self.hw[:-1]) + self.hw[-1]) if self.nHidden is not None else c

        # evaluate visible layer
        return util.segdot(z, self.vw[:-1]) + self.vw[-1]

    def penaltyError(self):
        if self.penalty is None:
            return 0.0

        weights = []
        weights += [cw[:-1].ravel() for cw in self.cws]
        if self.nHidden is not None:
            weights.append(self.hw[:-1].ravel())
        weights.append(self.vw[:-1].ravel())

        totalPenalty = 0.0
        for weight, penalty, elastic in zip(weights, self.penalty, self.elastic):
            totalPenalty += ( elastic      * penalty * weight.dot(weight)/weight.size + # L2
                             (1.0-elastic) * penalty * np.mean(np.abs(weight)) ) # L1

        return totalPenalty

    def penaltyGradient(self, layer):
        if self.penalty is None:
            return 0.0

        if layer == -1:
            weights = self.vw
        elif layer == -2:
            weights = self.hw
        else:
            weights = self.cws[layer]

        penalty = self.penalty[layer]
        elastic = self.elastic[layer]

        penMask = np.ones_like(weights)
        penMask[-1] = 0.0
        return ( elastic * 2.0 * penalty * penMask * weights/weights.size + # L2
                (1.0-elastic)  * penalty * penMask * np.sign(weights)/weights.size ) # L1

    def error(self, x, g):
        x = util.segmat(x)
        g = util.segmat(g)

        # evaluate network
        y = self.eval(x)

        trim = (g.shape[1] - y.shape[1]) // 2
        gTrim = g[:,:(g.shape[1]-trim)]
        gTrim = gTrim[:,-y.shape[1]:]

        # figure mse
        return np.mean((y-gTrim)**2) + self.penaltyError()

    def gradient(self, x, g, returnError=True):
        x = util.segmat(x)
        g = util.colmat(g)

        # packed views of the hidden and visible gradient matrices
        views = util.packedViews(self.layerDims, dtype=self.dtype)
        pg  = views[0]

        if self.nHidden is None:
            cgs = views[1:-1]
            hg  = None
            vg  = views[-1]
        else:
            cgs = views[1:-2]
            hg  = views[-2]
            vg  = views[-1]

        # forward pass
        c = x
        c1s = []
        cPrimes = []
        for l, cw in enumerate(self.cws):
            width = self.convWidths[l]
            phi = self.transFunc[l]

            c = util.timeEmbed(c, lags=width-1, axis=1)

            c1 = util.bias(c)
            c1s.append(c1)

            h = util.segdot(c1, cw)
            cPrime = phi(h, 1)
            cPrimes.append(cPrime)

            c = phi(h)
        
        c1 = util.bias(c)

        # evaluate hidden and visible layers
        if self.nHidden is None:
            y = util.segdot(c1, self.vw)
        else:
            h = util.segdot(c1, self.hw)
            z1 = util.bias(self.transFunc[-1](h))
            zPrime = self.transFunc[-1](h, 1)
            y = util.segdot(z1, self.vw)

        # error components
        trim = (g.shape[1] - y.shape[1]) // 2
        gTrim = g[:,:(g.shape[1]-trim)]
        gTrim = gTrim[:,-y.shape[1]:]

        # error components
        e = util.colmat(y - gTrim)
        delta = 2.0 * e / e.size

        if self.nHidden is None:
            # visible layer gradient
            c1f = c1.reshape((-1, c1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            vg[...] = c1f.T.dot(deltaf)
            vg += self.penaltyGradient(-1)

            delta = util.segdot(delta, self.vw[:-1].T)

        else:
            # visible layer gradient
            z1f = z1.reshape((-1, z1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            vg[...] = z1f.T.dot(deltaf)
            vg += self.penaltyGradient(-1)

            # hidden layer gradient
            c1f = c1.reshape((-1, c1.shape[-1]))
            delta = util.segdot(delta, self.vw[:-1].T) * zPrime
            deltaf = delta.reshape((-1, delta.shape[-1]))
            hg[...] = c1f.T.dot(deltaf)
            hg += self.penaltyGradient(-2)

            delta = util.segdot(delta, self.hw[:-1].T)

        # backward pass for convolutional layers
        for l in xrange(self.nConvLayers-1, -1, -1):
            c1 = c1s[l]
            cPrime = cPrimes[l]

            delta = delta[:,:cPrime.shape[1]] * cPrime

            c1f = c1.reshape((-1, c1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            cgs[l][...] = c1f.T.dot(deltaf)
            cgs[l] += self.penaltyGradient(l)

            if l > 0: # won't propigate back to inputs
                delta = util.segdot(delta, self.cws[l][:-1].T)
                delta = deltaDeEmbedSum(delta, self.convWidths[l])

        if returnError:
            error = np.mean(e**2) + self.penaltyError()
            return error, pg
        else:
            return pg

class CNR(ConvolutionalNetworkRegression):
    pass

def demoCNR():
    t = np.linspace(0,2,513)

    ns = 50

    phaseShifts = np.random.uniform(-np.pi, np.pi, size=(ns,1))
    slow   = np.sin(8*2*np.pi*t[None,:] + phaseShifts)
    medium = 0.5*np.sin(28*2*np.pi*t[None,:] + phaseShifts)
    fast   = 0.25*np.sin(60*2*np.pi*t[None,:] + phaseShifts)

    x = slow + medium
    g = slow

    #noise = np.random.normal(size=x.shape, scale=0.5)
    #x += noise

    x = x[...,None]
    g = g[...,None]

    x = x.astype(np.float32)
    g = g.astype(np.float32)

    ns2 = ns // 2
    xTrain = x[ns2:]
    gTrain = g[ns2:]
    xTest = x[:ns2]
    gTest = g[:ns2]

    print x.shape, g.shape

    #model = CNR(x=xTrain, g=gTrain, convs=((1,11),(1,9),(1,7)), nHidden=None,
    #            optimFunc=optim.scg, maxIter=100, transFunc=transfer.linear,
    #            precision=1.0e-7, accuracy=0.0, pTrace=True, eTrace=True, verbose=True)
    model = CNR(x=xTrain, g=gTrain, convs=((2,7),(4,5),(8,3)), nHidden=None,
                optimFunc=optim.scg, maxIter=200, transFunc=transfer.lecun,
                precision=1.0e-7, accuracy=0.0, pTrace=True, eTrace=True, verbose=True)

    yTest = model.eval(xTest)
    print yTest.shape, yTest.dtype

    trim = (len(t) - yTest.shape[1]) // 2
    tTrim = t[:(len(t)-trim)]
    tTrim = tTrim[-yTest.shape[1]:]

    fig = plt.figure(figsize=(20,6))
    axSigs = fig.add_subplot(model.nConvLayers,3, 1)
    axSigs.plot(t, xTest[0].T.squeeze(), color='blue', linewidth=2)
    #axSigs.plot(t, xTest.T.squeeze(), color='blue', alpha=0.1, linewidth=2)
    axSigs.plot(t, -3.0+gTest[0].T.squeeze(), color='red', linewidth=2)
    #axSigs.plot(t, 3.0-gTest.T.squeeze(), color='red', alpha=0.1, linewidth=2)
    axSigs.plot(tTrim, -5.5+yTest[0].T.squeeze(), color='red', linewidth=2)
    #axSigs.plot(tTrim, 5.5-yTest.T.squeeze(), color='red', alpha=0.1, linewidth=2)
    axSigs.set_title('')
    axSigs.set_xlabel('Time')
    axSigs.set_ylabel('Signal')
    axSigs.autoscale(tight=True)

    axETrace = fig.add_subplot(model.nConvLayers,3, 2)
    eTrace = np.array(model.trainResult['eTrace'])
    axETrace.plot(eTrace)

    axPTrace = fig.add_subplot(model.nConvLayers,3, 3)
    pTrace = np.array(model.trainResult['pTrace'])
    axPTrace.plot(pTrace)

    cs = model.evalConvs(xTest)
    for i in xrange(model.nConvLayers):
        axConvs = fig.add_subplot(model.nConvLayers,3, 4+i)
        c = cs[i][0,:,:]
        axConvs.plot(c+util.colsep(c), color='blue', linewidth=2, alpha=0.25)

        axConvs.autoscale(tight=True)

        #axRespon = fig.add_subplot(3,3, 7+i)
        #freqs, responses = zip(*[spsig.freqz(cw) for cw in model.cws[i].T])
        #freqs = np.array(freqs)
        #responses = np.array(responses)
        #axRespon.plot(freqs.T, np.abs(responses).T)

    print 'nParams: ', model.parameters().size

    fig.tight_layout()

if __name__ == '__main__':
    demoCNR()
    plt.show()
