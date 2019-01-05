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
from ..classifier import Classifier

from . import transfer
from .ddembed import *


class ConvolutionalNetworkAccum(Classifier, optim.Optable):
    def __init__(self, classData, convs=((8,16),(16,8)), nHidden=None,
                 poolSize=2, poolMethod='average', filtOrder=8,
                 transFunc=transfer.lecun, weightInitFunc=pinit.lecun,
                 penalty=None, elastic=1.0, optimFunc=optim.scg, **kwargs):
        Classifier.__init__(self, util.segmat(classData[0]).shape[2], len(classData))
        optim.Optable.__init__(self)

        self.dtype = np.result_type(*[cls.dtype for cls in classData])

        self.nConvHiddens, self.convWidths = zip(*convs)
        self.nConvLayers = len(convs)
        self.nHidden = nHidden

        self.layerDims = [(self.nIn*self.convWidths[0]+1, self.nConvHiddens[0]),]
        for l in range(1, self.nConvLayers):
            ni = self.nConvHiddens[l-1] * self.convWidths[l] + 1
            no = self.nConvHiddens[l]
            self.layerDims.append((ni, no))

        if self.nHidden is None:
            self.layerDims.append((self.nConvHiddens[-1]+1, self.nCls))
        else:
            self.layerDims.append((self.nConvHiddens[-1]+1, self.nHidden))
            self.layerDims.append((self.nHidden+1, self.nCls))

        self.poolSize = poolSize if util.isiterable(poolSize) \
                else (poolSize,) * self.nConvLayers
        assert len(self.poolSize) == self.nConvLayers

        self.poolMethod = poolMethod.lower()
        if self.poolMethod == 'lanczos':
            self.initLanczos(filtOrder)

        elif not self.poolMethod in ('stride', 'average'):
            raise RuntimeError('Invalid poolMethod %s.' % str(self.poolMethod))

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
            self.train(classData, optimFunc, **kwargs)

    def initLanczos(self, filtOrder):
        self.filtOrder = filtOrder

        if self.filtOrder % 2 != 0:
             raise RuntimeError('Invalid filtOrder: ' + str(self.filtOrder) +
                 ' Must be an even integer.')

        radius = self.filtOrder // 2
        win = np.sinc(np.linspace(-radius, radius, self.filtOrder+1) / float(radius)) # lanczos
        #win = spsig.hamming(self.filtOrder+1) # sinc-hamming

        # this should be automated somehow XXX - idfah
        if self.filtOrder <= 6:
            cutoff = 2*0.570

        elif self.filtOrder <= 8:
            cutoff = 2*0.676

        elif self.filtOrder <= 12:
            cutoff = 2*0.781

        elif self.filtOrder <= 16:
            cutoff = 2*0.836

        elif self.filtOrder <= 32:
            cutoff = 2*0.918

        elif self.filtOrder <= 64:
            cutoff = 2*0.959

        # need to fix for multiple pool sizes XXX - idfah
        cutoff /= float(self.poolSize)

        taps = cutoff * np.linspace(-radius, radius, self.filtOrder+1, dtype=self.dtype)
        impulseResponse = cutoff * np.sinc(taps) * win

        self.filters = []
        nReadoutLayers = 1 if self.nHidden is None else 2
        for ni, no in self.layerDims[:-nReadoutLayers]:
            noEmb = no*(self.filtOrder+1) # no outs after filter embedding

            filtMat = np.zeros(noEmb*2, dtype=self.dtype)
            filtMat[noEmb-1::-no] = impulseResponse

            # filters strided for embedding
            sz = filtMat.itemsize
            filtMat = npst.as_strided(filtMat, (no,noEmb), strides=(sz,sz))[::-1].T
            self.filters.append(filtMat.copy())

    def train(self, classData, optimFunc, **kwargs):
        x, g = label.indicatorsFromList(classData)
        x = np.require(x, requirements=['O', 'C'])
        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

        #dv = self.discrim(x, accum='mult')
        #self.normSoftmaxMean = dv.mean()
        #self.normSoftmaxStd = dv.std()
        #self.normSoftmaxMin = dv.min()
        #self.normSoftmaxMax = (dv-self.normSoftmaxMin).max()

    def parameters(self):
        return self.pw

    def evalConvs(self, x):
        x = util.segmat(x)

        c = x
        cs = []
        #for cw, width, filt in zip(self.cws, self.convWidths, self.filters):
        for l, cw in enumerate(self.cws):
            width = self.convWidths[l]
            phi = self.transFunc[l]
            poolSize = self.poolSize[l]

            if poolSize == 1:
                c = util.timeEmbed(c, lags=width-1, axis=1)
                #c = phi(c.dot(cw[:-1]) + cw[-1])
                c = phi(util.segdot(c, cw[:-1]) + cw[-1])

            elif self.poolMethod == 'stride':
                c = util.timeEmbed(c, lags=width-1, axis=1, stride=poolSize)
                #c = phi(c.dot(cw[:-1]) + cw[-1])
                c = phi(util.segdot(c, cw[:-1]) + cw[-1])

            elif self.poolMethod == 'average':
                c = util.timeEmbed(c, lags=width-1, axis=1)
                #c = phi(c.dot(cw[:-1]) + cw[-1])
                c = phi(util.segdot(c, cw[:-1]) + cw[-1])
                c = util.accum(c, poolSize, axis=1) / poolSize
                
            elif self.poolMethod == 'lanczos':
                c = util.timeEmbed(c, lags=width-1, axis=1)
                #c = phi(c.dot(cw[:-1]) + cw[-1])
                c = phi(util.segdot(c, cw[:-1]) + cw[-1])

                c = util.timeEmbed(c, lags=self.filtOrder, axis=1, stride=poolSize)
                #c = c.dot(self.filters[l])
                c = util.segdot(c, self.filters[l])

            cs.append(c)

        return cs

    def stepProbs(self, x):
        x = util.segmat(x)

        # evaluate convolutional layers
        c = self.evalConvs(x)[-1]

        # evaluate hidden layer
        #z = self.transFunc[-1](c.dot(self.hw[:-1]) + self.hw[-1]) if self.nHidden is not None else c
        z = self.transFunc[-1](util.segdot(c, self.hw[:-1]) + self.hw[-1]) if self.nHidden is not None else c

        # evaluate visible layer
        #v = z.dot(self.vw[:-1]) + self.vw[-1]
        v = util.segdot(z, self.vw[:-1]) + self.vw[-1]

        #return util.capZero(np.array([util.softmax(s) for s in v]))
        return util.capZero(
            util.softmax(v.reshape(-1, v.shape[-1])).reshape((v.shape[0], v.shape[1], -1)))

    def stepLikes(self, x):
        return np.log(self.stepProbs(x))

    def discrim(self, x, accum='prod'):
        x = util.segmat(x)
        accum = accum.lower()

        # sum loglikes (multiply probs) accumulation
        if accum in ('prod', 'mult'):
            return self.stepLikes(x).sum(axis=1)

        # sum probs accum accumulation
        elif accum in ('sum', 'add'):
            return self.stepProbs(x).sum(axis=1)

        # vote likes accumulation
        elif accum == 'vote':
            likes = self.stepLikes(x)

            votes = np.zeros_like(likes)
            maxi = np.argmax(likes, axis=2)
            for i,m in enumerate(maxi):
                votes[i][range(len(m)),m] = 1.0

            return votes.sum(axis=1)

        else:
            raise RuntimeError('Invalid discrim accum method: ' + str(accum))

    def probs(self, x, squash='softmax', accum='prod'):
        x = util.segmat(x)
        squash = squash.lower()

        dv = self.discrim(x, accum=accum)

        if squash == 'softmax':
            return util.softmax(dv)

        #elif squash == 'normsoftmax':
        #    #dv -= self.normSoftmaxMin
        #    #dv /= self.normSoftmaxMax
        #    dv -= self.normSoftmaxMean
        #    #dv /= 0.5*self.normSoftmaxStd
        #    dv /= self.normSoftmaxStd
        #    return util.softmax(dv)

        elif squash == 'frac':
            return dv / dv.sum(axis=1)[:,None]

        else:
            raise RuntimeError('Invalid probs squash method: ' + str(squash))

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
        g = util.colmat(g)

        likes = self.stepLikes(x)
        gs = np.repeat(g, likes.shape[1], axis=0).reshape(likes.shape)

        return -np.mean(gs*likes) + self.penaltyError()

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
            poolSize = self.poolSize[l]

            if poolSize == 1:
                c = util.timeEmbed(c, lags=width-1, axis=1)

            elif self.poolMethod == 'stride':
                c = util.timeEmbed(c, lags=width-1, axis=1, stride=poolSize)

            elif self.poolMethod in ('average', 'lanczos'):
                c = util.timeEmbed(c, lags=width-1, axis=1)

            c1 = util.bias(c)
            c1s.append(c1)

            #h = c1.dot(cw)
            h = util.segdot(c1, cw)
            cPrime = phi(h, 1)
            cPrimes.append(cPrime)

            c = phi(h)
       
            if poolSize == 1:
                pass
 
            elif self.poolMethod == 'average':
                c = util.accum(c, poolSize, axis=1) / poolSize

            elif self.poolMethod == 'lanczos':
                c = util.timeEmbed(c, lags=self.filtOrder, axis=1, stride=poolSize)
                #c = c.dot(self.filters[l])
                c = util.segdot(c, self.filters[l])

        c1 = util.bias(c)

        # evaluate hidden and visible layers
        if self.nHidden is None:
            #v = c1.dot(self.vw)
            v = util.segdot(c1, self.vw)
        else:
            #h = c1.dot(self.hw)
            h = util.segdot(c1, self.hw)
            z1 = util.bias(self.transFunc[-1](h))
            zPrime = self.transFunc[-1](h, 1)
            #v = z1.dot(self.vw)
            v = util.segdot(z1, self.vw)

        #probs = np.array([util.softmax(s) for s in v])
        probs = util.softmax(v.reshape(-1, v.shape[-1])).reshape((v.shape[0], v.shape[1], -1))
        gs = np.repeat(g, probs.shape[1], axis=0).reshape(probs.shape)

        # error components
        delta = (probs - gs) / probs.size

        if self.nHidden is None:
            # visible layer gradient
            c1f = c1.reshape((-1, c1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            vg[...] = c1f.T.dot(deltaf)
            vg += self.penaltyGradient(-1)

            #delta = delta.dot(self.vw[:-1].T)
            delta = util.segdot(delta, self.vw[:-1].T)

        else:
            # visible layer gradient
            z1f = z1.reshape((-1, z1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            vg[...] = z1f.T.dot(deltaf)
            vg += self.penaltyGradient(-1)

            #delta = delta.dot(self.vw[:-1].T) * zPrime
            delta = util.segdot(delta, self.vw[:-1].T) * zPrime

            # hidden layer gradient
            c1f = c1.reshape((-1, c1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            hg[...] = c1f.T.dot(deltaf)
            hg += self.penaltyGradient(-2)

            #delta = delta.dot(self.hw[:-1].T)
            delta = util.segdot(delta, self.hw[:-1].T)

        # backward pass for convolutional layers
        for l in range(self.nConvLayers-1, -1, -1):
            c1 = c1s[l]
            cPrime = cPrimes[l]
            poolSize = self.poolSize[l]

            if poolSize == 1:
                pass

            elif self.poolMethod == 'average':
                deltaPool = np.empty_like(cPrime)
                deltaPool[:,:delta.shape[1]*poolSize] = \
                    delta.repeat(poolSize, axis=1) / poolSize
                deltaPool[:,delta.shape[1]*poolSize:] = 0.0

                delta = deltaPool

            elif self.poolMethod == 'lanczos':
                filt = self.filters[l]

                #delta = delta.dot(filt.T)
                delta = util.segdot(delta, filt.T)

                deltaPoolShape = list(delta.shape)
                deltaPoolShape[1] *= poolSize
                deltaPool = np.zeros(deltaPoolShape, dtype=delta.dtype)
                deltaPool[:,::poolSize] = delta
                delta = deltaPool

                delta = deltaDeEmbedSum(delta, self.filtOrder+1)

            assert abs(delta.shape[1] - cPrime.shape[1]) <= poolSize
            delta = delta[:,:cPrime.shape[1]] * cPrime

            c1f = c1.reshape((-1, c1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            cgs[l][...] = c1f.T.dot(deltaf)
            cgs[l] += self.penaltyGradient(l)

            if l > 0: # won't propigate back to inputs
                #delta = delta.dot(self.cws[l][:-1].T)
                delta = util.segdot(delta, self.cws[l][:-1].T)

                if poolSize == 1:
                    pass

                elif self.poolMethod == 'stride':
                    deltaPoolShape = list(delta.shape)
                    deltaPoolShape[1] *= poolSize
                    deltaPool = np.zeros(deltaPoolShape, dtype=delta.dtype)
                    deltaPool[:,::poolSize] = delta
                    delta = deltaPool

                delta = deltaDeEmbedSum(delta, self.convWidths[l])

        if returnError:
            error = -np.mean(gs*np.log(util.capZero(probs))) + self.penaltyError()
            return error, pg
        else:
            return pg

class CNA(ConvolutionalNetworkAccum):
    pass

def demoCNA():
    def gauss_map(n, a=0.62, b=-0.5):
        v = np.empty(n)
        v[0] = np.random.uniform(0.05, 0.15)
        for i in range(1,n):
            v[i] = np.exp(-a*v[i-1]**2) + b
        return v
    ###s1 = np.vstack([gauss_map(len(x), a=6.2) for i in range(ns)])[:,:,None]
    ###s2 = np.vstack([gauss_map(len(x), a=6.0) for i in range(ns)])[:,:,None]

    if False:
        x = np.linspace(0.0, 6*np.pi, 256)
        #x = np.linspace(0.0, 2.0*np.pi, 105)

        ns1 = 30
        #s1 = np.array([np.vstack((np.sin(x+phaseShift),
        #                          np.cos(x+phaseShift))).T
        #                          for phaseShift in np.random.uniform(-np.pi, np.pi, size=ns1)])
        s1 = np.sin(x[None,:] + np.random.uniform(-np.pi, np.pi, size=(ns1,1)))
        s1 += np.random.normal(size=s1.shape, scale=0.8)

        ns2 = 50
        #s2 = np.array([np.vstack((np.sin(x+phaseShift),
        #                          np.sin(x+phaseShift))).T
        #                          for phaseShift in np.random.uniform(-np.pi, np.pi, size=ns2)])
        s2 = np.sin(2.0*x[None,:] + np.random.uniform(-np.pi, np.pi, size=(ns2,1)))
        s2 += np.random.normal(size=s2.shape, scale=0.1)

        s1 = s1[...,None]
        s2 = s2[...,None]
        #s1 = np.repeat(s1, 32).reshape((ns1,x.size,32))
        #s2 = np.repeat(s2, 32).reshape((ns2,x.size,32))

        trainData = [s1[:(ns1//2)], s2[:(ns2//2)]]
        testData = [s1[(ns1//2):], s2[(ns2//2):]]
    else:
        ns = 50

        nObs = 128
        x = np.arange(nObs)

        s1 = np.zeros((ns,nObs))
        i = np.random.randint(10,nObs-5-10, size=ns)
        s1[np.arange(ns),i] = 1.0
        s1[np.arange(ns),i+5] = 1.0

        s2 = np.zeros((ns,nObs))
        i = np.random.randint(10+3,nObs-10, size=ns)
        s2[np.arange(ns),i-3] = 1.0
        s2[np.arange(ns),i] = 1.0

        s1 = s1.astype(np.float32)
        s2 = s2.astype(np.float32)

        s1 = s1[...,None]
        s2 = s2[...,None]

        trainData = [s1[:(ns//2)], s2[:(ns//2)]]
        testData = [s1[(ns//2):], s2[(ns//2):]]

    print(s1.shape)
    print(s2.shape)

    print(trainData[0].shape, trainData[1].shape)

    standardizer = stand.ClassSegStandardizer(trainData)
    trainData = standardizer.apply(trainData)
    testData = standardizer.apply(testData)

    #model = CNA(trainData, convs=((2,11),(4,9),(6,7)), nHidden=2,
    model = CNA(trainData, convs=((4,9),(8,9)), nHidden=None,
                 poolSize=2, poolMethod='average', verbose=True,
                 optimFunc=optim.scg, maxIter=250, transFunc=transfer.rectifier,
                 #optimFunc=optim.rprop, maxIter=1000,
                 #optimFunc=optim.sciopt, method='Powell', maxIter=1000)
                 precision=1.0e-10, accuracy=0.0, pTrace=True, eTrace=True)

    print('Training Performance:')
    print('=======')
    print('Labels: ', model.labelKnown(trainData))
    print('ProbsA: ', model.probs(trainData[0]))
    print('ProbsB: ', model.probs(trainData[1]))
    print('CA:     ', model.ca(trainData))
    print('BCA:    ', model.bca(trainData))
    print('AUC:    ', model.auc(trainData))
    print()
    print('Test Performance:')
    print('=======')
    print('Labels: ', model.labelKnown(testData))
    print('ProbsA: ', model.probs(testData[0]))
    print('ProbsB: ', model.probs(testData[1]))
    print('CA:     ', model.ca(testData))
    print('BCA:    ', model.bca(testData))
    print('AUC:    ', model.auc(testData))
    print()

    fig = plt.figure(figsize=(20,6))
    axSigs = fig.add_subplot(3,3, 1)
    axSigs.plot(x, trainData[0][0].T.squeeze(), color='blue', linewidth=2)#, label=r'$\mathbf{sin}(x)$')
    axSigs.plot(x, trainData[0].T.squeeze(), color='blue', alpha=0.1, linewidth=2)
    axSigs.plot(x, 10.0+trainData[1][0].T.squeeze(), color='red', linewidth=2)#, label=r'$\mathbf{sin}(2x)$')
    axSigs.plot(x, 10.0+trainData[1].T.squeeze(), color='red', alpha=0.1, linewidth=2)
    axSigs.set_title('')
    axSigs.set_xlabel('Time')
    axSigs.set_ylabel('Signal')
    #axSigs.legend()
    axSigs.autoscale(tight=True)

    axETrace = fig.add_subplot(3,3, 2)
    eTrace = np.array(model.trainResult['eTrace'])
    axETrace.plot(eTrace)

    axPTrace = fig.add_subplot(3,3, 3)
    pTrace = np.array(model.trainResult['pTrace'])
    axPTrace.plot(pTrace)

    cs1 = model.evalConvs(trainData[0])
    cs2 = model.evalConvs(trainData[1])
    #for i in range(model.nConvLayers):
    for i in (0,1):
        axConvs = fig.add_subplot(3,3, 4+i)
        c1 = cs1[i][0,:,:]
        c2 = cs2[i][0,:,:]
        sep = util.colsep(np.vstack((c1,c2)))
        axConvs.plot(c1+sep, color='blue', linewidth=2, alpha=0.25)
        axConvs.plot(c2+sep, color='red', linewidth=2, alpha=0.25)

        axRespon = fig.add_subplot(3,3, 7+i)
        freqs, responses = zip(*[spsig.freqz(cw) for cw in model.cws[i].T])
        freqs = np.array(freqs)
        responses = np.array(responses)
        axRespon.plot(freqs.T, np.abs(responses).T)

    axProbs = fig.add_subplot(3,3, 6)
    probs1 = model.stepProbs(trainData[0])
    probs2 = model.stepProbs(trainData[1])
    p1 = probs1[0,:,0]
    p2 = probs2[0,:,1]
    axProbs.plot(p1, color='blue', linewidth=2)
    axProbs.plot(p2, color='red', linewidth=2)

    print('nParams: ', model.parameters().size)

    #for l,cw in enumerate(model.cws):
    #    plt.figure()
    #    plt.hist(cw.ravel())
    #    plt.title(str(l))

    #plt.figure()
    #plt.hist(model.vw.ravel())
    #plt.title('vw')

    #fig.tight_layout()

if __name__ == '__main__':
    demoCNA()
    plt.show()
