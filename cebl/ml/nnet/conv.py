import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.lib.stride_tricks as npst
import scipy as sp
import scipy.signal as spsig

from cebl import util
from cebl.ml import label
from cebl.ml import optim
from cebl.ml import paraminit as pinit
from cebl.ml import stand
from cebl.ml.classifier import Classifier

import transfer
from ddembed import *


class ConvolutionalNetwork(Classifier, optim.Optable):
    def __init__(self, classData, convs=((8,16), (16,8)), nHidden=8,
                 poolSize=2, poolMethod='stride', filtOrder=8,
                 transFuncs=transfer.lecun, weightInitFuncs=pinit.lecun,
                 penalties=None, elastics=1.0, optimFunc=optim.scg, **kwargs):
        seg = util.segmat(classData[0])
        nCls = len(classData)
        nSeg = seg.shape[0]
        nObs = seg.shape[1]
        nIn = seg.shape[2]

        Classifier.__init__(self, nIn, nCls)

        optim.Optable.__init__(self)

        self.dtype = np.result_type(*[cls.dtype for cls in classData])

        self.nConvHiddens, self.convWidths = zip(*convs)
        self.nConvLayers = len(convs)
        self.nHidden = nHidden
        self.poolSize = poolSize
        self.poolMethod = poolMethod.lower()

        self.layerDims = [(self.nIn*self.convWidths[0]+1, self.nConvHiddens[0]),]
        for l in xrange(1, self.nConvLayers):
            ni = self.nConvHiddens[l-1] * self.convWidths[l] + 1
            no = self.nConvHiddens[l]
            self.layerDims.append((ni, no))

        # figure length of inputs to final hidden layer
        # probably a more elegant way to do this XXX - idfah
        ravelLen = nObs
        if self.poolMethod == 'stride':
            for w in self.convWidths:
                ravelLen = int(np.ceil((ravelLen-w+1)/float(self.poolSize)))

        elif self.poolMethod == 'average':
            for w in self.convWidths:
                ravelLen = (ravelLen-w+1)//self.poolSize

        elif self.poolMethod == 'lanczos':
            for w in self.convWidths:
                ravelLen = int(np.ceil((ravelLen-filtOrder-w+1)/float(self.poolSize)))

        self.layerDims.append((ravelLen*self.nConvHiddens[-1]+1, self.nHidden))
        self.layerDims.append((self.nHidden+1, self.nCls))

        if self.poolMethod == 'lanczos':
            self.initLanczos(filtOrder)

        elif not self.poolMethod in ('stride', 'average'):
            raise Exception('Invalid poolMethod %s.' % str(self.poolMethod))

        self.transFuncs = transFuncs if util.isiterable(transFuncs) \
                else (transFuncs,) * (len(self.layerDims)-1)
        assert len(self.transFuncs) == (len(self.layerDims)-1)

        views = util.packedViews(self.layerDims, dtype=self.dtype)
        self.pw  = views[0]
        self.cws = views[1:-2]
        self.hw  = views[-2]
        self.vw  = views[-1]

        if not util.isiterable(weightInitFuncs):
            weightInitFuncs = (weightInitFuncs,) * (self.nConvLayers+2)
        assert len(weightInitFuncs) == (len(self.cws) + 2)

        self.penalties = penalties
        if self.penalties is not None:
            if not util.isiterable(self.penalties):
                self.penalties = (self.penalties,) * (self.nConvLayers+2)
        assert (self.penalties is None) or (len(self.penalties) == (len(self.cws) + 2))

        self.elastics = elastics if util.isiterable(elastics) \
                else (elastics,) * (self.nConvLayers+2)
        assert (len(self.elastics) == (len(self.cws) + 2))

        # initialize weights
        for cw, wif in zip(self.cws, weightInitFuncs):
            cw[...] = wif(cw.shape).astype(self.dtype, copy=False)
        self.hw[...] = weightInitFuncs[-2](self.hw.shape).astype(self.dtype, copy=False)
        self.vw[...] = weightInitFuncs[-1](self.vw.shape).astype(self.dtype, copy=False)

        # train the network
        if optimFunc is not None:
            self.train(classData, optimFunc, **kwargs)

    def initLanczos(self, filtOrder):
        self.filtOrder = filtOrder

        if self.filtOrder % 2 != 0:
             raise Exception('Invalid filtOrder: ' + str(self.filtOrder) +
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

        cutoff /= float(self.poolSize)

        taps = cutoff * np.linspace(-radius, radius, self.filtOrder+1, dtype=self.dtype)
        impulseResponse = cutoff * np.sinc(taps) * win

        self.filters = []
        for ni, no in self.layerDims[:-2]:
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

    def parameters(self):
        return self.pw

    def evalConvs(self, x):
        x = util.segmat(x)

        c = x
        cs = []
        for l, cw in enumerate(self.cws):
            width = self.convWidths[l]
            phi = self.transFuncs[l]

            if self.poolMethod == 'stride':
                c = util.timeEmbed(c, lags=width-1, axis=1, stride=self.poolSize)
                c = phi(c.dot(cw[:-1]) + cw[-1])

            elif self.poolMethod == 'average':
                c = util.timeEmbed(c, lags=width-1, axis=1)
                c = phi(c.dot(cw[:-1]) + cw[-1])
                c = util.accum(c, self.poolSize, axis=1) / self.poolSize

            elif self.poolMethod == 'lanczos':
                c = util.timeEmbed(c, lags=width-1, axis=1)
                c = phi(c.dot(cw[:-1]) + cw[-1])

                c = util.timeEmbed(c, lags=self.filtOrder, axis=1, stride=self.poolSize)
                c = c.dot(self.filters[l])

            cs.append(c)

        #print x.shape
        #print [c.shape for c in cs]
        return cs

    def probs(self, x):
        x = util.segmat(x)

        # evaluate convolutional layers
        c = self.evalConvs(x)[-1]

        # flatten to fully-connected
        c = c.reshape((c.shape[0], -1), order='F')

        # evaluate hidden layer
        z = self.transFuncs[-1](c.dot(self.hw[:-1]) + self.hw[-1])

        # evaluate visible layer
        v = z.dot(self.vw[:-1]) + self.vw[-1]

        # softmax to get probabilities
        return util.softmax(v)

    def penaltyError(self):
        if self.penalties is None:
            return 0.0

        cwsf = [cw[:-1].ravel() for cw in self.cws]
        hwf = self.hw[:-1].ravel()
        vwf = self.vw[:-1].ravel()

        weights = cwsf + [hwf,] + [vwf,]

        totalPenalty = 0.0
        for weight, penalty, elastic in zip(weights, self.penalties, self.elastics):
            totalPenalty += ( elastic      * penalty * weight.dot(weight)/weight.size + # L2
                             (1.0-elastic) * penalty * np.mean(np.abs(weight)) ) # L1

        return totalPenalty

    def penaltyGradient(self, layer):
        if self.penalties is None:
            return 0.0

        if layer == -1:
            weights = self.vw
        elif layer == -2:
            weights = self.hw
        else:
            weights = self.cws[layer]

        penalty = self.penalties[layer]
        elastic = self.elastics[layer]

        penMask = np.ones_like(weights)
        penMask[-1] = 0.0
        return ( elastic * 2.0 * penalty * penMask * weights/weights.size + # L2
                (1.0-elastic)  * penalty * penMask * np.sign(weights)/weights.size ) # L1

    def error(self, x, g):
        # evaluate network
        likes = np.log(util.capZero(self.probs(x)))

        return -np.mean(g*likes) + self.penaltyError()

    def gradient(self, x, g, returnError=True):
        x = util.segmat(x)
        g = util.colmat(g)

        # packed views of the hidden and visible gradient matrices
        views = util.packedViews(self.layerDims, dtype=self.dtype)
        pg  = views[0]
        cgs = views[1:-2]
        hg  = views[-2]
        vg  = views[-1]

        # forward pass
        c = x
        c1s = []
        cPrimes = []
        for l, cw in enumerate(self.cws):
            width = self.convWidths[l]
            phi = self.transFuncs[l]

            if self.poolMethod == 'stride':
                c = util.timeEmbed(c, lags=width-1, axis=1, stride=self.poolSize)

            elif self.poolMethod in ('average', 'lanczos'):
                c = util.timeEmbed(c, lags=width-1, axis=1)

            c1 = util.bias(c)
            c1s.append(c1)

            h = c1.dot(cw)
            cPrime = phi(h, 1)
            cPrimes.append(cPrime)

            c = phi(h)

            if self.poolMethod == 'average':
                c = util.accum(c, self.poolSize, axis=1) / self.poolSize

            elif self.poolMethod == 'lanczos':
                c = util.timeEmbed(c, lags=self.filtOrder, axis=1, stride=self.poolSize)
                c = c.dot(self.filters[l])

        # flatten to fully-connected
        c = c.reshape((c.shape[0], -1), order='F')
        c1 = util.bias(c)

        # evaluate hidden layer
        h = c1.dot(self.hw)
        z = self.transFuncs[-1](h)
        z1 = util.bias(z)
        zPrime = self.transFuncs[-1](h, 1)

        # evaluate visible layer
        v = z1.dot(self.vw)
        probs = util.softmax(v)

        # error components
        delta = (probs - g) / probs.size

        # visible layer gradient
        vg[...] = z1.T.dot(delta)
        vg += self.penaltyGradient(-1)

        # hidden layer gradient
        delta = delta.dot(self.vw[:-1].T) * zPrime
        hg[...] = c1.T.dot(delta)
        hg += self.penaltyGradient(-2)

        delta = delta.dot(self.hw[:-1].T)

        # unflatten deltas back to convolution
        delta = delta.reshape((delta.shape[0], -1, self.nConvHiddens[-1]), order='F')

        widths = list(self.convWidths[1:]) + [None,]

        # backward pass for convolutional layers
        for l in xrange(self.nConvLayers-1, -1, -1):
            c1 = c1s[l]
            cPrime = cPrimes[l]

            if self.poolMethod == 'average':
                deltaPool = np.empty_like(cPrime)
                deltaPool[:,:delta.shape[1]*self.poolSize] = \
                    delta.repeat(self.poolSize, axis=1) / self.poolSize
                deltaPool[:,delta.shape[1]*self.poolSize:] = 0.0

                delta = deltaPool

            elif self.poolMethod == 'lanczos':
                filt = self.filters[l]

                delta = delta.dot(filt.T)

                deltaPoolShape = list(delta.shape)
                deltaPoolShape[1] *= self.poolSize
                deltaPool = np.zeros(deltaPoolShape, dtype=delta.dtype)
                deltaPool[:,::self.poolSize] = delta
                delta = deltaPool

                delta = deltaDeEmbedSum(delta, self.filtOrder+1)

            assert abs(delta.shape[1] - cPrime.shape[1]) < self.poolSize
            delta = delta[:,:cPrime.shape[1]] * cPrime

            c1f = c1.reshape((-1, c1.shape[-1]))
            deltaf = delta.reshape((-1, delta.shape[-1]))
            cgs[l][...] = c1f.T.dot(deltaf)
            cgs[l] += self.penaltyGradient(l)

            if l > 0: # won't propigate back to inputs
                delta = delta.dot(self.cws[l][:-1].T)

                if self.poolMethod == 'stride':
                    deltaPoolShape = list(delta.shape)
                    deltaPoolShape[1] *= self.poolSize
                    deltaPool = np.zeros(deltaPoolShape, dtype=delta.dtype)
                    deltaPool[:,::self.poolSize] = delta
                    delta = deltaPool

                delta = deltaDeEmbedSum(delta, widths[l-1])

        if returnError:
            error = -np.mean(g*np.log(util.capZero(probs))) + self.penaltyError()
            return error, pg
        else:
            return pg

class CN(ConvolutionalNetwork):
    pass

def demoCN():
    def gauss_map(n, a=0.62, b=-0.5):
        v = np.empty(n)
        v[0] = np.random.uniform(0.05, 0.15)
        for i in xrange(1,n):
            v[i] = np.exp(-a*v[i-1]**2) + b
        return v
    ###s1 = np.vstack([gauss_map(len(x), a=6.2) for i in xrange(ns)])[:,:,None]
    ###s2 = np.vstack([gauss_map(len(x), a=6.0) for i in xrange(ns)])[:,:,None]

    if False:
        x = np.linspace(0.0, 6*np.pi, 256)
        #x = np.linspace(0.0, 2.0*np.pi, 105)

        ns1 = 30
        #s1 = np.array([np.vstack((np.sin(x+phaseShift),
        #                          np.cos(x+phaseShift))).T
        #                          for phaseShift in np.random.uniform(-np.pi, np.pi, size=ns1)])
        s1 = np.sin(x[None,:] + np.random.uniform(-np.pi, np.pi, size=(ns1,1)))
        s1 += np.random.normal(size=s1.shape, scale=0.1)

        ns2 = 50
        #s2 = np.array([np.vstack((np.sin(x+phaseShift),
        #                          np.sin(x+phaseShift))).T
        #                          for phaseShift in np.random.uniform(-np.pi, np.pi, size=ns2)])
        s2 = np.sin(2.0*x[None,:] + np.random.uniform(-np.pi, np.pi, size=(ns2,1)))
        s2 += np.random.normal(size=s2.shape, scale=0.8)

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
        #i = np.sort(i)
        s1[np.arange(ns),i] = 1.0
        s1[np.arange(ns),i+5] = 1.0

        s2 = np.zeros((ns,nObs))
        i = np.random.randint(10+3,nObs-10, size=ns)
        #i = np.sort(i)
        s2[np.arange(ns),i-3] = 1.0
        s2[np.arange(ns),i] = 1.0

        s1 = s1.astype(np.float32)
        s2 = s2.astype(np.float32)

        s1 = s1[...,None]
        s2 = s2[...,None]

        trainData = [s1[:(ns//2)], s2[:(ns//2)]]
        testData = [s1[(ns//2):], s2[(ns//2):]]
        #trainData = [s1[::2], s2[::2]]
        #testData = [s1[1::2], s2[1::2]]

    print s1.shape
    print s2.shape

    print trainData[0].shape, trainData[1].shape

    standardizer = stand.ClassSegStandardizer(trainData)
    trainData = standardizer.apply(trainData)
    testData = standardizer.apply(testData)

    model = CN(trainData, convs=((2,11),(4,9),(6,7)), nHidden=2,
               poolSize=2, poolMethod='average', filtOrder=6, verbose=True,
               optimFunc=optim.scg, maxIter=1000, transFuncs=transfer.lecun,
               precision=1.0e-16, accuracy=0.0, pTrace=True, eTrace=True)

    #from softmax import FNS
    #model = FNS(trainData, nHidden=20, optimFunc=optim.scg, maxIter=500, precision=1.0e-10, accuracy=1.0e-10, pTrace=True, eTrace=True, verbose=True)

    #from softmax import FSMN
    #trainData = [cls.reshape((cls.shape[0], -1)) for cls in trainData]
    #testData = [cls.reshape((cls.shape[0], -1)) for cls in testData]
    #model = FSMN(trainData, maxIter=500, nHidden=20,
    #             precision=0.0, accuracy=0.0, eTrace=True, pTrace=True, verbose=True)

    #from cebl.ml import ARC
    #model = ARC(trainData, order=20)

    print 'Training Performance:'
    print '======='
    print 'Labels: ', model.labelKnown(trainData)
    print 'CA:     ', model.ca(trainData)
    print 'BCA:    ', model.bca(trainData)
    print 'AUC:    ', model.auc(trainData)
    print
    print 'Test Performance:'
    print '======='
    print 'Labels: ', model.labelKnown(testData)
    print 'CA:     ', model.ca(testData)
    print 'BCA:    ', model.bca(testData)
    print 'AUC:    ', model.auc(testData)
    print

    nCol = max(model.nConvLayers, 3)

    fig = plt.figure(figsize=(20,6))
    axSigs = fig.add_subplot(3,nCol, 1)
    axSigs.plot(x, trainData[0][0].T.squeeze(), color='blue', linewidth=2)#, label=r'$\mathbf{sin}(x)$')
    axSigs.plot(x, trainData[0].T.squeeze(), color='blue', alpha=0.1, linewidth=2)
    axSigs.plot(x, 10.0+trainData[1][0].T.squeeze(), color='red', linewidth=2)#, label=r'$\mathbf{sin}(2x)$')
    axSigs.plot(x, 10.0+trainData[1].T.squeeze(), color='red', alpha=0.1, linewidth=2)
    axSigs.set_title('')
    axSigs.set_xlabel('Time')
    axSigs.set_ylabel('Signal')
    #axSigs.legend()
    axSigs.autoscale(tight=True)

    axETrace = fig.add_subplot(3,nCol, 2)
    eTrace = np.array(model.trainResult['eTrace'])
    axETrace.plot(eTrace)

    axPTrace = fig.add_subplot(3,nCol, 3)
    pTrace = np.array(model.trainResult['pTrace'])
    axPTrace.plot(pTrace)

    cs1 = model.evalConvs(trainData[0])
    cs2 = model.evalConvs(trainData[1])
    for i in xrange(model.nConvLayers):
        axConvs = fig.add_subplot(3,nCol, nCol+1+i)
        c1 = cs1[i][0,:,:]
        c2 = cs2[i][0,:,:]
        sep = util.colsep(np.vstack((c1,c2)))
        axConvs.plot(c1+sep, color='blue', linewidth=2, alpha=0.25)
        axConvs.plot(c2+sep, color='red', linewidth=2, alpha=0.25)
        #axConvs.set_xlim(0.0, 120)

        axRespon = fig.add_subplot(3,nCol, 2*nCol+1+i)
        freqs, responses = zip(*[spsig.freqz(cw) for cw in model.cws[i].T])
        freqs = np.array(freqs)
        responses = np.array(responses)
        axRespon.plot(freqs.T, np.abs(responses).T)

    #for l,cw in enumerate(model.cws):
    #    plt.figure()
    #    plt.hist(cw.ravel())
    #    plt.title(str(l))

    #plt.figure()
    #plt.hist(model.hw.ravel())
    #plt.title('hw')

    #plt.figure()
    #plt.hist(model.vw.ravel())
    #plt.title('vw')

    #fig.tight_layout()

if __name__ == '__main__':
    demoCN()
    plt.show()
