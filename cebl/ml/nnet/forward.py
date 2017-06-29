import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from cebl import util

from .. import optim
from .. import paraminit as pinit
from ..regression import Regression

from . import transfer


class ForwardNetwork(Regression, optim.Optable):
    def __init__(self, x, g, nHidden=10, transFunc=transfer.lecun,
                 weightInitFunc=pinit.lecun, penalty=None, elastic=1.0,
                 optimFunc=optim.scg, **kwargs):
        x = np.asarray(x)
        g = np.asarray(g)
        self.dtype = np.result_type(x.dtype, g.dtype)

        self.flattenOut = False if g.ndim > 1 else True

        Regression.__init__(self, util.colmat(x).shape[1],
                            util.colmat(g).shape[1])
        optim.Optable.__init__(self)

        self.nHidden = nHidden if util.isiterable(nHidden) else (nHidden,)
        self.nHLayers = len(self.nHidden)

        self.layerDims = [(self.nIn+1, self.nHidden[0])]
        for l in xrange(1, self.nHLayers):
            self.layerDims.append((self.nHidden[l-1]+1, self.nHidden[l]))
        self.layerDims.append((self.nHidden[-1]+1, self.nOut))

        self.transFunc = transFunc if util.isiterable(transFunc) \
                else (transFunc,) * self.nHLayers
        assert len(self.transFunc) == self.nHLayers

        views = util.packedViews(self.layerDims, dtype=self.dtype)
        self.pw  = views[0]
        self.hws = views[1:-1]
        self.vw  = views[-1]

        if not util.isiterable(weightInitFunc): 
            weightInitFunc = (weightInitFunc,) * (self.nHLayers+1)
        assert len(weightInitFunc) == (len(self.hws) + 1)

        # initialize weights
        for hw, wif in zip(self.hws, weightInitFunc):
            hw[...] = wif(hw.shape).astype(self.dtype, copy=False)
        self.vw[...] = weightInitFunc[-1](self.vw.shape).astype(self.dtype, copy=False)

        self.penalty = penalty
        if self.penalty is not None:
            if not util.isiterable(self.penalty):
                self.penalty = (self.penalty,) * (self.nHLayers+1)
        assert (self.penalty is None) or (len(self.penalty) == (len(self.hws) + 1))

        self.elastic = elastic if util.isiterable(elastic) \
                else (elastic,) * (self.nHLayers+1)
        assert (len(self.elastic) == (len(self.hws) + 1))

        # train the network
        if optimFunc is not None:
            self.train(x, g, optimFunc, **kwargs)

    def train(self, x, g, optimFunc, **kwargs):
        """Train the network to minimize the mean-squared-error between x and
        g using a given optimization routine.

        Args:
            x:              Input data.  A numpy array with shape
                            (nObs[,nDim]).

            g:              Target data.  A numpy array with shape
                            (nObs[,nDim]).

            optimFunc:      Function used to optimize the weight matrices.
                            See ml.optim for some candidate optimization
                            functions.

            kwargs:         Additional arguments passed to optimFunc.
        """
        x = np.asarray(x)
        g = np.asarray(g)

        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

    def parameters(self):
        """Return a 1d numpy array view of the parameters to optimize.
        This view will be modified in place.  This is part of the
        optim.Optable interface.
        """
        # return packed weights, generated in constructor
        return self.pw

    def evalHiddens(self, x):
        x = np.asarray(x)

        z = x
        zs = []
        for hw, phi in zip(self.hws, self.transFunc):
            z = phi(z.dot(hw[:-1]) + hw[-1])
            zs.append(z)

        return zs

    def eval(self, x):
        """Evaluate the network for given inputs.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nDim]).

        Returns:
            A numpy array with shape (nObs,nOut) containing the
            network outputs for each input in x.
        """
        x = np.asarray(x)

        z = self.evalHiddens(x)[-1]
        y = z.dot(self.vw[:-1]) + self.vw[-1]

        if self.flattenOut:
            y = y.ravel()

        return y

    def penaltyError(self):
        if self.penalty is None:
            return 0.0

        hwsf = [hw[:-1].ravel() for hw in self.hws]
        vwf = self.vw[:-1].ravel()

        weights = hwsf + [vwf,]

        totalPenalty = 0.0
        for weight, penalty, elastic in zip(weights, self.penalty, self.elastic):
            totalPenalty += ( elastic      * penalty * weight.dot(weight)/weight.size + # L2
                             (1.0-elastic) * penalty * np.mean(np.abs(weight)) ) # L1

        return totalPenalty

    def penaltyGradient(self, layer):
        if self.penalty is None:
            return 0.0

        weights = self.vw if layer == -1 else self.hws[layer]

        penalty = self.penalty[layer]
        elastic = self.elastic[layer]

        penMask = np.ones_like(weights)
        penMask[-1] = 0.0
        return ( elastic * 2.0 * penalty * penMask * weights/weights.size + # L2
                (1.0-elastic)  * penalty * penMask * np.sign(weights)/weights.size ) # L1

    def error(self, x, g):
        """Compute the mean-squared error (MSE) for given inputs and targets.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nDim]).

            g:  Target data.  A numpy array with shape (nObs[,nDim]).

        Returns:
            The scalar MSE.
        """
        x = np.asarray(x)
        g = np.asarray(g)

        # evaluate network
        y = self.eval(x)

        if self.flattenOut:
            g = g.ravel()

        # figure mse
        return np.mean((y-g)**2) + self.penaltyError()

    def gradient(self, x, g, returnError=True):
        """Compute the gradient of the mean-squared error with respect to the
        network weights for each layer and given inputs and targets.  Useful
        for optimization routines that make use of first-order gradients.

        Args:
            x:              Input data.  A numpy array with shape
                            (nObs[,nDim]).

            g:              Target data.  A numpy array with shape
                            (nObs[,nDim]).

            returnError:    If True (default) then also return the
                            mean-squared error.  This can improve
                            performance in some optimization routines
                            by avoiding an additional forward pass.

        Returns:
            If returnError is True, then return a tuple containing
            the error followed by a 1d numpy array containing the
            gradient of the packed weights.  If returnError is False,
            then only return the gradient.
        """
        x = np.asarray(x)
        g = np.asarray(g)

        if self.flattenOut:
            g = g.ravel()

        # packed views of the hidden and visible gradient matrices
        views = util.packedViews(self.layerDims, dtype=self.dtype)
        pg  = views[0]
        hgs = views[1:-1]
        vg  = views[-1]

        # forward pass
        z1 = util.bias(x)
        z1s = [z1]
        zPrimes = []
        for hw, phi in zip(self.hws, self.transFunc):
            h = z1.dot(hw)

            z1 = util.bias(phi(h))
            z1s.append(z1)

            zPrime = phi(h, 1)
            zPrimes.append(zPrime)

        y = z1.dot(self.vw)

        if self.flattenOut:
            y = y.ravel()

        # error components
        e = util.colmat(y - g)
        delta = 2.0 * e / e.size

        # visible layer gradient
        vg[...] = z1.T.dot(delta)
        vg += self.penaltyGradient(-1)

        # backward pass for hidden layers
        w = self.vw
        for l in xrange(self.nHLayers-1, -1, -1):
            delta = delta.dot(w[:-1,:].T) * zPrimes[l]
            hgs[l][...] = z1s[l].T.dot(delta)
            hgs[l] += self.penaltyGradient(l)
            w = self.hws[l]

        if returnError:
            error = np.mean(e**2) + self.penaltyError()
            return error, pg
        else:
            return pg

class FN(ForwardNetwork):
    pass

class ForwardNetworkL1(ForwardNetwork):
    def error(self, x, g):
        x = np.asarray(x)
        g = np.asarray(g)

        # evaluate network
        y = self.eval(x)

        if self.flattenOut:
            g = g.ravel()

        # figure mse
        return np.mean(np.abs(y-g)) + self.penaltyError()

    def gradient(self, x, g, returnError=True):
        x = np.asarray(x)
        g = np.asarray(g)

        if self.flattenOut:
            g = g.ravel()

        # packed views of the hidden and visible gradient matrices
        views = util.packedViews(self.layerDims, dtype=self.dtype)
        pg  = views[0]
        hgs = views[1:-1]
        vg  = views[-1]

        # forward pass
        z1 = util.bias(x)
        z1s = [z1]
        zPrimes = []
        for hw, phi in zip(self.hws, self.transFunc):
            h = z1.dot(hw)

            z1 = util.bias(phi(h))
            z1s.append(z1)

            zPrime = phi(h, 1)
            zPrimes.append(zPrime)

        y = z1.dot(self.vw)

        if self.flattenOut:
            y = y.ravel()

        # error components
        e = util.colmat(y - g)
        delta = np.sign(e) / e.size

        # visible layer gradient
        vg[...] = z1.T.dot(delta)
        vg += self.penaltyGradient(-1)

        # backward pass for hidden layers
        w = self.vw
        for l in xrange(self.nHLayers-1, -1, -1):
            delta = delta.dot(w[:-1,:].T) * zPrimes[l]
            hgs[l][...] = z1s[l].T.dot(delta)
            hgs[l] += self.penaltyGradient(l)
            w = self.hws[l]

        if returnError:
            error = np.mean(np.abs(e)) + self.penaltyError()
            return error, pg
        else:
            return pg

class FNL1(ForwardNetworkL1):
    pass

def demoFN1d():
    x = np.linspace(0.0, 5*np.pi, 100)[:,None]
    gClean = np.sin(x)
    g = gClean + np.random.normal(scale=0.3, size=x.shape)

    x -= np.mean(x)
    x /= np.std(x)
    g -= np.mean(g)
    g /= np.std(g)

    x = x.astype(np.float32)
    g = g.astype(np.float32)

    model = FN(x, g, nHidden=10,
              optimFunc=optim.scg, maxIter=250, precision=0.0,
              #sTrace=True, pTrace=True, eTrace=True, verbose=True)
              pTrace=True, eTrace=True, verbose=True)

    results = model.trainResult

    fig = plt.figure(figsize=(16,12))

    axFit = fig.add_subplot(3,2,1)
    axFit.plot(x, gClean, linewidth=2, color='blue')
    axFit.plot(x, g, linewidth=2, color='black')
    axFit.plot(x, model.eval(x), linewidth=2, color='red')
    axFit.legend(['True Target', 'Noisy Target', 'Network Output'])
    axFit.set_title('Network Output')
    axFit.set_xlabel('Input')
    axFit.set_ylabel('Output')

    axError = fig.add_subplot(3,2,2)
    axError.plot(results['eTrace'])
    axError.set_title('Training Error')
    axError.set_xlabel('Epoch')
    axError.set_ylabel('Mean-Squared Error')

    axHResponse = fig.add_subplot(3,2,3)
    axHResponse.plot(x, model.evalHiddens(x)[0], linewidth=2)
    axHResponse.set_title('Hidden Unit Response')
    axHResponse.set_xlabel('Input')
    axHResponse.set_ylabel('Hidden Unit Output')

    axHWeight = fig.add_subplot(3,2,4)
    img = axHWeight.imshow(model.hws[0], aspect='auto',
        interpolation='none', cmap=plt.cm.winter)
    cbar = plt.colorbar(img)
    cbar.set_label('Weight')
    axHWeight.set_title('Hidden Weights')
    axHWeight.set_xlabel('Hidden Unit')
    axHWeight.set_ylabel('Input')
    axHWeight.set_yticks(range(model.hws[0].shape[0]))
    axHWeight.set_yticklabels(list(range(1,model.hws[0].shape[0])) + ['bias'])

    pTrace = np.array(results['pTrace'])
    #sTrace = np.array(results['sTrace'])
    hwTrace = pTrace[:,:model.hws[0].size]
    #hwTrace = sTrace[:,:model.hws[0].size]
    #hwTrace = sTrace
    vwTrace = pTrace[:,model.vw.size:]

    axHWTrace = fig.add_subplot(3,2,5)
    axHWTrace.plot(hwTrace)
    axHWTrace.set_title('Hidden Weight Trace')
    axHWTrace.set_xlabel('Epoch')
    axHWTrace.set_ylabel('Weight')

    axVWTrace = fig.add_subplot(3,2,6)
    axVWTrace.plot(vwTrace)
    axVWTrace.set_title('Visible Weight Trace')
    axVWTrace.set_xlabel('Epoch')
    axVWTrace.set_ylabel('Weight')

    fig.tight_layout(pad=0.4)

def demoFN2d():
    def radialSinc(x):
        r = np.sqrt(np.sum(x**2, axis=1).ravel())

        s = np.ones_like(r)
        i = (np.abs(r) > np.finfo(r.dtype).eps)
        s[i] = np.sin(r[i]) / r[i]

        return s

    x1 = np.linspace(-6.0*np.pi, 6.0*np.pi, 150).astype(np.float32)
    x2 = np.linspace(-6.0*np.pi, 6.0*np.pi, 150).astype(np.float32)

    xx1, xx2 = np.meshgrid(x1, x2, copy=False)
    x = np.vstack((xx1.ravel(), xx2.ravel())).T

    xMean = x.mean(axis=0)
    xStd = x.std(axis=0)
    xStand = (x - xMean) / xStd

    g = radialSinc(x)
    #g += np.random.normal(scale=0.04, size=g.shape)
    gg = g.reshape((xx1.shape[0], xx2.shape[1]))

    gMean = g.mean()
    gStd = g.std()
    gStand = (g - gMean) / gStd

    model = FN(xStand, gStand, nHidden=(4,4,4), transFunc=transfer.gaussian,
            optimFunc=optim.scg, maxIter=1000, precision=0.0, accuracy=0.0,
            eTrace=True, pTrace=True, verbose=True)
    results = model.trainResult

    yStand = model.eval(xStand)
    y = (yStand * gStd) + gMean
    yy = y.reshape((xx1.shape[0], xx2.shape[1]))

    fig = plt.figure()

    axTargSurf = fig.add_subplot(2,3,1, projection='3d')
    targSurf = axTargSurf.plot_surface(xx1, xx2, gg, linewidth=0.0, cmap=plt.cm.jet)
    targSurf.set_edgecolor('black')

    axTargCont = fig.add_subplot(2,3,2)
    axTargCont.contour(x1, x2, gg, 40, color='black',
            marker='o', s=400, linewidth=3, cmap=plt.cm.jet)

    eTrace = results['eTrace']
    axError = fig.add_subplot(2,3,3)
    axError.plot(eTrace)
    axError.set_title('Training Error')
    axError.set_xlabel('Epoch')
    axError.set_ylabel('Mean-Squared Error')

    axPredSurf = fig.add_subplot(2,3,4, projection='3d')
    predSurf = axPredSurf.plot_surface(xx1, xx2, yy, linewidth=0.0, cmap=plt.cm.jet)
    predSurf.set_edgecolor('black')

    axPredCont = fig.add_subplot(2,3,5)
    axPredCont.contour(x1, x2, yy, 40, color='black',
            marker='o', s=400, linewidth=3, cmap=plt.cm.jet)

    pTrace = np.array(results['pTrace'])
    axHWTrace = fig.add_subplot(2,3,6)
    axHWTrace.plot(pTrace)
    axHWTrace.set_title('Weight Trace')
    axHWTrace.set_xlabel('Epoch')
    axHWTrace.set_ylabel('Weight')

if __name__ == '__main__':
    demoFN1d()
    #demoFN2d()
    plt.show()
