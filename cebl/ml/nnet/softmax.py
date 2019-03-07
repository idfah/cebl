"""Feedforward Neural Network with softmax visible layer for classification.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from cebl import util

from .. import label
from .. import optim
from .. import paraminit as pinit
from ..classifier import Classifier

from . import transfer


class ForwardNetworkSoftmax(Classifier, optim.Optable):
    """Feedforward Neural Network with softmax visible layer for classification.
    """
    def __init__(self, classData, nHidden=10, transFunc=transfer.lecun,
                 weightInitFunc=pinit.lecun, penalty=None, elastic=1.0,
                 optimFunc=optim.scg, **kwargs):
        """Construct a new feedforward neural network.

        Args:
            classData:      Training data.  This is a numpy array or list of numpy
                            arrays with shape (nCls, nObs[,nIn]).  If the
                            dimensions index is missing the data is assumed to be
                            one-dimensional.

            nHidden:        Number of hidden units.

            transFunc:      Hidden layer transfer function.  The default is
                            transfer.lecun.  See the transfer module for more.

            weightInitFunc: Function to initialize the weights in each layer.
                            If a single function is given, it will be repeated
                            for each layer with the visible layer last.  The
                            default function is the lecun function in the
                            paraminit module.  See the paraminit module
                            for more choices.

            penalty:

            elastic:       Penalty or weight decay.  The cost function is
                            then e + p * ||W||_2 where e is the prediction
                            error, p is the penalty and ||W||_2 denotes the
                            l2 norm of the weight matrix.  This regularizes
                            the network by pulling weights toward zero and
                            toward each other.
                                    1.0 is pure L2-norm
                                         --> between is elastic net
                                    0.0 is pure L1-norm

            optimFunc:      Function used to optimize the weight matrices.
                            If None, initial training will be skipped.
                            See ml.optim for some candidate optimization
                            functions.

            kwargs:         Additional arguments passed to optimFunc.

        Returns:
            A new, trained feedforward network.

        Refs:
            @incollection{lecun2012efficient,
              title={Efficient backprop},
              author={LeCun, Yann A and Bottou, L{\'e}on and Orr,
                      Genevieve B and M{\"u}ller, Klaus-Robert},
              booktitle={Neural networks: Tricks of the trade},
              pages={9--48},
              year={2012},
              publisher={Springer}
            }
        """
        Classifier.__init__(self, util.colmat(classData[0]).shape[1],
                            len(classData))
        optim.Optable.__init__(self)

        self.dtype = np.result_type(*[cls.dtype for cls in classData])

        self.nHidden = nHidden if util.isiterable(nHidden) else (nHidden,)
        self.nHLayers = len(self.nHidden)

        self.layerDims = [(self.nIn+1, self.nHidden[0])]
        for l in range(1, self.nHLayers):
            self.layerDims.append((self.nHidden[l-1]+1, self.nHidden[l]))
        self.layerDims.append((self.nHidden[-1]+1, self.nCls))

        self.transFunc = transFunc if util.isiterable(transFunc) \
                else (transFunc,) * self.nHLayers
        assert len(self.transFunc) == self.nHLayers

        views = util.packedViews(self.layerDims, dtype=self.dtype)
        self.pw = views[0]
        self.hws = views[1:-1]
        self.vw = views[-1]

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
            self.train(classData, optimFunc, **kwargs)

    def train(self, classData, optimFunc, **kwargs):
        """Train the network to minimize the mean-squared-error between x and
        g using a given optimization routine.

        Args:
            x:              Input data.  A numpy array with shape
                            (nObs[,nIn]).

            g:              Target data.  A numpy array with shape
                            (nObs[,nIn]).

            optimFunc:      Function used to optimize the weight matrices.
                            See ml.optim for some candidate optimization
                            functions.

            kwargs:         Additional arguments passed to optimFunc.
        """
        x, g = label.indicatorsFromList(classData)
        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

    def parameters(self):
        """Return a 1d numpy array view of the parameters to optimize.
        This view will be modified in place.  This is part of the
        optim.Optable interface.
        """
        # return packed weights, generated in constructor
        return self.pw

    def evalHiddens(self, x):
        """Evaluate the hidden layers for given inputs.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            A numpy array with shape (nObs, nHidden) containing the
            hidden layer activations for each input in x.
        """
        x = np.asarray(x)

        z = x
        zs = []
        for hw, phi in zip(self.hws, self.transFunc):
            z = phi(z.dot(hw[:-1]) + hw[-1])
            zs.append(z)

        return zs

    def probs(self, x):
        """Evaluate the network for given inputs.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs, nCls) containing the
            probability values.
        """
        x = np.asarray(x)

        z = self.evalHiddens(x)[-1]
        v = z.dot(self.vw[:-1]) + self.vw[-1]

        return util.softmax(v)

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
        """Compute the negative log likelyhood for given inputs and targets.
        Error function for Optable interface.

        Args:
            x:

            g:

        Returns:
            The scalar negative log likelyhood.
        """
        x = np.asarray(x)
        g = np.asarray(g)

        # evaluate network
        likes = np.log(util.capZero(self.probs(x)))

        return -np.mean(g*likes) + self.penaltyError()

    def gradient(self, x, g, returnError=True):
        """Compute the gradient of the mean-squared error with respect to the
        network weights for each layer and given inputs and targets.  Useful
        for optimization routines that make use of first-order gradients.

        Args:
            x:

            g:

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

        # packed views of the hidden and visible gradient matrices
        views = util.packedViews(self.layerDims, dtype=self.dtype)
        pg = views[0]
        hgs = views[1:-1]
        vg = views[-1]

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

        v = z1.dot(self.vw)
        probs = util.softmax(v)

        # error components
        delta = util.colmat(probs - g) / probs.size

        # visible layer gradient
        vg[...] = z1.T.dot(delta)
        vg += self.penaltyGradient(-1)

        # backward pass for hidden layers
        w = self.vw
        for l in range(self.nHLayers-1, -1, -1):
            delta = delta.dot(w[:-1,:].T) * zPrimes[l]
            hgs[l][...] = z1s[l].T.dot(delta)
            hgs[l] += self.penaltyGradient(l)
            w = self.hws[l]

        if returnError:
            error = -np.mean(g*np.log(util.capZero(probs))) + self.penaltyError()
            return error, pg
        else:
            return pg

class FNS(ForwardNetworkSoftmax):
    pass

def demoFNS2d():
    n1, n2, n3 = 500, 300, 300
    noiseScale = 0.8

    t1 = np.linspace(0.0, 2.0*np.pi, n1)
    x1 = 5.0*np.cos(t1) + np.random.normal(scale=noiseScale, size=n1)
    y1 = 5.0*np.sin(t1) + np.random.normal(scale=noiseScale, size=n1)
    red = np.vstack((x1, y1)).T

    x2 = np.linspace(-1.0, 3.0, n2)
    y2 = (x2-0.8)**2 - 2.5
    y2 += np.random.normal(scale=noiseScale, size=n2)
    green = np.vstack((x2, y2)).T

    x3 = np.linspace(-3.0, 1.0, n3)
    y3 = -(x3+0.8)**2 + 2.5
    y3 += np.random.normal(scale=noiseScale, size=n3)
    blue = np.vstack((x3, y3)).T

    classData = [red, green, blue]

    classData = [cls.astype(np.float32) for cls in classData]

    classData = [np.asfortranarray(cls) for cls in classData]

    # min and max training values
    mn = np.min(np.vstack(classData), axis=0)
    mx = np.max(np.vstack(classData), axis=0)

    # train model
    model = FNS(classData, nHidden=(4, 8, 16), optimFunc=optim.scg,
                transFunc=transfer.lecun, precision=1.0e-5,
                #transFunc=transfer.exprect, precision=1.0e-10,
                #transFunc=transfer.rectifierTwist, precision=1.0e-10,
                maxIter=1000, verbose=True)

    ##model = FNS(classData, nHidden=10, optimFunc=optim.minibatch,
    ##            penalty=0.01,
    ##            batchSize=15, maxRound=5, maxIter=5,
    ##            transFunc=transfer.lecun, precision=1.0e-10,
    ##            verbose=1)
    ##print('ca:', model.ca(classData))
    ##print('bca:', model.bca(classData))
    ##print('confusion:\n', model.confusion(classData))
    ##model.train(classData, optimFunc=optim.scg,
    ##            maxIter=10, precision=1.0e-10, verbose=True)

    # find class labels
    redLabel = model.label(red) # one at a time
    greenLabel = model.label(green)
    blueLabel = model.label(blue)

    print(model.probs(classData[0]).dtype)
    print(model.probs(classData[1]).dtype)
    print(model.probs(classData[2]).dtype)

    print('red labels\n-------')
    print(redLabel)
    print('\ngreen labels\n-------')
    print(greenLabel)
    print('\nblue labels\n-------')
    print(blueLabel)

    print('ca:', model.ca(classData))
    print('bca:', model.bca(classData))
    print('confusion:\n', model.confusion(classData))

    # first figure shows training data and class intersections
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Class Data')

    # training data
    ax.scatter(red[:,0], red[:,1],   color="red")
    ax.scatter(green[:,0], green[:,1], color="green")
    ax.scatter(blue[:,0], blue[:,1],  color="blue")

    # generate grid over training data
    sw = 0.025
    sx = np.arange(mn[0], mx[0], sw)
    sy = np.arange(mn[1], mx[1], sw)
    x, y = np.meshgrid(sx, sy)

    # get probability densities and labels for values in grid
    z = np.vstack((x.reshape((-1,)), y.reshape((-1,)))).T
    probs = model.probs(z)

    # red, green, blue and max probability densities
    pRed = np.reshape(probs[:,0,None], x.shape)
    pGreen = np.reshape(probs[:,1,None], x.shape)
    pBlue = np.reshape(probs[:,2,None], x.shape)
    pMax = np.reshape(np.max(probs, axis=1), x.shape)

    # class intersections
    diffRG = pRed   - pGreen
    diffRB = pRed   - pBlue
    diffGB = pGreen - pBlue
    ##ax.contour(x, y, diffRG, colors='black', levels=(0,))
    ##ax.contour(x, y, diffRB, colors='black', levels=(0,))
    ##ax.contour(x, y, diffGB, colors='black', levels=(0,))

    # second figure shows 3d plots of probability densities
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title('P(C = k)')

    # straight class colors for suface plots
    color = np.reshape([pRed, pGreen, pBlue], (3, x.shape[0], x.shape[1]))
    color = color.swapaxes(1, 2).T

    # flip colors to fade to white
    zro        = np.zeros_like(x)
    colorFlip  = np.ones((3, x.shape[0], x.shape[1]))
    colorFlip -= (np.array((zro, pRed, pRed)) +
                  np.array((pGreen, zro, pGreen)) +
                  np.array((pBlue, pBlue, zro)))
    colorFlip -= np.min(colorFlip)
    colorFlip /= np.max(colorFlip)
    colorFlip  = colorFlip.swapaxes(1, 2).T

    # probability density surface
    #surf = ax.plot_surface(x, y, pMax, cmap=matplotlib.cm.jet, linewidth=0)
    surf = ax.plot_surface(x, y, pMax, facecolors=colorFlip,
                           linewidth=0.02, shade=True)
    surf.set_edgecolor('black') # add edgecolor back in, bug?

    # third figure shows contours and color image of probability densities
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('max_K P(C = k)')

    #ax.pcolor(x, y, pMax)
    ax.imshow(colorFlip, origin='lower',
              extent=(mn[0], mx[0], mn[1], mx[1]), aspect='auto')

    # contours
    nLevel = 4
    cs = ax.contour(x, y, pMax, colors='black',
                    levels=np.linspace(np.min(pMax), np.max(pMax), nLevel))
    cs.clabel(fontsize=6)

    # fourth figure
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_title('argmax_K P(C = k)')

    labels = model.label(z)
    lMax = np.reshape(labels, x.shape)

    surf = ax.plot_surface(x, y, lMax, facecolors=colorFlip,
                           linewidth=0.02)#, antialiased=False)
    surf.set_edgecolor('black')

if __name__ == '__main__':
    demoFNS2d()
    plt.show()
