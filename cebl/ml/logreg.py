"""Linear logistic regression.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from cebl import util

from .classifier import Classifier
from . import label
from . import optim
from . import paraminit as pinit


class LogisticRegression(Classifier, optim.Optable):
    """Linear logistic regression.  Despite its name, logistic regression
    is typically used as a classifier.
    """
    def __init__(self, classData, weightInitFunc=pinit.runif,
                 optimFunc=optim.scg, **kwargs):
        """Create a new logistic regression classifier.

        Args:
            classData:      Training data.  This is a numpy array or list of numpy
                            arrays with shape (nCls, nObs[,nIn]).  If the
                            dimensions index is missing the data is assumed to
                            be one-dimensional.

            weightInitFunc: Function to initialize the model weights.
                            The default function is the runif function in the
                            paraminit module.  See the paraminit module for
                            more candidates.

            optimFunc:      Function used to optimize the model weights.
                            See ml.optim for some candidate optimization
                            functions.

            kwargs:         Additional arguments passed to optimFunc.


        Returns:

            A new, trained logistic regression classifier.
        """
        Classifier.__init__(self, util.colmat(classData[0]).shape[1],
                            len(classData))
        optim.Optable.__init__(self)

        self.dtype = np.result_type(*[cls.dtype for cls in classData])

        self.weights = weightInitFunc((self.nIn+1, self.nCls)).astype(self.dtype, copy=False)

        self.train(classData, optimFunc, **kwargs)

    def train(self, classData, optimFunc, **kwargs):
        """Train the network to maximize the log likelyhood
        using a given optimization routine.

        Args:
            classData:      Training data.  This is a numpy array or list of numpy
                            arrays with shape (nCls, nObs[,nIn]).  If the
                            dimensions index is missing the data is assumed to
                            be one-dimensional.

            optimFunc:      Function used to optimize the weight matrices.
                            If None, initial training will be skipped.
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
        return self.weights.ravel()

    def discrim(self, x):
        x = util.colmat(x)
        v = x @ (self.weights[:-1]) + self.weights[-1]
        return v

    def probs(self, x):
        """Compute class probabilities.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs, nIn) containing the probability values.
        """
        return util.softmax(self.discrim(x))

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

        likes = np.log(util.capZero(self.probs(x)))

        return -np.mean(g*likes)

    def gradient(self, x, g, returnError=True):
        x = np.asarray(x)
        g = np.asarray(g)

        probs = self.probs(x)

        delta = (probs - g) / probs.size

        grad = util.bias(x).T @ delta

        gf = grad.ravel()

        if returnError:
            err = -np.mean(g*np.log(util.capZero(probs)))

            return err, gf
        else:
            return gf

class LGR(LogisticRegression):
    pass

def demoLogisticRegression1d():
    c1 = np.random.normal(loc=-1.0, scale=0.3, size=60)
    c2 = np.random.normal(loc=0.0, scale=0.3, size=30)
    c3 = np.random.normal(loc=2.0, scale=0.7, size=90)
    classData = [c1, c2, c3]

    model = LogisticRegression(classData, verbose=True)

    c1Probs = model.probs(c1)
    c2Probs = model.probs(c2)
    c3Probs = model.probs(c3)

    print("c1:")
    print(model.label(c1))
    print("c2:")
    print(model.label(c2))
    print("c3:")
    print(model.label(c3))

    x = np.linspace(-2.0, 4.0, 500)
    xProbs = model.probs(x)

    plt.plot(x, xProbs, linewidth=2)
    plt.scatter(c1, np.zeros_like(c1), color="blue")
    plt.scatter(c2, np.zeros_like(c2), color="green")
    plt.scatter(c3, np.zeros_like(c3), color="red")

def demoLogisticRegression2d():
    # covariance matrix for each training class
    cov = [[1, -0.8],
           [-0.8, 1]]

    # red data
    red = np.random.multivariate_normal(
        (-1, -1), cov, 500)

    # green data
    green = np.random.multivariate_normal(
        (0, 0), cov, 300)

    # blue data
    blue = np.random.multivariate_normal(
        (1, 1), cov, 400)

    classData = [red, green, blue]

    # min and max training values
    mn = np.min(np.vstack(classData), axis=0)
    mx = np.max(np.vstack(classData), axis=0)

    # train model
    model = LogisticRegression(classData=classData, verbose=True)
        #optimFunc=optim.rprop, accuracy=0.0, precision=0.0, maxIter=100, penalty=0.3)
    #print(model.weights)
    #plt.imshow(np.abs(model.weights), interpolation="none")
    #plt.colorbar()

    # find class labels
    redLabel = model.label(red) # one at a time
    greenLabel = model.label(green)
    blueLabel = model.label(blue)

    print("red labels\n-------")
    print(redLabel)
    print("\ngreen labels\n-------")
    print(greenLabel)
    print("\nblue labels\n-------")
    print(blueLabel)

    print("ca:", model.ca(classData))
    print("bca:", model.bca(classData))
    print("confusion:\n", model.confusion(classData))

    # first figure shows training data and class intersections
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)

    # training data
    ax.scatter(red[:,0],   red[:,1],   color="red")
    ax.scatter(green[:,0], green[:,1], color="green")
    ax.scatter(blue[:,0],  blue[:,1],  color="blue")

    # generate grid over training data
    sw = 0.02
    sx = np.arange(mn[0], mx[0], sw)
    sy = np.arange(mn[1], mx[1], sw)
    x, y = np.meshgrid(sx, sy, copy=False)

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
    ax.contour(x, y, diffRG, colors="black", levels=(0,))
    ax.contour(x, y, diffRB, colors="black", levels=(0,))
    ax.contour(x, y, diffGB, colors="black", levels=(0,))

    # second figure shows 3d plots of probability densities
    ax = fig.add_subplot(2, 2, 2, projection="3d")

    # straight class colors for suface plots
    color = np.reshape([pRed, pGreen, pBlue], (3, x.shape[0], x.shape[1]))
    color = color.swapaxes(1, 2).T

    # flip colors to fade to white
    zro = np.zeros_like(x)
    colorFlip = np.ones((3, x.shape[0], x.shape[1]))
    colorFlip -= (
        np.array((zro, pRed, pRed)) +
        np.array((pGreen, zro, pGreen)) +
        np.array((pBlue, pBlue, zro))
    )
    colorFlip -= np.min(colorFlip)
    colorFlip /= np.max(colorFlip)
    colorFlip = colorFlip.swapaxes(1, 2).T

    # probability density surface
    #surf = ax.plot_surface(x, y, pMax, cmap=matplotlib.cm.jet, linewidth=0)
    surf = ax.plot_surface(x, y, pMax, facecolors=colorFlip,
                           linewidth=0.02, shade=True)
    surf.set_edgecolor("black") # add edgecolor back in, bug?

    # third figure shows contours and color image of probability densities
    ax = fig.add_subplot(2, 2, 3)

    #ax.pcolor(x, y, pMax)
    ax.imshow(colorFlip, origin="lower",
              extent=(mn[0], mx[0], mn[1], mx[1]), aspect="auto")

    # contours
    nLevel = 4
    cs = ax.contour(x, y, pMax, colors="black",
                    levels=np.linspace(np.min(pMax), np.max(pMax), nLevel))
    cs.clabel(fontsize=6)

    # fourth figure
    ax = fig.add_subplot(2, 2, 4, projection="3d")

    labels = model.label(z)
    lMax = np.reshape(labels, x.shape)

    surf = ax.plot_surface(x, y, lMax, facecolors=colorFlip,
                           linewidth=0.02)#, antialiased=False)
    surf.set_edgecolor("black")


class LogisticRegressionElastic(LogisticRegression):
    def __init__(self, classData, penalty=0.0, elastic=1.0, **kwargs):
        """
        Args:
            penalty:        l2 norm weight decay.  This regularizes
                            the model by pulling all weights toward
                            zero and toward each other.

            elastic:

            kwargs:
        """
        self.penalty = penalty
        self.elastic = elastic

        LogisticRegression.__init__(self, classData, **kwargs)

    def error(self, x, g):
        x = np.asarray(x)
        g = np.asarray(g)

        likes = np.log(util.capZero(self.probs(x)))

        pf = self.weights[:-1,:].ravel()
        return (-np.mean(g*likes) +
                self.elastic       * self.penalty * pf.dot(pf)/pf.size + # L2-norm penalty
                (1.0-self.elastic) * self.penalty * np.mean(np.abs(pf))) # L1-norm penalty

    def gradient(self, x, g, returnError=True):
        x = np.asarray(x)
        g = np.asarray(g)

        probs = self.probs(x)

        delta = (probs - g) / probs.size

        penMask = np.ones_like(self.weights)
        penMask[-1,:] = 0.0

        grad = (util.bias(x).T @ delta +
            self.elastic * 2.0 * self.penalty * penMask * self.weights / self.weights.size + # L2-norm penalty
            (1.0-self.elastic) * self.penalty * penMask * np.sign(self.weights) / self.weights.size) # L1-norm penalty

        gf = grad.ravel()

        if returnError:
            pf = self.weights[:-1,:].ravel()
            err = (-np.mean(g*np.log(util.capZero(probs))) +
                    self.elastic       * self.penalty * pf.dot(pf)/pf.size + # L2-norm penalty
                    (1.0-self.elastic) * self.penalty * np.mean(np.abs(pf))) # L1-norm penalty

            return err, gf
        else:
            return gf

class LGRE(LogisticRegressionElastic):
    pass


if __name__ == "__main__":
    demoLogisticRegression2d()
    plt.show()
