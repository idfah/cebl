import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import linalg as splinalg

from cebl import util

from . import optim
from . import paraminit as pinit
from .regression import Regression

class RidgeRegression(Regression):
    def __init__(self, x, g, penalty=0.0, pseudoInv=True):
        Regression.__init__(self, util.colmat(x).shape[1],
                            util.colmat(g).shape[1])

        self.dtype = np.result_type(x.dtype, g.dtype)

        self.penalty = penalty
        self.pseudoInv = pseudoInv

        self.train(x, g)

    def train(self, x, g):
        x = np.asarray(x)
        g = np.asarray(g)

        x1 = util.bias(x)

        penaltyMat = self.penalty * np.eye(x1.shape[1], dtype=self.dtype)
        penaltyMat[-1,-1] = 0.0

        a = x1.T @ x1 + penaltyMat
        b = x1.T @ g

        if self.pseudoInv is None:
            if np.linalg.cond(a) < 1.0/np.finfo(self.dtype).eps:
                pseudoInv = True
            else:
                pseudoInv = False
        else:
            pseudoInv = self.pseudoInv

        if pseudoInv:
            #self.weights, residuals, rank, s = \
            #    np.linalg.lstsq(a, b)

            #self.weights = np.linalg.pinv(a) @ b
            #self.weights = sp.linalg.pinv2(a) @ b

            # since x1.T @ x1 is symmetric, pinvh is equivalent but faster than pinv2
            self.weights = sp.linalg.pinvh(a) @ b

        else:
            #self.weights = sp.linalg.solve(a, b, sym_pos=True)
            self.weights = np.linalg.solve(a, b)

    def eval(self, x):
        x = np.asarray(x).reshape((x.shape[0], -1))
        return x @ self.weights[:-1] + self.weights[-1]

class RR(RidgeRegression):
    pass

class LinearRegression(RidgeRegression):
    pass

class LR(RidgeRegression):
    pass

def demoRidgeRegression1dQuad():
    x = np.linspace(0.0, 3.0, 50)
    y = (x-3)**2

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(x, y, marker='o', color='black')

    linearModel = RidgeRegression(x, y)

    ax.plot(x, linearModel.eval(x), color='green')

    x2 = x.repeat(2).reshape((-1,2))
    x2[:,1] **= 2
    quadraticModel = RidgeRegression(x2, y)

    ax.plot(x, quadraticModel.eval(x2), color='red')

def demoRidgeRegression1d():
    x = np.linspace(0.0, 3.0, 50)
    y = 2.0 * x
    y += np.random.normal(scale=0.3, size=y.size)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x, y)

    model = RidgeRegression(x, y)

    ax.plot(x, model.eval(x), color='green')


class LinearRegressionElastic(Regression, optim.Optable):
    def __init__(self, x, g,
                 elastic=1.0, penalty=0.0,
                 weightInitFunc=pinit.lecun,
                 optimFunc=optim.scg, **kwargs):
        x = np.asarray(x)
        g = np.asarray(g)
        self.dtype = np.result_type(x.dtype, g.dtype)

        if g.ndim > 1:
            self.flattenOut = False
        else:
            self.flattenOut = True

        self.elastic = elastic
        self.penalty = penalty

        Regression.__init__(self, util.colmat(x).shape[1],
                            util.colmat(g).shape[1])
        optim.Optable.__init__(self)

        self.weights = weightInitFunc((self.nIn+1, self.nOut)).astype(self.dtype, copy=False)

        if optimFunc is not None:
            self.train(x, g, optimFunc, **kwargs)

    def train(self, x, g, optimFunc, **kwargs):
        self.trainResult = optimFunc(self, x=x, g=g, **kwargs)

    def parameters(self):
        """Return a 1d numpy array view of the parameters to optimize.
        This view will be modified in place.  This is part of the
        optim.Optable interface.
        """
        return self.weights.ravel()

    def eval(self, x):
        x = np.asarray(x)
        y = x @ self.weights[:-1] + self.weights[-1]

        if self.flattenOut:
            y = y.ravel()

        return y

    def error(self, x, g):
        x = np.asarray(x)
        g = np.asarray(g)

        y = self.eval(x)

        if self.flattenOut:
            g = g.ravel()

        wf = self.weights[:-1,:].ravel()

        return (np.mean((y-g)**2) +
                self.elastic       * self.penalty * (wf @ wf)/wf.size + # L2-norm penalty
                (1.0-self.elastic) * self.penalty * np.mean(np.abs(wf))) # L1-norm penalty

    def gradient(self, x, g, returnError=True):
        x = np.asarray(x)
        g = np.asarray(g)

        if self.flattenOut:
            g = g.ravel()

        x1 = util.bias(x)
        y = x1 @ self.weights

        if self.flattenOut:
            y = y.ravel()

        e = util.colmat(y-g)
        delta = 2.0 * e / e.size

        penMask = np.ones_like(self.weights)
        penMask[-1,:] = 0.0
        grad = (x1.T @ delta +
            self.elastic * 2.0 * self.penalty * penMask * self.weights / self.weights.size +
            (1.0-self.elastic) * self.penalty * penMask * np.sign(self.weights) / self.weights.size)

        gf = grad.ravel()

        if returnError:
            wf = self.weights[:-1,:].ravel()

            error = (np.mean(e**2) +
                self.elastic       * self.penalty * (wf @ wf)/wf.size + # L2-norm penalty
                (1.0-self.elastic) * self.penalty * np.mean(np.abs(wf))) # L1-norm penalty
            return error, gf
        else:
            return gf

class LRE(LinearRegressionElastic):
    pass

def demoLinearRegressionElastic():
    n = 500

    x1 = np.linspace(-1.0, 1.0, n) + np.random.normal(scale=0.1, size=n) + 10.0
    x2 = np.linspace(1.0, -1.0, n) + np.random.normal(scale=0.1, size=n)
    x3 = np.random.normal(scale=0.1, size=n)
    x = np.vstack((x1,x2,x3)).T

    g1 = np.linspace(-1.0, 1.0, n) + np.random.normal(scale=0.1, size=n)
    g2 = np.linspace(1.0, -1.0, n) + np.random.normal(scale=0.1, size=n)
    g = np.vstack((g1,g2)).T

    model = LRE(x, g, elastic=0.0, penalty=0.7, verbose=True, optimFunc=optim.rprop,
                weightInitFunc=lambda size: np.random.uniform(-0.1, 0.1, size=size),
                precision=1e-16, maxIter=2000)

    y = model.eval(x)

    fig = plt.figure()
    axLines = fig.add_subplot(1,2,1)

    #ax.scatter(x, g1, color='red')
    #ax.scatter(x, g2, color='green')
    #ax.scatter(x, g3, color='blue')

    axLines.plot(y, color='green')

    axWeights = fig.add_subplot(1,2,2)
    img = axWeights.imshow(np.abs(model.weights), aspect='auto', interpolation='none')
    cbar = plt.colorbar(img)
    cbar.set_label('Weight Magnitude')

    axWeights.set_xticks((0,1))
    axWeights.set_xticklabels(('y1', 'y2'))
    axWeights.set_yticks((0,1,2,3))
    axWeights.set_yticklabels(('x1', 'x2', 'x3', 'bias'))

    print(model.weights)


if __name__ == '__main__':
    demoRidgeRegression1d()
    demoRidgeRegression1dQuad()
    demoLinearRegressionElastic()
    plt.show()
