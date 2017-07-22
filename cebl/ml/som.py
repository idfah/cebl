"""Self-organizing map.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as spdist
import time

from cebl import util


class SelfOrganizingMap(object):
    """Self-Organizing Map (SOM).
    """
    def __init__(self, x, latticeSize=(64,64), maxIter=5000,
                 distMetric='euclidean', learningRate=0.02, learningRateFinal=None,
                 radius=None, radiusFinal=None, weightRange=(0.0,1.0),
                 callback=None, verbose=False):
        """Construct a new Self-Organizing Map (SOM).

        Args:
            x:

            latticeSize:

            weightRange:

            maxIter:

            distMetric: can be any metric supported by scipy.spatial.distance.cdist

        References:
            http://www.ai-junkie.com/ann/som/som1.html
        """

        self.x = util.colmat(x)

        self.nDim = self.x.shape[1]

        self.latticeSize = latticeSize
        self.weightSize = tuple(list(latticeSize) + [self.nDim,])

        self.weights = np.random.uniform(weightRange[0], weightRange[1],
                                         size=self.weightSize)

        self.maxIter = maxIter

        def distFunc(x, w, metric=distMetric):
            w = w.reshape((-1, w.shape[-1]))
            return spdist.cdist(x, w, metric=distMetric)
        self.distFunc = distFunc

        # should support generic neighborhood function? XXX - idfah
        def neighborFunc(grid, center, sigma):
            # grid: (ls,ls,2)
            # center: (ns,2)
            # return: (ns,ls,ls)
            sumSqrs = np.power(grid - center, 2.0).sum(axis=-1)
            return np.exp(-sumSqrs / (2.0*sigma**2))
        self.neighborFunc = neighborFunc

        self.learningRate = learningRate
        if learningRateFinal is None:
            self.learningRateFinal = self.learningRate
        else:
            self.learningRateFinal = learningRateFinal
        self.learningRateDecay = ((self.learningRateFinal - self.learningRate) /
                float(self.maxIter))

        if radius is None:
            self.radius = np.max(latticeSize)/2.0
        else:
            self.radius = radius

        if radiusFinal is None:
            self.radiusFinal = 1.0
        else:
            self.radiusFinal = radiusFinal
        self.radiusDecay = ((self.radiusFinal - self.radius) /
                float(self.maxIter))

        self.callback = callback
        self.verbose = verbose

        self.train(x)

    def getWeights(self):
        return self.weights

    def getBMUIndices(self, x):
        dist = self.distFunc(x, self.weights)

        BMUIndex = np.unravel_index(dist.argmin(axis=1), self.latticeSize)
        BMUIndex = np.array(BMUIndex).T

        return BMUIndex

    def train(self, x):
        x = util.colmat(x)

        nObs = x.shape[0]

        grid = np.meshgrid(range(self.latticeSize[0]), range(self.latticeSize[1]))
        grid = np.array(grid).T

        if self.callback is not None:
            self.callback(0, self.weights, self.learningRate, self.radius)

        for iteration in xrange(1,self.maxIter+1):
            curObs = x[np.random.randint(0, nObs)]

            curLearningRate = self.learningRate + self.learningRateDecay * iteration
            curRadius = self.radius + self.radiusDecay * iteration

            BMUIndex = self.getBMUIndices(curObs[None,...])

            neighborHood = self.neighborFunc(grid, BMUIndex, curRadius)

            self.weights += curLearningRate * \
                neighborHood[...,None] * (curObs[None,None,:] - self.weights)

            if self.verbose:
                print '%d %.3f %.3f' % (iteration, curLearningRate, curRadius)

            if self.callback is not None:
                self.callback(iteration, self.weights, curLearningRate, curRadius)

class SOM(SelfOrganizingMap):
    pass

def demoSOM():
    #data = np.random.random((5000,3))
    #data =  np.array(
    #    ((1.0,0.0,0.0),
    #     (0.0,1.0,0.0),
    #     (0.0,0.0,1.0),
    #     (1.0,1.0,0.0),
    #     (1.0,0.0,1.0),
    #     (0.0,1.0,1.0),
    #     (1.0,1.0,1.0),
    #     (0.0,0.0,0.0)))

    n = 1000
    v = np.random.uniform(0.85, 1.0, size=(12,n))
    z = np.zeros(n)

    data = np.vstack((
        np.array((v[0], z,    z   )).T,
        np.array((z,    v[1], z   )).T,
        np.array((z,    z,    v[2])).T,
        np.array((v[3], v[4], z   )).T,
        np.array((v[5], z,    v[6])).T,
        np.array((z,    v[7], v[8])).T,
        np.array((v[9], v[10],v[11])).T))
        #np.array((z,    z,    z   )).T))

    def animFunc(iteration, weights, learningRate, radius):
        if iteration == 0:
            animFunc.fig = plt.figure(figsize=(10,10))
            animFunc.ax = animFunc.fig.add_subplot(1,1,1)
            animFunc.wimg = animFunc.ax.imshow(weights,
                    interpolation='none', origin='lower', animated=True)

            animFunc.fig.tight_layout()
            animFunc.fig.show()
            animFunc.fig.canvas.draw()

            animFunc.background = animFunc.fig.canvas.copy_from_bbox(animFunc.ax.bbox)

            animFunc.frame = 0

            time.sleep(1)

        if (iteration % 20) == 0:
            animFunc.fig.canvas.restore_region(animFunc.background)
            animFunc.wimg.set_array(weights)
            animFunc.ax.draw_artist(animFunc.wimg)
            animFunc.fig.canvas.blit(animFunc.ax.bbox)

            plt.savefig('frame-%04d.png' % animFunc.frame, dpi=200)
            animFunc.frame += 1

    som = SOM(data, latticeSize=(32,32), maxIter=20000,
              radius=16, radiusFinal=0.05, learningRate=0.05, learningRateFinal=0.005,
              callback=animFunc, verbose=True)

    rgb = np.vstack((
        np.array((1.0, 0.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
        np.array((0.0, 0.0, 1.0))))

    #bindex = np.vstack((
    #    som.getBMUIndices(r),
    #    som.getBMUIndices(g),
    #    som.getBMUIndices(b)))
    bindex = som.getBMUIndices(data)

    xlim = animFunc.ax.get_xlim()
    ylim = animFunc.ax.get_ylim()

    #plt.plot(bindex, linestyle='', marker='p', color=('red', 'green', 'blue'))
    animFunc.ax.scatter(bindex[:,1], bindex[:,0], c=data, s=100)

    animFunc.ax.set_xlim(xlim)
    animFunc.ax.set_ylim(ylim)
    
if __name__ == '__main__':
    demoSOM()
    plt.show()
