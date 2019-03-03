import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as spdist

from cebl import util

from .classifier import Classifier


class KNearestNeighbors(Classifier):
    def __init__(self, classData, k=1, distMetric='euclidean', **kwargs):
        Classifier.__init__(self, util.colmat(classData[0]).shape[1],
                            len(classData))

        self.k = k
        minObs = min([len(cls) for cls in classData])
        if self.k > minObs:
            raise RuntimeError('k=%d exceeds the number of examples in ' +
                'smallest training class %d.' % (k, minObs))

        if callable(distMetric):
            self.distFunc = lambda x1, x2: distMetric(x1, x2, **kwargs)
        else:
            self.distFunc = lambda x1, x2: spdist.cdist(x1, x2, metric=distMetric)

        self.train(classData)

    def train(self, classData):
        self.trainData = classData

    def probs(self, x):
        dists = np.hstack([self.distFunc(x, cls) for cls in self.trainData])
        indices = np.argpartition(dists, self.k, axis=1)[:,:self.k]

        #start = 0
        #votes = list()
        #for cls in self.trainData:
        #    end = start + cls.shape[0]
        #    votes.append(np.sum(np.logical_and(start <= indices, indices < end), axis=1))
        #    start = end

        ends = np.cumsum([len(cls) for cls in self.trainData])
        starts = ends - np.array([len(cls) for cls in self.trainData])
        votes = [np.sum(np.logical_and(start <= indices, indices < end), axis=1)
                 for start, end in zip(starts, ends)]
        votes = np.vstack(votes).T

        #probs = np.zeros((x.shape[0], self.nCls))
        #probs[np.arange(probs.shape[0]), np.argmax(votes, axis=1)] = 1.0
        ##probs = util.softmax(votes / float(self.k))
        probs = votes / float(self.k)

        return probs

class KNN(KNearestNeighbors):
    pass


def demoKNN():
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

    # min and max training values
    mn = np.min(np.vstack(classData), axis=0)
    mx = np.max(np.vstack(classData), axis=0)

    # train model
    model = KNN(classData, k=3)

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
    ax.scatter(red[:,0],   red[:,1],   color="red")
    ax.scatter(green[:,0], green[:,1], color="green")
    ax.scatter(blue[:,0],  blue[:,1],  color="blue")

    # generate grid over training data
    sw = 0.025
    sx = np.arange(mn[0], mx[0], sw)
    sy = np.arange(mn[1], mx[1], sw)
    x,y = np.meshgrid(sx, sy)

    # get probability densities and labels for values in grid
    z = np.vstack((x.reshape((-1,)), y.reshape((-1,)))).T
    probs = model.probs(z)

    # red, green, blue and max probability densities
    pRed = np.reshape(probs[:,0,None], x.shape)
    pGreen = np.reshape(probs[:,1,None], x.shape)
    pBlue = np.reshape(probs[:,2,None], x.shape)
    pMax = np.reshape(np.max(probs, axis=1), x.shape)

    # class intersections
    ##diffRG = pRed   - pGreen
    ##diffRB = pRed   - pBlue
    ##diffGB = pGreen - pBlue
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
    demoKNN()
    plt.show()
