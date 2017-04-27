"""Discriminant Analysis classifiers.
"""
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
from scipy import linalg as splinalg

from cebl import util

from classifier import Classifier


log2pi = np.log(2.0*np.pi)


class QuadraticDiscriminantAnalysis(Classifier):
    """Quadratic Discriminant Analysis Classifier.
    """
    def __init__(self, classData, average=0.0, shrinkage=0.0):
        """Construct a new Quadratic Discriminant Analysis (QDA) classifier.

        Args:
            classData:  Training data.  This is a numpy array or list of numpy
                        arrays with shape (nCls,nObs[,nIn]).  If the dimensions
                        index is missing the data is assumed to be
                        one-dimensional.

            average:    This parameter regularizes QDA by mixing the class
                        covariance matrices with the average covariance matrix.
                        A value of zero is pure QDA while a value of one
                        reduces to LDA.

            shrinkage:  This parameter regularizes QDA by shrinking each
                        covariance matrix toward the average eigenvalue of
                        the average covariance matrix.

        Returns:
            A trained QDA classifier.
        """
        Classifier.__init__(self, util.colmat(classData[0]).shape[1],
                            len(classData))

        self.dtype = np.result_type(*[cls.dtype for cls in classData])

        # average regularization parameter
        self.average = average

        # shrinkage regularization parameter
        self.shrinkage = shrinkage

        self.train(classData)

    def train(self, classData):
        """Train QDA classifier.

        Args:
            classData:  Training data.  This is a numpy array or list of numpy
                        arrays with shape (nCls,nObs[,nIn]).  If the dimensions
                        index is missing the data is assumed to be
                        one-dimensional.
        """
        # total number of observations
        if self.nIn == 1:
            totalObs = np.concatenate(classData).size
        else:
            totalObs = np.vstack(classData).shape[0]

        # log of class priors (nCls,)
        logPriors = np.log(np.array(
                [cls.shape[0]/float(totalObs) for cls in classData])).astype(self.dtype, copy=False)

        # class means (nCls,ndim)
        self.means = np.array(
                [np.mean(cls, axis=0) for cls in classData]).astype(self.dtype, copy=False)
        self.means = util.colmat(self.means)

        # covariance matrices
        covs = []

        # average covariance matrix
        avgCov = np.zeros((self.nIn, self.nIn), dtype=self.dtype)

        for i,cls in enumerate(classData):
            #dataZeroMean = cls - self.means[i]
            #cv = np.cov(dataZeroMean, rowvar=False).astype(self.dtype, copy=False)
            cv = np.cov(cls, rowvar=False).astype(self.dtype, copy=False)
            avgCov += cv
            covs.append(cv)

        # average covariance over number of classes
        avgCov /= self.nCls

        # mix with average
        covs = [(1.0-self.average) * cv +
                self.average * avgCov for cv in covs]

        # apply shrinkage
        covs = [((1.0-self.shrinkage) * cv +
                self.shrinkage * (np.trace(cv)/cv.shape[0])*np.identity(self.nIn, dtype=self.dtype))
                for cv in covs]

        self.invCovs = []
        self.intercepts = np.zeros(self.nCls, dtype=self.dtype)

        for i,cv in enumerate(covs):
            ##cvi = sp.linalg.pinvh(cv)
            #try:
            #    cvi = np.linalg.inv(cv)
            #except np.linalg.LinAlgError as e:
            #    raise Exception('Failed to invert covariance matrix, consider using shrinkage.')

            try:
                cvi = sp.linalg.pinvh(cv)
            except Exception as e:
                raise Exception('Failed to invert covariance matrix, consider using shrinkage.')

            self.invCovs.append(cvi)

            sign, logDet = np.linalg.slogdet(cv)
            if sign == 0:
                raise Exception('Covariance matrix has zero determinant, consider using shrinkage.')

            #self.intercepts[i] = logDet - 2.0*logPriors[i]

            #self.intercepts[i] = -0.5 * (self.nCls*log2pi + logDet) + logPriors[i]
            #self.intercepts[i] = -0.5 * (self.nIn*log2pi + logDet) + logPriors[i] # works
            self.intercepts[i] = -0.5*logDet + logPriors[i]

    def discrim(self, x):
        """Compute discriminant values.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs,nCls) containing the discriminant values.

        Notes:
            These values are the log of the evaluated discriminant functions
            with terms cancelled on both sides.  If you want probabilities,
            try the prob method instead.
        """
        x = util.colmat(x)

        # number of observations
        nObs = x.shape[0]

        # (nObs,nCls)
        dv = np.zeros((nObs,self.nCls), dtype=self.dtype)

        # could probably vectorize this? XXX - idfah
        for i in xrange(self.nCls):
            zm = x - self.means[i]
            dv[:,i] = np.sum(zm.dot(self.invCovs[i]) * zm, axis=1)
        dv *= -0.5
        dv += self.intercepts

        # (nObs,nCls)
        return dv

    def logDens(self, x):
        # find discriminant values
        dv = self.discrim(x)

        # find log densities by adding back in canceled terms
        return -0.5 * self.nCls*log2pi + dv

    def dens(self, x):
        """Compute class probability densities.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs,nCls) containing the density values.

        Notes:
            This is slower and less precise than discrim.  Only use probs if you
            need the class probability densities.
        """
        return np.exp(self.logDens(x))

    def probs(self, x):
        """Compute class probabilities.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs,nCls) containing the probability values.

        Notes:
            This is less precise than discrim.  Only use probs if you
            need the class probabilites for each observation.
        """
        # log probability densities
        logDens = self.logDens(x)

        # density_i*P(C=i) / sum_j(density_j*P(C=j))
        mx = np.max((np.max(logDens), 0.0))
        dens = util.capZero(np.exp(logDens-mx))
        return dens / dens.sum(axis=1)[:,None]

class QDA(QuadraticDiscriminantAnalysis):
    pass

def demoQDA2d():
    """QDA Example.
    """
    # covariance matrices
    covRed   = [[1,-0.9],
                [-0.9,1]]

    covGreen = [[0.8,-0.5],
                [-0.5,0.8]]

    covBlue  = [[0.3,0.0],
                [0.0,0.3]]

    # red data
    red = np.random.multivariate_normal(
        (-1.2,-1.2), covRed, 500)

    # green data
    green = np.random.multivariate_normal(
        (0,0), covGreen, 300)

    # blue data
    blue = np.random.multivariate_normal(
        (1.5,1.5), covBlue, 400)

    data = [red,green,blue]
    #data = [cls.astype(np.float32) for cls in data]

    # min and max training values
    mn = np.min(np.vstack((red,green,blue)), axis=0)
    mx = np.max(np.vstack((red,green,blue)), axis=0)

    # train model
    model = QuadraticDiscriminantAnalysis(data)

    # find class labels
    redLabel   = model.label(red) # one at a time
    greenLabel = model.label(green)
    blueLabel  = model.label(blue)

    print 'red labels\n-------'
    print redLabel
    print redLabel.shape
    print '\ngreen labels\n-------'
    print greenLabel
    print greenLabel.shape
    print '\nblue labels\n-------'
    print blueLabel
    print blueLabel.shape

    print 'ca: ', model.ca(data)
    print 'bca: ', model.bca(data)
    print 'confusion:\n', model.confusion(data)

    # first figure shows training data and class intersections
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)

    # training data
    ax.scatter(red[:,0],   red[:,1],   color="red")
    ax.scatter(green[:,0], green[:,1], color="green")
    ax.scatter(blue[:,0],  blue[:,1],  color="blue")

    # generate grid over training data
    sw = 0.02
    sx = np.arange(mn[0],mx[0], sw)
    sy = np.arange(mn[1],mx[1], sw)
    x,y = np.meshgrid(sx,sy)

    # get probabilities and labels for values in grid
    z = np.vstack((x.reshape((-1,)), y.reshape((-1,)))).T
    probs = model.probs(z)

    # red, green, blue and max probabilities
    pRed   = np.reshape(probs[:,0,np.newaxis], x.shape)
    pGreen = np.reshape(probs[:,1,np.newaxis], x.shape)
    pBlue  = np.reshape(probs[:,2,np.newaxis], x.shape)
    pMax   = np.reshape(np.max(probs, axis=1), x.shape)

    # red, green, blue and max probability densities
    densities = model.dens(z)
    dRed   = np.reshape(densities[:,0,np.newaxis], x.shape)
    dGreen = np.reshape(densities[:,1,np.newaxis], x.shape)
    dBlue  = np.reshape(densities[:,2,np.newaxis], x.shape)
    dMax   = np.reshape(np.max(densities, axis=1), x.shape)

    # class intersections
    diffRG = pRed   - pGreen
    diffRB = pRed   - pBlue
    diffGB = pGreen - pBlue
    ax.contour(x, y, diffRG, colors='black', levels=(0,))
    ax.contour(x, y, diffRB, colors='black', levels=(0,))
    ax.contour(x, y, diffGB, colors='black', levels=(0,))

    # second figure shows 3d plots of probability densities
    ax = fig.add_subplot(2,2,2, projection='3d')

    # straight class colors for suface plots
    color = np.reshape([dRed,dGreen,dBlue], (3, x.shape[0],x.shape[1]))
    color = color.swapaxes(1,2).T

    # flip colors to fade to white
    zro        = np.zeros_like(x)
    colorFlip  = np.ones((3, x.shape[0], x.shape[1]))
    colorFlip -= (np.array((zro,dRed,dRed)) +
                  np.array((dGreen,zro,dGreen)) +
                  np.array((dBlue,dBlue,zro)))
    colorFlip -= np.min(colorFlip)
    colorFlip /= np.max(colorFlip)
    colorFlip  = colorFlip.swapaxes(1,2).T

    # probability density surface
    #surf = ax.plot_surface(x,y, dMax, cmap=matplotlib.cm.jet, linewidth=0)
    surf = ax.plot_surface(x,y, dMax, facecolors=colorFlip,
                           linewidth=0.02, shade=True)
    surf.set_edgecolor('black') # add edgecolor back in, bug?

    # third figure shows 3d plots of probabilities
    ax = fig.add_subplot(2,2,3, projection='3d')

    # straight class colors for suface plots
    color = np.reshape([pRed,pGreen,pBlue], (3, x.shape[0],x.shape[1]))
    color = color.swapaxes(1,2).T

    # flip colors to fade to white
    zro        = np.zeros_like(x)
    colorFlip  = np.ones((3, x.shape[0], x.shape[1]))
    colorFlip -= (np.array((zro,pRed,pRed)) +
                  np.array((pGreen,zro,pGreen)) +
                  np.array((pBlue,pBlue,zro)))
    colorFlip -= np.min(colorFlip)
    colorFlip /= np.max(colorFlip)
    colorFlip  = colorFlip.swapaxes(1,2).T

    # probability density surface
    #surf = ax.plot_surface(x,y, pMax, cmap=matplotlib.cm.jet, linewidth=0)
    surf = ax.plot_surface(x,y, pMax, facecolors=colorFlip,
                           linewidth=0.02, shade=True)
    surf.set_edgecolor('black') # add edgecolor back in, bug?
    """
    # third figure shows contours and color image of probability densities
    ax = fig.add_subplot(2,2,3)

    #ax.pcolor(x,y,pMax)
    ax.imshow(colorFlip, origin='lower',
              extent=(mn[0],mx[0],mn[1],mx[1]), aspect='auto')

    # contours 
    nLevel = 6
    cs = ax.contour(x, y, pMax, colors='black',
                    levels=np.linspace(np.min(pMax),np.max(pMax),nLevel))
    cs.clabel(fontsize=6)
    """

    # fourth figure
    ax = fig.add_subplot(2,2,4, projection='3d')

    labels = model.label(z)
    lMax   = np.reshape(labels, x.shape)

    surf = ax.plot_surface(x,y, lMax, facecolors=colorFlip,
                           linewidth=0.02)#, antialiased=False)
    #surf.set_edgecolor(np.vstack(color))
    surf.set_edgecolor('black')

    fig.tight_layout()


#covCache = util.Cache(2)
class LinearDiscriminantAnalysis(Classifier):
    """Linear Discriminant Analysis Classifier.
    """
    def __init__(self, classData, shrinkage=0):
        """Construct a new Linear Discriminant Analysis (LDA) classifier.

        Args:
            classData:  Training data.  This is a numpy array or list of numpy
                        arrays with shape (nCls,nObs[,nIn]).  If the dimensions
                        index is missing the data is assumed to be
                        one-dimensional.

            shrinkage:  This parameter regularizes LDA by shrinking the average
                        covariance matrix toward its average eigenvalue:
                            covariance = (1-shrinkage)*covariance +
                            shrinkage*averageEigenvalue*identity
                        Behavior is undefined if shrinkage is outside [0,1].
                        This parameter has no effect if average is 0.

        Returns:
            A trained LDA classifier.
        """
        Classifier.__init__(self, util.colmat(classData[0]).shape[1],
                            len(classData))

        self.dtype = np.result_type(*[cls.dtype for cls in classData])

        self.shrinkage = shrinkage

        self.train(classData)

    def train(self, classData):
        """Train an LDA classifier.

        Args:
            classData:  Training data.  This is a numpy array or list of numpy
                        arrays with shape (nCls,nObs[,nIn]).  If the dimensions
                        index is missing the data is assumed to be
                        one-dimensional.
        """
        # total number of observations
        if self.nIn == 1:
            totalObs = np.concatenate(classData).size
        else:
            totalObs = np.vstack(classData).shape[0]

        # class priors (nCls,)
        logPriors = np.log(np.array(
                [cls.shape[0]/float(totalObs) for cls in classData])).astype(self.dtype, copy=False)

        # class means (nCls,ndim)
        means = np.array([np.mean(cls, axis=0) for cls in classData]).astype(self.dtype, copy=False)
        means = util.colmat(means)

        # average covariance matrix starts with zeros
        self.avgCov = np.zeros((self.nIn, self.nIn), dtype=self.dtype)

        # sum up class covariances
        for i,cls in enumerate(classData):
            self.avgCov += np.cov(cls, rowvar=False)
            #key = util.hashArray(cls)
            #cov = covCache[key]
            #if cov is None:
            #    cov = np.cov(cls, rowvar=False)
            #    covCache[key] = cov
            #    #print 'cache miss'
            #self.avgCov += cov

        # average covariance over number of classes
        self.avgCov /= self.nCls

        # apply shrinkage
        eigAvg = np.trace(self.avgCov)/self.avgCov.shape[0]
        self.avgCov = ((1.0 - self.shrinkage) * self.avgCov +
            self.shrinkage * eigAvg * np.identity(self.nIn, dtype=self.dtype))

        ##self.invCov = sp.linalg.pinvh(self.avgCov)
        #try:
        #    self.invCov = np.linalg.inv(self.avgCov)
        #except np.linalg.LinAlgError as e:
        #    raise Exception('Failed to invert covariance matrix, consider using shrinkage.')

        try:
            self.invCov = sp.linalg.pinvh(self.avgCov)
        except Exception as e:
            raise Exception('Failed to invert covariance matrix, consider using shrinkage.')

        sign, self.logDet = np.linalg.slogdet(self.avgCov)
        if sign == 0:
            raise Exception('Covariance matrix has zero determinant, consider using shrinkage.')

        # model coefficients
        # (ndim,nCls) = (ndim,ndim) x (ndim,nCls)
        self.weights = self.invCov.dot(means.T)

        # model intercepts (nCls,)
        #self.intercepts = np.array([-0.5 * means[cls,:].dot(self.weights[:,cls]) + logPriors[cls]
        #                  for cls in xrange(self.nCls)])
        self.intercepts = -0.5 * np.sum(self.weights * means.T, axis=0) + logPriors

    def parameters(self):
        return np.vstack((self.weights, self.intercepts))

    def discrim(self, x):
        """Compute discriminant values.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs,nCls) containing the discriminant values.

        Notes:
            These values are the log of the evaluated discriminant functions
            with terms cancelled on both sides.  If you want probabilities,
            try the prob method instead.
        """
        x = util.colmat(x)

        # discriminant values
        # (nObs,nCls) = (nObs,ndim) x (ndim,nCls) + (nObs,nCls)
        dv = x.dot(self.weights) + self.intercepts.reshape((1,-1))

        return dv

    def logDens(self, x):
        # find discriminant values
        dv = self.discrim(x)

        # find class probability densities by adding back in canceled terms
        xSx = np.sum(x.dot(self.invCov) * x, axis=1).reshape((-1,1))

        return -0.5 * (self.nCls*log2pi + self.logDet + xSx) + dv

    def dens(self, x):
        """Compute class probability densities.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs,nCls) containing the density values.

        Notes:
            This is slower and less precise than discrim.  Only use probs if you
            need the class probability densities.
        """
        return np.exp(self.logDens(x))

    def probs(self, x):
        """Compute class probabilities.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            Numpy array with shape (nObs,nCls) containing the probability values.

        Notes:
            This is less precise than discrim.  Only use probs if you
            need the class probabilites for each observation.
        """
        # log probability densities
        logDens = self.logDens(x)

        # density_i*P(C=i) / sum_j(density_j*P(C=j))
        mx = np.max((np.max(logDens), 0.0))
        dens = util.capZero(np.exp(logDens-mx))
        return dens / dens.sum(axis=1)[:,None]

    # hack, doesn't work for QDA or other algorithms where discrim is not comparable XXX - idfah
    # handles ties better since probs may be equal within precision but not log likelihoods (discrims)
    def auc(self, classData, *args, **kwargs):
        return util.auc(self.discrimKnown(classData, *args, **kwargs))
    def roc(self, classData, *args, **kwargs):
        return util.roc(self.discrimKnown(classData, *args, **kwargs))

class LDA(LinearDiscriminantAnalysis):
    pass

def demoLDA2d():
    """LDA 2d example.
    """
    # covariance matrix for each training class
    cov = [[1,-0.8],
           [-0.8,1]]

    # red data
    red = np.random.multivariate_normal(
        (-1,-1), cov, 500)

    # green data
    green = np.random.multivariate_normal(
        (0,0), cov, 300)

    # blue data
    blue = np.random.multivariate_normal(
        (1,1), cov, 400)

    data = [red,green,blue]

    # min and max training values
    mn = np.min(np.vstack((red,green,blue)), axis=0)
    mx = np.max(np.vstack((red,green,blue)), axis=0)

    # train model
    model = LinearDiscriminantAnalysis(data, shrinkage=0)

    # find class labels
    redLabel   = model.label(red) # one at a time
    greenLabel = model.label(green)
    blueLabel  = model.label(blue)

    print 'red labels\n-------'
    print redLabel
    print '\ngreen labels\n-------'
    print greenLabel
    print '\nblue labels\n-------'
    print blueLabel

    print 'ca: ', model.ca(data)
    print 'bca: ', model.bca(data)
    print 'confusion:\n', model.confusion(data)

    # first figure shows training data and class intersections
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)

    # training data
    ax.scatter(red[:,0],   red[:,1],   color="red")
    ax.scatter(green[:,0], green[:,1], color="green")
    ax.scatter(blue[:,0],  blue[:,1],  color="blue")

    # generate grid over training data
    sw = 0.02
    sx = np.arange(mn[0],mx[0], sw)
    sy = np.arange(mn[1],mx[1], sw)
    x,y = np.meshgrid(sx,sy)

    # get probability probabilities and labels for values in grid
    z = np.vstack((x.reshape((-1,)), y.reshape((-1,)))).T
    probs = model.probs(z)

    # red, green, blue and max probabilities
    pRed   = np.reshape(probs[:,0,np.newaxis], x.shape)
    pGreen = np.reshape(probs[:,1,np.newaxis], x.shape)
    pBlue  = np.reshape(probs[:,2,np.newaxis], x.shape)
    pMax   = np.reshape(np.max(probs, axis=1), x.shape)

    # class intersections
    diffRG = pRed   - pGreen
    diffRB = pRed   - pBlue
    diffGB = pGreen - pBlue
    ax.contour(x, y, diffRG, colors='black', levels=(0,))
    ax.contour(x, y, diffRB, colors='black', levels=(0,))
    ax.contour(x, y, diffGB, colors='black', levels=(0,))

    # red, green, blue and max probability densities
    densities = model.dens(z)
    dRed   = np.reshape(densities[:,0,np.newaxis], x.shape)
    dGreen = np.reshape(densities[:,1,np.newaxis], x.shape)
    dBlue  = np.reshape(densities[:,2,np.newaxis], x.shape)
    dMax   = np.reshape(np.max(densities, axis=1), x.shape)

    # second figure shows 3d plots of probability densities
    ax = fig.add_subplot(2,2,2, projection='3d')

    # straight class colors for suface plots
    color = np.reshape([dRed,dGreen,dBlue], (3, x.shape[0],x.shape[1]))
    color = color.swapaxes(1,2).T

    # flip colors to fade to white
    zro        = np.zeros_like(x)
    colorFlip  = np.ones((3, x.shape[0], x.shape[1]))
    colorFlip -= (np.array((zro,dRed,dRed)) +
                  np.array((dGreen,zro,dGreen)) +
                  np.array((dBlue,dBlue,zro)))
    colorFlip -= np.min(colorFlip)
    colorFlip /= np.max(colorFlip)
    colorFlip  = colorFlip.swapaxes(1,2).T

    # probability density surface
    #surf = ax.plot_surface(x,y, dMax, cmap=matplotlib.cm.jet, linewidth=0)
    surf = ax.plot_surface(x,y, dMax, facecolors=colorFlip,
                           linewidth=0.02, shade=True)
    surf.set_edgecolor('black') # add edgecolor back in, bug?

    # third figure shows 3d plots of probabilities
    ax = fig.add_subplot(2,2,3, projection='3d')

    # straight class colors for suface plots
    color = np.reshape([pRed,pGreen,pBlue], (3, x.shape[0],x.shape[1]))
    color = color.swapaxes(1,2).T

    # flip colors to fade to white
    zro        = np.zeros_like(x)
    colorFlip  = np.ones((3, x.shape[0], x.shape[1]))
    colorFlip -= (np.array((zro,pRed,pRed)) +
                  np.array((pGreen,zro,pGreen)) +
                  np.array((pBlue,pBlue,zro)))
    colorFlip -= np.min(colorFlip)
    colorFlip /= np.max(colorFlip)
    colorFlip  = colorFlip.swapaxes(1,2).T

    # probability density surface
    #surf = ax.plot_surface(x,y, pMax, cmap=matplotlib.cm.jet, linewidth=0)
    surf = ax.plot_surface(x,y, pMax, facecolors=colorFlip,
                           linewidth=0.02, shade=True)
    surf.set_edgecolor('black') # add edgecolor back in, bug?
    """
    # third figure shows contours and color image of probability densities
    ax = fig.add_subplot(2,2,3)

    #ax.pcolor(x,y,pMax)
    ax.imshow(colorFlip, origin='lower',
              extent=(mn[0],mx[0],mn[1],mx[1]), aspect='auto')

    # contours 
    nLevel=6
    cs = ax.contour(x, y, pMax, colors='black',
                    levels=np.linspace(np.min(pMax),np.max(pMax),nLevel))
    cs.clabel(fontsize=6)
    """

    # fourth figure
    ax = fig.add_subplot(2,2,4, projection='3d')

    labels = model.label(z)
    lMax   = np.reshape(labels, x.shape)

    surf = ax.plot_surface(x,y, lMax, facecolors=colorFlip,
                           linewidth=0.02)#, antialiased=False)
    #surf.set_edgecolor(np.vstack(color))
    surf.set_edgecolor('black')

    fig.tight_layout()


if __name__ == '__main__':
    demoLDA2d()
    demoQDA2d()
    plt.show()
