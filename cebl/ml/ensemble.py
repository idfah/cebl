import matplotlib.pyplot as plt
import numpy as np

from cebl import util

from cebl.ml.regression import Regression
from cebl.ml.classifier import Classifier
from cebl.ml import stand

from cebl.ml.linreg import RidgeRegression
from cebl.ml.logreg import LogisticRegression


class Ensemble(Regression):
    def __init__(self, x, g, nModels=10, obsFrac=0.5,
                 replacement=True, dimFrac=None,
                 regClass=RidgeRegression, *args, **kwargs):
        Regression.__init__(self, util.colmat(x).shape[1],
                            util.colmat(g).shape[1])

        self.nModels = nModels
        self.obsFrac = obsFrac
        self.replacement = replacement
        self.dimFrac = dimFrac

        self.train(x, g, regClass, *args, **kwargs)

    def train(self, x, g, regClass, *args, **kwargs):
        x = np.asarray(x)
        g = np.asarray(g)

        nObs = x.shape[0]

        if self.obsFrac is None:
            obsPerModel = None
        else:
            obsPerModel = int(self.obsFrac*nObs)

        if self.dimFrac is None:
            dimPerModel = None
        else:
            dimPerModel = int(self.dimFrac*self.nIn)

        self.models = []
        self.dimInds = []

        for m in xrange(self.nModels):
            if obsPerModel is not None:
                if self.replacement:
                    obsInd = np.random.random_integers(0,len(x)-1, size=obsPerModel)
                else:
                    obsInd = np.arange(nObs)
                    np.random.shuffle(obsInd)
                    obsInd = obsInd[:obsPerModel]
                xSub = x[obsInd]
                gSub = g[obsInd]
            else:
                xSub = x
                gSub = g

            if dimPerModel is not None:
                if xSub.ndim != 2:
                    raise Exception('Cannot subset dimensions for x with shape ' + \
                                    str(x.shape) + '.')

                dimInd = np.arange(self.nIn)
                np.random.shuffle(dimInd)
                dimInd = dimInd[:dimPerModel]

                xSub = xSub[:,dimInd]

                self.dimInds.append(dimInd)

            model = regClass(xSub, gSub, *args, **kwargs)
            self.models.append(model)

    def evalModels(self, x, *args, **kwargs):
        x = np.asarray(x)

        ys = []
        for m, mdl in enumerate(self.models):
            if self.dimFrac is not None:
                dimInd = self.dimInds[m]
                xSub = x[:,dimInd]
            else:
                xSub = x

            y = mdl.eval(xSub, *args, **kwargs)
            ys.append(y)

        return np.array(ys)

    def eval(self, x, method='mean', *args, **kwargs):
        ys = self.evalModels(x, *args, **kwargs)
        if method == 'mean':
            return np.mean(ys, axis=0)
        elif method == 'median':
            return np.median(ys, axis=0)
        else:
            raise Exception('Invalid method %s.' % str(method))

class ClassEnsemble(Classifier):
    def __init__(self, classData, nModels=10, obsFrac=0.5,
                 replacement=True, dimFrac=None,
                 clsClass=LogisticRegression, *args, **kwargs):
        Classifier.__init__(self, util.colmat(classData[0]).shape[1],
                            len(classData))

        self.nModels = nModels
        self.obsFrac = obsFrac
        self.replacement = replacement
        self.dimFrac = dimFrac

        self.train(classData, clsClass, *args, **kwargs)

    def train(self, classData, clsClass, *args, **kwargs):
        classData = [np.asarray(cls) for cls in classData]

        nObs = classData[0].shape[0]

        if self.dimFrac is None:
            dimPerModel = None
        else:
            dimPerModel = int(self.dimFrac*self.nIn)

        self.models = []
        self.dimInds = []

        for m in xrange(self.nModels):
            classDataSub = []
            if self.obsFrac is not None:
                for cls in classData:
                    nObs = len(cls)
                    obsPerModel = int(self.obsFrac*nObs)

                    if self.replacement:
                        obsInd = np.random.random_integers(0,len(cls)-1, size=obsPerModel)
                    else:
                        obsInd = np.arange(nObs)
                        np.random.shuffle(obsInd)
                        obsInd = obsInd[:obsPerModel]
                    classDataSub.append(cls[obsInd])
            else:
                classDataSub = classData

            if dimPerModel is not None:
                if classDataSub[0].ndim != 2:
                    raise Exception('Cannot subset dimensions with shape ' + \
                                    str(classDataSub[0].shape) + '.')

                dimInd = np.arange(self.nIn)
                np.random.shuffle(dimInd)
                dimInd = dimInd[:dimPerModel]

                classDataSub = [cls[:,dimInd] for cls in classDataSub]

                self.dimInds.append(dimInd)

            model = clsClass(classDataSub, *args, **kwargs)
            self.models.append(model)

    def probsModels(self, x, *args, **kwargs):
        x = np.asarray(x)

        allProbs = []
        for m, mdl in enumerate(self.models):
            if self.dimFrac is not None:
                dimInd = self.dimInds[m]
                xSub = x[:,dimInd]
            else:
                xSub = x

            p = mdl.probs(xSub, *args, **kwargs)
            allProbs.append(p)

        return np.array(allProbs)

    def probs(self, x, *args, **kwargs):
        allProbs= self.probsModels(x, *args, **kwargs)

        #probs = np.zeros((x.shape[0], self.nCls))
        #for labels in np.argmax(allProbs, axis=2):
        #    probs[np.arange(labels.shape[0]), labels] += 1.0
        #return probs / len(self.models)

        return util.softmax(np.sum(allProbs, axis=0))

        #likes = np.log(util.capZero(allProbs))
        #return util.softmax(np.sum(likes, axis=0))

def demoClassEnsemble():
    c1 = np.random.normal(loc=-1.0, scale=0.3, size=60)
    c2 = np.random.normal(loc=0.0, scale=0.3, size=30)
    c3 = np.random.normal(loc=2.0, scale=0.7, size=90)
    classData = [c1,c2,c3]

    model = ClassEnsemble(classData, nModels=10, obsFrac=0.5, replacement=False, dimFrac=None)

    c1Probs = model.probs(c1)
    c2Probs = model.probs(c2)
    c3Probs = model.probs(c3)

    print 'c1:'
    print model.label(c1)
    print 'c2:'
    print model.label(c2)
    print model.probs(c1)
    print 'c3:'
    print model.label(c3)
    print model.probs(c1)

    x = np.linspace(-2.0, 4.0, 500)
    xProbs = model.probs(x)

    plt.plot(x, xProbs, linewidth=2)
    plt.scatter(c1, np.zeros_like(c1), color='blue')
    plt.scatter(c2, np.zeros_like(c2), color='green')
    plt.scatter(c3, np.zeros_like(c3), color='red')

if __name__ == '__main__':
    demoClassEnsemble()
    plt.show()
