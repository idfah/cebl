import matplotlib.pyplot as plt
import numpy as np

from cebl import util

from .classifier import Classifier
from .autoreg import AutoRegression, RecurrentAutoRegression
from .logreg import LogisticRegression
from . import stand

from .nnet import esn

from .logreg import LogisticRegression
from .knn import KNN


class AutoRegressiveClassifierCosine(Classifier):
    def __init__(self, classData, k=5, autoRegClass=AutoRegression, **autoRegKwargs):
        # initialize Classifier base class
        Classifier.__init__(self, util.segmat(classData[0]).shape[2], len(classData))

        self.k = k

        self.autoRegClass = autoRegClass

        self.train(classData, **autoRegKwargs)

    def train(self, classData, **autoRegKwargs):
        self.models = \
            [self.autoRegClass(ss, **autoRegKwargs) for ss in classData]

        trainErrors = [self.modelErrors(ss) for ss in classData]
        self.errorModel = KNN(trainErrors, k=self.k, distMetric='cosine')

    def modelErrors(self, ss, *args, **kwargs):
        errors = []
        for mdl in self.models:
            preds, resids = mdl.eval(ss, *args, returnResid=True, **kwargs)
            errors.append(np.vstack([util.rmse(r) for r in resids]))
        return np.hstack(errors)

    def probs(self, ss, *args, **kwargs):
        return self.errorModel.probs(self.modelErrors(ss, *args, **kwargs))

class ARCC(AutoRegressiveClassifierCosine):
    pass

class AutoRegressiveClassifier3(Classifier):
    def __init__(self, classData, center='none',
                 autoRegClass=AutoRegression, **autoRegKwargs):
        # initialize Classifier base class
        Classifier.__init__(self, util.segmat(classData[0]).shape[2], len(classData))

        self.autoRegClass = autoRegClass

        self.train(classData, center=center, **autoRegKwargs)

    def train(self, classData, center, **autoRegKwargs):
        self.models = \
            [self.autoRegClass(ss, **autoRegKwargs) for ss in classData]

        self.center = 0.0
        trainErrors = [self.modelErrors(ss) for ss in classData]

        center = center.lower()

        if center == 'none':
            pass

        elif center == 'mean':
            self.center = np.mean([np.mean(te, axis=0) for te in trainErrors], axis=0)

        elif center == 'median':
            self.center = np.mean([np.median(te, axis=0) for te in trainErrors], axis=0)

        else:
            raise Exception('Invalid center method %s.' % str(center))

    def modelErrors(self, ss, *args, **kwargs):
        errors = [np.vstack([mdl.rmse(s, *args, **kwargs) for s in ss]) for mdl in self.models]
        return np.hstack(errors) - self.center

    #def discrim(self, ss, *args, **kwargs):
    #    return -self.modelErrors(ss, *args, **kwargs)

    def probs(self, ss, *args, **kwargs):
        errors = self.modelErrors(ss, *args, **kwargs)
        return util.softmax(-errors) # softmin

class AutoRegressiveClassifier(Classifier):
    def __init__(self, classData, autoRegClass=AutoRegression, **autoRegKwargs):
        # initialize Classifier base class
        Classifier.__init__(self, util.segmat(classData[0]).shape[2], len(classData))

        self.autoRegClass = autoRegClass

        self.train(classData, **autoRegKwargs)

    def train(self, classData, **autoRegKwargs):
        self.models = \
            [self.autoRegClass(ss, **autoRegKwargs) for ss in classData]

        #self.baselineErrors = np.empty(len(self.models))
        #for i,mdl in enumerate(self.models):
        #    preds, resids = mdl.eval(classData[i], returnResid=True)
        #    self.baselineErrors[i] = util.rmse(resids)

    def modelErrors(self, ss, *args, **kwargs):
        #errors = [np.vstack([mdl.rmse(s, *args, **kwargs) for s in ss]) for mdl in self.models]
        #return np.hstack(errors)
        errors = []
        for mdl in self.models:
            preds, resids = mdl.eval(ss, *args, returnResid=True, **kwargs)
            errors.append(np.vstack([util.rmse(r) for r in resids]))
            #errors.append(np.vstack([r.var(axis=1).mean() for r in resids]))
            #errors.append(np.vstack([util.rmse(r-r.mean(axis=1)[:,None]) for r in resids]))
            #errors.append(np.vstack([util.mse(r, axis=1).mean() for r in resids]))
        return np.hstack(errors)

    #def discrim(self, ss, *args, **kwargs):
    #    return -self.modelErrors(ss, *args, **kwargs)

    def probs(self, ss, *args, **kwargs):
        errors = self.modelErrors(ss, *args, **kwargs)
        return util.softmax(-errors) # softmin

class ARC(AutoRegressiveClassifier):
    pass

def demoARC():
    order = 5

    x = np.linspace(0.0, 6*np.pi, 101)
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

    print s1.shape
    print s2.shape

    trainData = [s1[:(ns1//2)], s2[:(ns2//2)]]
    testData = [s1[(ns1//2):], s2[(ns2//2):]]

    standardizer = stand.ClassSegStandardizer(trainData)
    trainData = standardizer.apply(trainData)
    testData = standardizer.apply(testData)

    model = ARC(trainData, order=order)

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

    fig = plt.figure(figsize=(20,6))
    axSigs = fig.add_subplot(1,3, 1)
    axSigs.plot(x, trainData[0][0].T, color='blue', linewidth=2, label=r'$\mathbf{sin}(x)$')
    axSigs.plot(x, trainData[0].T, color='blue', alpha=0.1, linewidth=2)
    axSigs.plot(x, 3.0+trainData[1][0].T, color='red', linewidth=2, label=r'$\mathbf{sin}(2x)$')
    axSigs.plot(x, 3.0+trainData[1].T, color='red', alpha=0.1, linewidth=2)
    axSigs.set_title('Noisy Sinusoids with Random Phase Shifts')
    axSigs.set_xlabel('Time')
    axSigs.set_ylabel('Signal')
    axSigs.legend()
    axSigs.autoscale(tight=True)

    trainErrors = [model.modelErrors(cls) for cls in trainData]
    testErrors = [model.modelErrors(cls) for cls in testData]
    trainProbs = [model.probs(cls) for cls in trainData]
    testProbs = [model.probs(cls) for cls in testData]

    #standardizer = stand.Standardizer(np.vstack(trainErrors))
    #trainErrors = [standardizer.apply(cls) for cls in trainErrors]
    #testErrors = [standardizer.apply(cls) for cls in testErrors]

    axTrainErrs = fig.add_subplot(1,3, 2)
    #axTrainErrs = fig.add_subplot(1,2, 1)
    axTrainErrs.scatter(trainErrors[0][:,0], trainErrors[0][:,1], color='blue')
    axTrainErrs.scatter(trainErrors[1][:,0], trainErrors[1][:,1], color='red')
    axTrainErrs.set_title('Training Relative Modeling Errors')
    axTrainErrs.set_xlabel('$\mathbf{sin}(x)$ model error')
    axTrainErrs.set_ylabel('$\mathbf{sin}(2x)$ model error')

    allTrainErrs = np.vstack(trainErrors)
    mn = allTrainErrs.min()
    mx = allTrainErrs.max()

    axTrainErrs.plot((mn,mx), (mn,mx), color='grey', linestyle='-.')
    axTrainErrs.grid()
    axTrainErrs.autoscale(tight=True)

    axTestErrs = fig.add_subplot(1,3, 3)
    #axTestErrs = fig.add_subplot(1,2, 2)
    axTestErrs.scatter(testErrors[0][:,0], testErrors[0][:,1], color='blue')
    axTestErrs.scatter(testErrors[1][:,0], testErrors[1][:,1], color='red')
    axTestErrs.set_title('Testing Relative Modeling Errors')
    axTestErrs.set_xlabel('$\mathbf{sin}(x)$ model error')
    axTestErrs.set_ylabel('$\mathbf{sin}(2x)$ model error')

    allTestErrs = np.vstack(testErrors)
    mn = allTestErrs.min()
    mx = allTestErrs.max()

    axTestErrs.plot((mn,mx), (mn,mx), color='grey', linestyle='-.')
    axTestErrs.grid()
    axTestErrs.autoscale(tight=True)

    fig.tight_layout()


class UnivariateAutoRegressiveClassifier(AutoRegressiveClassifier):
    def __init__(self, *args, **kwargs):
        AutoRegressiveClassifier.__init__(self, *args, autoRegClass=UnivariateAutoRegression, **kwargs)

class UARC(UnivariateAutoRegressiveClassifier):
    pass


class RecurrentAutoRegressiveClassifier(AutoRegressiveClassifier):
    def __init__(self, *args, **kwargs):
        AutoRegressiveClassifier.__init__(self, *args, autoRegClass=RecurrentAutoRegression, **kwargs)

class RARC(RecurrentAutoRegressiveClassifier):
    pass

def demoRARC():
    x = np.linspace(0.0, 6*np.pi, 101)
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

    print s1.shape
    print s2.shape

    trainData = [s1[:(ns1//2)], s2[:(ns2//2)]]
    testData = [s1[(ns1//2):], s2[(ns2//2):]]

    standardizer = stand.ClassSegStandardizer(trainData)
    trainData = standardizer.apply(trainData)
    testData = standardizer.apply(testData)

    model = RARC(trainData, nRes=512)

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

    fig = plt.figure(figsize=(20,6))
    axSigs = fig.add_subplot(1,3, 1)
    axSigs.plot(x, trainData[0][0].T, color='blue', linewidth=2, label=r'$\mathbf{sin}(x)$')
    axSigs.plot(x, trainData[0].T, color='blue', alpha=0.1, linewidth=2)
    axSigs.plot(x, 3.0+trainData[1][0].T, color='red', linewidth=2, label=r'$\mathbf{sin}(2x)$')
    axSigs.plot(x, 3.0+trainData[1].T, color='red', alpha=0.1, linewidth=2)
    axSigs.set_title('Noisy Sinusoids with Random Phase Shifts')
    axSigs.set_xlabel('Time')
    axSigs.set_ylabel('Signal')
    axSigs.legend()
    axSigs.autoscale(tight=True)

    trainErrors = [model.modelErrors(cls) for cls in trainData]
    testErrors = [model.modelErrors(cls) for cls in testData]
    trainProbs = [model.probs(cls) for cls in trainData]
    testProbs = [model.probs(cls) for cls in testData]

    #standardizer = stand.Standardizer(np.vstack(trainErrors))
    #trainErrors = [standardizer.apply(cls) for cls in trainErrors]
    #testErrors = [standardizer.apply(cls) for cls in testErrors]

    axTrainErrs = fig.add_subplot(1,3, 2)
    #axTrainErrs = fig.add_subplot(1,2, 1)
    axTrainErrs.scatter(trainErrors[0][:,0], trainErrors[0][:,1], color='blue')
    axTrainErrs.scatter(trainErrors[1][:,0], trainErrors[1][:,1], color='red')
    axTrainErrs.set_title('Training Relative Modeling Errors')
    axTrainErrs.set_xlabel('$\mathbf{sin}(x)$ model error')
    axTrainErrs.set_ylabel('$\mathbf{sin}(2x)$ model error')

    allTrainErrs = np.vstack(trainErrors)
    mn = allTrainErrs.min()
    mx = allTrainErrs.max()

    axTrainErrs.plot((mn,mx), (mn,mx), color='grey', linestyle='-.')
    axTrainErrs.grid()
    axTrainErrs.autoscale(tight=True)

    axTestErrs = fig.add_subplot(1,3, 3)
    #axTestErrs = fig.add_subplot(1,2, 2)
    axTestErrs.scatter(testErrors[0][:,0], testErrors[0][:,1], color='blue')
    axTestErrs.scatter(testErrors[1][:,0], testErrors[1][:,1], color='red')
    axTestErrs.set_title('Testing Relative Modeling Errors')
    axTestErrs.set_xlabel('$\mathbf{sin}(x)$ model error')
    axTestErrs.set_ylabel('$\mathbf{sin}(2x)$ model error')

    allTestErrs = np.vstack(testErrors)
    mn = allTestErrs.min()
    mx = allTestErrs.max()

    axTestErrs.plot((mn,mx), (mn,mx), color='grey', linestyle='-.')
    axTestErrs.grid()
    axTestErrs.autoscale(tight=True)

    fig.tight_layout()


if __name__ == '__main__':
    demoARC()
    plt.show()
