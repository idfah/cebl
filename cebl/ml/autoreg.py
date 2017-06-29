import matplotlib.pyplot as plt
import numpy as np

from cebl import util

from . import nnet
from .linreg import RidgeRegression

#from . import strans


class AutoRegressionBase(object):
    def __init__(self, ss, horizon, regClass, *args, **kwargs):
        """
        Args:
            ss:   (nSeg,nObs[,nDim])
        """
        ss = np.asarray(ss)
        self.horizon = horizon
        self.regClass = regClass

        self.train(ss, *args, **kwargs)

    def getInputs(self, ss):
        raise NotImplementedError('getInputs not implemented.')

    def getTargets(self, ss):
        raise NotImplementedError('getTargets not implemented.')

    def train(self, ss, *args, **kwargs):
        raise NotImplementedError('train not implemented.')

    def eval(self, ss, returnResid=False, *args, **kwargs):
        raise NotImplementedError('eval not implemented.')

    def resid(self, ss, *args, **kwargs):
        pred, resid = self.eval(ss, *args, returnResid=True, **kwargs)
        return resid

    def abe(self, ss, axis=None, *args, **kwargs):
        resid = self.resid(ss, *args, **kwargs)
        return util.abe(resid, axis=axis)

    def sse(self, ss, axis=None, *args, **kwargs):
        resid = self.resid(ss, *args, **kwargs)
        return util.sse(resid, axis=axis)

    def mse(self, ss, axis=None, *args, **kwargs):
        resid = self.resid(ss, *args, **kwargs)
        return util.mse(resid, axis=axis)

    def rmse(self, ss, axis=None, *args, **kwargs):
        resid = self.resid(ss, *args, **kwargs)
        return util.rmse(resid, axis=axis)

    def nrmse(self, ss, axis=None, *args, **kwargs):
        resid = self.resid(ss, *args, **kwargs)
        return util.nrmse(resid, axis=axis)


class AutoRegression(AutoRegressionBase):
    def __init__(self, ss, order, horizon=1, regClass=RidgeRegression, *args, **kwargs):
        self.order = order

        AutoRegressionBase.__init__(self, ss, horizon=horizon,
                regClass=regClass, *args, **kwargs)

    def getInputs(self, ss):
        ss = util.segmat(ss)
        return util.timeEmbed(ss, lags=self.order-1, axis=1)[:,:-self.horizon]

    def getTargets(self, ss):
        ss = util.segmat(ss)
        return ss[:,(self.order+self.horizon-1):]

    def train(self, ss, *args, **kwargs):
        xs = self.getInputs(ss)
        gs = self.getTargets(ss)

        x = xs.reshape((xs.shape[0]*xs.shape[1], -1))
        g = gs.reshape((gs.shape[0]*gs.shape[1], -1))

        self.model = self.regClass(x, g, *args, **kwargs)

    def eval(self, ss, returnResid=False, *args, **kwargs):
        xs = self.getInputs(ss)
        gs = self.getTargets(ss)

        preds = self.model.evals(xs, *args, **kwargs)

        if returnResid:
            resids = gs - preds
            return preds, resids
        else:
            return preds

class AR(AutoRegression):
    pass

def demoAutoRegressionSine():
    time = np.linspace(0.0,10.0*np.pi,5000)
    s = np.sin(time)

    #data = [s + np.random.normal(size=s.shape, scale=0.3) for i in xrange(5)]
    data = s[None,...]

    order = 2
    arFit = AutoRegression(data, order=order)

    print arFit.model.weights

    pred, resid = arFit.eval((s,), returnResid=True)

    plt.plot(time, s, color='blue')
    plt.plot(time[order:], pred[0], color='red')

def demoAutoRegressionMulti():
    time = np.linspace(0.0,10.0*np.pi,5000)

    # noisy cosine chirp
    s1 = np.cos(time**2/10.0)
    s1 += np.random.normal(size=len(time), scale=0.2)

    # chaotic gauss map
    a = 6.2
    b = -0.5
    n = len(time)
    s2 = np.empty(n)
    s2[0] = 0.1
    for i in xrange(1,n):
        s2[i] = np.exp(-a*s2[i-1]**2) + b

    s3 = np.random.normal(size=len(time), scale=0.3)

    data = np.vstack((s1, s2, s3)).T

    mid = int(data.shape[0]/2)

    timeTrain = time[:mid]
    timeTest = time[mid:]

    dataTrain = data[:mid,:][None,...]
    dataTest = data[mid:,:][None,...]

    order = 5
    arFit = AutoRegression(dataTrain, order=order, penalty=0.0)

    predTrain, residTrain = arFit.eval(dataTrain, returnResid=True)
    predTest, residTest = arFit.eval(dataTest, returnResid=True)

    print arFit.rmse(dataTest)

    sepTrain = np.arange(dataTrain.shape[2])*2.0*np.max(np.abs(data))
    sepTest = np.arange(dataTest.shape[2])*2.0*np.max(np.abs(data))

    fig = plt.figure(figsize=(19,8))

    axTrainPred = fig.add_subplot(2,3,1)
    axTrainPred.plot(timeTrain, dataTrain[0]-sepTrain, color='gray', linewidth=2)
    axTrainPred.plot(timeTrain[order:], predTrain[0]-sepTrain, linewidth=1)
    axTrainPred.autoscale(tight=True)
    axTrainPred.set_title('Train Predictions')
    axTrainPred.set_xlabel('Time')
    axTrainPred.set_yticks(-sepTrain)
    axTrainPred.set_yticklabels(['s1', 's2', 's3'])

    axTestPred = fig.add_subplot(2,3,2)
    axTestPred.plot(timeTest, dataTest[0]-sepTest, color='gray', linewidth=2)
    axTestPred.plot(timeTest[order:], predTest[0]-sepTest, linewidth=1)
    axTestPred.autoscale(tight=True)
    axTestPred.set_title('Test Predictions')
    axTestPred.set_xlabel('Time')
    axTestPred.set_yticks(-sepTrain)
    axTestPred.set_yticklabels(['s1', 's2', 's3'])

    axWeights = fig.add_subplot(2,3,3)
    img = axWeights.imshow(arFit.model.weights, aspect='auto', interpolation='none')
    cbar = plt.colorbar(img)
    cbar.set_label('Weight')
    axWeights.set_title('Model Weights')
    axWeights.set_xlabel('Output')
    axWeights.set_ylabel('Input')
    axWeights.set_xticks(range(arFit.model.weights.shape[1]))
    axWeights.set_xticklabels(['s1', 's2', 's3'])
    axWeights.set_yticks(range(arFit.model.weights.shape[0]))
    axWeights.set_yticklabels(list(range(1,arFit.model.weights.shape[0]) + ['bias']))
    axWeights.autoscale(tight=True)

    axTrainResid = fig.add_subplot(2,3,4)
    axTrainResid.plot(timeTrain[order:], residTrain[0]-sepTrain)
    axTrainResid.autoscale(tight=True)
    axTrainResid.set_title('Train Residuals')
    axTrainResid.set_xlabel('Time')
    axTrainResid.set_yticks(-sepTrain)
    axTrainResid.set_yticklabels(['s1', 's2', 's3'])

    axTestResid = fig.add_subplot(2,3,5)
    axTestResid.plot(timeTest[order:], residTest[0]-sepTest)
    axTestResid.autoscale(tight=True)
    axTestResid.set_title('Test Residuals')
    axTestResid.set_xlabel('Time')
    axTestResid.set_yticks(-sepTrain)
    axTestResid.set_yticklabels(['s1', 's2', 's3'])

    axTestResidDist = fig.add_subplot(2,3,6)
    #axTestResidDist.hist(residTest, histtype='stepfilled', normed=True)
    axTestResidDist.hist(residTest[0], stacked=True, normed=True)
    axTestResidDist.legend(['s1', 's2', 's3'])
    axTestResidDist.set_title('Test Residual Distribution')
    axTestResidDist.set_xlabel('Residual')
    axTestResidDist.set_ylabel('Density')

    fig.tight_layout()


class UnivariateAutoRegression(AutoRegression):
    def train(self, ss, *args, **kwargs):
        ss = util.segmat(ss)

        self.model = []
        for i in xrange(ss.shape[2]):
            v = ss[:,:,i]

            xs = self.getInputs(v)
            gs = self.getTargets(v)

            x = xs.reshape((xs.shape[0]*xs.shape[1], -1))
            g = gs.reshape((gs.shape[0]*gs.shape[1], -1))

            self.model.append(self.regClass(x, g, *args, **kwargs))

    def eval(self, ss, returnResid=False, *args, **kwargs):
        ss = util.segmat(ss)

        preds = []
        gi = []
        for i in xrange(ss.shape[2]):
            v = ss[:,:,i]

            xs = self.getInputs(v)
            gs = self.getTargets(v)

            preds.append(self.model[i].evals(xs, *args, **kwargs).squeeze(2))

            if returnResid:
                gi.append(gs.squeeze(2))

        preds = np.rollaxis(np.array(preds), 0,3)

        if returnResid:
            gs = np.rollaxis(np.array(gi), 0,3)
            resids = gs - preds
            return preds, resids
        else:
            return preds

class UAR(UnivariateAutoRegression):
    pass

def demoAutoRegressionUni():
    time = np.linspace(0.0,10.0*np.pi,5000)

    # noisy cosine chirp
    s1 = np.cos(time**2/10.0)
    s1 += np.random.normal(size=len(time), scale=0.2)

    # chaotic gauss map
    a = 6.2
    b = -0.5
    n = len(time)
    s2 = np.empty(n)
    s2[0] = 0.1
    for i in xrange(1,n):
        s2[i] = np.exp(-a*s2[i-1]**2) + b

    s3 = np.random.normal(size=len(time), scale=0.3)

    data = np.vstack((s1, s2, s3)).T

    mid = int(data.shape[0]/2)

    timeTrain = time[:mid]
    timeTest = time[mid:]

    dataTrain = data[:mid,:][None,...]
    dataTest = data[mid:,:][None,...]

    order = 5
    arFit = UnivariateAutoRegression(dataTrain, order=order, penalty=0.0)

    predTrain, residTrain = arFit.eval(dataTrain, returnResid=True)
    predTest, residTest = arFit.eval(dataTest, returnResid=True)

    print arFit.rmse(dataTest)

    sepTrain = np.arange(dataTrain.shape[2])*2.0*np.max(np.abs(data))
    sepTest = np.arange(dataTest.shape[2])*2.0*np.max(np.abs(data))

    fig = plt.figure(figsize=(19,8))

    axTrainPred = fig.add_subplot(2,3,1)
    axTrainPred.plot(timeTrain, dataTrain[0]-sepTrain, color='gray', linewidth=2)
    axTrainPred.plot(timeTrain[order:], predTrain[0]-sepTrain, linewidth=1)
    axTrainPred.autoscale(tight=True)
    axTrainPred.set_title('Train Predictions')
    axTrainPred.set_xlabel('Time')
    axTrainPred.set_yticks(-sepTrain)
    axTrainPred.set_yticklabels(['s1', 's2', 's3'])

    axTestPred = fig.add_subplot(2,3,2)
    axTestPred.plot(timeTest, dataTest[0]-sepTest, color='gray', linewidth=2)
    axTestPred.plot(timeTest[order:], predTest[0]-sepTest, linewidth=1)
    axTestPred.autoscale(tight=True)
    axTestPred.set_title('Test Predictions')
    axTestPred.set_xlabel('Time')
    axTestPred.set_yticks(-sepTrain)
    axTestPred.set_yticklabels(['s1', 's2', 's3'])

    axWeights = fig.add_subplot(2,3,3)
    #img = axWeights.imshow(arFit.model.weights, aspect='auto', interpolation='none')
    #cbar = plt.colorbar(img)
    #cbar.set_label('Weight')
    #axWeights.set_title('Model Weights')
    #axWeights.set_xlabel('Output')
    #axWeights.set_ylabel('Input')
    #axWeights.set_xticks(range(arFit.model.weights.shape[1]))
    #axWeights.set_xticklabels(['s1', 's2', 's3'])
    #axWeights.set_yticks(range(arFit.model.weights.shape[0]))
    #axWeights.set_yticklabels(list(range(1,arFit.model.weights.shape[0]) + ['bias']))
    #axWeights.autoscale(tight=True)

    axTrainResid = fig.add_subplot(2,3,4)
    axTrainResid.plot(timeTrain[order:], residTrain[0]-sepTrain)
    axTrainResid.autoscale(tight=True)
    axTrainResid.set_title('Train Residuals')
    axTrainResid.set_xlabel('Time')
    axTrainResid.set_yticks(-sepTrain)
    axTrainResid.set_yticklabels(['s1', 's2', 's3'])

    axTestResid = fig.add_subplot(2,3,5)
    axTestResid.plot(timeTest[order:], residTest[0]-sepTest)
    axTestResid.autoscale(tight=True)
    axTestResid.set_title('Test Residuals')
    axTestResid.set_xlabel('Time')
    axTestResid.set_yticks(-sepTrain)
    axTestResid.set_yticklabels(['s1', 's2', 's3'])

    axTestResidDist = fig.add_subplot(2,3,6)
    #axTestResidDist.hist(residTest, histtype='stepfilled', normed=True)
    axTestResidDist.hist(residTest[0], stacked=True, normed=True)
    axTestResidDist.legend(['s1', 's2', 's3'])
    axTestResidDist.set_title('Test Residual Distribution')
    axTestResidDist.set_xlabel('Residual')
    axTestResidDist.set_ylabel('Density')

    fig.tight_layout()


class RecurrentAutoRegression(AutoRegressionBase):
    def __init__(self, ss, horizon=1, transient=0, regClass=nnet.ESN, *args, **kwargs):
        self.transient = transient
        AutoRegressionBase.__init__(self, ss, horizon=horizon,
                regClass=regClass, *args, **kwargs)

    def getInputs(self, ss):
        ss = np.asarray(ss)
        return ss[:,:-self.horizon]

    def getTargets(self, ss):
        ss = np.asarray(ss)
        return ss[:,self.horizon:]

    def train(self, ss, *args, **kwargs):
        xs = self.getInputs(ss)
        gs = self.getTargets(ss)

        self.model = self.regClass(xs, gs, *args, **kwargs)

    def eval(self, ss, returnResid=False, *args, **kwargs):
        xs = self.getInputs(ss)
        gs = self.getTargets(ss)

        preds = self.model.eval(xs, *args, **kwargs)

        if returnResid:
            resids = gs - preds
            return preds[:,self.transient:], resids[:,self.transient:]
        else:
            return preds[:,self.transient:]

class RAR(RecurrentAutoRegression):
    pass

def demoRecurrentAutoRegression():
    time = np.linspace(0.0,10.0*np.pi,5000)

    # noisy cosine chirp
    s1 = np.cos(time**2/10.0)
    s1 += np.random.normal(size=len(time), scale=0.2)

    # chaotic gauss map
    a = 6.2
    b = -0.5
    n = len(time)
    s2 = np.empty(n)
    s2[0] = 0.1
    for i in xrange(1,n):
        s2[i] = np.exp(-a*s2[i-1]**2) + b

    s3 = np.random.normal(size=len(time), scale=0.3)

    data = np.vstack((s1, s2, s3)).T

    mid = int(data.shape[0]/2)

    timeTrain = time[:mid]
    timeTest = time[mid:]

    dataTrain = data[:mid,:][None,...]
    dataTest = data[mid:,:][None,...]

    horizon = 1
    rarFit = RecurrentAutoRegression(dataTrain, horizon=1, rwScale=0.9, penalty=0.0)

    predTrain, residTrain = rarFit.eval(dataTrain, returnResid=True)
    predTest, residTest = rarFit.eval(dataTest, returnResid=True)

    print rarFit.rmse(dataTest)

    sepTrain = np.arange(dataTrain.shape[2])*2.0*np.max(np.abs(data))
    sepTest = np.arange(dataTest.shape[2])*2.0*np.max(np.abs(data))

    fig = plt.figure(figsize=(19,8))

    axTrainPred = fig.add_subplot(2,3,1)
    axTrainPred.plot(timeTrain, dataTrain[0]-sepTrain, color='gray', linewidth=2)
    axTrainPred.plot(timeTrain[horizon:], predTrain[0]-sepTrain, linewidth=1)
    axTrainPred.autoscale(tight=True)
    axTrainPred.set_title('Train Predictions')
    axTrainPred.set_xlabel('Time')
    axTrainPred.set_yticks(-sepTrain)
    axTrainPred.set_yticklabels(['s1', 's2', 's3'])

    axTestPred = fig.add_subplot(2,3,2)
    axTestPred.plot(timeTest, dataTest[0]-sepTest, color='gray', linewidth=2)
    axTestPred.plot(timeTest[horizon:], predTest[0]-sepTest, linewidth=1)
    axTestPred.autoscale(tight=True)
    axTestPred.set_title('Test Predictions')
    axTestPred.set_xlabel('Time')
    axTestPred.set_yticks(-sepTrain)
    axTestPred.set_yticklabels(['s1', 's2', 's3'])

    axWeights = fig.add_subplot(2,3,3)
    rarFit.model.reservoir.plotActDensity(dataTest, ax=axWeights)

    axTrainResid = fig.add_subplot(2,3,4)
    axTrainResid.plot(timeTrain[horizon:], residTrain[0]-sepTrain)
    axTrainResid.autoscale(tight=True)
    axTrainResid.set_title('Train Residuals')
    axTrainResid.set_xlabel('Time')
    axTrainResid.set_yticks(-sepTrain)
    axTrainResid.set_yticklabels(['s1', 's2', 's3'])

    axTestResid = fig.add_subplot(2,3,5)
    axTestResid.plot(timeTest[horizon:], residTest[0]-sepTest)
    axTestResid.autoscale(tight=True)
    axTestResid.set_title('Test Residuals')
    axTestResid.set_xlabel('Time')
    axTestResid.set_yticks(-sepTrain)
    axTestResid.set_yticklabels(['s1', 's2', 's3'])

    axTestResidDist = fig.add_subplot(2,3,6)
    #axTestResidDist.hist(residTest, histtype='stepfilled', normed=True)
    axTestResidDist.hist(residTest[0], stacked=True, normed=True)
    axTestResidDist.legend(['s1', 's2', 's3'])
    axTestResidDist.set_title('Test Residual Distribution')
    axTestResidDist.set_xlabel('Residual')
    axTestResidDist.set_ylabel('Density')

    fig.tight_layout()


if __name__ == '__main__':
    demoAutoRegressionSine()
    demoAutoRegressionMulti()
    #demoAutoRegressionUni()
    demoRecurrentAutoRegression()
    plt.show()
