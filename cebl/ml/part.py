"""Dataset partitioning.
"""
import numpy as np


def classStratified(classData, nFold):
    """Partition generator for stratified cross validation
    for classification problems.
    """
    classData = [np.asarray(cls) for cls in classData]

    ns = [len(cls) // nFold for cls in classData]

    for fold in range(nFold):
        trainData = list()
        testData = list()

        for n, cls in zip(ns, classData):
            start = fold * n

            if fold < nFold-1:
                end = (fold+1) * n
            else:
                end = len(cls)

            trainData.append(cls[range(start)+range(end, len(cls))])
            testData.append(cls[start:end])

        yield fold, trainData, testData

def classRandomSubSample(classData, trainFrac, nFold=1):
    """Partition generator for random sub-sampling validation
    for classification problems.
    """
    # create a copy of data that we can shuffle in place
    classData = [np.array(cls, copy=True) for cls in classData]

    # for each repetition
    for rep in range(nFold):
        # create new lists to hold training and
        # validation partitions for each class
        trainData = list()
        testData = list()

        # for each class in data
        for cls in classData:
            # number of training observations
            nTrain = int(len(cls) * trainFrac)

            # shuffle in place
            np.random.shuffle(cls)

            # append training and test partitions
            trainData.append(cls[:nTrain])
            testData.append(cls[nTrain:])

        # yield the current partitions
        yield rep, trainData, testData

def classLeaveOneOut(classData):
    """Partition generator for leave-one-out cross validation
    for classification problems.
    """
    # works? XXX - idfah
    classData = [np.asarray(cls) for cls in classData]
    rep = 0
    for cls in classData:
        for i, testDatum in enumerate(cls):
            if cls.ndim > 1:
                testDatum = testDatum.reshape((1,-1))
            else:
                testDatum = np.array((testDatum,))

            clsMinusTestDatum = np.delete(cls, i, axis=0)
            trainData = [d if d is not cls else clsMinusTestDatum
                         for d in classData]
            testData = [testDatum if d is cls else d[0:0,...]
                        for d in classData]

            yield rep, trainData, testData

            rep += 1

def stratified(x, g, nFold):
    """Partition generator for stratified cross validation
    for regression problems.
    """
    x = np.asarray(x)
    g = np.asarray(g)

    nx = len(x) // nFold
    ng = len(g) // nFold

    if nx != ng:
        raise RuntimeError('size of x and g do not match.')

    for fold in range(nFold):
        start = fold * nx
        if fold < nFold-1:
            end = (fold+1) * nx
        else:
            end = len(x)

        keep = range(start) + range(end, len(x))

        yield fold, x[keep], g[keep], x[start:end], g[start:end]

def randomSubSample(x, g, trainFrac, nFold=1):
    """Partition generator for random sub-sampling validation
    for regression problems.
    """
    x = np.asarray(x)
    g = np.asarray(g)

    for rep in range(nFold):
        nTrain = int(len(x) * trainFrac)

        ind = np.arange(len(x))
        np.random.shuffle(ind)

        xShuffle = x[ind]
        gShuffle = g[ind]

        yield rep, xShuffle[:nTrain], gShuffle[:nTrain], xShuffle[nTrain:], gShuffle[nTrain:]
