import numpy as np

from .scg import scg


def minibatch(optable, x, g, batchSize=10, maxRound=10,
              maxTotalIter=np.inf, pTrace=False, eTrace=False,
              miniOptimFunc=scg, verbose=0, *args, **kwargs):
    '''
    only works for supervised problems, must have x and g

    args and kwargs are passed to miniOptimFunc, doesn't pass extra args to optable.error
    '''
    # make sure x and g are numpy arrays
    x = np.asarray(x)
    g = np.asarray(g)

    # make sure x and g have same size
    assert len(x) == len(g)

    # number of observations
    nObs = len(x)

    verbose = int(verbose)

    # parameter and error traces
    # evaluated here, not in miniOptimFunc
    paramTrace = []
    errorTrace = []

    # final termination reason
    reason = None

    # current round of minibatches
    # increments after miniOptimFunc
    # is applied to all observations
    curRound = 0

    # indices into x and g used to select minibatches
    batchInd = np.arange(nObs)

    # sum of all iterations performed in calls to miniOptimFunc
    totalIter = 0

    # total number of batches evaluated
    totalBatch = 0

    # for each round
    done = False
    while not done:
        if verbose > 0:
            print('=======')
            print('iterations: %d' % totalIter)
            print('error: %.5f' % optable.error(x=x, g=g))
            print('round: %d' % curRound)
            print('=======')

        # start index into current minibatch
        start = 0

        # randomly shuffle minibatches each round
        np.random.shuffle(batchInd)
        xShuff = x[batchInd]
        gShuff = g[batchInd]

        # for each minibatch
        curBatch = 0
        while True:
            if verbose > 0:
                print('minibatch: %d' % curBatch)

            # end index into current minibatch
            end = start + batchSize

            # don't process last minibatch
            # if smaller than batchSize
            if end > nObs:
                break

            # select current batch
            xMini = xShuff[start:end]
            gMini = gShuff[start:end]

            v = True if verbose > 1 else False

            # run miniOptimFunc on current minibatch
            miniResult = miniOptimFunc(optable, *args, x=xMini, g=gMini,
                                       pTrace=pTrace, eTrace=eTrace,
                                       verbose=v, **kwargs)

            # keep parameter history if requested
            if pTrace:
                #paramTrace += miniResult['pTrace']
                paramTrace.append(optable.parameters().copy())

            # keep error function history if requested
            if eTrace:
                #errorTrace += miniResult['eTrace']
                errorTrace.append(optable.error(x=x, g=g))

            # increment total iterations
            totalIter += miniResult['iteration']

            # increment batch counters
            curBatch += 1
            totalBatch += 1

            # terminate if maximum total iterations reached
            if totalIter >= maxTotalIter:
                reason = 'maxiter'
                done = True
                break

            # move mini-batch forward
            start += batchSize

        # increment round counter
        curRound += 1

        # terminate if maximum total rounds is reached
        if curRound >= maxRound:
            reason = 'maxround'
            done = True

    if verbose > 0:
        print('reason: %s' % reason)
        print('round: %d' % curRound)
        print('iterations: %d' % totalIter)

    # save result into a dictionary
    result = {}
    result['iteration'] = totalIter
    result['round'] = curRound
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if eTrace: result['eTrace'] = errorTrace

    return result
