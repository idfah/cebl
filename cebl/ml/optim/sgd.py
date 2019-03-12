import numpy as np


def sgd(optable, x, g, batchSize=30,
        learningRate=0.05, learningRateFinal=0.001, momentum=0.01,
        accuracy=0.0, precision=1.0e-10, divergeThresh=1.0e10,
        maxIter=5000, pTrace=False, eTrace=False,
        callback=None, verbose=False, *args, **kwargs):
    """Stochastic Gradient Descent.
    Requires a first-order gradient estimate.
    """
    x = np.asarray(x)
    g = np.asarray(g)

    params = optable.parameters()

    # set up learning rate decay
    if learningRateFinal is None:
        learningRateFinal = learningRate

    learningRateDecay = (np.log(learningRateFinal / float(learningRate)) /
                         -(maxIter-1))
    #learningRateDecay = (learningRateFinal - learningRate) / float(maxIter-1)

    velocity = np.zeros(params.shape)

    # initial value of the error function
    error = np.inf

    paramTrace = []
    errorTrace = []

    # termination reason
    reason = None

    batch = 0
    nObs = len(x)
    batchInd = np.arange(nObs)

    iteration = 0
    while True:
        start = 0
        np.random.shuffle(batchInd)
        if verbose:
            print("batch: %d" % batch)

        xShuff = x[batchInd]
        gShuff = g[batchInd]

        while True:
            end = start + batchSize
            if end > nObs:
                break

            xMini = xShuff[start:end]
            gMini = gShuff[start:end]

            # compute value of the error function and the gradient
            errorPrev = error
            error = optable.error(x, g, *args, **kwargs)
            grad = optable.gradient(xMini, gMini, *args, returnError=False, **kwargs)

            curLearningRate = learningRate * np.exp(-iteration * learningRateDecay)
            #curLearningRate = learningRate + learningRateDecay * iteration

            velocity[...] = momentum * velocity + curLearningRate * grad

            if verbose:
                print("%d %3f %6f" % (iteration, curLearningRate, error))

            if callback is not None:
                callback(optable, iteration, paramTrace, errorTrace)

            # keep parameter history if requested
            if pTrace:
                paramTrace.append(params.copy())

            # keep error function history if requested
            if eTrace:
                errorTrace.append(error)

            # terminate if desired accuracy reached
            if error < accuracy:
                reason = "accuracy"
                break

            # terminate if desired precision reached
            if np.abs(error - errorPrev) < precision:
                reason = "precision"
                break

            # terminate if the error function diverges
            if error > divergeThresh:
                reason = "diverge"
                break

            # terminate if maximum iterations reached
            if iteration >= maxIter:
                reason = "maxiter"
                break

            # move in direction of negative gradient
            #params -= curLearningRate * grad
            params -= velocity

            # increment iteration counter
            iteration += 1

            # move mini-batch forward
            start += batchSize

        if reason is not None:
            break

        # increment batch counter
        batch += 1

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {
        "error": error,
        "params": params,
        "error": error,
        "iteration": iteration,
        "reason": reason
    }

    # pylint: disable=multiple-statements
    if pTrace: result["pTrace"] = paramTrace
    if eTrace: result["eTrace"] = errorTrace

    return result
