import matplotlib.pyplot as plt
import numpy as np

from . import tests


def rprop(optable,
          stepInitial=0.05, stepUp=1.02, stepDown=0.6,
          stepMin=0.0, stepMax=50.0,
          accuracy=0.0, precision=1.0e-10,
          divergeThresh=1.0e10, maxIter=2500,
          pTrace=False, sTrace=False, eTrace=False,
          callback=None, verbose=False, *args, **kwargs):
    """Resilient backpropigation.
    Requires a first-order gradient estimate.

    Args:
        optable:

        stepInitial:        Initial step size.

        stepUp:             Scalar to multiply step size by when the dirction
                            of the gradient remains unchanged:
                            step <- step * stepUp

        stepDown:           Scalar to multiply step size by when the dirction
                            of the gradient changes:
                            step <- step * stepDown

        stepMin:            A lower bound on step sizes.

        stepMax:            An upper bound on step sizes.

        accuracy:           Terminate if current value of the error funciton
                            falls below this value.

        precision:          Terminate if change in the error function falls
                            below this value.

        divergeThresh:      Terminate if the value of the error function
                            exceeds this value.

        maxIter:            Terminate once current iteration reaches this value.

        pTrace:             If True, a list of matrices (one for each parameter
                            matrix) is included in the final results that
                            contains a history of the parameters during
                            optimization.  If False (default), then a history
                            is not kept.

        sTrace:             If True, a list of matrices (one for each parameter
                            matrix) is included in the final results that
                            contains a history of the step sizes used during
                            optimization.  If False (default), then a history
                            is not kept.

        eTrace:             If True, an array containing a history of the error
                            function during optimization is included in the
                            final results.  If False (default), then a history
                            is not kept.

        callback:

        verbose:            Print extra information to standard out during the
                            training procedure.

        args, kwargs:       Arguments passed to optable.gradients.

    Returns:
        A dictionary containing the following keys:

        params:     A numpy array containing the optimized parameters.

        error:      Final value of the error function.

        iteration:  The number of iterations performed.

        reason:     A string describing the reason for termination.

        eTrace:     A list containing the value of the error function at each
                    iteration.  Only returned if eTrace is True.

        pTrace:     A list containing a copy of the parameters at each
                    iteration.  Only returned if pTrace is True.

    Refs:
        @inproceedings{riedmiller1993direct,
          title={A direct adaptive method for faster backpropagation learning: The RPROP algorithm},
          author={Riedmiller, Martin and Braun, Heinrich},
          booktitle={IEEE International Conference on Neural Networks}
          pages={586--591},
          year={1993},
          organization={IEEE}
        }
    """
    params = optable.parameters()
    paramsStart = params.copy()

    # initialize all step sizes to stepInitial
    steps = np.ones_like(params) * stepInitial

    # initialize unknown error to infinity
    error = np.inf

    # initialize gradient to zero in order to yield no flips
    grad = np.zeros_like(params)

    paramTrace = []
    stepTrace = []
    errorTrace = []

    # termination reason
    reason = ''

    iteration = 0
    while True:
        # compute value of the error function and the gradient
        errorPrev = error
        gradPrev = grad
        error, grad = optable.gradient(*args, returnError=True, **kwargs)

        if verbose:
            print('%d %6f' % (iteration, error))

        if callback is not None:
            callback(optable, iteration, paramTrace, errorTrace)

        # keep parameter history if requested
        if pTrace:
            paramTrace.append(params.copy())

        # keep step trace if requested
        if sTrace:
            stepTrace.append(steps.copy())

        # keep error function history if requested
        if eTrace:
            errorTrace.append(error)

        # terminate if maximum iterations reached
        if iteration >= maxIter:
            reason = 'maxiter'
            break

        # terminate if desired accuracy reached
        if error < accuracy:
            reason = 'accuracy'
            break

        # terminate if desired precision reached
        if np.abs(error - errorPrev) < precision:
            reason = 'precision'
            break

        # terminate if the error function diverges
        if error > divergeThresh:
            reason = 'diverge'
            break

        flips = grad * gradPrev

        # decrease step sizes where gradient flipped
        flipsNeg = np.where(flips < 0.0)[0]
        steps[flipsNeg] *= stepDown
        steps[...] = np.maximum(steps, stepMin)

        # increase step sizes where gradient did not flip
        flipsPos = np.where(flips > 0.0)[0]
        steps[flipsPos] *= stepUp
        steps[...] = np.minimum(steps, stepMax)

        params[...] += steps * -np.sign(grad)

        # increment iteration counter
        iteration += 1

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {}
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if sTrace: result['sTrace'] = stepTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def demoRProp():
    rosen = tests.Rosen(optimFunc=rprop, maxIter=10000, verbose=True)
    rosen.plot()


def irprop(optable,
          stepInitial=0.05, stepUp=1.02, stepDown=0.6,
          #stepInitial=0.05, stepUp=1.015, stepDown=0.5,
          stepMin=0.0, stepMax=50.0,
          accuracy=0.0, precision=1.0e-10,
          divergeThresh=1.0e10, maxIter=2500,
          pTrace=False, sTrace=False, eTrace=False,
          callback=None, verbose=False, *args, **kwargs):
    """Resilient backpropigation.
    Requires a first-order gradient estimate.

    Args:
        optable:

        stepInitial:        Initial step size.

        stepUp:             Scalar to multiply step size by when the dirction
                            of the gradient remains unchanged:
                            step <- step * stepUp

        stepDown:           Scalar to multiply step size by when the dirction
                            of the gradient changes:
                            step <- step * stepDown

        stepMin:            A lower bound on step sizes.

        stepMax:            An upper bound on step sizes.

        accuracy:           Terminate if current value of the error funciton
                            falls below this value.

        precision:          Terminate if change in the error function falls
                            below this value.

        divergeThresh:      Terminate if the value of the error function
                            exceeds this value.

        maxIter:            Terminate once current iteration reaches this value.

        pTrace:             If True, a list of matrices (one for each parameter
                            matrix) is included in the final results that
                            contains a history of the parameters during
                            optimization.  If False (default), then a history
                            is not kept.

        sTrace:             If True, a list of matrices (one for each parameter
                            matrix) is included in the final results that
                            contains a history of the step sizes used during
                            optimization.  If False (default), then a history
                            is not kept.

        eTrace:             If True, an array containing a history of the error
                            function during optimization is included in the
                            final results.  If False (default), then a history
                            is not kept.

        callback:

        verbose:            Print extra information to standard out during the
                            training procedure.

        args, kwargs:       Arguments passed to optable.gradients.

    Returns:
        A dictionary containing the following keys:

        params:     A numpy array containing the optimized parameters.

        error:      Final value of the error function.

        iteration:  The number of iterations performed.

        reason:     A string describing the reason for termination.

        eTrace:     A list containing the value of the error function at each
                    iteration.  Only returned if eTrace is True.

        pTrace:     A list containing a copy of the parameters at each
                    iteration.  Only returned if pTrace is True.

    Refs:
        @inproceedings{riedmiller1993direct,
          title={A direct adaptive method for faster backpropagation learning: The RPROP algorithm},
          author={Riedmiller, Martin and Braun, Heinrich},
          booktitle={IEEE International Conference on Neural Networks}
          pages={586--591},
          year={1993},
          organization={IEEE}
        }

        @article{igel2003empirical,
          title={Empirical evaluation of the improved Rprop learning algorithms},
          author={Igel, Christian and H{\"u}sken, Michael},
          journal={Neurocomputing},
          volume={50},
          pages={105--123},
          year={2003},
          publisher={Elsevier}
        }
    """
    params = optable.parameters()

    # initialize all step sizes to stepInitial
    steps = np.ones_like(params) * stepInitial

    # initialize unknown error to infinity
    error = np.inf

    # initialize gradient to zero in order to yield no flips
    grad = np.zeros_like(params)

    paramTrace = []
    stepTrace = []
    errorTrace = []

    # termination reason
    reason = ''

    iteration = 0
    while True:
        # compute value of the error function and the gradient
        errorPrev = error
        gradPrev = grad
        error, grad = optable.gradient(*args, returnError=True, **kwargs)

        if verbose:
            print('%d %6f' % (iteration, error))

        if callback is not None:
            callback(optable, iteration, paramTrace, errorTrace)

        # keep parameter history if requested
        if pTrace:
            paramTrace.append(params.copy())

        # keep step trace if requested
        if sTrace:
            stepTrace.append(steps.copy())

        # keep error function history if requested
        if eTrace:
            errorTrace.append(error)

        # terminate if maximum iterations reached
        if iteration >= maxIter:
            reason = 'maxiter'
            break

        # terminate if desired accuracy reached
        if error < accuracy:
            reason = 'accuracy'
            break

        # terminate if desired precision reached
        if np.abs(error - errorPrev) < precision:
            reason = 'precision'
            break

        # terminate if the error function diverges
        if error > divergeThresh:
            reason = 'diverge'
            break

        flips = grad * gradPrev
        flipsNeg = np.where(flips < 0.0)[0]
        flipsPos = np.where(flips > 0.0)[0]

        if error > errorPrev:
            if verbose:
                print('No success.')

            # backtrack flipped steps
            params[flipsNeg] += steps[flipsNeg] * np.sign(grad[flipsNeg])

            # zero gradient
            # prevents weight update this iteration
            # and prevents re-flip next iteration
            grad[flipsNeg] = 0.0

        # decrease step sizes where gradient flipped
        steps[flipsNeg] *= stepDown

        # increase step sizes where gradient did not flip
        steps[flipsPos] *= stepUp

        # cap step sizes
        steps[...] = np.maximum(steps, stepMin)
        steps[...] = np.minimum(steps, stepMax)

        # update weights
        params[...] -= steps * np.sign(grad)

        # increment iteration counter
        iteration += 1

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {}
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if sTrace: result['sTrace'] = stepTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def demoIRProp():
    rosen = tests.Rosen(optimFunc=irprop, maxIter=10000, verbose=True)
    rosen.plot()


def srprop(optable, x, g, batchSize=10, maxRound=np.inf,
          stepInitial=0.05, stepUp=1.02, stepDown=0.6,
          stepMin=0.0, stepMax=50.0,
          accuracy=0.0, precision=1.0e-10,
          divergeThresh=1.0e10, maxIter=2500,
          pTrace=False, sTrace=False, eTrace=False,
          callback=None, verbose=True, *args, **kwargs):
    """Stochastic Resilient Backpropigation.
    """
    # make sure x and g are numpy arrays
    x = np.asarray(x)
    g = np.asarray(g)

    # make sure x and g have same size
    assert len(x) == len(g)

    # number of observations
    nObs = len(x)

    params = optable.parameters()
    paramsStart = params.copy()

    # initialize all step sizes to stepInitial
    steps = np.ones_like(params) * stepInitial

    # initialize unknown error to infinity
    error = np.inf

    # initialize gradient to zero in order to yield no flips
    grad = np.zeros_like(params)

    paramTrace = []
    stepTrace = []
    errorTrace = []

    # indices into x and g used to select minibatches
    batchInd = np.arange(nObs)

    # current round of minibatches
    curRound = 0

    iteration = 0

    # termination reason
    reason = ''

    # for each round
    done = False
    while not done:
        if verbose:
            print('=======')
            print('round: %d' % curRound)
            print('error: %.5f' % optable.error(x=x, g=g))
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
            # end index into current minibatch
            end = start + batchSize

            # don't process last minibatch
            # if smaller than batchSize
            if end > nObs:
                break

            # select current batch
            xMini = xShuff[start:end]
            gMini = gShuff[start:end]

            # compute value of the error function and the gradient
            errorPrev = error
            gradPrev = grad
            error, grad = optable.gradient(*args, x=x, g=g, returnError=True, **kwargs)

            if verbose:
                print('%d %d %d %6f' % (iteration, curRound, curBatch, error))

            if callback is not None:
                callback(optable, iteration, paramTrace, errorTrace)

            # keep parameter history if requested
            if pTrace:
                paramTrace.append(params.copy())

            # keep step trace if requested
            if sTrace:
                stepTrace.append(steps.copy())

            # keep error function history if requested
            if eTrace:
                errorTrace.append(error)

            # terminate if maximum iterations reached
            if iteration >= maxIter:
                reason = 'maxiter'
                done = True
                break

            # terminate if desired accuracy reached
            if error < accuracy:
                reason = 'accuracy'
                done = True
                break

            # terminate if desired precision reached
            if np.abs(error - errorPrev) < precision:
                reason = 'precision'
                done = True
                break

            # terminate if the error function diverges
            if error > divergeThresh:
                reason = 'diverge'
                done = True
                break

            flips = grad * gradPrev

            # decrease step sizes where gradient flipped
            flipsNeg = np.where(flips < 0.0)[0]
            steps[flipsNeg] *= stepDown
            steps[...] = np.maximum(steps, stepMin)

            # increase step sizes where gradient did not flip
            flipsPos = np.where(flips > 0.0)[0]
            steps[flipsPos] *= stepUp
            steps[...] = np.minimum(steps, stepMax)

            params[...] += steps * -np.sign(grad)

            # move mini-batch forward
            start += batchSize

            # increment iteration counter
            iteration += 1

            # increment batch counters
            curBatch += 1

        # increment round counter
        curRound += 1

        if curRound >= maxRound:
            reason = 'maxround'
            done = True
            break

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {}
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if sTrace: result['sTrace'] = stepTrace
    if eTrace: result['eTrace'] = errorTrace

    return result


if __name__ == '__main__':
    demoRProp()
    plt.show()
