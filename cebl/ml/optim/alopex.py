import matplotlib.pyplot as plt
import numpy as np

from cebl import util

from . import tests


def alopex(optable, stepSize=0.0075, tempInit=10000, tempIter=20,
           accuracy=0.0, precision=0.0, divergeThresh=1.0e10, maxIter=10000,
           pTrace=False, tTrace=False, eTrace=False, callback=None,
           verbose=False, *args, **kwargs):
    """ ALgorithm Of Pattern EXtraction (ALOPEX)

    Args:
        optable:

        stepSize:           Step size.

        tempInit:

        tempIter:

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

        tTrace:

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
    """
    params = optable.parameters()

    # initial error
    error = optable.error(*args, **kwargs)
    errorPrev = error

    # intial temperature
    temp = tempInit

    # running correlation
    corrRun = 0.0

    # probability of taking a negative step
    probs = np.ones_like(params) * 0.5

    # weight pertibations
    dw = np.empty_like(params)
    dwPrev = np.empty_like(params)

    paramTrace = [params.copy()]
    tempTrace = [temp]
    errorTrace = [error]

    # termination reason
    reason = ''

    iteration = 0

    if verbose:
        print('%d %6f' % (iteration, error))

    if callback is not None:
        callback(optable, iteration, paramTrace, errorTrace)

    while True:
        # corr err dw  action
        #  +    +   +   -
        #  -    -   +   +
        #  -    +   -   +
        #  +    -   -   -

        draw = np.random.random(params.shape)

        stepsNeg = np.where(draw <  probs)[0]
        stepsPos = np.where(draw >= probs)[0]

        dwPrev[...] = dw
        dw[stepsNeg] = -stepSize
        dw[stepsPos] =  stepSize

        params += dw

        errorPrev = error
        error = optable.error(*args, **kwargs)

        # increment iteration counter
        iteration += 1

        if verbose:
            print('%d %6f' % (iteration, error))

        if callback is not None:
            callback(optable, iteration, paramTrace, errorTrace)

        # keep parameter history if requested
        if pTrace:
            paramTrace.append(params.copy())

        # keep temperature trace if requested
        if tTrace:
            tempTrace.append(temp)

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

        # current change in error
        de = error - errorPrev

        # correlation metric
        corr = de * dw

        corrRun += (np.abs(de) * np.sum(np.abs(dw))) / params.size

        if (iteration % tempIter) == 0:
            # new temperature is average correlation
            # since the previous temperature update
            temp = corrRun / tempIter

            # reset running correlation
            corrRun = 0.0

            if verbose:
                print('Cooling: %f' % temp)

        # probability of taking negative step
        # is drawn from the Boltzman Distribution
        probs[...] = 1.0 / util.capZero(1.0 + np.exp(-corr/temp))

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {}
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if tTrace: result['tTrace'] = tTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def alopexb(optable, stepSize=0.005, forgetFactor=0.5,
            accuracy=0.0, precision=0.0, divergeThresh=1.0e10, maxIter=10000,
            pTrace=False, tTrace=False, eTrace=False, callback=None,
            verbose=False, *args, **kwargs):
    """ ALgorithm Of Pattern EXtraction (ALOPEX)

    Args:
        optable:

        stepSize:           Step size.

        forgetFactor:

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

        tTrace:

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
    """
    params = optable.parameters()

    # initial error
    error = optable.error(*args, **kwargs)
    errorPrev = error

    # decaying average of correlation
    corrRun = 0.0

    # probability of taking a negative step
    probs = np.ones_like(params) * 0.5

    # weight pertibations
    dw = np.empty_like(params)
    dwPrev = np.empty_like(params)

    paramTrace = [params.copy()]
    errorTrace = [error]

    # termination reason
    reason = ''

    iteration = 0

    if verbose:
        print('%d %6f' % (iteration, error))

    if callback is not None:
        callback(optable, iteration, paramTrace, errorTrace)

    while True:
        # corr err dw  action
        #  +    +   +   -
        #  -    -   +   +
        #  -    +   -   +
        #  +    -   -   -

        draw = np.random.random(params.shape)

        stepsNeg = np.where(draw <  probs)[0]
        stepsPos = np.where(draw >= probs)[0]

        dwPrev[...] = dw
        dw[stepsNeg] = -stepSize
        dw[stepsPos] =  stepSize

        params += dw

        errorPrev = error
        error = optable.error(*args, **kwargs)

        # increment iteration counter
        iteration += 1

        if verbose:
            print('%d %6f' % (iteration, error))

        if callback is not None:
            callback(optable, iteration, paramTrace, errorTrace)

        # keep parameter history if requested
        if pTrace:
            paramTrace.append(params.copy())

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

        # current change in error
        de = error - errorPrev

        corrRun = (((forgetFactor-1.0) * corrRun) +
                   (forgetFactor * (np.abs(de) * np.sum(np.abs(dw))) / params.size))

        # correlation metric
        corr = de * dw

        # probability of taking negative step
        # is drawn from the Boltzman Distribution
        probs[...] = 1.0 / util.capZero(1.0 + np.exp(-corr/corrRun))

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {}
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if tTrace: result['tTrace'] = tTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def demoALOPEX():
    rosen = tests.Rosen(optimFunc=alopex, accuracy=0.01, maxIter=np.inf, tempIter=20,
                        stepSize=0.0005, verbose=True)

    rosen.plot()

if __name__ == '__main__':
    demoALOPEX()
    plt.show()
