import matplotlib.pyplot as plt
import numpy as np

from . import tests


def steepest(optable,
        learningRate=0.01, learningRateFinal=None, momentum=0.1,
        accuracy=0.0, precision=1.0e-10, divergeThresh=1.0e10,
        maxIter=1000, pTrace=False, eTrace=False,
        callback=None, verbose=False, *args, **kwargs):
    """Steepest Gradient Descent.
    Requires a first-order gradient estimate.

    Args:
        optable:
        
        learningRates:      Initial learning learning rate.

        finalLearningRates: Final learning rate.  If None (default),
                            the learning rate remains constant its
                            initial value.  Otherwise, the learning
                            rate decays linearly so that it would
                            reach learningRateFinal after maxIter
                            iterations.

        accuracy:           Terminate if current value of the
                            error funciton falls below this value.

        precision:          Terminate if change in the error
                            function falls below this value.

        divergeThresh:      Terminate if the value of the error
                            function exceeds this value.

        maxIter:            Terminate once current iteration reaches
                            this value.

        pTrace:             If True, a tuple of matrices (one for
                            each parameter matrix) is included in
                            the final result that contains a
                            history of the parameters during
                            optimization.  If False (default),
                            then a history is not kept.

        eTrace:             If True, an array containing a history
                            of the error function during
                            optimization is included in the final
                            result.  If False (default), then a
                            history is not kept.

        verbose:            Print extra information to standard out
                            during the training procedure.
    
        args, kwargs:       Arguments passed to opt.gradients.

    Returns:
        A dictionary containing the following keys:

        error:      Final value of the error function.

        iteration:  The number of iterations performed.

        reason:     A string describing the reason for termination.

        eTrace:     A tuple containing the value of the error
                    function at each iteration.  Only returned
                    if eTrace is True.

        pTrace:     A tuple containing a copy of the parameters
                    at each iteration.  Only returned if pTrace
                    is True.
    """
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
    reason = ''

    iteration = 0
    while True:
        # compute value of the error function and the gradient
        errorPrev = error
        error, grad = optable.gradient(*args, returnError=True, **kwargs)

        curLearningRate = learningRate * np.exp(-iteration * learningRateDecay)
        #curLearningRate = learningRate + learningRateDecay * iteration

        if verbose:
            print '%d %3f %6f' % (iteration, curLearningRate, error)

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

        # terminate if maximum iterations reached
        if iteration >= maxIter:
            reason = 'maxiter'
            break

        # move in direction of negative gradient
        #params -= curLearningRate * grad
        velocity[...] = momentum * velocity + curLearningRate * grad

        params -= velocity

        # increment iteration counter
        iteration += 1

    if verbose:
        print reason

    # save result into a dictionary
    result = dict()
    result['error'] = error
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def demoSteepest():
    rosen = tests.Rosen(optimFunc=steepest, maxIter=20000,
                learningRate=0.0001, momentum=0.1, verbose=True)
    rosen.plot()


if __name__ == '__main__':
    demoSteepest()
    plt.show()
