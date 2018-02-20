import matplotlib.pyplot as plt
import numpy as np

from cebl import util

from . import tests


def scg(optable,
        betaMin=1.0e-15, betaMax=1.0e100,
        accuracy=0.0, precision=1.0e-10,
        divergeThresh=1.0e10, maxIter=1000,
        pTrace=False, eTrace=False,
        callback=None, verbose=False, *args, **kwargs):
    """Scaled Conjugate Gradients
    Requires a first-order gradient estimate.

    Args:
        optable:

        betaMin:            Lower bound on scale.

        betaMax:            Upper bound on scale.

        accuracy:           Terminate if current value of the error
                            funciton falls below this value.

        precision:          Terminate if change in the error function
                            falls below this value.

        divergeThresh:      Terminate if the value of the error function
                            exceeds this value.

        maxIter:            Terminate once current iteration reaches
                            this value.

        pTrace:             If True, a list of matrices (one for each
                            parameter matrix) is included in the final
                            results that contains a history of the
                            parameters during optimization.  If False
                            (default), then a history is not kept.

        eTrace:             If True, an array containing a history of
                            the error function during optimization
                            is included in the final results.  If False
                            (default), then a history is not kept.

        callback:           

        verbose:            Print extra information to standard out during
                            the training procedure.
    
        args, kwargs:       Additional arguments passed to optable.error
                            and optable.gradient.

    Returns:
        A dictionary containing the following keys:

        params:     A numpy array containing the optimized parameters.

        error:      Final value of the error function.

        iteration:  The number of iterations performed.

        reason:     A string describing the reason for termination.

        eTrace:     A list containing the value of the error
                    function at each iteration.  Only returned
                    if eTrace is True.

        pTrace:     A list containing a copy of the parameters
                    at each iteration.  Only returned if pTrace
                    is True.

    Refs:
        @article{moller1993scaled,
          title={A scaled conjugate gradient algorithm for fast supervised learning},
          author={M{\o}ller, Martin Fodslette},
          journal={Neural networks},
          volume={6},
          number={4},
          pages={525--533},
          year={1993},
          publisher={Elsevier}
        }

        @book{nabney2002netlab,
          title={NETLAB: algorithms for pattern recognition},
          author={Nabney, Ian},
          year={2002},
          publisher={Springer}
        }
    """
    params = optable.parameters()

    # total number of parameters to optimize
    nParam = params.size

    # machine precision for parameter data type
    #eps = np.finfo(params.dtype).eps
    tiny = np.finfo(params.dtype).tiny

    error, grad = optable.gradient(*args, returnError=True, **kwargs)

    errorPrev = error
    gradPrev = grad

    direction = -grad

    sigma0 = 1.0e-4

    # initial scale
    beta = 1.0

    # force calculation of directional derivatives
    success = True

    # total number of successes
    nSuccess = 0

    if pTrace:
        paramTrace = [params.copy()]
    else:
        paramTrace = []

    if eTrace:
        errorTrace = [error]
    else:
        errorTrace = []

    # termination reason
    reason = ''

    iteration = 0

    if verbose:
        print('%d %6f' % (iteration, error))

    if callback is not None:
        callback(optable, iteration, paramTrace, errorTrace, success)

    while True:
        paramsStart = params.copy()

        if success:
            mu = direction.dot(grad)
            if mu >= 0.0:
                direction = -grad
                mu = direction.dot(grad)

            #kappa = util.capZero(direction.dot(direction))
            kappa = direction.dot(direction)
            #if kappa < eps:
            if kappa < tiny:
                reason = 'kappa'
                break

            sigma = sigma0 / np.sqrt(kappa)

            params += sigma * direction
            gradPlus = optable.gradient(*args, returnError=False, **kwargs)
            params[...] = paramsStart

            theta = direction.dot(gradPlus - grad) / sigma

        # increase effective curvature and evaluate step size alpha
        delta = theta + beta * kappa
        if delta <= 0.0:
            delta = beta * kappa
            beta -= theta / kappa
        alpha = -mu / delta

        # calculate comparison ratio
        params += alpha * direction
        #errorTry, gradTry = optable.gradient(*args, returnError=True, **kwargs)
        errorTry = optable.error(*args, **kwargs)

        Delta = 2.0 * (errorTry - error) / (alpha * mu)
        if Delta >= 0.0:
            success = True
            nSuccess += 1

            errorPrev = error
            error = errorTry

            ##gradPrev = grad
            #grad = gradTry
            ##grad = optable.gradient(*args, returnError=False, **kwargs)

            # keep parameter history if requested
            if pTrace:
                paramTrace.append(params.copy())

            # keep error function history if requested
            if eTrace:
                errorTrace.append(error)

        else:
            success = False
            params[...] = paramsStart
            if verbose:
                print('No success')

        # increment iteration counter
        iteration += 1

        if verbose:
            print('%d %6f' % (iteration, error))

        if callback is not None:
            callback(optable, iteration, paramTrace, errorTrace)

        # terminate if maximum iterations reached
        if iteration >= maxIter:
            reason = 'maxiter'
            break

        if success:
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

            gradPrev = grad
            grad = optable.gradient(*args, returnError=False, **kwargs)

        # adjust beta according to comparison ratio
        if Delta < 0.25:
            beta = np.min((4.0*beta, betaMax))

        elif Delta > 0.75:
            beta = np.max((0.5*beta, betaMin))

        # update search direction using Polak-Ribiere formula, or
        # re-start in direction of negative gradient after nParam steps
        if nSuccess == nParam:
            direction = -grad
            nSuccess = 0

        elif success:
            gamma = (gradPrev - grad).dot(grad / mu)
            direction = gamma * direction - grad

    if verbose:
        print(reason)

    # save result into a dictionary
    result = {}
    result['params'] = params
    result['error'] = error
    result['iteration'] = iteration
    result['reason'] = reason

    if pTrace: result['pTrace'] = paramTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def demoSCG():
    rosen = tests.Rosen(optimFunc=scg, verbose=True)
    rosen.plot()


if __name__ == '__main__':
    demoSCG()
    plt.show()
