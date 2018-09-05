import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt

from . import tests


def sciopt(optable,
           method='CG', options=None, 
           maxIter=1000, precision=1.0e-10,
           pTrace=False, eTrace=False,
           callback=None, verbose=False,
           *args, **kwargs):
    """Wrapper for scipy optimization routines.
    """
    # get view of parameters to optimize
    params = optable.parameters()

    paramTrace = []
    errorTrace = []

    def errFunc(p):
        params.flat[...] = p
        return optable.error(*args, **kwargs)

    if method in ('CG', 'BFGS', 'Newton-CG', 'dogleg', 'trust-ncg'):
        def gradFunc(p):
            params.flat[...] = p
            return optable.gradient(*args, returnError=False, **kwargs)
    else:
        gradFunc = None

    if method == 'Newton-CG':
        def grad2Func(p):
            params.flat[...] = p
            return optable.gradient2(*args, returnError=False, **kwargs)
    else:
        grad2Func = None

    def cb(p):
        cb.iteration += 1

        if eTrace or verbose:
            error = optable.error(*args, **kwargs)

        if verbose:
            print '%d %6f' % (cb.iteration, error)

        # keep parameter history if requested
        if pTrace:
            paramTrace.append(params.copy())

        # keep error history if requested
        if eTrace:
            errorTrace.append(error)

        if callback is not None:
            callback(optable, cb.iteration, paramTrace, errorTrace)

    cb.iteration = 0

    if options is None:
        options = {}
    options['maxiter'] = maxIter

    optres = spopt.minimize(fun=errFunc, method=method,
                x0=params, tol=precision,
                jac=gradFunc, hess=grad2Func,
                options=options, callback=cb)

    if verbose:
        print optres
        print '\n'

    params.flat[...] = optres['x']

    result = {}
    result['error'] = optres['fun']
    result['params'] = params
    result['iteration'] = cb.iteration
    result['reason'] = optres['message']

    if pTrace: result['pTrace'] = paramTrace
    if eTrace: result['eTrace'] = errorTrace

    return result

def demoScioptPowell():
    rosen = tests.Rosen(optimFunc=sciopt, method='Powell', verbose=True, options={'maxfev': 1000})
    rosen.plot()

def demoScioptBFGS():
    rosen = tests.Rosen(optimFunc=sciopt, method='BFGS', verbose=True)
    rosen.plot()

def demoScioptCG():
    rosen = tests.Rosen(optimFunc=sciopt, method='CG', verbose=True)
    rosen.plot()


if __name__ == '__main__':
    #demoScioptPowell()
    #demoScioptBFGS()
    demoScioptCG()
    plt.show()
