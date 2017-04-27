import matplotlib.pyplot as plt
import matplotlib.cm as pltcm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import tests


def pso(optable, nParticles=10, pInit=0.5, vInit=0.01,
        #momentum=0.9, pAttract=0.3, gAttract=0.3,
        momentum=0.9, pAttract=0.3, gAttract=0.3,
        accuracy=0.0, precision=0.0,
        divergeThresh=1.0e10, maxIter=10000,
        eTrace=False, pTrace=False,
        callback=None, verbose=False,
        *args, **kwargs):
    """Particle Swarm Optimization (PSO).
    """
    params = optable.parameters()

    #pParams = [optable.parameters() + np.random.normal(scale=pInit, size=params.size)
    #           for i in xrange(nParticles)]
    pParams = [optable.parameters() + np.random.uniform(-pInit, pInit, size=params.size)
               for i in xrange(nParticles)]

    pVeloc = [np.random.uniform(-vInit, vInit) for i in xrange(nParticles)]

    pError = []
    pBest = [p for p in pParams]

    gError = np.inf
    for p in pParams:
        params[...] = p
        e = optable.error(*args, **kwargs)
        pError.append(e)

        if e < gError:
            gError = e
            gBest = p.copy()

    errorTrace = [gError]
    paramTrace = [np.vstack(pParams)]

    # termination reason
    reason = ''

    iteration = 0

    if verbose:
        print '%d %6f' % (iteration, gError)

    if callback is not None:
        callback(optable, iteration, paramTrace, errorTrace)

    while True:
        for i,p in enumerate(pParams):
            pr = np.random.random(params.size)
            gr = np.random.random(params.size)

            pVeloc[i] = (momentum * pVeloc[i] +
                         pAttract * pr * (pBest[i] - p) +
                         gAttract * gr * (gBest - p))

            p += pVeloc[i]

            params[...] = p
            e = optable.error(*args, **kwargs)

            if e < pError[i]:
                pError[i] = e

            if e < gError:
                gError = e
                gBest = p.copy()

        # increment iteration counter
        iteration += 1

        if verbose:
            print '%d %3f %6f' % (iteration, np.max(pVeloc), gError)

        if callback is not None:
            callback(optable, iteration, paramTrace, errorTrace)

        # keep error function history if requested
        if eTrace:
            errorTrace.append(gError)

        # keep error function history if requested
        if pTrace:
            paramTrace.append(np.vstack(pParams))

        # terminate if maximum iterations reached
        if iteration >= maxIter:
            reason = 'maxiter'
            break

        # terminate if desired accuracy reached
        if gError < accuracy:
            reason = 'accuracy'
            break

        # terminate if desired precision reached
        if np.abs(gError - gError) < precision:
            reason = 'precision'
            break

        # terminate if the error function diverges
        if gError > divergeThresh:
            reason = 'diverge'
            break

    params[...] = gBest

    if verbose:
        print reason

    # save result into a dictionary
    result = {}
    result['error'] = gError
    result['iteration'] = iteration
    result['reason'] = reason

    if eTrace: result['eTrace'] = errorTrace
    if pTrace: result['pTrace'] = paramTrace

    return result

def demoPSO():
    rosen = tests.Rosen(optimFunc=pso, nParticles=10, accuracy=0.01,
                maxIter=5000, verbose=True)#, initialSolution=(2.5, -2.5))

    n = 200
    rng=(-3.0,3.0, -4.0,8.0)

    x = np.linspace(rng[0], rng[1], n)
    y = np.linspace(rng[2], rng[3], n)

    xx, yy = np.meshgrid(x, y)

    points = np.vstack((xx.ravel(), yy.ravel())).T
    values = rosen.eval(points)
    zz = values.reshape((xx.shape[0], yy.shape[1]))

    fig = plt.figure(figsize=(12,6))
    axSurf = fig.add_subplot(1,2,1, projection='3d')

    surf = axSurf.plot_surface(xx, yy, zz, linewidth=1.0, cmap=pltcm.jet)
    surf.set_edgecolor('black')

    axCont = fig.add_subplot(1,2,2)
    axCont.contour(x, y, zz, 40, color='black')
    axCont.scatter(rosen.a, rosen.a**2, color='black', marker='o', s=400, linewidth=3)
    axCont.scatter(*rosen.solution, color='red', marker='x', s=400, linewidth=3)

    paramTrace = np.array(rosen.trainResult['pTrace'])
    for i in xrange(paramTrace.shape[1]):
        axCont.plot(paramTrace[:,i:,0], paramTrace[:,i:,1], color=plt.cm.jet(i/float(paramTrace.shape[1])), linewidth=1)

    fig.tight_layout()

if __name__ == '__main__':
    demoPSO()
    plt.show()
