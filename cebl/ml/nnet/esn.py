import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spsparse
import scipy.optimize as spopt

from cebl import util

from ..linreg import RidgeRegression
from ..regression import Regression

from . import transfer


def rayleigh(x, eigVal=None, eigVec=None, precision=1e-16, maxIter=500, verbose=False):
    """Use rayleigh quotent iterations to find the spectral radius
    of a square matrix.

    Args:
        x:          A numpy array containing the matrix for which we seek to
                    determine the spectral radius.

        eigVal:     An initial guess for the largest eigenvalue.  If None
                    (default) then the initial guess will be 0.5 times
                    the number of rows of x.

        eigVec:     An initial guess for the eigenvector associated with
                    the largest eigenvalue of x.  If None (default) then
                    the initial guess will be drawn from the random
                    uniform distribution with the same range as x.

        precision:  Terminate if the change in our estimate of the spectral
                    radius falls below this value.

        maxIter:    Terminate if the number of iterations exceeds this value.

        verbose:    Print progress information at each iteration.

    References:
        https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration
    """
    x = np.asarray(x)

    if eigVal is None:
        eigVal = x.shape[0] * 0.5

    if eigVec is None:
        eigVec = np.random.uniform(-np.min(x), np.max(x),
                    size=x.shape[1]).astype(x.dtype, copy=False)
    else:
        eigVec = eigVec.asarray(eigVec)

    def norm2(m):
        return np.linalg.norm(m, ord=2)

    iteration = 1
    while (True):
        if verbose:
            print iteration, eigVal

        eigVec = eigVec / norm2(eigVec)

        y = (x - eigVal * np.eye(x.shape[0], dtype=x.dtype))
        y = np.linalg.solve(y, eigVec)
        l = y.T.dot(eigVec)
        eigValPrev = eigVal
        eigVal += 1.0/l

        iteration += 1

        if np.abs(eigVal-eigValPrev) < precision:
            reason = 'precision'
            break

        if iteration > maxIter:
            reason = 'maxiter'
            break

    results = dict()
    results['eigVal'] = eigVal
    results['eigVec'] = eigVec
    results['reason'] = reason
    results['iterations'] = iteration

    return results

def miniRProp(x, errFunc, step=0.1, stepUp=1.01, stepDown=0.6,
              maxIter=1000, accuracy=0.0, precision=0.0, verbose=False):
    err = errFunc(x)

    iteration = 0
    while True:
        if verbose:
            print iteration, step, err

        x += step

        errPrev = err
        err = errFunc(x)

        if err > errPrev:
            x -= step
            step = -step * stepDown
        else:
            step *= stepUp

        iteration += 1

        if iteration >= maxIter:
            if verbose:
                print 'Maximum iterations %d reached.' % iteration
            break

        if np.abs(err) < accuracy:
            break

        if np.abs(err-errPrev) < precision:
            break

    return x


class EchoStateNetworkReservoir(object):
    """Echo State Network reservoir.
    """
    def __init__(self, x, nRes=1024, rwScale=0.95, rwConn=0.01,
                 iwScale=0.3, iwConn=0.2, transFunc=transfer.tanh,
                 sparse=None, actCacheSize=0, verbose=False):
        x = util.segmat(x)
        self.nIn = x.shape[2]
        self.dtype = x.dtype

        self.nRes = nRes
        self.transFunc = transFunc

        if sparse is None:
            if rwConn < 0.05:
                self.sparse = True
            else:
                self.sparse = False
        else:
            self.sparse = sparse

        self.actCache = util.Cache(actCacheSize)

        self.verbose = verbose

        # (ns, ni+nr) x (ni+nr, nr)
        #self.hw = np.empty((self.nIn+1+self.nRes,self.nRes), dtype=self.dtype)
        #iw = self.hw[:self.nIn+1,:]
        #rw = self.hw[self.nIn+1:,:]

        iw = self.initIW(iwScale, iwConn)
        rw = self.initRW(rwScale, rwConn)
        self.hw = np.vstack((iw,rw))

        if self.sparse:
            # is csc or csr faster? XXX - idfah
            self.hw = spsparse.csr_matrix(self.hw, dtype=self.dtype)

        self.scaleIW(x)

        ##iw = self.hw[:self.nIn+1,:]
        ##rw = self.hw[self.nIn+1:,:]
        ##if self.sparse:
        ##    iw = iw.todense()
        ##    rw = rw.todense()
        ##print 'iw.shape: ', iw.shape
        ##print 'iw min/max: ', (np.min(iw), np.max(iw))
        ##print 'iwMult: ', self.iwMult
        ##print 'rw.shape: ', rw.shape
        ##l = np.max(np.abs(np.linalg.eigvals(rw)))
        ##print 'rwScale: ', l

    def initIW(self, iwScale, iwConn):
        if self.verbose:
            print 'Initializing input weights...'

        self.iwMult = 1.0
        self.iwScale = iwScale
        self.iwConn = iwConn

        iw = np.random.uniform(-self.iwMult, self.iwMult,
                    size=(self.nIn+1,self.nRes))
        connMask = np.random.random(iw.shape) > self.iwConn
        connMask[0,0] = False
        iw[connMask] = 0.0

        return iw

    def initRW(self, rwScale, rwConn):
        if self.verbose:
            print 'Initializing recurrent weights...'

        self.rwScale = rwScale
        self.rwConn = rwConn

        rw = np.random.uniform(-1.0, 1.0,
                    size=(self.nRes, self.nRes))
        connMask = np.random.random(rw.shape) > self.rwConn
        #connMask[0,0] = False
        rw[connMask] = 0.0

        loneNeurons = np.where(np.all(rw == 0.0, axis=1))[0]
        newConns = np.random.randint(0, self.nRes, size=loneNeurons.size)
        rw[loneNeurons,newConns] = np.random.uniform(-1.0, 1.0, size=loneNeurons.size)

        if self.verbose:
            print 'Eliminated %d lone reservor units.' % loneNeurons.size

        if self.sparse:
            if self.verbose:
                print 'Using sparse linalg to find spectral radius...'
            try:
                ncv = int(np.max((10, rwConn*self.nRes)))
                l = np.abs(spsparse.linalg.eigs(spsparse.csr_matrix(rw, dtype=self.dtype),
                           k=1, ncv=ncv, tol=0, return_eigenvectors=False)[0])
            except spsparse.linalg.ArpackNoConvergence as e:
                if self.verbose:
                    print 'ARPACK did not converge, using dense matrices.'
                l = np.max(np.abs(np.linalg.eigvals(rw)))
        else:
            #if self.verbose:
            #    print 'Using dense Rayleigh iteration to find spectral radius...'

            # need to take a closer look at rayleigh, finding all eigenvalues for now XXX - idfah
            # rayleigh only works for positive matrices
            ##rayl = rayleigh(rw, eigVal=self.nRes*0.5*self.rwConn, verbose=self.verbose)
            #if rayl['reason'] != 'precision':
            #    print 'ESN Warning: rayleigh did not converge: ' + rayl['reason']
            #l = rayl['eigVal']

            if self.verbose:
                print 'Finding spectral radius using dense matrices...'
            l = np.max(np.abs(np.linalg.eigvals(rw)))

        rw[:,:] *= self.rwScale / l
        #rw[:,:] *= np.sign(np.random.random(rw.shape) - 0.5)

        return rw

    def eval(self, x, context=None, returncontext=False):
        x = util.segmat(x)

        cacheAct = False
        if returncontext is False and \
           context is None and \
           self.actCache.getMaxSize() > 0:
                key = util.hashArray(x)
                if key in self.actCache:
                    #print 'cache hit.'
                    return self.actCache[key]
                else:
                    #print 'cache miss.'
                    cacheAct = True

        nSeg = x.shape[0]
        nObs = x.shape[1]
        nIn  = x.shape[2]

        act = np.empty((nSeg, nObs, self.nRes))

        if context is None:
            context = np.zeros((nSeg, self.nRes), dtype=self.dtype)

        xt = np.empty((nSeg,nIn+self.nRes))

        hwT = self.hw[:-1].T

        for t in xrange(nObs):
            xt[:,:nIn] = x[:,t,:]
            xt[:,nIn:] = context

            if self.sparse:
                # need to have w first for sparse matrix
                # does not appear faster to convert xt to csr
                act[:,t,:] = self.transFunc(hwT.dot(xt.T).T + self.hw[-1])
            else:
                act[:,t,:] = self.transFunc(xt.dot(self.hw[:-1]) + self.hw[-1])

            context = act[:,t,:]

        if cacheAct:
            self.actCache[key] = act

        if returncontext:
            return act, context
        else:
            return act

    def setIWMult(self, mult):
        m = mult / float(self.iwMult)

        if self.sparse:
            v = spsparse.lil_matrix((self.hw.shape[0], self.hw.shape[0]))
            v.setdiag(np.ones(v.shape[0], dtype=self.dtype))
            v.setdiag([m,]*(self.nIn+1))

            self.hw = v*self.hw

            # good for debugging above
            #w = self.hw.todense()
            #w[:self.nIn+1,:] *= m
            #self.hw = spsparse.csc_matrix(w)
        else:
            self.hw[:self.nIn+1,:] *= m

        self.iwMult = mult

    def scaleIW(self, x, method='brentq'):
        if self.verbose:
            print 'Scaling input weights...'

        self.actCache.disable()

        scale = self.iwScale

        maxIter = 100
        accuracy = 1e-4

        method = method.lower()

        def stdErr(m):
            self.setIWMult(m)
            act = self.eval(x)
            err = np.abs(np.std(act)-scale)
            if self.verbose:
                print 'scale, mult: ', (np.std(act), m)
            return err

        def stdRoot(m):
            self.setIWMult(m)
            act = self.eval(x)
            err = np.std(act) - scale
            if self.verbose:
                print 'scale, mult: ', (np.std(act), m)
            return err

        if method == 'rprop':
            miniRProp(0.75, errFunc=stdErr,
                      maxIter=maxIter, accuracy=accuracy)
                      #verbose=self.verbose)

        elif method == 'brentq':
            m, r = spopt.brentq(stdRoot, 1.0e-5, 10.0, xtol=accuracy, full_output=True)
            if self.verbose:
                print 'brentq iterations: %d' % r.iterations

        elif method == 'simplex':
            r = spopt.minimize(stdErr, scale, method='Nelder-Mead', tol=accuracy,
                    options={'maxiter': 100})
            m = r.x

        else:
            raise Exception('Invalid scaleIW method %s.' % method)

        self.actCache.enable()

    def setRWScale(self, x, scale):
        # why does this method not work? XXX - idfah
        m = scale / float(self.rwScale)

        if self.sparse:
            v = spsparse.lil_matrix((self.hw.shape[0], self.hw.shape[0]))
            d = np.ones(v.shape[0], dtype=self.dtype)
            d[self.nIn+1:] = m
            v.setdiag(d)

            self.hw = v*self.hw

            # good for debugging above
            #w = self.hw.todense()
            #w[self.nIn+1:,:] *= m
            #self.hw = spsparse.csc_matrix(w)
        else:
            self.hw[self.nIn+1:,:] *= m

        self.scaleIW(x)

    def plotActDensity(self, x, ax=None, **kwargs):
        act = self.eval(x, **kwargs)

        t = np.arange(0.0,4.0,0.01)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel('Density')
            ax.set_ylabel('Reservoir Activation')

        n, bins, patches = ax.hist(act.ravel(), normed=True,
                        orientation='horizontal', label='Activations')

        lines = ax.plot(np.linspace(0.0,np.max(n),t.size), np.tanh(t-2.0),
                        linewidth=2, label=r'$\phi$') # label=r'$\phi='+self.phi.__name__)
        leg = ax.legend(loc='lower right')

        return {'ax': ax, 'n': n, 'bins': bins, 'patches': patches, 'lines': lines, 'leg': leg}

    def plotWeightImg(self, ax=None):
        if self.sparse:
            hw = self.hw.todense()
        else:
            hw = self.hw

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        img = ax.imshow(hw[self.nIn:,:], interpolation='none')

        return {'ax': ax, 'img': img}


class ESNReservoir(EchoStateNetworkReservoir):
    pass


class EchoStateNetworkFromReservoir(Regression):
    def __init__(self, x, g, reservoir, transient=0, sideTrack=False,
                 verbose=False, **kwargs):
        x = util.segmat(x)
        g = util.segmat(g)
        self.dtype = np.result_type(x.dtype, g.dtype)

        nIn = x.shape[2]
        nOut = g.shape[2]
        Regression.__init__(self, nIn, nOut)

        self.reservoir = reservoir
        self.transient = transient
        self.sideTrack = sideTrack

        self.verbose = verbose

        self.train(x, g, **kwargs)

    def train(self, x, g, readoutClass=RidgeRegression, **kwargs):
        x = util.segmat(x)
        g = util.segmat(g)

        act = self.reservoir.eval(x)
        act = self.addSideTrack(x, act)

        actf = act.reshape((-1, act.shape[-1]))

        if g.ndim == 3:
            gf = g.reshape((-1, g.shape[-1]))
        else:
            gf = g.ravel()

        if self.verbose:
            print 'Training readout layer...'

        self.readout = readoutClass(actf[self.transient:], gf[self.transient:], **kwargs)

    def eval(self, x, returnTransient=False, **kwargs):
        x = util.segmat(x)

        act = self.reservoir.eval(x)
        act = self.addSideTrack(x, act)
        actf = act.reshape((-1, act.shape[-1]))

        yf = self.readout.eval(actf, **kwargs)
        y = yf.reshape((act.shape[0], x.shape[1], -1))

        if x.ndim == 2:
            y = y.squeeze(axis=2)

        if not returnTransient:
            y = y[:,self.transient:]

        return y

    def iterate(self, context):
        pass

    def addSideTrack(self, x, act):
        x = util.segmat(x)

        if self.sideTrack:
            if self.verbose:
                print 'adding side track...'
            return np.concatenate((x, act), axis=2)
        else:
            return act

class ESNFromReservoir(EchoStateNetworkFromReservoir):
    pass


class EchoStateNetwork(EchoStateNetworkFromReservoir):
    def __init__(self, x, g, nRes=1024, rwScale=0.95, rwConn=0.01,
                 iwScale=0.3, iwConn=0.2, transFunc=transfer.tanh,
                 transient=0, sideTrack=False, sparse=None,
                 verbose=False, **kwargs):

        reservoir = EchoStateNetworkReservoir(x,
            nRes=nRes, rwScale=rwScale, rwConn=rwConn,
            iwScale=iwScale, iwConn=iwConn, transFunc=transFunc,
            sparse=sparse, verbose=verbose)

        EchoStateNetworkFromReservoir.__init__(self, x, g,
            reservoir=reservoir, transient=transient, sideTrack=sideTrack,
            verbose=verbose, **kwargs)

class ESN(EchoStateNetwork):
    pass


def demoESP():
    t = np.arange(20)

    sig = np.cos(t)

    impulse = np.zeros(20)
    impulse[5] = 10

    sigi = sig + impulse

    res = ESNReservoir((sig,), nRes=20, rwScale=0.95)

    act = res.eval((sig,))[0]
    acti = res.eval((sigi,))[0]

    fig = plt.figure()

    impulseAx = fig.add_subplot(2,1,1)
    impulseAx.plot(sig, color='grey', linewidth=3)
    impulseAx.plot(sigi, color='red')

    actAx = fig.add_subplot(2,1,2)
    actAx.plot(act, color='black', linewidth=2)
    actAx.plot(acti, color='red')

def demoESNTXOR():
    def xor(a, b):
        if (a == True) and (b == True):
            return False

        if a or b:
            return True

        return False

    n = 500
    horizon = 5
    transient = horizon+1

    x = np.random.randint(0,2, size=n).astype(np.float32)
    g = np.array([int(xor(x[i-horizon], x[i-horizon-1])) if i > horizon
            else 0 for i in xrange(len(x))], dtype=np.float32)

    net = ESN(x[None,...], g[None,...], nRes=2048, rwScale=0.85, rwConn=0.01, iwScale=0.3,
              transient=transient, sideTrack=False, sparse=True, verbose=True)

    # redo for test data
    x = np.random.randint(0,2, size=n)
    g = np.array([int(xor(x[i-horizon], x[i-horizon-1])) if i > horizon
            else 0 for i in xrange(len(x))], dtype=np.float)

    out = net.eval(x[None,...], returnTransient=True)[0]

    net.reservoir.plotActDensity(x[None,...])
    net.reservoir.plotWeightImg()

    fig = plt.figure()
    axTarg = fig.add_subplot(2,1,1)
    axTarg.bar(range(len(g)), g)
    axTarg.set_xlim((0, len(g)))
    axTarg.set_ylim((0.0,1.0))

    axOut = fig.add_subplot(2,1,2)
    axOut.bar(range(len(g)), out)
    axOut.set_xlim((0, len(g)))
    axOut.set_ylim((0.0,1.0))

def demoESNSine():
    time = np.linspace(0.0,10.0*np.pi,5000)
    s1 = np.sin(time)
    s2 = np.cos(time)
    s = np.vstack((s1,s2)).T

    x = s[None,:-1]
    g = s[None,1:]

    res = ESNReservoir(x, nRes=1024, rwScale=0.75, verbose=False)
    model = ESNFromReservoir(x, g, res, sideTrack=True, verbose=False)

    pred = model.eval(x, returnTransient=True)
    resid = g - pred

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(time, s, color='blue')
    ax.plot(time[1:], pred[0], color='red')

    ax.autoscale(tight=True)


if __name__ == '__main__':
    #demoESP()
    demoESNTXOR()
    #demoESNSine()
    plt.show()
