import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

from cebl import util

from . import bandpass
from . import windows


def downsample(s, factor):
    """Downsample a discrete signal by a given factor.

    Args:
        s:      Numpy array containing the discrete signal to downsample.
                Must have shape (nObs[,nDim]).

        factor: Integer downsampling factor.  For instance, a factor of 2
                will decrease the effective sampling rate to 1/2 of its
                previous value.

    Returns:
        A numpy array containing the downsampled signal with shape
        (nObs//factor[,nDim])

    Notes:
        This function does NOT apply a lowpass filter before downsampling.
        This means that if s contains frequencies above the nyquist rate
        of the downsampled signal, then aliasing will result.  Typically,
        this is prevented by applying a lowpass filter before downsampling.
        The decimate and resample functions found in this module combine
        these steps.
    """
    # ensure we have a numpy array
    s = np.asarray(s)

    # observations strided by factor
    return s[::factor]

def upsample(s, factor):
    if s.ndim > 1:
        flattenOut = False
    else:
        flattenOut = True

    s = util.colmat(s)

    sup = s.repeat(factor).reshape((-1, s.shape[1]), order='F')

    if flattenOut:
        sup = sup.ravel()

    return sup

def decimate(s, factor, lowpassFrac=0.625, **kwargs):
    #newNyquist = 1.0/(2.0*factor)
    newNyquist = 0.5/factor

    lowpassFilter = bandpass.BandpassFilter(lowFreq=0.0,
                        highFreq=lowpassFrac*newNyquist,
                        sampRate=1.0, **kwargs)
    sFiltered = lowpassFilter.filter(s)

    return downsample(sFiltered, factor)

def interpolate(s, factor, order=8, filtType='lanczos'):
    """Interpolate (upsample) a discrete signal by a given factor using a
    Finite Impulse Response (FIR) filter.

    Args:
        s:          Numpy array containing the discrete signal to downsample.
                    Must have shape (nObs[,nDim]).

        factor:     Integer upsampling factor.  For instance, a factor of 2
                    will increase the effective sampling rate to twice that
                    of its previous value.

        order:      Order of the FIR filter.  The width of the convolution
                    window is order+1, i.e., order//2 points on both sides.

        filtType:   The type of FIR filter used for interpolation.
                    The following are currently available:
                        lanczos:        (default) Sinc function multiplied by
                                        the window consisting of the central
                                        hump of another sinc.  Lanczos is
                                        often considered to be robust and it is
                                        commonly used.  However, it does
                                        contain stopband and passband ripple.

                        sinc-blackman:  The sinc function multiplied by a
                                        blackman window.  Sinc-blackman has
                                        has a slower rolloff than lanczos but
                                        little passband ripple and less
                                        stopband ripple.

    Returns:
        A numpy array containing the interpolated signal with shape
        (nObs*factor[,nDim])

    Notes:
        The interpolated signal may not align exactly with the original
        signal because of the fencepost problem.  Consider the following:

        Original observations (length 4):  |  |  |  |
        New observations (length 4*2=8):   |**|**|**|**
    """
    if order % 2 != 0:
        raise RuntimeError('Invalid order: ' + str(order) +
            ' Must be an even integer.')

    # ensure we have a numpy array
    s = np.asarray(s)

    # filtType should be case-insensitive
    filtType = filtType.lower()

    # find shape of interpolated signal
    slShape = list(s.shape)
    slShape[0] *= factor
    # could be used instead of above to eliminate
    # padding at end but new signal would not be
    # factor*nObs in length
    #slShape[0] = slShape[0]*factor - factor+1

    # initialize interpolated signal to s with zeros padded
    # between observations to give length of interpolated signal
    # i.e., {s0, 0, 0, s1, 0, 0, s2, 0, 0} for factor=3
    sl = np.zeros(slShape, dtype=s.dtype)
    sl[::factor] = s

    # determine filter order and tap locations
    radius = order//2
    newOrder = order*factor
    taps = np.linspace(-radius, radius, newOrder+1).astype(s.dtype, copy=False)

    # generate FIR filter
    if filtType == 'lanczos':
        impulseResponse = np.sinc(taps) * windows.lanczos(newOrder+1).astype(s.dtype, copy=False)
    elif filtType == 'sinc-blackman':
        impulseResponse = np.sinc(taps) * windows.blackman(newOrder+1).astype(s.dtype, copy=False)
    else:
        raise RuntimeError('Invalid filtType: ' + str(filtType))

    # convolve with FIR filter to smooth across zero padding
    # NOTE:  there is potential for performance improvement here since
    #        zeros could be excluded from the computation XXX - idfah
    # spsig.fftconvolve might also be faster for long signals XXX - idfah
    return np.apply_along_axis(lambda v:
               np.convolve(v, impulseResponse, mode='same'),
               axis=0, arr=sl)

def resample(s, factorDown, factorUp=1, interpKwargs=None, **decimKwargs):
    """Resample a discrete signal using a factor defined as a rational number.
    This is performed by first upsampling (if necessary) and then decimating the
    signal by given factors.

    Args:
        s:              Numpy array containing the discrete signal to resample.
                        Must have shape (nObs[,nDim]).

        factorUp:       Integer upsampling factor.  For instance, a factor of
                        2 will increase the effective sampling rate to twice
                        that of its previous value.  A value of one (default)
                        will skip upsampling.

        factorDown:     Integer decimation factor.  For instance, a factor of
                        2 will decrease the effective sampling rate to 1/2 of
                        its previous value.

        interpKwargs:   Additional keyword arguments passed to interpolate.

        decimKwargs:    Additional keyword arguments passed to decimate.

    Returns:
        A numpy array containing the interpolated signal with shape
        (nObs*factorUp//factorDown[,nDim])

    Notes:
        This function is essentially a wrapper around an interpolate followed
        by decimation.  This can be used to achieve a sampling rate of any
        rational number.  See the documentation for the interpolate and
        decimate functions in this module for more information.
    """
    if interpKwargs is None:
        interpKwargs = {}

    s = np.asarray(s)
    end = s.shape[0]*factorUp//factorDown

    if factorUp > 1:
        s = interpolate(s, factorUp, **interpKwargs)

    return decimate(s, factorDown, **decimKwargs)[:end]


def demoInterpolate():
    n = 50
    factor = 10

    s = np.cumsum(np.random.normal(size=n, scale=5))
    t = np.arange(0.0, n)
    tInterp = np.linspace(0.0, n, n*factor, endpoint=False)

    sLanczos4 = interpolate(s, factor, order=4)
    sLanczos8 = interpolate(s, factor, order=8)
    sLanczos16 = interpolate(s, factor, order=16)

    plt.plot(t, s, marker='o', color='lightgrey', linewidth=3, label='Original')
    plt.plot(tInterp, sLanczos4, color='blue', label='Lanczos4')
    plt.plot(tInterp, sLanczos8, color='red', label='Lanczos8')
    plt.plot(tInterp, sLanczos16, color='green', label='Lanczos16')
    plt.title('Interpolation of a Random Walk')
    plt.legend()
    plt.tight_layout()

def demoResample():
    sampRate = 256.0
    nyquist = sampRate/2.0

    factor = 3

    t = np.linspace(0.0, 2.0, 2.0*sampRate, endpoint=False).astype(np.float32)
    f = np.linspace(0.0, nyquist, 2.0*sampRate, endpoint=False).astype(np.float32)

    chirp = spsig.chirp(t, 0.0, 2.0, nyquist).astype(np.float32)

    fig = plt.figure()

    ##axChirp = fig.add_subplot(4, 1, 1)
    ##axChirp.plot(f, chirp, color='black')
    ##axChirp.set_title('Chirp')
    ##axChirp.set_xlabel('Frequency (Hz)')
    ##axChirp.set_ylabel('Signal')
    ##axChirp.autoscale(tight=True)

    #axChirpTwiny = axChirp.twiny()
    #axChirpTwiny.plot(t, chirp, alpha=0.0)
    #axChirpTwiny.set_xlabel('Time (s)')

    fDown = downsample(f, factor)
    chirpDown = downsample(chirp, factor)

    axDown = fig.add_subplot(2, 2, 1)
    axDown.plot(f, chirp, color='lightgrey', linewidth=2)
    axDown.plot(fDown, chirpDown, color='red')
    axDown.vlines(nyquist/factor, -1.0, 1.0, linewidth=2,
        linestyle='--', color='green', label='New Nyquist')
    axDown.set_title('Downsample factor %d' % factor)
    axDown.set_xlabel('Frequency (Hz)')
    axDown.set_ylabel('Signal')
    axDown.autoscale(tight=True)

    chirpDeci = decimate(chirp, factor)
    axDeci = fig.add_subplot(2, 2, 3)
    axDeci.plot(f, chirp, color='lightgrey', linewidth=2)
    axDeci.plot(fDown, chirpDeci, color='red')
    axDeci.vlines(nyquist/factor, -1.0, 1.0, linewidth=2,
        linestyle='--', color='green', label='New Nyquist')
    axDeci.set_title('Decimation factor %d' % factor)
    axDeci.set_xlabel('Frequency (Hz)')
    axDeci.set_ylabel('Signal')
    axDeci.autoscale(tight=True)

    fInterp = np.linspace(0.0, nyquist, 2.0*sampRate*factor, endpoint=False)
    chirpInterp = interpolate(chirp, factor)

    axInterp = fig.add_subplot(2, 2, 2)
    axInterp.plot(f, chirp, color='lightgrey', linewidth=2)
    axInterp.plot(fInterp, chirpInterp, color='red')
    #axInterp.vlines(nyquist*factor, -1.0, 1.0, linewidth=2,
    #  linestyle='--', color='green', label='New Nyquist')
    axInterp.set_title('Interpolation factor %d' % factor)
    axInterp.set_xlabel('Frequency (Hz)')
    axInterp.set_ylabel('Signal')
    axInterp.autoscale(tight=True)

    fResamp = np.linspace(0.0, nyquist, 2.0*sampRate*(2.0/factor), endpoint=False)
    chirpResamp = resample(chirp, factorUp=2, factorDown=factor)

    axResamp = fig.add_subplot(2, 2, 4)
    axResamp.plot(f, chirp, color='lightgrey', linewidth=2)
    axResamp.plot(fResamp, chirpResamp, color='red')
    axResamp.vlines((2.0/factor)*nyquist, -1.0, 1.0, linewidth=2,
      linestyle='--', color='green', label='New Nyquist')
    axResamp.set_title('Resample factor 2/%d' % factor)
    axResamp.set_xlabel('Frequency (Hz)')
    axResamp.set_ylabel('Signal')
    axResamp.autoscale(tight=True)

    fig.tight_layout()

if __name__ == '__main__':
    demoInterpolate()
    demoResample()
    plt.show()
