import numpy as np
import matplotlib.pyplot as plt

from cebl import util
from cebl.ml.autoreg import AutoRegression

from . import windows


class PSDBase(object):
    def __init__(self, freqs, powers, sampRate):
        self.freqs = np.asarray(freqs)
        self.powers = np.asarray(powers)
        self.sampRate = sampRate

        if self.powers.ndim < 2:
            self.nChan = 1
        else:
            self.nChan = self.powers.shape[1]

    def getNChan(self):
        return self.nChan

    def getSampRate(self):
        return self.sampRate

    def getFreqs(self):
        return self.freqs

    def getPowers(self):
        return self.powers

    def getFreqsPowers(self):
        return self.freqs, self.powers

    def plotPower(self, scale='log', ax=None, **kwargs):
        """Plot a PSD estimate on a log scale.

        Returns
        -------
        A tuple containing
        ax:         pyplot axis.
        lines:      lines on axis.
        """
        result = {}

        scale = scale.lower()
        if ax is None:
            fig = plt.figure()
            result['fig'] = fig
            ax = fig.add_subplot(1,1,1)
        result['ax'] = ax

        ax.grid()
        ax.set_title('Power Spectral Density')
        ax.set_xlabel(r'Freqency ($Hz$)')
        ax.set_xlim((np.min(self.freqs), np.max(self.freqs)))
        if scale in ('linear', 'log'):
            ax.set_ylabel(r'Power Density ($\mu V^2 / Hz$)')
        elif scale in ('db', 'decibels'):
            ax.set_ylabel(r'Power Density (dB)')
        if scale == 'log':
            ax.set_yscale('log')

        if scale in ('linear', 'log'):
            scaledPowers = self.powers
        elif scale in ('db', 'decibels'):
            scaledPowers = 10.0*np.log10(self.powers/np.max(self.powers))
        else:
            raise Exception('Invalid scale %s.' % str(scale))

        lines = ax.plot(self.freqs, scaledPowers, **kwargs)
        result['lines'] = lines

        return result


class WelchPSD(PSDBase):
    """PSD smoothed using Welch's method.
    """

    def __init__(self, s, sampRate=1.0, span=3.0, overlap=0.5, windowFunc=windows.hann, pad=False):
        """Construct a new PSD using Welch's method.

        Args:
            s:          Numpy array with shape (observations[,dimensions])
                        containing the signal data to use for generating the
                        PSD estimate.  Should be matrix-like.  Rows are
                        observations.  Columns are optional and, if present,
                        represent dimensions.

            sampRate:   Sampling frequency of s.

            span:       Span/width of the sub-windows to use in seconds.

            overlap:    Fraction of window overlap.

            windowFunc: Callback to window function, default is hann window.
                        The time-domain signal is multiplied by this window in
                        order to control leakage, edge effects, et cetra.
                        Takes a single argument that specifies the length of
                        the window in observations.  If None, a rectangular
                        window is used.

            pad:        If True, then each window is padded with zeros to be
                        a power of 2.  This improves performance of the FFT
                        but the powers "appear" to be less smooth.  If False
                        (default) then no padding is done.

        Refs:
            @article{heinzel2002spectrum,
              title={Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows},
              author={Heinzel, G. and R{\"u}diger, A. and Schilling, R. and Hannover, T.},
              journal={Max Plank Institute},
              year={2002}
            }
        """
        s = util.colmat(s)
        nObs, nChan = s.shape

        # number of observations per window
        wObs = int(span*sampRate)

        # check span parameter
        if wObs > nObs:
            raise Exception('Span of %.2f exceedes length of input %.2f.' %
                (span, nObs/float(sampRate)))
        if wObs < 7:
            raise Exception('Span of %.2f is too small.' % span)

        if pad:
            # find next largest power of two
            # padding to this improves FFT speed
            nPad = util.nextpow2(wObs)
        else:
            nPad = wObs

        # split into overlapping windows
        wins = util.slidingWindow(s, span=wObs, stride=wObs-int(overlap*wObs))

        if windowFunc is not None:
            # multiply by window function
            cWin = windowFunc(wObs).reshape(1, wObs, 1)
            wins = wins*cWin

            # PSD scaling denominator
            #scaleDenom = np.sum(cWin**2)*float(sampRate)
            scaleDenom = float(sampRate)*np.sum(np.abs(cWin))**2

        else:
            #scaleDenom = wObs*float(sampRate)
            scaleDenom = float(sampRate)*wObs**2

        # discrete fourier transform
        dft = np.fft.fft(wins, nPad, axis=1)

        # first half of dft
        dft = dft[:,:int(np.ceil(nPad/2.0)),:]

        # scale to power/Hz
        # numpy fft doesn't support complex64 so can't preserve float32 dtype XXX - idfah
        dftmag = np.abs(dft).astype(s.dtype, copy=False)
        powers = 2.0*(dftmag**2)/scaleDenom

        # average windows
        powers = np.mean(powers, axis=0)

        # find frequencies
        freqs = np.linspace(0, sampRate/2.0, powers.shape[0])

        # omit DC and nyquist components
        #freqs = freqs[1:-1]
        #powers = powers[1:-1]

        PSDBase.__init__(self, freqs, powers, sampRate)


class RawPSD(PSDBase):
    """PSD generated using the raw output of the FFT.
    Also known as a raw periodogram.
    """

    def __init__(self, s, sampRate=1.0, windowFunc=windows.hann, pad=False):
        """Construct a new PSD using an FFT.
        Args:
            s:          Numpy array with shape (observations[,dimensions])
                        containing the signal data to use for generating the
                        PSD estimate.  Should be matrix-like.  Rows are
                        observations.  Columns are optional and, if present,
                        represent dimensions.

            sampRate:   Sampling frequency of data.

            windowFunc: Callback to window function, default is a hann window.
                        The time-domain signal is multiplied by this window in
                        order to control leakage, edge effects, et cetra.
                        Takes a single argument that specifies the length of
                        the window in observations.  If None, a rectangular
                        window is used.

            pad:        If True, then each window is padded with zeros to be
                        a power of 2.  This improves performance of the FFT
                        but the powers "appear" to be less smooth.  If False
                        (default) then no padding is done.

        """
        s = util.colmat(s)
        nObs, nChan = s.shape

        if windowFunc is not None:
            # multiply by window function
            cWin = windowFunc(nObs).reshape(nObs, 1)
            s = s * cWin

            # PSD scaling denominator
            #scaleDenom = np.sum(cWin**2)*float(sampRate)
            scaleDenom = float(sampRate)*np.sum(np.abs(cWin))**2

        else:
            #scaleDenom = nObs*float(sampRate)
            scaleDenom = float(sampRate)*nObs**2

        if pad:
            # find next largest power of two
            # padding to this improves FFT speed
            nPad = util.nextpow2(nObs)
        else:
            nPad = nObs

        paddedS = np.zeros((nPad,nChan), dtype=s.dtype)
        paddedS[:s.shape[0],:] = s
        s = paddedS

        # find discrete fourier transform
        dft = np.fft.fft(s, nPad, axis=0)

        # scale to power/Hz
        dftmag = np.abs(dft).astype(s.dtype, copy=False)
        powers = 2.0*(dftmag**2)/scaleDenom

        # first half of fft
        powers = powers[:int(np.ceil(nPad/2.0))]

        # find frequencies
        freqs = np.linspace(0, sampRate/2.0, powers.shape[0])

        # omit DC and nyquist components
        #freqs = freqs[1:-1]
        #powers = powers[1:-1]

        PSDBase.__init__(self, freqs, powers, sampRate)


class AutoRegPSD(PSDBase):
    """PSD generated from the coefficients of univariate autoregressive models.
    """

    def __init__(self, s, sampRate=1.0, order=20, freqs=None, *args, **kwargs):
        """Construct a new PSD using univariate autoregressive models.

        Args:
            s:          Numpy array with shape (observations[,dimensions])
                        containing the signal data to use for generating the
                        PSD estimate.  Should be matrix-like.  Rows are
                        observations.  Columns are optional and, if present,
                        represent dimensions.

            sampRate:   Sampling frequency of data.

            order:      Order of the autoregressive models.

            freqs:      Freqencies at which to estimate the signal power.
                        If freqs is None (default) then estimate the power
                        at all integer frequencies between 1Hz and Nyquist.
                        If freqs is an integer then estimate the power at
                        freqs equally spaced frequencies above DC to
                        Nyquist.  If freqs is a list or numpy array, then
                        estimate power at the specified frequencies.
        """
        s = util.colmat(s)
        nObs, nChan = s.shape

        # AR model order
        order = order

        if freqs is None:
            freqs = np.arange(1.0, np.floor(sampRate/2.0))
        elif isinstance(freqs, (int,)):
            freqs = np.linspace(0.0, sampRate/2.0, freqs+1)[1:]
        else:
            freqs = np.asarray(freqs)

        freqs = freqs
        nFreq = len(freqs)

        # period of sampling frequency
        dt = 1.0/float(sampRate)

        # vector of model orders
        #orders = np.arange(1, order+1)[None,:]
        orders = np.arange(order, 0, -1)[None,:]

        weights = np.empty((order,nChan))
        iVar = np.empty(nChan)

        # for each channel
        for chanI, chanS in enumerate(s.T):
            # train an AR model
            arFit = AutoRegression(chanS[None,...], order=order, *args, **kwargs)

            # residual values of AR model
            resid = arFit.resid(chanS[None,...])[0]

            # model weights, ditch bias
            weights[:,chanI] = arFit.model.weights[:-1].ravel()

            # innovation variance
            iVar[chanI] = np.var(resid)

        # estimate spectrum, vectorized
        powersDenom = util.capZero(
            np.abs(1.0 - np.exp(-2.0j*np.pi*freqs[:,None]*orders*dt).dot(weights))**2)
        powers = (iVar[None,:] * dt) / powersDenom

        # scale to power density
        powers /= sampRate

        """
        # for each channel
        for chanI, chanS in enumerate(s.T):
            # train an AR model
            arFit = AutoRegression((chanS,), order=order, *args, **kwargs)

            # predicted and residual values of AR model
            pred, resid = arFit.eval(chanS, returnResid=True)

            # model weights, ditch bias
            weights = arFit.model.weights[:-1,None]

            # innovation variance
            iVar = np.var(resid)

            # estimate spectrum, in a loop
            #for i,f in enumerate(freqs):
            #    powers[i,chanI] = (iVar * dt) / (np.abs(1.0 - \
            #        np.sum(weights * np.exp(-2.0j*np.pi*f*orders) * dt)))**2)

            # estimate spectrum, vectorized
            powers[:,chanI] = ((iVar * dt) /
                (np.abs(1.0 - np.sum(weights *
                    np.exp(-2.0j*np.pi*freqs[:,None]*orders*dt), axis=0))**2))

            # estimate spectrum vectorized using sines and cosines instead of complex numbers
            #cs = np.sum(weights[:,None] * np.cos(2.0 * np.pi * freqsNorm*orders)), axis=0)
            #sn = np.sum(weights[:,None] * np.sin(2.0 * np.pi * freqsNorm*orders)), axis=0)
            #powers[:,chanI] = iVar / (sampRate * ((1.0 - cs)**2 + sn**2))
        """

        PSDBase.__init__(self, freqs, powers, sampRate)


def PowerSpectralDensity(s, method='welch', *args, **kwargs):
    method = method.lower()
    if method == 'welch':
        return WelchPSD(s, *args, **kwargs)
    elif method in ('raw', 'fft'):
        return RawPSD(s, *args, **kwargs)
    elif method in ('ar', 'autoreg'):
        return AutoRegPSD(s, *args, **kwargs)
    else:
        raise Exception('Unknown PSD estimation method: ' + str(method))

def PSD(*args, **kwargs):
    return PowerSpectralDensity(*args, **kwargs)


def demoPSD():
    sampRate = 500

    f1 = 60.0
    f2 = 160.0

    s = np.arange(0,10*np.pi,1.0/sampRate)

    noise1 = np.random.uniform(0,0.001, size=s.shape)
    noise2 = np.random.normal(loc=0.0, scale=0.2, size=s.shape)

    y = np.vstack((np.sin(2*f1*np.pi*s)+noise1,10.0*np.sin(2*f2*np.pi*s)+noise2)).T
    print 'True max power: ', np.mean(y**2, axis=0)

    scale = 'log'

    raw = RawPSD(y, sampRate)
    ax = raw.plotPower(scale=scale, label='raw')['ax']
    print 'Raw max power: ', np.max(raw.getPowers()*sampRate, axis=0)

    welch = WelchPSD(y, sampRate, span=4)
    welch.plotPower(scale=scale, ax=ax, label='welch')
    print 'Welch max power: ', np.max(welch.getPowers()*sampRate, axis=0)

    autoreg = AutoRegPSD(y, sampRate, order=10)
    autoreg.plotPower(scale=scale, ax=ax, label='autoreg')
    print 'AR max power: ', np.max(autoreg.getPowers()*sampRate, axis=0)

    ax.legend()

    ax.autoscale(tight=True)

if __name__ == '__main__':
    demoPSD()
    plt.show()
