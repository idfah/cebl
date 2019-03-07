import numpy as np
import scipy.signal as spsig

from cebl import util


class ContinuousWaveletTransform(object):
    def __init__(self, sampRate=256.0, freqs=None, span=5, dtype=None):
        """Continuous Wavelet Transform

        Refs:
            Partly adopted from the CWT matlab code found in FieldTrip
            http://fieldtrip.fcdonders.nl/

            @article{tallon1997oscillatory,
              title={Oscillatory $\gamma$-band (30--70 Hz) activity induced
                     by a visual search task in humans},
              author={Tallon-Baudry, Catherine and Bertrand, Olivier and Delpuech,
                      Claude and Pernier, Jacques},
              journal={Journal of Neuroscience},
              volume={17},
              number={2},
              pages={722--734},
              year={1997},
              publisher={Soc Neuroscience}
            }

            @book{addison2017illustrated,
              title={The illustrated wavelet transform handbook: introductory theory
                     and applications in science, engineering, medicine and finance},
              author={Addison, Paul S},
              year={2017},
              publisher={CRC press}
            }
        """
        self.dtype = dtype

        self.sampRate = sampRate

        if freqs is None:
            self.freqs = np.arange(1.0, np.floor(sampRate/2.0)).astype(self.dtype, copy=False)
        elif isinstance(freqs, (int,long)):
            self.freqs = np.linspace(0.0, sampRate/2.0, freqs+1).astype(self.dtype, copy=False)[1:]
        else:
            self.freqs = np.asarray(freqs)
            self.freqs = self.freqs[np.where(self.freqs > 0.0)] # DC not allowed
        self.freqs = self.freqs.astype(self.dtype, copy=False)

        self.nFreq = len(self.freqs)

        self.span = span

        scaledFreqs = self.freqs / float(self.span)
        timeScales = 1.0 / (2.0 * np.pi * scaledFreqs)

        dt = 1.0 / sampRate
        times = [np.arange(-3.5*ts, 3.5*ts+dt, dt, dtype=self.dtype) for ts in timeScales]
        #times = [np.linspace(-3.5*ts, 3.5*ts, sampRate, dtype=self.dtype) for ts in timeScales]

        self.wavelets = [self.morlet(f,t,ts) for f,t,ts in
                            zip(self.freqs, times, timeScales)]

    def morlet(self, freq, time, timeScale):
        """Morlet wavelet for given frequency and time vectors.
        The wavelet will be normalized so the total energy is 1.
        """

        dialation = 1.0 / np.sqrt(timeScale * np.sqrt(2.0*np.pi))
        return (dialation * np.exp(-time**2.0/(2.0*timeScale**2.0)) *
            np.exp(2.0j * np.pi * freq * time))

    ## #pure python.
    ## def apply(self, s):
    ##     ##cdef long nObs, nChan, i, padDiff, padFront, padBack
    ##     s = util.colmat(s)

    ##     # number of observations and channels
    ##     nObs, nChan = s.shape

    ##     # empty arrays to hold power and phase information
    ##     powers = np.zeros((nObs, self.nFreq, nChan), dtype=s.dtype)
    ##     phases = np.zeros((nObs, self.nFreq, nChan), dtype=s.dtype)

    ##     for i,wlet in enumerate(self.wavelets):
    ##         conv = np.apply_along_axis(lambda d:
    ##                     np.convolve(d, wlet, mode='full'),
    ##                     axis=0, arr=s)

    ##         padDiff = (conv.shape[0] - s.shape[0])
    ##         padFront = padDiff // 2
    ##         padBack = padDiff - padFront
    ##         conv = conv[padFront:-padBack,:]

    ##         ##conv = np.apply_along_axis(lambda d:
    ##         ##    spsig.fftconvolve(d, wlet, mode='same'),
    ##         ##    axis=0, arr=s)

    ##         powers[:,i,:] = 2.0*np.abs(conv)**2 / \
    ##                             np.sum(np.abs(wlet))**2
    ##         phases[:,i,:] = np.angle(conv)

    ##     powers /= self.sampRate

    ##     return self.freqs, powers, phases

    def apply(self, s):
        # XXX: need to measure if this is really any faster in cython - idfah
        cdef long nObs, nChan, nFreq, i, j, padDiff, padFront, padBack
        
        s = util.colmat(s)

        # number of observations, channels and frequencies
        nObs, nChan = s.shape
        nFreq = self.nFreq

        # empty arrays to hold power and phase information
        powers = np.zeros((nObs, nFreq, nChan), dtype=s.dtype)
        phases = np.zeros((nObs, nFreq, nChan), dtype=s.dtype)

        for i in range(nChan):
            for j in range(nFreq):
                conv = np.convolve(s[:,i], self.wavelets[j], mode='full')

                padDiff = (conv.shape[0] - s.shape[0])
                padFront = padDiff // 2
                padBack = padDiff - padFront
                conv = conv[padFront:-padBack]

                powers[:,j,i] = 2.0*np.abs(conv)**2 / \
                                    np.sum(np.abs(self.wavelets[j]))**2
                phases[:,j,i] = np.angle(conv)

        powers /= self.sampRate

        return self.freqs, powers, phases

class CWT(ContinuousWaveletTransform):
    pass
