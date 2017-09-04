import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

from . import windows

from cebl import util


class BandpassFilterBase(object):
    """Base class for all bandpass filters.
    This class must be extended in order to construct an actual filter.
    """
    def __init__(self, lowFreq, highFreq, sampRate=1.0, dtype=None):
        """Construct a new BandpassFilterBase instance.

        Args:
            lowFreq:        Low corner frequency in hertz.

            highFreq:       High corner frequency in hertz.

            sampRate:       Sampling rate in hertz.

        Vars:
            self.nyquist:   Nyquist rate for the given sampling rate.

            self.low:       Low corner as a fraction of the nyquist rate.

            self.high:      High corner as a fraction of the nyquist rate.

            self.bandType:  String containing the "band type" of the filter.
                            One of: 'highpass', 'lowpass' or 'bandpass'.
        """
        self.dtype = dtype

        self.lowFreq   = lowFreq
        self.highFreq  = highFreq
        self.sampRate  = sampRate

        self.nyquist = sampRate * 0.5

        self.low = lowFreq / self.nyquist
        if self.low > 1.0:
            raise Exception('Invalid lowFreq: ' + str(lowFreq) + '. Above nyquist rate.')
        if self.low < 0.0:
            raise Exception('Invalid lowFreq: ' + str(lowFreq) + '. Not positive.')

        self.high = highFreq / self.nyquist
        if self.high != np.Inf:
            if self.high > 1.0:
                raise Exception('Invalid highFreq: ' + str(highFreq) + '. Above nyquist rate.')
            if self.high < 0.0:
                raise Exception('Invalid highFreq: ' + str(highFreq) + '. Not positive.')

        if np.isclose(self.low, 0.0) and self.high == np.inf:
            self.bandType = 'allpass'
        elif np.isclose(self.low, self.high):
            self.bandType = 'allstop'
        elif np.isclose(self.low, 0.0) and self.high != np.Inf:
            self.bandType = 'lowpass'
        elif self.low > 0.0 and self.high == np.Inf:
            self.bandType = 'highpass'
        elif self.low > 0.0 and self.high != np.Inf and self.low < self.high:
            self.bandType = 'bandpass'
        elif self.low > 0.0 and self.high != np.Inf and self.high < self.low:
            self.bandType = 'bandstop'
        else:
            raise Exception('Invalid filter corners: ' +
                    str(lowFreq) + ', ' + str(highFreq) + '.')

    def frequencyResponse(self, freqs=None):
        raise NotImplementedException('frequencyResponse not implemented.')

    def filter(self, s, axis=0):
        raise NotImplementedException('filter not implemented.')

    def plotFreqResponse(self, freqs=None, scale='linear', 
                         showCorners=True,
                         label='Frequency Response',
                         ax=None, **kwargs):
        """Plot the frequency response of the filter.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        freqs, responses = self.frequencyResponse(freqs=freqs)
        freqs = freqs * self.sampRate * 0.5 / np.pi
        responseMags = np.abs(responses)

        scale = scale.lower()
        if scale == 'linear':
            ax.set_ylabel('Gain')
        elif scale == 'log':
            ax.set_ylabel('Gain')
            ax.set_yscale('symlog')
        elif scale == 'db':
            responseMags = 10.0*np.log10(util.capZero(responseMags**2))
            ax.set_ylabel('Gain (dB)')
        else:
            raise Exception('Invalid scale: ' + str(scale) + '.')
        
        lines = ax.plot(freqs, responseMags,
                        label=label, **kwargs)

        result = {'ax': ax, 'lines': lines}

        if showCorners:
            if scale == 'db':
                halfPow = 10.0*np.log10(0.5)
                halfAmp = 10.0*np.log10(0.5**2)
                mn = np.min(responseMags)
                mx = np.max(responseMags)
            else:
                halfPow = np.sqrt(0.5)
                halfAmp = 0.5
                mn = np.min(responseMags)
                mn = np.min((mn, 0.0))
                mx = np.max(responseMags)
                mx = np.max((mx, 1.0))

            halfPowerLines = ax.hlines(halfPow, 0.0, 0.5*self.sampRate,
                color='red', linestyle='-.', label='Half Power')
            result['halfPowerLines'] = halfPowerLines

            halfAmpLines = ax.hlines(halfAmp, 0.0, 0.5*self.sampRate,
                color='orange', linestyle=':', label='Half Amplitude')
            result['halfAmpLines'] = halfAmpLines

            cornerLines = ax.vlines((self.lowFreq,self.highFreq),
                mn, mx, color='violet', linestyle='--', label='Corners')
            result['cornerLines'] = cornerLines

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylim((0.0, 1.0))

        return result

    def plotPhaseResponse(self, freqs=None, scale='radians',
                         showCorners=True,
                         label='Frequency Response',
                         ax=None, **kwargs):
        """Plot the frequency response of the filter.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        freqs, responses = self.frequencyResponse(freqs=freqs)
        freqs = freqs * self.sampRate * 0.5 / np.pi
        responseAngles = np.unwrap(np.angle(responses))

        scale = scale.lower()
        if scale == 'radians':
            ax.set_ylabel('Phase (Radians)')
        elif scale == 'cycles':
            ax.set_ylabel('Phase (Cycles)')
            responseAngles /= 2.0*np.pi
        elif scale == 'degrees':
            ax.set_ylabel('Phase (Degrees)')
            responseAngles = responseAngles*180.0 / np.pi
        else:
            raise Exception('Invalid scale: ' + str(scale) + '.')

        lines = ax.plot(freqs, responseAngles,
                        label=label, **kwargs)

        result = {'ax': ax, 'lines': lines}

        if showCorners:
            cornerLines = ax.vlines((self.lowFreq,self.highFreq),
                np.min(responseAngles), np.max(responseAngles),
                color='violet', linestyle='--', label='Corners')
            result['cornerLines'] = cornerLines

        ax.set_xlabel('Frequency (Hz)')

        return result


class BandpassFilterIIR(BandpassFilterBase):
    """Infinite Impulse Response (IIR) bandpass filter.
    """
    def __init__(self, lowFreq, highFreq, sampRate=1.0,
                 order=3, filtType='butter', zeroPhase=True,
                 dtype=None, **kwargs):
        """Construct a new IIR bandpass filter.
        """
        BandpassFilterBase.__init__(self, lowFreq, highFreq, sampRate, dtype=dtype)

        self.order     = order
        self.filtType  = filtType.lower()
        self.zeroPhase = zeroPhase

        if self.bandType not in ('allpass', 'allstop'):
            if self.bandType == 'lowpass':
                self.Wn = self.high
            elif self.bandType == 'highpass':
                self.Wn = self.low
            elif self.bandType == 'bandpass':
                self.Wn = (self.low, self.high)
            elif self.bandType == 'bandstop':
                self.Wn = (self.high, self.low)
            else:
                raise Exception('Invalid bandType: ' + str(self.bandType))

            self.numCoef, self.denomCoef = spsig.iirfilter(order, self.Wn,
                ftype=filtType, btype=self.bandType, **kwargs)

            self.numCoef = self.numCoef.astype(self.dtype, copy=False)
            self.denomCoef = self.denomCoef.astype(self.dtype, copy=False)

            self.initZi()

    def initZi(self):
        # lfilter_zi does not preserve dtype of arguments, bug that should  be reported XXX - idfah
        # if above was fixed, use don't need astype below
        self.zi = spsig.lfilter_zi(self.numCoef, self.denomCoef).astype(self.dtype, copy=False)

    def scaleZi(self, s, axis):
        ziShape = [1,] * s.ndim
        ziShape[axis] = self.zi.size
        zi = self.zi.reshape(ziShape)

        a = [slice(None),] * s.ndim
        a[axis] = 0
        s0 = s[a]
        s0Shape = list(s.shape)
        s0Shape[axis] = 1
        s0 = s0.reshape(s0Shape)

        return zi * s0

    def frequencyResponse(self, freqs=None):
        if self.bandType == 'allpass':
            return spsig.freqz(1, worN=freqs)
        if self.bandType == 'allstop':
            return spsig.freqz(0, worN=freqs)

        numCoef = self.numCoef
        denomCoef = self.denomCoef
        
        if self.zeroPhase:
            # http://www.mathworks.com/matlabcentral/newsreader/view_thread/245017
            numCoef = np.convolve(numCoef,numCoef[::-1])
            denomCoef = np.convolve(denomCoef,denomCoef[::-1])

        # freqz does not preserve dtype of arguments, report bug XXX - idfah
        return spsig.freqz(numCoef, denomCoef, worN=freqs)

    def filter(self, s, axis=0):
        """Filter new data.
        """
        if self.bandType == 'allpass':
            return s
        if self.bandType == 'allstop':
            return np.zeros_like(s)

        """ Should be very close to filtfilt, padding? XXX - idfah
        if self.zeroPhase:
            rev = [slice(None),]*s.ndim
            rev[axis] = slice(None,None,-1)

            #ziScaled = self.scaleZi(s[rev], axis)

            y, newZi = spsig.lfilter(self.numCoef, self.denomCoef, s[rev], axis=axis, zi=newZi)
            y = y[rev]
        """

        # if zeroPhase and signal is shorter than padlen (default in filtfilt function)
        if self.zeroPhase and \
            (3*max(len(self.numCoef), len(self.denomCoef))) < s.shape[axis]:

            # need astype below since filtfilt calls lfilter_zi, which does not preserve dtype XXX - idfah
            return spsig.filtfilt(self.numCoef, self.denomCoef,
                        s, axis=axis, padtype='even').astype(self.dtype, copy=False)

        else:
            ziScaled = self.scaleZi(s, axis)

            # even padding to help reduce edge effects
            nPad = 3*max(len(self.numCoef), len(self.denomCoef))
            sPad = np.apply_along_axis(np.pad, axis, s, pad_width=nPad, mode='reflect') # edge for constant padding
            slc = [slice(nPad,-nPad) if i == axis else slice(None) for i in range(s.ndim)]

            y, newZi = spsig.lfilter(self.numCoef, self.denomCoef, sPad, axis=axis, zi=ziScaled)

            return y[slc]

class BandpassFilterIIRStateful(BandpassFilterIIR):
    def __init__(self, *args, **kwargs):
        if ('zeroPhase' in kwargs) and (kwargs['zeroPhase'] == True):
            raise Exception('Stateful IIR filter cannot have linear phase.')

        BandpassFilterIIR.__init__(self, *args, zeroPhase=False, **kwargs)
        self.ziSaved = False

    def filter(self, s, axis=0):
        if self.bandType == 'allpass':
            return s
        if self.bandType == 'allstop':
            return np.zeros_like(s)

        if not self.ziSaved:
            # use padding to find initial zi? XXX - idfah

            self.zi = self.scaleZi(s, axis)
            self.ziSaved = True

        y, self.zi = spsig.lfilter(nc, dc, s, axis=axis, zi=self.zi)
        return y

def demoBandpassFilterIIR():
    order =  5
    sampRate = 256.0
    nyquist = sampRate / 2.0
    lowFreq = 1.5
    highFreq = 45.0
    zeroPhase = True

    butter = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='butter', zeroPhase=zeroPhase)
    #cheby1 = BandpassFilter(0.0, highFreq, sampRate, order, filtType='cheby1', rp=1.0, zeroPhase=zeroPhase)
    cheby2 = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='cheby2', rs=20.0, zeroPhase=zeroPhase)
    ellip  = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='ellip', rp=1.0, rs=20.0, zeroPhase=zeroPhase)
    bessel = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='bessel', zeroPhase=zeroPhase)

    fig = plt.figure(figsize=(18,10))
    fig.canvas.set_window_title('IIR Bandpass Filter Demo')
    axLn = fig.add_subplot(2,2, 1)
    axDb = fig.add_subplot(2,2, 2)

    for ax, scale in zip((axLn, axDb), ('linear', 'db')):
        butter.plotFreqResponse(showCorners=False, scale=scale, label='Butterworth', ax=ax, linewidth=2)
        #cheby1.plotFreqResponse(showCorners=False, scale=scale, label='Cbebyshev-I', ax=ax, linewidth=2)
        cheby2.plotFreqResponse(showCorners=False, scale=scale, label='Cbebyshev-II', ax=ax, linewidth=2)
        ellip.plotFreqResponse(showCorners=False, scale=scale, label='Elliptical', ax=ax, linewidth=2)
        bessel.plotFreqResponse(showCorners=True, scale=scale, label='Bessel', ax=ax, linewidth=2)
        #ax.grid()

    axLn.autoscale(tight=True)
    axLn.set_title('Amplitude Response')
    axLn.legend(loc='upper right')

    axDb.set_xlim((0.0, nyquist))
    axDb.set_ylim((-100.0, 0.0))
    axDb.set_title('Power Response')

    axPh = fig.add_subplot(2,2, 3)
    scale='radians'
    butter.plotPhaseResponse(showCorners=False, scale=scale, label='Butterworth', ax=axPh, linewidth=2)
    #cheby1.plotPhaseResponse(showCorners=False, scale=scale, label='Chebyshev-I', ax=axPh, linewidth=2)
    cheby2.plotPhaseResponse(showCorners=False, scale=scale, label='Chebyshev-II', ax=axPh, linewidth=2)
    ellip.plotPhaseResponse(showCorners=False, scale=scale, label='Elliptical', ax=axPh, linewidth=2)
    bessel.plotPhaseResponse(showCorners=True, scale=scale, label='Bessel', ax=axPh, linewidth=2)
    axPh.autoscale(tight=True)
    axPh.set_title('Phase Response')

    t = np.linspace(0.0, 2.0, 2.0*sampRate, endpoint=False)
    f = np.linspace(0.0, nyquist, 2.0*sampRate, endpoint=False)
    chirp = spsig.chirp(t, 0.0, 2.0, nyquist)

    chirpButter = butter.filter(chirp)
    #chirpCheby1 = cheby1.filter(chirp)
    chirpCheby2 = cheby2.filter(chirp)
    chirpEllip = ellip.filter(chirp)
    chirpBessel = bessel.filter(chirp)

    sep = -np.arange(0, 5)*2.0
    chirpAll = np.vstack((chirpButter, chirpCheby2, chirpEllip, chirpBessel, chirp)).T + sep

    axCh = fig.add_subplot(2,2, 4)
    axCh.plot(f, chirpAll)
    axCh.vlines(lowFreq, 1, -9, color='violet', linestyle='--')
    axCh.vlines(highFreq, 1, -9, color='violet', linestyle='--')
    axCh.set_yticks([])
    axCh.set_xlabel('Frequency (Hz)')
    axCh.set_ylabel('Chirp')
    axCh.autoscale(tight=True)
    axChTwiny = axCh.twiny()
    axChTwiny.hlines(sep, 0.0, t[-1], linestyle='--', color='black')
    axChTwiny.set_xlabel('Time (s)')

    fig.tight_layout()


class BandpassFilterFIR(BandpassFilterBase):
    """Finite Impulse Response (FIR) bandpass filter.
    """
    def __init__(self, lowFreq, highFreq, sampRate=1.0,
                 order=20, filtType='sinc-blackman', dtype=None):
        """Construct a new FIR bandpass filter.
        """
        BandpassFilterBase.__init__(self, lowFreq, highFreq, sampRate, dtype)

        if order % 2 != 0:
            raise Exception('Invalid order: ' + str(order) +
                ' Must be an even integer.')

        self.order = order
        self.radius = order//2
        self.taps = np.linspace(-self.radius,self.radius, self.order+1)

        self.filtType = filtType.lower()
        if self.filtType == 'lanczos':
            self.initImpulseResponse(windows.lanczos(self.order+1, radius=self.radius))
        elif self.filtType == 'sinc-blackman':
            self.initImpulseResponse(windows.blackman(self.order+1))
        elif self.filtType == 'sinc-hamming':
            self.initImpulseResponse(windows.hamming(self.order+1))
        elif self.filtType == 'sinc-hann':
            self.initImpulseResponse(windows.hann(self.order+1))
        else:
            raise Exception('Invalid filtType: ' + str(filtType))

    def initImpulseResponse(self, window):
        if self.bandType == 'allpass':
            self.impulseResponse = windows.kroneckerDelta(self.order+1)

        elif self.bandType == 'allstop':
            self.impulseResponse = np.zeros_like(window)

        elif self.bandType == 'lowpass':
            hightaps = self.high*self.taps
            self.impulseResponse = self.high*np.sinc(hightaps) * window

        elif self.bandType == 'highpass':
            lowtaps = self.low*self.taps
            self.impulseResponse = (-self.low*np.sinc(lowtaps) * window +
                windows.kroneckerDelta(self.order+1))

        elif self.bandType == 'bandpass':
            lowtaps = self.low*self.taps
            hightaps = self.high*self.taps
            self.impulseResponse = (self.high*np.sinc(hightaps) -
                self.low*np.sinc(lowtaps)) * window

        elif self.bandType == 'bandstop':
            lowtaps = self.low*self.taps
            hightaps = self.high*self.taps
            self.impulseResponse = ((self.high*np.sinc(hightaps) -
                self.low*np.sinc(lowtaps)) * window +
                windows.kroneckerDelta(self.order+1))

        else:
            raise Exception('Invalid bandType: ' + str(self.bandType))

        self.impulseResponse = self.impulseResponse.astype(self.dtype, copy=False)

    def frequencyResponse(self, freqs=None):
        return spsig.freqz(self.impulseResponse, worN=freqs)

    def filter(self, s, axis=0, mode='same'):
        return np.apply_along_axis(lambda v:
                    np.convolve(v, self.impulseResponse, mode='same'), axis=axis, arr=s)

def demoBandpassFilterFIR():
    order = 20
    sampRate = 256.0
    nyquist = sampRate / 2.0
    lowFreq = 1.5
    highFreq = 40.0

    sincBla = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='sinc-blackman')
    sincHan = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='sinc-hann')
    sincHam = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='sinc-hamming')
    lanczos = BandpassFilter(lowFreq, highFreq, sampRate, order, filtType='lanczos')

    fig = plt.figure(figsize=(18,10))
    fig.canvas.set_window_title('FIR Bandpass Filter Demo')
    axLn = fig.add_subplot(2,2, 1)
    axDb = fig.add_subplot(2,2, 2)

    for ax, scale in zip((axLn, axDb), ('linear', 'db')):
        sincBla.plotFreqResponse(showCorners=True, scale=scale, label='Sinc-Blackman', ax=ax, linewidth=2)
        sincHan.plotFreqResponse(showCorners=False, scale=scale, label='Sinc-Hann', ax=ax, linewidth=2)
        sincHam.plotFreqResponse(showCorners=False, scale=scale, label='Sinc-Hamming', ax=ax, linewidth=2)
        lanczos.plotFreqResponse(showCorners=False, scale=scale, label='Lanczos', ax=ax, linewidth=2)
        #ax.grid()

    axLn.autoscale(tight=True)
    axLn.set_title('Amplitude Response')
    axLn.legend(loc='upper right')

    axDb.set_xlim((0.0, nyquist))
    axDb.set_ylim((-100.0, 0.0))
    axDb.set_title('Power Response')

    axPh = fig.add_subplot(2,2, 3)
    scale='radians'
    sincBla.plotPhaseResponse(showCorners=False, scale=scale, label='Sinc-Blackman', ax=axPh, linewidth=2)
    sincHan.plotPhaseResponse(showCorners=False, scale=scale, label='Sinc-Hann', ax=axPh, linewidth=2)
    sincHam.plotPhaseResponse(showCorners=False, scale=scale, label='Sinc-Hamming', ax=axPh, linewidth=2)
    lanczos.plotPhaseResponse(showCorners=True, scale=scale, label='Lanczos', ax=axPh, linewidth=2)
    axPh.autoscale(tight=True)
    axPh.set_title('Phase Response')

    t = np.linspace(0.0, 2.0, 2.0*sampRate, endpoint=False)
    f = np.linspace(0.0, nyquist, 2.0*sampRate, endpoint=False)
    chirp = spsig.chirp(t, 0.0, 2.0, nyquist)

    chirpSincBla = sincBla.filter(chirp)
    chirpSincHan = sincHan.filter(chirp)
    chirpSincHam = sincHam.filter(chirp)
    chirpLanczos = lanczos.filter(chirp)

    sep = -np.arange(0, 5)*2.0
    chirpAll = np.vstack((chirpSincBla, chirpSincHan, chirpSincHam, chirpLanczos, chirp)).T + sep

    axCh = fig.add_subplot(2,2, 4)
    axCh.plot(f, chirpAll)
    axCh.vlines(lowFreq, 1, -9, color='violet', linestyle='--')
    axCh.vlines(highFreq, 1, -9, color='violet', linestyle='--')
    axCh.set_yticks([])
    axCh.set_xlabel('Frequency (Hz)')
    axCh.set_ylabel('Chirp')
    axCh.autoscale(tight=True)

    axChTwiny = axCh.twiny()
    axChTwiny.hlines(sep, 0.0, t[-1], linestyle='--', color='black')
    axChTwiny.set_xlabel('Time (s)')

    fig.tight_layout()


def BandpassFilter(lowFreq, highFreq, sampRate=1.0, order=None, filtType='butter', **kwargs):
    filtType = filtType.lower()
    if filtType in ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel'):
        if order is None: order = 3
        return BandpassFilterIIR(lowFreq=lowFreq, highFreq=highFreq,
                    sampRate=sampRate, order=order, filtType=filtType, **kwargs)
    elif filtType in ('lanczos', 'sinc-blackman', 'sinc-hamming', 'sinc-hann'):
        if order is None: order = 20
        return BandpassFilterFIR(lowFreq=lowFreq, highFreq=highFreq,
                    sampRate=sampRate, order=order, filtType=filtType, **kwargs)
    else:
        raise Exception('Invalid filter type: ' + str(filtType) + '.')


if __name__ == '__main__':
    demoBandpassFilterIIR()
    demoBandpassFilterFIR()
    plt.show()
