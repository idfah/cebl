import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as pltLogNorm
import numpy as np

from cebl import util

# pylint: disable=no-name-in-module
from .cwt import CWT

from . import windows


class SpectrogramBase:
    def __init__(self, freqs, powers, phases, sampRate):
        self.freqs = np.asarray(freqs)
        self.powers = np.asarray(powers)
        self.phases = np.asarray(phases)
        self.sampRate = sampRate

        self.nObs = self.powers.shape[0]
        self.nFreq = self.powers.shape[1]

        if self.powers.ndim < 3:
            self.nChan = 1
        else:
            self.nChan = self.powers.shape[2]

        self.times = np.linspace(0.0,
            self.nObs/float(self.sampRate), self.nObs)

    def getNObs(self):
        return self.nObs

    def getNFreq(self):
        return self.nFreq

    def getNChan(self):
        return self.nChan

    def getSampRate(self):
        return self.sampRate

    def getFreqs(self):
        return self.freqs

    def getTimes(self):
        return self.times

    def getPowers(self):
        return self.powers

    def getPhases(self, scale='radians'):
        scale = scale.lower()

        if scale == 'radians':
            return self.phases
        elif scale == 'cycles':
            return self.phases / (2.0*np.pi)
        elif scale == 'degrees':
            return self.phases * 180.0 / np.pi
        else:
            raise RuntimeError('Invalid phase scale: %s.' % str(scale))

    def getFreqsPowers(self):
        return self.freqs, self.powers

    def getFreqsPowersPhases(self):
        return self.freqs, self.powers, self.phases

    def plotPower(self, scale='log', chanNames=None, colorbar=True, axs=None):
        if chanNames is None:
            chanNames = [str(i) for i in range(self.nChan)]

        if scale == 'linear':
            powers = self.powers
            norm = plt.Normalize(np.min(powers), np.max(powers))
            zlabel = (r'Power Density ($\mu V^2 / Hz$)')

        elif scale == 'log':
            powers = self.powers
            norm = pltLogNorm(np.min(powers), np.max(powers))
            zlabel = (r'Power Density ($\mu V^2 / Hz$)')

        elif scale == 'db':
            me = np.max((np.min(self.powers), np.finfo(self.powers.dtype).tiny))
            powers = 10.0*np.log10(self.powers/me)
            norm = plt.Normalize(np.min(powers), np.max(powers))
            zlabel = 'Power (db)'

        else:
            raise RuntimeError('Invalid scale %s.' % str(scale))

        nRows = int(np.sqrt(self.nChan))
        if nRows*nRows < self.nChan:
            nRows += 1
        nCols = nRows

        if axs is None:
            fig = plt.figure()#figsize=(18, 10))
            newAxs = True
            axs = []
        else:
            newAxs = False
        imgs = []

        for i, chanName in enumerate(chanNames):
            if newAxs:
                ax = fig.add_subplot(nRows, nCols, i+1)
                axs.append(ax)
            else:
                ax = axs[i]

            img = ax.imshow(powers[:,:,i].T, interpolation='bicubic', origin='lower',
                            cmap=plt.cm.get_cmap('jet'), aspect='auto', norm=norm,
                            extent=(self.times[0], self.times[-1],
                                    self.freqs[0], self.freqs[-1]))
            imgs.append(img)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')

            if len(chanNames) > 1:
                ax.set_title(chanName)
            else:
                ax.set_title('Spectrogram')

        result = {'axs': axs, 'imgs': imgs}

        if colorbar:
            if newAxs:
                fig.tight_layout()
                cbar = fig.colorbar(imgs[-1], norm=norm, ax=axs,
                                    pad=0.025, fraction=0.1)#, anchor=(0.0, 0.1))
            else:
                cbar = axs[-1].colorbar(imgs[-1], norm=norm)
            cbar.set_label(zlabel)
            result['cbar'] = cbar

        return result


class CWTSpectrogram(SpectrogramBase):
    def __init__(self, s, sampRate=1.0, **kwargs):
        s = util.colmat(s)

        transform = CWT(sampRate=sampRate, **kwargs)
        freqs, powers, phases = transform.apply(s)

        SpectrogramBase.__init__(self, freqs, powers, phases, sampRate)


class FFTSpectrogram(SpectrogramBase):
    def __init__(self, s, sampRate=1.0, span=0.1, overlap=0.5,
                 windowFunc=windows.hanning, pad=False):
        s = util.colmat(s)
        nObs, nChan = s.shape

        wObs = int(span*sampRate)

        # check span parameter
        if wObs > nObs:
            raise RuntimeError('Span of %.2f exceedes length of input %.2f.' %
                (span, nObs/float(sampRate)))
        if wObs < 7:
            raise RuntimeError('Span of %.2f is too small.' % span)

        if pad:
            # find next largest power of two
            # padding to this improves FFT speed
            nPad = util.nextpow2(wObs)
        else:
            nPad = wObs

        # split into overlapping window
        wins = util.slidingWindow(s, span=wObs, stride=wObs-int(overlap*wObs))

        if windowFunc is not None:
            # multiply by window function
            cWin = windowFunc(wObs).reshape(1, wObs, 1)
            wins = wins*cWin

            # scaling denominator
            scaleDenom = float(sampRate)*np.sum(np.abs(cWin))**2

        else:
            scaleDenom = float(sampRate)*wObs**2

        # discrete fourier transform
        dft = np.fft.fft(wins, nPad, axis=1)

        # first half of dft
        dft = dft[:,:int(np.ceil(nPad/2.0)),:]

        # scale to power/Hz
        powers = 2.0*(np.abs(dft)**2)/scaleDenom

        # phase angles
        phases = np.angle(dft)

        # find frequencies
        freqs = np.linspace(0, sampRate/2.0, powers.shape[1])

        # omit DC and nyquist components
        freqs = freqs[1:-1]
        powers = powers[:,1:-1]
        phases = phases[:,1:-1]

        SpectrogramBase.__init__(self, freqs, powers, phases, sampRate)


# wrapper around class constructors
# pylint: disable=invalid-name
def Spectrogram(s, *args, method='cwt', **kwargs):
    method = method.lower()
    if method == 'cwt':
        return CWTSpectrogram(s, *args, **kwargs)
    elif method in ('fft', 'stft', 'stfft'):
        return FFTSpectrogram(s, *args, **kwargs)
    else:
        raise RuntimeError('Unknown Spectrogram  estimation method: ' + str(method))


def demoCWT():
    sampRate = 256
    freqs = np.arange(0.25, 128, 0.25)
    #span = 7
    span = 30

    #transform = CWT(sampRate, freqs, span, dtype=np.float32)

    t = np.linspace(0.0, sampRate*10.0, sampRate*10.0)

    s1 = np.sin(t*2.0*np.pi*20.0/float(sampRate))
    s2 = np.sin(t*2.0*np.pi*60.0/float(sampRate)) + \
            np.random.normal(scale=0.02, size=t.size)
    s3 = np.cumsum(np.random.normal(scale=0.05, size=t.size))
    s = np.vstack((s1, s2, s3)).T

    cwtSpecgram = CWTSpectrogram(s1, sampRate)

    cwtSpecgram.plotPower()

    #s = s.astype(np.float32)
    #freqs, powers, phases = transform.apply(s)

    #transform.plotPower(s, chanNames=('20Hz Sinusoid', 'Noisy 60Hz Sinusoid', 'Random Walk'))

    #plt.tight_layout()

if __name__ == '__main__':
    demoCWT()
    plt.show()
