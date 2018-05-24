"""Module for processing unsegmented eeg.
"""

import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as pltLogNorm
import matplotlib.cm as pltcm
import numpy as np
import scipy.stats as spstats

from cebl import ml
from cebl import sig
from cebl import util

from .base import EEGBase
from . import chanlocs
from . import head
from . import readbdf
from . import seg


class EEG(EEGBase):
    """Class for processing unsegmented eeg.
    """
    def __init__(self, data, sampRate=256.0, chanNames=None,
                 markers=None, deviceName='', dtype=None, copy=False):
        """Construct a new EEG instance.

        Args:
            data:       A 2D numpy array of floats containing the eeg data.
                        The first dimension (rows) correspond to observations
                        (i.e., time steps) while the second dimension (cols)
                        corresponds to the different channels.

            sampRate:   The sampling rate (frequency) in samples-per-second
                        (Hertz) of the eeg data.  This defaults to 256Hz.

            chanNames:  A list of names of the channels in the eeg data.
                        If None (default) then the channel names are set
                        to '1', '2', ... 'nChan'.

            markers:    EEG event markers.  This is a list or tuple of floats
                        that mark events in the eeg data.  There should be one
                        marker for each time step.  The interpretation of these
                        marks is up to the up to the user.  If None (default)
                        then markers are set to zero.

            deviceName: The name of the device used to record the eeg data.
                        Defaults to the empty string.

            dtype:      The data type used to store the signal.  Must be
                        a floating point type, e.g., np.float32 or np.float64.
                        If None (default) the data type is determined from
                        the data argument.

            copy:       If False (default) then data will not be copied if
                        possible.  If True, then the data definitely be 
                        copied.  Warning:  If multiple EEG instances use
                        the same un-copied data array, then modifying one
                        EEG instance may lead to undefined behavior in
                        the other instances.
        """
        # ensure we have numpy array with two axes
        # copy and cast if necessary
        self.data = util.colmat(data, dtype=dtype, copy=copy)
        self.dtype = self.data.dtype

        # if given an int, assume its the channel in the data
        if isinstance(markers, (int,)):
            markerChan = markers
            markers = self.data[:,markerChan]
            self.data = np.delete(self.data, markerChan, axis=1)

        # initialize the eeg base class
        EEGBase.__init__(self, self.data.shape[0], self.data.shape[1],
            sampRate=sampRate, chanNames=chanNames, deviceName=deviceName)

        # initialze the event markers
        self.setMarkers(markers, copy=copy)

    def copy(self, dtype=None):
        return EEG(data=self.data, sampRate=self.sampRate, chanNames=self.chanNames,
                   markers=self.markers, deviceName=self.deviceName, dtype=dtype, copy=True)

    def getData(self):
        """Get the current data as a numpy array of shape (nSeg,nObs,nChan).
        """
        return self.data

    def getMarkers(self):
        """Get the current markers as a numpy array of floats.
        """
        return self.markers

    def setMarkers(self, markers, copy=False):
        """Set the eeg event markers.

        Args:
            markers:    EEG event markers.  This is a list or tuple of floats
                        that mark events in the eeg data.  The interpretation
                        of these marks is up to the up to the user.  If None
                        (default) then markers are set to zero.
        """
        if markers is None:
            self.markers = np.linspace(0.0, self.nSec, self.nObs)
        else:
            self.markers = np.array(markers, copy=copy)

        self.markers = self.markers.astype(self.dtype, copy=False)

        if len(self.markers) != self.nObs:
            raise Exception('Length of markers ' + str(len(self.markers)) + \
                            ' does not match number of observations ' + str(self.nObs))

        return self

    def bandpass(self, lowFreq, highFreq, **kwargs):
        bp = sig.BandpassFilter(lowFreq=lowFreq, highFreq=highFreq,
                sampRate=self.sampRate, dtype=self.dtype, **kwargs)
        self.data = bp.filter(self.data)
        return self

    def cap(self, level):
        self.data[self.data >  level] =  level
        self.data[self.data < -level] = -level
        return self

    def commonAverageReference(self, *args, **kwargs):
        self.data = sig.commonAverageReference(self.data, *args, **kwargs)
        return self

    def car(self, *args, **kwargs):
        return self.commonAverageReference(*args, **kwargs)

    def meanSeparate(self, recover=False):
        self.data = sig.meanSeparate(self.data, recover=recover)

        if recover:
            self.chanNames[-1] = 'recovered'
        else:
            self.chanNames[-1] = 'mean'

        return self

    def EOGRegress(self, vChan1, vChan2, hChan1, hChan2, model=None):
        vChan1, vChan2 = self.getChanIndices((vChan1, vChan2))
        hChan1, hChan2 = self.getChanIndices((hChan1, hChan2))

        # report which chan?  do this elsewhere? XXX - idfah
        if None in (vChan1, vChan2, hChan1, hChan2):
            raise Exception('Invalid channel.')

        veog = self.data[:,vChan1] - self.data[:,vChan2]
        heog = self.data[:,hChan1] - self.data[:,hChan2]
        eog = np.vstack((veog,heog)).T

        bp = sig.BandpassFilter(0.0, 20.0, order=2, sampRate=self.sampRate)
        eogFilt = bp.filter(eog);

        if model is None:
            model = ml.RidgeRegression(eogFilt, self.data)

        self.data -= model.eval(eogFilt)

        return self, model

    def EOGRegress2(self, vChan1, vChan2, hChan1, hChan2, model=None):
        vChan1, vChan2 = self.getChanIndices((vChan1, vChan2))
        hChan1, hChan2 = self.getChanIndices((hChan1, hChan2))

        # report which chan?  do this elsewhere? XXX - idfah
        if None in (vChan1, vChan2, hChan1, hChan2):
            raise Exception('Invalid channel.')

        eog = self.data[:,(vChan1,vChan2,hChan1,hChan2)]

        if model is None:
            model = ml.RidgeRegression(eog, self.data)

        self.data -= model.eval(eog)

        return self, model

    def EOGRegress3(self, chan, model=None):
        chan = self.getChanIndices((chan,))[0]

        if chan is None:
            raise Exception('Invalid channel.')

        eog = self.data[:,chan][:,None]

        if model is None:
            model = ml.RidgeRegression(eog, self.data)

        self.data -= model.eval(eog)

        return self, model

    def sharpen(self, coord='sphere', *args, **kwargs):
        locs = np.asarray([chanlocs.chanLocs3d[cn.lower()] for cn in self.getChanNames()],
                          dtype=self.dtype)

        coord = coord.lower()
        if coord == '2d':
            # steriographic projection
            x = locs[:,0]
            y = locs[:,1]
            z = locs[:,2]
            xy = np.vstack((x/(1.0+z), y/(1.0+z))).T

            dist = head.euclidDist(xy, xy)

        elif coord == '3d':
            dist = head.euclidDist(locs, locs)

        elif coord == 'sphere':
            dist = head.sphereDist(locs, locs)

        else:
            raise Exception('Invalid coord %s.', str(coord))

        self.data = sig.sharpen(self.data, dist=dist, *args, **kwargs)
        return self

    def decimate(self, factor, *args, **kwargs):
        self.data = sig.decimate(self.data, factor, *args, **kwargs)
        self.markers = sig.downsample(self.markers, factor)

        self.sampRate /= float(factor)
        self.nObs = self.data.shape[0]
        self.nSec = self.nObs / float(self.sampRate)

        return self

    def downsample(self, factor):
        self.data = sig.downsample(self.data, factor)
        self.markers = sig.downsample(self.markers, factor)

        self.sampRate /= float(factor)
        self.nObs = self.data.shape[0]
        self.nSec = self.nObs / float(self.sampRate)

        return self

    def interpolate(self, factor, *args, **kwargs):
        self.data = sig.interpolate(self.data, factor, *args, **kwargs)
        self.markers = sig.upsample(self.markers, factor)

        self.sampRate *= float(factor)
        self.nObs = self.data.shape[0]
        self.nSec = self.nObs / float(self.sampRate)

        return self

    def resample(self, factorDown, factorUp=1, interpKwargs=dict(), **decimKwargs):
        self.data = sig.resample(self.data, factorDown, factorUp, interpKwargs=interpKwargs, **decimKwargs)

        self.markers = sig.upsample(self.markers, factorUp)
        self.markers = sig.downsample(self.markers, factorDown)

        self.sampRate *= float(factorUp)
        self.sampRate /= float(factorDown)
        self.nObs = self.data.shape[0]
        self.nSec = self.nObs / float(self.sampRate)

        return self

    def deleteChans(self, chans):
        chans = self.getChanIndices(chans)
        self.data = np.delete(self.data, chans, axis=1)
        self.nChan -= len(chans)
        self.chanNames = [c for i,c in enumerate(self.chanNames) if i not in chans]
        return self

    def keepChans(self, chans):
        chans = self.getChanIndices(chans)
        delChan = [c for c in xrange(self.nChan) if c not in chans]
        self.deleteChans(delChan)
        return self

    def demean(self):
        self.data -= self.data.mean(axis=0)
        return self

    def getStandardizer(self, **kwargs):
        return ml.Standardizer(self.data)

    def standardize(self, standardizer=None, **kwargs):
        if standardizer is None:
            standardizer = self.getStandardizer(**kwargs)

        self.data = standardizer.apply(self.data)

        return self

    def detrend(self):
        pass

    def ica(self):
        pass

    def movingAverage(self, *args, **kwargs):
        self.data = sig.movingAverage(self.data, *args, **kwargs)
        return self

    def wiener(self, *args, **kwargs):
        self.data = sig.wiener(self.data, *args, **kwargs)
        return self

    def ma(self, *args, **kwargs):
        return self.movingAverage(*args, **kwargs)

    def icaFilter(self, comp, remove=False, lags=0, returnICA=False, **kwargs):
        ica = ml.ICA(self.data, lags=lags, **kwargs)
        if ica.reason == 'diverge':
            raise Exception('ICA training diverged.  Try a smaller learning rate.')

        self.data = ica.filter(self.data, comp=comp, remove=remove)

        if returnICA:
            return self, ica
        else:
            return self

    def icaTransform(self, comp=None, remove=False, lags=0, returnICA=False, **kwargs):
        ica = ml.ICA(self.data, lags=lags, **kwargs)
        if ica.reason == 'diverge':
            raise Exception('ICA training diverged.  Try a smaller learning rate.')

        newData = ica.transform(self.data, comp=comp, remove=remove)

        chanNames = [str(c) for c in range(newData.shape[1])]

        newEEG = EEG(newData, sampRate=self.sampRate,
                     chanNames=chanNames, markers=self.markers[lags:],
                     deviceName=self.deviceName,
                     dtype=self.dtype, copy=False)

        if returnICA:
            return newEEG, ica
        else:
            return newEEG

    def msfFilter(self, comp, remove=False, lags=0, returnMSF=False):
        msf = ml.MSF(self.data, lags=lags)

        self.data = msf.filter(self.data, comp=comp, remove=remove)

        if returnMSF:
            return self, msf
        else:
            return self

    def msfTransform(self, comp=None, remove=False, lags=0, returnMSF=False):
        msf = ml.MSF(self.data, lags=lags)

        newData = msf.transform(self.data, comp=comp, remove=remove)

        chanNames = [str(c) for c in range(newData.shape[1])]

        newEEG = EEG(newData, sampRate=self.sampRate,
                     chanNames=chanNames, markers=self.markers[lags:],
                     deviceName=self.deviceName,
                     dtype=self.dtype, copy=False)

        if returnMSF:
            return newEEG, msf
        else:
            return newEEG

    def pcaFilter(self, comp, remove=False, lags=0, returnPCA=False):
        pca = ml.PCA(self.data, lags=lags)

        self.data = pca.filter(self.data, comp=comp, remove=remove)

        if returnPCA:
            return self, pca
        else:
            return self

    def pcaTransform(self, comp=None, remove=False, lags=0, returnPCA=False):
        pca = ml.PCA(self.data, lags=lags)

        newData = pca.transform(self.data, comp=comp, remove=remove)

        chanNames = [str(c) for c in range(newData.shape[1])]

        newEEG = EEG(newData, sampRate=self.sampRate,
                     chanNames=chanNames, markers=self.markers[lags:],
                     deviceName=self.deviceName,
                     dtype=self.dtype, copy=False)

        if returnPCA:
            return newEEG, pca
        else:
            return newEEG

    def psd(self, *args, **kwargs):
        return sig.PSD(self.data, *args, sampRate=self.sampRate, **kwargs)

    def power(self, *args, **kwargs):
        return self.psd(*args, **kwargs).getFreqsPowers()

    def spectrogram(self, *args, **kwargs):
        return sig.Spectrogram(self.data, *args, sampRate=self.sampRate, **kwargs)

    def reference(self, chans):
        chans = self.getChanIndices(chans)

        ref = self.data[:,chans]
        if len(chans) > 1:
            ref = ref.mean(axis=1)

        self.data -= ref[:,None]

        return self

    def bipolarReference(self, pairs):
        for pair in pairs:
            if len(pair) > 2:
                raise Exception('Bipolar reference assumes pairs of electrodes but got %s.' % pair)

            pair = self.getChanIndices(pair)

            ref = self.data[:,pair].mean(axis=1)
            self.data[:,pair] = ref.reshape((-1,1))

        chanNames = []
        for pair in pairs:
            pair = self.getChanNames(pair)
            chanNames.append('-'.join(pair))

        self.deleteChans([r for l,r in pairs])
        self.setChanNames(chanNames)

        return self

    def arbitraryReference(self, chanRefs):
        # not arbitrary enough, think about this XXX - idfah
        for chans, refs in chanRefs:
            chans = self.getChanIndices(chans)
            refs = self.getChanIndices(refs)

            ref = self.data[:,refs]
            if len(refs) > 1:
                ref = ref.mean(axis=1)
            self.data[:,chans] -= ref.reshape((-1,1))

        return self

    def trim(self, start=None, end=None):
        if (start is not None) and (start != 0.0):
            # adjust start to fit sample rate
            start = int(start*float(self.sampRate))/self.sampRate

            if start < 0.0:
                raise Exception('start %f is less than zero.' % start)

            startTrimSamp = int(start*self.sampRate)
        else:
            startTrimSamp = None

        if end is not None:
            # adjust end to fit sample rate
            end = int(end*float(self.sampRate))/self.sampRate

            if end > self.nSec:
                raise Exception('end %f is greater than length of data %f.' % (end, self.nSec))

            endTrimSamp = int((end-self.nSec)*self.sampRate)
        else:
            endTrimSamp = None

        self.data = self.data[startTrimSamp:endTrimSamp]
        self.markers = self.markers[startTrimSamp:endTrimSamp]

        self.nObs = self.data.shape[0]
        self.nSec = self.nObs / float(self.sampRate)

        return self

    def segment(self, *args, **kwargs):
        return seg.SegmentedEEGFromEEG(self, *args, **kwargs)

    def segmentSingle(self, *args, **kwargs):
        return seg.SegmentEEGFromSingleEEG(self, *args, **kwargs)

    def split(self, nSec, overlap=0.0, **kwargs):
        span = int(self.sampRate*nSec)
        overlap = int(overlap*span)

        data = util.slidingWindow(self.data,
            span=span, stride=span-overlap, axis=0).reshape((-1,span,self.nChan))

        return seg.SegmentedEEG(self, data=data, sampRate=self.sampRate,
            chanNames=self.chanNames, deviceName=self.deviceName, **kwargs)

    def timeEmbed(self, *args, **kwargs):
        return util.timeEmbed(self.data, *args, **kwargs)

    def autoCorrelation(self, *args, **kwargs):
        return sig.autoCorrelation(self.data, *args, **kwargs)

    def plotAutoCorrelation(self, chans=None, lags=None, ax=None, **kwargs):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        s = self.data[:, chans].copy()

        if lags is None:
            lags = s.shape[0]
        else:
            lags = min(s.shape[0], lags)

        ac = sig.autoCorrelation(s)[:lags]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        ax.grid()
        ax.set_xlabel(r'Lag')
        ax.set_ylabel(r'Correlation')

        lines = ax.plot(np.arange(ac.shape[0]), ac, **kwargs)

        ax.autoscale(tight=True)

        return {'ax': ax, 'lines': lines}

    def plotCWT(self, chans=None, start=None, end=None,
                method='cwt', colorbar=True, *args, **kwargs):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)
        chanNames = self.getChanNames(chans)

        # need to sync this with sig.cwt.plotCWT XXX - idfah
        nChan = len(chans)

        if start is None:
            start = 0.0
            startSamp = None
        else:
            startSamp = int(start*self.sampRate)

        if end is None:
            end = self.nSec*self.sampRate
            endSamp = None
        else:
            endSamp = int(end*self.sampRate)

        s = self.data[startSamp:endSamp,chans]

        time = np.linspace(0, end-start, s.shape[0]).astype(self.dtype, copy=False)

        transform = sig.CWT(sampRate=self.sampRate, *args, **kwargs)
        freqs = transform.freqs
        powers, phases = transform.apply(s)

        nChanSqrt = np.sqrt(nChan)
        plotRows = int(np.ceil(nChanSqrt))
        plotCols = int(nChanSqrt)

        fig = plt.figure()
        axs = []
        ims = []

        for i,cn in enumerate(chanNames):
            ax = fig.add_subplot(plotRows, plotCols, i+1)
            axs += [ax]

            im = ax.imshow(powers[:,:,i].T, interpolation='lanczos',
                            origin='lower', cmap=pltcm.get_cmap('jet'),
                            norm=pltLogNorm(), aspect='auto',
                            extent=(0.0, d.shape[0]/float(self.sampRate),
                            np.min(freqs), np.max(freqs)))
            ims += [im]

            #if i == plotRows:
            #    plt.colorbar(ax)

            #plt.colorbar()
            ax.autoscale(tight=True)
            ax.set_title('Channel: %s' % cn)
            ax.set_xlabel("Seconds")
            ax.set_ylabel("Frequencies (Hz)")
            #locs = range(nObs)
            #if len(locs) > 20:
            #    locs = locs[0:len(locs):len(locs)/20]
            #labels = ['{:6.0g}'.format(v/float(self.sampRate)) for v in locs]
            #print labels
            #plt.xticks(locs, labels)
            #locs = range(self.nFreq)
            #if len(locs) > 20:
            #    locs = locs[0:len(locs):len(locs)/20]
            #labels = ['{:6.2g}'.format(self.freqs[v]) for v in locs]
            #plt.yticks(locs, labels)
            #ax.set_title(chanNames[i])

            if colorbar:
                cbar = plt.colorbar(im)
                cbar.set_label('Power')

        return {'axs': axs, 'ims': ims}

    def plotPairs(self, chans=None, start=None, end=None, bins=30):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        startSamp = None
        if start is not None:
            startSamp = int(start*self.sampRate)

        endSamp = None
        if end is not None:
            endSamp = int(end*self.sampRate)

        s = util.colmat(self.data[startSamp:endSamp,chans])

        nObs = s.shape[0]
        nDim = s.shape[1]

        mx = np.max(np.abs(s))

        fig = plt.figure()
        axs = []

        for r in xrange(nDim):
            for c in xrange(nDim):
                ax = fig.add_subplot(nDim, nDim, r*nDim+c+1+nDim*(nDim-(r%nDim)*2-1))
                axs.append(ax)

                sx = s[:,r]
                sy = s[:,c]

                if (r == c):
                    plt.hist(sx, bins=bins, normed=False)
                    ax.set_xlim(-mx/2.0,mx/2.0)
                else:
                    ax.scatter(sx, sy, alpha=0.5, s=10, marker='.')
                    ax.plot((-mx,mx),(-mx,mx), color='grey', linestyle='dashed')
                    pearsonr, pearsonp = spstats.pearsonr(sx,sy)
                    pearsons = ".%2d" % np.round(pearsonr*100)
                    ax.text(0.9,0.1,pearsons,
                        transform=ax.transAxes,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        fontsize=8)
                    ax.set_ylim(-mx,mx)
                    ax.set_xlim(-mx,mx)

                if r == 0:
                #if r == ndim-1:
                    ax.set_xlabel(self.chanNames[c])
                    ax.set_xticks([])
                else:
                    #ax.set_xlabel('')
                    ax.get_xaxis().set_visible(False)

                if c == 0:
                    ax.set_ylabel(self.chanNames[r])
                    ax.set_yticks([])
                else:
                    ax.get_yaxis().set_visible(False)

        return {'fig': fig, 'axs': axs}

    def plotPSD(self, chans=None, ax=None, psdKwargs={}, **kwargs):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        psd = sig.PSD(self.data[:,chans], sampRate=self.sampRate, **psdKwargs)
        return psd.plotPower(ax=ax, **kwargs)

    def plotTrace(self, start=None, end=None, chans=None, drawZero=False, scale=None, ax=None, **kwargs):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        if start is None:
            start = 0.0
            startSamp = None
        else:
            startSamp = int(start*self.sampRate)

        if end is None:
            end = self.nSec
            endSamp = None
        else:
            endSamp = int(end*self.sampRate)

        s = self.data[startSamp:endSamp, chans].copy()
        ##s -= s.mean(axis=0)
        #time = np.linspace(0,end-start,s.shape[0]).astype(self.dtype, copy=False)
        time = np.linspace(start,end,s.shape[0]).astype(self.dtype, copy=False)

        sep, scale = util.colsep(s, scale=scale, returnScale=True)

        if ax is None:
            #fig = plt.figure(figsize=(14,8.5))
            #fig = plt.figure(figsize=(9,5.5))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        ax.set_xlabel(r'Time ($s$)')
        ax.set_ylabel(r'Signal ($\mu V$)')
        if len(chans) > 1:
            ax.set_yticklabels([c for i,c in enumerate(self.chanNames) if i in chans])
            ax.set_yticks(sep)
            ##ax.set_ylim(-scale, sep[-1] + scale)

        if len(chans) > 1:
            s += sep

        lines = ax.plot(time, s, **kwargs)

        if drawZero:
            ax.hlines(sep, time[0], time[-1], linewidth=2, linestyle='--', color='grey')

        ax.autoscale(tight=True)

        return {'ax': ax, 'lines': lines, 'scale': scale, 'sep': sep}

    def plotLags(self, chans=None, lags=(1,2,4,8,16,32,64,128,256), **kwargs):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        s = self.data[:, chans].copy()

        fig = plt.figure(figsize=(10,8))
        #fig.subplots_adjust(hspace=0.15, wspace=0.25,
        #        left=0.05, right=0.92, top=0.97, bottom=0.06)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.35

        nCols = np.ceil(np.sqrt(len(lags)))
        nRows = np.ceil(len(lags)/float(nCols))

        mn = np.min(s)
        mx = np.max(s)

        for i,lag in enumerate(lags):
            ax = fig.add_subplot(nRows, nCols, i+1)

            lines = []
            for j in xrange(s.shape[1]):
                lines += ax.plot(s[lag:,j], s[:-lag,j], **kwargs)
                                 #color=plt.cm.jet(j/float(s.shape[1]), alpha=0.2))

            ax.set_xlabel(r'$x_t$')
            ax.set_ylabel(r'$x_{t-%d}$' % lag)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim((mn, mx))
            ax.set_ylim((mn, mx))

            #if i == nCols-1:
            #    leg = ax.legend(lines, self.getChanNames(chans), labelspacing=0.34, prop={'size': 12},
            #                    bbox_to_anchor=(1.35, 0.8))
            if i == 0:
                leg = ax.legend(lines, self.getChanNames(chans), loc='upper left', prop={'size': 12})

        for l in leg.legendHandles:
            l.set_alpha(1.0)
            l.set_linewidth(2)

        return {'ax': ax, 'lines': lines, 'legend': leg}

    def saveFile(self, fileName):
        fileNameLower = fileName.lower()
        dotSplit = fileNameLower.rsplit('.')

        if dotSplit[-1] in util.compressedExtensions:
            fileFormat = dotSplit[-2]
        else:
            fileFormat = dotSplit[-1]

        if fileFormat == 'pkl':
            data = np.hstack((self.data, self.markers[:,None]))
            with util.openCompressedFile(fileName, 'w') as fileHandle:
                pickle.dump(data, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception('Unknown file format ' + str(fileFormat))

class EEGFromPickledMatrix(EEG):
    def __init__(self, fileName, sampRate, chanNames=None, markers=-1, transpose=False, *args, **kwargs):
        with util.openCompressedFile(fileName, 'r') as fileHandle:
            data = np.asarray(pickle.load(fileHandle))

        if transpose:
            data = data.T

        EEG.__init__(self, data=data,
                     sampRate=sampRate, chanNames=chanNames,
                     markers=markers, *args, **kwargs)

class EEGFromJSON(EEG):
    def __init__(self, fileName, protocol, trial=1, *args, **kwargs):

        # should be able to give keys as argument XXX - idfah

        fileHandle = util.openCompressedFile(fileName, 'r')
        jData = json.load(fileHandle)
        fileHandle.close()

        if isinstance(protocol, (int,)):
            jData = jData[protocol]
        else:
            found = False

            for d in jData:
                if d['protocol'] == protocol:
                    jData = d
                    found = True
                    break

            if not found:
                raise Exception('Invalid protocol: %s.' % str(protocol))

        sampRate = jData['sample rate']
        chanNames = [str(cn) for cn in jData['channels']]
        deviceName = jData['device']

        self.notes = jData['notes']
        self.date = jData['date']
        self.location = jData['location']
        self.impairment = jData['impairment']
        self.subjectNumber = jData['subject']

        trialName = 'trial %d' % trial
        data = np.asarray(jData['eeg'][trialName]).T
        markers = len(chanNames)

        EEG.__init__(self, data=data, sampRate=sampRate, chanNames=chanNames,
                     markers=markers, deviceName=deviceName, *args, **kwargs)

class EEGFromBDF(EEG):
    def __init__(self, fileName, *args, **kwargs):
        info = readbdf.readBDF(fileName)

        self.date = info['startDate']
        self.time = info['startTime']

        EEG.__init__(self, data=info['data'], sampRate=info['sampRate'],
                     chanNames=info['chanNames'], deviceName='biosemi',
                     markers=info['markers'], *args, **kwargs)
