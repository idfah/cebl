"""Module for processing unsegmented eeg.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs
import scipy.io as spio

from cebl import ml
from cebl import sig
from cebl import util

from base import EEGBase
import chanlocs
import head


class SegmentedEEG(EEGBase):
    def __init__(self, data, sampRate, chanNames=None, markers=None,
                 start=0.0, deviceName='', dtype=None, copy=False):
        """Construct a new SegmentedEEG instance for processing eeg data
        that has been split into segments of equal length.

        Args:
            data:       A 3D numpy array of floats of shape (nSeg,nObs[,nDim])
                        containing the eeg segments.  The first axis
                        corresponds to the eeg segments.  The second axis
                        corresponds to the observations (i.e., time steps).
                        The third axis is optional and corresponds to eeg
                        channels.

            sampRate:   The sampling rate (frequency) in samples-per-second
                        (Hertz) of the eeg data.  This defaults to 256Hz.

            chanNames:  A list of names of the channels in the eeg data.
                        If None (default) then the channel names are set
                        to '1', '2', ... 'nChan'.

            markers:    EEG event markers.  This is a list or tuple of floats
                        that mark each eeg segment.  There should be one marker
                        for each segment.  The interpretation of these marks is
                        up to the up to the user.  If None (default) then
                        markers are set to 1, 2, ..., nSeg.

            start:      Starting time in seconds of the segments.  Defaults
                        to 0.0.  This is useful if the data were segmented
                        using an offset from an event.  For example, if an
                        ERP is segmented starting at -0.2 seconds before
                        the stimulus onset.

            deviceName: The name of the device used to record the eeg data.

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
        # ensure we have numpy array with three axes
        # copy and cast if necessary
        self.data = util.segmat(data, dtype=dtype, copy=copy)
        self.dtype = self.data.dtype

        self.nSeg = data.shape[0]

        EEGBase.__init__(self, data.shape[1], data.shape[2],
            sampRate=sampRate, chanNames=chanNames, deviceName=deviceName)

        self.setMarkers(markers, copy=copy)
        self.setStart(start)

    def copy(self, dtype=None):
        return SegmentedEEG(self.data, sampRate=self.sampRate, chanNames=self.chanNames,
                            markers=self.markers, start=self.start,
                            deviceName=self.deviceName, dtype=dtype, copy=True)

    def getData(self):
        """Get the current data as a numpy array of shape (nSeg,nObs,nChan).
        """
        return self.data

    def getMarkers(self):
        """Get the current markers as a numpy array of floats.
        """
        return self.markers

    def setMarkers(self, markers, copy=False):
        if markers is None:
            self.markers = np.arange(self.nSeg)
        else:
            self.markers = np.array(markers, copy=copy)

        self.markers = self.markers.astype(self.dtype, copy=False)

        if len(self.markers) != self.nSeg:
            raise Exception('Length of markers ' + str(len(self.markers)) + \
                            ' does not match number of segments ' + str(self.nSeg))
        return self

    def setStart(self, start=None):
        if start is None:
            start = self.start
        self.start = int(np.floor(start*float(self.sampRate)))/float(self.sampRate)
        return self

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.start + self.nObs / float(self.sampRate)

    def getNSeg(self):
        return self.nSeg

    def append(self, newSeg):
        self.data = np.vstack((self.data, newSeg.data))
        self.nSeg = self.data.shape[0]
        self.markers = np.append(self.markers, newSeg.markers)
        return self

    def select(self, matchFunc, copy=False):
        indicators = np.asarray(map(matchFunc, self.markers), dtype=np.bool)
        return SegmentedEEG(data=self.data[indicators],
            sampRate=self.sampRate, markers=self.markers[indicators], start=self.getStart(),
            chanNames=self.chanNames, deviceName=self.deviceName, copy=copy)

    def selectChr(self, character, sign=0, *args, **kwargs):
        """ Note np.abs in documentation XXX - idfah
        """
        def matchFunc(mark):
            chrMatches = chr(int(np.abs(mark))) == character
            if sign == 0:
                return chrMatches
            elif sign == 1:
                return (mark > 0) & chrMatches
            elif sign == -1:
                return (mark < 0) & chrMatches

        return self.select(matchFunc, *args, **kwargs)

    def selectNear(self, value, *args, **kwargs):
        """ Note np.abs in documentation XXX - idfah
        """
        def matchFunc(mark):
            return np.isclose(np.abs(mark), value)

        return self.select(matchFunc, *args, **kwargs)

    def trim(self, start=None, end=None):
        startOrig = self.getStart()
        endOrig = self.getEnd()

        if start is not None:
            # adjust start to fit sample rate
            start = int(start*float(self.sampRate))/self.sampRate

            if start < startOrig:
                raise Exception('Cannot trim to start %f before original start %f.' %
                                (start, startOrig))

            startTrimSamp = int((start-startOrig)*self.sampRate)
        else:
            startTrimSamp = None
            start = startOrig

        if end is not None:
            # adjust end to fit sample rate
            end = int(end*float(self.sampRate))/self.sampRate

            if end > endOrig:
                raise Exception('Cannot trim to end %f before original end %f.' %
                                (end, endOrig))

            endTrimSamp = int((end-endOrig)*self.sampRate)
        else:
            endTrimSamp = None

        self.data = self.data[:,startTrimSamp:endTrimSamp,:]

        self.nObs = self.data.shape[1]
        self.nSec = self.nObs / float(self.sampRate)

        self.setStart(start)

        return self

    '''
    def split(self, nSec):
        nObs = self.sampRate*nSec
        rem = np.remainder(self.nObs, nObs)

        self.data = self.data[:,:(self.nObs-rem-1)]

        nSplit = self.data.shape[1] // nObs

        self.data = self.data.reshape((nSplit*self.nSeg, nObs, -1))

        self.nSeg = self.data.shape[0]
        self.nObs = self.data.shape[1]
        self.nSec = self.nObs / float(self.sampRate)

        return self
    '''

    def split(self, nSec, overlap=0.0):
        span = int(self.sampRate*nSec)
        overlap = int(overlap*span)

        windows = util.slidingWindow(self.data,
            span=span, stride=span-overlap, axis=1)

        # test this? XXX - idfah
        self.markers = np.repeat(self.markers, windows.shape[1])

        self.data = windows.reshape((-1,span,self.nChan))

        self.nSeg = self.data.shape[0]
        self.nObs = self.data.shape[1]
        self.nSec = self.nObs / float(self.sampRate)

        return self

    def avgERP(self):
        return np.mean(self.data, axis=0)

    def demean(self):
        self.data -= self.data.mean(axis=1).reshape(self.nSeg, 1, -1)
        return self

    def getStandardizer(self, **kwargs):
        return ml.SegStandardizer(self.data, **kwargs)

    def standardize(self, standardizer=None, **kwargs):
        if standardizer is None:
            standardizer = self.getStandardizer(**kwargs)

        self.data = standardizer.apply(self.data)

        return self

    def bandpass(self, lowFreq, highFreq, *args, **kwargs):
        bp = sig.BandpassFilter(lowFreq=lowFreq, highFreq=highFreq,
                sampRate=self.sampRate, *args, **kwargs)
        self.data = bp.filter(self.data, axis=1)
        return self

    def deleteChans(self, chans):
        chans = self.getChanIndices(chans)
        self.data = np.delete(self.data, chans, axis=2)
        self.nChan -= len(chans)
        self.chanNames = [c for i,c in enumerate(self.chanNames) if i not in chans]
        return self

    def keepChans(self, chans):
        chans = self.getChanIndices(chans)
        delChan = [c for c in xrange(self.nChan) if c not in chans]
        self.deleteChans(delChan)
        return self

    def reference(self, chans):
        chans = self.getChanIndices(chans)
        ref = self.data[:,:,chans]
        if len(chans) > 1:
            ref = ref.mean(axis=2)
        self.data -= util.segmat(ref)
        return self

    def bipolarReference(self, pairs):
        for pair in pairs:
            if len(pair) > 2:
                raise Exception('Bipolar reference assumes pairs of electrodes but got %s.' % pair)

            pair = self.getChanIndices(pair)

            ref = self.data[:,:,pair].mean(axis=2)
            self.data[:,:,pair] = util.segmat(ref)

        chanNames = []
        for pair in pairs:
            pair = self.getChanNames(pair)
            chanNames.append('-'.join(pair))

        self.deleteChans([r for l,r in pairs])
        self.setChanNames(chanNames)

        return self

    def commonAverageReference(self, *args, **kwargs):
        # faster to loop?  look for this elsewhere XXX - idfah
        #self.data[...] = np.array([sig.commonAverageReference(seg, *args, **kwargs)
        #                           for seg in self.data])

        for i in xrange(self.data.shape[0]):
            self.data[i,...] = sig.commonAverageReference(self.data[i,...], *args, **kwargs)

        return self

    def car(self, *args, **kwargs):
        return self.commonAverageReference(*args, **kwargs)

    def meanSeparate(self, recover=False):
        for i in xrange(self.data.shape[0]):
            self.data[i,...] = sig.meanSeparate(self.data[i,...], recover=recover)

        if recover:
            self.chanNames[-1] = 'recovered'
        else:
            self.chanNames[-1] = 'mean'

        return self

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

        self.data = np.array([sig.sharpen(seg, dist=dist, *args, **kwargs) for seg in self.data])
        return self

    def baselineCorrect(self, t=None):
        if t is None:
            if self.getStart() >= 0.0:
                raise Exception('Cannot baselineCorrect with positive start time ' +
                                'unless t is given explicitely')
            tSamp = int(np.abs(self.getStart()) * self.sampRate)
        else:
            tSamp = int(t * self.sampRate)

        self.data -= np.mean(self.data[:,:tSamp,:],axis=1).reshape((self.nSeg,1,self.nChan))    

        return self

    def downsample(self, factor):
        self.data = np.asarray([sig.downsample(seg, factor) for seg in self.data],
                                   dtype=self.dtype)

        self.sampRate /= float(factor)
        self.nObs = self.data.shape[1]
        self.nSec = self.nObs / float(self.sampRate)

        self.setStart()

        return self

    def resample(self, factorDown, factorUp=1, interpKwargs=dict(), **decimKwargs):
        self.data = np.asarray(
            [sig.resample(seg, factorDown=factorDown, factorUp=factorUp,
                          interpKwargs=interpKwargs, **decimKwargs)
             for seg in self.data], dtype=self.dtype)

        self.markers = sig.upsample(self.markers, factorUp)
        self.markers = sig.downsample(self.markers, factorDown)

        self.sampRate *= float(factorUp)
        self.sampRate /= float(factorDown)
        self.nObs = self.data.shape[1]
        self.nSec = self.nObs / float(self.sampRate)

        self.setStart()

        return self

    def chanEmbed(self):
        return self.data.reshape((self.data.shape[0], -1), order='F')

    def timeEmbed(self, *args, **kwargs):
        return util.timeEmbed(self.data, *args, axis=1, **kwargs)

    def psd(self, *args, **kwargs):
        return [sig.PSD(s, *args, sampRate=self.sampRate, **kwargs)
                for s in self.data]

    def power(self, *args, **kwargs):
        freqs = sig.PSD(self.data[0], *args, sampRate=self.sampRate, **kwargs).getFreqs()
        powers = np.array([sig.PSD(s, *args, sampRate=self.sampRate, **kwargs).getPowers()
                    for s in self.data], dtype=self.dtype)
        return freqs, powers

    def spectrogram(self, *args, **kwargs):
        return [sig.Spectrogram(s, *args, sampRate=self.sampRate, **kwargs)
                for s in self.data]

    def reference(self, chans):
        chans = self.getChanIndices(chans)
        ref = self.data[:,:,chans]
        if len(chans) > 1:
            ref = ref.mean(axis=2)
        self.data -= util.segmat(ref)
        return self

    def plotSegs(self, chan=0, drawZeroTime=False, drawZeroVolt=True,
                 timeUnit='ms',
                 segLineColor=(0.05,0.05,0.2,0.1), segLineWidth=3,
                 meanLineColor='red', meanLineWidth=2,
                 ax=None, *args, **kwargs):
        chan, = self.getChanIndices((chan,))

        segs = self.data[:,:,chan].T

        timeUnit = timeUnit.lower()
        if timeUnit in ('s', 'ms'):
            time = np.linspace(self.getStart(),self.getEnd(),
                        segs.shape[0]).astype(self.dtype, copy=False)
        elif timeUnit == 'obs':
            time = np.arange(self.nObs)
        else:
            raise Exception('Invalid timeUnit %s.' + str(timeUnit))

        if timeUnit == 'ms':
            time *= 1000.0

        if ax is None:
            #fig = plt.figure(figsize=(9,5.5))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        if timeUnit == 's':
            ax.set_xlabel('Time (s)')
        elif timeUnit == 'ms':
            ax.set_xlabel('Time (ms)')
        elif timeUnit == 'obs':
            ax.set_xlabel('Observation')
        ax.set_ylabel(r'Signal ($\mu V$)')

        segLines = ax.plot(time, segs, color=segLineColor, linewidth=segLineWidth, *args, **kwargs)
        segLines[-1].set_label('Single Trial')
        ax.plot(time, np.mean(segs, axis=1), color='white', linewidth=meanLineWidth*2, *args, **kwargs)
        meanLine = ax.plot(time, np.mean(segs, axis=1), color=meanLineColor, linewidth=meanLineWidth, label='Mean', *args, **kwargs)

        vertLine = None
        if drawZeroTime:
            vertLine = ax.vlines(0.0, np.min(segs), np.max(segs), color='red', linewidth=2, linestyle='--')

        if drawZeroVolt:
            ax.hlines(0.0, time[0], time[-1], linewidth=2, linestyle='--', color='grey')

        ax.autoscale(tight=True)

        return {'ax': ax, 'segLines': segLines, 'meanLine': meanLine, vertLine: 'vertLine'}

    def plotAvg(self, chans=None, drawZeroTime=False, drawZeroVolt=True, timeUnit='ms', scale=None, ax=None, **kwargs):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        chans = self.getChanIndices(chans)

        avg = self.data[:,:,chans].mean(axis=0)

        timeUnit = timeUnit.lower()
        if timeUnit in ('s', 'ms'):
            time = np.linspace(self.getStart(),self.getEnd(),
                        avg.shape[0]).astype(self.dtype, copy=False)
        elif timeUnit == 'obs':
            time = np.arange(self.nObs)
        else:
            raise Exception('Invalid timeUnit %s.' + str(timeUnit))

        if timeUnit == 'ms':
            time *= 1000.0

        sep = util.colsep(avg, scale=scale)

        if ax is None:
            #fig = plt.figure(figsize=(9,5.5))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        if timeUnit == 's':
            ax.set_xlabel('Time (s)')
        elif timeUnit == 'ms':
            ax.set_xlabel('Time (ms)')
        elif timeUnit == 'obs':
            ax.set_xlabel('Observation')

        if len(chans) > 1:
            ax.set_yticklabels([c for i,c in enumerate(self.chanNames) if i in chans])
            ax.set_yticks(sep)
        else:
            ax.set_ylabel(r'Signal ($\mu V$)')

        lines = ax.plot(time, avg+sep, **kwargs)

        ax.autoscale(tight=True)

        if drawZeroTime:
            ylim = ax.get_ylim()
            #ax.vlines(0.0, np.min(avg+sep), np.max(avg+sep), color='red', linewidth=2, linestyle='--')
            ax.vlines(0.0, ylim[0], ylim[1], color='red', linewidth=2, linestyle='--')

        if drawZeroVolt:
            ax.hlines(0.0, time[0], time[-1], linewidth=2, linestyle='--', color='grey')

        return {'ax': ax, 'lines': lines, 'scale': scale}

    def plotAvgDiffHead(self, other, times=(0.0, 0.1, 0.2, 0.3), chans=None):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        chans = self.getChanIndices(chans)
        chanNames = self.getChanNames(chans)

        nTimes = len(times)

        avgSelf = self.data[:,:,chans].mean(axis=0)
        avgOther = other.data[:,:,chans].mean(axis=0)
        avgDiff = avgSelf - avgOther

        if nTimes <= 4:
            nRow = 1
            nCol = nTimes
        else:
            nRow = (nTimes // 4)+1
            nCol = 4

        fig = plt.figure()
        axs = []

        for i,t in enumerate(times):
            ax = fig.add_subplot(nRow, nCol, i+1)
            ax.set_title('%.0fms' % (t*1000.0))
            axs.append(ax)

            timeStep = int(t*self.sampRate)

            head.plotHeadInterp(chanNames=chanNames, magnitudes=avgDiff[timeStep,:], ax=ax)

        return {'fig': fig, 'axs': axs}

    def plotAvgAndHead(self, chan=0, times=(100,200,300,400,500,600,700), timeUnit='ms',
                       mn=None, mx=None, avgKwargs={}, **kwargs):
        chan, = self.getChanIndices((chan,))


        timeUnit = timeUnit.lower()
        nHead = len(times)

        fig = plt.figure(figsize=(14,8))
        fig.subplots_adjust(hspace=0.32, wspace=0.02,
            left=0.065, right=0.95, top=0.97, bottom=0.18)

        gs = pltgs.GridSpec(2,nHead)
        axAvg = fig.add_subplot(gs[0,:])
        axHead = [fig.add_subplot(gs[1,i]) for i in xrange(nHead)]
        axCBar = fig.add_axes((0.05, 0.08, 0.9, 0.05))

        avgPlot = self.plotAvg(chans=(chan,), ax=axAvg, timeUnit=timeUnit, **avgKwargs)

        avgMn, avgMx = avgPlot['ax'].get_ylim()
        axAvg.vlines(times, avgMn, avgMx, linestyle='--', linewidth=2, color='red')
        axAvg.set_title('Channel %s Average' % self.getChanNames((chan,))[0])

        avg = np.mean(self.data, axis=0)

        headPlots = []
        for t,axH in zip(times, axHead):
            startObs = int(self.start*self.sampRate)
            if timeUnit == 's':
                i = int(self.sampRate*t)
                fmt = '%.2f'
            elif timeUnit == 'ms':
                i = int(self.sampRate*t/1000.0)
                fmt = '%.0f'
            elif timeUnit == 'obs':
                i = t
                fmt = '%.0f'
            else:
                raise Exception('Invalid timeUnit %s.' % str(timeUnit))
            i -= startObs

            if mn is None:
                mn = np.min(avg)

            if mx is None:
                mx = np.max(avg)

            hp = head.plotHeadInterp(chanNames=self.getChanNames(),
                    magnitudes=avg[i,:], mn=mn, mx=mx,
                    colorbar=False, ax=axH, **kwargs)
            axH.set_title((fmt + timeUnit) % t)
            headPlots.append(hp)

        cbar = plt.colorbar(hp['im'], ax=axAvg, orientation='horizontal', cax=axCBar)
        cbar.set_label(r'Signal ($\mu V$)')

        return {'axAvg': axAvg, 'axHead': axHead, 'axCBar': axCBar,
                'avgPlot': avgPlot, 'headPlots': headPlots, 'cbar': cbar}

    def plotAvgPSDByChan(self, scale='log', plotChanNames=True, lowFreq=0, highFreq=np.inf, ax=None, psdKwargs={}, **kwargs):
        psds = self.psd(**psdKwargs)

        powers = np.array([psd.getPowers() for psd in psds])
        freqs = psds[0].getFreqs()

        lowMask = freqs < 40.0
        highMask = freqs > 0.5
        freqMask = lowMask & highMask

        powers = powers[:,freqMask]
        powers = powers.mean(axis=0)

        freqs = freqs[freqMask]

        scale = scale.lower()
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.grid()
            ax.set_title('Power Spectral Density')
            ax.set_xlabel(r'Freqency ($Hz$)')
            ax.set_xlim((np.min(freqs), np.max(freqs)))
            if scale in ('linear', 'log'):
                ax.set_ylabel(r'Power Density ($\mu V^2 / Hz$)')
            elif scale in ('db', 'decibels'):
                ax.set_ylabel(r'Power Density (dB)')
            if scale == 'log':
                ax.set_yscale('log')

        if scale in ('linear', 'log'):
            pass
        elif scale in ('db', 'decibels'):
            powers = 10.0*np.log10(powers/np.max(powers))
        else:
            raise Exception('Invalid scale %s.' % str(scale))

        powersFlat = powers.reshape((-1,), order='F')
        lines = ax.plot(powersFlat, **kwargs)

        nFreq = len(freqs)
        mn = np.min(powersFlat)
        mx = np.max(powersFlat)
        chanNames = self.getChanNames()

        if plotChanNames:
            for i,cn in enumerate(chanNames):
                if i > 0:
                    ax.vlines(i*float(nFreq), mn, mx, linestyle='--')
                ax.text((i+0.25)*float(nFreq), mx-0.38*(mx-mn), cn, fontsize=14)

        tickStride = int(np.ceil(nFreq/3.0))
        tickFreqs = freqs[::tickStride]
        tickPlaces = np.arange(nFreq)[::tickStride]
        tickLocs = np.concatenate(
                        [tickPlaces+nFreq*i for i,c in enumerate(chanNames)])
        tickLabels = np.round(np.tile(tickFreqs, len(chanNames))).astype(np.int)

        ax.set_xticks(tickLocs)
        ax.set_xticklabels(tickLabels)

        ax.autoscale(tight=True)

        return {'freqs': freqs, 'powers': powers, 'lines': lines, 'ax': ax}

    def plotImg(self, chans):
        if chans is None:
            chans = self.getChanNames()
        chans = self.getChanIndices(chans)

        chans = self.getChanIndices(chans)

        #if ax is None:
        ##    fig = plt.figure(figsize=(14,8.5))
        #    fig = plt.figure(figsize=(9,5.5))
        #    ax = fig.add_subplot(1,1,1)
        #    ax.set_yticklabels([c for i,c in enumerate(self.chanNames) if i in chans])
        #    ax.set_yticks(sep)
        #    ax.set_xlabel('Time (s)')
        #    ax.set_ylim(-scale, sep[-1] + scale)


class SegmentedEEGFromEEG(SegmentedEEG):
    def __init__(self, unSegmentedEEG, start=0.0, end=0.8,
                 #startsFunc=lambda m: np.where(np.diff(np.abs(m)) > 0.0)[0],
                 startsFunc=lambda m: np.where(~np.isclose(np.diff(m), 0.0))[0],
                 *args, **kwargs):

        unSegmentedData = unSegmentedEEG.getData() 
        sampRate = unSegmentedEEG.getSampRate()
        chanNames = unSegmentedEEG.getChanNames()
        markers = unSegmentedEEG.getMarkers()

        startSamp = int(np.floor(start*float(sampRate)))
        start = startSamp/float(sampRate)

        endSamp = int(np.ceil(end*float(sampRate)))
        end = endSamp/float(sampRate)

        #segStarts = np.where(np.diff(np.abs(markers)) > 0.0)[0]
        segStarts = startsFunc(markers)

        #print np.diff(segStarts)

        # if first segment is too short, ditch it
        # this feels hacky ? XXX - idfah
        while segStarts[0] + startSamp < 0:
            ##print 'ditching first segment'
            segStarts = segStarts[1:]

        # if last segment is too short, ditch it
        while segStarts[-1] + endSamp >= unSegmentedData.shape[0]:
            ##print 'ditching last segment'
            segStarts = segStarts[:-1]

        indices = np.asarray([range(s+startSamp, s+endSamp) for s in segStarts],
                             dtype=np.int)

        data = unSegmentedData[indices]

        SegmentedEEG.__init__(self, data=data, sampRate=sampRate, chanNames=chanNames,
                              markers=markers[segStarts+1], start=start, *args, **kwargs)

class SegmentEEGFromSingleEEG(SegmentedEEG):
    def __init__(self, singleEEG, *args, **kwargs):
        markers = np.array((0,))
        data = singleEEG.getData()
        data = data.reshape((1, data.shape[0], data.shape[1]))

        sampRate = singleEEG.getSampRate()
        chanNames = singleEEG.getChanNames()
        
        SegmentedEEG.__init__(self, data=data, sampRate=sampRate, chanNames=chanNames,
                              markers=markers, *args, **kwargs)

class SegmentedEEGFromMatFiles(SegmentedEEG):
    def __init__(self, fileNames, dataKey='data', sampRate=('key','freq'),
                 chanNames=('key','channels'), markers=('arg',None), start=('arg',0.0),
                 transpose=False, deviceName=('arg',None), *args, **kwargs):

        firstMat = spio.loadmat(fileNames[0])
        firstSeg = util.colmat(firstMat[dataKey])
        if transpose:
            firstSeg = firstSeg.T
        firstShape = firstSeg.shape

        def keyOrArg(spec):
            koa = spec[0]
            val = spec[1]

            if koa == 'key':
                return firstMat[val]
            elif koa == 'arg':
                return val
            else:
                raise Exception('Invalid spec %s.' % spec)

        sampRate = int(keyOrArg(sampRate))
        chanNames = [str(chanName[0]) for chanName in keyOrArg(chanNames)[0][0]]
        markers = keyOrArg(markers)
        start = float(keyOrArg(start))
        deviceName = str(keyOrArg(deviceName))

        data = []

        for fileName in fileNames:
            mat = spio.loadmat(fileName)
            seg = util.colmat(mat[dataKey])

            if transpose:
                seg = seg.T

            if seg.shape != firstShape:
                raise Exception('Shape of first segment %s %s does not not match shape of segment %s %s.' %
                    (str(fileNames[0]), str(firstShape), str(fileName), str(seg.shape)))

            data.append(seg)

        data = np.asarray(data)

        SegmentedEEG.__init__(self, data=data, sampRate=sampRate, chanNames=chanNames,
                              markers=markers, start=start, *args, **kwargs)
