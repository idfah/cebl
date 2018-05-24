"""Source to generate random data.
"""
import multiprocessing as mp
import numpy as np
import time
import wx

from cebl import util
from cebl.rt import widgets

from .source import Source, SourceConfigPanel


chans8 = ('F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2')

chans19 = ('FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
           'F7','F8','T3','T4','P7','P8','CZ','FZ','PZ')

chans32 = ('FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7',
           'P3','PZ','PO3','O1','OZ','O2','PO4','P4','P8','CP6','CP2',
           'C4','T8','FC6','FC2','F4','F8','AF4','FP2','FZ','CZ')

chans40 = ('FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7',
           'P3','PZ','PO3','O1','OZ','O2','PO4','P4','P8','CP6','CP2',
           'C4','T8','FC6','FC2','F4','F8','AF4','FP2','FZ','CZ',
           'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8')

chans64 = ('Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',  'FT7',
           'FC5', 'FC3', 'FC1', 'C1',  'C3',  'C5',  'T7',  'TP7',
           'CP5', 'CP3', 'CP1', 'P1',  'P3',  'P5',  'P7',  'P9',
           'PO7', 'PO3', 'O1',  'Iz',  'Oz',  'POz', 'Pz',  'CPz',
           'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',  'F2',  'F4',
           'F6',  'F8',  'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
           'C2',  'C4',  'C6',  'T8',  'TP8', 'CP6', 'CP4', 'CP2',
           'P2',  'P4',  'P6',  'P8',  'P10', 'PO8', 'PO4', 'O2')

chans72 = ('Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',  'FT7',
           'FC5', 'FC3', 'FC1', 'C1',  'C3',  'C5',  'T7',  'TP7',
           'CP5', 'CP3', 'CP1', 'P1',  'P3',  'P5',  'P7',  'P9',
           'PO7', 'PO3', 'O1',  'Iz',  'Oz',  'POz', 'Pz',  'CPz',
           'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',  'F2',  'F4',
           'F6',  'F8',  'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
           'C2',  'C4',  'C6',  'T8',  'TP8', 'CP6', 'CP4', 'CP2',
           'P2',  'P4',  'P6',  'P8',  'P10', 'PO8', 'PO4', 'O2',
           'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8')

def rnorm(scale, size):
    return np.random.normal(scale=scale, size=size)

def runif(scale, size):
    return np.random.uniform(low=-scale, high=scale, size=size)

def rtriang(scale, size):
    return np.random.triangular(left=-scale, mode=0, right=scale, size=size)

distributions = {
    'normal': rnorm,
    'uniform': runif,
    'triangular': rtriang
}


class RandomConfigPanel(SourceConfigPanel):
    """wx panel containing controls for configuring random signal sources.
    """
    def __init__(self, parent, src, *args, **kwargs):
        """Construct a new configuration panel for a random signal source.

        Args:
            parent:         wx parent of this panel.

            src:            Source to configure.

            args, kwargs:   Additional arguments passed to the SourcePanel
                            base class.
        """
        SourceConfigPanel.__init__(self, parent=parent, src=src, *args, **kwargs)

        self.initSigControls()
        self.initRateControls()
        self.initLayout()

    def initSigControls(self):
        """Initialize signal controls.
        """
        sigSizer = wx.BoxSizer(orient=wx.VERTICAL)

        distControlBox = widgets.ControlBox(self, label='Distribution', orient=wx.VERTICAL)

        self.distComboBox = wx.ComboBox(self, choices=list(distributions.keys()),
            value='uniform')#, style=wx.CB_SORT | wx.CB_READONLY)
        self.distComboBox.Bind(wx.EVT_COMBOBOX, self.setDist, self.distComboBox)
        distControlBox.Add(self.distComboBox, proportion=0, flag=wx.ALL, border=10)

        self.walkCheckBox = wx.CheckBox(self, label='Walk')
        self.walkCheckBox.Bind(wx.EVT_CHECKBOX, self.setWalk, self.walkCheckBox)
        distControlBox.Add(self.walkCheckBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT, border=10)

        sigSizer.Add(distControlBox, proportion=0, flag=wx.ALL, border=10)

        scaleControlBox = widgets.ControlBox(self, label='Scale', orient=wx.HORIZONTAL)

        self.scaleText = wx.StaticText(self, label='%4.1f' % 1.0)
        scaleTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        scaleTextSizer.Add(self.scaleText, proportion=1, flag=wx.EXPAND)
        self.scaleSlider = wx.Slider(self, style=wx.SL_HORIZONTAL, value=10, minValue=1, maxValue=300)
        self.Bind(wx.EVT_SLIDER, self.setScale, self.scaleSlider)

        scaleControlBox.Add(scaleTextSizer, proportion=0, flag=wx.ALL, border=10)
        scaleControlBox.Add(self.scaleSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        sigSizer.Add(scaleControlBox, proportion=0,
                flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border=10)

        self.sizer.Add(sigSizer)

    def initRateControls(self):
        """Initialize the sample rate and poll size controls.
        """
        rateSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        # poll size
        pollSizeControlBox = widgets.ControlBox(self, label='Poll Size', orient=wx.HORIZONTAL)
        self.pollSizeSpinCtrl = wx.SpinCtrl(self, style=wx.SP_WRAP,
                value=str(self.src.pollSize), min=1, max=32)
        pollSizeControlBox.Add(self.pollSizeSpinCtrl, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SPINCTRL, self.setPollSize, self.pollSizeSpinCtrl)

        rateSizer.Add(pollSizeControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)

        # samp rates
        sampRates = np.array((32,64,128,256,512,1024))

        self.sampRateRadios = [wx.RadioButton(self, label=str(sampRates[0])+'Hz', style=wx.RB_GROUP)] +\
                              [wx.RadioButton(self, label=str(sr)+'Hz') for sr in sampRates[1:]]

        self.sampRateRadios[3].SetValue(True)

        sampRateControlBox = widgets.ControlBox(self, label='Sample Rate', orient=wx.VERTICAL)

        for sr,rbtn in zip(sampRates,self.sampRateRadios):
            def sampRadioWrapper(event, sr=sr):
                try:
                    self.src.setSampRate(sr)
                except Exception as e:
                    wx.LogError('Failed to set sample rate: ' + str(e.message))

            self.Bind(wx.EVT_RADIOBUTTON, sampRadioWrapper, id=rbtn.GetId())

        for rbtn in self.sampRateRadios[:-1]:
            sampRateControlBox.Add(rbtn, proportion=0,
                    flag=wx.TOP | wx.LEFT | wx.RIGHT, border=10)
        sampRateControlBox.Add(self.sampRateRadios[-1], proportion=0, flag=wx.ALL, border=10)

        rateSizer.Add(sampRateControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)

        # number of channels
        nChans = (2, 4, 8, 19, 32, 40, 64, 72)
        self.nChanRadios = [wx.RadioButton(self, label=str(nChans[0]), style=wx.RB_GROUP)] +\
                           [wx.RadioButton(self, label=str(sr)) for sr in nChans[1:]]

        self.nChanRadios[4].SetValue(True)

        nChanControlBox = widgets.ControlBox(self, label='Chans', orient=wx.VERTICAL)

        for nc,rbtn in zip(nChans, self.nChanRadios):
            def nChanRadioWrapper(event, nc=nc):
                try:
                    self.src.setNChan(nc)
                except Exception as e:
                    wx.LogError('Failed to set number of channels: ' + str(e.message))

            self.Bind(wx.EVT_RADIOBUTTON, nChanRadioWrapper, id=rbtn.GetId())

        nChanControlBox.Add(self.nChanRadios[0], proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        for rbtn in self.nChanRadios[1:]:
            nChanControlBox.Add(rbtn, proportion=0,
                    flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        rateSizer.Add(nChanControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)

        self.sizer.Add(rateSizer)

    def setDist(self, event=None):
        """Set distribution in src.
        """
        self.src.setDist(self.distComboBox.GetValue())

    def setWalk(self, event=None):
        """Set walk in src.
        """
        self.src.setWalk(self.walkCheckBox.GetValue())

    def setScale(self, event=None):
        """Set scale in src.
        """
        scale = self.scaleSlider.GetValue() / 10.0

        self.src.setScale(scale)
        self.scaleText.SetLabel('%4.1f' % scale)

    def setPollSize(self, event=None):
        self.src.pollSize = self.pollSizeSpinCtrl.GetValue()

    def beforeStart(self):
        self.pollSizeSpinCtrl.Disable()
        for rbtn in self.sampRateRadios:
            rbtn.Disable()

    def afterStop(self):
        self.pollSizeSpinCtrl.Enable()
        for rbtn in self.sampRateRadios:
            rbtn.Enable()

class Random(Source):
    """Random signal source.
    """
    def __init__(self, mgr, sampRate=256, chans=chans19,
                 dist='uniform', scale=1.0, walk=False, pollSize=2):
        """
        Construct a new source for generating random signals.

        Args:
            sampRate:   Floating point value of the initial sampling frequency.

            chans:      Tuple of strings containing the initial channel
                        configuration.

            dist:       String describing the distribution of random data.
                        May be 'uniform' or 'normal' or 'triangular'

            scale:      Floating point scale parameter for distribution.
                        The impact of this argument is distribution dependent.
                        See rnorm, runif and rtriang in rt.sources.random for
                        more information.

            walk:       If True then data takes a random walk.  If False (default)
                        then the data is just random.

            pollSize:   Number of data samples collected during each poll.
                        Higher values result in better timing and marker
                        resolution but more CPU usage while higher values
                        typically use less CPU but worse timing results.
        """

        self.allChans = list(chans)
        self.dist = mp.Value('I', 0)
        self.scale = mp.Value('d', scale)
        self.walk = mp.Value('b', walk)
        self.pollSize = pollSize
        self.lock = mp.Lock()

        Source.__init__(self, mgr=mgr, sampRate=sampRate, chans=chans,
            configPanelClass=RandomConfigPanel)

        self.setDist(dist)

        self.initWalk()
        self.setWalk(walk)

    def initWalk(self):
        self.walk0Array = mp.Array('d', self.getNChan())
        self.walk0 = np.frombuffer(self.walk0Array.get_obj())
        self.walk0[:] = 0.0 # set start of random walk to zero

    def setNChan(self, nChan):
        if nChan == 2:
            self.setChans([str(chan) for chan in range(2)])
        elif nChan == 4:
            self.setChans([str(chan) for chan in range(4)])
        elif nChan == 8:
            self.setChans(chans8)
        elif nChan == 19:
            self.setChans(chans19)
        elif nChan == 32:
            self.setChans(chans32)
        elif nChan == 40:
            self.setChans(chans40)
        elif nChan == 64:
            self.setChans(chans64)
        elif nChan == 72:
            self.setChans(chans72)
        else:
            raise Exception('Invalid number of channels: ' + str(nChan))

        #self.setChans(self.allChans[:nChan] + self.allChans[-8:])
        self.initWalk()
        self.mgr.updateSources()

    def setDist(self, dist):
        """Set the distribution of the data.

        Args:
            dist:   String describing the distribution of random data.
                    May be 'uniform' or 'normal' or 'triangular'
        """
        dist = dist.lower()

        with self.lock:
            try:
                # index into keys gives us an integer id
                self.dist.value = list(distributions.keys()).index(dist)
            except ValueError:
                raise ValueError('Invalid distribution %s.' % str(dist))

    def setWalk(self, walk):
        """Set whether the data is random or a random walk.

        Args:
            walk:   If True then data takes a random walk.  If False (default)
                    then the data is just random.
        """
        with self.lock:
            self.walk.value = walk
            self.walk0[:] = 0.0

    def setScale(self, scale):
        """Set the scale of the data.

        Args:
            scale:  Floating point scale parameter for distribution.
                    The impact of this argument is distribution dependent.
                    See rnorm, runif and rtriang in rt.sources.random for
                    more information.
        """
        with self.lock:
            self.scale.value = scale

    def beforeRun(self):
        """Set things up for starting the source.
        """
        self.shift = self.pollSize / float(self.sampRate)
        self.pollDelay = -1.0
        self.lastPollTime = -1.0

    def pollData(self):
        """Poll for new data.  This method sleeps in order to ensure
        that self.pollSize observations are generated at a realistic rate.
        """
        # figure time between polls using exponentially weighted moving average
        curTime = time.time()
        if self.pollDelay >= 0.0:
            self.pollDelay = 0.8*self.pollDelay + 0.2*(curTime - self.lastPollTime)
        else:
            self.pollDelay = 0.0
            self.lastPollTime = curTime - self.shift
        sleepTime = np.max((0.0, self.shift - self.pollDelay))
        self.lastPollTime = curTime + sleepTime
        time.sleep(sleepTime)

        with self.lock:
            distFunc = distributions[distributions.keys()[self.dist.value]]

            # generate some random data
            data = distFunc(self.scale.value, (self.pollSize,self.nChan))

            if self.walk.value:
                # cumulative sum over pollSize
                data = np.cumsum(data, axis=0)

                # shift by end of previous poll
                data += self.walk0

                # save end of current poll
                self.walk0[:] = data[-1]

        return data
