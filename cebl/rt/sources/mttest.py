"""Source to generate Mental Task test data.
"""
import multiprocessing as mp
import numpy as np
import scipy.signal as spsig
import time
import wx

from cebl import util
from cebl.rt import widgets

from .source import Source, SourceConfigPanel


class MTTestConfigPanel(SourceConfigPanel):
    """wx panel containing controls
    """
    def __init__(self, parent, src, *args, **kwargs):
        """Construct a new configuration panel

        Args:
            parent:         wx parent of this panel.

            src:            Source to configure.

            args, kwargs:   Additional arguments passed to the SourceConfigPanel
                            base class.
        """
        SourceConfigPanel.__init__(self, parent=parent, src=src, *args, **kwargs)

        self.initTriggerControls()
        self.initSigControls()

        self.initLayout()

    def initTriggerControls(self):
        """Initialize trigger controls.
        """
        triggerSizer = wx.BoxSizer(orient=wx.VERTICAL)

        triggerControlBox = widgets.ControlBox(self,
                label='Manual Trigger', orient=wx.VERTICAL)

        self.trigger0Button = wx.Button(self, label='Class 0')
        self.Bind(wx.EVT_BUTTON, self.trigger0, self.trigger0Button)
        triggerControlBox.Add(self.trigger0Button, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.trigger1Button = wx.Button(self, label='Class 1')
        self.Bind(wx.EVT_BUTTON, self.trigger1, self.trigger1Button)
        triggerControlBox.Add(self.trigger1Button, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.trigger2Button = wx.Button(self, label='Class 2')
        self.Bind(wx.EVT_BUTTON, self.trigger2, self.trigger2Button)
        triggerControlBox.Add(self.trigger2Button, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.trigger3Button = wx.Button(self, label='Class 3')
        self.Bind(wx.EVT_BUTTON, self.trigger3, self.trigger3Button)
        triggerControlBox.Add(self.trigger3Button, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.trigger4Button = wx.Button(self, label='Class 4')
        self.Bind(wx.EVT_BUTTON, self.trigger4, self.trigger4Button)
        triggerControlBox.Add(self.trigger4Button, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        triggerSizer.Add(triggerControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(triggerSizer)

    def trigger0(self, event=None):
        self.src.trigger(0)

    def trigger1(self, event=None):
        self.src.trigger(1)

    def trigger2(self, event=None):
        self.src.trigger(2)

    def trigger3(self, event=None):
        self.src.trigger(3)

    def trigger4(self, event=None):
        self.src.trigger(4)

    def initSigControls(self):
        """Initialize signal controls.
        """
        sigSizer = wx.BoxSizer(orient=wx.VERTICAL)

        scaleControlBox = widgets.ControlBox(self, label='Noise', orient=wx.VERTICAL)

        self.scaleText = wx.StaticText(self, label='%4.1f' % self.src.scale.value)
        scaleTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        scaleTextSizer.Add(self.scaleText, proportion=1, flag=wx.EXPAND)
        self.scaleSlider = wx.Slider(self, style=wx.SL_VERTICAL,
                value=self.src.scale.value*10.0, minValue=1, maxValue=100)
        self.Bind(wx.EVT_SLIDER, self.setScale, self.scaleSlider)

        scaleControlBox.Add(scaleTextSizer, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        scaleControlBox.Add(self.scaleSlider, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        sigSizer.Add(scaleControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(sigSizer, proportion=1, flag=wx.EXPAND)

    def setScale(self, event=None):
        """Set scale in src.
        """
        scale = self.scaleSlider.GetValue() / 10.0

        self.src.setScale(scale)
        self.scaleText.SetLabel('%4.1f' % scale)


class MTTest(Source):
    """Mental tasks test source.
    """
    def __init__(self, mgr, sampRate=256,
                 chans=('Fz','Cz','P3','Pz','P4','P7','Oz','P8'),
                 pollSize=2):
        """
        Construct a new wave generator source.

        Args:
            sampRate:   Floating point value of the initial sampling frequency.

            chans:      Tuple of strings containing the initial channel
                        configuration.

            pollSize:    Number of data samples collected during each poll.
                        Higher values result in better timing and marker
                        resolution but more CPU usage while higher values
                        typically use less CPU but worse timing results.
        """

        self.trig = mp.Value('i', 0)
        self.scale = mp.Value('d', 1.0)
        self.t0 = mp.Value('d', 0.0)
        self.t0.value = 0.0
        self.pollSize = pollSize
        self.lock = mp.Lock()

        Source.__init__(self, mgr=mgr, sampRate=sampRate, chans=chans,
            configPanelClass=MTTestConfigPanel)
        self.freqs = [0.2, 0.4, 0.6, 0.8, 1.0]

        s = np.random.get_state()
        np.random.seed(779)
        self.mixMats = [np.random.random((self.getNChan(), self.getNChan()))
                        for i in range(4)]
        np.random.set_state(s)
        self.mixMats.insert(0, np.identity(self.getNChan()))

    def setScale(self, scale):
        """Set the scale of the data.

        Args:
            scale:  Floating point scale parameter for scale of the
                    signal noise.
        """
        with self.lock:
            self.scale.value = scale

    def setMarker(self, marker):
        if np.any(np.isclose(marker, (0,1,2,3,4))):
            self.trigger(int(marker))

        if np.any(np.isclose(marker/10.0, (1,2,3,4))):
            self.trigger(int(marker/10.0))

        print(self.trig.value)

        Source.setMarker(self, marker)

    def trigger(self, trig):
        with self.lock:
            self.trig.value = trig

    def beforeRun(self):
        self.t = np.arange(0.0, self.pollSize/float(self.sampRate), 1.0/self.sampRate)
        self.t = util.colmat(self.t)

        self.shift = self.pollSize / float(self.sampRate)
        self.pollDelay = -1.0
        self.lastPollTime = -1.0

    def pollData(self):
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
            rand = np.random.uniform(-self.scale.value, self.scale.value,
                                     size=(self.pollSize,self.nChan))

            mixMat = self.mixMats[self.trig.value]
            freq = self.freqs[self.trig.value]

            freqs = freq * (2.0*np.pi*np.power(2.0, np.arange(self.nChan))/2.0)
            freqs = freqs[self.activeChanIndex]

            newData = np.sin(freqs * (self.t + self.t0.value)).dot(mixMat) + rand
            self.t0.value += self.shift

        return newData

    def __repr__(self):
        r = Source.__repr__(self)
        return r
