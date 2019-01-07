"""Source to generate ERP test data.
"""
import multiprocessing as mp
import numpy as np
import time
import wx

from cebl import util
from cebl.rt import widgets

from .source import Source, SourceConfigPanel


class ERPTestConfigPanel(SourceConfigPanel):
    """wx panel containing controls for configuring ERP test sources.
    """
    def __init__(self, parent, src, *args, **kwargs):
        """Construct a new configuration panel for a ERP test source.

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

        scaleControlBox = widgets.ControlBox(self, label='Noise Scale', orient=wx.HORIZONTAL)

        self.scaleText = wx.StaticText(self, label='%4.1f' % self.src.scale.value)
        scaleTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        scaleTextSizer.Add(self.scaleText, proportion=1, flag=wx.EXPAND)
        self.scaleSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
                value=self.src.scale.value*10.0, minValue=1, maxValue=300)
        self.Bind(wx.EVT_SLIDER, self.setScale, self.scaleSlider)

        scaleControlBox.Add(scaleTextSizer, proportion=0, flag=wx.ALL, border=10)
        scaleControlBox.Add(self.scaleSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        sigSizer.Add(scaleControlBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        erpSpeedControlBox = widgets.ControlBox(self, label='ERP Speed', orient=wx.HORIZONTAL)

        self.erpSpeedText = wx.StaticText(self, label='%4.1f' % self.src.erpSpeed.value)
        erpSpeedTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        erpSpeedTextSizer.Add(self.erpSpeedText, proportion=1, flag=wx.EXPAND)
        self.erpSpeedSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
                value=self.src.erpSpeed.value*10.0, minValue=1, maxValue=300)
        self.Bind(wx.EVT_SLIDER, self.setERPSpeed, self.erpSpeedSlider)

        erpSpeedControlBox.Add(erpSpeedTextSizer, proportion=0, flag=wx.ALL, border=10)
        erpSpeedControlBox.Add(self.erpSpeedSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        sigSizer.Add(erpSpeedControlBox, proportion=0,
                flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border=10)

        triggerControlBox = widgets.ControlBox(self, label='Trigger', orient=wx.HORIZONTAL)

        self.triggerValueTextCtrl = wx.TextCtrl(self,
                value=str(self.src.trigger.value))#, style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT, self.setTrigger, self.triggerValueTextCtrl)

        self.triggerButton = wx.Button(self, label='Manual')
        self.Bind(wx.EVT_BUTTON, self.manualTrigger, self.triggerButton)

        triggerControlBox.Add(self.triggerValueTextCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        triggerControlBox.Add(self.triggerButton, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

        sigSizer.Add(triggerControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(sigSizer)

    def initRateControls(self):
        """Initialize the sample rate and poll size controls.
        """
        rateSizer = wx.BoxSizer(orient=wx.VERTICAL)

        pollSizeControlBox = widgets.ControlBox(self, label='Poll Size', orient=wx.HORIZONTAL)
        self.pollSizeSpinCtrl = wx.SpinCtrl(self, style=wx.SP_WRAP,
                value=str(self.src.pollSize), min=1, max=32)
        pollSizeControlBox.Add(self.pollSizeSpinCtrl, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SPINCTRL, self.setPollSize, self.pollSizeSpinCtrl)

        rateSizer.Add(pollSizeControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)

        sampRates = np.array((128,256,512,1024))

        self.sampRateRadios = [wx.RadioButton(self, label=str(sampRates[0])+'Hz', style=wx.RB_GROUP)] +\
                              [wx.RadioButton(self, label=str(sr)+'Hz') for sr in sampRates[1:]]

        self.sampRateRadios[0].SetValue(True)

        sampRateControlBox = widgets.ControlBox(self, label='Sample Rate', orient=wx.VERTICAL)

        for sr,rbtn in zip(sampRates, self.sampRateRadios):
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
                flag=wx.BOTTOM | wx.RIGHT, border=10)

        self.sizer.Add(rateSizer)

    def setScale(self, event=None):
        """Set scale in src.
        """
        scale = self.scaleSlider.GetValue() / 10.0

        self.src.setScale(scale)
        self.scaleText.SetLabel('%4.1f' % scale)

    def setERPSpeed(self, event=None):
        erpSpeed = self.erpSpeedSlider.GetValue() / 10.0

        self.src.setERPSpeed(erpSpeed)
        self.erpSpeedText.SetLabel('%4.1f' % erpSpeed)

    def setTrigger(self, event=None):
        # should check if chr XXX - idfah

        value = self.triggerValueTextCtrl.GetValue()

        if len(value) == 0:
            fValue = 0.0
        else:
            try:
                fValue = float(value)
            except ValueError:
                fValue = float(ord(value[0]))

        self.src.setTrigger(fValue)

    def manualTrigger(self, event=None):
        self.src.erpReset()

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

class ERPTest(Source):
    """ERP test source.
    """
    def __init__(self, mgr, sampRate=128,
                 #chans=('Fz','C3','Cz','C4','P3','Pz','P4','Oz'),
                 chans=('Fz','Cz','P3','Pz','P4','P7','Oz','P8'),
                 scale=1.0, erpSpeed=6.0, pollSize=2):
        """
        Construct a new source for generating ERP test signals.

        Args:
            sampRate:   Floating point value of the initial sampling frequency.

            chans:      Tuple of strings containing the initial channel
                        configuration.

            scale:      Floating point scale parameter for scale of the
                        signal noise.

            pollSize:    Number of data samples collected during each poll.
                        Higher values result in better timing and marker
                        resolution but more CPU usage while higher values
                        typically use less CPU but worse timing results.
        """
        self.scale = mp.Value('d', scale)
        self.trigger = mp.Value('d', 1.0)
        self.erpStart = mp.Value('d', 1024.0)
        self.erpSpeed = mp.Value('d', erpSpeed)
        self.pollSize = pollSize
        self.lock = mp.Lock()

        Source.__init__(self, mgr=mgr, sampRate=sampRate, chans=chans,
            configPanelClass=ERPTestConfigPanel)

        self.setTrigger(1.0)

    def erpReset(self):
        with self.lock:
            self.erpStart.value = -2.0

    def setERPSpeed(self, speed):
        with self.lock:
            self.erpSpeed.value = speed

    def setTrigger(self, trigger):
        with self.lock:
            self.trigger.value = trigger

    def setScale(self, scale):
        """Set the scale of the data.

        Args:
            scale:  Floating point scale parameter for scale of the
                    signal noise.
        """
        with self.lock:
            self.scale.value = scale

    def setMarker(self, marker):
        if np.isclose(marker, self.trigger.value):
            #print('resetting erp')
            self.erpReset()

        Source.setMarker(self, marker)

    def beforeRun(self):
        """Set things up for starting the source.
        """
        with self.lock:
            self.erpStart.value = 1024.0

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
            # generate some random data
            data = np.random.uniform(-self.scale.value, self.scale.value,
                                     size=(self.pollSize,self.nChan))

            erpEnd = self.erpStart.value +\
                self.erpSpeed.value *\
                data.shape[0]/float(self.sampRate)

            erp = np.linspace(self.erpStart.value, erpEnd, data.shape[0])
            erp = np.repeat(erp, data.shape[1]).reshape((-1,data.shape[1]))
            erp = erp * 0.5*(np.arange(data.shape[1])+1.0)
            erp = np.sinc(erp)

            data += erp

            self.erpStart.value = erpEnd

        return data
