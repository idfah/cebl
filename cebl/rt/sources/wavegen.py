"""Source to generate various periodic waveforms.
"""
import multiprocessing as mp
import numpy as np
import scipy.signal as spsig
import time
import wx

from cebl import util
from cebl.rt import widgets

from source import Source, SourceConfigPanel


waveforms = {
    'sinusoid': np.sin,
    'sawtooth': spsig.sawtooth,
    'square': spsig.square
}

class WaveGenConfigPanel(SourceConfigPanel):
    """wx panel containing controls for configuring wave generator signal sources.
    """
    def __init__(self, parent, src, *args, **kwargs):
        """Construct a new configuration panel for a wave generator signal source.

        Args:
            parent:         wx parent of this panel.

            src:            Source to configure.

            args, kwargs:   Additional arguments passed to the SourceConfigPanel
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

        waveformControlBox = widgets.ControlBox(self, label='Waveform', orient=wx.VERTICAL)

        self.waveformComboBox = wx.ComboBox(self, id=wx.ID_ANY, choices=waveforms.keys(),
            value='sinusoid', style=wx.CB_SORT | wx.CB_READONLY)
        self.waveformComboBox.Bind(wx.EVT_COMBOBOX, self.setWaveform)

        waveformControlBox.Add(self.waveformComboBox, proportion=0, flag=wx.ALL, border=10)

        sigSizer.Add(waveformControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        freqControlBox = widgets.ControlBox(self, label='Base Frequency', orient=wx.HORIZONTAL)

        self.freqText = wx.StaticText(self, label='%4.1f(Hz)' % 1.0)
        freqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        freqTextSizer.Add(self.freqText, proportion=1, flag=wx.EXPAND)
        self.freqSlider = wx.Slider(self, style=wx.SL_HORIZONTAL, value=10, minValue=1, maxValue=300)
        self.Bind(wx.EVT_SLIDER, self.setFreq, self.freqSlider)

        freqControlBox.Add(freqTextSizer, proportion=0, flag=wx.ALL, border=10)
        freqControlBox.Add(self.freqSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        sigSizer.Add(freqControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        mixControlBox = widgets.ControlBox(self, label='Channel Mixer', orient=wx.HORIZONTAL)

        self.mixNoneButton = wx.RadioButton(self, label='None', style=wx.RB_GROUP)
        #mixControlBox.Add(self.mixNoneButton, proportion=0, flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)
        mixControlBox.Add(self.mixNoneButton, proportion=0, flag=wx.ALL, border=10)
        self.mixNoneButton.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setMixNone, self.mixNoneButton)

        #self.mixEqualButton = wx.RadioButton(self, label='Equal')
        #mixControlBox.Add(self.mixEqualButton, proportion=0, flag=wx.ALL, border=10)
        #self.Bind(wx.EVT_RADIOBUTTON, self.setMixEqual, self.mixEqualButton)

        self.mixRandomButton = wx.RadioButton(self, label='Random')
        #mixControlBox.Add(self.mixRandomButton, proportion=0, flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)
        mixControlBox.Add(self.mixRandomButton, proportion=0, flag=wx.ALL, border=10)
        self.Bind(wx.EVT_RADIOBUTTON, self.setMixRandom, self.mixRandomButton)

        sigSizer.Add(mixControlBox, proportion=0,
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

        sampRateControlBox= widgets.ControlBox(self, label='Sample Rate', orient=wx.VERTICAL)

        for sr,rbtn in zip(sampRates, self.sampRateRadios):
            def sampRadioWrapper(event, sr=sr):
                self.src.setSampRate(sr)

            self.Bind(wx.EVT_RADIOBUTTON, sampRadioWrapper, id=rbtn.GetId())

        for rbtn in self.sampRateRadios[:-1]:
            sampRateControlBox.Add(rbtn, proportion=0,
                    flag=wx.TOP | wx.LEFT | wx.RIGHT, border=10)
        sampRateControlBox.Add(self.sampRateRadios[-1], proportion=0, flag=wx.ALL, border=10)

        rateSizer.Add(sampRateControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT, border=10)

        self.sizer.Add(rateSizer)

    def setWaveform(self, event=None):
        """Set the waveform in src.
        """
        self.src.setWaveform(self.waveformComboBox.GetValue())

    def setFreq(self, event=None):
        """Set base frequency in src.
        """
        freq = self.freqSlider.GetValue() / 10.0

        self.src.setFreq(freq)
        self.freqText.SetLabel('%4.1f(Hz)' % freq)

    def setMixNone(self, event=None):
        """Set mix to none in src.
        """
        self.src.setMix('none')

    def setMixEqual(self, event=None):
        """Set mix to equal in src.
        """
        self.src.setMix('equal')

    def setMixRandom(self, event=None):
        """Set mix to random in src.
        """
        self.src.setMix('random')

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


class WaveGen(Source):
    """Wave generator source.
    """
    def __init__(self, mgr, sampRate=128,
                 chans=[str(n)+'x' for n in np.power(2, np.arange(8))/2.0],
                 waveform='sinusoid', freq=1.0, mix='none', pollSize=2):
        """
        Construct a new wave generator source.

        Args:
            sampRate:   Floating point value of the initial sampling frequency.

            chans:      Tuple of strings containing the initial channel
                        configuration.

            waveform:   String describing the type of waveform to produce.
                        May be 'sinusoid' or 'sawtooth' or 'square'

            freq:       Base frequency.  Each channel is a power-of-two
                        multiple of this frequency.

            pollSize:    Number of data samples collected during each poll.
                        Higher values result in better timing and marker
                        resolution but more CPU usage while higher values
                        typically use less CPU but worse timing results.
        """

        self.waveform = mp.Value('I', 0)
        self.freq = mp.Value('d', freq)
        self.t0 = mp.Value('d', 0.0)
        self.t0.value = 0.0
        self.pollSize = pollSize
        self.lock = mp.Lock()

        Source.__init__(self, mgr=mgr, sampRate=sampRate, chans=chans,
            configPanelClass=WaveGenConfigPanel)

        self.setWaveform(waveform)

        self.mixArr = mp.Array('d', self.getNChan()*self.getNChan())
        self.mixMat = (np.frombuffer(self.mixArr.get_obj())
                        .reshape((-1,self.getNChan())))
        self.setMix(mix)

    def setWaveform(self, waveform):
        """Set the periodic waveform to generate.

        Args:
            waveform:   String describing the type of waveform to produce.
                        May be 'sinusoid' or 'sawtooth' or 'square'
        """
        waveform = waveform.lower()

        with self.lock:
            try:
                # index into keys gives us an integer id
                self.waveform.value = waveforms.keys().index(waveform)
            except ValueError:
                raise ValueError('Invalid waveform %s.' % str(waveform))

    def setFreq(self, freq):
        with self.lock:
            self.freq.value = freq

    def setMix(self, mix='none'):
        wx.LogMessage('Setting mixing matrix to %s.' % mix)
        with self.lock:
            mix = mix.lower()
            if mix == 'none':
                self.mixMat[:,:] = np.eye(self.mixMat.shape[0])
            elif mix == 'equal':
                self.mixMat[:,:] = 1.0 / self.mixMat.shape[0]
            elif mix == 'random':
                self.mixMat[:,:] = np.random.random(self.mixMat.shape)
                self.mixMat /= self.mixMat.sum(axis=0)
            else:
                raise Exception('Invalid mix mode %s.' % str(mix))

    def beforeRun(self):
        #self.t = np.linspace(0.0, self.pollSize/float(self.sampRate), self.pollSize)
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
            waveFunc = waveforms[waveforms.keys()[self.waveform.value]]

            freqs = self.freq.value * (2.0*np.pi*np.power(2.0, np.arange(self.nChan))/2.0)
            freqs = freqs[self.activeChanIndex]

            newData = waveFunc(freqs * (self.t + self.t0.value)).dot(self.mixMat)
            self.t0.value += self.shift

        return newData

    def __repr__(self):
        r = Source.__repr__(self)
        r  += '\nOptions:\n' + \
              '====================\n'  + \
              'Base Frequency: '        + str(self.freq.value)  + '\n' + \
              'Mixing Matrix:\n'         + str(np.round(self.mixMat,2))      + '\n' + \
              '====================\n'

        return r
