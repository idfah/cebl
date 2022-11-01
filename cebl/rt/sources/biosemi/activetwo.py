import ctypes
import importlib
import os
import time

import numpy as np
import wx

from cebl import util
from cebl.rt import widgets

from cebl.rt.sources.source import Source, SourceConfigPanel

# find the .so file for the biosemi interface
# we have to search all possible extensions for this build, feels gross
# but I'm unsure how else to reliably find the file name
for _so_ext in importlib.machinery.EXTENSION_SUFFIXES:
    _biosemi_libpath = os.path.join(os.path.dirname(__file__), f'libactivetwo{_so_ext}')
    if os.path.isfile(_biosemi_libpath):
        _biosemi = ctypes.cdll.LoadLibrary(_biosemi_libpath)
        break
else:
    raise FileNotFoundError('Unable to find libactivetwo shared library.')

#bs = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/libactivetwo.so')
## bs = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/libactivetwo.cpython-36m-x86_64-linux-gnu.so')

_biosemi.bs_poll.argtypes = [ctypes.POINTER(ctypes.c_double),]


chans32 = ['Fp1', 'AF3', 'F7',  'F3',  'FC1', 'FC5', 'T7',  'C3',
           'CP1', 'CP5', 'P7',  'P3',  'Pz',  'PO3', 'O1',  'Oz',
           'O2',  'PO4', 'P4',  'P8',  'CP6', 'CP2', 'C4',  'T8',
           'FC6', 'FC2', 'F4',  'F8',  'AF4', 'Fp2', 'Fz',  'Cz',
           'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

chans64 = ['Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',  'FT7',
           'FC5', 'FC3', 'FC1', 'C1',  'C3',  'C5',  'T7',  'TP7',
           'CP5', 'CP3', 'CP1', 'P1',  'P3',  'P5',  'P7',  'P9',
           'PO7', 'PO3', 'O1',  'Iz',  'Oz',  'POz', 'Pz',  'CPz',
           'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',  'F2',  'F4',
           'F6',  'F8',  'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
           'C2',  'C4',  'C6',  'T8',  'TP8', 'CP6', 'CP4', 'CP2',
           'P2',  'P4',  'P6',  'P8',  'P10', 'PO8', 'PO4', 'O2',
           'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

class ActiveTwoConfigPanel(SourceConfigPanel):
    def __init__(self, parent, src, *args, **kwargs):
        SourceConfigPanel.__init__(self, parent=parent, src=src, orient=wx.VERTICAL, *args, **kwargs)

        self.initRateControls()
        self.initLayout()

    def initRateControls(self):
        """Initialize the poll size control.
        """
        rateSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        # poll rate
        pollSizeControlBox = widgets.ControlBox(self, label='Poll Size', orient=wx.HORIZONTAL)
        self.pollSizeSpinCtrl = wx.SpinCtrl(self, style=wx.SP_WRAP,
                value=str(self.src.pollSize), min=1, max=32)
        pollSizeControlBox.Add(self.pollSizeSpinCtrl, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SPINCTRL, self.setPollSize, self.pollSizeSpinCtrl)

        rateSizer.Add(pollSizeControlBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        # speed mode
        speedModeControlBox = widgets.ControlBox(self, label='Speed Mode', orient=wx.HORIZONTAL)
        self.speedModeSpinCtrl = wx.SpinCtrl(self, style=wx.SP_WRAP,
                value=str(self.src.speedMode.value), min=4, max=7)
        speedModeControlBox.Add(self.speedModeSpinCtrl, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SPINCTRL, self.setSpeedMode, self.speedModeSpinCtrl)

        rateSizer.Add(speedModeControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=10)

        # number of channels
        nChans = (32, 64)
        self.nChanRadios = [wx.RadioButton(self, label=str(nChans[0]), style=wx.RB_GROUP)] +\
                           [wx.RadioButton(self, label=str(sr)) for sr in nChans[1:]]

        self.nChanRadios[-1].SetValue(True)

        nChanControlBox = widgets.ControlBox(self, label='Num Chans', orient=wx.HORIZONTAL)

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
                    flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=10)

        rateSizer.Add(nChanControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=10)

        self.sizer.Add(rateSizer)

    def setPollSize(self, event=None):
        self.src.pollSize = self.pollSizeSpinCtrl.GetValue()

    def setSpeedMode(self, event=None):
        self.src.setSpeedMode(self.speedModeSpinCtrl.GetValue())

    def beforeStart(self):
        self.pollSizeSpinCtrl.Disable()
        self.speedModeSpinCtrl.Disable()

    def afterStop(self):
        self.speedModeSpinCtrl.Enable()

class ActiveTwo(Source):
    def __init__(self, mgr, sampRate=1024, chans=chans64, pollSize=4):
        Source.__init__(self, mgr=mgr, sampRate=sampRate, chans=chans,
            name='ActiveTwo', configPanelClass=ActiveTwoConfigPanel)

        self.connected = False
        self.pollSize = pollSize

        self.speedMode = ctypes.c_int(4)
        self.setSpeedMode(4)

    def setSpeedMode(self, speedMode):
        # only support speedMode 4 for now
        if speedMode != 4:
            wx.LogError('Only speedMode=4 supported for now.')

        self.speedMode.value = 4
        self.nDataChan = 256
        self.nAuxChan = 24
        self.nExgChan = 8

    def setNChan(self, nc):
        #self.setChans(self.allChans[:nc] + self.allChans[-8:])
        if nc == 32:
            self.setChans(chans32)
        elif nc == 64:
            self.setChans(chans64)
        else:
            self.setChans([str(c) for c in range(nc)])

        self.mgr.updateSources()

    def connect(self):
        wx.LogMessage(self.getName() + ': connecting.')

        try:
            if not self.connected:
                if _biosemi.bs_open():
                    raise RuntimeError('Failed to open ActiveTwo.')

                self.connected = True

        except Exception as e:
            self.connected = False
            raise

    def disconnect(self):
        wx.LogMessage(self.getName() + ': disconnecting.')
        try:
            if self.connected:
                if _biosemi.bs_close():
                    raise RuntimeError('Failed to close ActiveTwo.')

        except Exception as e:
            raise

        finally:
            self.connected = False

    def configure(self):
        if _biosemi.bs_setScansPerPoll(self.pollSize*2):
            raise RuntimeError('Failed to set pollSize.')
        #if _biosemi.bs_setScansPerPoll(self.pollSize):
        #    raise RuntimeError('Failed to set pollSize.')

        # only speedmode 4 is supported for now XXX - idfah
        # need to be able to change number of channels after initializing source XXX - idfah
        if _biosemi.bs_setSpeedMode(4):
            raise RuntimeError('Failed to set speedMode.')

    def query(self):
        try:
            self.configure()
            self.connect()

        except Exception as e:
            raise RuntimeError('Failed to query ActiveTwo: ' + str(e))

        finally:
            self.disconnect()

        return repr(self)

    def beforeRun(self):
        try:
            self.initChanMask()
            self.configure()
            self.connect()

            if _biosemi.bs_start():
                raise RuntimeError('Failed to start ActiveTwo acquisition.')

        except Exception as e:
            self.disconnect()
            raise

    def afterRun(self):
        if _biosemi.bs_stop():
            raise RuntimeError('Failed to stop ActiveTwo acquisition.')

        self.disconnect()

    def initChanMask(self):
        self.chanMask = np.arange(self.getNChan())
        self.chanMask[-self.nExgChan:] = self.nDataChan + \
                                         np.arange(self.nExgChan)
        self.chanMask += 2

    def pollData(self):
        #data = np.zeros((self.pollSize, 2+self.nDataChan+self.nAuxChan), dtype=np.float64)
        data = np.zeros((2*self.pollSize, 2+self.nDataChan+self.nAuxChan), dtype=np.float64)

        if _biosemi.bs_poll(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))):
            raise RuntimeError('Failed to poll ActiveTwo.')

        return data[::2,self.chanMask] # ::2 is a hack to downsample XXX - idfah

    def __repr__(self):
        r = Source.__repr__(self)
        r  += '\nConfiguration:\n' + \
              '====================\n' + \
              'SpeedMode:        '     + str(self.speedMode.value) + '\n' + \
              'Data Channels:    '     + str(self.nDataChan)      + '\n' + \
              'Aux Channels:     '     + str(self.nAuxChan)       + '\n' + \
              'Exg Channels:     '     + str(self.nExgChan)       + '\n'

        return r
