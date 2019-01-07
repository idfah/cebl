import ctypes
import numpy as np
import os
import time
import wx

from cebl import util
from cebl.rt import widgets

from cebl.rt.sources.source import Source, SourceConfigPanel


#ms = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/libmindset24r.so')
ms = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/libmindset24r.cpython-36m-x86_64-linux-gnu.so')

ms.ms_Open.argtypes = [ctypes.POINTER(ctypes.c_char)]
ms.ms_ReadNextDataBlock.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

# enums in libmindset24r.h, is there a better way? XXX - idfah
BLOCKSIZE96  = ctypes.c_int(1)
BLOCKSIZE192 = ctypes.c_int(2)
BLOCKSIZE384 = ctypes.c_int(4)
BLOCKSIZE768 = ctypes.c_int(8)

SAMPLERATE0    = ctypes.c_int(0)
SAMPLERATE1024 = ctypes.c_int(1)
SAMPLERATE512  = ctypes.c_int(2)
SAMPLERATE256  = ctypes.c_int(3)
SAMPLERATE128  = ctypes.c_int(4)
SAMPLERATE64   = ctypes.c_int(5)

class Mindset24RConfigPanel(SourceConfigPanel):
    def __init__(self, parent, src, *args, **kwargs):
        SourceConfigPanel.__init__(self, parent=parent, src=src, orient=wx.VERTICAL, *args, **kwargs)

        self.initDeviceControls()
        self.initRateControls()
        self.initLayout()

    def initDeviceControls(self):
        deviceControlBox = widgets.ControlBox(self, label='Device Path', orient=wx.HORIZONTAL)

        self.deviceTextCtrl = wx.TextCtrl(self, value=self.src.getDeviceName())
        self.Bind(wx.EVT_TEXT, self.setDeviceName, self.deviceTextCtrl)

        deviceControlBox.Add(self.deviceTextCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(deviceControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

    def setDeviceName(self, event):
        deviceName = self.deviceTextCtrl.GetValue()
        self.src.setDeviceName(deviceName)

    def initRateControls(self):
        # sample rate config
        rateSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        sampRates = np.array((64,128,256,512,1024))

        self.sampRateRadios = [wx.RadioButton(self, label=str(sampRates[0])+'Hz', style=wx.RB_GROUP)] +\
                              [wx.RadioButton(self, label=str(sr)+'Hz') for sr in sampRates[1:]]

        self.sampRateRadios[2].SetValue(True)

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
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT, border=10)

        # block size config
        blockSizes = np.array((96, 192, 384, 768))

        self.blockSizeRadios = [wx.RadioButton(self, label=str(blockSizes[0])+'KB', style=wx.RB_GROUP)] +\
                               [wx.RadioButton(self, label=str(bs)+'KB') for bs in blockSizes[1:]]

        self.blockSizeRadios[3].SetValue(True)

        blockSizeControlBox = widgets.ControlBox(self, label='Block Size', orient=wx.VERTICAL)

        for bs,rbtn in zip(blockSizes,self.blockSizeRadios):
            def sampRadioWrapper(event, bs=bs):
                try:
                    self.src.setBlockSize(bs)
                except Exception as e:
                    wx.LogError('Failed to set block size: ' + str(e.message))

            self.Bind(wx.EVT_RADIOBUTTON, sampRadioWrapper, id=rbtn.GetId())

        for rbtn in self.blockSizeRadios[:-1]:
            blockSizeControlBox.Add(rbtn, proportion=0,
                    flag=wx.TOP | wx.LEFT | wx.RIGHT, border=10)
        blockSizeControlBox.Add(self.blockSizeRadios[-1], proportion=0, flag=wx.ALL, border=10)

        rateSizer.Add(blockSizeControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT, border=10)

        self.sizer.Add(rateSizer)

    def beforeStart(self):
        self.deviceTextCtrl.Disable()
        for rbtn in self.sampRateRadios:
            rbtn.Disable()
        for rbtn in self.blockSizeRadios:
            rbtn.Disable()

    def afterStop(self):
        self.deviceTextCtrl.Enable()
        for rbtn in self.sampRateRadios:
            rbtn.Enable()
        for rbtn in self.blockSizeRadios:
            rbtn.Enable()

class Mindset24R(Source):
    def __init__(self, mgr, sampRate=256,
                 chans=('FP1','FP2','F3','F4','C3','C4',
                        'P3','P4','O1','O2','F7','F8',
                        'T3','T4','P7','P8','CZ','FZ','PZ',
                        None, None, None, None, None)):
                        #'EX1', 'EX2', 'EX3', 'EX4', 'EX5')):
        Source.__init__(self, mgr=mgr, sampRate=sampRate, chans=chans,
            name='Mindset24R', configPanelClass=Mindset24RConfigPanel)

        self.connected = False
        self.deviceName = '/dev/mindset'
        self.blockSizeKey = BLOCKSIZE768
        self.actualBlockSize = -1.0
        self.sampRateKey = SAMPLERATE256

    def connect(self):
        wx.LogMessage(self.getName() + ': connecting.')
        try:
            if not self.connected:
                ms.ms_Open(ctypes.c_char_p(self.deviceName))
                if ms.ms_Ready() < 0:
                    raise RuntimeError('The mindset24r is not ready.')
                self.connected = True
        except Exception as e:
            self.connected = False
            raise RuntimeError('Failed to connect to mindset24r: ' + str(e))

    def disconnect(self):
        wx.LogMessage(self.getName() + ': disconnecting.')
        try:
            if self.connected:
                ms.ms_Close()
        except Exception as e:
            raise RuntimeError('Failed to disconnect from mindset24r: ' + str(e))
        finally:
            self.connected = False

    def configure(self):
        try:
            ms.ms_SetBlockSize(self.blockSizeKey)
            self.actualBlockSize = ms.ms_ActualBlockSize(self.blockSizeKey)

            ms.ms_SetSampleRate(self.sampRateKey)
            self.sampRate = ms.ms_ActualSampleRate(self.sampRateKey)
        except Exception as e:
            raise RuntimeError('Failed to configure mindset24r: ' + str(e))

    def query(self):
        try:
            self.connect()
            self.configure()
        except Exception as e:
            raise RuntimeError('Failed to query mindset24r: ' + str(e))
        finally:
            self.disconnect()

        return repr(self)

    def beforeStart(self):
        try:
            self.connect()
            self.configure()
        except Exception as e:
            self.disconnect()
            raise RuntimeError('Failed to start mindset24r acquisition: ' + str(e))

    def afterStop(self):
        self.disconnect()

    def pollData(self):
        sampPerBlock = int(self.actualBlockSize / 2 / 24)

        data = np.zeros((sampPerBlock, 24), dtype=np.float64)

        # better way to handle errors? XXX - idfah
        tryCount = 0
        while ms.ms_ReadNextDataBlock(
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                    ctypes.c_int(sampPerBlock)) < sampPerBlock:
            if tryCount < 10:
                tryCount += 1
            else:
                raise RuntimeError('Failed to keep up with mindset buffer.')

        return data

    def getDeviceName(self):
        return self.deviceName

    def setDeviceName(self, deviceName):
        self.deviceName = deviceName

    def setSampRate(self, sampRate=256):
        if sampRate == 64:
            self.sampRateKey = SAMPLERATE64
        elif sampRate == 128:
            self.sampRateKey = SAMPLERATE128
        elif sampRate == 256:
            self.sampRateKey = SAMPLERATE256
        elif sampRate == 512:
            self.sampRateKey = SAMPLERATE512
        elif sampRate == 1024:
            self.sampRateKey = SAMPLERATE1024
        else:
            raise RuntimeError('Invalid sample rate ' + str(sampRate))

    def setBlockSize(self, blockSize=768):
        if blockSize == 96:
            self.blockSizeKey = BLOCKSIZE96
        elif blockSize == 192:
            self.blockSizeKey = BLOCKSIZE192
        elif blockSize == 384:
            self.blockSizeKey = BLOCKSIZE384
        elif blockSize == 768:
            self.blockSizeKey = BLOCKSIZE768
        else:
            raise RuntimeError('Invalid block size ' + str(blockSize))

    def __repr__(self):
        r = Source.__repr__(self)
        r  += '\nConfiguration:\n' + \
              '====================\n' + \
              'SampRateKey:     '      + str(self.sampRateKey)     + '\n' + \
              'ActualSampRate:  '      + str(self.sampRate)        + '\n' + \
              'BlockSizeKey:    '      + str(self.blockSizeKey)    + '\n' + \
              'ActualBlockSize: '      + str(self.actualBlockSize) + '\n'

        return r
