from collections import OrderedDict as odict
import numpy as np
import wx

from cebl import sig

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

MovingAverageName = 'Moving Average'


class MovingAverageConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.initOptions()
        self.initKernel()

        self.initLayout()

    def initOptions(self):
        optionsSizer = wx.BoxSizer(wx.HORIZONTAL)

        kernTypeControlBox = widgets.ControlBox(self, label='Kernel Type', orient=wx.HORIZONTAL)
        self.kernTypeComboBox = wx.ComboBox(self, choices=list(self.flt.kernMap.keys()),
            value=self.flt.kernType, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_COMBOBOX, self.setKernType, self.kernTypeComboBox)
        kernTypeControlBox.Add(self.kernTypeComboBox, proportion=1, flag=wx.ALL, border=8)
        optionsSizer.Add(kernTypeControlBox, proportion=1,
                flag=wx.ALL | wx.ALIGN_CENTER, border=8)

        widthControlBox = widgets.ControlBox(self, label='Width', orient=wx.HORIZONTAL)
        self.widthSpinCtrl = wx.SpinCtrl(self, value=str(self.flt.width), min=2, max=100)
        self.Bind(wx.EVT_SPINCTRL, self.setWidth, self.widthSpinCtrl)
        widthControlBox.Add(self.widthSpinCtrl, proportion=1, flag=wx.ALL, border=8)
        optionsSizer.Add(widthControlBox, proportion=0,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        self.sizer.Add(optionsSizer, proportion=0)#, flag=wx.EXPAND)

    def setKernType(self, event):
        kernType = self.kernTypeComboBox.GetValue()

        if kernType not in self.flt.kernMap.keys():
            raise RuntimeError('Invalid kernel type: %s.' % str(kernType))

        self.flt.kernType = kernType

    def setWidth(self, event):
        self.flt.width = int(self.widthSpinCtrl.GetValue())

    def initKernel(self):
        pass

class MovingAverage(Filter):
    def __init__(self, *args, **kwargs):
        self.initConfig()

        Filter.__init__(self, *args, name=MovingAverageName,
                        configPanelClass=MovingAverageConfigPanel, **kwargs)

    def initConfig(self):
        self.width = 10

        self.kernMap = odict()
        self.kernMap['Boxcar'] = sig.windows.boxcar
        self.kernMap['Gaussian'] = lambda w: sig.windows.gaussian(w, std=0.12*w) # configurable std XXX - idfah
        self.kernMap['Triangular'] = sig.windows.triang

        self.kernType = 'Boxcar'

    def apply(self, cap):
        return cap.ma(width=self.width, kernelFunc=self.kernMap[self.kernType])
