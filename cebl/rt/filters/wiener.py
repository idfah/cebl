from collections import OrderedDict as odict
import numpy as np
import wx

from cebl import sig

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

WienerName = "Wiener"


class WienerConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.initOptions()

        self.initLayout()

    def initOptions(self):
        optionsSizer = wx.BoxSizer(wx.HORIZONTAL)

        # sizes should only be odd, easiest way is to implement this as slider with odd values XXX - idfah

        sizeControlBox = widgets.ControlBox(self, label="Size", orient=wx.HORIZONTAL)
        self.sizeSpinCtrl = wx.SpinCtrl(self, value=str(self.flt.size), min=3, max=100)
        self.Bind(wx.EVT_SPINCTRL, self.setSize, self.sizeSpinCtrl)
        sizeControlBox.Add(self.sizeSpinCtrl, proportion=1, flag=wx.ALL, border=8)
        optionsSizer.Add(sizeControlBox, proportion=0,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        # need to add slider for noise parameter XXX - idfah

        self.sizer.Add(optionsSizer, proportion=0)#, flag=wx.EXPAND)

    def setSize(self, event):
        self.flt.size = int(self.sizeSpinCtrl.GetValue())

class Wiener(Filter):
    def __init__(self, *args, **kwargs):
        self.initConfig()

        Filter.__init__(self, *args, name=WienerName,
                        configPanelClass=WienerConfigPanel, **kwargs)

    def initConfig(self):
        self.size = 10
        self.noise = None

    def apply(self, cap):
        return cap.wiener(size=self.size, noise=self.noise)
