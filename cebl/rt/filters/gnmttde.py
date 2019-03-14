import numpy as np
import wx

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

GNautilusMTTDEName = "GNautilusMTTDE"


class GNautilusMTTDEConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label="Hello World!")
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class GNautilusMTTDE(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=GNautilusMTTDEName,
                        configPanelClass=GNautilusMTTDEConfigPanel, **kwargs)

    def apply(self, cap):
        #cap.keepChans(("F3", "F4", "C3", "C4", "P3", "P4", "PO3", "PO4"))
        cap.keepChans(("C3", "C4", "CZ", "F3", "F4", "F7", "F8", "FP1", "FP2",
                       "FZ", "Oz", "P3", "P4", "P7", "P8", "PZ", "T7",  "T8"))

        cap.demean()
        #cap.bandpass(0.0, 28.0, order=3).downsample(3)
        cap.bandpass(60.5, 59.5, order=2)
        cap.bandpass(0.0, 80.0, order=3)
        cap.bandpass(1.0, np.inf, order=3)

        return cap
