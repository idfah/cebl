import numpy as np
import wx

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

GNautilusP3Name = "GNautilusP3"


class GNautilusP3ConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label="GNautilus P300 Filter.")
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class GNautilusP3(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=GNautilusP3Name,
                        configPanelClass=GNautilusP3ConfigPanel, **kwargs)

    def apply(self, cap):
        cap.demean()
        cap.keepChans(("Fz", "Cz", "P3", "Pz", "P4", "P7", "Oz", "P8"))

        #cap.bandpass(0.0, 80.0, order=3)

        return cap
