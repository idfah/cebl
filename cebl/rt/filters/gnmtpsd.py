import numpy as np
import wx

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

GNautilusMTPSDName = 'GNautilusMTPSD'


class GNautilusMTPSDConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label='Hello World!')
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class GNautilusMTPSD(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=GNautilusMTPSDName,
                        configPanelClass=GNautilusMTPSDConfigPanel, **kwargs)

    def apply(self, cap):
        cap.keepChans(('F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'PO3', 'PO4'))

        cap.demean()
        cap.bandpass(0.0, 20.0, order=3)
        cap.bandpass(1.0, np.inf, order=3).car()

        return cap
