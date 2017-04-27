import numpy as np
import wx

from cebl.rt import widgets

from filt import Filter, FilterConfigPanel

BiosemiMTCNAName = 'BioSemiMTCNA'


class BiosemiMTCNAConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label='Hello World!')
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class BiosemiMTCNA(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=BiosemiMTCNAName,
                        configPanelClass=BiosemiMTCNAConfigPanel, **kwargs)

    def apply(self, cap):
        cap.demean().reference(('EXG5','EXG6')).demean()
        cap.keepChans(('F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'))
        #cap.keepChans(('C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2',
        #               'FZ', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'PZ', 'T7', 'T8'))

        cap.bandpass(0.0, 80.0, order=3).downsample(4)
        cap.bandpass(1.0, np.inf, order=3)

        return cap
