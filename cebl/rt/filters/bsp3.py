import numpy as np
import wx

from cebl.rt import widgets

from filt import Filter, FilterConfigPanel

BiosemiP3Name = 'BioSemiP3'


class BiosemiP3ConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label='BioSemi P300 Filter.')
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class BiosemiP3(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=BiosemiP3Name,
                        configPanelClass=BiosemiP3ConfigPanel, **kwargs)

    def apply(self, cap):
        cap.demean().reference(('EXG5','EXG6')).demean()
        #cap.keepChans(('C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2',
        #               'FZ', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'PZ', 'T7', 'T8'))
        cap.keepChans(('Fz', 'Cz', 'P3', 'Pz', 'P4', 'P7', 'Oz', 'P8'))

        # used in rt sub1 day1
        #cap.bandpass(0.5, 10.0, order=2).downsample(32)

        cap.bandpass(0.0, 80.0, order=3).downsample(4)

        return cap
