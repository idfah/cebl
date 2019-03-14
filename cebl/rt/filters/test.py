import numpy as np
import wx

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

TestName = "Test"


class TestConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label="Test Filter.")
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class Test(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=TestName,
                        configPanelClass=TestConfigPanel, **kwargs)

    def apply(self, cap):
        cap.keepChans(("C1", "C2"))
        return cap
