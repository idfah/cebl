import numpy as np
import wx

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

ReferenceName = 'Reference'


class ReferenceConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.hello = wx.StaticText(self, label='Hello World!')
        self.sizer.Add(self.hello, proportion=1, flag=wx.EXPAND, border=10)

        self.initLayout()

class Reference(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=ReferenceName,
                        configPanelClass=ReferenceConfigPanel, **kwargs)

    def apply(self, cap):
        return cap.car()
