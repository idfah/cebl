import wx
import wx.lib.agw.floatspin as agwfs


class ControlBox(wx.StaticBoxSizer):
    def __init__(self, parent, label='', orient=wx.VERTICAL):
        wx.StaticBoxSizer.__init__(self, wx.StaticBox(parent, label=label), orient=orient)

'''class LabeledFloatSpinCtrl(wx.Panel):
    def __init__(self, parent, label='', digits=3, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent)

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.label = wx.StaticText(parent=self, label=label)
        self.sizer.Add(self.label, proportion=0, flag=wx.BOTTOM, border=5)

        self.spinner = agwfs.FloatSpin(parent=self, *args, **kwargs)
        self.spinner.SetFormat("%f")
        self.spinner.SetDigits(digits)
        self.sizer.Add(self.spinner, proportion=0)#, flag=wx.ALL)#, border=10)

        self.SetSizer(self.sizer)
        self.Layout()
'''
