import wx

from cebl import ml
from cebl.rt import widgets

from .strans import STrans, STransConfigPanel

IndependentComponentsName = "Independent Components"


class ICAConfigPanel(STransConfigPanel):
    #def __init__(self, *args, **kwargs):
    #    STransConfigPanel.__init__(self, *args, **kwargs)

    def initOptions(self):
        STransConfigPanel.initOptions(self)

        optionsSizer = wx.BoxSizer(wx.HORIZONTAL)

        kurtosisControlBox = widgets.ControlBox(self, label="Kurtosis", orient=wx.VERTICAL)
        self.kurtosisComboBox = wx.ComboBox(self, choices=("Adapt", "Sub", "Super"),
            value=self.flt.kurtosis, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_COMBOBOX, self.setKurtosis, self.kurtosisComboBox)
        kurtosisControlBox.Add(self.kurtosisComboBox, proportion=1, flag=wx.ALL, border=8)
        optionsSizer.Add(kurtosisControlBox, proportion=1,
                flag=wx.LEFT | wx.RIGHT, border=8)

        maxIterControlBox = widgets.ControlBox(self, label="Max Iter.", orient=wx.HORIZONTAL)
        self.maxIterSpinCtrl = wx.SpinCtrl(self, value=str(self.flt.maxIter), min=50, max=3500)
        self.Bind(wx.EVT_SPINCTRL, self.setMaxIter, self.maxIterSpinCtrl)
        maxIterControlBox.Add(self.maxIterSpinCtrl, proportion=1,
                flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=8)
        optionsSizer.Add(maxIterControlBox, proportion=0,
                flag=wx.RIGHT | wx.EXPAND, border=8)

        #lrControlBox = widgets.ControlBox(self, label="Learning Rate.", orient=wx.HORIZONTAL)

        self.sizer.Add(optionsSizer, proportion=0)

    def setKurtosis(self, event):
        self.flt.kurtosis = self.kurtosisComboBox.GetValue()

    def setMaxIter(self, event):
        self.flt.maxIter = int(self.maxIterSpinCtrl.GetValue())

class IndependentComponents(STrans):
    def __init__(self, *args, **kwargs):

        self.initICAConfig()

        STrans.__init__(self, *args, stransClass=self.ICAWrapper, name=IndependentComponentsName,
                configPanelClass=ICAConfigPanel, **kwargs)

    def initICAConfig(self):
        self.kurtosis = "Adapt"
        self.learningRate = 0.01
        self.maxIter = 1500

    def ICAWrapper(self, *args, **kwargs):
        return ml.ICA(*args, kurtosis=self.kurtosis.lower(), learningRate=self.learningRate,
                maxIter=self.maxIter, callback=self.ICACallback, **kwargs)

    def ICACallback(self, iteration, wtol):
        if (iteration % 50) == 0:
            percent = int(100*iteration/float(self.maxIter))
            self.dialog.Update(percent, "Complete: %d%%\nwtol: %f" % (percent,wtol))

    def updateFilter(self):
        if self.trainCap is not None:
            self.dialog = wx.ProgressDialog("Training ICA",
                    "Training", maximum=101,
                    style=wx.PD_ELAPSED_TIME | wx.PD_SMOOTH)

            STrans.updateFilter(self)

            #self.dialog.Update(101, "Reason: %s" % self.stransFilter.reason)
            self.dialog.Destroy()

        else:
            STrans.updateFilter(self)

