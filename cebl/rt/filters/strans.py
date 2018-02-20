import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import time
import wx

from cebl import eeg
from cebl import sig

from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel


class STransConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        self.initCollect()
        self.initOptions()
        self.initTrace()
        self.initLayout()

        if self.flt.filteredTrain is not None:
            self.updateTrace()

    def initCollect(self):
        self.collecting = False
        self.rawView = True

        self.collectRefreshDelay = 0.025
        self.collectTimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.updateTrace, self.collectTimer)

    def initTrace(self):
        self.tracePlot = widgets.TracePlot(self)

        self.sizer.Add(self.tracePlot, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

    def updateTrace(self, event=None):
        if self.collecting:
            # make sure we're not passing through the current strans filter
            self.flt.stransFilter = None
            self.flt.trainCap = self.getCap()

        if self.rawView:
            data = self.flt.trainCap.data
            chanNames = self.flt.trainCap.getChanNames()

        else:
            data = self.flt.filteredTrain
            chanNames = self.flt.getOutChans()

        data = data - data.mean(axis=0)

        #dFactor = int(np.log2(2*data.size/float(256*8*10))) + 1
        dFactor = (data.size // (256*8*10)) + 1
        if dFactor > 1:
            data = sig.decimate(data, factor=dFactor)#, lowpassFrac=0.75, order=4)

        time = self.flt.trainCap.getNSec()

        scale = np.max(2*data.std(axis=0))
        self.tracePlot.draw(data, time, chanNames=chanNames, scale=scale)

    def updateFilt(self, event=None):
        # retrain the filter and update the trace
        self.flt.updateFilter()
        self.updateTrace()

    def updateFiltTrain(self, event=None):
        # just update the filtered training data and the trace
        self.flt.updateFilteredTrain()
        self.updateTrace()

    def initOptions(self):
        optionsSizer = wx.BoxSizer(wx.HORIZONTAL)

        collectControlBox = widgets.ControlBox(self, label='Collect Data', orient=wx.HORIZONTAL)
        self.collectButton = wx.Button(self, label='Start')
        self.Bind(wx.EVT_BUTTON, self.toggleCollect, self.collectButton)
        collectControlBox.Add(self.collectButton, proportion=1, flag=wx.ALL, border=8)
        optionsSizer.Add(collectControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=8)

        trainControlBox = widgets.ControlBox(self, label='Retrain', orient=wx.HORIZONTAL)
        self.trainButton = wx.Button(self, label='Update')
        self.Bind(wx.EVT_BUTTON, self.updateFilt, self.trainButton)
        trainControlBox.Add(self.trainButton, proportion=1, flag=wx.ALL, border=8)
        optionsSizer.Add(trainControlBox, proportion=0,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        viewControlBox = widgets.ControlBox(self, label='View', orient=wx.HORIZONTAL)
        self.rawViewRbtn = wx.RadioButton(self, label='Raw', style=wx.RB_GROUP)
        self.rawViewRbtn.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setRawView, self.rawViewRbtn)
        self.filteredViewRbtn = wx.RadioButton(self, label='Filtered')
        self.Bind(wx.EVT_RADIOBUTTON, self.setFilteredView, self.filteredViewRbtn)

        viewControlBox.Add(self.rawViewRbtn, proportion=1,
                flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=8)
        viewControlBox.Add(self.filteredViewRbtn, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, border=8)

        optionsSizer.Add(viewControlBox, proportion=0,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        if self.flt.filteredTrain is None:
            self.trainButton.Disable()
            self.rawViewRbtn.Disable()
            self.filteredViewRbtn.Disable()

        compControlBox = widgets.ControlBox(self, label='Components', orient=wx.HORIZONTAL)
        self.compTextCtrl = wx.TextCtrl(parent=self, style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.setComp, self.compTextCtrl)
        self.compTextCtrl.Bind(wx.EVT_KILL_FOCUS, self.setComp, self.compTextCtrl)
        compControlBox.Add(self.compTextCtrl, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        self.removeCheckBox = wx.CheckBox(self, label='Remove')
        self.removeCheckBox.SetValue(self.flt.remove)
        self.Bind(wx.EVT_CHECKBOX, self.setRemove, self.removeCheckBox)
        compControlBox.Add(self.removeCheckBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        self.transformCheckBox = wx.CheckBox(self, label='Transform')
        self.transformCheckBox.SetValue(self.flt.transform)
        self.Bind(wx.EVT_CHECKBOX, self.setTransform, self.transformCheckBox)
        compControlBox.Add(self.transformCheckBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        lagsControlBox = widgets.ControlBox(self, label='Lags', orient=wx.HORIZONTAL)
        self.lagsSpinCtrl = wx.SpinCtrl(self, value=str(self.flt.lags), min=0, max=20)
        self.Bind(wx.EVT_SPINCTRL, self.setLags, self.lagsSpinCtrl)
        lagsControlBox.Add(self.lagsSpinCtrl, proportion=1,
                flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=8)
        optionsSizer.Add(lagsControlBox, proportion=0,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        optionsSizer.Add(compControlBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)

        self.sizer.Add(optionsSizer, proportion=0)#, flag=wx.EXPAND)

    def getCap(self):
        return self.pg.src.getEEGSecs(float(time.time() - self.collectTime), filter=True)

    def toggleCollect(self, event):
        if self.collecting:
            # get capture
            self.flt.trainCap = self.getCap()

            # stop collecting
            self.pg.stop()
            self.collecting = False
            self.collectTimer.Stop()
            self.updateFilt()

            # set button label to start
            self.collectButton.SetLabel('Start')

            # enable view buttons
            self.trainButton.Enable()
            self.rawViewRbtn.Enable()
            self.filteredViewRbtn.Enable()

            # set to filtered view
            self.filteredViewRbtn.SetValue(True)
            self.setFilteredView()

        else:
            # set button label to stop
            self.collectButton.SetLabel('Stop')

            # start collecting
            self.pg.start()
            self.collecting = True
            self.collectTime = time.time()
            self.collectTimer.Start(1000.0*self.collectRefreshDelay)

            # set to raw view
            self.rawViewRbtn.SetValue(True)
            self.rawView = True

    def setRawView(self, event=None):
        self.rawView = True
        self.updateTrace()

    def setFilteredView(self, event=None):
        self.rawView = False
        self.updateTrace()

    def setComp(self, event):
        self.flt.comp = []
        compStr = self.compTextCtrl.GetValue()

        if len(compStr) != 0:
            try:
                toks = compStr.replace(' ', '').split(',')

                for tok in toks:
                    if '-' in tok:
                        low, high = (int(c) for c in tok.split('-'))
                        comp = range(low, high+1)
                    else:
                        comp = [int(tok,)]

                    self.flt.comp += comp

            except Exception as e:
                self.flt.comp = []
                wx.LogWarning('Invalid component config: %s.' % str(compStr))

        self.flt.updateFilteredTrain()
        self.updateTrace()

    def setRemove(self, event):
        self.flt.remove = self.removeCheckBox.GetValue()
        self.updateFiltTrain()

    def setTransform(self, event):
        self.flt.transform = self.transformCheckBox.GetValue()
        self.updateFiltTrain()

    def setLags(self, event):
        self.flt.lags = int(self.lagsSpinCtrl.GetValue())
        self.updateFilt()

class STrans(Filter):
    def __init__(self, inSampRate, inChans, stransClass, name, configPanelClass, *args, **kwargs):
        self.stransClass = stransClass

        Filter.__init__(self, inSampRate, inChans, *args, name=name,
                        configPanelClass=configPanelClass, **kwargs)

        self.comp = []
        self.remove = True
        self.transform = True
        self.lags = 0

        self.trainCap = None
        self.filteredTrain = None
        self.updateFilter()

    def updateFilter(self):
        if self.trainCap is not None:
            self.stransFilter = self.stransClass(self.trainCap.data, lags=self.lags)
            self.updateFilteredTrain()

        else:
            self.stransFilter = None

    def updateFilteredTrain(self):
        self.filteredTrain, outChans = self.applySTrans(self.trainCap.data)
        self.setOutChans(outChans)

    def applySTrans(self, data):
        if self.transform:
            filteredData = self.stransFilter.transform(data,
                    comp=self.comp, remove=self.remove)

            nComp = len(self.getInChans()) * (self.lags+1)
            if self.remove:
                outChans = [str(c) for c in range(nComp) if c not in self.comp]
            else:
                outChans = [str(c) for c in range(nComp) if c in self.comp]

        else:
            filteredData = self.stransFilter.filter(data,
                    comp=self.comp, remove=self.remove)

            outChans = self.getInChans()

        assert len(outChans) == filteredData.shape[1]

        return filteredData, outChans

    def apply(self, cap):
        if self.stransFilter is None:
            return cap

        filteredData, outChans = self.applySTrans(cap.data)

        # pass markers except trunkation from lags
        markers = cap.getMarkers()[:filteredData.shape[0]]

        filteredCap = eeg.EEG(filteredData, sampRate=self.getOutSampRate(),
                chanNames=self.getOutChans(), markers=markers,
                deviceName='Max Signal Fraction')

        return filteredCap
