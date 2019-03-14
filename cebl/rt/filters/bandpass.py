from collections import OrderedDict as odict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import wx

from cebl import sig
from cebl.rt import widgets

from .filt import Filter, FilterConfigPanel

IIRBandpassName = "IIR Bandpass Filter"
FIRBandpassName = "FIR Bandpass Filter"


class IIRBandpassConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        # options go in top-level sizer
        self.initOptions()

        # other stuff split horizontally by bottomSizer
        self.bottomSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.initSliders()
        self.initResponse()
        self.sizer.Add(self.bottomSizer, proportion=1, flag=wx.EXPAND)

        self.initLayout()

    def initOptions(self):
        optionsSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.filtTypeComboBox = wx.ComboBox(self, choices=list(self.flt.filtMap.keys()),
            value=self.flt.filtType, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_COMBOBOX, self.setFiltType, self.filtTypeComboBox)
        optionsSizer.Add(self.filtTypeComboBox, proportion=1,
                flag=wx.LEFT | wx.TOP | wx.RIGHT | wx.ALIGN_CENTER, border=20)

        self.zeroPhaseCheckBox = wx.CheckBox(self, label="Zero Phase")
        self.zeroPhaseCheckBox.SetValue(self.flt.zeroPhase)
        self.Bind(wx.EVT_CHECKBOX, self.setZeroPhase, self.zeroPhaseCheckBox)
        optionsSizer.Add(self.zeroPhaseCheckBox, proportion=1,
                flag=wx.LEFT | wx.TOP | wx.ALIGN_CENTER, border=20)

        self.sizer.Add(optionsSizer, proportion=0)#, flag=wx.EXPAND)

    def setFiltType(self, event):
        filtType = self.filtTypeComboBox.GetValue()

        if filtType not in self.flt.filtMap.keys():
            raise RuntimeError("Invalid filter type: %s." % str(filtType))

        self.flt.filtType = filtType
        self.updateResponse()

    def setZeroPhase(self, event):
        self.flt.zeroPhase = self.zeroPhaseCheckBox.GetValue()
        self.updateResponse()

    def initSliders(self):
        sliderSizer = wx.BoxSizer(wx.HORIZONTAL)

        lowFreqControlBox = widgets.ControlBox(self, label="lowFreq", orient=wx.VERTICAL)
        self.lowFreqText = wx.StaticText(self, label="%6.2f(Hz)" % self.flt.lowFreq)
        lowFreqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        lowFreqTextSizer.Add(self.lowFreqText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.lowFreqSlider = wx.Slider(self, style=wx.SL_VERTICAL | wx.SL_INVERSE,
                minValue=0, maxValue=int(self.flt.nyquist*4), value=int(self.flt.lowFreq*4))
        self.Bind(wx.EVT_SLIDER, self.setLowFreq, self.lowFreqSlider)
        lowFreqControlBox.Add(lowFreqTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        lowFreqControlBox.Add(self.lowFreqSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)

        sliderSizer.Add(lowFreqControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        highFreqControlBox = widgets.ControlBox(self, label="highFreq", orient=wx.VERTICAL)
        self.highFreqText = wx.StaticText(self, label="%6.2f(Hz)" % self.flt.highFreq)
        highFreqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        highFreqTextSizer.Add(self.highFreqText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.highFreqSlider = wx.Slider(self, style=wx.SL_VERTICAL | wx.SL_INVERSE,
                minValue=0, maxValue=int(self.flt.nyquist*4), value=int(self.flt.highFreq*4))
        self.Bind(wx.EVT_SLIDER, self.setHighFreq, self.highFreqSlider)
        highFreqControlBox.Add(highFreqTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        highFreqControlBox.Add(self.highFreqSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)

        sliderSizer.Add(highFreqControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        orderControlBox = widgets.ControlBox(self, label="Order", orient=wx.VERTICAL)
        self.orderText = wx.StaticText(self, label="%2d" % self.flt.order)
        orderTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        orderTextSizer.Add(self.orderText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.orderSlider = wx.Slider(self, style=wx.SL_VERTICAL | wx.SL_INVERSE,
                minValue=1, maxValue=10, value=self.flt.order)
        self.Bind(wx.EVT_SLIDER, self.setOrder, self.orderSlider)
        orderControlBox.Add(orderTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        orderControlBox.Add(self.orderSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)

        sliderSizer.Add(orderControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.bottomSizer.Add(sliderSizer, proportion=1, flag=wx.EXPAND)

    def setLowFreq(self, event):
        self.flt.lowFreq = self.lowFreqSlider.GetValue() / 4.0
        self.lowFreqText.SetLabel("%6.2f(Hz)" % self.flt.lowFreq)
        self.updateResponse()

    def setHighFreq(self, event):
        self.flt.highFreq = self.highFreqSlider.GetValue() / 4.0
        self.highFreqText.SetLabel("%6.2f(Hz)" % self.flt.highFreq)
        self.updateResponse()

    def setOrder(self, event):
        self.flt.order = self.orderSlider.GetValue()
        self.orderText.SetLabel("%2d" % self.flt.order)
        self.updateResponse()

    def initResponse(self):
        self.freqResponseFig = plt.Figure()
        self.freqResponseCanvas = FigureCanvas(parent=self,
                id=wx.ID_ANY, figure=self.freqResponseFig)
        self.freqResponseAx = self.freqResponseFig.add_subplot(1,1,1)
        #self.freqResponseFig.tight_layout()

        self.phaseResponseFig = plt.Figure()
        self.phaseResponseCanvas = FigureCanvas(parent=self,
                id=wx.ID_ANY, figure=self.phaseResponseFig)
        self.phaseResponseAx = self.phaseResponseFig.add_subplot(1,1,1)
        #self.freqResponseFig.tight_layout()

        responseSizer = wx.BoxSizer(wx.VERTICAL)

        freqResponseControlBox = widgets.ControlBox(self,
                label="Freqency Response", orient=wx.VERTICAL)
        freqResponseControlBox.Add(self.freqResponseCanvas, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=8)
        responseSizer.Add(freqResponseControlBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

        phaseResponseControlBox = widgets.ControlBox(self,
                label="Phase Response", orient=wx.VERTICAL)
        phaseResponseControlBox.Add(self.phaseResponseCanvas, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=8)
        responseSizer.Add(phaseResponseControlBox, proportion=1,
                flag=wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

        self.bottomSizer.Add(responseSizer, proportion=1, flag=wx.EXPAND)

        self.freqResponseCanvas.SetMinSize((0,0))
        self.phaseResponseCanvas.SetMinSize((0,0))

        # could we prevent resize when panel is not visible? XXX - idfah
        self.freqResponseLastSize = (0,0)
        self.freqResponseCanvas.Bind(wx.EVT_SIZE, self.freqResponseResize)
        self.phaseResponseLastSize = (0,0)
        self.phaseResponseCanvas.Bind(wx.EVT_SIZE, self.phaseResponseResize)

        self.updateResponse()

    def freqResponseResize(self, event):
        # prevents handling extra resize events, hack XXX - idfah
        size = self.freqResponseCanvas.GetSize()
        if self.freqResponseLastSize == size:
            return

        self.freqResponseLastSize = size
        event.Skip()

    def phaseResponseResize(self, event):
        # prevents handling extra resize events, hack XXX - idfah
        size = self.phaseResponseCanvas.GetSize()
        if self.phaseResponseLastSize == size:
            return

        self.phaseResponseLastSize = size
        event.Skip()

    def updateResponse(self):
        self.flt.updateFilter()

        self.freqResponseAx.cla()
        self.flt.bp.plotFreqResponse(ax=self.freqResponseAx, linewidth=2)
        self.freqResponseAx.autoscale(tight=True)
        self.freqResponseAx.legend(prop={"size": 12})
        self.freqResponseCanvas.draw()

        self.phaseResponseAx.cla()
        self.flt.bp.plotPhaseResponse(ax=self.phaseResponseAx, linewidth=2)
        self.phaseResponseAx.legend(prop={"size": 12})
        self.phaseResponseAx.autoscale(tight=True)

        #if self.flt.zeroPhase:
        #    self.phaseResponseAx.set_ylim((-6,6))
        #ymn, ymx = self.phaseResponseAx.get_ylim()
        #ymn = min(ymn, -0.5)
        #ymx = max(ymx, 0.5)
        #self.phaseResponseAx.set_ylim((ymn, ymx))

        self.phaseResponseCanvas.draw()

class IIRBandpass(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=IIRBandpassName,
                        configPanelClass=IIRBandpassConfigPanel, **kwargs)

        self.filtMap = odict()
        self.filtMap["Butterworth"] = "butter"
        self.filtMap["Chebyshev I"] = "cheby1"
        self.filtMap["Chebyshev II"] = "cheby2"
        self.filtMap["Elliptic"] = "ellip"
        self.filtMap["Bessel"] = "bessel"

        self.filtType = "Butterworth"
        self.nyquist = self.inSampRate/2.0
        self.lowFreq = 0.0
        self.highFreq = self.nyquist
        self.order = 3
        self.zeroPhase = True

        self.rp = 0.25
        self.rs = 20.0

        self.updateFilter()

    def updateFilter(self):
        # handle this logic in the sig.bandpass class? XXX - idfah

        lowFreq = self.lowFreq
        if np.isclose(lowFreq, 0.0, atol=0.0001):
            lowFreq = 0.0

        highFreq = self.highFreq
        if np.isclose(highFreq, 0.0, atol=0.0001):
            highFreq = 0.0
        if np.isclose(highFreq, self.nyquist, atol=0.0001):
            highFreq = np.inf

        filtType = self.filtMap[self.filtType]

        # should be configurable XXX - idfah
        kwargs = {}
        if filtType in ("ellip", "cheby1"):
            kwargs["rp"] = self.rp
        if filtType in ("ellip", "cheby2"):
            kwargs["rs"] = self.rs

        # need dtype argument XXX - idfah
        self.bp = sig.BandpassFilterIIR(lowFreq=lowFreq, highFreq=highFreq, order=self.order,
            filtType=filtType, zeroPhase=self.zeroPhase, sampRate=self.inSampRate, **kwargs)

    def apply(self, cap):
        cap.data = self.bp.filter(cap.data)
        return cap

        #return cap.bandpass(lowFreq=lowFreq, highFreq=highFreq, order=self.order,
        #        filtType=filtType, zeroPhase=self.zeroPhase)


class FIRBandpassConfigPanel(FilterConfigPanel):
    def __init__(self, *args, **kwargs):
        FilterConfigPanel.__init__(self, *args, **kwargs)

        # options go in top-level sizer
        self.initOptions()

        # other stuff split horizontally by bottomSizer
        self.bottomSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.initSliders()
        self.initResponse()
        self.sizer.Add(self.bottomSizer, proportion=1, flag=wx.EXPAND)

        self.initLayout()

    def initOptions(self):
        optionsSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.filtTypeComboBox = wx.ComboBox(self, choices=list(self.flt.filtMap.keys()),
            value=self.flt.filtType, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_COMBOBOX, self.setFiltType, self.filtTypeComboBox)
        optionsSizer.Add(self.filtTypeComboBox, proportion=1,
                flag=wx.LEFT | wx.TOP | wx.RIGHT | wx.ALIGN_CENTER, border=20)

        self.sizer.Add(optionsSizer, proportion=0)#, flag=wx.EXPAND)

    def setFiltType(self, event):
        filtType = self.filtTypeComboBox.GetValue()

        if filtType not in self.flt.filtMap.keys():
            raise RuntimeError("Invalid filter type: %s." % str(filtType))

        self.flt.filtType = filtType
        self.updateResponse()

    def initSliders(self):
        sliderSizer = wx.BoxSizer(wx.HORIZONTAL)

        lowFreqControlBox = widgets.ControlBox(self, label="lowFreq", orient=wx.VERTICAL)
        self.lowFreqText = wx.StaticText(self, label="%6.2f(Hz)" % self.flt.lowFreq)
        lowFreqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        lowFreqTextSizer.Add(self.lowFreqText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.lowFreqSlider = wx.Slider(self, style=wx.SL_VERTICAL | wx.SL_INVERSE,
                minValue=0, maxValue=int(self.flt.nyquist*4), value=int(self.flt.lowFreq*4))
        self.Bind(wx.EVT_SLIDER, self.setLowFreq, self.lowFreqSlider)
        lowFreqControlBox.Add(lowFreqTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        lowFreqControlBox.Add(self.lowFreqSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)

        sliderSizer.Add(lowFreqControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        highFreqControlBox = widgets.ControlBox(self, label="highFreq", orient=wx.VERTICAL)
        self.highFreqText = wx.StaticText(self, label="%6.2f(Hz)" % self.flt.highFreq)
        highFreqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        highFreqTextSizer.Add(self.highFreqText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.highFreqSlider = wx.Slider(self, style=wx.SL_VERTICAL | wx.SL_INVERSE,
                minValue=0, maxValue=int(self.flt.nyquist*4), value=int(self.flt.highFreq*4))
        self.Bind(wx.EVT_SLIDER, self.setHighFreq, self.highFreqSlider)
        highFreqControlBox.Add(highFreqTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        highFreqControlBox.Add(self.highFreqSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)

        sliderSizer.Add(highFreqControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        orderControlBox = widgets.ControlBox(self, label="Order", orient=wx.VERTICAL)
        self.orderText = wx.StaticText(self, label="%2d" % self.flt.order)
        orderTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        orderTextSizer.Add(self.orderText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.orderSlider = wx.Slider(self, style=wx.SL_VERTICAL | wx.SL_INVERSE,
                minValue=2, maxValue=50, value=self.flt.order // 2)
        self.Bind(wx.EVT_SLIDER, self.setOrder, self.orderSlider)
        orderControlBox.Add(orderTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        orderControlBox.Add(self.orderSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)

        sliderSizer.Add(orderControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.bottomSizer.Add(sliderSizer, proportion=1, flag=wx.EXPAND)

    def setLowFreq(self, event):
        self.flt.lowFreq = self.lowFreqSlider.GetValue() / 4.0
        self.lowFreqText.SetLabel("%6.2f(Hz)" % self.flt.lowFreq)
        self.updateResponse()

    def setHighFreq(self, event):
        self.flt.highFreq = self.highFreqSlider.GetValue() / 4.0
        self.highFreqText.SetLabel("%6.2f(Hz)" % self.flt.highFreq)
        self.updateResponse()

    def setOrder(self, event):
        self.flt.order = self.orderSlider.GetValue() * 2
        self.orderText.SetLabel("%2d" % self.flt.order)
        self.updateResponse()

    def initResponse(self):
        self.freqResponseFig = plt.Figure()
        self.freqResponseCanvas = FigureCanvas(parent=self,
                id=wx.ID_ANY, figure=self.freqResponseFig)
        self.freqResponseAx = self.freqResponseFig.add_subplot(1,1,1)
        #self.freqResponseFig.tight_layout()

        self.phaseResponseFig = plt.Figure()
        self.phaseResponseCanvas = FigureCanvas(parent=self,
                id=wx.ID_ANY, figure=self.phaseResponseFig)
        self.phaseResponseAx = self.phaseResponseFig.add_subplot(1,1,1)
        #self.freqResponseFig.tight_layout()

        responseSizer = wx.BoxSizer(wx.VERTICAL)

        freqResponseControlBox = widgets.ControlBox(self,
                label="Freqency Response", orient=wx.VERTICAL)
        freqResponseControlBox.Add(self.freqResponseCanvas, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=8)
        responseSizer.Add(freqResponseControlBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

        phaseResponseControlBox = widgets.ControlBox(self,
                label="Phase Response", orient=wx.VERTICAL)
        phaseResponseControlBox.Add(self.phaseResponseCanvas, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=8)
        responseSizer.Add(phaseResponseControlBox, proportion=1,
                flag=wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

        self.bottomSizer.Add(responseSizer, proportion=1, flag=wx.EXPAND)

        self.freqResponseCanvas.SetMinSize((0,0))
        self.phaseResponseCanvas.SetMinSize((0,0))

        # could we prevent resize when panel is not visible? XXX - idfah
        self.freqResponseLastSize = (0,0)
        self.freqResponseCanvas.Bind(wx.EVT_SIZE, self.freqResponseResize)
        self.phaseResponseLastSize = (0,0)
        self.phaseResponseCanvas.Bind(wx.EVT_SIZE, self.phaseResponseResize)

        self.updateResponse()

    def freqResponseResize(self, event):
        # prevents handling extra resize events, hack XXX - idfah
        size = self.freqResponseCanvas.GetSize()
        if self.freqResponseLastSize == size:
            return

        self.freqResponseLastSize = size
        event.Skip()

    def phaseResponseResize(self, event):
        # prevents handling extra resize events, hack XXX - idfah
        size = self.phaseResponseCanvas.GetSize()
        if self.phaseResponseLastSize == size:
            return

        self.phaseResponseLastSize = size
        event.Skip()

    def updateResponse(self):
        self.flt.updateFilter()

        self.freqResponseAx.cla()
        self.flt.bp.plotFreqResponse(ax=self.freqResponseAx, linewidth=2)
        self.freqResponseAx.autoscale(tight=True)
        self.freqResponseAx.legend(prop={"size": 12})
        self.freqResponseCanvas.draw()

        self.phaseResponseAx.cla()
        self.flt.bp.plotPhaseResponse(ax=self.phaseResponseAx, linewidth=2)
        self.phaseResponseAx.legend(prop={"size": 12})
        self.phaseResponseAx.autoscale(tight=True)

        self.phaseResponseCanvas.draw()

class FIRBandpass(Filter):
    def __init__(self, *args, **kwargs):
        Filter.__init__(self, *args, name=FIRBandpassName,
                        configPanelClass=FIRBandpassConfigPanel, **kwargs)

        self.filtMap = odict()
        self.filtMap["Lanczos"] = "lanczos"
        self.filtMap["Sinc Blackman"] = "sinc-blackman"
        self.filtMap["Sinc Hamming"] = "sinc-hamming"
        self.filtMap["Sinc Hann"] = "sinc-hann"

        self.filtType = "Sinc Blackman"
        self.nyquist = self.inSampRate/2.0
        self.lowFreq = 0.0
        self.highFreq = self.nyquist
        self.order = 20

        self.updateFilter()

    def updateFilter(self):
        # handle this logic in the sig.bandpass class? XXX - idfah

        lowFreq = self.lowFreq
        if np.isclose(lowFreq, 0.0, atol=0.0001):
            lowFreq = 0.0

        highFreq = self.highFreq
        if np.isclose(highFreq, 0.0, atol=0.0001):
            highFreq = 0.0
        if np.isclose(highFreq, self.nyquist, atol=0.0001):
            highFreq = np.inf

        filtType = self.filtMap[self.filtType]

        # need dtype argument XXX - idfah
        self.bp = sig.BandpassFilterFIR(lowFreq=lowFreq, highFreq=highFreq, order=self.order,
            filtType=filtType, sampRate=self.inSampRate)

    def apply(self, cap):
        cap.data = self.bp.filter(cap.data)
        return cap
