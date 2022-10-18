import numpy as np
import time
import wx
from wx.lib.agw import aui

from cebl.rt import widgets
from cebl import sig

from .standard import StandardConfigPanel, StandardMonitorPage


class ConfigPanel(StandardConfigPanel):
    """Panel containing configuration widgets.  This is intimately
    related to this specific page.
    """
    def __init__(self, *args, **kwargs):
        StandardConfigPanel.__init__(self, *args, **kwargs)

        self.initTopbar()
        self.initBottombar()
        self.initStandardLayout()

    def initTopbar(self):
        """Initialize controls at top of panel.
        """
        # sizer for top of config panel
        self.topbar = wx.BoxSizer(orient=wx.VERTICAL)

        # combobox for selecting filtered or non-filtered signal
        filterControlBox = widgets.ControlBox(self, label='Signal', orient=wx.VERTICAL)
        self.filterComboBox = wx.ComboBox(self, choices=('Raw', 'Filtered'),
            value='FIltered', style=wx.CB_READONLY)
        self.Bind(wx.EVT_COMBOBOX, self.setFilter, self.filterComboBox)
        filterControlBox.Add(self.filterComboBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        # add filterControlBox to topbar sizer
        self.topbar.Add(filterControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        # radio buttons for turning marker channel on and off
        markerControlBox = widgets.ControlBox(self, label='Marker Channel', orient=wx.HORIZONTAL)

        markerOnRbtn = wx.RadioButton(self, label='On', style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, self.setMarkerOn, markerOnRbtn)
        markerControlBox.Add(markerOnRbtn, proportion=0, flag=wx.ALL, border=10)

        markerOffRbtn = wx.RadioButton(self, label='Off')
        markerOffRbtn.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setMarkerOff, markerOffRbtn)
        markerControlBox.Add(markerOffRbtn, proportion=0, flag=wx.ALL, border=10)

        # add markerControlBox to topbar sizer
        self.topbar.Add(markerControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        # radio buttons for selecting plot quality
        qualityControlBox = widgets.ControlBox(self, label='Plot Quality', orient=wx.HORIZONTAL)

        # low quality radio button
        # first button determines group using wx.RB_GROUP
        lowQualityRbtn = wx.RadioButton(self, label='Low', style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, self.setLowQuality, lowQualityRbtn)
        qualityControlBox.Add(lowQualityRbtn, proportion=0, flag=wx.ALL, border=10)

        # medium quality radio button
        mediumQualityRbtn = wx.RadioButton(self, label='Medium')
        mediumQualityRbtn.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setMediumQuality, mediumQualityRbtn)
        qualityControlBox.Add(mediumQualityRbtn, proportion=0, flag=wx.ALL, border=10)

        # high quality radio button
        highQualityRbtn = wx.RadioButton(self, label='High')
        self.Bind(wx.EVT_RADIOBUTTON, self.setHighQuality, highQualityRbtn)
        qualityControlBox.Add(highQualityRbtn, proportion=0, flag=wx.ALL, border=10)

        # add qualityControlBox to topbar sizer
        self.topbar.Add(qualityControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        # radio buttons for selection decimation factor
        decimateControlBox = widgets.ControlBox(self, label='Decimation', orient=wx.HORIZONTAL)

        # list of buttons, first button determines group
        rbtns = [wx.RadioButton(self, label='None', style=wx.RB_GROUP)] + \
                [wx.RadioButton(self, label=l) for l in ('1/2', '1/4', '1/8')]
        rbtns[0].SetValue(True) # select first button to start

        # bind callbacks to each radio button with appropriate factors
        for rbtn,factor in zip(rbtns,(1, 2, 4, 8)):
            # Uses lexical scoping to save ratio for each button.
            def setDecimationWrapper(event, factor=factor):
                self.setDecimation(factor=factor)
            self.Bind(wx.EVT_RADIOBUTTON, setDecimationWrapper, rbtn)
            #decimateControlBox.Add(rbtn, proportion=0, flag=wx.ALL, border=10)

        for rbtn in rbtns[:-1]:
            decimateControlBox.Add(rbtn, proportion=0, flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)
        decimateControlBox.Add(rbtns[-1], proportion=0, flag=wx.ALL, border=10)

        # add decimateControlBox to topbar sizer
        self.topbar.Add(decimateControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(self.topbar, proportion=0, flag=wx.EXPAND)

    def initBottombar(self):
        """Initialize controls at bottom of panel.
        """
        # sizer for bottom of configPanel
        self.bottombar = wx.BoxSizer(orient=wx.HORIZONTAL)

        # slider for controlling width of trace
        # since sliders are int, we use divide by 4 to get float value
        widthControlBox = widgets.ControlBox(self, label='Width', orient=wx.VERTICAL)
        self.widthText = wx.StaticText(self, label='%6.2f(s)' % self.pg.width)
        widthTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        widthTextSizer.Add(self.widthText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.widthSlider = wx.Slider(self, style=wx.SL_VERTICAL,
            value=int(self.pg.width * 4), minValue=1, maxValue=240)
        widthControlBox.Add(widthTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        widthControlBox.Add(self.widthSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)
        self.Bind(wx.EVT_SLIDER, self.setWidth, self.widthSlider)

        # add widthControlBox to bottombar sizer
        self.bottombar.Add(widthControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        # slider for controlling scale of trace
        # since sliders are int, we use divide by 4 to get float value
        # zero results in automagic scaling
        scaleControlBox = widgets.ControlBox(self, label='Scale', orient=wx.VERTICAL)
        self.scaleText = wx.StaticText(self, label='auto       ')
        scaleTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        scaleTextSizer.Add(self.scaleText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.scaleSlider = wx.Slider(self, style=wx.SL_VERTICAL,
            value=int(self.pg.scale), minValue=0, maxValue=5000)
        scaleControlBox.Add(scaleTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        scaleControlBox.Add(self.scaleSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)
        self.Bind(wx.EVT_SLIDER, self.setScale, self.scaleSlider)

        # add scaleControlBox to bottombar sizer
        self.bottombar.Add(scaleControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        # slider for selecting delay between consecutive refreshes of the trace
        refreshControlBox = widgets.ControlBox(self, label='Refresh', orient=wx.VERTICAL)
        self.refreshText = wx.StaticText(self, label='%3d(ms)' % int(self.pg.getRefreshDelay()))
        refreshTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        refreshTextSizer.Add(self.refreshText, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.refreshSlider = wx.Slider(self, style=wx.SL_VERTICAL,
            value=int(self.pg.getRefreshDelay()/10), minValue=1, maxValue=200)
        refreshControlBox.Add(refreshTextSizer, proportion=0,
                flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=8)
        refreshControlBox.Add(self.refreshSlider, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=25)
        self.Bind(wx.EVT_SLIDER, self.setRefresh, self.refreshSlider)

        # add refreshControlBox to bottombar sizer
        self.bottombar.Add(refreshControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        # make sure sliders don't collapse vertically all the way when undocked
        minX, minY = self.bottombar.GetMinSize()
        minY *= 3
        self.bottombar.SetMinSize((minX, minY))

        self.sizer.Add(self.bottombar, proportion=1, flag=wx.EXPAND)

    def setFilter(self, event=None):
        """Set filter status.
        """
        filterStatus = self.filterComboBox.GetValue().lower()

        if filterStatus == 'raw':
            self.pg.filter = False
        else:
            self.pg.filter = True

    def setMarkerOn(self, event=None):
        self.pg.showMarker = True

    def setMarkerOff(self, event=None):
        self.pg.showMarker = False

    def setLowQuality(self, event=None):
        """Set plot to low quality in page.
        """
        self.pg.plot.setLowQuality()

    def setMediumQuality(self, event=None):
        """Set plot to medium quality in page.
        """
        self.pg.plot.setMediumQuality()

    def setHighQuality(self, event=None):
        """Set plot to high quality in page.
        """
        self.pg.plot.setHighQuality()

    def setDecimation(self, factor):
        """Set decimation factor in page.
        """
        self.pg.decimationFactor = factor

    def setWidth(self, event=None):
        """Set plot width in page.
        Divide slider value by 4 to get float.
        """
        self.pg.width = self.widthSlider.GetValue() / 4.0
        self.widthText.SetLabel('%6.2f(s)' % self.pg.width)

    def setScale(self, event=None):
        """Set scale in page.
        Divide slider value by 4 to get float.
        Zero starts automagic scaling.
        """
        curScale = self.scaleSlider.GetValue()

        if curScale == 0:
            self.pg.scale = 0
            self.scaleText.SetLabel('auto       ')
        else:
            self.pg.scale = curScale / 20.0
            self.scaleText.SetLabel('%6.2f(uv)' % self.pg.scale)

    def setRefresh(self, event=None):
        """Set refreshDelay in page.
        Multiply by a factor of 10 to get to ms scale.
        """
        refreshDelay = float(self.refreshSlider.GetValue()*10)
        self.pg.setRefreshDelay(refreshDelay)
        self.refreshText.SetLabel('%4d(ms)' % int(refreshDelay))

class Trace(StandardMonitorPage):
    """Main class for a page that generates trace plots of EEG.
    Trace is configurable and allows still plots to be captured and
    EEG to be saved to file.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new Trace page.

        Args:
            *args, **kwargs:  Arguments to pass to the Page base class.
        """
        self.initConfig()

        StandardMonitorPage.__init__(self, name='Trace',
                configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initConfig(self):
        """Initialize configuration values.
        """
        self.filter = True          # use raw or filtered signal
        self.showMarker = False     # flag to indicate whether or not to plot the marker channel
        self.decimationFactor = 1   # decimation factor, e.g., 2 will decimate to half sampRate
        self.width = 5.0            # width of the trace plot in seconds
        self.scale = 0.0            # scale of the plot (space between zeros for each channel)

    def initPlots(self):
        """Initialize a new TracePlot widgets for EEG and marker.
        """
        self.plot = widgets.TracePlot(self)

    def initLayout(self):
        self.initStandardLayout()

        # plot pane
        plotPaneAuiInfo = aui.AuiPaneInfo().Name('plot').Caption('EEG Trace').CenterPane()
        self.auiManager.AddPane(self.plot, plotPaneAuiInfo)

        self.auiManager.Update()

    def updatePlot(self, event=None):
        """Draw the trace plot.
        """
        # get EEG data from current source
        cap = self.src.getEEGSecs(self.width, filter=self.filter, copy=True)

        # decimate EEG
        if self.decimationFactor > 1:
            cap.decimate(self.decimationFactor)

        # remove channel means
        cap.demean()

        data = cap.data
        chanNames = cap.getChanNames()

        if self.showMarker:
            markers = cap.markers

            # if markers are not all zero
            if not np.all(np.isclose(markers, 0.0)):
                # if data is not all zero
                if not np.all(np.isclose(data, 0.0)):
                    # marker is always autoscaled
                    if np.isclose(self.scale, 0.0):
                        markerScale = np.max(np.abs(data))
                    else:
                        markerScale = self.scale
                    markers = 0.9 * markers * markerScale / np.max(np.abs(markers))

            data = np.hstack((data,markers[:,None]))
            chanNames = chanNames + ['Mk']

        # tell trace plot widget to draw
        self.plot.draw(data, self.width, scale=self.scale, chanNames=chanNames)

    def captureImage(self, event=None):
        self.plot.saveFile()
