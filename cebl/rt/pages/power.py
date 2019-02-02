import wx
from wx.lib.agw import aui

from cebl import util
from cebl.rt import widgets

from .standard import StandardConfigPanel, StandardMonitorPage


class WelchConfigPanel(wx.Panel):
    def __init__(self, parent, pg, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.pg = pg

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.initControls()
        self.Layout()

    def initControls(self):
        # slider for controlling the length of the spans in Welch's method
        # since sliders are int, we use divide by 4 to get float value
        spanControlBox = widgets.ControlBox(self, label='Span', orient=wx.HORIZONTAL)
        self.spanText = wx.StaticText(self, label='%4.2f(s)' % self.pg.welchConfig.span)
        spanTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        spanTextSizer.Add(self.spanText, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.spanSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.welchConfig.span*4), minValue=1, maxValue=12)
        spanControlBox.Add(spanTextSizer, proportion=0, flag=wx.TOP, border=10)
        spanControlBox.Add(self.spanSlider, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setSpan, self.spanSlider)

        self.sizer.Add(spanControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setSpan(self, event=None):
        """Set span in page.
        Divide slider value by 4 to get float.
        """
        self.pg.welchConfig.span = self.spanSlider.GetValue() / 4.0
        self.spanText.SetLabel('%4.2f(s)' % self.pg.welchConfig.span)

class AutoregConfigPanel(wx.Panel):
    def __init__(self, parent, pg, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.pg = pg

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.initControls()

        self.Layout()

    def initControls(self):
        orderControlBox = widgets.ControlBox(self, label='Model Order', orient=wx.HORIZONTAL)
        self.orderText = wx.StaticText(self, label='%3d' % self.pg.autoregConfig.order)
        orderTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        orderTextSizer.Add(self.orderText, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.orderSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=self.pg.autoregConfig.order, minValue=1, maxValue=100)
        orderControlBox.Add(orderTextSizer, proportion=0, flag=wx.TOP, border=10)
        orderControlBox.Add(self.orderSlider, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setOrder, self.orderSlider)

        self.sizer.Add(orderControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        nFreqControlBox = widgets.ControlBox(self, label='Num Freqs', orient=wx.HORIZONTAL)
        self.nFreqText = wx.StaticText(self, label='%3d' % self.pg.autoregConfig.nFreq)
        nFreqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        nFreqTextSizer.Add(self.nFreqText, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.nFreqSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=self.pg.autoregConfig.nFreq/5, minValue=1, maxValue=100)
        nFreqControlBox.Add(nFreqTextSizer, proportion=0, flag=wx.TOP, border=10)
        nFreqControlBox.Add(self.nFreqSlider, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setNFreq, self.nFreqSlider)

        self.sizer.Add(nFreqControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setOrder(self, event=None):
        self.pg.autoregConfig.order = self.orderSlider.GetValue()
        self.orderText.SetLabel('%3d' % self.pg.autoregConfig.order)

    def setNFreq(self, event=None):
        self.pg.autoregConfig.nFreq = self.nFreqSlider.GetValue() * 5
        self.nFreqText.SetLabel('%3d' % self.pg.autoregConfig.nFreq)

class ConfigPanel(StandardConfigPanel):
    """Panel containing configuration widgets.
    This is intimately related to this specific page.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new panel containing configuration widgets.

        Args:
            parent: Parent in wx hierarchy.

            pg:     Page to be configured.

            *args, **kwargs:  Additional arguments passed
                              to the wx.Panel base class.
        """
        StandardConfigPanel.__init__(self, *args, **kwargs)

        self.initControls()
        self.initLayout()

    def initControls(self):
        # combobox for selecting filtered or non-filtered signal
        self.filterControlBox = widgets.ControlBox(self, label='Signal', orient=wx.VERTICAL)
        self.filterComboBox = wx.ComboBox(self, choices=('Raw', 'Filtered'),
            value='Filtered', style=wx.CB_READONLY)
        self.filterControlBox.Add(self.filterComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_COMBOBOX, self.setFilter, self.filterComboBox)

        # slider for controlling width of window
        # since sliders are int, we use divide by 4 to get float value
        self.widthControlBox = widgets.ControlBox(self, label='Width', orient=wx.HORIZONTAL)
        self.widthText = wx.StaticText(self, label='%5.2f(s)' % self.pg.width)
        widthTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        widthTextSizer.Add(self.widthText, proportion=1, flag=wx.EXPAND)
        self.widthSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.width*4.0), minValue=12, maxValue=120)
        self.widthControlBox.Add(widthTextSizer, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        self.widthControlBox.Add(self.widthSlider, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setWidth, self.widthSlider)

        # radio buttons for selection decimation factor
        self.decimateControlBox = widgets.ControlBox(self,
                label='Decimation', orient=wx.HORIZONTAL)

        # list of buttons, first button determines group
        rbtns = [wx.RadioButton(self, label='None', style=wx.RB_GROUP)] + \
                [wx.RadioButton(self, label=l) for l in ('1/2', '1/4', '1/8')]
        rbtns[0].SetValue(True) # select first button to start

        # bind callbacks to each radio button with appropriate factors
        for rbtn, factor in zip(rbtns, (1,2,4,8)):
            # Uses lexical scoping to save ratio for each button.
            def setDecimationWrapper(event, factor=factor):
                self.setDecimation(factor=factor)
            self.Bind(wx.EVT_RADIOBUTTON, setDecimationWrapper, rbtn)

        for rbtn in rbtns[:-1]:
            self.decimateControlBox.Add(rbtn, proportion=0,
                    flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)
        self.decimateControlBox.Add(rbtns[-1], proportion=0,
                flag=wx.ALL, border=10)

        # radio buttons for selecting plot quality
        self.qualityControlBox = widgets.ControlBox(self,
                label='Plot Quality', orient=wx.HORIZONTAL)

        # low quality radio button
        # first button determines group using wx.RB_GROUP
        lowQualityRbtn = wx.RadioButton(self, label='Low', style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, self.setLowQuality, lowQualityRbtn)
        self.qualityControlBox.Add(lowQualityRbtn, proportion=0,
                flag=wx.ALL, border=10)

        # medium quality radio button
        mediumQualityRbtn = wx.RadioButton(self, label='Medium')
        mediumQualityRbtn.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setMediumQuality, mediumQualityRbtn)
        self.qualityControlBox.Add(mediumQualityRbtn, proportion=0,
                flag=wx.ALL, border=10)

        # high quality radio button
        highQualityRbtn = wx.RadioButton(self, label='High')
        self.Bind(wx.EVT_RADIOBUTTON, self.setHighQuality, highQualityRbtn)
        self.qualityControlBox.Add(highQualityRbtn, proportion=0, flag=wx.ALL, border=10)

        # slider for selecting delay between consecutive refreshes of the plot
        self.refreshControlBox = widgets.ControlBox(self, label='Refresh', orient=wx.HORIZONTAL)
        self.refreshText = wx.StaticText(self, label='%4d(ms)' % int(self.pg.getRefreshDelay()), style=wx.ALIGN_CENTER)
        refreshTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        refreshTextSizer.Add(self.refreshText, proportion=1,
                flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.refreshSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.getRefreshDelay()/10), minValue=1, maxValue=200)
        self.refreshControlBox.Add(refreshTextSizer, proportion=0,
                flag=wx.TOP, border=10)
        self.refreshControlBox.Add(self.refreshSlider, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setRefresh, self.refreshSlider)

        # combobox for selecting PSD estimation method
        self.methodControlBox = widgets.ControlBox(self, label='Method', orient=wx.VERTICAL)
        self.methodComboBox = wx.ComboBox(self, choices=('FFT+Welch', 'AR Model'),
            value='FFT+Welch', style=wx.CB_READONLY)
        self.methodControlBox.Add(self.methodComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_COMBOBOX, self.setMethod, self.methodComboBox)

        self.methodConfigSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.welchPanel = WelchConfigPanel(self, pg=self.pg)
        self.methodConfigSizer.Add(self.welchPanel, proportion=1, flag=wx.EXPAND)

        self.autoregPanel = AutoregConfigPanel(self, pg=self.pg)
        self.methodConfigSizer.Add(self.autoregPanel, proportion=1, flag=wx.EXPAND)

        self.methodConfigPanel = self.welchPanel

    def initLayout(self):
        controlSizer = wx.BoxSizer(orient=wx.VERTICAL)

        controlSizer.Add(self.filterControlBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        controlSizer.Add(self.widthControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        controlSizer.Add(self.decimateControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        controlSizer.Add(self.qualityControlBox, proportion=0, flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        controlSizer.Add(self.refreshControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        controlSizer.Add(self.methodControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT| wx.EXPAND, border=10)
        controlSizer.Add(self.methodConfigSizer, proportion=1, flag=wx.EXPAND)

        self.sizer.Add(controlSizer, proportion=0, flag=wx.EXPAND)

        # set sizer and figure layout
        self.initStandardLayout()

        self.FitInside()
        self.autoregPanel.Hide()
        self.FitInside()

    def setFilter(self, event=None):
        """Set filter status.
        """
        filterStatus = self.filterComboBox.GetValue().lower()

        if filterStatus == 'raw':
            self.pg.filter = False
        else:
            self.pg.filter = True

    def setWidth(self, event=None):
        """Set plot width in page.
        Divide slider value by 4 to get float.
        """
        self.pg.width = self.widthSlider.GetValue() / 4.0
        self.widthText.SetLabel('%5.2f(s)' % self.pg.width)

    def setDecimation(self, factor):
        """Set decimation factor in page.
        """
        self.pg.decimationFactor = factor

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

    def setRefresh(self, event=None):
        """Set refreshDelay in page.
        Multiply by a factor of 10 to get to ms scale.
        """
        refreshDelay = float(self.refreshSlider.GetValue()*10)
        self.pg.setRefreshDelay(refreshDelay)
        self.refreshText.SetLabel('%4d(ms)' % int(refreshDelay))

    def setMethod(self, event=None):
        method = self.methodComboBox.GetValue()
        self.pg.method = method

        self.methodConfigPanel.Hide()
        if method == 'FFT+Welch':
            self.methodConfigPanel = self.welchPanel
        elif method == 'AR Model':
            self.methodConfigPanel = self.autoregPanel
        else:
            raise RuntimeError('Unknown PSD method: ' + str(method))
        self.methodConfigPanel.Show()

        self.FitInside()

class Power(StandardMonitorPage):
    """Main class for a page that generates real-time power-spectral
    density plots of EEG.  Power is configurable and allows raster plots
    to be captured to file.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new Power page.

        Args:
            *args, **kwargs:  Arguments to pass to the Page base class.
        """
        self.initConfig()

        StandardMonitorPage.__init__(self, name='Power',
            configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlot()
        self.initLayout()

    def initConfig(self):
        """Initialize configuration values.
        """
        self.filter = True          # use raw or filtered signal
        self.decimationFactor = 1   # decimation factor, e.g., 2 will decimate to half sampRate

        self.width = 3.0            # width of window to use for computing PSD

        self.method = 'FFT+Welch'

        self.welchConfig = util.Holder(
            span=1.0    # width of spans/sub-windows used in Welch's method
        )

        self.autoregConfig = util.Holder(
            order=20,
            nFreq=150
        )

    def initPlot(self):
        """Initialize a new PowerPlot widget.
        """
        self.plot = widgets.PowerPlot(self)

    def initLayout(self):
        self.initStandardLayout()

        # plot pane
        plotPaneAuiInfo = aui.AuiPaneInfo().Name('plot').Caption('Power').CenterPane()
        self.auiManager.AddPane(self.plot, plotPaneAuiInfo)

        self.auiManager.Update()

    def updatePlot(self, event=None):
        """Draw the PSD plot.
        """
        # get EEG capture from current source
        cap = self.src.getEEGSecs(self.width, filter=self.filter, copy=False)

        # decimate EEG
        if self.decimationFactor > 1:
            cap.decimate(self.decimationFactor)

        if self.method == 'FFT+Welch':
            spectrum = cap.psd(method='welch', span=self.welchConfig.span)
        elif self.method == 'AR Model':
            spectrum = cap.psd(method='ar', order=self.autoregConfig.order,
                               freqs=self.autoregConfig.nFreq)
        else:
            raise RuntimeError('Unknown PSD estimation method: ' + str(method))

        freqs, powers = spectrum.getFreqsPowers()

        # tell plot widget to draw
        self.plot.draw(freqs, powers, chanNames=cap.getChanNames(), wxYield=False)

    def captureImage(self, event=None):
        self.plot.saveFile()
