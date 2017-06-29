import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
from matplotlib.colors import LogNorm as pltLogNorm
from matplotlib.colors import Normalize as pltLinNorm
import numpy as np
import os
import time
import wx
from wx.lib.agw import aui

from cebl import sig
from cebl import util
from cebl.rt import widgets

from .standard import StandardConfigPanel, StandardMonitorPage


class FourierConfigPanel(wx.Panel):
    def __init__(self, parent, pg, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.pg = pg

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.initControls()
        self.Layout()

    def initControls(self):
        hello = wx.StaticText(self, label='Hello FFT!')
        self.sizer.Add(hello)

class WaveletConfigPanel(wx.Panel):
    def __init__(self, parent, pg, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.pg = pg

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.initControls()
        self.Layout()

    def initControls(self):
        nFreqControlBox = widgets.ControlBox(self, label='Num Freqs', orient=wx.HORIZONTAL)
        self.nFreqText = wx.StaticText(self, label='%3d' % self.pg.waveletConfig.nFreq)
        nFreqTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        nFreqTextSizer.Add(self.nFreqText, proportion=1, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.nFreqSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=self.pg.waveletConfig.nFreq/5, minValue=1, maxValue=100)
        nFreqControlBox.Add(nFreqTextSizer, proportion=0, flag=wx.TOP, border=10)
        nFreqControlBox.Add(self.nFreqSlider, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setNFreq, self.nFreqSlider)

        self.sizer.Add(nFreqControlBox, proportion=0, flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        spanControlBox = widgets.ControlBox(self, label='Span', orient=wx.HORIZONTAL)
        self.spanText = wx.StaticText(self, label='%3d' % self.pg.waveletConfig.span)
        spanTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        spanTextSizer.Add(self.spanText, proportion=1, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.spanSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=self.pg.waveletConfig.span, minValue=1, maxValue=50)
        spanControlBox.Add(spanTextSizer, proportion=0, flag=wx.TOP, border=10)
        spanControlBox.Add(self.spanSlider, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setSpan, self.spanSlider)

        self.sizer.Add(spanControlBox, proportion=0, flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setSpan(self, event=None):
        self.pg.waveletConfig.span = self.spanSlider.GetValue()
        self.spanText.SetLabel('%3d' % self.pg.waveletConfig.span)
        self.pg.needsFirstPlot = True

    def setNFreq(self, event=None):
        self.pg.waveletConfig.nFreq = self.nFreqSlider.GetValue() * 5
        self.nFreqText.SetLabel('%3d' % self.pg.waveletConfig.nFreq)
        self.pg.needsFirstPlot = True

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
        self.filterControlBox.Add(self.filterComboBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_COMBOBOX, self.setFilter, self.filterComboBox)

        # combobox for selecting the displayed channel
        self.chanControlBox = widgets.ControlBox(self, label='Channel', orient=wx.VERTICAL)
        self.chanComboBox = wx.ComboBox(self, choices=[''], value='', style=wx.CB_READONLY)
        self.updateChannels()
        self.chanControlBox.Add(self.chanComboBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_COMBOBOX, self.setChannel, self.chanComboBox)

        # slider for controlling width of window
        # since sliders are int, we use divide by 4 to get float value
        self.widthControlBox = widgets.ControlBox(self, label='Width', orient=wx.HORIZONTAL)
        self.widthText = wx.StaticText(self, label='%5.2f(s)' % self.pg.width)
        widthTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        widthTextSizer.Add(self.widthText, proportion=1, flag=wx.EXPAND)
        self.widthSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.width*4.0), minValue=12, maxValue=120)
        self.widthControlBox.Add(widthTextSizer, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)
        self.widthControlBox.Add(self.widthSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setWidth, self.widthSlider)

        # scaleControlBox contains slider and radio buttons
        self.scaleControlBox = widgets.ControlBox(self, label='Scale', orient=wx.VERTICAL)

        # slider for adjusting the scale of power density in the plot
        scaleSliderSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.scaleText = wx.StaticText(self, label='10^%4.2f' % int(self.pg.scale), style=wx.ALIGN_CENTER)
        scaleTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        scaleTextSizer.Add(self.scaleText, proportion=1, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.scaleSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.scale*4), minValue=-16, maxValue=40)
        scaleSliderSizer.Add(scaleTextSizer, proportion=0, flag=wx.TOP, border=8)
        scaleSliderSizer.Add(self.scaleSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=8)
        self.Bind(wx.EVT_SLIDER, self.setScale, self.scaleSlider)
        self.scaleControlBox.Add(scaleSliderSizer, proportion=1, flag=wx.ALL | wx.EXPAND)

        # radio buttons for selecting log or linear scale
        scaleRBtnSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        linScaleRbtn = wx.RadioButton(self, label='Linear', style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, self.setLinScale, linScaleRbtn)
        scaleRBtnSizer.Add(linScaleRbtn, proportion=0, flag=wx.LEFT | wx.BOTTOM | wx.RIGHT, border=8)

        logScaleRbtn = wx.RadioButton(self, label='Log10')
        logScaleRbtn.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setLogScale, logScaleRbtn)
        scaleRBtnSizer.Add(logScaleRbtn, proportion=0, flag=wx.LEFT | wx.BOTTOM | wx.RIGHT, border=8)

        self.scaleControlBox.Add(scaleRBtnSizer, proportion=0, flag=wx.ALL | wx.ALIGN_CENTER_HORIZONTAL)

        # radio buttons for selection decimation factor
        self.decimateControlBox = widgets.ControlBox(self, label='Decimation', orient=wx.HORIZONTAL)

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

        for rbtn in rbtns[:-1]:
            self.decimateControlBox.Add(rbtn, proportion=0, flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)
        self.decimateControlBox.Add(rbtns[-1], proportion=0, flag=wx.ALL, border=10)

        # radio buttons for selecting plot interpolation
        self.interpControlBox = widgets.ControlBox(self, label='Interpolation', orient=wx.HORIZONTAL)

        # no interpolation radio button
        # first button determines group using wx.RB_GROUP
        noInterpRbtn = wx.RadioButton(self, label='None', style=wx.RB_GROUP)
        noInterpRbtn.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.setNoInterp, noInterpRbtn)
        self.interpControlBox.Add(noInterpRbtn, proportion=0,
                flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)

        # bicubic interpolation radio button
        nearInterpRbtn = wx.RadioButton(self, label='Bicubic')
        self.Bind(wx.EVT_RADIOBUTTON, self.setBicubicInterp, nearInterpRbtn)
        self.interpControlBox.Add(nearInterpRbtn, proportion=0,
                flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)

        # lanczos interpolation radio button
        lanczosInterpRbtn = wx.RadioButton(self, label='Lanczos')
        self.Bind(wx.EVT_RADIOBUTTON, self.setLanczosInterp, lanczosInterpRbtn)
        self.interpControlBox.Add(lanczosInterpRbtn, proportion=0,
                flag=wx.ALL, border=10)

        # slider for selecting delay between consecutive refreshes of the plot
        self.refreshControlBox = widgets.ControlBox(self, label='Refresh', orient=wx.HORIZONTAL)
        self.refreshText = wx.StaticText(self, label='%4d(ms)' % int(self.pg.getRefreshDelay()), style=wx.ALIGN_CENTER)
        refreshTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        refreshTextSizer.Add(self.refreshText, proportion=1, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=8)
        self.refreshSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.getRefreshDelay()/10), minValue=1, maxValue=200)
        self.refreshControlBox.Add(refreshTextSizer, proportion=0, flag=wx.TOP, border=10)
        self.refreshControlBox.Add(self.refreshSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SLIDER, self.setRefresh, self.refreshSlider)

        # combobox for selecting PSD estimation method
        self.methodControlBox = widgets.ControlBox(self, label='Method', orient=wx.VERTICAL)
        self.methodComboBox = wx.ComboBox(self, choices=('Wavelet', 'Fourier'),
            value='Wavelet', style=wx.CB_READONLY)
        self.methodControlBox.Add(self.methodComboBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_COMBOBOX, self.setMethod, self.methodComboBox)

        self.methodConfigSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.waveletPanel = WaveletConfigPanel(self, pg=self.pg)
        self.methodConfigSizer.Add(self.waveletPanel, proportion=1, flag=wx.EXPAND)

        self.fourierPanel = FourierConfigPanel(self, pg=self.pg)
        self.methodConfigSizer.Add(self.fourierPanel, proportion=1, flag=wx.EXPAND)

        self.methodConfigPanel = self.waveletPanel

    def initLayout(self):
        controlSizer = wx.BoxSizer(orient=wx.VERTICAL)

        controlSizer.Add(self.filterControlBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)
        controlSizer.Add(self.chanControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        controlSizer.Add(self.widthControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        controlSizer.Add(self.scaleControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        controlSizer.Add(self.decimateControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        controlSizer.Add(self.interpControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        controlSizer.Add(self.refreshControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        controlSizer.Add(self.methodControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT| wx.EXPAND, border=10)
        controlSizer.Add(self.methodConfigSizer, proportion=1, flag=wx.EXPAND)

        self.sizer.Add(controlSizer, proportion=0, flag=wx.EXPAND)

        # set sizer and figure layout
        self.initStandardLayout()

        self.FitInside()
        self.fourierPanel.Hide()
        self.FitInside()

    def setFilter(self, event=None):
        """Set filter status.
        """
        filterStatus = self.filterComboBox.GetValue().lower()

        if filterStatus == 'raw':
            self.pg.filter = False
        else:
            self.pg.filter = True

        self.updateChannels()

    def updateChannels(self, event=None):
        """Update list of currently available channels.
        """
        # needs fixed to handle filter chain XXX - idfah

        chanNames = self.pg.src.getChanNames()

        self.chanComboBox.Clear()
        self.chanComboBox.AppendItems(chanNames)

        self.chanComboBox.SetValue(chanNames[0])
        self.pg.chanIndex = 0

    def setChannel(self, event=None):
        self.pg.chanIndex = self.pg.src.getChanNames().index(self.chanComboBox.GetValue())

    def setWidth(self, event=None):
        """Set plot width in page.
        Divide slider value by 4 to get float.
        """
        self.pg.width = self.widthSlider.GetValue() / 4.0
        self.widthText.SetLabel('%5.2f(s)' % self.pg.width)
        self.pg.canvas.draw()

    def setScale(self, event=None):
        """Set scale in page.
        """
        self.pg.scale = self.scaleSlider.GetValue() / 4.0
        self.scaleText.SetLabel('10^%4.2f' % self.pg.scale)
        self.pg.needsFirstPlot = True

    def setLinScale(self, event=None):
        self.pg.normScale = 'linear'
        self.pg.needsFirstPlot = True

    def setLogScale(self, event=None):
        self.pg.normScale = 'log'
        self.pg.needsFirstPlot = True

    def setDecimation(self, factor):
        """Set decimation factor in page.
        """
        self.pg.decimationFactor = factor
        self.pg.needsFirstPlot = True

    def setNoInterp(self, event=None):
        """Set plot to use no interpolation.
        """
        self.pg.interpolation = 'none'
        self.pg.needsFirstPlot = True

    def setBicubicInterp(self, event=None):
        """Set plot to use bicubic interpolation.
        """
        self.pg.interpolation = 'bicubic'
        self.pg.needsFirstPlot = True

    def setLanczosInterp(self, event=None):
        """Set plot to use lanczos interpolation.
        """
        self.pg.interpolation = 'lanczos'
        self.pg.needsFirstPlot = True

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
        if method == 'Wavelet':
            self.methodConfigPanel = self.waveletPanel
        elif method == 'Fourier':
            self.methodConfigPanel = self.fourierPanel
        else:
            raise Exception('Unknown method: ' + str(method))
        self.methodConfigPanel.Show()

        self.FitInside()

class Spectrogram(StandardMonitorPage):
    """Main class for a page that generates real-time spectrogram plots of EEG.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new Spectrogram page.

        Args:
            *args, **kwargs:  Arguments to pass to the Page base class.
        """
        self.initConfig()

        # initialize Page base class
        StandardMonitorPage.__init__(self, name='Spectrogram',
            configPanelClass=ConfigPanel, *args, **kwargs)

        self.initCanvas()
        self.initLayout()

    def initConfig(self):
        self.filter = True          # use raw or filtered signal
        self.chanIndex = 0          # index of channel to show
        self.width = 5.0            # width of window to use for computing PSD

        self.decimationFactor = 1   # decimation factor, e.g., 2 will decimate to half sampRate

        self.interpolation = 'none'

        self.normScale = 'log'
        self.scale = -2

        self.method = 'Wavelet'

        self.setRefreshDelay(200)

        self.waveletConfig = util.Holder(
            nFreq = 100,
            span = 10
        )

        self.fourierConfig = util.Holder()

    def initCanvas(self):
        """Initialize a new matplotlib canvas, figure and axis.
        """
        self.plotPanel = wx.Panel(self)
        self.plotPanel.SetBackgroundColour('white')
        plotSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.plotPanel.SetSizer(plotSizer)

        self.fig = plt.Figure(facecolor='white')
        #self.canvas = FigureCanvas(parent=self, id=wx.ID_ANY, figure=self.fig)
        self.canvas = FigureCanvas(parent=self.plotPanel, id=wx.ID_ANY, figure=self.fig)

        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        self.cbAx = self.fig.add_axes([0.91, 0.05, 0.03, 0.93])

        #self.fig.subplots_adjust(hspace=0.0, wspace=0.0,
        #    left=0.035, right=0.92, top=0.98, bottom=0.05)

        self.adjustMargins()

        self.firstPlot()

        self.lastSize = (0,0)
        self.needsResizePlot = True
        self.canvas.Bind(wx.EVT_SIZE, self.resizePlot)
        self.canvas.Bind(wx.EVT_IDLE, self.idleResizePlot)

        ##self.plotToolbar = widgets.PyPlotNavbar(self.canvas)
        ##plotSizer.Add(self.plotToolbar, proportion=0, flag=wx.EXPAND)
        plotSizer.Add(self.canvas, proportion=1, flag=wx.EXPAND)

        #self.plotToolbar.Hide()

    def initLayout(self):
        self.initStandardLayout()

        plotPaneAuiInfo = aui.AuiPaneInfo().Name('canvas').Caption('Spectrogram').CenterPane()
        #self.auiManager.AddPane(self.canvas, plotPaneAuiInfo)
        self.auiManager.AddPane(self.plotPanel, plotPaneAuiInfo)

        self.auiManager.Update()

        self.canvas.Hide()

    def afterUpdateSource(self):
        self.configPanel.updateChannels()

    def afterStart(self):
        # make sure canvas is visible
        self.canvas.Show()
        self.plotPanel.Layout()

        # trigger initial plot update
        self.needsFirstPlot = True

    def getCap(self):
        cap = self.src.getEEGSecs(self.width, filter=self.filter, copy=False)
        if self.decimationFactor > 1:
            cap.decimate(self.decimationFactor)

        return cap

    def getSpectrum(self, cap):
        # configurable XXX - idfah
        data = cap.data[:,self.chanIndex] * sig.windows.tukey(cap.data.shape[0]) # tukey or hann? XXX - idfah

        freqs, powers, phases = self.cwt.apply(data)

        # configurable XXX - idfah
        powers = np.clip(powers, 1.0e-10, np.inf)

        return freqs, powers

    def firstPlot(self, event=None):
        cap = self.getCap()

        self.cwt = sig.CWT(sampRate=cap.getSampRate(),
                           freqs=self.waveletConfig.nFreq,
                           span=self.waveletConfig.span)

        if self.isRunning():
            freqs, powers = self.getSpectrum(cap)
        else:
            freqs = np.arange(1,self.src.getSampRate()//2+1)
            powers = np.zeros((128,10,1))
            powers[0,0,0] = 1.0

        self.ax.cla()
        self.cbAx.cla()

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        self.wimg = self.ax.imshow(powers[:,:,0].T, interpolation=self.interpolation,
            origin='lower', aspect='auto', norm=self.getNorm(),
            extent=self.getExtent(cap, freqs),
            cmap=plt.cm.get_cmap('jet'), animated=True)

        self.cbar = self.fig.colorbar(self.wimg, cax=self.cbAx)
        self.cbar.set_label(r'Power Density ($V^2 / Hz$)')

        #self.updateNorm(powers)

        self.canvas.draw()

        #self.background = self.canvas.copy_from_bbox(self.fig.bbox)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.needsFirstPlot = False

    def adjustMargins(self):
        self.fig.subplots_adjust(hspace=0.0, wspace=0.0,
            left=0.045, right=0.90, top=0.98, bottom=0.07)

    def resizePlot(self, event):
        # prevents handling extra resize events, hack XXX - idfah
        size = self.canvas.GetSize()
        if self.lastSize == size:
            return
        else:
            self.lastSize = size

        # this is all a hack to do resizing on idle when page is not running
        # should this be a custom FigureCanvas derived widget? XXX - idfah
        if self.isRunning():
            # when running, just do event.Skip() this will
            # call canvas._onSize since it is second handler
            self.needsResizePlot = False
            event.Skip()
        else:
            # flag to resize on next idle event
            self.needsResizePlot = True

    def idleResizePlot(self, event):
        # if not running and flagged for resize
        if not self.isRunning() and self.needsResizePlot:
            ##self.adjustMargins()
            self.needsResizePlot = False
            # call canvas resize method manually
            # hack alert, we just pass None as event
            # since it's not used anyway
            self.canvas._onSize(None)

    def getExtent(self, cap, freqs):
        return (0.0, cap.getNObs()/float(cap.getSampRate()),
                np.min(freqs), np.max(freqs))

    def getNorm(self):
        mx = 10**self.scale

        if self.normScale == 'linear':
            mn = 0.0
            norm = pltLinNorm(mn,mx)

        elif self.normScale == 'log':
            mn = 1e-10
            norm = pltLogNorm(mn,mx)

        else:
            raise Exception('Invalid norm %s.' % norm)

        return norm

    def updatePlot(self, event=None):
        """Draw the spectrogram plot.
        """
        if self.needsFirstPlot:
            self.firstPlot()

        else:
            cap = self.getCap()
            freqs, powers = self.getSpectrum(cap)

            #self.updateNorm(powers)

            self.canvas.restore_region(self.background)
            self.wimg.set_array(powers[:,:,0].T)
            self.wimg.set_extent(self.getExtent(cap, freqs))
            self.ax.draw_artist(self.wimg)

            ##self.cbAx.draw_artist(self.cbar.patch)
            ##self.cbAx.draw_artist(self.cbar.solids)

            #self.cbar.draw_all()
            #self.canvas.blit(self.cbAx.bbox)

            #self.canvas.blit(self.fig.bbox)
            self.canvas.blit(self.ax.bbox)

            # for debugging, redraws everything
            ##self.canvas.draw()

    def captureImage(self, event=None):
        ## Parts borrowed from backends_wx.py from matplotlib
        # Fetch the required filename and file type.
        filetypes, exts, filter_index = self.canvas._get_imagesave_wildcards()
        default_file = self.canvas.get_default_filename()
        dlg = wx.FileDialog(self, "Save to file", "", default_file,
                            filetypes,
                            wx.SAVE|wx.OVERWRITE_PROMPT)
        dlg.SetFilterIndex(filter_index)
        if dlg.ShowModal() == wx.ID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            format = exts[dlg.GetFilterIndex()]
            basename, ext = os.path.splitext(filename)
            if ext.startswith('.'):
                ext = ext[1:]
            if ext in ('svg', 'pdf', 'ps', 'eps', 'png') and format!=ext:
                #looks like they forgot to set the image type drop
                #down, going with the extension.
                format = ext
            self.canvas.print_figure(
                os.path.join(dirname, filename), format=format)
