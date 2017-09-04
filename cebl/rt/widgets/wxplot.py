import numpy as np
import wx
#import wx.lib.plot as wxplt

from . import wxlibplot as wxplt # cython rebuild

from cebl import util


class wxPlotCanvas(wxplt.PlotCanvas):
    def __init__(self, *args, **kwargs):
        wxplt.PlotCanvas.__init__(self, *args, **kwargs)

        self.lastSize = (0,0)
        self.Bind(wx.EVT_SIZE, self.resizeHack)

    def resizeHack(self, event):
        # hack alert, to prevent multiple consecutive resize events XXX - idfah
        # should this be reported as a wx bug? XXX - idfah
        size = self.GetSize()
        if size != self.lastSize:
            self.lastSize = size
            event.Skip()

class wxPlot(wx.Panel):
    def __init__(self, parent, title, xLabel, yLabel, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel

        self.initCanvas()
        self.initLayout()
        self.initCanvasSettings()

    def initCanvas(self):
        self.canvas = wxPlotCanvas(self)

    def initCanvasSettings(self):
        self.canvas.SetDoubleBuffered(True)
        self.canvas.SetFontSizeAxis(12)
        self.setMediumQuality()

    def initLayout(self):
        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.sizer.Add(self.canvas, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Layout()

    def draw(self):
        raise NotImplementedError('draw not implemented.')

    def saveFile(self, event=None, fileName=''):
        return self.canvas.SaveFile(fileName=fileName)

    def setHighQuality(self):
        self.canvas.SetEnableHiRes(True)
        self.canvas.SetEnableAntiAliasing(True)

    def setMediumQuality(self):
        self.canvas.SetEnableHiRes(False)
        self.canvas.SetEnableAntiAliasing(True)

    def setLowQuality(self):
        self.canvas.SetEnableHiRes(False)
        self.canvas.SetEnableAntiAliasing(False)


class TracePlotCanvas(wxPlotCanvas):
    def __init__(self, *args, **kwargs):
        wxPlotCanvas.__init__(self, *args, **kwargs)

        self.chanNames = None
        self.scale = 0.0

    def _yticks(self, *args):
        if self.chanNames is None:
            return wxplt.PlotCanvas._yticks(self, *args)

        locs = -np.arange(len(self.chanNames)) * self.scale
        return zip(locs, self.chanNames)

    def setChanNames(self, chanNames, scale):
        self.chanNames = chanNames
        self.scale = scale

class TracePlot(wxPlot):
    def __init__(self, parent, title='',
                 xLabel='Time (s)', yLabel='Channel',
                 *args, **kwargs):
        wxPlot.__init__(self, parent=parent,
                        title=title, xLabel=xLabel, yLabel=yLabel,
                        *args, **kwargs)

    def initCanvas(self):
        self.canvas = TracePlotCanvas(self)

    def draw(self, data, t=None, scale=None, chanNames=None,
             colors=('black', 'red', 'violet', 'blue', 'green', 'orange'),
             #colors=('black', 'blue', 'green', 'red', 'turquoise', 'blue violet', 'maroon', 'orange'),
             wxYield=False):

        data = util.colmat(data)
        nObs, nChan = data.shape

        if t is None:
            t = np.arange(nObs)
        else:
            t = np.linspace(0,t,nObs)

        colsep = util.colsep(data, scale=scale)
        scale = colsep[1]

        yMin = scale * (-nChan + 0.5)
        yMax = scale * 0.5

        data = data - colsep

        if chanNames is None:
            chanNames = (None,) * nObs

        self.canvas.setChanNames(chanNames, scale)

        colors = util.cycle(colors, nChan)

        if wxYield:
            wx.Yield()
        data = data.T
        lines = [wxplt.PolyLine(zip(t,d), legend=chan, colour=col, width=2)
            for d,col,chan in zip(data, colors, chanNames)]
        gc = wxplt.PlotGraphics(lines, title=self.title,
            xLabel=self.xLabel, yLabel=self.yLabel)

        if wxYield:
            wx.Yield()
        self.canvas.Draw(gc,
            xAxis=(np.min(t),np.max(t)),
            yAxis=(yMin,yMax))


class PowerPlot(wxPlot):
    def __init__(self, parent, title='',
                 xLabel='Frequency (Hz)', yLabel='Power (uV^2/Hz)',
                 *args, **kwargs):
        wxPlot.__init__(self, parent=parent,
                        title=title, xLabel=xLabel, yLabel=yLabel,
                        *args, **kwargs)

    def initCanvasSettings(self):
        wxPlot.initCanvasSettings(self)
        self.canvas.setLogScale((False,True))
        self.canvas.SetEnableLegend(True)
        self.canvas.SetFontSizeLegend(12)

    def draw(self, freqs, powers, chanNames=None,
             colors = ('black', 'red', 'violet', 'blue', 'green', 'orange'),
             wxYield=False):

        if chanNames is None:
            chanNames = (None,) * powers.shape[0]

        colors = util.cycle(colors, powers.shape[1])

        powers = util.colmat(powers)
        powers = powers.T

        # cap so we don't break wxplt.PlotGraphics with inf
        # Note: we need to use finfo.max/10.0 since
        #   wxplt.PlotGraphics does some log10 processing
        #   before figuring tick marks
        finfo = np.finfo(powers.dtype)
        powers[powers < finfo.eps] = finfo.eps
        powers[powers > (finfo.max/10.0)] = (finfo.max/10.0)


        if wxYield:
            wx.Yield()
        lines = [wxplt.PolyLine(zip(freqs,p), legend=chan, colour=col, width=2)
            for p,col,chan in zip(powers, colors, chanNames)]

        #lines += [wxplt.PolyLine(( (60.0,np.min(powers)), (60.0,np.max(powers)) ), legend='60Hz', colour='black', width=1)]

        if wxYield:
            wx.Yield()
        gc = wxplt.PlotGraphics(lines, title=self.title,
            xLabel=self.xLabel, yLabel=self.yLabel)

        self.canvas.Draw(gc,
            xAxis=(freqs[0], freqs[-1]),
            yAxis=(np.min(powers), np.max(powers)))


class BMUPlot(wxPlot):
    def __init__(self, parent, title='', xLabel='', yLabel='', *args, **kwargs):
        wxPlot.__init__(self, parent=parent,
                title=title, xLabel=xLabel, yLabel=yLabel, *args, **kwargs)

    def draw(self, data, latticeSize, color='black', wxYield=False):
        if wxYield:
            wx.Yield()

        data = np.asarray(data)
        data = data + 1

        points = (wxplt.PolyMarker(data, colour=color),)
        gc = wxplt.PlotGraphics(points, title=self.title,
            xLabel=self.xLabel, yLabel=self.yLabel)

        if wxYield:
            wx.Yield()

        self.canvas.Draw(gc,
            xAxis=(0,latticeSize[0]+1),
            yAxis=(0,latticeSize[1]+1))
