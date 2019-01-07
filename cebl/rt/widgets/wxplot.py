import time as _time
import numpy as np
import wx

#import wx.lib.plot as wxplt

from . import wxlibplot as wxplt # cython rebuild

from cebl import util


class wxPlotCanvas(wxplt.PlotCanvas):
    def __init__(self, *args, **kwargs):
        wxplt.PlotCanvas.__init__(self, *args, **kwargs)

        self.lastSize = (0,0)
        ##self.Bind(wx.EVT_SIZE, self.resizeHack)

    ##def resizeHack(self, event):
    ##    # hack alert, to prevent multiple consecutive resize events XXX - idfah
    ##    # should this be reported as a wx bug? XXX - idfah
    ##    size = self.GetSize()
    ##    if size != self.lastSize:
    ##        self.lastSize = size
    ##        event.Skip()

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
        self.canvas.fontSizeAxis = 12
        self.canvas.enableGrid = False
        self.setMediumQuality()

    def initLayout(self, event=None):
        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.sizer.Add(self.canvas, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Layout()

    def draw(self):
        raise NotImplementedError('draw not implemented.')

    def saveFile(self, event=None, fileName=''):
        return self.canvas.SaveFile(fileName=fileName)

    def setHighQuality(self):
        self.canvas.enableHiRes = True
        self.canvas.enableAntiAliasing = True

    def setMediumQuality(self):
        self.canvas.enableHiRes = False
        self.canvas.enableAntiAliasing = True

    def setLowQuality(self):
        self.canvas.enableHiRes = False
        self.canvas.enableAntiAliasing = False


class TracePlotCanvas(wxPlotCanvas):
    def __init__(self, *args, **kwargs):
        wxPlotCanvas.__init__(self, *args, **kwargs)

        self.chanNames = None
        self.scale = 0.0

    def _yticks(self, *args):
        if self.chanNames is None:
            return wxplt.PlotCanvas._yticks(self, *args)

        locs = -np.arange(len(self.chanNames)) * self.scale
        return list(zip(locs, self.chanNames))

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

    def initCanvas(self, event=None):
        ##if not self.windowCreated:
        ##    self.windowCreated = True
        ##    self.canvas = TracePlotCanvas(self)
        ##    self.sizer.Add(self.canvas, proportion=1, flag=wx.EXPAND)
        ##    self.Layout()
        ##wx.CallAfter(self.initCanvasSettings)
        self.canvas = TracePlotCanvas(self)

    def draw(self, data, time=None, scale=None, chanNames=None,
             colors=('black', 'red', 'violet', 'blue', 'green', 'orange'),
             #colors=('black', 'blue', 'green', 'red', 'turquoise', 'blue violet', 'maroon', 'orange'),
             wxYield=False):

        data = util.colmat(data)
        nObs, nChan = data.shape

        if time is None:
            time = np.arange(nObs)
        else:
            time = np.linspace(0, time, nObs)

        colsep = util.colsep(data, scale=scale)
        scale = colsep[1]

        yMin = scale * (-nChan + 0.5)
        yMax = scale * 0.5

        data = data - colsep
        #time = time.repeat(data.shape[1]).reshape(data.shape)
        #comb = np.array((time,data)).swapaxes(1,2).swapaxes(0,1)

        if chanNames is None:
            chanNames = (None,) * nObs

        self.canvas.setChanNames(chanNames, scale)

        colors = util.cycle(colors, nChan)

        if wxYield:
            wx.Yield()

        # so many ways to slice this XXX - idfah
        #lines = [wxplt.PolyLine(pts.T, legend=chan, colour=col, width=2)
        #    for pts,col,chan in zip(comb, colors, chanNames)]
        #lines = [wxplt.PolyLine(zip(time,d), legend=chan, colour=col, width=2)
        #    for d,col,chan in zip(data.T, colors, chanNames)]
        #lines = [wxplt.PolyLine(np.vstack((time,d)).T, legend=chan, colour=col, width=2)
        #    for d,col,chan in zip(data.T, colors, chanNames)]
        #t = _time.time()
        lines = []
        for i in range(nChan):
            lines.append(wxplt.PolyLine(np.vstack((time,data[:,i])).T,
                legend=chanNames[i], colour=colors[i], width=2))
        #print("time making lines: ", _time.time()-t)

        #t = _time.time()
        graphics = wxplt.PlotGraphics(lines, title=self.title,
            xLabel=self.xLabel, yLabel=self.yLabel)
        #print("time making PlotGraphics: ", _time.time()-t)

        if wxYield:
            wx.Yield()
        #t = _time.time()
        self.canvas.Draw(graphics,
            xAxis=(np.min(time),np.max(time)),
            yAxis=(yMin,yMax))
        #print("time drawing at top level: ", _time.time()-t)


class PowerPlot(wxPlot):
    def __init__(self, parent, title='',
                 xLabel='Frequency (Hz)', yLabel='Power (uV^2/Hz)',
                 *args, **kwargs):
        wxPlot.__init__(self, parent=parent,
                        title=title, xLabel=xLabel, yLabel=yLabel,
                        *args, **kwargs)

    def initCanvasSettings(self, event=None):
        wxPlot.initCanvasSettings(self)
        self.canvas.logScale = (False,True)
        self.canvas.enableLegend = True
        self.canvas.fontSizeLegend = 12

    def draw(self, freqs, powers, chanNames=None,
             colors = ('black', 'red', 'violet', 'blue', 'green', 'orange'),
             wxYield=False):

        if chanNames is None:
            chanNames = (None,) * powers.shape[0]

        colors = util.cycle(colors, powers.shape[1])

        powers = util.colmat(powers)

        # cap so we don't break wxplt.PlotGraphics with inf
        # Note: we need to use finfo.max/10.0 since
        #   wxplt.PlotGraphics does some log10 processing
        #   before figuring tick marks
        finfo = np.finfo(powers.dtype)
        powers[powers < finfo.eps] = finfo.eps
        powers[powers > (finfo.max/10.0)] = (finfo.max/10.0)

        if wxYield:
            wx.Yield()
        lines = [wxplt.PolyLine(list(zip(freqs,p)), legend=chan, colour=col, width=2)
            for p,col,chan in zip(powers.T, colors, chanNames)]

        #lines += [wxplt.PolyLine(( (60.0,np.min(powers)), (60.0,np.max(powers)) ), legend='60Hz', colour='black', width=1)]

        if wxYield:
            wx.Yield()
        graphics = wxplt.PlotGraphics(lines, title=self.title,
            xLabel=self.xLabel, yLabel=self.yLabel)

        self.canvas.Draw(graphics,
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
        graphics = wxplt.PlotGraphics(points, title=self.title,
            xLabel=self.xLabel, yLabel=self.yLabel)

        if wxYield:
            wx.Yield()

        self.canvas.Draw(graphics,
            xAxis=(0,latticeSize[0]+1),
            yAxis=(0,latticeSize[1]+1))
