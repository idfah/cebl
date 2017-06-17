import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import os
import time
import wx
from wx.lib.agw import aui
import wx.lib.agw.floatspin as agwfs

from cebl import eeg
from cebl import ml
from cebl.rt import widgets
from cebl.rt.widgets import grid

from standard import StandardConfigPanel, StandardBCIPage

class ConfigPanel(StandardConfigPanel):
    """Panel containing configuration widgets.  This is intimately
    related to this specific page.  Extends wx.Panel.
    """
    def __init__(self, parent, pg, *args, **kwargs):
        """Construct a new panel containing configuration widgets.

        Args:
            parent: Parent in wx hierarchy.

            pg:     Page to be configured.

            *args, **kwargs:  Additional arguments passed
                              to the wx.Panel base class.
        """
        StandardConfigPanel.__init__(self, parent=parent, pg=pg, *args, **kwargs)

        self.initCopy()
        self.initNTrial()
        self.initIntervals()
        self.initSegWindow()
        self.initColors()
        self.initGridLayout()
        self.initClassifier()
        self.initStandardLayout()

    def initCopy(self):
        copySizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        copyControlBox = widgets.ControlBox(self, label='Copy Text', orient=wx.VERTICAL)
        self.copyTextCtrl = wx.TextCtrl(self)
        self.Bind(wx.EVT_TEXT, self.setCopyText, self.copyTextCtrl)         
        self.offlineControls += [self.copyTextCtrl]
        copyControlBox.Add(self.copyTextCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        copySizer.Add(copyControlBox, proportion=1,
                flag =wx.ALL | wx.EXPAND, border=10)
      
        self.sizer.Add(copySizer, proportion=0, flag=wx.EXPAND)

    def setCopyText(self, event):
        copyText = self.copyTextCtrl.GetLineText(0)
        self.pg.testText = copyText
        self.pg.gridSpeller.setCopyText(copyText)
        if len(copyText) == 0:
            self.pg.freeSpelling = True
        else:
            self.pg.freeSpelling = False

    def initNTrial(self):
        trialSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        
        trialControlBox = widgets.ControlBox(self, label='Num Trials', orient=wx.VERTICAL)
        self.trialSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.nTrials), min=1, max=100)
        self.Bind(wx.EVT_SPINCTRL, self.setNTrial, self.trialSpinCtrl)
        self.offlineControls += [self.trialSpinCtrl]
        trialControlBox.Add(self.trialSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        trialSizer.Add(trialControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)

        pauseControlBox = widgets.ControlBox(self,
                label='Pause Secs', orient=wx.VERTICAL)

        self.pauseFloatSpin = agwfs.FloatSpin(self, min_val=0.5, max_val=5.0,
                increment=0.5, value=self.pg.pause)
        self.pauseFloatSpin.SetFormat("%f")
        self.pauseFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setPause, self.pauseFloatSpin)
        self.offlineControls += [self.pauseFloatSpin]
        pauseControlBox.Add(self.pauseFloatSpin, proportion=0,
            flag=wx.ALL | wx.EXPAND, border=12)

        trialSizer.Add(pauseControlBox, proportion=1,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(trialSizer, proportion=0, flag=wx.EXPAND)

    def setNTrial(self, event):
        self.pg.nTrials = self.trialSpinCtrl.GetValue()

    def setPause(self, event):
        self.pg.pause = self.pauseFloatSpin.GetValue()

    def initIntervals(self):
        intervalSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        siControlBox = widgets.ControlBox(self, label='Stim Secs', orient=wx.VERTICAL)

        self.siFloatSpin = agwfs.FloatSpin(self, min_val=0.025, max_val=0.5,
                increment=1/40.0, value=self.pg.si)
        self.siFloatSpin.SetFormat("%f")
        self.siFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setSI, self.siFloatSpin)
        self.offlineControls += [self.siFloatSpin]
        siControlBox.Add(self.siFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        intervalSizer.Add(siControlBox, proportion=1,
                 flag=wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)

        isiControlBox = widgets.ControlBox(self, label='Inter-Stim Secs', orient=wx.VERTICAL)
        self.isiFloatSpin = agwfs.FloatSpin(self, min_val=0.05, max_val=1.0,
                increment=1/40.0, value=self.pg.isi)
        self.isiFloatSpin.SetFormat("%f")
        self.isiFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setISI, self.isiFloatSpin)
        self.offlineControls += [self.isiFloatSpin]
        isiControlBox.Add(self.isiFloatSpin, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)

        intervalSizer.Add(isiControlBox, proportion=1,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(intervalSizer, proportion=0, flag=wx.EXPAND)

    def setSI(self, event):
        self.pg.si = self.siFloatSpin.GetValue()
        self.pg.setTrained(False)

    def setISI(self, event):
        self.pg.isi = self.isiFloatSpin.GetValue()
        self.pg.setTrained(False)

    def initSegWindow(self):
        windowSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        windowStartControlBox = widgets.ControlBox(self, label='Window Start', orient=wx.VERTICAL)

        self.windowStartFloatSpin = agwfs.FloatSpin(self, min_val=0.0, max_val=0.25,
                increment=1/40.0, value=self.pg.windowStart)
        self.windowStartFloatSpin.SetFormat("%f")
        self.windowStartFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setWindowStart, self.windowStartFloatSpin)
        self.offlineControls += [self.windowStartFloatSpin]
        windowStartControlBox.Add(self.windowStartFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        windowSizer.Add(windowStartControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)

        windowEndControlBox = widgets.ControlBox(self, label='Window End', orient=wx.VERTICAL)
        self.windowEndFloatSpin = agwfs.FloatSpin(self, min_val=0.3, max_val=1.5,
                increment=1/20.0, value=self.pg.windowEnd)
        self.windowEndFloatSpin.SetFormat("%f")
        self.windowEndFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setWindowEnd, self.windowEndFloatSpin)
        self.offlineControls += [self.windowEndFloatSpin]
        windowEndControlBox.Add(self.windowEndFloatSpin, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)

        windowSizer.Add(windowEndControlBox, proportion=1,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(windowSizer, proportion=0, flag=wx.EXPAND)

    def setWindowStart(self, event):
        self.pg.windowStart = self.windowStartFloatSpin.GetValue()
        self.pg.requireRetrain()

    def setWindowEnd(self, event):
        self.pg.windowEnd = self.windowEndFloatSpin.GetValue()
        self.pg.requireRetrain()

    def initColors(self):
        # first row
        colorSizer1 = wx.BoxSizer(orient=wx.HORIZONTAL)
        
        gridColorControlBox = widgets.ControlBox(self,
                label='Grid color', orient=wx.VERTICAL)
        self.gridColorCtrl = wx.ColourPickerCtrl(self)
        self.gridColorCtrl.SetColour(self.pg.gridSpeller.getGridColor())
        self.Bind(wx.EVT_COLOURPICKER_CHANGED, self.setGridColor, self.gridColorCtrl)
        self.offlineControls += [self.gridColorCtrl]
        gridColorControlBox.Add(self.gridColorCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        colorSizer1.Add(gridColorControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)

        backgroundColorControlBox = widgets.ControlBox(self,
                label='Background color', orient=wx.VERTICAL)
        self.backgroundColorCtrl = wx.ColourPickerCtrl(self)
        self.backgroundColorCtrl.SetColour(self.pg.gridSpeller.getBackground())
        self.Bind(wx.EVT_COLOURPICKER_CHANGED, self.setBackgroundColor, self.backgroundColorCtrl)
        self.offlineControls += [self.backgroundColorCtrl]
        backgroundColorControlBox.Add(self.backgroundColorCtrl, proportion=1,
                 flag=wx.ALL | wx.EXPAND, border=10)

        colorSizer1.Add(backgroundColorControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(colorSizer1, proportion=0, flag=wx.EXPAND)

        # second row
        colorSizer2 = wx.BoxSizer(orient=wx.HORIZONTAL)

        copyColorControlBox = widgets.ControlBox(self,
                label='Copy color', orient=wx.VERTICAL)
        self.copyColorCtrl = wx.ColourPickerCtrl(self)
        self.copyColorCtrl.SetColour(self.pg.gridSpeller.getCopyColor())
        self.Bind(wx.EVT_COLOURPICKER_CHANGED, self.setCopyColor, self.copyColorCtrl)
        self.offlineControls += [self.copyColorCtrl]
        copyColorControlBox.Add(self.copyColorCtrl, proportion=1,
                 flag=wx.ALL | wx.EXPAND, border=10)

        colorSizer2.Add(copyColorControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)

        feedColorControlBox = widgets.ControlBox(self,
                label='Feed color', orient=wx.VERTICAL)
        self.feedColorCtrl = wx.ColourPickerCtrl(self)
        self.feedColorCtrl.SetColour(self.pg.gridSpeller.getFeedColor())
        self.Bind(wx.EVT_COLOURPICKER_CHANGED, self.setFeedColor, self.feedColorCtrl)
        self.offlineControls += [self.feedColorCtrl]
        feedColorControlBox.Add(self.feedColorCtrl, proportion=1,
                 flag=wx.ALL | wx.EXPAND, border=10)

        colorSizer2.Add(feedColorControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(colorSizer2, proportion=0, flag=wx.EXPAND)

        # third row
        colorSizer3 = wx.BoxSizer(orient=wx.HORIZONTAL)

        highlightColorControlBox = widgets.ControlBox(self,
                label='Highlight color', orient=wx.VERTICAL)
        self.highlightColorCtrl = wx.ColourPickerCtrl(self)
        self.highlightColorCtrl.SetColour(self.pg.gridSpeller.getHighlightColor())
        self.Bind(wx.EVT_COLOURPICKER_CHANGED, self.setHighlightColor, self.highlightColorCtrl)
        self.offlineControls += [self.highlightColorCtrl]
        highlightColorControlBox.Add(self.highlightColorCtrl, proportion=1,
                 flag=wx.ALL | wx.EXPAND, border=10)

        colorSizer3.Add(highlightColorControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)

        unhighlightColorControlBox = widgets.ControlBox(self,
                label='Unhighlight color', orient=wx.VERTICAL)
        self.unhighlightColorCtrl = wx.ColourPickerCtrl(self)
        self.unhighlightColorCtrl.SetColour(self.pg.gridSpeller.getUnhighlightColor())
        self.Bind(wx.EVT_COLOURPICKER_CHANGED, self.setUnhighlightColor, self.unhighlightColorCtrl)
        self.offlineControls += [self.unhighlightColorCtrl]
        unhighlightColorControlBox.Add(self.unhighlightColorCtrl, proportion=1,
                 flag=wx.ALL | wx.EXPAND, border=10)

        colorSizer3.Add(unhighlightColorControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(colorSizer3, proportion=0, flag=wx.EXPAND)

    def setGridColor(self, event):
        self.pg.gridSpeller.setGridColor(self.gridColorCtrl.GetColour())

    def setBackgroundColor(self, event):
        self.pg.gridSpeller.setBackground(self.backgroundColorCtrl.GetColour())
 
    def setHighlightColor(self, event):
        self.pg.gridSpeller.setHighlightColor(self.highlightColorCtrl.GetColour())

    def setUnhighlightColor(self, event):
        self.pg.gridSpeller.setUnhighlightColor(self.unhighlightColorCtrl.GetColour())

    def setCopyColor(self, event):
        self.pg.gridSpeller.setCopyColor(self.copyColorCtrl.GetColour())

    def setFeedColor(self, event):
        self.pg.gridSpeller.setFeedColor(self.feedColorCtrl.GetColour())
    
    def initGridLayout(self):
        gridLayoutControlBox = widgets.ControlBox(self, label='Layout', orient=wx.VERTICAL)

        topSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        bottomSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.gridLayoutLowButton = wx.Button(self, label='Low')
        self.gridLayoutUppButton = wx.Button(self, label='Upp')
        self.gridLayoutNumButton = wx.Button(self, label='Num')
        self.gridLayoutEtcButton = wx.Button(self, label='Etc')
        self.gridLayoutSymButton = wx.Button(self, label='Sym')

        topSizer.Add(self.gridLayoutLowButton, proportion=0,
                flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)
        self.Bind(wx.EVT_BUTTON, self.setGridLayoutLow, self.gridLayoutLowButton)
        topSizer.Add(self.gridLayoutUppButton, proportion=0,
                flag=wx.TOP | wx.BOTTOM, border=10)
        self.Bind(wx.EVT_BUTTON, self.setGridLayoutUpp, self.gridLayoutUppButton)
        topSizer.Add(self.gridLayoutNumButton, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)
        self.Bind(wx.EVT_BUTTON, self.setGridLayoutNum, self.gridLayoutNumButton)

        bottomSizer.Add(self.gridLayoutEtcButton, proportion=0,
                flag=wx.LEFT | wx.BOTTOM, border=10)
        self.Bind(wx.EVT_BUTTON, self.setGridLayoutEtc, self.gridLayoutEtcButton)
        bottomSizer.Add(self.gridLayoutSymButton, proportion=0,
                flag=wx.RIGHT | wx.BOTTOM , border=10)
        self.Bind(wx.EVT_BUTTON, self.setGridLayoutSym, self.gridLayoutSymButton)

        gridLayoutControlBox.Add(topSizer, proportion=0, flag=wx.EXPAND)
        gridLayoutControlBox.Add(bottomSizer, proportion=0, flag=wx.EXPAND)

        self.sizer.Add(gridLayoutControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setGridLayoutLow(self, event):
        self.pg.gridSpeller.setGridLower()

    def setGridLayoutUpp(self, event):
        self.pg.gridSpeller.setGridUpper()

    def setGridLayoutNum(self, event):
        self.pg.gridSpeller.setGridNum()
    
    def setGridLayoutEtc(self, event):
        self.pg.gridSpeller.setGridEtc()

    def setGridLayoutSym(self, event):
        self.pg.gridSpeller.setGridSym()

    def initClassifier(self):
        classifierKindControlBox = widgets.ControlBox(self, label='Classifier', orient=wx.VERTICAL)
        self.classifierKindComboBox = wx.ComboBox(self, value=self.pg.classifierKind,
                style=wx.CB_READONLY, choices=self.pg.classifierChoices)
        self.Bind(wx.EVT_COMBOBOX, self.setClassifierKind, self.classifierKindComboBox)
        self.offlineControls += [self.classifierKindComboBox]
        classifierKindControlBox.Add(self.classifierKindComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(classifierKindControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setClassifierKind(self, event):
        self.pg.classifierKind = self.classifierKindComboBox.GetValue()
        self.pg.requireRetrain()

class PlotPanel(wx.Panel):
    def __init__(self, parent, pg, *args, **kwargs):
        # initialize wx.Panel parent class
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.pg = pg

        # various initialization routines
        self.initGridSpeller()
        self.initERPCanvas()
        self.initLayout()

    def initGridSpeller(self):
        self.gridSpeller = widgets.GridSpeller(self)
        self.gridSpeller.setGridUpper()

    def initERPCanvas(self):
        self.erpFig = plt.Figure()
        self.erpFig.subplots_adjust(hspace=0.32, wspace=0.02,
            left=0.065, right=0.95, top=0.97, bottom=0.18)
        gs = pltgs.GridSpec(2,4)
        self.erpAx = self.erpFig.add_subplot(gs[0,:])
        self.h1Ax  = self.erpFig.add_subplot(gs[1,0])
        self.h2Ax  = self.erpFig.add_subplot(gs[1,1])
        self.h3Ax  = self.erpFig.add_subplot(gs[1,2])
        self.h4Ax  = self.erpFig.add_subplot(gs[1,3])
        self.cbAx  = self.erpFig.add_axes([0.05, 0.08, 0.9, 0.05])
        self.erpCanvas = FigureCanvas(parent=self, id=wx.ID_ANY, figure=self.erpFig)

    def initLayout(self):
        plotSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        plotSizer.Add(self.gridSpeller, proportion=1, flag=wx.EXPAND)
        plotSizer.Add(self.erpCanvas, proportion=1, flag=wx.EXPAND)

        self.SetSizer(plotSizer)

        self.Layout()

        self.showGrid()

    def showGrid(self):
        self.erpCanvas.Hide()
        self.gridSpeller.Show()
        self.Layout()

    def showERPCanvas(self):
        self.erpCanvas.Show()
        #self.erpCanvas.draw()
        self.gridSpeller.Hide()
        self.Layout()

    def plotERP(self, cap):
        chanIndex = cap.getChanIndices(('cz',))[0]
        if chanIndex is None:
            chans = (0,)
            chanIndex = 0
            wx.LogWarning('Could not find channel Cz.  Using first channel instead.')
        else:
            chans = ('Cz',)

        cap = self.pg.bandpass(cap)
        seg = cap.segment(start=-0.2, end=0.75)
        seg = self.pg.downsample(seg)

        targ = seg.select(matchFunc=lambda mark: mark > 0.0)
        nonTarg = seg.select(matchFunc=lambda mark: mark < 0.0)

        for ax in (self.erpAx, self.h1Ax, self.h2Ax, self.h3Ax, self.h4Ax):
            ax.cla()

        targPlot = targ.plotAvg(chans=chans, ax=self.erpAx, linewidth=2, color='blue')
        targPlot['lines'][0].set_label('Target ERP')

        nonTargPlot = nonTarg.plotAvg(chans=chans, ax=self.erpAx, linewidth=2, color='green')
        nonTargPlot['lines'][0].set_label('Non-Target ERP')

        erp = np.mean(targ.data, axis=0)

        mn = np.min(erp[:,chanIndex])
        mx = np.max(erp[:,chanIndex])

        self.erpAx.hlines(0.0, 0.0, 0.8, linestyle='--', linewidth=2, color='grey')
        self.erpAx.vlines(0.0, mn, mx, linestyle='--', linewidth=2, color='grey')
        self.erpAx.vlines((200, 300, 400, 500), mn, mx,
                          linestyle='--', linewidth=1, color='red')

        self.erpAx.legend()
        self.erpAx.set_xlabel('Time (s)')
        self.erpAx.set_ylabel(r'Signal ($\mu V$)')

        sampRate = targ.getSampRate()
        erp1 = erp[int((0.2+0.2)*sampRate),:]
        erp2 = erp[int((0.2+0.3)*sampRate),:]
        erp3 = erp[int((0.2+0.4)*sampRate),:]
        erp4 = erp[int((0.2+0.5)*sampRate),:]

        erpAll = np.concatenate((erp1, erp2, erp3, erp4))

        mn = np.min(erpAll)
        mx = np.max(erpAll)

        interpMethod = 'multiquadric'
        coord = '3d'
        h1 = eeg.plotHeadInterp(erp1, chanNames=targ.getChanNames(),
            method=interpMethod, coord=coord, mn=mn, mx=mx, ax=self.h1Ax)
        self.h1Ax.set_title('200ms')
        h2 = eeg.plotHeadInterp(erp2, chanNames=targ.getChanNames(),
            method=interpMethod, coord=coord, mn=mn, mx=mx, ax=self.h2Ax)
        self.h2Ax.set_title('300ms')
        h3 = eeg.plotHeadInterp(erp3, chanNames=targ.getChanNames(),
            method=interpMethod, coord=coord, mn=mn, mx=mx, ax=self.h3Ax)
        self.h3Ax.set_title('400ms')
        h4 = eeg.plotHeadInterp(erp4, chanNames=targ.getChanNames(),
            method=interpMethod, coord=coord, mn=mn, mx=mx, ax=self.h4Ax)
        self.h4Ax.set_title('500ms')

        cbar = plt.colorbar(h1['im'], ax=self.erpAx, orientation='horizontal', cax=self.cbAx)

        cbar.set_label(r'Mean ERP ($\mu V$)')

        self.showERPCanvas()

class P300Grid(StandardBCIPage):
    """P300-speller interface for writing using EEG cap
    """
    def __init__(self, *args, **kwargs):
        """Construct a new P300Grid page.

        Args:
            *args, **kwargs:  Arguments to pass to the Page base class.
        """
        self.initConfig()

        StandardBCIPage.__init__(self, name='P300Grid',
           configPanelClass=ConfigPanel,  *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initConfig(self):
        """Initialize configuration values.
        """
        # shape of our grid
        # this should be configurable XXX - idfah
        self.nRows = 6
        self.nCols = 6

        # row and column that are currently highlighted
        self.curRow = -1
        self.curCol = -1

        # number of flashes per symbol
        self.nTrials = 10

        # train and test text
        self.trainText = 'COLORADO'
        self.testText = 'STATE'
        self.freeSpelling = False

        # stimulus interval and inter-stimulus interval
        self.si = 0.100
        self.isi = 0.100

        # time to pause after showing a symbol
        self.pause = 3.0

        # beginning and end of ERP window in seconds
        self.windowStart = 0.0
        self.windowEnd = 0.75

        # classifier parameters
        self.classifierChoices = ('Linear Discriminant', 
                                  'K-Nearest Euclidean',
                                  'K-Nearest Cosine',
                                  'Linear Logistic',
                                  'Neural Network')
        self.classifierKind = self.classifierChoices[0]

        self.nFold = 10

        # capture of training eeg data, none if no data
        self.trainCap = None

    def initCurStimList(self):
        """Initialize and randomize list of current stimuli indices to draw from.
        """
        #self.curStimList = list(range(1, self.nRows + self.nCols + 1))
        #self.curStimList *= self.nTrials
        #np.random.shuffle(self.curStimList)

        # permute individually and then append
        ind = np.arange(self.nRows + self.nCols) + 1
        self.curStimList = sum([list(np.random.permutation(ind))
                               for trial in xrange(self.nTrials)], [])

        #self.curStimList = sum([list(np.random.permutation(ind[::2])) +
        #                        list(np.random.permutation(ind[1::2]))
        #                            for trial in xrange(self.nTrials)], [])

        #rind = list(range(1,self.nRows+1))
        #cind = list(range(self.nRows+1,self.nCols+self.nRows+1))

        #self.curStimList = sum([list(np.random.permutation(rind)) +
        #                        list(np.random.permutation(cind))
        #                            for trial in xrange(self.nTrials)], [])

        #self.curStimList = sum([list(np.random.permutation(rind[::2] + cind[1::2])) +
        #                        list(np.random.permutation(cind[::2] + rind[1::2]))
        #                            for trial in xrange(self.nTrials)], [])

        #self.curStimList = sum([list(np.random.permutation(rind[::2])) +
        #                        list(np.random.permutation(cind[1::2])) +
        #                        list(np.random.permutation(rind[1::2])) +
        #                        list(np.random.permutation(cind[::2]))
        #                            for trial in xrange(self.nTrials)], [])

        #self.curStimList = sum([list(np.random.permutation(rind[::2])) +
        #                        list(np.random.permutation(rind[1::2])) +
        #                        list(np.random.permutation(cind[1::2])) +
        #                        list(np.random.permutation(cind[::2]))
        #                            for trial in xrange(self.nTrials)], [])

    def initPlots(self):
        """Initialize PieMenu and ERP plots.
        """
        self.plotPanel = PlotPanel(self, self)
        self.gridSpeller = self.plotPanel.gridSpeller
        self.gridSpeller.setCopyText(self.trainText)

    def initLayout(self):
        self.initStandardLayout()

        # plot pane
        plotPaneAuiInfo = aui.AuiPaneInfo().Name('plot').Caption('P300 Grid Speller').CenterPane()
        self.auiManager.AddPane(self.plotPanel, plotPaneAuiInfo)

        self.auiManager.Update()

    def stimToMark(self, mark):
        if self.freeSpelling:
            return -mark

        if mark == self.curRow+1 or mark == self.nRows+self.curCol+1:
            return mark
        else:
            return -mark

    def saveCap(self):
        cap = self.src.getEEGSecs(self.getSessionTime(), filter=False)
        saveDialog = wx.FileDialog(self, message='Save EEG data.',
            wildcard='Pickle (*.pkl)|*.pkl|All Files|*',
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        try:
            if saveDialog.ShowModal() == wx.ID_CANCEL:
                return
            cap.saveFile(saveDialog.GetPath())
        except Exception:
            wx.LogError('Save failed!')
            raise
        finally:
            saveDialog.Destroy()

    def saveResultText(self, resultText):
        saveDialog = wx.FileDialog(self, message='Save Result Text.',
            wildcard='Text (*.txt)|*.txt|All Files|*',
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        try:
            if saveDialog.ShowModal() == wx.ID_CANCEL:
                return
            with open(saveDialog.GetPath(), 'w') as fd:
                fd.write(resultText)
        except Exception:
            wx.LogError('Save failed!')
            raise
        finally:
            saveDialog.Destroy()

    #def decimate(self, cap):
    #    #cap = cap.demean().bandpass(0.5, 20.0, order=3)

    #    # original
    #    #cap = cap.copy().demean().bandpass(0.5, 12.0, order=3)
    #    # biosemi hack XXX - idfah
    #    cap = cap.copy().demean().reference((36,37)).deleteChans(range(32,40))
    #    cap.keepChans(('Fz', 'Cz', 'P3', 'Pz', 'P4', 'P7', 'Oz', 'P8'))

    #    # kind of a hack XXX - idfah
    #    if cap.getSampRate() > 32.0:
    #        decimationFactor = int(np.round(cap.getSampRate()/32.0))
    #        cap = cap.downsample(decimationFactor)

    #    return cap

    def bandpass(self, cap):
        cap = cap.copy()

        if cap.getSampRate() > 32.0:
            cap.bandpass(0.0, 10.0, order=3)
            cap.bandpass(0.5, np.inf, order=3)

        return cap

    def downsample(self, seg):
        if seg.getSampRate() > 32.0:
            decimationFactor = int(np.round(seg.getSampRate()/32.0))
            seg = seg.downsample(decimationFactor)

        return seg

    def beforeTrain(self):
        self.curRep = 0
        self.curStimList = []
        self.gridSpeller.setSelectColor(self.gridSpeller.getCopyColor(), refresh=False)
        self.gridSpeller.setCopyText(self.trainText) # in setTrained XXX - idfah
        self.gridSpeller.setFeedText('')
        self.src.setMarker(0.0)

    def afterTrain(self, earlyStop):
        if not earlyStop:
            self.trainCap = self.src.getEEGSecs(self.getSessionTime())
            self.saveCap()

            self.gridSpeller.setCopyText(self.testText) # should be set setTrained XXX - idfah
            self.gridSpeller.setFeedText('')

    def showTrainSymbol(self):
        # random, no bottom row
        #self.curRow = np.random.randint(0,5)
        #self.curCol = np.random.randint(0,6)

        trainSyms = [sym if sym != ' ' else grid.space for sym in self.trainText]
        sym = trainSyms[(self.curRep-1) % len(trainSyms)]
        self.curRow, self.curCol = self.gridSpeller.getGridLocation(sym)

        self.gridSpeller.selectSymbol(self.curRow, self.curCol)

        wx.CallLater(1000.0*self.pause, self.gridSpeller.removeHighlight)
        wx.CallLater(1000.0*(self.pause+self.windowEnd), self.trainClearStim)

    def trainEpoch(self):
        # if the stim list is empty
        if len(self.curStimList) == 0:
            # increment current repetition
            self.curRep += 1

            # if we have done all reps, then quit
            if self.curRep > len(self.trainText):
                 self.gridSpeller.removeHighlight()
                 ##wx.CallLater(1000.0*self.windowEnd*1.1-1000.0*self.si, self.endTrain)
                 wx.CallLater(1000.0*self.windowEnd*1.1, self.endTrain)

            # otherwise, reset stim list and show another training symbol
            else:
                self.initCurStimList()
                #self.showTrainSymbol()
                ##wx.CallLater(1000.0*self.windowEnd*1.1-1000.0*self.si, self.showTrainSymbol)
                wx.CallLater(1000.0*self.windowEnd*1.1, self.showTrainSymbol)

        # if stim list still had elements to show
        else:
            # grab next symbol index and set marker
            curStim = self.curStimList.pop()
            self.src.setMarker(self.stimToMark(curStim))

            # highlight row or column
            if curStim <= self.nRows:
                self.gridSpeller.highlightRow(curStim - 1)
            else:
                self.gridSpeller.highlightCol(curStim - self.nRows - 1)

            # clear after si seconds
            wx.CallLater(1000.0*self.si, self.trainClearStim)

    def trainClearStim(self, event=None):
        self.gridSpeller.removeHighlight()
        self.src.setMarker(0.0)

        wx.CallLater(1000.0*self.isi, self.runTrainEpoch)

    def trainClassifier(self):
        if self.trainCap is None:
            raise Exception('No data available for training.')

        self.plotPanel.plotERP(self.trainCap)

        dialog = wx.ProgressDialog('Training Classifier',
            'Featurizing', maximum=self.nFold+1,
            style=wx.PD_ELAPSED_TIME | wx.PD_SMOOTH)

        #cap = self.decimate(self.trainCap)
        cap = self.bandpass(self.trainCap)
        seg = cap.segment(start=self.windowStart, end=self.windowEnd)
        seg = self.downsample(seg)
        print 'nSeg: ', seg.getNSeg()

        ##print 'markers: ', seg.markers

        targ = seg.select(matchFunc=lambda mark: mark > 0.0)
        nTarg = targ.getNSeg()
        print 'nTarg: ', nTarg

        nonTarg = seg.select(matchFunc=lambda mark: mark < 0.0)
        nNonTarg = nonTarg.getNSeg()
        print 'nNonTarg: ', nNonTarg

        classData = [targ.chanEmbed(), nonTarg.chanEmbed()]

        #self.standardizer = ml.ClassStandardizer(classData)
        #classData = self.standardizer.apply(classData)
        self.standardizer = ml.Standardizer(np.vstack(classData))
        classData = [self.standardizer.apply(cls) for cls in classData]

        if self.classifierKind == 'Linear Discriminant':
            self.trainLDA(classData, dialog)
        elif self.classifierKind == 'K-Nearest Euclidean':
            self.trainKNN(classData, dialog, metric='euclidean')
        elif self.classifierKind == 'K-Nearest Cosine':
            self.trainKNN(classData, dialog, metric='cosine')
        elif self.classifierKind == 'Linear Logistic':
            self.trainLGR(classData, dialog)
        elif self.classifierKind == 'Neural Network':
            self.trainNN(classData, dialog)
        else:
            raise Exception('Invalid classifier kind: %s.' % str(self.classifierKind))

        self.plotPanel.showGrid()

    def trainLDA(self, classData, dialog):
        #shrinkages = np.insert(np.logspace(-3.0, -0.00001, num=100), 0, 0.0)
        #shrinkages = np.insert(np.power(10.0, np.linspace(-3.0, 0.0, 51)), 0, 0.0)
        shrinkages = np.linspace(0.0, 1.0, 51)

        trainAUC = np.zeros(shrinkages.shape)
        validAUC = np.zeros(shrinkages.shape)

        partitionGenerator = ml.part.classStratified(classData, self.nFold)

        for fold, trainData, validData in partitionGenerator:
            dialog.Update(fold, 'Validation Fold: %d' % fold)
            #print 'fold: ', fold

            for i, sh in enumerate(shrinkages):
                #print 'shrinkage: ', sh
                classifier = ml.LDA(trainData, shrinkage=sh)

                trainAUC[i] += classifier.auc(trainData)
                validAUC[i] += classifier.auc(validData)

        dialog.Update(self.nFold, 'Training Final Classifier')

        trainAUC /= self.nFold
        validAUC /= self.nFold

        print 'train AUC: ', trainAUC
        print 'valid AUC: ', validAUC

        bestShrinkage = shrinkages[np.argmax(validAUC)]
        print 'best shrinkage: ', bestShrinkage

        self.classifier = ml.LDA(classData, shrinkage=bestShrinkage)

        finalAUC = self.classifier.auc(classData)

        dialog.Destroy()

        resultText = (('Best Shrinkage: %f\n' % bestShrinkage) +
                      ('Mean Train AUC: %f\n' % np.max(trainAUC)) +
                      ('Mean Validation AUC: %f\n' % np.max(validAUC)) +
                      ('Final Training AUC: %f\n' % finalAUC))

        wx.MessageBox(message=resultText,
                      caption='Training Completed!',
                      style=wx.OK | wx.ICON_INFORMATION)

        self.saveResultText(resultText)

    def trainKNN(self, classData, dialog, metric):
        ks = np.arange(1,10)

        trainAUC = np.zeros(ks.shape)
        validAUC = np.zeros(ks.shape)

        partitionGenerator = ml.part.classStratified(classData, self.nFold)

        for fold, trainData, validData in partitionGenerator:
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            for i, k in enumerate(ks):
                classifier = ml.KNN(trainData, k=k, distMetric=metric)

                trainAUC[i] += classifier.auc(trainData)
                validAUC[i] += classifier.auc(validData)

        dialog.Update(self.nFold, 'Training Final Classifier')

        trainAUC /= self.nFold
        validAUC /= self.nFold

        print 'train AUC: ', trainAUC
        print 'valid AUC: ', validAUC

        bestK = ks[np.argmax(validAUC)]
        print 'best K: ', bestK

        self.classifier = ml.KNN(classData, k=bestK, distMetric=metric)

        finalAUC = self.classifier.auc(classData)

        dialog.Destroy()

        resultText = (('Best K: %d\n' % bestK) +
                      ('Mean Train AUC: %f\n' % np.max(trainAUC)) +
                      ('Mean Validation AUC: %f\n' % np.max(validAUC)) +
                      ('Final Training AUC: %f\n' % finalAUC))

        wx.MessageBox(message=resultText,
                      caption='Training Completed!',
                      style=wx.OK | wx.ICON_INFORMATION)

        self.saveResultText(resultText)

    def trainLGR(self, classData, dialog):
        penalties = np.insert(np.power(10.0, np.linspace(-2.0, 4.5, 51)), 0, 0.0)
        seed = np.random.randint(0, 1000000)

        trainAUC = np.zeros(penalties.shape)
        validAUC = np.zeros(penalties.shape)

        partitionGenerator = ml.part.classStratified(classData, self.nFold)

        for fold, trainData, validData in partitionGenerator:
            #print 'trainData shape: ', [cls.shape for cls in trainData]
            #print 'validData shape: ', [cls.shape for cls in validData]
            #print 'validData shape: ', [cls.shape for cls in validData]
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            for i, pen in enumerate(penalties):
                s = np.random.get_state()
                np.random.seed(seed)
                classifier = ml.LGRE(trainData, penalty=pen, optimFunc=ml.optim.scg,
                                     accuracy=0.0, precision=1.0e-10, maxIter=250)
                np.random.set_state(s)

                trainAUC[i] += classifier.auc(trainData)
                validAUC[i] += classifier.auc(validData)

        dialog.Update(self.nFold, 'Training Final Classifier')

        trainAUC /= self.nFold
        validAUC /= self.nFold

        print 'train AUC: ', trainAUC
        print 'valid AUC: ', validAUC

        bestPenalty = penalties[np.argmax(validAUC)]
        print 'best penalty: ', bestPenalty

        s = np.random.get_state()
        np.random.seed(seed)
        self.classifier = ml.LGRE(classData, penalty=bestPenalty,
                                  optimFunc=ml.optim.scg,
                                  accuracy=0.0, precision=1.0e-10, maxIter=250)
        np.random.set_state(s)

        finalAUC = self.classifier.auc(classData)

        dialog.Destroy()

        resultText = (('Best Penalty: %f\n' % bestPenalty) +
                      ('Mean Train AUC: %f\n' % np.max(trainAUC)) +
                      ('Mean Validation AUC: %f\n' % np.max(validAUC)) +
                      ('Final Training AUC: %f\n' % finalAUC))

        wx.MessageBox(message=resultText,
                      caption='Training Completed!',
                      style=wx.OK | wx.ICON_INFORMATION)

        self.saveResultText(resultText)

    def trainNN(self, classData, dialog):
        maxIter = 1000
        nHidden = 30
        seed = np.random.randint(0, 1000000)

        trainAUC = np.zeros((self.nFold, maxIter+1))
        validAUC = np.zeros((self.nFold, maxIter+1))

        partitionGenerator = ml.part.classStratified(classData, self.nFold)

        for fold, trainData, validData in partitionGenerator:
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            def validTraceCB(optable, iteration, paramTrace, errorTrace, success=True):
                if success:
                    trainAUC[fold,validTraceCB.it:] = optable.auc(trainData)
                    validAUC[fold,validTraceCB.it:] = optable.auc(validData)
                    validTraceCB.it += 1
            validTraceCB.it = 0

            s = np.random.get_state()
            np.random.seed(seed)
            classifier = ml.FNS(trainData, accuracy=0.0, precision=1.0e-10,
                                nHidden=nHidden, maxIter=maxIter, optimFunc=ml.optim.scg,
                                callback=validTraceCB, eTrace=False, verbose=False)
            np.random.set_state(s)

        dialog.Update(self.nFold, 'Training Final Classifier')

        meanValidAUC  = np.mean(validAUC, axis=0)

        bestIter = np.argmax(meanValidAUC)

        bestMeanValidAUC = meanValidAUC[bestIter]

        s = np.random.get_state()
        np.random.seed(seed)
        self.classifier = ml.FNS(classData, accuracy=0.0, precision=1.0e-10,
                                 nHidden=nHidden, maxIter=bestIter, optimFunc=ml.optim.scg,
                                 eTrace=False, verbose=False)
        np.random.set_state(s)

        finalAUC = self.classifier.auc(classData)

        dialog.Destroy()

        resultText = (('Best Iteration: %f\n' % bestIter) +
                      ('Mean Train AUC: %f\n' % np.max(trainAUC)) +
                      ('Mean Validation AUC: %f\n' % bestMeanValidAUC) +
                      ('Final Training AUC: %f\n' % finalAUC))

        wx.MessageBox(message=resultText,
                      caption='Training Completed!',
                      style=wx.OK | wx.ICON_INFORMATION)

        self.saveResultText(resultText)

    def beforeTest(self):
        # start with rep 1 because we don't want to trigger testClassify
        self.curRep = 1
        self.initCurStimList()

        if not self.freeSpelling:
            self.testSyms = [sym if sym != ' ' else grid.space for sym in self.testText]
            sym = self.testSyms.pop(0)
            self.curRow, self.curCol = self.gridSpeller.getGridLocation(sym)
        else:
            self.curRow = self.curCol = 0
            self.testSyms = [' ',] * 100 # will stop after 100 chars XXX - idfah

        self.testTime = time.time() # time that testing repetitions started
        self.gridSpeller.setSelectColor(self.gridSpeller.getFeedColor(), refresh=False)
        self.gridSpeller.removeHighlight(refresh=False)

        self.gridSpeller.setCopyText(self.testText)
        self.gridSpeller.setFeedText('') # should be set setTrained XXX - idfah

        self.src.setMarker(0.0)

    def afterTest(self, earlyStop):
        self.gridSpeller.removeHighlight()

        if not earlyStop:
            self.saveCap()
            self.gridSpeller.saveFile()

    def testEpoch(self):
        curStim = self.curStimList.pop()
        self.src.setMarker(self.stimToMark(curStim))

        if curStim <= self.nRows:
            self.gridSpeller.highlightRow(curStim - 1)
        else:
            self.gridSpeller.highlightCol(curStim - self.nRows - 1)

        wx.CallLater(1000.0*self.si, self.testClearStim)

    def testClearStim(self, event=None):
        self.gridSpeller.removeHighlight()
        self.src.setMarker(0.0)

        if len(self.curStimList) == 0:
            self.initCurStimList()

            wx.CallLater(1000.0*self.windowEnd*1.1, self.testClassify)
        else:
            wx.CallLater(1000.0*self.isi, self.runTestEpoch)

    def controlSpeller(self, rowChoice, colChoice):
        choice = self.gridSpeller.getGridValue(rowChoice, colChoice)
        self.gridSpeller.selectSymbol(rowChoice, colChoice)

        if self.freeSpelling:
            if choice in (grid.etc,):
                self.gridSpeller.appendFeedText('_')
            else:
                self.gridSpeller.appendFeedText(choice)
        else:
            if not len(choice) > 1 and choice not in (grid.num, grid.etc, grid.sym, grid.enter, grid.back, grid.upper, grid.lower):
                self.gridSpeller.appendFeedText(choice)
            else:
                self.gridSpeller.appendFeedText('_')

        if len(self.testSyms) == 0:
             ##wx.CallLater(1000.0*self.windowEnd*1.1-1000.0*self.si, self.endTest)
             wx.CallLater(1000.0*self.windowEnd*1.1, self.endTest)

        else:
            sym = self.testSyms.pop(0)

            if not self.freeSpelling:
                self.curRow, self.curCol = self.gridSpeller.getGridLocation(sym)

            wx.CallLater(1000.0*self.pause, self.gridSpeller.removeHighlight)
            ##wx.CallLater(1000.0*(self.pause+self.windowEnd), self.testClearStim)
            wx.CallLater(1000.0*2.0*self.pause, self.testClearStim)

    def testClassify(self):
        cap = self.src.getEEGSecs(time.time() - self.testTime)
        self.testTime = time.time()

        #cap = self.decimate(cap)
        cap = self.bandpass(cap)
        seg = cap.segment(start=self.windowStart, end=self.windowEnd)
        seg = self.downsample(seg)

        print 'nSeg: ', seg.getNSeg()

        #x = self.standardizer.apply([seg.chanEmbed(),])
        x = self.standardizer.apply(seg.chanEmbed())

        dv = self.classifier.discrim(x)

        # finding probabiites of selecting each letter using classification
        probabilities = np.zeros((self.nRows, self.nCols), dtype=np.float64)

        for m, d in zip(np.abs(seg.getMarkers()), dv):
            if m <= self.nRows:
                probabilities[m-1,:] += d[0]
            else:
                probabilities[:,m-self.nRows-1] += d[0]

        print 'probabilities: \n', probabilities

        resultRow, resultCol = np.unravel_index(probabilities.argmax(), probabilities.shape)

        #resultRow = np.random.randint(0,6)
        #resultCol = np.random.randint(0,6)
        self.controlSpeller(resultRow,resultCol)
