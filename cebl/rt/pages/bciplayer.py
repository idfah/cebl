import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import os
import random
import time
import wx
from wx.lib.agw import aui
import wx.lib.agw.floatspin as agwfs

from cebl import eeg
from cebl import ml
from cebl.rt import widgets

from standard import StandardConfigPanel, StandardBCIPage


rightArrow = u'\u2192'
leftArrow = u'\u2190'

class ConfigPanel(StandardConfigPanel):
    """Panel containing configuration widgets.  This is intimately
    related to this specific page.  Extends wx.Panel.
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

        self.initMediaPath()
        self.initNTrial()
        self.initIntervals()
        self.initSegWindow()
        self.initClassifierKind()
        self.initStandardLayout()

    def initMediaPath(self):
        mediaPathControlBox = widgets.ControlBox(self, label='Media Path', orient=wx.HORIZONTAL)
        
        self.mediaPathTextCtrl = wx.TextCtrl(parent=self, style=wx.TE_PROCESS_ENTER)
        self.mediaPathTextCtrl.SetValue(self.pg.defaultMusicDir)
        mediaPathControlBox.Add(self.mediaPathTextCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_TEXT_ENTER, self.setMediaPath, self.mediaPathTextCtrl)
        self.offlineControls += [self.mediaPathTextCtrl]

        self.mediaBrowseButton = wx.Button(self, label='Browse')
        mediaPathControlBox.Add(self.mediaBrowseButton, proportion=0,
                flag=wx.TOP | wx.RIGHT | wx.BOTTOM, border=10)
        self.Bind(wx.EVT_BUTTON, self.mediaBrowse, self.mediaBrowseButton)
        self.offlineControls += [self.mediaBrowseButton]

        self.sizer.Add(mediaPathControlBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

    def setMediaPath(self, event):
        path = os.path.expanduser(self.mediaPathTextCtrl.GetValue())

        if not os.path.isdir(path):
            raise Exception('Path %s is not a valid directory!' % str(path))

        self.pg.mplayer.setCWD(self.mediaPathTextCtrl.GetValue())

    def mediaBrowse(self, event):
        dialog = wx.DirDialog(self, 'Choose media directory', '',
                    style=wx.DD_DEFAULT_STYLE)

        try:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = dialog.GetPath()
        except Exception:
            wx.LogError('Failed to open directory!')
            raise
        finally:
            dialog.Destroy()

        if len(path) > 0:
            self.mediaPathTextCtrl.SetValue(path)
            self.pg.mplayer.setCWD(path)

    def initNTrial(self):
        trialSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        trainTrialControlBox = widgets.ControlBox(self, label='Train Trials', orient=wx.VERTICAL)
        self.trainTrialSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.nTrainTrial), min=10, max=100)
        self.Bind(wx.EVT_SPINCTRL, self.setNTrainTrial, self.trainTrialSpinCtrl)
        self.offlineControls += [self.trainTrialSpinCtrl]
        trainTrialControlBox.Add(self.trainTrialSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        trialSizer.Add(trainTrialControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        testTrialControlBox = widgets.ControlBox(self, label='Test Trials', orient=wx.VERTICAL)
        self.testTrialSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.nTestTrial), min=1, max=100)
        self.Bind(wx.EVT_SPINCTRL, self.setNTestTrial, self.testTrialSpinCtrl)
        self.offlineControls += [self.testTrialSpinCtrl]
        testTrialControlBox.Add(self.testTrialSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        trialSizer.Add(testTrialControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(trialSizer, proportion=0, flag=wx.EXPAND)

    def setNTrainTrial(self, event):
        self.pg.nTrainTrial = self.trainTrialSpinCtrl.GetValue()
        self.pg.setTrained(False)

    def setNTestTrial(self, event):
        self.pg.nTestTrial = self.testTrialSpinCtrl.GetValue()

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
                flag=wx.RIGHT | wx.BOTTOM | wx.LEFT | wx.EXPAND, border=10)

        isiControlBox = widgets.ControlBox(self, label='Inter-Stim Secs', orient=wx.VERTICAL)
        self.isiFloatSpin = agwfs.FloatSpin(self, min_val=0.05, max_val=1.0,
                increment=1/20.0, value=self.pg.isi)
        self.isiFloatSpin.SetFormat("%f")
        self.isiFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setISI, self.isiFloatSpin)
        self.offlineControls += [self.isiFloatSpin]
        isiControlBox.Add(self.isiFloatSpin, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)
        intervalSizer.Add(isiControlBox, proportion=1,
            flag=wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

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
                flag=wx.RIGHT | wx.BOTTOM | wx.LEFT | wx.EXPAND, border=10)

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
            flag=wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=10)

        self.sizer.Add(windowSizer, proportion=0, flag=wx.EXPAND)

    def setWindowStart(self, event):
        self.pg.windowStart = self.windowStartFloatSpin.GetValue()
        self.pg.requireRetrain()

    def setWindowEnd(self, event):
        self.pg.windowEnd = self.windowEndFloatSpin.GetValue()
        self.pg.requireRetrain()

    def initClassifierKind(self):
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
        self.initPieMenu()
        self.initMPlayer()
        self.initERPCanvas()
        self.initLayout()

    def initPieMenu(self):
        font = wx.Font(pointSize=10, family=wx.FONTFAMILY_SWISS,
                       style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_BOLD,
                       #underline=False, face='DejaVu Serif')
                       underline=False, face='Utopia')

        self.pieMenu = widgets.PieMenu(self,
                choices=self.pg.choices, #font=font,
                rotation=4.0*np.pi/len(self.pg.choices),
                colors=('turquoise', 'red', 'blue violet', 'orange',
                        'blue', 'crimson'))
                #colors=('turquoise', 'red', 'blue', 'green',
                #        'yellow', 'blue violet'))

    def initMPlayer(self):
        self.mplayer = widgets.MPlayerPanel(self,
                cwd=self.pg.defaultMusicDir, style=wx.SUNKEN_BORDER)

    def initERPCanvas(self):
        #self.erpFig = plt.Figure()
        #self.erpAx = self.erpFig.add_subplot(1,1,1)
        #self.erpCanvas = FigureCanvas(parent=self, id=wx.ID_ANY, figure=self.erpFig)
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

        plotSizer.Add(self.pieMenu, proportion=1, flag=wx.EXPAND)
        plotSizer.Add(self.mplayer, proportion=0, flag=wx.EXPAND)
        plotSizer.Add(self.erpCanvas, proportion=1, flag=wx.EXPAND)

        self.SetSizer(plotSizer)

        self.Layout()

        ##self.showPieMenu()
        self.showMPlayer()

    def showPieMenu(self):
        self.erpCanvas.Hide()
        self.pieMenu.Show()
        self.mplayer.Hide()
        self.Layout()

    def showMPlayer(self):
        self.erpCanvas.Hide()
        self.pieMenu.Show()
        self.mplayer.Show()
        self.Layout()

    def showERPCanvas(self):
        self.erpCanvas.Show()
        #self.erpCanvas.draw()
        self.pieMenu.Hide()
        self.mplayer.Hide()
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

        targ = seg.select(matchFunc=lambda mark: self.pg.markToStim(mark) == 'Play')
        nonTarg = seg.select(matchFunc=lambda mark: self.pg.markToStim(mark) == 'Preview')

        for ax in (self.erpAx, self.h1Ax, self.h2Ax, self.h3Ax, self.h4Ax):
            ax.cla()

        targPlot = targ.plotAvg(chans=chans, ax=self.erpAx, linewidth=2, color='blue')
        targPlot['lines'][0].set_label('Play ERP')

        nonTargPlot = nonTarg.plotAvg(chans=chans, ax=self.erpAx, linewidth=2, color='green')
        nonTargPlot['lines'][0].set_label('Preview ERP')

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
        #interpMethod = 'none'
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

        cbar.set_label(r'Target ERP ($\mu V$)')

        self.showERPCanvas()

class BCIPlayer(StandardBCIPage):
    """PieMenu with P300-speller interface for controlling a music/video player.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new BCIPlayer page.

        Args:
            *args, **kwargs:  Arguments to pass to the Page base class.
        """
        self.initConfig()

        StandardBCIPage.__init__(self, name='BCIPlayer',
            configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initConfig(self):
        self.choices = ['Play', 'Album ' + rightArrow, 'Song ' + rightArrow,
                        'Preview', leftArrow + ' Song', leftArrow + ' Album']

        self.defaultMusicDir = '~/music/Keller_Williams/'
        if not os.path.isdir(os.path.expanduser(self.defaultMusicDir)):
            self.defaultMusicDir = '~'

        self.classifierChoices = ('Linear Discriminant', 
                                  'K-Nearest Euclidean',
                                  'K-Nearest Cosine',
                                  'Linear Logistic',
                                  'Neural Network')
        self.classifierKind = self.classifierChoices[0]

        self.nTrainTrial = 30
        self.nTestTrial = 4

        self.nFold = 10
        self.trainFrac = 0.8

        self.windowStart = 0.05
        self.windowEnd = 0.7

        self.si = 0.1
        self.isi = 0.550

        self.trainCap = None
        
    def initCurStimList(self):
        self.curStimList = copy.copy(self.choices)
        np.random.shuffle(self.curStimList)

    def initPlots(self):
        """Initialize PieMenu and ERP plots.
        """
        self.plotPanel = PlotPanel(self, self)
        self.mplayer = self.plotPanel.mplayer
        self.pieMenu = self.plotPanel.pieMenu

        self.Bind(widgets.EVT_MPLAYER_FINISHED, self.mplayerFinished)

    def initLayout(self):
        self.initStandardLayout()

        # plot pane
        plotPaneAuiInfo = aui.AuiPaneInfo().Name('plot').Caption('P300 Pie Menu').CenterPane()
        self.auiManager.AddPane(self.plotPanel, plotPaneAuiInfo)

        self.auiManager.Update()

    ##def setTrained(self, trained):
    ##    StandardBCIPage.setTrained(self, trained)
    ##    if self.isTrained():
    ##        self.plotPanel.showMPlayer()
    ##    else:
    ##        self.plotPanel.showPieMenu()

    def markToStim(self, mark):
        return self.choices[int(mark)-1]

    def stimToMark(self, stim):
        return float(self.choices.index(stim)+1)

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

    ##def decimate(self, cap):
    ##    #cap = cap.demean().bandpass(0.5, 10.0, order=3)
    ##    cap = cap.copy().demean().bandpass(0.5, 12.0, order=3)

    ##    # kind of a hack XXX - idfah
    ##    if cap.getSampRate() > 32.0:
    ##        decimationFactor = int(np.round(cap.getSampRate()/32.0))
    ##        cap = cap.downsample(decimationFactor)

    ##    return cap

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

    def clearPieMenu(self):
        self.pieMenu.clearAllHighlights(refresh=False)
        self.pieMenu.zeroBars()

    def beforeTrain(self):
        wx.MessageBox(
            message='Please note each time the "Play" section of the menu is highlighted.',
            caption='Instructions', style=wx.OK | wx.ICON_INFORMATION)

        self.curTrainTrial = 0
        self.initCurStimList()
        self.src.setMarker(0.0)

    def afterTrain(self, earlyStop):
        if not earlyStop:
            self.trainCap = self.src.getEEGSecs(self.getSessionTime())
            self.saveCap()

    def trainEpoch(self):
        if len(self.curStimList) == 0:
            self.curTrainTrial += 1
            self.initCurStimList()

        if self.curTrainTrial >= self.nTrainTrial:
            wx.CallLater(1000.0*self.windowEnd*1.05, self.endTrain)
            return

        curStim = self.curStimList.pop()

        self.src.setMarker(self.stimToMark(curStim))
        self.pieMenu.highlight(curStim, style='jump')

        wx.CallLater(1000.0*self.si, self.trainClearStim)

    def trainClearStim(self, event=None):
        self.pieMenu.clearAllHighlights()

        self.src.setMarker(0.0)

        wx.CallLater(1000.0*self.isi, self.runTrainEpoch)

    def trainClassifier(self):
        if self.trainCap is None:
            raise Exception('No data available for training.')

        self.plotPanel.plotERP(self.trainCap)

        dialog = wx.ProgressDialog('Training Classifier',
            'Featurizing', maximum=self.nFold+1,
            style=wx.PD_ELAPSED_TIME | wx.PD_SMOOTH)

        cap = self.bandpass(self.trainCap)
        seg = cap.segment(start=self.windowStart, end=self.windowEnd)
        seg = self.downsample(seg)
        print 'nSeg: ', seg.getNSeg()

        targ = seg.select(matchFunc=lambda mark: self.markToStim(mark) == 'Play')
        nTarg = targ.getNSeg()
        #print 'nTarg: ', nTarg

        #nonTarg = seg.select(matchFunc=lambda mark: self.markToStim(mark) == 'Backward')
        nonTarg = seg.select(matchFunc=lambda mark: self.markToStim(mark) != 'Play')
        nNonTarg = nonTarg.getNSeg()
        #print 'nNonTarg: ', nNonTarg

        classData = [targ.chanEmbed(), nonTarg.chanEmbed()]

        self.standardizer = ml.Standardizer(np.vstack(classData))
        classData = [self.standardizer.apply(cls) for cls in classData]
        #print 'classData shape', [cls.shape for cls in classData]

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

        self.plotPanel.showMPlayer()

    def trainLDA(self, classData, dialog):
        #shrinkages = np.insert(np.logspace(-3.0, -0.00001, num=100), 0, 0.0)
        #shrinkages = np.insert(np.power(10.0, np.linspace(-3.0, 0.0, 51)), 0, 0.0)
        shrinkages = np.linspace(0.0, 1.0, 51)

        trainAUC = np.zeros(shrinkages.shape)
        validAUC = np.zeros(shrinkages.shape)

        partitionGenerator = ml.part.classRandomSubSample(classData,
                self.trainFrac, self.nFold)

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

        wx.MessageBox(message=('Best Shrinkage: %f\n' % bestShrinkage) + \
            ('Mean Validation AUC: %f\n' % np.max(validAUC)) +
            ('Final Training AUC: %f' % finalAUC),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def trainKNN(self, classData, dialog, metric):
        ks = np.arange(1,10)

        trainAUC = np.zeros(ks.shape)
        validAUC = np.zeros(ks.shape)

        partitionGenerator = ml.part.classRandomSubSample(classData,
                self.trainFrac, self.nFold)

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

        wx.MessageBox(message=('Best K: %d\n' % bestK) + \
            ('Mean Validation AUC: %f\n' % np.max(validAUC)) +
            ('Final Training AUC: %f' % finalAUC),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def trainLGR(self, classData, dialog):
        penalties = np.insert(np.power(10.0, np.linspace(-2.0, 4.5, 51)), 0, 0.0)
        seed = np.random.randint(0, 1000000)

        trainAUC = np.zeros(penalties.shape)
        validAUC = np.zeros(penalties.shape)

        partitionGenerator = ml.part.classRandomSubSample(classData,
                self.trainFrac, self.nFold)

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

        wx.MessageBox(message=('Best Penalty: %f\n' % bestPenalty) + \
            ('Mean Validation AUC: %f\n' % np.max(validAUC)) +
            ('Final Training AUC: %f' % finalAUC),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def trainNN(self, classData, dialog):
        maxIter = 1000
        nHidden = 30
        seed = np.random.randint(0, 1000000)

        trainAUC = np.zeros((self.nFold, maxIter+1))
        validAUC = np.zeros((self.nFold, maxIter+1))

        partitionGenerator = ml.part.classRandomSubSample(classData,
                self.trainFrac, self.nFold)

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

        wx.MessageBox(message=('Best Iteration: %f\n' % bestIter) +
            ('Mean Validation AUC: %f\n' % bestMeanValidAUC) +
            ('Final Training AUC: %f' % finalAUC),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def beforeTest(self):
        self.initCurStimList()
        self.testTime = time.time()
        self.src.setMarker(0.0)

    def afterTest(self, earlyStop):
        self.pieMenu.clearAllHighlights(refresh=False)
        self.pieMenu.zeroBars()
        self.mplayer.stop()

    def testEpoch(self):
        curStim = self.curStimList.pop()

        self.src.setMarker(self.stimToMark(curStim))
        self.pieMenu.highlight(curStim, style='jump')

        wx.CallLater(1000.0*self.si, self.testClearStim)

    def testClearStim(self, event=None):
        self.pieMenu.clearAllHighlights()
        self.src.setMarker(0.0)

        if len(self.curStimList) == 0:
            self.initCurStimList()
            wx.CallLater(1000.0*self.windowEnd*1.05, self.testClassify)
        else:
            wx.CallLater(1000.0*self.isi, self.runTestEpoch)

    def testClassify(self):
        cap = self.src.getEEGSecs(time.time() - self.testTime)
        self.testTime = time.time()

        cap = self.bandpass(cap)
        seg = cap.segment(start=self.windowStart, end=self.windowEnd)
        seg = self.downsample(seg)

        # if verbose XXX - idfah
        #wx.LogMessage('nSeg: %d' % seg.getNSeg())
        assert seg.getNSeg() == len(self.choices)

        stim = [self.markToStim(m) for m in seg.getMarkers()]

        x = self.standardizer.apply(seg.chanEmbed())

        dv = self.classifier.discrim(x)
        choice = stim[np.argmax(dv, axis=0)[0]]

        if self.pieMenu.growBar(choice, amount=1.0/self.nTestTrial):
            wx.CallAfter(self.controlPlayer, choice)
        else:
            wx.CallLater(1000.0*self.isi, self.runTestEpoch)

    def controlPlayer(self, choice):
        self.pieMenu.highlight(choice, style='pop')

        def moveOn():
            wx.CallLater(1000.0*1.0, self.clearPieMenu)
            wx.CallLater(1000.0*2.0, self.runTestEpoch)

        if choice == 'Play':
            self.mplayer.play()
            wx.CallLater(1000.0*2.0, self.pieMenu.zeroBars)
        elif choice == 'Album ' + rightArrow:
            self.mplayer.forAlbum()
            moveOn()
        elif choice == leftArrow + ' Album':
            self.mplayer.rewAlbum()
            moveOn()
        elif choice == 'Song ' + rightArrow:
            self.mplayer.forSong()
            moveOn()
        elif choice == leftArrow + ' Song':
            self.mplayer.rewSong()
            moveOn()
        elif choice == 'Preview':
            self.mplayer.preview()
            wx.CallLater(1000.0*2.0, self.pieMenu.zeroBars)
        else:
            raise Exception('Invalid choice: %s.' % str(choice))

    def mplayerFinished(self, event=None):
        if self.isRunning():
            self.pieMenu.clearAllHighlights()
            wx.CallLater(1000.0*1.0, self.runTestEpoch)
            wx.LogMessage('restarting after EVT_MPLAYER_FINISHED')
