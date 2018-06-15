import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import random
import socket
import time
import wx
from wx.lib.agw import aui
import wx.lib.agw.floatspin as agwfs

from cebl import eeg
from cebl import ml
from cebl.rt import widgets

from .standard import StandardConfigPanel, StandardBCIPage


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

        self.initRobot()
        self.initNTrial()
        self.initIntervals()
        self.initSegWindow()
        self.initClassifierKind()
        self.initStandardLayout()

    def initRobot(self):
        # robot kind
        robotKindControlBox = widgets.ControlBox(self, label='Robot Kind', orient=wx.VERTICAL)

        self.robotKindComboBox = wx.ComboBox(self, value='ER1',
                style=wx.CB_READONLY, choices=('ER1', 'Baxter'))
        self.Bind(wx.EVT_COMBOBOX, self.setRobotKind, self.robotKindComboBox)
        self.offlineControls += [self.robotKindComboBox]
        robotKindControlBox.Add(self.robotKindComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(robotKindControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        robotHostControlBox = widgets.ControlBox(self, label='Robot Hostname', orient=wx.HORIZONTAL)

        # robot host
        self.robotHostTextCtrl = wx.TextCtrl(parent=self, value='IP or host name')
        #self.robotHostTextCtrl.SetHint('IP or host name') # in next version of wxpython? XXX - idfah
        robotHostControlBox.Add(self.robotHostTextCtrl, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)

        # robot connect button
        self.robotConnectButton = wx.Button(self, label='Connect')
        robotHostControlBox.Add(self.robotConnectButton, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)
        self.Bind(wx.EVT_BUTTON, self.setRobotHost, self.robotConnectButton)

        self.sizer.Add(robotHostControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setRobotKind(self, event):
        self.pg.robotKind = self.robotKindComboBox.GetValue()

        if self.pg.robotKind == 'ER1':
            self.pg.choices = self.pg.er1Choices

            targStr = 'Forward'
            nonTargStr = 'Backward'

            self.pg.pieMenu.setChoices(self.pg.choices, refresh=False)
            self.pg.pieMenu.setRotation(4.0*np.pi/len(self.pg.choices))

        elif self.pg.robotKind == 'Baxter':
            self.pg.choices = self.pg.baxterChoices

            targStr = 'Both Up'
            nonTargStr = 'Both Down'

            self.pg.pieMenu.setChoices(self.pg.choices, refresh=False)
            self.pg.pieMenu.setRotation(np.pi/len(self.pg.choices)+np.pi/2.0)

        else:
            raise Exception('Invalid robot kind: %s.' % str(self.pg.robotKind))

        self.pg.requireRetrain()

    def setRobotHost(self, event):
        robotHost = self.robotHostTextCtrl.GetValue()
        self.pg.connectToRobot(robotHost)

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
        classifierControlBox = widgets.ControlBox(self, label='Classifier', orient=wx.VERTICAL)
        self.classifierKindComboBox = wx.ComboBox(self, value=self.pg.classifierKind,
                style=wx.CB_READONLY, choices=self.pg.classifierChoices)
        self.Bind(wx.EVT_COMBOBOX, self.setClassifier, self.classifierKindComboBox)
        self.offlineControls += [self.classifierKindComboBox]
        classifierControlBox.Add(self.classifierKindComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(classifierControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setClassifier(self, event):
        self.pg.classifierKind = self.classifierKindComboBox.GetValue()
        self.pg.requireRetrain()

class PlotPanel(wx.Panel):
    def __init__(self, parent, pg, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.pg = pg

        # various initialization routines
        self.initPieMenu()
        self.initERPCanvas()
        self.initLayout()

    def initPieMenu(self):
        self.pieMenu = widgets.PieMenu(self,
                choices=self.pg.choices,
                rotation=4.0*np.pi/len(self.pg.choices),
                colors=('turquoise', 'red', 'blue violet', 'orange',
                        'blue', 'yellow'))
                #colors=('turquoise', 'red', 'blue', 'green',
                #        'yellow', 'blue violet'))

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
        plotSizer = wx.BoxSizer(orient=wx.VERTICAL)

        plotSizer.Add(self.pieMenu, proportion=1, flag=wx.EXPAND)
        plotSizer.Add(self.erpCanvas, proportion=1, flag=wx.EXPAND)

        self.SetSizer(plotSizer)

        self.erpCanvas.Hide()
        self.Layout()

    def showPieMenu(self):
        self.erpCanvas.Hide()
        self.pieMenu.Show()
        self.Layout()

    def showERPCanvas(self):
        self.erpCanvas.Show()
        #self.erpCanvas.draw()
        self.pieMenu.Hide()
        self.Layout()

    def plotERP(self, cap):
        chanIndex = cap.getChanIndices(('cz',))[0]
        if chanIndex is None:
            chans = (0,)
            chanIndex = 0
            wx.LogWarning('Could not find channel Cz.  Using first channel instead.')
        else:
            chans = ('Cz',)

        cap = cap.copy().bandpass(0.5, np.inf, order=3)

        seg = cap.segment(start=-0.2, end=0.75)

        targ = seg.select(matchFunc=lambda mark: self.pg.markToStim(mark) == self.pg.targStr)
        nonTarg = seg.select(matchFunc=lambda mark: self.pg.markToStim(mark) == self.pg.nonTargStr)

        for ax in (self.erpAx, self.h1Ax, self.h2Ax, self.h3Ax, self.h4Ax):
            ax.cla()

        targPlot = targ.plotAvg(chans=chans, ax=self.erpAx, linewidth=2, color='blue')
        targPlot['lines'][0].set_label(self.pg.targStr + ' ERP')

        nonTargPlot = nonTarg.plotAvg(chans=chans, ax=self.erpAx, linewidth=2, color='green')
        nonTargPlot['lines'][0].set_label(self.pg.nonTargStr + ' ERP')

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

        cbar.set_label(r'Target ERP ($\mu V$)')

        self.showERPCanvas()

class P300Bot(StandardBCIPage):
    """PieMenu with P300-speller interface for driving the ER1 robot.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new P300bot page.

        Args:
            *args, **kwargs:  Arguments to pass to the Page base class.
        """
        self.initConfig()

        StandardBCIPage.__init__(self, name='P300Bot',
            configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initConfig(self):
        self.er1Choices = ['Forward', 'Right 30', 'Right 60', 'Backward', 'Left 60', 'Left 30']
        self.baxterChoices = ['Both Up', 'Right Up', 'Both Down', 'Left Up']

        self.robotKind = 'ER1'
        self.choices = self.er1Choices
        self.targStr = 'Forward'
        self.nonTargStr = 'Backward'

        self.classifierKind = 'LDA'
        self.classifierChoices = ('LDA', 'KNNE', 'KNNC', 'LGR', 'NN')

        self.nTrainTrial = 25
        self.nTestTrial = 4

        self.nFold = 10
        self.trainFrac = 0.8

        self.windowStart = 0.05
        self.windowEnd = 0.7

        self.si = 0.1
        self.isi = 0.550

        self.trainCap = None
        self.robotSock = None

    def initCurStimList(self):
        self.curStimList = copy.copy(self.choices)
        np.random.shuffle(self.curStimList)

    def initPlots(self):
        """Initialize PieMenu and ERP plots.
        """
        self.plotPanel = PlotPanel(self, self)
        self.pieMenu = self.plotPanel.pieMenu

    def initLayout(self):
        self.initStandardLayout()

        # plot pane
        plotPaneAuiInfo = aui.AuiPaneInfo().Name('plot').Caption('P300 Pie Menu').CenterPane()
        self.auiManager.AddPane(self.plotPanel, plotPaneAuiInfo)

        self.auiManager.Update()

    def connectToRobot(self, host, port=7799):
        if self.robotSock is not None:
            try:
                self.robotSock.close()
            except Exception:
                pass

        try:
            self.robotSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.robotSock.connect((host,port))

        except Exception as e:
            self.robotSock = None
            wx.LogError('Failed to connect to robot: ' + str(e))

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

    def decimate(self, cap):
        cap = cap.copy().demean()

        cap.bandpass(0.0, 10.0, order=3)
        cap.bandpass(0.5, np.inf, order=3)

        # kind of a hack XXX - idfah
        if cap.getSampRate() > 32.0:
            decimationFactor = int(np.round(cap.getSampRate()/32.0))
            cap = cap.downsample(decimationFactor)

        return cap

    def clearPieMenu(self):
        self.pieMenu.clearAllHighlights(refresh=False)
        self.pieMenu.zeroBars()

    def beforeTrain(self):
        self.curTrainTrial = 0
        self.initCurStimList()
        self.src.setMarker(0.0)

    def afterTrain(self, earlyStop):
        if not earlyStop:
            self.trainCap = self.src.getEEGSecs(self.getSessionTime())
            ##self.saveCap()

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

        cap = self.decimate(self.trainCap)
        seg = cap.segment(start=self.windowStart, end=self.windowEnd)
        #print 'nSeg: ', seg.getNSeg()

        targ = seg.select(matchFunc=lambda mark: self.markToStim(mark) == self.targStr)
        nTarg = targ.getNSeg()
        #print 'nTarg: ', nTarg

        #nonTarg = seg.select(matchFunc=lambda mark: self.markToStim(mark) == self.nonTargStr)
        nonTarg = seg.select(matchFunc=lambda mark: self.markToStim(mark) != self.targStr)
        nNonTarg = nonTarg.getNSeg()
        #print 'nNonTarg: ', nNonTarg

        classData = [targ.chanEmbed(), nonTarg.chanEmbed()]

        self.standardizer = ml.Standardizer(np.vstack(classData))
        classData = [self.standardizer.apply(cls) for cls in classData]
        #print 'classData shape', [cls.shape for cls in classData]

        if self.classifierKind == 'LDA':
            self.trainLDA(classData, dialog)
        elif self.classifierKind == 'KNNE':
            self.trainKNN(classData, dialog, metric='euclidean')
        elif self.classifierKind == 'KNNC':
            self.trainKNN(classData, dialog, metric='cosine')
        elif self.classifierKind == 'LGR':
            self.trainLGR(classData, dialog)
        elif self.classifierKind == 'NN':
            self.trainNN(classData, dialog)
        else:
            raise Exception('Invalid classifier kind: %s.' % str(self.classifierKind))

        self.plotPanel.showPieMenu()

    def trainLDA(self, classData, dialog):
        #shrinkages = np.insert(np.logspace(-3.0, -0.00001, num=100), 0, 0.0)
        #shrinkages = np.insert(np.power(10.0, np.linspace(-3.0, 0.0, 100)), 0, 0.0)
        shrinkages = np.linspace(0.0, 1.0, 51)

        trainAUC = np.zeros(shrinkages.shape)
        validAUC = np.zeros(shrinkages.shape)

        partitionGenerator = ml.part.classRandomSubSample(classData,
                self.trainFrac, self.nFold)

        for fold, trainData, validData in partitionGenerator:
            #print 'trainData shape: ', [cls.shape for cls in trainData]
            #print 'validData shape: ', [cls.shape for cls in validData]
            #print 'validData shape: ', [cls.shape for cls in validData]
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            for i, sh in enumerate(shrinkages):
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
        maxIter = 250
        nHidden = 10
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
                                callback=validTraceCB, eTrace=True, verbose=False)
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

        cap = self.decimate(cap)
        seg = cap.segment(start=self.windowStart, end=self.windowEnd)

        assert seg.getNSeg() == len(self.choices)

        stim = [self.markToStim(m) for m in seg.getMarkers()]

        x = self.standardizer.apply(seg.chanEmbed())

        dv = self.classifier.discrim(x)
        choice = stim[np.argmax(dv, axis=0)[0]]

        if self.pieMenu.growBar(choice, amount=1.0/self.nTestTrial):
            wx.CallAfter(self.moveRobot, choice)
        else:
            wx.CallLater(1000.0*self.isi, self.runTestEpoch)

    def moveRobot(self, choice):
        if self.robotSock is not None:
            if self.robotKind == 'ER1':
                if choice == 'Forward' or \
                   choice == 'Backward':
                    choice += ' 12'

            self.robotSock.send(choice + '\n')

        self.pieMenu.highlight(choice, style='pop')

        wx.CallLater(1000.0*2.0, self.clearPieMenu)

        if self.robotKind == 'ER1':
            wx.CallLater(1000.0*3.0, self.runTestEpoch)
        else:
            wx.CallLater(1000.0*8.0, self.runTestEpoch)
