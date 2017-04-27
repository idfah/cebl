import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import wx
from wx.lib.agw import aui
import wx.lib.agw.floatspin as agwfs

from cebl import ml
from cebl.rt import widgets

from standard import StandardConfigPanel, StandardBCIPage


class ConfigPanel(StandardConfigPanel):
    def __init__(self, parent, pg, *args, **kwargs):
        StandardConfigPanel.__init__(self, parent=parent, pg=pg, *args, **kwargs)

        self.initChoices()
        self.initNTrial()
        self.initTrialSecs()
        self.initARParam()
        self.initClassifier()
        self.initStandardLayout() 

    def initChoices(self):
        choiceControlBox = widgets.ControlBox(self, label='Choices', orient=wx.VERTICAL)

        self.choiceTextCtrl = wx.TextCtrl(parent=self, value=', '.join(self.pg.choices),
                style=wx.TE_PROCESS_ENTER)
        choiceControlBox.Add(self.choiceTextCtrl, proportion=1,
                             flag=wx.ALL | wx.EXPAND, border=10)
        self.choiceTextCtrl.Bind(wx.EVT_KILL_FOCUS, self.setChoices, self.choiceTextCtrl)
        self.offlineControls += [self.choiceTextCtrl]

        self.sizer.Add(choiceControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

    def setChoices(self, event=None):
        choiceString = self.choiceTextCtrl.GetValue()
        choices = [c.strip() for c in choiceString.split(',')]
        self.pg.setChoices(choices)
        event.Skip()

    def initNTrial(self):
        trialSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        trainTrialControlBox = widgets.ControlBox(self, label='Train Trials', orient=wx.VERTICAL)
        self.trainTrialSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.nTrainTrial), min=5, max=100)
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

    def initTrialSecs(self):
        secsSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        trialSecsControlBox = widgets.ControlBox(self, label='Trial Secs', orient=wx.VERTICAL)

        self.trialSecsFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=50.0,
                increment=1/4.0, value=self.pg.trialSecs)
        self.trialSecsFloatSpin.SetFormat("%f")
        self.trialSecsFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setTrialSecs, self.trialSecsFloatSpin)
        self.offlineControls += [self.trialSecsFloatSpin]
        trialSecsControlBox.Add(self.trialSecsFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        secsSizer.Add(trialSecsControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        itiSecsControlBox = widgets.ControlBox(self, label='Inter-Trial Secs', orient=wx.VERTICAL)
        self.itiFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=10.0,
                increment=1/4.0, value=self.pg.iti)
        self.itiFloatSpin.SetFormat("%f")
        self.itiFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setITI, self.itiFloatSpin)
        self.offlineControls += [self.itiFloatSpin]
        itiSecsControlBox.Add(self.itiFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        secsSizer.Add(itiSecsControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(secsSizer, proportion=0, flag=wx.EXPAND)

    def setTrialSecs(self, event):
        self.pg.trialSecs = self.trialSecsFloatSpin.GetValue()
        self.pg.setTrained(False)

    def setITI(self, event):
        self.pg.iti = self.itiFloatSpin.GetValue()
        self.pg.setTrained(False)

    def initARParam(self):
        lowHighFreqSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        lowFreqControlBox = widgets.ControlBox(self, label='Low Freq', orient=wx.VERTICAL)
        self.lowFreqFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=100.0,
                increment=1/4.0, value=self.pg.lowFreq)
        self.lowFreqFloatSpin.SetFormat("%f")
        self.lowFreqFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setLowFreq, self.lowFreqFloatSpin)
        self.offlineControls += [self.lowFreqFloatSpin]
        lowFreqControlBox.Add(self.lowFreqFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        lowHighFreqSizer.Add(lowFreqControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        highFreqControlBox = widgets.ControlBox(self, label='High Freq', orient=wx.VERTICAL)

        self.highFreqFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=100.0,
             increment=1/4.0, value=self.pg.highFreq)
        self.highFreqFloatSpin.SetFormat("%f")
        self.highFreqFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setHighFreq, self.highFreqFloatSpin)
        self.offlineControls += [self.highFreqFloatSpin]
        highFreqControlBox.Add(self.highFreqFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        lowHighFreqSizer.Add(highFreqControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(lowHighFreqSizer, proportion=0, flag=wx.EXPAND)

        freqOrderSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        nFreqControlBox = widgets.ControlBox(self, label='Num Freqs', orient=wx.VERTICAL)
        self.nFreqSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.nFreq), min=1, max=500)
        self.Bind(wx.EVT_SPINCTRL, self.setNFreq, self.nFreqSpinCtrl)
        self.offlineControls += [self.nFreqSpinCtrl]
        nFreqControlBox.Add(self.nFreqSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        freqOrderSizer.Add(nFreqControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        orderControlBox = widgets.ControlBox(self, label='AR Order', orient=wx.VERTICAL)
        self.orderSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.order), min=2, max=200)
        self.Bind(wx.EVT_SPINCTRL, self.setOrder, self.orderSpinCtrl)
        self.offlineControls += [self.orderSpinCtrl]
        orderControlBox.Add(self.orderSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        freqOrderSizer.Add(orderControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(freqOrderSizer, proportion=0, flag=wx.EXPAND)

    def setLowFreq(self, event=None):
        self.pg.lowFreq = self.lowFreqFloatSpin.GetValue()
        self.pg.requireRetrain()

    def setHighFreq(self, event=None):
        self.pg.highFreq = self.highFreqFloatSpin.GetValue()
        self.pg.requireRetrain()

    def setNFreq(self, event=None):
        self.pg.nFreq = self.nFreqSpinCtrl.GetValue()
        self.pg.requireRetrain()

    def setOrder(self, event=None):
        self.pg.order = self.orderSpinCtrl.GetValue()
        self.pg.requireRetrain()

    def initClassifier(self):
        classifierKindControlBox = widgets.ControlBox(self, label='Classifier', orient=wx.VERTICAL)
        self.classifierKindComboBox = wx.ComboBox(self, value=self.pg.classifierKind.upper(),
                style=wx.CB_READONLY, choices=('LDA', 'NN'))
        self.Bind(wx.EVT_COMBOBOX, self.setClassifierKind, self.classifierKindComboBox)
        self.offlineControls += [self.classifierKindComboBox]
        classifierKindControlBox.Add(self.classifierKindComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(classifierKindControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def setClassifierKind(self, event=None):
        self.pg.classifierKind = self.classifierKindComboBox.GetValue().lower()
        self.pg.requireRetrain()

class PlotPanel(wx.Panel):
    def __init__(self, parent, choices, rotation, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.initPieMenu(choices, rotation)
        self.initFeatureCanvas()
        self.initLayout()

    def initPieMenu(self, choices, rotation):
        self.pieMenu = widgets.PieMenu(self, choices=choices, rotation=rotation)

    def initFeatureCanvas(self):
        self.featureFig = plt.Figure()
        ##self.featureFig.subplots_adjust(hspace=0.32, wspace=0.02,
        ##    left=0.065, right=0.95, top=0.97, bottom=0.18)

        self.featureAx = self.featureFig.add_subplot(1,1,1)

        self.featureCanvas = FigureCanvas(parent=self, id=wx.ID_ANY, figure=self.featureFig)

    def initLayout(self):
        plotSizer = wx.BoxSizer(orient=wx.VERTICAL)

        plotSizer.Add(self.pieMenu, proportion=1, flag=wx.EXPAND | wx.ALL)
        plotSizer.Add(self.featureCanvas, proportion=1, flag=wx.EXPAND | wx.ALL)

        self.SetSizer(plotSizer)

        self.featureCanvas.Hide()
        self.Layout()

    def showPieMenu(self):
        self.featureCanvas.Hide()
        self.pieMenu.Show()
        self.Layout()

    def showFeatureCanvas(self):
        self.featureCanvas.Show()
        self.pieMenu.Hide()
        self.Layout()

    def plotFeatures(self, trainData, freqs, choices, chanNames):
        self.featureAx.cla()

        meanFeat = [np.mean(cls, axis=0) for cls in trainData]
        for cls, choice in zip(meanFeat, choices):
            self.featureAx.plot(cls, label=choice, marker='o', linewidth=2)

        self.featureAx.set_xlabel(r'Frequency ($Hz$)')
        self.featureAx.set_ylabel(r'Power ($uV^2 / Hz$)')
        self.featureAx.legend()

        nFreq = len(freqs)
        mn = np.min(np.concatenate(meanFeat))
        mx = np.max(np.concatenate(meanFeat))
        for i,cn in enumerate(chanNames):
            if i > 0:
                self.featureAx.vlines(i*float(nFreq), mn, mx, linestyle='--')
            self.featureAx.text((i+0.25)*float(nFreq), 0.8*mx, cn, fontsize=14)

        tickStride = int(np.ceil(nFreq/3.0))
        tickFreqs = freqs[::tickStride]
        tickPlaces = np.arange(nFreq)[::tickStride]
        tickLocs = np.concatenate(
                        [tickPlaces+nFreq*i for i,c in enumerate(chanNames)])
        tickLabels = np.round(np.tile(tickFreqs, len(chanNames))).astype(np.int)

        self.featureAx.set_xticks(tickLocs)
        self.featureAx.set_xticklabels(tickLabels)

        self.featureAx.autoscale(tight=True)
        self.featureFig.tight_layout()

        self.showFeatureCanvas()

class PieERN(StandardBCIPage):
    def __init__(self, *args, **kwargs):
        self.initConfig()

        StandardBCIPage.__init__(self, name='PieERN',
            configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initConfig(self):
        self.nTrainTrial = 20
        self.nTestTrial = 20

        self.trialSecs = 3.0
        self.iti = 1.0

        self.lowFreq = 4.0
        self.highFreq = 50.0
        self.nFreq = 47
        self.order = 10

        self.nFold = 5
        self.classifierKind = 'lda'
        self.classifier = None

        self.trainCap = None

        self.choices = ['Left', 'Right']
        self.nChoices = len(self.choices)
        self.confusion = np.zeros((self.nChoices,self.nChoices), dtype=np.float32)

    def initPlots(self):
        self.plotPanel = PlotPanel(self, choices=self.choices, rotation=-np.pi/2.0)
        self.pieMenu = self.plotPanel.pieMenu

    def initLayout(self):
        self.initStandardLayout()

        stimPaneAuiInfo = aui.AuiPaneInfo().Name('stim').Caption(self.name + ' Stimulus').CenterPane()
        self.auiManager.AddPane(self.plotPanel, stimPaneAuiInfo)

        self.auiManager.Update()

    def setChoices(self, choices):
        if len(choices) < 2:
            wx.LogError('Page %s: Cannot use less than 2 choices.' % self.name)
        elif self.isRunning():
            wx.LogError('Page %s: Cannot change choices while running.' % self.name)
        else:
            self.choices = choices
            self.nChoices = len(self.choices)
            self.confusion = np.zeros((self.nChoices,self.nChoices), dtype=np.float32)
            self.pieMenu.setChoices(choices)
            self.setTrained(False)

    def markToStim(self, mark):
        index = int(mark%10) - 1
        return self.choices[index]

    def getCap(self, secs):
        cap = self.src.getEEGSecs(secs)

        cap = cap.demean().bandpass(0.0, 80.0, order=3).bandpass(0.5, np.inf, order=3)
        if cap.getSampRate() > 256.0:
            decimationFactor = int(np.round(cap.getSampRate()/256.0))
            cap = cap.downsample(decimationFactor)

        return cap

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

    def powerize(self, segs):
        freqs = np.linspace(self.lowFreq, self.highFreq, self.nFreq)

        # generate PSD object for each seg in each class
        psds = [cls.psd(method='ar', order=self.order, freqs=freqs) for cls in segs]
        #psds = [cls.psd(method='welch', span=0.5) for cls in segs]

        # extract powers into a single matrix for each class
        powers = [np.array([segPsd.getPowers().T for segPsd in cls]) for cls in psds]
        freqs = psds[0][0].getFreqs()

        ##nFreq = np.sum(freqs < 60)
        ##powers = [cls[:,:,:nFreq] for cls in powers]

        # use sqrt power to put in amplitude units
        powers = [np.sqrt(cls) for cls in powers]

        # embed channels
        powers = [cls.reshape(cls.shape[0], -1) for cls in powers]

        return freqs, powers

    def beforeTrain(self):
        self.curChoices = self.choices*self.nTrainTrial
        np.random.shuffle(self.curChoices)

        self.src.setMarker(0.0)

    def afterTrain(self, earlyStop):
        if not earlyStop:
            self.trainCap = self.getCap(self.getSessionTime())
            self.saveCap()

    def trainEpoch(self):
        choice = self.curChoices.pop()
        self.pieMenu.highlight(choice, style='pop')

        self.src.setMarker(self.choices.index(choice)+1.0)

        wx.CallLater(1000.0*self.trialSecs, self.trainClearTrial)

    def trainClearTrial(self, event=None):
        self.pieMenu.clearAllHighlights()

        self.src.setMarker(0.0)

        if len(self.curChoices) > 0:
            wx.CallLater(1000.0*self.iti, self.runTrainEpoch)
        else:
            wx.CallLater(1000.0*self.iti, self.endTrain)

    def trainClassifier(self):
        if self.trainCap is None:
            raise Exception('No data available for training.')

        dialog = wx.ProgressDialog('Training Classifier',
                    'Featurizing', maximum=self.nFold+1,
                    style=wx.PD_ELAPSED_TIME | wx.PD_SMOOTH)

        segmented = self.trainCap.segment(start=0.0, end=self.trialSecs)
        segs = [segmented.select(matchFunc=lambda mark: self.markToStim(mark) == choice)
                for choice in self.choices]

        assert segs[0].getNSeg() == self.nTrainTrial

        ##print 'nSegs:'
        ##for sg in segs:
        ##    print sg.getNSeg(), sg.data.shape

        freqs, trainData = self.powerize(segs)

        self.plotPanel.plotFeatures(trainData, freqs, self.choices,
                                    self.trainCap.getChanNames())

        ##print trainData[0].mean(axis=0)
        ##print trainData[0].mean(axis=0).shape
        ##print trainData[1].mean(axis=0)
        ##print trainData[1].mean(axis=0).shape

        if self.classifierKind == 'lda':
            self.trainLDA(trainData, dialog)
        elif self.classifierKind == 'nn':
            self.trainNN(trainData, dialog)
        else:
            raise Exception('Invalid classifier kind: %s.' % str(self.classifierKind))

        self.plotPanel.showPieMenu()

    def trainLDA(self, trainData, dialog):
        penalties = np.insert(np.power(10.0, np.linspace(-3.0, 0.0, 50)), 0, 0.0)
        nPenalties = len(penalties)

        trnCA = np.zeros((self.nFold, nPenalties))
        valCA = np.zeros((self.nFold, nPenalties))

        for fold, trnData, valData in ml.part.classStratified(trainData, nFold=self.nFold):
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            stand = ml.ClassStandardizer(trnData)
            trnData = stand.apply(trnData)
            valData = stand.apply(valData)

            for i,penalty in enumerate(penalties):
                classifier = ml.LDA(trnData, shrinkage=penalty)

                trnCA[fold,i] = classifier.ca(trnData)
                valCA[fold,i] = classifier.ca(valData)

            meanTrnCA = np.mean(trnCA, axis=0)

        dialog.Update(self.nFold, 'Training Final Classifier')

        meanValCA = np.mean(valCA, axis=0)

        bestPenaltyIndex = np.argmax(meanValCA)
        bestPenalty = penalties[bestPenaltyIndex]

        bestMeanTrnCA = meanTrnCA[bestPenaltyIndex]
        bestMeanValCA = meanValCA[bestPenaltyIndex]

        self.stand = ml.ClassStandardizer(trainData)
        trainData = self.stand.apply(trainData)

        self.classifier = ml.LDA(trainData, shrinkage=bestPenalty)

        trainCA = self.classifier.ca(trainData)
        trainConfusion = self.classifier.confusion(trainData)

        dialog.Destroy()

        wx.MessageBox(message=('Best Shrinkage: %f\n' % bestPenalty) +
            ('Mean Validation CA: %f\n' % bestMeanValCA) +
            ('Final Training CA: %f' % trainCA),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def trainNN(self, trainData, dialog):
        maxIter = 250
        nHiddens = 10
        seed = np.random.randint(0, 1000000)

        trnCA = np.zeros((self.nFold, maxIter+1))
        valCA = np.zeros((self.nFold, maxIter+1))

        for fold, trnData, valData in ml.part.classStratified(trainData, nFold=self.nFold):
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            stand = ml.ClassStandardizer(trnData)
            trnData = stand.apply(trnData)
            valData = stand.apply(valData)

            def valTraceCB(optable, iteration, paramTrace, errorTrace, success=True):
                if success:
                    trnCA[fold,valTraceCB.it] = optable.ca(trnData)
                    valCA[fold,valTraceCB.it] = optable.ca(valData)
                    valTraceCB.it += 1
            valTraceCB.it = 0

            np.random.seed(seed)
            classifier = ml.FNS(trnData, accuracy=0.0, precision=0.0,
                                nHiddens=nHiddens, maxIter=maxIter, optimFunc=ml.optim.scg,
                                callback=valTraceCB, eTrace=True, verbose=False)

        dialog.Update(self.nFold, 'Training Final Classifier')

        meanValCA = np.mean(valCA, axis=0)

        bestIter = np.argmax(meanValCA)

        bestMeanValCA = meanValCA[bestIter]

        self.stand = ml.ClassStandardizer(trainData)
        trainData = self.stand.apply(trainData)

        np.random.seed(seed)
        self.classifier = ml.FNS(trainData, accuracy=0.0, precision=0.0,
                                 nHiddens=nHiddens, maxIter=bestIter, optimFunc=ml.optim.scg,
                                 eTrace=False, verbose=False)

        trainCA = self.classifier.ca(trainData)
        trainConfusion = self.classifier.confusion(trainData)

        dialog.Destroy()

        wx.MessageBox(message=('Best Iteration: %f\n' % bestIter) +
            ('Mean Validation CA: %f\n' % bestMeanValCA) +
            ('Final Training CA: %f' % trainCA),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def beforeTest(self):
        self.curChoices = self.choices*self.nTestTrial
        np.random.shuffle(self.curChoices)

        self.confusion[...] = 0.0

        self.src.setMarker(0.0)

    def afterTest(self, earlyStop):
        if not earlyStop:
            self.saveCap()
            ca = np.mean(np.diag(self.confusion))/self.nTestTrial
            wx.MessageBox(('Test CA: %f\n' % ca) +
                           'Confusion Matrix:\n' + str(self.confusion/self.nTestTrial),
                           'Testing Complete', wx.OK | wx.ICON_INFORMATION)

    def testEpoch(self):
        self.curChoice = self.curChoices.pop()
        self.pieMenu.highlight(self.curChoice, style='pop')

        self.src.setMarker(10.0*(self.choices.index(self.curChoice)+1.0))

        # a little extra at the end to make sure we get the last segment
        wx.CallLater(1000.0*self.trialSecs*1.1, self.testClassify)

    def testClassify(self):
        testCap = self.getCap(np.min((self.getSessionTime(), 2.0*self.trialSecs)))

        segs = testCap.segment(start=0.0, end=self.trialSecs)

        # select the last segment only
        segs = (segs.setMarkers(None)
            .select(lambda mark: np.isclose(mark, np.max(segs.markers))))

        assert segs.getNSeg() == 1

        ##print 'test nSeg: ', segs.nSeg
        ##print segs.data.shape
        ##print ''

        freqs, testData = self.powerize((segs,))

        testData = self.stand.apply(testData)[0]

        label = self.classifier.label(testData)
        selection = self.choices[label]

        self.src.incrementMarker(label+1)
        self.pieMenu.growBar(selection, 1.0)

        self.confusion[label, self.choices.index(self.curChoice)] += 1.0

        wx.CallLater(1000.0*self.iti, self.testClearTrial)

    def testClearTrial(self, event=None):
        self.pieMenu.zeroBars()
        self.pieMenu.clearAllHighlights()

        self.src.setMarker(0.0)

        if len(self.curChoices) > 0:
            wx.CallLater(1000.0*self.iti, self.runTestEpoch)
        else:
            wx.CallLater(1000.0*self.iti, self.endTest)
