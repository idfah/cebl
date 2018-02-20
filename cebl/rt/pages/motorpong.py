import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
import wx
from wx.lib.agw import aui
import wx.lib.agw.floatspin as agwfs

from cebl import ml
from cebl import util
from cebl.rt import widgets

# StandardPage only needed for toolbar setup XXX - idfah
from .standard import StandardConfigPanel, StandardBCIPage, StandardPage


class WelchConfigPanel(wx.Panel):
    def __init__(self, parent, pg, cp, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.pg = pg
        self.cp = cp

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.initFeatures()
        self.initFreqs()
        self.Layout()

    def initFeatures(self):
        featureSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        spanControlBox = widgets.ControlBox(self, label='Welch Span', orient=wx.VERTICAL)
        self.spanFloatSpin = agwfs.FloatSpin(self, min_val=0.1, max_val=3.0,
            increment=0.05, value=self.pg.welchConfig.span)
        self.spanFloatSpin.SetFormat('%f')
        self.spanFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setSpan, self.spanFloatSpin)
        self.cp.offlineControls += [self.spanFloatSpin]
        spanControlBox.Add(self.spanFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        featureSizer.Add(spanControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        
        # radio buttons for turning log transform on and off
        logTransControlBox = widgets.ControlBox(self, label='Log Trans', orient=wx.HORIZONTAL)

        logTransOnRbtn = wx.RadioButton(self, label='On', style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, self.setLogTransOn, logTransOnRbtn)
        logTransControlBox.Add(logTransOnRbtn, proportion=0, flag=wx.ALL, border=10)
        self.cp.offlineControls += [logTransOnRbtn]

        logTransOffRbtn = wx.RadioButton(self, label='Off')
        self.Bind(wx.EVT_RADIOBUTTON, self.setLogTransOff, logTransOffRbtn)
        logTransControlBox.Add(logTransOffRbtn, proportion=0, flag=wx.ALL, border=10)
        self.cp.offlineControls += [logTransOffRbtn]

        if self.pg.welchConfig.logTrans:
            logTransOnRbtn.SetValue(True)
        else:
            logTransOffRbtn.SetValue(True)

        featureSizer.Add(logTransControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(featureSizer, proportion=0, flag=wx.EXPAND)

    def setSpan(self, event):
        self.pg.welchConfig.span = self.spanFloatSpin.GetValue()
        self.pg.requireRetrain()

    def setLogTransOn(self, event=None):
        self.pg.welchConfig.logTrans = True

    def setLogTransOff(self, event=None):
        self.pg.welchConfig.logTrans = False

    def initFreqs(self):
        lowHighFreqSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        lowFreqControlBox = widgets.ControlBox(self, label='Low Freq', orient=wx.VERTICAL)
        self.lowFreqFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=100.0,
                increment=1/4.0, value=self.pg.welchConfig.lowFreq)
        self.lowFreqFloatSpin.SetFormat('%f')
        self.lowFreqFloatSpin.SetDigits(4)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setLowFreq, self.lowFreqFloatSpin)
        self.cp.offlineControls += [self.lowFreqFloatSpin]
        lowFreqControlBox.Add(self.lowFreqFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        lowHighFreqSizer.Add(lowFreqControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        highFreqControlBox = widgets.ControlBox(self, label='High Freq', orient=wx.VERTICAL)

        self.highFreqFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=100.0,
             increment=1/4.0, value=self.pg.welchConfig.highFreq)
        self.highFreqFloatSpin.SetFormat('%f')
        self.highFreqFloatSpin.SetDigits(4)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setHighFreq, self.highFreqFloatSpin)
        self.cp.offlineControls += [self.highFreqFloatSpin]
        highFreqControlBox.Add(self.highFreqFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        lowHighFreqSizer.Add(highFreqControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(lowHighFreqSizer, proportion=0, flag=wx.EXPAND)

    def setLowFreq(self, event):
        self.pg.welchConfig.lowFreq = self.lowFreqFloatSpin.GetValue()
        self.pg.requireRetrain()

    def setHighFreq(self, event):
        self.pg.welchConfig.highFreq = self.highFreqFloatSpin.GetValue()
        self.pg.requireRetrain()

class AutoregConfigPanel(wx.Panel):
    def __init__(self, parent, pg, cp, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.pg = pg
        self.cp = cp

        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.Layout()

class ConfigPanel(StandardConfigPanel):
    def __init__(self, *args, **kwargs):
        StandardConfigPanel.__init__(self, *args, **kwargs)

        self.initChoices()
        self.initNTrial()
        self.initTrialSecs()
        self.initTiming()
        self.initGainLoss()
        self.initMethod()
        self.initLayout()

    def initChoices(self):
        choiceControlBox = widgets.ControlBox(self, label='Choices', orient=wx.VERTICAL)

        choiceGridSizer = wx.GridSizer(1, 2, 10, 10)

        self.leftChoiceTextCtrl = wx.TextCtrl(parent=self, value=self.pg.choices[0],
                style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.setChoices, self.leftChoiceTextCtrl)
        self.leftChoiceTextCtrl.Bind(wx.EVT_KILL_FOCUS, self.setChoices, self.leftChoiceTextCtrl)
        self.offlineControls += [self.leftChoiceTextCtrl]
        choiceGridSizer.Add(self.leftChoiceTextCtrl, proportion=0,
                             flag=wx.ALL | wx.EXPAND, border=10)

        self.rightChoiceTextCtrl = wx.TextCtrl(parent=self, value=self.pg.choices[1],
                style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.setChoices, self.rightChoiceTextCtrl)
        self.rightChoiceTextCtrl.Bind(wx.EVT_KILL_FOCUS, self.setChoices, self.rightChoiceTextCtrl)
        self.offlineControls += [self.rightChoiceTextCtrl]
        choiceGridSizer.Add(self.rightChoiceTextCtrl, proportion=0,
                             flag=wx.ALL | wx.EXPAND, border=10)

        choiceControlBox.Add(choiceGridSizer, proportion=1,
                             flag=wx.ALL | wx.EXPAND, border=0)
        
        self.sizer.Add(choiceControlBox, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

    def setChoices(self, event):
        choices = [self.leftChoiceTextCtrl.GetValue(), self.rightChoiceTextCtrl.GetValue()]

        if len(choices) < 2:
            wx.LogError('Page %s: Cannot use less than 2 choices.' % self.name)
        else:
            self.pg.choices = choices
            self.pg.confusion = np.zeros((len(choices), len(choices)))
            self.pg.pieMenu.setChoices(choices)
            self.pg.setTrained(False)

        #event.Skip()

    def initNTrial(self):
        trialSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        trainTrialControlBox = widgets.ControlBox(self, label='Train Trials', orient=wx.VERTICAL)
        self.trainTrialSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.nTrainTrial), min=2, max=100)
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

        trainTrialSecsControlBox = widgets.ControlBox(self,
                label='Train Trial Secs', orient=wx.VERTICAL)

        self.trainTrialSecsFloatSpin = agwfs.FloatSpin(self, min_val=2.00, max_val=60.0,
                increment=1/4.0, value=self.pg.trainTrialSecs)
        self.trainTrialSecsFloatSpin.SetFormat('%f')
        self.trainTrialSecsFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setTrialSecs, self.trainTrialSecsFloatSpin)
        self.offlineControls += [self.trainTrialSecsFloatSpin]
        trainTrialSecsControlBox.Add(self.trainTrialSecsFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        secsSizer.Add(trainTrialSecsControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        pauseSecsControlBox = widgets.ControlBox(self, label='Pause Secs', orient=wx.VERTICAL)
        self.pauseSecsFloatSpin = agwfs.FloatSpin(self, min_val=0.25, max_val=10.0,
                increment=1/4.0, value=self.pg.pauseSecs)
        self.pauseSecsFloatSpin.SetFormat('%f')
        self.pauseSecsFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setPauseSecs, self.pauseSecsFloatSpin)
        self.offlineControls += [self.pauseSecsFloatSpin]
        pauseSecsControlBox.Add(self.pauseSecsFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        secsSizer.Add(pauseSecsControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(secsSizer, proportion=0, flag=wx.EXPAND)

    def setTrialSecs(self, event):
        self.pg.trainTrialSecs = self.trainTrialSecsFloatSpin.GetValue()
        self.pg.setTrained(False)

    def setPauseSecs(self, event):
        self.pg.pauseSecs = self.pauseSecsFloatSpin.GetValue()
        self.pg.setTrained(False)

    def initTiming(self):
        timingSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        widthControlBox = widgets.ControlBox(self, label='Width Secs', orient=wx.VERTICAL)
        self.widthFloatSpin = agwfs.FloatSpin(self, min_val=0.2, max_val=5.0,
            increment=0.05, value=self.pg.width)
        self.widthFloatSpin.SetFormat('%f')
        self.widthFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setWidth, self.widthFloatSpin)
        self.offlineControls += [self.widthFloatSpin]
        widthControlBox.Add(self.widthFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        timingSizer.Add(widthControlBox, proportion=1,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        decisionSecsControlBox = widgets.ControlBox(self, label='Decision Secs', orient=wx.VERTICAL)
        self.decisionSecsFloatSpin = agwfs.FloatSpin(self, min_val=0.025, max_val=5.0,
                increment=0.025, value=self.pg.decisionSecs)
        self.decisionSecsFloatSpin.SetFormat('%f')
        self.decisionSecsFloatSpin.SetDigits(4)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setDecisionSecs, self.decisionSecsFloatSpin)
        self.offlineControls += [self.decisionSecsFloatSpin]
        decisionSecsControlBox.Add(self.decisionSecsFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        timingSizer.Add(decisionSecsControlBox, proportion=1,
            flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(timingSizer, proportion=0, flag=wx.EXPAND)

    def setDecisionSecs(self, event):
        self.pg.decisionSecs = self.decisionSecsFloatSpin.GetValue()
        self.pg.initOverlap()
        self.pg.requireRetrain()

    def setWidth(self, event):
        self.pg.width = self.widthFloatSpin.GetValue()
        self.pg.initOverlap()
        self.pg.requireRetrain()

    def initGainLoss(self):
        gainLossSizer = wx.BoxSizer(orient=wx.VERTICAL)

        gainControlBox = widgets.ControlBox(self, label='Gain', orient=wx.HORIZONTAL)
        self.gainText = wx.StaticText(self, label='%0.2f' % self.pg.gain)
        gainTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        gainTextSizer.Add(self.gainText, proportion=1, flag=wx.EXPAND)
        self.gainSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.gain*100.0), minValue=1, maxValue=100)
        self.Bind(wx.EVT_SLIDER, self.setGain, self.gainSlider)
        gainControlBox.Add(gainTextSizer, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)
        gainControlBox.Add(self.gainSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        gainLossSizer.Add(gainControlBox,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        lossControlBox = widgets.ControlBox(self, label='Loss', orient=wx.HORIZONTAL)
        self.lossText = wx.StaticText(self, label='%0.2f' % self.pg.loss)
        lossTextSizer = wx.BoxSizer(orient=wx.VERTICAL)
        lossTextSizer.Add(self.lossText, proportion=1, flag=wx.EXPAND)
        self.lossSlider = wx.Slider(self, style=wx.SL_HORIZONTAL,
            value=int(self.pg.loss*100.0), minValue=1, maxValue=100)
        self.Bind(wx.EVT_SLIDER, self.setLoss, self.lossSlider)
        lossControlBox.Add(lossTextSizer, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)
        lossControlBox.Add(self.lossSlider, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        gainLossSizer.Add(lossControlBox,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(gainLossSizer, proportion=0, flag=wx.EXPAND)

    def setGain(self, event):
        self.pg.gain = self.gainSlider.GetValue() / 100.0
        self.gainText.SetLabel('%0.2f' % self.pg.gain)

    def setLoss(self, event):
        self.pg.loss= self.lossSlider.GetValue() / 100.0
        self.lossText.SetLabel('%0.2f' % self.pg.loss)

    def initMethod(self):
        methodControlBox = widgets.ControlBox(self, label='Method', orient=wx.VERTICAL)
        self.methodComboBox = wx.ComboBox(self, value=self.pg.method,
                style=wx.CB_READONLY, choices=('Welch Power', 'Autoregressive'))
        self.Bind(wx.EVT_COMBOBOX, self.setMethod, self.methodComboBox)
        self.offlineControls += [self.methodComboBox]
        methodControlBox.Add(self.methodComboBox, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.sizer.Add(methodControlBox, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.methodConfigSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.welchPanel = WelchConfigPanel(self, pg=self.pg, cp=self)
        self.methodConfigSizer.Add(self.welchPanel, proportion=1, flag=wx.EXPAND)

        self.autoregPanel = AutoregConfigPanel(self, pg=self.pg, cp=self)
        self.methodConfigSizer.Add(self.autoregPanel, proportion=1, flag=wx.EXPAND)

        self.sizer.Add(self.methodConfigSizer, proportion=1, flag=wx.EXPAND)

        self.methodConfig = self.welchPanel

    def setMethod(self, event):
        method = self.methodComboBox.GetValue()
        self.method = method

        self.methodConfig.Hide()
        if method == 'Welch Power':
            self.methodConfig = self.welchPanel
        elif method == 'Autoregressive':
            self.methodConfig = self.autoregPanel
        else:
            raise Exception('Unknown method: ' + str(method))
        self.methodConfig.Show()

        self.FitInside()
        self.pg.requireRetrain()

    def initLayout(self):
        self.initStandardLayout() 

        self.FitInside()
        self.autoregPanel.Hide()
        self.FitInside()

class PlotPanel(wx.Panel):
    def __init__(self, parent, pg, choices, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.pg = pg

        self.initPieMenu(choices)
        self.initFeatureCanvas()
        self.initPongGame()
        self.initLayout()

    def initPieMenu(self, choices):
        self.pieMenu = widgets.PieMenu(self, choices=choices, rotation=-np.pi/2.0,
                #colors=('red', (50,220,50), 'yellow', 'blue'))
                colors=('turquoise', 'red', 'blue violet', 'orange',
                        'blue', 'yellow'))

    def initFeatureCanvas(self):
        self.featureFig = plt.Figure()
        ##self.featureFig.subplots_adjust(hspace=0.32, wspace=0.02,
        ##    left=0.065, right=0.95, top=0.97, bottom=0.18)

        self.featureAx = self.featureFig.add_subplot(1,1,1)

        self.featureCanvas = FigureCanvas(parent=self, id=wx.ID_ANY, figure=self.featureFig)

    def initPongGame(self):
        self.pongGame = widgets.Pong(self) 
        
        # for some reason pongGame.Hide() is not called in showPieMenu. I don't know why
        self.pongGame.Hide()

    def initLayout(self):
        plotSizer = wx.BoxSizer(orient=wx.VERTICAL)

        plotSizer.Add(self.pieMenu, proportion=1, flag=wx.EXPAND | wx.ALL)
        plotSizer.Add(self.featureCanvas, proportion=1, flag=wx.EXPAND | wx.ALL)
        plotSizer.Add(self.pongGame, proportion=1, flag=wx.EXPAND | wx.ALL)

        self.SetSizer(plotSizer)

        self.featureCanvas.Hide()
        self.Layout()

    def showPieMenu(self):
        self.featureCanvas.Hide()
        self.pongGame.Hide()
        self.pieMenu.Show()
        self.Layout()

    def showFeatureCanvas(self):
        self.featureCanvas.Show()
        self.pongGame.Hide()
        self.pieMenu.Hide()
        self.Layout()

    def showPong(self):
        self.featureCanvas.Hide()
        self.pongGame.Show()
        self.pieMenu.Hide()
        self.Layout()

    def plotFeatures(self, trainData, freqs, choices, chanNames):
        self.featureAx.cla()

        meanFeat = [np.mean(cls, axis=0) for cls in trainData]
        for cls, choice in zip(meanFeat, choices):
            self.featureAx.plot(cls, label=choice, marker='o', linewidth=2)

        self.featureAx.set_xlabel(r'Frequency ($Hz$)')
        self.featureAx.set_ylabel(r'Mean Log$_{10}$ Power $(uV^2 / Hz)^{\frac{1}{2}}$')
        self.featureAx.legend()

        nFreq = len(freqs)
        mn = np.min(np.concatenate(meanFeat))
        mx = np.max(np.concatenate(meanFeat))
        for i,cn in enumerate(chanNames):
            if i > 0:
                self.featureAx.vlines(i*float(nFreq), mn, mx, linestyle='--')
            self.featureAx.text((i+0.25)*float(nFreq), mx-0.1*(mx-mn), cn, fontsize=14)

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

    def toggleGame(self, event=None):
        if self.pg.gameActive:
            self.showPieMenu()
            self.pg.gameActive = False
        else:
            self.showPong()
            self.pg.gameActive = True

class MotorPong(StandardBCIPage):
    def __init__(self, *args, **kwargs):
        self.initConfig()

        StandardBCIPage.__init__(self, name='MotorPong',
            configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initStandardToolbarControls(self):

        ## hack alert, need a better way to configure order of buttons in StandardPage XXX - idfah

        # button to start training process
        self.trainButton = wx.Button(self.toolbar, label='Train')
        self.toolbar.AddControl(self.trainButton, label='Train')
        self.Bind(wx.EVT_BUTTON, self.toggleTrain, self.trainButton)
        
        # button to re-train classifier
        self.retrainButton = wx.Button(self.toolbar, label='Retrain')
        self.retrainButton.Disable()
        self.toolbar.AddControl(self.retrainButton, label='Retrain')
        self.Bind(wx.EVT_BUTTON, self.retrain, self.retrainButton)

        # button to start and stop the interface
        self.testButton = wx.Button(self.toolbar, label='Test')
        self.testButton.Disable()
        self.toolbar.AddControl(self.testButton, label='Test')
        self.Bind(wx.EVT_BUTTON, self.toggleTest, self.testButton)

        # button to start put into play mode
        self.playButton = wx.Button(self.toolbar, label='Play')
        self.playButton.Disable()
        self.toolbar.AddControl(self.playButton, label='Play')
        self.Bind(wx.EVT_BUTTON, self.plotPanel.toggleGame, self.playButton)

        # init toolbar controls in base class
        StandardPage.initStandardToolbarControls(self)

    def setTrained(self, trained=True):
        StandardBCIPage.setTrained(self, trained)

        if self.trained:
            self.playButton.Enable()
        else:
            self.playButton.Disable()

    def requireRetrain(self):
        StandardBCIPage.requireRetrain(self)

        if not self.trained:
            self.playButton.Disable

    def initConfig(self):
        self.nTrainTrial = 10
        self.trainTrialSecs = 5.0
        self.pauseSecs = 1.0

        self.nTestTrial = 5

        self.width = 1.0
        self.decisionSecs = 0.5
        self.initOverlap()

        self.gain = 0.25
        self.loss = 0.2

        self.method = 'Welch Power'

        self.choices = ['Left', 'Right']

        self.welchConfig = util.Holder(
            classifierKind = 'Linear Discrim',
            span = 0.5,
            logTrans = True,

            lowFreq = 0.5,
            highFreq = 40.0
        )

        # autoregression config
        self.autoregConfig = util.Holder(
            horizon = 1
        )

        self.classifier = None
        self.trainCap = None
        self.confusion = np.zeros((len(self.choices), len(self.choices)))

        self.gameActive = False

    def initOverlap(self):
        self.overlap = 1.0 - (self.decisionSecs/float(self.width))

    def initPlots(self):
        self.plotPanel = PlotPanel(self, pg=self, choices=self.choices)
        self.pieMenu = self.plotPanel.pieMenu
        self.pongGame = self.plotPanel.pongGame

    def initLayout(self):
        self.initStandardLayout()

        stimPaneAuiInfo = aui.AuiPaneInfo().Name('stim').Caption(self.name + ' Stimulus').CenterPane()
        self.auiManager.AddPane(self.plotPanel, stimPaneAuiInfo)

        self.auiManager.Update()

    def beforeStart(self):
        self.configPanel.disable()

    def afterStop(self):
        self.configPanel.enable()

    def markToStim(self, mark):
        index = int(mark%10) - 1
        return self.choices[index]

    def getCap(self, secs):
        cap = self.src.getEEGSecs(secs)

        # hack, should be handled in filter chain XXX - idfah
        if cap.getSampRate() > 256.0:
            cap = cap.demean().bandpass(0.0, 100.0, order=3)
            decimationFactor = int(np.round(cap.getSampRate()/256.0))
            cap = cap.downsample(decimationFactor)
        #cap.car()
        #cap.bandpass(0.5, 40).downsample(3)

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
        # generate PSD object for each seg in each class
        psds = [cls.psd(method='welch', span=self.welchConfig.span) for cls in segs]
        #psds = [cls.psd(method='raw') for cls in segs]

        # get freqencies, should be same for all classes
        freqs = psds[0][0].getFreqs()

        # mask where freqencies fall between lowFreq and highFreq
        freqMask = (freqs > self.welchConfig.lowFreq) & (freqs < self.welchConfig.highFreq)

        freqs = freqs[freqMask]

        # extract powers into a single matrix for each class
        powers = [np.array([segPsd.getPowers().T for segPsd in cls]) for cls in psds]

        # grab powers between lowFreq and highFreq
        powers = [cls[:,:,freqMask] for cls in powers]

        if self.welchConfig.logTrans:
            # use log power to put near amplitude units
            powers = [np.log10(cls) for cls in powers]

        # embed channels
        powers = [cls.reshape(cls.shape[0], -1) for cls in powers]

        return freqs, powers

    def beforeTrain(self):
        self.curTrial = 0
        self.curChoices = []

        self.src.setMarker(0.0)

        self.plotPanel.showPieMenu()

    def afterTrain(self, earlyStop):
        if not earlyStop:
            self.trainCap = self.getCap(self.getSessionTime())
            self.saveCap()

    def trainEpoch(self):
        if len(self.curChoices) == 0:
            self.curChoices = copy.copy(self.choices)
            np.random.shuffle(self.curChoices)
            self.curTrial += 1

        choice = self.curChoices.pop()
        self.pieMenu.highlight(choice, style='pop')

        self.src.setMarker(self.choices.index(choice)+1.0)

        wx.CallLater(1000.0*self.trainTrialSecs, self.trainClearTrial)

    def trainClearTrial(self, event=None):
        self.pieMenu.clearAllHighlights()

        self.src.setMarker(0.0)

        if self.curTrial == self.nTrainTrial and len(self.curChoices) == 0:
            wx.CallLater(1000.0*self.pauseSecs, self.endTrain)
        else:
            wx.CallLater(1000.0*self.pauseSecs, self.runTrainEpoch)

    def trainClassifier(self):
        if self.trainCap is None:
            raise Exception('No data available for training.')

        nFold = self.nTrainTrial
        dialog = wx.ProgressDialog('Training Classifier',
                    'Featurizing', maximum=nFold+1,
                    style=wx.PD_ELAPSED_TIME | wx.PD_SMOOTH)

        segmented = self.trainCap.segment(start=0.0, end=self.trainTrialSecs)
        segs = [segmented.select(matchFunc=lambda mark: self.markToStim(mark) == choice)
                for choice in self.choices]

        assert segs[0].getNSeg() == self.nTrainTrial
        assert segs[1].getNSeg() == self.nTrainTrial

        # split segments
        segs = [cls.split(self.width, self.overlap) for cls in segs]

        ##print('nSplit segs: ', segs[0].getNSeg())

        if self.method == 'Welch Power':
            self.trainWelch(segs, dialog)
        elif self.method == 'Autoregressive':
            self.trainAutoreg(segs, dialog)
        else:
            raise Exception('Invalid method: %s.' % str(self.method))

        if self.gameActive:
            self.plotPanel.showPong()
        else:
            self.plotPanel.showPieMenu()

    def trainWelch(self, segs, dialog):
        freqs, trainData = self.powerize(segs)

        self.plotPanel.plotFeatures(trainData, freqs, self.choices,
                                    self.trainCap.getChanNames())

        if self.welchConfig.classifierKind == 'Linear Discrim':
            self.trainWelchLDA(trainData, dialog)

        elif self.welchConfig.classifierKind == 'Neural Net':
            self.trainWelchNN(trainData, dialog)

        else:
            raise Exception('Invalid classifier kind: %s.' % str(self.welchConfig.classifierKind))

    def trainWelchLDA(self, trainData, dialog):
        self.stand = ml.ClassStandardizer(trainData)
        trainDataStd = self.stand.apply(trainData)

        #penalties = np.insert(np.power(10.0, np.linspace(-3.0, 0.0, 50)), 0, 0.0)
        #penalties = np.linspace(0.0, 1.0, 51)
        penalties = np.linspace(0.00001, 1.0, 51) # XXX - idfah

        nFold = self.nTrainTrial

        trnCA = np.zeros((nFold, penalties.size))
        valCA = np.zeros((nFold, penalties.size))

        for fold, trnData, valData in ml.part.classStratified(trainDataStd, nFold=nFold):
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            for i, penalty in enumerate(penalties):
                classifier = ml.LDA(trnData, shrinkage=penalty)

                trnCA[fold,i] = classifier.ca(trnData)
                valCA[fold,i] = classifier.ca(valData)

        dialog.Update(nFold, 'Training Final Classifier')

        meanTrnCA = np.mean(trnCA, axis=0)
        meanValCA = np.mean(valCA, axis=0)

        bestPenaltyIndex = np.argmax(meanValCA)
        bestPenalty = penalties[bestPenaltyIndex]

        bestMeanTrnCA = meanTrnCA[bestPenaltyIndex]
        bestMeanValCA = meanValCA[bestPenaltyIndex]

        self.classifier = ml.LDA(trainDataStd, shrinkage=bestPenalty)

        trainCA = self.classifier.ca(trainDataStd)
        trainConfusion = np.round(100*self.classifier.confusion(trainDataStd))

        dialog.Destroy()

        wx.MessageBox(message=('Best Shrinkage: %f\n' % bestPenalty) +
            ('Best Mean Training CA: %f\n' % bestMeanTrnCA) +
            ('Best Mean Validation CA: %f\n' % bestMeanValCA) +
            ('Final Training CA: %f\n' % trainCA) +
            ('Confusion Matrix:\n' + str(trainConfusion)),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def trainWelchNN(self, trainData, dialog):
        maxIter = 250
        nHidden = 10
        seed = np.random.randint(0, 1000000)

        self.stand = ml.ClassStandardizer(trainData)
        trainDataStd = self.stand.apply(trainData)

        nFold = self.nTrainTrial

        trnCA = np.zeros((nFold, maxIter+1))
        valCA = np.zeros((nFold, maxIter+1))

        for fold, trnData, valData in ml.part.classStratified(trainDataStd, nFold=nFold):
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
                                nHidden=nHidden, maxIter=maxIter, optimFunc=ml.optim.scg,
                                callback=valTraceCB, eTrace=True, verbose=False)

        dialog.Update(nFold, 'Training Final Classifier')

        meanTrnCA = np.mean(trnCA, axis=0)
        meanValCA = np.mean(valCA, axis=0)

        bestIter = np.argmax(meanValCA)

        bestMeanTrnCA = meanTrnCA[bestIter]
        bestMeanValCA = meanValCA[bestIter]

        np.random.seed(seed)
        self.classifier = ml.FNS(trainDataStd, accuracy=0.0, precision=0.0,
                                 nHidden=nHidden, maxIter=bestIter, optimFunc=ml.optim.scg,
                                 eTrace=False, verbose=False)

        trainCA = self.classifier.ca(trainDataStd)
        trainConfusion = np.round(100*self.classifier.confusion(trainDataStd))

        dialog.Destroy()

        wx.MessageBox(message=('Best Num Iterations: %f\n' % bestIter) +
            ('Best Mean Training CA: %f\n' % bestMeanTrnCA) +
            ('Best Mean Validation CA: %f\n' % bestMeanValCA) +
            ('Final Training CA: %f\n' % trainCA) +
            ('Confusion Matrix:\n' + str(trainConfusion)),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def trainAutoreg(self, segs, dialog):
        trainData = [seg.data for seg in segs]

        self.trainAutoregRR(trainData, dialog)

    def trainAutoregRR(self, trainData, dialog):
        self.stand = ml.ClassSegStandardizer(trainData)
        trainDataStd = self.stand.apply(trainData)

        orders = np.arange(2,30)

        nFold = self.nTrainTrial

        trnCA = np.zeros((nFold, orders.size))
        valCA = np.zeros((nFold, orders.size))

        for fold, trnData, valData in ml.part.classStratified(trainDataStd, nFold=nFold):
            dialog.Update(fold, 'Validation Fold: %d' % fold)

            for i, order in enumerate(orders):
                classifier = ml.ARC(trnData, order=order)

                trnCA[fold,i] = classifier.ca(trnData)
                valCA[fold,i] = classifier.ca(valData)

        dialog.Update(nFold, 'Training Final Classifier')

        meanTrnCA = np.mean(trnCA, axis=0)
        meanValCA = np.mean(valCA, axis=0)

        bestOrderIndex = np.argmax(meanValCA)
        bestOrder = orders[bestOrderIndex]

        bestMeanTrnCA = meanTrnCA[bestOrderIndex]
        bestMeanValCA = meanValCA[bestOrderIndex]

        self.classifier = ml.ARC(trainDataStd, order=bestOrder)

        trainCA = self.classifier.ca(trainDataStd)
        trainConfusion = np.round(100*self.classifier.confusion(trainDataStd))

        dialog.Destroy()

        wx.MessageBox(message=('Best Order: %f\n' % bestOrder) +
            ('Best Mean Training CA: %f\n' % bestMeanTrnCA) +
            ('Best Mean Validation CA: %f\n' % bestMeanValCA) +
            ('Final Training CA: %f\n' % trainCA) +
            ('Confusion Matrix:\n' + str(trainConfusion)),
            caption='Training Completed!', style=wx.OK | wx.ICON_INFORMATION)

    def beforeTest(self):
        self.curTrial = 0
        self.curChoices = []
        self.curDecision = -1

        self.confusion[...] = 0.0

        self.src.setMarker(0.0)

        if self.gameActive:
            self.pongGame.newGame()

    def afterTest(self, earlyStop):
        self.pieMenu.zeroBars(refresh=False)
        self.pieMenu.clearAllHighlights()
        self.pongGame.stopGame()

        if not earlyStop:
            self.saveCap()
            ca = np.mean(np.diag(self.confusion))/self.nTestTrial
            confusion = np.round(100*self.confusion/self.nTestTrial)
            wx.MessageBox(('Test Selection CA: %f\n' % ca) +
                           'Confusion Matrix:\n' + str(confusion),
                           'Testing Complete', wx.OK | wx.ICON_INFORMATION)

    def testEpoch(self):
        if not self.gameActive and self.curDecision == -1:
            self.src.setMarker(0.0)
            self.highlightTestTarget()
            self.curDecision += 1
            wx.CallLater(1000.0*self.pauseSecs*1.1, self.runTestEpoch)

        else:
            wx.CallLater(1000.0*self.decisionSecs*1.1, self.testClassify)

    def highlightTestTarget(self):
        if len(self.curChoices) == 0:
            self.curChoices = copy.copy(self.choices)
            np.random.shuffle(self.curChoices)
            self.curTrial += 1

        self.curChoice = self.curChoices.pop()
        self.pieMenu.highlight(self.curChoice, style='pop')

        self.src.setMarker(10.0*(self.choices.index(self.curChoice)+1.0))

    def testClassify(self):
        testCap = self.getCap(self.width)

        seg = testCap.segmentSingle()

        if self.method == 'Welch Power':
            freqs, testData = self.powerize((seg,))

        elif self.method == 'Autoregressive':
            testData = (seg.data,)

        testDataStd = self.stand.apply(testData)[0]

        #label = self.classifier.label(testDataStd)
        probs = self.classifier.probs(testDataStd)[0]
        #label = np.argmax(probs)
        #selection = self.choices[label]

        if self.gameActive:
            self.gameDecision(probs)

        else:
            self.testDecision(probs)

    def testDecision(self, probs):
        for i,choice in enumerate(self.choices):
            self.pieMenu.growBar(choice, self.gain*(probs[i]-self.loss), refresh=False)
        self.pieMenu.refresh()

        self.curDecision += 1

        finalSelection = self.pieMenu.getSelection()

        if finalSelection is None:
            wx.CallAfter(self.runTestEpoch)

        else:
            self.pieMenu.clearAllHighlights(refresh=False)
            self.pieMenu.highlight(finalSelection, style='jump', secs=self.pauseSecs)
            finalLabel = self.choices.index(finalSelection)
            self.src.incrementMarker(finalLabel+1)
            self.confusion[finalLabel, self.choices.index(self.curChoice)] += 1.0

            wx.CallLater(1000.0*self.pauseSecs, self.testClearTrial)

    def testClearTrial(self):
        self.pieMenu.zeroBars(refresh=False)
        self.pieMenu.clearAllHighlights()
        self.curDecision = -1

        if self.curTrial == self.nTestTrial and len(self.curChoices) == 0:
            wx.CallLater(1000.0*self.pauseSecs, self.endTest)
        else:
            wx.CallLater(1000.0*self.pauseSecs, self.runTestEpoch)

    def gameDecision(self, probs):
        if probs[0] > probs[1]:
            self.pongGame.movePaddleLeft()
        else:
            self.pongGame.movePaddleRight()

        if self.decisionSecs > 0.25:
            wx.CallLater(1000.0*self.decisionSecs*0.85, self.pongGame.stopPaddle)

        if np.sum(self.pongGame.getScore()) < 10:
            wx.CallLater(1000.0*self.decisionSecs*1.1, self.runTestEpoch)

        else:
            wx.CallLater(1000.0*self.pauseSecs, self.endTest)
