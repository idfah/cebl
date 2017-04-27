
# need to check exceptions XXX - idfah

import time
import traceback
import wx
from wx.lib.agw import aui
from wx.lib.scrolledpanel import ScrolledPanel

from page import Page


class StandardConfigPanel(ScrolledPanel):
    def __init__(self, parent, pg, *args, **kwargs):
        """Construct a new panel containing configuration widgets.

        Args:
            parent: Parent in wx hierarchy.

            pg:   Page to be configured.

            *args, **kwargs:  Additional arguments passed
                              to the wx.Panel base class.
        """
        # initialize wx.Panel base class
        ScrolledPanel.__init__(self, parent=parent, *args, **kwargs)

        # page to be configured
        self.pg = pg

        # main sizer
        self.sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.SetupScrolling()#scroll_x=False, scroll_y=True)

        self.offlineControls = []

    def initStandardLayout(self):
        self.Layout()

    def disable(self):
        for ctrl in self.offlineControls:
            ctrl.Disable()

    def enable(self):
        for ctrl in self.offlineControls:
            ctrl.Enable()

class StandardPage(Page):
    def __init__(self, name, configPanelClass=StandardConfigPanel, *args, **kwargs):
        self.configPanelClass = configPanelClass
        Page.__init__(self, name=name, *args, **kwargs)

    def initStandardToolbarControls(self):
        # button to show and hide the config pane
        self.configButton = wx.Button(self.toolbar, label='Configure')
        self.toolbar.AddControl(self.configButton, label='Config')
        self.Bind(wx.EVT_BUTTON, self.toggleShowConfig, self.configButton)

    def initStandardLayout(self):
        self.configPanel = self.configPanelClass(parent=self, pg=self)

        # set up this page to be managed by AUI
        self.auiManager = aui.AuiManager()
        self.auiManager.SetManagedWindow(self)

        # new aui toolbar
        self.toolbar = aui.AuiToolBar(self)

        self.initStandardToolbarControls()

        # realize the new toolbar
        self.toolbar.Realize()

        # add toolbar pane
        toolbarAuiInfo = (aui.AuiPaneInfo().Name('toolbar').Caption(self.name + ' Tools')
            .ToolbarPane().Top().CloseButton(False).LeftDockable(False).RightDockable(False))
        self.auiManager.AddPane(self.toolbar, toolbarAuiInfo)

        # add configPanel pane
        configPaneAuiInfo = (aui.AuiPaneInfo().Name('config').Caption(self.name + ' Configuration')
            .Right().TopDockable(False).BottomDockable(False))

        # setup best size, leave room for scrollbars
        configPanelSizer = self.configPanel.GetSizer()
        if configPanelSizer is not None:
            minSize = configPanelSizer.GetMinSize()

            bestSize = (minSize[0]+wx.SystemSettings_GetMetric(wx.SYS_VSCROLL_X),
                        minSize[1]+wx.SystemSettings_GetMetric(wx.SYS_HSCROLL_Y))
            configPaneAuiInfo.BestSize(bestSize)

        # add config pane to aui manager
        self.auiManager.AddPane(self.configPanel, configPaneAuiInfo)

        # start with config pane hidden
        configPaneAuiInfo.Hide()

        # update aui manager for changes to take effect
        self.auiManager.Update()

    def toggleShowConfig(self, event=None):
        """If the config panel is visible, hide it.
        If it is hidden, show it.

        Args:
            event:  Optional wx.Event.
                    Default is None.
        """
        # get a handle for the config pane
        configPane = self.auiManager.GetPane(self.configPanel)

        # toggle visibility
        configPane.Show(not configPane.IsShown())

        # update AUI to render changes
        self.auiManager.Update()

    def start(self, event=None):
        self.configPanel.disable()
        Page.start(self, event)

    def stop(self, event=None):
        self.configPanel.enable()
        Page.stop(self, event)

class StandardMonitorPage(StandardPage):
    def __init__(self, *args, **kwargs):
        self.paused = False         # pause the plot without stopping data acquisition
        self.refreshDelay = 50.0    # milliseconds between plot updates, does not include draw time
        self.recordingTime = None   # start time for EEG recording, None indicates not started

        StandardPage.__init__(self, *args, **kwargs)

        # wx timer for updating the monitor plot
        self.updateTimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.runUpdatePlot, self.updateTimer)

    def initStandardToolbarControls(self):
        # button to start and stop the page
        self.runButton = wx.Button(self.toolbar, label='Start')
        self.toolbar.AddControl(self.runButton, label='Run')
        self.Bind(wx.EVT_BUTTON, self.toggleRunning, self.runButton)

        # toolbar button to pause without stopping data acquisition
        self.pauseButton = wx.Button(self.toolbar, label='Pause')
        self.pauseButton.Disable()
        self.toolbar.AddControl(self.pauseButton, label='Pause')
        self.Bind(wx.EVT_BUTTON, self.togglePause, self.pauseButton)

        # toolbar button to start and stop EEG recording
        self.recordButton = wx.Button(self.toolbar, label='Start Recording')
        self.recordButton.Disable()
        self.toolbar.AddControl(self.recordButton, label='Record EEG')
        self.Bind(wx.EVT_BUTTON, self.toggleRecord, self.recordButton)

        # toolbar button to save an image of the current plot
        self.captureButton = wx.Button(self.toolbar, label='Capture Image')
        self.toolbar.AddControl(self.captureButton, label='Capture Image')
        self.Bind(wx.EVT_BUTTON, self.captureImage, self.captureButton)

        # init toolbar controls in base class
        StandardPage.initStandardToolbarControls(self)

    def getRefreshDelay(self):
        return self.refreshDelay

    def setRefreshDelay(self, delay):
        self.refreshDelay = delay

    def toggleRunning(self, event=None):
        if self.isRunning(): # stop monitor
            try:
                self.runButton.SetLabel('Start')
                self.pauseButton.Disable()
                self.recordButton.Disable()

                self.stop(self)

                # just let timer expire since we always use oneShot

            except Exception:
                self.runButton.SetLabel('Stop')
                self.pauseButton.Enable()
                self.recordButton.Enable()
                raise

            finally:
                self.unPause()
                self.stopRecording()

        else: # start monitor
            try:
                self.runButton.SetLabel('Stop')
                self.pauseButton.Enable()
                self.recordButton.Enable()

                self.start(self)

                # start update timer
                self.updateTimer.Start(self.refreshDelay, oneShot=True)


            except Exception:
                self.runButton.SetLabel('Start')
                self.pauseButton.Disable()
                self.recordButton.Disable()
                raise

            finally:
                self.unPause()
                self.recordingTime = None
                self.stopRecording()

    def togglePause(self, event=None):
        if self.isRunning():
            if self.paused:
                self.unPause()
            else:
                self.pause()

    def pause(self):
        self.pauseButton.SetLabel('Resume')
        self.paused = True

    def unPause(self):
        self.pauseButton.SetLabel('Pause ')
        self.paused = False

    def toggleRecord(self, event):
        if self.isRunning():
            if self.recordingTime is None:
                self.startRecording()

            else:
                self.stopRecording()

    def startRecording(self):
        if self.isRunning():
            self.recordingTime = time.time()

        self.recordButton.SetLabel('Stop Recording')

    def stopRecording(self):
        if self.recordingTime is not None:
            secsToSave = time.time() - self.recordingTime

            wx.LogMessage('Page %s saving %f secs of EEG' % (self.name, secsToSave))

            cap = self.src.getEEGSecs(secsToSave, filter=False)

            saveDialog = wx.FileDialog(self, message='Save EEG data.',
                wildcard='Pickle (*.pkl)|*.pkl|All Files|*',
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

            try:
                if saveDialog.ShowModal() != wx.ID_CANCEL:
                    cap.saveFile(saveDialog.GetPath())
            except Exception as e:
                wx.LogError('Save failed!')
                raise
            finally:
                saveDialog.Destroy()

        self.recordButton.SetLabel('Start Recording')
        self.recordingTime = None

    def runUpdatePlot(self, event):
        if not self.paused:
            self.updatePlot()

        # if still running
        if self.isRunning():
            # set to run again
            self.updateTimer.Start(self.refreshDelay, oneShot=True)
        else:
            # unPuase if no longer running
            self.unPause()

    def updatePlot(self, event=None):
        raise NotImplementedError('updatePlot not implemented!')

    def captureImage(self, event=None):
        raise NotImplementedError('captureImage not implemented')

class StandardBCIPage(StandardPage):
    def __init__(self, *args, **kwargs):
        self.earlyStopFlag = True
        self.trained = False

        StandardPage.__init__(self, *args, **kwargs)

    def initStandardToolbarControls(self):
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

        # init toolbar controls in base class
        StandardPage.initStandardToolbarControls(self)

    def getSessionTime(self):
        if self.isRunning():
            return time.time() - self.startTime
        else:
            return 0.0

    def isTrained(self):
        return self.trained

    def setTrained(self, trained=True):
        if self.isRunning():
            raise Exception('Cannot change state while running!')

        self.trained = trained
        if self.trained:
            self.retrainButton.Enable()
            self.testButton.Enable()
        else:
            self.retrainButton.Disable()
            self.testButton.Disable()

    def requireRetrain(self):
        if self.isRunning():
            raise Exception('Cannot change state while running!')

        if self.trained:
            self.trained = False
            self.testButton.Disable()
            self.retrainButton.Enable()

    def toggleTrain(self, event=None):
        if self.isRunning():
            self.earlyStopFlag = True
            self.trainButton.Disable()

            wx.LogMessage('%s: Training stopped early.' % self.name)
        else:
            try:
                self.earlyStopFlag = False
                self.setTrained(False)
                self.trainButton.SetLabel('Stop')
                self.start()

                self.startTime = time.time()
                self.beforeTrain()

                wx.CallLater(1000.0*2.0, self.runTrainEpoch)

            except Exception:
                self.earlyStopFlag = True
                self.trainButton.SetLabel('Start')
                raise

    def endTrain(self, event=None):
        if self.isRunning():
            self.afterTrain(False)

            try:
                self.stop()
                self.trainButton.SetLabel('Train')

            except Exception:
                self.trainButton.SetLabel('Stop')
                raise

            try:
                self.trainClassifier()
                self.setTrained(True)

            except Exception:
                self.setTrained(False)
                raise

    def beforeTrain(self):
        pass

    def afterTrain(self, earlyStop):
        pass

    def retrain(self, event=None):
        try:
            self.trainClassifier()
            self.setTrained(True)
        except Exception:
            self.setTrained(False)
            raise

    def trainClassifier(self):
        raise NotImplementedException('trainClassifier is not implemented!')

    def runTrainEpoch(self, event=None):
        if self.earlyStopFlag:
            self.afterTrain(self.earlyStopFlag)

            try:
                self.stop()
                self.trainButton.SetLabel('Train')

            except Exception:
                self.trainButton.SetLabel('Stop')
                raise

            finally:
                self.trainButton.Enable()
                self.setTrained(False)

        else:
            self.trainEpoch()

    def trainEpoch(self):
        raise NotImplementedError('trainEpoch not implemented!')

    def toggleTest(self, event=None):
        if self.isRunning():
            self.earlyStopFlag = True
            self.testButton.Disable()

            wx.LogMessage('%s: Testing stopped early.' % self.name)
        else:
            try:
                self.earlyStopFlag = False
                self.trainButton.Disable()
                self.retrainButton.Disable()
                self.testButton.SetLabel('Stop')
                self.start()

                self.startTime = time.time()
                self.beforeTest()

                wx.CallLater(1000.0*2.0, self.runTestEpoch)

            except Exception:
                self.earlyStopFlag = True
                self.trainButton.Enable()
                self.retrainButton.Enable()
                self.testButton.SetLabel('Test')
                raise

    def endTest(self, event=None):
        if self.isRunning():
            self.afterTest(False)

            try:
                self.stop()
                self.trainButton.Enable()
                self.retrainButton.Enable()
                self.testButton.SetLabel('Test')

            except Exception:
                self.trainButton.Disable()
                self.retrainButton.Disable()
                self.testButton.SetLabel('Stop')
                raise

            finally:
                self.testButton.Enable()

            self.testClassifier()

    def beforeTest(self):
        pass

    def afterTest(self, earlyStop):
        pass

    def testClassifier(self):
        pass

    def runTestEpoch(self, event=None):
        if self.earlyStopFlag:
            self.afterTest(self.earlyStopFlag)

            try:
                self.stop()
                self.trainButton.Enable()
                self.retrainButton.Enable()
                self.testButton.SetLabel('Test')

            except Exception:
                self.trainButton.Disable()
                self.retrainButton.Disable()
                self.testButton.SetLabel('Stop')

            finally:
                self.testButton.Enable()

        else:
            self.testEpoch()

    def testEpoch(self):
        raise NotImplementedError('trainEpoch not implemented!')

    def afterUpdateSource(self):
        self.setTrained(False)

    def close(self, event=None):
        if self.isRunning():
            self.earlyStopFlag = True
            self.trainButton.Disable()
            self.retrainButton.Disable()
            self.testButton.Disable()

        if event is not None:
            event.Skip()
