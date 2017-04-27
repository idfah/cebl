import numpy as np
import time
import wx
from wx.lib.agw import aui
import wx.lib.agw.floatspin as agwfs

from cebl import ml
from cebl.rt import widgets

from standard import StandardConfigPanel, StandardMonitorPage


class ConfigPanel(StandardConfigPanel):
    """Panel containing configuration widgets.  This is intimately
    related to this specific page.
    """
    def __init__(self, *args, **kwargs):
        # initialize base class
        StandardConfigPanel.__init__(self, *args, **kwargs)

        self.initTiming()
        self.initLattice()

        self.initStandardLayout()

    def initTiming(self):
        timingSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        widthControlBox = widgets.ControlBox(self, label='Train Width', orient=wx.VERTICAL)
        self.widthFloatSpin = agwfs.FloatSpin(self, min_val=0.5, max_val=60.0,
            increment=0.25, value=self.pg.width)
        self.widthFloatSpin.SetFormat('%f')
        self.widthFloatSpin.SetDigits(3)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setWidth, self.widthFloatSpin)
        self.offlineControls += [self.widthFloatSpin]
        widthControlBox.Add(self.widthFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        timingSizer.Add(widthControlBox, proportion=1,
            flag=wx.ALL | wx.EXPAND, border=10)

        retrainDelayControlBox = widgets.ControlBox(self, label='Retrain Delay', orient=wx.VERTICAL)
        self.retrainDelayFloatSpin = agwfs.FloatSpin(self, min_val=2.0, max_val=60.0,
                increment=0.25, value=self.pg.retrainDelay)
        self.retrainDelayFloatSpin.SetFormat('%f')
        self.retrainDelayFloatSpin.SetDigits(4)
        self.Bind(agwfs.EVT_FLOATSPIN, self.setRetrainDelay, self.retrainDelayFloatSpin)
        self.offlineControls += [self.retrainDelayFloatSpin]
        retrainDelayControlBox.Add(self.retrainDelayFloatSpin, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        timingSizer.Add(retrainDelayControlBox, proportion=1,
            flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=10)

        self.sizer.Add(timingSizer, proportion=0, flag=wx.EXPAND)

    def setWidth(self, event):
        self.pg.width = self.widthFloatSpin.GetValue()

    def setRetrainDelay(self, event):
        self.pg.retrainDelay = self.retrainDelayFloatSpin.GetValue()

    def initLattice(self):
        latticeSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        latticeSizeXControlBox = widgets.ControlBox(self, label='Lattice Size X', orient=wx.VERTICAL)
        self.latticeSizeXSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.latticeSize[0]), min=1, max=512)
        self.Bind(wx.EVT_SPINCTRL, self.setLatticeSizeX, self.latticeSizeXSpinCtrl)
        self.offlineControls += [self.latticeSizeXSpinCtrl]
        latticeSizeXControlBox.Add(self.latticeSizeXSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        latticeSizer.Add(latticeSizeXControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        latticeSizeYControlBox = widgets.ControlBox(self, label='Lattice Size Y', orient=wx.VERTICAL)
        self.latticeSizeYSpinCtrl = wx.SpinCtrl(self, #style=wx.SP_WRAP,
                value=str(self.pg.latticeSize[0]), min=1, max=512)
        self.Bind(wx.EVT_SPINCTRL, self.setLatticeSizeY, self.latticeSizeYSpinCtrl)
        self.offlineControls += [self.latticeSizeYSpinCtrl]
        latticeSizeYControlBox.Add(self.latticeSizeYSpinCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        latticeSizer.Add(latticeSizeYControlBox, proportion=1,
                flag=wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(latticeSizer, proportion=0, flag=wx.EXPAND)

    def setLatticeSizeX(self, event):
        self.pg.latticeSize[0] = self.latticeSizeXSpinCtrl.GetValue()
        self.pg.som = None # force training from scratch

    def setLatticeSizeY(self, event):
        self.pg.latticeSize[1] = self.latticeSizeYSpinCtrl.GetValue()
        self.pg.som = None # force training from scratch

class FreeSOM(StandardMonitorPage):
    def __init__(self, *args, **kwargs):
        self.initConfig()

        StandardMonitorPage.__init__(self, name='FreeSOM',
                configPanelClass=ConfigPanel, *args, **kwargs)

        self.initPlots()
        self.initLayout()

    def initConfig(self):
        """Initialize configuration values.
        """
        self.width = 15.0
        self.retrainDelay = 20.0

        self.latticeSize = [16, 16]
        self.lags = 10
        self.nPoints = 50
        self.maxIter = 5000
        self.distMetric = 'euclidean'
        self.learningRate=0.02
        self.learningRateFinal=0.01
        self.radius = 8
        self.radiusFinal = 1

    def initPlots(self):
        """Initialize a new BMUPlot widgets for EEG and marker.
        """
        self.plot = widgets.BMUPlot(self)

    def initLayout(self):
        self.initStandardLayout()

        # plot pane
        plotPaneAuiInfo = aui.AuiPaneInfo().Name('plot').Caption('SOM Best Matching Unit').CenterPane()
        self.auiManager.AddPane(self.plot, plotPaneAuiInfo)

        self.auiManager.Update()

    def beforeStart(self):
        self.som = None
        self.lastTrainTime = time.time()

    def updatePlot(self, event=None):
        """Draw the BMU plot.
        """

        # get EEG data from current source
        cap = self.src.getEEGSecs(self.width, copy=False)

        data = cap.timeEmbed(lags=self.lags)

        if (time.time() - self.lastTrainTime) > self.retrainDelay:
            progressDialog = wx.ProgressDialog('Training Classifier',
                                    'Training', maximum=self.maxIter // 50 + 1,
                                    style=wx.PD_ELAPSED_TIME | wx.PD_SMOOTH)

            def updateProgressDialog(iteration, weights, learningRate, radius):
                if not (iteration % 50):
                    progressDialog.Update(updateProgressDialog.i, 'Training')
                    updateProgressDialog.i += 1
            updateProgressDialog.i = 0

            if self.som is None:
                self.som = ml.SOM(data, latticeSize=self.latticeSize,
                                  maxIter=self.maxIter, distMetric=self.distMetric,
                                  learningRate=self.learningRate,
                                  learningRateFinal=self.learningRateFinal,
                                  radius=self.radius, radiusFinal=self.radiusFinal,
                                  callback=updateProgressDialog, verbose=False)
            else:
                self.som.callback = updateProgressDialog
                self.som.train(data)

            progressDialog.Destroy()

            self.lastTrainTime = time.time()

        if self.som is not None:
            points = self.som.getBMUIndices(data[-self.nPoints:,:])

        else:
            pointsX = np.round(np.random.uniform(0, self.latticeSize[0]-1, size=self.nPoints))
            pointsY = np.round(np.random.uniform(0, self.latticeSize[1]-1, size=self.nPoints))
            points = np.vstack((pointsX,pointsY)).T

        self.plot.draw(points, latticeSize=self.latticeSize)

    def captureImage(self, event=None):
        self.plot.saveFile()
