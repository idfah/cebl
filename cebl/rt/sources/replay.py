"""Source to replay EEG saved to file.
"""
import multiprocessing as mp
import numpy as np
import time
import wx

from cebl import eeg
from cebl import util
from cebl.rt import widgets

from .source import Source, SourceConfigPanel


class ReplayConfigPanel(SourceConfigPanel):
    """wx panel containing controls for configuring replay signal sources.
    """
    def __init__(self, parent, src, *args, **kwargs):
        """Construct a new configuration panel for a replay signal source.

        Args:
            parent:         wx parent of this panel.

            src:            Source to configure.

            args, kwargs:   Additional arguments passed to the SourcePanel
                            base class.
        """
        SourceConfigPanel.__init__(self, parent=parent, src=src, *args, **kwargs)

        self.initFileControls()
        self.initRateControls()
        self.initLayout()

    def initFileControls(self):
        fileSizer = wx.BoxSizer(orient=wx.VERTICAL)

        fileControlBox = widgets.ControlBox(self, label='Data File', orient=wx.VERTICAL)

        self.fileTextCtrl = wx.TextCtrl(self,
                value=str('SAMPLE DATA'), style=wx.TE_READONLY)

        self.fileBrowseButton = wx.Button(self, label='Browse')

        fileControlBox.Add(self.fileTextCtrl, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)
        fileControlBox.Add(self.fileBrowseButton, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        fileSizer.Add(fileControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.sizer.Add(fileSizer, proportion=1, flag=wx.EXPAND)

    def initRateControls(self):
        """Initialize the sample rate and poll size controls.
        """
        rateSizer = wx.BoxSizer(orient=wx.VERTICAL)

        pollSizeControlBox = widgets.ControlBox(self, label='Poll Size', orient=wx.HORIZONTAL)
        self.pollSizeSpinCtrl = wx.SpinCtrl(self, style=wx.SP_WRAP,
                value=str(self.src.pollSize), min=1, max=32)
        pollSizeControlBox.Add(self.pollSizeSpinCtrl, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SPINCTRL, self.setPollSize, self.pollSizeSpinCtrl)

        rateSizer.Add(pollSizeControlBox, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP, border=10)

        self.sizer.Add(rateSizer)

    def setPollSize(self, event=None):
        self.src.pollSize = self.pollSizeSpinCtrl.GetValue()

    def beforeStart(self):
        self.pollSizeSpinCtrl.Disable()

    def afterStop(self):
        self.pollSizeSpinCtrl.Enable()

class Replay(Source):
    """Replay signal source.
    """
    def __init__(self, mgr, pollSize=2):
        """
        Construct a new source for replaying EEG saved to file.

            pollSize:   Number of data samples collected during each poll.
                        Higher values result in better timing and marker
                        resolution but more CPU usage while higher values
                        typically use less CPU but worse timing results.
        """
        self.pollSize = pollSize
        self.lock = mp.Lock()

        self.replayData = None
        self.startIndex = 0

        Source.__init__(self, mgr=mgr, sampRate=256, chans=[str(i) for i in range(8)],
            configPanelClass=ReplayConfigPanel)

    def loadFile(self, event=None):
        openFileDialog = wx.FileDialog(None, message='Load EEG data.',
            wildcard='JSON (*.json)|*.json|All Files|*',
            style=wx.FD_OPEN)

        try:
            if openFileDialog.ShowModal() == wx.ID_CANCEL:
                return wx.ID_CANCEL
        except Exception:
            wx.LogError('Save failed!')
            raise
        finally:
            openFileDialog.Destroy()

        cap = eeg.EEGFromJSON(openFileDialog.GetPath(), protocol='3minutes')
        self.replayData = cap.data

        self.setSampRate(cap.getSampRate())
        self.setChans(cap.getChanNames())

    def beforeStart(self):
        """Set things up for starting the source.
        """
        if self.replayData is None:
            try:
                if self.loadFile() == wx.ID_CANCEL:
                    wx.LogError('Failed to load data!')
                    # maybe play some default data instead of bombing? XXX - idfah
                    raise Exception('Critical file load canceled.')
            except Exception:
                raise

        self.shift = self.pollSize / float(self.sampRate)
        self.pollDelay = -1.0
        self.lastPollTime = -1.0

    def pollData(self):
        """Poll for new data.  This method sleeps in order to ensure
        that self.pollSize observations are generated at a realistic rate.
        """
        # figure time between polls using exponentially weighted moving average
        curTime = time.time()
        if self.pollDelay >= 0.0:
            self.pollDelay = 0.8*self.pollDelay + 0.2*(curTime - self.lastPollTime)
        else:
            self.pollDelay = 0.0
            self.lastPollTime = curTime - self.shift
        sleepTime = np.max((0.0, self.shift - self.pollDelay))
        self.lastPollTime = curTime + sleepTime
        time.sleep(sleepTime)

        with self.lock:
            endIndex = self.startIndex + self.pollSize

            # wrap back to beginning if at end of data
            if not endIndex < self.replayData.shape[0]:
                self.startIndex = 0
                endIndex = self.pollSize

            data = self.replayData[self.startIndex:endIndex,:]
            self.startIndex = endIndex

        return data
