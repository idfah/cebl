import copy
import string
import time

import numpy as np
import wx
from wx.lib.agw import aui

from cebl.rt import widgets

from .page import Page


class TextStim(Page):
    def __init__(self, *args, **kwargs):
        self.initConfig()

        Page.__init__(self, name="TextStim", *args, **kwargs)

        self.initAui()
        self.initToolbar()
        self.initTextStim()
        self.initLayout()

    def initAui(self):
        self.auiManager = aui.AuiManager()
        self.auiManager.SetManagedWindow(self)

    def initToolbar(self):
        self.toolbar = aui.AuiToolBar(self)

        self.startButton = wx.Button(self.toolbar, label="Start")
        self.toolbar.AddControl(self.startButton, label="Run")
        self.Bind(wx.EVT_BUTTON, self.toggleRunning, self.startButton)

        self.subjectTextCtrl = wx.TextCtrl(self.toolbar)
        self.subjectTextCtrl.SetValue("s")
        self.toolbar.AddControl(self.subjectTextCtrl, label="Subject")

        self.protocolComboBox = wx.ComboBox(self.toolbar, choices=self.protocols,
                value=self.protocol, style=wx.CB_READONLY)
        self.toolbar.AddControl(self.protocolComboBox, label="Protocol")
        self.Bind(wx.EVT_COMBOBOX, self.setProtocolFromComboBox, self.protocolComboBox)

        #self.toolbar.Realize()

    def initConfig(self):
        #self.stimColor = (179, 179, 36)
        self.stimColor = (255, 255, 10)

        self.stimFont = wx.Font(pointSize=196, family=wx.FONTFAMILY_SWISS,
                            style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)

        self.protocols = ("3minutes", "letter practice b",
                          "letter b", "letter d", "letter p",
                          "letter m", "letter t", "letter x",
                          "motortasks practice",
                          "motortasks trial1", "motortasks trial2",
                          "mentaltasks practice",
                          "mentaltasks trial1",
                          "mentaltasks trial2",
                          "mentaltasks trial3",
                          "mentaltasks trial4",
                          "mentaltasks trial5")
        self.nProtocols = len(self.protocols)

        self.setProtocol("3minutes")

        self.startPause = 2.0

    def initTextStim(self):
        self.stimArea = widgets.TextStim(self, stimText="",
            stimColor=self.stimColor, stimFont=self.stimFont)

        self.stimTimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.changeStim, self.stimTimer)

    def initLayout(self):
        """Initialize layout of main page and initialize layout.
        """
        toolbarAuiInfo = (aui.AuiPaneInfo().Name("toolbar").Caption(self.name + " Tools")
            .ToolbarPane().Top().CloseButton(False).LeftDockable(False).RightDockable(False))
        self.auiManager.AddPane(self.toolbar, toolbarAuiInfo)
        self.toolbar.Realize()

        stimPaneAuiInfo = aui.AuiPaneInfo().Name("stim").Caption(self.name + "Stimulus").CenterPane()
        self.auiManager.AddPane(self.stimArea, stimPaneAuiInfo)

        self.auiManager.Update()

    def setProtocol(self, protocol):
        self.protocol = protocol

        if self.protocol == "3minutes":
            self.letter = ""

            self.stims = ["*",]

            self.si = 3*60.0
            self.isi = 0.0

            self.instructions = "Relax and look at the\n\n%s for 3 minutes." % self.stims[0]

        elif self.protocol.startswith("letter"):
            self.letter = self.protocol[-1]

            if self.protocol.startswith("letter practice"):
                chars = "bdpfnpdpbddsbakbbdb"

            elif self.letter in ("b", "d", "p"):
                chars = "bdpfnpdpbddsbakbbdbmbbpadtdtbdpvdnpbddpp" + \
                        "bsppdimddppdbpbbbdpbdpdpkibdpfdpeebpbbpv" + \
                        "vddbpdbcbpdpbbykcdpp"

            elif self.letter == "m":
                chars = "zijovpmmlhvyummpcmthtdmbpkmimnuomtnmbsq" + \
                        "mglcmmanmqgluakqmnoumhfmimrjfjlmhrntmyjw"

            elif self.letter == "t":
                chars = "tbmdfaootfrsqyjptotutrslttxpfejtqontmtdh" + \
                        "pwhtrweesqvaprbatmtztlrztktsutthtwpvtvne"

            elif self.letter == "x":
                chars = "fjgxaxgunxzuyrxkqphxiddoyxqcccacxbtxxtxv" + \
                        "ecplmunxrxcxxzbexyfztojwmxybxnxhtpwxxwrz"

            else:
                stims = string.ascii_lowercase*3

            self.stims = list(chars)

            self.si = 0.100
            self.isi = 0.750

            self.instructions = \
                ("Count the number of times the letter\n\n%s" % self.letter) + \
                " appears in the center of the screen."

        elif self.protocol.startswith("motortasks"):
            self.letter = ""

            #mtasks = ["Left", "Right"]
            #if self.protocol.startswith("motortasks practice"):
            #    nTrials = 1
            #else:
            #    nTrials = 5
            #self.stims = sum([list(np.random.permutation(mtasks)) for i in range(nTrials)], [])

            if self.protocol.startswith("motortasks practice"):
                self.stims = ["Left", "Right"]
            else:
                self.stims = ["Right", "Right",
                              "Left", "Right",
                              "Left", "Left",
                              "Right", "Left"]

            self.si = 10.0
            self.isi = 5.0

            self.instructions = '''In your mind only, please perform one of the following tasks
when one of the following cues appears. When the screen is blank, relax and think of nothing.


"Left"  think about repeatedly raising and lowering your left arm over your head.

"Right"  think about repeatedly raising and lowering your right arm over your head.'''

        elif self.protocol.startswith("mentaltasks"):
            self.letter = ""

            #mtasks = ["Count", "Fist", "Rotate", "Song"]
            #if self.protocol.startswith("mentaltasks practice"):
            #    nTrials = 1
            #else:
            #    nTrials = 3
            #self.stims = sum([list(np.random.permutation(mtasks)) for i in range(nTrials)], [])

            #if self.protocol.startswith("mentaltasks practice"):
            #    self.stims = ["Count", "Rotate", "Song", "Fist"]
            #else:
            #    self.stims = ["Count", "Song", "Rotate", "Count",
            #                  "Fist", "Song", "Rotate", "Fist",
            #                  "Count", "Fist", "Rotate", "Song",
            #                  "Rotate", "Count", "Song", "Count",
            #                  "Fist", "Rotate", "Song", "Fist"]
            self.stims = ["Count", "Rotate", "Song", "Fist"]
            np.random.shuffle(self.stims)

            self.si = 10.0
            self.isi = 5.0

            self.instructions = '''In your mind only, please perform one of the following tasks
when one of the following cues appears. When the screen is blank, relax and think of nothing.


"Count"  think about counting backwards from 100 by 3

"Fist"   think about repeatedly clenching and opening your right hand

"Rotate" think about a rotating cube suspended in air

"Song"   sing a favorite song silently to yourself'''

        else:
            raise RuntimeError("Invalid protocol: " % protocol)

    def toggleRunning(self, event=None):
        if self.isRunning():
            self.stopFlag = True
            self.startButton.Disable()
        else:
            self.start()
            self.startButton.SetLabel("Stop")

    def setProtocolFromComboBox(self, event):
        self.setProtocol(self.protocolComboBox.GetValue())
        #self.stimArea.setStimText(self.letter)

    def beforeStart(self):
        instructionDialog = wx.MessageDialog(self, self.instructions,
                    "Instructions", style=wx.OK | wx.CENTER)
        instructionDialog.ShowModal()
        instructionDialog.Destroy()

    def afterStart(self):
        self.nextBlank = True
        self.stopFlag = False
        self.availStims = copy.copy(self.stims)
        self.stimArea.setStimText("")

        self.startTime = time.time()

        self.stimTimer.Start(1000.0*self.startPause, oneShot=True)

    def changeStim(self, event=None):
        if self.stopFlag:
            curStim = ""
            self.src.setMarker(0.0)

            #print("stopping")
            self.stop()
            self.startButton.SetLabel("Start")
            self.startButton.Enable()

        elif not self.availStims:
            self.stimTimer.Start(1000.0*self.startPause, oneShot=True)

            curStim = ""
            self.stopFlag = True
            self.src.setMarker(0.0)

        elif self.nextBlank:
            if self.isRunning():
                self.stimTimer.Start(1000.0*self.isi, oneShot=True)

            curStim = ""
            self.nextBlank = False
            self.src.setMarker(0.0)

        else:
            self.stimTimer.Start(1000.0*self.si, oneShot=True)

            curStim = self.availStims.pop(0)
            if self.protocol.startswith("letter"):
                if curStim == self.letter:
                    sign = 1
                else:
                    sign = -1
            else:
                sign = 1
            self.nextBlank = True

            #print(curStim[0], curStim)
            self.src.setMarkerChr(curStim[0], sign)

        self.stimArea.setStimText(curStim)

    def beforeStop(self):
        if not "practice" in self.protocol:
            cap = self.src.getEEGSecs(time.time() - self.startTime, filter=False)

            fileName = self.subjectTextCtrl.GetValue() + "-" + \
                    self.protocol.replace(" ", "-") + ".pkl"

            saveDialog = wx.FileDialog(self, message="Save EEG data.",
                wildcard="Pickle (*.pkl)|*.pkl|All Files|*",
                defaultFile=fileName,
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

            try:
                if saveDialog.ShowModal() == wx.ID_CANCEL:
                    return
                cap.saveFile(saveDialog.GetPath())
            except Exception:
                wx.LogError("Save failed!")
                raise
            finally:
                saveDialog.Destroy()

    def afterStop(self):
        curProtoIndex = self.protocolComboBox.GetSelection()

        # we get -1 if no item was selected, i.e., first initial value
        if curProtoIndex == -1:
            nextProtoIndex = 1
        else:
            nextProtoIndex = (curProtoIndex+1) % self.nProtocols

        if curProtoIndex+1 == self.nProtocols:
            self.sayThankYou()

        self.protocolComboBox.SetSelection(nextProtoIndex)
        self.setProtocol(self.protocols[nextProtoIndex])

    def sayThankYou(self):
        thankYou = "This session is complete.  Thank you for participating!!"

        thanksDialog = wx.MessageDialog(self, thankYou,
                    "Thank You!", style=wx.OK | wx.CENTER)
        thanksDialog.ShowModal()
        thanksDialog.Destroy()
