"""Classes and routines for making configuration changes to sources, channels, logging, et cetra.
"""
import wx
from wx.lib.scrolledpanel import ScrolledPanel

from cebl.rt import sources
from cebl.rt import widgets

from page import Page

class Config(Page):
    """Page for making configuration changes.  This includes source, channel
    configuration as well as a logging console.

    Note:
        This page constructs all source instances and adds them to the manager.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new Config page.
        """
        Page.__init__(self, name='Config', *args, **kwargs)

        self.scrolledPanel = ScrolledPanel(self)

        self.initSourceConfig()
        self.initChannelConfig()
        self.initMessageArea()
        self.initLayout()

        self.selectSource()

    def initSourceConfig(self):
        """Initialize the source configuration area.
        """
        # Generate a dictionary of configuration panels for each source.
        # dictionary mapping source name to a configuration panel
        self.srcPanels = {src.getName(): src.genConfigPanel(parent=self.scrolledPanel)
                            for src in self.mgr.getAllSources()}

        self.curSrcPanel = self.srcPanels[self.src.getName()]

        # sizer for source configuration options general to all sources
        sourceGeneralSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        # source selector
        sourceControlBox = widgets.ControlBox(self.scrolledPanel, label='Source', orient=wx.VERTICAL)
        choices = sorted(self.srcPanels.keys(), key=lambda k: k.lower(), reverse=True)
        self.sourceComboBox = wx.ComboBox(self.scrolledPanel, id=wx.ID_ANY, choices=choices,
            value=self.src.getName(), style=wx.CB_READONLY)
        self.sourceComboBox.Bind(wx.EVT_COMBOBOX, self.selectSource, self.sourceComboBox)
        sourceControlBox.Add(self.sourceComboBox, proportion=0, flag=wx.ALL, border=10)

        # query button
        self.sourceQueryButton = wx.Button(self.scrolledPanel, label='Query')
        sourceControlBox.Add(self.sourceQueryButton, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        self.Bind(wx.EVT_BUTTON, self.querySource, self.sourceQueryButton)

        # reset button
        self.sourceResetButton = wx.Button(self.scrolledPanel, label='Reset')
        sourceControlBox.Add(self.sourceResetButton, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        #self.Bind(wx.EVT_BUTTON, self.resetSource, self.sourceResetButton)
        sourceGeneralSizer.Add(sourceControlBox, proportion=0,
                flag=wx.RIGHT | wx.TOP | wx.LEFT, border=10)

        bufferSizer = wx.BoxSizer(orient=wx.VERTICAL)

        # buffer seconds selector
        bufferSecsControlBox = widgets.ControlBox(self.scrolledPanel,
                label='Buffer Size', orient=wx.HORIZONTAL)

        self.bufferRollSpinCtrl = wx.SpinCtrl(self.scrolledPanel, style=wx.SP_WRAP,
                value=str(3), min=2, max=10, size=(100,28))
        bufferSecsControlBox.Add(self.bufferRollSpinCtrl, proportion=0,
                flag=wx.ALL | wx.CENTER, border=10)

        xStaticText = wx.StaticText(self.scrolledPanel, label='X')
        bufferSecsControlBox.Add(xStaticText, proportion=0, flag=wx.CENTER)

        self.bufferSecsSpinCtrl = wx.SpinCtrl(self.scrolledPanel, style=wx.SP_WRAP,
                value=str(300), min=60, max=1000, size=(125,28))
        bufferSecsControlBox.Add(self.bufferSecsSpinCtrl, proportion=0,
                flag=wx.ALL | wx.CENTER, border=10)

        bufferSizer.Add(bufferSecsControlBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.EXPAND, border=10)

        precisControlBox = widgets.ControlBox(self.scrolledPanel, label='Data Precision', orient=wx.HORIZONTAL)
        self.precisSingleButton = wx.RadioButton(self.scrolledPanel, label='Single', style=wx.RB_GROUP)
        precisControlBox.Add(self.precisSingleButton, proportion=0,
                flag=wx.TOP | wx.LEFT | wx.BOTTOM | wx.CENTER, border=10)

        self.precisDoubleButton = wx.RadioButton(self.scrolledPanel, label='Double')
        precisControlBox.Add(self.precisDoubleButton, proportion=0,
                flag=wx.ALL | wx.CENTER, border=10)
        self.precisDoubleButton.SetValue(True)
        #self.Bind(wx.EVT_RADIOBUTTON, self.setExtendedPrecision, self.precisExtendedButton)

        bufferSizer.Add(precisControlBox, proportion=1,
                flag=wx.TOP | wx.RIGHT | wx.EXPAND, border=10)

        sourceGeneralSizer.Add(bufferSizer, proportion=0, flag=wx.EXPAND)

        sourceSpecificSizer = wx.BoxSizer(orient=wx.VERTICAL)

        # add each source configuration panel (we'll hide/show as needed)
        for sp in self.srcPanels.values():
            sourceSpecificSizer.Add(sp, proportion=1)#, flag=wx.EXPAND)

        # sizer for both general and specific configurations
        self.sourceSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.sourceSizer.Add(sourceGeneralSizer, proportion=0, flag=wx.EXPAND)
        self.sourceSizer.Add(sourceSpecificSizer, proportion=1, flag=wx.EXPAND)

    def initChannelConfig(self):
        """Initialize the channel configuration area.
        """
        # controlbox to surround the area
        chanControlBox = widgets.ControlBox(self.scrolledPanel,
                label='Channels', orient=wx.HORIZONTAL)

        # only supports two columns, this could probably be done better XXX - idfah
        ## # left column
        ## leftChanSizer = wx.BoxSizer(orient=wx.VERTICAL)

        ## # create text controls
        ## self.chanTextCtrls = []
        ## for c in xrange(16):
        ##     curChanTextCtrl = wx.TextCtrl(self.scrolledPanel)
        ##     self.chanTextCtrls.append(curChanTextCtrl)
        ##     leftChanSizer.Add(curChanTextCtrl, proportion=0,
        ##         flag=wx.RIGHT | wx.TOP | wx.LEFT, border=10)

        ## # add to sizer
        ## chanControlBox.Add(leftChanSizer)

        ## # right column
        ## rightChanSizer = wx.BoxSizer(orient=wx.VERTICAL)

        ## # create text controls
        ## for c in xrange(16):
        ##     curChanTextCtrl = wx.TextCtrl(self.scrolledPanel)
        ##     self.chanTextCtrls.append(curChanTextCtrl)
        ##     rightChanSizer.Add(curChanTextCtrl, proportion=0,
        ##         flag=wx.RIGHT | wx.TOP | wx.LEFT, border=10)

        ## # add to sizer
        ## chanControlBox.Add(rightChanSizer)

        self.chanSizer = wx.GridSizer(20, 2, 10, 10)
        #self.chanSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.chanTextCtrls = [wx.TextCtrl(self.scrolledPanel) for i in xrange(36*2)]
        self.chanSizer.AddMany(self.chanTextCtrls)
        #for ctc in self.chanTextCtrls:
        #    self.chanSizer.Add(ctc, proportion=0, flag=wx.TOP | wx.LEFT | wx.RIGHT, border=2)

        chanControlBox.Add(self.chanSizer, flag=wx.ALL, border=10)

        # sizer for channel configuration area
        self.chanSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.chanSizer.Add(chanControlBox, proportion=1,
                           flag=wx.TOP | wx.BOTTOM, border=10)

    def initMessageArea(self):
        """Initialize the message log area.
        """
        # font for messages
        msgFont = wx.Font(pointSize=11, family=wx.FONTFAMILY_MODERN,
            style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL,
            underline=False)

        # font for CEBL introduction message
        helloFont = wx.Font(pointSize=24, family=wx.FONTFAMILY_ROMAN,
            style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_BOLD, underline=True)

        # the message log
        messageControlBox = widgets.ControlBox(self.scrolledPanel,
                label='Message Log', orient=wx.VERTICAL)
        self.messageArea = wx.TextCtrl(self.scrolledPanel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)
        self.messageArea.SetMinSize((150,150))
        messageControlBox.Add(self.messageArea, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        # intro message
        self.messageArea.SetDefaultStyle(
                wx.TextAttr(font=helloFont, alignment=wx.TEXT_ALIGNMENT_LEFT))
        self.messageArea.AppendText('Welcome to CEBL!\n\n')

        # setup message style
        self.messageArea.SetDefaultStyle(wx.TextAttr())
        self.messageArea.SetDefaultStyle(wx.TextAttr(font=msgFont))

        # add the message area text ctrl widget as a log target
        self.mgr.logger.addTextCtrl(self.messageArea)

        messageControlSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        # button for saving the message log to a file
        self.saveMessagesButton = wx.Button(self.scrolledPanel, label='Save')
        messageControlSizer.Add(self.saveMessagesButton, proportion=0,
            flag=wx.LEFT | wx.BOTTOM | wx.RIGHT, border=10)
        self.Bind(wx.EVT_BUTTON, self.saveMessages, self.saveMessagesButton)

        # button for clearing the message log
        self.clearMessagesButton = wx.Button(self.scrolledPanel, label='Clear')
        messageControlSizer.Add(self.clearMessagesButton, proportion=0,
            flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.Bind(wx.EVT_BUTTON, self.clearMessages, self.clearMessagesButton)

        # set up verbose logging
        self.verboseMessagesCheckBox = wx.CheckBox(self.scrolledPanel, label='Verbose')
        messageControlSizer.Add(self.verboseMessagesCheckBox, proportion=0,
            flag=wx.BOTTOM | wx.RIGHT, border=10)

        messageControlBox.Add(messageControlSizer, proportion=0, flag=wx.EXPAND)

        # sizer for message log area
        self.messageSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.messageSizer.Add(messageControlBox, proportion=1,
                              flag=wx.ALL | wx.EXPAND, border=10)

    def initLayout(self):
        """Initialize the page layout.
        """
        scrolledSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        scrolledSizer.Add(self.sourceSizer, proportion=0)#, flag=wx.EXPAND)
        scrolledSizer.Add(self.chanSizer, proportion=0, flag=wx.EXPAND)
        scrolledSizer.Add(self.messageSizer, proportion=1, flag=wx.EXPAND)
        self.scrolledPanel.SetSizer(scrolledSizer)

        # main sizer
        sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        sizer.Add(self.scrolledPanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)

        self.scrolledPanel.Layout()
        self.scrolledPanel.FitInside()
        self.scrolledPanel.SetupScrolling()

        # hide after layout (prevents gtk warnings)
        for sp in self.srcPanels.values():
            sp.deselect()

    def updateChanText(self):
        for ctc in self.chanTextCtrls:
            ctc.Clear()
            ctc.Hide()

        chanNames = self.src.getChanNames()
        ctls = self.chanTextCtrls[:len(chanNames)]

        for chan, ctl in zip(chanNames, ctls[0::2] + ctls[1::2]):
            ctl.Show()
            ctl.AppendText(chan)

    def afterUpdateSource(self):
        self.updateChanText()

    def selectSource(self, event=None):
        """Set a new source from the source selection combobox.
        """
        srcName = self.sourceComboBox.GetValue()

        # deselect previous source config panel
        self.curSrcPanel.deselect()

        # select new source config panel
        self.curSrcPanel = self.srcPanels[srcName]
        self.curSrcPanel.select()

        # set the source in the manager
        self.mgr.setSource(srcName)

        # add source description to the message log
        wx.LogMessage(repr(self.src) + '\n')

        # update the text in the channel configuration area
        self.updateChanText()

        # adjust layout since we have shown and hidden panels
        self.scrolledPanel.Layout()
        self.scrolledPanel.FitInside()

    def querySource(self, event=None):
        """Call the query method on the current source
        and put the output in the message log.
        """
        try:
            wx.LogMessage(self.src.query())
        except Exception as e:
            wx.LogError('Failed to query source: ' + str(e))

    def clearMessages(self, event=None):
        """Clear the message log.
        """
        self.messageArea.Clear()

    def saveMessages(self, event=None):
        """Save the message log to file.
        """
        saveDialog = wx.FileDialog(self.scrolledPanel, message='Save Message Log',
            wildcard='Text Files (*.txt)|*.txt|All Files|*',
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        try:
            if saveDialog.ShowModal() == wx.ID_CANCEL:
                return
            with open(saveDialog.GetPath(), 'w') as fileHandle:
                fileHandle.write(self.messageArea.GetValue())
        except Exception as e:
            wx.LogError('Save failed!')
            raise
        finally:
            saveDialog.Destroy()
