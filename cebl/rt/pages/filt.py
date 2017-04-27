import wx
import wx.gizmos

from cebl.rt import sources
from cebl.rt import widgets
from cebl.rt import filters

from page import Page


# the way filters are managed here is messy.  Needs work. XXX - idfah

class Filter(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, name='Filter', *args, **kwargs)

        self.filterChain = self.src.getFilterChain()

        self.initSelectorArea()
        self.initConfigArea()
        self.initLayout()

    def initSelectorArea(self):
        self.selectorSizer = wx.BoxSizer(orient=wx.VERTICAL)

        filterListControlBox = widgets.ControlBox(self,
                label='Available Filters', orient=wx.VERTICAL)
        self.filterListBox = wx.ListBox(self, choices=filters.filterChoices.keys(),
                style=wx.LB_SORT | wx.LB_SINGLE)
        filterListControlBox.Add(self.filterListBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        self.pushFilterButton = wx.Button(self, label='Push Filter')
        self.Bind(wx.EVT_BUTTON, self.pushFilter, self.pushFilterButton)
        filterListControlBox.Add(self.pushFilterButton, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.selectorSizer.Add(filterListControlBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        chainListControlBox = widgets.ControlBox(self, label='Filter Chain', orient=wx.VERTICAL)
        self.chainListBox = wx.ListBox(self, choices=[], style=wx.LB_SINGLE)
        ##self.chainListBox = wx.ListCtrl(self, style=wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)
        self.Bind(wx.EVT_LISTBOX, self.configFilter, self.chainListBox)
        chainListControlBox.Add(self.chainListBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=10)

        #self.configFilterButton = wx.Button(self, label='Configure Filter')
        #self.Bind(wx.EVT_BUTTON, self.configFilter, self.configFilterButton)
        #chainListControlBox.Add(self.configFilterButton, proportion=0,
        #        flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=10)

        self.popFilterButton = wx.Button(self, label='Pop Filter')
        self.Bind(wx.EVT_BUTTON, self.popFilter, self.popFilterButton)
        chainListControlBox.Add(self.popFilterButton, proportion=0,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.selectorSizer.Add(chainListControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

    def initConfigArea(self):
        self.configSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.filterConfigPanel = None

    def initLayout(self):
        sizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        sizer.Add(self.selectorSizer, proportion=0, flag=wx.EXPAND)
        sizer.Add(self.configSizer, proportion=1, flag=wx.EXPAND)

        self.SetSizer(sizer)
        self.Layout()

    def pushFilter(self, event):
        filterChoiceIndex = self.filterListBox.GetSelection()
        if filterChoiceIndex == wx.NOT_FOUND:
            wx.LogError('Page %s: No filter selected.' % self.name)
            return

        filterChoice = self.filterListBox.GetString(filterChoiceIndex)
        self.chainListBox.Append(filterChoice)

        self.filterChain.push(filters.filterChoices[filterChoice])

    def popFilter(self, event=None):
        self.chainListBox.Delete(self.chainListBox.GetCount()-1)
        self.filterChain.pop()

    def configFilter(self, event=None):
        filterChoiceIndex = self.chainListBox.GetSelection()

        if self.filterConfigPanel is not None:
            self.filterConfigPanel.Destroy()

        if filterChoiceIndex != wx.NOT_FOUND:
            flt = self.filterChain.getFilter(filterChoiceIndex)
            self.filterConfigPanel = flt.genConfigPanel(parent=self, pg=self)

            self.configSizer.Add(self.filterConfigPanel, proportion=1, flag=wx.EXPAND)

        else:
            self.filterConfigPanel = None

        self.Layout()

    def afterUpdateSource(self):
        self.chainListBox.Clear()
        self.configFilter()

        self.filterChain = self.src.getFilterChain()

        for name in self.filterChain.getFilterNames():
            self.chainListBox.Append(name)
