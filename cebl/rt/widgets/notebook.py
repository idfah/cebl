import wx
from wx.lib.agw import aui
#from wx import aui

from cebl.rt import events


'''class FloatableNotebook(aui.AuiNotebook):
    """A wx AUI notebook with pages that can be dragged outside
    the notebook and floated in a new frame.  The page is put
    back into the notebook when it is closed.

    Note:  This may be available in future version of AuiNotebook
        with the AUI_NB_TAB_FLOAT option.

    This module is derived from AuiNotebookWithFloatingPages
        Author: Frank Niessink <frank@...>
        License: wxWidgets license
        Version: 0.1
        Date: August 8, 2007
    """
    def __init__(self, parent,
            style=aui.AUI_NB_TOP | aui.AUI_NB_TAB_SPLIT | aui.AUI_NB_TAB_MOVE, *args, **kwargs):
        aui.AuiNotebook.__init__(self, parent=parent, *args, style=style, **kwargs)

        self.auiManager = self.GetAuiManager()

        self.dragging = False

        #self.Bind(aui.EVT_AUINOTEBOOK_BEGIN_DRAG, self.beginDrag)
        self.Bind(aui.EVT_AUINOTEBOOK_DRAG_MOTION, self.dragMotion)
        self.Bind(aui.EVT_AUINOTEBOOK_END_DRAG, self.endDrag)
        self.Bind(wx.EVT_SIZE, self.resize)

    def resize(self, event):
        # XXX idfah - why is this needed?
        event.Skip()

    def mouseOutsideWindow(self):
        screenRect = self.GetScreenRect()
        screenRect.Inflate(50, 50)
        return not screenRect.Contains(wx.GetMousePosition())

    def dragMotion(self, event):
        self.auiManager.HideHint()
        if self.mouseOutsideWindow():
            x,y = wx.GetMousePosition()
            hintRect = wx.Rect(x, y, 200, 200)
            wx.CallAfter(self.auiManager.ShowHint, hintRect)
        else:
            event.Skip()

    def endDrag(self, event):
        wx.CallAfter(self.auiManager.HideHint)
        if self.mouseOutsideWindow() and self.GetPageCount() > 1:
            # CallAfter allows base class to handle event first
            wx.CallAfter(self.floatPage, self.Selection)
        else:
            event.Skip()

    def floatPage(self, pageIndex):
        pageTitle = self.GetPageText(pageIndex)
        pageContents = self.GetPage(pageIndex)

        frame = wx.Frame(self, title=pageTitle,
            size=self.GetClientSize(),
            style=wx.DEFAULT_FRAME_STYLE)

        pageContents.Reparent(frame)
        self.RemovePage(pageIndex)

        frame.Bind(wx.EVT_CLOSE, self.closeFloatPage)
        frame.Move(wx.GetMousePosition())
        frame.Show()

    def closeFloatPage(self, event):
        self.floated = False
        event.Skip()

        frame = event.GetEventObject()

        pageTitle = frame.GetTitle()
        pageContents = frame.GetChildren()[0]
        pageContents.Reparent(self)

        self.AddPage(pageContents, pageTitle, select=True)'''

class FloatingFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        #self.Bind(wx.EVT_KEY_UP, self.onKeyUp)
        self.Bind(events.EVT_FULLSCREEN, self.toggleFullScreen)

    #def onKeyUp(self, event):
    #    key = event.GetKeyCode()
    #    if key == wx.WXK_F11:
    #        wx.PostEvent(self, events.FullScreenEvent(id=wx.ID_ANY))

    def toggleFullScreen(self, event=None):
        self.ShowFullScreen(not self.IsFullScreen(), wx.FULLSCREEN_NOCAPTION)

class FloatableNotebook(aui.AuiNotebook):
    def __init__(self, parent, agwStyle=aui.AUI_NB_TOP | aui.AUI_NB_TAB_SPLIT):
        aui.AuiNotebook.__init__(self, parent=parent, agwStyle=agwStyle | aui.AUI_NB_TAB_FLOAT)

    #def FloatPage(self, page_index):
    #    pageTitle = self.GetPageText(page_index)
    #    pageContents = self.GetPage(page_index)

    #    frame = wx.Frame(self, title=pageTitle,
    #        size=self.GetClientSize(),
    #        style=wx.DEFAULT_FRAME_STYLE)

    #    self.RemovePage(page_index)
    #    pageContents.Reparent(frame)

    #    frame.Bind(wx.EVT_CLOSE, self.closeFloatPage)
    #    frame.Move(wx.GetMousePosition())
    #    frame.Show()

    #def closeFloatPage(self, event):
    #    self.floated = False
    #    event.Skip()

    #    frame = event.GetEventObject()

    #    pageTitle = frame.GetTitle()
    #    pageContents = frame.GetChildren()[0]
    #    pageContents.Reparent(self)

    #    self.AddPage(pageContents, pageTitle, select=True)

    def FloatPage(self, page_index):
        root_manager = aui.GetManager(self)
        page_title = self.GetPageText(page_index)
        page_contents = self.GetPage(page_index)
        page_bitmap = self.GetPageBitmap(page_index)
        text_colour = self.GetPageTextColour(page_index)
        info = self.GetPageInfo(page_index)

        frame = FloatingFrame(self, title=page_title,
            size=self.GetClientSize(), # request add to standard AuiNotebook? XXX - idfah
            style=wx.DEFAULT_FRAME_STYLE)
                         #style=wx.DEFAULT_FRAME_STYLE|wx.FRAME_TOOL_WINDOW|
                         #      wx.FRAME_FLOAT_ON_PARENT | wx.FRAME_NO_TASKBAR)

        if info.control:
            info.control.Reparent(frame)
            info.control.Hide()

        frame.bitmap = page_bitmap
        frame.page_index = page_index
        frame.text_colour = text_colour
        frame.control = info.control
        page_contents.Reparent(frame)
        frame.Bind(wx.EVT_CLOSE, self.OnCloseFloatingPage)
        frame.Move(wx.GetMousePosition())
        frame.Show()
        page_contents.SetFocus()

        self.RemovePage(page_index)
        self.RemoveEmptyTabFrames()

        wx.CallAfter(self.RemoveEmptyTabFrames)

    #def FloatPage(self, page_index):
    #    page_contents = self.GetPage(page_index)
    #    size = page_contents.GetSize()

    #    aui.AuiNotebook.FloatPage(self, page_index)

    #    page_contents.GetParent().SetSize(size)

    def OnCloseFloatingPage(self, event):
            event.Skip()
            frame = event.GetEventObject()
            page_title = frame.GetTitle()
            page_contents = list(frame.GetChildren())[-1]
            #self.InsertPage(frame.page_index, page_contents, page_title, select=True, bitmap=frame.bitmap, control=frame.control)
            #self.InsertPage(frame.page_index, wx.StaticText(self, label='hello world'), page_title, select=True)#, bitmap=frame.bitmap, control=frame.control)
            page_contents.Reparent(self)
            wx.Yield() # report this bug XXX - idfah
            self.InsertPage(frame.page_index, page_contents, page_title, select=True,
                    bitmap=frame.bitmap, control=frame.control)

            if frame.control:
                src_tabs, idx = self.FindTab(page_contents)
                frame.control.Reparent(src_tabs)
                frame.control.Hide()
                frame.control = None

            self.SetPageTextColour(frame.page_index, frame.text_colour)
