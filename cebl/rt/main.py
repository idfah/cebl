"""Main entry point for GUI, top-level frame and state management.
"""
import os
import wx

import sys

from . import events
from . import widgets

from .manager import *


class CEBLApp(wx.App):
    """Main wx app.  Simply creates a CEBLMain frame to kick things off.
    """
    def OnInit(self):
        """Create a new CEBLMain frame.
        """
        self.SetAppName('CEBL')
        main = CEBLMain()
        return True

    def OnExit(self):
        """Send gracefull exit notice.
        This should go in final release XXX - idfah
        """
        print 'Gracefull exit.'

class CEBLMain(wx.Frame):
    """Top-level CEBL frame.  Holds the notebook and source manager and maintains the general state of CEBL.
    """
    def __init__(self):
        """Initialize the main GUI frame.
        """
        # set frame size to reasonable value
        ##displaySize = [int(0.75*s) for s in wx.DisplaySize()]

        # hack to help on dual-screen, need something better XXX - idfah
        displaySize = wx.DisplaySize()
        displaySize = 1.2*displaySize[1], 0.75*displaySize[1]

        # call base class constructor
        wx.Frame.__init__(self, parent=None, title='CEBL3', size=displaySize)

        # initialize the main notebook
        self.initNotebook()

        # initialize the page and source manager
        self.mgr = Manager(pageParent=self.notebook, statusPanel=self)

        # add all pages to the notebook
        self.initNotebookPages()

        # initialize the status bar
        self.initStatusBar()

        # initialize the top-level layout
        self.initLayout()

        # cleanup when main frame is closed
        self.Bind(wx.EVT_CLOSE, self.close)

        # toggle full screen on EVT_FULLSCREEN events
        self.Bind(events.EVT_FULLSCREEN, self.toggleFullScreen)

        # update the status bar on EVT_UPDATE_STATUS events
        self.Bind(events.EVT_UPDATE_STATUS, self.updateStatusBar)

    def initNotebook(self):
        """Initialize main notebook.
        """
        # use custom widget with floatable tabs
        self.notebook = widgets.FloatableNotebook(self)

    def initNotebookPages(self):
        for pg in self.mgr.pages:
            self.notebook.AddPage(pg, pg.getName())

    def initStatusBar(self):
        """Initialize main status bar.
        """
        nFields = 5
        self.statusBar = self.CreateStatusBar()
        self.statusBar.SetFieldsCount(nFields)
        self.statusBar.SetStatusStyles([wx.SB_FLAT]*nFields)

        # update status bar at least once per second
        self.updateStatusTimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.updateStatusBar, id=self.updateStatusTimer.GetId())
        self.updateStatusTimer.Start(1000*1)

    def initLayout(self):
        """Initialize main layout.
        """
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Layout()
        self.Center()
        self.Show()

    def Show(self):
        """Show the main frame and splash screen.
        """
        Splash(self)
        wx.Frame.Show(self)

    def close(self, event):
        self.mgr.closeAllPages()
        event.Skip()

    def toggleFullScreen(self, event=None):
        self.ShowFullScreen(not self.IsFullScreen(), wx.FULLSCREEN_NOCAPTION)

    def updateStatusBar(self, event=None):
        src = self.mgr.getSource()

        if src is None:
            curRollCount, curBuffFill = 0, 0.0
            sampRate = 0.0
        else:
            curRollCount, curBuffFill = src.getBufferStats()
            sampRate = src.getEffectiveSampRate()

        self.statusBar.SetStatusText('Source: %s' % str(src), 0)
        self.statusBar.SetStatusText('Running Pages: %d' % self.mgr.getNRunningPages(), 1)
        self.statusBar.SetStatusText('Buffer: %d/%d%%' % (curRollCount, int(curBuffFill*100)), 2)
        self.statusBar.SetStatusText('Sampling Rate: %.2fHz' % sampRate, 3)
        self.statusBar.SetStatusText('Version: 3.0.0a', 4)

class Splash(wx.SplashScreen):
    def __init__(self, parent):
        logo = wx.Image(os.path.dirname(__file__) + '/images/CEBL3_splash.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        wx.SplashScreen.__init__(self,
            parent=parent, milliseconds=2000, bitmap=logo,
            splashStyle=wx.SPLASH_CENTER_ON_SCREEN | wx.SPLASH_TIMEOUT)

def run():
    bci = CEBLApp()
    bci.MainLoop()

if __name__ == '__main__':
    run()
