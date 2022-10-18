import sys
import traceback
import wx

#import warnings
#warnings.simplefilter('always')

from . import events
from . import logging
from . import pages
from . import sources


def logExceptionHook(etype, e, trace):
    wx.LogError(''.join(traceback.format_exception(etype, e, trace)) + 'Uncaught.\n')

class Manager:
    def __init__(self, pageParent, statusPanel=None,
                 sourceList=sources.sourceList,
                 defaultSource=sources.defaultSource,
                 pageList=pages.pageList):
        self.pageParent = pageParent
        self.statusPanel = statusPanel

        self.sourceList = sourceList
        self.defaultSource = defaultSource
        self.pageList = pageList

        self.pages = []
        self.src = None

        self.initLogger()

        self.initSources()

        self.initPages()
        self.runningPages = []

    def initLogger(self):
        # set up custom message logger for wx.
        self.logger = logging.LogTarget()
        #self.logger.this.disown() # hack, something is broken in LogTextCtrl XXX - idfah
        wx.Log.SetActiveTarget(self.logger)
        sys.excepthook = logExceptionHook # log error on uncaught exception

    def initSources(self):
        self.sources = {src.getName(): src for src in
                        [srcClass(self) for srcClass in self.sourceList]}

        self.src = self.sources[self.defaultSource]
        self.src.initBuffer()

    def initPages(self):
        self.pages = [pgClass(parent=self.pageParent, mgr=self)
                      for pgClass in self.pageList]

    def setSource(self, srcName):
        self.src.delBuffer() # buffer is only active for current source
        self.src = self.sources[srcName]
        self.src.initBuffer()

        self.updateSources()

        if self.statusPanel is not None:
            wx.PostEvent(self.statusPanel, events.UpdateStatusEvent(id=wx.ID_ANY))

    def getSource(self):
        return self.src

    def getAllSources(self):
        return self.sources.values()

    def getAllPages(self):
        return self.pages

    def updateSources(self):
        for pg in self.pages:
            pg.updateSource()

    def addRunningPage(self, page):
        """Add a page and start current source if
        when number of running pages goes above zero.
        """
        if self.getNRunningPages() == 0:
            self.src.start()

        self.runningPages.append(page)

        if self.statusPanel is not None:
            wx.PostEvent(self.statusPanel, events.UpdateStatusEvent(id=wx.ID_ANY))

    def getNRunningPages(self):
        return len(self.runningPages)

    def remRunningPage(self, page):
        """Remove a page and stop current source if
        number of running pages reaches zero.
        """
        self.runningPages.remove(page)

        if self.getNRunningPages() == 0:
            self.src.stop()

        if self.statusPanel is not None:
            wx.PostEvent(self.statusPanel, events.UpdateStatusEvent(id=wx.ID_ANY))

    def closeAllPages(self):
        # should probably be stopping sources instead of pages if this is only for cleanup?  This still hangs sometimes XXX - idfah
        if self.getNRunningPages() > 0:
            for pg in self.runningPages:
                pg.close()

            self.runningPages = []

            if self.statusPanel is not None:
                wx.PostEvent(self.statusPanel, events.UpdateStatusEvent(id=wx.ID_ANY))
