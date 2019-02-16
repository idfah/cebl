"""Base classes and routines common to all pages.
"""
import wx

from cebl.rt import events


class Page(wx.Panel):
    """Base class for all pages.  Handles starting, stopping and
    state management of pages.

    Notes:
        Consider using standard.StandardMonitorPage or
        standard.StandardBCIPage for quick creating of
        pages with the standard look and feel.
    """
    def __init__(self, parent, name, mgr, *args, **kwargs):
        """Construct a new page.

        Args:
            name:           The string name of this page as it will appear
                            to the user.

            mgr:            The cebl.manager.Manager instance that manages
                            all sources and pages.

            args, kwargs:   Additional arguments to pass to wx.Panel.
        """
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.name = name
        self.running = False

        self.mgr = mgr
        self.src = self.mgr.getSource()

        self.Bind(wx.EVT_KEY_UP, self.onKeyUp)

    def getName(self):
        """Get the string name of this page as seen by the user.
        """
        return self.name

    def updateSource(self, event=None):
        self.src = self.mgr.getSource()
        self.afterUpdateSource()

    def afterUpdateSource(self):
        pass

    def getSource(self):
        return self.src

    def isRunning(self):
        """True if this page is currently running, false otherwise.
        """
        return self.running

    def toggleRunning(self, event=None):
        """If this page is running, stop it.  If it is stopped, start it.
        """
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self, event=None):
        """Start this page and notify the manager.

        Note:
            This method calls the beforeStart method before it is started and
            afterStart after it is started.  Use these hooks to add to the
            startup procedure in sub-classes.
        """
        if not self.running:
            wx.LogMessage('Starting page %s.' % self.name)
            try:
                self.beforeStart()
                self.running = True
                self.mgr.addRunningPage(self)
                self.afterStart()

            except Exception as e:
                # better way to stop page in case of failure? XXX - idfah
                self.running = False
                raise
        else:
            wx.LogError('Cannot start page %s because it is already running.' % self.name)

    def beforeStart(self):
        """Called by start before the page is started.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def afterStart(self):
        """Called by start after the page is started.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def stop(self, event=None):
        """Stop this page and notify the manager.

        Note:
            This method calls the beforeStop method before it is stopped and
            afterStop after it is stopped.  Use these hooks to add to the
            stop procedure in sub-classes.
        """
        if self.running:
            wx.LogMessage('Stopping page %s.' % self.name)

            try:
                self.beforeStop()
                self.running = False
                self.mgr.remRunningPage(self)
                self.afterStop()

            except Exception as e:
                # better way to stop page in case of failure? XXX - idfah
                self.running = False
                raise
        else:
            wx.LogError('Cannot stop page %s because it is not running.' % self.name)

    def beforeStop(self):
        """Called by stop before the page is stopped.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def afterStop(self):
        """Called by stop after the page is stopped.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def onKeyUp(self, event):
        """Handle keypress events.
        """
        key = event.GetKeyCode()

        # check for F11 to start full screen mode
        if key == wx.WXK_F11:
            # post a an EVT_FULL_SCREEN event
            # the change should be handled by the parent
            wx.PostEvent(self, events.FullScreenEvent(id=wx.ID_ANY))

    def close(self, event=None):
        if self.running:
            self.stop()

        if event is not None:
            event.Skip()
