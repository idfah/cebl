import wx
import wx.lib.agw.floatspin as agwfs


class SourceConfigPanel(wx.Panel):
    """Simple source configuration panel.  Contains controls for adjusting
    buffer parameters and the polling rate.  Typically, all source config
    panels should extend this class.  This class may be used alone if there
    is no special configuration for a source.
    """
    def __init__(self, parent, src, orient=wx.HORIZONTAL, *args, **kwargs):
        """Initialize a new source configuration panel.

        Args:
            parent:         wx parent of this panel.

            src:            Source to configure.

            args, kwargs:   Additional arguments to pass to the wx.Panel
                            base class constructor.
        """
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)
        self.src = src

        self.sizer = wx.BoxSizer(orient=orient)
        self.SetSizer(self.sizer)

    def initLayout(self):
        self.Layout()

    def select(self):
        """Select this panel to be active.  This involves showing the
        panel and updating the layout.
        """
        self.Show()
        self.Layout()

    def deselect(self):
        """Deselect this panel and make it inactive.  This currently only
        involves hiding the panel.
        """
        self.Hide()

    def beforeStart(self):
        pass

    def afterStart(self):
        pass

    def beforeStop(self):
        pass

    def afterStop(self):
        pass
