"""Classes and functions that make it easier to implement new widgets that rely
on the wx drawing and graphics capabilities.

Refs:
        http://wiki.wxpython.org/DoubleBufferedDrawing
        by
            Chris Barker
            2011.03.15
            Chris.Barker@noaa.gov
"""
import wx


class DrawablePanel(wx.Panel):
    """This class is a wx.Panel that contains a wx.DC drawing context.
    Various wx events are handled so that the user can simply modify
    the drawing context.

    This class is typically used as a base-class for widgets that
    use a wx drawing context.

    Notes:
        The wx drawing context is fairly primative as far as graphics
        are concerned.  In order to draw more sophisticated vector
        graphics, consider using GraphicsPanel instead.
    """
    def __init__(self, parent, background='black', style=0, *args, **kwargs):
        """Initialize a new DrawablePanel.

        Args:
            parent:         wx parent object.

            style:          Style arguments passed the the wx.Panel base class.
                            The wx.NO_FULL_REPAINT_ON_RESIZE argument is added
                            to the given style arguments.

            args, kwargs:   Additional arguments passed to the wx.Panel
                            base class.
        """
        wx.Panel.__init__(self, parent=parent, style=style | wx.NO_FULL_REPAINT_ON_RESIZE,
                          *args, **kwargs)

        self.background = background

        self.lastSize = (0,0)

        # initial resize creates initial drawing
        # buffer and triggers first draw
        self.resize()

        self.Bind(wx.EVT_PAINT, self.repaint)
        self.Bind(wx.EVT_SIZE, self.resize)

    def getBackground(self):
        return self.background

    def setBackground(self, background=(0,0,0), refresh=True):
        self.background = background

        if refresh:
            self.refresh()

    def resize(self, event=None):
        """Handle wx resize events.

        Notes:
            This method may be called outside the wx event buffer
            in order to initialize the panel or reset the drawing
            buffer.
        """
        size = self.winWidth, self.winHeight = self.GetSize()
        if size != self.lastSize: # hack to mitigate multiple consecutive resize events
            self.winRadius = min((self.winWidth/2.0, self.winHeight/2.0))
            self.drawingBuffer = wx.Bitmap(self.winWidth, self.winHeight)
            #self.drawingBuffer = wx.EmptyBitmap(self.winWidth, self.winHeight)
            self.lastSize = size
            self.refresh()

        if event is not None:
            event.Skip()

    def repaint(self, event):
        """Handle wx repaint events.

        Notes:
            This should only be called from the wx event loop.  If you need to
            manually trigger a repaint, call self.triggerRepaint instead to
            post an EVT_REPAINT event.
        """
        wx.BufferedPaintDC(self, self.drawingBuffer)

    def triggerRepaint(self):
        """Trigger a repaint of the drawing area.
        """
        self.Refresh(eraseBackground=False)
        self.Update()

    def refresh(self):
        """Refresh the drawing area after a change has been made.
        This method sets up a drawing context, calls self.draw
        to update the drawing and then calls self.triggerRepaint
        in order to update the drawing area on the screen.

        This method should be called each time a change is
        made that requires the drawing area to be updated.
        """
        dc = wx.MemoryDC(self.drawingBuffer)
        dc.SelectObject(self.drawingBuffer)

        dc.SetBackground(wx.Brush(self.background, style=wx.SOLID))
        dc.Clear()

        # do not draw if window is very small, right solution? XXX - idfah
        if self.winRadius < 1.0e-3:
            return

        #dc.BeginDrawing()
        self.draw(dc)
        #dc.EndDrawing()

        #del dc
        dc.SelectObject(wx.NullBitmap)
        self.triggerRepaint()

    def draw(self, dc):
        """This method draws the widget using the given drawing context.
        This method should be overridden if you want to draw anything other
        than a blank background.

        Args:
            dc: wx drawing context.
        """
        pass

    def saveFile(self):
        self.refresh()

        saveDialog = wx.FileDialog(self, message='Save Image',
                wildcard='Portable Network Graphics (*.png)|*.png|All Files|*',
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        try:
            if saveDialog.ShowModal() != wx.ID_CANCEL:
                img = self.drawingBuffer.ConvertToImage()
                img.SaveFile(saveDialog.GetPath(), wx.BITMAP_TYPE_PNG)

        except Exception as e:
            wx.LogError('Save failed!')
            raise

        finally:
            saveDialog.Destroy()

class GraphicsPanel(DrawablePanel):
    def refresh(self):
        dc = wx.MemoryDC(self.drawingBuffer)
        dc.SelectObject(self.drawingBuffer)

        dc.SetBackground(wx.Brush(self.background, style=wx.SOLID))
        dc.Clear()

        gc = wx.GraphicsContext.Create(dc)

        # do not draw if window is very small, right solution? XXX - idfah
        if self.winRadius < 1.0e-3:
            return

        self.draw(gc)

        dc.SelectObject(wx.NullBitmap)
        self.triggerRepaint()

    def draw(self, gc):
        """This method draws the widget using the given drawing context.
        This method should be overridden if you want to draw anything other
        than the blank background.

        Args:
            gc: wx graphics context.
        """
        pass
