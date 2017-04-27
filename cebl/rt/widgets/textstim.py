"""Text-stimulus widgets that present text stimuli to the screen.
"""

import numpy as np
import random
import string
import wx

from cebl import util

from wxgraphics import DrawablePanel


class TextStim(DrawablePanel):
    """Present centered text or letter on the center of the screen along with
    a feedback message in the corner.  Flashing text stimuli can be created
    by turning the message on and off.
    """
    def __init__(self, parent, stimText='', stimColor=(255,255,0), stimFont=None,
                 feedText='', feedColor=(255,0,0), feedFont=None, feedLength=10,
                 background=(0,0,0), style=0, *args, **kwargs):
        """Initialize a new TextStim widget.

        Args:
            parent:             wx parent object.

            stimText:

            stimColor:          A wx.Color object specifying the color of the
                                stimulus text.  Defaults to yellow.

            stimFont:

            feedText:

            feedColor:          A wx.Color object specifying the color of the
                                feedback message.  Defaults to red.

            feedFont:

            feedLength:         Maximum number of characters in the feedback
                                message.  If this length is exceeded, the
                                message will be truncated at the beginning.

            args, kwargs:       Additional arguments passed to DrawablePanel
                                base class.
        """
        self.feedLength = feedLength

        self.stimText = stimText
        self.stimColor = stimColor

        self.feedText = feedText
        self.feedColor = feedColor
        self.feedLength = feedLength

        self.initDefaultFonts()
        if stimFont is None:
            self.stimFont = self.defaultStimFont
        else:
            self.stimFont = stimFont

        if feedFont is None:
            self.feedFont = self.defaultFeedFont
        else:
            self.feedFont = feedFont

        DrawablePanel.__init__(self, parent=parent, *args, **kwargs)

    def initDefaultFonts(self):
        """Initialize default fonts.
        """
        fontEnum = wx.FontEnumerator()
        fontEnum.EnumerateFacenames()
        faceList = fontEnum.GetFacenames()
        if 'Utopia' in faceList:
            self.defaultStimFont = wx.Font(pointSize=196, family=wx.FONTFAMILY_ROMAN,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL,
                underline=False, face='Utopia')
            self.defaultFeedFont = wx.Font(pointSize=32, family=wx.FONTFAMILY_ROMAN,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL,
                underline=True, face='Utopia')
        else:
            self.defaultStimFont = wx.Font(pointSize=196, family=wx.FONTFAMILY_ROMAN,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL, underline=False)
            self.defaultFeedFont = wx.Font(pointSize=32, family=wx.FONTFAMILY_ROMAN,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL, underline=True)

    def setStimText(self, stimText=None, refresh=True):
        """Set the stimulus text.

        Args:
            stimText:   String containing the stimulus text.
                        If None (default) the stimulus text
                        will not be drawn.
        """
        self.stimText  = stimText

        if refresh:
            self.refresh()

    def setStimColor(self, color=(255,255,0), refresh=True):
        """Set the stimulus color.

        Args:
            color:  A wx.Color object specifying the color of the stimulus
                    text.  Defaults to yellow.
        """
        self.stimColor = color

        if refresh:
            self.refresh()

    def setStimFont(self, font=None, refresh=True):
        """Set the stimulus font.

        Args:
            font:   A wx.Font object specifying the font for the stimulus
                    text.  If None (default) then the font will be set to
                    a reasonable default.
        """
        if font is None:
            self.stimFont = self.defaultStimFont
        else:
            self.stimFont = font

        if refresh:
            self.refresh()

    def clearStim(self):
        """Set the stimulus text to be empty.
        """
        self.stimText = None

        if refresh:
            self.refresh()

    def setFeedText(self, feedText=None, refresh=True):
        """Set the feedback message text.

        Args:
            feedText:   String containing the feedback message.
                        If None (default) the feedback message
                        will not be drawn.
        """
        self.feedText  = feedText

        if refresh:
            self.refresh()

    def setFeedColor(self, color=(255,0,0), refresh=True):
        """Set the feedback color.

        Args:
            color:  A wx.Color object specifying the color of the feedback
                    message.  Defaults to red.
        """
        self.feedColor = color

        if refresh:
            self.refresh()

    def setFeedFont(self, font=None, refresh=True):
        """Set the feedback font.

        Args:
            font:   A wx.Font object specifying the font for the feedback
                    message.  If None (default) then the font will be set
                    to a reasonable default.
        """
        if font is None:
            self.feedFont = self.defaultFeedFont
        else:
            self.feedFont = font

        if refresh:
            self.refresh()

    def setFeedLength(self, length, refresh=True):
        """Set the length of the feedback message.  If the feedback text
        exceeds this length, the message will be truncated at the beginning.

        Args:
            length:     Integer specifying the maximum length of the
                        feedback message.
        """
        self.feedLength = length
        self.refresh()

    def clearFeed(self, refresh=True):
        """Set the feedback message to be empty.
        """
        self.feedText = None

        if refresh:
            self.refresh()

    def draw(self, dc):
        """Draw all items.  This class should only be called automatically
        when self.refresh is called.

        Args:
            dc:     wx.DC drawing context.  Calling the methods of dc will
                    modify the drawing accordingly.
        """
        self.drawStim(dc)
        self.drawFeed(dc)

    def drawStim(self, dc):
        """Draw the stimulus text.

        Args:
            dc:     wx.DC drawing context.  Calling the methods of dc will
                    modify the drawing accordingly.
        """
        if self.stimText is None:
            return

        dc.SetTextForeground(self.stimColor)
        dc.SetFont(self.stimFont)

        textWidth, textHeight = dc.GetTextExtent(self.stimText)
        posX, posY = (self.winWidth-textWidth)//2, (self.winHeight-textHeight)//2

        dc.DrawText(self.stimText, posX, posY)

    def drawFeed(self, dc):
        """Draw the feedback message.

        Args:
            dc:     wx.DC drawing context.  Calling the methods of dc will
                    modify the drawing accordingly.
        """
        if self.feedText is None:
            return

        dc.SetTextForeground(self.feedColor)
        dc.SetFont(self.feedFont)

        trimmedFeedText = \
            self.feedText[max(len(self.feedText)-self.feedLength,0):]

        dc.DrawText(trimmedFeedText, 10, 5)


class IdleTextStim(TextStim):
    """Present a single centered character or text message as an
    idle fixation stimulus.
    """
    def __init__(self, parent, stim='*', *args, **kwargs):
        """Initialize a new IdleTextStim.

        Args:
            parent:         wx parent object.

            stim:           String containing the idle stimulus.
    
            args, kwargs:   Additional arguments passed to TextStim
                            base class.
        """
        self.stim = stim

        TextStim.__init__(self, parent=parent, *args, **kwargs)

    def showStim(self):
        """Show the idle stimulus.
        """
        self.setStimText(self.stim)

    def hideStim(self):
        """Hide the idle stimulus.
        """
        self.clearStim()
