"""Text-stimulus widgets that present text stimuli to the screen.
"""

import copy
import numpy as np
import string
import wx

from cebl import util

from .wxgraphics import DrawablePanel


grid = util.Holder()

grid.enter = u'\u21B5'
grid.back  = u'\u232B'
grid.space = u'__'
grid.upper = u'\u21E7'
grid.lower = u'\u21E9'
grid.ellip = u'\u2026'
grid.left  = u'\u2190'
grid.right = u'\u2192'
grid.up    = u'\u2191'
grid.down  = u'\u2193'

grid.num   = '123'
grid.etc   = 'Etc'
grid.sym   = 'Sym'

grid.normal, grid.highlighted, grid.unhighlighted, grid.selected = range(4)


class GridSpeller(DrawablePanel):
    """Presents a grid of characters.  Rows, columns or groups of characters
    can be highlighted to for P300 speller type applications.
    """

    def __init__(self, parent, copyText='',
                 gridColor=(120,120,120), highlightColor=(240,240,240),
                 unhighlightColor=(40,40,40), selectColor=(255,102,0),
                 copyColor=(255,0,0), feedColor=(255,255,0),
                 gridFont=None, feedFont=None, *args, **kwargs):
        """Initialize a new GridSpeller widget.

        Args:
            parent:             wx parent object.

            copyText:           Text to show above the feedback line for
                                "copy spelling."  If None (default) then no
                                copy text line will be displayed.

            gridColor:

            highlightColor:

            unhighlightColor:

            selectColor:

            copyColor:          A wx.Color specifying the color of the copy
                                text.  Red is the default.

            feedColor:          A wx.Color specifying the color of the feedback
                                text.  Yellow is the default.

            gridFont:

            feedFont:           A wx.Font to use for the feedback text.
                                If None (default) then a default font will be
                                chosen automatically.

            args, kwargs:       Additional arguments passed to DrawablePanel
                                base class.
        """
        self.initDefaultFonts()

        self.gridColor = gridColor
        self.highlightColor = highlightColor
        self.unhighlightColor = unhighlightColor
        self.selectColor = selectColor
        self.copyColor = copyColor
        self.feedColor = feedColor

        self.feedMax = 35

        self.copyText = copyText

        self.gridFont = gridFont
        if gridFont is None:
            self.gridFont = self.defaultStimFont

        self.feedFont = feedFont
        if feedFont is None:
                self.feedFont = self.defaultFeedFont

        self.feedText = ''

        self.setGridLower(refresh=False)
        self.nRows, self.nCols = self.grid.shape
        self.marked = np.zeros_like(self.grid, dtype=np.int)

        DrawablePanel.__init__(self, parent=parent, *args, **kwargs)

    def initDefaultFonts(self):
        """Initialize default fonts.
        """
        fontEnum = wx.FontEnumerator()
        fontEnum.EnumerateFacenames()
        faceList = fontEnum.GetFacenames()
        if 'Utopia' in faceList:
            self.defaultStimFont = wx.Font(pointSize=50, family=wx.FONTFAMILY_ROMAN,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL,
                underline=False, faceName='Utopia')
            #self.defaultFeedFont = wx.Font(pointSize=36, family=wx.FONTFAMILY_ROMAN,
            #    style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL,
            #    underline=True, faceName='Utopia')
        else:
            self.defaultStimFont = wx.Font(pointSize=50, family=wx.FONTFAMILY_ROMAN,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL, underline=False)
            #self.defaultFeedFont = wx.Font(pointSize=32, family=wx.FONTFAMILY_ROMAN,
            #    style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL, underline=True)

        self.defaultFeedFont = wx.Font(pointSize=32, family=wx.FONTFAMILY_MODERN,
            style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL, underline=False)

    def getGridColor(self):
        return self.gridColor

    def setGridColor(self, color, refresh=True):
        self.gridColor = color

        if refresh:
            self.refresh()

    def getGridLayout(self):
        return self.gridLayout

    def setGridLower(self, refresh=True):
        self.grid = np.array([['a', 'b', 'c', 'd', 'e',  'f'],
                              ['g', 'h', 'i', 'j', 'k',  'l'],
                              ['m', 'n', 'o', 'p', 'q',  'r'],
                              ['s', 't', 'u', 'v', 'w',  'x'],
                              ['y', 'z', grid.space, ',', '.', grid.ellip],
                              [grid.upper, grid.num, grid.etc, grid.sym, grid.back, grid.enter]])

        self.gridLayout = grid.lower

        if refresh:
            self.refresh()

    def setGridUpper(self, refresh=True):
        self.grid = np.array([['A', 'B', 'C', 'D', 'E',  'F'],
                              ['G', 'H', 'I', 'J', 'K',  'L'],
                              ['M', 'N', 'O', 'P', 'Q',  'R'],
                              ['S', 'T', 'U', 'V', 'W',  'X'],
                              ['Y', 'Z', grid.space, ',', '.', grid.ellip],
                              [grid.lower, grid.num, grid.etc, grid.sym, grid.back, grid.enter]])

        self.gridLayout = grid.upper

        if refresh:
            self.refresh()

    def setGridNum(self, refresh=True):
        self.grid = np.array([['1', '2', '3', '4', '5',  '6'],
                              ['7', '8', '9', '0', '-',  '='],
                              ['!', '@', '#', '$', '%',  '^'],
                              ['&', '*', '(', ')', '_',  '+'],
                              [':', ';', '"', '?', '/',  '@'],
                              [grid.lower, grid.upper, grid.etc, grid.sym, grid.back, grid.enter]])

        self.gridLayout = grid.num

        if refresh:
            self.refresh()

    def setGridEtc(self, refresh=True):
        self.grid = np.array([['Esc',  ')',    '~',   '[',    'Hom', '|'],
                              ['Tab',  '(',    'Ins', ']',    'End', '/'],
                              ['Ctl',  '}',    'Del', '<',   'Pgu',  '\\'],
                              ['Alt',  '{',    'Win', '>',   'Pgd',  '\''],
                              [grid.left, grid.right, grid.up, grid.down, '`',   '\"'],
                              [grid.lower, grid.num, grid.upper, grid.sym, grid.back, grid.enter]])

        self.gridLayout = grid.etc

        if refresh:
            self.refresh()

    def setGridSym(self, refresh=True):
        self.grid = np.array([[ u'\U0001F44D', u'\U0001F4D6',  u'\U0001F60C', u'\U0001F615', u'\U0001F60E', u'\u2615'    ],
                              [ u'\U0001F44E', u'\U0001F4FA',  u'\U0001F603', u'\U0001F61C', u'\u26C8',     u'\U0001F354'],
                              [ u'\u267F',     u'\u260E',      u'\U0001F61E', u'\U0001F622', u'\u263C',     u'\U0001F355'],
                              [ u'\u266B',     u'\u26A0',      u'\U0001F620', u'\U0001F60D', u'\U0001F319', u'\U0001F36D'],
                              [ u'\U0001F3AE',  u'\U0001F48A', u'\U0001F62E', u'\U0001F607', u'\U0001F320', u'\U0001F377'],
                              [grid.lower, grid.num, grid.etc, grid.upper, grid.back, grid.enter]], dtype=np.unicode)

        self.gridLayout = grid.sym

        if refresh:
            self.refresh()

    def getGridValue(self, row, col):
        return self.grid[row,col]

    def getGridLocation(self, symbol):
        #return np.where(self.grid == unicode(symbol))
        return np.where(self.grid == symbol)

    def getCopyColor(self):
        return self.copyColor

    def setCopyColor(self, color, refresh=True):
        self.copyColor = color

        if refresh:
            self.refresh()

    def getFeedColor(self):
        return self.feedColor

    def setFeedColor(self, color, refresh=True):
        self.feedColor = color

        if refresh:
            self.refresh()

    def getCopyText(self):
        return self.copyText

    def setCopyText(self, copyText, refresh=True):
        """Set the text for the copy line.

        Args:
            copyText:   A string of text to display in the copy line.  If None
                        then no copy line will be displayed.
        """
        self.copyText = copyText

        if len(self.copyText) > self.feedMax:
            self.copyText = self.copyText[-self.feedMax:]

        if refresh:
            self.refresh()

    def getFeedText(self):
        return self.feedText

    def setFeedText(self, feedText, refresh=True):
        """Set the text for the feedback line.

        Args:
            feedText:   A string of text to display in the feedback line.
        """
        self.feedText = feedText

        if refresh:
            self.refresh()

    def appendFeedText(self, symbol, refresh=True):
        """Append a character to the feedback line or
        send an instruction to the grid, e.g., backspace or change grid.

        Args:
            symbol:   A string or instruction.
        """
        if symbol is None:
            return

        isText = False

        if symbol == grid.upper:
            self.setGridUpper(refresh=False)

        elif symbol == grid.lower:
            self.setGridLower(refresh=False)

        elif symbol == grid.num:
            self.setGridNum(refresh=False)

        elif symbol == grid.etc:
            self.setGridEtc(refresh=False)

        elif symbol == grid.sym:
            self.setGridSym(refresh=False)

        elif symbol == grid.back:
            self.feedText = self.feedText[:-1]

        elif symbol == grid.enter:
            pass

        elif symbol == grid.space:
            self.feedText += ' '
            isText = True

        elif len(symbol) > 1:
            isText = True

        else:
            self.feedText += symbol
            isText = True

        if len(self.feedText) > self.feedMax:
            self.feedText = self.feedText[-self.feedMax:]

        if refresh:
            self.refresh()

        return isText

    def getHighlightColor(self):
        return self.highlightColor

    def setHighlightColor(self, color, refresh=True):
        self.highlightColor = color

        if refresh:
            self.refresh()

    def getUnhighlightColor(self):
        return self.unhighlightColor

    def setUnhighlightColor(self, color, refresh=True):
        self.unhighlightColor = color

        if refresh:
            self.refresh()

    def highlightCol(self, col, refresh=True):
        """Highlight a given row.  All previous highlights will be cleared.

        Args:
            row:    The index of the row to highlight.
        """
        self.removeHighlight(refresh=False)

        # updating the highlights setting
        self.marked[:,col] = grid.highlighted

        if refresh:
            self.refresh()

    def highlightRow(self, row, refresh=True):
        """Highlight a given column.  All previous highlights will be cleared.

        Args:
            col:    The index of the column to highlight.
        """
        self.removeHighlight(False) # removing previous highlight without refreshing the page

        # updating the highlights settings
        self.marked[row,:] = grid.highlighted

        if refresh:
            self.refresh()

    def getSelectColor(self):
        return self.selectColor

    def setSelectColor(self, color, refresh=True):
        self.selectColor = color

        if refresh:
            self.refresh()

    def selectSymbol(self, row, column, refresh=True):
        """Highlihgt a symbol in given row and column.
        All other symbols will be unhighlighted with dark color

        Args:
            row:    The index of the row of symbol to highlight
            col:    The index of the column of symbol to highlight
        """
        self.marked = np.zeros((self.nRows, self.nCols), dtype=np.int)
        self.marked[:,:] = grid.unhighlighted
        self.marked[row,column] = grid.selected

        if refresh:
            self.refresh()

    def highlight(self, how, refresh=True):
        """Highlights the grid according to the input matrix

        Args:
            how:    A numpy array that gives a way to highlight the grid:
                        (0) grid.normal:  normal
                        (1) grid.highlighted:  highlighted
                        (2) grid.unhighlighted:  unhighlighted
                        (3) grid.selected:  selected
        """
        # checking, if it has the same dimensions
        if how.shape == self.marked.shape:
            self.marked = how.copy()
        else:
            raise Exception('Shape of how %s does not match grid shape %s.' %\
                            (str(how.show), str(self.marked.shape)))

        if refresh:
            self.refresh()

    def removeHighlight(self, refresh=True):
        """removes every highlight in the grid
        """
        self.marked[:,:] = grid.normal

        if refresh:
            self.refresh()

    def draw(self, dc):
        """Draw all items.  This class should only be called automatically
        when self.refresh is called.

        Args:
            dc:     wx.DC drawing context for drawing the graphics in this widget.
                    Calling the methods of dc will modify the drawing accordingly.
        """
        self.drawGrid(dc) # draw the grid and highlights
        self.drawCopy(dc) # draw the copy line
        self.drawFeed(dc) # draw the feedback line

    def drawGrid(self, dc):
        """Draw the grid characters and highlights.

        Args:
            dc:     wx.DC drawing context.  Calling the methods of dc will
                    modify the drawing accordingly.
        """
        ##dc.SetFont(self.gridFont) # setting the font for the text

        # figuring out what is the y-offset for the grid of symbols
        yOffset = self.winHeight*32/340.0
        if len(self.copyText) == 0:
            yOffset = self.winHeight * 14/340.0

        # figuring out the distances between each symbol horizontaly and vertically
        dx = (self.winWidth+0.0)/(self.nCols+1)
        dy = (self.winHeight-yOffset)/(self.nRows+1)

        for i in range(self.nRows):
            for j in range(self.nCols):
                # select the color and font for the next symbol
                mark = self.marked[i,j]

                if mark == grid.normal:
                    dc.SetTextForeground(self.gridColor)

                elif mark == grid.highlighted:
                    dc.SetTextForeground(self.highlightColor)
                    self.gridFont.SetWeight(wx.FONTWEIGHT_BOLD)

                elif mark == grid.unhighlighted:
                    dc.SetTextForeground(self.unhighlightColor)

                elif mark == grid.selected:
                    dc.SetTextForeground(self.selectColor)
                    self.gridFont.SetWeight(wx.FONTWEIGHT_BOLD)

                else:
                    raise Exception('Invalid mark value %d.' % mark)

                dc.SetFont(self.gridFont)

                # get extents of symbol
                text = self.grid[i,j]
                textWidth, textHeight = dc.GetTextExtent(text)

                # draw next symbol
                #dc.DrawText(self.grid[i,j], (j+0.7)*dx, (i+0.7)*dy+yOffset)
                dc.DrawText(self.grid[i,j],
                        (j+1.0)*dx-textWidth/2.0,
                        (i+1.0)*dy+yOffset-textHeight/2.0)
                self.gridFont.SetWeight(wx.FONTWEIGHT_NORMAL)

    def drawCopy(self, dc):
        """Draw the copy text.

        Args:
            dc:     wx.DC drawing context.  Calling the methods of dc will
                    modify the drawing accordingly.
        """
        if len(self.copyText) == 0:
            return

        dx = (self.winWidth+0.0)/(self.nCols+1)

        dc.SetFont(self.feedFont)
        dc.SetTextForeground(self.copyColor)

        #dc.DrawText(self.copyText, dx*0.7,4)
        cornerWidth, cornerHeight = dc.GetTextExtent(self.grid[0,0])
        dc.DrawText(self.copyText, dx-cornerWidth/2.0,4)

    def drawFeed(self, dc):
        """Draw the feedback message.

        Args:
            dc:     wx.DC drawing context.  Calling the methods of dc will
                    modify the drawing accordingly.
        """
        dc.SetFont(self.feedFont)
        dc.SetTextForeground(self.feedColor)

        yOffset = 27*self.winHeight/340.0

        if len(self.copyText) == 0:
            yOffset = 4

        dx = (self.winWidth+0.0)/(self.nCols+1)

        dc.SetFont(self.feedFont)
        dc.SetTextForeground(self.feedColor)

        #dc.DrawText(self.feedText,dx*0.7, yOffset)
        cornerWidth, cornerHeight = dc.GetTextExtent(self.grid[0,0])
        dc.DrawText(self.feedText,dx-cornerWidth/2.0, yOffset)
