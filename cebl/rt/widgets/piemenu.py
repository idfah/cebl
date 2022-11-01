import numpy as np
import wx

from cebl import util

from .wxgraphics import GraphicsPanel


# does this need to be a command event? XXX - idfah
wxEVT_PIEMENU_SELECT = wx.NewEventType()
EVT_PIEMENU_SELECT = wx.PyEventBinder(wxEVT_PIEMENU_SELECT, 1)
class PieMenuSelectEvent(wx.PyCommandEvent):
    def __init__(self, choice, id=wx.ID_ANY):
        wx.PyCommandEvent.__init__(self, id=id,
            eventType=wxEVT_PIEMENU_SELECT)
        self.choice = choice

    def getChoice(self):
        return self.choice


class PieMenu(GraphicsPanel):
    """The PieMenu widget consists of a number of menu choices placed around
    a circular pie-shaped menu with bars that can grow toward the choices.
    """

    def __init__(self, parent,
            choices=('fist', 'leg', 'count', 'song', 'place', 'relax', 'talk', 'face'),
            colors=('blue', 'green', 'red', 'yellow', 'turquoise', 'blue violet', 'maroon', 'orange'),
            background='white', font=None, rotation=np.pi/2.0, **kwargs):
        """Initialize a new PieMenu.

        Args:
            parent:     wx parent.

            choices:    Tuple of strings containing titles for each
                        cell in the pie menu.

            colors:     Tuple of colors for each cell in the pie
                        menu.  Each color can be a string, RGB hex
                        code or tuple of RGB integer values.

            background: Background color.  May be a string, RGB hex
                        code or tuple of RGB integer values.

            font:       wx.Font used for drawing text.  If None, a
                        default font will be chosen automagically.

            rotation:   Amount in radians to rotate the cells of
                        the menu.  Cell boundaries start at the top
                        by default.

            **kwargs:   Passed to wxgraphics.GraphicsPanel parent class.
        """
        # setup menu choices
        self.choices = choices
        self.nChoices = len(self.choices)

        # menu colors (before cycling)
        self.colors = colors

        # initialize font for menu choices
        self.initFont(font)

        # menu rotation
        self.rotation = rotation

        # initialize menu
        self.initPie()

        GraphicsPanel.__init__(self, parent=parent, background=background, **kwargs)

    def initFont(self, font):
        """Initialize font for drawing text.
        """
        # if none, use SWISS font
        if font is None:
            self.font = \
                wx.Font(pointSize=10, family=wx.FONTFAMILY_SWISS,
                        style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_BOLD,
                        underline=False)#, faceName='Utopia')

        # otherwise, use given font
        else:
            self.font = font

    def initPie(self):
        """Initialize angles between menu cells.
        """
        # initialize menu bars
        self.bars = {c: 0.0 for c in self.choices}

        # sets containing names of cells currently
        # highlighted with jump and pop effects.
        self.highlightPop = set()
        self.highlightJump = set()

        # colors
        self.curColors = util.cycle(self.colors, self.nChoices)

        # angles between ends of pie menu choices
        self.angle = 2.0 * np.pi / float(self.nChoices)
        self.angles = self.angle * np.arange(self.nChoices+1)

        # angles between centers of pie menu choices
        self.midAngles = self.angles + self.angle / 2.0

    def setChoices(self, choices, refresh=True):
        self.choices = choices
        self.nChoices = len(self.choices)

        self.initPie()

        if refresh:
            self.refresh()

    def setRotation(self, rotation, refresh=True):
        self.rotation = rotation

        self.initPie()

        if refresh:
            self.refresh()

    def zeroBars(self, refresh=True):
        """Set all menu bars to zero.
        """
        for choice in self.choices:
            self.bars[choice] = 0.0

        if refresh:
            self.refresh()

    def clearAllHighlights(self, refresh=True):
        """Clear all highlights from all cells.
        """
        self.highlightPop.clear()
        self.highlightJump.clear()

        if refresh:
            self.refresh()

    def clearHighlight(self, choice, refresh=True):
        """Clear all highlight from a single cell.

        Args:
            choice: Name of cell where highlights should
                    be cleared.
        """
        if choice in self.highlightPop:
            self.highlightPop.remove(choice)

        if choice in self.highlightJump:
            self.highlightJump.remove(choice)

        if refresh:
            self.refresh()

    def highlight(self, choice, style='pop', secs=None, refresh=True):
        """Turn on a highlight for a given cell.

        Args:
            choice: Name of cell where highlight should be
                    turned on.

            style:  String describing the type of highlight.
                    Current possible values are 'jump' or
                    'pop'.

            secs:   Floating point seconds until the
                    highlight should be turned off.  If
                    None, highlight will be left on until
                    manually cleared.
        """
        if style.lower() == 'pop':
            self.highlightPop.add(choice)
        elif style.lower() == 'jump':
            self.highlightJump.add(choice)
        else:
            raise RuntimeError('Unknown highlight style %s.' % style)

        if refresh:
            self.refresh()

        if secs is not None:
            wx.CallLater(int(1000 * secs), self.clearHighlight, choice=choice, refresh=refresh)

    def growBar(self, choice, amount, refresh=True):
        """Grow a selection bar toward a menu cell.

        Args:
            choice: String name of the menu cell to grow
                    the bar toward.

            amount: Floating point amount to grow the
                    bar.  Must be between 0.0 and 1.0
                    with 1.0 growing the bar all the
                    way to the cell and 0.0 not
                    growing the bar at all.

        Returns:
            True if the bar for choice meets or exceeds
            1.0 and False otherwise.

        Events:
            A PieMenuSelectEvent is posted if the bar
            length for choice meets or exceeds 1.0.
        """
        self.bars[choice] += amount

        if self.bars[choice] < 0.0:
            self.bars[choice] = 0.0

        if refresh:
            self.refresh()

        if np.isclose(self.bars[choice], 1.0) or self.bars[choice] > 1.0:
            self.bars[choice] = 1.0
            wx.PostEvent(self, PieMenuSelectEvent(choice=choice, id=wx.ID_ANY))
            return True
        else:
            return False

    def setBar(self, choice, amount, refresh=True):
        """Set the length of the selection bar associated
        with a given menu cell.

        Args:
            choice: String name of the menu cell to set
                    the bar length for.

            amount: Floating point length of the bar.
                    Must be between 0.0 and 1.0 with
                    1.0 being all the way to the cell
                    and 0.0 having no length at all.
        """
        self.bars[choice] = amount

        if refresh:
            self.refresh()

    def getSelection(self):
        for choice in self.choices:
            if np.isclose(self.bars[choice], 1.0) or self.bars[choice] > 1.0:
                return choice

        return None

    def draw(self, gc):
        """Draw the widget.
        """
        gc.PushState()

        # center of the pie menu
        centerX = self.winWidth / 2.0
        centerY = self.winHeight / 2.0

        # translate to center and rotate
        gc.Translate(centerX, centerY)
        gc.Rotate(-self.rotation)

        self.drawBars(gc)
        self.drawCenterPolygon(gc)
        self.drawCells(gc)
        self.drawCellText(gc)

        gc.PopState()

    def drawCells(self, gc):
        """Draw the menu cells.
        """
        gc.PushState()

        # setup the pen for drawing the
        # outline of the cells
        gc.SetPen(wx.Pen((80,80,80), 2))

        # for each cell
        for i,choice in enumerate(self.choices):
        #in range(len(self.angles)-1):
            gc.PushState()

            # set color for filling cell
            gc.SetBrush(wx.Brush(self.curColors[i]))

            # cell border is a contigous path
            path = gc.CreatePath()

            # inner and outer radii of cells
            smallRadius = 0.70 * self.winRadius
            bigRadius = 0.90 * self.winRadius

            if choice in self.highlightPop:
                smallRadius = 0.65 * self.winRadius
                bigRadius = 0.95 * self.winRadius

            if choice in self.highlightJump:
                gc.Scale(1.1, 1.1)

            # draw arcs defining cell border
            path.AddArc(0, 0, bigRadius, self.angles[i], self.angles[i+1], True)
            path.AddArc(0, 0, smallRadius, self.angles[i+1], self.angles[i], False)

            # close the path and draw it
            path.CloseSubpath()
            gc.DrawPath(path)

            gc.PopState()

        gc.PopState()

    def drawCellText(self, gc):
        """Draw the text in each cell.
        """
        # centers for text in each cell
        radius = 0.8 * self.winRadius
        textCenters = radius * np.array((np.cos(self.midAngles), np.sin(self.midAngles))).T

        gc.SetFont(self.font, wx.Colour('black'))

        # for each cell
        for i,choice in enumerate(self.choices):
            gc.PushState()

            # translate and rotate to text center
            gc.Translate(textCenters[i,0], textCenters[i,1])
            gc.Rotate(self.rotation)

            # find width and height of text
            textW, textH = gc.GetTextExtent(choice)
            textWPlus = textW + 0.2 * textW
            textHPlus = textH + 0.1 * textH

            # set pen color for text
            gc.SetPen(wx.Pen((80,80,80), 2))

            # set fill color for text backdrop
            #gc.SetBrush(wx.Brush((240,240,240)))
            gc.SetBrush(wx.Brush((255,255,255,200)))

            # draw a box behind the text to increase visibility
            scale = 0.01 * self.winRadius
            gc.DrawRoundedRectangle(-scale*textWPlus/2.0,
                                    -scale*textHPlus/2.0,
                                     scale*textWPlus,
                                     scale*textHPlus, 1)

            # scale and draw text
            gc.Scale(scale, scale)
            gc.DrawText(choice, -textW/2.0, -textH/2.0)

            gc.PopState()

    def drawCenterPolygon(self, gc):
        """Draw the polygon at the center of the pie menu.
        """
        gc.PushState()

        # set border and fill colors
        gc.SetPen(wx.Pen((80,80,80), 2))
        gc.SetBrush(wx.Brush((0,0,0)))

        # radius of circle that circumscribes the polygon
        radius = 0.1 * self.winRadius
        points = radius * np.array((np.cos(self.angles), np.sin(self.angles))).T

        # draw the polygon
        gc.DrawLines(points)

        gc.PopState()

    def drawBars(self, gc):
        """Draw the bars reaching to the menu cells.
        """
        gc.SetPen(wx.Pen((80,80,80), 2))

        # figure bar width using number of bars and apothem of polygon
        r = 0.1 * self.winRadius
        barWidth = np.sqrt((r*(np.cos(self.angle)-1.0))**2+(r*np.sin(self.angle))**2)
        apothem = r * np.cos(self.angle/2.0)

        # for each angle, color, choice triple
        for angle, color, choice in zip(self.midAngles, self.curColors, self.choices):
            # set fill color
            gc.SetBrush(wx.Brush(color))

            # figure bar length
            barLength = self.bars[choice] * (0.70*self.winRadius - apothem)
                                            # smallRadius

            # rotate and draw bar
            gc.PushState()
            gc.Rotate(angle)
            gc.DrawRectangle(apothem, -barWidth/2.0, barLength, barWidth)
            gc.PopState()
