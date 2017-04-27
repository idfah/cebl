import wx

import matplotlib
#matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg


class PyPlotNavbar(NavigationToolbar2WxAgg):
    def __init__(self, canvas):
        self.canvas = canvas
        NavigationToolbar2WxAgg.__init__(self, self.canvas)
        self.DeleteToolByPos(7) # remove margin config
