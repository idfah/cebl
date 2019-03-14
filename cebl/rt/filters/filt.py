import wx


# note: FilterConfigPanel must be stateless because they are created and destroyed by the Filter page.  Think about whether or not this is wise XXX - idfah
class FilterConfigPanel(wx.Panel):
    def __init__(self, parent, pg, flt, *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.pg = pg
        self.flt = flt

        self.sizer = wx.BoxSizer(wx.VERTICAL)

    def initLayout(self):
        self.SetSizer(self.sizer)
        self.Layout()

class Filter:
    # need to be able to handle different dtypes XXX - idfah
    def __init__(self, inSampRate, inChans, name=None, configPanelClass=FilterConfigPanel):
        self.inSampRate = inSampRate
        self.outSampRate = inSampRate

        self.inChans = inChans
        self.nIn = len(inChans)

        self.outChans = inChans
        self.nOut = self.nIn

        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        self.configPanelClass = configPanelClass

    def getInSampRate(self):
        return self.inSampRate

    def getOutSampRate(self):
        return self.outSampRate

    def setOutSampRate(self, sampRate):
        self.outSampRate = sampRate

    def getInChans(self):
        return self.inChans

    def getOutChans(self):
        return self.outChans

    def setOutChans(self, outChans):
        self.outChans = outChans

    def getNInChans(self):
        return self.nIn

    def getNOutChans(self):
        return self.nOut

    def getName(self):
        """Get the name of this filter.

        Returns:
            The string name describing this filter.
        """
        return self.name

    def apply(self, s):
        raise NotImplementedError("apply not implemented.")

    def genConfigPanel(self, parent, pg, *args, **kwargs):
        """Generate an instance of the configPanelClass, given as an
        argument to the constructor, that can be used to configure
        this filter.
        """
        return self.configPanelClass(parent=parent, pg=pg, flt=self, *args, **kwargs)

class FilterChain:
    def __init__(self, src):
        self.src = src
        self.filters = []

    def push(self, filterClass):
        if not self.filters:
            inSampRate = self.src.getSampRate()
            inChans = self.src.getChanNames()
        else:
            lastFilt = self.filters[-1]
            inSampRate = lastFilt.getOutSampRate()
            inChans = lastFilt.getOutChans()

        newFilt = filterClass(inSampRate, inChans)

        self.filters.append(newFilt)

    def pop(self):
        return self.filters.pop()

    def apply(self, cap):
        for filt in self.filters:
            cap = filt.apply(cap)
        return cap

    def getFilters(self):
        return self.filters

    def getFilter(self, i):
        return self.filters[i]

    def getFilterNames(self):
        return [filt.getName() for filt in self.filters]
