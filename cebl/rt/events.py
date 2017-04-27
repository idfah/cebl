"""Custom top-level events.  Only events that do not
clearly belong to a single module should go here.
Otherwise, the event should go in the associated module.
"""

import wx
from wx.lib import newevent


#SetSourceEvent, EVT_SET_SOURCE = newevent.NewCommandEvent()
#wxEVT_SET_SOURCE = wx.NewEventType()
#EVT_SET_SOURCE = wx.PyEventBinder(wxEVT_SET_SOURCE, 1)
#class SetSourceEvent(wx.PyCommandEvent):
#    def __init__(self, sourceName, id=wx.ID_ANY):
#        wx.PyCommandEvent.__init__(self, id=id,
#            eventType=wxEVT_SET_SOURCE)
#        self.sourceName = sourceName

UpdateStatusEvent, EVT_UPDATE_STATUS = newevent.NewCommandEvent()
FullScreenEvent, EVT_FULLSCREEN = newevent.NewCommandEvent()
