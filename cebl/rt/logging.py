import wx
import sys


class LogTarget(wx.PyLog):
    def __init__(self, *args, **kwargs):
        wx.PyLog.__init__(self, *args, **kwargs)

        self.textCtrls = []

    # should use DoLogTextAtLevel in newer versions of wxPython XXX - idfah
    def DoLog(self, level, msg, time):
        if level == wx.LOG_Warning:
            caption = 'Warning'
        elif level == wx.LOG_Error:
            caption = 'Error'
        else:
            caption = 'Message'

        fullMessage = caption + ': ' + msg + '\n'
        if level == wx.LOG_Error:
            sys.stderr.write(fullMessage)
            sys.stderr.flush()

        for tctrl in self.textCtrls:
            tctrl.AppendText(fullMessage)

        if level <= wx.LOG_Warning:
            dialog = wx.MessageDialog(None, message=msg, caption=caption,
                style=wx.ICON_ERROR | wx.OK)
            dialog.ShowModal()
            dialog.Destroy()

    def addTextCtrl(self, tctrl):
        self.textCtrls.append(tctrl)
