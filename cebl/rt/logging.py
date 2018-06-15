import wx
import sys


class LogTarget(wx.Log):
    def __init__(self, *args, **kwargs):
        wx.Log.__init__(self, *args, **kwargs)
        self.textCtrls = []

    def DoLogTextAtLevel(self, level, msg):
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
