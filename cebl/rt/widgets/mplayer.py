import os

import wx
from wx.lib import newevent
import wx.media as wxm

from .control import ControlBox


MPlayerFinishedEvent, EVT_MPLAYER_FINISHED = newevent.NewCommandEvent()


def _filesAndPaths(cwd):
    files = os.listdir(cwd)
    files = [f for f in files if not f.startswith(".")]
    filesPath = [cwd + os.path.sep + f for f in files]

    return files, filesPath

class MPlayerPanel(wx.Panel):
    def __init__(self, parent=None, cwd="~", *args, **kwargs):
        wx.Panel.__init__(self, parent=parent, *args, **kwargs)

        self.initNavbar()
        self.initMusicLists()
        self.initMediaCtrl()
        self.initLayout()

        self.previewSecs = 10.0
        self.setCWD(cwd)

    def initMediaCtrl(self):
        #self.mediaControlBox = ControlBox(self, label="Media", orient=wx.HORIZONTAL)

        self.mediaCtrl = wxm.MediaCtrl(self, style=wx.SIMPLE_BORDER)
        self.mediaCtrl.Hide()
        #self.mediaControlBox.Add(self.mediaCtrl, proportion=1,
        #        flag=wx.ALL | wx.EXPAND, border=10)

        self.Bind(wxm.EVT_MEDIA_STOP, self.onStop)

    def initNavbar(self):
        self.navControlBox = ControlBox(self, label="Navigation",
            orient=wx.HORIZONTAL)

        self.playButton = wx.Button(self, label="Play")
        self.Bind(wx.EVT_BUTTON, self.play, self.playButton)
        self.navControlBox.Add(self.playButton, proportion=0,
                flag=wx.ALL | wx.EXPAND, border=5)

        self.rewAlbumButton = wx.Button(self, label="Album <-")
        self.Bind(wx.EVT_BUTTON, self.rewAlbum, self.rewAlbumButton)
        self.navControlBox.Add(self.rewAlbumButton, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

        self.forAlbumButton = wx.Button(self, label="Album ->")
        self.Bind(wx.EVT_BUTTON, self.forAlbum, self.forAlbumButton)
        self.navControlBox.Add(self.forAlbumButton, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

        self.rewSongButton = wx.Button(self, label="Song <-")
        self.Bind(wx.EVT_BUTTON, self.rewSong, self.rewSongButton)
        self.navControlBox.Add(self.rewSongButton, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

        self.forSongButton = wx.Button(self, label="Song ->")
        self.Bind(wx.EVT_BUTTON, self.forSong, self.forSongButton)
        self.navControlBox.Add(self.forSongButton, proportion=0,
                flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

        #self.previewButton = wx.Button(self, label="Preview")
        #self.Bind(wx.EVT_BUTTON, self.preview, self.previewButton)
        #self.navControlBox.Add(self.previewButton, proportion=0,
        #        flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

        #self.stopButton = wx.Button(self, label="Stop")
        #self.Bind(wx.EVT_BUTTON, self.stop, self.stopButton)
        #self.navControlBox.Add(self.stopButton, proportion=0,
        #        flag=wx.BOTTOM | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

    def initMusicLists(self):
        self.albumControlBox = ControlBox(self, label="Albums", orient=wx.VERTICAL)
        self.albumListBox = wx.ListBox(self, choices=[], style=wx.LB_SINGLE)
        self.albumControlBox.Add(self.albumListBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=5)

        self.songControlBox = ControlBox(self, label="Songs", orient=wx.VERTICAL)
        self.songListBox = wx.ListBox(self, choices=[], style=wx.LB_SINGLE)
        self.songControlBox.Add(self.songListBox, proportion=1,
                flag=wx.ALL | wx.EXPAND, border=5)

    def initLayout(self):
        sizer = wx.BoxSizer(orient=wx.VERTICAL)

        sizer.Add(self.navControlBox, proportion=0,
                flag=wx.ALL, border=10)
        sizer.Add(self.albumControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        sizer.Add(self.songControlBox, proportion=1,
                flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)
        #sizer.Add(self.mediaControlBox, proportion=1,
        #        flag=wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.EXPAND, border=10)

        self.SetSizer(sizer)
        self.Layout()

    def getCWD(self):
        return self.cwd

    def setCWD(self, cwd):
        self.cwd = os.path.expanduser(cwd)
        self.updateAlbums()
        self.updateSongs()

    def updateAlbums(self):
        files, filesPath = _filesAndPaths(self.cwd)

        self.albumList = [f for f, fp in zip(files, filesPath)
                          if os.path.isdir(fp)]

        self.albumList.sort()

        self.albumListBox.Clear()

        if self.albumList:
            self.albumListBox.AppendItems(self.albumList)
            self.albumListBox.SetSelection(0)
            self.albumListBox.EnsureVisible(0)

    def updateSongs(self):
        album = self.albumList[self.albumListBox.GetSelection()]
        albumPath = self.cwd + os.path.sep + album

        files, filesPath = _filesAndPaths(albumPath)

        self.songList = [f for f, fp in zip(files, filesPath)
                         if os.path.isfile(fp) and fp.endswith(".flac")]

        self.songList.sort()

        self.songListBox.Clear()

        if self.songList:
            self.songListBox.AppendItems(self.songList)
            self.songListBox.SetSelection(0)
            self.albumListBox.EnsureVisible(0)

    def loadAndPlay(self):
        #self.mediaCtrl.Load("/home/idfah/tests/python/wx/reggae.wav")
        song = self.songList[self.songListBox.GetSelection()]
        album = self.albumList[self.albumListBox.GetSelection()]

        songPath = self.cwd + os.path.sep + album + os.path.sep + song
        status = self.mediaCtrl.Load(songPath)
        if not status:
            raise RuntimeError("Failed to load song %s." % str(songPath))

        self.mediaCtrl.ShowPlayerControls()
        self.mediaCtrl.Play()

    def play(self, event=None):
        self.loadAndPlay()
        wx.CallLater(1000.0*3.0*self.previewSecs, self.stop)

    def stop(self, event=None):
        state = self.mediaCtrl.GetState()
        if state != wxm.MEDIASTATE_STOPPED:
            self.nextOnStop = False
            self.mediaCtrl.Stop()

    def onStop(self, event=None):
        if self.nextOnStop:
            self.forSong()
        self.nextOnStop = True

        wx.PostEvent(self, MPlayerFinishedEvent(id=wx.ID_ANY))

    def rewAlbum(self, event=None):
        curIndex = self.albumListBox.GetSelection()
        self.albumListBox.SetSelection((curIndex - 1) %
                len(self.albumList))
        self.updateSongs()

    def forAlbum(self, event=None):
        curIndex = self.albumListBox.GetSelection()
        self.albumListBox.SetSelection((curIndex + 1) %
                len(self.albumList))
        self.updateSongs()

    def rewSong(self, event=None):
        curIndex = self.songListBox.GetSelection()
        self.songListBox.SetSelection((curIndex - 1) %
                len(self.songList))

    def forSong(self, event=None):
        curIndex = self.songListBox.GetSelection()
        self.songListBox.SetSelection((curIndex + 1) %
                len(self.songList))

    def preview(self, event=None):
        self.loadAndPlay()
        wx.CallLater(1000.0*self.previewSecs, self.stop)
