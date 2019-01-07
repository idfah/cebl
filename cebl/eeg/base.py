"""Module containing EEG base class and related routines.
"""

class EEGBase:
    """Base class for all EEG types.
    """
    def __init__(self, nObs, nChan, sampRate=256.0, chanNames=None, deviceName=''):
        """Construct a new EEGBase instance.

        Args:
            nObs:       The number of observations in the eeg data.

            nChan:      The number of channels in the eeg data.

            sampRate:   The sampling rate (frequency) in samples-per-second
                        (Hertz) of the eeg data.  This defaults to 256Hz.

            chanNames:  A list of names of the channels in the eeg data.
                        If None (default) then the channel names are set
                        to '1', '2', ... 'nChan'.

            deviceName: The name of the device used to record the eeg data.
        """
        self.nObs = nObs
        self.nChan = nChan
        self.setSampRate(sampRate)
        self.setChanNames(chanNames)
        self.setDeviceName(deviceName)

        self.nSec = self.nObs / float(self.sampRate)

    def getSampRate(self):
        """Get the sampling rate of the eeg data in samples-per-second (Hertz).
        """
        return self.sampRate

    def setSampRate(self, sampRate):
        """Set the sampling rate of the eeg data in samples-per-second (Hertz).
        """
        self.sampRate = float(sampRate)
        return self

    def getNChan(self):
        """Get the number of channels in the eeg data.
        """
        return self.nChan

    def getChanNames(self, chans=None):
        """Get a list of strings containing the names of the channels
        in the eeg data.

        Args:
            chans:  A tuple containing string channel names (case insensetive)
                    or integer channel indices or any combination of the two.
                    If None (default) then all channel names are assumed.

        Returns:
            A list of strings specifying the names of the channels given
            by the chans argument.  The list will contain None entries
            where a non-existent channel name or index is given.
        """
        if chans is None:
            return self.chanNames

        chanNames = []
        lowerChanNames = [c.lower() for c in self.chanNames]
        for c in chans:
            if isinstance(c, str):
                if c.lower() in lowerChanNames:
                    i = lowerChanNames.index(c.lower())
                    chanNames.append(self.chanNames[i])
                else:
                    chanNames.append(None)
            else:
                if c < self.nChan and c >= 0:
                    chanNames.append(self.chanNames[c])
                else:
                    chanNames.append(None)

        return chanNames

    def setChanNames(self, chanNames=None):
        """Set the names of the channels for the eeg data.

        Args:
            chanNames:  A list or tuple of channel names.  If None (default)
                        then the channel names are set to '1', '2', ... 'nChan'.
        """
        if chanNames is None:
            chanNames = [str(i) for i in range(self.nChan)]

        if len(chanNames) != self.nChan:
            raise RuntimeError('Length of chanNames ' + str(len(chanNames)) + \
                            ' does not match number of channels ' + str(self.nChan))

        self.chanNames = list(chanNames)

    def getChanIndices(self, chans=None):
        """Get a list of integer indices into the eeg data for the given
        channel names.

        Args:
            chans:  A tuple containing string channel names (case insensetive)
                    or integer channel indices or any combination of the two.
                    If None (default) then all channel names are assumed.

        Returns:
            A list of integer indices into the eeg data for the channel names
            given in the chans argument.  The list will contain None entries
            where a non-existent channel name or index is given.
        """
        if chans is None:
            return range(self.nChan)

        chanIndices = []
        lowerChanNames = [c.lower() for c in self.chanNames]
        for c in chans:
            if isinstance(c, str):
                if c.lower() in lowerChanNames:
                    chanIndices.append(lowerChanNames.index(c.lower()))
                else:
                    chanIndices.append(None)
            else:
                if c < self.nChan and c >= 0:
                    chanIndices.append(c)
                else:
                    chanIndices.append(None)

        return chanIndices

    def getDeviceName(self):
        """Return the name of the device used to record the eeg data.
        """
        return self.deviceName

    def setDeviceName(self, deviceName):
        """Set the name of the device used to record the eeg data.
        """
        if deviceName is None:
            deviceName = ''
        self.deviceName = str(deviceName)
        return self

    def getNObs(self):
        """Get the number of observations in the eeg data.
        """
        return self.nObs

    def getNSec(self):
        """Get the number of seconds in the eeg data.
        """
        return self.nSec
