"""Base classes and routines common to all sources.
"""
##from importlib import import_module
import multiprocessing as mp
import numpy as np
import time
import wx

from cebl import eeg
from cebl import util

from cebl.rt import widgets
from cebl.rt import filters

from sourcepanel import SourceConfigPanel


cdef class Source(object):
    """Base class for all data sources.  Responsible for starting
    and stopping the source and polling for new data.

    Note:
        This source object spawns a sub-process to perform data collection
        once it is started.  Shared memory is used for the data buffer and
        the marker.  Most other parameters cannot be configured on a running
        source.
    """
    cdef double buffSecs, rollBuffSecs
    cdef long buffObs, rollBuffObs

    cdef double esrTime
    cdef long esrObs

    def __init__(self, mgr, sampRate, chans,
                 buffSecs=1200.0, rollBuffSecs=None,
                 name=None, configPanelClass=SourceConfigPanel):
        """Construct a new data source.

        Args:
            mgr:                Manager.

            sampRate:           Sampling rate in observations per second.

            chans:              List of strings containing the initial names
                                of the EEG channels.  A value of None
                                indicates the channel should be deactivated.

            buffSecs:           Maximum seconds of data that can be retrieved
                                at any given time.

            rollBuffSecs:       Floating point number of seconds of data to
                                be kept in the buffer.  When the buffer fills,
                                a copy will be performed to roll the data
                                back while preserving buffSecs of data.  A
                                larger value of rollBuffSecs will result in
                                larger copies but they will be less frequent.
                                The default value of None will make
                                rollBuffSecs 3x buffSecs.

            name:               String name describing this source.  If None
                                (default) then the __name__ of the current
                                class will be used.

            configPanelClass:   Class used for generating a wx panel used
                                for configuring this source.  This class
                                must extend SourceConfigPanel.  The
                                SourceConfigPanel (default) can be used
                                alone if the source has no special
                                configuration needs.
        """
        # manager
        self.mgr = mgr

        # process for data polling loop
        self.dataProcess = None

        # array to hold data, None when not initialized
        self.data = None

        # source sampling rate
        self.sampRate = sampRate

        # channel configuration
        self.setChans(chans)

        # init buffer parameters
        self.buffSecs = buffSecs
        if rollBuffSecs is None:
            rollBuffSecs = 3.0*self.buffSecs
        self.rollBuffSecs = rollBuffSecs
        self.initBuffParams()

        # lock held on startup until the first poll completes
        self.startLock = mp.Lock()

        # lock held when buffer rolls
        self.dataLock = util.ReadWriteLock()

        # flag to notify dataProcess to stop
        self.stopFlag = mp.Event()

        # current index to the tail of the buffer
        self.dataIndex = mp.Value('i', self.buffObs)

        # number of times buffer has rolled
        self.rollCount = mp.Value('i', 0)

        # total observations collected since source start
        self.totalObs = mp.Value('i', 0)

        # accounting for effective sampling rate
        self.esrTime = 1.0
        self.esrObs = 0

        # maximum number of data queues that can be used
        self.maxNumQs = 16

        # initialize shared queues
        self.initQs()

        # current value of marker
        self.marker = mp.Value('d', 0.0)

        # source string name
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        # class for generating configuration panels
        self.configPanelClass = configPanelClass

        # list of generated configuration panels
        self.configPanels = []

        # filter chain
        self.filterChain = filters.FilterChain(self)

    def initBuffParams(self):
        """Initialize buffer parameters.
        """
        self.initRollBuffObs()
        self.initBuffObs()

    def initBuffObs(self):
        """Initialize the maximum number of observations that can be
        garunteed to be retrievable from the buffer at any given time.
        """
        if self.buffSecs >= self.rollBuffSecs:
            raise Exception('buffSecs must be less than rollBuffSecs.')
        self.buffObs = int(self.buffSecs * self.sampRate)

    def initRollBuffObs(self):
        """Initialize the total number of observations held in the buffer.
        """
        if self.rollBuffSecs <= self.buffSecs:
            raise Exception('rollBuffSecs must be greater than buffSecs.')
        self.rollBuffObs = int(self.rollBuffSecs * self.sampRate)

    def initBuffer(self):
        """Initialize a shared-memory buffer.  This is called each time the 
        """
        if self.isRunning():
            raise Exception('Cannot initialize buffer for running source.')

        if self.data is not None:
            raise Exception('Buffer already initialized.')

        self.initBuffParams()

        # shared array to hold data buffer
        ##self.dataArray = mp.Array('f', (self.nChan+1) * self.rollBuffObs)
        self.dataArray = mp.Array('d', (self.nChan+1) * self.rollBuffObs)

        # numpy view of dataArray
        ##self.data = (np.frombuffer(self.dataArray.get_obj(), dtype=np.float32)
        ##    .reshape((-1,(self.nChan+1))))
        self.data = (np.frombuffer(self.dataArray.get_obj())
            .reshape((-1,(self.nChan+1))))

        # initialize first buffObs to zero
        # not needed, mp.Array zeros - XXX idfah
        #self.data[:self.buffObs,:] = 0.0

        self.dataIndex.value = self.buffObs
        self.rollCount.value = 0
        self.totalObs.value = 0

    def delBuffer(self):
        """Delete the shared memory buffer.  Since the buffer is often quite
        large and since we may have multiple ready sources, it is a good idea
        to only allocate it when a source is running.
        """
        if self.isRunning():
            raise Exception('Cannot delete buffer for running source.')

        if self.data is None:
            raise Exception('Buffer already deleted.')

        self.dataArray = None
        self.data = None

    def resetBuffer(self):
        if self.data is None:
            raise Exception('Cannot reset uninitialized buffer.')

        self.delBuffer()
        self.initBuffer()

    def initQs(self):
        """Initialize shared queues for retrieving the data
        stream as poll is called.

        Note:
            self.maxNumQs queues are created when the source starts.
            Each of these queues may be retrieved using getDataQ but
            more queues cannot be created while the source is running.
        """

        # shared data queues
        self.dataQs = [mp.Queue() for q in xrange(self.maxNumQs)]

        # number of shared data queues currently used
        self.nActiveQs = mp.Value('i', 0)

    def delQs(self):
        """Delete the shared queues when not in use.
        """
        self.dataQs = None
        self.nActiveQs = None

    ## Configuration
    #####################################

    def getSampRate(self):
        """Get the current sampling rate

        Returns:
            The integer sampling rate in observations per second.
        """
        return self.sampRate

    def setSampRate(self, sampRate):
        """Set the sampling rate.

        Args:
            sampRate:   The integer sampling rate in observations per second.
        """
        if self.isRunning():
            raise Exception('Cannot change sample rate while source is running.')

        sampRateOrig = self.sampRate
        try:
            self.sampRate = sampRate

        except Exception as e:
            self.sampRate = sampRateOrig
            raise

        if self.data is not None:
            self.resetBuffer()

    def getNChan(self):
        """Get the number of channels in the current source.
        """
        return self.nChan

    def getChans(self):
        """Get current channel configuration.

        Returns:
            List of strings describing each channel.  A value of
            None indicates that the channel is not active.
        """
        return self.chans

    def setChans(self, chans):
        """Set the channel configuration.

        Args:
            chans:  A list of strings describing each channel.
                    A value of None indicates that the channel should be
                    deactivated.
        """
        if self.isRunning():
            raise Exception('Cannot change channel configuration while source is running.')

        self.nChan = len(chans)
        self.chans = chans

        # indices of active channels
        self.activeChanIndex = [i for i,j in enumerate(self.chans) if j != None]

        # names of channels, ommiting None values
        self.chanNames = [j for j in self.chans if j != None]

        # number of active channels
        self.nActiveChan = len(self.chanNames)

        if self.data is not None:
            self.resetBuffer()

    def getChanNames(self):
        """Get active channel names.

        Returns:
            A list of strings describing each active channel.
        """
        return self.chanNames

    def getNSecs(self):
        """Get the maximum number of seconds of data that can be retrieved
        from the buffer.

        Returns:
            Floating point seconds of data that can be retrieved.
        """
        return self.buffSecs

    def setNSecs(self, buffSecs):
        """Set the maxiumum number of seconds of data that can be retrieved
        from the buffer.

        Args:
            buffSecs:    Floating point seconds of data that can be retrieved.
        """
        if self.isRunning():
            raise Exception('Cannot change buffSecs while source is running.')

        buffSecsOrig = self.buffSecs
        try:
            self.buffSecs = buffSecs

        except Exception as e:
            self.buffSecs = buffSecsOrig
            raise

        if self.data is not None:
            self.resetBuffer()

    def getBuffSecs(self):
        """Get the maximum number of seconds of data to be stored in
        the buffer before performing a copy to roll the buffer back.

        Returns:
            Floating point number of seconds to store.
        """
        return self.rollBuffSecs

    def setBuffSecs(self, rollBuffSecs=None):
        """Set the maximum number of seconds of data to be stored in
        the buffer before performing a copy to roll the buffer back.

        Args:
            rollBuffSecs:   Floating point number of seconds of data to be kept
                        in the buffer.  When the buffer fills, a copy will
                        be performed to roll the data back while preserving
                        buffSecs of data.  A larger value of rollBuffSecs will
                        result in larger copies but they will be less frequent.
                        The default value of None will make rollBuffSecs 5x
                        buffSecs.
        """
        if self.isRunning():
            raise Exception('Cannot change rollBuffSecs while source is running.')

        rollBuffSecsOrig = self.buffSecs
        try:
            self.rollBuffSecs = self.buffSecs

        except Exception as e:
            self.rollBuffSecs = rollBuffSecsOrig
            raise

        if self.data is not None:
            self.resetBuffer()

    def getBufferStats(self):
        """Get the number of times that the buffer has rolled
        and the fraction that it is full.

        Returns:
            A tuple containing the number of times the buffer
            has rolled followed by the fraction [0,1] that the
            buffer is currently full.  (0, 0.0) is returned for
            stopped sources.
        """
        if (not self.isRunning()) or (self.data is None):
            return (0, 0.0)

        buffFill = ((self.dataIndex.value - self.buffObs) /
                     float(self.rollBuffObs - self.buffObs))

        return (self.rollCount.value, buffFill)

    def getEffectiveSampRate(self):
        if not self.isRunning():
            return 0.0

        curTime = time.time()
        curObs = self.totalObs.value

        deltaObs = curObs - self.esrObs
        deltaTime = curTime - self.esrTime

        self.esrTime = curTime
        self.esrObs = curObs

        return deltaObs / deltaTime

    def getESR(self):
        return self.getEffectiveSampRate()

    def getName(self):
        """Get the current source name.

        Returns:
            The string name of describing this source.
        """
        return self.name

    def genConfigPanel(self, parent, *args, **kwargs):
        """Generate an instance of the configPanelClass, given as an
        argument to the constructor, that can be used to configure
        this source.
        """
        #className = type(self).__name__ + 'ConfigPanel'
        #moduleName = type(self).__module__
        #module = import_module(moduleName)
        #configPanelClass = getattr(module, className)
        #return configPanelClass(parent=parent, src=self, *args, **kwargs)
        configPanel = self.configPanelClass(parent=parent, src=self, *args, **kwargs)
        self.configPanels.append(configPanel)
        return configPanel

    def getFilterChain(self):
        return self.filterChain

    def query(self):
        """Query the source for additional information.

        Returns:
            String containing a description of the source.
            The default is to use repr(self) but subclassed
            sources may decide to include more or less
            information.
        """
        return repr(self)

    ## State management
    #####################################

    def isRunning(self):
        """Determine if the device is currently running.

        Returns:
            True if the device is currently running.  False otherwise.
        """
        # dataProcess gets set after child process joins
        if self.dataProcess is None:
            return False
        else:
            return True

    def start(self):
        """Start data collection.  This spawns a child process that
        executes the run method.

        Note:
            This method calls the beforeStart method before it is started and
            afterStart after it is started in the parent process.  The
            beforeRun and afterRun methods are called in the child process.
            Use these hooks to add source startup and teardown in sub-classes.
        """
        # should marker value be set to zero when source starts? XXX - idfah

        if self.isRunning():
            raise Exception('Cannot start source %s because it is already running.' % self.name)
        else:
            wx.LogMessage('Starting source %s.' % self.name)

        # initialize shared variables
        #self.initBuffer() # done in manager XXX - idfah

        for configPanel in self.configPanels:
            configPanel.beforeStart()

        self.beforeStart()

        # clear the stop flag
        self.stopFlag.clear()

        # get the start lock
        self.startLock.acquire()

        self.esrTime = time.time()
        self.esrObs = 0

        # fork child process to run polling loop
        self.dataProcess = mp.Process(target=self.run)
        self.dataProcess.start()

        # wait for first poll to release lock
        self.startLock.acquire()
        self.startLock.release()

        self.afterStart()

        for configPanel in self.configPanels:
            configPanel.afterStart()

    def beforeStart(self):
        """Called by start in the parent process before the page is started.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def afterStart(self):
        """Called by start in the parent process after the page is started.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def stop(self):
        """Stop data collection and wait on the child process.

        Note:
            This method calls the beforeStop method before it is stopped and
            afterStop after it is stopped.  Use these hooks to add to the
            stop procedure in sub-classes.
        """
        if not self.isRunning():
            raise Exception('Cannot stop source %s because it is not running.' % self.name)
        else:
            wx.LogMessage('Stopping source %s.' % self.name)

        for configPanel in self.configPanels:
            configPanel.beforeStop()

        self.beforeStop()

        # stop the polling loop and wait for it to join
        self.stopFlag.set()
        self.dataProcess.join()
        self.dataProcess = None

        self.afterStop()

        for configPanel in self.configPanels:
            configPanel.afterStop()

        # self.delBuffer() # done in manager XXX - idfah

    def beforeStop(self):
        """called by stop in the parent process before the page is stopped.
        nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def afterStop(self):
        """Called by stop in the parent process after the page is stopped.
        Nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    ## Data management
    #####################################

    def clearData(self):
        """Reset and zero the data buffer.
        """
        self.data[:,:] = 0.0
        self.dataIndex.value = self.buffObs

    def getDataObs(self, long n, bint copy=True):
        """Retrieve a number of data observations from the tail of the buffer.

        Args:
            n:  Number of observations to return.
        """
        cdef long dataIndex

        if n > self.buffObs:
            raise Exception(('n %d exceeds observations that can be retrieved' % n) + \
                ('from buffer with buffObs %d.' % self.buffObs))

        with self.dataLock.getReadLock():
            if self.data is None:
                raise Exception('Failed to get data: Buffer not initialized.')

            dataIndex = self.dataIndex.value
            curData = self.data[(dataIndex-n):dataIndex,self.activeChanIndex]
            if copy:
                curData = curData.copy()

        # return markers? be sure to copy if so.  XXX - idfah
        return curData

    def getDataSecs(self, double secs, bint copy=True):
        """Retrieve a number of seconds of data from the tail of the buffer.

        Args:
            secs:   Number of seconds to retrieve.

        Returns:
            Numpy array of floats with shape (secs*sampRate,nChan) containing
            the last secs seconds of the buffer.
        """
        return self.getDataObs(int(float(secs)*self.sampRate), copy=copy)

    def getEEGSecs(self, double secs, bint filter=True, bint copy=True):
        """Get an eeg.EEG instance containing a given number of seconds of
        data from the current buffer.

        Args:
            secs:   Number of seconds of data to retrieve from the buffer.

        Returns:
            An eeg.EEG instance containing secs seconds of data.
            The end of the data is at the current moment in time.
        """
        cdef long dataIndex, n = int(float(secs)*self.sampRate)

        if n > self.buffObs:
            raise Exception(('n %d exceeds observations that can be retrieved' % n) + \
                ('from buffer with buffObs %d.' % self.buffObs))

        with self.dataLock.getReadLock():
            if self.data is None:
                raise Exception('Failed to get eeg data: Buffer not initialized.')

            dataIndex = self.dataIndex.value
            curData = self.data[(dataIndex-n):dataIndex,:]
            markers = curData[:,-1]
            curData = curData[:,self.activeChanIndex]

        cap = eeg.EEG(curData, sampRate=self.sampRate,
                      chanNames=self.chanNames,
                      markers=markers, copy=copy)

        if filter:
            return self.filterChain.apply(cap)
        else:
            return cap

    def getDataQ(self):
        """Get an instance of a multiprocessing.queue that will have the
        current data placed in it each time poll is called.  This can
        be used to get streaming data from the source.

        Note:
            self.maxNumQs queues are created when the source starts.
            Each of these queues may be retrieved using getDataQ but
            more queues cannot be created while the source is running.
        """
        with self.nActiveQs.get_lock():
            i = self.nActiveQs.value
            if i >= self.maxNumQs:
                raise Exception('No more data queues available.')
            q = self.dataQs[i]
            self.nActiveQs.value += 1
        return q

    def addDataToBuffer(self, newData):
        """Add data to the tail of the buffer.

        Args:
            newData:    Numpy array of floats with shape
                        (nSec*sampRate, nChan) containing data to
                        add to the tail of the buffer.
        """
        if not self.isRunning():
            raise Exception('Cannot add to buffer of stopped source.')

        # number of observations to append
        cdef long n = newData.shape[0]

        with self.dataLock.getWriteLock():
            # if new data puts us over buffObs then
            # roll the data back to buffObs
            if self.dataIndex.value+n > self.rollBuffObs:
                # it would probably be faster to just copy the data
                # we need instead of rolling everything? - XXX idfah
                #self.data[:,:] = np.roll(self.data,
                #    -self.dataIndex.value+self.buffObs, axis=0)

                ##with self.rollLock:
                self.data[:self.buffObs,:] = \
                    self.data[(self.dataIndex.value-self.buffObs):self.dataIndex.value,:]
                self.dataIndex.value = self.buffObs

                self.rollCount.value += 1

            # append the new data to the buffer
            self.data[self.dataIndex.value:(self.dataIndex.value+n)] = \
                newData

            # adjust the tail index accordingly
            self.dataIndex.value += n

    def addDataToQueues(self, newData):
        """Add data to all active queues.

        Args:
            newData:    Numpy array of floats with shape
                        (nSec*sampRate, nChan) containing data to
                        add to the tail of the buffer.
        """
        if not self.isRunning():
            raise Exception('Cannot add to data queue of stopped source.')

        for i in xrange(self.nActiveQs.value):
            self.dataQs[i-1].put(newData)

    def run(self):
        """Loop polling for new data.  Runs in the child process spawned
        by the start method.
        """
        cdef bint firstPoll = True
        cdef long n
        #cdef double curTime, curESR

        self.beforeRun()

        # poll until stopFlag is set
        while not self.stopFlag.is_set():
            # get new data
            newData = self.pollData()
            n = newData.shape[0]

            # grab active channels
            newData = util.colmat(newData)

            # add marker channel
            newData = util.bias(newData, self.marker.value)

            # add data to shared buffer
            self.addDataToBuffer(newData)

            # add data to any active queues
            # extract active channels and marker here
            self.addDataToQueues(newData[:,self.activeChanIndex+[-1,]])

            # figure esr using exponentially weighted moving average
            #curTime = time.time()
            #curESR = n / (curTime - self.esrTime)
            #self.esr.value = 0.99*self.esr.value + 0.01*curESR
            #self.esrTime = curTime

            #curTime = time.time()
            #deltaTime = curTime - self.esrTime
            #curESR = n / deltaTime
            #alpha = 1.0 - np.exp(-(deltaTime)/100.0)
            #self.esr.value = alpha*self.esr.value + (1.0-alpha)*curESR
            #self.esrTime = curTime

            self.totalObs.value += n

            # release the start lock after first poll
            if firstPoll:
                self.startLock.release()
                firstPoll = False

        self.afterRun()

    def beforeRun(self):
        """called by run in the child process before the page is started.
        nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def afterRun(self):
        """called by run in the child process after the page is stopped.
        nothing is actually done in this method, it is here as a hook for
        adding functionality in sub-classes.
        """
        pass

    def pollData(self):
        """Poll for new data.  This method should block until new data
        is available.  Runs in the child process.  MUST be overriden
        by all subclasses implementing a source.

        Returns:
            A numpy array of floats with shape (pollSize,nChan)
            containing new data.
        """
        raise NotImplementedError('pollData is not implemented.')

    ## Marker management
    #####################################

    def setMarker(self, double marker=0.0):
        """Set the value of the marker.

        Args:
            marker:     Floating point marker value.
        """
        #with self.dataLock.getWriteLock():
        self.marker.value = marker

    def incrementMarker(self, double amount=1.0):
        """Increment the marker, i.e., add a value to it.

        Args:
            amount:     Floating point value to add to the current
                        marker value.  Default is 1.0.
        """
        # should be OK as long as we're not setting marker in rapid succession XXX - idfah
        #with self.dataLock.getWriteLock():
        self.marker.value += amount

    def setMarkerChr(self, character, int sign=1):
        self.setMarker(float(np.sign(sign)*ord(character)))

    def getMarkerObs(self, long n):
        """Retrieve a number of observations from the tail of the marker.

        Args:
            n:      Number of observations to retrieve.

        Returns:
            Numpy array of floats with shape (n,) containing
            the last n observations of the marker.
        """
        cdef long dataIndex

        if n > self.buffObs:
            raise Exception(('n %d exceeds markers that can be retrieved' % n) + \
                ('for buffer with buffObs %d.' % self.buffObs))

        with self.dataLock.getReadLock():
            if self.data is None:
                raise Exception('Failed to get marker data: Buffer not initialized.')

            dataIndex = self.dataIndex.value
            return self.data[(dataIndex-n):dataIndex,:-1].copy()

    def getMarkerSecs(self, double secs):
        """Retrieve a number of seconds of data from the tail of the marker.

        Args:
            secs:   Number of seconds to retrieve.

        Returns:
            Numpy array of floats with shape (secs*sampRate,) containing
            the last secs seconds of the marker.
        """
        return self.getMarkerObs(int(float(secs)*self.sampRate))

    ## Magic
    #####################################

    def __repr__(self):
        """Return a string fully describing this source.
        """
        # string containing important variables
        r = self.name + '\n' + \
            '====================\n'        + \
            'running: '                     + str(self.isRunning())     + '\n' + \
            'Sample Rate: '                 + str(self.sampRate)        + '\n' + \
            'Total Channels: '              + str(self.nChan)           + '\n' + \
            'Active Channels: '             + str(self.nActiveChan)     + '\n' + \
            'Channel Names: '               + str(self.chanNames)       + '\n' + \
            'Buffered Seconds: '            + str(self.buffSecs)        + '\n' + \
            'Buffered Observations: '       + str(self.buffObs)         + '\n' + \
            'Roll Buffered Seconds: '       + str(self.rollBuffSecs)    + '\n' + \
            'Roll Buffered Observations: '  + str(self.rollBuffObs)     + '\n'

        return r

    def __str__(self):
        """Return the name of this source.
        """
        return self.name
