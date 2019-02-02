# from open_bci_v3.py in OpenBCI's python-sdk, as is most of the openbci-specific  code here.

import serial
import numpy as np
import struct
import time
import wx
import sys
import glob

from cebl import util
from cebl.rt import widgets

from cebl.rt.sources.source import Source, SourceConfigPanel


SAMPLE_RATE = 250.0  # Hz
START_BYTE = 0xA0  # start of data packet
END_BYTE = 0xC0  # end of data packet
ADS1299_VREF = 4.5  #reference voltage for ADC in ADS1299.  set by its hardware
ADS1299_GAIN = 24.0  #assumed gain setting for ADS1299.  set by its Arduino code
SCALE_uVOLTS_PER_COUNT = ADS1299_VREF/float((pow(2,23)-1))/ADS1299_GAIN*1000000.
SCALE_ACCEL_G_PER_COUNT = 0.002 /(pow(2,4)) #assume set to +/4G, so 2 mG

class OpenBCIConfigPanel(SourceConfigPanel):
    def __init__(self, parent, src, *args, **kwargs):
        SourceConfigPanel.__init__(self, parent=parent, src=src, *args, **kwargs)

        self.initPollSizeSelector()
        self.initLayout()

    def initPollSizeSelector(self):
        pollSizeControlBox = widgets.ControlBox(self, label='Poll Size', orient=wx.HORIZONTAL)
        self.pollSizeSpinCtrl = wx.SpinCtrl(self, style=wx.SP_WRAP,
                value=str(self.src.pollSize), min=1, max=32)
        pollSizeControlBox.Add(self.pollSizeSpinCtrl, flag=wx.ALL | wx.EXPAND, border=10)
        self.Bind(wx.EVT_SPINCTRL, self.setPollSize, self.pollSizeSpinCtrl)

        self.sizer.Add(pollSizeControlBox, proportion=0,
                flag=wx.ALL, border=10)

    def setPollSize(self, event=None):
        self.src.pollSize = self.pollSizeSpinCtrl.GetValue()

class OpenBCI(Source):
    """OpenBCI data source.
    """
    def __init__(self, mgr, sampRate=250,
                 chans=('EEG1','EEG2','EEG3','EEG4','EEG5','EEG6','EEG7','EEG8'),
                 pollSize=1):
        """Construct a new OpenBCI source.
        """
        # initialize source parent class

        # chans = ('FZ', 'CZ', 'PZ', 'OZ', 'P3', 'P4', 'P7', 'P8')
        chans = ('C1','C2', 'C3', 'C4', 'C5', 'C7', 'CZ', 'PZ')


        Source.__init__(self, mgr, sampRate=sampRate, chans=chans,
            name='OpenBCI', configPanelClass=OpenBCIConfigPanel)

        # self.batteryScale = 4.2 - 1.28 / 16
        self.acceloScale = SCALE_ACCEL_G_PER_COUNT
        self.dataScale = SCALE_uVOLTS_PER_COUNT

        self.baudrate = 115200
        self.timeout = 4
        self.connected = False
        self.handshaking = False
        self.description = 'OpenBCI V3 - 8'
        self.device = None

        # observations collected in each poll
        self.pollSize = pollSize
        self.firstPoll = True

    ## Connection
    #####################################

    def find_port(self):   # from openbci code
        # Finds the serial port names
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i+1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        print('ports',ports)
        openbci_port = ''
        for port in ports:
            try:
                s = serial.Serial(port= port, baudrate = self.baudrate, timeout=self.timeout)
                time.sleep(1)
                s.write(b's') # Chuck added. didn't work without this.
                time.sleep(1)
                # flush
                n = s.inWaiting()
                if n > 0:
                    b = s.read(n)
                s.write(b'v')
                openbci_serial = self.openbci_id(s)
                s.close()
                print(openbci_serial)
                if openbci_serial:
                    openbci_port = port;
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port')
        else:
            return openbci_port

    def connect(self):
        wx.LogMessage(self.getName() + ': connecting.')
        try:
            if not self.connected:
                port = '/dev/ttyUSB0'  # self.find_port()  # like '/dev/ttyUSB0'
                self.device = serial.Serial(port = port, baudrate = 115200, timeout = None)
                # print('device is',self.device)
                time.sleep(1)
                self.device.write(b's') # stop
                time.sleep(1)
                self.device.write(b'v')  # init 32-bit board
                time.sleep(1)
                # self.description = self.getConfig()
                # print('desc is',self.description)

                self.device.baudrate = 115200 # has no effect in above statement
                # self.device.open()

                self.stopAcquisition()

                # self.configuration = self.getConfig()

                self.connected = True
        except Exception as e:
            self.connected = False
            raise RuntimeError('Failed to connect to OpenBCI: ' + str(e))

    # def read_incoming_text(self):
    #     if self.device.inWaiting():
    #         lines = ''
    #         line = self.device.read().decode('utf-8')
    #         # Look for end sequence $$$
    #         while '$$$' not in line:
    #             lines += line.replace('$$$','')
    #         return lines
    #     else:
    #         return ''

    def disconnect(self):
        wx.LogMessage(self.getName() + ': disconnecting.')
        if self.connected is False or not self.device.isOpen():
            wx.LogMessage(self.getName() + ': already disconnected.')
            self.connected = False
            self.device = None
            return

        try:
            #time.sleep(1)
            self.device.close()

        except Exception as e:
            raise RuntimeError('Failed to disconnect from OpenBCI: ' + str(e))

        finally:
            self.connected = False
            self.device = None

    def startAcquisition(self):
        # send start command
        #  self.stopAcquisition()   Now done in beforeStart
        print('sending start command b')
        self.device.write(b'b')

    def stopAcquisition(self):
        # send stop command
        self.device.write(b's')
        time.sleep(1)

        try:
            n = self.device.inWaiting()
            print('stopAcquisition has',n,'bytes waiting and will read them')
            if n > 0:
                dumpBuffer = self.device.read(n)
                print( 'dumpBuffer len: ', len(dumpBuffer))

        except Exception as e:
            pass

    ## Configuration
    #####################################

    def getConfig(self):
        self.device.write(b'?')
        print('waiting after ? command')
        time.sleep(3)
        reply = ''
        n = self.device.inWaiting()
        print('there are',n,'bytes waiting')
        if n > 0:
            reply = self.device.read(n)

        # ack, pan_id_bs, pan_id_hs, addr_bs, addr_hs, channel_bs, channel_hs, handshaking, free_channels, huh, checksum = \
        #         struct.unpack('>bHHHHBB?H?H',reply)

        # return pan_id_bs, pan_id_hs, addr_bs, addr_hs, channel_bs, channel_hs, handshaking, free_channels
        print('getConfig returned')
        print(reply)
        self.description = reply
        return reply

    def query(self):
        try:
            self.connect()
            time.sleep(1)
            self.disconnect()
            time.sleep(1)

        except Exception as e:
            raise RuntimeError('Failed to query OpenBCI: ' + str(e))

        return repr(self)

    ## Data management
    #####################################

    def beforeStart(self):
        try:
            print('beforeStart')
            self.connect()
            print('beforeState after connect')
            self.stopAcquisition()
            # # self.startAcquisition()
            # print('inWaiting..',self.device.inWaiting())
            # if self.device.inWaiting() > 0:
            #     junk = self.device.read(self.device.inWaiting())
            #     time.sleep(1)
            # print('after read: inWaiting..',self.device.inWaiting())
            self.startAcquisition()
            print('returned from startAcquisition')

        except Exception as e:
            raise RuntimeError('Failed to start OpenBCI acquisition: ' + str(e))

    def afterStop(self):
        try:
            self.stopAcquisition()

        except:
            raise

        finally:
            self.disconnect()

    def pollData(self):
        ## Incoming Packet Structure:
        ##   Start Byte(1), Sample ID(1), Channel Data(24), Acc Data(6), End Byte(1)
        ##   0xA0,  0-255,     8 3-byte signed ints, 3 2-byte signed ints, 0xC0
        ## total packet size is 1+1+24+6+1 = 33

        ## This code only extracts the 8 EEG channels.
        # print('pollData')
        scanSize = 33  # bytes per scan, fixed by hardware

        NCHANNELS = len(self.chans)

        eeg = np.empty((self.pollSize, NCHANNELS))
        # accelerometers = np.empty((self.pollSize, 3))
        # ids = np.empty((self.pollSize,1))

        eegIndices = np.array([2,5,8,11,14,17,20,23])[:NCHANNELS]
        # eegIndices = np.array([2,5]) #eegIndices([2,5]) #s:NCHANNELS)

        # accFirstIndex = 26

        # print('pollData before read of',scanSize*self.pollSize)
        reply = self.device.read(scanSize * self.pollSize)
        # print('pollData after read',len(reply))

        startByte, sampleId = struct.unpack('BB', reply[:2])

        # print('sampleId',sampleId,'startbyte',startByte,'inwaiting',self.device.inWaiting())

        for polli in range(self.pollSize):
            # big-endian

            eeg[polli,:] = [struct.unpack('>i', (b'\x00' if reply[i] < 0x80 else b'\xff') +
                                          reply[i:i+3])[0] for i in eegIndices+(polli*scanSize)]
            #eeg[polli,:] = np.array([struct.unpack('>i', (b'\x00' if reply[i] < 0x80 else b'\xff') + reply[i:i+3])[0] for i in eegIndices+(polli*self.pollSize)])

            # acci = accFirstIndex + polli * self.pollSize
            # accelerometers[polli,:] = struct.unpack('>hhh', packet[acci:acci+6])
            # ids[polli,:] = packet[1 + (polli * self.pollSize)]

        eeg *= SCALE_uVOLTS_PER_COUNT

        return eeg

    ## Magic
    #####################################

    def __repr__(self):
        r = Source.__repr__(self)
        r  += '\nHardware:\n' + \
              '====================\n' + \
              'Description: '   + str(self.description) + '\n' + \
              '====================\n'

        self.connected = False
        self.description = ''
        return r
