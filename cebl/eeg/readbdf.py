"""Work in progress: read bdf data from a file.
"""
import matplotlib.pyplot as plt
import numpy as np
import struct


__all__ = ['readBDF',]


def unpackStrings(binaryData, start, length, count=1):
    totalLength = length * count
    strings = struct.unpack(str(totalLength) + 's',
                binaryData[start:start+totalLength])[0].decode('utf-8')

    if count > 1:
        strings = [strings[i:i+length] for i in range(0,totalLength,length)]
        strings = [s.rstrip().encode('ascii') for s in strings] # ascii will need to go in python3 XXX - idfah

    return strings, start + length*count

def unpackInts(binaryData, start, length, count=1):
    str, nextByte = unpackStrings(binaryData, start, length, count)

    if count == 1:
        return int(str), nextByte
    else:
        return [int(s) for s in str], nextByte

def unpackInt24s(binaryData, start, length, count, nChan, sampRate):
    totalLength = length * count
    bytes = (np.array(bytearray(binaryData[start:start+totalLength]))
                .reshape((-1,3)).astype(np.int))

    negatives = bytes[:,2] > -128
    bytes[negatives,2] -= 256

    ints = bytes[:,0] + bytes[:,1] * 2**8 + bytes[:,2] * 2**16
    ints = np.array(ints).reshape((-1, nChan, sampRate))
    ints = np.swapaxes(ints, 1, 2)
    ints = ints.reshape((-1, nChan))

    return ints, start + totalLength
    
def readBDF(fileName, verbose=False):
    with open(fileName, 'rb') as fileHandle:
        binaryData = fileHandle.read()

    nextByte = 1
    idCode, nextByte = unpackStrings(binaryData, nextByte, 7)

    if idCode != 'BIOSEMI':
        raise RuntimeError('readBDF:  idCode is', idcode, 'which is not BIOSEMI. Cannot read this file.')

    subjectId, nextByte = unpackStrings(binaryData, nextByte, 80)
    if verbose: print('subjectId is', subjectId)    

    recordingId, nextByte = unpackStrings(binaryData, nextByte, 80)
    if verbose: print('recordingId is', recordingId)    

    startDate, nextByte = unpackStrings(binaryData, nextByte, 8)
    if verbose: print('startDate is', startDate)    

    startTime, nextByte = unpackStrings(binaryData, nextByte, 8)
    if verbose: print('startTime is', startTime)    

    nBytesHeader, nextByte = unpackInts(binaryData, nextByte, 8)
    if verbose: print('nBytesHeader is', nBytesHeader)

    versionDataFmt, nextByte = unpackStrings(binaryData, nextByte,44)
    if verbose: print('versionDataFmt is', versionDataFmt,'. Should be BIOSEMI?')

    nDataRecords, nextByte = unpackInts(binaryData, nextByte, 8)
    if verbose: print('nDataRecords is', nDataRecords,'. If -1, not known')

    nSecondsPerDataRecord, nextByte = unpackInts(binaryData, nextByte, 8)
    if verbose: print('nSecondsPerDataRecord is', nSecondsPerDataRecord)

    nChan, nextByte = unpackInts(binaryData, nextByte, 4)
    if verbose: print('nChan is', nChan)

    chanNames, nextByte = unpackStrings(binaryData, nextByte, 16, nChan)
    if verbose: print('chanNames are', chanNames)

    chanTypes, nextByte = unpackStrings(binaryData, nextByte, 80, nChan)
    if verbose: print('chanTypes are', chanTypes)

    chanUnits, nextByte = unpackStrings(binaryData, nextByte, 8, nChan)
    if verbose: print('chanUnits are', chanUnits)

    chanMinimums, nextByte = unpackInts(binaryData, nextByte, 8, nChan)
    if verbose: print('chanMinimums are', chanMinimums)

    chanMaximums, nextByte = unpackInts(binaryData, nextByte, 8, nChan)
    if verbose: print('chanMaximums are', chanMaximums)

    chanDigitalMinimums, nextByte = unpackInts(binaryData, nextByte, 8, nChan)
    if verbose: print('chanDigitalMinimums are', chanDigitalMinimums)

    chanDigitalMaximums, nextByte = unpackInts(binaryData, nextByte, 8, nChan)
    if verbose: print('chanDigitalMaximums are', chanDigitalMaximums)

    chanMinimums = np.array(chanMinimums, dtype=np.float64)
    chanMaximums = np.array(chanMaximums, dtype=np.float64)
    chanDigitalMinimums = np.array(chanDigitalMinimums, dtype=np.float64)
    chanDigitalMaximums = np.array(chanDigitalMaximums, dtype=np.float64)
    chanGains = (chanMaximums - chanMinimums) / (chanDigitalMaximums - chanDigitalMinimums)
    if verbose: print('chanGains is', chanGains)

    chanPrefilter, nextByte = unpackStrings(binaryData, nextByte, 80, nChan)
    if verbose: print('chanPrefilter are', chanPrefilter)

    sampRates, nextByte = unpackInts(binaryData, nextByte, 8, nChan)
    sampRates = [sr/nSecondsPerDataRecord for sr in sampRates]
    if verbose: print('sampRates are', sampRates)

    if not np.all(np.isclose(sampRates, sampRates[0])):
        raise RuntimeError('readBDF: Chan sample rates are not all the same. They are', sampRates)
    sampRate = int(sampRates[0])

    nextByte = (nChan + 1) * 256  # first byte past header
    data, nextByte = unpackInt24s(binaryData, nextByte, 3,
                                  nDataRecords * nSecondsPerDataRecord * nChan * sampRate,
                                  nChan, sampRate)

    markers = data[:,-1].astype(np.int16)
    data = data[:,:-1] * chanGains[:-1]
    chanNames = chanNames[:-1]

    return {'data': data,
            'markers': markers,
            'idCode': idCode,
            'subjectId': subjectId,
            'recordingId': recordingId,
            'startDate': startDate,
            'startTime': startTime,
            'nBytesHeader': nBytesHeader,
            'versionDataFmt': versionDataFmt,
            'nDataRecords': nDataRecords,
            'nSecondsPerDataRecord': nSecondsPerDataRecord,
            'nChan': nChan,
            'chanNames': chanNames,
            'chanTypes': chanTypes,
            'chanUnits': chanUnits,
            'chanMinimums': chanMinimums,
            'chanMaximums': chanMaximums,
            'chanDigitalMinimums': chanDigitalMinimums,
            'chanDigitalMaximums': chanDigitalMaximums,
            'chanGains': chanGains,
            'chanPrefilter': chanPrefilter,
            'sampRate': sampRate}

if __name__ == '__main__':
    #fileName = 'Newtest17-256.bdf'
    #contents = readBDF(fileName)
    #print('From', fileName, 'read',contents['nDataRecords']*contents['nSecondsPerDataRecord'],'seconds of',contents['nChan'],'chans of data at',contents['sampRate'],'samples per second. EEG matrix is',contents['data'].shape)
    #n = 500
    #eeg = contents['data']
    #plt.figure(1)
    #plt.clf()
    #eeg = eeg - eeg.mean(0)
    #plt.plot(eeg[:n,:])

    fileName = 's11-letter-b.bdf'
    contents = readBDF(fileName, verbose=True)
    print('From', fileName, 'read', contents['nDataRecords']*contents['nSecondsPerDataRecord'],
          'seconds of', contents['nChan'], 'chans of data at', contents['sampRate'],
          'samples per second. EEG matrix is',contents['data'].shape)
    n = 25000
    eeg = contents['data']
    plt.figure(2)
    plt.clf()
    eeg = eeg - eeg.mean(0)
    plt.plot(eeg[:n,:])

    plt.show()
