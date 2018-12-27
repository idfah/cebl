import bz2
import gzip
import lzma


compressedExtensions = ('xz', 'bz2', 'gz')

def openCompressedFile(fileName, mode='rb', *args, **kwargs):
    fileNameLower = fileName.lower()

    if fileNameLower.endswith('.xz'):
        return lzma.open(fileName, mode, *args, **kwargs)
    elif fileNameLower.endswith('.bz2'):
        return bz2.open(fileName, mode, *args, **kwargs)
    elif fileNameLower.endswith('.gz'):
        return gzip.open(fileName, mode, *args, **kwargs)
    else:
        return open(fileName, mode, *args, **kwargs)
