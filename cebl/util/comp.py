import bz2
import gzip
import lzma

compressedExtensions = ('xz', 'bz2', 'gz')

def openCompressedFile(fileName, mode='r', *args, **kwargs):
    fileNameLower = fileName.lower()

    if fileNameLower.endswith('.xz'):
        fh = lzma.LZMAFile(fileName, mode, *args, **kwargs)
        dir(fh) # fix for wierd bug https://bugs.launchpad.net/pyliblzma/+bug/1219296 hopefully fixed in next release XXX - idfah
        return fh
    elif fileNameLower.endswith('.bz2'):
        return bz2.BZ2File(fileName, mode, *args, **kwargs)
    elif fileNameLower.endswith('.gz'):
        return gzip.GzipFile(fileName, mode, *args, **kwargs)
    else:
        return open(fileName, mode, *args, **kwargs)
