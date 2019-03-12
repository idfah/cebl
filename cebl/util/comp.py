"""Handle file compression easily.
"""
import bz2
import gzip
import lzma


compressedExtensions = ("xz", "bz2", "gz")

def openCompressedFile(fileName, mode="rb", **kwargs):
    """Open a compressed file using an algorithm derived
    from its file extension.

    Args:
        fileName:   The name of the file to open.

        mode:       The mode to use when opening the file.
                    See the documentation for the standard
                    open function.

        **kwargs:   Additional arguments to pass to the
                    library used for opening.  Generally,
                    these arguments are the same as in the
                    standard open function.

    Returns:
        A handle to the decompressed file stream.

    Notes:
        The following compression methods are suppored:
            xz, bz2, gz

        No decompression will be used, and the open function
        will be used if the file does not end in one of
        these file extensions.
    """
    fileNameLower = fileName.lower()

    if fileNameLower.endswith(".xz"):
        return lzma.open(fileName, mode, **kwargs)
    elif fileNameLower.endswith(".bz2"):
        return bz2.open(fileName, mode, **kwargs)
    elif fileNameLower.endswith(".gz"):
        return gzip.open(fileName, mode, **kwargs)
    else:
        return open(fileName, mode, **kwargs)
