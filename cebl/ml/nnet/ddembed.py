import numpy as np
import numpy.lib.stride_tricks as npst

###def _deltaDeEmbedSum(delta, width):
###    # yuck - XXX idfah
###    nObs = delta.shape[0]
###    nDim = delta.shape[1]
###
###    lags = width-1
###
###    origDim = nDim // width
###
###    pad = np.zeros((lags,nDim))
###    delta = np.vstack((pad,delta,pad))
###
###    d = list()
###    sz = delta.itemsize
###    for i in range(origDim):
###        rowStride = width
###        colStride = rowStride+1
###        d.append(npst.as_strided(delta[:,i::origDim].copy(),
###                shape=(nObs+lags,width), strides=(rowStride*sz,colStride*sz)))
###
###    return np.array(d).sum(axis=2).T

##def _deltaDeEmbedSum(delta, width):
##    # yuck - XXX idfah
##    nSeg = delta.shape[0]
##    nObs = delta.shape[1]
##    nDim = delta.shape[2]
##
##    lags = width-1
##
##    origDim = nDim // width
##
##    pad = np.zeros((nSeg,lags,nDim))
##    delta = np.concatenate((pad,delta,pad), axis=1)
##
##    d = list()
##    sz = delta.itemsize
##
##    segStride = delta.shape[1] * width * sz
##    rowStride = width * sz
##    colStride = (width+1) * sz
##
##    for i in range(origDim):
##        d.append(npst.as_strided(delta[:,:,i::origDim].copy(),
##                shape=(nSeg,nObs+lags,width), strides=(segStride,rowStride,colStride)))
##
##    d = np.array(d).sum(axis=3)
##    d = np.rollaxis(d, 2, 1).T
##
##    return d

def deltaDeEmbedSum(delta, width):
    nSeg = delta.shape[0]
    nObs = delta.shape[1]
    nDim = delta.shape[2]

    lags = width-1

    origDim = nDim // width

    pad = np.zeros((nSeg,lags,nDim), dtype=delta.dtype)
    delta = delta[:,::-1,:]
    delta = np.concatenate((pad,delta,pad), axis=1)

    flags = delta.flags
    assert flags.c_contiguous

    sz = delta.itemsize

    deEmb = npst.as_strided(delta,
        shape=(width, nSeg, nObs+lags, origDim),
        strides=(origDim*(width+1)*sz,
                 delta.shape[1]*delta.shape[2]*sz,
                 width*origDim*sz, sz))[:,:,::-1,:]

    return deEmb.sum(axis=0)#[:,::-1,:]
