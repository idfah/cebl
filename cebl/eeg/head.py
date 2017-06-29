import matplotlib.pyplot as plt
import matplotlib.patches as pltPatches
import matplotlib.colors as pltColors
import numpy as np
from scipy import interpolate as spinterp

from cebl import util

from .chanlocs import chanLocs3d


def sphereDist(x1, x2):
    """Great-circle distance between two 3d (cartesian) points.
    (nPts, nDim)

    nDim, nPts, 1
    nDim, 1, nPts

    (nPts, 1, nDim)
    (1, nPts, nDim)
    """
    # make sure coordinates have unit magnitude
    # this needs to be asserted as a precondition because it isn't true for euclidDist in 2d coords XXX - idfah
    #x1 /= np.sqrt(np.sum(x1**2, axis=1))[:,None]
    #x2 /= np.sqrt(np.sum(x2**2, axis=1))[:,None]

    x1 = x1[:,None,:]
    x2 = x2[None,:,:]

    # cross product
    cross = np.cross(x2,x1, axisa=2, axisb=2)
    cross = np.sqrt(np.sum(cross**2, axis=2))

    # dot product
    dot = (x1 * x2).sum(axis=2)

    return np.arctan2(cross, dot)

def euclidDist(x1, x2):
    x1 = x1[:,None,:]
    x2 = x2[None,:,:]

    return np.sqrt( ((x1 - x2)**2).sum(axis=2) )

def cacheDist(x1, x2, coord):
    coord = coord.lower()
    if coord in ('2d', '3d'):
        distFunc = euclidDist
    elif coord == 'sphere':
        distFunc = sphereDist
    else:
        raise Exception('Invalid coord %s.' % str(coord))

    hx1 = util.hashArray(x1)
    hx2 = util.hashArray(x2)
    hx = coord + hx1 + hx2

    if not hasattr(cacheDist, 'cache'):
        cacheDist.cache = util.Cache(10)

    if hx in cacheDist.cache:
        print 'cache hit'
        return cacheDist.cache[hx]

    result = distFunc(x1, x2)
    cacheDist.cache[hx] = result
    return result


def plotHeadOutline(ax=None, radius=1.2):
    # if new axis not given, create one
    if ax is None:
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(1,1,1, aspect='equal')

    extent = (-0.16-radius,0.16+radius,-0.01-radius,0.20+radius)

    ax.set_aspect('equal')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    leftEar = pltPatches.Ellipse((-0.075-radius,0.0), width=0.15, height=0.4, angle=0.0, edgecolor='dimgrey', facecolor='white', linewidth=3, zorder=10, fill=False)
    ax.add_patch(leftEar)

    rightEar = pltPatches.Ellipse((0.075+radius,0.0), width=0.15, height=0.4, angle=0.0, edgecolor='dimgrey', facecolor='white', linewidth=3, zorder=10, fill=False)
    ax.add_patch(rightEar)

    noseLength = 0.18
    noseWidth = 0.12
    noseIntersect = 2.0*radius-np.sqrt(noseWidth**2+radius**2)

    noseLeft, = ax.plot((0.0,-noseWidth), (radius+noseLength,noseIntersect), color='dimgrey', linewidth=3, zorder=10)
    noseRight, = ax.plot((0.0,noseWidth), (radius+noseLength,noseIntersect), color='dimgrey', linewidth=3, zorder=10)

    noseLeft.set_solid_capstyle('round')
    noseRight.set_solid_capstyle('round')

    head = pltPatches.Circle((0.0,0.0), radius, edgecolor='dimgrey', facecolor='white', linewidth=3, zorder=10, fill=False)
    ax.add_patch(head)

    ax.set_xticks([])
    ax.set_yticks([])

    return {'ax': ax, 'leftEar': leftEar, 'rightEar': rightEar,
            'noseLeft': noseLeft, 'noseRight': noseRight,
            'head': head, 'extent': extent}

def plotHead(chanNames=('F3','F4','C3','C4','P3','P4','O1','O2'),
             radius=0.01, fillColor='black', lineColor='black',
             drawLabels=True, clip=True, fontSize=12, ax=None, **kwargs):

    # plot head outline and save result dict
    result = plotHeadOutline(ax)
    ax = result['ax']

    # get chanNames that we have a location for
    chanNames = [chanName for chanName in chanNames if chanName.lower() in chanLocs3d.keys()]
    chanNamesLower = [chanName.lower() for chanName in chanNames] # lower case

    # if no valid chanNames then just draw the outline
    if len(chanNames) == 0:
        return result

    # get 3d cartesian coordinates for each channel
    xyz = np.asarray([chanLocs3d[chanName] for chanName in chanNamesLower])

    # make sure coordinates have unit magnitude
    xyz = xyz / np.sqrt(np.sum(xyz**2, axis=1))[:,None]

    # rotate by pi/2 around z axis
    cos90 = np.cos(np.pi/2.0)
    rot = np.asarray(((cos90,-1.0,0.0),(1.0,cos90,0.0),(0.0,0.0,1.0))).T
    xyz = xyz.dot(rot)

    # stereographic projection
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    xy = np.vstack((x/(1.0+z), y/(1.0+z))).T

    # list of circles and chan labels to store in result
    circles = []
    chanLabels = []

    # for each channel
    for chanName, coord in zip(chanNames, xy):
        x, y = coord

        c = pltPatches.Circle((x,y), radius, edgecolor=lineColor,
                              facecolor=fillColor, linewidth=1, fill=True, zorder=100)

        ax.add_patch(c)
        circles.append(c)

        # draw channel labels
        if drawLabels:
            if fontSize is not None:
                txt = ax.text(x,y+0.081, chanName, size=fontSize,
                        horizontalalignment='center', verticalalignment='center', zorder=100)
                chanLabels.append(txt)

    # save labels and circles in result
    result['chanLabels'] = chanLabels
    result['circles'] = circles

    return result

def plotHeadLines(magnitudes, chanNames=('F3','F4','C3','C4','P3','P4','O1','O2'),
                    method='lines', width=0.1, height=0.075, fillColor='black',
                    lineColor='black', mn=None, mx=None, drawLabels=True, clip=True,
                    fontSize=11, xlabel=None, ylabel=None, ax=None, **kwargs):

    # plot head outline and save result dict
    result = plotHeadOutline(ax)
    ax = result['ax']

    # get chanNames that we have a location for
    chanNames = [chanName for chanName in chanNames if chanName.lower() in chanLocs3d.keys()]
    chanNamesLower = [chanName.lower() for chanName in chanNames] # lower case

    mn = np.min(magnitudes) if mn is None else mn
    mx = np.max(magnitudes) if mx is None else mx
    result['mn'] = mn
    result['mx'] = mx

    # if no valid chanNames then just draw the outline
    if len(chanNames) == 0:
        return result

    # get 3d cartesian coordinates for each channel
    xyz = np.asarray([chanLocs3d[chanName] for chanName in chanNamesLower])

    # make sure coordinates have unit magnitude
    xyz = xyz / np.sqrt(np.sum(xyz**2, axis=1))[:,None]

    # rotate by pi/2 around z axis
    cos90 = np.cos(np.pi/2.0)
    rot = np.asarray(((cos90,-1.0,0.0),(1.0,cos90,0.0),(0.0,0.0,1.0))).T
    xyz = xyz.dot(rot)

    # stereographic projection
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    xy = np.vstack((x/(1.0+z), y/(1.0+z))).T

    # list of circles and chan labels to store in result
    circles = []
    chanLabels = []

    # for each channel
    for chanName, coord, mags in zip(chanNames, xy, magnitudes.T):
        x, y = coord

        axc = plt.gcf().transFigure.inverted().transform(ax.transData.transform((x,y)))
        sax = plt.axes([axc[0]-width/2.0, axc[1], width, height])
        sax.set_ylim(mn, mx)

        sax.set_xticks([])
        sax.set_yticks([])
        sax.spines['top'].set_visible(False)
        sax.spines['right'].set_visible(False)

        if xlabel is not None:
            sax.set_xlabel(xlabel)

        if ylabel is not None:
            sax.set_ylabel(ylabel)

        if (xlabel is None) and (ylabel is None):
            sax.axis('off')
        else:
            sax.axis('on')

        if method == 'bars':
            bars = sax.bar(np.arange(len(mags)), mags, width=1, **kwargs)
            result['bars'] = bars
        elif method == 'lines':
            lines = sax.plot(mags, **kwargs)
            result['lines'] = lines
        else:
            raise Exception('Invalid method: ' + str(method))

        if drawLabels:
            sax.set_title(chanName, fontsize=fontSize)

    # save labels and circles in result
    result['chanLabels'] = chanLabels
    result['circles'] = circles

    return result

def plotHeadInterp(magnitudes, chanNames=('Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3',
                    'Cz','C4','T8','T5','P3','Pz','P4','T6','O1','O2'),
                   drawLabels=True, method='none', coord='2d', n=512,
                   scale='linear', mn=None, mx=None, clip=True, fontSize=12,
                   colorbar=True, cmap=plt.cm.jet, ax=None, cache=False, **kwargs):
    """
        kwargs:     Additional arguments passed to scipy.interpolate.Rbf
                    if method is not 'nearest' or 'none'.
                    Note:  If the argument epsilon is not given, it will
                    be set to a default of 0.2.
    """
    method = method.lower()
    coord = coord.lower()

    # plot head outline and save result dict
    result = plotHeadOutline(ax)
    ax = result['ax']

    # get plot extent from head outline
    extent = result['extent']

    # if no magnitudes given, use zeros
    if magnitudes is None:
        magnitudes = [0.0,]*len(chanNames)

    # pull out only magnitudes that we have channel locations for
    magnitudes = [mag for mag,chanName in zip(magnitudes,chanNames) if chanName.lower() in chanLocs3d.keys()]
    magnitudes = np.asarray(magnitudes)

    # if no valid mags then just draw the outline
    if len(magnitudes) == 0:
        return result

    # get min and max magnitudes if not provided
    mn = np.min(magnitudes) if mn is None else mn
    mx = np.max(magnitudes) if mx is None else mx
    result['mn'] = mn
    result['mx'] = mx

    if scale == 'linear':
        norm = pltColors.Normalize(mn,mx)
    elif scale == 'log':
        norm = pltColors.LogNorm(mn,mx)
    else:
        raise Exception('Invalid scale ' + str(scale))

    # get chanNames that we have a location for
    chanNames = [chanName for chanName in chanNames if chanName.lower() in chanLocs3d.keys()]
    chanNamesLower = [chanName.lower() for chanName in chanNames]

    # get 3d cartesian coordinates for each channel
    xyz = np.asarray([chanLocs3d[chanName] for chanName in chanNamesLower])

    # put sensors on "southern" hemisphere
    # we use -z instead XXX - idfah
    ##xyz[:,2] *= -1.0

    # make sure coordinates have unit magnitude
    xyz = xyz / np.sqrt(np.sum(xyz**2, axis=1))[:,None]

    # rotate by pi/2 around z axis
    cos90 = np.cos(np.pi/2.0)
    rot = np.asarray(((cos90,-1.0,0.0),(1.0,cos90,0.0),(0.0,0.0,1.0))).T
    xyz = xyz.dot(rot)

    # stereographic projection
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    #xy = np.vstack((x/(1.0-z), y/(1.0-z))).T
    xy = np.vstack((x/(1.0+z), y/(1.0+z))).T

    # points at which to interpolate
    xi = np.linspace(extent[0], extent[1], n)
    yi = np.linspace(extent[2], extent[3], n)

    if method == 'none':
        im = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        im._A = magnitudes

    elif method == 'nearest':
        if coord != '2d':
            raise Exception('Interpolation method %s only supports coord 2d, cannot use %s.' %
                (str(method), str(coord)))

        # interpolation over 2d grid
        magi = spinterp.griddata(xy, magnitudes, (xi[None,:], yi[:,None]), method=method)

        magi[magi > mx] = mx
        magi[magi < mn] = mn

        im = ax.imshow(magi, extent=extent, interpolation='bicubic',
                       origin='lower', cmap=cmap, zorder=1, norm=norm)

        if clip:
            im.set_clip_path(result['head'])

    elif method in ('multiquadric', 'inverse', 'gaussian',
                    'linear', 'cubic', 'quintic', 'thin_plate'):

        if not 'epsilon' in kwargs.keys():
            kwargs['epsilon'] = 0.2

        def thinPlate(self, r):
            # Hack to prevent thin plate from throwing warning.
            # Should report this as a bug XXX - idfah
            return r**2 * np.log(util.capZero(r))

        if method == 'thin_plate':
            ifunc = thinPlate
        else:
            ifunc = method

        if coord == '2d':
            if cache:
                kwargs['norm'] = (lambda x1, x2:
                    cacheDist(x1.squeeze(2).T, x2.squeeze(1).T, coord=coord))

            rbfModel = spinterp.Rbf(xy[:,0], xy[:,1], magnitudes,
                        function=ifunc, **kwargs)

            xxi, yyi = np.meshgrid(xi,yi)
            xxi = xxi.reshape((-1,))
            yyi = yyi.reshape((-1,))

            magi = rbfModel(xxi, yyi).reshape((n,n))

        elif coord == '3d':
            if cache:
                kwargs['norm'] = (lambda x1, x2:
                    cacheDist(x1.squeeze(2).T, x2.squeeze(1).T, coord=coord))

            rbfModel = spinterp.Rbf(xyz[:,0], xyz[:,1], xyz[:,2], magnitudes,
                        function=ifunc, **kwargs)

            xxi, yyi = np.meshgrid(xi,yi)
            xxi = xxi.reshape((-1,))
            yyi = yyi.reshape((-1,))

            # inverse stereographic projection
            denom = (1.0 + xxi**2 + yyi**2)
            xi3 = 2.0 * xxi / denom
            yi3 = 2.0 * yyi / denom
            #zi3 = (denom - 2.0) / denom
            zi3 = -(denom - 2.0) / denom

            magi = rbfModel(xi3, yi3, zi3).reshape((n,n))

        elif coord == 'sphere':
            if cache:
                kwargs['norm'] = (lambda x1, x2:
                    cacheDist(x1.squeeze(2).T, x2.squeeze(1).T, coord=coord))
            else:
                kwargs['norm'] = (lambda x1, x2:
                    sphereDist(x1.squeeze(2).T, x2.squeeze(1).T))

            rbfModel = spinterp.Rbf(xyz[:,0], xyz[:,1], xyz[:,2], magnitudes,
                        function=ifunc, **kwargs)

            # can all this be vectorized XXX - idfah

            xxi, yyi = np.meshgrid(xi,yi)
            xxi = xxi.reshape((-1,))
            yyi = yyi.reshape((-1,))

            # inverse stereographic projection
            denom = (1.0 + xxi**2 + yyi**2)
            xi3 = 2.0 * xxi / denom
            yi3 = 2.0 * yyi / denom
            #zi3 = (denom - 2.0) / denom
            zi3 = -(denom - 2.0) / denom

            # ensure we have unit length vectors
            # necessary XXX - idfah
            l = np.sqrt((xi3 + yi3 + zi3)**2)
            xi3 /= l
            yi3 /= l
            zi3 /= l

            magi = rbfModel(xi3, yi3, zi3).reshape((n,n))

        else:
            raise Exception('Invalid coord %s.', str(coord))

        magi[magi > mx] = mx
        magi[magi < mn] = mn

        im = ax.imshow(magi, extent=extent, interpolation='none', origin='lower',
                       cmap=cmap, zorder=1, norm=norm)

        if clip:
            im.set_clip_path(result['head'])

    # save image or scalar mappable in result
    result['im'] = im

    # list of circles and chan labels to store in result
    circles = []
    chanLabels = []

    magnitudesCircle = magnitudes.copy()
    eps = np.finfo(magnitudesCircle.dtype).eps
    magnitudesCircle[magnitudesCircle < eps] = eps
    magnitudesCircle = magnitudes * (0.06 / float(np.max(np.abs((mn, mx)))))

    # for each channel
    for chanName, coord, mag, magc in zip(chanNames, xy, magnitudes, magnitudesCircle):
        x, y = coord

        # if interpolation method is none, draw scaled circles
        if method == 'none':
            c = pltPatches.Circle((x,y), magc, edgecolor='black',
                facecolor=im.to_rgba(mag), linewidth=1, fill=True, zorder=100)

        # otherwise, draw small circles
        else:
            c = pltPatches.Circle((x,y), 0.01, edgecolor='black',
                facecolor='black', linewidth=1, fill=True, zorder=100)

        ax.add_patch(c)
        circles.append(c)

        # draw channel labels
        if drawLabels:
            if fontSize is not None:
                txt = ax.text(x,y+0.081, chanName, size=fontSize,
                        horizontalalignment='center', verticalalignment='center', zorder=100)
                chanLabels.append(txt)

    # save labels and circles in result
    result['chanLabels'] = chanLabels
    result['circles'] = circles

    # draw colorbar if requested
    if colorbar:
        cb = plt.colorbar(im)
        cb.set_label(r'Microvolts ($\mu V$)')
        result['colorbar'] = cb

    return result

def demoPlotHeadLines():
    hp = plotHeadLines(np.random.random((9,19)), linewidth=2, mn=-1, mx=1)
    for i in range(3):
        plotHeadLines(np.random.random((9,19)), ax=hp['ax'], linewidth=2, mn=0, mx=1)

def demoPlotHeadInterp():
    chanNames = list(set(chanLocs3d.keys()) - set(('t7','t8','t5','t6')))
    #chanNames = ('Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'P7', 'P8')
    #chanNames = ('F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2')
    #chanNames = ('Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2')
    #chanNames = ('O1', 'O2', 'F2')
    mags = np.random.uniform(-2.0,2.0, size=len(chanNames))

    fig = plt.figure(figsize=(18,12))

    axNone = fig.add_subplot(2,3, 1)
    plotHeadInterp(mags, chanNames=chanNames,
        method='none', coord='2d', cache=True, colorbar=True, ax=axNone)
    axNone.set_title('None')

    axNer2d = fig.add_subplot(2,3, 2)
    plotHeadInterp(mags, chanNames=chanNames,
        method='nearest', coord='2d', cache=True, colorbar=True, ax=axNer2d)
    axNer2d.set_title('Nearest 2d')

    axCub2d = fig.add_subplot(2,3, 3)
    plotHeadInterp(mags, chanNames=chanNames,
        method='cubic', coord='2d', cache=True, colorbar=True, ax=axCub2d)
    axCub2d.set_title('Cubic 2d')

    axMul2d = fig.add_subplot(2,3, 4)
    plotHeadInterp(mags, chanNames=chanNames,
        method='multiquadric', coord='2d', cache=True, colorbar=True, ax=axMul2d)
    axMul2d.set_title('Multiquadric 2d')

    axMul3d = fig.add_subplot(2,3, 5)
    plotHeadInterp(mags, chanNames=chanNames,
        method='multiquadric', coord='3d', cache=True, colorbar=True, ax=axMul3d)
    axMul3d.set_title('Multiquadric 3d')

    axMulSp = fig.add_subplot(2,3, 6)
    plotHeadInterp(mags, chanNames=chanNames,
        method='multiquadric', coord='sphere', cache=True, colorbar=True, ax=axMulSp)
    axMulSp.set_title('Multiquadric Sphere')

    fig.tight_layout()

if __name__ == '__main__':
    demoPlotHeadInterp()
    demoPlotHeadLines()
    plt.show()
