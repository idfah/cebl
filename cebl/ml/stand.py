import numpy as np

from cebl import util


class Standardizer:
    def __init__(self, x, method='zmus'):
        """

        Args:
            method: 
                zmus:   Zero mean, unit standard deviation

                range:  Range of [-1,1]
        """
        method = method.lower()
        if method == 'zmus':
            self.initZmus(x)
        elif method == 'range':
            self.initRange(x)
        else:
            raise RuntimeError('Unknown method: %s.' % method)

    def initZmus(self, x):
        x = np.asarray(x)

        self.shift = np.mean(x, axis=0)
        #self.scale = util.capZero(np.std(x, axis=0))
        self.scale = np.std(x, axis=0)

        # best way to handle this? XXX - idfah
        if np.any(np.isclose(self.scale, 0.0)):
            print('Standardizer Warning: Some dimensions are constant, capping zeros.')
            self.scale = util.capZero(self.scale)

    def initRange(self, x):
        x = np.asarray(x)

        mn = np.min(x, axis=0)
        mx = np.max(x, axis=0)
        self.shift = 0.5 * (mx + mn)
        self.scale = 0.5 * (mx - mn)

    def apply(self, x):
        x = np.asarray(x)
        return (x - self.shift) / self.scale

    def unapply(self, x):
        x = np.asarray(x)
        return (x * self.scale) + self.shift

class ClassStandardizer(Standardizer):
    def __init__(self, classData, *args, **kwargs):
        Standardizer.__init__(self, np.vstack(classData), *args, **kwargs)

    def apply(self, classData):
        return [Standardizer.apply(self, cls) for cls in classData]

    def unapply(self, classData):
        return [Standardizer.unapply(self, cls) for cls in classData]

class SegStandardizer(Standardizer):
    def __init__(self, x, *args, **kwargs):
        Standardizer.__init__(self, self.segToX(x), *args, **kwargs)

    def segToX(self, x):
        x = np.asarray(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        return np.vstack(x)

    def apply(self, x):
        x = np.asarray(x)
        return Standardizer.apply(self, self.segToX(x)).reshape(x.shape)

    def unapply(self, x):
        x = np.asarray(x)
        return Standardizer.unapply(self, self.segToX(x)).reshape(x.shape)

class ClassSegStandardizer(SegStandardizer):
    def __init__(self, classData, *args, **kwargs):
        collapsedClassData = [self.segToX(cls) for cls in classData]
        SegStandardizer.__init__(self, np.vstack(collapsedClassData), *args, **kwargs)

    def apply(self, classData):
        return [SegStandardizer.apply(self, cls) for cls in classData]

    def unapply(self, classData):
        return [SegStandardizer.unapply(self, cls) for cls in classData]
