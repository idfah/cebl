import numpy as np

from cebl import util
from cebl.util.clsm import *

import label


class Classifier(object):
    """Base class for classifiers.
    """
    def __init__(self, nIn, nCls):
        """Construct a new classifier.

        Args:
            nCls:   Number of classes.

            nIn:   Number of input dimensions.
        """
        self.nIn = nIn
        self.nCls = nCls

    def train(self):
        """Train a new classifier.  This method is not implemented in the
        Classifier base class and should be overridden by most classifiers.

        Args:
            classData:  Training data as a list of numpy arrays with shape
                        (nCls,nObs[,nIn]).
        """
        raise NotImplementedError('train not implemented.')

    def discrim(self, x, *args, **kwargs):
        return self.probs(x, *args, **kwargs)

    def discrimKnown(self, classData, *args, **kwargs):
        return [self.discrim(cls, *args, **kwargs) for cls in classData]

    def probs(self, x):
        """Find class membership probabilities.  This method is not
        implemented in the Classifier base class.  It MUST be overridden
        by ALL classifiers.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            A numpy array of shape (nObs,nCls) containing the
            estimated probability that each observation belongs
            to each class.

        Notes:
            If a classifier does not follow a probabilistic model,
            it can simple return 1's for positive labels and zeros
            everywhere else.
        """
        raise NotImplementedError('probs not implemented.')

    def probsKnown(self, classData, *args, **kwargs):
        return [self.probs(cls, *args, **kwargs) for cls in classData]

    def label(self, x, method='single', *args, **kwargs):
        """Assign class labels to novel data.

        Args:
            x:  Input data.  A numpy array with shape (nObs[,nIn]).

        Returns:
            A numpy array containing the integer class label found
            for each observation.
        """
        # are we going to do this everywhere? XXX - idfah
        if np.array(x).size == 0:
            return np.array([])

        method = method.lower()

        if method == 'single':
            return self.labelSingle(x, *args, **kwargs)
        elif method == 'vote':
            return self.labelVote(x, *args, **kwargs)
        elif method == 'intersect':
            return self.labelIntersect(x, *args, **kwargs)
        elif method == 'union':
            return self.labelUnion(x, *args, **kwargs)
        else:
            raise Exception('Unknown method.')

    def labelSingle(self, x, *args, **kwargs):
        dv = self.discrim(x, *args, **kwargs)
        return np.argmax(dv, axis=1)

    def labelVote(self, x, n, truncate=True, *args, **kwargs):
        """Assign class labels by voting across successive class labels.

        Args:
            x:          Input data.  A numpy array with shape (nObs[,nIn]).

            n:          Number of consecutive observations to use.  The
                        observations are combined in blocks of size n by
                        winner-takes-all voting.

            truncate:   Specifies how to handle trailing observations in the
                        case that the number of observations is not a multiple
                        of n.  If True (default), trailing observations will
                        be truncated.  If False, trailing observations will be
                        used but last label assignment may use fewer than n
                        observations.

            args, kwargs:   Additional arguments to pass to the label method.

        Returns:
            Numpy array with length nObs//n containing the predicted class labels.
        """
        labels = self.label(x, *args, **kwargs)

        def voteCount(lbl, axis=None):
            def vc(l):
                return np.bincount(np.array(l, dtype=np.int)).argmax()
            return np.apply_along_axis(func1d=vc, axis=axis, arr=lbl)

        return util.accum(labels, n, accumf=voteCount, truncate=truncate, axis=None)

    def labelIntersect(self, x, n, truncate=True, *args, **kwargs):
        """Assign class labels using the intersection of independent
        probabilities across successive observations.

        Args:
            x:          Input data.  A numpy array with shape (nObs[,nIn]).

            n:          Number of consecutive observations to use.  The
                        observations are combined in blocks of size n by
                        assuming that class memberships are independent
                        and finding the probability that ALL of the
                        segments belong to the target class.  In other
                        words, use the joint probability, i.e., the product
                        of the probabilites.

            truncate:   Specifies how to handle trailing observations in the
                        case that the number of observations is not a multiple
                        of n.  If True (default), trailing observations will
                        be truncated.  If False, trailing observations will be
                        used but last label assignment may use fewer than n
                        observations.

            args, kwargs:   Additional arguments to pass to the label method.

        Returns:
            Numpy array of length nObs//n containing the predicted integer
            class labels.

        Notes:
            Requires that the probs method is implemented.
        """
        # find the class membership probabilities
        probs = self.probs(x, *args, **kwargs)

        # use log probabilities for performance and stability
        logProbs = np.log(util.capZero(probs))

        # accumulate by summing log probs across n observations
        # equivalent to multiplying probs since we are only interested in the argmax
        intersect = util.accum(probs, n, accumf=np.sum, truncate=truncate, axis=0)

        # label is argmax of accumulated/summed log probabilities
        return np.argmax(intersect, axis=1)

    def labelUnion(self, x, n, truncate=True, *args, **kwargs):
        """Assign class labels using the union of independent probabilities
        across successive observations.

        Args:
            x:          Input data.  A numpy array with shape (nObs[,nIn]).

            n:          Number of consecutive observations to use.  The
                        observations are combined in blocks of size n by
                        assuming that class memberships are independent
                        and finding the probability that ANY of the
                        segments belong to the target class. In other
                        words, use the inclusion-exclusion rule to
                        compute the probability unions.

            truncate:   Specifies how to handle trailing observations in the
                        case that the number of observations is not a multiple
                        of n.  If True (default), trailing observations will
                        be truncated.  If False, trailing observations will be
                        used but last label assignment may use fewer than n
                        observations.

            args, kwargs:   Additional arguments to pass to the label method.

        Returns:
            Numpy array of length nObs//n containing the predicted integer
            class labels.

        Notes:
            Requires that the probs method is implemented.
        """
        # find class membership probability densities
        probs = self.probs(x, *args, **kwargs)

        # accumulate across n observations
        probs = util.accum(probs, n, accumf=util.punion, truncate=truncate, axis=0)

        # label is max probability density
        return np.argmax(probs, axis=1)

    def labelKnown(self, classData, *args, **kwargs):
        return [self.label(cls, *args, **kwargs) for cls in classData]

    def auc(self, classData, *args, **kwargs):
        return auc(self.probsKnown(classData, *args, **kwargs))

    def bca(self, classData, *args, **kwargs):
        return bca(self.labelKnown(classData, *args, **kwargs))

    def ca(self, classData, *args, **kwargs):
        return ca(self.labelKnown(classData, *args, **kwargs))

    def confusion(self, classData, normalize=True, *args, **kwargs):
        return confusion(self.labelKnown(classData, *args, **kwargs), normalize=True)

    def itr(self, classData, decisionRate=60.0, *args, **kwargs):
        return itr(self.labelKnown(classData, *args, **kwargs), decisionRate=decisionRate)

    def lloss(self, classData, *args, **kwargs):
        classProbs = self.probsKnown(classData, *args, **kwargs)
        probs, g = label.indicatorsFromList(classProbs)
        return lloss(probs, g)

    def roc(self, classData, *args, **kwargs):
        return roc(self.probsKnown(classData, *args, **kwargs))
