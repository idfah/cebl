"""Classification performance metrics.
"""

import numpy as np

from .arr import capZero


def roc(classProbs):
    if len(classProbs) > 2:
        raise RuntimeError('roc is only valid for two-class problems.')

    probs = np.concatenate([cls[:,1] for cls in classProbs])
    labels = np.ones(probs.size, dtype=np.bool)
    labels[:classProbs[0].shape[0]] = False
    
    idx = np.argsort(probs, kind='mergesort')[::-1]
    labels = labels[idx]
   
    fprCum = np.cumsum(labels == False)
    fprTotal = np.sum(labels == False).astype(probs.dtype)
    fpr = (fprCum / fprTotal) if fprTotal > 0.0 else np.zeros_like(probs)

    tprCum = np.cumsum(labels == True)
    tprTotal = np.sum(labels == True).astype(probs.dtype)
    tpr = (tprCum / tprTotal) if tprTotal > 0.0 else np.zeros_like(probs)

    return fpr, tpr
    
#def auc(classProbs):
#    if len(classProbs) > 2:
#        raise RuntimeError('auc is only valid for two-class problems.')
#
#    fpr, tpr = roc(classProbs)
#    return np.sum((fpr[1:] - fpr[:-1]) * tpr[1:])

def auc(classProbs):
    """Area under the roc curve
    """

    if len(classProbs) > 2:
        raise RuntimeError('auc is only implemented for two-class problems.')

    denom = classProbs[0].shape[0]*classProbs[1].shape[0]
    if denom == 0:
        return 0.0

    #score = 0.0
    #for pi in classProbs[0][:,0]:
    #    for pj in classProbs[1][:,0]:
    #        score += (pi > pj)

    # broadcast to get all pairs
    # fast but uses O(nObs0*nObs1) memory
    pi = classProbs[0][:,0]
    pj = classProbs[1][:,0]
    score = np.sum(pi[:,None] > pj)
    score += 0.5*np.sum(pi[:,None] == pj) # 50/50 tie breaker

    return score / float(denom)

def bca(classLabels):
    """Balanced classification accuracy
    """
    con = confusion(classLabels, normalize=False)
    return np.mean(np.diag(con) / np.sum(con, axis=0))

def ca(classLabels):
    """Compute the classification accuracy using predicted class labels
    with known true labels.

    Args:
        classLabels:    A list with length equal to the number of classes
                        with one element per class.  Each element of
                        this list contains a list of predictec class labels.

    Returns:
        Scalar classification accuracy as the fraction of correct labels
        over incorrect labels.  Multiply by 100 to get percent correct.
    """
    nCorrect = 0
    nTotal = 0

    for trueLabel, foundLabels in enumerate(classLabels):
        foundLabels = np.asarray(foundLabels)
        nCorrect += np.sum(foundLabels == trueLabel)
        nTotal += len(foundLabels)

    return nCorrect/float(nTotal)

def confusion(classLabels, normalize=True):
    """Find the confusion matrix using predicted class labels with
    known true labels.
    
    Args:
        classLabels:    A list with length equal to the number of classes
                        with one element per class.  Each element of
                        this list contains a list of predicted class labels.

        normalize:      If True (default) then each cell in the confusion
                        matrix is a fraction of the predicted labels over
                        the total labels for the given class, i.e., the
                        columns of the confusion matrix sum to one.  If
                        False then each cell is a count of class labels.

    Returns:
        The confusion matrix where each cell represents:

            row: predicted label
            col: actual label

        If the normalize argument (described above) is true then each cell
        is a fraction out of the total labels for the corresponding class.
        Otherwise, each cell is a label count.

    Examples:
        >>> from cebl import util
        >>> import numpy as np
        
        >>> a = [[0,0,0,1], [1,1,1,1,1,0], [2,2]]

        >>> con = util.confusion(a)

        >>> con
        array([[ 0.75      ,  0.16666667,  0.        ],
               [ 0.25      ,  0.83333333,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])

        >>> np.sum(con, axis=0)
        array([ 1.,  1.,  1.])

        >>> util.confusion(a, normalize=False)
        array([[ 3.,  1.,  0.],
               [ 1.,  5.,  0.],
               [ 0.,  0.,  2.]])
    """
    nCls = len(classLabels)
    confMat = np.zeros((nCls, nCls))

    for trueLabel, foundLabels in enumerate(classLabels):
        for foundLabel in foundLabels:
            confMat[foundLabel, trueLabel] += 1

    if normalize:
        counts = [len(l) for l in classLabels]
        confMat /= counts

    return confMat

def itrSimple(accuracy, nCls, decisionRate):
    if accuracy < 0.0 or np.isclose(accuracy, 0.0):
        return 0.0

    left = np.log2(nCls)
    middle = accuracy*np.log2(accuracy)

    right = 0.0 if np.isclose(accuracy, 1.0) else \
                (1.0-accuracy)*np.log2((1.0-accuracy)/(nCls-1.0))

    return decisionRate * (left + middle + right)

def itr(classLabels, decisionRate=60.0):
    """Information transfer rate in bits per minute

    Args:
        classLabels:    A list with length equal to the number of classes
                        with one element per class.  Each element of
                        this list contains a list of predicted class labels.

        decisionRate:   Scalar rate at which labels are assigned
                        in decisions per minute.

    Returns:
        Scalar information transfer rate in bits per minute.

    Refs:
        @book{pierce1980,
          title={An introduction to information theory: symbols, signals \& noise},
          author={Pierce, John},
          isbn={0486240614},
          pages={145--165},
          year={1980},
          publisher={Dover}
        }

        @article{wolpaw1998326,
          title={{EEG}-based communication: improved accuracy by response verification},
          author={Wolpaw, Jonathan and Ramoser, Herbert and McFarland, Dennis and Pfurtscheller, Gert},
          journal={{IEEE} Transactions on Rehabilitation Engineering},
          volume={6},
          number={3},
          pages={326--333},
          issn={1063--6528},
          year={1998},
          publisher={IEEE}
        }
    """
    nCls = len(classLabels)
    accuracy = ca(classLabels)

    return itrSimple(accuracy, nCls, decisionRate)

def lloss(probs, g):
    logLike = np.log(capZero(probs))
    return -np.mean(g*logLike)
